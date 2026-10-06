#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Schedule a stage's experiment scripts across the cluster's GPUs.

    python experiments/tools/run_queue.py --stage A
    python experiments/tools/run_queue.py --stage A --scale dev
    python experiments/tools/run_queue.py --stage A --dry-run
    python experiments/tools/run_queue.py --only A_ecapa1024_aamsoftmax_si_s42
    python experiments/tools/run_queue.py --stage A --local        # this host's GPU

One worker thread per GPU slot pulls from a shared job queue, so a stage's runs
proceed at whatever rate the hardware allows without any manual babysitting.
Placement goes through ``tools/gpurun.sh -n <node> -g <gpu>``, which is the
project's existing mechanism -- /mnt/ricproject3 is NFS-mounted at the same path
on every node, so no data moves.

Restartability
--------------
A job whose ``final.json`` records ``exit_code == 0`` is skipped.  Killing the
queue and restarting it therefore resumes rather than redoing work, which
matters when a stage runs for days.  ``--redo`` overrides.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.join(REPO_ROOT, "experiments", "scripts")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
GPURUN = os.path.join(REPO_ROOT, "tools", "gpurun.sh")

PRINT_LOCK = threading.Lock()

# Interrupt handling ---------------------------------------------------------
# Each job is launched with start_new_session=True, which puts its ssh client in
# its own process group. Without that, Ctrl+C goes to the whole foreground
# group and reaches the ssh clients directly, and what happens next is not
# uniform: on some nodes SIGINT propagates through to the remote python and it
# dies (recorded as rc=255), on others the remote process is orphaned and keeps
# training with nothing left to record its result. Both happened in the same
# interrupted Stage A run on 2026-08-12.
#
# With the jobs isolated, SIGINT reaches only this process and the handler below
# decides deliberately what to do with them.
STOP = threading.Event()
RUNNING = {}            # exp_id -> (node, gpu, Popen)
RUNNING_LOCK = threading.Lock()
KEEP_ON_INTERRUPT = False

# Retry of INFRASTRUCTURE failures ------------------------------------------
# A failed job used to be consumed, never retried. On 2026-08-13 all three
# compute nodes rebooted mid-stage; for the ~4 minutes they were booting, sshd
# answered with pam_nologin ("System is booting up. Unprivileged users are not
# permitted to log in yet"), so every remaining job failed with rc=255 in under
# a second. Five experiments were burned that way and the queue then had nothing
# left to schedule, on a cluster that came back healthy minutes later.
#
# ssh reserves exit code 255 for its own errors, so rc=255 with NO final.json
# means the experiment never started -- an infrastructure problem, not a result.
# Those are worth retrying. A job that ran and genuinely failed writes a
# final.json, and is not retried, because repeating it would just repeat the
# failure.
ATTEMPTS = {}
ATTEMPTS_LOCK = threading.Lock()
SSH_ERROR_RC = 255


def log(msg):
    with PRINT_LOCK:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _remote_kill(node, exp_id):
    """Stop an experiment's processes on `node` (same matching as stop_queue.py)."""
    try:
        # NOTE: `pkill -f <pattern>` cannot be used here. The remote `bash -c`
        # carries the pattern in its own command line, so pkill matches itself
        # and kills its own shell before reaching the trainer -- verified on
        # compute-node-1. Collect PIDs first and skip this shell and its parent.
        script = (
            f"for p in $(pgrep -f 'experiments/scripts/{exp_id}[._]' "
            f"           ; pgrep -f 'exps/{exp_id}[ /_]'); do "
            f"  if [ \"$p\" != \"$$\" ] && [ \"$p\" != \"$PPID\" ]; then "
            f"    kill -TERM \"$p\" 2>/dev/null; "
            f"  fi; "
            f"done; true"
        )
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, script],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pass


def _handle_interrupt(signum, frame):
    """First Ctrl+C stops the queue; a second one is a hard exit."""
    if STOP.is_set():
        with PRINT_LOCK:
            print("\nsecond interrupt -- exiting immediately; any surviving "
                  "remote jobs can be listed with "
                  "`python experiments/tools/stop_queue.py`", flush=True)
        os._exit(130)

    STOP.set()
    with PRINT_LOCK:
        print("\ninterrupt: no further jobs will be launched.", flush=True)

    with RUNNING_LOCK:
        running = list(RUNNING.items())

    if KEEP_ON_INTERRUPT:
        with PRINT_LOCK:
            print(f"leaving {len(running)} running job(s) alone (--keep-running). "
                  f"They will finish and write their results, but this queue will "
                  f"not record them. Track with `stop_queue.py`.", flush=True)
        return

    for exp_id, (node, gpu, proc) in running:
        with PRINT_LOCK:
            print(f"  stopping {exp_id} on {node}:gpu{gpu}", flush=True)
        _remote_kill(node, exp_id)          # the remote trainer
        try:
            proc.terminate()                # the local ssh client
        except Exception:
            pass
    with PRINT_LOCK:
        print(f"stopped {len(running)} job(s). Re-run the same command to "
              f"resume; completed experiments are skipped.", flush=True)


def discover_slots(min_free_mb=8000):
    """Ask each candidate node which GPUs are free.

    Returns [(node, gpu_index, free_mb)]. The free-memory figure is carried
    through because the cluster is heterogeneous -- 15 GB T4s on compute-node-1,
    23 GB A10s on node-3, a 46 GB A40 on node-4 -- and a scheduler that ignores
    it will send a 112M-parameter hybrid to a T4 and lose the run to an OOM
    several minutes in.
    """
    slots = []
    out = subprocess.run(
        [GPURUN, "--status"], capture_output=True, text=True, timeout=180
    ).stdout
    node = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            node = line.strip("= ").strip()
            continue
        parts = [p.strip() for p in line.split(",")]
        if node and len(parts) >= 4 and parts[0].isdigit():
            try:
                free_mb = int(parts[2].split()[0])
            except (ValueError, IndexError):
                continue
            if free_mb >= min_free_mb:
                slots.append((node, int(parts[0]), free_mb))
    return slots


def gpu_requirement(exp_id):
    """Free GPU memory (MiB) this experiment needs, from the registry."""
    try:
        sys.path.insert(0, REPO_ROOT)
        from experiments import registry as R

        man = os.path.join(SCRIPT_DIR, f"{exp_id}.py")
        if os.path.isfile(man):
            ns = {}
            with open(man, encoding="utf-8") as fh:
                src = fh.read()
            # The generated script carries its resolved spec as a literal; read
            # it without importing, so this never triggers a training run.
            start = src.index("EXPERIMENT = ")
            literal = src[start + len("EXPERIMENT = "):src.index("\nif __name__")]
            spec = ast.literal_eval(literal.strip())
            if spec.get("min_gpu_mb"):
                return int(spec["min_gpu_mb"])
            arch = R.ARCHITECTURES.get(spec.get("architecture"), {})
            return int(arch.get("min_gpu_mb", R.MIN_GPU_MB_DEFAULT))
    except Exception:
        pass
    return 9000


def _final_exit_code(exp_id, scale):
    """Exit code recorded by the experiment itself, or None if not finished."""
    name = exp_id if scale == "full" else f"{exp_id}__{scale}"
    path = os.path.join(RESULTS_DIR, name, "final.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("exit_code")
    except Exception:
        return None


def _wait_for(proc, exp_id, scale, grace_s=180):
    """Wait for a job, tolerating an ssh client that outlives its command.

    `gpurun.sh` runs `ssh -t`. When the remote experiment exits, stray
    dataloader workers can keep the pty open, and the ssh client then never
    returns -- observed on 2026-08-13, where three slots sat blocked for 5.5
    hours on experiments that had already written `exit_code 0`, so the queue
    stopped scheduling and most of the cluster went idle.

    The experiment itself records its result in final.json, so that file is the
    authority on completion. Once it appears, the ssh client gets a grace period
    to close on its own and is then closed here.
    """
    waited_after_final = 0
    while True:
        try:
            return proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        if STOP.is_set():
            continue
        rc = _final_exit_code(exp_id, scale)
        if rc is None:
            continue
        waited_after_final += 20
        if waited_after_final >= grace_s:
            log(f"{exp_id} recorded exit_code={rc} but its ssh client is still "
                f"open after {grace_s}s; closing it and freeing the slot")
            try:
                proc.terminate()
                proc.wait(timeout=20)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return rc


def live_on_cluster():
    """Experiment ids that already have a process on some node.

    run_queue decides what to run from final.json alone, which says nothing
    about whether a job is running RIGHT NOW on another node. Two queues over
    overlapping job sets therefore each start their own copy, and both write the
    same save_path and the same epochs.jsonl -- observed twice on 2026-08-16
    (en_full, then resnetse34v2_en_matched). Checking the cluster before
    scheduling closes that hole.
    """
    import re
    live = set()
    try:
        with open(GPURUN, encoding="utf-8") as fh:
            nodes = []
            for line in fh:
                if line.strip().startswith("CANDIDATE_NODES="):
                    nodes = [n for n in line.split("=", 1)[1].strip().strip("()").split()
                             if n and not n.startswith("#")]
                    break
    except Exception:
        return live
    for node in nodes:
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node,
                 "pgrep -af 'experiments/scripts/' || true"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        for line in out.splitlines():
            m = re.search(r"experiments/scripts/([A-Za-z0-9_]+)\.py", line)
            if m:
                live.add(m.group(1))
    return live


def is_done(exp_id, scale):
    name = exp_id if scale == "full" else f"{exp_id}__{scale}"
    path = os.path.join(RESULTS_DIR, name, "final.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("exit_code") == 0
    except Exception:
        return False


def jobs_for(stage=None, only=None):
    if only:
        names = [o if o.endswith(".py") else f"{o}.py" for o in only]
        paths = [os.path.join(SCRIPT_DIR, n) for n in names]
    else:
        paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, f"{stage}_*.py")))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "missing script(s): "
            + ", ".join(os.path.basename(m) for m in missing)
            + f"\ngenerate them first: python experiments/tools/gen_scripts.py --stage {stage}"
        )
    return paths


def build_cmd(script, node, gpu, scale, local, python, resume=False):
    inner = [python, os.path.relpath(script, REPO_ROOT), "--scale", scale, "--quiet"]
    if resume:
        inner.append("--resume")
    if local:
        return inner, {"CUDA_VISIBLE_DEVICES": str(gpu)}
    # gpurun.sh pins CUDA_VISIBLE_DEVICES on the remote side, so the script's
    # own --gpu is not passed here; forcing both would be redundant.
    return [GPURUN, "-n", node, "-g", str(gpu), "--"] + inner, {}


def worker(slot, pending, lock, args, state):
    """One worker per GPU slot.

    Instead of a plain queue, each worker scans the pending list for the first
    job its slot has the memory to run. A shared FIFO would hand a 20 GB SSL job
    to a 15 GB T4 and lose it to an OOM, while the A40 sat idle.
    """
    node, gpu, free_mb = slot
    while True:
        if STOP.is_set():
            return
        script = None
        with lock:
            for i, cand in enumerate(pending):
                if gpu_requirement(os.path.splitext(os.path.basename(cand))[0]) <= free_mb:
                    script = pending.pop(i)
                    break
            if script is None:
                # Nothing left that fits here. If jobs remain, they are waiting
                # on a bigger card, not on this one.
                if pending:
                    blocked = [os.path.splitext(os.path.basename(c))[0] for c in pending]
                    log(f"slot {node}:gpu{gpu} ({free_mb} MiB) idle -- "
                        f"{len(blocked)} job(s) need a larger GPU")
                return
        exp_id = os.path.splitext(os.path.basename(script))[0]
        need = gpu_requirement(exp_id)
        cmd, extra_env = build_cmd(script, node, gpu, args.scale, args.local,
                                   args.python, resume=args.resume)

        if args.dry_run:
            log(f"[dry-run] {node}:gpu{gpu} ({free_mb} MiB, needs {need})  "
                f"{' '.join(cmd)}")
            continue

        log(f"START  {exp_id}  ->  {node}:gpu{gpu} ({free_mb} MiB free, "
            f"needs {need})")
        t0 = time.time()
        env = os.environ.copy()
        env.update(extra_env)
        logdir = os.path.join(REPO_ROOT, "experiments", "queue_logs")
        os.makedirs(logdir, exist_ok=True)
        logpath = os.path.join(logdir, f"{exp_id}.{args.scale}.log")
        with open(logpath, "w", encoding="utf-8", buffering=1) as fh:
            # start_new_session isolates the ssh client from this terminal's
            # process group, so Ctrl+C cannot reach it (or the remote trainer)
            # by accident -- see the note at the top of this file.
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, env=env, stdout=fh,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            with RUNNING_LOCK:
                RUNNING[exp_id] = (node, gpu, proc)
            try:
                rc = _wait_for(proc, exp_id, args.scale)
            finally:
                with RUNNING_LOCK:
                    RUNNING.pop(exp_id, None)

        if STOP.is_set():
            log(f"STOPPED {exp_id} (queue interrupted)")
            return
        dt = (time.time() - t0) / 3600.0

        # Transient? Put it back on the queue rather than burning it.
        # A deliberate stop_queue.py --kill also surfaces as rc=255. The marker
        # it leaves is the only way to tell that apart from a node dropping out.
        _name = exp_id if args.scale == "full" else f"{exp_id}__{args.scale}"
        _marker = os.path.join(RESULTS_DIR, _name, "STOPPED_BY_USER")
        if os.path.isfile(_marker):
            log(f"{exp_id} was stopped deliberately (STOPPED_BY_USER); not retrying")
            os.remove(_marker)
            continue

        # The NFS boot race is ALSO transient but does not give rc=255: after a
        # node reboots, /mnt/ricproject3 can be briefly absent, so gpurun's
        # `cd` fails and bash exits 1. Observed 2026-08-17 on compute-node-2,
        # whose mount was healthy again minutes later. Detect it from the log
        # rather than the exit code, which cannot distinguish it from a genuine
        # failure.
        _mount_race = False
        if rc != 0 and _final_exit_code(exp_id, args.scale) is None:
            try:
                with open(logpath, encoding="utf-8", errors="replace") as fh:
                    tail = fh.read()[-2000:]
                _mount_race = ("cd: /mnt/ricproject" in tail
                               and "No such file or directory" in tail)
            except Exception:
                pass

        if ((rc == SSH_ERROR_RC or _mount_race)
                and _final_exit_code(exp_id, args.scale) is None
                and not STOP.is_set()):
            with ATTEMPTS_LOCK:
                ATTEMPTS[exp_id] = ATTEMPTS.get(exp_id, 0) + 1
                tries = ATTEMPTS[exp_id]
            if tries <= args.max_retries:
                backoff = min(60 * 2 ** (tries - 1), 600)
                log(f"RETRY  {exp_id}  rc=255 with no result written "
                    f"(node unreachable?) -- attempt {tries}/{args.max_retries}, "
                    f"requeuing in {backoff}s")
                # Wait before requeuing so a rebooting node has time to come
                # back; other slots keep working meanwhile.
                for _ in range(backoff):
                    if STOP.is_set():
                        return
                    time.sleep(1)
                with lock:
                    pending.append(script)
                continue
            log(f"GIVE UP {exp_id} after {tries} attempt(s) at rc=255")

        state[exp_id] = {"rc": rc, "hours": round(dt, 3),
                         "slot": f"{node}:gpu{gpu}", "gpu_free_mb": free_mb,
                         "gpu_required_mb": need,
                         "attempts": ATTEMPTS.get(exp_id, 0) + 1}
        log(
            f"{'DONE ' if rc == 0 else 'FAIL '} {exp_id}  rc={rc}  {dt:.2f} h  "
            f"(log: {os.path.relpath(logpath, REPO_ROOT)})"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default=None,
                    choices=["A", "B", "C", "D", "E", "F", "G", "H", "M"])
    ap.add_argument("--only", nargs="+", default=None, help="explicit experiment ids")
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    ap.add_argument("--redo", action="store_true", help="re-run completed experiments")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="run on this host instead of dispatching over ssh")
    ap.add_argument("--nodes", default=None,
                    help="comma-separated nodes to use, e.g. compute-node-4. "
                         "Lets a second queue run alongside a first one without "
                         "double-booking the same GPUs.")
    ap.add_argument("--min-free-mb", type=int, default=8000,
                    help="a GPU must have at least this much free to be used")
    ap.add_argument("--slots", type=int, default=None,
                    help="cap concurrent jobs (default: one per free GPU)")
    ap.add_argument("--python", default="python")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="how many times to requeue a job that failed with "
                         "rc=255 and wrote no result (node rebooting, ssh "
                         "unreachable). Genuine experiment failures are never "
                         "retried.")
    ap.add_argument("--resume", action="store_true",
                    help="continue interrupted runs in place: keeps the existing "
                         "results directory and appends to epochs.jsonl instead "
                         "of moving it aside. The trainer independently resumes "
                         "from the highest checkpoint in exps/<exp_id>/model/.")
    ap.add_argument("--keep-running", action="store_true",
                    help="on Ctrl+C, leave already-started jobs running on their "
                         "nodes instead of stopping them (default is to stop "
                         "them, so the cluster is left clean)")
    args = ap.parse_args()

    global KEEP_ON_INTERRUPT
    KEEP_ON_INTERRUPT = args.keep_running
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    if not args.stage and not args.only:
        ap.error("pass --stage or --only")

    scripts = jobs_for(args.stage, args.only)
    pending = [
        s for s in scripts
        if args.redo or not is_done(os.path.splitext(os.path.basename(s))[0], args.scale)
    ]
    skipped = len(scripts) - len(pending)

    # Never schedule something another queue is already running: two processes
    # sharing one save_path corrupt each other's checkpoints.
    already = live_on_cluster()
    if already:
        clash = [s for s in pending
                 if os.path.splitext(os.path.basename(s))[0] in already]
        if clash:
            for c in clash:
                print(f"  SKIP {os.path.basename(c)[:-3]} — already running on the "
                      f"cluster (another queue owns it)")
            pending = [s for s in pending if s not in clash]

    if args.local:
        slots = [("localhost", 0, 10 ** 6)]
    else:
        slots = discover_slots(args.min_free_mb)
        if args.nodes:
            want = {n.strip() for n in args.nodes.split(",")}
            slots = [s for s in slots if s[0] in want]
        if not slots:
            raise SystemExit(
                "no free GPU found. Check `tools/gpurun.sh --status`, or pass "
                "--local to run on this host."
            )
    if args.slots:
        slots = slots[: args.slots]

    print(f"stage {args.stage or 'custom'}: {len(scripts)} experiments, "
          f"{skipped} already complete, {len(pending)} to run")
    print("slots: " + ", ".join(f"{n}:gpu{g} ({m} MiB)" for n, g, m in slots))
    biggest = max((m for _, _, m in slots), default=0)
    too_big = [os.path.basename(s) for s in pending
               if gpu_requirement(os.path.splitext(os.path.basename(s))[0]) > biggest]
    if too_big:
        print(f"WARNING: {len(too_big)} job(s) need more GPU memory than any "
              f"free slot has ({biggest} MiB) and will not be run: "
              + ", ".join(too_big[:3]) + (" ..." if len(too_big) > 3 else ""))
    if not pending:
        print("nothing to do")
        return 0

    shared = list(pending)
    lock = threading.Lock()

    state, threads = {}, []
    t0 = time.time()
    for slot in slots:
        t = threading.Thread(target=worker, args=(slot, shared, lock, args, state),
                             daemon=True)
        t.start()
        threads.append(t)
    # join with a timeout so the signal handler can run promptly; a bare
    # join() blocks signal delivery in the main thread on some platforms.
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.5)
        if STOP.is_set() and not any(t.is_alive() for t in threads):
            break
    if STOP.is_set():
        log("queue stopped. Re-run the same command to resume "
            "(completed experiments are skipped).")
        return 130

    ok = sum(1 for v in state.values() if v["rc"] == 0)
    total_h = round((time.time() - t0) / 3600.0, 2)
    print(f"\n{ok}/{len(state)} succeeded in {total_h} h wall")
    for k, v in sorted(state.items()):
        mark = "ok  " if v["rc"] == 0 else "FAIL"
        print(f"  {mark} {k:46s} {v['hours']:6.2f} h  {v['slot']}"
              f"  ({v.get('gpu_required_mb')} MiB needed)")

    summary = {
        "stage": args.stage,
        "scale": args.scale,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_hours": total_h,
        "results": state,
    }
    out = os.path.join(RESULTS_DIR, f"_queue_{args.stage or 'custom'}_{args.scale}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0 if ok == len(state) else 1


if __name__ == "__main__":
    sys.exit(main())
