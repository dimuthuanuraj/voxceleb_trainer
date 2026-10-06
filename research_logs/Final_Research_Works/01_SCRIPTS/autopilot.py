#!/usr/bin/env python3
"""Autopilot — work the task graph down, filling every available GPU.

    python3 autopilot.py                 # run until nothing is left to do
    python3 autopilot.py --dry-run       # show the schedule, run nothing
    python3 autopilot.py --jobs 4        # cap concurrency (default: usable GPUs)
    python3 autopilot.py --status        # what the autopilot is doing right now

How it differs from `runner.py --run`
-------------------------------------
`runner.py` runs exactly one task and returns; that is the right tool when a
person is watching. The autopilot is the unattended form: it picks every task
whose dependencies are satisfied, runs several at once, and keeps going.

What it will NOT do, deliberately
---------------------------------
* **Manual tasks** (`W0.4` — the driver ticket) need a person, not a GPU.
* **Decision tasks** (`G1` — retire or re-run the Phase I claim) require a
  recorded human choice; the autopilot must not pick one by default.
* **Tasks that cannot fit** — it checks free VRAM against each task's
  `gpu_mb` before launching, so a 30 GB arbiter run is never dispatched onto a
  15 GB card where it would OOM half an hour in.
* **Retry forever** — a task that runs cleanly but never satisfies its exit
  test is *parked* after a few attempts rather than looping. Parking is
  visible; an infinite retry loop is not.

Long tasks
----------
Several tasks dispatch detached training and return in seconds. They are
re-attempted after a cooldown so they can score what has since finished. That
is why `A4`, `D1`, `D2`, `D3` may appear several times in the log — each pass
picks up more completed arms.

Logs
----
Every attempt writes `03_RESULTS/<ID>/autopilot_<n>_<timestamp>.log`, and
`03_RESULTS/<ID>/attempts.json` records the history. Nothing is overwritten.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(HERE)
STATE = os.path.join(FRW, "02_STATE")
RESULTS = os.path.join(FRW, "03_RESULTS")
AUTOSTATE = os.path.join(STATE, "autopilot.json")

sys.path.insert(0, HERE)
import runner  # noqa: E402

# Tasks the autopilot must never choose on its own.
MANUAL = {"W0.4"}            # needs the cluster administrator
DECISION = {"G1"}            # needs a recorded human decision (retire vs re-run)

MAX_ATTEMPTS = 3             # normal tasks
MAX_ATTEMPTS_LONG = 8        # tasks that dispatch detached work and re-score
COOLDOWN_LONG = 900          # seconds before re-attempting a long task
POLL = 20                    # seconds between scheduler passes


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def load_attempts(tid: str) -> dict:
    p = os.path.join(RESULTS, tid, "attempts.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"task": tid, "attempts": []}


def save_attempts(tid: str, data: dict) -> None:
    d = os.path.join(RESULTS, tid)
    os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "attempts.json"), "w"), indent=2)


def write_autostate(payload: dict) -> None:
    os.makedirs(STATE, exist_ok=True)
    json.dump(payload, open(AUTOSTATE, "w"), indent=2)



def idle_gpu_count() -> int:
    """GPUs with zero compute processes across the cluster.

    `runner.probe_gpu_count()` reports cards that ANSWER nvidia-smi, which is a
    different question: a card busy with a detached training job is usable but
    not free. Dispatching against the former is how five T-series arms ended up
    on one node with another node idle (2026-09-14).
    """
    nodes = ["compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4"]
    idle = 0
    for node in nodes:
        try:
            out = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node,
                 "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | "
                 "while read -r i; do "
                 "  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i 2>/dev/null | wc -l); "
                 "  echo \"$i $n\"; done"],
                capture_output=True, text=True, timeout=30)
            for line in out.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == "0":
                    idle += 1
        except Exception:
            pass
    return idle


class Job:
    def __init__(self, tid: str, task: dict, proc, logpath: str, n: int):
        self.tid, self.task, self.proc = tid, task, proc
        self.logpath, self.n = logpath, n
        self.started = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started


def eligible(data: dict, st: dict, running: set, largest_mb: int, free_slots: int):
    """Tasks that may be launched right now, in register order."""
    out = []
    for t in data["tasks"]:
        tid = t["id"]
        if tid in running or st[tid]["passed"] or not st[tid]["ready"]:
            continue
        if tid in MANUAL or tid in DECISION:
            continue
        need = int(t.get("gpu_mb") or 0)
        if need > largest_mb:
            continue                       # cannot fit on any live card
        if need > 0 and free_slots <= 0:
            continue                       # no GPU headroom this pass

        a = load_attempts(tid)
        tries = len(a["attempts"])
        cap = MAX_ATTEMPTS_LONG if t.get("long") else MAX_ATTEMPTS
        if tries >= cap:
            continue                       # parked
        if t.get("long") and a["attempts"]:
            last = a["attempts"][-1].get("finished_epoch", 0)
            if time.time() - last < COOLDOWN_LONG:
                continue                   # still cooling down
        out.append(t)
    return out


def launch(t: dict) -> Job:
    tid = t["id"]
    a = load_attempts(tid)
    n = len(a["attempts"]) + 1
    d = os.path.join(RESULTS, tid)
    os.makedirs(d, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logpath = os.path.join(d, f"autopilot_{n:02d}_{stamp}.log")

    fh = open(logpath, "w")
    fh.write(f"# {tid} — {t['title']}\n")
    fh.write(f"# attempt {n}, started {now()}\n")
    fh.write(f"# script {t['script']}   cost {t['cost']}   gpu_mb {t['gpu_mb']}\n")
    fh.write(f"# source {t['source']}\n{'=' * 78}\n\n")
    fh.flush()

    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "runner.py"), "--run", tid],
        cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    proc._logfh = fh                       # keep the handle alive
    log(f"LAUNCH {tid:7s} attempt {n}  -> {os.path.relpath(logpath, FRW)}")
    return Job(tid, t, proc, logpath, n)


def finish(job: Job, data: dict) -> bool:
    rc = job.proc.returncode
    try:
        job.proc._logfh.close()
    except Exception:
        pass
    passed, detail = runner.check_exit(job.task["exit_test"])

    a = load_attempts(job.tid)
    a["attempts"].append({
        "n": job.n, "rc": rc, "passed": passed, "exit_test": detail,
        "log": os.path.relpath(job.logpath, FRW),
        "started": datetime.datetime.fromtimestamp(
            job.started, datetime.timezone.utc).isoformat(timespec="seconds"),
        "finished": now(), "finished_epoch": time.time(),
        "elapsed_s": round(job.elapsed, 1),
    })
    save_attempts(job.tid, a)

    mark = "PASS" if passed else ("ran, incomplete" if rc == 0 else f"FAILED rc={rc}")
    log(f"{'DONE  ' if passed else 'END   '}{job.tid:7s} {mark}  "
        f"({job.elapsed/60:.1f} min)  {detail[:70]}")
    if not passed:
        cap = MAX_ATTEMPTS_LONG if job.task.get("long") else MAX_ATTEMPTS
        if len(a["attempts"]) >= cap:
            log(f"PARK  {job.tid:7s} {len(a['attempts'])} attempts without passing — "
                f"not retrying. See {os.path.relpath(job.logpath, FRW)}")
    return passed


def cmd_status() -> int:
    if not os.path.exists(AUTOSTATE):
        print("autopilot has not run yet")
        return 1
    d = json.load(open(AUTOSTATE))
    print(json.dumps(d, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=0,
                    help="max concurrent tasks (default: number of usable GPUs, min 2)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-passes", type=int, default=0,
                    help="stop after N scheduler passes (0 = until nothing is left)")
    a = ap.parse_args()

    if a.status:
        return cmd_status()

    data = runner.load_tasks()
    n_gpu, gpu_detail, largest = runner.probe_gpu_count()
    jobs_cap = a.jobs or max(2, n_gpu)

    log(f"autopilot starting — {n_gpu} usable GPU(s), {idle_gpu_count()} idle, "
        f"largest {largest} MiB")
    log(f"  {gpu_detail}")
    log(f"  concurrency: {jobs_cap}")
    log(f"  never auto-run: manual {sorted(MANUAL)}, decision {sorted(DECISION)}")

    st = runner.compute_status(data)
    cant_fit = [t["id"] for t in data["tasks"]
                if int(t.get("gpu_mb") or 0) > largest and not st[t["id"]]["passed"]]
    if cant_fit:
        log(f"  cannot fit on any live card ({largest} MiB): {', '.join(cant_fit)}")

    if a.dry_run:
        elig = eligible(data, st, set(), largest, jobs_cap)
        print(f"\n  {len(elig)} task(s) would start now:")
        for t in elig[:jobs_cap]:
            print(f"    {t['id']:7s} {t['title'][:58]:58s} {t['cost']:>14s}")
        if len(elig) > jobs_cap:
            print(f"    ... and {len(elig)-jobs_cap} more queued behind them")
        blocked = [t["id"] for t in data["tasks"]
                   if not st[t["id"]]["passed"] and not st[t["id"]]["ready"]]
        print(f"\n  blocked on dependencies: {', '.join(blocked) or 'none'}")
        return 0

    running: dict[str, Job] = {}
    passes = 0
    started_at = time.time()

    try:
        while True:
            passes += 1
            st = runner.compute_status(data)
            n_gpu, gpu_detail, largest = runner.probe_gpu_count()

            # Free slots = cards with NO compute process, minus the GPU tasks this
            # scheduler already has in flight. Counting only our own jobs would
            # ignore the detached training that earlier passes dispatched, and
            # we would oversubscribe the cluster within a couple of passes.
            idle_now = idle_gpu_count()
            gpu_running = sum(1 for j in running.values()
                              if int(j.task.get("gpu_mb") or 0) > 0)
            free_slots = max(0, idle_now - gpu_running)

            done = [tid for tid, j in running.items() if j.proc.poll() is not None]
            for tid in done:
                finish(running.pop(tid), data)

            if done:
                st = runner.compute_status(data)

            while len(running) < jobs_cap:
                elig = eligible(data, st, set(running), largest, free_slots)
                if not elig:
                    break
                t = elig[0]
                job = launch(t)
                running[t["id"]] = job
                if int(t.get("gpu_mb") or 0) > 0:
                    free_slots -= 1
                st = runner.compute_status(data)

            n_pass = sum(1 for tid in st if st[tid]["passed"])
            write_autostate({
                "updated": now(), "pass": passes,
                "running": {tid: {"attempt": j.n, "elapsed_min": round(j.elapsed / 60, 1),
                                  "log": os.path.relpath(j.logpath, FRW)}
                            for tid, j in running.items()},
                "passed": n_pass, "total": len(st),
                "gpus": gpu_detail, "largest_free_mb": largest,
                "elapsed_h": round((time.time() - started_at) / 3600, 2),
            })

            if not running:
                st = runner.compute_status(data)
                if not eligible(data, st, set(), largest, max(1, n_gpu)):
                    log("nothing left that the autopilot may run")
                    break

            if a.max_passes and passes >= a.max_passes:
                log(f"stopping after {passes} passes as requested")
                break
            time.sleep(POLL)

    except KeyboardInterrupt:
        log("interrupted — leaving running tasks alone; re-run to resume")

    st = runner.compute_status(data)
    n_pass = sum(1 for tid in st if st[tid]["passed"])
    log(f"autopilot stopping — {n_pass}/{len(st)} tasks pass, "
        f"{(time.time()-started_at)/3600:.2f} h elapsed")

    todo = [(tid, s) for tid, s in st.items() if not s["passed"]]
    if todo:
        log("outstanding:")
        for tid, s in sorted(todo):
            a_ = load_attempts(tid)
            why = ("parked after %d attempts" % len(a_["attempts"])) if a_["attempts"] else ""
            if tid in MANUAL:
                why = "MANUAL — needs the cluster administrator"
            elif tid in DECISION:
                why = "DECISION — needs a recorded human choice"
            elif s["blocked_by"]:
                why = "waits on " + ", ".join(s["blocked_by"])
            log(f"  {tid:7s} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
