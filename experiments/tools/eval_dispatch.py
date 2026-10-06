#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spread the outstanding evaluations over every free GPU on the cluster.

    python experiments/tools/eval_dispatch.py            # show the plan
    python experiments/tools/eval_dispatch.py --run      # dispatch it
    python experiments/tools/eval_dispatch.py --run --redo-all

"Outstanding" means an experiment that trained successfully and either has no
``test_eval.json`` at all, or has one whose ``checkpoint_load.backbone_skipped``
is non-empty -- i.e. it was scored against a model that did not match its own
checkpoint, so the numbers came from partly-random weights.

Why not just run ``evaluate.py --all``
--------------------------------------
``--all`` is one serial process.  Sharding it by stage helps, but stages are
wildly uneven (A is 28 experiments, H is 4), so the long stage keeps a GPU busy
while the others idle.  This assigns *individual experiments* round-robin across
every GPU, which is the granularity that actually balances.

Each slot runs its own list serially in one detached process, so a dropped ssh
connection cannot kill it -- that is how the previous evaluation driver died,
with a broken pipe, after two hours of work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "experiments", "results")
QLOGS = os.path.join(REPO, "experiments", "queue_logs")
GPURUN = os.path.join(REPO, "tools", "gpurun.sh")
PY = "/home/anuraj/anaconda2025/envs/SL_SPV/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
MIN_FREE_MB = 4000


def nodes():
    try:
        with open(GPURUN, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("CANDIDATE_NODES="):
                    inner = line.split("=", 1)[1].strip().strip("()")
                    return [n for n in inner.split() if n and not n.startswith("#")]
    except Exception:
        pass
    return ["compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4"]


def outstanding(redo_all=False):
    out = []
    for d in sorted(os.listdir(RESULTS)):
        if "__smoke" in d or ".superseded-" in d:
            continue
        path = os.path.join(RESULTS, d)
        if not os.path.isdir(path):
            continue
        fin = os.path.join(path, "final.json")
        if not os.path.isfile(fin):
            continue
        try:
            if json.load(open(fin, encoding="utf-8")).get("exit_code") != 0:
                continue
        except Exception:
            continue
        ev = os.path.join(path, "test_eval.json")
        if redo_all or not os.path.isfile(ev):
            out.append((d, "never scored" if not os.path.isfile(ev) else "redo-all"))
            continue
        try:
            j = json.load(open(ev, encoding="utf-8"))
        except Exception:
            out.append((d, "unreadable"))
            continue
        if (j.get("checkpoint_load") or {}).get("backbone_skipped"):
            out.append((d, "contaminated"))
    return out


def gpu_slots(node):
    """Free GPUs on one node, and whether an evaluation is already running."""
    try:
        r = subprocess.run(
            SSH + [node,
                   "nvidia-smi --query-gpu=index,memory.used,memory.total "
                   "--format=csv,noheader 2>/dev/null; echo '###'; "
                   "pgrep -cf '[e]valuate.py|trainSpeakerNet.py' || true"],
            capture_output=True, text=True, timeout=45)
    except Exception:
        return [], -1
    if r.returncode != 0:
        return [], -1
    head, _, tail = r.stdout.partition("###")
    slots = []
    for line in head.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[0].isdigit():
            used = int(re.sub(r"\D", "", p[1]) or 0)
            total = int(re.sub(r"\D", "", p[2]) or 0)
            if total - used >= MIN_FREE_MB:
                slots.append((node, int(p[0]), total - used))
    busy = int("".join(c for c in tail if c.isdigit()) or 0)
    return slots, busy


def stop_existing(ns):
    def kill(node):
        subprocess.run(SSH + [node,
                              "pids=$(pgrep -f '[e]valuate.py'); "
                              "[ -n \"$pids\" ] && kill $pids 2>/dev/null; sleep 4; "
                              "pids=$(pgrep -f '[e]valuate.py'); "
                              "[ -n \"$pids\" ] && kill -9 $pids 2>/dev/null; true"],
                       capture_output=True, text=True, timeout=60)
    with ThreadPoolExecutor(max_workers=len(ns)) as ex:
        list(ex.map(kill, ns))


def dispatch(slot, exps):
    node, gpu, _ = slot
    log = os.path.join(QLOGS, f"eval-dispatch-{node.replace('compute-node-','n')}g{gpu}.log")
    inner = " ; ".join(
        f"{PY} -u experiments/tools/evaluate.py --exp {e} --redo" for e in exps)
    cmd = (f"cd {REPO} && setsid nohup env CUDA_VISIBLE_DEVICES={gpu} "
           f"PYTHONPATH={REPO}/experiments/common/runtime_patches "
           f"SLSPV_REPO_ROOT={REPO} "
           f"bash -c '{inner}' > {log} 2>&1 < /dev/null &")
    # -n and stdin=DEVNULL: without them ssh keeps the session open waiting on
    # the backgrounded child's inherited descriptors, and the call blocks until
    # its timeout even though the job started fine.
    #
    # A timeout here is NOT a failure to launch -- the remote command has
    # already run by then. Raising would abort the loop and leave the remaining
    # GPUs unused, which is exactly what happened on the first attempt: one slot
    # dispatched, four left idle. So a timeout is reported and the loop goes on;
    # the caller verifies from the logs.
    try:
        r = subprocess.run(SSH + ["-n", node, cmd], capture_output=True,
                           text=True, timeout=45, stdin=subprocess.DEVNULL)
        note = "started" if r.returncode == 0 else f"rc={r.returncode}"
    except subprocess.TimeoutExpired:
        note = "started (ssh did not return; verify from log)"
    except Exception as exc:
        note = f"FAILED {exc!r}"
    return node, gpu, log, note


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="actually dispatch")
    ap.add_argument("--redo-all", action="store_true",
                    help="re-score every experiment, not just the outstanding ones")
    ap.add_argument("--stop-first", action="store_true", default=True,
                    help="stop evaluations already running (avoids two writers "
                         "on one test_eval.json)")
    args = ap.parse_args()

    work = outstanding(args.redo_all)
    if not work:
        print("  nothing outstanding — every trained experiment has a clean evaluation.")
        return 0
    print(f"  {len(work)} experiment(s) outstanding:")
    for d, why in work:
        print(f"     {why:14s} {d}")

    ns = nodes()
    with ThreadPoolExecutor(max_workers=len(ns)) as ex:
        res = list(ex.map(gpu_slots, ns))
    slots = [s for r, _ in res for s in r]
    down = [n for n, (r, b) in zip(ns, res) if b < 0]
    if down:
        print(f"  unreachable: {', '.join(down)}")
    if not slots:
        print("  no GPU with enough free memory.")
        return 1
    print(f"\n  {len(slots)} GPU slot(s): "
          + ", ".join(f"{n.replace('compute-node-','n')}:gpu{g}" for n, g, _ in slots))

    plan = {s: [] for s in slots}
    for i, (d, _) in enumerate(work):
        plan[slots[i % len(slots)]].append(d)

    print("\n  plan:")
    for s, exps in plan.items():
        if exps:
            print(f"     {s[0].replace('compute-node-','n')}:gpu{s[1]}  {len(exps)}: "
                  + ", ".join(e.replace('_aamsoftmax', '')[:30] for e in exps))
    if not args.run:
        print("\n  (dry run — pass --run to dispatch)")
        return 0

    if args.stop_first:
        print("\n  stopping evaluations already running ...")
        stop_existing(ns)

    print("  dispatching:")
    for s, exps in plan.items():
        if not exps:
            continue
        node, gpu, log, ok = dispatch(s, exps)
        print(f"     {node.replace('compute-node-','n')}:gpu{gpu}  {ok}  "
              f"-> {os.path.relpath(log, REPO)}")
    print("\n  watch with: python3 experiments/tools/status.py --watch 60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
