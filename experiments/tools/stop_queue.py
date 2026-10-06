#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find and stop experiment jobs left running on the cluster.

    python experiments/tools/stop_queue.py                 # list what is running
    python experiments/tools/stop_queue.py --kill           # stop all of them
    python experiments/tools/stop_queue.py --kill --exp A_ecapa1024_aamsoftmax_si_s42
    python experiments/tools/stop_queue.py --kill --node compute-node-1

Why this is needed
------------------
``run_queue.py`` dispatches over plain ``ssh`` without a TTY. When the queue is
interrupted, what happens to the remote training processes is **not uniform**:

* the local ``ssh`` client dies immediately;
* on some nodes SIGINT reaches the remote process and it exits (the queue then
  records ``rc=255``);
* on others the remote ``python`` is simply **orphaned** and keeps training,
  holding its GPU, with nothing left to report its result.

An orphaned run is the bad case: it consumes a GPU the scheduler now believes is
free, and its ``final.json`` will never be written, so ``run_queue.py`` will
happily start the *same* experiment again on top of it — two processes writing
the same ``save_path``.

This tool finds those processes by matching the command line against the
experiment scripts and ``trainSpeakerNet.py``, and stops them cleanly (SIGTERM,
then SIGKILL for anything that ignores it).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPURUN = os.path.join(REPO_ROOT, "tools", "gpurun.sh")

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def candidate_nodes():
    """Read the node list from gpurun.sh, so there is one source of truth."""
    try:
        with open(GPURUN, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("CANDIDATE_NODES="):
                    inner = line.split("=", 1)[1].strip().strip("()")
                    return [n for n in inner.split() if n and not n.startswith("#")]
    except Exception:
        pass
    return ["compute-node-1", "compute-node-3", "compute-node-4"]


def scan(node):
    """Return [(pid, exp_id, cmd)] for experiment processes running on `node`."""
    cmd = [
        "ssh", *SSH_OPTS, node,
        # -f matches the full command line; the two patterns cover the harness
        # wrapper and the trainer it spawns (plus its dataloader workers).
        "pgrep -af 'experiments/scripts/|trainSpeakerNet.py' || true",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
    except Exception as exc:
        print(f"[{node}] unreachable: {exc!r}", file=sys.stderr)
        return []

    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "pgrep" in line:
            continue
        pid, _, rest = line.partition(" ")
        if not pid.isdigit():
            continue
        # Recover the experiment id from either the script path or --save_path.
        m = re.search(r"experiments/scripts/([A-Za-z0-9_]+)\.py", rest)
        if not m:
            m = re.search(r"--save_path\s+\S*/exps/([A-Za-z0-9_]+)", rest)
        exp_id = m.group(1) if m else "?"
        found.append((int(pid), exp_id, rest))
    return found


def kill(node, pids, signal="TERM"):
    if not pids:
        return
    subprocess.run(
        ["ssh", *SSH_OPTS, node, f"kill -{signal} {' '.join(map(str, pids))} 2>/dev/null || true"],
        capture_output=True, text=True, timeout=40,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kill", action="store_true",
                    help="actually stop them (default is to list only)")
    ap.add_argument("--exp", default=None, help="restrict to one experiment id")
    ap.add_argument("--node", default=None, help="restrict to one node")
    ap.add_argument("--force", action="store_true",
                    help="skip SIGTERM and go straight to SIGKILL")
    args = ap.parse_args()

    nodes = [args.node] if args.node else candidate_nodes()
    total = {}
    for node in nodes:
        procs = scan(node)
        if args.exp:
            procs = [p for p in procs if p[1] == args.exp]
        if procs:
            total[node] = procs

    if not total:
        print("no experiment processes running on " + ", ".join(nodes))
        return 0

    for node, procs in total.items():
        print(f"\n=== {node} ===")
        by_exp = {}
        for pid, exp_id, _ in procs:
            by_exp.setdefault(exp_id, []).append(pid)
        for exp_id, pids in sorted(by_exp.items()):
            print(f"  {exp_id:46s} {len(pids)} process(es): "
                  f"{', '.join(map(str, pids[:6]))}"
                  f"{' ...' if len(pids) > 6 else ''}")

    n_proc = sum(len(v) for v in total.values())
    if not args.kill:
        print(f"\n{n_proc} process(es) across {len(total)} node(s). "
              f"Re-run with --kill to stop them.")
        return 0

    # Mark these as deliberately stopped BEFORE killing them. Killing through
    # ssh returns rc=255, which run_queue's retry heuristic reads as "node
    # unreachable, transient" and requeues -- on 2026-08-16 that silently
    # started a second copy of en_full on another node, two processes writing
    # one save_path. The marker lets an intentional stop be told apart from an
    # infrastructure failure, which the exit code alone cannot express.
    import time as _t
    for _n, _procs in total.items():
        for _pid, _exp, _ in _procs:
            if _exp and _exp != "?":
                d = os.path.join(REPO_ROOT, "experiments", "results", _exp)
                if os.path.isdir(d):
                    with open(os.path.join(d, "STOPPED_BY_USER"), "w") as fh:
                        fh.write(f"{_t.strftime('%Y-%m-%dT%H:%M:%S')} stop_queue.py --kill\n")

    for node, procs in total.items():
        pids = [p[0] for p in procs]
        if args.force:
            kill(node, pids, "KILL")
            print(f"[{node}] SIGKILL sent to {len(pids)} process(es)")
            continue
        kill(node, pids, "TERM")
        print(f"[{node}] SIGTERM sent to {len(pids)} process(es)")

    if not args.force:
        # Torch dataloader workers occasionally ignore SIGTERM; sweep once more.
        import time

        time.sleep(6)
        for node in list(total):
            remaining = scan(node)
            if args.exp:
                remaining = [p for p in remaining if p[1] == args.exp]
            if remaining:
                kill(node, [p[0] for p in remaining], "KILL")
                print(f"[{node}] SIGKILL sent to {len(remaining)} survivor(s)")

    print("\nNote: an interrupted run leaves no final.json, so run_queue.py will "
          "treat it as incomplete and start it again from scratch. Its partial "
          "checkpoints under exps/<exp_id>/model/ are still there and the "
          "trainer will resume from the highest-numbered one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
