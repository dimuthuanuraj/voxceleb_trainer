#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Status and auto-resume for the whole experiment programme.

    python experiments/tools/autorun.py                  # status only
    python experiments/tools/autorun.py --resume         # restart if stopped
    python experiments/tools/autorun.py --watch 15       # check every 15 min, forever

Why this exists
---------------
The cluster rebooted three times in four days (2026-08-13 12:56, 08-14 14:54,
08-16 07:44). A reboot kills every queue driver on the head node -- nothing
survives it, not even tmux -- so each one needed a manual diagnosis and restart.
The training itself always survived, because checkpoints are written every epoch
and the recorder appends on resume, but somebody had to notice.

This is that somebody. ``--watch`` polls, and whenever it finds pending work with
no queue running, it starts one.

Design notes
------------
* **One queue, not several.** Earlier restarts partitioned work across nodes by
  hand (``--nodes compute-node-2`` etc.) to stop two queues double-booking the
  same GPU. A single queue owning every slot cannot collide with itself, and
  ``run_queue.py`` is already memory-aware, so it places the >=20 GB SSL jobs on
  the A40/A10s and the light ones on the T4s without being told.
* **Safe to run repeatedly.** If a queue is already alive it reports and exits.
  Cron-safe and idempotent.
* **Never restarts a running job.** Completion is judged by ``final.json`` with
  ``exit_code == 0``; anything else is pending, and ``--resume`` is passed so
  partial runs continue from their checkpoints rather than starting over.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.join(REPO_ROOT, "experiments", "scripts")
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
GPURUN = os.path.join(REPO_ROOT, "tools", "gpurun.sh")
LOG_DIR = os.path.join(REPO_ROOT, "experiments", "queue_logs")

STAGES = [
    ("A_*_si_s42", "A  si (slr52)", 7),
    ("A_*_ta_s42", "A  ta (slr127)", 7),
    ("A_*_si_celeb_s42", "A  si_celeb", 7),
    ("A_*_si_pooled_s42", "A  si_pooled", 7),
    ("F_*", "F  front ends", 14),
    ("E_*", "E  english", 8),
    ("H_*", "H  hybrids", 4),
]
STALL_MIN = 25          # no epoch written for this long => not actually running


def _is_complete(exp_id: str) -> bool:
    p = os.path.join(RESULTS_DIR, exp_id, "final.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("exit_code") == 0
    except Exception:
        return False


def _scan(pattern: str):
    """-> (complete, [(exp_id, epochs, idle_minutes)])"""
    done, partial = [], []
    now = time.time()
    for d in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern) + "/")):
        exp_id = os.path.basename(d.rstrip("/"))
        if ".superseded-" in exp_id or exp_id.endswith(("__smoke", "__dev")):
            continue
        epochs_path = os.path.join(d, "epochs.jsonl")
        n_ep = 0
        if os.path.isfile(epochs_path):
            with open(epochs_path, encoding="utf-8") as fh:
                n_ep = sum(1 for _ in fh)
        if _is_complete(exp_id):
            done.append(exp_id)
        elif n_ep:
            partial.append((exp_id, n_ep, (now - os.path.getmtime(epochs_path)) / 60))
    return done, partial


def pending_experiments():
    """Every generated experiment that has not completed successfully."""
    out = []
    for p in sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.py"))):
        exp_id = os.path.basename(p)[:-3]
        if not _is_complete(exp_id):
            out.append(exp_id)
    return out


def queue_running():
    try:
        r = subprocess.run(["pgrep", "-af", "run_queue.py"],
                           capture_output=True, text=True, timeout=20)
        return [l for l in r.stdout.splitlines() if "pgrep" not in l]
    except Exception:
        return []


def live_experiments():
    """Experiment ids with a real process on some node.

    File mtime alone is NOT a liveness test: a resumed run can go ~10 minutes
    before it writes its first epoch, so a freshly restarted SSL job looks
    hours-idle. A status tool that reports those as STALLED trains you to ignore
    it, so liveness is taken from the processes themselves.
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
        nodes = []
    for node in nodes:
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node,
                 "pgrep -af 'experiments/scripts/|trainSpeakerNet.py' || true"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        for line in out.splitlines():
            m = re.search(r"experiments/scripts/([A-Za-z0-9_]+)\.py", line)
            if not m:
                m = re.search(r"--save_path\s+\S*/exps/([A-Za-z0-9_]+)", line)
            if m:
                live.add(m.group(1))
    return live


def node_health():
    """-> [(node, gpu, free_mb)] plus the set of nodes that answered."""
    slots, alive = [], set()
    try:
        out = subprocess.run([GPURUN, "--status"], capture_output=True,
                             text=True, timeout=180).stdout
    except Exception:
        return slots, alive
    node = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            node = line.strip("= ").strip()
            continue
        parts = [p.strip() for p in line.split(",")]
        if node and len(parts) >= 4 and parts[0].isdigit():
            try:
                slots.append((node, int(parts[0]), int(parts[2].split()[0])))
                alive.add(node)
            except (ValueError, IndexError):
                pass
    return slots, alive


def print_status():
    print(f"\n{'=' * 78}\n  {datetime.now():%Y-%m-%d %H:%M:%S}   experiment programme status\n{'=' * 78}")

    live = live_experiments()
    total_done = total = 0
    stalled = []
    for pattern, label, expected in STAGES:
        done, partial = _scan(pattern)
        total_done += len(done)
        total += expected
        bar = "#" * int(18 * len(done) / expected) + "." * (18 - int(18 * len(done) / expected))
        print(f"  {label:18s} [{bar}] {len(done):>2}/{expected}")
        for exp_id, n_ep, idle in partial:
            # A run counts as alive if it has a process, regardless of how long
            # since its last epoch row.
            if exp_id in live:
                state = "running"
            elif idle < STALL_MIN:
                state = "starting"
            else:
                state = "STALLED"
                stalled.append(exp_id)
            print(f"       {state:7s} {exp_id.replace('_aamsoftmax', ''):48s} "
                  f"{n_ep:>3d} ep, {idle:>4.0f} min idle")

    n_eval = len(glob.glob(os.path.join(RESULTS_DIR, "*", "test_eval.json")))
    print(f"\n  trained {total_done}/{total}   ·   evaluated {n_eval}")

    slots, alive = node_health()
    expected_nodes = {"compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4"}
    print(f"\n  GPUs available: {len(slots)}")
    for n, g, mb in slots:
        print(f"       {n}:gpu{g}  {mb:,} MiB free")
    for n in sorted(expected_nodes - alive):
        print(f"       {n}  DOWN")

    q = queue_running()
    print(f"\n  queue drivers: {len(q)}" + ("" if q else "   <-- nothing scheduling work"))

    pend = pending_experiments()
    print(f"  pending experiments: {len(pend)}")
    if pend and not q:
        print("\n  ==> work is pending and no queue is running. "
              "Re-run with --resume to start one.")
    if stalled:
        print(f"  ==> {len(stalled)} run(s) idle >{STALL_MIN} min with no queue attached.")
    return pend, q


def do_resume(dry_run=False, max_retries=3):
    pend = pending_experiments()
    if not pend:
        print("  nothing pending — the programme is complete.")
        return 0
    running = queue_running()
    if running:
        print(f"  a queue is already running ({len(running)}); leaving it alone.")
        print("  (use --force to start another anyway — risks double-booking GPUs)")
        return 0
    slots, _ = node_health()
    if not slots:
        print("  no GPU slots available; nothing to do.")
        return 1

    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"autorun-{datetime.now():%Y%m%d-%H%M%S}.log")
    # ONE queue over ALL nodes: it owns every slot, so it cannot collide with
    # itself, and run_queue places jobs by their GPU-memory requirement.
    cmd = [sys.executable, os.path.join(REPO_ROOT, "experiments", "tools", "run_queue.py"),
           "--resume", "--max-retries", str(max_retries), "--only"] + pend
    print(f"  starting one queue for {len(pend)} pending experiment(s)")
    print(f"  log: {os.path.relpath(log, REPO_ROOT)}")
    if dry_run:
        print("  [dry-run] " + " ".join(cmd[:6]) + f" ... ({len(pend)} ids)")
        return 0
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    time.sleep(20)
    with open(log, encoding="utf-8") as fh:
        for line in fh.read().splitlines()[:6]:
            print(f"    {line[:104]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resume", action="store_true",
                   help="start a queue if work is pending and none is running")
    ap.add_argument("--watch", type=int, metavar="MIN",
                   help="poll every MIN minutes and resume automatically")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                   help="start a queue even if one is already running")
    ap.add_argument("--quiet", action="store_true", help="only print when acting")
    args = ap.parse_args()

    if args.watch:
        print(f"watching every {args.watch} min — Ctrl+C to stop")
        while True:
            try:
                pend, q = (print_status() if not args.quiet
                           else (pending_experiments(), queue_running()))
                if pend and (not q or args.force):
                    print(f"\n[{datetime.now():%H:%M:%S}] resuming")
                    do_resume(args.dry_run)
                elif not pend:
                    print(f"[{datetime.now():%H:%M:%S}] programme complete; stopping watch")
                    return 0
            except Exception as exc:      # a transient ssh failure must not end the watch
                print(f"[{datetime.now():%H:%M:%S}] check failed: {exc!r}")
            time.sleep(args.watch * 60)

    print_status()
    if args.resume:
        print()
        return do_resume(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
