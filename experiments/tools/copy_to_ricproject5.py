#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy the training corpora to ricproject5, without starving other jobs.

    python experiments/tools/copy_to_ricproject5.py            # wait, then copy
    python experiments/tools/copy_to_ricproject5.py --now      # copy immediately
    python experiments/tools/copy_to_ricproject5.py --status

ricproject5 lives on compute-node-5 (10.222.1.125): a dedicated, otherwise idle
8 TB spindle with 6.9 TB free, reached over 10 GbE.  Its value is *contention*,
not raw speed -- both it and ricproject3 are spinning disks, so this does not
change the random-IOPS class.  What it buys is taking dataset reads off the disk
that also serves checkpoints, code and results.

Why this waits instead of throttling
------------------------------------
The source is /dev/sde on the head node, which uses the **mq-deadline**
scheduler.  ``ionice`` classes are honoured only by CFQ/BFQ, so ``ionice -c3``
is silently a no-op here -- the first attempt at this copy ran "idle-priority"
straight through the evaluations and cost them ~5x throughput (stage A fell from
98.8 to 18.4 files/s).  Sequencing is the only lever that actually works, so
this polls until no evaluation is running and copies then.

``rsync`` is incremental, so an interrupted copy resumes cheaply; ``-L``
dereferences symlinks, because musan/RIRS live on /mnt/ricproject and
slceleb2026/wav is itself a symlink -- the copy must be self-contained.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = "/mnt/ricproject5/slspv_data"
NODES = ["compute-node-1", "compute-node-2", "compute-node-4"]
CORPORA = ["slr52_sinhala", "slr127_tamil", "slceleb2026_sinhala", "musan",
           "RIRS_NOISES", "kathbath_tamil", "nisp_tamil", "slr65_tamil"]
QUIET_CHECKS = 2          # consecutive all-clear polls before starting
POLL_S = 300


def busy():
    """How many evaluation/training processes are live cluster-wide.

    Returns -1 if any node could not be reached: an unreachable node is *not*
    evidence that nothing is running, and treating it as zero is exactly how the
    first version of this gate started copying on top of a live evaluation.
    """
    total = 0
    for node in NODES:
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
                 "pgrep -cf '[e]valuate.py|experiments/scripts/' || true"],
                capture_output=True, text=True, timeout=45)
        except Exception:
            return -1
        if out.returncode != 0:
            return -1
        digits = "".join(c for c in out.stdout if c.isdigit())
        if not digits:
            return -1
        total += int(digits)
    return total


def do_copy():
    os.makedirs(DST, exist_ok=True)
    for name in CORPORA:
        src = os.path.join(REPO, "data", name)
        if not os.path.exists(src):
            print(f"--- {name}: missing, skipped", flush=True)
            continue
        print(f"--- {name} {datetime.now():%H:%M:%S}", flush=True)
        rc = subprocess.call(["rsync", "-aL", "--partial", "--info=stats2",
                              src, DST + "/"])
        if rc != 0:
            print(f"FAILED {name} (rc={rc})", flush=True)
    print(f"=== copy finished {datetime.now():%F %T} ===", flush=True)
    subprocess.call(["du", "-sh", DST])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--now", action="store_true",
                    help="copy immediately, even if jobs are running")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        n = busy()
        size = subprocess.run(["du", "-sh", DST], capture_output=True,
                              text=True).stdout.split()[0] if os.path.isdir(DST) else "0"
        run = subprocess.run(["pgrep", "-cf", "[r]sync -aL"],
                             capture_output=True, text=True).stdout.strip()
        print(f"  copied      {size}")
        print(f"  rsync live  {run if run else '0'}")
        print(f"  cluster     {'unreachable' if n < 0 else str(n) + ' job(s) running'}")
        return 0

    if not args.now:
        clear = 0
        while clear < QUIET_CHECKS:
            n = busy()
            if n == 0:
                clear += 1
                print(f"[{datetime.now():%H:%M:%S}] cluster idle "
                      f"({clear}/{QUIET_CHECKS})", flush=True)
            else:
                if clear:
                    print(f"[{datetime.now():%H:%M:%S}] jobs reappeared, resetting",
                          flush=True)
                clear = 0
            if clear < QUIET_CHECKS:
                time.sleep(POLL_S)
        print(f"=== cluster idle; starting copy {datetime.now():%F %T} ===", flush=True)
    do_copy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
