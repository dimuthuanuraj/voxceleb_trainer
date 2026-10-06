#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-screen status of the experiment programme.

    python experiments/tools/status.py              # snapshot
    python experiments/tools/status.py --watch 60   # refresh every 60 s
    python experiments/tools/status.py --quiet      # only what is running

Shows, in one place:

* training coverage per stage, and evaluation coverage split by whether the
  result came from the corrected evaluator
* every live experiment process: which node and GPU it is on, how long it has
  been running, and that GPU's utilisation and memory
* per-epoch timing and a remaining-time estimate for anything still training
* progress and rate of each evaluation shard
* the ricproject5 corpus copy

Why the GPU column matters
--------------------------
A job can hold a GPU and do nothing.  Both H hybrid runs sat in a dataloader
deadlock for five days with their GPUs at 0 %, and nothing in a progress bar
said so -- the processes were alive and the epoch counter simply never moved.
Utilisation next to elapsed time makes that visible at a glance: a run that has
been up for hours at 0 % is stuck, not slow.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "experiments", "results")
SCRIPTS = os.path.join(REPO, "experiments", "scripts")
QLOGS = os.path.join(REPO, "experiments", "queue_logs")
GPURUN = os.path.join(REPO, "tools", "gpurun.sh")
COPY_DST = "/mnt/ricproject5/slspv_data"
COPY_TARGET_GB = 70.0

STAGES = [("A", 28), ("F", 14), ("E", 8), ("H", 4)]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]

C = {"dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "y": "\033[33m",
     "r": "\033[31m", "c": "\033[36m", "0": "\033[0m"}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def hms(seconds):
    """Elapsed seconds -> compact human form."""
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def nodes():
    """Node list from gpurun.sh, so there is one source of truth."""
    try:
        with open(GPURUN, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().startswith("CANDIDATE_NODES="):
                    inner = line.split("=", 1)[1].strip().strip("()")
                    return [n for n in inner.split() if n and not n.startswith("#")]
    except Exception:
        pass
    return ["compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4"]


# --------------------------------------------------------------------------
PROBE = r"""
echo '###PROCS'
ps -eo pid,etimes,pcpu,args --no-headers 2>/dev/null | grep -E 'experiments/(scripts|tools)/' | grep -v grep
echo '###GPUS'
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo '###APPS'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader 2>/dev/null
"""


def probe(node):
    """-> dict(node=..., procs=[...], gpus=[...], apps=[...]) or None if down."""
    try:
        out = subprocess.run(SSH + [node, PROBE], capture_output=True,
                             text=True, timeout=45).stdout
    except Exception:
        return None
    if not out.strip():
        return None
    sec, cur = {"PROCS": [], "GPUS": [], "APPS": []}, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("###"):
            cur = s[3:]
            continue
        if cur and s:
            sec[cur].append(s)

    procs = []
    for line in sec["PROCS"]:
        parts = line.split(None, 3)
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        pid, et, cpu, args = int(parts[0]), int(parts[1]), parts[2], parts[3]
        m = (re.search(r"experiments/scripts/([A-Za-z0-9_]+)\.py", args)
             or re.search(r"--save_path\s+\S*/exps/([A-Za-z0-9_]+)", args))
        if m:
            kind, name = "train", m.group(1)
        elif "evaluate.py" in args:
            st = re.search(r"--stage\s+([A-Z])", args)
            ex = re.search(r"--exp\s+([A-Za-z0-9_]+)", args)
            kind = "eval"
            name = f"stage {st.group(1)}" if st else (ex.group(1) if ex else "eval --all")
        else:
            continue
        procs.append({"pid": pid, "elapsed": et, "cpu": cpu, "kind": kind, "name": name})

    gpus, uuid2idx = [], {}
    for line in sec["GPUS"]:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and p[0].isdigit():
            idx = int(p[0])
            uuid2idx[p[1]] = idx
            gpus.append({"idx": idx,
                         "util": int(re.sub(r"\D", "", p[2]) or 0),
                         "used": int(re.sub(r"\D", "", p[3]) or 0),
                         "total": int(re.sub(r"\D", "", p[4]) or 0)})

    apps = []
    for line in sec["APPS"]:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[1].isdigit():
            apps.append({"gpu": uuid2idx.get(p[0]), "pid": int(p[1]),
                         "mem": int(re.sub(r"\D", "", p[2]) or 0)})
    return {"node": node, "procs": procs, "gpus": gpus, "apps": apps}


# --------------------------------------------------------------------------
def real_experiments():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        e = os.path.basename(p)
        if "__smoke" in e or ".superseded-" in e or not os.path.isdir(p):
            continue
        f = os.path.join(p, "final.json")
        if not os.path.isfile(f):
            continue
        try:
            if json.load(open(f, encoding="utf-8")).get("exit_code") == 0:
                out.append(e)
        except Exception:
            pass
    return out


def eval_state(exp):
    """-> (has_result, is_clean, evaluated_utc)"""
    p = os.path.join(RESULTS, exp, "test_eval.json")
    if not os.path.isfile(p):
        return False, False, None
    try:
        j = json.load(open(p, encoding="utf-8"))
    except Exception:
        return True, False, None
    skipped = (j.get("checkpoint_load") or {}).get("backbone_skipped") or []
    return True, not skipped, j.get("evaluated_utc")


def epoch_info(exp):
    """-> (last_epoch, max_epoch, median_min_per_epoch, best_val_eer)"""
    p = os.path.join(RESULTS, exp, "epochs.jsonl")
    if not os.path.isfile(p):
        return 0, None, None, None
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        return 0, None, None, None
    mx = None
    try:
        mx = json.load(open(os.path.join(RESULTS, exp, "manifest.json"),
                            encoding="utf-8"))["resolved_parameters"].get("max_epoch")
    except Exception:
        pass
    last = max((r.get("epoch", 0) for r in rows if isinstance(r.get("epoch"), int)), default=0)
    durs = sorted(r["epoch_duration_s"] / 60 for r in rows if r.get("epoch_duration_s"))
    med = durs[len(durs) // 2] if durs else None
    vals = [r["val_eer"] for r in rows if r.get("val_eer") is not None]
    return last, mx, med, (min(vals) if vals else None)


def shard_progress():
    """Evaluation shard logs -> (stage, written, last_line, mtime_age_s)."""
    out = []
    for path in sorted(glob.glob(os.path.join(QLOGS, "eval-redo-*.log"))):
        stage = os.path.basename(path).replace("eval-redo-", "").replace(".log", "")
        try:
            data = open(path, "rb").read()
        except Exception:
            continue
        wrote = data.count(b"] wrote ")
        tail = data[-4000:].replace(b"\r", b"\n").decode("utf-8", "replace").strip().splitlines()
        last = tail[-1][:58] if tail else ""
        age = time.time() - os.path.getmtime(path)
        out.append((stage, wrote, last, age))
    return out


def copy_status():
    if not os.path.isdir(COPY_DST):
        return None
    try:
        n = subprocess.run(["du", "-sb", COPY_DST], capture_output=True,
                           text=True, timeout=120).stdout.split()[0]
        gb = int(n) / 1073741824
    except Exception:
        return None
    running = subprocess.run(["pgrep", "-cf", "[r]sync -aL"],
                             capture_output=True, text=True).stdout.strip()
    return gb, (running not in ("", "0"))


# --------------------------------------------------------------------------
def render():
    print(f"\n{C['b']}{'=' * 76}\n  {datetime.now():%Y-%m-%d %H:%M:%S}   SL_SPV experiment status\n{'=' * 76}{C['0']}")

    exps = real_experiments()
    total_scripts = len(glob.glob(os.path.join(SCRIPTS, "*.py")))

    # ---- training + evaluation coverage -------------------------------
    print(f"\n{C['b']}TRAINING{C['0']}")
    for st, expected in STAGES:
        done = [e for e in exps if e.startswith(st + "_")]
        bar = "#" * int(18 * min(len(done), expected) / expected)
        print(f"  stage {st:2s} [{bar:<18s}] {len(done):>2}/{expected}")
    print(f"  {C['g'] if len(exps) >= total_scripts else C['y']}"
          f"{len(exps)}/{total_scripts} complete{C['0']}")

    have = clean = 0
    dirty, missing = [], []
    for e in exps:
        ok, cl, _ = eval_state(e)
        have += ok
        clean += (ok and cl)
        if ok and not cl:
            dirty.append(e)
        if not ok:
            missing.append(e)
    print(f"\n{C['b']}EVALUATION{C['0']}")
    print(f"  scored          {have}/{len(exps)}")
    print(f"  clean rebuild   {C['g'] if clean == have else C['y']}{clean}/{len(exps)}{C['0']}")
    if dirty:
        print(f"  {C['r']}contaminated    {len(dirty)}{C['0']} (backbone tensors did not load)")
        for e in dirty[:4]:
            print(f"       {e}")
    if missing:
        print(f"  not yet scored  {len(missing)}")
        for e in missing[:4]:
            print(f"       {e}")
    hist = len(glob.glob(os.path.join(RESULTS, "*", "eval_history", "*.json")))
    if hist:
        print(f"  {C['dim']}previous results kept in eval_history/: {hist}{C['0']}")

    # ---- live processes ------------------------------------------------
    ns = nodes()
    with ThreadPoolExecutor(max_workers=len(ns)) as ex:
        info = list(ex.map(probe, ns))

    print(f"\n{C['b']}RUNNING NOW{C['0']}")
    any_live = False
    for res, node in zip(info, ns):
        if res is None:
            print(f"  {C['r']}{node:16s} DOWN{C['0']}")
            continue
        pid2gpu = {a["pid"]: a for a in res["apps"]}
        gpu_by_idx = {g["idx"]: g for g in res["gpus"]}
        # One logical job shows up as several processes: the launcher, the
        # trainer/evaluator, and its dataloader workers. Collapse them to the
        # one holding the GPU, so the display has a row per job rather than a
        # row per pid -- and keep the longest-lived pid, which is the launcher's
        # start time and therefore the job's true elapsed time.
        by_job = {}
        for p in res["procs"]:
            key = (p["kind"], p["name"])
            prev = by_job.get(key)
            if prev is None:
                by_job[key] = p
                continue
            p_gpu, prev_gpu = p["pid"] in pid2gpu, prev["pid"] in pid2gpu
            if p_gpu and not prev_gpu:
                p["elapsed"] = max(p["elapsed"], prev["elapsed"])
                by_job[key] = p
            else:
                prev["elapsed"] = max(prev["elapsed"], p["elapsed"])
        for p in by_job.values():
            if p["kind"] == "train" and p["name"].endswith(".py"):
                continue
            app = pid2gpu.get(p["pid"])
            g = gpu_by_idx.get(app["gpu"]) if app else None
            gpu_txt = "  cpu-only"
            if g is not None:
                col = C['r'] if g["util"] == 0 else (C['y'] if g["util"] < 30 else C['g'])
                gpu_txt = (f"gpu{g['idx']} {col}{g['util']:>3d}%{C['0']} "
                           f"{app['mem']:>6,}MiB")
            warn = ""
            if g is not None and g["util"] == 0 and p["elapsed"] > 1800:
                warn = f"  {C['r']}<-- 0% for {hms(p['elapsed'])}, likely stuck{C['0']}"
            print(f"  {node.replace('compute-node-', 'n'):4s} {gpu_txt}  "
                  f"{hms(p['elapsed']):>7s}  {p['kind']:5s} {p['name'][:38]}{warn}")
            any_live = True
    if not any_live:
        print(f"  {C['dim']}nothing running{C['0']}")

    # ---- idle GPUs ------------------------------------------------------
    free = []
    for res in info:
        if not res:
            continue
        busy = {a["gpu"] for a in res["apps"]}
        for g in res["gpus"]:
            if g["idx"] not in busy:
                free.append(f"{res['node'].replace('compute-node-', 'n')}:gpu{g['idx']}"
                            f" ({g['total'] - g['used']:,}MiB)")
    if free:
        print(f"  {C['dim']}idle GPUs: {', '.join(free)}{C['0']}")

    # ---- anything still training ---------------------------------------
    training = [e for e in glob.glob(os.path.join(SCRIPTS, "*.py"))
                if os.path.basename(e)[:-3] not in exps]
    if training:
        print(f"\n{C['b']}TRAINING PROGRESS / ETA{C['0']}")
        for path in training:
            e = os.path.basename(path)[:-3]
            last, mx, med, best = epoch_info(e)
            eta = f"{(mx - last) * med / 60:.1f} h" if (mx and med) else "?"
            bv = f"best val EER {best:.3f}%" if best is not None else "no val yet"
            print(f"  {e[:44]:46s} ep {last}/{mx}  {med:.0f} min/ep  ETA {eta}  {bv}"
                  if med else f"  {e[:44]:46s} ep {last}/{mx}  {bv}")

    # ---- evaluation shards ---------------------------------------------
    shards = shard_progress()
    if shards:
        print(f"\n{C['b']}EVALUATION SHARDS{C['0']}")
        for stage, wrote, last, age in shards:
            stale = f"  {C['y']}(log idle {hms(age)}){C['0']}" if age > 900 else ""
            print(f"  stage {stage:2s} {wrote:>3d} written   {C['dim']}{last}{C['0']}{stale}")

    # ---- corpus copy -----------------------------------------------------
    cp = copy_status()
    if cp:
        gb, running = cp
        state = f"{C['g']}copying{C['0']}" if running else f"{C['y']}paused{C['0']}"
        print(f"\n{C['b']}RICPROJECT5 COPY{C['0']}")
        print(f"  {gb:.1f} GB of ~{COPY_TARGET_GB:.0f} GB  ({gb / COPY_TARGET_GB * 100:.0f}%)  {state}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--watch", type=int, metavar="SEC",
                    help="refresh every SEC seconds until interrupted")
    ap.add_argument("--quiet", action="store_true", help="reserved")
    args = ap.parse_args()
    if args.watch:
        try:
            while True:
                os.system("clear")
                render()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    render()
    return 0


if __name__ == "__main__":
    sys.exit(main())
