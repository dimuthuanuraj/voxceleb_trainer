#!/usr/bin/env python3
"""One-task-at-a-time driver for the Final Research Works plan.

Why this exists
---------------
On 2026-08-24 a dispatcher launched fourteen jobs and died with the cluster.
Nineteen days later five arms were still sitting at 45-100 % complete with
nobody aware. This runner is the opposite design on purpose:

  * it runs **one** task at a time and returns;
  * it refuses a task whose dependencies have not passed their **exit test**;
  * the exit test is evaluated **from the filesystem**, not from a status flag a
    crashed process was supposed to write -- the same principle as each strand's
    ``scaffold.py --check``, which "prints this from the filesystem, so it cannot
    drift from what is written here".

So a crash costs the running task and nothing else, and the answer to "what
happened while I was away" is always ``runner.py --status``.

Usage
-----
    python3 runner.py --status        # ledger: passed / ready / blocked
    python3 runner.py --next          # name the next ready task, do not run it
    python3 runner.py --run W0.2      # run one task
    python3 runner.py --run-next      # run the next ready task
    python3 runner.py --graph         # dependency graph and critical path
    python3 runner.py --run B1 --force    # override a blocked dependency (logged)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(HERE)
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
STATE = os.path.join(FRW, "02_STATE")
LEDGER = os.path.join(STATE, "ledger.json")
TASKS = os.path.join(HERE, "tasks.json")

C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "b": "\033[34m",
     "d": "\033[2m", "B": "\033[1m", "0": "\033[0m"}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_tasks() -> dict:
    with open(TASKS, encoding="utf-8") as fh:
        return json.load(fh)


def load_ledger() -> dict:
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _ledger_lock():
    """Advisory lock around ledger read-modify-write.

    The autopilot runs several tasks concurrently, each shelling out to this
    module, so two processes can update different keys of the same file at the
    same time. Without the lock the later write silently drops the earlier one
    and a completed task looks unstarted.
    """
    import fcntl
    os.makedirs(STATE, exist_ok=True)
    fh = open(os.path.join(STATE, ".ledger.lock"), "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def update_ledger(tid: str, fields: dict) -> None:
    """Merge `fields` into one task's ledger entry, under the lock."""
    lk = _ledger_lock()
    try:
        led = load_ledger()
        led.setdefault(tid, {}).update(fields)
        with open(LEDGER, "w", encoding="utf-8") as fh:
            json.dump(led, fh, indent=2, sort_keys=True)
    finally:
        lk.close()


def save_ledger(led: dict) -> None:
    lk = _ledger_lock()
    try:
        with open(LEDGER, "w", encoding="utf-8") as fh:
            json.dump(led, fh, indent=2, sort_keys=True)
    finally:
        lk.close()


# --------------------------------------------------------------- exit tests
#
# Each returns (passed: bool, detail: str). They read the filesystem only --
# never the ledger -- so a task marked done by hand still shows as incomplete
# unless its artefact is really there.

def _resolve(root: str | None, path: str) -> str:
    base = SL_SPV if root == "SL_SPV" else FRW
    return path if os.path.isabs(path) else os.path.join(base, path)


def check_exit(test: dict) -> tuple[bool, str]:
    kind = test.get("type")

    if kind == "json_true":
        p = _resolve(test.get("root"), test["path"])
        if not os.path.exists(p):
            return False, f"missing {os.path.relpath(p, FRW)}"
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            return False, f"unreadable: {exc}"
        key = test.get("key", "ok")
        return bool(d.get(key)), f"{os.path.basename(p)}[{key}]={d.get(key)!r}"

    if kind == "file_glob":
        p = _resolve(test.get("root"), test["path"])
        hits = glob.glob(p)
        return bool(hits), (f"{len(hits)} match" if hits else
                            f"no match for {os.path.relpath(p, SL_SPV if test.get('root') else FRW)}")

    if kind == "all_globs":
        missing = []
        for rel in test["paths"]:
            if not glob.glob(_resolve(test.get("root"), rel)):
                missing.append(os.path.basename(rel))
        if missing:
            return False, f"{len(missing)} missing: {', '.join(missing[:3])}" + (" ..." if len(missing) > 3 else "")
        return True, f"all {len(test['paths'])} present"

    if kind == "gpu_count":
        n, detail, largest = probe_gpu_count()
        need_n = int(test.get("min", 1))
        need_mb = int(test.get("min_free_mb", 0))
        ok = n >= need_n and largest >= need_mb
        if need_mb and largest < need_mb:
            detail += f"  [largest {largest} MiB < {need_mb} MiB required]"
        return ok, detail

    return False, f"unknown exit test {kind!r}"


def probe_gpu_count() -> tuple[int, str, int]:
    """-> (usable GPU count, per-node summary, largest free MiB on any one card).

    The largest-single-card figure matters as much as the count: the arbiter run
    needs 30 GB, so four recovered 15 GB T4s leave it exactly as blocked as two
    did. Reporting only the count would make a blocked task look ready.
    """
    nodes = ["compute-node-1", "compute-node-2", "compute-node-3", "compute-node-4"]
    total, per, largest = 0, [], 0
    for node in nodes:
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node,
                 "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=25)
            lines = [l for l in out.stdout.strip().splitlines()
                     if l.strip() and l.split(",")[0].strip().isdigit()]
            if lines:
                total += len(lines)
                free = [int(l.split(",")[1]) for l in lines]
                largest = max(largest, max(free))
                per.append(f"{node.replace('compute-node-', 'n')}:{len(lines)}gpu/{max(free)}MiB")
            else:
                per.append(f"{node.replace('compute-node-', 'n')}:DEAD")
        except Exception:
            per.append(f"{node.replace('compute-node-', 'n')}:UNREACHABLE")
    return total, " ".join(per), largest


# --------------------------------------------------------------- status

def compute_status(data: dict) -> dict:
    """-> {id: {task, passed, detail, ready, blocked_by}}"""
    tasks = {t["id"]: t for t in data["tasks"]}
    out: dict[str, dict] = {}
    for tid, t in tasks.items():
        passed, detail = check_exit(t["exit_test"])
        out[tid] = {"task": t, "passed": passed, "detail": detail}
    for tid, t in tasks.items():
        blocked = [d for d in t["deps"] if not out.get(d, {}).get("passed")]
        out[tid]["blocked_by"] = blocked
        out[tid]["ready"] = (not blocked) and (not out[tid]["passed"])
    return out


def cmd_status(data: dict) -> int:
    st = compute_status(data)
    led = load_ledger()
    waves: dict[str, list] = {}
    for tid, s in st.items():
        waves.setdefault(s["task"]["wave"], []).append((tid, s))

    print(f"\n{C['B']}Final Research Works — status{C['0']}  {now()}")
    n_gpu, gpu_detail, largest_mb = probe_gpu_count()
    colour = C["g"] if n_gpu >= 3 else (C["y"] if n_gpu else C["r"])
    print(f"  cluster: {colour}{n_gpu} GPU(s) usable{C['0']}  {C['d']}{gpu_detail}{C['0']}")
    big = C["g"] if largest_mb >= 30000 else C["r"]
    print(f"  largest single card: {big}{largest_mb} MiB{C['0']}"
          f"{'' if largest_mb >= 30000 else C['d'] + '  (arbiter B1/B2 needs 30000)' + C['0']}")

    done = ready = blocked = 0
    for wave in sorted(waves):
        print(f"\n  {C['B']}Wave {wave}{C['0']}")
        for tid, s in sorted(waves[wave], key=lambda x: x[0]):
            t = s["task"]
            if s["passed"]:
                mark, col = "PASS", C["g"]; done += 1
            elif s["ready"]:
                mark, col = "READY", C["y"]; ready += 1
            else:
                mark, col = "BLOCK", C["d"]; blocked += 1
            note = led.get(tid, {}).get("status", "")
            extra = f" {C['d']}({note}){C['0']}" if note and not s["passed"] else ""
            print(f"    {col}{mark:5s}{C['0']} {tid:6s} {t['title'][:62]:62s} {C['d']}{t['cost']:>14s}{C['0']}{extra}")
            if s["blocked_by"]:
                print(f"          {C['d']}waits on: {', '.join(s['blocked_by'])}{C['0']}")
            elif not s["passed"]:
                print(f"          {C['d']}{s['detail']}{C['0']}")

    print(f"\n  {C['g']}{done} passed{C['0']}  {C['y']}{ready} ready{C['0']}  {C['d']}{blocked} blocked{C['0']}"
          f"  of {len(st)} tasks")

    mt = data.get("manual_tracking", [])
    if mt:
        print(f"\n  {C['B']}Tracked, not scriptable{C['0']}")
        for m in mt:
            print(f"    {C['d']}{m['id']:4s} {m['title'][:70]:70s} [{m['owner']}]{C['0']}")
    print()
    return 0


def next_ready(data: dict):
    st = compute_status(data)
    order = [t["id"] for t in data["tasks"]]
    for tid in order:
        if st[tid]["ready"]:
            return tid, st[tid]
    return None, None


def cmd_next(data: dict) -> int:
    tid, s = next_ready(data)
    if tid is None:
        print("nothing ready — everything is either passed or blocked. See --status.")
        return 1
    t = s["task"]
    print(f"\n  next ready: {C['B']}{tid}{C['0']} — {t['title']}")
    print(f"  script    : {t['script']}")
    print(f"  cost      : {t['cost']}   gpu_mb: {t['gpu_mb']}")
    print(f"  source    : {t['source']}")
    if t.get("manual"):
        print(f"  {C['y']}MANUAL — this task needs a person, not a GPU{C['0']}")
    if t.get("long"):
        print(f"  {C['y']}LONG — runs for hours/days; resumable from checkpoints{C['0']}")
    print(f"\n  run it with: python3 runner.py --run {tid}\n")
    return 0


def cmd_graph(data: dict) -> int:
    st = compute_status(data)
    print(f"\n{C['B']}Dependency graph{C['0']}\n")
    for t in data["tasks"]:
        tid = t["id"]
        mark = f"{C['g']}o{C['0']}" if st[tid]["passed"] else (
            f"{C['y']}>{C['0']}" if st[tid]["ready"] else f"{C['d']}.{C['0']}")
        deps = " <- " + ", ".join(t["deps"]) if t["deps"] else ""
        print(f"  {mark} {tid:6s} {t['title'][:56]:56s}{C['d']}{deps}{C['0']}")

    # longest dependency chain by task count
    tasks = {t["id"]: t for t in data["tasks"]}
    memo: dict[str, list] = {}

    def chain(tid: str) -> list:
        if tid in memo:
            return memo[tid]
        best: list = []
        for d in tasks[tid]["deps"]:
            c = chain(d)
            if len(c) > len(best):
                best = c
        memo[tid] = best + [tid]
        return memo[tid]

    longest = max((chain(t) for t in tasks), key=len)
    print(f"\n  {C['B']}Longest chain{C['0']} ({len(longest)} tasks): {' -> '.join(longest)}")
    gpu_blocked = [t["id"] for t in data["tasks"]
                   if t["gpu_mb"] > 15000 and not st[t["id"]]["passed"]]
    if gpu_blocked:
        print(f"  {C['r']}Needs >15 GB VRAM (A40 only):{C['0']} {', '.join(gpu_blocked)}")
    print()
    return 0


def cmd_run(data: dict, tid: str, force: bool) -> int:
    tasks = {t["id"]: t for t in data["tasks"]}
    if tid not in tasks:
        print(f"unknown task {tid!r}. Known: {', '.join(tasks)}")
        return 2
    t = tasks[tid]
    st = compute_status(data)

    if st[tid]["passed"] and not force:
        print(f"{C['g']}{tid} already passes its exit test{C['0']} — {st[tid]['detail']}")
        print("  use --force to run it again")
        return 0

    blocked = st[tid]["blocked_by"]
    if blocked and not force:
        print(f"{C['r']}{tid} is blocked{C['0']} — waits on: {', '.join(blocked)}")
        print("  use --force to override (it is written into the ledger)")
        return 1

    need = int(t.get("gpu_mb") or 0)
    if need > 0:
        n, detail, largest = probe_gpu_count()
        if n == 0 and not force:
            print(f"{C['r']}no usable GPU on the cluster{C['0']} — {detail}")
            print("  this is blocker B-1; see 00_PLANNING/04_BLOCKERS_AND_RISKS.md")
            return 1
        if largest < need and not force:
            print(f"{C['r']}no GPU large enough{C['0']} — {tid} needs {need} MiB, "
                  f"largest free is {largest} MiB")
            print(f"  {detail}")
            print("  this is blocker B-1; see 00_PLANNING/04_BLOCKERS_AND_RISKS.md")
            return 1

    script = os.path.join(HERE, t["script"])
    if not os.path.exists(script):
        print(f"{C['r']}script missing:{C['0']} {t['script']}")
        return 2

    entry = {"status": "running", "started": now(), "script": t["script"],
             "forced": bool(force and (blocked or st[tid]["passed"]))}
    if entry["forced"]:
        entry["forced_over"] = blocked
    update_ledger(tid, entry)

    print(f"\n{C['B']}=== {tid}: {t['title']} ==={C['0']}")
    print(f"  script {t['script']}   cost {t['cost']}\n")

    cmd = ([sys.executable, script] if script.endswith(".py") else ["bash", script])
    rc = subprocess.call(cmd, cwd=HERE)

    passed, detail = check_exit(t["exit_test"])
    update_ledger(tid, {
        "status": "passed" if passed else ("ran-but-incomplete" if rc == 0 else "failed"),
        "rc": rc, "finished": now(), "exit_test": detail})

    if passed:
        print(f"\n{C['g']}{tid} PASSED{C['0']} — {detail}")
    elif rc == 0:
        print(f"\n{C['y']}{tid} ran (rc=0) but its exit test does not pass yet{C['0']} — {detail}")
        print("  for a long/resumable task this is normal: re-run to continue.")
    else:
        print(f"\n{C['r']}{tid} FAILED{C['0']} rc={rc} — {detail}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true", help="ledger: passed / ready / blocked")
    g.add_argument("--next", action="store_true", help="name the next ready task")
    g.add_argument("--run", metavar="ID", help="run one task")
    g.add_argument("--run-next", action="store_true", help="run the next ready task")
    g.add_argument("--graph", action="store_true", help="dependency graph and critical path")
    ap.add_argument("--force", action="store_true", help="override a blocked dependency (logged)")
    a = ap.parse_args()

    data = load_tasks()
    if a.status:
        return cmd_status(data)
    if a.next:
        return cmd_next(data)
    if a.graph:
        return cmd_graph(data)
    if a.run_next:
        tid, _ = next_ready(data)
        if tid is None:
            print("nothing ready. See --status.")
            return 1
        return cmd_run(data, tid, a.force)
    return cmd_run(data, a.run, a.force)


if __name__ == "__main__":
    sys.exit(main())
