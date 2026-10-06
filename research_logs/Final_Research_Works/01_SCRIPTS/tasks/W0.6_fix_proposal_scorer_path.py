#!/usr/bin/env python3
"""W0.6 -- Put `_trainer_shim/` on sys.path in `proposals/evaluate_proposal.py`.

The defect (found 2026-09-12 while running task A1)
---------------------------------------------------
`evaluate_proposal.py` rebuilds a model from the literal argv that trained it --
correct provenance discipline, and the reason it exists. But it sets up only::

    sys.path.insert(0, HERE)                                    # proposals/
    sys.path.insert(0, os.path.join(TRAINER, "experiments", "tools"))
    sys.path.insert(0, TRAINER)                                 # voxceleb_trainer/

It never adds `proposals/_trainer_shim/`. Training does -- `common/train.py:98`
builds `PYTHONPATH = [SHIM, PATCHES, TRAINER]` for the child process -- so a
proposal that supplies its own `MainModel` **trains fine and cannot be scored**::

    ModuleNotFoundError: No module named 'models.PLLW'

Scope of the defect
-------------------
It hits every proposal that supplies a MODEL through the shim:

    A7 PLLW, A4 LSCAM, A5 DKCAMPP, A1 DA2LoRA   -- affected
    A9 (supplies a LOSS, standard model)         -- unaffected, which is why
                                                    A9 is the one that scored

So this single missing path is why three fully-trained runs sat unscored, and it
would have blocked D1/D2 as well. It is not a new bug introduced by this plan --
it has been latent since the scorer was written, and only shows up on the first
model-supplying proposal anyone tried to score.

The fix
-------
Do exactly what `transformer_sv/evaluate_transformer.py:35-41` already does, and
for the reason its own comment gives: *"the shim must precede the trainer so
`models` resolves here"*. That file is the working precedent; this brings the
A-series scorer in line with it.
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
TARGET = os.path.join(SL_SPV, "proposals", "evaluate_proposal.py")
OUT = os.path.join(FRW, "03_RESULTS", "W0.6")

OLD = """HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
"""

NEW = '''HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "_trainer_shim")


def _use_proposal_shim() -> None:
    """Put `proposals/_trainer_shim/` ahead of the trainer on sys.path.

    `models.<Proposal>` must resolve to the shim rather than to
    voxceleb_trainer/models/. Training already arranges this --
    common/train.py builds PYTHONPATH = [SHIM, PATCHES, TRAINER] for the child
    -- but this scorer did not, so any proposal supplying its own MainModel
    trained fine and then failed to score with
    `ModuleNotFoundError: No module named 'models.PLLW'`. That is why A7, A4 and
    A5 sat trained-but-unscored; A9 supplies a LOSS and a standard model, which
    is why it was the only one unaffected.

    WHY THIS IS A FUNCTION AND NOT A MODULE-LEVEL INSERT
    ---------------------------------------------------
    `transformer_sv/evaluate_transformer.py` imports THIS module as a library to
    reuse `score()`, and it has its own `_trainer_shim/` supplying
    `models.MFAConformer` and friends. Both shims contribute to the same `models`
    NAMESPACE package, so whichever directory reaches sys.path first wins for
    every subsequent lookup. Inserting at import time made importing this module
    silently hijack `models` for the T-series and broke
    `models.MFAConformer` there.

    So the path change belongs to the ENTRY POINT, not to the import. Run
    directly, this module sets it up in main(); imported as a library, it
    touches nothing and the importer's own shim keeps priority. [W0.6, 2026-09-12]
    """
    for _p in (SHIM, HERE):
        if _p not in sys.path:
            sys.path.insert(0, _p)


if HERE not in sys.path:
    sys.path.insert(0, HERE)
'''


MAIN_OLD = """    args = ap.parse_args()
"""

MAIN_NEW = """    args = ap.parse_args()

    # Entry-point-only shim setup: see _use_proposal_shim(). Doing this here
    # rather than at import time is what keeps `transformer_sv`'s own shim
    # working when it imports this module as a library. [W0.6]
    _use_proposal_shim()
"""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report = {"task": "W0.6", "ok": False,
              "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "file": os.path.relpath(TARGET, SL_SPV)}

    if not os.path.exists(TARGET):
        print(f"  MISSING {TARGET}")
        json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1

    src = open(TARGET, encoding="utf-8").read()

    if "W0.6" in src and "SHIM" in src:
        print("  already patched")
        report["status"] = "already-patched"
    elif OLD not in src:
        print("  FAILED: the sys.path preamble does not match the expected source.")
        report["status"] = "NO-ANCHOR"
        json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1
    else:
        backup = TARGET + ".pre-W0.6"
        if not os.path.exists(backup):
            shutil.copy2(TARGET, backup)
        new_src = src.replace(OLD, NEW, 1)
        if MAIN_OLD not in new_src:
            print("  FAILED: could not find `args = ap.parse_args()` in main()")
            report["status"] = "NO-MAIN-ANCHOR"
            json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
            return 1
        new_src = new_src.replace(MAIN_OLD, MAIN_NEW, 1)
        open(TARGET, "w", encoding="utf-8").write(new_src)
        ast.parse(open(TARGET, encoding="utf-8").read())
        print(f"  patched evaluate_proposal.py (backup: {os.path.basename(backup)})")
        print("  - added _use_proposal_shim() helper")
        print("  - called it from main(), NOT at import time")
        report["status"] = "patched"

    # ------------------------------------------------------------- verify
    print("\n  verifying both strands' shim models import, and that neither")
    print("  shadows the other...")
    probe = os.path.join(OUT, "_probe.py")
    open(probe, "w").write(f'''
"""Verify BOTH strands in ISOLATED subprocesses.

They must be separate processes: `models` is a namespace package and its
__path__ is fixed at first import, so testing both in one interpreter would
measure only whichever ran first -- which is the very bug being fixed.
"""
import json, subprocess, sys, os

SL = {SL_SPV!r}
OUT = {OUT!r}

PROP_TEST = r"""
import glob, importlib, json, os, sys
sys.path.insert(0, os.path.join({SL_SPV!r}, "proposals"))
import evaluate_proposal as ep
ep._use_proposal_shim()                 # what main() does
res = {{}}
for f in sorted(glob.glob(os.path.join({SL_SPV!r}, "proposals", "_trainer_shim", "models", "*.py"))):
    n = os.path.splitext(os.path.basename(f))[0]
    if n.startswith("__"): continue
    try:
        m = importlib.import_module("models." + n)
        res[n] = "ok" if hasattr(m, "MainModel") else "no MainModel"
    except Exception as e:
        res[n] = "FAIL: %s: %s" % (type(e).__name__, e)
print(json.dumps(res))
"""

TRANS_TEST = r"""
import glob, importlib, json, os, sys
sys.path.insert(0, os.path.join({SL_SPV!r}, "transformer_sv"))
import evaluate_transformer as et       # imports evaluate_proposal as a library
sys.path.insert(0, os.path.join(et.layout.TRAINER, "experiments", "tools"))
import evaluate as ev
res = {{}}
for f in sorted(glob.glob(os.path.join({SL_SPV!r}, "transformer_sv", "_trainer_shim", "models", "*.py"))):
    n = os.path.splitext(os.path.basename(f))[0]
    if n.startswith("__"): continue
    try:
        m = importlib.import_module("models." + n)
        res[n] = "ok" if hasattr(m, "MainModel") else "no MainModel"
    except Exception as e:
        res[n] = "FAIL: %s: %s" % (type(e).__name__, e)
print(json.dumps(res))
"""

def run(src, cwd):
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, cwd=cwd)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {{"_error": (p.stderr or p.stdout)[-600:]}}

prop = run(PROP_TEST, os.path.join(SL, "proposals"))
trans = run(TRANS_TEST, os.path.join(SL, "transformer_sv"))

res = {{"proposals_shim_models": prop, "transformer_shim_models": trans}}
res["proposals_ok"] = bool(prop) and all(v == "ok" for v in prop.values())
res["transformer_ok"] = bool(trans) and all(v == "ok" for v in trans.values())
res["no_cross_shadowing"] = res["proposals_ok"] and res["transformer_ok"]
res["all_ok"] = res["no_cross_shadowing"]
print(json.dumps(res, indent=2))
open(os.path.join(OUT, "probe.json"), "w").write(json.dumps(res, indent=2))
''')
    subprocess.call([sys.executable, probe], cwd=os.path.join(SL_SPV, "proposals"))

    ppath = os.path.join(OUT, "probe.json")
    p = json.load(open(ppath)) if os.path.exists(ppath) else {}
    report["probe"] = p
    ok = bool(p.get("all_ok"))

    if ok:
        print(f"\n  VERIFIED in isolated subprocesses:")
        print(f"    proposals shim  : {len(p['proposals_shim_models'])} model(s) import")
        print(f"    transformer shim: {len(p['transformer_shim_models'])} model(s) import")
        print("    neither shim shadows the other")
    else:
        print("\n  VERIFICATION FAILED -- inspect probe.json")
        for strand in ("proposals_shim_models", "transformer_shim_models"):
            for k, v in (p.get(strand) or {}).items():
                if v != "ok":
                    print(f"    {strand}/{k}: {v}")

    report["ok"] = ok
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    print(f"\n  -> {os.path.join(OUT, 'result.json')}  ok={ok}")
    print("\n  Unblocks: A1 (A7 PLLW), A2 (A4 LSCAM, A5 DKCAMPP), and later D1/D2.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
