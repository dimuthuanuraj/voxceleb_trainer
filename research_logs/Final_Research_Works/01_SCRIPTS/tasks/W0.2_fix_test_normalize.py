#!/usr/bin/env python3
"""W0.2 -- Fix `test_normalize` on the A1/A8 wrapper losses.

The defect
----------
A1 (DA2-LoRA) and A8 (CD-NDAL) each supply a custom ``LossFunction`` that
*wraps* the trainer's AAM-Softmax in ``self.speaker``. The trainer's evaluation
path reads the attribute directly off the loss object::

    SpeakerNet.py:694    if self.__model__.module.__L__.test_normalize:

The wrapper never exposes it, so all four runs die with::

    AttributeError: 'LossFunction' object has no attribute 'test_normalize'

They die at their **first validation**, which is why both proposals have zero
checkpoints: the failure is at the start of the run, not the end, so the whole
run is lost. Four runs, one missing attribute.

The fix
-------
Delegate to the wrapped AAM, which sets ``test_normalize = True`` at
``loss/aamsoftmax.py:15``. Delegating rather than hard-coding ``True`` keeps the
wrapper honest if the inner loss ever changes its convention.

This writes only inside ``proposals/_trainer_shim/``. ``voxceleb_trainer/`` is
untouched -- execution plan rule 6.
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
SHIM = os.path.join(SL_SPV, "proposals", "_trainer_shim", "loss")
OUT = os.path.join(FRW, "03_RESULTS", "W0.2")

TARGETS = ["da2_lora_adv.py", "cd_ndal_adv.py"]

# Inserted immediately after `self.speaker = AAM(...)` in each wrapper.
PATCH = """
        # The trainer's eval path reads `__L__.test_normalize` directly
        # (SpeakerNet.py:694). This wrapper delegates to the AAM it wraps rather
        # than hard-coding True, so it stays correct if the inner loss ever
        # changes convention. Without it the run dies at its FIRST validation
        # with AttributeError and leaves no checkpoint at all. [W0.2]
        self.test_normalize = self.speaker.test_normalize
"""


def find_speaker_assignment(src: str) -> int | None:
    """Return the line index just after the `self.speaker = AAM(...)` statement."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and tgt.attr == "speaker"
                        and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
                    return node.end_lineno          # 1-based, inclusive
    return None


def already_patched(src: str) -> bool:
    return re.search(r"self\.test_normalize\s*=", src) is not None


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report = {"task": "W0.2", "files": [], "ok": False,
              "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    for name in TARGETS:
        path = os.path.join(SHIM, name)
        entry = {"file": os.path.relpath(path, SL_SPV)}
        if not os.path.exists(path):
            entry["status"] = "MISSING"
            report["files"].append(entry)
            print(f"  MISSING {path}")
            continue

        src = open(path, encoding="utf-8").read()

        if already_patched(src):
            entry["status"] = "already-patched"
            report["files"].append(entry)
            print(f"  already patched: {name}")
            continue

        line = find_speaker_assignment(src)
        if line is None:
            entry["status"] = "NO-ANCHOR"
            entry["detail"] = "could not locate `self.speaker = AAM(...)`"
            report["files"].append(entry)
            print(f"  FAILED {name}: no `self.speaker = AAM(...)` assignment found")
            continue

        backup = path + ".pre-W0.2"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)

        lines = src.splitlines(keepends=True)
        lines.insert(line, PATCH.lstrip("\n"))
        open(path, "w", encoding="utf-8").write("".join(lines))

        entry.update({"status": "patched", "after_line": line,
                      "backup": os.path.basename(backup)})
        report["files"].append(entry)
        print(f"  patched {name}: inserted after line {line}")

    # ---- verification: parse, and assert the attribute is really reachable
    print("\n  verifying...")
    all_ok = True
    for name in TARGETS:
        path = os.path.join(SHIM, name)
        if not os.path.exists(path):
            all_ok = False
            continue
        src = open(path, encoding="utf-8").read()
        try:
            ast.parse(src)
        except SyntaxError as exc:
            print(f"    {name}: SYNTAX ERROR after patch -- {exc}")
            all_ok = False
            continue
        if not already_patched(src):
            print(f"    {name}: attribute still absent")
            all_ok = False
            continue
        # the assignment must sit inside __init__, after self.speaker exists
        tree = ast.parse(src)
        ok_here = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                spk = tn = None
                for st in ast.walk(node):
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                                    and t.value.id == "self":
                                if t.attr == "speaker":
                                    spk = st.lineno
                                elif t.attr == "test_normalize":
                                    tn = st.lineno
                if spk and tn and tn > spk:
                    ok_here = True
        if ok_here:
            print(f"    {name}: OK -- test_normalize assigned in __init__ after self.speaker")
        else:
            print(f"    {name}: assignment present but not ordered after self.speaker")
            all_ok = False

    # ---- isolation guard
    trainer = os.path.join(SL_SPV, "voxceleb_trainer")
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=trainer,
                           capture_output=True, text=True).stdout
    touched = [l for l in dirty.splitlines()
               if l[3:].startswith(("loss/", "SpeakerNet.py", "trainSpeakerNet.py"))
               and not l.startswith("??")]
    report["isolation_ok"] = not touched
    if touched:
        print(f"\n  ISOLATION WARNING: trainer loss/ or core files modified:")
        for t in touched:
            print(f"    {t}")
        all_ok = False
    else:
        print("\n  isolation OK -- voxceleb_trainer/ loss and core untouched")

    report["ok"] = all_ok
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    print(f"\n  -> {os.path.join(OUT, 'result.json')}  ok={all_ok}")
    print("\n  Unblocked by this fix: A1 anchored/control, A8 adv0.3/control (tasks D2, D3).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
