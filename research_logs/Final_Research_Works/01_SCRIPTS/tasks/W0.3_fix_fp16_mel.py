#!/usr/bin/env python3
"""W0.3 -- Fix the fp16 mel-spectrogram overflow in the CAM++ family (A4, A5).

The defect
----------
``A5_dk_campp/DK_CAMPP.py`` builds its **own** ``MelSpectrogram`` inside the
model rather than taking the trainer's, and computes it with no autocast guard::

    def features(self, x):
        with torch.no_grad():
            x = self.torchfb(x) + 1e-6      # <-- fp16 under --mixedprec
            if self.log_input: x = x.log()
            x = self.instancenorm(x)

Measured over 30,000 real augmented samples (annual report 7.2): MUSAN mixing
and RIR convolution push |x| to 5.66, and the fp16 power spectrogram peaks at
**30,592 against fp16's 65,504 ceiling -- 2.1x headroom, not orders of
magnitude.** One ``inf`` there reaches ``InstanceNorm1d`` and then BatchNorm's
running buffers, which are written in the *forward* pass where ``GradScaler``
does not guard. Eval then returns NaN for every input **while training accuracy
still looks healthy**, and the run dies at its next validation.

A4 inherits it: ``ls_cam.py:223`` builds a ``DK_CAMPP`` trunk.

The fix
-------
Compute the front end in fp32 with autocast explicitly disabled -- the same
remedy ``transformer_sv/common/attn.py:229`` already applies to ``q @ k^T``.
In fp32 ``.float()`` is the identity, so **no existing number changes** and no
asserted equivalence is disturbed. Only the two runs that never produced a
number are affected.

Verification
------------
This script does not merely patch. It drives the patched ``features()`` under
autocast with inputs at the measured worst case (|x| up to 8.0, above the 5.66
observed) and asserts:
  1. zero non-finite values in the output;
  2. the fp32 path is numerically unchanged (max abs diff below tolerance).
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
TARGET = os.path.join(SL_SPV, "proposals", "A5_dk_campp", "DK_CAMPP.py")
OUT = os.path.join(FRW, "03_RESULTS", "W0.3")

OLD = """    def features(self, x):
        \"\"\"Waveform -> normalised log-mel, matching the trainer's convention.\"\"\"
        with torch.no_grad():
            x = self.torchfb(x) + 1e-6
            if self.log_input:
                x = x.log()
            x = self.instancenorm(x)
        return x
"""

NEW = '''    def features(self, x):
        """Waveform -> normalised log-mel, matching the trainer's convention.

        Computed in fp32 with autocast explicitly disabled. [W0.3]

        This model builds its own MelSpectrogram rather than taking the
        trainer's, and the power spectrogram is the one quantity here with no
        bound on its magnitude. Measured over 30,000 real augmented samples,
        MUSAN + RIR push |x| to 5.66 and the fp16 power spectrogram peaks at
        30,592 against fp16's 65,504 ceiling -- 2.1x headroom, not orders of
        magnitude. A single `inf` propagates through InstanceNorm into
        BatchNorm's *running buffers*, which are written in the forward pass
        where GradScaler does not guard, so eval returns NaN for every input
        while training accuracy still looks healthy. Observed 2026-08-24 on
        DKCAMPP_ta and LSCAM_ta (which inherits this trunk).

        In fp32 `.float()` is the identity, so this changes no published number
        and no asserted equivalence -- only the runs that never produced one.
        """
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            x = self.torchfb(x.float()) + 1e-6
            if self.log_input:
                x = x.log()
            x = self.instancenorm(x)
        return x
'''


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report = {"task": "W0.3", "ok": False,
              "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    if not os.path.exists(TARGET):
        print(f"  MISSING {TARGET}")
        json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1

    src = open(TARGET, encoding="utf-8").read()
    report["file"] = os.path.relpath(TARGET, SL_SPV)

    if "amp.autocast('cuda', enabled=False)" in src and "W0.3" in src:
        print("  already patched")
        report["status"] = "already-patched"
    elif OLD not in src:
        print("  FAILED: features() does not match the expected source.")
        print("  Inspect DK_CAMPP.py:features() by hand -- it may already differ.")
        report["status"] = "NO-ANCHOR"
        json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1
    else:
        backup = TARGET + ".pre-W0.3"
        if not os.path.exists(backup):
            shutil.copy2(TARGET, backup)
        open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW))
        ast.parse(open(TARGET, encoding="utf-8").read())
        print(f"  patched {os.path.basename(TARGET)} (backup: {os.path.basename(backup)})")
        report["status"] = "patched"

    # ------------------------------------------------------------ verify
    print("\n  verifying under autocast at the measured worst case...")
    verify = os.path.join(OUT, "_verify.py")
    open(verify, "w").write(f'''
import sys, json, torch
sys.path.insert(0, {SL_SPV!r})
sys.path.insert(0, {os.path.join(SL_SPV, "proposals")!r})
from A5_dk_campp.DK_CAMPP import DK_CAMPP

res = {{}}
torch.manual_seed(0)
m = DK_CAMPP(nOut=256).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev)
res["device"] = dev

# |x| up to 8.0 -- above the 5.66 observed over 30,000 real augmented samples.
x = (torch.randn(4, 32000, device=dev) * 2.5).clamp(-8.0, 8.0)
res["input_absmax"] = float(x.abs().max())

with torch.no_grad():
    f32 = m.features(x)
res["fp32_finite"] = bool(torch.isfinite(f32).all())

if dev == "cuda":
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        f16 = m.features(x)
    res["amp_dtype"] = str(f16.dtype)
    res["amp_finite"] = bool(torch.isfinite(f16).all())
    res["amp_nonfinite_count"] = int((~torch.isfinite(f16)).sum())
    res["max_abs_diff_vs_fp32"] = float((f16.float() - f32).abs().max())
    # raw power spectrogram headroom, the quantity that actually overflows
    with torch.no_grad():
        p = m.torchfb(x.float())
    res["power_spec_peak"] = float(p.max())
    res["fp16_ceiling"] = 65504.0
    res["headroom_x"] = 65504.0 / max(float(p.max()), 1e-9)
else:
    res["amp_finite"] = None
    res["note"] = "no CUDA on this host: autocast path not exercised. Re-run W0.3 on a GPU node to complete verification."

print(json.dumps(res, indent=2))
open({os.path.join(OUT, "verify.json")!r}, "w").write(json.dumps(res, indent=2))
''')

    # The head node has no GPU, and the assertion that matters is the autocast
    # one. Dispatch the verifier to a node that has a live card, via the
    # project's own gpurun.sh -- /mnt/ricproject3 is NFS at the same path
    # everywhere, so nothing is copied.
    gpurun = os.path.join(SL_SPV, "voxceleb_trainer", "tools", "gpurun.sh")
    if os.path.exists(gpurun):
        print("  dispatching verifier to a GPU node via gpurun.sh ...")
        rc = subprocess.call(["bash", gpurun, "-m", "4000", "--",
                              "python", verify], cwd=os.path.dirname(gpurun))
        if rc != 0:
            print("  gpurun dispatch failed; falling back to this host (CPU only)")
            rc = subprocess.call([sys.executable, verify])
    else:
        rc = subprocess.call([sys.executable, verify])
    vpath = os.path.join(OUT, "verify.json")
    v = json.load(open(vpath)) if os.path.exists(vpath) else {}
    report["verification"] = v

    if v.get("device") == "cuda":
        ok = bool(v.get("fp32_finite")) and bool(v.get("amp_finite")) \
             and v.get("max_abs_diff_vs_fp32", 1.0) < 1e-3
        if ok:
            print(f"\n  VERIFIED on GPU: 0 non-finite under autocast; "
                  f"max |amp - fp32| = {v['max_abs_diff_vs_fp32']:.2e}")
            print(f"  power spectrogram peak {v['power_spec_peak']:.0f}, "
                  f"fp16 headroom {v['headroom_x']:.2f}x")
        else:
            print("\n  VERIFICATION FAILED -- do not launch D1 until this passes")
    else:
        ok = False
        print("\n  PARTIAL: patched, and the fp32 path is verified -- but the autocast")
        print("  assertion did not run, and that is the assertion that matters here.")
        print("  W0.3 stays INCOMPLETE until it runs on a live GPU. Re-run when one is up.")
        report["partial"] = True

    # isolation
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=os.path.join(SL_SPV, "voxceleb_trainer"),
                           capture_output=True, text=True).stdout
    touched = [l for l in dirty.splitlines()
               if not l.startswith("??") and re.search(r"models/|SpeakerNet|trainSpeakerNet", l)]
    report["isolation_ok"] = not touched
    print("  isolation OK -- voxceleb_trainer/ models untouched" if not touched
          else f"  ISOLATION WARNING: {touched}")

    report["ok"] = bool(ok and not touched)
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    print(f"\n  -> {os.path.join(OUT, 'result.json')}  ok={report['ok']}")
    print("\n  Unblocked by this fix: A4 LSCAM_ta, A5 DKCAMPP_ta (task D1).")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
