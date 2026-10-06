
"""Verify BOTH strands in ISOLATED subprocesses.

They must be separate processes: `models` is a namespace package and its
__path__ is fixed at first import, so testing both in one interpreter would
measure only whichever ran first -- which is the very bug being fixed.
"""
import json, subprocess, sys, os

SL = '/mnt/ricproject3/2026/SL_SPV'
OUT = '/mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/Final_Research_Works/03_RESULTS/W0.6'

PROP_TEST = r"""
import glob, importlib, json, os, sys
sys.path.insert(0, os.path.join('/mnt/ricproject3/2026/SL_SPV', "proposals"))
import evaluate_proposal as ep
ep._use_proposal_shim()                 # what main() does
res = {}
for f in sorted(glob.glob(os.path.join('/mnt/ricproject3/2026/SL_SPV', "proposals", "_trainer_shim", "models", "*.py"))):
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
sys.path.insert(0, os.path.join('/mnt/ricproject3/2026/SL_SPV', "transformer_sv"))
import evaluate_transformer as et       # imports evaluate_proposal as a library
sys.path.insert(0, os.path.join(et.layout.TRAINER, "experiments", "tools"))
import evaluate as ev
res = {}
for f in sorted(glob.glob(os.path.join('/mnt/ricproject3/2026/SL_SPV', "transformer_sv", "_trainer_shim", "models", "*.py"))):
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
        return {"_error": (p.stderr or p.stdout)[-600:]}

prop = run(PROP_TEST, os.path.join(SL, "proposals"))
trans = run(TRANS_TEST, os.path.join(SL, "transformer_sv"))

res = {"proposals_shim_models": prop, "transformer_shim_models": trans}
res["proposals_ok"] = bool(prop) and all(v == "ok" for v in prop.values())
res["transformer_ok"] = bool(trans) and all(v == "ok" for v in trans.values())
res["no_cross_shadowing"] = res["proposals_ok"] and res["transformer_ok"]
res["all_ok"] = res["no_cross_shadowing"]
print(json.dumps(res, indent=2))
open(os.path.join(OUT, "probe.json"), "w").write(json.dumps(res, indent=2))
