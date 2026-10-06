#!/usr/bin/env bash
# A6 -- The combined-condition shared-layer-weight control. Makes A7 interpretable.
#
# WHY THIS TASK EXISTS
# --------------------
# A1 scored A7 PLLW and got a real number: 3.024 % cosine / 2.631 % AS-Norm on
# the held-out si test set, against its reference baseline's 3.297 / 2.964 -- on
# the SAME 19,838 trials, so the pairing is exact.
#
# But the two runs differ in TWO things, not one:
#
#   A7 PLLW                        trained on combined_si_ta, 782 speakers
#   F_ssl_wavlm_lw_..._si_s42      trained on si only,        336 speakers
#
# So the -0.27 pp cannot be attributed to per-language layer weighting. It
# confounds the METHOD with 2.3x more training data, and more speakers alone
# would be expected to help.
#
# Verified 2026-09-12: `experiments/analysis/v1-final/results_table.csv` contains
# ZERO rows with condition == "combined". The control does not exist anywhere in
# the project -- which is why A7's number has never been interpretable, and why
# scoring it (annual report open item #3) was necessary but not sufficient.
#
# THE CONTROL: ssl_wavlm_lw -- SHARED layer weights, one 13-vector for both
# languages -- trained on combined_si_ta, scored on the same si test trials.
# Everything else identical. Then:
#
#   A7 (per-language weights) vs A6 (shared weights)   on combined  <- the METHOD
#   A6 (shared, combined)     vs v1 (shared, si)        on si        <- the DATA
#
# Two single-factor contrasts instead of one confounded pair. A7 adds ~39
# parameters over A6, so the difference needs the speaker-clustered bootstrap.
#
# Cost: SSL encoder frozen, so ~8 GPU-h and it fits a T4. Runnable today.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/A6"; mkdir -p "$OUTDIR"
EXP_ID=F_ssl_wavlm_lw_aamsoftmax_combined_s42

step "Confirming the control really is absent"
python3 - <<'PY'
import csv, os
p = os.path.join(os.environ["EXPERIMENTS"], "analysis", "v1-final", "results_table.csv")
n = 0
with open(p) as fh:
    for row in csv.DictReader(fh):
        if row.get("condition") == "combined":
            n += 1
            print(f"  found: {row['exp_id']} {row['eval_set']} eer={row['test_eer']}")
print(f"  combined-condition rows in v1: {n}")
if n:
    print("  A control may already exist -- inspect before spending GPU-hours.")
PY

step "Is 'combined' a registered condition?"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["EXPERIMENTS"])
import registry as r
print(f"  conditions: {list(r.CONDITIONS)}")
if "combined" not in r.CONDITIONS:
    raise SystemExit("`combined` is not a registry condition -- add it before running")
c = r.CONDITIONS["combined"]
print(f"  combined -> {c}")
PY

require_gpu_mb 8000
trainer_guard_snapshot
cd "$EXPERIMENTS"

step "Generating the control"
set +e
python3 tools/gen_scripts.py --stage F --archs ssl_wavlm_lw --conditions combined --seeds 42 \
    2>&1 | tee "$OUTDIR/gen.log"
set -e

step "Training"
if [[ -f "results/$EXP_ID/final.json" ]]; then
    log "  already complete"
else
    set +e
    bash "$GPURUN" -d -m 8000 -- python "scripts/${EXP_ID}.py" 2>&1 | tee "$OUTDIR/train.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
    log "  LAUNCHED DETACHED (~8 GPU-h). Re-run A6 to score."
fi

trainer_guard_verify "registry\.py|scripts/|results/"

step "The two contrasts, once the control exists"
python3 - "$OUTDIR" <<'PY'
import csv, json, os, sys
outdir = sys.argv[1]
EXPD = os.environ["EXPERIMENTS"]
A7 = {"eer_cosine": 3.024498437342474, "eer_asnorm": 2.6313136404879534}
V1_SI = {"eer_cosine": 3.296703296703297, "eer_asnorm": 2.9640084685956247}

ctrl = None
f = os.path.join(EXPD, "results", "F_ssl_wavlm_lw_aamsoftmax_combined_s42", "final.json")
if os.path.exists(f):
    try:
        d = json.load(open(f))
        ctrl = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    except Exception:
        pass

out = {"task": "A6", "ok": ctrl is not None,
       "a7_pllw_per_language": A7, "v1_shared_si_only": V1_SI,
       "control_shared_combined": ctrl}

if ctrl is None:
    print("  control not trained yet -- the two contrasts cannot be formed.")
    print("\n  Until then, the ONLY safe statement about A7 is its absolute number:")
    print("    A7 PLLW: 3.024 % cosine / 2.631 % AS-Norm, held-out si, 19,838 trials.")
    print("  Whether per-language weighting caused any of it is UNRESOLVED.")
else:
    method = A7["eer_cosine"] - ctrl
    data = ctrl - V1_SI["eer_cosine"]
    print(f"  METHOD effect (A7 per-language vs shared, both combined): {method:+.3f} pp")
    print(f"  DATA   effect (shared combined vs shared si-only)       : {data:+.3f} pp")
    print(f"  the confounded raw difference was                       : "
          f"{A7['eer_cosine'] - V1_SI['eer_cosine']:+.3f} pp")
    out["method_effect_pp"] = method
    out["data_effect_pp"] = data
    print("\n  Run the speaker-clustered bootstrap on the METHOD contrast before")
    print("  claiming it: A7 adds ~39 parameters and the effect is well under 1 pp.")

json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  WHAT A7 ACTUALLY ANSWERS, once A6 lands
  ---------------------------------------
  Not "is A7 better" but "do Sinhala and Tamil want different SSL depths". The
  fitted per-language weight vectors are the primary output; the EER is
  secondary. `experiments/tools/layer_weights.py` reads them out.

  If the two languages' optimal depths differ, one shared weight vector is
  leaving performance on the table -- the direct extension of Finding III-3 for
  ~39 extra parameters, and the reason the annual report ranked this third among
  all open items.
NOTE
