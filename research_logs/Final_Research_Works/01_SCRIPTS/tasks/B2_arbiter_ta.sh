#!/usr/bin/env bash
# B2 -- The arbiter run, Tamil. Same design as B1; see that script for rationale.
#
# WHAT IT DECIDES
# ---------------
# v1 measured two SSL configurations that differ on TWO axes at once:
#
#     ssl_wavlm_lw   layer-weighted, FROZEN        si 2.369 held-out
#     ssl_wavlm_ft   last-layer,     FINE-TUNED    si 5.404 held-out
#
# The diagonal -- layer-weighted AND fine-tuned -- was never run. The annual
# report section 11 item 1:
#
#   "v1 measured layer-weighted+frozen (2.369 si) and single-layer+fine-tuned
#    (5.404 si) but never the diagonal that P3's headline claim actually rests
#    on -- and that claim currently underpins papers/ieee_spl."
#
# So this run either CONFIRMS or WITHDRAWS a claim in a paper already drafted.
# That is why it outranks completeness.
#
# The Tamil arm matters independently: Finding III-4 showed the last-layer
# penalty SCALES WITH LINGUISTIC DISTANCE (1.04x en, 1.88x si, 5.23x ta), so
# Tamil is where layer weighting buys the most -- and therefore where
# fine-tuning has the least left to add. P21 predicts the smallest gain here.
#
# PRE-REGISTERED PREDICTION (P21): layer-weighted + fine-tuned will beat
# layer-weighted + frozen on si, but by LESS than the 2.369 -> 1.69 gap that full
# fine-tuning showed at last-layer. Rationale: if layer weighting already
# recovers most of what depth costs (Finding III-3), fine-tuning has less left to
# recover. A NULL would be the more useful outcome -- it would mean layer
# weighting is the cheap substitute for fine-tuning.
#
# HARDWARE: 30 GB. No live GPU can hold this today (largest is 14,914 MiB).
# Blocked on W0.4. The script refuses rather than silently placing it wrong.

. "$(dirname "$0")/../lib/common.sh"

EXP_ID=F_ssl_wavlm_lw_ft_aamsoftmax_ta_s42
OUTDIR="$RESULTS/B2"; mkdir -p "$OUTDIR"

step "Preconditions"
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.environ["EXPERIMENTS"])
import registry as r
if "ssl_wavlm_lw_ft" not in r.FRONTENDS:
    raise SystemExit("ssl_wavlm_lw_ft is not registered -- run W0.5 first")
f = r.FRONTENDS["ssl_wavlm_lw_ft"]
print(f"  model      {f['model']}")
print(f"  flags      {f['flags']}")
print(f"  overrides  {f.get('overrides')}")
print(f"  min_gpu_mb {f['min_gpu_mb']}")
PY

require_gpu_mb 30000
trainer_guard_snapshot

step "Generating the experiment script"
cd "$EXPERIMENTS"
python3 tools/gen_scripts.py --stage F --archs ssl_wavlm_lw_ft --conditions ta --seeds 42 \
    2>&1 | tee "$OUTDIR/gen.log" || warn "gen_scripts reported a problem"

step "Training"
set +e
bash "$GPURUN" -d -m 30000 -- python "scripts/${EXP_ID}.py" 2>&1 | tee "$OUTDIR/train.log"
log "  dispatch rc=${PIPESTATUS[0]}"
set -e

trainer_guard_verify "registry\.py|scripts/|results/"

cat <<'NOTE'

  LAUNCHED DETACHED. This is a ~20 GPU-hour run on an A40.

  Monitor:   python3 tools/status.py --watch
  Resume:    re-run B1 -- the trainer auto-resumes from its highest checkpoint
  Then:      B3 (the 2x2 analysis)

  DO NOT quote a number from this run until B3 has placed it in the 2x2 against
  ssl_wavlm_lw and ssl_wavlm_ft on identical trials with the paired bootstrap.
  A single absolute EER at S ~ 91 carries +/-5-6 pp; only the paired contrast
  resolves anything at this scale.
NOTE
