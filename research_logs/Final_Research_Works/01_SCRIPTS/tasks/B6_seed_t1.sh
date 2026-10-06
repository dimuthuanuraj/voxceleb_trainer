#!/usr/bin/env bash
# B6 -- Seed replication for T1 MFAConformer full_si.
#
# The T1 result -- a from-scratch Conformer loses to a parameter-matched ECAPA by
# 0.665 pp -- CONFIRMS the strand's pre-registered prediction and supports the
# programme's cleanest self-contained argument:
#
#   "On low-resource languages the value of a transformer is in its pretraining,
#    not in its architecture." The same family loses by 0.665 pp from scratch and
#    wins by 1.92 pp pretrained and correctly read, on identical trials.
#
# But the annual report's own caveat is explicit: "single seed, one language. The
# paired bootstrap has not been run on this contrast, and at S = 91 the +0.665 pp
# gap is NEAR THE RESOLUTION LIMIT established in section 6.8.2. Treat as
# directional pending the ta arm and a seed replication."
#
# A4 supplies the ta arm; A5 supplies the bootstrap; this supplies the seeds.
# Together they convert a directional claim into a reportable one.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/B6"; mkdir -p "$OUTDIR"
require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"
cd "$TRANSFORMER"

for SEED in 123 7; do
    step "MFAConformer full si seed=$SEED"
    set +e
SLOT="$(next_idle_slot)"
    if [[ -z "$SLOT" ]]; then
        warn "  every GPU is busy -- deferring the rest of this task to the next pass"
        warn "  (this task is idempotent: re-running it picks up where it stopped)"
        break
    fi
    read -r SNODE SGPU <<< "$SLOT"
    log "  slot: $SNODE gpu$SGPU"
    bash "$GPURUN" -n "$SNODE" -g "$SGPU" -d -m 8000 -- \
        python T1_mfa_conformer/run.py --lang si --arm full --seed "$SEED" \
        2>&1 | tee "$OUTDIR/s${SEED}.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "Scoring and aggregating"
set +e
bash "$GPURUN" -m 8000 -- python evaluate_transformer.py --all --device cuda 2>&1 | tee "$OUTDIR/score.log"
set -e

python3 - "$OUTDIR" <<'PY'
import glob, json, os, statistics as st, sys
outdir = sys.argv[1]
T = os.environ["TRANSFORMER"]
rows = []
for f in sorted(glob.glob(os.path.join(T, "T1_mfa_conformer", "out", "MFAConformer_full_si_s*.eval.json"))):
    tag = os.path.basename(f).replace(".eval.json", "")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    if eer is not None:
        rows.append((tag, float(eer)))
print(f"  {len(rows)} seed(s) of MFAConformer_full_si")
for t, v in rows:
    print(f"    {t:44s} {v:.4f}")
out = {"task": "B6", "seeds": rows, "ok": len(rows) >= 3,
       "baseline_ecapa512_si_test_eer": 4.285,
       "single_seed_gap_pp": 0.665}
if len(rows) > 1:
    vals = [v for _, v in rows]
    m, s = st.mean(vals), st.stdev(vals)
    print(f"\n  mean {m:.4f} +/- {s:.4f}   baseline A_ecapa512 = 4.285")
    out["mean"], out["sd"] = m, s
    print(f"  gap vs baseline: {m - 4.285:+.4f} pp")
    if s > 0.665:
        print("\n  SEED SPREAD EXCEEDS THE CLAIMED GAP.")
        print("  The T1 contrast is not resolvable at one seed and the directional")
        print("  reading must be withdrawn, not merely caveated.")
    else:
        print("\n  Seed spread is smaller than the gap -- the contrast survives replication.")
json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY
