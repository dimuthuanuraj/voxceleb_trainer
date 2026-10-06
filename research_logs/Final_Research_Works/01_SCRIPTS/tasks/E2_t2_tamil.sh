#!/usr/bin/env bash
# E2 -- T2 tamil
#
# The Tamil arm of the cleanest single-factor contrast in the folder.
#
# tf0 is an ASSERTED EXACT REDUCTION of tf4 (stem only, bit-identical path), so
# the pair isolates attention-as-refinement over convolutional features with no
# approximation anywhere. Protocol rule 8: a control must be an exact reduction,
# asserted rather than asserted-to-be-similar.
#
# Tamil matters here for the same reason as everywhere else in this programme:
# Finding III-4 showed the cost of a wrong default SCALES WITH LINGUISTIC
# DISTANCE, and Tamil is the far end of that scale.
#
# Ordering note: transformer_sv/EXPERIMENT_PLAN.md section 4 fixes the priority
# P1-P6 and it should not be reshuffled for convenience. P1-P3 change one factor
# against a trustworthy baseline; P5 and P6 change more at once and cost more.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/E2"; mkdir -p "$OUTDIR"
EXP=T2_res2former
ARMS="ta:tf4 ta:tf0"

require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"
cd "$TRANSFORMER"

for spec in $ARMS; do
    IFS=':' read -r LANG ARM <<< "$spec"
    step "$EXP $LANG $ARM"

    # Already scored?
    if ls "$TRANSFORMER/$EXP/out/"*"${ARM}_${LANG}"*.eval.json >/dev/null 2>&1; then
        log "  already scored -- skipping"
        continue
    fi

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
        python "$EXP/run.py" --lang "$LANG" --arm "$ARM" \
        2>&1 | tee -a "$OUTDIR/${ARM}_${LANG}.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "Scoring anything that has finished"
set +e
bash "$GPURUN" -m 8000 -- python evaluate_transformer.py --all --device cuda \
    2>&1 | tee "$OUTDIR/score.log"
set -e

python3 - "$OUTDIR" "$EXP" <<'PY'
import glob, json, os, sys
outdir, exp = sys.argv[1], sys.argv[2]
T = os.environ["TRANSFORMER"]
hits = sorted(glob.glob(os.path.join(T, exp, "out", "*.eval.json")))
print(f"  {len(hits)} scored arm(s) in {exp}")
res = {}
for h in hits:
    try:
        d = json.load(open(h))
    except Exception:
        continue
    tag = os.path.basename(h).replace(".eval.json", "")
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    print(f"    {tag:44s} eer={eer}")
    res[tag] = d
json.dump({"task": os.path.basename(outdir), "ok": bool(res), "results": res},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

echo
echo "  Launched detached. Re-run E2 after training to score."
echo "  scaffold.py --check is the figure to trust for what has produced numbers."
