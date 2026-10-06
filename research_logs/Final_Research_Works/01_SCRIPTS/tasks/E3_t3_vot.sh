#!/usr/bin/env bash
# E3 -- T3 vot
#
# P5 -- which time scale carries speaker evidence.
#
# EXPERIMENT_PLAN.md: "Worth it mainly for the branch weights, which answer a
# question v1s fixed --max_frames 200 never asked. Its EER is a secondary output."
#
# So report the BRANCH WEIGHTS as the result. r1 is the single-branch control
# with the gate pinned to 1.0; it is not parameter-matched and must say so.
#
# Ordering note: transformer_sv/EXPERIMENT_PLAN.md section 4 fixes the priority
# P1-P6 and it should not be reshuffled for convenience. P1-P3 change one factor
# against a trustworthy baseline; P5 and P6 change more at once and cost more.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/E3"; mkdir -p "$OUTDIR"
EXP=T3_vot
ARMS="si:r124 si:r1"

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
echo "  Launched detached. Re-run E3 after training to score."
echo "  scaffold.py --check is the figure to trust for what has produced numbers."
