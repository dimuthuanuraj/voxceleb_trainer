#!/usr/bin/env bash
# D1 -- Retrain A4 LSCAM_ta and A5 DKCAMPP_ta under the fp32 mel fix (W0.3).
#
# Annual report section 11 item 5: "Fix and re-run A4/A5 on Tamil. The diagnosis
# and the one-line fix are recorded; the runs simply never produced a number."
#
# W0.3 applied the fix and verified it on a GPU: 0 non-finite values under
# autocast, and the fp32 path bit-identical (max |amp - fp32| = 0.0). The measured
# fp16 headroom at |x| = 8.0 was only 1.42x, so the overflow was never
# hypothetical.
#
# BEFORE RE-LAUNCHING: the annual report notes all eight existing A-series
# checkpoints were verified clean (0 poisoned BN buffers, 0 NaN weights), so no
# published A-series number is affected. These two runs simply never produced
# one. Do not "recover" the old dead checkpoints -- delete them and start clean,
# because a poisoned BatchNorm buffer is not recoverable by resuming.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/D1"; mkdir -p "$OUTDIR"

step "Verifying the W0.3 fix is in place"
grep -q "W0.3" "$PROPOSALS/A5_dk_campp/DK_CAMPP.py" || \
    die "DK_CAMPP.py does not carry the W0.3 fp32 fix -- run W0.3 first"
grep -q "amp.autocast('cuda', enabled=False)" "$PROPOSALS/A5_dk_campp/DK_CAMPP.py" || \
    die "the fp32 autocast guard is missing from DK_CAMPP.features()"
log "fp32 mel guard present"

V="$RESULTS/W0.3/verify.json"
if [[ -f "$V" ]]; then
    python3 - "$V" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
if v.get("device") != "cuda":
    raise SystemExit("W0.3 was only verified on CPU. Re-run W0.3 on a GPU node "
                     "before spending 70 GPU-hours on these arms.")
assert v.get("amp_finite"), "W0.3 GPU verification did not pass"
print(f"  W0.3 verified on GPU: 0 non-finite, headroom {v['headroom_x']:.2f}x")
PY
else
    die "no W0.3 verification record"
fi

step "Clearing the dead runs"
# A poisoned BatchNorm running buffer is written in the forward pass and is not
# recoverable by resuming -- the buffers are already wrong. Start clean.
for d in "$PROPOSALS/A4_ls_cam/out/exps/LSCAM_ta_s42" \
         "$PROPOSALS/A5_dk_campp/out/exps/DKCAMPP_ta_s42"; do
    if [[ -d "$d" ]]; then
        n=$(ls "$d"/model/model0*.model 2>/dev/null | wc -l)
        if (( n > 0 )); then
            warn "  $d has $n checkpoints from the NaN run"
            mv "$d" "${d}.nan-$(date -u +%Y%m%d-%H%M%S)"
            log "  moved aside (not deleted)"
        else
            log "  $(basename "$d"): no checkpoints, nothing to clear"
        fi
    fi
done

require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"
cd "$PROPOSALS"

for spec in "A4_ls_cam|LSCAM_ta_s42|--lang ta" "A5_dk_campp|DKCAMPP_ta_s42|--lang ta"; do
    IFS='|' read -r DIR TAG ARGS <<< "$spec"
    step "$TAG"
    set +e
    # shellcheck disable=SC2086
SLOT="$(next_idle_slot)"
    if [[ -z "$SLOT" ]]; then
        warn "  every GPU is busy -- deferring the rest of this task to the next pass"
        warn "  (this task is idempotent: re-running it picks up where it stopped)"
        break
    fi
    read -r SNODE SGPU <<< "$SLOT"
    log "  slot: $SNODE gpu$SGPU"
    bash "$GPURUN" -n "$SNODE" -g "$SGPU" -d -m 8000 -- python "$DIR/run.py" $ARGS \
        2>&1 | tee "$OUTDIR/$TAG.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

cat <<'NOTE'

  LAUNCHED DETACHED -- ~35 GPU-hours each.

  READ THE FIRST EPOCH BEFORE WALKING AWAY. transformer_sv/EXPERIMENT_PLAN.md
  section 2b records the lesson these very runs taught:

    "a mixed-precision NaN in an architecture the benchmark has never trained is
     invisible in the harness -- no exception, no exit code, just `Loss nan`
     scrolling past."

  Check:  grep -c 'nan' proposals/A*/out/*_ta_s42.train.log
  Expect: 0. If not, stop and diagnose -- do not let it run to the end.

  Re-run D1 after training to score the arms.
NOTE
