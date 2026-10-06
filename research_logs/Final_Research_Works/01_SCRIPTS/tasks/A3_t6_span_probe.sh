#!/usr/bin/env bash
# A3 -- T6 span probe on MFAConformer_full_si_s42.
#
# WHY THIS IS THE BEST COST/BENEFIT TASK IN THE REGISTER
# ------------------------------------------------------
# It requires NO TRAINING. It caps attention span on an already-trained,
# already-scored checkpoint and re-scores, so it costs GPU-minutes and runs on
# a T4 today.
#
# `transformer_sv/EXPERIMENT_PLAN.md` P3 states the stakes:
#
#   "It tests the property transformers are *chosen* for. If a 41-frame cap
#    (ECAPA-1024's receptive field) costs nothing, the long-range advantage is
#    not doing work on these corpora, and that single number reframes the whole
#    strand -- including for the SSL systems, whose front-ends are also
#    attention-based."
#
# The mechanism exists and is asserted: MultiHeadSelfAttention.set_mode() switches
# cost profile AFTER training without touching parameters, and window attention
# with w >= T-1 is asserted BIT-IDENTICAL to full -- so the widest span is an
# exact control, not an approximation.
#
# PRE-REGISTERED PREDICTION (P22, recorded before the run):
#   a 41-frame cap will cost < 0.3 pp on this model.
# Justification: Finding III-8 showed the quadratic term is a minority of cost
# at these lengths, and a from-scratch Conformer already LOST to ECAPA here --
# if long range were doing work, it should not have.

. "$(dirname "$0")/../lib/common.sh"

TAG=MFAConformer_full_si_s42
OUTDIR="$RESULTS/A3"; mkdir -p "$OUTDIR"

step "Preconditions"
CK="$TRANSFORMER/T1_mfa_conformer/out/exps/$TAG/model"
[[ -d "$CK" ]] || die "no checkpoints for $TAG"
log "checkpoints: $(ls "$CK"/model0*.model 2>/dev/null | wc -l), best val $(cat "$CK/model_best.eer" 2>/dev/null)"
[[ -f "$TRANSFORMER/T1_mfa_conformer/out/$TAG.eval.json" ]] \
    || warn "$TAG has no eval.json -- the probe compares against it; run evaluate_transformer first"

step "Probe machinery self-check"
cd "$TRANSFORMER"
python3 T6_span_probe/run.py --self-check 2>&1 | tee "$OUTDIR/self_check.log" || \
    warn "self-check reported a problem -- read the log before trusting the numbers"

require_gpu_mb 8000
trainer_guard_snapshot

# ---------------------------------------------------------------------------
# HARDWARE-REPRODUCIBILITY GUARD  [added 2026-09-14 after the first A3 run]
#
# T6 refuses to write a result unless its UNCAPPED rung reproduces the published
# eval.json EER to within TOL=1e-6. That guard is correct and must not be
# disabled: a capped EER is only interpretable against an uncapped one from the
# same code path, which is the defect class the 22 August evaluation-integrity
# rebuild closed.
#
# What it did not anticipate is the probe running on DIFFERENT SILICON from the
# scorer. Measured on the first attempt, 2026-09-12:
#
#   published (eval.json)  4.950095775783849   <- scored on node-4's A40
#   probe uncapped         4.960177437241657   <- probed on node-1's T4
#   difference             0.010082 pp = EXACTLY 2 trials of 19,838
#
# 231 tensors loaded identically from the same checkpoint, so this is cuDNN
# kernel selection and reduction order across architectures, not a rebuild error.
#
# Resolution, in order of preference:
#   A. probe on a large card -- the class that produced the published number
#   B. re-score on THIS card first, so baseline and probe share hardware
#
# B is taken automatically below when no large card is available. The published
# eval.json is BACKED UP and RESTORED afterwards: T1's 4.950 is quoted in the
# annual report and must not be silently overwritten by a T4 re-score.
# ---------------------------------------------------------------------------
EVAL_JSON="$TRANSFORMER/T1_mfa_conformer/out/$TAG.eval.json"
HAVE_BIG=$(probe_gpus | awk '$3 > 20000 {print "yes"; exit}')
RESTORE_EVAL=0

if [[ "$HAVE_BIG" == "yes" ]]; then
    log "a >20 GB card is available -- probing on the same class that scored T1"
else
    warn "only small cards available; the published number came from an A40."
    warn "Re-scoring T1 on this card so the probe's baseline shares its hardware."
    cp -p "$EVAL_JSON" "$EVAL_JSON.published-a40" 2>/dev/null || true
    RESTORE_EVAL=1
    set +e
    bash "$GPURUN" -m 8000 -- python evaluate_transformer.py --tag "$TAG" --device cuda --redo \
        2>&1 | tee "$OUTDIR/rescore.log"
    log "  re-score rc=${PIPESTATUS[0]}"
    set -e
    python3 "$SCRIPTS/lib/compare_eer.py" "$EVAL_JSON" "$EVAL_JSON.published-a40" || true
fi

step "Running the span probe"
set +e
bash "$GPURUN" -m 8000 -- python T6_span_probe/run.py --tag "$TAG" --device cuda \
    2>&1 | tee "$OUTDIR/probe.log"
RC=${PIPESTATUS[0]}
set -e
log "probe rc=$RC"

trainer_guard_verify

if (( RESTORE_EVAL )); then
    step "Restoring the published eval.json"
    # The probe has taken its baseline; put the A40-scored number back so
    # the annual report's 4.950 stays the number of record.
    mv "$EVAL_JSON.published-a40" "$EVAL_JSON"
    log "  published (A40) eval.json restored"
fi

step "Result"
python3 - "$TAG" "$OUTDIR" <<'PY'
import glob, json, os, sys
tag, outdir = sys.argv[1], sys.argv[2]
sl = os.environ["SL_SPV"]
hits = [h for h in glob.glob(os.path.join(sl, "transformer_sv/T6_span_probe/out/*.json"))
        if tag in os.path.basename(h) or "self_check" not in os.path.basename(h)]
hits = [h for h in hits if "self_check" not in os.path.basename(h)]
if not hits:
    print("  no probe output -- see probe.log")
    raise SystemExit(1)
summary = {"task": "A3", "tag": tag, "prediction_P22": "a 41-frame cap costs < 0.3 pp",
           "files": [os.path.relpath(h, sl) for h in hits]}
for h in sorted(hits):
    d = json.load(open(h))
    print(f"\n  {os.path.basename(h)}")
    print("  " + json.dumps(d, indent=2)[:2500])
    summary.setdefault("probe", {})[os.path.basename(h)] = d
json.dump(summary, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  SCORE THE PREDICTION
  --------------------
  P22 said: a 41-frame cap costs < 0.3 pp. Write the outcome into
  02_STATE/predictions.json whichever way it came out. The annual report scores
  ~40 % of its pre-registered predictions as falsified and treats that as a
  healthy signal -- a plan that drops its own falsified predictions would be the
  first regression from that standard.

  If the cap costs NOTHING: the long-range advantage is not doing work on these
  corpora, and that conclusion extends to the SSL front-ends too.
  If the cap HURTS: attention span is load-bearing, and T3/T4 become more
  interesting rather than less.
NOTE
