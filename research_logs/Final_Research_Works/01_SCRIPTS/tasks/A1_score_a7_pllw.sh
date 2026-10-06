#!/usr/bin/env bash
# A1 -- Score A7_pllw_combined_s42.
#
# WHY THIS IS FIRST
# -----------------
# The annual report names "A7 PLLW on combined_si_ta" as open item #3, describing
# it as "the direct extension of Finding III-3 for ~39 extra parameters. If the
# optimal layer differs between Sinhala and Tamil, one shared weight vector is
# leaving performance on the table."
#
# The run is ALREADY TRAINED. It reached epoch 53, early-stopped on patience 16,
# best validation EER 1.9217, and 54 checkpoints sit on disk. It has never been
# scored -- the dispatcher died before the eval pass. The GPU-days are spent;
# this recovers the result for ~25 GPU-minutes.
#
# Note on conditions: A7 is run on `combined` DELIBERATELY. Annual report Finding
# III-7 -- per-language layer weighting is undefined on a single-language
# condition, and the si arm that was attempted is retired in
# out/_degenerate_single_language/ for exactly that reason. Do not "fix" this by
# scoring the si arm.

. "$(dirname "$0")/../lib/common.sh"

TAG=A7_pllw_combined_s42
OUTDIR="$RESULTS/A1"; mkdir -p "$OUTDIR"

step "Preconditions"
CKPT_DIR="$PROPOSALS/A7_pllw/out/exps/$TAG/model"
[[ -d "$CKPT_DIR" ]] || die "no checkpoint directory: $CKPT_DIR"
N=$(ls "$CKPT_DIR"/model0*.model 2>/dev/null | wc -l)
log "checkpoints: $N"
(( N > 0 )) || die "no checkpoints for $TAG"
[[ -f "$CKPT_DIR/model_best.model" ]] && log "model_best.model present, val EER $(cat "$CKPT_DIR/model_best.eer" 2>/dev/null)"

require_gpu_mb 8000
trainer_guard_snapshot

step "Scoring $TAG"
cd "$PROPOSALS"
set +e
bash "$GPURUN" -m 8000 -- python evaluate_proposal.py --tag "$TAG" --device cuda \
    2>&1 | tee "$OUTDIR/score.log"
RC=${PIPESTATUS[0]}
set -e
log "evaluate_proposal rc=$RC"

trainer_guard_verify

step "Result"
python3 - "$TAG" "$OUTDIR" <<'PY'
import glob, json, os, sys
tag, outdir = sys.argv[1], sys.argv[2]
sl = os.environ["SL_SPV"]
hits = glob.glob(os.path.join(sl, "proposals/A7_pllw/out", f"{tag}*.eval.json"))
hits += glob.glob(os.path.join(sl, "proposals/A7_pllw/out/scores", tag, "*.json"))
if not hits:
    print("  no eval JSON produced -- see score.log")
    raise SystemExit(1)
summary = {"task": "A1", "tag": tag, "files": [os.path.relpath(h, sl) for h in hits]}
for h in sorted(hits):
    try:
        d = json.load(open(h))
    except Exception:
        continue
    print(f"\n  {os.path.basename(h)}")
    for k in ("eer", "EER", "eer_cosine", "eer_asnorm", "mindcf", "minDCF", "set", "condition"):
        if k in d:
            print(f"    {k:12s} {d[k]}")
    summary.setdefault("results", {})[os.path.basename(h)] = {
        k: v for k, v in d.items() if isinstance(v, (int, float, str))}
json.dump(summary, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  INTERPRETATION -- read before quoting this number
  -------------------------------------------------
  A7 learns a SEPARATE 13-layer weight vector per language. The question it
  answers is not "is A7 better" but "do Sinhala and Tamil want different SSL
  depths". Extract the fitted weight vectors and compare them; the EER is the
  secondary output. `tools/layer_weights.py` in the experiments tree reads them.

  Power caveat (annual report 6.8.2): at S ~ 91 an absolute EER carries +/-5-6 pp.
  Only the PAIRED contrast against the matched baseline resolves anything at
  this scale, and A7 adds ~39 parameters -- a difference that small needs the
  speaker-clustered bootstrap, not a raw comparison.
NOTE
