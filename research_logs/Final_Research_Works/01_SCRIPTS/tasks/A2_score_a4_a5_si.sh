#!/usr/bin/env bash
# A2 -- Score LSCAM_si_s42 (A4) and DKCAMPP_si_s42 (A5).
#
# Both trained to 60 epochs and were never scored. Best validation EER 4.3417
# (LSCAM) and 4.0416 (DKCAMPP). Their Tamil counterparts died on the fp16 mel
# overflow that W0.3 fixes; the Sinhala arms survived and are complete.
#
# A4 feeds paper P3, A5 feeds P4 (see 00_PLANNING/05_PUBLICATION_MAP.md).
#
# CAVEAT that must travel with A4's number: LS-CAM segments by LANGUAGE, and the
# `si` condition contains one language. Annual report Finding III-7 applies --
# the segmentation degenerates to a single segment and the method reduces toward
# its own trunk. Report the si arm as a TRUNK result (DK-CAM++ with LS-CAM's
# head), not as evidence about language segmentation. The informative A4 arm is
# the combined/ta one, which D1 produces.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/A2"; mkdir -p "$OUTDIR"
declare -a TAGS=(LSCAM_si_s42 DKCAMPP_si_s42)
declare -a DIRS=(A4_ls_cam A5_dk_campp)

step "Preconditions"
for i in 0 1; do
    d="$PROPOSALS/${DIRS[$i]}/out/exps/${TAGS[$i]}/model"
    [[ -d "$d" ]] || die "no checkpoints: $d"
    log "${TAGS[$i]}: $(ls "$d"/model0*.model 2>/dev/null | wc -l) ckpts, best val $(cat "$d/model_best.eer" 2>/dev/null)"
done

require_gpu_mb 8000
trainer_guard_snapshot

cd "$PROPOSALS"
for t in "${TAGS[@]}"; do
    step "Scoring $t"
    set +e
    bash "$GPURUN" -m 8000 -- python evaluate_proposal.py --tag "$t" --device cuda \
        2>&1 | tee "$OUTDIR/$t.log"
    log "  rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "Results"
python3 - "$OUTDIR" "${TAGS[@]}" <<'PY'
import glob, json, os, sys
outdir, *tags = sys.argv[1:]
sl = os.environ["SL_SPV"]
summary = {"task": "A2", "results": {}}
for tag in tags:
    hits = glob.glob(os.path.join(sl, "proposals/A*/out", f"{tag}*.eval.json"))
    hits += glob.glob(os.path.join(sl, "proposals/A*/out/scores", tag, "*.json"))
    print(f"\n  {tag}: {len(hits)} artefact(s)")
    for h in sorted(hits):
        try:
            d = json.load(open(h))
        except Exception:
            continue
        vals = {k: v for k, v in d.items()
                if k.lower() in ("eer", "eer_cosine", "eer_asnorm", "mindcf", "set")}
        if vals:
            print(f"    {os.path.basename(h)}: {vals}")
        summary["results"].setdefault(tag, {})[os.path.basename(h)] = vals
    if not hits:
        print("    NONE -- see the log")
json.dump(summary, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  Reminder on A4's si arm: report it as a trunk result, not as evidence about
  language segmentation. See the header of this script and annual report 7.1.
NOTE
