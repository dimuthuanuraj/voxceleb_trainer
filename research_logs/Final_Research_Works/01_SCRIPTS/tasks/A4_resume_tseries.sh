#!/usr/bin/env bash
# A4 -- Resume the five interrupted T-series arms, then score them.
#
# WHY RESUME RATHER THAN SCORE WHERE THEY STAND
# ---------------------------------------------
# The tempting move is to declare these "done enough" and score their best
# checkpoints. Measured on 2026-09-12, that is not safe:
#
#   arm                                ep  best@  since_best  converged?
#   MFAConformer_full_ta_s42           30     30          0   NO -- still improving
#   MFAConformer_no_attention_si_s42   27     19          8   no -- patience 16 not reached
#   MFAConformer_no_attention_ta_s42    7      7          0   NO -- barely started
#   Res2Former_tf0_si_s42              33     32          1   NO -- still improving
#   Res2Former_tf4_si_s42              32     21         11   no -- patience not reached
#
# Three of five had their BEST epoch as their LAST epoch when the cluster took
# them down. Scoring them now would put artificially pessimistic numbers in the
# same table as MFAConformer_full_si_s42, which did early-stop properly at 31.
# That is precisely the "good number measured on something other than what it
# claims" failure the programme exists to avoid.
#
# RESUMING IS FREE
# ----------------
# trainSpeakerNet.py:474-486 globs model0*.model in the save path, loads the
# highest, and sets the start epoch to its index + 1. So re-issuing the identical
# run.py command continues from where it stopped. No new code, no lost epochs.
#
# This script is RESUMABLE AND IDEMPOTENT: run it, let it work, re-run it after
# any reboot. It skips arms that have already reached their stopping criterion.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/A4"; mkdir -p "$OUTDIR"
cd "$TRANSFORMER"

# tag : experiment dir : run.py args
ARMS=(
  "MFAConformer_full_ta_s42|T1_mfa_conformer|--lang ta --arm full"
  "MFAConformer_no_attention_si_s42|T1_mfa_conformer|--lang si --arm no_attention"
  "MFAConformer_no_attention_ta_s42|T1_mfa_conformer|--lang ta --arm no_attention"
  "Res2Former_tf0_si_s42|T2_res2former|--lang si --arm tf0"
  "Res2Former_tf4_si_s42|T2_res2former|--lang si --arm tf4"
)

step "Current state of every arm"
python3 - <<'PY'
import glob, os, re
T = os.environ["TRANSFORMER"]
for exp in ("T1_mfa_conformer", "T2_res2former"):
    for d in sorted(glob.glob(os.path.join(T, exp, "out", "exps", "*", "model"))):
        tag = os.path.basename(os.path.dirname(d))
        eers = {}
        for f in glob.glob(os.path.join(d, "model0*.eer")):
            try:
                eers[int(re.search(r"model0*(\d+)\.eer", f).group(1))] = float(open(f).read().split()[0])
            except Exception:
                pass
        if not eers:
            print(f"  {tag:38s} no eval checkpoints")
            continue
        last, best = max(eers), min(eers, key=eers.get)
        scored = os.path.exists(os.path.join(T, exp, "out", f"{tag}.eval.json"))
        print(f"  {tag:38s} ep={last:2d} best={eers[best]:.4f}@{best:2d} "
              f"since_best={last-best:2d} scored={'yes' if scored else 'NO'}")
PY

require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"

for entry in "${ARMS[@]}"; do
    IFS='|' read -r TAG EXP ARGS <<< "$entry"
    step "$TAG"

    if [[ -f "$TRANSFORMER/$EXP/out/$TAG.eval.json" ]]; then
        log "  already scored -- skipping"
        continue
    fi

    # Already finished training? The dispatcher writes train.json only on a
    # clean exit, so its presence is the completion marker.
    if [[ -f "$TRANSFORMER/$EXP/out/$TAG.train.json" ]] && \
       grep -q '"trained"' "$TRANSFORMER/$EXP/out/$TAG.train.json" 2>/dev/null; then
        log "  training complete; scoring only"
    else
        log "  resuming training (auto-resume from the highest checkpoint)"
        set +e
        # shellcheck disable=SC2086
        SLOT="$(next_idle_slot)"
        if [[ -z "$SLOT" ]]; then
            warn "  every GPU is busy -- deferring the rest to the next pass"
            break
        fi
        read -r SNODE SGPU <<< "$SLOT"
        log "  slot: $SNODE gpu$SGPU"
        bash "$GPURUN" -n "$SNODE" -g "$SGPU" -d -m 8000 -- python "$EXP/run.py" $ARGS \
            2>&1 | tee -a "$OUTDIR/$TAG.resume.log"
        log "  dispatch rc=${PIPESTATUS[0]}"
        set -e
        log "  NOTE: launched DETACHED. Re-run A4 after it finishes to score it."
        continue
    fi

    log "  scoring"
    set +e
    bash "$GPURUN" -m 8000 -- python evaluate_transformer.py --tag "$TAG" --device cuda \
        2>&1 | tee "$OUTDIR/$TAG.eval.log"
    log "  eval rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "Summary"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
T = os.environ["TRANSFORMER"]
tags = ["MFAConformer_full_ta_s42", "MFAConformer_no_attention_si_s42",
        "MFAConformer_no_attention_ta_s42", "Res2Former_tf0_si_s42",
        "Res2Former_tf4_si_s42"]
res, done = {}, 0
for tag in tags:
    hits = glob.glob(os.path.join(T, "T*", "out", f"{tag}.eval.json"))
    if hits:
        d = json.load(open(hits[0]))
        res[tag] = d
        done += 1
        eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
        print(f"  {tag:38s} SCORED  eer={eer}")
    else:
        print(f"  {tag:38s} pending")
json.dump({"task": "A4", "scored": done, "of": len(tags), "results": res},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  {done}/{len(tags)} scored -> {outdir}/result.json")
if done < len(tags):
    print("  Re-run A4 once the detached training jobs finish.")
PY

cat <<'NOTE'

  WHAT THESE ARMS ANSWER
  ----------------------
  full_ta          does the si result (Conformer loses to ECAPA by 0.665 pp)
                   hold on Tamil? The annual report calls the si contrast
                   "directional pending the ta arm".
  no_attention_*   ABLATION, not a matched control -- separates "the Conformer
                   topology helps" from "self-attention helps". It is capacity-
                   unmatched and must say so wherever its number appears.
  tf0 vs tf4       the cleanest single-factor contrast in the folder: tf0 is an
                   asserted EXACT REDUCTION of tf4 (stem only), so the pair
                   isolates attention-as-refinement over conv features.
NOTE
