#!/usr/bin/env bash
# F3 -- The mel-scale question. Annual report open item #11.
#
#   "The mel-scale question is untested since RawNet3 was removed on request --
#    whether a scale fitted to English and European perceptual data suits Sinhala
#    and Tamil remains open, and is a genuinely publishable question."
#
# WHY IT IS A REAL QUESTION, NOT A CURIOSITY
# ------------------------------------------
# The mel scale is an empirical fit to perceptual experiments run on speakers of
# English and European languages. Every system in this benchmark reads audio
# through it. The programme has already shown, twice, that a default fitted
# elsewhere carries a cost that scales with linguistic distance:
#
#   Finding III-3  the conventional last-layer SSL default is the WORST choice
#   Finding III-4  its penalty scales with distance: 1.04x en, 1.88x si, 5.23x ta
#
# The mel scale is the same shape of assumption one level lower in the stack, and
# it has never been tested here. Finding III-4 is the reason to expect an effect
# and the reason to expect it to be larger on Tamil than on Sinhala.
#
# DESIGN: hold everything else fixed and vary only the filterbank warping --
# mel (control), linear, and Bark or ERB. Front-end parity against the trainer's
# own mel factory is already asserted by the strands' self-tests, so the control
# arm is exact rather than approximate.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/F3"; mkdir -p "$OUTDIR"

step "Where the filterbank is defined"
grep -rn "melscale\|mel_scale\|MelSpectrogram\|n_mels" "$TRAINER/models/_frontend.py" 2>/dev/null | head -10 || \
    warn "no _frontend.py hits -- locate the filterbank before implementing"

cat <<'PLAN'

  IMPLEMENTATION PLAN (this task is scaffolding; the arms need writing)
  --------------------------------------------------------------------
  1. Add a `--fb_scale {mel,linear,bark,erb}` front-end option that changes ONLY
     the filter centre frequencies. Same n_mels, same window, same hop, same
     normalisation -- otherwise the contrast is confounded.

  2. Assert the control is an EXACT reduction: `--fb_scale mel` must be
     bit-identical to the current path. Protocol rule 8. Without that assertion
     the comparison is worthless.

  3. Run 3 scales x 2 languages on A_ecapa512 (the cheapest trustworthy backbone,
     and the one the capacity null says is not capacity-limited).

  4. Score with the paired speaker-clustered bootstrap on identical trials.

  5. PRE-REGISTER before running. Finding III-4 gives a specific, falsifiable
     expectation: if a non-mel scale helps, it should help TAMIL MORE THAN
     SINHALA, because the penalty of an English-fitted default scales with
     linguistic distance. A uniform effect across languages would falsify the
     mechanism even if the scale change itself wins.

  Cost: ~40 GPU-hours for 6 arms.

PLAN

python3 - "$OUTDIR" <<'PY'
import json, os, sys
outdir = sys.argv[1]
json.dump({"task": "F3", "ok": False, "status": "scaffolded, not implemented",
           "source": "annual report section 11 item 11",
           "blocking": "needs a --fb_scale front-end option and an exactness assertion",
           "prediction_to_register":
               "if a non-mel scale helps, it helps Tamil more than Sinhala "
               "(Finding III-4: the penalty of an English-fitted default scales "
               "with linguistic distance). A uniform effect falsifies the mechanism."},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"  -> {outdir}/result.json (task scaffolded, marked not-ok until implemented)")
PY
