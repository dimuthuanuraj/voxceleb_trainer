#!/usr/bin/env bash
# D3 -- A8 CD-NDAL, adv0.3 + control, on `combined`.
#
# Unblocked by W0.2 (the `test_normalize` delegation). Both arms previously died
# at their FIRST validation with:
#     AttributeError: 'LossFunction' object has no attribute 'test_normalize'
# which is why A1 has zero checkpoints -- the failure was at the start, not the
# end, so nothing was salvageable.
#
# CONDITION: `combined` ONLY. CD-NDAL is adversarial over CORPUS, and si/ta each
# contain a single corpus -- the discriminator would have one class and the
# gradient-reversal term would carry no information. The shim raises SystemExit
# rather than run, which is the guard added after A7 taught the lesson.
#
# The `control` arm is the identical model with the adversary disabled
# (adv_weight = 0.0). That is the matched comparison.
#
# PRE-REGISTERED PREDICTION (P25): the corpus adversary will NOT reverse PLDA's
# sign on si. MASTER_SYNTHESIS section 6 risk 2 names this as the falsifiable
# half of the channel-confound hypothesis, and A6 already showed the confound is
# real but SHALLOW -- linearly decodable at 91.2 %, yet not dominant in local
# neighbourhood structure. If CD-NDAL also fails to move it, the hypothesis is
# falsified and effort should move to label reliability, where A9 shows signal.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/D3"; mkdir -p "$OUTDIR"

step "Verifying the W0.2 fix"
grep -q "test_normalize" "$PROPOSALS/_trainer_shim/loss/cd_ndal_adv.py" || \
    die "cd_ndal_adv.py still lacks test_normalize -- run W0.2 first"
log "test_normalize delegation present"

step "Verifying the domain guard will pass on 'combined'"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["PROPOSALS"]))
from common.domains import DomainTable
tl = os.path.join(os.environ["EXPERIMENTS"], "splits", "combined_si_ta", "train_list.txt")
t = DomainTable(tl, "/")
print(f"  languages: {t.n_languages} {t.languages}")
print(f"  corpora  : {t.n_corpora} {t.corpora}")
if t.n_languages < 2 or t.n_corpora < 2:
    raise SystemExit("combined does not supply both a language and a corpus variable -- "
                     "DA2-LoRA is undefined here and the shim will refuse it")
print("  guard will pass: both anchors are defined on this condition")
PY

require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"
cd "$PROPOSALS"

for ARM in adv0.3 control; do
    step "A8 $ARM"
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
        python A8_cd_ndal/run.py --arm "$ARM" --condition combined \
        2>&1 | tee "$OUTDIR/A8_$ARM.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

cat <<'NOTE'

  LAUNCHED DETACHED -- ~35 GPU-hours each.

  The adv-vs-control contrast is the result, and P25 is scored on it. An absolute EER from either arm
  alone is uninterpretable at S ~ 91. Report the PAIRED difference with the
  speaker-clustered bootstrap, and state that the condition is `combined` -- a
  pooled-corpus condition in which domain is confounded with language, which is
  the constraint annual report 7.1 identifies as the project's clearest data
  priority (see H2).

  Re-run D3 after training to score both arms.
NOTE
