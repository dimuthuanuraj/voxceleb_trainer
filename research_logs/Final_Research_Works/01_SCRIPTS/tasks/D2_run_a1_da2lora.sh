#!/usr/bin/env bash
# D2 -- A1 DA2-LoRA, anchored + control, on `combined`.
#
# Unblocked by W0.2 (the `test_normalize` delegation). Both arms previously died
# at their FIRST validation with:
#     AttributeError: 'LossFunction' object has no attribute 'test_normalize'
# which is why A1 has zero checkpoints -- the failure was at the start, not the
# end, so nothing was salvageable.
#
# CONDITION: `combined` ONLY, and this is not a convenience choice.
# Annual report Finding III-7: DA2-LoRA has a language anchor AND a channel
# anchor, and si/ta each contain one language and one corpus. On a single-language
# condition both discriminators are constant and both reversal terms carry no
# information -- the method silently degrades to its own baseline while still
# emitting a plausible number. The shim now raises SystemExit rather than run.
#
# The `control` arm sets all four anchor weights to 0: plain LoRA, same streams,
# same rank, no anchors. That is the matched comparison; the anchored arm's
# number means nothing without it.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/D2"; mkdir -p "$OUTDIR"

step "Verifying the W0.2 fix"
grep -q "test_normalize" "$PROPOSALS/_trainer_shim/loss/da2_lora_adv.py" || \
    die "da2_lora_adv.py still lacks test_normalize -- run W0.2 first"
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

for ARM in anchored control; do
    step "A1 $ARM"
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
        python A1_da2_lora/run.py --arm "$ARM" --condition combined \
        2>&1 | tee "$OUTDIR/A1_$ARM.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

cat <<'NOTE'

  LAUNCHED DETACHED -- ~35 GPU-hours each.

  The anchored-vs-control contrast is the result. An absolute EER from either arm
  alone is uninterpretable at S ~ 91. Report the PAIRED difference with the
  speaker-clustered bootstrap, and state that the condition is `combined` -- a
  pooled-corpus condition in which domain is confounded with language, which is
  the constraint annual report 7.1 identifies as the project's clearest data
  priority (see H2).

  Re-run D2 after training to score both arms.
NOTE
