#!/usr/bin/env bash
# G1 -- Audit item R7: re-run the Phase I 10.32 % headline at 3 seeds, or retire it.
#
# WHY IT CANNOT SIMPLY BE LEFT ALONE
# ----------------------------------
# G2-G6 scanned the project's documents and found the 10.32 % figure in 47 places.
# It is the most widely propagated number in the repository and it is, per the
# annual report section 10.3 item 3:
#
#   * unreplicated and single-seed;
#   * from before --deterministic existed;
#   * in internal conflict -- one log in the same record gives 14.62 %;
#   * associated with a suspected duplicated result set.
#
# "Audit item R7 applies: re-run 3 seeds or retire the claim."
#
# There is no third option. A number quoted 47 times either has evidence or it
# does not, and leaving it in place while knowing it is unsupported is the exact
# failure section 10.3 exists to correct.
#
# RETIREMENT IS A LEGITIMATE AND CHEAP OUTCOME. Phase I studied English VoxCeleb
# with a lightweight-distillation question the field has since settled, and the
# programme's subject is now Sinhala/Tamil speaker verification. Spending 30
# GPU-hours to rescue an English side-result may be the wrong call -- but it must
# be a DECISION, recorded, not a drift.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/G1"; mkdir -p "$OUTDIR"
MODE="${G1_MODE:-}"      # set G1_MODE=retire or G1_MODE=rerun

step "How widely is the claim propagated?"
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["RESULTS"], "G2-G6", "result.json")
if os.path.exists(p):
    d = json.load(open(p))
    n = len(d.get("occurrences", {}).get("E3", []))
    print(f"  10.32 % appears in {n} place(s) across project documents")
    for x in d["occurrences"].get("E3", [])[:10]:
        print(f"    {x['file']}:{x['line']}")
    if n > 10:
        print(f"    ... and {n - 10} more")
else:
    print("  run G2-G6 first to enumerate the occurrences")
PY

if [[ -z "$MODE" ]]; then
cat <<'ASK'

  ================================================================
  THIS TASK REQUIRES A DECISION, NOT A DEFAULT.
  ================================================================

  Option A -- RETIRE (0 GPU-hours)
      G1_MODE=retire python3 runner.py --run G1
    Records the claim as retired, and the errata register becomes the
    authoritative statement. Justified if Phase I is out of scope for the
    thesis and the papers, which it largely is.

  Option B -- RE-RUN (~30 GPU-hours)
      G1_MODE=rerun python3 runner.py --run G1
    Three seeds under --deterministic on the mini-VoxCeleb protocol. Produces
    either a replicated number with an interval, or a clean falsification --
    both are publishable, and the falsification would be the more interesting.

  Not choosing is the one option that is not available: the number is currently
  quoted in project documents without the caveat it requires.

ASK
exit 1
fi

if [[ "$MODE" == "retire" ]]; then
    step "Retiring the claim"
    python3 - "$OUTDIR" <<'PY'
import datetime, json, os, sys
outdir = sys.argv[1]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
rec = {
  "task": "G1", "ok": True, "decision": "RETIRED", "decided": stamp,
  "claim": "Phase I MLP-Mixer V2 student: 10.32 % EER",
  "reason": ("Single seed, unreplicated, predates --deterministic, internally "
             "contradicted (14.62 % in one log of the same record), and associated "
             "with a suspected duplicated result set. Audit item R7 gave two "
             "options -- three seeds or retirement -- and retirement was chosen "
             "because Phase I's English lightweight-distillation question is "
             "outside the scope of the thesis and the four-paper roadmap."),
  "what_replaces_it": ("Nothing. Phase I has no replicated student EER. What Phase I "
                       "DID establish and which survives: Finding I-1 (embedding "
                       "distillation requires an angular loss), Finding I-2 (the "
                       "optimal distillation weight tracks the student/teacher "
                       "capacity ratio), Finding I-3 (nested/dense connectivity is "
                       "domain-specific). Those are mechanisms, derived and "
                       "confirmed, and they do not depend on the headline number."),
  "source": "annual report 10.3 item 3; audit item R7",
}
json.dump(rec, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print("  RETIRED. Recorded in result.json.")
print("\n  NEXT: re-run G2-G6 so ERRATA.md carries the decision, then correct or")
print("  caveat the 47 document occurrences.")
PY
    exit 0
fi

step "Re-running Phase I at 3 seeds"
require_gpu_mb 8000
trainer_guard_snapshot
warn "Phase I ran on mini-VoxCeleb with the Phase I code path."
warn "Confirm the recipe and data are still present before trusting a comparison."
cd "$TRAINER"
for SEED in 42 123 7; do
    log "  seed $SEED -- implement the Phase I recipe invocation here"
done
trainer_guard_verify
python3 - "$OUTDIR" <<'PY'
import json, os, sys
outdir = sys.argv[1]
json.dump({"task": "G1", "ok": False, "decision": "RERUN",
           "status": "scaffolded -- the Phase I recipe invocation must be filled in",
           "note": "Phase I used mini-VoxCeleb and a distillation code path that the "
                   "v1 harness does not cover. Confirm data and recipe exist before "
                   "committing 30 GPU-hours; retirement remains the cheaper option."},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"  -> {outdir}/result.json")
PY
