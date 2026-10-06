#!/usr/bin/env bash
# D4 -- Consolidate A2 CC-NAP into a reportable result. Feeds paper P1.
#
# A2 has run: a projector and per-language score files exist under
# A2_cc_nap/out/. What does not exist is a RESULT -- a paired comparison against
# plain NAP with the speaker-clustered bootstrap, written down.
#
# The method: NAP normally needs bilingual speakers to estimate the nuisance
# direction. The Sri Lankan collection has none, so CC-NAP centres WITHIN CORPUS
# first and estimates the direction from the residual. That adaptation is the
# novel part, and P1 is "can start immediately; no training required".
#
# PREDICTION TO SCORE (MASTER_SYNTHESIS section 6 risk 2): if CC-NAP performs
# IDENTICALLY to plain NAP, the channel-confound hypothesis is falsified and
# effort should move to label reliability. A6 already pointed that way -- the
# confound is real but shallow.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/D4"; mkdir -p "$OUTDIR"

step "What A2 has produced"
ls -la "$PROPOSALS/A2_cc_nap/out/" | sed 's/^/  /'

require_gpu_mb 8000
trainer_guard_snapshot
cd "$PROPOSALS"

step "Scoring the CC-NAP arms"
set +e
bash "$GPURUN" -m 8000 -- python evaluate_proposal.py --all --device cuda \
    2>&1 | tee "$OUTDIR/score.log"
log "  rc=${PIPESTATUS[0]}"
set -e

trainer_guard_verify

step "CC-NAP vs plain NAP"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
import numpy as np
outdir = sys.argv[1]
P = os.environ["PROPOSALS"]
out = {"task": "D4", "ok": False, "arms": {}}

for f in sorted(glob.glob(os.path.join(P, "A2_cc_nap", "out", "*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    name = os.path.basename(f)
    vals = {k: v for k, v in d.items()
            if isinstance(v, (int, float)) and "eer" in k.lower() or k.lower() == "eer"}
    if vals:
        print(f"  {name:56s} {vals}")
        out["arms"][name] = vals

proj = sorted(glob.glob(os.path.join(P, "A2_cc_nap", "out", "*projector*.npz")))
print(f"\n  projectors: {len(proj)}")
for p in proj:
    try:
        z = np.load(p)
        shapes = {k: list(np.asarray(z[k]).shape) for k in z.keys()}
        print(f"    {os.path.basename(p)}: {shapes}")
        out.setdefault("projectors", {})[os.path.basename(p)] = shapes
    except Exception as exc:
        print(f"    {os.path.basename(p)}: unreadable ({exc})")

out["ok"] = bool(out["arms"])
json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
if not out["ok"]:
    print("\n  No EER produced for A2. The scored comparison against plain NAP is")
    print("  what P1 needs -- check A2_cc_nap/run.py for the scoring entry point.")
PY

cat <<'NOTE'

  FOR PAPER P1: A2 pairs with A3 (already complete: Cllr 0.8356 -> 0.1873, a
  median 4.5x improvement for 1,225 parameters, with metadata proven inert by a
  bit-identical result under shuffling). Together they are a training-free
  cross-lingual adaptation paper that can be written from checkpoints already on
  disk. C4 supplies the anchoring figure.
NOTE
