#!/usr/bin/env bash
# A5 -- Paired speaker-clustered bootstrap on the T1 transformer contrast.
#
# THE CAVEAT THIS REMOVES
# -----------------------
# T1 measured MFAConformer 4.950 against A_ecapa512's 4.285 on si -- a +0.665 pp
# gap confirming the strand's pre-registered prediction. The annual report's own
# caveat on it:
#
#   "single seed, one language. The paired bootstrap has not been run on this
#    contrast, and at S = 91 the +0.665 pp gap is NEAR THE RESOLUTION LIMIT
#    established in section 6.8.2. Treat as directional pending the ta arm and a
#    seed replication."
#
# WHY THE PAIRED BOOTSTRAP AND NOT A T-TEST
# -----------------------------------------
# Trials that share a speaker are correlated, so the effective sample size is
# bounded by SPEAKER count, not trial count. The project's bootstrap resamples
# SPEAKERS, not trials, and compares two systems on the SAME trials -- which is
# why both systems must have been scored on identical trial lists in identical
# order. It was validated against synthetic ground truth: EER matches the
# repository's own tuneThreshold exactly, a known injected difference is
# detected, and a system against itself correctly finds nothing. It is 2.6x
# tighter than absolute EER.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/A5"; mkdir -p "$OUTDIR"

step "Locating the paired score vectors"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
T, EXPD = os.environ["TRANSFORMER"], os.environ["EXPERIMENTS"]
pairs = [("MFAConformer_full_si_s42", "A_ecapa512_aamsoftmax_si_s42"),
         ("MFAConformer_full_ta_s42", "A_ecapa512_aamsoftmax_ta_s42")]
found = {}
for prop, base in pairs:
    p = glob.glob(os.path.join(T, "T*", "out", "scores", prop, "*.npz")) or \
        glob.glob(os.path.join(T, "T*", "out", f"{prop}*.npz"))
    b = glob.glob(os.path.join(EXPD, "results", base, "scores", "test.npz")) or \
        glob.glob(os.path.join(EXPD, "results", base, "*test*.npz"))
    print(f"  {prop:36s} scores={len(p)}   baseline {base}: {len(b)}")
    if p and b:
        found[prop] = {"proposal": p[0], "baseline": b[0], "base_exp": base}
json.dump(found, open(os.path.join(outdir, "pairs.json"), "w"), indent=2)
if not found:
    print("\n  No paired score vectors yet. A4 must score the arms first, and the")
    print("  scorer must have RETAINED the per-trial vectors -- a scalar EER cannot")
    print("  support this test.")
PY

step "Running the bootstrap"
cd "$EXPERIMENTS"
set +e
python3 tools/analyze.py --stage F --out "$OUTDIR/analysis" --n-boot 10000 \
    2>&1 | tee "$OUTDIR/analyze.log"
log "  rc=${PIPESTATUS[0]}"
set -e

python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
res = {"task": "A5", "ok": False,
       "contrast": "MFAConformer (from-scratch transformer) vs A_ecapa512 (matched CNN)",
       "single_seed_gap_pp": 0.665,
       "caveat_being_removed": "paired bootstrap not run; gap near the S=91 resolution limit"}
comp = glob.glob(os.path.join(outdir, "analysis", "comparisons.json"))
if comp:
    d = json.load(open(comp[0]))
    hits = [c for c in (d if isinstance(d, list) else d.get("comparisons", []))
            if isinstance(c, dict) and "MFAConformer" in json.dumps(c)]
    for c in hits:
        print(f"  {json.dumps(c)[:300]}")
    res["comparisons"] = hits
    res["ok"] = bool(hits)
else:
    print("  no comparisons.json produced -- see analyze.log")
json.dump(res, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  HOW TO READ THE OUTCOME
  -----------------------
  If the CI EXCLUDES zero: the transformer-loses-from-scratch result is
  reportable, and combined with Finding III-3 (a pretrained transformer read
  through learned layer weights reaches 2.369 on the same trials) it gives the
  programme its cleanest self-contained argument -- the value of a transformer on
  low-resource languages is in its PRETRAINING, not its architecture.

  If the CI INCLUDES zero: the gap is not resolvable at S = 91 and the claim must
  be stated as "no measurable difference", NOT as "the transformer lost". That is
  still a useful result -- a from-scratch transformer failing to BEAT a CNN at
  matched parameters supports the same conclusion more weakly -- but it must be
  worded honestly. Section 6.8.2 exists precisely so this distinction is not
  blurred.
NOTE
