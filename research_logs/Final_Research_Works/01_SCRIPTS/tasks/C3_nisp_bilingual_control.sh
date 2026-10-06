#!/usr/bin/env bash
# C3 -- The nisp_tamil bilingual control.
#
# THE QUESTION IT SETTLES
# -----------------------
# Tamil verifies better than Sinhala on every system measured, and the gap GROWS
# with system quality: 1.03x on bispectrum, 6.41x on ECAPA. A language-intrinsic
# advantage would shift all systems together. What behaves this way is HEADROOM.
#
# Measured audio quality supports the confound reading -- Tamil is cleaner on all
# four periodicity measures (HNR 3.77 vs 3.00 dB, jitter 0.036 vs 0.057).
#
# So the annual report says twice that "Tamil verifies 6x better than Sinhala"
# MUST NOT be quoted as a language finding, and feature_level_testing/RESULTS.md
# section 5.5 names the fix:
#
#   "...should not be reported as a language effect without a within-corpus
#    bilingual control (nisp_tamil has 65 bilingual speakers and is the obvious
#    next step)."
#
# nisp_tamil holds the SAME speakers recorded in two languages, so language
# varies while speaker, channel and session do not. That is the only design here
# that can separate the two.
#
# PRE-REGISTERED PREDICTION (P23): the Tamil advantage will SHRINK BY MORE THAN
# HALF when measured within one corpus on the same speakers.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/C3"; mkdir -p "$OUTDIR"
SPLIT="$EXPERIMENTS/splits/nisp_tamil"

step "Preconditions"
[[ -d "$SPLIT" ]] || die "nisp_tamil split missing: $SPLIT"
ls "$SPLIT" | sed 's/^/  /'
for f in "$SPLIT"/*trials*.txt; do
    [[ -f "$f" ]] && log "  $(basename "$f"): $(wc -l < "$f") trials"
done

step "Is the corpus actually bilingual within speaker?"
python3 - "$SPLIT" "$OUTDIR" <<'PY'
import collections, glob, json, os, sys
split, outdir = sys.argv[1], sys.argv[2]
tl = os.path.join(split, "train_list.txt")
rows = []
for p in ([tl] if os.path.exists(tl) else []) + sorted(glob.glob(os.path.join(split, "*list*.txt"))):
    try:
        rows += [l.split() for l in open(p, encoding="utf-8") if l.strip()]
    except Exception:
        pass
spk_lang = collections.defaultdict(set)
for r in rows:
    if len(r) < 2:
        continue
    spk, path = r[0], r[-1]
    for tag in ("english", "tamil", "hindi", "/en/", "/ta/", "_en_", "_ta_"):
        if tag in path.lower():
            spk_lang[spk].add(tag.strip("/_"))
bi = {s: sorted(l) for s, l in spk_lang.items() if len(l) > 1}
print(f"  speakers seen            : {len(spk_lang)}")
print(f"  speakers with >1 language: {len(bi)}")
if bi:
    for s, l in list(bi.items())[:8]:
        print(f"    {s}: {l}")
    print("\n  Bilingual structure confirmed -- the within-speaker control is possible.")
else:
    print("\n  No within-speaker language variation detected from the paths.")
    print("  The language tag may be encoded elsewhere; inspect the list format")
    print("  before concluding the control cannot be built.")
json.dump({"speakers": len(spk_lang), "bilingual_speakers": len(bi),
           "example": {k: v for k, v in list(bi.items())[:20]}},
          open(os.path.join(outdir, "corpus_structure.json"), "w"), indent=2)
PY

require_gpu_mb 8000
trainer_guard_snapshot

step "Scoring the existing cross-lingual probe on nisp"
# The v1 evaluator already emits a `crosslingual_nisp` probe set for the ECAPA
# systems. Harvest those first -- they may answer the question with no new compute.
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
tr = os.environ["TRAINER"]
hits = glob.glob(os.path.join(tr, "experiments", "results", "*", "scores", "*nisp*.npz"))
hits += glob.glob(os.path.join(tr, "experiments", "results", "*", "*nisp*.json"))
print(f"  existing nisp probe artefacts: {len(hits)}")
for h in sorted(hits)[:12]:
    print(f"    {os.path.relpath(h, tr)}")
json.dump({"existing_nisp_artefacts": [os.path.relpath(h, tr) for h in hits]},
          open(os.path.join(outdir, "existing_probes.json"), "w"), indent=2)
PY

step "Evaluating the best system on the within-speaker bilingual trials"
cd "$EXPERIMENTS"
set +e
bash "$GPURUN" -m 8000 -- python tools/evaluate.py \
    --exp F_ssl_mhubert_lw_aamsoftmax_ta_s42 --probes --device cuda \
    2>&1 | tee "$OUTDIR/evaluate.log"
log "  rc=${PIPESTATUS[0]}"
set -e

trainer_guard_verify

step "Result"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
tr = os.environ["TRAINER"]
res = {"task": "C3", "ok": False, "prediction_P23":
       "the Tamil advantage shrinks by more than half within one corpus"}
hits = glob.glob(os.path.join(tr, "experiments", "results", "*", "*.json"))
nisp = [h for h in hits if "nisp" in h.lower()]
res["nisp_results"] = []
for h in sorted(nisp):
    try:
        d = json.load(open(h))
    except Exception:
        continue
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    if eer is not None:
        print(f"  {os.path.relpath(h, tr):70s} EER={eer}")
        res["nisp_results"].append({"file": os.path.relpath(h, tr), "eer": eer})
res["ok"] = bool(res["nisp_results"])
if not res["ok"]:
    print("  No nisp EER produced. Check evaluate.log; the probe set may need")
    print("  building with tools/build_extra_splits.py first.")
json.dump(res, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY

cat <<'NOTE'

  HOW TO READ THIS
  ----------------
  Compare the WITHIN-corpus si-vs-ta gap here against the ACROSS-corpus gap
  (si 3.63 / ta 0.57 on ECAPA, a 6.4x ratio). If the within-corpus ratio is much
  smaller, the across-corpus gap was mostly recording quality, and every Tamil
  number in the programme keeps its channel caveat permanently.

  Score prediction P23 in 02_STATE/predictions.json either way.
NOTE
