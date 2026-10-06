#!/usr/bin/env bash
# B4 -- Seed replication for the top 3 systems per language. Report open item #2.
#
# WHY THIS IS NOT OPTIONAL
# ------------------------
# Every v1, proposal and T-series number is SINGLE SEED. The measured power
# ceiling (annual report 6.8.2) says a single absolute EER at S ~ 91 carries
# +/-5-6 pp, and that effective sample size is bounded by SPEAKER count, not
# trial count. Differences of 0.09 pp are not resolvable; 2.0 pp differences are.
#
# Without replication, Stage D's question -- "is the ordering stable?" -- is
# simply unanswered, and every ranking in the programme is an assertion.
#
# PRE-REGISTERED PREDICTION (P24): the ORDERING of the top 3 per language will be
# stable, but at least one ABSOLUTE EER will move by more than 0.5 pp.
#
# Cost: 12 runs, ~180 GPU-h. This is the largest single block in the register and
# it is why W0.4 matters -- on 2 x T4 it is 4 days of pure compute.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/B4"; mkdir -p "$OUTDIR"
SEEDS="123 7"

step "Selecting the top 3 per language from measured validation EER"
cd "$EXPERIMENTS"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
exp = os.environ["EXPERIMENTS"]
best = {"si": [], "ta": []}
for fj in glob.glob(os.path.join(exp, "results", "*", "final.json")):
    name = os.path.basename(os.path.dirname(fj))
    if ".superseded" in name or "__smoke" in name:
        continue
    lang = "si" if "_si_" in name else ("ta" if "_ta_" in name else None)
    if not lang:
        continue
    try:
        d = json.load(open(fj))
    except Exception:
        continue
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    if eer is None:
        for sub in d.values():
            if isinstance(sub, dict):
                eer = sub.get("eer") or sub.get("EER")
                if eer is not None:
                    break
    if eer is not None:
        best[lang].append((float(eer), name))
sel = {}
for lang in ("si", "ta"):
    best[lang].sort()
    sel[lang] = [n for _, n in best[lang][:3]]
    print(f"  {lang}: " + ", ".join(f"{n} ({e:.3f})" for e, n in best[lang][:3]))
json.dump(sel, open(os.path.join(outdir, "selection.json"), "w"), indent=2)
PY

require_gpu_mb 8000
trainer_guard_snapshot

step "Generating seed replicas"
python3 tools/gen_scripts.py --stage D --seeds $SEEDS --top 3 \
    2>&1 | tee "$OUTDIR/gen.log" || warn "gen_scripts reported a problem -- check --stage D semantics"

step "Dispatching"
set +e
python3 tools/run_queue.py --stage D --resume --keep-running 2>&1 | tee "$OUTDIR/queue.log"
log "  rc=${PIPESTATUS[0]}"
set -e

trainer_guard_verify "registry\.py|scripts/|results/"

step "Aggregating whatever has finished"
python3 tools/aggregate_seeds.py 2>&1 | tee "$OUTDIR/aggregate.log" || \
    warn "aggregate_seeds reported a problem"

python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
exp = os.environ["EXPERIMENTS"]
runs = [os.path.basename(os.path.dirname(f))
        for f in glob.glob(os.path.join(exp, "results", "*", "final.json"))
        if "_s123" in f or "_s7_" in f or f.endswith("_s7/final.json")]
print(f"  seed-replica runs complete: {len(runs)}")
for r in sorted(runs):
    print(f"    {r}")
json.dump({"task": "B4", "ok": len(runs) >= 12, "completed_replicas": sorted(runs),
           "target": 12, "prediction_P24":
           "ordering stable, at least one absolute EER moves > 0.5 pp"},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
if len(runs) < 12:
    print(f"  {12 - len(runs)} replica(s) outstanding. Re-run B4 -- it resumes.")
PY
