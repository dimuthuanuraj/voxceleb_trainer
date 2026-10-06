#!/usr/bin/env bash
# C6 -- The dialect gap: train on Indian Tamil, evaluate on Sri Lankan Tamil.
#
# MASTER_SYNTHESIS section 5: "Train on Indian Tamil, evaluate on Sri Lankan
# Tamil, report the number with the channel caveat. Nobody has published it."
#
# WHY IT MATTERS MORE THAN ITS COST
# ---------------------------------
# Annual report section 10.2 forbids "any Sri Lankan Tamil claim" because every
# large Tamil corpus in use is INDIAN Tamil. That is blocker B-2, and it is
# currently a silent limitation -- readers see Tamil numbers and assume Sri
# Lankan Tamil.
#
# This task converts the blocker from silent to MEASURED. A stated, quantified
# dialect penalty is an honest limitation; an unstated one is a misrepresentation.
# It needs no training: the Indian-Tamil-trained checkpoints already exist and
# slr65_tamil / slceleb2026 are already held-out probes.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/C6"; mkdir -p "$OUTDIR"

step "Which Tamil corpora are which dialect?"
cat <<'TXT'
  slr127_tamil   Indian Tamil  (IIT-Madras / OpenSLR)   -- TRAINING corpus
  kathbath       Indian Tamil  (AI4Bharat)              -- held-out probe
  nisp_tamil     Indian Tamil  (NISP, bilingual)        -- held-out probe
  slr65_tamil    Indian Tamil  (OpenSLR crowdsourced)   -- held-out probe
  slceleb2026    SRI LANKAN    (this project's corpus)  -- held-out probe

  Only the last is Sri Lankan, which is exactly why H1 (SLCeleb) is the binding
  blocker on any Sri Lankan claim.
TXT

require_gpu_mb 8000
trainer_guard_snapshot
cd "$EXPERIMENTS"

step "Evaluating Indian-Tamil-trained systems on every probe"
for s in F_ssl_mhubert_lw_aamsoftmax_ta_s42 A_ecapa512_aamsoftmax_ta_s42; do
    [[ -d "results/$s" ]] || { warn "missing $s"; continue; }
    set +e
    bash "$GPURUN" -m 8000 -- python tools/evaluate.py --exp "$s" --probes --device cuda \
        2>&1 | tee "$OUTDIR/${s}.log"
    log "  rc=${PIPESTATUS[0]}"
    set -e
done

trainer_guard_verify

step "The gap"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
EXPD = os.environ["EXPERIMENTS"]
DIALECT = {"slr127": "indian", "kathbath": "indian", "nisp": "indian",
           "slr65": "indian", "slceleb": "SRI LANKAN", "test": "indian(in-domain)"}
rows = {}
for f in glob.glob(os.path.join(EXPD, "results", "*ta_s42*", "*.json")):
    exp = os.path.basename(os.path.dirname(f))
    setn = os.path.splitext(os.path.basename(f))[0]
    try:
        d = json.load(open(f))
    except Exception:
        continue
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    if eer is None:
        continue
    dia = next((v for k, v in DIALECT.items() if k in setn.lower()), None)
    if dia:
        rows.setdefault(exp, {})[setn] = (float(eer), dia)

out = {"task": "C6", "ok": False, "systems": {}}
for exp, sets in sorted(rows.items()):
    print(f"\n  {exp}")
    ind, sl = [], []
    for setn, (eer, dia) in sorted(sets.items()):
        mark = "  <-- SRI LANKAN" if "SRI" in dia else ""
        print(f"    {setn:32s} {dia:18s} EER={eer:7.3f}{mark}")
        (sl if "SRI" in dia else ind).append(eer)
    out["systems"][exp] = {k: v[0] for k, v in sets.items()}
    if ind and sl:
        gap = sum(sl) / len(sl) - sum(ind) / len(ind)
        print(f"    Indian mean {sum(ind)/len(ind):.3f} | Sri Lankan {sum(sl)/len(sl):.3f}"
              f" | DIALECT GAP {gap:+.3f} pp")
        out["systems"][exp]["dialect_gap_pp"] = gap
        out["ok"] = True
json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
if not out["ok"]:
    print("\n  No Sri Lankan probe scored. SLCeleb (H1) is the binding blocker --")
    print("  without it this measurement cannot be made, which is itself the finding")
    print("  to report.")
PY

cat <<'NOTE'

  THE CHANNEL CAVEAT IS MANDATORY HERE. A cross-corpus difference is dialect
  PLUS channel, and the v1 channel probe predicts corpus-of-origin from the
  embedding at 91.2 % against 20 % chance. So the number this produces is an
  UPPER BOUND on the dialect effect, not an estimate of it. Say so.
NOTE
