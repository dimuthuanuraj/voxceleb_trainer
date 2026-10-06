#!/usr/bin/env bash
# F1 -- Run V4 (Squeeze-Excitation) and V5 (Res2Net) properly under the v1 harness.
#
# Annual report section 11 item 9: "Run V4 (SE) and V5 (Res2Net) properly. The
# Phase II designs were never executed but the rationales are sound and both are
# cheap under the v1 harness."
#
# This is the ONLY part of Phase II worth resurrecting. Section 5 of the annual
# report establishes that the Jan-Jun 2026 experimental record is not
# reproducible from the repository and is contradicted by the July audit, so
# nothing from that period may be cited. But the DESIGNS were never the problem
# -- they were never run. V4 and V5 are standard, well-motivated, and cheap.
#
# CRITICAL: these are NEW runs under the v1 open-set protocol. They are not a
# recovery of any Phase II number and must never be presented as confirming one.
# Section 10.3 item 1 forbids citing the Phase II results at all.
#
# ResNetSE34V2 already provides SE blocks in the v1 registry, so V4 is largely a
# matter of running the existing architecture on the open-set splits; V5 needs
# the Res2Net variant confirmed present before launching.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/F1"; mkdir -p "$OUTDIR"

step "Which architectures are available?"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["EXPERIMENTS"])
import registry as r
have = list(r.ARCHITECTURES)
print(f"  registry architectures: {have}")
se = [a for a in have if "se34" in a.lower() or "se" == a.lower()[-2:]]
res2 = [a for a in have if "res2" in a.lower()]
print(f"\n  V4 candidates (SE)     : {se or 'NONE'}")
print(f"  V5 candidates (Res2Net): {res2 or 'NONE -- needs implementing'}")
if not res2:
    print("\n  V5 has no registry entry. Implement Res2Net as a new arch_key before")
    print("  launching, or run V4 alone and record V5 as deferred with the reason.")
PY

require_gpu_mb 8000
trainer_guard_snapshot
cd "$EXPERIMENTS"

step "V4 -- SE blocks, open-set, both languages"
set +e
python3 tools/gen_scripts.py --stage A --archs resnetse34v2 --conditions si ta --seeds 42 \
    2>&1 | tee "$OUTDIR/gen_v4.log"
for c in si ta; do
    E="A_resnetse34v2_aamsoftmax_${c}_s42"
    if [[ -f "results/$E/final.json" ]]; then
        log "  $E already complete"
        continue
    fi
    bash "$GPURUN" -d -m 8000 -- python "scripts/${E}.py" 2>&1 | tee "$OUTDIR/${E}.log"
done
set -e

trainer_guard_verify "registry\.py|scripts/|results/"

step "Status"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
EXPD = os.environ["EXPERIMENTS"]
done = {}
for c in ("si", "ta"):
    f = os.path.join(EXPD, "results", f"A_resnetse34v2_aamsoftmax_{c}_s42", "final.json")
    if os.path.exists(f):
        try:
            d = json.load(open(f))
            done[c] = d.get("eer") or d.get("EER") or d.get("eer_cosine")
        except Exception:
            pass
print(f"  V4 complete: {done}")
json.dump({"task": "F1", "ok": len(done) == 2, "V4_resnetse34v2": done,
           "V5_res2net": "deferred -- no registry entry; see gen log",
           "caveat": "NEW open-set runs. NOT a recovery of any Phase II number; "
                     "annual report 10.3 item 1 forbids citing that period."},
          open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY
