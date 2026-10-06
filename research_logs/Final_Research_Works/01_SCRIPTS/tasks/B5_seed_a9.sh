#!/usr/bin/env bash
# B5 -- Seed replication for A9 ARI-SubCenter. Report open item #2.
#
# A9 is the project's FIRST POSITIVE proposal result and it is mechanistically
# coherent with the project's own data audit: si 4.375 -> 3.962, Delta -0.413 pp,
# CI [-0.677, -0.015], p = 0.038; ta null (0.884 -> 0.934, p = 0.668).
#
# The asymmetry follows the QC audit exactly. SLR127 (Tamil) is the corpus whose
# identity error the audit FOUND AND CORRECTED (638 speakers had been collapsed
# into 531). SLR52 (Sinhala) is the corpus the audit FLAGGED AS UNRESOLVED.
# Sub-centres are the standard response to residual within-class identity
# heterogeneity, and they improved the corpus where it is suspected but
# uncorrected while doing nothing where it was found and fixed.
#
# WHY IT NEEDS SEEDS: the annual report holds it as "promising, not settled --
# p = 0.038 with an interval reaching -0.015 pp, single seed." An interval whose
# upper bound is -0.015 pp is one seed away from crossing zero.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/B5"; mkdir -p "$OUTDIR"
require_gpu_mb 8000
trainer_guard_snapshot
log "GPU slots: $(refresh_slots) total, $(idle_slot_count) idle"
cd "$PROPOSALS"

for SEED in 123 7; do
  for LANG in si ta; do
    TAG="A9_subcenter_measured_${LANG}_s${SEED}"
    step "$TAG"
    if [[ -f "A9_ari_subcenter/out/${TAG}.eval.json" ]]; then
        log "  already scored -- skipping"; continue
    fi
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
        python A9_ari_subcenter/run.py --lang "$LANG" --seed "$SEED" \
        2>&1 | tee "$OUTDIR/$TAG.log"
    log "  dispatch rc=${PIPESTATUS[0]}"
    set -e
  done
done

trainer_guard_verify

step "Aggregating across seeds"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, statistics as st, sys
outdir = sys.argv[1]
P = os.environ["PROPOSALS"]
by_lang = {}
for f in sorted(glob.glob(os.path.join(P, "A9_ari_subcenter", "out", "*.eval.json"))):
    tag = os.path.basename(f).replace(".eval.json", "")
    lang = "si" if "_si_" in tag else ("ta" if "_ta_" in tag else "?")
    try:
        d = json.load(open(f))
    except Exception:
        continue
    eer = d.get("eer") or d.get("EER") or d.get("eer_cosine")
    if eer is not None:
        by_lang.setdefault(lang, []).append((tag, float(eer)))
out = {"task": "B5", "seeds": by_lang}
for lang, rows in sorted(by_lang.items()):
    vals = [v for _, v in rows]
    print(f"\n  {lang}: {len(vals)} seed(s)")
    for t, v in rows:
        print(f"    {t:44s} {v:.4f}")
    if len(vals) > 1:
        m, s = st.mean(vals), st.stdev(vals)
        print(f"    mean {m:.4f} +/- {s:.4f}")
        out.setdefault("summary", {})[lang] = {"n": len(vals), "mean": m, "sd": s}
        if lang == "si" and s > 0.413:
            print("    NOTE: seed spread EXCEEDS the -0.413 pp effect A9 claims.")
            print("    The single-seed result does not survive replication as stated.")
out["ok"] = sum(len(v) for v in by_lang.values()) >= 6   # 3 seeds x 2 languages
json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
PY
