#!/usr/bin/env bash
# B3 -- The 2x2 analysis: {last-layer, layer-weighted} x {frozen, fine-tuned}.
#
# This is where the arbiter run becomes an answer rather than a number. The four
# cells, all on identical trials:
#
#                      FROZEN                    FINE-TUNED
#   last-layer      ssl_wavlm (A-stage)        ssl_wavlm_ft      si 5.404
#   layer-weighted  ssl_wavlm_lw   si 2.369    ssl_wavlm_lw_ft   <- B1/B2
#
# The two main effects and their interaction are the result:
#   * layer weighting effect, held at each freeze setting
#   * fine-tuning effect, held at each read-out setting
#   * INTERACTION -- does layer weighting still pay once the encoder can adapt?
#
# If the interaction is strongly negative, layer weighting is a SUBSTITUTE for
# fine-tuning rather than a complement, and the cheap frozen configuration is the
# right recommendation for low-resource deployment. That is the outcome with the
# most practical value, and P21 predicts it.

. "$(dirname "$0")/../lib/common.sh"

OUTDIR="$RESULTS/B3"; mkdir -p "$OUTDIR"
cd "$EXPERIMENTS"

step "Running the paired analysis over all SSL systems"
set +e
python3 tools/analyze.py --stage F --out "$OUTDIR/analysis" 2>&1 | tee "$OUTDIR/analyze.log"
log "  rc=${PIPESTATUS[0]}"
set -e

step "Assembling the 2x2"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
outdir = sys.argv[1]
exp = os.environ["EXPERIMENTS"]

CELLS = {
    ("last", "frozen"):      "A_ssl_wavlm_aamsoftmax_{lang}_s42",
    ("last", "finetuned"):   "F_ssl_wavlm_ft_aamsoftmax_{lang}_s42",
    ("weighted", "frozen"):  "F_ssl_wavlm_lw_aamsoftmax_{lang}_s42",
    ("weighted", "finetuned"): "F_ssl_wavlm_lw_ft_aamsoftmax_{lang}_s42",
}

def eer_of(exp_id):
    for cand in (os.path.join(exp, "results", exp_id, "final.json"),
                 os.path.join(exp, "results", exp_id, "test.json")):
        if os.path.exists(cand):
            try:
                d = json.load(open(cand))
            except Exception:
                continue
            for k in ("eer", "EER", "eer_cosine", "test_eer"):
                if k in d:
                    return float(d[k])
            for sub in d.values():
                if isinstance(sub, dict):
                    for k in ("eer", "EER", "eer_cosine"):
                        if k in sub:
                            return float(sub[k])
    return None

out = {"task": "B3", "ok": False, "cells": {}, "prediction_P21":
       "lw+ft beats lw+frozen on si, but by less than the last-layer ft gain"}
complete = True
for lang in ("si", "ta"):
    print(f"\n  --- {lang} ---")
    print(f"  {'':16s}{'FROZEN':>12s}{'FINE-TUNED':>14s}")
    grid = {}
    for read in ("last", "weighted"):
        row = []
        for frz in ("frozen", "finetuned"):
            v = eer_of(CELLS[(read, frz)].format(lang=lang))
            grid[(read, frz)] = v
            row.append(f"{v:12.3f}" if v is not None else f"{'--':>12s}")
            if v is None:
                complete = False
        print(f"  {read:16s}" + "".join(row))
    out["cells"][lang] = {f"{r}_{f}": grid[(r, f)] for r in ("last", "weighted")
                          for f in ("frozen", "finetuned")}

    lf, lt = grid[("last", "frozen")], grid[("last", "finetuned")]
    wf, wt = grid[("weighted", "frozen")], grid[("weighted", "finetuned")]
    if None not in (lf, lt, wf, wt):
        ft_at_last = lf - lt          # >0 means fine-tuning helped
        ft_at_weighted = wf - wt
        interaction = ft_at_weighted - ft_at_last
        print(f"\n  fine-tuning gain at last-layer     : {ft_at_last:+.3f} pp")
        print(f"  fine-tuning gain at layer-weighted : {ft_at_weighted:+.3f} pp")
        print(f"  INTERACTION                        : {interaction:+.3f} pp")
        out["cells"][lang]["ft_gain_at_last"] = ft_at_last
        out["cells"][lang]["ft_gain_at_weighted"] = ft_at_weighted
        out["cells"][lang]["interaction"] = interaction
        if ft_at_weighted < ft_at_last:
            print("\n  Layer weighting SUBSTITUTES for fine-tuning: once the read-out is")
            print("  correct, adapting the encoder buys less. The cheap frozen config is")
            print("  the right low-resource recommendation. (P21 supported.)")
        else:
            print("\n  The two are COMPLEMENTARY: fine-tuning pays at least as much once")
            print("  the read-out is correct. (P21 falsified -- record it as such.)")

out["ok"] = complete
json.dump(out, open(os.path.join(outdir, "result.json"), "w"), indent=2)
print(f"\n  -> {outdir}/result.json")
if not complete:
    print("\n  Grid incomplete: B1/B2 have not produced their final.json yet.")
PY

cat <<'NOTE'

  THE CONSEQUENCE FOR papers/ieee_spl
  -----------------------------------
  P3's headline claim rests on the cell that was never measured. Once the grid
  is complete, one of two things must happen and neither is optional:

    * the claim is CONFIRMED -- say so, and cite this grid rather than the
      two-axis-at-once comparison it previously rested on; or
    * the claim is WITHDRAWN -- the paper is corrected before submission.

  Annual report section 10.3 exists because claims that should have been
  withdrawn were not. Do not let this become item 7.
NOTE
