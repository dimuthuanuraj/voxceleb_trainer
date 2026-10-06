#!/usr/bin/env python3
"""C4 -- Reproduce Thienpondt et al.'s Figure 1 (the score-shift histogram) for si/ta.

`MASTER_SYNTHESIS.md` section 5, P1: *"reproduce Thienpondt et al.'s Figure 1
(the score-shift histogram) for Sinhala/Tamil. It is an afternoon's work on
scores already on disk and it is the figure that anchors the whole narrative."*

What the figure shows
---------------------
Target and non-target score distributions per language on one axis. The point is
not that the distributions separate -- it is that the *same* decision threshold
sits in a different place for each language, which is the visual form of the
project's Finding III-1 (per-language calibration is mandatory, Cllr inflation up
to 5.7x, a shared threshold costs Tamil +1.65 pp).

Inputs: per-trial score vectors already written by the v1 evaluator. No GPU.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
TRAINER = os.path.join(SL_SPV, "voxceleb_trainer")
OUT = os.path.join(FRW, "03_RESULTS", "C4")


def find_score_files() -> dict[str, list[str]]:
    """Locate per-trial score archives, grouped by language."""
    pats = [
        os.path.join(TRAINER, "experiments", "results", "*", "scores", "*.npz"),
        os.path.join(TRAINER, "experiments", "results", "*", "*.npz"),
        os.path.join(SL_SPV, "proposals", "*", "out", "scores", "*", "*.npz"),
    ]
    found: dict[str, list[str]] = {"si": [], "ta": [], "other": []}
    for p in pats:
        for f in glob.glob(p):
            low = f.lower()
            if "_si_" in low or "sinhala" in low or "slr52" in low:
                found["si"].append(f)
            elif "_ta_" in low or "tamil" in low or "slr127" in low:
                found["ta"].append(f)
            else:
                found["other"].append(f)
    return found


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    try:
        import numpy as np
    except ImportError:
        print("  numpy unavailable")
        return 1

    files = find_score_files()
    print(f"  score archives found: si={len(files['si'])} ta={len(files['ta'])} "
          f"other={len(files['other'])}\n")

    if not files["si"] and not files["ta"]:
        print("  No per-trial score archives found.")
        print("  They are written by experiments/tools/evaluate.py, which retains the")
        print("  per-trial score vector precisely so the paired bootstrap and this")
        print("  figure are possible. Run evaluate.py --all first, or point this")
        print("  script at the v1 analysis outputs.")
        json.dump({"task": "C4", "ok": False, "reason": "no score archives found"},
                  open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1

    stats: dict[str, dict] = {}
    for lang in ("si", "ta"):
        for f in sorted(files[lang])[:6]:
            try:
                z = np.load(f, allow_pickle=True)
            except Exception as exc:
                print(f"  skip {os.path.basename(f)}: {exc}")
                continue
            keys = list(z.keys())
            sc = next((z[k] for k in ("scores", "score", "sc") if k in keys), None)
            lb = next((z[k] for k in ("labels", "label", "lab") if k in keys), None)
            if sc is None or lb is None:
                print(f"  skip {os.path.basename(f)}: keys={keys}")
                continue
            sc, lb = np.asarray(sc).ravel(), np.asarray(lb).ravel()
            if sc.shape != lb.shape:
                continue
            tgt, non = sc[lb == 1], sc[lb == 0]
            if not len(tgt) or not len(non):
                continue
            # .../results/<exp>/scores/<set>.npz -- the experiment name is two
            # levels up when the file sits in a scores/ subdirectory.
            d = os.path.dirname(f)
            exp = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "scores" \
                else os.path.basename(d)
            setname = os.path.splitext(os.path.basename(f))[0]
            # The language of a RESULT is the language of the trials it was
            # scored on, not the language the model was trained on. A Sinhala
            # model evaluated on `probe_nisp_tamil` is a Tamil measurement.
            # Getting this backwards is exactly the "good number measured on
            # something other than what it claims" failure mode. Only the
            # in-language held-out `test` set is used for the shift.
            eval_lang = lang
            sl_ = setname.lower()
            if "tamil" in sl_ or sl_.endswith("_ta"):
                eval_lang = "ta"
            elif "sinhala" in sl_ or sl_.endswith("_si"):
                eval_lang = "si"
            name = f"{eval_lang}:{exp}/{setname}"
            # d-prime: the standard separability statistic, and the quantity
            # the annual report uses for the same-recording confound (3.46 -> 2.44)
            pooled_sd = np.sqrt((tgt.var() + non.var()) / 2) or 1e-9
            stats[name] = {
                "lang": eval_lang, "train_condition": lang,
                "eval_set": setname, "matched_test": setname == "test",
                "file": os.path.relpath(f, SL_SPV),
                "n_target": int(len(tgt)), "n_nontarget": int(len(non)),
                "target_mean": float(tgt.mean()), "target_sd": float(tgt.std()),
                "nontarget_mean": float(non.mean()), "nontarget_sd": float(non.std()),
                "d_prime": float((tgt.mean() - non.mean()) / pooled_sd),
            }
            print(f"  {name:52s} tgt {tgt.mean():+.3f}+/-{tgt.std():.3f}  "
                  f"non {non.mean():+.3f}+/-{non.std():.3f}  d'={stats[name]['d_prime']:.2f}")

    if not stats:
        print("\n  archives found but none had a usable (scores, labels) pair.")
        json.dump({"task": "C4", "ok": False, "reason": "no usable score/label pairs",
                   "archives": {k: len(v) for k, v in files.items()}},
                  open(os.path.join(OUT, "result.json"), "w"), indent=2)
        return 1

    # ---- the actual point of the figure: where does ONE threshold land?
    print("\n  " + "=" * 66)
    print("  THE SHIFT -- one threshold, two languages")
    print("  " + "=" * 66)
    # ONLY matched in-language held-out test sets. Cross-lingual probes measure
    # a different quantity and must not be averaged into a per-language midpoint.
    si = [v for v in stats.values() if v["lang"] == "si" and v["matched_test"]]
    ta = [v for v in stats.values() if v["lang"] == "ta" and v["matched_test"]]
    print(f"  using matched held-out test sets only: si={len(si)} ta={len(ta)}")
    print("  (cross-lingual probe sets are excluded -- they measure a different thing)\n")
    shift = None
    if si and ta:
        si_mid = sum((v["target_mean"] + v["nontarget_mean"]) / 2 for v in si) / len(si)
        ta_mid = sum((v["target_mean"] + v["nontarget_mean"]) / 2 for v in ta) / len(ta)
        shift = ta_mid - si_mid
        print(f"  Sinhala score midpoint : {si_mid:+.4f}")
        print(f"  Tamil   score midpoint : {ta_mid:+.4f}")
        print(f"  SHIFT                  : {shift:+.4f}")
        print("\n  A single threshold set at one language's operating point sits at the")
        print("  wrong place for the other by this much. That is Finding III-1 in one")
        print("  number, and this figure is its visual form.")
    else:
        print("  Not enough matched in-language test sets to compute the shift.")
        print("  Cross-lingual probes are present but deliberately excluded; run")
        print("  evaluate.py so each system has its own in-language `test` set.")

    report = {"task": "C4", "ok": True,
              "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "systems": stats, "midpoint_shift_ta_minus_si": shift,
              "shift_basis": {"si_systems": [v["eval_set"] for v in si],
                              "ta_systems": [v["eval_set"] for v in ta],
                              "matched_test_only": True},
              "source": "MASTER_SYNTHESIS.md section 5 P1 (Thienpondt et al. Fig. 1)",
              "note": "Per-trial score distributions per language. The figure anchors "
                      "paper P1's narrative; the shift is the visual form of Finding III-1."}
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    print(f"\n  -> {OUT}/result.json")
    print("\n  To draw it: histogram target vs non-target per language on shared axes,")
    print("  with each language's own EER threshold marked, plus the pooled threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
