#!/usr/bin/env python3
"""C5 -- The PEFT two-stage fine-tuning confound check. A PUBLICATION GATE.

Why this is a gate rather than an analysis
------------------------------------------
`My_refered_papers/_deep_study/00_MASTER_SYNTHESIS.md` section 6, risk 3, states
it plainly:

    "Three papers say two-stage beats one-stage. If SL_SPV's arms differ in this
     respect, the negative PEFT result is not safe to publish as-is. Check first."

The annual report's prediction scoreboard currently records prediction 9 --
"PEFT >= full FT under language shift" -- as **Falsified**, on the evidence that
full fine-tuning won at every seed by 2.2x. That verdict is only sound if the
PEFT arm and the full-FT arm were trained under the SAME staging discipline.

If the full-FT arm was initialised from a first-stage checkpoint and the PEFT arm
was not (or vice versa), then what was measured is staging, not PEFT, and the
falsification must be withdrawn pending a matched re-run.

What this checks, from argv rather than from prose
--------------------------------------------------
For every SSL experiment in the v1 tree and every recorded P3 run, it extracts:

  --initial_model     was the run warm-started, and from what?
  --finetune          the trainer's own two-stage flag
  --lr, --llrd*       optimiser deviations that co-occur with staging
  --ssl_freeze        which arm is PEFT and which is full FT

and reports whether the arms being compared are matched on staging.

This reads command.json / train.command.json files only. No GPU, no training,
~30 minutes including the reading.
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
OUT = os.path.join(FRW, "03_RESULTS", "C5")

STAGING_FLAGS = ["--initial_model", "--finetune", "--no_finetune",
                 "--lr", "--llrd", "--no_llrd", "--llrd_decay",
                 "--ssl_freeze", "--no_ssl_freeze", "--max_epoch"]


def argv_of(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return []
    if isinstance(d, list):
        return [str(x) for x in d]
    for k in ("command", "argv", "cmd"):
        v = d.get(k)
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def extract(argv: list[str]) -> dict:
    out: dict[str, object] = {}
    for i, tok in enumerate(argv):
        if tok in STAGING_FLAGS:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not str(nxt).startswith("--"):
                out[tok] = nxt
            else:
                out[tok] = True
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print("  scanning every run's literal argv for staging flags...\n")

    runs: dict[str, dict] = {}

    # v1 experiment tree
    for cj in sorted(glob.glob(os.path.join(TRAINER, "experiments", "results", "*", "command.json"))):
        exp = os.path.basename(os.path.dirname(cj))
        if ".superseded" in exp or "__smoke" in exp:
            continue
        argv = argv_of(cj)
        if argv:
            runs[exp] = {"source": "experiments/results", "flags": extract(argv)}

    # proposal + transformer runs
    for pat in ("proposals/*/out/*.train.command.json",
                "transformer_sv/*/out/*.train.command.json"):
        for cj in sorted(glob.glob(os.path.join(SL_SPV, pat))):
            tag = os.path.basename(cj).replace(".train.command.json", "")
            argv = argv_of(cj)
            if argv:
                runs[tag] = {"source": pat.split("/")[0], "flags": extract(argv)}

    print(f"  {len(runs)} runs with a recorded argv\n")

    # ---- the arms that matter: SSL fine-tuned vs SSL frozen/PEFT
    ssl = {k: v for k, v in runs.items() if "ssl" in k.lower()}
    ft = {k: v for k, v in ssl.items() if "--no_ssl_freeze" in v["flags"]}
    frozen = {k: v for k, v in ssl.items() if "--ssl_freeze" in v["flags"]}

    print(f"  SSL runs: {len(ssl)}   fine-tuned arms: {len(ft)}   frozen arms: {len(frozen)}\n")

    def warm(v: dict) -> bool:
        return bool(v["flags"].get("--initial_model")) or bool(v["flags"].get("--finetune"))

    print("  --- FINE-TUNED (full FT) arms ---")
    for k, v in sorted(ft.items()):
        print(f"    {k:52s} warm_start={warm(v)!s:5s} "
              f"initial_model={str(v['flags'].get('--initial_model'))[:40]} "
              f"lr={v['flags'].get('--lr')}")
    print("\n  --- FROZEN / PEFT arms ---")
    for k, v in sorted(frozen.items()):
        print(f"    {k:52s} warm_start={warm(v)!s:5s} "
              f"initial_model={str(v['flags'].get('--initial_model'))[:40]} "
              f"lr={v['flags'].get('--lr')}")

    ft_warm = {k: warm(v) for k, v in ft.items()}
    fz_warm = {k: warm(v) for k, v in frozen.items()}
    any_ft_warm = any(ft_warm.values())
    any_fz_warm = any(fz_warm.values())
    matched = (any_ft_warm == any_fz_warm)

    print("\n  " + "=" * 66)
    print("  VERDICT")
    print("  " + "=" * 66)
    print(f"  any fine-tuned arm warm-started : {any_ft_warm}")
    print(f"  any frozen/PEFT arm warm-started: {any_fz_warm}")

    if matched:
        print("\n  MATCHED on staging.")
        print("  The arms do not differ in two-stage-ness, so the confound named in")
        print("  MASTER_SYNTHESIS section 6 risk 3 does NOT apply to this comparison.")
        print("  Prediction 9's 'Falsified' verdict stands on this axis.")
        verdict = "matched-staging"
    else:
        print("\n  *** MISMATCHED on staging. ***")
        print("  One arm was warm-started and the other was not. What was measured")
        print("  is staging, not PEFT-vs-full-FT.")
        print("\n  REQUIRED ACTION: withdraw prediction 9's 'Falsified' verdict pending")
        print("  a staging-matched re-run, and do NOT include the PEFT negative in P2")
        print("  until it is rerun. (MASTER_SYNTHESIS section 6, risk 3.)")
        verdict = "MISMATCHED-staging"

    # Also flag optimiser deviations that co-occur with the comparison, since the
    # registry itself records lr as a declared deviation for the fine-tuned arm.
    lrs_ft = sorted({v["flags"].get("--lr") for v in ft.values() if v["flags"].get("--lr")})
    lrs_fz = sorted({v["flags"].get("--lr") for v in frozen.values() if v["flags"].get("--lr")})
    print(f"\n  learning rates -- fine-tuned {lrs_ft}  frozen {lrs_fz}")
    if lrs_ft and lrs_fz and set(lrs_ft) != set(lrs_fz):
        print("  NOTE: the arms also differ in learning rate. registry.py declares this")
        print("  as a necessary deviation (1e-3 destroys a pretrained transformer), but")
        print("  it MUST be stated wherever the PEFT contrast is reported -- it is a")
        print("  second uncontrolled factor alongside whatever staging shows.")

    report = {
        "task": "C5", "ok": True, "verdict": verdict, "staging_matched": matched,
        "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_runs_scanned": len(runs),
        "fine_tuned_arms": ft_warm, "frozen_arms": fz_warm,
        "lr_fine_tuned": lrs_ft, "lr_frozen": lrs_fz,
        "lr_matched": set(lrs_ft) == set(lrs_fz) if (lrs_ft and lrs_fz) else None,
        "consequence": (
            "Prediction 9 'Falsified' stands on the staging axis."
            if matched else
            "Withdraw prediction 9's Falsified verdict pending a staging-matched re-run."),
        "source": "My_refered_papers/_deep_study/00_MASTER_SYNTHESIS.md section 6 risk 3",
    }
    json.dump(report, open(os.path.join(OUT, "result.json"), "w"), indent=2)
    json.dump(runs, open(os.path.join(OUT, "all_run_flags.json"), "w"), indent=2, sort_keys=True)
    print(f"\n  -> {OUT}/result.json")
    print(f"  -> {OUT}/all_run_flags.json  (staging flags for all {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
