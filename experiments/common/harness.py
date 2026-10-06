#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared entry point for every generated experiment script.

Each script in ``experiments/scripts/`` is a thin, self-documenting file that
carries its registry coordinates plus a frozen copy of its resolved parameters,
and hands both to :func:`main` here.  Keeping the execution logic in one place
means a fix to the recording format applies to all ~36 experiments at once,
while the generated scripts stay readable as the record of what was run.

Drift detection
---------------
A generated script freezes the parameters it was created with.  At run time the
harness re-resolves them from ``registry.py`` and compares.  If the registry has
changed since generation, the run stops and asks for regeneration rather than
silently producing a result whose script no longer describes it -- the failure
mode that makes an experiment archive untrustworthy months later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments import registry as R  # noqa: E402
from experiments.common import lossspec, modelcard  # noqa: E402
from experiments.common.recorder import ExperimentRecorder, fingerprint_file  # noqa: E402
from experiments.common.runner import TrainerRunner  # noqa: E402


def rebuild(spec: dict) -> R.Experiment:
    """Reconstruct the Experiment object from a script's stored coordinates."""
    return R.Experiment(
        exp_id=spec["exp_id"],
        stage=spec["stage"],
        arch_key=spec["architecture"],
        loss_key=spec["loss"],
        condition_key=spec["condition"],
        seed=spec.get("seed"),
        overrides=spec.get("overrides"),
        note=spec.get("note", ""),
    )


def check_drift(exp: R.Experiment, frozen: dict, scale: str):
    if not frozen or scale != "full":
        return
    live = exp.params("full")
    diffs = {
        k: {"script": frozen.get(k), "registry": live.get(k)}
        for k in set(frozen) | set(live)
        if str(frozen.get(k)) != str(live.get(k))
    }
    if diffs:
        raise SystemExit(
            "registry.py has changed since this script was generated:\n"
            + json.dumps(diffs, indent=2)
            + "\n\nRegenerate with:  python experiments/tools/gen_scripts.py --stage "
            + exp.stage
        )


def build_manifest(exp: R.Experiment, params: dict, argv, flags, scale: str) -> dict:
    """Everything knowable before the first epoch."""
    kw = dict(exp.arch["args"])
    for ck, cv in exp.extra_cli().items():
        kw[ck.lstrip("-")] = cv

    cond = exp.condition
    data_fp = {
        "train_list": fingerprint_file(params.get("train_list"), head_lines=2),
        "val_or_test_list": fingerprint_file(params.get("test_list"), head_lines=2),
        "heldout_test_list": fingerprint_file(cond.get("test_list")),
        "asnorm_cohort": fingerprint_file(cond.get("cohort_list")),
    }
    split_manifest_path = os.path.join(R.SPLITS, cond["corpus"], "manifest.json")
    split_manifest = None
    if os.path.isfile(split_manifest_path):
        with open(split_manifest_path, encoding="utf-8") as fh:
            split_manifest = json.load(fh)

    return {
        "experiment": exp.describe(),
        "scale": scale,
        "resolved_parameters": params,
        "flags": flags,
        "command": argv,
        "architecture_spec": modelcard.get_arch(exp.arch["model"]),
        "model_card": modelcard.build_card(exp.arch["model"], kw),
        "loss_spec": lossspec.get(exp.loss_key),
        "data": {
            "condition": {k: v for k, v in cond.items() if isinstance(v, (str, int, list))},
            "fingerprints": data_fp,
            "split_manifest": split_manifest,
        },
        "controlled_factors": {
            "embedding_dim": R.EMBED_DIM,
            "max_frames": params.get("max_frames"),
            "eval_frames": params.get("eval_frames"),
            "optimizer": params.get("optimizer"),
            "lr": params.get("lr"),
            "lr_decay": params.get("lr_decay"),
            "weight_decay": params.get("weight_decay"),
            "augmentation": "--augment" in flags,
            "as_norm_during_training": False,
            "model_selection": "validation trials only; test set touched once by evaluate.py",
        },
    }


def main(script_path: str, spec: dict):
    ap = argparse.ArgumentParser(description=f"Run experiment {spec['exp_id']}")
    ap.add_argument("--scale", default="full", choices=sorted(R.SCALES),
                    help="smoke = 2 tiny epochs to validate plumbing; "
                         "dev = 12 epochs on a data subset; full = the real run")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu", default=None, help="value for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing results dir instead of moving it aside")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    ap.add_argument("--quiet", action="store_true", help="do not echo trainer output")
    ap.add_argument("--no-runtime-patches", action="store_true",
                    help="do not inject experiments/common/runtime_patches via "
                         "PYTHONPATH (they only fix the trainer's ROC plot; "
                         "metrics are identical either way)")
    args = ap.parse_args()

    exp = rebuild(spec)
    check_drift(exp, spec.get("resolved_parameters_full"), args.scale)

    exp_id = exp.exp_id if args.scale == "full" else f"{exp.exp_id}__{args.scale}"

    # Reduced-scale runs must not share a checkpoint directory with the real
    # run: trainSpeakerNet.py resumes by globbing model0*.model in save_path, so
    # a leftover 2-epoch smoke checkpoint would be silently resumed by the
    # subsequent full run and quietly corrupt its result.
    scale_override = {} if args.scale == "full" else {
        "save_path": os.path.join(R.REPO_ROOT, "exps", exp_id)
    }
    exp.overrides.update(scale_override)

    params = exp.params(args.scale)
    flags = exp.flags(args.scale)
    argv = exp.argv(python=args.python, scale=args.scale)

    if args.dry_run:
        print(" \\\n    ".join(argv))
        return 0
    rec = ExperimentRecorder(exp_id, root=args.results_root, resume=args.resume)
    rec.copy_script(script_path)

    manifest = build_manifest(exp, params, argv, flags, args.scale)
    manifest["experiment"]["exp_id"] = exp_id
    rec.write_manifest(manifest)

    try:
        import yaml

        rec.write_config(yaml.safe_dump(params, sort_keys=True))
    except Exception:
        rec.write_config(json.dumps(params, indent=2, sort_keys=True))

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Runtime patches for the trainer subprocess, applied via sitecustomize on
    # PYTHONPATH rather than by editing trainSpeakerNet.py. Currently this fixes
    # the per-epoch ROC plot, which fails on every run because
    # ComputeErrorRates returns lists and the plotting code computes `1 - fnrs`.
    # See experiments/common/runtime_patches/sitecustomize.py -- the numbers are
    # unaffected either way; only the image is.
    if not args.no_runtime_patches:
        patch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "runtime_patches")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{patch_dir}{os.pathsep}{existing}" if existing else patch_dir
        env["SLSPV_REPO_ROOT"] = _REPO_ROOT

    card = manifest["model_card"]
    print(f"\n=== {exp_id} ===")
    print(f"  architecture : {exp.arch_key} ({exp.arch['model']}) "
          f"{card.get('parameters_total_human', '?')} params")
    print(f"  loss         : {exp.loss_key} ({manifest['loss_spec']['family']})")
    print(f"  condition    : {exp.condition_key} -- {exp.condition['label']}, "
          f"{exp.condition['n_classes']} classes")
    print(f"  scale        : {args.scale}   seed: {exp.seed}")
    print(f"  results      : {rec.dir}\n")

    runner = TrainerRunner(rec, argv, env=env, cwd=_REPO_ROOT, echo=not args.quiet)
    rc = runner.run()

    summary = rec.write_final({"exit_code": rc, "events": runner.events})
    best = summary.get("best")
    print(f"\n=== {exp_id} finished (rc={rc}) ===")
    if best:
        print(f"  best val EER {best['val_eer']:.4f}% @ epoch {best['epoch']}  "
              f"(minDCF {best.get('val_mindcf')})")
    print(f"  wall {summary['total_wall_h']:.2f} h over "
          f"{summary['n_epochs_completed']} epochs")
    print(f"  records -> {rec.dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit("this module is called by the generated experiment scripts")
