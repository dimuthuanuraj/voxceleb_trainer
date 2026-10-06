#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract the fitted SSL layer weights from layer-weighted experiments.

    python experiments/tools/layer_weights.py --all
    python experiments/tools/layer_weights.py --exp F_ssl_wavlm_lw_aamsoftmax_si_s42

Why this is a result and not a diagnostic
-----------------------------------------
``SSLFrontendSpeakerLW`` learns a softmax weighting over all L+1 hidden states of
its encoder (index 0 is the convolutional feature extractor, 1..L the transformer
blocks).  After training, that distribution answers a question no EER can:

    at what depth of this encoder does speaker information live,
    for this language?

The prediction under test is that masked-prediction pretraining pushes the upper
layers toward phonetic content — for which speaker identity is nuisance — so the
mass should concentrate LOW.  If the fitted weights peak near the top instead,
that prediction is wrong and the last-layer default was fine all along.

Because the same encoder can be run on Sinhala, Tamil and scale-matched English,
the *profiles can be compared*: a language whose peak sits at a different depth
is evidence that the encoder's representation of speaker identity is itself
language-dependent, which is a stronger and more interesting claim than a
difference in EER.

Outputs a table, a JSON summary, and (unless --no-figure) an overlay plot of
every profile.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "experiments", "analysis")


def weights_from_checkpoint(path):
    """Return the softmax-normalised layer distribution, or None."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    for key, val in state.items():
        if key.endswith("layer_weights"):
            return torch.softmax(val.detach().float(), dim=0).numpy()
    return None


def summarise(w):
    """Descriptive statistics of the fitted distribution."""
    n = len(w)
    idx = numpy.arange(n)
    uniform = 1.0 / n
    # Entropy in bits, normalised: 1.0 = uniform (no preference at all), 0 = all
    # mass on a single layer. Says how *decisively* the model chose a depth.
    ent = float(-(w * numpy.log2(numpy.clip(w, 1e-12, None))).sum() / numpy.log2(n))
    return {
        "n_states": int(n),
        "argmax_layer": int(w.argmax()),
        "max_weight": round(float(w.max()), 5),
        "centre_of_mass": round(float((w * idx).sum()), 3),
        "centre_of_mass_relative": round(float((w * idx).sum() / (n - 1)), 3),
        "mass_lower_half": round(float(w[: n // 2].sum()), 4),
        "mass_upper_half": round(float(w[n // 2:].sum()), 4),
        "normalised_entropy": round(ent, 4),
        "l1_deviation_from_uniform": round(float(numpy.abs(w - uniform).sum()), 5),
        "weights": [round(float(x), 5) for x in w],
    }


def collect(exp_ids=None, include_reduced=False):
    out = {}
    for man_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "manifest.json"))):
        exp_id = os.path.basename(os.path.dirname(man_path))
        if not include_reduced and (exp_id.endswith("__smoke") or exp_id.endswith("__dev")):
            continue
        if exp_ids and exp_id not in exp_ids:
            continue
        try:
            with open(man_path, encoding="utf-8") as fh:
                man = json.load(fh)
        except Exception:
            continue
        if "LW" not in man["experiment"].get("model", ""):
            continue
        ckpt = os.path.join(man["resolved_parameters"]["save_path"], "model",
                            "model_best.model")
        if not os.path.isfile(ckpt):
            continue
        w = weights_from_checkpoint(ckpt)
        if w is None:
            continue
        rec = summarise(w)
        rec.update({
            "architecture": man["experiment"]["architecture"],
            "condition": man["experiment"]["condition"],
            "encoder": man["resolved_parameters"].get("ssl_encoder_name", "?"),
            "checkpoint": ckpt,
        })
        out[exp_id] = rec
    return out


def figure(records, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figure] skipped: {exc!r}")
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for exp_id, r in sorted(records.items()):
        w = r["weights"]
        ax.plot(range(len(w)), w, marker="o", ms=3, lw=1.3,
                label=f"{r['architecture']} / {r['condition']}")
    if records:
        n = len(next(iter(records.values()))["weights"])
        ax.axhline(1.0 / n, ls="--", c="grey", lw=1,
                   label=f"uniform ({1.0 / n:.3f})")
    ax.set_xlabel("encoder hidden state (0 = CNN feature extractor)")
    ax.set_ylabel("learned weight")
    ax.set_title("Where speaker information lives in the SSL encoder")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-reduced", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out", default=ANALYSIS_DIR)
    args = ap.parse_args()

    if not args.exp and not args.all:
        ap.error("pass --exp ID... or --all")

    records = collect(set(args.exp) if args.exp else None, args.include_reduced)
    if not records:
        raise SystemExit(
            "no layer-weighted experiments with checkpoints found. Run Stage F "
            "first: python experiments/tools/run_queue.py --stage F"
        )

    print(f"{'experiment':46s} {'peak':>5s} {'CoM':>6s} {'lower':>6s} "
          f"{'upper':>6s} {'entropy':>8s}")
    for exp_id, r in sorted(records.items()):
        print(f"{exp_id:46s} {r['argmax_layer']:>5d} {r['centre_of_mass']:>6.2f} "
              f"{r['mass_lower_half']:>6.3f} {r['mass_upper_half']:>6.3f} "
              f"{r['normalised_entropy']:>8.4f}")

    print("\n  peak    = highest-weighted hidden state (0 = CNN output)")
    print("  CoM     = centre of mass; below (n-1)/2 means the model prefers "
          "lower layers")
    print("  entropy = 1.0 is uniform (no preference); lower means a decisive "
          "choice of depth")

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "ssl_layer_weights.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"\njson -> {os.path.relpath(out_json, REPO_ROOT)}")

    if not args.no_figure:
        fig_dir = os.path.join(args.out, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        p = figure(records, os.path.join(fig_dir, "ssl_layer_weights.png"))
        if p:
            print(f"figure -> {os.path.relpath(p, REPO_ROOT)}")


if __name__ == "__main__":
    main()
