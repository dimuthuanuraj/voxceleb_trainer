#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure how much speaker information sits at each layer of an SSL encoder.

    python experiments/tools/ssl_layer_probe.py --condition si
    python experiments/tools/ssl_layer_probe.py --condition ta --encoder mhubert-147
    python experiments/tools/ssl_layer_probe.py --condition si --n-trials 4000

Why this exists
---------------
``SSLFrontendSpeaker`` defaults to ``--ssl_layer -1``: the encoder's LAST hidden
state.  Masked-prediction pretraining rewards reconstructing a masked frame from
its context, which drives the upper layers toward phonetic and lexical content
and treats speaker identity as nuisance.  The last layer is therefore expected to
be close to the *worst* choice for speaker verification -- but "expected" is not
a number, and the fix (which layer instead?) needs one.

This probe answers it directly and cheaply: **no training at all.**  It runs the
frozen encoder once per file, keeps all L+1 hidden states, mean-pools each over
time, L2-normalises, and scores the validation trials with plain cosine.  The
resulting EER-per-layer curve is a direct read-out of where speaker identity
survives in that encoder, for this language.

What the numbers do and do not mean
-----------------------------------
Absolute EERs here are pessimistic: a trained attentive-pooling head does far
better than mean pooling, and the trained models in Stage A/F sit well below
these values.  The *shape* of the curve is the result -- which depth carries the
information, and how much is lost by reading the top instead of the best layer.
That ratio is what justifies changing ``--ssl_layer`` or moving to learned layer
weights, and it is measured in minutes rather than a training run per layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ANALYSIS_DIR = os.path.join(REPO_ROOT, "experiments", "analysis")


def eer_from_scores(scores, labels):
    scores = numpy.asarray(scores, dtype=numpy.float64)
    labels = numpy.asarray(labels, dtype=numpy.int32)
    order = numpy.argsort(-scores)
    lab = labels[order]
    n_t, n_n = int((lab == 1).sum()), int((lab == 0).sum())
    if not n_t or not n_n:
        return float("nan")
    fnr = 1.0 - numpy.cumsum(lab == 1) / n_t
    fpr = numpy.cumsum(lab == 0) / n_n
    i = int(numpy.nanargmin(numpy.abs(fnr - fpr)))
    return float(100.0 * 0.5 * (fnr[i] + fpr[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--condition", default="si", choices=["si", "ta"])
    ap.add_argument("--encoder", default="wavlm-base-plus",
                    help="directory name under models/weights/")
    ap.add_argument("--n-trials", type=int, default=4000,
                    help="validation trials to score (subset, for speed)")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="seconds of audio per file")
    ap.add_argument("--out", default=ANALYSIS_DIR)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel
    from DatasetLoader import loadWAV
    from experiments import registry as R

    cond = R.CONDITIONS[args.condition]
    trials_path = cond["val_list"]
    root = cond["test_path"]

    labels, a, b = [], [], []
    with open(trials_path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 3:
                labels.append(int(p[0])); a.append(p[1]); b.append(p[2])
    if args.n_trials and args.n_trials < len(labels):
        rng = numpy.random.default_rng(42)
        keep = rng.choice(len(labels), args.n_trials, replace=False)
        labels = [labels[i] for i in keep]
        a = [a[i] for i in keep]; b = [b[i] for i in keep]
    files = sorted(set(a) | set(b))

    enc_path = os.path.join(REPO_ROOT, "models", "weights", args.encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(enc_path, output_hidden_states=True).to(device).eval()
    n_states = int(model.config.num_hidden_layers) + 1
    print(f"[probe] {args.encoder}: {n_states} hidden states, "
          f"{len(files)} files, {len(labels)} trials, {args.condition}, {device}")

    # One forward pass per file; every layer's mean-pooled embedding is kept.
    embs = {l: {} for l in range(n_states)}
    n_samples = int(args.seconds * 16000)
    t0 = time.time()
    with torch.no_grad():
        for i, f in enumerate(files):
            path = f if os.path.isabs(f) else os.path.join(root, f)
            wav = loadWAV(path, int(args.seconds * 100), evalmode=False)
            x = torch.FloatTensor(wav[:, :n_samples]).to(device)
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
            out = model(x)
            for l, h in enumerate(out.hidden_states):
                e = h.mean(dim=1)                       # mean-pool over time
                e = torch.nn.functional.normalize(e, p=2, dim=1)
                embs[l][f] = e[0].cpu().numpy()
            if i % 200 == 0:
                print(f"    {i}/{len(files)}  ({i / max(time.time() - t0, 1e-9):.1f} files/s)",
                      flush=True)

    rows = []
    for l in range(n_states):
        E = embs[l]
        sc = [float(numpy.dot(E[x], E[y])) for x, y in zip(a, b)]
        rows.append({"layer": l, "eer": round(eer_from_scores(sc, labels), 4)})

    best = min(rows, key=lambda r: r["eer"])
    last = rows[-1]
    print(f"\n  {'layer':>6s} {'EER %':>8s}   (0 = CNN feature extractor)")
    for r in rows:
        mark = "  <-- BEST" if r["layer"] == best["layer"] else (
            "  <-- default --ssl_layer -1" if r["layer"] == last["layer"] else "")
        bar = "#" * int(max(0, 60 - r["eer"]) / 2)
        print(f"  {r['layer']:>6d} {r['eer']:>8.3f}  {bar}{mark}")

    ratio = last["eer"] / best["eer"] if best["eer"] else float("nan")
    print(f"\n  best layer {best['layer']} = {best['eer']:.3f}%  vs  "
          f"last layer {last['layer']} = {last['eer']:.3f}%")
    print(f"  the default reads a layer {ratio:.2f}x worse than the best available")

    os.makedirs(args.out, exist_ok=True)
    out = {
        "encoder": args.encoder, "condition": args.condition,
        "n_trials": len(labels), "n_files": len(files),
        "seconds_per_file": args.seconds,
        "protocol": "frozen encoder, mean-pool over time, cosine, NO training",
        "layers": rows,
        "best_layer": best, "last_layer": last,
        "last_over_best_ratio": round(ratio, 3),
    }
    p = os.path.join(args.out, f"ssl_layer_probe_{args.encoder}_{args.condition}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  json -> {os.path.relpath(p, REPO_ROOT)}")


if __name__ == "__main__":
    main()
