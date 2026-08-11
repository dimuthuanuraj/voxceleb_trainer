#!/usr/bin/env python3
"""Compute t-SNE and UMAP projections of speaker embeddings, for the QC reports.

Both algorithms are run on the *same* speaker subsample so the two views are
directly comparable -- that is the whole point of showing both.

Why a subsample of speakers rather than of utterances: a 2-D scatter of 500
speakers is an unreadable smear, and the question these plots answer is
"do utterances of one speaker group together, and do groups stay apart?".
That is best judged on ~12-20 speakers with all their utterances present.

Mathematics, briefly:

  t-SNE converts distances to conditional probabilities
      p_{j|i} = exp(-||x_i - x_j||^2 / 2 sigma_i^2) / sum_{k!=i} exp(...)
  with sigma_i set so the perplexity 2^{H(P_i)} matches a target, and finds a
  low-dimensional Y minimising KL(P || Q) where
      q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / sum_{k!=l} (1 + ||y_k - y_l||^2)^{-1}
  The heavy-tailed Student-t in Q is what stops distinct clusters collapsing.
  t-SNE preserves LOCAL neighbourhoods; between-cluster distances are not
  meaningful, so never read "these two blobs are far apart" as a claim.

  UMAP builds a fuzzy simplicial set from a k-NN graph with a local connectivity
  correction, membership
      mu(x_i, x_j) = exp(-(d(x_i, x_j) - rho_i) / sigma_i)
  and minimises the cross-entropy between the high- and low-dimensional fuzzy
  graphs. It preserves more GLOBAL structure than t-SNE and is far faster, so
  disagreement between the two views is itself informative.

Distances are cosine throughout, matching how the embeddings are scored.

Usage:
    python tools/projections.py --all --speakers 15
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(REPO, "data", "_qc", "embeddings")
OUT_DIR = os.path.join(REPO, "data", "_qc", "projections")


def pick_speakers(spk, n_speakers, rng, min_utts=8):
    labs, counts = np.unique(spk, return_counts=True)
    eligible = labs[counts >= min_utts]
    if len(eligible) == 0:
        eligible = labs
    if len(eligible) > n_speakers:
        eligible = rng.choice(eligible, n_speakers, replace=False)
    return set(eligible.tolist())


def compute(emb, seed, perplexity, n_neighbors, min_dist):
    from sklearn.manifold import TSNE
    n = len(emb)
    perp = float(min(perplexity, max(5, (n - 1) / 3)))
    tsne = TSNE(n_components=2, metric="cosine", init="pca",
                perplexity=perp, learning_rate="auto",
                max_iter=1000, random_state=seed)
    xy_tsne = tsne.fit_transform(emb)

    xy_umap, umap_err = None, None
    try:
        import umap
        red = umap.UMAP(n_components=2, metric="cosine",
                        n_neighbors=int(min(n_neighbors, max(2, n - 1))),
                        min_dist=min_dist, random_state=seed)
        xy_umap = red.fit_transform(emb)
    except Exception as exc:
        umap_err = str(exc)
    return xy_tsne, xy_umap, perp, umap_err


def run(npz_path, args):
    name = os.path.splitext(os.path.basename(npz_path))[0]
    z = np.load(npz_path, allow_pickle=False)
    emb, spk, lang = z["emb"], z["spk"], z["lang"]
    key = np.array([f"{l}/{s}" for l, s in zip(lang, spk)])
    rng = np.random.default_rng(args.seed)

    keep_spk = pick_speakers(key, args.speakers, rng, args.min_utts)
    mask = np.array([k in keep_spk for k in key])
    if args.max_points and mask.sum() > args.max_points:
        idx = np.where(mask)[0]
        idx = rng.choice(idx, args.max_points, replace=False)
        mask = np.zeros(len(key), bool)
        mask[idx] = True

    E = emb[mask]
    print(f"[proj] {name}: {mask.sum()} points, "
          f"{len(set(key[mask]))} speakers", flush=True)
    xy_t, xy_u, perp, umap_err = compute(E, args.seed, args.perplexity,
                                         args.n_neighbors, args.min_dist)
    if umap_err:
        print(f"[proj] {name}: UMAP unavailable ({umap_err})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, f"{name}.npz"),
        tsne=xy_t.astype(np.float32),
        umap=(xy_u.astype(np.float32) if xy_u is not None
              else np.zeros((0, 2), np.float32)),
        spk=key[mask], sess=z["sess"][mask], gender=z["gender"][mask],
        lang=lang[mask], path=z["path"][mask], dur=z["dur"][mask])
    meta = {"dataset": name, "points": int(mask.sum()),
            "speakers": len(set(key[mask].tolist())),
            "tsne_perplexity": perp, "umap_n_neighbors": args.n_neighbors,
            "umap_min_dist": args.min_dist, "metric": "cosine",
            "seed": args.seed, "umap_available": xy_u is not None,
            "umap_error": umap_err}
    with open(os.path.join(args.out, f"{name}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--speakers", type=int, default=15)
    ap.add_argument("--min-utts", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=1800)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    targets = args.emb
    if args.all or not targets:
        targets = [os.path.join(args.emb_dir, f)
                   for f in sorted(os.listdir(args.emb_dir)) if f.endswith(".npz")]
    metas = [run(t, args) for t in targets]
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(metas, fh, indent=2)
    print(f"[proj] -> {args.out}")


if __name__ == "__main__":
    main()
