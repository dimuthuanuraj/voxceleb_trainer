#!/usr/bin/env python3
"""Layer 3 of the dataset quality assessment: speaker-label reliability.

Consumes the .npz files written by tools/extract_embeddings.py and answers:
are the speaker labels actually correct?

Mathematics used (documented here so the reports can cite it):

  Embeddings are L2-normalised, so cosine similarity is an inner product:
      s(a, b) = <a, b> / (||a|| ||b||) = <a, b>

  Leave-one-out speaker centroid, for utterance i of speaker S with n_S utterances:
      c_S^(-i) = (1 / (n_S - 1)) * sum_{j in S, j != i} e_j
      score(i)  = s(e_i, c_S^(-i) / ||c_S^(-i)||)
  Low score = the utterance does not sound like the rest of its own speaker,
  i.e. a mislabel candidate. Leave-one-out matters: including e_i in its own
  centroid biases the score upward, and does so hardest exactly for the
  singleton speakers we most want to test.

  Silhouette, with a(i) the mean intra-speaker distance and b(i) the mean
  distance to the nearest other speaker (cosine distance d = 1 - s):
      sil(i) = (b(i) - a(i)) / max(a(i), b(i))    in [-1, 1]

  Agglomerative clustering vs the given labels, U = clusters, V = labels:
      NMI(U, V) = 2 I(U; V) / (H(U) + H(V))
      ARI       = (RI - E[RI]) / (max RI - E[RI])
      purity    = (1/N) * sum_k max_j |u_k ∩ v_j|

  EER-free diagnostic of separability (used in the reports):
      d' = (mu_within - mu_between) / sqrt((var_within + var_between) / 2)

Usage:
    python tools/label_audit.py --emb data/_qc/embeddings/slr52_sinhala.npz
    python tools/label_audit.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_samples, v_measure_score)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(REPO, "data", "_qc", "embeddings")
OUT_DIR = os.path.join(REPO, "data", "_qc", "label_audit")


def load(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def loo_centroid_scores(emb, spk):
    """Leave-one-out cosine to own-speaker centroid, for every utterance."""
    scores = np.full(len(emb), np.nan, dtype=np.float32)
    for s in np.unique(spk):
        idx = np.where(spk == s)[0]
        if len(idx) < 2:
            continue
        E = emb[idx]
        total = E.sum(axis=0)
        loo = (total[None, :] - E) / (len(idx) - 1)
        loo /= (np.linalg.norm(loo, axis=1, keepdims=True) + 1e-12)
        scores[idx] = np.einsum("ij,ij->i", E, loo)
    return scores


def pair_distributions(emb, spk, rng, max_pairs=400_000):
    """Sampled within-speaker and between-speaker cosine similarities."""
    n = len(emb)
    order = np.argsort(spk, kind="stable")
    spk_sorted = spk[order]
    bounds = {}
    start = 0
    for i in range(1, len(spk_sorted) + 1):
        if i == len(spk_sorted) or spk_sorted[i] != spk_sorted[start]:
            bounds[spk_sorted[start]] = (start, i)
            start = i

    within = []
    for s, (a, b) in bounds.items():
        idx = order[a:b]
        if len(idx) < 2:
            continue
        k = min(len(idx) * 4, 2000)
        i1 = rng.choice(idx, k)
        i2 = rng.choice(idx, k)
        m = i1 != i2
        if m.any():
            within.append(np.einsum("ij,ij->i", emb[i1[m]], emb[i2[m]]))
    within = np.concatenate(within) if within else np.array([0.0])
    if len(within) > max_pairs:
        within = rng.choice(within, max_pairs, replace=False)

    k = min(max_pairs, n * 20)
    i1 = rng.integers(0, n, k)
    i2 = rng.integers(0, n, k)
    m = spk[i1] != spk[i2]
    between = np.einsum("ij,ij->i", emb[i1[m]], emb[i2[m]])
    return within.astype(np.float32), between.astype(np.float32)


def dprime(within, between):
    return float((within.mean() - between.mean()) /
                 np.sqrt((within.var() + between.var()) / 2 + 1e-12))


def purity(labels_true, labels_pred):
    total = 0
    for c in np.unique(labels_pred):
        m = labels_pred == c
        vals, counts = np.unique(labels_true[m], return_counts=True)
        total += counts.max()
    return float(total / len(labels_true))


def cluster_vs_labels(emb, spk, rng, max_n=6000):
    """AHC on a subsample, scored against the given speaker labels."""
    n = len(emb)
    if n > max_n:
        idx = rng.choice(n, max_n, replace=False)
    else:
        idx = np.arange(n)
    E, S = emb[idx], spk[idx]
    n_spk = len(np.unique(S))
    if n_spk < 2:
        return {"status": "fewer than 2 speakers in subsample"}

    # cluster count fixed to the true speaker count: this asks "given the right
    # number of groups, do they line up with the labels?"
    ac = AgglomerativeClustering(n_clusters=n_spk, metric="cosine",
                                 linkage="average")
    pred = ac.fit_predict(E)

    # and a threshold-based run, which asks "how many speakers does the audio
    # think there are?" -- the diagnostic for split/merged identities
    ac2 = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5,
                                  metric="cosine", linkage="average")
    pred2 = ac2.fit_predict(E)

    return {
        "subsample": int(len(idx)),
        "true_speakers": int(n_spk),
        "fixed_k": {
            "NMI": round(float(normalized_mutual_info_score(S, pred)), 4),
            "ARI": round(float(adjusted_rand_score(S, pred)), 4),
            "V_measure": round(float(v_measure_score(S, pred)), 4),
            "purity": round(purity(S, pred), 4),
        },
        "threshold_0.5": {
            "clusters_found": int(len(np.unique(pred2))),
            "clusters_over_speakers": round(len(np.unique(pred2)) / n_spk, 3),
            "NMI": round(float(normalized_mutual_info_score(S, pred2)), 4),
            "ARI": round(float(adjusted_rand_score(S, pred2)), 4),
        },
    }


def silhouette(emb, spk, rng, max_n=5000):
    n = len(emb)
    idx = rng.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    E, S = emb[idx], spk[idx]
    if len(np.unique(S)) < 2:
        return None, None
    vals = silhouette_samples(E, S, metric="cosine")
    per_spk = {}
    for s in np.unique(S):
        per_spk[str(s)] = float(vals[S == s].mean())
    return float(vals.mean()), per_spk


def run(npz_path, args):
    name = os.path.splitext(os.path.basename(npz_path))[0]
    d = load(npz_path)
    emb, spk, lang = d["emb"], d["spk"], d["lang"]
    # speakers are namespaced per language for bilingual corpora
    spk_key = np.array([f"{l}/{s}" for l, s in zip(lang, spk)])
    rng = np.random.default_rng(args.seed)
    print(f"[audit] {name}: {len(emb)} embeddings, "
          f"{len(np.unique(spk_key))} speaker-language groups", flush=True)

    loo = loo_centroid_scores(emb, spk_key)
    valid = ~np.isnan(loo)
    within, between = pair_distributions(emb, spk_key, rng)
    sil_mean, sil_per_spk = silhouette(emb, spk_key, rng, args.silhouette_n)
    clus = cluster_vs_labels(emb, spk_key, rng, args.cluster_n)

    # mislabel shortlist: lowest LOO-centroid scores, for human listening
    order = np.argsort(np.where(valid, loo, np.inf))
    k = max(20, int(len(emb) * args.outlier_frac))
    shortlist = [
        {"path": str(d["path"][i]), "spk": str(spk_key[i]),
         "session": str(d["sess"][i]), "loo_centroid_cos": round(float(loo[i]), 4),
         "duration_s": float(d["dur"][i])}
        for i in order[:k] if valid[i]
    ]

    lo = loo[valid]
    report = {
        "dataset": name,
        "embeddings": int(len(emb)),
        "speaker_groups": int(len(np.unique(spk_key))),
        "embedding_model": "speechbrain/spkrec-ecapa-voxceleb (192-d, L2-normalised)",
        "within_speaker_cosine": {
            "mean": round(float(within.mean()), 4), "std": round(float(within.std()), 4),
            "p05": round(float(np.percentile(within, 5)), 4),
            "median": round(float(np.median(within)), 4),
        },
        "between_speaker_cosine": {
            "mean": round(float(between.mean()), 4), "std": round(float(between.std()), 4),
            "p95": round(float(np.percentile(between, 95)), 4),
            "median": round(float(np.median(between)), 4),
        },
        "separability_dprime": round(dprime(within, between), 4),
        "loo_centroid_cosine": {
            "mean": round(float(lo.mean()), 4), "std": round(float(lo.std()), 4),
            "p01": round(float(np.percentile(lo, 1)), 4),
            "p05": round(float(np.percentile(lo, 5)), 4),
            "median": round(float(np.median(lo)), 4),
            "frac_below_0.3": round(float((lo < 0.3).mean()), 5),
            "frac_below_0.2": round(float((lo < 0.2).mean()), 5),
            "frac_negative": round(float((lo < 0).mean()), 5),
            "utterances_scored": int(valid.sum()),
        },
        "silhouette": {
            "mean": round(sil_mean, 4) if sil_mean is not None else None,
            "worst_10_speakers": sorted(sil_per_spk.items(),
                                        key=lambda kv: kv[1])[:10] if sil_per_spk else [],
        },
        "clustering_vs_labels": clus,
        "mislabel_shortlist_size": len(shortlist),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(args.out, f"{name}_shortlist.json"), "w",
              encoding="utf-8") as fh:
        json.dump(shortlist, fh, indent=2)
        fh.write("\n")
    np.savez_compressed(
        os.path.join(args.out, f"{name}_dists.npz"),
        within=within, between=between, loo=lo)
    print(f"[audit] {name}: d'={report['separability_dprime']} "
          f"NMI={clus.get('fixed_k', {}).get('NMI')} "
          f"sil={report['silhouette']['mean']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cluster-n", type=int, default=6000)
    ap.add_argument("--silhouette-n", type=int, default=5000)
    ap.add_argument("--outlier-frac", type=float, default=0.005)
    args = ap.parse_args()

    targets = args.emb
    if args.all or not targets:
        targets = [os.path.join(args.emb_dir, f)
                   for f in sorted(os.listdir(args.emb_dir)) if f.endswith(".npz")]

    summary = {}
    for t in targets:
        summary[os.path.splitext(os.path.basename(t))[0]] = run(t, args)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"[audit] summary -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
