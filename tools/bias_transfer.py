#!/usr/bin/env python3
"""Layer 6: subgroup fairness, and cross-corpus distribution shift.

Two questions the earlier layers cannot answer:

  (a) Does the system work equally well for everyone in the corpus?
  (b) How far apart are these corpora in embedding space -- i.e. how much of a
      domain gap would a model trained on one face on another?

Mathematics
-----------
Subgroup error rates at a SHARED threshold (a deployed system has exactly one
operating point, so per-subgroup EER alone is misleading):

    FNMR_g(t) = |{targets in g : s < t}| / |targets in g|
    FMR_g(t)  = |{impostors in g : s >= t}| / |impostors in g|

with t fixed at the pooled EER threshold. The Fairness Discrepancy Rate
(Sixta et al.; used by Hutiri & Ding for speaker recognition) combines the
worst-case gaps:

    FDR(t) = 1 - [ a * max_{g,h}|FMR_g - FMR_h| + (1-a) * max_{g,h}|FNMR_g - FNMR_h| ]

with a = 0.5 here. FDR = 1 is perfect parity; lower is worse.

Cross-corpus shift, on L2-normalised embeddings:

  * Frechet distance between Gaussian fits (the FID construction), which
    captures both mean and covariance shift:
        d^2 = ||mu_A - mu_B||^2 + tr(S_A + S_B - 2 (S_A S_B)^{1/2})
  * Linear channel probe: multinomial logistic regression predicting
    corpus-of-origin from the embedding. Chance is 1/n_corpora. Accuracy near
    1.0 means the embedding encodes recording channel at least as strongly as
    it encodes anything corpus-independent -- so cross-corpus comparisons are
    partly comparisons of microphones.
  * Centroid cosine between corpora, as a scale-free companion.

Usage:
    python tools/bias_transfer.py
"""
from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
from scipy import linalg

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
QC = os.path.join(DATA, "_qc")
EMB = os.path.join(QC, "embeddings")
OUT = os.path.join(QC, "bias_transfer")


# --------------------------------------------------------------------------- #
def eer_threshold(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    n_t, n_i = lab.sum(), len(lab) - lab.sum()
    fnr = 1.0 - np.cumsum(lab) / n_t
    fpr = np.cumsum(1 - lab) / n_i
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(scores[order][i]), float((fnr[i] + fpr[i]) / 2)


def subgroup_rates(scores, labels, groups, thr):
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        t = m & (labels == 1)
        i = m & (labels == 0)
        if t.sum() < 20 or i.sum() < 20:
            continue
        out[g] = {
            "targets": int(t.sum()), "impostors": int(i.sum()),
            "FNMR": round(float((scores[t] < thr).mean()), 4),
            "FMR": round(float((scores[i] >= thr).mean()), 4),
        }
    return out


def fdr(rates, alpha=0.5):
    if len(rates) < 2:
        return None
    fmr = [v["FMR"] for v in rates.values()]
    fnmr = [v["FNMR"] for v in rates.values()]
    return round(1.0 - (alpha * (max(fmr) - min(fmr))
                        + (1 - alpha) * (max(fnmr) - min(fnmr))), 4)


def bias_for(ds_name):
    """Per-gender FNMR/FMR at the pooled EER threshold, using cached trial scores."""
    lists_dir = os.path.join(DATA, ds_name, "lists")
    cache_dir = os.path.join(QC, "trial_emb")
    import csv
    gpath = os.path.join(DATA, ds_name, "metadata", "utterances.csv")
    gender = {}
    with open(gpath, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gender[r["path"]] = r["gender"]
    if len(set(gender.values()) - {"unk"}) < 2:
        return {"status": "no gender labels — subgroup analysis not possible"}

    res = {}
    for f in sorted(os.listdir(lists_dir)):
        if not (f.startswith("test_list") and f.endswith(".txt")):
            continue
        cache = os.path.join(cache_dir, f"{ds_name}__{os.path.splitext(f)[0]}.npz")
        if not os.path.isfile(cache):
            continue
        z = np.load(cache, allow_pickle=False)
        emb, paths = z["emb"], z["path"]
        idx = {p: i for i, p in enumerate(paths)}
        sc, lb, gr = [], [], []
        for line in open(os.path.join(lists_dir, f), encoding="utf-8"):
            p = line.split()
            if len(p) != 3 or p[1] not in idx or p[2] not in idx:
                continue
            ga, gb = gender.get(p[1], "unk"), gender.get(p[2], "unk")
            if ga == "unk" or gb == "unk" or ga != gb:
                continue          # subgroup = the gender of a same-gender pair
            sc.append(float(np.dot(emb[idx[p[1]]], emb[idx[p[2]]])))
            lb.append(int(p[0]))
            gr.append(ga)
        if len(sc) < 200:
            continue
        sc, lb, gr = np.array(sc), np.array(lb), np.array(gr)
        thr, eer = eer_threshold(sc, lb)
        rates = subgroup_rates(sc, lb, gr, thr)
        res[f] = {"pooled_EER_pct": round(eer * 100, 3),
                  "threshold": round(thr, 4),
                  "subgroups": rates, "FDR": fdr(rates),
                  "note": "Subgroup = gender of a same-gender trial pair; rates "
                          "are at the single pooled EER threshold."}
    return res or {"status": "no scored trial caches available"}


# --------------------------------------------------------------------------- #
def frechet(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    sa = np.cov(a, rowvar=False)
    sb = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(sa.dot(sb), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(sa + sb - 2 * covmean))


def channel_probe(sets, seed=42, n_per=2500):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, (name, e) in enumerate(sets):
        k = min(n_per, len(e))
        sel = rng.choice(len(e), k, replace=False)
        X.append(e[sel]); y.append(np.full(k, i))
    X = np.concatenate(X); y = np.concatenate(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=seed,
                                          stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte)), 1.0 / len(sets)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sorted(f[:-4] for f in os.listdir(EMB) if f.endswith(".npz"))
    sets = []
    for n in names:
        z = np.load(os.path.join(EMB, f"{n}.npz"), allow_pickle=False)
        sets.append((n, z["emb"]))
    print(f"[L6] {len(sets)} corpora loaded")

    # --- bias
    bias = {}
    for n in names:
        bias[n] = bias_for(n)
        st = bias[n].get("status")
        if st:
            print(f"[L6] {n}: {st}")
        else:
            for k, v in bias[n].items():
                print(f"[L6] {n}/{k}: FDR={v['FDR']} "
                      f"subgroups={ {g: (r['FNMR'], r['FMR']) for g, r in v['subgroups'].items()} }")

    # --- shift
    shift = {"frechet": {}, "centroid_cosine": {}}
    cents = {}
    for n, e in sets:
        c = e.mean(0); cents[n] = c / np.linalg.norm(c)
    for (na, ea), (nb, eb) in combinations(sets, 2):
        key = f"{na} | {nb}"
        rng = np.random.default_rng(0)
        a = ea[rng.choice(len(ea), min(4000, len(ea)), replace=False)]
        b = eb[rng.choice(len(eb), min(4000, len(eb)), replace=False)]
        shift["frechet"][key] = round(frechet(a, b), 4)
        shift["centroid_cosine"][key] = round(float(np.dot(cents[na], cents[nb])), 4)
        print(f"[L6] shift {key}: Frechet={shift['frechet'][key]} "
              f"centroid_cos={shift['centroid_cosine'][key]}")

    acc, chance = channel_probe(sets)
    shift["channel_probe"] = {
        "accuracy": round(acc, 4), "chance": round(chance, 4),
        "n_corpora": len(sets),
        "interpretation": ("Accuracy near 1.0 means corpus-of-origin is linearly "
                           "decodable from the speaker embedding, so the space "
                           "encodes recording channel as well as voice and "
                           "cross-corpus comparisons are partly comparisons of "
                           "microphones."),
    }
    print(f"[L6] channel probe: {acc:.3f} (chance {chance:.3f})")

    with open(os.path.join(OUT, "bias.json"), "w", encoding="utf-8") as fh:
        json.dump(bias, fh, indent=2)
    with open(os.path.join(OUT, "shift.json"), "w", encoding="utf-8") as fh:
        json.dump(shift, fh, indent=2)
    print(f"[L6] -> {OUT}")


if __name__ == "__main__":
    main()
