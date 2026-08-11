#!/usr/bin/env python3
"""Cllr and minCllr for every scored trial list (the L5 gap).

EER says how well a system RANKS. Cllr says how well its scores work as
likelihood ratios -- which is what matters the moment a threshold is fixed, and
what the earlier cross-language calibration finding in this project was about.

Mathematics
-----------
For log-likelihood ratios llr over targets T and non-targets N:

    Cllr = 1/2 [ (1/|T|) sum_{i in T} log2(1 + e^{-llr_i})
               + (1/|N|) sum_{j in N} log2(1 + e^{+llr_j}) ]

Cllr = 1 is the useless system (always answering "no information"); lower is
better. Raw cosine similarities are NOT log-likelihood ratios, so they must be
calibrated first. We fit the standard 1-D logistic map

    llr = a * s + b

by maximum likelihood on the same trials, which gives Cllr, and separately
compute minCllr by replacing the scores with the optimal monotonic mapping found
by the pool-adjacent-violators (PAV) algorithm. PAV is order-preserving, so
minCllr is the discrimination loss alone -- the part no threshold choice can
remove -- and

    calibration loss = Cllr - minCllr

is what a better-calibrated system could recover for free.

Note both numbers here are optimistic: the calibration is fitted on the very
trials it is scored against, so this is a floor on Cllr, not an estimate of what
a held-out calibration would give.

Usage:
    python tools/calibration.py
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
QC = os.path.join(DATA, "_qc")
CACHE = os.path.join(QC, "trial_emb")
OUT = os.path.join(QC, "calibration")


def cllr(llr_t, llr_n):
    return float(0.5 * (np.mean(np.log2(1 + np.exp(-llr_t)))
                        + np.mean(np.log2(1 + np.exp(llr_n)))))


def pav(y, w=None):
    """Pool-adjacent-violators: least-squares monotonic fit to y."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    val, wt, idx = [], [], []
    for i in range(n):
        val.append(y[i]); wt.append(w[i]); idx.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2, i2 = val.pop(), wt.pop(), idx.pop()
            v1, w1, i1 = val.pop(), wt.pop(), idx.pop()
            val.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wt.append(w1 + w2); idx.append(i1 + i2)
    out = np.empty(n)
    p = 0
    for v, c in zip(val, idx):
        out[p:p + c] = v
        p += c
    return out


def min_cllr(scores, labels, eps=1e-6):
    """PAV-based minCllr: optimal monotonic score->posterior, then to LLR."""
    order = np.argsort(scores, kind="mergesort")
    lab = labels[order].astype(float)
    post = np.clip(pav(lab), eps, 1 - eps)
    prior = labels.mean()
    llr = np.log(post / (1 - post)) - np.log(prior / (1 - prior))
    lt = llr[lab == 1]
    ln = llr[lab == 0]
    return cllr(lt, ln)


def fit_logistic(scores, labels, iters=200):
    """1-D logistic calibration s -> llr = a*s + b, by Newton/IRLS."""
    x = np.column_stack([scores, np.ones_like(scores)])
    beta = np.zeros(2)
    for _ in range(iters):
        z = x @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        g = x.T @ (labels - p)
        w = np.clip(p * (1 - p), 1e-9, None)
        h = (x * w[:, None]).T @ x + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    prior = labels.mean()
    return beta, float(np.log(prior / (1 - prior)))


def run():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".npz"):
            continue
        ds, lst = f[:-4].split("__", 1)
        list_path = os.path.join(DATA, ds, "lists", f"{lst}.txt")
        if not os.path.isfile(list_path):
            continue
        z = np.load(os.path.join(CACHE, f), allow_pickle=False)
        emb, paths = z["emb"], z["path"]
        idx = {p: i for i, p in enumerate(paths)}
        sc, lb = [], []
        for line in open(list_path, encoding="utf-8"):
            p = line.split()
            if len(p) != 3 or p[1] not in idx or p[2] not in idx:
                continue
            sc.append(float(np.dot(emb[idx[p[1]]], emb[idx[p[2]]])))
            lb.append(int(p[0]))
        if len(sc) < 200:
            continue
        sc = np.asarray(sc, dtype=float)
        lb = np.asarray(lb, dtype=float)

        beta, logit_prior = fit_logistic(sc, lb)
        llr = beta[0] * sc + beta[1] - logit_prior
        c = cllr(llr[lb == 1], llr[lb == 0])
        mc = min_cllr(sc, lb)
        res.setdefault(ds, {})[f"{lst}.txt"] = {
            "Cllr": round(c, 4),
            "minCllr": round(mc, 4),
            "calibration_loss": round(c - mc, 4),
            "trials": int(len(sc)),
            "calibration": {"a": round(float(beta[0]), 4),
                            "b": round(float(beta[1]), 4)},
            "note": "Calibration fitted on the same trials it is scored on, so "
                    "Cllr here is a floor, not a held-out estimate.",
        }
        print(f"[cal] {ds}/{lst}: Cllr={c:.4f} minCllr={mc:.4f} "
              f"loss={c - mc:.4f}", flush=True)

    with open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"[cal] -> {os.path.join(OUT, 'calibration.json')}")


if __name__ == "__main__":
    run()
