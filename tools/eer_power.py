#!/usr/bin/env python3
"""Layer 5: difficulty calibration and statistical power for each trial list.

Answers the question the whole assessment exists for: *can this corpus resolve
the EER differences our feature ablation is trying to rank?*

Mathematics:

  Scores are cosine similarities between L2-normalised ECAPA embeddings.

  With threshold t, over targets T and impostors I:
      FNR(t) = |{s in T : s < t}| / |T|      (miss rate)
      FPR(t) = |{s in I : s >= t}| / |I|     (false-alarm rate)
      EER    = FNR(t*) = FPR(t*) where FNR(t*) = FPR(t*)
  We report EER_avg = (FNR + FPR)/2 at the crossing (literature standard) and
  EER_max = max(FNR, FPR) (this repo's conservative convention).

  Normalised detection cost, for prior P and unit costs:
      DCF(t) = P * FNR(t) + (1 - P) * FPR(t)
      minDCF = min_t DCF(t) / min(P, 1 - P)

  SPEAKER-CLUSTERED BOOTSTRAP. Trials are not independent: every trial involving
  speaker S shares S's voice, channel and recording conditions, so resampling
  trials independently pretends we have more information than we do and produces
  intervals that are too narrow. We instead resample *speakers* with replacement:

      for b in 1..B:
          S_b   = sample(speakers, n_speakers, replace=True)
          T_b   = concat(trials grouped by speaker, for each speaker in S_b)
          EER_b = EER(T_b)
      CI_95 = [percentile(EER_b, 2.5), percentile(EER_b, 97.5)]
      SE    = std(EER_b)

  MINIMUM DETECTABLE EFFECT. For a two-sided test at alpha with power 1-beta,
  comparing two systems whose EERs correlate rho on the same trials:
      SE_diff = SE * sqrt(2 * (1 - rho))
      MDE     = (z_{1-alpha/2} + z_{1-beta}) * SE_diff
  With alpha=0.05 and power=0.8 the constant is 1.960 + 0.842 = 2.802.
  rho is unknown until two real systems are scored on the same list, so we
  report MDE at rho = 0 (conservative; systems unrelated) and rho = 0.7
  (typical for two variants of one architecture). The truth lies between.

Usage:
    python tools/eer_power.py --dataset data/slr65_tamil            # all its lists
    python tools/eer_power.py --all --bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
OUT_DIR = os.path.join(DATA, "_qc", "power")
Z_ALPHA, Z_POWER = 1.959964, 0.841621


# --------------------------------------------------------------------------- #
def eer_and_dcf(scores, labels, p_targets=(0.01, 0.05)):
    """scores: float array; labels: 1 = target, 0 = impostor."""
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    n_t = int(lab.sum())
    n_i = int(len(lab) - n_t)
    if n_t == 0 or n_i == 0:
        return None

    # sweep the threshold from high to low
    tp = np.cumsum(lab)                       # targets accepted
    fp = np.cumsum(1 - lab)                   # impostors accepted
    fnr = 1.0 - tp / n_t
    fpr = fp / n_i

    i = int(np.nanargmin(np.abs(fnr - fpr)))
    out = {
        "EER_avg": float((fnr[i] + fpr[i]) / 2),
        "EER_max": float(max(fnr[i], fpr[i])),
        "n_targets": n_t, "n_impostors": n_i,
    }
    for p in p_targets:
        dcf = p * fnr + (1 - p) * fpr
        out[f"minDCF_p{p}"] = float(dcf.min() / min(p, 1 - p))
    return out


def eer_only(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    n_t = lab.sum()
    n_i = len(lab) - n_t
    if n_t == 0 or n_i == 0:
        return np.nan
    fnr = 1.0 - np.cumsum(lab) / n_t
    fpr = np.cumsum(1 - lab) / n_i
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def speaker_clustered_bootstrap(scores, labels, spk, B, rng):
    """Resample speakers with replacement; recompute EER on the pooled trials."""
    speakers = np.unique(spk)
    groups = {s: np.where(spk == s)[0] for s in speakers}
    n = len(speakers)
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        draw = rng.choice(speakers, n, replace=True)
        idx = np.concatenate([groups[s] for s in draw])
        out[b] = eer_only(scores[idx], labels[idx])
    return out[~np.isnan(out)]


# --------------------------------------------------------------------------- #
def load_embeddings_for(paths, ds, cache, batch=96):
    """Embed exactly the utterances a trial list references (cached)."""
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        # materialise BOTH arrays once: NpzFile is lazy, so z["emb"][i] inside a
        # comprehension re-decompresses the whole array on every lookup
        cached_emb = z["emb"]
        cached_paths = z["path"]
        have = {p: i for i, p in enumerate(cached_paths)}
        if all(p in have for p in set(paths)):
            return {p: cached_emb[have[p]] for p in set(paths)}

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_embeddings import load_model, load_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    uniq = sorted(set(paths))
    embs, kept = [], []
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        wav, rel, ok = load_batch(chunk, ds)
        if wav is None:
            continue
        with torch.no_grad():
            e = model.encode_batch(wav.to(device), rel.to(device))
        e = e.squeeze(1).float().cpu().numpy()
        e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        embs.append(e)
        kept.extend(ok)
        if (i // batch) % 25 == 0:
            print(f"    embedding {i}/{len(uniq)}", flush=True)
    emb = np.concatenate(embs).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, emb=emb, path=np.array(kept))
    return {p: e for p, e in zip(kept, emb)}


def score_list(ds, list_path, cache_dir):
    trials = []
    with open(list_path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) == 3:
                trials.append((int(p[0]), p[1], p[2]))
    if not trials:
        return None

    name = os.path.splitext(os.path.basename(list_path))[0]
    paths = [t[1] for t in trials] + [t[2] for t in trials]
    cache = os.path.join(cache_dir, f"{os.path.basename(ds)}__{name}.npz")
    emb = load_embeddings_for(paths, ds, cache)

    keep = [t for t in trials if t[1] in emb and t[2] in emb]
    if len(keep) < len(trials):
        print(f"    note: {len(trials) - len(keep)} trials dropped (missing audio)")
    scores = np.array([float(np.dot(emb[a], emb[b])) for _, a, b in keep])
    labels = np.array([t[0] for t in keep])
    # primary speaker = speaker of the first utterance: <lang>/<spk>/<sess>/<utt>
    spk = np.array([t[1].split("/")[1] for t in keep])
    return scores, labels, spk


def run(ds, args):
    name = os.path.basename(ds.rstrip("/"))
    lists_dir = os.path.join(ds, "lists")
    if not os.path.isdir(lists_dir):
        return {}
    rng = np.random.default_rng(args.seed)
    results = {}
    for f in sorted(os.listdir(lists_dir)):
        if not (f.startswith("test_list") and f.endswith(".txt")):
            continue
        lp = os.path.join(lists_dir, f)
        print(f"[power] {name}/{f}", flush=True)
        got = score_list(ds, lp, args.cache_dir)
        if got is None:
            continue
        scores, labels, spk = got
        base = eer_and_dcf(scores, labels)
        if base is None:
            continue
        boot = speaker_clustered_bootstrap(scores, labels, spk, args.bootstrap, rng)
        se = float(boot.std(ddof=1))
        const = Z_ALPHA + Z_POWER
        results[f] = {
            **base,
            "distinct_speakers": int(len(np.unique(spk))),
            "bootstrap": {
                "B": int(len(boot)),
                "mean": round(float(boot.mean()) * 100, 4),
                "SE_pct": round(se * 100, 4),
                "CI95_pct": [round(float(np.percentile(boot, 2.5)) * 100, 4),
                             round(float(np.percentile(boot, 97.5)) * 100, 4)],
                "method": "speaker-clustered (resample speakers with replacement)",
            },
            "MDE_pct": {
                "rho_0.0_conservative": round(const * se * np.sqrt(2.0) * 100, 4),
                "rho_0.7_typical_paired": round(const * se * np.sqrt(2 * 0.3) * 100, 4),
                "alpha": 0.05, "power": 0.8,
            },
        }
        for k in ("EER_avg", "EER_max"):
            results[f][k] = round(results[f][k] * 100, 4)
        r = results[f]
        print(f"    EER {r['EER_avg']:.2f}%  CI95 {r['bootstrap']['CI95_pct']}  "
              f"MDE(rho=.7) {r['MDE_pct']['rho_0.7_typical_paired']:.2f}pp  "
              f"spk={r['distinct_speakers']}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=os.path.join(DATA, "_qc", "trial_emb"))
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    targets = args.dataset
    if args.all or not targets:
        targets = [os.path.join(DATA, d) for d in sorted(os.listdir(DATA))
                   if os.path.isdir(os.path.join(DATA, d, "lists"))]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "power_summary.json")
    # MERGE into any existing summary: running with --dataset for one corpus
    # must not wipe the results already computed for the others
    summary = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}
    for ds in targets:
        ds = ds if os.path.isabs(ds) else os.path.join(REPO, ds)
        summary[os.path.basename(ds.rstrip("/"))] = run(ds, args)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"[power] -> {os.path.join(args.out, 'power_summary.json')}")


if __name__ == "__main__":
    main()
