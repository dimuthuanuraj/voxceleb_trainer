#!/usr/bin/env python3
"""
P2 (2026-07-03 roadmap) — training-free backend adaptation ablation:
AS-Norm, PLDA, score calibration, alone and composed (thesis RQ4).

Runs entirely on cached embeddings (no audio, no GPU):
    test-file embeddings   from tools/zeroshot_eval.py runs   (--test_cache)
    cohort + PLDA-train    from tools/extract_files.py runs   (--train_cache)

Conditions per model and per trial list:
    cosine                  raw baseline (must reproduce the P1 numbers)
    cosine+asnorm           adaptive S-norm, top-K cohort (Matejka 2017)
    plda                    TwoCovPLDA (repo FEATURE-010) fit on SL train embs
    plda+asnorm             PLDA scoring, AS-Norm with PLDA cohort scores
Each condition is then calibrated (logistic regression -> LLR) on a held-out
calibration half of the trials, evaluated on the other half:
    self-cal                calibrator fit on the same language
    cross-cal               calibrator fit on the OTHER language (quantifies
                            the cross-lingual score shift, arXiv:2110.09150)

Metrics: EER_avg, minDCF(0.01/0.05) on full lists; actDCF(0.01) and Cllr on
the eval half (before/after calibration).

Usage:
    python backend_adapt.py \
        --test_cache emb_cache --train_cache emb_cache_train \
        --trials test_list_si.txt test_list_ta.txt \
        --cohort p2_cohort.txt --plda_list p2_plda_list.txt \
        --models speechbrain_ecapa redimnet:b1 redimnet:b6 \
        --out results/p2_backend.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf  # noqa: E402
from plda import TwoCovPLDA  # noqa: E402


# ---------------------------------------------------------------- embeddings

def cache_path(cache_dir, spec):
    safe = spec.replace(':', '_').replace('/', '_')
    return Path(cache_dir) / f"{safe}.npz"


def load_cache(cache_dir, spec):
    z = np.load(cache_path(cache_dir, spec), allow_pickle=False)
    return {k: z[k] for k in z.files}


def read_trials(path):
    trials = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) == 3:
                trials.append((int(parts[0]), parts[1], parts[2]))
    return trials


def read_labeled(path):
    rows = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) == 2:
                rows.append((int(parts[0]), parts[1]))
    return rows


# ---------------------------------------------------------------- scoring

def cosine_scores(embs, trials):
    return np.array([float(np.dot(embs[a], embs[b])) for _l, a, b in trials])


def plda_score_matrix(plda, X, Y):
    """All-pairs PLDA scores between rows of X and rows of Y (fast path
    using the fitted quadratic forms)."""
    ZX = plda._preprocess(np.asarray(X, dtype=np.float64))
    ZY = plda._preprocess(np.asarray(Y, dtype=np.float64))
    qX = np.einsum('nd,de,ne->n', ZX, plda.Q, ZX)
    qY = np.einsum('md,de,me->m', ZY, plda.Q, ZY)
    cross = ZX @ plda.P @ ZY.T
    return 0.5 * (qX[:, None] + qY[None, :]) + cross


def asnorm(scores, trials, file_stats):
    out = []
    for s, (_l, a, b) in zip(scores, trials):
        mu_a, sd_a = file_stats[a]
        mu_b, sd_b = file_stats[b]
        out.append(0.5 * ((s - mu_a) / sd_a + (s - mu_b) / sd_b))
    return np.array(out)


def topk_stats(sim_matrix, files, k):
    """file -> (mu, sd) of its top-K cohort similarities."""
    stats = {}
    for i, f in enumerate(files):
        row = sim_matrix[i]
        top = np.sort(row)[-k:]
        stats[f] = (float(top.mean()), float(top.std() + 1e-8))
    return stats


# ---------------------------------------------------------------- metrics

def base_metrics(scores, labels):
    res = tuneThresholdfromScore(list(scores), list(labels), [1, 0.1])
    fnrs, fprs, ths = ComputeErrorRates(list(scores), list(labels))
    dcf01, _ = ComputeMinDcf(fnrs, fprs, ths, 0.01, 1, 1)
    dcf05, _ = ComputeMinDcf(fnrs, fprs, ths, 0.05, 1, 1)
    return {'eer_avg': round(res[5], 4),
            'mindcf_p01': round(dcf01, 4), 'mindcf_p05': round(dcf05, 4)}


def act_dcf(llrs, labels, p_target):
    thr = np.log((1 - p_target) / p_target)
    llrs = np.asarray(llrs); labels = np.asarray(labels)
    p_miss = float(np.mean(llrs[labels == 1] < thr)) if (labels == 1).any() else 0.0
    p_fa = float(np.mean(llrs[labels == 0] >= thr)) if (labels == 0).any() else 0.0
    return (p_target * p_miss + (1 - p_target) * p_fa) / min(p_target, 1 - p_target)


def cllr(llrs, labels):
    llrs = np.asarray(llrs); labels = np.asarray(labels)
    tar = llrs[labels == 1]; non = llrs[labels == 0]
    c = 0.5 * (np.mean(np.log2(1 + np.exp(-tar)))
               + np.mean(np.log2(1 + np.exp(non))))
    return float(c)


def fit_calibrator(scores, labels):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6)  # near-unregularized affine calibration
    lr.fit(np.asarray(scores).reshape(-1, 1), np.asarray(labels))
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return lambda s: a * np.asarray(s) + b, (a, b)


def split_half(trials, seed=42):
    """Stratified 50/50 calibration/eval split."""
    import random as _r
    rng = _r.Random(seed)
    tar = [i for i, t in enumerate(trials) if t[0] == 1]
    non = [i for i, t in enumerate(trials) if t[0] == 0]
    rng.shuffle(tar); rng.shuffle(non)
    cal = set(tar[:len(tar)//2] + non[:len(non)//2])
    return cal


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="P2 backend adaptation ablation.")
    p.add_argument('--test_cache', required=True)
    p.add_argument('--train_cache', required=True)
    p.add_argument('--trials', nargs='+', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--plda_list', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--topk', type=int, default=300)
    p.add_argument('--plda_dim', type=int, default=150)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cohort_files = [ln.split()[-1] for ln in open(args.cohort) if ln.strip()]
    plda_rows = read_labeled(args.plda_list)
    trial_sets = {Path(t).stem.replace('test_list_', ''): read_trials(t)
                  for t in args.trials}

    results = {}
    for spec in args.models:
        print(f"\n===== {spec} =====", flush=True)
        test_embs = load_cache(args.test_cache, spec)
        train_embs = load_cache(args.train_cache, spec)
        coh = np.stack([train_embs[f] for f in cohort_files])
        plda_X = np.stack([train_embs[rel] for _l, rel in plda_rows])
        plda_y = np.array([l for l, _r in plda_rows])

        plda = TwoCovPLDA(lda_dim=args.plda_dim)
        plda.fit(plda_X, plda_y)

        per_model = {}
        for lname, trials in trial_sets.items():
            labels = np.array([l for l, _a, _b in trials])
            files = sorted({f for _l, a, b in trials for f in (a, b)})
            T = np.stack([test_embs[f] for f in files])

            conditions = {}
            # cosine
            conditions['cosine'] = cosine_scores(test_embs, trials)
            # cosine + AS-Norm
            cos_sim = T @ coh.T
            stats_c = topk_stats(cos_sim, files, args.topk)
            conditions['cosine+asnorm'] = asnorm(
                conditions['cosine'], trials, stats_c)
            # PLDA
            idx = {f: i for i, f in enumerate(files)}
            Pm = plda_score_matrix(plda, T, T)
            conditions['plda'] = np.array(
                [Pm[idx[a], idx[b]] for _l, a, b in trials])
            # PLDA + AS-Norm (cohort scored with PLDA)
            Pc = plda_score_matrix(plda, T, coh)
            stats_p = topk_stats(Pc, files, args.topk)
            conditions['plda+asnorm'] = asnorm(
                conditions['plda'], trials, stats_p)

            per_model[lname] = {}
            for cname, scores in conditions.items():
                m = base_metrics(scores, labels)
                # calibration on held-out half
                cal_idx = split_half(trials, args.seed)
                s_cal = [scores[i] for i in range(len(trials)) if i in cal_idx]
                y_cal = [labels[i] for i in range(len(trials)) if i in cal_idx]
                s_ev = [scores[i] for i in range(len(trials)) if i not in cal_idx]
                y_ev = [labels[i] for i in range(len(trials)) if i not in cal_idx]
                calib, ab = fit_calibrator(s_cal, y_cal)
                llr_ev = calib(s_ev)
                m['selfcal'] = {
                    'actdcf_p01': round(act_dcf(llr_ev, y_ev, 0.01), 4),
                    'cllr': round(cllr(llr_ev, y_ev), 4),
                    'coef': [round(x, 4) for x in ab]}
                m['raw_eval_half'] = {
                    'actdcf_p01': round(act_dcf(np.asarray(s_ev), y_ev, 0.01), 4),
                    'cllr': round(cllr(np.asarray(s_ev), y_ev), 4)}
                per_model[lname][cname] = m
                print(f"[p2] {spec} | {lname} | {cname}: "
                      f"EER {m['eer_avg']}%  minDCF01 {m['mindcf_p01']}  "
                      f"Cllr {m['selfcal']['cllr']}", flush=True)

        # cross-lingual calibration: fit on one language's cal half, apply to
        # the other's eval half (per condition) — the score-shift measurement.
        langs = list(trial_sets.keys())
        if len(langs) == 2:
            xc = {}
            for cname in ['cosine', 'cosine+asnorm', 'plda', 'plda+asnorm']:
                xc[cname] = {}
                for src, dst in [(langs[0], langs[1]), (langs[1], langs[0])]:
                    tr_s, tr_d = trial_sets[src], trial_sets[dst]
                    lab_s = np.array([l for l, _a, _b in tr_s])
                    lab_d = np.array([l for l, _a, _b in tr_d])
                    # recompute scores per condition (cheap, cached embs)
                    def cond_scores(trials, lname):
                        fs = sorted({f for _l, a, b in trials for f in (a, b)})
                        Tm = np.stack([test_embs[f] for f in fs])
                        ix = {f: i for i, f in enumerate(fs)}
                        if cname.startswith('cosine'):
                            s = cosine_scores(test_embs, trials)
                            if cname.endswith('asnorm'):
                                st = topk_stats(Tm @ coh.T, fs, args.topk)
                                s = asnorm(s, trials, st)
                        else:
                            M = plda_score_matrix(plda, Tm, Tm)
                            s = np.array([M[ix[a], ix[b]] for _l, a, b in trials])
                            if cname.endswith('asnorm'):
                                st = topk_stats(plda_score_matrix(plda, Tm, coh),
                                                fs, args.topk)
                                s = asnorm(s, trials, st)
                        return s
                    s_src = cond_scores(tr_s, src)
                    s_dst = cond_scores(tr_d, dst)
                    cal_idx = split_half(tr_s, args.seed)
                    calib, _ab = fit_calibrator(
                        [s_src[i] for i in sorted(cal_idx)],
                        [lab_s[i] for i in sorted(cal_idx)])
                    ev_idx = [i for i in range(len(tr_d))
                              if i not in split_half(tr_d, args.seed)]
                    llr = calib([s_dst[i] for i in ev_idx])
                    y = [lab_d[i] for i in ev_idx]
                    xc[cname][f'{src}->{dst}'] = {
                        'actdcf_p01': round(act_dcf(llr, y, 0.01), 4),
                        'cllr': round(cllr(llr, y), 4)}
            per_model['crosscal'] = xc
        results[spec] = per_model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'topk': args.topk, 'plda_dim': args.plda_dim,
                   'results': results}, f, indent=2)
    # markdown summary
    md = out.with_suffix('.md')
    with open(md, 'w') as f:
        f.write("| Model | Lang | Condition | EER % | minDCF .01 | minDCF .05 "
                "| Cllr (self-cal) | actDCF .01 (self-cal) |\n|---|---|---|---|---|---|---|---|\n")
        for spec, per in results.items():
            for lname, conds in per.items():
                if lname == 'crosscal':
                    continue
                for cname, m in conds.items():
                    f.write(f"| {spec} | {lname} | {cname} | {m['eer_avg']} "
                            f"| {m['mindcf_p01']} | {m['mindcf_p05']} "
                            f"| {m['selfcal']['cllr']} "
                            f"| {m['selfcal']['actdcf_p01']} |\n")
    print(f"\n[p2] wrote {out} and {md}")


if __name__ == '__main__':
    main()
