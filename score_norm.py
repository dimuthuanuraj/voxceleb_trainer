#! /usr/bin/python
# -*- encoding: utf-8 -*-
"""
FEATURE-002 - Adaptive Symmetric Normalisation (AS-Norm) for trial scores.

Pure functions; no Trainer / model class coupling. The encoder is passed in
as a callable so this module is independently testable with a mock model.

Score metric matches evaluateFromList:
    score(a, b) = -mean(cdist(a_eval, b_eval))    over num_eval segments
where a_eval / b_eval have shape [num_eval, dim]. Cohort scoring uses the
identical formula so the AS-Norm anchors are on the same scale as the
trial. See docs/bugfixes/FEATURE-002-as-norm-score-normalisation.md.
"""

import os
import sys
import time

import numpy
import torch
import torch.nn.functional as F

from DatasetLoader import test_dataset_loader


# ----- Cohort cache I/O -------------------------------------------------------

def _save_cohort_cache(path, cohort_feats, cohort_paths):
    torch.save({'feats': cohort_feats, 'paths': cohort_paths}, path)


def _load_cohort_cache(path):
    # BUGFIX-019: weights_only=True. We wrote this file ourselves moments
    # earlier; the strict loader is a no-op for the expected payload but
    # blocks the pickle-RCE vector should the cohort file be tampered with.
    blob = torch.load(path, map_location='cpu', weights_only=True)
    return blob['feats'], blob['paths']


# ----- Cohort embedding extraction --------------------------------------------

def _read_cohort_paths(cohort_list):
    """Read cohort list file. Each non-empty line yields one path; if the
    line has multiple whitespace-separated fields, the LAST one is taken
    (so a `<spk> <path>` or VoxCeleb `<label> <enrol> <test>` line works,
    though only the final field is consumed).

    De-duplicates while preserving first-seen order.
    """
    seen = set()
    paths = []
    with open(cohort_list) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()[-1]
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def extract_cohort_embeddings(
    model,
    cohort_list,
    cohort_path,
    nDataLoaderThread,
    num_eval=10,
    eval_frames=0,
    sample_rate=16000,
    cache_file=None,
    device='cuda',
    rank=0,
    print_interval=50,
    **kwargs,
):
    """Run `model` over every utterance in cohort_list.

    Returns (cohort_feats, paths) where cohort_feats has shape
    [cohort_size, num_eval, dim] and paths is the matching list.

    `model` is the inner module (so callers should pass `trainer.__model__`),
    invoked with a tensor of shape [num_eval, T]. Output is reshaped to
    [num_eval, dim].
    """
    paths = _read_cohort_paths(cohort_list)
    if not paths:
        raise ValueError(f"AS-Norm cohort list {cohort_list} is empty.")

    if cache_file and os.path.exists(cache_file):
        feats, cached_paths = _load_cohort_cache(cache_file)
        if cached_paths == paths:
            if rank == 0:
                print(f"[AS-Norm] Loaded {len(paths)} cohort embeddings from {cache_file}")
            return feats, paths
        if rank == 0:
            print(f"[AS-Norm] Cohort cache {cache_file} stale (paths differ); re-extracting.")

    dataset = test_dataset_loader(
        paths, cohort_path,
        eval_frames=eval_frames, num_eval=num_eval, sample_rate=sample_rate,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=nDataLoaderThread, drop_last=False,
    )

    feats_list = []
    was_training = model.training
    model.eval()
    tstart = time.time()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            inp = data[0][0].to(device, non_blocking=True)  # [num_eval, T]
            emb = model(inp).detach().cpu()                 # [num_eval, dim]
            feats_list.append(emb)
            if rank == 0 and (idx % print_interval == 0):
                elapsed = max(1e-6, time.time() - tstart)
                sys.stdout.write(
                    f"\r[AS-Norm] cohort {idx + 1}/{len(paths)} "
                    f"({(idx + 1) / elapsed:.1f} Hz, dim {emb.size(-1)})"
                )
                sys.stdout.flush()
    if rank == 0:
        print()
    if was_training:
        model.train()

    dim = feats_list[0].size(-1)
    feats = torch.stack(feats_list, dim=0).reshape(len(paths), num_eval, dim)

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        _save_cohort_cache(cache_file, feats, paths)
        if rank == 0:
            print(f"[AS-Norm] Saved cohort cache to {cache_file}")

    return feats, paths


# ----- Per-file cohort statistics --------------------------------------------

def compute_file_cohort_stats(
    file_iter,
    cohort_feats,
    top_k=300,
    normalize=True,
    device='cuda',
    chunk_size=64,
    rank=0,
    print_interval=200,
):
    """For every file embedding produced by `file_iter`, compute (mu, sigma)
    of the top-K cohort scores.

    Parameters
    ----------
    file_iter : iterable of (path: str, feat: Tensor[num_eval, dim])
    cohort_feats : Tensor[cohort_size, num_eval, dim]
    top_k : int; clamped to cohort_size
    normalize : bool. If True, L2-normalise both file and cohort features
        before scoring. This MUST match the trial-loop branch
        (`if self.__model__.module.__L__.test_normalize`).
    chunk_size : files batched per cdist call.

    Returns
    -------
    dict mapping path -> (mu, sigma) of top-K cohort scores. sigma is
    floored at 1e-8 to keep the AS-Norm division numerically safe.
    """
    cohort_size, num_eval, dim = cohort_feats.shape
    cohort_dev = cohort_feats.to(device).reshape(cohort_size * num_eval, dim)
    if normalize:
        cohort_dev = F.normalize(cohort_dev, p=2, dim=1)
    top_k_eff = min(int(top_k), cohort_size)

    stats = {}
    buffer_paths = []
    buffer_feats = []
    total_done = 0
    tstart = time.time()

    def _flush():
        nonlocal total_done
        if not buffer_paths:
            return
        batch = torch.stack(buffer_feats, dim=0).to(device)        # [B, num_eval, dim]
        B = batch.size(0)
        flat = batch.reshape(B * num_eval, dim)
        if normalize:
            flat = F.normalize(flat, p=2, dim=1)
        dist = torch.cdist(flat, cohort_dev)                       # [B*num_eval, C*num_eval]
        dist = dist.reshape(B, num_eval, cohort_size, num_eval)
        scores = -dist.mean(dim=(1, 3))                            # [B, cohort_size]
        topk = scores.topk(top_k_eff, dim=1).values                # [B, K]
        mu = topk.mean(dim=1).cpu().numpy()
        sigma = topk.std(dim=1).cpu().numpy()
        for i, p in enumerate(buffer_paths):
            stats[p] = (float(mu[i]), float(sigma[i]) + 1e-8)
        total_done += B
        if rank == 0:
            elapsed = max(1e-6, time.time() - tstart)
            sys.stdout.write(
                f"\r[AS-Norm] file stats {total_done} ({total_done / elapsed:.1f} Hz, top_K={top_k_eff})"
            )
            sys.stdout.flush()
        buffer_paths.clear()
        buffer_feats.clear()

    for path, feat in file_iter:
        # Accept [num_eval, dim] or flat [num_eval * dim]; reshape defensively.
        f = feat.reshape(num_eval, dim) if feat.dim() == 1 else feat
        if f.shape != (num_eval, dim):
            f = f.reshape(num_eval, dim)
        buffer_paths.append(path)
        buffer_feats.append(f)
        if len(buffer_paths) >= chunk_size:
            _flush()
    _flush()
    if rank == 0:
        print()
    return stats


# ----- AS-Norm application ----------------------------------------------------

def apply_as_norm(scores, trials, file_stats):
    """Apply the AS-Norm formula to a list of raw trial scores.

    Parameters
    ----------
    scores : list[float] of length N
    trials : list[str] of length N; each "enrol_path test_path"
    file_stats : dict path -> (mu, sigma)

    Returns
    -------
    list[float] of normalised scores. KeyError if a trial file has no
    stats — that surfaces a bug (cohort stats were not computed for some
    file) rather than silently leaving the score raw.
    """
    out = []
    for s, trial in zip(scores, trials):
        enrol, test = trial.split()
        mu_e, sd_e = file_stats[enrol]
        mu_t, sd_t = file_stats[test]
        s_as = 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)
        out.append(float(s_as))
    return out
