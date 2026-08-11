#! /usr/bin/python
# -*- encoding: utf-8 -*-
"""
FEATURE-010 - Two-covariance PLDA (Probabilistic Linear Discriminant Analysis)
for speaker verification scoring.

Pure NumPy; no Trainer / model class coupling. The encoder is passed in
as a callable so this module is independently testable. Preprocessing
pipeline (centre -> length-normalise -> LDA -> centre) is fixed inside
the PLDA object so the same transform is applied at fit time and at
score time.

Math reference:
- Sizov, Lee, Kinnunen "Unifying PLDA Variants" (S+SSPR 2014), eqs 11-16.
- Garcia-Romero, Espy-Wilson, "Analysis of i-vector Length Normalisation"
  (Interspeech 2011) for the length-norm preprocessing.

Scoring formula (for centred + length-normed + LDA'd + centred x1, x2):
    score = 0.5 * (x1.T Q x1 + x2.T Q x2) + x1.T P x2
with
    Q = (Sigma_b + Sigma_w)^-1 - (Sigma_b + 2 Sigma_w)^-1
    P = (Sigma_b + 2 Sigma_w)^-1 - Q   (after some algebra)
Constants dropped (they cancel in EER / DCF).
"""

import os
import sys
import time
import pickle

import numpy as np
import torch

from DatasetLoader import test_dataset_loader


# ----- PLDA class -------------------------------------------------------------

class TwoCovPLDA:
    """Simplified two-covariance PLDA. Preprocessing + fit + score in one
    pickle-able object. Use cases match the FEATURE-002 AS-Norm sibling:
    fit once after main training, persist to <save_path>/plda.pkl, load
    on subsequent eval runs.
    """

    def __init__(self, lda_dim=200, eig_floor=1e-6):
        self.lda_dim = int(lda_dim)
        self.eig_floor = float(eig_floor)
        # Set by fit():
        self.mean1 = None        # [D] global mean before LDA
        self.lda_W = None        # [D, lda_dim] LDA projection
        self.mean2 = None        # [lda_dim] mean after LDA
        self.Sigma_b = None      # [lda_dim, lda_dim]
        self.Sigma_w = None      # [lda_dim, lda_dim]
        self.Q = None            # [lda_dim, lda_dim] scoring matrix
        self.P = None            # [lda_dim, lda_dim] scoring matrix
        self.n_speakers = 0
        self.n_utts = 0

    # -- preprocessing ---------------------------------------------------------

    def _length_norm(self, X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, 1e-12, None)

    def _preprocess(self, X):
        """Apply the four-step preprocessing pipeline. X: [N, D]."""
        X = X - self.mean1                # 1. centre
        X = self._length_norm(X)          # 2. length-norm
        X = X @ self.lda_W                # 3. LDA
        X = X - self.mean2                # 4. centre again
        return X

    # -- fit -------------------------------------------------------------------

    def fit(self, embeddings, labels):
        """Closed-form fit.

        Parameters
        ----------
        embeddings : np.ndarray [N, D]
        labels     : np.ndarray [N] (integer speaker IDs; values don't
                     need to be contiguous)
        """
        X = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels)
        N, D = X.shape
        unique = np.unique(y)
        S = len(unique)
        if S < 2 * self.lda_dim:
            raise ValueError(
                f"PLDA fit needs >= 2*lda_dim={2*self.lda_dim} speakers; "
                f"got {S}. Reduce --plda_dim or extend the PLDA train list."
            )

        # 1. global mean
        self.mean1 = X.mean(axis=0)
        Xc = X - self.mean1
        # 2. length-norm
        Xc = self._length_norm(Xc)
        # 3. LDA fit
        self.lda_W = self._fit_lda(Xc, y, unique)
        Xl = Xc @ self.lda_W
        # 4. centre in LDA space
        self.mean2 = Xl.mean(axis=0)
        Xl = Xl - self.mean2

        # 5. covariances. Group speakers, compute per-speaker means.
        per_speaker_mean = {}
        per_speaker_count = {}
        for s in unique:
            mask = y == s
            per_speaker_mean[s] = Xl[mask].mean(axis=0)
            per_speaker_count[s] = int(mask.sum())

        # Within-speaker: Sigma_w = 1/(N - S) * sum_{i,s} (x_is - mu_s) (x_is - mu_s).T
        Sw = np.zeros((self.lda_dim, self.lda_dim), dtype=np.float64)
        for s in unique:
            mu_s = per_speaker_mean[s]
            diff = Xl[y == s] - mu_s
            Sw += diff.T @ diff
        Sw /= max(1, N - S)

        # Between-speaker: Sigma_b = 1/(S - 1) * sum_s n_s * (mu_s - mu_global) (mu_s - mu_global).T
        # mu_global = 0 after step 4.
        Sb = np.zeros((self.lda_dim, self.lda_dim), dtype=np.float64)
        for s in unique:
            mu_s = per_speaker_mean[s]
            n_s = per_speaker_count[s]
            Sb += n_s * np.outer(mu_s, mu_s)
        Sb /= max(1, S - 1)

        self.Sigma_b = Sb
        self.Sigma_w = Sw

        # 6. precompute scoring matrices. Floor eigenvalues for safety.
        # See Sizov et al. eq. 16:
        #   score = 0.5 (x1.T Q x1 + x2.T Q x2) + x1.T P x2 + const
        # where
        #   A_tot = Sigma_b + Sigma_w     (single-utterance total covariance)
        #   A_2   = Sigma_b + 2*Sigma_w   (joint covariance under H_diff)
        # After algebra:
        #   Q = A_tot^-1 - A_2^-1
        #   P = something derived from A_2^-1 and Sigma_b
        # The cleanest form (also Sizov):
        #   P = A_2^-1 Sigma_b A_tot^-1
        #   Q = A_tot^-1 - A_2^-1
        A_tot = Sb + Sw
        A_2 = Sb + 2 * Sw
        A_tot_inv = self._safe_inv(A_tot)
        A_2_inv = self._safe_inv(A_2)
        self.Q = A_tot_inv - A_2_inv
        self.P = A_2_inv @ Sb @ A_tot_inv

        self.n_speakers = S
        self.n_utts = N
        return self

    def _safe_inv(self, M):
        # Symmetric eigen-decomposition with eigenvalue flooring before
        # inversion — robust to near-singular covariance from small train
        # sets without ever crashing.
        w, V = np.linalg.eigh((M + M.T) / 2)
        w = np.clip(w, self.eig_floor, None)
        return (V * (1.0 / w)) @ V.T

    def _fit_lda(self, X, y, unique):
        """Standard LDA: solve Sb v = lambda Sw v, take top-`lda_dim` eigvecs.
        Input X is assumed already length-normed and centred."""
        N, D = X.shape
        Sw = np.zeros((D, D), dtype=np.float64)
        Sb = np.zeros((D, D), dtype=np.float64)
        mu_global = X.mean(axis=0)
        for s in unique:
            Xs = X[y == s]
            mu_s = Xs.mean(axis=0)
            d = Xs - mu_s
            Sw += d.T @ d
            md = mu_s - mu_global
            Sb += len(Xs) * np.outer(md, md)
        # Sw^{-1/2} Sb Sw^{-1/2} eigenvectors
        # Use eigh on Sw and decompose stably.
        w_w, V_w = np.linalg.eigh((Sw + Sw.T) / 2 + self.eig_floor * np.eye(D))
        w_w = np.clip(w_w, self.eig_floor, None)
        Sw_inv_sqrt = V_w @ np.diag(1.0 / np.sqrt(w_w)) @ V_w.T
        M = Sw_inv_sqrt @ Sb @ Sw_inv_sqrt
        w_m, V_m = np.linalg.eigh((M + M.T) / 2)
        # Take top lda_dim eigenvectors (eigh returns ascending; flip).
        idx = np.argsort(w_m)[::-1][: self.lda_dim]
        W = Sw_inv_sqrt @ V_m[:, idx]
        return W

    # -- score -----------------------------------------------------------------

    def score(self, x1, x2):
        """Score one or many trial pairs.

        Parameters
        ----------
        x1, x2 : np.ndarray of shape [D] or [N, D] (pre-preprocessing).
                 Each row is one trial-side embedding.

        Returns
        -------
        float (single pair) or np.ndarray [N] (batched).
        """
        single = (x1.ndim == 1)
        x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
        x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
        z1 = self._preprocess(x1)
        z2 = self._preprocess(x2)
        # Batched: row i of z1 paired with row i of z2.
        q1 = np.einsum('nd,de,ne->n', z1, self.Q, z1)
        q2 = np.einsum('nd,de,ne->n', z2, self.Q, z2)
        p12 = np.einsum('nd,de,ne->n', z1, self.P, z2)
        s = 0.5 * (q1 + q2) + p12
        return float(s[0]) if single else s

    # -- save / load -----------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'lda_dim': self.lda_dim,
                'eig_floor': self.eig_floor,
                'mean1': self.mean1, 'lda_W': self.lda_W, 'mean2': self.mean2,
                'Sigma_b': self.Sigma_b, 'Sigma_w': self.Sigma_w,
                'Q': self.Q, 'P': self.P,
                'n_speakers': self.n_speakers, 'n_utts': self.n_utts,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        obj = cls(lda_dim=d['lda_dim'], eig_floor=d['eig_floor'])
        for k in ('mean1', 'lda_W', 'mean2', 'Sigma_b', 'Sigma_w',
                  'Q', 'P', 'n_speakers', 'n_utts'):
            setattr(obj, k, d[k])
        return obj


# ----- Embedding extraction for the PLDA train set ----------------------------

def extract_plda_train_embeddings(
    model,
    train_list_path,
    train_path,
    nDataLoaderThread,
    num_eval=10,
    eval_frames=0,
    sample_rate=16000,
    device='cuda',
    rank=0,
    print_interval=200,
    **kwargs,
):
    """Run `model` over every utterance in train_list_path, return
    (embeddings: [N, D] float64, labels: [N] int64).

    train_list_path format: '<spk_label_int> <relative_path>' per line
    (the standard repo train_list format). Mean-pools across the num_eval
    crops to produce one embedding per file.
    """
    paths = []
    labels = []
    with open(train_list_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            labels.append(int(parts[0]))
            paths.append(parts[1])

    dataset = test_dataset_loader(
        paths, train_path,
        eval_frames=eval_frames, num_eval=num_eval, sample_rate=sample_rate,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=nDataLoaderThread, drop_last=False,
    )

    feats = []
    was_training = model.training
    model.eval()
    tstart = time.time()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            inp = data[0][0].to(device, non_blocking=True)   # [num_eval, T]
            emb = model(inp).detach().cpu().numpy()         # [num_eval, D]
            feats.append(emb.mean(axis=0))                  # mean-pool over num_eval
            if rank == 0 and (idx % print_interval == 0):
                elapsed = max(1e-6, time.time() - tstart)
                sys.stdout.write(
                    f"\r[PLDA] extracting {idx + 1}/{len(paths)} "
                    f"({(idx + 1) / elapsed:.1f} Hz)"
                )
                sys.stdout.flush()
    if rank == 0:
        print()
    if was_training:
        model.train()

    return np.stack(feats, axis=0).astype(np.float64), np.asarray(labels, dtype=np.int64)
