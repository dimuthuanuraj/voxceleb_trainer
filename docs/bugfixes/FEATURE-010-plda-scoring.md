# FEATURE-010 — Probabilistic Linear Discriminant Analysis (PLDA) scoring

**Status:** in progress (this PR)
**Type:** feature (eval-time scoring backend)
**Relates to:** §3.3 #13 of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-002 (AS-Norm — composes on top of PLDA scores); FEATURE-003 (per-language eval — applies independently to PLDA scores)
**Owner:** SL-SPV
**Touches:** `plda.py` (new), `SpeakerNet.py`, `trainSpeakerNet.py`

## 1. Motivation

§3.3 #13 of the analysis doc proposes PLDA scoring on top of speaker embeddings. PLDA is the historical NIST SRE standard backend and remains useful for two scenarios our setup is likely to hit:

- **Short-utterance trials** (< 2 s) — common in telephony deployment. Single embeddings from short audio have high variance; PLDA models this explicitly via the within-speaker covariance, while cosine treats every embedding as equally trustworthy.
- **Channel mismatch** between enrolment and test (mic vs. phone vs. codec) — PLDA's residual covariance term captures channel variability that cosine ignores.

For modern AAM-Softmax + ECAPA/SSL embeddings on long, clean utterances, PLDA over cosine usually adds **0–3 % relative EER** — modest. For short-utterance and channel-mismatched trials, **5–10 % relative EER** is realistic. Shipping it now means the backend is ready the moment the deployment-channel question becomes concrete (§3.1 #5 telephony scope is still open per §9 action 1).

## 2. What this is and is not

**Is:**
- A pure post-training, eval-time scoring backend. No effect on training or the saved model.
- Simplified (two-covariance) PLDA — speaker covariance `Σ_b` and within-speaker covariance `Σ_w`, no separate channel subspace `G`. The standard modern choice (Sizov et al.); near-identical EER to the full PLDA at half the complexity.
- Preprocessing pipeline: centre → length-normalise → LDA dim-reduce → centre-again.
- Closed-form fit (no EM). Sufficient for typical SV-train-set sizes (10k–100k embeddings).
- Pickle-able save/load with cache invalidation by training-list path.
- Composes with AS-Norm: PLDA produces a score per trial, then AS-Norm normalises that score using cohort statistics. This is the standard NIST pipeline order.

**Is not:**
- Heavy-duty PLDA with channel subspace (Kenny 2010 full PLDA). Two-cov is enough.
- A replacement for AAM-Softmax or any other discriminative loss. PLDA scores embeddings; it doesn't change how they're learned.
- A new training step. The PLDA *fit* uses the trained model to extract embeddings over the labelled PLDA-train list once, then runs closed-form linear algebra — no gradient updates.

## 3. Design

### 3.1 Module layout

A new file `plda.py` at repo root, sibling to `score_norm.py` and `tuneThreshold.py`. Exposes one class with the fit / save / load / score API:

```python
class TwoCovPLDA:
    def __init__(self, lda_dim=200): ...
    def fit(self, embeddings, speaker_labels): ...          # closed-form
    def score(self, x1, x2) -> float | np.ndarray: ...      # batched
    def save(self, path): ...                               # pickle
    @classmethod
    def load(cls, path) -> 'TwoCovPLDA': ...
```

Plus a top-level helper `extract_plda_train_embeddings(model, train_list, ...)` that runs the trained model over the PLDA-fit data and returns `(embeddings: np.ndarray[N, D], labels: np.ndarray[N])`.

### 3.2 Math

For embedding `x` (after preprocessing — centred, length-normalised, LDA-reduced, centred):

- Speaker covariance: `Σ_b = (1/S) Σ_speakers (μ_s − μ)(μ_s − μ)ᵀ`
- Within covariance: `Σ_w = (1/N) Σ_utts (x_i − μ_{s(i)})(x_i − μ_{s(i)})ᵀ`

Score for trial `(x1, x2)`:
```
score = log p(x1, x2 | same) − log p(x1, x2 | diff)
      = 0.5 · [ x1ᵀ·Q·x1 + x2ᵀ·Q·x2 ] + x1ᵀ·P·x2 + const
```
where `Q` and `P` are closed-form matrices derived from `Σ_b` and `Σ_w` (Garcia-Romero & Espy-Wilson 2011, Sizov et al. 2014). Precomputed once at `fit` time; scoring is two quadratic forms + a bilinear form per pair.

### 3.3 Preprocessing pipeline

PLDA assumes Gaussian residuals — the discriminatively-trained embeddings out of AAM-Softmax aren't strictly Gaussian, but the standard four-step preprocessing makes them close enough:

1. **Subtract global mean.**
2. **Length-normalise** (`x / ||x||₂`). Crucial; without this the variance is dominated by length.
3. **LDA** to `plda_dim` (default 200; embeddings are 192–512 dim so this is a small reduction). Picks the directions of maximum between-speaker variance — the directions PLDA models well.
4. **Subtract LDA-output mean.**

These steps are stored in the PLDA object and applied identically at score time.

### 3.4 Hook point

`SpeakerNet.evaluateFromList` — same place AS-Norm was inserted. When `plda: true`:

1. After main feature extraction (rank 0 only), check if `<save_path>/plda.pkl` exists and matches the current `plda_train_list` hash.
2. If yes → load. If no → extract embeddings over `plda_train_list`, fit PLDA, save to cache.
3. In the trial loop, replace `score = -mean(cdist(ref, com))` with `score = self.plda.score(ref_mean, com_mean)` where `ref_mean` / `com_mean` are the per-utterance means over the `num_eval` segments (matching the existing path).
4. AS-Norm (FEATURE-002) runs **after** PLDA scoring if both are enabled — it normalises the PLDA scores using cohort PLDA scores. This is the standard order.

### 3.5 CLI / config surface

```
--plda                              # bool, default False
--plda_train_list path.txt          # required when --plda is set; same format as train_list.txt
--plda_train_path path/             # root for files in plda_train_list (defaults to --train_path)
--plda_dim 200                      # LDA reduction dim
--plda_save                         # cache the fit to <save_path>/plda.pkl (default True)
--no_plda_save
```

YAML:
```yaml
plda: true
plda_train_list: data/sl_celeb/plda_train_list.txt
plda_train_path: data/sl_celeb/wav
plda_dim: 200
plda_save: true
```

### 3.6 Composition with existing features

| Feature | Composes? | Order |
|---|---|---|
| FEATURE-002 (AS-Norm) | ✅ | PLDA → AS-Norm (PLDA produces score, AS-Norm normalises against cohort PLDA scores) |
| FEATURE-003 (per-language eval) | ✅ | Each per-lang test list runs PLDA scoring independently; PLDA fit is shared across lists |
| FEATURE-004 / 005 (fine-tune / LLRD) | ✅ | Training-side; orthogonal to PLDA |
| FEATURE-006 (ECAPA backbone) | ✅ | Pure backbone swap |
| FEATURE-007 / 008 (aux / DANN) | ✅ | Training-side; PLDA sees only the final embedding |

## 4. Risk and rollback

- **Default off** — `plda: false` (default) is byte-identical to today.
- **Singular covariance matrices** on small PLDA-train sets. Mitigation: floor eigenvalues at 1e-6 before inversion. Raise an explicit error if the rank is too low (fewer than 2× `plda_dim` speakers in the train list).
- **Stale cache** — cache stores a hash of the PLDA train list path + the model checkpoint mtime. Mismatch ⇒ refit.
- **Length-normalisation conflict** — the trial loop already L2-normalises when `test_normalize=true`. PLDA needs its own length-norm AFTER its own centring. Implementation skips the trial-loop L2-norm when PLDA is on (PLDA owns preprocessing).
- **Rollback** — single revert; no on-disk format changes.

## 5. Validation plan (not blocking this PR)

1. **Smoke test in this PR**: synthetic embeddings — generate `N` speakers × `K` utterances × `D` dims from a known generative process (Gaussian per-speaker means, Gaussian residuals). Fit PLDA. Assert that:
   - `plda.score(x_same_spk_pair)` > `plda.score(x_diff_spk_pair)` on held-out test pairs.
   - Fit + save + load round-trips byte-identically.
2. On the SL pilot: cosine vs. PLDA EER with all else equal. Expect 0–3 % relative improvement on long utterances; revisit when short-utterance / telephony eval is in scope.
3. With AS-Norm: PLDA + AS-Norm vs. cosine + AS-Norm. The PLDA contribution may shrink under AS-Norm — both interventions reduce score variance.

## 6. Follow-ups (intentionally not in this PR)

- **Full PLDA with channel subspace `G`.** Useful if PLDA + AS-Norm has plateaued AND clear channel mismatch is observed.
- **PLDA score recalibration.** A logistic regression on `(plda_score, language_id)` for paper-grade reporting.
- **Heavy-tailed PLDA** (Kenny 2010) — uses Student-t residuals; ~1 % EER on heavy-tailed embeddings.

## 7. Cross-references

- `score_norm.py` (FEATURE-002) — sibling eval-time backend.
- `tuneThreshold.py` — consumes the score list PLDA produces.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.3 #13 — original ask; §3.1 #5 telephony scope is the strongest empirical motivator.

## 8. References

- Sizov, Lee, Kinnunen, *Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric Authentication*, S+SSPR 2014.
- Garcia-Romero, Espy-Wilson, *Analysis of i-vector Length Normalization in Speaker Recognition Systems*, Interspeech 2011.
- Kenny, *Bayesian Speaker Verification with Heavy-Tailed Priors*, Odyssey 2010.
