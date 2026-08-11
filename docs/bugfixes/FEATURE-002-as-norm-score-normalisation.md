# FEATURE-002 — Adaptive Symmetric Normalisation (AS-Norm)

**Status:** in progress (this PR)
**Type:** feature
**Relates to:** §9 of SL_LANGUAGE_SPV_ANALYSIS.md; MINDCF_IMPROVEMENT_GUIDE.md (open item)
**Owner:** SL-SPV
**Touches:** `score_norm.py` (new), `SpeakerNet.py`, `trainSpeakerNet.py`. *Scoped down from the original plan*: the performance-updated and distillation trainers store per-file embeddings as `[dim]` (mean-pooled in their batched eval optimisation) rather than `[num_eval, dim]`, so the cohort-scoring path does not compose cleanly there. Note added at the top of each sibling trainer; extend in a follow-up if needed.

## 1. Motivation

`MINDCF_IMPROVEMENT_GUIDE.md` flags AS-Norm as a high-leverage missing piece: 10–20 % relative MinDCF reduction at zero training cost. It is the standard cohort-based score post-processing step for SV systems (Matejka et al., Interspeech 2017) and is independent of the encoder, loss, or front-end. For a low-resource target language like Sinhala — where every percentage point matters and we cannot afford to retrain — AS-Norm is essentially free improvement.

The trainer currently has no score-normalisation hook at all; trial scores go straight to `tuneThresholdfromScore`. This feature adds one cleanly, behind config flags, without changing default behaviour.

## 2. Math (one paragraph)

For each trial `(e, t)` with raw score `s = score(e, t)`, pick a fixed cohort `C` of unrelated impostor utterances. Compute `S_e = {score(e, c) : c ∈ C}` and `S_t = {score(t, c) : c ∈ C}`, keep the top-K largest of each (the "adaptive" step — most similar impostors anchor the local statistics), and normalise:

```
s_as = 0.5 * ((s − μ(top_K(S_e))) / σ(top_K(S_e))
            + (s − μ(top_K(S_t))) / σ(top_K(S_t)))
```

Identical scoring function on both sides ⇒ symmetric. `top_K` ≈ 300 is the literature default for cohorts of size 1k–10k; for our smaller (~500-speaker) SL cohort we default to `min(300, cohort_size)`.

## 3. Design

### 3.1 Module layout

A new file `score_norm.py` at repo root, sibling to `tuneThreshold.py`. It exposes three pure functions:

- `extract_cohort_embeddings(model, cohort_list, cohort_path, ...) → (cohort_feats, paths)`
  Runs the trained encoder over a list of cohort wav paths, returning a `[cohort_size, num_eval, dim]` tensor and the matching path list. Caches to `<save_path>/asnorm_cohort.pt` (and validates by path list on reload, so a changed cohort invalidates the cache automatically).
- `compute_file_cohort_stats(file_iter, cohort_feats, top_k, normalize) → dict[path → (μ, σ)]`
  For every trial-file embedding, computes top-K cohort scores using the **same scoring metric the trial loop uses** (`-mean(cdist(ref, x))` averaged over `num_eval`), then top-K, then `(mean, std)`. Pure GPU; chunked for memory.
- `apply_as_norm(scores, trials, file_stats) → list[float]`
  The closed-form formula above. ε = 1e-8 floor on σ to avoid divide-by-zero on degenerate cohorts.

This module has **no Trainer/model knowledge** beyond the forward pass — it accepts a model and a path list. It is independently testable.

### 3.2 Hook point

Inside `SpeakerNet.evaluateFromList` (canonical trainer only — see scoping note in the header), right before the final `return (all_scores, all_labels, all_trials)`:

1. If `as_norm` flag is false → return unchanged (zero overhead).
2. Else (rank 0 only):
   1. Read cohort list, extract cohort embeddings (cached after first run).
   2. Walk every unique file in `setfiles` (or `stream_keys` under `eval_streaming`), compute (μ, σ) per file.
   3. Stash raw scores on `self._last_raw_scores` (so the trainer can still report raw EER for diagnostics).
   4. Replace `all_scores` with AS-Norm scores.

The 4-line trainer `sc, lab, _ = trainer.evaluateFromList(...)` call site does **not** change shape. When `as_norm` is on, `sc` is already normalised; raw scores are accessible via `trainer._last_raw_scores` for the side-by-side `[AS-Norm] Raw VEER X / AS-Norm VEER Y` print.

This minimises blast radius: every existing call site keeps working untouched.

### 3.3 Compatibility with streaming eval (BUGFIX-018)

Cohort extraction always builds the cohort tensor in memory (≤500 speakers × ~10×512 floats ≈ 10 MB — trivial). Trial-file feature retrieval mirrors the existing branch: `feats[fname]` when in-memory, `_load_feat(fname)` when streaming. So AS-Norm composes cleanly with `--eval_streaming`.

### 3.4 Distributed eval

Cohort extraction and AS-Norm computation run on rank 0 only, after the existing `all_gather` step has consolidated `feats` on rank 0. Other ranks see no change.

### 3.5 Config / CLI surface

```
--as_norm                       # bool flag, default False
--as_norm_cohort_list PATH      # required when --as_norm is set
--as_norm_cohort_path PATH      # optional; falls back to --test_path
--as_norm_top_k INT             # default 300
--as_norm_save_cohort           # bool, default True → cache cohort embeddings
--no_as_norm_save_cohort        # explicit disable
```

YAML keys match exactly (snake_case via the existing `_CAMEL_YAML_ALIASES` machinery from BUGFIX-022).

The cohort list format is one wav path per line; lines may also be `<spk-id>\t<path>` — only the last whitespace-separated field is used (so the existing VoxCeleb-style `<label> <enrol> <test>` test_list format would also work if you reuse a file). One utterance per cohort entry; the user is responsible for picking ~one-per-speaker for a clean cohort.

### 3.6 What is intentionally NOT included

- **Multiple cohorts** (per-language, per-domain). Single cohort suffices for the immediate goal (move MinDCF down on SL eval). Add later if a per-domain pattern emerges.
- **Z-norm / T-norm separately.** AS-Norm subsumes them; offering all three would clutter the CLI for no gain on this corpus.
- **Calibrated probabilities.** A separate concern; AS-Norm only normalises scores for thresholding, not for downstream Bayesian decisions.
- **PLDA backend.** Out of scope — AS-Norm is a 30-line post-processing step, PLDA is a separate scoring backend.

## 4. Risk and rollback

- **Default off** — behaviour unchanged unless `as_norm: true` is set in a config or `--as_norm` on the CLI. Existing experiment scripts continue producing identical numbers.
- **Cache invalidation** — cohort cache validates by exact path list. If the cohort list file changes (a path is added/removed/reordered), the cache is rebuilt automatically. No silent stale-cache failure mode.
- **Numerical safety** — σ has a 1e-8 floor; an all-identical-score cohort (impossible in practice) cannot NaN the output.
- **Rollback** — single git revert removes the feature; no shared state is migrated, no on-disk format is committed-to (the `asnorm_cohort.pt` cache file is regenerable and lives under `save_path`, which is already covered by the exps/ retention policy from BUGFIX-026).

## 5. Validation plan (post-merge, not blocking this PR)

1. Run an existing checkpoint on VoxCeleb1-O **without** `--as_norm` → record baseline EER/MinDCF.
2. Run the same checkpoint **with** `--as_norm` and a 500-speaker VoxCeleb1-dev cohort → expect MinDCF down ≥ 5 % relative (literature says 10–20 %, but our cohort is on the small side).
3. Vary `--as_norm_top_k ∈ {100, 200, 300, 400}` → confirm a broad plateau (not a knife-edge optimum).
4. Once the SL corpus exists, swap to an in-language cohort and re-measure.

Numbers go into `research_logs/` under the experiment folder, not into this doc.

## 6. Cross-references

- `docs/bugfixes/BUGFIX-018-streaming-evaluation.md` — composes with `--eval_streaming`.
- `docs/bugfixes/BUGFIX-022-camel-snake-case-aliases.md` — YAML key normalisation reused for new flags.
- `MINDCF_IMPROVEMENT_GUIDE.md` — original flag of the gap.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §9 — research roadmap; AS-Norm now ships, so this is the first §9 item closed.
