# FEATURE-003 — Per-language evaluation (threshold calibration)

**Status:** in progress (this PR)
**Type:** feature
**Relates to:** §3.1 #4 of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-002 (composes cleanly)
**Owner:** SL-SPV
**Touches:** `trainSpeakerNet.py`

## 1. Motivation

§3.1 #4 of the analysis doc asks for per-language threshold calibration: fit a separate decision threshold for Sinhala-only, Tamil-only, and code-switched trial sets so the deployment EER reflects each language's operating point rather than a global compromise. The trainer currently exposes only a single `--test_list`, so per-language reporting requires three separate runs and manual aggregation.

This feature adds first-class per-language evaluation: one `--per_lang_test_lists` flag, the trainer iterates each list, reports per-language EER/MinDCF/threshold (plus AS-Norm diagnostics when enabled), and prints a pooled combined number for the overall picture. Default behaviour is unchanged — the new flag is opt-in.

## 2. What this is and is not

**Is:**
- A way to evaluate against N test lists in one trainer invocation.
- Per-language EER, threshold, and MinDCF in a single report.
- Composes with `--as_norm` (each list reuses the same cohort, gets its own normalised scores).
- Pooled metrics across all lists for the "overall" line.

**Is not:**
- A Platt / logistic-regression calibrator. That's a separate concern requiring a held-out dev set + fit + apply pipeline; documented as a follow-up below.
- A per-language cohort path. AS-Norm uses one cohort for all lists (the SL cohort). Per-language cohorts can be added later by repeating the flag with `--as_norm_cohort_list` overrides.
- Confidence calibration (raw scores → probabilities). Out of scope.

## 3. Design

### 3.1 CLI / config surface

```
--per_lang_test_lists "si:/data/test_list_si.txt,ta:/data/test_list_ta.txt,cs:/data/test_list_cs.txt"
```

Comma-separated `lang:path` pairs. Lang labels are free strings — used only for log lines. When empty (default), the trainer keeps the existing single-list flow.

YAML form:
```yaml
per_lang_test_lists:
  si: /data/test_list_si.txt
  ta: /data/test_list_ta.txt
  cs: /data/test_list_cs.txt
```

(The trainer already routes dict-valued YAML scalars through `args.__dict__[k] = v` without type coercion — see the `isinstance(v, (dict, list))` branch added for BUGFIX-016.)

### 3.2 Eval flow

Inside the `if args.eval == True:` block of `trainSpeakerNet.py`:

1. If `per_lang_test_lists` is empty / unset → unchanged path.
2. Otherwise iterate `(lang, path)`:
   1. Temporarily override `args.test_list = path`.
   2. Call `trainer.evaluateFromList(**vars(args))`.
   3. Compute EER, MinDCF, threshold, raw vs AS-Norm pair (if `--as_norm`).
   4. Print a banner like
      `[per-lang si] VEER X.XXXX, MinDCF Y.YYYYY, Threshold T   (AS-Norm: ...)`
   5. Accumulate `(scores, labels)` for pooling.
3. After the loop, compute pooled metrics on the concatenated `(scores, labels)` and print
   `[per-lang POOLED] VEER X.XXXX, MinDCF Y.YYYYY, Threshold T`.
4. Restore `args.test_list`.

The per-epoch evaluation block (during training, not just `--eval`) gets the same treatment so the per-language picture is visible across the training curve.

### 3.3 Composition with AS-Norm (FEATURE-002)

The AS-Norm cohort is extracted once per `evaluateFromList` call. With per-language eval, each call to `evaluateFromList` re-uses the same cohort because the cached cohort file (`asnorm_cohort.pt` under `save_path`) is hit on the second and third pass. Cost: one cohort extraction total, three sets of per-file stats (cheap, GPU-batched).

If you want different cohorts per language, run the trainer three separate times today; a future iteration can add per-language cohort overrides.

### 3.4 Edge cases

- One of the per-lang lists is empty → fail loudly with a `ValueError`; do not silently produce zero-pair metrics.
- One list path doesn't exist → fail loudly at startup (before any model load).
- `--test_list` is also set → `--per_lang_test_lists` takes precedence; a warning is printed.
- Distributed eval → unchanged. Each list's `evaluateFromList` already handles distributed gathering; per-language reporting runs on rank 0 only.

## 4. Risk and rollback

- **Default off** — empty `--per_lang_test_lists` keeps the existing single-list flow intact.
- **No new dependencies.** Pure orchestration over the existing `evaluateFromList`.
- **Rollback** — single revert; no schema migration, no on-disk artefact change.

## 5. Validation plan (not blocking this PR)

1. Run on the current English VoxCeleb1-O list passed three times under different aliases (`vox1-1`, `vox1-2`, `vox1-3`) → all three per-language EERs must match exactly, and the pooled number must also match the single-list run.
2. Once the SL pilot exists, run with `si:test_list_si.txt, ta:test_list_ta.txt, cs:test_list_cs.txt` → expect:
   - Three distinct per-language EERs.
   - cs EER ≥ max(si, ta) EER (code-switched should be hardest).
   - With `--as_norm`, each language's MinDCF drops; the per-language deltas reported on the AS-Norm diagnostic lines.

## 6. Follow-ups (intentionally not in this PR)

- **Platt logistic calibrator.** Separate `fit_platt_calibrator.py` that reads (score, label, lang) triples from a dev run, fits a sigmoid `P(target) = σ(a·score + b·lang_id + c)`, and dumps weights as JSON. An eval-time `--platt_calibrator path.json` flag applies it before thresholding.
- **Per-language AS-Norm cohorts.** `--as_norm_cohort_list_per_lang "si:cohort_si.txt,ta:cohort_ta.txt"` overriding the single cohort.
- **Per-language threshold persistence.** Save the EER-point threshold per language as `<save_path>/thresholds.json` so a deployment script can load it directly.

## 7. Cross-references

- `docs/bugfixes/FEATURE-002-as-norm-score-normalisation.md` — composes naturally; one cohort, three per-list stats.
- `docs/bugfixes/BUGFIX-018-streaming-evaluation.md` — eval-streaming applies independently to each per-lang list.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 #4, §9 action 6 — the original ask.
