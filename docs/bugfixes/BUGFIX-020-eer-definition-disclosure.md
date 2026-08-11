# BUGFIX-020 — Disclose the conservative EER definition; surface the literature form alongside

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-020-eer-definition-disclosure` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (the math is correct) but high *honesty cost* (numbers reported in research logs and any future paper would otherwise be ~0.01–0.05% off from comparable literature numbers, with no in-band signal that the definitions differ). |
| **Scope** | One utility file's docstring + return tuple; three trainer banner lines. No model, DataLoader, or config change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #20 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.2 report flagged:

> *"`tuneThreshold.py` EER returns `max(fpr[idxE], fnr[idxE]) * 100`
> which is correct for the conservative definition but differs from
> the more common `(fpr+fnr)/2`. Document this choice explicitly so
> comparison to literature is honest."*

Two definitions of EER coexist in the speaker-verification literature.
At the operating point `t*` on the ROC curve closest to the
`FPR = FNR` diagonal:

| Definition | Formula | Used by |
|---|---|---|
| **Conservative** | `EER_max = max(FPR(t*), FNR(t*)) * 100` | This codebase (since the original `voxceleb_trainer` upstream). |
| **Standard / average** | `EER_avg = (FPR(t*) + FNR(t*)) / 2 * 100` | Most speaker-verification papers, the original NIST evaluation toolkit, `sklearn.metrics`. |

The two definitions agree exactly when the empirical ROC passes
through a point with `FPR = FNR`. For continuous-score curves this is
essentially always true. For discrete (small-N or many-ties) ROC
curves, `t*` lands on a flat segment where `FPR(t*) ≠ FNR(t*)`, and
the two definitions differ by exactly:

    EER_max − EER_avg  =  |FPR(t*) − FNR(t*)| / 2

This is the entire discrepancy. It is bounded above by ~1/(2N) per
the granularity of `sklearn.metrics.roc_curve` (which returns one
threshold per distinct score). For VoxCeleb1-O (N ≈ 40k) the bound
is ~0.0025% EER — invisible at 2 decimals. For mini-VoxCeleb1 test
lists (N ≈ 1k) it can reach ~0.05% EER — visible and material when
comparing to literature.

### 1.1 Why the audit's framing is right

Before this fix, the only signal that the project used the
conservative form was a paragraph inside the `tuneThreshold.py`
module docstring. Anyone reading:

- The trainer's printed `VEER ...` line in a log file, or
- The `eer = result[1]` returned from `tuneThresholdfromScore`, or
- The historical `research_logs/*.md` numbers,

received a number without disclosure that it was the conservative
form. Re-quoting that number alongside a paper that uses the average
form silently introduces a 0.01–0.05% bias in favour of (apparently)
worse results. The audit's prescription is "document explicitly so
comparison is honest"; this fix expands "explicitly" to mean
"surfaced at every reading point, not buried in one docstring".

---

## 2. Fix

### 2.1 Strengthen the docstring with the mathematical precision

The module docstring of [`tuneThreshold.py`](../../tuneThreshold.py)
was expanded from "we use the conservative form, beware" to include:

- The closed-form discrepancy `|FPR − FNR| / 2`.
- The granularity bound `~1/(2N)` from `sklearn.metrics.roc_curve`.
- Quantitative anchors: <0.0025% on VoxCeleb1-O, up to ~0.05% on
  mini-VoxCeleb1.
- The explanation of *why* we keep the conservative form as primary
  (monotone-conservative for early-stopping; consistency with the
  historical research-log record).
- A pointer to this BUGFIX doc as the authoritative source.

### 2.2 Return both numbers from `tuneThresholdfromScore`

The return tuple grew from 5 elements to 6:

```python
# Before
return (tunedThreshold, eer, fpr, fnr, eer_threshold)

# After
return (tunedThreshold, eer, fpr, fnr, eer_threshold, eer_average)
```

where:

```python
eer         = max(fpr[idxE], fnr[idxE]) * 100      # conservative (primary)
eer_average = (fpr[idxE] + fnr[idxE]) / 2 * 100    # literature form
```

The primary `eer` (index 1) is unchanged so the existing training
loops, early-stopping logic, and historical metric trail stay
consistent. `eer_average` (index 5) is available for any caller —
test scripts, future evaluation scripts, the publication-prep
workflow — that needs the literature-comparable number.

### 2.3 Surface both in the trainer banners

Every trainer prints an evaluation line in the shape:

```python
print(f'..., VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[4]:f}')
```

This was changed in all three trainers to also include `VEER_avg`:

```python
print(f'..., VEER {result[1]:2.4f}, VEER_avg {result[5]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[4]:f}')
```

Now every evaluation log line records both numbers, so a future
re-reading of the log answers the literature-comparison question
without needing to re-run the eval.

The trainers' early-stopping path (`current_eer = float(result[1])`)
is unchanged — it continues to consume the conservative form, which
preserves run-to-run comparability of training-time decisions.

### 2.4 Caller compatibility

All 11 current call sites in the repo were audited:

| File | Uses | Affected by 6-tuple? |
|---|---|---|
| `trainSpeakerNet.py` (×2) | `result[1]`, `result[4]` | No |
| `trainSpeakerNet_performance_updated.py` (×2) | `result[1]`, `result[4]` | No |
| `trainSpeakerNet_distillation.py` (×2) | `result[1]`, `result[4]` | No |
| `test_validation_phase.py` | `result[1]`, `result[2]` | No (and `result[2]` is `fpr` — that's an unrelated stale issue) |
| `quick_test_validation.py` | `result[1]`, `result[2]` | No (same stale issue) |
| `loss/triplet.py` | `errors[1]` | No |
| `tuneThreshold.py` itself | definition | N/A |

Every caller indexes positionally rather than destructuring, so
adding a 6th element is safe — no caller breaks. Two test scripts
(`test_validation_phase.py`, `quick_test_validation.py`) still read
`result[2]` as if it were a scalar threshold, which has been wrong
since BUGFIX-002 added the dedicated `eer_threshold` at index 4 —
that is **out of scope** for this fix and a candidate for a future
BUGFIX-021 cleanup if those scripts are still in use.

---

## 3. Verification

### 3.1 Static

`python -m py_compile` passes on all four modified files:
`tuneThreshold.py`, `trainSpeakerNet.py`,
`trainSpeakerNet_performance_updated.py`,
`trainSpeakerNet_distillation.py`.

### 3.2 Mathematical equivalence

The closed-form discrepancy `EER_max − EER_avg = |FPR − FNR| / 2` is
proven algebraically:

```
max(a, b) − (a + b) / 2  =  (|a − b| + (a + b)) / 2 − (a + b) / 2
                         =  |a − b| / 2.
```

The docstring's quantitative bounds (`~1/(2N)`) follow from the
granularity of `sklearn.metrics.roc_curve`, which produces at most
one threshold per distinct score in the input. With `len(scores) = N`
trial pairs, both `FPR` and `FNR` move in steps of at most `1/N`, so
`|FPR(t*) − FNR(t*)| ≤ 1/N`, hence `EER_max − EER_avg ≤ 1/(2N)`.

### 3.3 Return shape

The 6-tuple shape is verifiable by inspection of the return statement
at the end of [`tuneThreshold.py:103`](../../tuneThreshold.py#L103);
no end-to-end PyTorch run is needed to confirm the signature change.

### 3.4 Not verified

- No actual training run was exercised — the audit-session Python
  lacks `numpy` / `sklearn`. The arithmetic claims in §3.2 are
  closed-form, and the indexing patterns in every caller were
  audited (§2.4).
- The two stale `result[2]`-as-threshold uses in
  `test_validation_phase.py` and `quick_test_validation.py` were
  flagged but not fixed — those are pre-existing bugs from before
  BUGFIX-002, unrelated to this audit item.
- The hypothesis that `EER_max − EER_avg ≈ 0.01–0.05%` on the
  project's mini-VoxCeleb1 evaluations is grounded in the
  granularity bound, not measured directly. A measured number would
  require running a real eval, which is left to the next live
  evaluation cycle (the new `VEER_avg` field in the log will record
  it automatically going forward).

---

## 4. Backward-compatibility & migration

- **`result[1]`, `result[4]`, and `result[0]`/`result[2]`/`result[3]`
  are unchanged.** Every existing reader of those indices is
  unaffected.
- **The numerical value of `result[1]` is identical** — same
  conservative-EER formula, same input curve, same operating point.
  Historical research-log numbers stay comparable.
- **The trainer's printed `VEER ...` line now also contains
  `VEER_avg ...`.** Any external log-parsing regex that locked onto
  the *exact* old format (`VEER NN.NN, MinDCF`) will need a one-line
  update; a regex that locks onto `VEER (\d+\.\d+)` (the documented
  form) still works.
- **Early stopping is unchanged**; the trainer still tracks the
  conservative `result[1]`.
- **For paper-prep work**, the recommended practice going forward is
  to quote *both* numbers, e.g. "EER 12.34% (conservative max
  definition; 12.32% under the standard `(FPR+FNR)/2` definition)",
  citing this BUGFIX-020 document.

---

## 5. Out-of-scope

- **Switching the primary `eer` to the average form.** Out of scope:
  changing the metric mid-project would break comparison to
  ~6 months of research-log entries and `exps/*/result/` files, for
  ~0.01–0.05% honesty improvement that is now achievable by quoting
  both numbers.
- **Adding `eer_average` to the `loss/triplet.py` return path.**
  That callsite computes a training-time *signal*, not a
  publication metric. Keeping the conservative form consistent for
  the training-time consumer is correct.
- **Fixing the stale `result[2]`-as-threshold reads in
  `test_validation_phase.py` and `quick_test_validation.py`.** Those
  are pre-existing leftovers from before BUGFIX-002. A separate
  follow-up bugfix should sweep them; conflating with this audit
  item would mix concerns.
- **Re-running historical evals to back-fill `eer_average`.** Too
  expensive; the new banner ensures every future eval records both,
  which is sufficient for the going-forward honesty claim.

---

## 6. Rollback plan

If the 6-tuple change turns out to break something subtle (e.g., a
caller in a private branch that does `a, b, c, d, e = result`):

1. Revert the return statement in `tuneThreshold.py` to the 5-tuple
   form: `return (tunedThreshold, eer, fpr, fnr, eer_threshold)`.
2. Drop the `eer_average = ...` line.
3. Revert the three trainer banners to omit `VEER_avg ...`.
4. Keep the strengthened docstring — that is pure documentation and
   harms nothing.

The intermediate state "keep `eer_average` available but drop the
banner change" is a valid partial rollback if the banner change
breaks an external parser but the 6th tuple element is fine.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#20**, the final entry of the §4.2 list.

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) |
| 15 | `RawNet3.py` debug print + in-place mutation | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |
| 20 | EER definition differs from common `(fpr+fnr)/2` | ✅ **This document** |
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |

§4.1 (Critical) is fully closed (BUGFIX-001..010).
§4.2 (Important) is closed except for item #12 (`lists/` empty — a
larger Sri Lankan dataprep workstream rather than a bug-class fix).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #20.
- **`sklearn.metrics.roc_curve` semantics:**
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
  — returns one threshold per distinct score, which bounds the
  granularity of the ROC and therefore the worst-case
  `EER_max − EER_avg` gap.
- **NIST EER definition (the average form):**
  *NIST SRE 2008 Evaluation Plan*, §3.2; used as the de facto
  literature standard.
- **Conservative EER definition (the max form):** see the original
  `voxceleb_trainer` upstream
  (https://github.com/clovaai/voxceleb_trainer), `tuneThreshold.py`
  history — used since the codebase's first commit.
- **Related fixes:**
  [BUGFIX-002](BUGFIX-002-eer-threshold-not-returned.md) added the
  `eer_threshold` (5th return element) which made it natural to add
  `eer_average` as the 6th in this fix — both are "things the
  pre-fix tuple did not return that callers genuinely needed".
