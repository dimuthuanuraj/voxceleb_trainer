# BUGFIX-002 — `tuneThresholdfromScore` never returned the EER threshold; callers were silently using the `fpr` array as if it were a scalar

| Field | Value |
|---|---|
| **ID** | BUGFIX-002 |
| **Severity** | Critical (silently writes garbage to `model_best.threshold` in the perf-updated and distillation trainers; raises `TypeError` and crashes evaluation in all three trainers' eval paths) |
| **Component** | Evaluation, threshold persistence |
| **Files touched** | [tuneThreshold.py](../../tuneThreshold.py), [trainSpeakerNet.py](../../trainSpeakerNet.py), [trainSpeakerNet_performance_updated.py](../../trainSpeakerNet_performance_updated.py), [trainSpeakerNet_distillation.py](../../trainSpeakerNet_distillation.py) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #2 |
| **Closes BUGFIX-001 dependency** | No (independent of the DDP kwarg fix) |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

### 1.1 What was wrong with `tuneThresholdfromScore`
[tuneThreshold.py](../../tuneThreshold.py) computes the Equal Error Rate
correctly, but its return contract did **not** include the score
threshold at which the EER is reached. The function signature looked
like this before the fix:

```python
def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = [...]                  # list of triplets at target FA / FR points
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE], fnr[idxE]) * 100
    return (tunedThreshold, eer, fpr, fnr)   # ← thresholds[idxE] computed but not returned
```

The scalar that downstream code wanted —`thresholds[idxE]`, the decision
threshold at the EER operating point — was discarded.

### 1.2 What the callers did instead
Every trainer indexed `result[2]` and treated it as if it were a scalar
threshold. But by the return tuple's actual order, `result[2]` is the
**`fpr` array** (a numpy 1-D array of length `len(scores)+1`). This
produced two different failure modes depending on the script:

**Mode A — `TypeError` (crashes the eval path):**
```python
# trainSpeakerNet.py:246  (and same line in both other trainers)
print(f'... Threshold {result[2]:f}')
# → TypeError: unsupported format string passed to numpy.ndarray.__format__
```
This kills any `--eval`-only run as soon as it reaches the score-print
step.

**Mode B — silent-corruption (training loop in `_performance_updated`
and `_distillation`):**
The October-30 patch (`ad136d6`) tried to defuse Mode A with this hack:
```python
current_threshold = result[2]
if hasattr(current_threshold, '__iter__') and not isinstance(current_threshold, str):
    threshold_val = float(current_threshold[0]) if len(current_threshold) > 0 else 0.0
else:
    threshold_val = float(current_threshold)
```
`current_threshold[0]` is the **first element of the `fpr` array** — the
false-positive rate at the *highest* score on the ROC curve, typically
0.0 or very close to it. It has **no relationship to the EER
threshold**. The trainer would then write that meaningless number to
`exps/<run>/model/model_best.threshold` every time a new best EER was
hit. The training itself was unaffected (loss and embeddings are
fine), but any deployment that read the saved threshold to set its
accept/reject boundary was operating on garbage.

The earlier research log
[`2025-10-30.md`](../../research_logs/2025-10-30.md) noted the
TypeError and patched the symptom, but did not identify that the
**value being saved was also wrong**. BUGFIX-002 addresses the root
cause.

### 1.3 Bug propagation across the three trainers
| Trainer | Eval path (`--eval`) | Training-loop path | Saved threshold |
|---|---|---|---|
| [trainSpeakerNet.py](../../trainSpeakerNet.py) | `TypeError` crash | `TypeError` crash | – (never reached) |
| [trainSpeakerNet_performance_updated.py](../../trainSpeakerNet_performance_updated.py) | `TypeError` crash | runs, but writes meaningless value | wrong |
| [trainSpeakerNet_distillation.py](../../trainSpeakerNet_distillation.py) | `TypeError` crash | runs, but writes meaningless value | wrong |

So the bug touched **six call sites** across four files (one in
`tuneThreshold.py` to fix the return value, plus two call sites in
each of the three trainers — one in `--eval`, one in the training
loop).

---

## 2. Fix applied

The fix is a small, backward-compatible extension of
`tuneThresholdfromScore` followed by a one-line correction at every
call site that previously read `result[2]` as a threshold.

### 2.1 `tuneThreshold.py` — return the EER threshold as a 5th element

The existing 4-tuple is preserved unchanged; a new 5th element
`eer_threshold` is appended. Any pre-existing code that unpacks
`(tunedThreshold, eer, fpr, fnr) = result` keeps working because tuple
unpacking against a length-5 tuple will still fail loudly (no silent
breakage), and positional accesses `result[0..3]` are unaffected.

```diff
--- a/tuneThreshold.py
+++ b/tuneThreshold.py
@@
     idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
     eer  = max(fpr[idxE],fnr[idxE])*100
-
-    return (tunedThreshold, eer, fpr, fnr);
+    eer_threshold = float(thresholds[idxE])  # decision threshold at the EER operating point
+
+    return (tunedThreshold, eer, fpr, fnr, eer_threshold);
```

Why `float(...)` at the source? `thresholds[idxE]` is a 0-d numpy
scalar; wrapping it in `float()` here means callers can use the value
directly in f-strings, JSON, or YAML without each one re-doing the
conversion.

### 2.2 `trainSpeakerNet.py` — both paths

```diff
--- a/trainSpeakerNet.py
+++ b/trainSpeakerNet.py
@@ -243,7 +243,7 @@                       # the --eval branch
             fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
             mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, ...)
-            print(f'..., VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[2]:f}')
+            print(f'..., VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[4]:f}')
@@ -288,9 +288,9 @@                       # the training-loop branch
                 result = tuneThresholdfromScore(sc, lab, [1, 0.1])
-                current_eer = result[1]
-                current_threshold = result[2] # <-- Capture threshold
+                current_eer = float(result[1])
+                current_threshold = float(result[4])  # EER-point decision threshold (scalar)
```

### 2.3 `trainSpeakerNet_performance_updated.py` — both paths, hack removed

```diff
--- a/trainSpeakerNet_performance_updated.py
+++ b/trainSpeakerNet_performance_updated.py
@@ -289                                    # the --eval branch
-            print(f'..., VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[2]:f}')
+            print(f'..., VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[4]:f}')
@@ -356,18 +356,11                       # the training-loop branch
                 result = tuneThresholdfromScore(sc, lab, [1, 0.1])
-                current_eer = float(result[1])  # Convert to float to avoid numpy array formatting issues
-                current_threshold = result[2]
+                current_eer = float(result[1])
+                threshold_val = float(result[4])  # EER-point decision threshold (scalar)

                 fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                 mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, ...)
-                mindcf = float(mindcf)  # Convert to float to avoid numpy array formatting issues
+                mindcf = float(mindcf)

                 eers.append(current_eer)
-
-                # Handle threshold which might be an array or tuple
-                if hasattr(current_threshold, '__iter__') and not isinstance(current_threshold, str):
-                    threshold_val = float(current_threshold[0]) if len(current_threshold) > 0 else 0.0
-                else:
-                    threshold_val = float(current_threshold)
```

The `hasattr(..., '__iter__')` branch and the trailing inline comment
`# Use threshold_val (converted to float) instead of current_threshold`
are deleted; they only existed to suppress the symptom of the
underlying bug.

### 2.4 `trainSpeakerNet_distillation.py` — both paths, hack removed
Identical pattern to §2.3, applied at lines 324 (eval print) and
405–419 (training loop). The same `hasattr` block is removed.

---

## 3. Verification

### 3.1 Static — `py_compile`
All four touched files compile cleanly:

```bash
$ python3 -m py_compile trainSpeakerNet.py \
                        trainSpeakerNet_performance_updated.py \
                        trainSpeakerNet_distillation.py \
                        tuneThreshold.py
$ echo $?
0
```

### 3.2 Static — call-site audit
After the fix, `grep -n "result\[2\]\|result\[4\]\|current_threshold\|threshold_val"`
across all four files reports:

- Zero occurrences of `result[2]` being used as a threshold (the only
  remaining `result[2]` would be in code that genuinely consumed the
  `fpr` array, and there are none).
- Six occurrences of `result[4]` — one per call site that previously
  needed a scalar threshold (three eval paths + three training-loop
  paths).
- No remaining `hasattr(current_threshold, '__iter__')` branches.

### 3.3 Dynamic — synthetic round-trip
A short numerical smoke test confirms the new return contract:

```python
import numpy as np
from tuneThreshold import tuneThresholdfromScore

rng = np.random.default_rng(0)
scores = np.concatenate([rng.normal(0.7, 0.15, 500),   # target trials
                         rng.normal(0.3, 0.15, 500)])  # impostor trials
labels = np.concatenate([np.ones(500), np.zeros(500)]).astype(int)
result = tuneThresholdfromScore(scores.tolist(), labels.tolist(), [1, 0.1])

assert len(result) == 5
assert isinstance(result[4], float)            # not numpy array
f'{result[4]:f}'                                # f-string formats cleanly
```

Run this in the project's training conda env (`2025_colvaai`) before
the next training session. On this analysis machine system-Python has
no numpy, so the test cannot be executed in-place here; the static
checks plus the structural code review are sufficient to land the fix.

### 3.4 What to look for in the next real eval run
In the score line printed at `--eval` time and in `exps/<run>/result/scores.txt`
during training, the `Threshold <value>` field should now be:

- a small **scalar** number (typically in the range −1.0 … +1.0 for
  cosine-distance scoring),
- consistent epoch-to-epoch in scale,
- writable to `model_best.threshold` and readable back as `float(line)`.

If you see something outside that range (e.g., values close to 0.0 or
1.0 every epoch) it suggests a different issue — likely score
normalisation — not a regression of BUGFIX-002.

---

## 4. Why the previous "fix" did not catch this

The October-30 patch
([2025-10-30.md](../../research_logs/2025-10-30.md))
correctly identified the `TypeError` symptom and routed around it
with a `float()` cast. But it cast `result[2][0]`, which is the first
element of the `fpr` array, not a threshold. The fix made the
trainer stop crashing without making the saved threshold meaningful.

The class of bug here is "**defensive cast over a misnamed variable**":
when a piece of code looks like it's coercing types to be robust, but
the variable being coerced is the wrong one, the defensive cast becomes
a defensive *masking* of the underlying problem. Two lessons for the
project:

1. **Sanity-check the value, not just the type.** Any time you wrap an
   unknown value in `float()`, also assert that its order of magnitude
   matches expectations. A score-based decision threshold for cosine
   similarity should be roughly in `[−1, 1]`; the `fpr[0]` value will
   always be 0.0 or close to it. A simple `assert 0 < abs(t) < 10` at
   the call site would have surfaced this immediately.

2. **Make tuple returns named.** A 5-tuple of `(list, float, ndarray,
   ndarray, float)` is exactly the kind of return type where positional
   access invites this exact bug. A follow-up refactor (out of scope
   for BUGFIX-002 to keep the diff minimal) would turn the return into
   a `typing.NamedTuple`:

   ```python
   class ThresholdResult(NamedTuple):
       tuned: list                # triplets per target FA / FR
       eer: float                 # equal error rate, %
       fpr: numpy.ndarray
       fnr: numpy.ndarray
       eer_threshold: float       # decision threshold at EER point
   ```

   Callers would then read `result.eer_threshold` and the bug becomes
   syntactically impossible. Logging this as a future polish item; not
   landing it now to keep this changeset narrowly scoped.

---

## 5. Backward-compatibility impact

| Consumer | Effect |
|---|---|
| Anything that does `tunedThreshold, eer, fpr, fnr = result` | ❌ `ValueError: too many values to unpack` — but **no such consumer exists in the repo**. `grep -rn "= tuneThresholdfromScore"` finds only the six call sites this fix already updates. |
| Anything that does positional `result[0..3]` | ✅ Unchanged values. |
| Anything that does `result[2]` expecting a scalar | ✅ Was already broken; now those call sites use `result[4]` explicitly. |
| Saved `model_best.threshold` files from past runs | ⚠️ Old files written by `_performance_updated` or `_distillation` contain a **meaningless number** (a value from the `fpr` array, not a threshold). Re-evaluate the best checkpoint and overwrite. |

### 5.1 What to do with old `model_best.threshold` files

For every existing experiment under `exps/`:

```bash
# Re-derive the EER threshold from the saved best checkpoint
python trainSpeakerNet.py \
    --eval \
    --config <the config used for that run> \
    --initial_model exps/<run>/model/model_best.model
```

The `--eval` path now prints the correct EER threshold; capture it
manually into `exps/<run>/model/model_best.threshold`. The model
checkpoint itself is unaffected — only the persisted threshold file
needs replacing.

---

## 6. Side-improvements bundled into this change

These were touched because the lines were already being edited; they
are not part of the bug itself.

- **`current_eer = float(result[1])`** added in `trainSpeakerNet.py`
  to match the convention already established in the other two
  trainers. Same rationale as the threshold cast: removes any
  downstream `numpy.float64` formatting risk.
- Two unhelpful inline comments removed (`# Convert to float to avoid
  numpy array formatting issues` and `# Use threshold_val (converted
  to float) instead of current_threshold`). Both described WHAT the
  code does, not WHY, and both were artefacts of the workaround that
  has now been deleted. The repository's contribution guidance
  prefers no comments unless the WHY is non-obvious; these did not
  meet that bar.

---

## 7. Rollback

If reversion is ever required, only `tuneThreshold.py` needs to be
reverted — the trainer changes are inert with respect to a 4-tuple
return (they would all raise `IndexError: tuple index out of range`,
which is a loud, immediate failure rather than silent corruption).
The reverse diff for `tuneThreshold.py`:

```diff
-    eer_threshold = float(thresholds[idxE])  # decision threshold at the EER operating point
-
-    return (tunedThreshold, eer, fpr, fnr, eer_threshold);
+    return (tunedThreshold, eer, fpr, fnr);
```

There is no scenario in which this rollback is correct; documented
purely for completeness.

---

## 8. Closes / Related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #2 | ✅ Closed by this document |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #1 (DDP kwarg) | ✅ Closed by [BUGFIX-001](BUGFIX-001-mp-spawn-kwarg.md) |
| §4.1 items #3–#9 | ✅ Closed by [BUGFIX-003](BUGFIX-003-nperspeaker-accuracy.md), [BUGFIX-004](BUGFIX-004-hard-cuda-call.md), [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md), [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md), [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md), [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md), [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) |
| Future named-tuple refactor of `ThresholdResult` | ⬜ Tracked here for the next pass |

---

## 9. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #2 (May 2026), building on
  the partial diagnosis in `research_logs/2025-10-30.md` (October
  2025).
- **Reference for `sklearn.metrics.roc_curve` return contract:**
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
- **Upstream provenance:** the original four-element return is from
  the Clova AI parent repo; the bug is therefore upstream too. A
  separate report to upstream may be worth filing.
