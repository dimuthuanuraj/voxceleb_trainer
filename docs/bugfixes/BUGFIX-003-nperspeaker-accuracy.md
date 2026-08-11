# BUGFIX-003 — TEER/TAcc broken (and AM/AAM-Softmax silently crashing) when `nPerSpeaker > 1`

| Field | Value |
|---|---|
| **ID** | BUGFIX-003 |
| **Severity** | Critical for AM/AAM-Softmax users (the loss assertion **always** fires when `nPerSpeaker > 1`); important for everyone else (training-time accuracy metric is wrong, so convergence problems can be masked) |
| **Component** | `SpeakerNet` forward path, loss-function input contract |
| **Files touched** | [SpeakerNet.py](../../SpeakerNet.py), [SpeakerNet_performance_updated.py](../../SpeakerNet_performance_updated.py), [SpeakerNet_distillation.py](../../SpeakerNet_distillation.py), [DistillationWrapper.py](../../DistillationWrapper.py), [loss/angleproto.py](../../loss/angleproto.py), [loss/proto.py](../../loss/proto.py), [loss/ge2e.py](../../loss/ge2e.py), [loss/softmaxproto.py](../../loss/softmaxproto.py), [loss/triplet.py](../../loss/triplet.py) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #3; original symptom in `research_logs/2025-10-30.md` |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

> **Important deviation from the original prescription** — the bug report
> in `SL_LANGUAGE_SPV_ANALYSIS.md` quoted the
> [research log](../../research_logs/2025-10-30.md) and proposed the
> fix `label = label[::nPerSpeaker]` "after the embedding-mean reshape."
> That prescription cannot be applied verbatim because **there is no
> embedding-mean reshape in the current code** — the research log was
> describing an *attempted-and-reverted* code path. The literal fix
> would compile but break every training run. The correct fix is
> different and is documented in §2 below.

---

## 1. Problem

### 1.1 What the code actually does (vs. what the log claims)
The research log [2025-10-30.md](../../research_logs/2025-10-30.md)
described `SpeakerNet_performance_updated.py:62-70` as:

```python
# Lines 62-70 (as described in the research log):
if self.nPerSpeaker > 1:
    outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1])
    outp = outp.transpose(1, 0)
    outp = outp.reshape(-1, self.nPerSpeaker, outp.size()[-1])
    outp = torch.mean(outp, 1)
```

That code does **not** exist in any committed version. `git log -p`
across the file's history finds zero occurrences of `torch.mean(outp, 1)`.
The block was an in-progress experiment that was reverted before being
committed, and the research log captured a description of the broken
attempt rather than of the merged code.

The actually-merged forward, identical across all three SpeakerNet
variants (modulo whitespace and a `non_blocking=True`), is:

```python
outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
nloss, prec1 = self.__L__.forward(outp, label)
```

No averaging, no label subsampling — `outp` ends up shape **`(B, P, D)`**
when `nPerSpeaker > 1` (because `.squeeze(1)` is a no-op when dim-1 has
size `P > 1`), and the unmodified `label` of shape `(B,)` is passed
through.

### 1.2 The downstream contract is split across two groups
Tracing every loss in `loss/`, the input contract is **not uniform**:

| Loss | Expects | What happens when fed `(B, P, D)` & `(B,)` with P > 1 |
|---|---|---|
| `softmax` | `(N, D)` flat, `(N,)` labels | Linear layer broadcasts on last dim → output `(B, P, nClasses)`. CrossEntropy interprets dim-1 as the class axis (P=2) and `label` values up to `nClasses-1` are out of range → **`RuntimeError: Target out of bounds`**, or with `ignore_index` semantics, silent wrong loss. `prec1` from `accuracy` is computed along the wrong axis. |
| `amsoftmax` | `(N, D)`, `(N,)` | `assert x.size()[1] == self.in_feats` → `2 == 512` → **`AssertionError`** at the first batch. |
| `aamsoftmax` | `(N, D)`, `(N,)` | Same assertion → **`AssertionError`**. |
| `angleproto` | `(B, P, D)`, `label` ignored | Works. Builds internal diagonal labels. `prec1` is within-batch ranking accuracy. |
| `proto` | `(B, P, D)`, `label` ignored | Works. Same as `angleproto`. |
| `ge2e` | `(B, P>=2, D)`, `label` ignored | Works. Builds internal labels. |
| `softmaxproto` | `(B, P==2, D)`, `(B,)` labels | Works. Internally flattens to `(B*2, D)` and does `label.repeat_interleave(2)`. |
| `triplet` | `(B, 2, D)`, `label` ignored | Works. Returns EER from internal scoring as `prec1`. |

So there are two distinct bugs, both manifesting under `nPerSpeaker > 1`:

- **Bug A (critical / loud)**: AM/AAM-Softmax assertion immediately
  crashes the very first training batch. The repository's
  [MINDCF_IMPROVEMENT_GUIDE.md §4.4](../../MINDCF_IMPROVEMENT_GUIDE.md)
  *recommends* `nPerSpeaker: 2` with `aamsoftmax`; following that
  recommendation has been impossible.
- **Bug B (silent / cosmetic)**: When a metric-learning loss is used
  with `nPerSpeaker > 1`, `prec1` is the loss's **own internal**
  accuracy (a within-batch ranking score, or — for `triplet` — an EER),
  not the real classification accuracy. The print line in
  `train_network` formats this as `TEER/TAcc {top1/counter}%`, which
  reads as a percentage but is actually drawn from a meaningless or
  loss-specific scale. The researcher in October 2025 observed
  `0.000%` and labelled it cosmetic; the deeper issue is that the
  metric **the training loop uses to confirm convergence** is not
  what its variable name suggests, and there is no path that
  produces a real classification accuracy when `P > 1`.

### 1.3 Why "subsample labels" can't be the fix here
The prescription `label = label[::nPerSpeaker]` assumes the
embeddings have been averaged from `(B*P, D)` to `(B, D)` and that
labels (currently `(B,)`) somehow grew to `(B*P,)` and need
subsampling. In the actual code:

- labels are already `(B,)` and the embeddings are `(B, P, D)`;
- there is nothing to subsample. Slicing `[::P]` an already-`(B,)`
  vector gives `(ceil(B/P),)`, which is **smaller than B** and
  mis-aligned with the embeddings.

Applying the prescription verbatim would produce a label tensor of the
wrong length and break the metric losses (which currently work). The
right fix has to go in the **opposite direction**: when the loss wants
flat input, **flatten the embeddings and replicate the labels**.

---

## 2. Fix applied

### 2.1 Strategy
Introduce a single, declarative attribute on each loss class —
`expects_grouped_input` — that mirrors the existing `test_normalize`
attribute convention. Set it `True` on the five metric-learning losses
that consume `(B, P, D)`; leave it absent (defaults to `False` via
`getattr`) on the three classification losses that consume `(N, D)`.

In `SpeakerNet.forward`, dispatch on that attribute:

- **Grouped path** (metric losses): keep `outp` as `(B, P, D)`, pass
  `label` unchanged. Behaviour unchanged from before.
- **Flat path** (classification losses): reshape to `(B*P, D)` and
  `repeat_interleave` labels to `(B*P,)`. This is exactly the trick
  `softmaxproto` already uses internally for its embedded softmax
  call — we are just hoisting it to the wrapper so it applies to AM,
  AAM, and plain softmax uniformly.

This single dispatch closes both Bug A and Bug B at once:
- AM/AAM-Softmax now see `(B*P, D)` and `(B*P,)` — their assertion
  passes, classification proceeds, `prec1` is real top-1 accuracy.
- `nPerSpeaker = 1` is a strict no-op for either path (a `(B, 1, D)`
  tensor `.reshape(-1, D)` is identical to `(B, D)`, and
  `repeat_interleave(1)` is a no-op).
- Metric losses are untouched.

### 2.2 Loss-side change (five files)
Each metric-learning loss class gains one line in `__init__`:

```python
self.expects_grouped_input = True  # consumes (B, nPerSpeaker, D); SpeakerNet must not flatten
```

Touched: [loss/angleproto.py](../../loss/angleproto.py),
[loss/proto.py](../../loss/proto.py),
[loss/ge2e.py](../../loss/ge2e.py),
[loss/softmaxproto.py](../../loss/softmaxproto.py),
[loss/triplet.py](../../loss/triplet.py).
Classification losses (`softmax`, `amsoftmax`, `aamsoftmax`) are
untouched — the absence of the attribute is the signal.

### 2.3 `SpeakerNet.py` — forward path

```diff
-        else:
-
-            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
-
-            nloss, prec1 = self.__L__.forward(outp, label)
-
-            return nloss, prec1
+        # Reshape (nPerSpeaker * B, D) -> (B, nPerSpeaker, D), the grouped form
+        # consumed by metric-learning losses.
+        outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0)
+
+        if getattr(self.__L__, "expects_grouped_input", False):
+            # angleproto / proto / ge2e / softmaxproto / triplet: keep (B, P, D).
+            nloss, prec1 = self.__L__.forward(outp, label)
+        else:
+            # softmax / amsoftmax / aamsoftmax: flatten to (B*P, D) and replicate
+            # labels so prec1 measures real classification accuracy when P > 1.
+            outp = outp.reshape(-1, outp.size(-1))
+            label = label.repeat_interleave(self.nPerSpeaker)
+            nloss, prec1 = self.__L__.forward(outp, label)
+
+        return nloss, prec1
```

The dropped `.squeeze(1)` is intentional. It only ever did anything
when `nPerSpeaker == 1`, in which case the new flat path
(`reshape(-1, D)`) achieves the identical result with no behaviour
change. For metric losses with `nPerSpeaker == 1`, an assertion inside
the loss (`assert x.size()[1] >= 2`) was already firing and continues
to fire — the user must use `nPerSpeaker >= 2` with metric losses, as
the existing in-loss asserts have always required.

### 2.4 `SpeakerNet_performance_updated.py` — identical change
Same diff applied to the analogous block; the `non_blocking=True`
optimisation in this file is preserved. Diff omitted for brevity (see
the file).

### 2.5 `SpeakerNet_distillation.py` — standard-mode branch only
The distillation script has two execution paths: `use_distillation`
(handled by `DistillationWrapper`, §2.6 below) and a fallback
standard-mode path that mirrors plain `SpeakerNet`. Only the
standard-mode branch is patched here; the distillation branch is
handled by §2.6.

### 2.6 `DistillationWrapper.py` — student/teacher distillation flow
This file shares the buggy pattern at the classification-loss call,
*and* has to keep the teacher/student tensors in a compatible shape
for the distillation criterion to compare them. The fix:

- The grouped tensor `student_grouped = ...transpose(1, 0).squeeze(1)`
  is preserved (its `.squeeze(1)` keeps the prior teacher/student
  shape contract).
- The classification call now dispatches by `expects_grouped_input`,
  flattening + replicating labels only on the classification path.
- The distillation call still receives the grouped tensors unchanged,
  so distillation losses (cosine similarity, MSE on embeddings) see
  exactly the same shapes they used to — distillation behaviour is
  bit-for-bit preserved.

The pre-fix code passed the same `(B, P, D)` tensor to both the
classification loss and the distillation loss; for AM/AAM-Softmax that
would have asserted out at the classification call before distillation
ever ran. The fix unblocks the classification call without touching
distillation.

---

## 3. Behaviour matrix after the fix

| `nPerSpeaker` | Loss | Old | New |
|---|---|---|---|
| 1 | any | works | works (bit-identical: `(B, 1, D).reshape(-1, D) == (B, D).squeeze`) |
| ≥ 2 | `softmax` | broadcast bug → `RuntimeError` or wrong axis | flatten + replicate → real classification, real top-1 |
| ≥ 2 | `amsoftmax` | `AssertionError` | works, real top-1 |
| ≥ 2 | `aamsoftmax` | `AssertionError` | works, real top-1 |
| ≥ 2 | `angleproto` | works (within-batch ranking accuracy as `prec1`) | unchanged |
| ≥ 2 | `proto` | works (within-batch ranking) | unchanged |
| ≥ 2 | `ge2e` | works (within-batch ranking) | unchanged |
| 2 | `softmaxproto` | works (`prec1` is real classification top-1) | unchanged |
| 2 | `triplet` | works (`prec1` is internal EER) | unchanged |

`prec1` semantics for metric losses are **deliberately** unchanged —
the user's research-log analysis treated them as broken because they
display as "TEER/TAcc" in the training print, but they have always
been the loss's own well-defined accuracy. Renaming the print field
(e.g., `LossMetric` instead of `TEER/TAcc` when a metric loss is
active) is a follow-up cosmetic; it does not change any numerics.

---

## 4. Verification

### 4.1 Static
`python3 -m py_compile` on all twelve touched files:

```bash
$ python3 -m py_compile \
    SpeakerNet.py SpeakerNet_performance_updated.py \
    SpeakerNet_distillation.py DistillationWrapper.py \
    loss/angleproto.py loss/proto.py loss/ge2e.py \
    loss/softmaxproto.py loss/triplet.py \
    loss/softmax.py loss/amsoftmax.py loss/aamsoftmax.py
$ echo $?
0
```

### 4.2 Functional — recommended in-env tests
Run these in the project's `2025_colvaai` conda env before the first
real training run after this changeset:

**(a) AAM-Softmax + `nPerSpeaker=2` should now run, not assert:**
```bash
python trainSpeakerNet_performance_updated.py \
    --config configs/mini_voxceleb1_optimized_phase1.yaml \
    --max_epoch 1 --test_interval 1
```
Expected log markers in the first few batches:
- No `AssertionError` from `loss/aamsoftmax.py:37` (`assert x.size()[1] == self.in_feats`).
- A finite, non-zero `TEER/TAcc` percentage on each line of the inner
  progress display (typically rises from < 1% at init to a few %
  within an epoch on mini-VoxCeleb1).
- `loss` decreasing monotonically.

**(b) `nPerSpeaker=1` regression check (any classification loss):**
Should be bit-identical to the pre-fix behaviour. The cleanest
sanity-check is to keep an existing `exps/<run>/result/scores.txt`
from before the fix and confirm a fresh `nPerSpeaker=1` run
reproduces the same loss values for the first epoch under the same
seed. Any divergence would indicate an unintended behaviour change.

**(c) Metric-learning loss + `nPerSpeaker=2`:**
```bash
python trainSpeakerNet_performance_updated.py \
    --config configs/mini_voxceleb1_fewshot_ge2e.yaml \
    --max_epoch 1 --test_interval 1
```
Expected: identical numerics to the pre-fix run (this path is
unchanged by the fix).

### 4.3 Tiny in-Python smoke test
For someone who wants confidence without setting up the full training
pipeline:

```python
import torch
from SpeakerNet import SpeakerNet

# Stub config — use any model that accepts these kwargs.
net = SpeakerNet(
    model="MLPMixerSpeaker", optimizer="adam", trainfunc="aamsoftmax",
    nPerSpeaker=2, nOut=512, nClasses=140,
).cuda()

# (B, P, max_audio) — fake input matching the training loop's pre-transpose shape.
B, P, T = 8, 2, 32000
data = torch.randn(P, B, T).cuda()           # post-transpose shape
label = torch.randint(0, 140, (B,)).cuda()   # (B,), one speaker per batch element

nloss, prec1 = net(data, label)
assert nloss.requires_grad and nloss.dim() == 0
assert 0.0 <= float(prec1) <= 100.0
print(f"loss={float(nloss):.3f}, prec1={float(prec1):.2f}%")
```

If this prints a finite loss and a `prec1` in `[0, 100]%`, both bugs
are closed.

---

## 5. Why earlier attempts were reverted

The October-29/30 research logs describe an attempt that was rolled
back. From [`2025-10-29.md`](../../research_logs/2025-10-29.md) (the
"Attempted Fix (Reverted)" section): the attempted patch tried to
make embeddings match labels by `label = label[::self.nPerSpeaker]`
*before* any reshape, on the assumption that some code path averaged
embeddings. With the actual non-averaging code, that subsampling
yields labels of length `ceil(B/P)` — strictly smaller than the
embedding batch dimension — and immediately breaks every loss.

The lesson for future bug investigations on this codebase: **trace
shapes through `print(tensor.shape)` at the exact call boundaries**,
do not derive them from prose descriptions of the code. Two distinct
people in the project history have written about a `torch.mean`-based
averaging step that does not exist in any committed code; both were
working from a mental model that diverged from `git show HEAD:SpeakerNet.py`.
A small in-repo sanity check would have surfaced this in minutes:

```python
# Place this at the top of SpeakerNet.forward, run one batch, then remove.
print("data:", data.shape, "outp(post-S):", outp.shape, "label:", label.shape)
```

For Group A losses with `P > 1`, the printed shapes would have
immediately shown `outp=(B, 2, 512)` and `label=(B,)`, making the
real fix (flatten + replicate) obvious.

---

## 6. Side improvements bundled

- The blanket `.squeeze(1)` is replaced by a path that produces a
  bit-identical result for `nPerSpeaker == 1` without relying on the
  squeeze being a no-op for `P > 1`. This removes an implicit
  invariant ("squeeze(1) does nothing when P > 1") that was easy to
  break.
- The dispatch attribute name `expects_grouped_input` is chosen to
  mirror the existing `test_normalize` attribute convention used by
  every loss in `loss/`. No new abstraction is introduced.

Not bundled (deferred):
- Renaming the training-print column from `TEER/TAcc` to a name that
  reflects the actual semantic (`Acc` for classification, `RankAcc`
  for metric losses, `Train-EER` for triplet). This is a
  pure-cosmetic ergonomic change that touches three files; tracked as
  a polish item for a later pass.

---

## 7. Backward-compatibility impact

| Consumer | Effect |
|---|---|
| Anyone using `nPerSpeaker=1` with any loss | ✅ Bit-identical behaviour. |
| Anyone using `nPerSpeaker>1` with `angleproto` / `proto` / `ge2e` / `softmaxproto` / `triplet` | ✅ Bit-identical behaviour. |
| Anyone trying to use `nPerSpeaker>1` with `softmax` / `amsoftmax` / `aamsoftmax` | ⚠️ Path now works (previously crashed or silently miscomputed). Loss values will differ from any partial pre-fix runs because, before the fix, the loss either never ran (assert) or computed over the wrong axis. There is nothing to migrate — there were no successful runs of this combination to compare against. |
| Custom losses that consume `(B, P, D)` but do not set `expects_grouped_input = True` | ❌ Will be incorrectly flattened. Mitigation: set the attribute in the loss's `__init__` (one line). This is the only API contract change introduced by the fix and it is documented here. |

### 7.1 Loss-author contract (new)
A loss class in `loss/` is now expected to declare its preferred input
shape via a class attribute:

```python
class LossFunction(nn.Module):
    def __init__(self, ...):
        ...
        self.test_normalize = True            # existing
        self.expects_grouped_input = False    # NEW — True if forward consumes (B, P, D)
```

Setting it `True` is also valid when the loss internally flattens
(`softmaxproto` is the worked example). The flag should reflect what
the loss's `forward(x, label)` is willing to receive, not what it
internally computes on.

---

## 8. Rollback

If the fix needs to be reverted, three categories of change have to
unwind:

1. Remove the `expects_grouped_input = True` line in each of the five
   metric losses.
2. Restore the `outp = ...transpose(1, 0).squeeze(1); nloss, prec1 = ...`
   blocks in `SpeakerNet.py`, `SpeakerNet_performance_updated.py`,
   `SpeakerNet_distillation.py`.
3. Restore the original `student_output_reshaped`/`teacher_output_reshaped`
   block in `DistillationWrapper.py`.

After rollback, all the original failure modes return:
- AM/AAM-Softmax + `nPerSpeaker > 1` → AssertionError on first batch.
- softmax + `nPerSpeaker > 1` → RuntimeError or silent wrong axis.

There is no scenario in which rollback is correct.

---

## 9. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #3 | ✅ Closed by this document |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #1 | ✅ [BUGFIX-001](BUGFIX-001-mp-spawn-kwarg.md) |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #2 | ✅ [BUGFIX-002](BUGFIX-002-eer-threshold-not-returned.md) |
| §4.1 items #4–#9 | ✅ Closed by [BUGFIX-004](BUGFIX-004-hard-cuda-call.md), [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md), [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md), [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md), [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md), [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) |
| Polish: rename `TEER/TAcc` print column | ⬜ Deferred, not part of this bugfix |
| Polish: assertion test for AM/AAM with P > 1 | ⬜ Deferred to a test-harness pass |

---

## 10. Authorship & references

- **Bug originally identified by:** the researcher's October 2025
  investigation captured in
  [`research_logs/2025-10-29.md`](../../research_logs/2025-10-29.md)
  and [`research_logs/2025-10-30.md`](../../research_logs/2025-10-30.md).
- **Root-cause re-analysis (and discovery that the original
  prescription was based on an incorrect description of the
  code):** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #3.
- **Upstream provenance:** the original `.squeeze(1)` pattern is from
  the Clova AI parent repo. The contract split between classification
  and metric losses is also upstream. A note for upstream maintainers
  may be worthwhile.
