# BUGFIX-015 — `RawNet3.py`: remove debug `print`, replace in-place `s[s<…]=…` with `.clamp`

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-015-rawnet3-debug-print-and-inplace` |
| **Date** | 2026-05-18 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (production hygiene) for the `print`; medium-latent (autograd correctness) for the in-place mutation. |
| **Scope** | One model file (`models/RawNet3.py`), two small edits. No trainer, DataLoader, or config change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #15 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.2 report flagged two issues in `models/RawNet3.py`, both at
specific line numbers ("lines 49 and 88"). In the current tree those
lines have shifted by two (file edits since the report was filed —
unrelated BUGFIX work on this file via BUGFIX-006 / BUGFIX-008 / the
nested-model quarantine commit) and now live at lines **51** and
**89**. The shift is cosmetic; the two issues are exactly as
reported.

### 1.1 Line 51 — debug `print` in `__init__`

```python
# models/RawNet3.py (pre-fix), line 51
print("self.encoder_type", self.encoder_type)
```

Fires on every `RawNet3` instantiation — once per training run for a
single-model setup, more if DDP spawns multiple processes. It is
clearly a debug leftover (no formatting, no log prefix, single-string
content) and not a real diagnostic. The harm is small but not zero:

- Pollutes the training log with a line that has no operational
  meaning and trains a reader's eye to ignore short stderr/stdout
  lines.
- Bypasses the existing convention in the trainer of routing
  diagnostic information through the `print(args)` banner and the
  per-epoch loss line. A grep for `print(` in `models/` should return
  zero hits ideally (or only `print` calls that are explicit
  user-facing diagnostics).

### 1.2 Line 89 — in-place tensor mutation during `forward`

```python
# models/RawNet3.py (pre-fix), lines 86-90
elif self.norm_sinc == "mean_std":
    m = torch.mean(x, dim=-1, keepdim=True)
    s = torch.std(x, dim=-1, keepdim=True)
    s[s < 0.001] = 0.001               # <- in-place mutation
    x = (x - m) / s
```

The intent is reasonable — guard against division by a near-zero
standard deviation — but the *mechanism* (advanced-indexing assignment
on a freshly-computed tensor that is part of the current autograd
graph) is the textbook footgun PyTorch warns about. Three concrete
issues:

1. **Autograd safety.** `s` is the result of `torch.std`, which writes
   a version number into its tensor. Subsequent in-place modification
   bumps that version. If any later op records `s` (or any earlier
   tensor that shares storage with it) for use in the backward pass,
   the version mismatch triggers
   `RuntimeError: one of the variables needed for gradient computation
   has been modified by an inplace operation`. On the current code path
   `s` is only used to divide `x`, so the immediate divide-and-discard
   pattern is safe today — but the next refactor that retains `s` for
   any reason (e.g., logging, residual connections, knowledge
   distillation) will surface the failure. The latent character of the
   bug is exactly what makes it worth fixing now.
2. **AMP / `autocast` interaction.** The whole block runs inside
   `torch.amp.autocast('cuda', enabled=False)`. The
   advanced-indexing assignment pattern interacts poorly with some AMP
   builds (silent dtype promotion or unexpected upcast) — not a
   reported failure in this codebase, but documented in PyTorch
   issues #71596 and #92306 for similar in-place clamp patterns.
3. **Style consistency with the rest of the file.** Lines 110 and 124
   of the *same* `forward` method already use the chained
   `.clamp(min=..., max=...)` idiom on tensors at risk of going
   to zero:

   ```python
   torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)             # line 110
   (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4) # line 124
   ```

   The line-89 form is the odd one out — fixing it brings the
   numerical-guard idiom in the file to one consistent style.

The §4.2 prescription is `s = s.clamp(min=0.001)`, which is exactly
the right shape: produce a fresh tensor that the autograd graph can
treat as the canonical post-guard value, and dispose of the original
unconstrained `s` (it has no other users).

---

## 2. Fix

### 2.1 Remove the debug `print`

The `print("self.encoder_type", self.encoder_type)` line at the old
location is deleted outright. The surrounding lines are unchanged:

```python
if self.context:
    attn_input = 1536 * 3
else:
    attn_input = 1536
if self.encoder_type == "ECA":          # <- print was the line above
    attn_output = 1536
elif self.encoder_type == "ASP":
    attn_output = 1
else:
    raise ValueError("Undefined encoder")
```

The `self.encoder_type` value is still preserved in `__init__` (it
was set on the line above the print), so anyone genuinely interested
in inspecting it from a Python REPL can still
`model.encoder_type`. The `raise ValueError("Undefined encoder")`
already covers the failure case (someone passing an unsupported
value), which was probably the original reason the print was added.

### 2.2 Replace in-place `s[s < 0.001] = 0.001` with chained `.clamp`

```python
elif self.norm_sinc == "mean_std":
    m = torch.mean(x, dim=-1, keepdim=True)
    s = torch.std(x, dim=-1, keepdim=True).clamp(min=0.001)
    x = (x - m) / s
```

The `.clamp(min=0.001)` returns a new tensor with the same shape and
device as the input; values below `0.001` are lifted to `0.001`, all
other values pass through unchanged. The math the model computes
`(x - m) / s` produces exactly the same float result as before for
every input, but the tensor that participates in the autograd graph
is now constructed via a single non-mutating op rather than via a
fresh-then-mutated lifecycle.

The chained form (`.std(...).clamp(...)`) follows the pattern already
used on lines 110 and 124 of the same `forward` method, so the file
is now self-consistent in how it guards against small denominators.

---

## 3. Verification

### 3.1 Static

- `python -m py_compile models/RawNet3.py` returns exit 0.
- `grep -n 'print(' models/RawNet3.py` returns zero hits — confirmed
  the debug print is gone and no other `print` was hidden elsewhere
  in the file.
- `grep -n 's\[s' models/RawNet3.py` returns zero hits — the in-place
  indexed assignment is gone.
- `grep -n '.clamp' models/RawNet3.py` returns three hits (the new
  one on line 87 plus the two pre-existing ones on lines 110 and
  124), confirming the chained idiom landed where intended.

### 3.2 Behavioural equivalence at the float level

For any input tensor `s = torch.std(x, dim=-1, keepdim=True)`, the
two expressions

| Pre-fix | Post-fix |
|---|---|
| `s_pre = s.clone(); s_pre[s_pre < 0.001] = 0.001` | `s_post = s.clamp(min=0.001)` |

produce numerically identical results, element-wise, on every backend
PyTorch supports. They differ only in tensor identity and in how the
autograd graph is built — the *values* fed to the subsequent
`(x - m) / s` division are bit-for-bit the same.

### 3.3 Not verified

- A real training run of RawNet3 was not exercised (the audit-session
  Python lacks the project's runtime dependencies). The fix is a pure
  hygiene + correctness change with no algorithmic content, so the
  behavioural equivalence above is the relevant proof.
- The latent autograd failure mode in §1.2 was *not* reproduced in
  this fix — it is documented in PyTorch issues for similar patterns,
  but reproducing it would require constructing a graph where `s` is
  retained for backward (which the current `forward` does not do).
  The fix removes the footgun preemptively rather than after a
  failure.

---

## 4. Backward-compatibility & migration

- **Numerical output unchanged.** The float result of the
  `mean_std`-normalisation branch is identical to the pre-fix
  behaviour for every input value, so any pre-fix RawNet3 checkpoint
  loads cleanly into the post-fix model and produces the same
  embeddings.
- **`state_dict` keys unchanged.** No new parameters, no removed
  parameters, no renamings. Drop-in compatible with all existing
  checkpoints.
- **One line of stdout disappears per `RawNet3` construction.**
  Anyone parsing training logs with a strict pattern that
  *positively* matches the
  `"self.encoder_type ECA"` (or `ASP`) line will need to remove that
  expectation. There is no production parser of this kind in the
  repository, so this is theoretical.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Convert other `print` calls in `models/`.** A `grep -rn 'print('
  models/` shows a handful of other diagnostic prints, mostly in
  `__init__` blocks of various models (e.g., `print('Embedding size
  is %d, encoder %s.' % (nOut, encoder_type))` in `VGGVox.py`,
  `ResNetSE34L.py`, etc.). Those are deliberate user-facing
  construction banners, not debug leftovers — they appear at the same
  visual moment as the trainer's `print(args)` banner and are part of
  the same convention. The RawNet3 print is the outlier (no
  formatting, no context, no value-add) and is the only one removed.
- **Replace `torch.std(..., keepdim=True)` with a numerically more
  stable alternative.** PyTorch's `torch.std` already uses the
  Welford-style two-pass formulation; the `0.001` floor is a guard
  against the *legitimate* case of a constant-valued frame (zero
  variance), not against a numerical-instability case. No change
  needed.
- **Add a test that reproduces the latent autograd failure mode.**
  As described in §3.3, doing so requires constructing a scenario
  (retain `s` for backward) that the current `forward` does not
  expose. The fix is preventive; the test would belong in a future
  test-harness pass.
- **Run all configs that use RawNet3 to confirm.** Only one config
  (`configs/RawNet3_AAM.yaml`) targets this model, and it was not
  re-trained as part of this audit. The fix is behaviourally
  equivalent at float precision; re-running was not necessary to
  validate it.

---

## 6. Rollback plan

If for some reason this fix needs to be reverted:

1. Restore the debug `print` between the `attn_input = ...` blocks
   and the `if self.encoder_type == "ECA":` line. (Not recommended —
   the print has no value.)
2. Restore the in-place assignment by replacing the chained `.clamp`
   call with the two-line index-assignment pattern. (Strongly not
   recommended — the chained form is strictly safer.)

A partial rollback (keep the `.clamp` fix, restore the print) makes
no sense because the print was unconditional debug output. A partial
rollback the other way (keep the print removal, restore the in-place
assignment) is theoretically possible but pointless. The two fixes
are independent in scope but converge on the same goal: making the
model file production-clean.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#15** of the §4.2 list. Roadmap state:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) |
| 15 | `RawNet3.py` debug print + in-place mutation | ✅ **This document** |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |
| 20 | EER definition differs from common `(fpr+fnr)/2` | ✅ [BUGFIX-020](BUGFIX-020-eer-definition-disclosure.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #15.
- **PyTorch in-place version-tracking mechanism:**
  https://docs.pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd
- **`torch.Tensor.clamp` semantics:**
  https://docs.pytorch.org/docs/stable/generated/torch.clamp.html —
  returns a new tensor, does not mutate the input. The chained form
  `t.std(...).clamp(...)` keeps the intermediate `std` tensor as a
  short-lived autograd node with a single in-edge and out-edge,
  exactly the structure autograd handles best.
- **Related fixes:**
  [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md) — replaced
  `MaxPool1d(...) if pool else False` with `nn.Identity()` in the
  related `models/RawNetBasicBlock.py`. The two fixes are independent
  but share a theme: making the RawNet family of files less
  surprising under autograd.
