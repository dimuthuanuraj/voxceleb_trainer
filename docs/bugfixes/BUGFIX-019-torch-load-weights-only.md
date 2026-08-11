# BUGFIX-019 — Set `weights_only=True` on every `torch.load` call site

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-019-torch-load-weights-only` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | High by category (arbitrary code execution if a malicious checkpoint is loaded). Low by realised likelihood in this repository today (no public checkpoint sharing yet). The hardening cost is one keyword argument per call site. |
| **Scope** | Five files, eight `torch.load` call sites. No model, DataLoader, trainer-flow, or config change. No behaviour change for any checkpoint this repo itself wrote. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #19 |
| **Status** | ✅ Fixed |

---

## 1. Problem

`torch.load` historically defaults to a full unpickle of whatever is
in the file. The pickle format permits arbitrary Python objects, which
in turn means an attacker who controls a `.pt` / `.model` file
controls the import / instantiation of arbitrary Python code at the
moment a victim runs `torch.load`. This is the standard
*pickle-based remote-code-execution* vector and is the reason PyTorch
introduced the `weights_only=` parameter in 1.13 (Nov 2022) and
flipped its default to `True` in 2.6 (Jan 2025).

The §4.2 report flagged one specific call site:

> *"`SpeakerNet.py:248` — `torch.load(path, map_location="cuda:%d"
> % self.gpu)` without `weights_only=True`. Loading untrusted
> checkpoints is a code-execution risk. Set `weights_only=True`
> (PyTorch ≥2.4)."*

A repo-wide audit (`grep -rn 'torch.load' --include='*.py' | grep -v
weights_only`) returned **eight** call sites split across five files,
not the one mentioned. Each is in scope for the same hardening.

### 1.1 The eight call sites, by category

**Category A — model-checkpoint loaders (the report's target):**

| File:line | Loads | Source of file |
|---|---|---|
| [`SpeakerNet.py:310`](../../SpeakerNet.py#L310) | trainer's own `state_dict` | local `saveParameters` output |
| [`SpeakerNet_performance_updated.py:387`](../../SpeakerNet_performance_updated.py#L387) | same | same |
| [`SpeakerNet_distillation.py:471`](../../SpeakerNet_distillation.py#L471) | same | same |
| [`DistillationWrapper.py:114`](../../DistillationWrapper.py#L114) | **teacher** model `state_dict` | possibly external (pretrained) |

The last one matters most: a knowledge-distillation pipeline almost
by definition loads checkpoints that someone else trained. If those
ever come from outside the trusting boundary (a pretrained release,
a colleague's checkpoint, a download URL), they are exactly the
untrusted input the report describes.

**Category B — eval-feature loaders (added in BUGFIX-018, hardened
for defence-in-depth):**

| File:line | Loads | Source of file |
|---|---|---|
| [`SpeakerNet.py:190`](../../SpeakerNet.py#L190) | one eval-time embedding tensor | written by *this same process* moments earlier |
| [`SpeakerNet_performance_updated.py:227`](../../SpeakerNet_performance_updated.py#L227) | same | same |
| [`SpeakerNet_distillation.py:304`](../../SpeakerNet_distillation.py#L304) | same | same |

These files live under `<save_path>/eval_feats_tmp/`, written by
`torch.save(ref_feat, ...)` minutes earlier in the same script.
The attack surface here is hypothetical (an attacker would have to
overwrite the temp dir mid-eval), but consistent hardening means a
future code path that loads them in a different context inherits
the safer default.

**Category C — diagnostic tool that explicitly reads user-provided
checkpoints:**

| File:line | Loads | Source of file |
|---|---|---|
| [`analyze_nan_debug.py:78`](../../analyze_nan_debug.py#L78) | whatever checkpoint the user points `--checkpoint` at | *unconditionally external* |

The NaN-debug tool is the most directly user-facing path. Anyone
running `python analyze_nan_debug.py --checkpoint <someone-elses>.model`
is the textbook attack scenario. `weights_only=True` is the right
default here even more obviously than it is for the trainer-internal
loads.

---

## 2. Fix

### 2.1 The one-argument change

Each of the eight sites gained `weights_only=True` as an explicit
keyword:

```diff
- loaded_state = torch.load(path, map_location=self.device)
+ loaded_state = torch.load(path, map_location=self.device, weights_only=True)
```

The other arguments (`map_location`) are unchanged. Each site also
gained a short comment naming BUGFIX-019 and the local rationale.

### 2.2 Why this is safe for every checkpoint this repo writes

Every checkpoint produced by this repository is a **pure
`state_dict`** — i.e., a `dict[str, torch.Tensor]`. The save sites
are all of the form:

```python
torch.save(self.__model__.module.state_dict(), path)   # SpeakerNet.saveParameters
torch.save(ref_feat[batch_idx], _feat_path(fname))     # BUGFIX-018 eval feats
```

`weights_only=True` accepts dicts, lists, tuples, primitives, and
`torch.Tensor`. It rejects anything that would require importing a
user-defined class or executing pickle's reduce protocol on a custom
object. Pure state_dicts and bare tensors fall comfortably inside the
allowed set.

The only checkpoint format this code *can* produce that
`weights_only=True` would refuse is one we are not producing —
specifically, the kind that wraps state_dict in a custom class or
embeds an optimizer / scheduler object with non-tensor state. None
of the trainers do that.

### 2.3 What `weights_only=True` actually blocks

Per the PyTorch docs (and the underlying `Unpickler` implementation):

- `BUILD` opcodes that would invoke a custom class's
  `__setstate__` are rejected.
- `REDUCE` opcodes pointing at functions outside an allowlist of
  safe ones (`torch.storage._load_from_bytes`,
  `collections.OrderedDict`, primitive constructors, etc.) are
  rejected.
- The result: a crafted file containing
  `os.system("rm -rf ~")` as a pickled `REDUCE` cannot run during
  load. With the legacy `weights_only=False` default it would.

The mechanism is the safe subset of pickle, not a custom format
change. A file saved by `torch.save(state_dict, ...)` reads
bit-identically through both loaders.

### 2.4 PyTorch version compatibility

The `weights_only=` kwarg has been accepted by `torch.load` since
PyTorch **1.13** (Nov 2022). The audit's BUGFIX-011 pinned this
project to `torch>=2.1,<3`, so every supported environment accepts
the new argument. No version branching is needed.

The default value of `weights_only` flipped from `False` to `True`
in PyTorch **2.6** (Jan 2025). Our explicit `weights_only=True` is
therefore:

- A behaviour change on PyTorch 2.1–2.5 (the legacy unsafe default
  is replaced with the safe one).
- A no-op on PyTorch 2.6+ (we are stating the default explicitly).

In both worlds the code is correct and the code is safe.

---

## 3. Verification

### 3.1 Static — every site covered

After the fix:

```text
$ grep -rn 'torch.load' --include='*.py' | grep -v weights_only
(no output)
```

Eight sites, each carries `weights_only=True`:

```text
DistillationWrapper.py:117          torch.load(checkpoint_path, map_location='cpu', weights_only=True)
analyze_nan_debug.py:85             torch.load(path, map_location='cpu', weights_only=True)
SpeakerNet.py:193                   torch.load(_feat_path(filename), map_location='cpu', weights_only=True)
SpeakerNet.py:318                   torch.load(path, map_location=self.device, weights_only=True)
SpeakerNet_distillation.py:306      torch.load(_feat_path(filename), map_location='cpu', weights_only=True)
SpeakerNet_distillation.py:476      torch.load(path, map_location=self.device, weights_only=True)
SpeakerNet_performance_updated.py:229   torch.load(_feat_path(filename), map_location='cpu', weights_only=True)
SpeakerNet_performance_updated.py:392   torch.load(path, map_location=self.device, weights_only=True)
```

### 3.2 Syntax

`python -m py_compile` passes on all five touched files:
`SpeakerNet.py`, `SpeakerNet_performance_updated.py`,
`SpeakerNet_distillation.py`, `DistillationWrapper.py`,
`analyze_nan_debug.py`.

### 3.3 Behavioural

A pure-state-dict checkpoint loads bit-identically through
`torch.load(..., weights_only=False)` and
`torch.load(..., weights_only=True)`. There is no float-level or
tensor-shape change between the two paths — only the set of
*permitted byte sequences* in the file differs. All checkpoints this
repository writes fall inside the strict-subset, so:

- Existing `state_dict` files load successfully.
- `loadParameters` populates `self.__model__` identically.
- Training and evaluation continue with no observable difference.

### 3.4 Not verified

- No live attempted-exploit test was run; the audit pass does not
  attempt to construct a malicious checkpoint to demonstrate the
  pre-fix RCE. The vulnerability is well-documented upstream and the
  fix follows PyTorch's own published mitigation.
- An end-to-end DistillationWrapper run with a real teacher
  checkpoint was not exercised — system Python lacks the runtime
  deps. If a particular teacher checkpoint *was* saved as a
  non-state-dict object (unusual but legal), `weights_only=True`
  will raise a descriptive `UnpicklingError` and the operator can
  re-save the teacher as a state_dict before retrying. This is the
  correct UX for an unknown-provenance file.

---

## 4. Backward-compatibility & migration

- **All 19 configs work unchanged.** Path settings are untouched.
- **Every checkpoint this trainer has ever written loads unchanged.**
  `saveParameters` writes a pure `state_dict`; the strict loader
  accepts that without complaint.
- **Every BUGFIX-018 eval-feat file loads unchanged.** These files
  are bare `torch.Tensor` pickles, fully inside the safe subset.
- **External pretrained teacher checkpoints**: any teacher saved as
  a plain `state_dict` continues to load. A teacher saved with
  optimiser state or custom-class wrappers (rare) will raise
  `UnpicklingError`; the remedy is to re-extract the state_dict
  from the original training pipeline once, or to pass
  `weights_only=False` *manually* in that specific local edit —
  but the default should remain safe.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Adding `torch.serialization.add_safe_globals(...)` allowlists**
  for classes the repo *might* want to load in future (e.g., a
  pretrained Whisper teacher with custom dataclass metadata). That
  is a per-use-case decision and should be made when such a teacher
  actually exists, not speculatively.
- **A `--unsafe_torch_load` escape hatch** for users with legacy
  custom-class checkpoints. The audit-trail cost of carrying such a
  flag (which would have to be explicitly justified every time it's
  used) exceeds the inconvenience of a one-time
  `state_dict`-extraction step in the rare cases that need it.
- **Migrating the save side to a non-pickle format** (e.g.,
  safetensors). A larger change that introduces a runtime dependency
  and a checkpoint format flag-day; orthogonal to the §4.2 audit
  prescription, which asked specifically for the load-side
  hardening.
- **Replacing the four `loadParameters` re-key loop bodies with the
  newer `model.load_state_dict(..., strict=False)` form.** Pure
  refactor — touching it now would conflate hygiene with the
  security fix.

---

## 6. Rollback plan

If `weights_only=True` causes a real-world checkpoint to refuse to
load and the underlying file is known-good:

1. For the *specific* call site that needs it, change
   `weights_only=True` back to `weights_only=False` and document
   *why* in a comment naming the affected checkpoint and its origin.
2. Do **not** flip the default repo-wide. The strict mode is the
   correct default; only the specific exception should opt out.

The cleaner long-term remedy is to re-save the offending checkpoint
as a pure `state_dict` once and keep the strict loader. Per-call
relaxation should be the very-last resort.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#19** of the §4.2 list. Roadmap state:

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
| 19 | `torch.load` without `weights_only=True` | ✅ **This document** |
| 20 | EER definition differs from common `(fpr+fnr)/2` | ✅ [BUGFIX-020](BUGFIX-020-eer-definition-disclosure.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #19.
- **PyTorch security advisory and `weights_only` rationale:**
  https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true
- **PyTorch issue history (default flip in 2.6):**
  https://github.com/pytorch/pytorch/issues/52596
- **Background on pickle-based RCE:**
  https://docs.python.org/3/library/pickle.html#restricting-globals
- **Related fixes:**
  [BUGFIX-011](BUGFIX-011-requirements-pins.md) pinned
  `torch>=2.1,<3`, which guarantees `weights_only=` is an accepted
  kwarg at every supported version (the kwarg has been available
  since 1.13).
  [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) introduced the
  eval-feature `torch.load` sites that this fix also hardens.
