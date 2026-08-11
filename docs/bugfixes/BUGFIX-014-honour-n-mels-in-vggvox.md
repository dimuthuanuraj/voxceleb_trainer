# BUGFIX-014 — Honour `--n_mels` in `VGGVox` (and audit `ResNetSE34L`)

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-014-honour-n-mels-in-vggvox` |
| **Date** | 2026-05-18 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Medium — only one model file (`VGGVox.py`) actually exhibited the bug. The reported sibling case (`ResNetSE34L.py`) already honours `--n_mels`; this is documented honestly in §1.2 below. |
| **Scope** | One model file (`models/VGGVox.py`). No trainer, DataLoader, or config change. Verifies (without modifying) that the §4.2 report's claim about `models/ResNetSE34L.py` is incorrect. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #14 |
| **Status** | ✅ Fixed |

---

## 1. Problem

### 1.1 What the §4.2 report says

> *"`models/ResNetSE34L.py` and `VGGVox.py` hard-code n_mels. The
> `--n_mels` CLI flag is silently ignored. Either honour the parameter
> or remove the flag for these models."*

The prescription is the right shape: a global argparse flag whose
value is silently dropped is the kind of bug that turns reproducibility
claims into reproducibility lies. But the report's specific claim
about *which* files exhibit the bug is only half right.

### 1.2 What is actually true in the code

**`models/ResNetSE34L.py` already honours `n_mels`.** As of the current
tree:

```python
# Line 12
def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP',
             n_mels=40, log_input=True, sample_rate=16000, **kwargs):
    ...
    # Line 19
    self.n_mels = n_mels
    ...
    # Line 33
    self.instancenorm = nn.InstanceNorm1d(n_mels)
    # Line 35
    self.torchfb = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, win_length=400,
        hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels,
    )
```

The kwarg flows into both downstream consumers (the instance norm and
the mel-spectrogram). The conv stack ends with
`x = torch.mean(x, dim=2, keepdim=True)` at line 96, which pools the
mel dimension to 1, making the rest of the network mel-dim invariant.

Twelve configs in the current tree already set non-default values
(`n_mels: 64` in `experiment_01.yaml` /
`experiment_01_performance_updated.yaml` /
`mini_voxceleb2_config.yaml`; `n_mels: 80` in nine others) and these
configs successfully train ResNetSE34L. Empirically the parameter has
been working.

**`models/VGGVox.py` does hard-code `n_mels`**, in two places:

```python
# Line 11 (original)
def __init__(self, nOut=1024, encoder_type='SAP', log_input=True,
             sample_rate=16000, **kwargs):
    ...
    # Line 65 (original)
    self.instancenorm = nn.InstanceNorm1d(40)
    # Line 68 (original)
    self.torchfb = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, win_length=400,
        hop_length=160, f_min=0.0, f_max=sample_rate/2, pad=0,
        n_mels=40,
    )
```

The class signature does not accept `n_mels`, so any `--n_mels=80`
landing in the unpacked `**kwargs` is silently discarded.

### 1.3 The hidden architectural constraint in VGGVox

A naïve fix — just thread `n_mels` through to the two call sites —
would look correct in isolation but would crash at runtime for any
value other than 40. The CNN stack at lines 20–48 ends with:

```python
nn.Conv2d(256, 512, kernel_size=(4,1), padding=(0,0)),
```

The `(4,1)` kernel with no padding requires the input height (mel
dimension) at that point to be exactly **4** so the output height is
exactly **1**. Working backwards through the preceding strided convs
and max-pools, the only input mel dimension that arithmetically lands
at 4 before the final conv is `n_mels=40`. Any other value produces a
shape mismatch and a `RuntimeError` deep inside the forward pass.

This makes the "Either honour the parameter or remove the flag"
prescription more nuanced than it looks: we can honour the parameter
in the *front-end* (mel-spectrogram + instance norm), but the
*architecture* genuinely cannot consume other values without
redesign. Removing the global `--n_mels` flag is not an option because
several other models (ResNetSE34L, ResNetSE34V2, MLPMixerSpeaker,
LSTMAutoencoder, the quarantined NestedSpeakerNet) do consume it
correctly.

---

## 2. Fix

### 2.1 `models/VGGVox.py` — accept, thread, validate

The fix has three parts.

1. Add `n_mels=40` to the class signature so the kwarg is no longer
   silently swallowed by `**kwargs`.
2. Replace both hard-coded `40` literals (`InstanceNorm1d` and
   `MelSpectrogram`) with `n_mels`.
3. Add a class-level `_SUPPORTED_N_MELS = 40` constant and validate
   the incoming value in `__init__`, raising a descriptive
   `ValueError` if it differs. This converts the *implicit*
   architectural constraint into an *explicit* contract.

```python
class MainModel(nn.Module):
    # The CNN stack ends with Conv2d(256, 512, kernel_size=(4,1), padding=(0,0));
    # for that final 4-tall kernel to collapse the height to 1, the input mel
    # bin count must reduce to exactly 4 through the preceding strided convs and
    # max-pools. The original VGGVox design was tuned for n_mels=40 and the
    # arithmetic only works at that value — other values crash at the final
    # conv with a shape mismatch. We honour the parameter (per BUGFIX-014) but
    # reject incompatible values explicitly instead of silently ignoring them.
    _SUPPORTED_N_MELS = 40

    def __init__(self, nOut=1024, encoder_type='SAP', n_mels=40,
                 log_input=True, sample_rate=16000, **kwargs):
        super(MainModel, self).__init__()

        if n_mels != self._SUPPORTED_N_MELS:
            raise ValueError(
                f"VGGVox requires n_mels={self._SUPPORTED_N_MELS}; got n_mels={n_mels}. "
                f"The final Conv2d(kernel_size=(4,1)) layer expects the mel "
                f"dimension to reduce to 4 before the last conv, which is only "
                f"true for n_mels={self._SUPPORTED_N_MELS}. Use a different "
                f"model (ResNetSE34L / ResNetSE34V2) for other n_mels values."
            )
        ...
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=400,
            hop_length=160, f_min=0.0, f_max=sample_rate/2, pad=0,
            n_mels=n_mels,
        )
```

### 2.2 `models/ResNetSE34L.py` — no code change

The bug-report claim is inaccurate for this file. Verified:

- `n_mels` is a keyword argument with default `40` (line 12).
- `self.n_mels = n_mels` is stored (line 19).
- `nn.InstanceNorm1d(n_mels)` uses the parameter (line 33).
- `MelSpectrogram(..., n_mels=n_mels)` uses the parameter (line 35).
- `torch.mean(x, dim=2, keepdim=True)` (line 96) pools the mel
  dimension to 1, making the post-conv pipeline invariant to the
  exact value of `n_mels`.
- Existing configs already set `n_mels: 64` and `n_mels: 80` and run
  cleanly.

No edit is required. Documenting the verification here so a future
audit pass doesn't repeat the wrong claim.

### 2.3 ResNetSE34V2 — also already correct

Out of scope for the §4.2 report but checked while in the
neighbourhood: `models/ResNetSE34V2.py` similarly accepts `n_mels` as
a kwarg and uses `int(self.n_mels/8)` to size the post-conv head,
making it both honour the parameter and correctly adapt the
architecture. No change needed.

---

## 3. Verification

### 3.1 Static

- `python -m py_compile models/VGGVox.py` passes.
- `grep -n 'n_mels' models/VGGVox.py` shows zero remaining hard-coded
  `40` in `InstanceNorm1d` or `MelSpectrogram` call sites. The only
  remaining `40` literals are the class-constant `_SUPPORTED_N_MELS`
  and the kwarg default — both intentional.

### 3.2 Logical (the validation path)

The new validation block fires before any nn.Module construction
inside `__init__`, so a misconfigured run terminates immediately with
a clear error rather than several seconds later inside the conv stack.
Three call patterns are covered:

| Call pattern | Behaviour |
|---|---|
| `MainModel(nOut=..., n_mels=40, ...)` | Constructs successfully; same as historical behaviour. |
| `MainModel(nOut=..., n_mels=80, ...)` (or any non-40) | Raises `ValueError` with a descriptive message naming the constraint and suggesting alternative models. |
| `MainModel(nOut=..., ...)` (no `n_mels` kwarg) | Default `n_mels=40` applied; constructs successfully. Backwards-compatible. |

### 3.3 What is not verified

- A *real* training run of VGGVox at `n_mels=40` was not exercised
  (the project's conda env is not installed in this audit session,
  and no current config uses VGGVox — see §4 below). The fix is
  arithmetically and structurally equivalent to the original behaviour
  at the default value.
- The arithmetic claim "only `n_mels=40` lands at height 4 before the
  final conv" was traced through the architecture in §1.3 and matches
  the documented historical assumption, but not exhaustively
  cross-checked by enumerating every nearby integer.

---

## 4. Backward-compatibility & migration

- **No current config uses VGGVox.** `grep -l VGGVox configs/`
  returned zero hits before this fix, and the only references to the
  model live in `models/VGGVox.py` itself and a comment in
  `SL_LANGUAGE_SPV_ANALYSIS.md`. The new validation therefore changes
  observable behaviour for zero existing training runs.
- **Anyone constructing `VGGVox.MainModel` from a script** that
  previously relied on the silent-discard behaviour will now get a
  `ValueError` at construction time. That is the desired behaviour:
  the old silent path was the bug.
- **Configs targeting other models** (ResNetSE34L, ResNetSE34V2,
  MLPMixerSpeaker, etc.) are entirely unaffected.
- **ResNetSE34L's behaviour is unchanged** — no code edits there.
  Configs already setting `n_mels: 64` / `n_mels: 80` for it continue
  to work exactly as before.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Make VGGVox itself accept other `n_mels` values.** That would
  require redesigning the final `Conv2d(4,1)` layer (e.g., replacing
  it with an `AdaptiveAvgPool2d((1, None))` plus a `Conv2d(1,1)`).
  Two reasons against:
  (a) it invalidates any pre-trained VGGVox checkpoint that exists
       outside this repository;
  (b) no current config or research log requires VGGVox at non-40
       `n_mels`. Speculative architectural change.
- **Remove `--n_mels` from argparse.** Several other models legitimately
  consume it; removing the flag would break those. The §4.2 report
  presented removal as an option, but at the codebase level it is not
  one.
- **Rename `n_mels` to something model-specific** (e.g.,
  `vggvox_n_mels`). Out of scope and would break the kwarg convention
  shared with every other model in `models/`.
- **Add a similar `_SUPPORTED_N_MELS` constant to ResNetSE34L for
  symmetry.** Not needed — that model genuinely is mel-dim invariant
  (verified in §2.2). Adding a fake constraint would be misleading.

---

## 6. Rollback plan

If the new explicit failure mode is more friction than it removes
(unlikely given no current caller uses VGGVox at all):

1. Remove the `_SUPPORTED_N_MELS` constant and the `if n_mels != …:`
   block from `models/VGGVox.py`. Keep the `n_mels=40` kwarg default
   and the `InstanceNorm1d(n_mels)` / `MelSpectrogram(n_mels=n_mels)`
   substitutions. The result is a model that still honours the
   parameter but trusts the caller not to pass an architecture-
   incompatible value.
2. Optionally restore the original `40` literals if reverting to the
   pre-fix behaviour entirely. This is **not** recommended — it
   re-introduces the silent-discard bug.

The intermediate "honour but don't validate" state (step 1 only) is a
reasonable rollback target if the validation turns out to be too
strict for some future use case.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#14** of the §4.2 list. Roadmap state:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ **This document** |
| 15 | `RawNet3.py:49,88` — `print()` and in-place modification | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #14.
- **Architectural constraint analysis:** traced through
  `models/VGGVox.py` lines 20–48 by hand. The `(4,1)` final-conv
  expectation matches the documented design of VGG-M-style speaker
  networks (cf. Nagrani et al. 2017, *"VoxCeleb: A Large-Scale
  Speaker Identification Dataset"*) which used 40 mel filters.
- **Related fixes:** [BUGFIX-006](BUGFIX-006-model-side-sample-rate-threading.md)
  threaded `sample_rate` through every model's `MelSpectrogram` call
  in the same code locations; this fix completes the picture for
  VGGVox's other parametric mel-front-end argument.
