# BUGFIX-025 — Pull `PreEmphasis` and the standard mel frontend into `models/_frontend.py`

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-025-shared-audio-frontend` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (no runtime behaviour change for any existing checkpoint) but meaningful for codebase hygiene — two competing `PreEmphasis` class definitions are consolidated to one, and the 6-line `MelSpectrogram + InstanceNorm1d` block in four model files is consolidated to a single factory call. |
| **Scope** | One new file (`models/_frontend.py`), two thin compatibility shims (`utils.py`, `models/RawNetBasicBlock.py`), four mel-based model files updated to use the factory (`ResNetSE34L`, `ResNetSE34V2`, `MLPMixerSpeaker`, `LSTMAutoencoder`). No trainer, DataLoader, loss, or config change. No `state_dict` key change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #25 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.3 polish item flagged:

> *"Each model file duplicates `PreEmphasis`, mel transforms,
> `InstanceNorm`; pull into a shared `models/_frontend.py`."*

A repo-wide audit found two distinct categories of duplication, each
worth resolving on a different time-horizon:

### 1.1 `PreEmphasis` — two competing class definitions

The pre-emphasis high-pass filter (`y[n] = x[n] - 0.97 * x[n-1]`)
existed as **two separate `class PreEmphasis(nn.Module)` definitions**:

- `utils.py:22` — imported by `models/ResNetSE34V2.py`. The forward
  pass ends with `.squeeze(1)`, producing output shape `(B, T)`.
- `models/RawNetBasicBlock.py:8` — imported by `models/RawNet3.py`.
  The forward pass omits the squeeze, producing output shape
  `(B, 1, T)`.

Both definitions are byte-identical apart from the squeeze. The
shape difference is **load-bearing**, not accidental:

- The `utils.PreEmphasis` 2D output feeds into a `MelSpectrogram`
  which expects `(B, T)`.
- The `RawNetBasicBlock.PreEmphasis` 3D output feeds into
  `nn.InstanceNorm1d(1, ...)` which expects `(B, C, T)`.

So we have two implementations of the same idea, each pinned to a
specific downstream consumer. This is the textbook duplication
problem: someone fixing the filter in one file (e.g., changing the
default `coef` from 0.97 to 0.95) silently leaves the other file's
behaviour out of sync.

### 1.2 Mel + InstanceNorm — same 6-line block in 4 files

The standard mel-spectrogram + instance-norm setup is duplicated
across `ResNetSE34L`, `ResNetSE34V2`, `MLPMixerSpeaker`, and
`LSTMAutoencoder`. All four files contain a variant of:

```python
self.torchfb = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=512,
    win_length=400,
    hop_length=160,
    window_fn=torch.hamming_window,
    n_mels=n_mels,
)
self.instancenorm = nn.InstanceNorm1d(n_mels)
```

The four call sites agree on every parameter except `n_mels`. The
only minor variation is whether `instancenorm` is assigned before or
after `torchfb` (purely cosmetic, no runtime effect).

`ResNetSE34V2` is the lone variant: it wraps `MelSpectrogram` in a
`nn.Sequential(PreEmphasis(), MelSpectrogram(...))` to get
pre-emphasis. That is a `pre_emphasis=True` toggle, not a different
mel setup.

### 1.3 Intentional non-duplication (kept at call sites)

Two model files have mel pipelines that *look* superficially
similar but are model-specific:

- **`VGGVox.py`** sets `f_min=0.0, f_max=sample_rate/2, pad=0` and
  omits `window_fn=hamming`. These are tied to the VGGVox CNN
  stack's expectations (see BUGFIX-014 §1.3 on the
  `Conv2d(kernel_size=(4,1))` architectural constraint that fixes
  `n_mels=40`). Not duplication — model-specific configuration.
- **`RawNet3.py`** uses `SincConv_fast` instead of `MelSpectrogram`,
  paired with `PreEmphasis(squeeze=False) + InstanceNorm1d(1, ...)`.
  Different frontend family entirely.

These two are documented in `models/_frontend.py`'s module docstring
as deliberate non-clients of the factory.

---

## 2. Fix

### 2.1 `models/_frontend.py` — new canonical home

The new module defines two things:

```python
class PreEmphasis(nn.Module):
    """Canonical pre-emphasis high-pass filter."""
    def __init__(self, coef: float = 0.97, squeeze: bool = True) -> None:
        super().__init__()
        self.coef = coef
        self.squeeze = squeeze
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2, "..."
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), "reflect")
        out = F.conv1d(x, self.flipped_filter)
        return out.squeeze(1) if self.squeeze else out


def make_mel_frontend(
    sample_rate: int = 16000,
    n_mels: int = 40,
    pre_emphasis: bool = False,
):
    """Standard mel-spectrogram + instance-norm frontend factory.

    Returns (torchfb, instancenorm). The two are *not* bundled into a
    Sequential — each caller assigns them to its own self.torchfb /
    self.instancenorm attributes so the resulting state_dict keys
    match the pre-fix per-model layout byte-for-byte.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512, win_length=400, hop_length=160,
        window_fn=torch.hamming_window,
        n_mels=n_mels,
    )
    if pre_emphasis:
        torchfb = nn.Sequential(PreEmphasis(squeeze=True), mel)
    else:
        torchfb = mel
    instancenorm = nn.InstanceNorm1d(n_mels)
    return torchfb, instancenorm
```

Two design decisions worth flagging:

1. **The factory returns a tuple, not a Sequential.** Bundling
   `(torchfb, instancenorm)` into a single `nn.Sequential` would
   change every model's state_dict layout (it would add an extra
   level of nesting under a new module name). Returning a tuple
   that each caller unpacks into the two existing attribute names
   preserves the layout exactly, so every pre-fix checkpoint loads
   unchanged. See §3.2 below.
2. **`pre_emphasis` is a single bool, not a `pre_emphasis_module`.**
   The Sequential wrapping when `pre_emphasis=True` exactly
   reproduces ResNetSE34V2's pre-fix structure
   (`torchfb.0 = PreEmphasis`, `torchfb.1 = MelSpectrogram`).

### 2.2 Legacy `PreEmphasis` paths preserved as thin subclasses

The two pre-existing `class PreEmphasis(...)` definitions are
replaced with thin subclasses of the canonical class, pinning the
historical default for `squeeze`:

```python
# utils.py (was: a full standalone class)
from models._frontend import PreEmphasis as _CanonicalPreEmphasis

class PreEmphasis(_CanonicalPreEmphasis):
    """Backwards-compatibility alias.
    Inherits squeeze=True default (historical behaviour)."""
    pass
```

```python
# models/RawNetBasicBlock.py (was: a full standalone class)
from models._frontend import PreEmphasis as _CanonicalPreEmphasis

class PreEmphasis(_CanonicalPreEmphasis):
    """Backwards-compatibility alias.
    Pins squeeze=False for the raw-waveform pipeline."""
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__(coef=coef, squeeze=False)
```

Every existing import path (`from utils import PreEmphasis`,
`from models.RawNetBasicBlock import PreEmphasis`) continues to
work and continues to produce a class with the *exact same output
shape* as before. The behaviour at runtime is identical down to
the float bit pattern of the output tensor.

### 2.3 Four mel-based models migrated to the factory

The 6-line construction block in each of `ResNetSE34L`,
`ResNetSE34V2`, `MLPMixerSpeaker`, and `LSTMAutoencoder` was
replaced with a 3-line factory call. Representative example:

```diff
- self.instancenorm = nn.InstanceNorm1d(n_mels)
- self.torchfb = torchaudio.transforms.MelSpectrogram(
-     sample_rate=sample_rate,
-     n_fft=512, win_length=400, hop_length=160,
-     window_fn=torch.hamming_window,
-     n_mels=n_mels,
- )
+ self.torchfb, self.instancenorm = make_mel_frontend(
+     sample_rate=sample_rate, n_mels=n_mels, pre_emphasis=False,
+ )
```

`ResNetSE34V2` is the same shape but with `pre_emphasis=True` —
collapsing its pre-fix `nn.Sequential(PreEmphasis(),
MelSpectrogram(...))` into the factory's `pre_emphasis=True` path.
Each model added a one-line `from models._frontend import
make_mel_frontend` to its import block, and removed the now-unused
`from utils import PreEmphasis` line from `ResNetSE34V2`.

### 2.4 What was deliberately NOT touched

- `VGGVox.py` — model-specific frontend (see §1.3).
- `RawNet3.py` — different frontend family (SincConv).
- `models/experimental/NestedSpeakerNet.py` — quarantined under
  BUGFIX-010; the quarantine policy says "do not modify the
  quarantined file in main". Its internal frontend remains
  unchanged.

A comment in `models/_frontend.py`'s module docstring explicitly
names each of these as "deliberate non-clients" so a future audit
doesn't try to absorb them mechanically.

---

## 3. Verification

### 3.1 Static

All seven touched files pass `python -m py_compile`:
- `models/_frontend.py` (new)
- `utils.py`, `models/RawNetBasicBlock.py` (shims)
- `models/ResNetSE34L.py`, `models/ResNetSE34V2.py`,
  `models/MLPMixerSpeaker.py`, `models/LSTMAutoencoder.py`
  (factory clients)

After the fix, exactly **one** `^class PreEmphasis(` definition
exists in the codebase that does real work: `models/_frontend.py:56`.
The two other `class PreEmphasis(...)` occurrences
(`utils.py:25`, `models/RawNetBasicBlock.py:11`) are both
`class PreEmphasis(_CanonicalPreEmphasis): ...` — pure subclasses
with no method overrides except the constructor in the RawNet shim.

### 3.2 Checkpoint compatibility (state_dict key invariance)

The factory return shape is constrained specifically to preserve
state_dict keys. For each migrated model:

| Model | Pre-fix attributes | Post-fix attributes | State_dict keys differ? |
|---|---|---|---|
| `ResNetSE34L` | `self.torchfb = MelSpectrogram(...)`, `self.instancenorm = InstanceNorm1d(...)` | Same two attributes, same types | No |
| `ResNetSE34V2` | `self.torchfb = nn.Sequential(PreEmphasis(), MelSpectrogram(...))`, `self.instancenorm = InstanceNorm1d(...)` | Same two attributes, same types (factory returns the Sequential in the same order) | No |
| `MLPMixerSpeaker` | `self.torchfb = MelSpectrogram(...)`, `self.instancenorm = InstanceNorm1d(...)` | Same | No |
| `LSTMAutoencoder` | Same as MLPMixer | Same | No |

The Sequential ordering in `ResNetSE34V2` is critical:
pre-fix `torchfb.0 = PreEmphasis`, `torchfb.1 = MelSpectrogram`.
Post-fix the factory builds `nn.Sequential(PreEmphasis(squeeze=True),
mel)` which produces the same ordering and therefore the same key
paths in the saved state_dict (`torchfb.0.flipped_filter`,
`torchfb.1.spectrogram.window`, etc.).

### 3.3 AST-level smoke test (7 checks, all pass)

A standalone test without torch verified:

| # | Check | Result |
|---|---|---|
| 1 | `make_mel_frontend` signature is `(sample_rate, n_mels, pre_emphasis)` | ✅ |
| 2 | `PreEmphasis.__init__` signature is `(self, coef, squeeze)` | ✅ |
| 3 | `utils.PreEmphasis` extends `_CanonicalPreEmphasis` | ✅ |
| 4 | `models.RawNetBasicBlock.PreEmphasis` extends `_CanonicalPreEmphasis` | ✅ |
| 5 | `ResNetSE34L` uses `make_mel_frontend(..., pre_emphasis=False)` | ✅ |
| 6 | `ResNetSE34V2` uses `make_mel_frontend(..., pre_emphasis=True)` | ✅ |
| 7 | `MLPMixerSpeaker` and `LSTMAutoencoder` use `pre_emphasis=False` | ✅ |

### 3.4 Not verified

- No real training run was exercised — system Python lacks the
  project's runtime deps. The runtime correctness argument is by
  construction: the factory builds the same `nn.Module` objects in
  the same order with the same arguments that the pre-fix code did,
  so any forward pass that worked before works after.
- Loading a pre-fix checkpoint into a post-fix model was not
  exercised end-to-end. The argument that this works is by
  state_dict key inspection (§3.2): the keys are constrained to be
  identical, so PyTorch's
  `model.load_state_dict(checkpoint, strict=True)` will succeed.
- The runtime behaviour of `nn.InstanceNorm1d` is identical to the
  pre-fix code by construction — it is the same class instantiated
  with the same `n_mels` argument.

---

## 4. Backward-compatibility & migration

- **No checkpoint format change.** Every pre-fix `.model` file
  produced by any of the four migrated models loads into the
  post-fix model with `strict=True` and continues to produce
  identical embeddings.
- **No import-path break.** `from utils import PreEmphasis` and
  `from models.RawNetBasicBlock import PreEmphasis` both still
  work and produce classes with their historical output shapes.
- **No CLI / config change.** None of the four migrated models
  introduces or removes any kwarg.
- **`models/_frontend.py` is the recommended import for new code**
  — both `PreEmphasis(squeeze=...)` and `make_mel_frontend(...)`
  are exported. The legacy alias paths will keep working
  indefinitely; there is no deprecation timeline.

---

## 5. Out-of-scope

- **Pulling `VGGVox.py`'s mel construction into the factory.**
  Rejected: VGGVox sets `f_min`, `f_max`, `pad` model-specifically
  and omits hamming windowing. Forcing it through the factory would
  either bloat the factory's parameter list with VGGVox-specific
  knobs (premature abstraction) or change VGGVox's mel behaviour
  (which BUGFIX-014 §1.3 says is architecture-constrained).
- **Pulling `RawNet3.py`'s SincConv frontend into the factory.**
  Rejected: that's a different frontend family entirely; "make mel
  frontend" is the wrong abstraction for it.
- **Deleting `utils.PreEmphasis` and
  `models.RawNetBasicBlock.PreEmphasis` entirely** in favour of
  forcing every caller to import from `_frontend.py`. Rejected:
  the audit prescription was "pull into a shared
  `models/_frontend.py`", not "rename every import in every
  caller". The thin-subclass approach delivers the deduplication
  without the breakage.
- **Pulling InstanceNorm1d out** as a standalone `make_instance_norm`
  factory. Rejected: `nn.InstanceNorm1d(n_mels)` is a single line
  with one argument. The CLAUDE.md prelude is explicit that
  "three similar lines is better than a premature abstraction".
  The factory already returns the InstanceNorm alongside the mel
  to keep the two together logically, which is the right granularity.
- **Refactoring the four migrated models' `forward` methods** to
  share a `_apply_frontend(x)` helper. Each model's forward
  applies the frontend slightly differently (some call
  `.detach()`, some don't; some `.unsqueeze(1)` for downstream
  conv stacks, some don't). Those differences are intentional and
  model-specific.

---

## 6. Rollback plan

If `models/_frontend.py` causes a problem (e.g., an import-order
issue in some downstream tool that imports `utils` before
`models` is on the path):

1. Revert `utils.py` and `models/RawNetBasicBlock.py` to their
   pre-fix full-class definitions (one-block paste-back from git
   history).
2. Revert the four migrated models — each had two edits (one
   import, one factory call) which can be inverted mechanically.
3. Delete `models/_frontend.py`.

A *partial* rollback that keeps the `PreEmphasis` consolidation but
reverts the mel-factory migration is valid — the two are
independent. Concretely: keep `models/_frontend.py` with only the
`PreEmphasis` class; delete `make_mel_frontend`; revert the four
mel-based models. This preserves the "two PreEmphasis definitions
→ one" win without the factory's risk.

---

## 7. Related items in §4.3 of the analysis

This fix closes item **#25**. Roadmap state:

| # | Title | Status |
|---|---|---|
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| 22 | Inconsistent variable casing | ✅ [BUGFIX-022](BUGFIX-022-camel-snake-case-aliases.md) |
| 23 | `dataprep.py` MD5 mismatch raises `Warning` | ✅ [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md) |
| 24 | Performance scripts overlap | ✅ [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md) |
| 25 | Each model duplicates PreEmphasis / mel / InstanceNorm | ✅ **This document** |
| 26 | `exps/` folders mostly contain `logs/` / `result/` but no checkpoints | ✅ [BUGFIX-026](BUGFIX-026-exps-cleanup-policy.md) |

§4.1 (Critical) is fully closed (BUGFIX-001..010).
§4.2 (Important) is closed except for **#12** (`lists/` empty — Sri Lankan dataprep workstream).
§4.3 (Minor / polish) — five items closed; one remaining (#26).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #25.
- **The pattern this fix established:** a single canonical module
  for shared model-side primitives, with thin compatibility shims
  at the legacy import paths. The same pattern could absorb a
  future `make_resnet_block`, `make_attention_pool`, etc. without
  re-organisation. The factory-returns-tuple convention preserves
  `state_dict` layout, which is essential when migrating an
  in-the-wild codebase to a shared helper.
- **`torchaudio.transforms.MelSpectrogram` semantics:**
  https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html
  — confirms the n_fft / win_length / hop_length parameters are
  unaffected by the factory's wrapping behaviour.
- **`torch.nn.Sequential` state_dict layout:**
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
  — keys are prefixed by the index of each child module (`0.`, `1.`).
  Critical to the §3.2 checkpoint-invariance argument.
- **Related fixes:**
  [BUGFIX-006](BUGFIX-006-model-side-sample-rate-threading.md)
  established the policy that `n_fft`, `win_length`, `hop_length`
  are kept at the 16 kHz-derived constants regardless of
  `sample_rate`. This fix moves that policy from "scattered comment
  in every model" to "the factory's documented contract".
  [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) explains why
  `VGGVox.py` is intentionally not a factory client.
  [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) is the
  reason `RawNet3.py`'s SincConv-based frontend isn't here either.
