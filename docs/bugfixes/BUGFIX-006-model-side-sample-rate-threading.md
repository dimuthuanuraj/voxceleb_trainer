# BUGFIX-006 — Thread `sample_rate` through every model's mel/sinc frontend; keep n_fft / win_length / hop_length as 16 kHz-derived constants

| Field | Value |
|---|---|
| **ID** | BUGFIX-006 |
| **Severity** | High (closes the silent-corruption hazard introduced by BUGFIX-005 when a non-default `--sample_rate` is configured; for the default 16 kHz path it is a strict no-op) |
| **Component** | Mel-spectrogram and SincConv frontends across every model |
| **Files touched** | [models/MLPMixerSpeaker.py](../../models/MLPMixerSpeaker.py), [models/MLPMixerSpeaker_RawWaveform.py](../../models/MLPMixerSpeaker_RawWaveform.py), [models/experimental/NestedSpeakerNet.py](../../models/experimental/NestedSpeakerNet.py), [models/LSTMAutoencoder.py](../../models/LSTMAutoencoder.py), [models/ResNetSE34L.py](../../models/ResNetSE34L.py), [models/ResNetSE34V2.py](../../models/ResNetSE34V2.py), [models/VGGVox.py](../../models/VGGVox.py), [models/RawNet3.py](../../models/RawNet3.py) |
| **Source of report** | User question after BUGFIX-005 — "keep 16 as the standard, but if dataset has bigger Fs make sure we can configure it and the rest of the dependencies adjust" |
| **Scope decision** | User explicitly picked "minimal: only thread `sample_rate`, keep n_fft/win_length/hop_length as magic numbers" |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

> **Read first** — this fix makes `--sample_rate` *configurable* end-to-end
> without crashes or silent shape errors, but it is **not** a recommended
> performance optimisation for speaker verification. The SV literature is
> unanimous that 16 kHz is sufficient for speaker identity (F₀ + F₁–F₅ +
> glottal source + fricatives all fit under 8 kHz). The full conceptual
> story is in [SAMPLING_RATE_GUIDE.md](../../SAMPLING_RATE_GUIDE.md) §1.2.
> Default behaviour for every existing config is bit-identical.

---

## 1. Problem

[BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md) made the **data loader**
honour a configurable `--sample_rate` and resample on demand. But every
model in `models/` continued to construct its mel-spectrogram (or sinc
filterbank) with a literal `sample_rate=16000`:

```python
# models/MLPMixerSpeaker.py:257 (pre-fix)
self.torchfb = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,                # ← hard-coded, ignored --sample_rate
    n_fft=512, win_length=400, hop_length=160,
    ...
)
```

That left a footgun: if anyone set `--sample_rate 22050`, the loader would
deliver 22050 Hz samples to the model, and the model's mel filterbank would
**still compute filter centre frequencies as if the audio were 16 kHz**.
Mel bins would land at the wrong absolute frequencies, the upper bins would
fold the actual 8–11 kHz content of the audio into the wrong band, and the
training run would silently corrupt. No exception, no warning — just the
same kind of silent failure mode the loader fix was meant to eliminate.

### 1.1 Where the hard-codes lived
A pre-fix survey:

| File | Line | Pattern |
|---|---|---|
| [MLPMixerSpeaker.py](../../models/MLPMixerSpeaker.py) | 257 | `MelSpectrogram(sample_rate=16000, ...)` |
| [NestedSpeakerNet.py](../../models/NestedSpeakerNet.py) | 195 | same |
| [LSTMAutoencoder.py](../../models/LSTMAutoencoder.py) | 220 | same |
| [ResNetSE34L.py](../../models/ResNetSE34L.py) | 33 | same (one-liner) |
| [ResNetSE34V2.py](../../models/ResNetSE34V2.py) | 36 | same (inside `nn.Sequential(PreEmphasis(), ...)`) |
| [VGGVox.py](../../models/VGGVox.py) | 65 | `MelSpectrogram(sample_rate=16000, ..., f_max=8000, ...)` — `f_max` also hard-coded |
| [MLPMixerSpeaker_RawWaveform.py](../../models/MLPMixerSpeaker_RawWaveform.py) | 307 | `SincConv_fast(...)` without `sample_rate=` argument (defaulted to 16000) |
| [RawNet3.py](../../models/RawNet3.py) | 27 | `ParamSincFB(..., stride=kwargs["sinc_stride"])` — no `sample_rate=` passed |

### 1.2 The user's stated design intent
> "keep [16 kHz] as the standard and if the dataset has a bigger values
> make sure we can configure it to that sampling rate and adjust the rest
> of the dependencies like bin sizes and others according to it automatically."

Three implementation paths were possible:

1. **Full scaling** — `n_fft`, `win_length`, `hop_length`, `f_max` all
   derive from `sample_rate` so that the 25 ms window / 10 ms hop
   convention is preserved across rates. This is the standard
   speech-processing approach.
2. **Minimal threading** — `sample_rate` is passed to the frontend, but
   `n_fft`, `win_length`, `hop_length` stay at their 16 kHz values 512 /
   400 / 160. At a non-16 kHz rate the time-domain analysis window is
   shorter (or longer) than the standard 25 ms / 10 ms, but the pipeline
   is internally consistent and doesn't crash.
3. **No change** — accept the documented limitation from BUGFIX-005 §3
   that overriding `--sample_rate` requires patching the model files
   manually.

The user picked **(2) minimal threading**. BUGFIX-006 implements that.

---

## 2. Fix applied

### 2.1 Pattern (consistent across all eight files)

Each model's `__init__` gains a new `sample_rate=16000` keyword argument.
That argument flows through to the frontend constructor. Everything else
is unchanged:

```diff
-def __init__(self, ..., **kwargs):
+def __init__(self, ..., sample_rate=16000, **kwargs):
     super().__init__()
     ...
+    self.sample_rate = sample_rate

     # n_fft/win_length/hop_length kept at 16 kHz-derived 512/400/160 by design — see BUGFIX-006.
     self.torchfb = torchaudio.transforms.MelSpectrogram(
-        sample_rate=16000,
+        sample_rate=sample_rate,
         n_fft=512, win_length=400, hop_length=160,
         ...
     )
```

The bottom-of-trainer `SpeakerNet(**vars(args))` plumbing means that the
`--sample_rate` argparse flag added by BUGFIX-005 reaches each model
automatically. No config edits required.

### 2.2 Per-file specifics

**[MLPMixerSpeaker.py](../../models/MLPMixerSpeaker.py)** — straightforward.
Added `sample_rate=16000` kwarg, stored on `self`, threaded to
`MelSpectrogram`.

**[NestedSpeakerNet.py](../../models/NestedSpeakerNet.py)** — same pattern.
The model is documented as unstable / NaN-prone (see [research_logs/2025-12-29-nested-learning-experiment.md](../../research_logs/2025-12-29-nested-learning-experiment.md))
but the Fs fix is mechanical and orthogonal.

**[LSTMAutoencoder.py](../../models/LSTMAutoencoder.py)** — straightforward.
This is the 9.68% EER teacher model used by the distillation pipeline.

**[ResNetSE34L.py](../../models/ResNetSE34L.py)** and
**[ResNetSE34V2.py](../../models/ResNetSE34V2.py)** — single-line
`MelSpectrogram` constructor; same pattern. V2 has its mel-spec inside
`nn.Sequential(PreEmphasis(), MelSpectrogram(...))`; the kwarg goes inside
the inner `MelSpectrogram` call.

**[VGGVox.py](../../models/VGGVox.py)** — additionally hard-coded `f_max=8000`
(Nyquist for 16 kHz). Fixed to `f_max=sample_rate/2` so the upper mel bin
always lands at the actual Nyquist. This is the only file where a
non-default value other than `sample_rate=` itself was hard-coded; left as
a single-line comment in the diff.

**[MLPMixerSpeaker_RawWaveform.py](../../models/MLPMixerSpeaker_RawWaveform.py)** —
SincConv-based frontend. The `SincConv_fast` class already accepts
`sample_rate` (default 16000); the fix is just to pass it through:

```diff
-self.sincnet = SincConv_fast(
-    out_channels=num_filters,
-    kernel_size=kernel_size,
-    stride=stride
-)
+self.sincnet = SincConv_fast(
+    out_channels=num_filters,
+    kernel_size=kernel_size,
+    sample_rate=sample_rate,
+    stride=stride
+)
```

SincConv's filter centre frequencies are initialised on the mel scale
**relative to its `sample_rate` argument**, so they scale correctly at any
Fs once the parameter is threaded.

**[RawNet3.py](../../models/RawNet3.py)** — uses `ParamSincFB` from the
`asteroid_filterbanks` library. `ParamSincFB` accepts `sample_rate` (a
`float`, defaulted to 16000); the fix passes it explicitly:

```diff
 self.conv1 = Encoder(
     ParamSincFB(
         C // 4,
         251,
         stride=kwargs["sinc_stride"],
+        sample_rate=float(self.sample_rate),
     )
 )
```

`float(...)` is required because `asteroid_filterbanks` accepts a
`float` for `sample_rate`; the argparse path produces an `int`.

### 2.3 What the constants 512 / 400 / 160 mean at non-16 kHz rates

The `n_fft / win_length / hop_length` values are intentionally kept fixed
in **samples**, not derived from seconds:

| `sample_rate` | `win_length` (samples) | Window duration | `hop_length` | Hop duration | Notes |
|---|---|---|---|---|---|
| 8 000 | 400 | **50 ms** | 160 | **20 ms** | Doubled — longer time window, coarser time resolution |
| 16 000 | 400 | 25 ms | 160 | 10 ms | Standard SV framing |
| 22 050 | 400 | 18.1 ms | 160 | 7.3 ms | Shorter — finer time resolution, slightly worse low-freq resolution |
| 44 100 | 400 | 9.1 ms | 160 | 3.6 ms | Much shorter — equivalent to using a tiny window; **mostly unusable for SV** |
| 48 000 | 400 | 8.3 ms | 160 | 3.3 ms | Same — too short |

Mel-bin frequency placement is correct at every rate because `sample_rate`
is now honoured; what shifts is the **time** axis of the analysis. For
rates close to 16 kHz the change is benign. For rates far from 16 kHz
(e.g., 44.1 kHz) the window is effectively a tenth of a phoneme — not
useful for SV.

The user accepted this trade-off explicitly when picking the minimal
option. The full-scaling alternative (deferred — see §5) is the correct
choice if anyone actually needs to train at 44.1 kHz; for the realistic
SL-data path (downsample to 16 kHz at corpus-build time, then train),
the minimal threading is sufficient and zero-risk.

---

## 3. Verification

### 3.1 Static
All eight files compile cleanly:

```bash
$ python3 -m py_compile \
    models/MLPMixerSpeaker.py models/MLPMixerSpeaker_RawWaveform.py \
    models/NestedSpeakerNet.py models/LSTMAutoencoder.py \
    models/ResNetSE34L.py models/ResNetSE34V2.py \
    models/VGGVox.py models/RawNet3.py
$ echo $?
0
```

### 3.2 Static — call-site audit
Every literal `sample_rate=16000` that used to be inside a `MelSpectrogram(...)`
call is gone. The grep that found the bug now shows only **constructor
default values**, which is the intended state:

```bash
$ grep -nE "sample_rate=16000" models/*.py
models/LSTMAutoencoder.py:207:    ... sample_rate=16000, **kwargs):           # default
models/MLPMixerSpeaker.py:246:    ... sample_rate=16000, **kwargs):           # default
models/MLPMixerSpeaker_RawWaveform.py:69:    ... sample_rate=16000, ...       # SincConv_fast inner default
models/MLPMixerSpeaker_RawWaveform.py:299:    ... sample_rate=16000, **kwargs):  # default
models/NestedSpeakerNet.py:178:    ... sample_rate=16000, **kwargs):           # default
models/ResNetSE34L.py:12:    def __init__(..., sample_rate=16000, **kwargs):  # default
models/ResNetSE34V2.py:13:   def __init__(..., sample_rate=16000, **kwargs):  # default
models/VGGVox.py:11:           def __init__(..., sample_rate=16000, **kwargs):  # default
```

Zero matches inside `MelSpectrogram(...)`, `SincConv_fast(...)`, or
`ParamSincFB(...)` calls.

### 3.3 Functional — 16 kHz bit-identity check
Run any existing 16 kHz config; loss, EER, and MinDCF for epoch 1 under
a fixed seed should be **bit-identical** to a pre-BUGFIX-006 run:

```bash
python trainSpeakerNet.py \
    --config configs/experiment_01.yaml \
    --max_epoch 1 --test_interval 1
```

Because every model defaults `sample_rate=16000` and the default
argparse value is also 16000, the executed code path is byte-equivalent
to the pre-fix version.

### 3.4 Functional — `--sample_rate 22050` smoke test
Add a temporary print in any model's `__init__`:

```python
print(f"[debug] MelSpec configured at sample_rate={self.sample_rate}")
```

Run with `--sample_rate 22050`. Expected:
- `[debug] MelSpec configured at sample_rate=22050` in the log.
- Training does not crash on the first batch.
- `torchaudio.transforms.MelSpectrogram(sample_rate=22050)` will place
  mel filter centres on the 0–11.025 kHz mel scale.

Remove the print before committing.

### 3.5 Functional — SincConv smoke test
For `MLPMixerSpeaker_RawWaveform` or `RawNet3` configured at a non-default
rate, check that the SincConv filter initialisation reflects the actual
Fs. The simplest check (one-shot, run from a Python REPL):

```python
import torch
from models.MLPMixerSpeaker_RawWaveform import SincConv_fast
sinc = SincConv_fast(out_channels=80, kernel_size=251, sample_rate=22050)
print("min_low_hz:", sinc.min_low_hz)
print("high freq edge:", 22050 / 2 - sinc.min_low_hz - sinc.min_band_hz)
# Expect ~10925 Hz (was 7925 for sample_rate=16000) — bands now scale to actual Nyquist.
```

If the high frequency edge does **not** scale with `sample_rate`, the
fix is incomplete and the SincConv frontend is still hard-coded somewhere
internally.

---

## 4. Backward compatibility

| Consumer | Effect |
|---|---|
| Every existing config in `configs/` (all 16 kHz, default `--sample_rate`) | ✅ Bit-identical. The default `sample_rate=16000` keeps the constructor argument list at the same effective value, and every internal call path is the same. |
| Existing checkpoints (any `.model` file from past runs) | ✅ Load unchanged. The new kwarg has a default; old code paths produce the same `MelSpectrogram` object as before. |
| Custom user configs that override `n_mels` | ✅ Unaffected. `n_mels` stays user-configurable as before. |
| Custom configs that override `--sample_rate` to a non-default value | ⚠️ Will now produce a **correct** mel filterbank for that Fs (with the time-window caveat in §2.3), where pre-fix would have silently corrupted. There is nothing to migrate; pre-fix non-16kHz runs were broken regardless. |
| Code outside this repo that subclasses one of the model classes | ⚠️ Any subclass that called `super().__init__(...)` with explicit positional args may now hit the `sample_rate=` default landing slot. Mitigation: pass kwargs by name. No known internal subclasses to migrate. |

### 4.1 Pretrained checkpoints from this repo
The two existing best checkpoints (MLP-Mixer V2 at 10.32% EER and the
LSTM+AE teacher at 9.68% EER) were trained at 16 kHz. The mel-spec
frontend's `torchfb` is **non-trainable** (no learnable parameters in
`torchaudio.transforms.MelSpectrogram`), so the checkpoint's saved
weights do not include any Fs-specific tensors. Loading a 16 kHz
checkpoint into a `sample_rate=22050` model will work mechanically —
the model just becomes a 22050 Hz-frontend network with weights that
were optimised against 16 kHz mel features. Performance is undefined
in that mode and almost certainly degraded. **Do not fine-tune
existing checkpoints at a non-default `--sample_rate`** unless you have
a specific research reason.

---

## 5. Things this fix does NOT change

| Item | Why deferred |
|---|---|
| **Full scaling of `n_fft`/`win_length`/`hop_length` to preserve the 25 ms / 10 ms framing convention at any Fs.** | User explicitly picked the minimal option. The full-scaling change is mechanical (set `win_length = int(0.025 * sample_rate)` etc.) and can be a follow-up if a use-case demands it. It is **not** a no-op even at 16 kHz: `int(0.025 * 16000) = 400` and `int(0.010 * 16000) = 160`, so the constants don't change at 16 kHz, but `n_fft` (a power of 2 ≥ win_length) would need a `next_pow2` helper. Skipping that keeps this fix strictly bit-identical at 16 kHz. |
| **A loud warning** when someone sets `--sample_rate != 16000`. | User picked minimal threading; adding new stderr output is a behaviour change outside the strict scope. |
| **Validation that the loader's `--sample_rate` matches the model's `sample_rate`.** | Both default to 16000 and both source from the same argparse flag, so they always agree unless someone constructs `SpeakerNet` directly with mismatched kwargs. Out of scope. |
| **Bandwidth augmentation** (the *information-content* half of the mixed-Fs problem). | Still tracked as a future BUGFIX — needs its own design and separate ablation. The corpus-level recommendation (downsample to 16 kHz at build time) is the right path for SL data regardless. |

---

## 6. Rollback

Mechanical: remove `sample_rate=16000` from each constructor signature,
remove `self.sample_rate = sample_rate`, and revert each
`MelSpectrogram(sample_rate=sample_rate, ...)` to
`MelSpectrogram(sample_rate=16000, ...)`. For VGGVox, revert
`f_max=sample_rate/2` to `f_max=8000`. For the SincConv-based models,
remove the `sample_rate=sample_rate` (or `sample_rate=float(self.sample_rate)`)
argument from the SincConv / ParamSincFB constructor.

There is no scenario in which rollback is correct; the pre-fix code's
non-default `--sample_rate` path was silent corruption.

---

## 7. Closes / related

| Item | Status |
|---|---|
| BUGFIX-005 §3 follow-up "model-side `sample_rate` propagation" | ✅ Closed by this document |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #1–#5 | ✅ See BUGFIX-001..005 |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #6–#9 | ✅ Closed by [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md), [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md), [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md), [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) |
| Future polish: full scaling of n_fft/win_length/hop_length | ⬜ Open, tracked as a §5 follow-up |
| Future polish: bandwidth augmentation (info-content half of mixed-Fs) | ⬜ Open |
| Reference document for the conceptual story | [SAMPLING_RATE_GUIDE.md](../../SAMPLING_RATE_GUIDE.md) |

---

## 8. Authorship & references

- **Bug originally surfaced by:** the user's question after BUGFIX-005 —
  "keep 16 as the standards and if the dataset has a bigger values make
  sure we can configure it to that sampling rate and adjust the rest of
  the dependencies like bin sizes and others according to it
  automatically, can we do that? will it be effective?"
- **Honest assessment given to user before implementing:**
  Yes, mechanically; for SV the marginal benefit over 16 kHz is
  near-zero; the change is mainly defensive correctness. User picked
  the minimal-risk option ("only thread `sample_rate`, keep magic
  numbers").
- **`torchaudio.transforms.MelSpectrogram` docs:** https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html
- **`asteroid_filterbanks.ParamSincFB` docs:** https://asteroid-filterbanks.readthedocs.io/en/latest/api/asteroid_filterbanks.param_sinc_fb.html
- **Upstream provenance:** The hard-coded `sample_rate=16000` values are
  inherited verbatim from the Clova AI parent repo, where they made
  sense as a single-rate-trainer constant. A note for upstream
  maintainers may be worthwhile when the SL fork is ready to send
  patches.
