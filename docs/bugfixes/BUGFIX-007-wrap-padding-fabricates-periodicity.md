# BUGFIX-007 — `numpy.pad(audio, ..., 'wrap')` fabricates periodic features that the model can shortcut-learn

| Field | Value |
|---|---|
| **ID** | BUGFIX-007 |
| **Severity** | High (silent — the model trains and reports normal-looking metrics while learning a non-speaker shortcut feature) |
| **Component** | `loadWAV` short-clip handling in both data loaders |
| **Files touched** | [DatasetLoader.py](../../DatasetLoader.py), [DatasetLoader_performance_updated.py](../../DatasetLoader_performance_updated.py) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #6 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

[DatasetLoader.py:38](../../DatasetLoader.py#L38) (pre-fix) read:

```python
if audiosize <= max_audio:
    shortage = max_audio - audiosize + 1
    audio    = numpy.pad(audio, (0, shortage), 'wrap')
    audiosize = audio.shape[0]
```

`numpy.pad` with `mode='wrap'` **tiles the original audio onto itself** —
for a 1-second clip padded to 2 seconds the result is the original 1 s
followed by another copy of the same 1 s.

### 1.1 Why this is a real problem

1. **Perfect periodicity is a shortcut feature.** A model trained on
   speaker-classification (AM/AAM-Softmax) is incentivised to use *any*
   reliable per-utterance feature. A 1-second clip wrapped to 2 seconds
   produces a waveform with an exact `T = 1 s` periodicity. The model
   can learn "if I see this exact T periodicity, I'm looking at speaker
   X who tends to have short clips." This is precisely the shortcut-
   learning failure mode that Geirhos et al. (*Nat. Mach. Intell.* 2020)
   document — a non-causal feature that correlates with the label and
   the model uses it instead of the intended concept.

2. **Phantom phoneme transitions.** At the wrap seam (`t = audiosize`)
   the signal jumps discontinuously back to whatever was at `t = 0`.
   For speech this is almost always a phonetic boundary that the
   original speaker never produced. The model sees a phoneme transition
   that doesn't physically exist.

3. **Frequency-domain artefacts.** A periodic signal of period `T` has
   spectral lines at `1/T, 2/T, 3/T, ...`. For a 1-second wrap-padded
   clip that's spectral lines at every integer Hz. These lines land
   inside the mel filterbank as a deterministic comb pattern that
   correlates with the original-clip duration — yet another non-speaker
   shortcut.

4. **Random-crop interaction.** In training mode, after wrap-padding
   the loader takes a random crop. With audio padded to length ~`max_audio + 1`
   the random offset reduces to `0`, so the same crop is taken every
   epoch. The model sees the same `(audio || audio || ...)` tile every
   time — no data augmentation from the crop, despite the loader
   acting like it's augmenting.

### 1.2 When this actually bites
Pre-fix the bug was silent in two senses:
- No error ever raised.
- It only triggers on clips shorter than `max_audio` (≈ 2 s at the
  default `max_frames=200, sample_rate=16000`).

VoxCeleb1/2 clips are mostly 4–15 seconds, so the bug almost never
fires there — which is why the upstream Clova AI repo never noticed.
Sri Lankan corpora and OpenSLR Sinhala / Tamil are more likely to
contain short clips (TTS prompts, news segments, KYC snippets), and
MUSAN noise files have variable durations. The first SL training run
on a corpus with short clips would silently exhibit this shortcut
without any log signal.

### 1.3 Pre-fix code-path inventory
The buggy pattern occurred at:
- [DatasetLoader.py:38](../../DatasetLoader.py#L38) — main `loadWAV`
- [DatasetLoader_performance_updated.py:104](../../DatasetLoader_performance_updated.py#L104) — perf-updated `loadWAV` (LRU-cached path)

The `AugmentWAV.additive_noise` path uses `loadWAV(noise, ..., evalmode=False)`
to load MUSAN noise, so it also exercised the same broken short-clip code
when a MUSAN file happened to be shorter than `max_audio`. The fix covers
both call sites automatically because the helper is invoked once per
`loadWAV` call.

---

## 2. Fix applied

### 2.1 Strategy
Two suggestions appeared in the bug report:

1. `mode='constant'` (silence pad) with an energy-based VAD warning.
2. Repeat with small Gaussian noise to avoid perfectly periodic features.

The selected design is a **hybrid** that captures the best of both:

- **Silence pad** as the base operation — it is the honest answer to
  "I don't have enough audio; here is the absence of audio." No
  fabricated content, no fabricated periodicity, no spectral comb.
- **Tiny Gaussian dither** in the padded region (std `≈ 1e-4`, about
  −80 dBFS) — comfortably below the natural noise floor of 16-bit PCM
  audio (quantisation floor ≈ −96 dBFS), inaudible, but non-zero. This
  avoids a perfectly-zero block that can produce zero-variance windows
  in downstream `InstanceNorm` and zero-energy frames in mel-statistics.

The "energy-based VAD warning" suggested in the report is implemented
as a **one-time stderr warning** the first time any worker process
sees a short clip — a corpus-quality red flag without the implementation
overhead of a real VAD. Subsequent short clips are padded silently to
avoid log spam at 8-worker DataLoader scale.

### 2.2 Module-level helpers (both loader files)

```python
# Module-level flag so the short-clip warning fires at most once per worker process.
_short_clip_warned = False


def _pad_short_with_dither(audio, target_length, dither_std=1e-4):
    """Right-pad an audio array with silence plus tiny Gaussian dither.

    Pre-BUGFIX-007 the loader used ``numpy.pad(..., 'wrap')`` which tiles the
    original audio onto itself and fabricates perfectly periodic features that
    the model can learn as a shortcut for short-clip speakers. Silence pad is
    the honest default. The small Gaussian dither (std ~1e-4, about -80 dBFS)
    avoids exact-zero blocks that can produce zero-variance windows downstream.
    """
    shortage = target_length - audio.shape[0]
    if shortage <= 0:
        return audio
    pad = (numpy.random.randn(shortage) * dither_std).astype(audio.dtype, copy=False)
    return numpy.concatenate([audio, pad], axis=0)


def _maybe_warn_short_clip(filename, audiosize, target):
    """Emit a one-time stderr warning the first time a short clip is padded."""
    global _short_clip_warned
    if _short_clip_warned:
        return
    _short_clip_warned = True
    sys.stderr.write(
        "[DatasetLoader] First short clip encountered "
        f"({filename}: {audiosize} samples < {target} required). "
        "Padding with silence + dither. Subsequent short clips will be padded silently. "
        "Consider filtering clips shorter than max_audio at corpus-prep time.\n"
    )
```

Duplicated across the two loader files for the same reason `_resolve_device`
was in BUGFIX-004: the repo has no shared utilities module, and each loader
is a drop-in interchangeable variant.

`numpy.random.randn` uses the global NumPy RNG, which is seeded per-worker
by the existing `worker_init_fn`:
```python
def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)
```
So the dither is reproducible from the worker seed — runs with the same
seed produce the same dither, runs with different seeds get different
dither, matching the existing augmentation behaviour.

### 2.3 Call-site rewrite (both files)

```diff
 audiosize = audio.shape[0]

 if audiosize <= max_audio:
-    shortage    = max_audio - audiosize + 1
-    audio       = numpy.pad(audio, (0, shortage), 'wrap')
-    audiosize   = audio.shape[0]
+    target_length = max_audio + 1  # +1 keeps the historical invariant that audiosize-max_audio >= 1
+    _maybe_warn_short_clip(filename, audiosize, target_length)
+    audio = _pad_short_with_dither(audio, target_length)
+    audiosize = audio.shape[0]
```

The `+1` is preserved. It exists so that the downstream
`numpy.linspace(0, audiosize-max_audio, num=num_eval)` (eval) and
`numpy.int64(random.random()*(audiosize-max_audio))` (train) have a
strictly positive range argument. Removing it would change the
random-crop semantics for exactly-`max_audio`-length clips; that's a
behaviour change orthogonal to BUGFIX-007 and is not part of this fix.

### 2.4 Choice of dither amplitude (`dither_std = 1e-4`)
| Reference level | Approximate dBFS |
|---|---|
| Maximum-amplitude speech sample (`±1.0` in float32) | 0 dBFS |
| Natural recording silence floor (studio condenser) | ≈ −60 to −70 dBFS |
| 16-bit PCM quantisation floor | ≈ −96 dBFS |
| Our dither (`std = 1e-4`) | **≈ −80 dBFS** |

`−80 dBFS` is below natural recording silence and well above the
16-bit floor. Audibly indistinguishable from genuine silence; just
non-zero enough that `numpy.mean(audio**2)` over a frame is never
exactly zero.

### 2.5 Choice of one-time warning over a recurring warning
Eight DataLoader workers × hundreds of short clips per epoch ×
`print` per occurrence → log floods. A single line per process is
enough to surface the corpus-quality issue at the first encounter.
For deeper triage the user runs `check_corrupted_audio.py` (already
in the repo) or writes a one-off `duration_audit.py` —the warning's
job is just to notify, not to enumerate.

If anyone *does* want every occurrence logged for triage, the
`_short_clip_warned` global can be reset to `False` mid-run, or the
warning relocated. Out of scope for this fix.

---

## 3. Behaviour matrix

| Scenario | Pre-fix | Post-fix |
|---|---|---|
| Clip ≥ `max_audio` (the common VoxCeleb case) | Unchanged path; no padding | Bit-identical — `if audiosize <= max_audio` is False, helpers never called |
| Clip < `max_audio`, training mode | Wrapped onto itself; periodic comb in spectrum; random crop reduces to fixed `[0:max_audio]` | Silence pad with dither; same fixed-crop reduction (preserves training behaviour for short clips); model sees real audio followed by silence — no shortcut periodicity |
| Clip < `max_audio`, eval mode | Linspaced start frames over wrap-padded audio | Linspaced start frames over silence-padded audio; eval EER unchanged on long clips, more honest on short clips |
| MUSAN noise file shorter than max_audio (rare) | Wrapped onto itself; injects periodic noise into augmented audio | Silence-padded; noise becomes "real noise + silence" — closer to deployment reality |
| First short clip seen by a worker | (silent) | One stderr warning per worker process |

### 3.1 What this fix does NOT change
- Random-crop diversity for very short clips. With audio padded only
  to `max_audio + 1`, the crop start is constrained to `[0, 1)` and
  rounds to `0`. Same as pre-fix. If meaningful crop diversity is
  desired for short clips, the right answer is to **drop them at
  corpus-prep time**, not pad them to a longer length and lie about
  available content. This is what the warning is pointing at.
- Numerical determinism for runs with the same seed. The dither is
  drawn from the worker-seeded NumPy RNG; identical seeds → identical
  dither.

---

## 4. Verification

### 4.1 Static
Both touched files compile cleanly:

```bash
$ python3 -m py_compile DatasetLoader.py DatasetLoader_performance_updated.py
$ echo $?
0
```

### 4.2 Static — call-site audit
Zero remaining `numpy.pad(..., 'wrap')` calls. The only `'wrap'`
references left are inside the helper's docstring, explaining the
pre-fix behaviour for future readers:

```bash
$ grep -nE "numpy\.pad.*['\"]wrap['\"]" DatasetLoader.py DatasetLoader_performance_updated.py
# (no output)
```

### 4.3 Functional — helper sanity (run in `2025_colvaai`)
```python
import numpy
from DatasetLoader import _pad_short_with_dither

# Case 1: short audio gets padded with silence+dither
a = numpy.ones(100, dtype=numpy.float32)
b = _pad_short_with_dither(a, 32241)
assert b.shape[0] == 32241
assert (b[:100] == 1.0).all(), "first 100 samples must be the original audio"
pad = b[100:]
assert 0 < pad.std() < 1e-3, "dither std should be tiny but nonzero"
assert abs(pad).max() < 5e-4, "dither amplitude well below recording noise floor"
assert b.dtype == numpy.float32, "dtype preserved"

# Case 2: already long enough → no-op
c = _pad_short_with_dither(a, 50)
assert c.shape[0] == 100
print("helper functional check OK")
```

### 4.4 Functional — warning fires once
```python
from DatasetLoader import _maybe_warn_short_clip
# First call prints; subsequent calls are silent.
_maybe_warn_short_clip("/tmp/short1.wav", 100, 32241)   # → prints to stderr
_maybe_warn_short_clip("/tmp/short2.wav", 200, 32241)   # → silent
```

### 4.5 Functional — periodicity is gone
A simple FFT probe on a pre-fix vs post-fix padded signal makes the
fix visible:

```python
import numpy, soundfile
audio, sr = soundfile.read("some_short_clip.wav")           # < 2 s of speech

# Pre-fix (do NOT use)
old = numpy.pad(audio, (0, 32241 - len(audio) + 1), 'wrap')

# Post-fix
from DatasetLoader import _pad_short_with_dither
new = _pad_short_with_dither(audio.astype(numpy.float32), 32241 + 1)

# Look at the spectrum's variance over time:
# - pre-fix has comb structure at f = k / T_audio for integer k
# - post-fix shows the original speech spectrum followed by a flat low-level noise floor
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].specgram(old, NFFT=512, Fs=sr); ax[0].set_title("'wrap' pad")
ax[1].specgram(new, NFFT=512, Fs=sr); ax[1].set_title("silence + dither")
plt.show()
```

Expected: the right-hand spectrogram shows real speech followed by a
near-uniform low-energy band; the left-hand spectrogram shows real
speech followed by the original spectrum tiled — visibly periodic.

---

## 5. Backward compatibility

| Consumer | Effect |
|---|---|
| Every existing config (VoxCeleb 16 kHz, clips ≥ 2 s) | ✅ Bit-identical — the `if audiosize <= max_audio` branch is never taken for typical VoxCeleb clips, so the helper is not invoked. |
| Configs with `max_frames` large enough to trigger padding on average VoxCeleb clips | ⚠️ Numerics change. Pre-fix runs were already shortcut-learning the wrap periodicity; post-fix runs see honest silence pads. **Pre-fix checkpoints on such configs are suspect; retrain.** |
| Mini-VoxCeleb1 (140 speakers) experiments documented in `research_logs/` | ✅ Mini-VoxCeleb1 clips are typical VoxCeleb duration. No change. |
| Sri Lankan corpus with short clips (the actual SL use-case) | ✅ Strict improvement — pre-fix would have silently shortcut-learned the wrap periodicity; post-fix surfaces the issue at the first short clip via stderr and produces honest training data. |
| MUSAN noise augmentation | ✅ When a MUSAN file is shorter than `max_audio`, it now pads with silence instead of being tiled. Cleaner augmentation; effectively a slight reduction in noise duration. Negligible quantitatively. |

### 5.1 Existing checkpoints
Models trained pre-fix on a corpus where padding never fired (i.e., all
clips ≥ `max_audio`): unaffected. The internal frontend is identical.

Models trained pre-fix on a corpus where padding *did* fire (i.e., a
significant fraction of short clips): probably shortcut-learning. The
fix doesn't migrate them automatically; the signal that a checkpoint
might be tainted is "EER was good on the source corpus but cratered on
a cross-corpus evaluation." If retraining is unavailable, the
mitigation is to evaluate the existing checkpoint *only* on a test set
whose clips are uniformly ≥ `max_audio`.

---

## 6. Rollback

Replace each helper call with the original three-line `shortage / pad('wrap')`
block; delete the helper definitions and the `_short_clip_warned` global
from both loader files. There is no scenario in which rollback is correct;
the pre-fix code's silent shortcut-learning was a real correctness
problem.

---

## 7. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #6 (`numpy.pad(..., 'wrap')`) | ✅ Closed by this document |
| §4.1 #1–#5 | ✅ BUGFIX-001..005 |
| BUGFIX-005 / 006 follow-ups | ✅ BUGFIX-006 |
| §4.1 #7 — SincConv buffer placement on every forward | ✅ [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) |
| §4.1 #8 — `MaxPool1d(...) if pool else False` returns literal `False` | ✅ [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md) |
| §4.1 #9 — NestedSpeakerNet NaN cascades | ✅ [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) (quarantined to `models/experimental/`) |
| Future polish: real energy-based VAD instead of one-time stderr warning | ⬜ Open |
| Future polish: corpus-prep tool to filter clips shorter than `max_audio` | ⬜ Open |

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #6.
- **Shortcut-learning literature:** Geirhos, R., Jacobsen, J.-H., et al.
  *Shortcut learning in deep neural networks*, Nat. Mach. Intell. 2 (2020).
  https://www.nature.com/articles/s42256-020-00257-z
- **`numpy.pad` mode semantics:** https://numpy.org/doc/stable/reference/generated/numpy.pad.html
- **Upstream provenance:** the `'wrap'` pad is inherited verbatim from the
  Clova AI parent repo. Worth a separate upstream report — this is a real,
  silent training-time bug that affects any corpus with short clips.
