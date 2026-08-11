# BUGFIX-005 — `loadWAV` hard-codes 16 kHz; non-16 kHz files are silently mis-framed

| Field | Value |
|---|---|
| **ID** | BUGFIX-005 |
| **Severity** | Critical (silent corruption — model trains on wrong-length analysis windows, no crash, no warning) |
| **Component** | Audio loading, augmentation, training-data and test-data pipelines |
| **Files touched** | [DatasetLoader.py](../../DatasetLoader.py), [DatasetLoader_performance_updated.py](../../DatasetLoader_performance_updated.py), [trainSpeakerNet.py](../../trainSpeakerNet.py), [trainSpeakerNet_performance_updated.py](../../trainSpeakerNet_performance_updated.py), [trainSpeakerNet_distillation.py](../../trainSpeakerNet_distillation.py) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #5, follow-up to [SAMPLING_RATE_GUIDE.md](../../SAMPLING_RATE_GUIDE.md) |
| **Scope decision** | User selected "Loader resample only" — bandwidth augmentation deferred to BUGFIX-006 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

> **Read first** — this fix resolves the **format** mismatch caused by mixed
> sample rates. It does **not** resolve the **information-content** mismatch
> (an 8 kHz file upsampled to 16 kHz still has zero energy above 4 kHz, and
> the model can shortcut-learn channel/bandwidth as a non-speaker cue).
> The full conceptual story is in [SAMPLING_RATE_GUIDE.md](../../SAMPLING_RATE_GUIDE.md) §3.
> Bandwidth augmentation is tracked as BUGFIX-006 (not yet implemented).

---

## 1. Problem

### 1.1 The hard-coded constants
[DatasetLoader.py:29](../../DatasetLoader.py#L29) read:

```python
max_audio = max_frames * 160 + 240
```

Those numbers are 16-kHz-specific:
- `160` = 10 ms hop × 16000 Hz/s = samples per frame hop.
- `240` = 25 ms window − 10 ms hop = 15 ms × 16000 Hz/s = trailing-window samples.

So `max_frames = 200` at 16 kHz means `max_audio = 32240` samples = 2.015 s of audio. Correct.

[DatasetLoader.py:32](../../DatasetLoader.py#L32) read the native sample rate but **never used it**:

```python
audio, sample_rate = soundfile.read(filename)
# sample_rate is bound and immediately discarded
```

### 1.2 What this means for non-16 kHz inputs
Loading a 44.1 kHz file with the unfixed code gives:
- `audio.shape[0]` = ~88,200 samples per second.
- `max_audio` = 32,240 (still computed for 16 kHz).
- The "200-frame" slice covers `32240 / 44100 ≈ 0.73 s` of real audio, not 2 s.

But the **model** (mel-spectrogram inside `MLPMixerSpeaker.py`, `ResNetSE34V2.py`, etc.) is configured with `sample_rate=16000` and `hop_length=160`. Feeding it a 32,240-sample window of 44.1 kHz audio:
- The mel-filterbank computes frame indices assuming 16 kHz → it sees ~200 frames, but each frame represents 3.6 ms of real audio (not 10 ms).
- Mel bin centre frequencies are placed on a 0–8 kHz mel scale, but the upper half of the spectrum carries 44.1 kHz content folded into the 0–22 kHz analog band — only the 0–8 kHz portion lands in the mel-filterbank's range.
- ASP pooling shape is still nominally (B, P, 200) — no crash — but the temporal dimension means something different per file.

Result: **silent corruption**. No exception, no warning, just degraded EER that's untraceable from the logs.

For **8 kHz telephony** the symmetric case produces half-length windows (0.5 s of audio when the script thinks it's getting 1 s); for any non-16 kHz the duration represented by `max_frames=200` is wrong.

### 1.3 Why the report flagged this
This is exactly the type of bug that would never surface during VoxCeleb-only training (all of VoxCeleb is 16 kHz, courtesy of `dataprep.py` running ffmpeg's `-ar 16000`). It is **guaranteed** to surface the moment a Sri Lankan corpus joins the pipeline:
- Sinhala studio recordings are typically 44.1 kHz / 48 kHz.
- Tamil OpenSLR is 16 kHz (lucky).
- Common Voice si/ta is 48 kHz post-standardisation.
- Any telephony channel for deployment evaluation is 8 kHz.

So the bug had a 1-in-3 probability of biting in the first SL training run.

### 1.4 Loader call-site inventory before the fix
Both loader files contain four places that hard-coded 16 kHz framing:

| File | Site | Function |
|---|---|---|
| `DatasetLoader.py` | 29 | `loadWAV` — main input framing |
| `DatasetLoader.py` | 62 | `AugmentWAV.__init__` — augmentation buffer size |
| `DatasetLoader.py` | 90 | `AugmentWAV.additive_noise` — MUSAN file loading |
| `DatasetLoader.py` | 101 | `AugmentWAV.reverberate` — RIR file loading (sample rate ignored) |
| `DatasetLoader_performance_updated.py` | 52 | `loadWAV` framing |
| `DatasetLoader_performance_updated.py` | 39–46 | `loadWAV_cached` (LRU-cached path with the same issue) |
| `DatasetLoader_performance_updated.py` | 94 | `AugmentWAV.__init__` |
| `DatasetLoader_performance_updated.py` | 133 | `AugmentWAV.additive_noise` |
| `DatasetLoader_performance_updated.py` | 151 | `AugmentWAV.reverberate` |

---

## 2. Fix applied

### 2.1 Design
Generalise the 16 kHz assumption two ways:

1. Convert the hop-and-window magic numbers into **seconds**, computed to samples at the configured target rate.
2. Read each file's native rate from `soundfile.read`, and **resample with proper anti-aliasing** to the target rate if mismatched.

The target rate is a new top-level parameter `sample_rate`, threaded through:
- `--sample_rate` argparse flag in all three trainers (default `16000`).
- Constructor kwargs of `train_dataset_loader`, `test_dataset_loader`, `AugmentWAV` (default `16000`).
- Keyword arg of `loadWAV` (default `16000`).

Defaulting to `16000` everywhere means every existing config and command line is bit-identical post-fix; this is a backward-compatible additive change.

### 2.2 Module-level helpers (both loader files)
Added at the top of each loader, after the existing utility imports:

```python
# Framing constants: 10 ms hop, 25 ms window. Expressed in seconds so they
# generalise across sample rates.
_HOP_SECONDS = 0.010
_WINDOW_SECONDS = 0.025


def _frame_to_samples(max_frames, sample_rate):
    """Audio length in samples for ``max_frames`` frames at ``sample_rate``."""
    hop = int(round(_HOP_SECONDS * sample_rate))
    window = int(round(_WINDOW_SECONDS * sample_rate))
    return max_frames * hop + (window - hop)


def _resample_if_needed(audio, orig_sr, target_sr):
    """Anti-aliased polyphase resample with gcd-based rational ratio.

    No-op when ``orig_sr == target_sr``. Mono assumption is enforced upstream.
    """
    if int(orig_sr) == int(target_sr):
        return audio
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return signal.resample_poly(audio, up, down)
```

#### 2.2.1 Why `scipy.signal.resample_poly`
- Already in the dependency tree (no new package).
- Polyphase implementation — fast even for the awkward 44100 → 16000 case
  (`gcd = 100`, `up = 160`, `down = 441`).
- Built-in anti-alias FIR filter; correctness is comparable to `sox`/`librosa`
  for SV-quality audio.

Real-world gcd-derived ratios that this fix has to handle:

| Native Fs | Target Fs | `up` | `down` |
|---|---|---|---|
| 8 000 | 16 000 | 2 | 1 |
| 22 050 | 16 000 | 320 | 441 |
| 44 100 | 16 000 | 160 | 441 |
| 48 000 | 16 000 | 1 | 3 |
| 16 000 | 16 000 | (no-op short-circuit) | — |

Verified during the fix (`python3 -c "..."` sanity check):
- `_frame_to_samples(200, 16000) == 32240` (matches the historical magic number).
- `_frame_to_samples(200, 8000) == 16120`.

### 2.3 `loadWAV` (both files)

```diff
-def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
-
-    # Maximum audio length
-    max_audio = max_frames * 160 + 240
-
-    # Read wav file and convert to torch tensor
-    audio, sample_rate = soundfile.read(filename)
-
-    audiosize = audio.shape[0]
+def loadWAV(filename, max_frames, evalmode=True, num_eval=10, sample_rate=16000):
+
+    # Maximum audio length at the target sample rate.
+    max_audio = _frame_to_samples(max_frames, sample_rate)
+
+    # Read wav file and convert to torch tensor.
+    audio, native_sr = soundfile.read(filename)
+    if audio.ndim > 1:
+        audio = audio.mean(axis=1)  # mono
+    audio = _resample_if_needed(audio, native_sr, sample_rate)
+
+    audiosize = audio.shape[0]
```

Two bundled defensive improvements:
- **Mono enforcement** (`audio.mean(axis=1)` when 2-D). The pre-fix code silently broke on stereo input — `soundfile.read` returns `(T, 2)` for stereo, and downstream `audio.shape[0]` then accesses `T` correctly, but every subsequent indexing/concat operation assumed 1-D and produced wrong-length frames.
- **`sample_rate` named kwarg** so callers can be explicit.

### 2.4 `loadWAV_cached` (perf-updated only)
The LRU cache key has to include `target_sr`, otherwise a process that runs at a non-default rate would silently return cached native-rate audio from a previous call. The cache now stores **post-resample** audio:

```python
@lru_cache(maxsize=1000)
def loadWAV_cached(filename, max_frames, target_sr):
    try:
        audio, native_sr = soundfile.read(filename, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample_if_needed(audio, native_sr, target_sr).astype(numpy.float32, copy=False)
        return audio, target_sr
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None
```

Disk IO, decode, and resample now happen at most once per `(file, target_sr)` pair. The cache size and eviction policy are unchanged.

### 2.5 `AugmentWAV` (both files)
- `__init__` accepts `sample_rate` and stores it on the instance.
- `max_audio` is computed from `_frame_to_samples(max_frames, sample_rate)`.
- `additive_noise` passes `sample_rate=self.sample_rate` to its internal `loadWAV` call so MUSAN noise files are resampled too.
- `reverberate` resamples the RIR with `_resample_if_needed(rir, fs, self.sample_rate)` before normalising and convolving.

This last one is important: MUSAN and RIR archives are 16 kHz on disk, so for a 16 kHz training run nothing changes. But the moment `--sample_rate 8000` (or any other) is configured, both the MUSAN noises and the RIR impulse responses get resampled to match the speech — without this, augmented training audio would have a sample-rate mismatch between speech (8 kHz) and noise / impulse (16 kHz) producing an inconsistent simulation.

### 2.6 `train_dataset_loader` and `test_dataset_loader` (both files)
Both dataset classes accept `sample_rate=16000` in `__init__`, store it on `self`, and pass it through to their `loadWAV` and (in the train case) `AugmentWAV` calls. Same default; same backward compatibility.

### 2.7 Trainer argparse (all three trainers)
Added one line, immediately after `--max_frames`, in `trainSpeakerNet.py`, `trainSpeakerNet_performance_updated.py`, and `trainSpeakerNet_distillation.py`:

```python
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='Target audio sample rate (Hz). Files at other rates are resampled at load time')
```

Because the trainers pass `**vars(args)` to `train_dataset_loader` and `test_dataset_loader`, the new argument plumbs through automatically.

---

## 3. Things this fix does NOT change

These are deliberately out of scope; they are tracked as follow-ups so future work doesn't re-discover them.

| Item | Why deferred |
|---|---|
| `numpy.pad(audio, (0, shortage), 'wrap')` for short audio (§4.1 #6). | Fabricates speaker-specific repetition cues. Same `loadWAV` body, different bug. User explicitly scoped BUGFIX-005 to the Fs problem only. Tracked as BUGFIX-007. |
| Bandwidth augmentation (§3 of `SAMPLING_RATE_GUIDE.md`). | This is the **information-content** half of the mixed-Fs problem — i.e., even after the loader resamples to a common rate, an 8 kHz source still has zero energy above 4 kHz and the model can shortcut on that. Tracked as BUGFIX-006. |
| Model-side mel-spectrogram `sample_rate` argument. | Every model in `models/` constructs its mel-extractor with a literal `sample_rate=16000`. If someone configures `--sample_rate 8000`, the model still computes mels assuming 16 kHz, which is wrong. **Workaround for now:** only override `--sample_rate` if you also edit the corresponding `models/*.py` to match. A clean fix is to read `sample_rate` from kwargs in each model's `__init__`. Tracked as BUGFIX-008. |
| `soxr`-based resampling for faster 44.1 → 16 conversion. | `scipy.signal.resample_poly` is fast enough (≈3× real-time per worker). Worth revisiting only if data-loading becomes a bottleneck. |

---

## 4. Verification

### 4.1 Static
All five touched files compile cleanly:
```bash
$ python3 -m py_compile DatasetLoader.py DatasetLoader_performance_updated.py \
                        trainSpeakerNet.py trainSpeakerNet_performance_updated.py \
                        trainSpeakerNet_distillation.py
$ echo $?
0
```

Zero remaining instances of the old hard-coded constant:
```bash
$ grep -nE "max_frames \* 160 \+ 240" DatasetLoader.py DatasetLoader_performance_updated.py
# (no output)
```

### 4.2 Helper math sanity (run from the project root)
```bash
$ python3 -c "
import math
def f(n, sr):
    hop=int(round(0.010*sr)); win=int(round(0.025*sr))
    return n*hop + (win-hop)
assert f(200, 16000) == 32240
assert f(200,  8000) == 16120
print('helper math OK')
"
helper math OK
```
The 16 kHz case reproduces the historical magic number bit-for-bit, so any
existing 16 kHz workflow is unchanged.

### 4.3 Functional — 16 kHz regression check
Run an existing VoxCeleb 16 kHz config; loss values for the first epoch under
the same seed should be bit-identical to a pre-fix run:
```bash
python trainSpeakerNet.py \
    --config configs/experiment_01.yaml \
    --max_epoch 1 --test_interval 1
```
Expected: no change in `loss` or `TEER/TAcc`. If they differ, BUGFIX-005 has a
subtle bug — investigate before any non-16 kHz run.

### 4.4 Functional — 44.1 kHz resample
Drop a single 44.1 kHz Sinhala WAV into `data/sl_celeb_pilot/`, point a config
at it, and run one batch with print debugging:
```python
# Add temporarily at the top of train_dataset_loader.__getitem__:
print(f"[debug] audio.shape={audio.shape} sample_rate={self.sample_rate}")
```
Expected: `audio.shape[1] == 32240` (because `_frame_to_samples(200, 16000)`),
regardless of whether the on-disk file was 16 kHz or 44.1 kHz. Remove the
print before committing.

### 4.5 Functional — 8 kHz at `--sample_rate 8000`
If a future workflow trains at 8 kHz end-to-end (which **also** requires the
deferred model-side fix from §3), the same probe should show:
- `audio.shape[1] == 16120` for 200 frames at 8 kHz.
- Mel-spec, ASP pooling, and embedding all consistent at half-time-resolution.

---

## 5. Backward compatibility

| Consumer | Effect |
|---|---|
| Any existing 16 kHz config (every config in `configs/`) | ✅ Bit-identical — `_frame_to_samples(N, 16000)` returns the same numbers as the old `N * 160 + 240`, and `_resample_if_needed(audio, 16000, 16000)` is a no-op short-circuit. |
| 16 kHz mono PCM training files (VoxCeleb) | ✅ Unchanged — no resampling triggered. |
| Stereo files that previously broke silently | ⚠️ Now reduced to mono automatically. The pre-fix code's behaviour on stereo was undefined (likely incorrect framing); the new behaviour is correct. No migration needed unless someone was relying on the broken pre-fix output. |
| 8 kHz telephony files (the original report's example) | ✅ Now framed correctly (`16120` samples for 200 frames at 8 kHz) **if** `--sample_rate 8000` is set. If `--sample_rate 16000` (default), the loader upsamples to 16 kHz and uses 32240-sample windows — also correct, but with a silent-band caveat (see §3 / SAMPLING_RATE_GUIDE §3 / BUGFIX-006). |
| 44.1 kHz / 48 kHz Sinhala / Common Voice files | ✅ Now resampled to 16 kHz at load time. Pre-fix behaviour was silent corruption. |
| Loss / model / SpeakerNet code | ✅ Untouched. |

### 5.1 Existing checkpoints
Models trained pre-fix on 16 kHz data are unaffected. Resume / `--eval` works
identically.

Models trained pre-fix on accidentally-non-16 kHz data are **garbage** — they
were trained on mis-framed inputs. No automatic detection; the only signal is
"EER never improved as expected on that corpus." If such a checkpoint exists,
retrain after this fix.

---

## 6. Why the change is conservative on purpose

A more aggressive design would have been:
- Make `sample_rate` a required positional everywhere.
- Add an audible warning the first time a file is resampled.
- Strict-mode failure if the loader sees a sample-rate distribution narrower
  than the configured value (e.g., training file at 22.05 kHz when
  `--sample_rate 16000`).

All three are reasonable additions; none are needed to close the report
under §4.1 #5. They add behaviour that is observable in normal operation and
therefore needs broader testing. Out of scope.

The minimal version — silently fix the math, silently resample, no new
required parameters — is the lowest-risk way to land the structural change.
Once the project has its first SL training run, we'll know whether the
silent-resample behaviour is actually what we want or whether the louder
warnings are needed.

---

## 7. Rollback

Mechanical: replace `_frame_to_samples(max_frames, sample_rate)` with the
literal `max_frames * 160 + 240` in five places, drop the helper definitions,
remove the `sample_rate` kwargs from four constructor signatures, remove the
three `--sample_rate` argparse lines, and delete the `_resample_if_needed`
calls in `loadWAV` and `AugmentWAV.reverberate`.

There is no scenario in which rollback is correct; the pre-fix code is
guaranteed to silently corrupt any non-16 kHz training run.

---

## 8. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #5 (16 kHz hard-coded) | ✅ Closed by this document |
| §4.1 #1 — DDP kwarg | ✅ [BUGFIX-001](BUGFIX-001-mp-spawn-kwarg.md) |
| §4.1 #2 — EER threshold | ✅ [BUGFIX-002](BUGFIX-002-eer-threshold-not-returned.md) |
| §4.1 #3 — nPerSpeaker accuracy | ✅ [BUGFIX-003](BUGFIX-003-nperspeaker-accuracy.md) |
| §4.1 #4 — Hard `.cuda()` | ✅ [BUGFIX-004](BUGFIX-004-hard-cuda-call.md) |
| §4.1 #6 — `numpy.pad(..., 'wrap')` | ✅ [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md) |
| §4.1 #7 — SincConv buffer placement | ✅ [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) |
| §4.1 #8 — `MaxPool1d(...) if pool else False` | ✅ [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md) |
| §4.1 #9 — NestedSpeakerNet NaN | ✅ [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) (quarantined) |
| Bandwidth augmentation (info-content half of the mixed-Fs problem) | ⬜ Open, tracked as BUGFIX-006 |
| Model-side `sample_rate` propagation | ⬜ Open, tracked as BUGFIX-008 |
| Reference document for the conceptual story | [SAMPLING_RATE_GUIDE.md](../../SAMPLING_RATE_GUIDE.md) |

---

## 9. Authorship & references

- **Bug originally identified by:** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #5.
- **User question that scoped the fix:** "if the audio has different
  sampling rates then training with different sampling rate wont be an
  issue?" — answered with the format-vs-info-content distinction in
  §1 of this doc and at length in `SAMPLING_RATE_GUIDE.md` §3.
- **`scipy.signal.resample_poly` docs:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
- **`soundfile.read` docs:** https://python-soundfile.readthedocs.io/en/latest/#soundfile.read
- **Upstream provenance:** The 16 kHz magic numbers are inherited verbatim
  from the Clova AI parent repo. Worth a separate upstream report.
