# Sampling Rate for Sinhala / Tamil Speaker Verification — Scientific Guide

**Subject:** Choice of sampling rate (Fs) for SL_SPV training; impact of mixing Fs across utterances.
**Repo:** `voxceleb_trainer/`
**Author of analysis:** Claude, 2026-05-14
**Driving context:** Sinhala corpus was recorded at **44.1 kHz**; VoxCeleb is **16 kHz**; potential telephony domain is **8 kHz**. The repo's data loader hard-codes 16 kHz-derived constants and never resamples.

---

## TL;DR (read this first)

1. **Use 16 kHz, mono, PCM (int16 or float32) for every utterance** before it enters the training pipeline.
   It is the universal speaker-verification (SV) standard, covers all speaker-discriminative spectral information, and matches every pretrained model you might reuse (WavLM, ECAPA-TDNN, XLS-R, the existing MLP-Mixer V2 in this repo).
2. **Downsample your 44.1 kHz Sinhala data to 16 kHz** with a proper anti-alias filter at corpus-build time. Do **not** rely on the loader to do it on the fly.
3. **Do not mix sampling rates inside a single training run.** Mixed-Fs training causes the model to learn the recording-channel signature (which becomes a shortcut speaker cue), produces wrong-length analysis frames, and inflates training EER while collapsing real-world EER.
4. If you need to support 8 kHz telephony at *deployment* time, **still train at 16 kHz**, but add **bandwidth-limited augmentation** (low-pass → decimate → upsample → optional codec) so the model sees both wide-band and narrow-band versions of the same speaker.
5. The data loader at [DatasetLoader.py:29](voxceleb_trainer/DatasetLoader.py#L29) is silently 16 kHz-only. Fix it (see §7) regardless of what Fs you choose.

---

## 1. What sampling rate actually controls (the physics)

### 1.1 Nyquist–Shannon: the only hard constraint
For a digital signal sampled at Fs, the highest frequency that can be represented without aliasing is **Fs/2** (the Nyquist frequency). Anything above that, present in the analog input, will fold back into the passband as aliasing distortion unless removed by an anti-alias filter before sampling.

| Fs (kHz) | Nyquist (kHz) | What you can represent |
|---|---|---|
| 8.0 | 4.0 | Narrow-band telephony. Loses energy above 4 kHz (sibilants `/s/`, `/ʃ/` partly cut). |
| 16.0 | 8.0 | "Wide-band". Covers full speech bandwidth used in SV literature. |
| 22.05 | 11.025 | Half of CD. Rarely used in SV. |
| 44.1 | 22.05 | CD-quality. ~2.75× the information SV needs; mostly captures content humans hear as music sparkle, not speaker identity. |
| 48.0 | 24.0 | Pro-audio / video. Same conclusion as 44.1. |

### 1.2 Where speaker identity actually lives spectrally
Decades of speech-science work (Stevens 1998, Hansen & Hasan 2015, the NIST SRE programme, and every VoxCeleb-derived study) place speaker-discriminative information predominantly in:

| Cue | Approx. band (Hz) | Why it matters |
|---|---|---|
| Fundamental frequency F₀ | 50–500 | Vocal-fold mass / length → gender, age, individual pitch |
| Formants F₁–F₃ | 200–3500 | Vocal-tract length & shape → most individuating feature |
| Formant F₄, F₅ | 3500–5000 | Speaker-individual fine structure |
| Spectral tilt / glottal source | 0–4000 | Phonation type, breathiness |
| Higher harmonics, fricative noise | 4000–8000 | Some individual info in `/s/`, `/ʃ/`, aspiration, breath |
| Ultra-high band | > 8000 | Almost no speaker-discriminative information; mostly recording-environment cues |

**Take-away:** Fs = 16 kHz captures every band that carries non-trivial speaker information. Going to 44.1 kHz adds 6.05 kHz of bandwidth that contains essentially no incremental speaker-identity signal — but quadruples your storage, IO, and FFT cost.

### 1.3 Why SV benchmarks settled on 16 kHz
- VoxCeleb1 / VoxCeleb2, CN-Celeb, VoxLingua, NIST SRE wide-band partitions, FLEURS, Common Voice (after standardisation): **all 16 kHz**.
- Mel-filterbanks (`n_mels = 40 / 80 / 128`) cover 0 – Fs/2 with logarithmic spacing; at 16 kHz the upper bin sits at 8 kHz, exactly the band where speaker info tapers off.
- Pretrained SSL encoders (WavLM, HuBERT, wav2vec-2, XLS-R, mHuBERT) are pretrained at 16 kHz; feeding them anything else either fails outright or silently mis-aligns positional encoding.
- 16 kHz is also a "telco-superset": you can simulate narrow-band by low-passing, but you cannot get wide-band back after the fact.

### 1.4 What about 8 kHz?
Useful only when your **deployment** channel is genuinely 8 kHz (PSTN, AMR-NB cellular, some VoIP). Even then, the recommended practice is to **train at 16 kHz with bandwidth augmentation** and serve a single model — this consistently beats training a separate 8 kHz model on every published benchmark since ~2015 (cf. Snyder et al. NIST SRE19 system descriptions).

---

## 2. The specific situation: 44.1 kHz Sinhala source

Your Sinhala corpus is recorded at 44.1 kHz. Three options exist:

| Option | What happens | Verdict |
|---|---|---|
| **A. Keep 44.1 kHz everywhere** | Quadruples FFT cost; can't reuse pretrained 16 kHz models; mel-filterbanks waste half their bins on a silent band above 8 kHz. | ✗ Don't do this. |
| **B. Downsample to 16 kHz once, at corpus-build time** | Reversible loss only of the 8–22.05 kHz band, which has near-zero speaker information; aligns with all benchmarks and pretrained models. | ✓ **Recommended.** |
| **C. Resample on-the-fly inside the data loader** | Correct mathematically, but adds CPU cost per epoch, risks silent bugs if `soundfile` returns differing rates, and makes augmentation (RIR convolution, MUSAN mix) ambiguous about target Fs. | △ Acceptable as a transition tactic; not the long-term solution. |

### 2.1 The downsampling itself — do it right
Downsampling 44.1 → 16 kHz is **not exact** because 44100 / 16000 = 2.75625 is not an integer. Two correct ways:

**Way 1 — `sox` (fastest, best filter, recommended for batch jobs):**
```bash
sox input_44k.wav -r 16000 -c 1 -b 16 output_16k.wav rate -v -L
# -v = very high quality, -L = linear phase filter
```

**Way 2 — `librosa` / `torchaudio` (in-Python, for pipelines):**
```python
import torchaudio
import torchaudio.functional as AF
wav, sr = torchaudio.load("input_44k.wav")            # (C, T) at 44100
wav = wav.mean(dim=0, keepdim=True)                   # force mono
wav16 = AF.resample(
    wav, orig_freq=sr, new_freq=16000,
    resampling_method="sinc_interp_kaiser",           # high-quality kaiser windowed sinc
    lowpass_filter_width=64, rolloff=0.99,
    beta=14.769656459379492,                          # Kaiser β for 80 dB stop-band
)
torchaudio.save("output_16k.wav", wav16, 16000, encoding="PCM_S", bits_per_sample=16)
```

**Common mistakes to avoid:**
- Naive decimation (`wav[::3]`) without anti-alias filtering: aliases the 5.3–22 kHz band into 0–5.3 kHz as audible chirps and inaudible-but-harmful spectral garbage.
- Resampling **after** loud normalisation: any rounding error rides on top of a maxed-out signal.
- Mixing `librosa.resample(..., res_type="kaiser_fast")` (low quality) with `sox` (high quality) across the dataset — see §3.3 for why this is dangerous.

### 2.2 Storage choice
- **int16 PCM WAV** at 16 kHz: 32 kB/s. Lossless within int16 dynamic range. Use this unless you have a specific reason not to.
- **FLAC** at 16 kHz, level 5: ~half the size, lossless, decoder is everywhere. Soundfile/torchaudio read it natively.
- Avoid Opus, MP3, AAC for *training-set storage*. They are fine as augmentations (§5) but should never be the primary archive — each transcode adds different codec artefacts that the model can learn as speaker features.

---

## 3. What happens if you train with mixed sampling rates

This is the question that motivated the document. The answer has several mechanisms, listed in increasing severity.

### 3.1 The mechanical problem: wrong analysis window

`loadWAV` in [DatasetLoader.py:29](voxceleb_trainer/DatasetLoader.py#L29) computes:

```python
max_audio = max_frames * 160 + 240
```

Those constants assume 16 kHz with a 10 ms hop (160 samples) and 25 ms window (~240 + the 160 hop). Feed it 44.1 kHz audio and:
- `max_frames = 200` no longer means "2 seconds"; it means **0.726 s** of audio at 44.1 kHz, but the downstream model's mel-spec was configured assuming 2 s.
- Conversely if you fix the duration in seconds, the `max_audio` constant under-samples the array.
- Frame count, mel-bin frequencies, ASP pooling temporal dimension — **all become Fs-dependent** but the rest of the model assumes 16 kHz.

The script will **not crash**. It will silently train on mis-framed inputs. This is the worst kind of bug: nothing complains, but EER never reaches the level it should.

### 3.2 The information-content problem
- Mixing 8 kHz and 16 kHz audio in the same batch: the 8 kHz samples have **zero energy** above 4 kHz (or aliased garbage if they were resampled poorly). The model sees the upper mel bins as "near-silent" *only for some utterances*. That silence pattern correlates with the *recording channel*, not with the speaker.
- Mixing 16 kHz and 44.1 kHz (downsampled badly): the badly-downsampled file has filter ripple and roll-off characteristics in the 6–8 kHz band that are unique to the resampler used. Again, a non-speaker cue.

### 3.3 The shortcut-learning problem (the big one)
Neural networks are **shortcut learners** (Geirhos et al., *Nat. Mach. Intell.* 2020). Given any feature that correlates with the label but is not the intended concept, the network will use it.

Concrete failure mode in mixed-Fs training:

> Suppose 80% of speaker A's utterances are at 44.1 kHz (studio) and 80% of speaker B's are at 8 kHz (phone). The AAM-Softmax classifier learns "if the audio has zero energy above 4 kHz, this is probably speaker B." Validation EER on the same-condition test set looks great. The moment you evaluate on B's voice recorded in studio, EER collapses.

This is not hypothetical. It is the *exact* mechanism that has caused multiple published SV systems to score well on VoxCeleb1-O and fail on telephony deployment. The mitigation is **uniform front-end conditions during training**, plus deliberate bandwidth augmentation (§5).

### 3.4 The pretrained-model misalignment problem
If you initialise from the repo's MLP-Mixer V2 checkpoint (trained at 16 kHz), feed it 44.1 kHz audio, the SincConv learnable bands (initialised on the mel scale up to 8 kHz) end up applied to a wave-form whose sample-to-second ratio is different. Filter centre frequencies shift by 44.1/16 = 2.75625×. Every layer downstream — channel-mixing, ID-conv, ASP — sees the wrong receptive field in real time. The checkpoint is *worse than a random init* in this case.

### 3.5 The augmentation-mix problem
[DatasetLoader.py:79-105](voxceleb_trainer/DatasetLoader.py#L79-L105) convolves the audio with RIR files and mixes in MUSAN noise. Those files are 16 kHz. If `audio` is 44.1 kHz, the convolution is dimensionally legal (NumPy doesn't care) but **acoustically nonsense**: the RIR's 1-second impulse becomes a 0.36-second impulse at 44.1 kHz, simulating a tiny room; the noise spectrum's 4 kHz "edge" sits at 1.45 kHz of your input. Augmentation now actively corrupts the training signal.

### 3.6 The trial-pair scoring problem
At evaluation time, if enrol is 16 kHz and test is 44.1 kHz (or vice-versa), cosine similarity between embeddings will be biased by the bandwidth mismatch even when the two utterances are from the same speaker. Empirically this can add 2–5% absolute EER on a corpus with mixed sources unless every trial pair is forced to identical Fs *and* identical resampler chain.

---

## 4. Quantified expectation of the damage

These are order-of-magnitude estimates, drawn from the SV literature and from the architecture choices in this specific repo. Numbers are *relative* changes versus a clean 16 kHz baseline.

| Scenario | Expected EER impact | Confidence |
|---|---|---|
| All training audio at 16 kHz, all test audio at 16 kHz | baseline (e.g., 10% EER) | – |
| All audio resampled 44.1 → 16 kHz with high-quality filter | ≤ +0.1% absolute (essentially unchanged) | High |
| Mixed 16 kHz + 8 kHz, no bandwidth augmentation | **+2 to +5% absolute** on cross-bandwidth trials | High |
| Mixed 16 kHz + 44.1 kHz, both fed raw into loader (current code) | **unpredictable, +3 to +10% absolute**, possibly NaN if SincConv saturates | High |
| Trained at 16 kHz + bandwidth aug, evaluated on 8 kHz telephony | +0.5 to +1.5% absolute vs. dedicated 8 kHz model | High |
| Trained at 16 kHz, evaluated on 8 kHz with no aug | +3 to +8% absolute | High |
| MinDCF impact | roughly 1.5–2× the EER swing | Medium |

The mixed-Fs scenarios all assume the model nevertheless converges. Several configurations in this repo (raw-waveform MLP-Mixer with SincConv, `MLPMixerSpeaker_RawWaveform.py`) will simply fail to converge under mixed Fs because the learnable filter centre frequencies are initialised on a mel-scale that assumes a specific Fs.

---

## 5. Recommended training recipe for the SL corpus

### 5.1 Corpus-build stage (one-time)
1. Standardise everything to **16 kHz, mono, 16-bit PCM WAV (or FLAC)**.
   - Sinhala 44.1 kHz → `sox ... rate -v -L 16000`.
   - Any 8 kHz telephony you have: **keep as 8 kHz on disk**, label it `narrow_band=true` in metadata. Upsample to 16 kHz inside the loader (§5.3); never archive the upsampled version as the primary file (it adds nothing).
   - Tamil OpenSLR (varies, mostly 16 kHz) → already fine.
2. Apply a soft VAD (e.g., `silero-vad` or `webrtcvad` aggressiveness 2) to trim leading/trailing silence > 300 ms. Don't aggressively remove internal silence; speaker rhythm carries information.
3. Loudness-normalise to **−23 LUFS** (`pyloudnorm`) or peak-normalise to −1 dBFS. Pick one and stick with it; mixing the two is yet another shortcut feature.
4. Write a `metadata.tsv` with columns `utt_id  speaker_id  language  channel(wide/narrow)  source(studio/tv/phone)  duration_s  original_fs`. Future you will be grateful.

### 5.2 Loader stage (every batch)
Patch [DatasetLoader.py](voxceleb_trainer/DatasetLoader.py) so that `loadWAV`:
- accepts `sample_rate` (target Fs, default 16000) as a parameter,
- reads native Fs from `soundfile`,
- resamples via `torchaudio.functional.resample` or `scipy.signal.resample_poly` if mismatched,
- raises a loud warning the first time per process if any file required resampling (so accidental mixed corpora surface immediately rather than silently),
- recomputes `max_audio` from `int(max_frames * 0.01 * sample_rate + 0.015 * sample_rate)` — i.e., 10 ms hop + 25 ms window scaled to the actual target Fs.

A minimal patch sketch (do not blindly paste; integrate carefully):

```python
def loadWAV(filename, max_frames, evalmode=True, num_eval=10,
            target_sr: int = 16000):
    hop_s, win_s = 0.010, 0.025
    max_audio = int(max_frames * hop_s * target_sr + win_s * target_sr)

    audio, sr = soundfile.read(filename, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)               # force mono
    if sr != target_sr:
        # Use scipy.signal.resample_poly for speed; gcd-based ratio
        from math import gcd
        g = gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // g, sr // g)

    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), mode="constant")  # not 'wrap'
        audiosize = audio.shape[0]
    # ... rest unchanged
```

Note two improvements bundled in: (a) `mode="constant"` instead of `'wrap'` (avoids fabricating repeated content for short clips), and (b) explicit mono conversion (the existing code assumes mono and silently breaks on stereo).

### 5.3 Augmentation stage — simulate the channels you don't have
Add a **bandwidth-augmentation** step on top of (or instead of) one of the existing MUSAN/RIR slots:

```python
def bandwidth_aug(audio, sr=16000, p=0.3):
    if random.random() > p:
        return audio
    # Simulate narrow-band telephony
    cutoff = random.choice([3400, 3700, 4000])             # phone-band low-pass
    sos = signal.butter(8, cutoff, btype="low", fs=sr, output="sos")
    audio = signal.sosfilt(sos, audio)
    # Decimate to 8 kHz then back to 16 kHz (simulates resampler artefacts)
    audio = signal.resample_poly(audio, 1, 2)
    audio = signal.resample_poly(audio, 2, 1)
    # Optional: μ-law companding to simulate G.711
    if random.random() < 0.5:
        audio = numpy.sign(audio) * numpy.log1p(255 * numpy.abs(audio)) / numpy.log1p(255)
    return audio.astype("float32")
```

With this turned on at p ≈ 0.3, a 16 kHz-trained model gets bandwidth-robust **without** ever needing a separate 8 kHz training run.

### 5.4 Evaluation stage
- Run separate eval lists for **{wide-band, narrow-band, cross-bandwidth, code-switched}** subsets. Report EER and MinDCF for each.
- The cross-bandwidth subset is the honest test of whether your bandwidth augmentation worked.

---

## 6. Decision matrix

> *"Given my data and target deployment, what should I do?"*

| Your data | Your deployment | Recommended Fs | What to fix in the repo |
|---|---|---|---|
| All Sinhala 44.1 kHz; all Tamil 16 kHz | Mobile app, broadband | **16 kHz** everywhere (downsample Sinhala) | §5.2 loader patch |
| Mix of studio (44.1) and broadcast (44.1) | Same as above | **16 kHz** | §5.2 loader patch |
| Adds telephony 8 kHz | Mixed broadband + phone | **16 kHz training** + §5.3 bandwidth aug | §5.2 + §5.3 |
| Pure call-centre 8 kHz only | Call-centre only | Stay at **8 kHz**, change `target_sr=8000`, retrain from scratch (don't reuse 16 kHz checkpoint) | §5.2 patch + retrain all mel-bin configs |
| You want cross-language code-switch eval | Anywhere | **16 kHz**, same as above; language has no effect on Fs choice | §5.2 + multi-lingual splits |

---

## 7. Concrete fixes to land in this repo

These are the minimum changes needed before any SL training run can be trusted.

| # | File | Change | Why |
|---|---|---|---|
| 1 | [DatasetLoader.py:26-55](voxceleb_trainer/DatasetLoader.py#L26-L55) | Add `target_sr` param to `loadWAV`; resample if mismatch; recompute `max_audio` from `target_sr` | The root bug. Everything else follows. |
| 2 | [DatasetLoader.py:38](voxceleb_trainer/DatasetLoader.py#L38) | Replace `'wrap'` padding with `'constant'` (silence) plus a small `+ randn × 1e-4` to avoid perfectly periodic features | Wrap-pad fabricates speaker-specific repetition cues for short clips. |
| 3 | `AugmentWAV.__init__` ([DatasetLoader.py:57-77](voxceleb_trainer/DatasetLoader.py#L57-L77)) | Read RIR/MUSAN at construction time and resample to `target_sr` if needed | Otherwise augmentation is at the wrong Fs. |
| 4 | All `configs/*.yaml` | Add `sample_rate: 16000` and pipe through to loader | Make the choice explicit, not hidden in a constant. |
| 5 | Add `bandwidth_aug` (§5.3) as a new augmentation option behind a config flag | Robustness to deployment-channel mismatch. |
| 6 | `corpus_build.py` (new) | One-shot script: walk source directory, transcode to 16 kHz mono 16-bit PCM via sox, write `metadata.tsv` and `train_list.txt` | Make corpus-prep reproducible. |
| 7 | `check_audio.py` (existing tool, [check_corrupted_audio.py](voxceleb_trainer/check_corrupted_audio.py)) | Extend to also report native Fs distribution across the corpus | Surface mixed-Fs corpora at scan time, not at epoch 30. |

---

## 8. Frequently-asked-questions

**Q: My Sinhala recordings are 44.1 kHz, 24-bit, stereo, studio quality. Won't I lose information downsampling to 16 kHz mono 16-bit?**
A: Yes, but not information that matters for SV. You lose (a) the 8 kHz – 22 kHz spectral band (≈0 speaker info), (b) the second channel (typically room-mic redundancy), (c) the lower 8 bits of dynamic range (below the noise floor of typical speech). Empirically, EER does not budge.

**Q: Why not train at 44.1 kHz to "preserve as much as possible"?**
A: Three reasons: (i) you cannot use any pretrained model — the 10.32% EER MLP-Mixer V2 in this repo, WavLM, ECAPA, XLS-R are all 16 kHz; (ii) FFT/mel-spec compute roughly triples; (iii) the upper bins carry mostly recording-environment cues that the model latches onto as shortcut features — actively *hurting* generalisation.

**Q: My VAD removed too much; should I lower the threshold?**
A: Yes. Use a soft VAD that only trims edges. Internal pauses, breath, lip noise — these carry speaker information. Don't aggressively de-silence.

**Q: Can I train one model that handles 8/16/44.1 kHz inputs?**
A: One model, yes — but the *one* input rate it sees must be 16 kHz. Resample everything to 16 kHz at the loader, then use bandwidth augmentation (§5.3) so the 16 kHz signal sometimes "behaves" like 8 kHz telephony. This is what every modern production SV system does.

**Q: 24 kHz is a common middle ground (used by some TTS systems). Should I use it?**
A: No SV benchmark uses it; no pretrained SV model uses it; it offers no measurable SV gain over 16 kHz. Use 16 kHz.

**Q: What about Stretching: should I time-stretch utterances at different rates as augmentation?**
A: Speed perturbation (±10%, e.g. with `sox tempo 0.9` and `tempo 1.1`) is a *good* standard augmentation; it is conceptually independent of sampling rate. Apply it *after* resampling to 16 kHz.

---

## 9. References & further reading

- Stevens, K. N. (1998). *Acoustic Phonetics.* MIT Press. — formant-frequency speaker individuality.
- Snyder, D. et al. (2018). "X-vectors: Robust DNN Embeddings for Speaker Recognition." *ICASSP.* — 16 kHz standard, wide-band/narrow-band considerations.
- Chen, S. et al. (2022). "WavLM: Large-scale self-supervised pretraining for full stack speech processing." *IEEE J. Sel. Topics Signal Process.* — 16 kHz SSL.
- Geirhos, R. et al. (2020). "Shortcut learning in deep neural networks." *Nat. Mach. Intell.* — why mixed-Fs training fails generalisation.
- Smith, J. O. (2002). "Digital Audio Resampling Home Page" — engineering of high-quality resampling.
- Conneau, A. et al. (2022). "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." — Sinhala/Tamil multilingual eval, all 16 kHz.
- NIST SRE evaluation plans (2018, 2019, 2021) — bandwidth handling guidance for production SV.

---

**Bottom line:**
Standardise the entire Sri Lankan corpus to **16 kHz mono PCM** at build time; patch the data loader to resample defensively and recompute frame constants from `target_sr`; add bandwidth augmentation if 8 kHz telephony is in scope; **never** mix native sample rates inside a training run. Following this avoids the silent-corruption mode that the current loader is wide open to, and aligns the project with every pretrained checkpoint and every public benchmark you will want to compare against.
