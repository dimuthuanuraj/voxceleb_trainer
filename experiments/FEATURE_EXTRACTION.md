# What should the network actually be fed?

Front-end options for Sinhala/Tamil speaker verification — what this project
already uses, what the alternatives are, and which of them Stage F measures.

---

## 1. What VoiceID actually does today

Answering the direct question first, because the two stacks in this project do
**not** use the same design and nothing had measured the difference.

`SL_SPV/voiceid/backend/app/speaker_model.py` (`SSLSpeakerNet`) is:

```
waveform (16 kHz)
  → WavLM-base-plus            94M params, 13 hidden states (CNN output + 12 transformer layers)
  → learned layer weighting    w = softmax(θ) ∈ R¹³,  h = Σ_l w_l · H_l
  → ASP pooling                attentive mean + std over time
  → Linear → 192-d embedding
  → AAM-Softmax head           (training only; discarded at inference)
```

So: **not** a two-stage "extract embeddings, then train a separate classifier".
It is one network trained end to end, in which WavLM is the *front end* — it
replaces the mel-spectrogram block, and the speaker embedding is the pooled
output, not WavLM's output. The checkpoint is `p3_full_s42/best.pt`, from the
P3 PEFT work, and its encoder was **fully fine-tuned** (full FT beat LoRA:
1.69/1.41 vs 3.73/3.10 EER).

**No Whisper anywhere.** Whisper models are in the HF cache
(`openai/whisper-small`, `whisper-large-v3`, `Systran/faster-whisper-*`) but
those belong to the transcription project (`SLT_Zoom_Project`), not to speaker
verification. Section 4 explains why that is the right call.

### The gap this study closes

The research stack's `models/SSLFrontendSpeaker.py` reads **one** layer
(`--ssl_layer`, default `-1` = last). VoiceID learns a weighting over **all**
layers. That is a real architectural difference, in the direction that matters,
and no experiment in this repository had ever compared them. Stage F does, via
the new `models/SSLFrontendSpeakerLW.py`, which ports the production design into
the trainer.

---

## 2. The four families

| family | what it is | trainable? | dim/frame | relative cost |
|---|---|---|---|---|
| **Cepstral** (MFCC) | DCT of the log-mel spectrum | no | 40–80 | ~1× |
| **Filterbank** (log-mel) | mel-scaled spectrogram, log-compressed | no | 80 | ~1× |
| **Learned filterbank** (SincConv) | band-pass filters with learned cut-offs | yes | 80–256 | ~2× |
| **Self-supervised** (WavLM, mHuBERT, XLS-R) | pretrained transformer features | pretrained, optionally fine-tuned | 768–1024 | 4–20× |

### 2.1 MFCC — the classical choice, and why it is now the wrong one

MFCCs apply a DCT to the log-mel spectrum:

```
c_n = Σ_{k=1}^{M} log(E_k) · cos( n(k − ½)π / M ),    n = 0 … N−1
```

The DCT approximately **decorrelates** the filterbank channels. That property
was essential for GMM-UBM and i-vector systems: they used diagonal covariance
matrices, so correlated inputs violated their core modelling assumption.

A CNN or TDNN has no such constraint, and the situation inverts:

* Convolution **wants** correlations between neighbouring frequency bands —
  that is where formant structure lives, and a local filter is precisely a
  device for exploiting it. The DCT scatters that locality across coefficients.
* Truncating to N < M coefficients keeps the smooth spectral envelope and
  discards fine spectral detail — including much of the glottal-source and
  vocal-tract detail that distinguishes speakers rather than phonemes.

**Prediction: MFCC loses to log-mel.** Stage F runs it anyway, in two
configurations (`mfcc80` static, and the classical `mfcc40d` with Δ + ΔΔ),
because "we don't use MFCCs because the literature moved on" is an assertion,
while "MFCC costs us X.X pp EER on Sinhala, 95% CI [a, b]" is a result — and it
is the one a reviewer or examiner will ask for.

### 2.2 Log-mel — the right default

80-bin log-mel is the baseline here and the standard input for ECAPA-TDNN,
ResNet-SE and every other mel-based backbone in `models/`. Mel spacing is
approximately logarithmic above 1 kHz, which matches both auditory resolution
and the way vocal-tract resonances scale.

One caveat worth stating plainly, because it bears on this project's research
question: **the mel scale was fitted to perceptual judgements from English and
other European-language listeners.** Whether its warping is optimal for Sinhala
and Tamil is not established. That is a genuine open question — it is what a
learned filterbank would answer.

> **Note.** The learned-filterbank arm (RawNet3) was removed from the study on
> request. `models/RawNet3.py` and its architecture spec remain in the
> repository, so restoring it is a one-line change to `ARCHITECTURES` in
> `experiments/registry.py`. Until then the mel-scale question stays open, and
> the honest position is that this study does not test it.

### 2.3 Self-supervised features — the strong option, and its real cost

An SSL encoder replaces the hand-designed front end with representations
learned from large unlabelled multilingual audio. Two things decide whether it
helps here, and they are separable:

**(a) Which layer you read.** This matters more than most people expect, and it
is where the research stack's default is actively bad.

SSL encoders are trained by masked prediction: recover a masked frame from its
context. That objective rewards **phonetic and lexical** information in the
upper layers — and speaker identity is *nuisance* for it. Speaker information
therefore concentrates in the **lower and middle** layers, near the
convolutional feature extractor. Taking the last layer — `--ssl_layer -1`, the
default — is close to the worst available choice for a speaker task: it is the
layer the pretraining objective worked hardest to make speaker-invariant.

Stage F tests three points on this axis:

| run | what it reads |
|---|---|
| `ssl_wavlm_last` | layer 12 (the current default) |
| `ssl_wavlm_mid` | layer 6, a fixed mid-stack choice |
| `ssl_wavlm_lw` | learned softmax weighting over all 13 states — the VoiceID design |

The fitted weights are themselves a result. `SSLFrontendSpeakerLW.layer_weights_()`
returns the distribution, and its argmax is a direct quantitative statement
about *where speaker information lives in this encoder for Sinhala and Tamil* —
comparable against the same encoder's profile on English.

**(b) Which encoder — coverage vs capacity.** These are easy to confound, so
the study separates them deliberately:

| encoder | params | pretraining languages | role |
|---|---|---|---|
| WavLM-base-plus | 94M | English only | control |
| **mHuBERT-147** | **94M** | **147, incl. si + ta** | **the clean contrast** |
| XLS-R-300m | 300M | 128, incl. si + ta | confounded — bigger *and* multilingual |

mHuBERT-147 is the informative one: **the same parameter count as WavLM**, so a
difference between them is attributable to pretraining coverage rather than to
capacity. If the multilingual encoder gains more on Tamil than on English, that
is direct evidence that coverage — not architecture — drives the transfer.
XLS-R is available via `tools/fetch_ssl_encoder.py` but should be read alongside
mHuBERT, never instead of it.

**The cost is real.** WavLM-base-plus is 94.98M parameters and 22.1 GFLOPs per
2 s of audio, against ECAPA-1024's 14.46M and 5.17 GFLOPs — **6.6× the
parameters, 4.3× the compute**, and measured at ~101 s/epoch against ECAPA's
~29 s in the smoke runs. For the VoiceID product, where CPU inference latency
is ~270 ms today, that is a deployment decision and not only an accuracy one.
The Stage F table reports EER against both parameters and GFLOPs so the
trade-off is visible rather than implicit.

---

## 3. Recommendation

1. **Keep log-mel as the default** for mel-based backbones. It is the right
   input for convolutional models and it is cheap.
2. **Fix the SSL layer question first.** Of everything on this list, moving from
   last-layer to learned-layer-weighted features is the change most likely to
   pay, and it costs 13 extra scalar parameters. It also aligns the research
   stack with what VoiceID already deploys, which is worth having for its own
   sake.
3. **Test mHuBERT-147 against WavLM before assuming multilingual pretraining
   helps.** Same size, so the comparison is clean. This is a cheap experiment
   with a genuinely uncertain outcome.
4. **Run MFCC once** to get the number, then stop using it.
5. **Do not use Whisper for embeddings** — see §4.

---

## 4. Why not Whisper

Whisper is an encoder-decoder **ASR** model trained on 680k hours of supervised
transcription. It is a strong feature extractor for *what was said*. Speaker
verification asks *who said it*, and these objectives pull in opposite
directions:

* ASR training explicitly rewards **speaker invariance**. A good transcriber
  maps the same words from different voices onto the same output — speaker
  identity is exactly the nuisance variable it is trained to discard. Masked
  prediction (WavLM, HuBERT) has no such incentive and leaves far more speaker
  information intact, particularly in the lower layers.
* Whisper's encoder consumes fixed 30-second windows, padding shorter input.
  Speaker verification operates on 2–8 s segments, so most of every forward pass
  would be padding — wasteful, and a distribution mismatch against the
  pretraining condition.
* It is large: whisper-large-v3 is ~1.55B parameters, ~16× WavLM-base-plus.

The literature that does use Whisper for speaker tasks generally takes *early*
encoder layers, and reports it trailing WavLM-style SSL models at equal size.
Given limited GPU budget, mHuBERT-147 is the better use of a run: same
parameter count as the existing WavLM baseline, explicitly covers Sinhala and
Tamil, and tests a hypothesis that is actually open.

**If you still want it measured**, the cheapest defensible version is a Whisper
encoder in `SSLFrontendSpeakerLW` with learned layer weights (`whisper-small`,
244M) on `si` only — one run, one number, and the layer weights would show
whether any usable speaker information survives and at what depth. Say the word
and it goes in as a Stage F entry; it is deliberately excluded for now because
the prior is poor and the run is not cheap.

---

## 5. Stage F — the front-end sweep

Backbone, loss, splits, augmentation and schedule are all fixed; only the input
representation changes.

| run | family | what it isolates |
|---|---|---|
| `mel80` *(reused from Stage A)* | filterbank | baseline |
| `mfcc80` | cepstral | the DCT alone — same channel count as mel80 |
| `mfcc40d` | cepstral | the classical 40 + Δ + ΔΔ configuration |
| `ssl_wavlm_last` *(reused from Stage A)* | SSL, single layer | the trainer's current default |
| `ssl_wavlm_mid` | SSL, single layer | layer depth (layer 6 vs 12) |
| `ssl_wavlm_lw` | SSL, layer-weighted | the VoiceID design vs single-layer |
| `ssl_mhubert_lw` | SSL, layer-weighted | pretraining coverage at equal capacity |

Two entries are **reused from Stage A rather than re-run**: they are already
exactly the configurations Stage F would launch, and the paired bootstrap
compares the *same* trained system, so retraining would only add seed noise.

```bash
python experiments/tools/fetch_ssl_encoder.py --all     # encoders as safetensors
python experiments/tools/gen_scripts.py --stage F
python experiments/tools/run_queue.py    --stage F
```

---

## 6. Stage G — the techniques already implemented here

The repo carries a set of universal feature flags (FEATURE-002…008) that were
previously exercised only by the closed-set `configs/sl_feature_*.yaml` recipes.
Stage G re-runs them on the **speaker-disjoint** splits, one technique per run,
so each contribution is separable.

| technique | flags | requires | hypothesis |
|---|---|---|---|
| `baseline` | — | — | the reference row |
| `plda` | `--plda --plda_dim 200` | PLDA train list | models within/between-speaker covariance explicitly; should help most where data is thinnest (ta) |
| `finetune_en` | `--finetune --finetune_lr_multiplier 0.1` | English checkpoint | do 5,871 English speakers buy more than 336 Sinhala ones? |
| `finetune_en_llrd` | `+ --llrd --llrd_decay 0.9` | English checkpoint | layer-wise decay should beat plain FT when the source/target gap is large |
| `lang_aux` | `--lang_aux --lang_aux_weight 0.3` | combined only | make the embedding language-**aware** |
| `dann_lang` | `--dann_lang --dann_lang_lambda 0.1` | combined only | make the embedding language-**invariant** |

`lang_aux` and `dann_lang` encode **opposite** hypotheses, and running both is
how that question gets settled rather than argued. Techniques whose
requirements are unmet are skipped, not run degraded — a language-adversarial
head on a monolingual corpus would train happily and mean nothing.

**AS-Norm is deliberately not here.** It is an inference-time score transform,
already measured as an explicit on/off factor in `tools/evaluate.py`, where it
is attributable on its own rather than confounded with a training effect.

```bash
python experiments/tools/gen_scripts.py --stage G \
    --initial-model exps/E_ecapa1024_aamsoftmax_en_full_s42/model/model_best.model
python experiments/tools/run_queue.py --stage G
```

---

## 7. English, and why there are two English conditions

Comparing architecture rankings across languages only means something if data
volume is held constant. English gets two conditions because one cannot do both
jobs:

| condition | speakers | train | test | job |
|---|---|---|---|---|
| `en_full` | 5,871 | 1.07M utts | VoxCeleb1-O (canonical) | **positive control** |
| `en_matched` | **336** | 47.9k utts / ~107 h | 100 held-out speakers | **fair cross-language comparison** |

**`en_full`** keeps the canonical VoxCeleb1-O trial list, so its EER is directly
comparable with published numbers. If ECAPA lands near the ~1 % this recipe is
known to give, the whole harness — loader, augmentation, scoring, checkpoint
selection — is validated end to end. If it does not, every Sinhala and Tamil
number is suspect. That is its real job, and it is also the checkpoint Stage G's
fine-tune techniques consume.

**`en_matched`** subsamples VoxCeleb2 to **exactly** the Sinhala speaker count
(336). Without it, "ECAPA ranks differently on Sinhala than on English" would
just be measuring how each architecture scales from 336 to 5,871 speakers.

Matching is on **speakers (exact)** and **total speech hours (approximate)**,
not utterance count: VoxCeleb utterances average 8.04 s against slr52's 4.38 s,
so equal counts would mean unequal audio, and duration is what bounds how much
distinct speech a speaker contributes under random-crop sampling.

**One honest limitation.** VoxCeleb2 speakers average ~182 utterances, so 336 of
them top out near 137 h against slr52's 157 h — `en_matched` reaches ~107 h,
**68 %** of the reference. Adding speakers would fix the hours but break the
exact speaker-count match, which is the more important half of the control.
The shortfall is recorded in `experiments/splits/en_matched/manifest.json`
(`hours_match_pct`, `hours_ceiling_for_this_speaker_count`). **Quote the
architecture *ranking* across conditions, not the absolute EER gap** — the
ranking is what the paired bootstrap supports and what survives this mismatch.

```bash
python experiments/tools/build_en_splits.py
python experiments/tools/gen_scripts.py --stage E --conditions en_matched,en_full
python experiments/tools/run_queue.py --stage E
```

---

## 8. Summary of what was added

| file | why |
|---|---|
| `models/ECAPA_TDNN_MFCC.py` | 80 static MFCCs — isolates the DCT |
| `models/ECAPA_TDNN_MFCC40D.py` | classical 40 + Δ + ΔΔ |
| `models/SSLFrontendSpeakerLW.py` | learned layer weighting — ports the VoiceID design into the trainer |
| `models/weights/mhubert-147/` | 147-language encoder, same size as WavLM |
| `experiments/tools/fetch_ssl_encoder.py` | materialises encoders as safetensors (torch 2.5.1 cannot load `.bin`) |
| `experiments/tools/build_en_splits.py` | the two English conditions |

`trainSpeakerNet.py` remains untouched. Every new model is a separate module,
because `n_mfcc`, `mfcc_deltas` and `channels` are not registered argparse
options and the YAML loader discards unknown keys — the same constraint that
produced `ECAPA_TDNN_C512`.
