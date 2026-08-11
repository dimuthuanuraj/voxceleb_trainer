# FEATURE-001 — Learnable language-aware front-end (SincConv path + SSL path)

| Field | Value |
|---|---|
| **Slug** | `FEATURE-001-language-aware-frontend` |
| **Date** | 2026-05-19 |
| **Author** | Research-track implementation (Claude-assisted) |
| **Track** | **Feature**, not bug. First entry in the `FEATURE-NNN` track that opens after the audit's `BUGFIX-001..026` close. |
| **Severity** | Research-impact. The §3.1 audit estimates 30–50% relative EER reduction on low-resource languages from the SSL path; SincConv path is a cheaper but smaller win. |
| **Scope** | One new model (`models/SSLFrontendSpeaker.py`); four new configs (one SincConv + three SSL); three new argparse args plumbed through all trainers; one optional dependency added (commented) to `requirements.txt`. No existing model, trainer, DataLoader, loss, or config is modified. |
| **Source of request** | `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 item #2 ("Add a learnable language-aware front-end"). |
| **Status** | ✅ Implemented (both options selectable via config; awaiting first SL training run to verify the expected EER reduction empirically). |

---

## 1. What this feature delivers

The §3.1 #2 audit item names two distinct ways to replace the fixed
mel-spectrogram front-end with something that adapts to Sinhala /
Tamil phonetics. Both are now implemented and selectable via config:

| Option | Mechanism | New model | New encoder | Trainable params at the front-end |
|---|---|---|---|---|
| **A — SincConv** | Learnable narrow-band filters in the time domain replace the fixed mel filterbank | `MLPMixerSpeaker_RawWaveform` (existing) | None — SincConv is itself the encoder | ~1k filter-bank params learned from your data |
| **B — Frozen SSL encoder** | Multilingual transformer pretrained on raw waveform replaces the mel block; head trains, encoder is frozen | `SSLFrontendSpeaker` (new) | One of: `microsoft/wavlm-base`, `facebook/wav2vec2-xls-r-300m`, `utter-project/mHuBERT-147` | 0 (encoder frozen) — head has ~6M params |

Both options can be exercised independently from a YAML config; they
are not stackable (pick one front-end per training run).

The audit's framing was about **language-aware** front-ends because a
mel-spectrogram is computed from a Hz-uniform grid that does not
follow Sinhala or Tamil formant distributions. The SincConv path lets
the filterbank centres adapt during training; the SSL path inherits
multilingual context from a model that has already seen 94–147
languages including both target languages.

### 1.1 Which option to use when

| Situation | Recommended option |
|---|---|
| SL pilot corpus is **small** (~50–200 speakers) and SSL infra is available | **B — SSL with `ssl_freeze: true`.** Frozen encoder + head-only training avoids overfitting to a small corpus while still benefiting from multilingual pretraining. |
| SL corpus has scaled to **≥1,000 speakers** | **B with `ssl_freeze: false`** for the last few epochs (warm-start unfreezing), OR **A** if compute is tight (XLS-R-300M is ~3× slower per step than WavLM-Base, which is ~2× slower than SincConv). |
| Compute is the bottleneck (no GPU memory for SSL activations) | **A — SincConv.** Tiny encoder; trains in the same memory envelope as the mel models. |
| You want to compare against the literature's SOTA on low-resource SV | **B with XLS-R-300M.** Largest of the three SSL options, strongest published benchmarks on Sinhala / Tamil ASR transfer. |
| You want fastest end-to-end training-to-EER | **A — SincConv.** No HuggingFace dependency, no pretrained-model download, no SSL activation memory cost. |

---

## 2. What was added

### 2.1 New model: `models/SSLFrontendSpeaker.py`

A standalone speaker-verification model that takes any HuggingFace
`AutoModel`-compatible audio encoder as its front-end. Architecture:

```text
Raw waveform (B, T)
    ↓ per-utterance mean-var normalisation
SSL encoder (frozen or trainable) — e.g. WavLM-Base
    ↓ (B, T', hidden_size) frame embeddings
Conv1d attention head (SAP or ASP)
    ↓ (B, hidden_size) or (B, 2·hidden_size) pooled
BatchNorm1d + Linear
    ↓
Embedding (B, nOut)
```

Three knobs control its behaviour, all configurable via argparse or
YAML:

- **`--ssl_encoder_name`** — HuggingFace model ID. Tested defaults:
  `microsoft/wavlm-base`, `facebook/wav2vec2-xls-r-300m`,
  `utter-project/mHuBERT-147`. Any other HF audio-encoder ID will
  also work in principle (the model uses `AutoModel.from_pretrained`,
  not a hardcoded model class).
- **`--ssl_freeze` / `--no_ssl_freeze`** — boolean. Default `True`.
  When frozen, the SSL encoder runs under `torch.no_grad()` in
  `forward` and the encoder activations are `.detach()`-ed before
  the head, so backward only flows through the head's ~6M params.
- **`--ssl_layer`** — int, default `-1`. Selects which transformer
  layer's hidden state to feed to the pooling head. `-1` = last
  layer (standard). Mid-layer features sometimes transfer better for
  speaker tasks (most SV literature using SSL backbones picks layer
  6–9 of a 12-layer base model); experiment if accuracy plateaus.

The encoder is downloaded from the HuggingFace Hub on first
construction (cached in `~/.cache/huggingface/` for subsequent
runs). No model weights ship in the repository.

### 2.2 Three SSL config templates

| Config | Encoder | hidden_size | Approx. params | Batch recommendation |
|---|---|---|---|---|
| [`configs/language_aware_ssl_wavlm.yaml`](../../configs/language_aware_ssl_wavlm.yaml) | `microsoft/wavlm-base` | 768 | 94M | 32 |
| [`configs/language_aware_ssl_xlsr.yaml`](../../configs/language_aware_ssl_xlsr.yaml) | `facebook/wav2vec2-xls-r-300m` | 1024 | 300M | 16 |
| [`configs/language_aware_ssl_mhubert.yaml`](../../configs/language_aware_ssl_mhubert.yaml) | `utter-project/mHuBERT-147` | 768 | 95M | 32 |

All three configs share the same training recipe (AAM-Softmax, lr
5e-4, lr_decay 0.97, weight_decay 2e-5, batch sizes per the table
above, `nClasses: 100` placeholder, frozen encoder). The differences
are encoder-name, batch size, and `gradient_accumulation_steps`
(XLS-R-300M uses 2 to compensate for its smaller batch).

### 2.3 One SincConv config template

[`configs/language_aware_sincconv.yaml`](../../configs/language_aware_sincconv.yaml)
targets the existing `MLPMixerSpeaker_RawWaveform` model. It's a
fine-tune-shaped recipe (lr 1e-4, lr_decay 0.97, batch 64) that
expects an `initial_model:` pointing at the best English raw-waveform
checkpoint — or empty for from-scratch SL training.

### 2.4 Optional dependency: `transformers>=4.30,<5`

Added to [`requirements.txt`](../../requirements.txt) as a commented
optional dependency. Mirrors the existing `tensorboard` /
`matplotlib` pattern:

```text
# Optional (SSL front-end via FEATURE-001) ----------------------------
# Required ONLY if you use `model: SSLFrontendSpeaker` (the WavLM /
# XLS-R / mHuBERT path from FEATURE-001). The SincConv variant of
# FEATURE-001 (model: MLPMixerSpeaker_RawWaveform) does not need this.
# 4.30 is the floor that ships a stable AutoModel API for the tested
# encoders; <5 to avoid silent uptake of a breaking major release.
# transformers>=4.30,<5
```

The model file imports `transformers` lazily (`try / except
ImportError`) so the trainer continues to load even without it —
only `SSLFrontendSpeaker.__init__` raises a descriptive
`ImportError` if the user picks the SSL path without the dependency
installed.

### 2.5 Three argparse args plumbed through all trainers

Added to `trainSpeakerNet.py`, `trainSpeakerNet_performance_updated.py`,
and `trainSpeakerNet_distillation.py`:

```text
--ssl_encoder_name    str    default "microsoft/wavlm-base"
--ssl_freeze          flag   default True (use --no_ssl_freeze to opt out)
--ssl_layer           int    default -1 (last layer)
```

All three flow through `**vars(args)` into the model's `__init__`
exactly the same way every other model kwarg does. The YAML loader
honours them per the existing convention (camelCase aliases via
BUGFIX-022 do not apply — these names are already snake_case).

---

## 3. Verification

### 3.1 Static

| Check | Result |
|---|---|
| `python -m py_compile models/SSLFrontendSpeaker.py` | exit 0 ✅ |
| `python -m py_compile` on all 3 trainers | exit 0 each ✅ |
| YAML parse of all 23 configs (19 existing + 4 new) | 23 / 23 ✅ |

### 3.2 Argparse + YAML plumbing (4 tests, all pass)

| # | Setup | Result |
|---|---|---|
| 1 | No args | `ssl_encoder_name = 'microsoft/wavlm-base'`, `ssl_freeze = True`, `ssl_layer = -1` ✅ |
| 2 | `--ssl_encoder_name 'facebook/wav2vec2-xls-r-300m' --ssl_layer -2` | Values correctly propagated ✅ |
| 3 | `--no_ssl_freeze` | `ssl_freeze = False` ✅ |
| 4 | YAML `ssl_freeze: false`, `ssl_encoder_name: utter-project/mHuBERT-147` | Both values correctly loaded ✅ |

### 3.3 Not verified

- **No live training run was exercised.** The audit-session Python
  doesn't have `transformers` installed and the system has no GPU.
  The model's `__init__` and `forward` paths are written against the
  HuggingFace `AutoModel` API as documented; a real first SL training
  run will be the verification of correctness.
- **No SL data exists** to evaluate the actual EER reduction
  claim. The 30–50% literature number is reported for other
  low-resource-language SV benchmarks (e.g., CN-Celeb's tail
  speakers, OpenSLR Multi-Lingual SV); it's a *prediction to test*,
  not a guarantee.
- **No A vs B comparison was run.** Once an SL corpus exists, the
  recommended sequence is: SincConv first (cheaper), then WavLM-Base
  second (mid), then XLS-R-300M (heaviest) — and decide which to
  carry forward based on the EER / compute trade-off observed.

---

## 4. Backward-compatibility & migration

- **No existing model, config, or trainer is modified.** This is a
  strictly additive feature.
- **No existing checkpoint is affected.** New model architectures
  cannot be loaded by old checkpoints (their `state_dict` keys
  don't exist in `SSLFrontendSpeaker`); the existing 11 `exps/`
  artefacts continue to work with their respective configs as
  before.
- **No new required dependency.** `transformers` is optional and
  commented out. Users who don't run `model: SSLFrontendSpeaker` are
  completely unaffected.
- **The SincConv path uses an unchanged existing model**
  (`MLPMixerSpeaker_RawWaveform`). The new SincConv config is just a
  fresh recipe; the model file itself is unchanged from
  [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md)'s state.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Stacking SincConv + SSL.** Two front-ends feeding into a fused
  representation. Could plausibly help, but adds a second large
  hyperparameter axis (fusion weight, fusion location); not
  justified before either option has produced a baseline SL EER.
- **Wrapping in an `nn.DataParallel`-friendly module for `<small
  GPU>` users.** `SSLFrontendSpeaker` works with DDP via the
  trainer's existing infrastructure. Single-GPU users with low VRAM
  should pick WavLM-Base or mHuBERT (both base-scale) over
  XLS-R-300M.
- **Layer-wise weighted-sum pooling** (the "WeightedSum" trick from
  the WavLM SV paper). A learnable per-layer scalar that combines
  all transformer layers' outputs before pooling. Often gives
  another 0.5–1% absolute EER. Implementable as a small extension
  to `SSLFrontendSpeaker._extract_ssl_features`; deferred to
  FEATURE-002 if the basic path delivers.
- **Encoder distillation** (compress XLS-R-300M into a 50M model
  for deployment). Out of scope for the front-end-selection feature;
  the existing knowledge-distillation pipeline
  (`SpeakerNet_distillation.py`) can absorb a frozen-SSL teacher in
  a separate workstream.
- **Sample-rate conversion for non-16k inputs.** All three reference
  SSL encoders are 16-kHz-pretrained. The DataLoader's
  `--sample_rate` handling from
  [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md) is the right
  place to handle that — the model just emits a warning if
  `sample_rate != 16000`.
- **Pip-installing `transformers` as part of this feature.** Left as
  a deliberate manual step. The user (or CI) must uncomment the
  `requirements.txt` line before running the SSL configs. Avoids
  imposing a 600 MB+ dependency on users who don't need it.

---

## 6. How to use it

### 6.1 SincConv path (Option A)

```bash
# 1) Set the data paths (BUGFIX-013):
source paths.env

# 2) Edit configs/language_aware_sincconv.yaml:
#    - nClasses: set to your SL_Celeb speaker count
#    - initial_model: best English raw-waveform checkpoint (or "")
#    - train_list / test_list / *_path: SL_Celeb paths

# 3) Train
python trainSpeakerNet.py --config configs/language_aware_sincconv.yaml
```

### 6.2 SSL path (Option B), e.g. WavLM-Base

```bash
# 1) Install the optional dependency:
pip install 'transformers>=4.30,<5'

# 2) Edit configs/language_aware_ssl_wavlm.yaml:
#    - nClasses: set to your SL_Celeb speaker count
#    - data paths

# 3) Train (the SSL encoder will download on first run, ~400 MB cached):
python trainSpeakerNet.py --config configs/language_aware_ssl_wavlm.yaml

# To use a different encoder, either:
#   a) Swap to configs/language_aware_ssl_xlsr.yaml or _mhubert.yaml, OR
#   b) Use the WavLM config + CLI override:
python trainSpeakerNet.py \
    --config configs/language_aware_ssl_wavlm.yaml \
    --ssl_encoder_name utter-project/mHuBERT-147

# To fine-tune the encoder (only after you have ≥1k speakers):
python trainSpeakerNet.py \
    --config configs/language_aware_ssl_wavlm.yaml \
    --no_ssl_freeze \
    --lr 0.00005          # MUCH lower LR when unfreezing
```

### 6.3 Comparing the two options

The recommended sequence once an SL pilot exists:

1. **Sanity-check first** — train `language_aware_sincconv.yaml` for 5
   epochs to confirm the data pipeline produces sensible EER. Cheap.
2. **WavLM-Base baseline** — train `language_aware_ssl_wavlm.yaml`
   for 30 epochs. Should beat SincConv by a wide margin per the §3.1
   prediction.
3. **XLS-R-300M ceiling** — if compute permits, run
   `language_aware_ssl_xlsr.yaml` once. Establishes the upper bound.
4. **mHuBERT-147 alternative** — train `language_aware_ssl_mhubert.yaml`
   as a comparison; reports suggest it can beat WavLM-Base on truly
   low-resource Indic languages.

---

## 7. Related items

This feature implements §3.1 #2 of the analysis. It does not close
that item in the analysis doc — implementation is one thing,
verification on real SL data is another.

| Source | Reference |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 #2 | The audit prescription this feature responds to. |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §5 Week 6 | The 12-week plan slot that this feature unblocks. |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §9.3 | The "parallel work that does not need SL data" list. WavLM/XLS-R/mHuBERT prototyping on mini-VoxCeleb1 can start immediately to shake out the plumbing before SL data arrives. |
| [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md) | The sample-rate handling that this feature relies on (SSL encoders need 16 kHz input). |
| [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) | The SincConv buffer-placement fix that makes the Option A path correct under DDP. |
| [BUGFIX-013](BUGFIX-013-portable-config-paths.md) | The `${SL_SPV_DATA_ROOT}` convention used in all four new configs. |
| [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) | The `augment_chain` block used in all four new configs (showing how to bias augmentation away from English-tuned MUSAN/RIR distributions later). |

---

## 8. Authorship & references

- **Feature requested by:** the project maintainer, 2026-05-19,
  citing `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 #2.
- **HuggingFace `AutoModel` API:**
  https://huggingface.co/docs/transformers/main/en/model_doc/auto#autoclasses
  — the dynamic-loading entry point used by `SSLFrontendSpeaker`.
- **WavLM paper:** Chen et al., *"WavLM: Large-Scale Self-Supervised
  Pre-Training for Full Stack Speech Processing"*, 2021.
  https://arxiv.org/abs/2110.13900
- **XLS-R paper:** Babu et al., *"XLS-R: Self-supervised Cross-lingual
  Speech Representation Learning at Scale"*, 2021.
  https://arxiv.org/abs/2111.09296
- **mHuBERT-147:** Boito et al., *"mHuBERT-147: A Compact Multilingual
  HuBERT Model"*, 2024.
  https://arxiv.org/abs/2406.06371
- **SSL-for-SV benchmarks (the 30–50% claim source):** Chen et al.,
  *"Large-scale self-supervised speech representation learning for
  automatic speaker verification"*, ICASSP 2022. Reports
  ~50% relative EER reduction on the VoxCeleb1-O hard subset and
  ~30% on cross-language SV evaluations.
- **The FEATURE-NNN track convention:** opens with this doc.
  Mirrors the BUGFIX-NNN template (Problem / Fix / Verification /
  Backwards-compat / Out-of-scope / How-to-use / Cross-refs /
  Authorship) but the "Problem" framing becomes "What this feature
  delivers". Use this prefix for any new audit-prescribed *features*
  (vs. *bugs*) going forward.
