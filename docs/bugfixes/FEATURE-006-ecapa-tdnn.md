# FEATURE-006 — ECAPA-TDNN model

**Status:** in progress (this PR)
**Type:** feature (new model)
**Relates to:** §3.2 #6 of SL_LANGUAGE_SPV_ANALYSIS.md
**Owner:** SL-SPV
**Touches:** `models/ECAPA_TDNN.py` (new), `configs/ecapa_tdnn.yaml` (new), `SpeakerNet.py` (one-line LLRD alias addition)

## 1. Motivation

§3.2 #6 of the analysis doc identifies ECAPA-TDNN (Desplanques et al., Interspeech 2020) as the strongest practical SV architecture today — consistently 0.8–1.5% EER on VoxCeleb1-O and the de-facto baseline in the SV literature since 2020. The repo today ships `ResNetSE34L`, `ResNetSE34V2`, `RawNet3`, `VGGVox`, `MLPMixerSpeaker`, and FEATURE-001 SSL frontends, but **not** ECAPA. The analysis doc estimates **2–4% absolute EER improvement on SL data** vs the current `ResNetSE34L` baseline.

ECAPA-TDNN is also the natural reference architecture for any future ablations and paper write-ups: nearly every cross-lingual SV paper since 2021 reports ECAPA numbers, so having it in the repo gives us a directly comparable baseline.

## 2. What this is and is not

**Is:**
- A clean implementation of ECAPA-TDNN following Desplanques et al. 2020.
- Uses the shared `make_mel_frontend` helper (BUGFIX-025), so the front-end stays consistent with the other mel-based models.
- Pluggable via `model: ECAPA_TDNN` in YAML; works with all existing trainSpeakerNet flags (AAM-Softmax loss, distributed training, FEATURE-002 AS-Norm, FEATURE-003 per-language eval, FEATURE-004 fine-tune, FEATURE-005 LLRD).
- One config: `configs/ecapa_tdnn.yaml`.
- One LLRD pattern alias added (`ecapa`) so layer-wise decay works when fine-tuning ECAPA.

**Is not:**
- ECAPA-TDNN-Large (3-stage MFA, 1024 channels) is the default; we don't ship a separate "small" variant. Set `channels: 512` in YAML for the smaller version.
- A pretrained checkpoint. Training happens on the user's hardware.
- A pre-emphasis-free path. ECAPA standard practice includes pre-emphasis at 0.97; the config uses the shared frontend's `pre_emphasis=True`.

## 3. Design

### 3.1 Architecture (matches the paper)

```
[B, T] raw audio
  ↓ make_mel_frontend (PreEmphasis + Mel + InstanceNorm)
[B, n_mels=80, T]
  ↓ Conv1d(80→C, kernel=5, dilation=1) + ReLU + BN          (layer1)
[B, C=1024, T]
  ↓ SE-Res2Block(C, kernel=3, dilation=2, scale=8)          (layer2)
  ↓ SE-Res2Block(prev + x1, kernel=3, dilation=3, scale=8)  (layer3)
  ↓ SE-Res2Block(prev + x1, kernel=3, dilation=4, scale=8)  (layer4)
[B, C, T] (each of layer2/3/4)
  ↓ MFA: concat(x2, x3, x4) → Conv1d(3C → 1536) + ReLU
[B, 1536, T]
  ↓ Attentive Statistics Pooling (channel-dependent, BN)
[B, 3072]
  ↓ Linear(3072 → nOut) + BN
[B, nOut]
```

### 3.2 SE-Res2Block

Each block is `Conv1dReluBn → Res2Conv1dReluBn → Conv1dReluBn → SE_Connect`, with a residual skip from input. The Res2 wiring splits the channel dim into 8 groups (`scale=8`), applies dilated `Conv1d` to each group with hierarchical accumulation, and concatenates — exactly the original Res2Net formulation adapted to 1D for TDNNs.

The SE block uses linear bottleneck (default 128 dims) over time-averaged channel activations.

### 3.3 Pooling

Channel-dependent attentive statistics pooling: 1×1 `Conv1d` produces per-channel attention weights, softmaxed over time, then weighted mean + std are concatenated. Output dim is `2 × in_dim = 3072`.

### 3.4 LLRD alias

ECAPA layer names are `layer1` / `layer2` / `layer3` / `layer4`. The new `ecapa` alias in `_LLRD_PATTERN_ALIASES` matches `\.layer(\d+)\.` so FEATURE-005 LLRD works on ECAPA configs. The effective decay across 4 layers with `llrd_decay=0.9` is modest (top vs bottom block ratio ≈ 0.73), as expected for shallower stacks — but the wiring is consistent and composable.

### 3.5 Defaults

| Param | Default | Source |
|---|---|---|
| `channels` (C) | `1024` | Paper "ECAPA-TDNN-Large" |
| `n_mels` | `80` | Paper |
| `nOut` | `192` | Paper embedding dim |
| `encoder_type` | `ASP` | Always attentive-stat pool (ignored for compat) |
| `log_input` | `true` | Standard mel-log input |
| `pre_emphasis` | `true` (in shared frontend) | Paper |

### 3.6 Config

`configs/ecapa_tdnn.yaml`: `lr=0.001`, AAM-Softmax (margin 0.2, scale 30), `nClasses=5994` (override per corpus), `max_epoch=80`. Augmentation chain inherits the BUGFIX-016 default (clean/reverb/music/speech/noise = 0.30/0.20/0.15/0.15/0.20).

## 4. Risk and rollback

- **No existing code paths changed.** New file under `models/`, new config under `configs/`, and one entry added to `_LLRD_PATTERN_ALIASES`. Nothing else.
- **Composability** with existing features is tested via the LLRD pattern alias and by inheriting the shared frontend; no special-case code in the trainers.
- **Rollback** is `git rm models/ECAPA_TDNN.py configs/ecapa_tdnn.yaml` + revert the LLRD alias addition.

## 5. Validation plan (not blocking this PR)

1. Smoke test in this PR: instantiate `ECAPA_TDNN(nOut=192)`, push a `[2, 32000]` tensor through, assert output is `[2, 192]` and finite. Done in the implementation step below.
2. Train on VoxCeleb1 dev → expect 1.0–1.5% EER on VoxCeleb1-O after ~80 epochs.
3. Cross-lingual fine-tune (FEATURE-004 + FEATURE-005) from the English checkpoint to the SL pilot → expect lower EER than `ResNetSE34L` per the literature.

## 6. Cross-references

- `models/_frontend.py` (BUGFIX-025) — shared mel frontend.
- `docs/bugfixes/FEATURE-001-language-aware-frontend.md` — alternative path (SSL frontend); ECAPA is the strong mel-based baseline you compare SSL against.
- `docs/bugfixes/FEATURE-005-llrd.md` — the new `ecapa` LLRD pattern alias.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.2 #6 — original ask.

## 7. References

- Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*. Interspeech 2020. https://arxiv.org/abs/2005.07143
- Shang-Hua Gao et al., *Res2Net: A New Multi-scale Backbone Architecture*. TPAMI 2021.
