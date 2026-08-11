# FEATURE-009 — Self-supervised pretraining on unlabelled SL audio (DESIGN-ONLY)

**Status:** **DEFERRED — design document only, no code in this PR**
**Type:** scaffolding (future feature)
**Relates to:** §3.3 #12 of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-001 (off-the-shelf SSL — the *cheap* alternative this defers to)
**Owner:** SL-SPV
**Touches:** none yet — when implemented, will add `pretrain.py` (~400 LOC), an unlabelled audio loader, and `configs/pretrain_*.yaml`

## 0. Why this file exists without code

§3.3 #12 of the analysis doc proposes wav2vec2-style SSL pretraining on hundreds of hours of unlabelled Sinhala/Tamil audio, then SV fine-tuning. This is a real opportunity but not the right next step:

1. The bulk of the SSL benefit is **already captured** by FEATURE-001 — frozen WavLM-Base / XLS-R-300M / mHuBERT-147, all multilingual and covering Sinhala+Tamil out of the box.
2. The marginal gain from continued pretraining (CPT) on SL audio is real but smaller (~5–15% relative EER on top of FEATURE-001) and requires significant prerequisites: ≥500 h unlabelled SL audio assembled AND a measured baseline showing FEATURE-001 has plateaued.
3. Implementing the code now means it sits untested for the corpus-assembly weeks. The design is more durable than the stub.

This document captures the design so future-you can land the implementation in a day once prerequisites are met. It is **not** a TODO to act on now.

## 1. The three SSL paths

| Path | What | Cost | When it wins |
|---|---|---|---|
| **A. Off-the-shelf SSL** (already done — FEATURE-001) | Load WavLM-Base / XLS-R-300M / mHuBERT-147 from HuggingFace, freeze, train pooling+head on labelled SL | $0; 1 GPU-day to fine-tune the SV head | Your first move. mHuBERT-147 explicitly covers Sinhala + Tamil with upweighted low-resource balancing. |
| **B. Continued pretraining (CPT)** — *this feature* | Take an off-the-shelf SSL checkpoint, run the wav2vec2 contrastive objective for more steps on your SL audio (no labels), then SV-fine-tune | ~3–5 GPU-days per 100 h | Off-the-shelf SSL EER has plateaued AND you have ≥500 h unlabelled SL audio AND target language is under-represented in the off-the-shelf model. |
| **C. From-scratch SSL** | Train wav2vec2 from random weights on SL audio only | ~weeks; needs ≥5,000 h | Almost never. You cannot beat 100,000 h of multilingual pretraining with 5,000 h of monolingual data. |

This feature implements **path B**. Path A is shipping. Path C is intentionally out of scope.

## 2. Prerequisites checklist (gate this feature behind these)

- [ ] **§9 action 2 — corpus assembled.** ≥500 hours of unlabelled Sinhala+Tamil audio collected (podcast / broadcast / lecture / parliamentary sources). Speaker grouping not required.
- [ ] **§9 action 5 — baseline EER measured** under FEATURE-001 (WavLM-Base, XLS-R-300M, mHuBERT-147 all run, best one chosen as baseline).
- [ ] **Compute budget approved.** 3–5 GPU-days per CPT run, with at least one round of hyper-parameter sweep — call it 15 GPU-days total before you see a number.
- [ ] **Expected EER improvement is the rate-limiting question.** If the FEATURE-001 baseline already meets the deployment target, this feature is not needed.

If any box is unchecked when you come back to this doc, **do not implement**. The design only justifies itself when all four hold.

## 3. Design (when implementation is unblocked)

### 3.1 New files

- `pretrain.py` — top-level script, structurally parallel to `trainSpeakerNet.py` but with the wav2vec2 contrastive head instead of a speaker loss.
- `models/SSLPretrainWrapper.py` — wraps a HuggingFace `Wav2Vec2ForPreTraining` (or `WavLMForPreTraining`) module; exposes a `forward(wav)` that returns the contrastive loss.
- `DatasetLoader.py` — extend with `unlabelled_dataset_loader` (audio paths only, no speaker grouping). Reuse the existing augmentation chain (BUGFIX-016) but skip speaker-level sampling.
- `configs/pretrain_wavlm.yaml`, `configs/pretrain_xlsr.yaml`, `configs/pretrain_mhubert.yaml` — one per starting checkpoint.

### 3.2 Workflow

```
unlabelled_sl_audio/   ─→ pretrain.py ─→ checkpoint.pt   ─→ trainSpeakerNet.py  ─→ fine-tuned SV
                                          (HF format)        --ssl_encoder_name <path>
                                                             (FEATURE-001 already supports this)
```

The fine-tune step is **already implemented** via FEATURE-001's `SSLFrontendSpeaker` — point `ssl_encoder_name` at the local CPT checkpoint directory and FEATURE-001 picks it up unchanged.

### 3.3 Hyper-parameters (sensible CPT defaults)

| Knob | CPT default | Notes |
|---|---|---|
| Starting checkpoint | `utter-project/mHuBERT-147` | best multilingual coverage for SL |
| Learning rate | `5e-5` | 1/10 of the original pretraining LR; CPT should not destabilise prior |
| Batch size | as large as memory allows (typical: 32 × 6-second clips on a 24GB GPU) | wav2vec2 likes large batches for diverse negatives |
| Mask span | 10 frames, p=0.065 | wav2vec2 paper defaults |
| Steps | 100k–250k | enough to adapt without overfitting |
| Audio length | 6 seconds | balance of context vs memory |
| Augmentation | re-use BUGFIX-016 chain (clean/reverb/music/speech/noise) | builds robustness to deployment channels |

### 3.4 Output artefact

A HuggingFace-format checkpoint directory (config.json + safetensors), so FEATURE-001 can load it as `ssl_encoder_name: ./exps/cpt_mhubert_sl/checkpoint-final`. No new loading code needed.

### 3.5 What can be reused from existing features

| Existing | Reused for SSL pretraining |
|---|---|
| FEATURE-001 (SSL encoder loader) | Final fine-tune step (the SV head). CPT outputs an HF checkpoint; FEATURE-001 consumes it. |
| BUGFIX-005 / 006 (sample rate threading) | The CPT loader needs sample-rate handling. |
| BUGFIX-016 (augment_chain) | CPT augmentation policy. |
| BUGFIX-018 (eval streaming) | Not directly applicable; CPT has no eval pass. |
| FEATURE-004/005 (fine-tune freeze + LLRD) | The SV-fine-tune step downstream of CPT can use these as today. |

## 4. Empirical expectation (literature)

For a typical low-resource language with ≥500 h of CPT audio, on top of an off-the-shelf multilingual SSL:

| Metric | FEATURE-001 (off-the-shelf) baseline | After CPT (this feature) |
|---|---|---|
| EER | X% | X% × (0.85 to 0.95) — i.e. 5–15% relative improvement |
| MinDCF | Y | Y × (0.85 to 0.95) |
| Cross-lingual gap (si EER vs ta EER) | Z | usually shrinks |

These numbers are not promises. They are the literature's median outcome on comparable low-resource SV setups (e.g. Pratap et al. 2023 *Scaling Speech Technology to 1,000+ Languages*).

## 5. Risks (when implementation lands)

- **Catastrophic forgetting.** CPT with too high LR can degrade the multilingual prior. The `5e-5` default is conservative; sweep `{1e-5, 5e-5, 1e-4}` before declaring a result.
- **Data quality.** Podcast/broadcast audio has variable SNR, music beds, ad breaks. A simple energy-based VAD filter is required before CPT; otherwise the contrastive objective wastes capacity on silence/music.
- **Compute waste.** A 250k-step run that produces a worse SV EER than the FEATURE-001 baseline is a real outcome. **Always re-measure with the baseline as the floor**, not as the comparison target.
- **Sunk-cost trap.** If after ≥2 CPT runs the SV-fine-tune EER does not improve over FEATURE-001, abandon the path. Path A may simply be sufficient.

## 6. Cross-references

- `docs/bugfixes/FEATURE-001-language-aware-frontend.md` — the cheap alternative that captures most of the SSL benefit.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.3 #12 — original ask; §9 actions 2 & 5 — the prerequisites that gate this feature.

## 7. References

- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020. https://arxiv.org/abs/2006.11477
- Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, IEEE/ACM TASLP 2021.
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages*, JMLR 2024 (MMS). https://arxiv.org/abs/2305.13516
- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022.

## 8. Status when this is unblocked

When the §2 prerequisites are met, update this header from `DEFERRED — design document only` to `in progress` and land the implementation per §3. Strike the warning, write the validation results in §4, and cross-ref the implementation PR.
