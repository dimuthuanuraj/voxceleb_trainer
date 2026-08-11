# Project Audit & SOTA-Aligned Roadmap — Sinhala/Tamil Speaker Verification

**Date:** 2026-07-03
**Scope:** Full audit of the SL_SPV/voxceleb_trainer codebase and research plan, cross-checked
against 2022–2026 literature (six research strands: architectures, SSL front-ends,
low-resource/cross-lingual adaptation, datasets, evaluation protocol, prior Sinhala/Tamil work,
edge deployment). Each roadmap item below is designed to become one proof-of-concept run plus
one research document.

---

## 1. Where the project stands

- Engineering hygiene is genuinely strong: 25/26 BUGFIX items closed, 11 FEATURE modules
  shipped (ECAPA-TDNN, SSL front-ends, AS-Norm, per-language eval, LLRD fine-tuning, DANN,
  lang-aux head, PLDA, streaming eval, `sl_dataprep.py`), portable configs, deterministic flag.
- **The research phase has not started.** Every EER number to date is English VoxCeleb.
  There is no Sinhala/Tamil audio in the tree and no SL EER number.
- The current English baseline (`exps/EN_p0_baseline_seed42`) is **blocked**: 564 missing
  VoxCeleb2 wavs (`bad_wavs.txt` / `drop_relpaths.txt`), 8 launch attempts on 2026-05-27,
  `result/scores.txt` empty, no checkpoints.

## 2. The single most important audit finding

**The project plan under-uses assets the group already owns, and the literature has moved past
the plan's core bets.**

1. **SLCeleb already exists.** The group's own dataset (IEEE DataPort, CC BY 4.0): 280
   Sinhala/Tamil celebrities (dev: 110 si + 100 ta; test: 40 + 40), 34k utterances, 4 genres,
   CN-Celeb-style in-the-wild collection. The 12-week plan treats "no SL corpus" as the blocker
   and proposes assembling a 100-speaker pilot from OpenSLR — while a purpose-built, larger,
   multi-genre corpus is already published by this group.
2. **OpenSLR SLR52 (Sinhala) has 478 speakers** (verified by counting `utt_spk_text.tsv`:
   185,293 utts, speaker IDs in column 2) — not the ~70 assumed in `SL_LANGUAGE_SPV_ANALYSIS.md`.
   Plus Tamil: SLR65 (50 spk, studio), IISc-MILE/SLR127 (531 spk), Kathbath (AI4Bharat, official
   SV benchmark splits), Vaani (Tamil Nadu districts, speaker IDs + metadata).
3. **The only published Sinhala SV paper is the group's own SLAAI-ICAI 2022 paper**
   (IEEE 10002663). Nobody else has published Sinhala SV, and **SLCeleb has no published
   baseline table** — a systematic benchmark paper is an open, low-effort, citable contribution.

## 3. Issues found (audit)

### 3.1 Research-strategy issues

| # | Issue | Evidence | What the literature says |
|---|---|---|---|
| R1 | Custom MLP-Mixer + LSTM-AE distillation track is far off SOTA | Best repo result: 10.32% EER @2.66M params (mini-Vox); teacher 9.68% | ReDimNet-B1 (Interspeech 2024, MIT weights): **0.85% EER @2.2M params**; even 1.0M ReDimNet-B0 gets 1.16%. Training small models from scratch on VoxCeleb is a solved-and-lost race; the field fine-tunes released checkpoints. |
| R2 | RQ1 plans from-scratch ECAPA on 100 SL speakers | `thesis_chapters/01_introduction.tex` | Consistent evidence this fails: frozen SSL/pretrained encoder + light head or PEFT halves EER vs from-scratch under matched data, and the gap grows as labeled data shrinks (WavLM arXiv:2110.13900; UniPET-SPK arXiv:2501.16542). Keep from-scratch only as a reported lower bound. |
| R3 | No zero-shot baseline planned before training anything | 12-week plan | The cheapest first SL EER table is pure evaluation of released checkpoints (SpeechBrain ECAPA, ReDimNet, CAM++, WavLM+ECAPA). CN-Celeb precedent says expect 2–4× EER inflation vs English (arXiv:1911.01799) — that degradation curve is itself a publishable result for Sinhala. |
| R4 | FEATURE-007 (lang-aux) vs FEATURE-008 (DANN/GRL) have opposite objectives on language | `docs/bugfixes/` feature notes | Current cross-lingual SOTA pattern is GRL + PEFT (Dual-LoRA arXiv:2604.26327; TidyVoice 2026 arXiv:2603.08092). Pick DANN/GRL for language invariance; drop the lang-aux head or use it only as an ablation arm. |
| R5 | Backend/calibration levers under-planned | AS-Norm/PLDA implemented but no cohort/calibration strategy | Cross-lingual trials suffer a systematic *target-score shift* (arXiv:2110.09150). AS-Norm with a target-language cohort ≈ 20–30% relative gain (Matejka 2017); PLDA beats cosine specifically under domain shift (arXiv:2204.11403); language-aware logistic calibration + Cllr reporting closes the rest. All are training-free. |
| R6 | 100-speaker benchmark is smaller than the data available | plan §2.3 | With SLCeleb (280) + SLR52 (478) + MILE Tamil (531) + Kathbath, a train set of several hundred speakers per language and a proper held-out eval is feasible now. |
| R7 | Headline 10.32% MLP-Mixer number unreplicated | predates `--deterministic`; single seed | Re-run 3 seeds or retire the claim from the thesis. |

### 3.2 Evaluation-protocol issues

| # | Issue | Fix |
|---|---|---|
| E1 | Session confound risk: read-speech corpora are near single-session per speaker → same-recording target trials inflate performance (8–17% of VoxCeleb1-H target pairs share a recording; Interspeech 2022/2024, arXiv:2408.13614) | Enforce cross-session (at minimum cross-video/cross-recording) enroll–test pairing in `sl_dataprep.py` trial generation; same-gender non-targets. |
| E2 | minDCF only at p_target=0.05 (`trainSpeakerNet.py:78`) | Report both p=0.01 (NIST-style) and p=0.05 (VoxCeleb-style); add actDCF and Cllr/minCllr after calibration. |
| E3 | Conservative `EER_max` reported as headline "VEER" | Report literature-standard `EER_avg` as the primary number (already computed), keep EER_max as diagnostic. |
| E4 | Missing trial labels → `random.randint(0,1)` appended silently (`SpeakerNet.py:675`) | Fail loudly on unlabeled trials. |
| E5 | No mono/cross-lingual trial partitions defined | Trial lists: {si–si, ta–ta, si↔ta same-speaker cross-lingual, pooled}, SdSV/DeepMine protocol as template. |

### 3.3 Engineering issues

| # | Issue | Fix |
|---|---|---|
| C1 | EN baseline blocked by 564 missing wavs | Filter `drop_relpaths.txt` out of `train_list.txt` (or re-download); the run then proceeds. |
| C2 | Three near-duplicate trainer stacks (`trainSpeakerNet*.py`, `SpeakerNet*.py`, `DatasetLoader*.py`) | Consolidate on the main stack (BUGFIX-024 already recommends this); delete `_performance_updated` copies after porting anything unique. |
| C3 | `max_test_pairs` silently ignored by the main trainer (only implemented in distillation/performance variants) | Implement in main eval path or remove from configs. |
| C4 | Seed default mismatch: CLI `--seed 10` vs configs `seed: 42` | Change CLI default to 42. |
| C5 | No pytest suite / CI; only ad-hoc root test scripts | Move to `tests/`, add a GitHub Actions smoke run (1-batch train + eval on synthetic data). |
| C6 | Machine-specific absolute paths (`paths.env`, `data/` symlinks) | Already env-var based for configs; document symlink recreation in SETUP.md. |
| C7 | NestedSpeakerNet (failed, NaN, quarantined) + its configs still present | Archive to a branch; remove `nested_*.yaml` from `configs/`. |
| C8 | LaTeX build artifacts and large binaries committed | gitignore `*.aux/.log/.out/.toc/.bbl/.blg`, move PDFs to releases or LFS. |

## 4. What the 2023–2026 literature says (condensed)

### Architectures (all Vox1-O, Vox2-dev training)
- **ReDimNet** (Interspeech 2024, arXiv:2407.18223, MIT weights): B0 1.0M/0.43GMACs → 1.16%;
  B1 2.2M → 0.85%; B2 4.7M → 0.57%; B6 15M → 0.40%. Dominates every size class.
  ReDimNet2 (arXiv:2603.11841, 2026) improves the whole front (B6 → 0.29%); weights TBD.
- ECAPA2 27M → 0.34% but **CC-BY-NC** (no commercial use). CAM++ 7.2M → 0.65–0.73%
  (Apache-2.0, 3D-Speaker/WeSpeaker). ERes2NetV2 → 0.61% full / 0.98% @3s (best short-utterance
  recipe). SpeechBrain ECAPA HF checkpoint: 0.80%.

### SSL front-ends
- WavLM-Large + ECAPA: frozen 0.617%, fine-tuned + LMFT + calibration 0.383%
  (arXiv:2110.13900); reproducible via ESPnet-SPK (0.39%, arXiv:2401.17230).
- **w2v-BERT 2.0** (600M, 143 languages incl. South Asian): current effective SOTA 0.12% with
  LoRA + layer adapters; 80% structured pruning costs +0.04% (arXiv:2510.04213). Best
  cross-lingual coverage for Sinhala/Tamil.
- WavLM ≫ wav2vec2/HuBERT for speaker tasks (denoising objective). mHuBERT-147 unproven for SV.
- Caution: ABC/BUT SRE24 found XLS-R-type backbones *underperformed* a VoxBlink2-pretrained
  ResNet fine-tuned on target data (arXiv:2505.15320) — multilingual SSL is not automatically
  better; data-side adaptation + fine-tuning is.

### Low-resource / cross-lingual adaptation (most relevant to us)
- Language mismatch costs 2–4× EER (CN-Celeb: 4.78% → 15.52%, arXiv:1911.01799).
- Training-free stack: AS-Norm w/ target-language cohort (~20–30% rel., Matejka 2017) →
  PLDA + CORAL++/unsupervised adaptation (~10–15% rel., arXiv:2202.01092) → language/duration-aware
  logistic calibration (fixes cross-lingual score shift, arXiv:2110.09150).
- **PEFT beats full fine-tuning under language shift**: UniPET-SPK 13.12% vs full-FT 14.49% on
  CN-Celeb transfer, 5.4% trainable params (arXiv:2501.16542). Dual-LoRA adds GRL language
  disentanglement inside adapters (arXiv:2604.26327).
- Unlabeled-audio route: DINO-lineage self-distillation — SDPN 1.80% Vox1-O without labels
  (arXiv:2308.02774; +score-norm 1.29%, arXiv:2505.13826); RDINO code in 3D-Speaker.
- Indian-language precedent: I-MSV 2022 (arXiv:2302.13209) winners = pretrained + weight-transfer
  fine-tuning + fusion (NPU-HC, arXiv:2211.16694); LASE (arXiv:2605.00777) quantifies and closes
  cross-script similarity drop incl. Tamil.

### Datasets (verified)
| Corpus | Lang | Speakers | Role |
|---|---|---|---|
| **SLCeleb** (ours, IEEE DataPort, CC BY 4.0) | si+ta | 280 (dev 210 / test 80) | in-the-wild train + THE eval set |
| OpenSLR SLR52 (CC BY-SA) | si | **478**, 185k utts, ~224h | training (read speech) |
| OpenSLR SLR65 / SLR30 | ta / si | 50 / 12 | clean sanity-check eval only |
| IISc-MILE SLR127 | ta | 531, 150h | training (verify spk IDs in tar) |
| Kathbath / IndicSUPERB (CC0, gated HF) | ta (+11) | 1,218 total | training + standardized ta SV test |
| Vaani (CC BY 4.0) | ta districts | large | training/augment |
| VoxBlink2 (CC BY-NC-SA) | multi | 110k | pretraining base (via released ckpts) |
| Shrutilipi, FLEURS | — | — | **unusable** (no speaker labels) |
| SiTa (CHiPSAL 2025, Moratuwa) | si+ta | — | diarization; possible extra wild audio |

### Edge deployment (if on-device matters later)
ReDimNet-B0/B1 or CAM++ → optional WavLM-teacher KD (arXiv:2309.14838) → ONNX INT8
(4-bit near-lossless demonstrated, arXiv:2406.05359) → sherpa-onnx runtime (Android/RPi).

## 5. Roadmap — each phase = one PoC + one research document

Ranked by (impact × feasibility on single-GPU). Phases 0–2 need **no training at all**.

### P0 — Unblock & data foundation (days)
- Filter the 564 `drop_relpaths.txt` entries from the EN train list; relaunch EN baseline
  (still useful as the English reference point).
- Ingest SLCeleb + SLR52 + SLR65 + Kathbath-ta into the `sl_celeb/<lang>/<spk>/<utt>.wav`
  layout `tools/sl_dataprep.py` expects.
- Extend `sl_dataprep.py` trial generation: cross-session constraint (E1), same-gender
  non-targets, 4 partitions (E5), cohort + PLDA lists.
- **Doc:** "SL-SPV Benchmark Design: corpora, trial protocol, and metrics" (becomes thesis §3
  + the dataset/protocol section of the benchmark paper).

### P1 — Zero-shot SOTA baseline table (days, GPU only for inference)
- Evaluate off-the-shelf checkpoints on the new benchmark: SpeechBrain ECAPA, ReDimNet-B1/B2/B6,
  CAM++, ERes2NetV2, WavLM+ECAPA (ESPnet-SPK/WeSpeaker), optionally w2v-BERT 2.0 SV.
- Deliverable: first-ever Sinhala EER table across modern architectures; English→SL degradation
  curve (CN-Celeb analogue).
- **Doc/paper:** "Benchmarking modern speaker verification on SLCeleb" — fills the documented
  gap (no SLCeleb baseline table exists); target IEEE SPL or the scaffolded IJST paper.

### P2 — Training-free adaptation stack (days)
- AS-Norm with SL cohort (FEATURE-002, exists) → PLDA trained/adapted on SL dev
  (FEATURE-010 + add CORAL+ interpolation, small code change) → language-aware logistic
  calibration (new small module; report Cllr/actDCF).
- Ablation: each lever alone and composed → directly answers RQ4.
- **Doc:** "Score-domain adaptation for cross-lingual SV in Sinhala and Tamil."

### P3 — PEFT fine-tuning (weeks, 1 GPU)
- LoRA/adapter fine-tuning of the best P1 backbone on SL train speakers; arms:
  frozen / LoRA / full-FT / LLRD (FEATURE-005 exists). 3 seeds.
- Expected from literature: PEFT ≥ full-FT under shift, at ~5% trainable params.
- **Doc:** "Parameter-efficient adaptation of pretrained SV models to Sri Lankan languages"
  — answers RQ2, likely the strongest thesis chapter.

### P4 — Language-invariant training & the Sinhala↔Tamil cross-lingual study (weeks)
- GRL/DANN head (FEATURE-008 exists) on top of P3, Dual-LoRA pattern; drop lang-aux arm (R4).
- Unique angle nobody has published: **bilingual Sri Lankan speakers enrolled in one language,
  tested in the other** (si↔ta cross-lingual same-speaker trials) — answers RQ3+RQ5.
- **Doc/paper:** the flagship cross-lingual paper (scaffolded IJB/IJST target).

### P5 — Self-supervised on unlabeled SL audio (optional, if ≥100s of hours collectable)
- SDPN/RDINO-style self-distillation on unlabeled Sinhala/Tamil (broadcast/SiTa-style audio),
  pseudo-label + iterate; supersedes FEATURE-009's design-only status.
- **Doc:** "Label-free speaker representation learning for Sinhala/Tamil."

### P6 — Edge PoC (optional, aligns with the original distillation thread)
- Distill P3/P4 best model into ReDimNet-B0/CAM++, INT8 ONNX export, sherpa-onnx demo.
- **Doc:** "On-device speaker verification for Sri Lankan languages" (ADCAIJ target fits).

### Retire / de-prioritize
- NestedSpeakerNet (failed, documented) — archive.
- From-scratch MLP-Mixer training race — keep the *distillation methodology* finding
  (cosine ≫ MSE on normalized embeddings; α ∝ capacity ratio) as a thesis contribution, but
  stop pushing the architecture against ReDimNet-class checkpoints.

## 6. Publication map

| Paper (scaffold exists in `papers/`) | Feeds from | Novelty claim |
|---|---|---|
| SLCeleb benchmark paper (IEEE SPL) | P0+P1 | first modern-SV baseline table for Sinhala; degradation analysis |
| Score-domain adaptation (IJST) | P2 | first cross-lingual score-shift study for si/ta |
| PEFT adaptation (IJB) | P3 | first PEFT-for-SV result on Sri Lankan languages |
| Cross-lingual si↔ta flagship | P4 | first bilingual Sri Lankan cross-lingual SV study — no prior work exists |
| Deployment/toolkit (ADCAIJ) | P6 + engineering | reproducible low-resource SV toolkit |

## 7. Key references (verified during this audit)

ReDimNet arXiv:2407.18223 · ReDimNet2 arXiv:2603.11841 · ECAPA2 arXiv:2401.08342 ·
CAM++ arXiv:2303.00332 · ERes2NetV2 arXiv:2406.02167 · WavLM arXiv:2110.13900 ·
ESPnet-SPK arXiv:2401.17230 · w2v-BERT2-SV arXiv:2510.04213 · UniPET-SPK arXiv:2501.16542 ·
Dual-LoRA arXiv:2604.26327 · score-shift arXiv:2110.09150 · CORAL++ arXiv:2202.01092 ·
backend selection arXiv:2204.11403 · Matejka AS-norm Interspeech 2017 · CN-Celeb arXiv:1911.01799 ·
SDPN arXiv:2308.02774 · ABC SRE24 arXiv:2505.15320 · I-MSV arXiv:2302.13209 ·
NPU-HC arXiv:2211.16694 · LASE arXiv:2605.00777 · bias/session-confound arXiv:2408.13614 ·
TidyVoice arXiv:2603.08092 · SLCeleb IEEE DataPort · SLAAI-ICAI 2022 IEEE 10002663 ·
SiTa CHiPSAL 2025 (aclanthology 2025.chipsal-1.8) · OpenSLR 52/65/30/127 · Kathbath arXiv:2208.11761 ·
non-target KD arXiv:2309.14838 · SV quantization arXiv:2406.05359 · sherpa-onnx (k2-fsa).
