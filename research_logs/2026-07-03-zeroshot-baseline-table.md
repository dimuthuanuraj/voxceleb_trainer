# Zero-Shot SOTA Baseline Table on the SL Benchmark (P1)

**Date started:** 2026-07-03 · **Status:** COMPLETE for 4 systems (ECAPA, ReDimNet-B1/B2/B6); WeSpeaker/WavLM deferred to v1
**Roadmap:** P1 of `2026-07-03-project-audit-sota-roadmap.md` ·
**Benchmark:** v0, see `2026-07-03-sl-benchmark-v0-design.md`

## 1. Question

How well do released, VoxCeleb-trained speaker-verification checkpoints work
on Sinhala and Tamil **without any adaptation** — and how large is the
English→SL degradation relative to their published VoxCeleb1-O numbers?
(CN-Celeb precedent predicts 2–4× EER inflation on in-the-wild data;
read-speech v0 is expected to be kinder.)

## 2. Systems under test

| Spec (`tools/zeroshot_eval.py --models`) | Model | Params | Published Vox1-O EER | Source |
|---|---|---|---|---|
| `speechbrain_ecapa` | ECAPA-TDNN (Vox1+2 train) | ~20M | 0.80% | HF speechbrain/spkrec-ecapa-voxceleb |
| `redimnet:b1` | ReDimNet-B1 ft_lm | 2.2M | 0.73% | torch.hub IDRnD/ReDimNet (MIT) |
| `redimnet:b2` | ReDimNet-B2 ft_lm | 4.7M | 0.52% | torch.hub IDRnD/ReDimNet (MIT) |
| `redimnet:b6` | ReDimNet-B6 ft_lm | 15M | 0.37% | torch.hub IDRnD/ReDimNet (MIT) |
| `wespeaker:english` | WeSpeaker ResNet34/ECAPA | — | ~0.72–0.78% | pip wespeaker (GPU node) |
| (planned) WavLM+ECAPA | SSL front-end | 316M+ | 0.39% | ESPnet-SPK / WeSpeaker (GPU node) |

Scoring: cosine on L2-normalized full-utterance embeddings; no score
normalization, no calibration (those are the P2 levers — this table is the
"raw transfer" reference they improve on).

## 3. Results

### v0 smoke validation (2026-07-03, CPU, 200 Tamil trials — sanity only)

| Model | Trial set | EER_avg | minDCF p=.01 | minDCF p=.05 |
|---|---|---|---|---|
| speechbrain_ecapa | ta smoke (100 tgt + 100 imp) | **4.0%** | 0.06 | 0.06 |

First Sinhala/Tamil verification number ever produced by this project.
Interpretation: 5× the checkpoint's Vox1-O EER (0.8% → 4.0%) even on clean
single-session read speech with same-gender impostors — consistent with the
cross-lingual degradation literature. Not citable (200 trials, and the raw
smoke output was not retained — console numbers only); the full table below
replaces it.

### v0 full table (si: 478 spk / ta: 49 spk; 3k targets + 9k impostors per lang)

| Model | test_list_si EER_avg | minDCF .01/.05 | test_list_ta EER_avg | minDCF .01/.05 |
|---|---|---|---|---|
| speechbrain_ecapa | 4.78% | 0.471 / 0.291 | 3.21% | 0.482 / 0.207 |
| redimnet:b1 | 3.57% | 0.306 / 0.209 | 1.67% | 0.406 / 0.162 |
| redimnet:b2 | 4.17% | 0.422 / 0.244 | 1.97% | 0.454 / 0.170 |
| redimnet:b6 | **2.71%** | 0.238 / 0.153 | **1.48%** | 0.422 / 0.160 |
| wespeaker/WavLM | _deferred to v1_ (wespeaker install + 316M staging not worth it before SLCeleb lands) | | | |

Raw outputs: `/home/anuraj/sl_spv_bench/results/zeroshot_v0_full.json` (staged
on the head node — compute nodes have no NFS mount, see USER_ACTION_ITEMS §2;
copy back to `/mnt/ricproject3/2025/data/sl_celeb/results/` once remounted).
Embedding cache: `/home/anuraj/sl_spv_bench/emb_cache/` (reused by P2/P3).

## 4. Analysis (filled 2026-07-03, run complete)

- [x] **Degradation factor** (SL EER ÷ published Vox1-O EER):
  ECAPA 6.0× (si) / 4.0× (ta); B1 4.9× / 2.3×; B2 8.0× / 3.8×; B6 7.3× / 4.0×.
  Tamil sits within the 2–4× band the CN-Celeb literature reports; Sinhala
  (4.9–8.0×) exceeds it — despite v0 being clean read speech, language shift
  alone costs a multiple, not a margin.
- [x] **Ranking is mostly preserved** (B6 best everywhere, ECAPA worst) with
  one inversion: **B1 beats B2 on both languages** despite B2's better Vox1-O
  number — mild evidence that Vox1-O rank does not fully transfer.
- [x] **Parameter count does not monotonically buy robustness**: B1 (2.2M)
  beats both B2 (4.7M) and ECAPA (~20M); only B6 (15M) justifies its size.
- [x] **si vs ta**: ta EER is uniformly lower (studio audio, 49 speakers) but
  ta minDCF(.01) is uniformly *worse* for the ReDimNets (e.g. B6: 0.238 si vs
  0.422 ta) — with only 49 speakers the impostor pool is small and a few hard
  same-genre impostors dominate the low-FA operating point. Report both
  metrics; EER alone overstates how solved ta is.
- [x] **minDCF(.01) is poor everywhere (0.24–0.48)** while EERs look usable —
  the score-calibration signature that motivates P2 (confirmed there: without
  touching the embeddings, AS-Norm alone reaches minDCF(.01) 0.167 si / 0.376
  ta, and adding PLDA takes ta to 0.297 — a large but not complete repair).

## 5. Publication note

This table + the v1 (SLCeleb) rerun is the core of the planned benchmark
paper (IEEE SPL target): SLCeleb has no published baseline table, and no
Sinhala results exist for any of these architectures anywhere in the
literature (verified in the 2026-07-03 literature sweep).
