# P2 — Training-Free Backend Adaptation: AS-Norm, PLDA, Calibration (RQ4)

**Date started:** 2026-07-03 · **Status:** COMPLETE for v0 (main grid §3, calibration §3.2, P2×P3 composition §5); §5b items blocked on SLCeleb (v1)
**Roadmap:** P2 of `2026-07-03-project-audit-sota-roadmap.md` ·
**Benchmark:** v0 (`2026-07-03-sl-benchmark-v0-design.md`) ·
**Inputs:** cached embeddings from P1 (`2026-07-03-zeroshot-baseline-table.md`)

---

## 1. Research question (thesis RQ4)

Do **AS-Norm** and **PLDA** compose or overlap when adapting VoxCeleb-trained
speaker embeddings to Sinhala/Tamil — and how much of the cross-lingual
degradation is a fixable **score-domain** problem (shift/scale) rather than an
embedding-quality problem? All levers here are **training-free**: no
fine-tuning, no GPU beyond one embedding-extraction pass.

Why this matters given P1: zero-shot EERs were usable (1.5–4.8%) but
**minDCF(p=0.01) was poor everywhere (0.24–0.48)** — the classic signature of
uncalibrated, badly-scaled scores in a mismatched domain.

Literature predictions under test:

| Prediction | Source |
|---|---|
| AS-Norm with a target-domain cohort ≈ 20–30% relative gain | Matejka et al., Interspeech 2017 |
| PLDA (vs cosine) wins under domain shift, is redundant in-domain | Wang et al., Interspeech 2022 (arXiv:2204.11403) |
| Cross-lingual trials suffer a systematic target-score shift fixable by language-aware calibration | Thienpondt et al., ICASSP 2022 (arXiv:2110.09150) |
| Composition (PLDA + AS-Norm + calibration) is the standard low-resource stack | SdSV/IDLab (arXiv:2007.07689), NIST SRE practice |

## 2. Method (`tools/backend_adapt.py`, new; reuses repo FEATURE-002/010 code)

### 2.1 Data (all from the staged v0 benchmark; no test-file leakage)

| Component | Contents | Source |
|---|---|---|
| Cohort | 527 utts (1 per speaker, si+ta train pool) | `p2_cohort.txt` |
| PLDA train | 10,480 utts (≤20/spk, 527 speakers, labels) | `p2_plda_list.txt` |
| Trials | full 12,000-pair si and ta lists (same as P1/P3) | `lists/test_list_{si,ta}.txt` |

Cohort and PLDA sets are drawn from the **train** utterance split only —
disjoint from all trial files. Cohort follows the Matejka recommendation:
multi-language, target-domain.

### 2.2 Conditions (per model × per language)

1. `cosine` — raw baseline (sanity check: must reproduce P1 exactly).
2. `cosine+asnorm` — adaptive S-norm: per trial side, mean/std of its
   **top-K (K=300)** cohort similarities;
   `s' = ½[(s−μ_e)/σ_e + (s−μ_t)/σ_t]`.
3. `plda` — TwoCovPLDA (repo `plda.py`), LDA dim 150, length-norm +
   centering, fit on the 10,480 SL train embeddings. Scored with a
   vectorized all-pairs fast path over the fitted quadratic forms.
4. `plda+asnorm` — AS-Norm applied to PLDA scores (cohort scored via PLDA).

### 2.3 Calibration protocol

- Stratified 50/50 calibration/eval split of each trial list (seed 42).
- Affine logistic-regression calibration (score → LLR) per condition.
- **Self-cal:** fit and evaluate within the same language.
- **Cross-cal:** fit on language A's calibration half, evaluate on language
  B's eval half — the *cross-lingual score shift* measurement: if cross-cal
  Cllr ≫ self-cal Cllr, the score domains of si and ta differ systematically
  and deployment needs language-aware calibration.
- Metrics: EER_avg + minDCF(0.01/0.05) on full lists (calibration-invariant);
  **Cllr** and **actDCF(0.01)** on the eval half, raw vs calibrated.

### 2.4 Models

`speechbrain_ecapa`, `redimnet:b1`, `redimnet:b6` (the P1 extremes + best).
Extension to the fine-tuned P3 checkpoints is a follow-up (their embeddings
need one extraction pass with the P3 model code).

## 3. Results (2026-07-03, run complete)

Raw outputs: `/home/anuraj/sl_spv_bench/results/p2_backend.{json,md}`
(mirrored to `/mnt/ricproject3/2025/data/sl_celeb/results/`).
Sanity check passed: every `cosine` row reproduces the P1 table exactly.

### 3.1 Main ablation grid — EER % (minDCF p=0.01), full 12k-pair lists

| Model | Cond | si EER (minDCF01) | ta EER (minDCF01) |
|---|---|---|---|
| speechbrain_ecapa | cosine | 4.78 (0.471) | 3.21 (0.482) |
| | cosine+**asnorm** | **3.19 (0.330)** | 2.24 (0.480) |
| | plda | 6.00 (0.598) | 2.46 (0.495) |
| | plda+asnorm | 5.80 (0.482) | **2.06 (0.374)** |
| redimnet:b1 | cosine | 3.57 (0.306) | 1.67 (0.406) |
| | cosine+**asnorm** | **2.38 (0.245)** | 1.47 (0.433) |
| | plda | 5.34 (0.452) | 1.77 (0.419) |
| | plda+asnorm | 5.23 (0.382) | **1.49 (0.297)** |
| redimnet:b6 | cosine | 2.71 (0.238) | 1.48 (0.422) |
| | cosine+**asnorm** | **2.07 (0.167)** | **0.90 (0.376)** |
| | plda | 3.84 (0.293) | 1.11 (0.340) |
| | plda+asnorm | 3.56 (0.254) | 1.16 (0.314) |

**Best training-free v0 system: ReDimNet-B6 + AS-Norm — si 2.07%, ta 0.90%.**

### 3.2 Calibration: self-language vs cross-language (redimnet:b6, Cllr)

| Condition | si self-cal | si with **ta-fit** calibrator | ta self-cal | ta with **si-fit** calibrator |
|---|---|---|---|---|
| cosine | 0.129 | 0.489 (3.8×) | 0.095 | 0.149 (1.6×) |
| cosine+asnorm | 0.099 | 0.565 (5.7×) | 0.071 | 0.167 (2.4×) |
| plda | 0.171 | 0.390 (2.3×) | 0.073 | 0.084 (1.2×) |
| plda+asnorm | 0.163 | 0.442 (2.7×) | 0.066 | 0.094 (1.4×) |

## 4. Findings (v0 verdicts on the literature predictions)

1. **AS-Norm: confirmed, 12–39% relative EER gain** (si: 33/33/23% for
   ECAPA/B1/B6; ta: 30/12/39%) — bracketing the Matejka 20–30% band (B1 ta
   is the 12% outlier), and it improves minDCF(.01) in five of six
   model–language cells (best: B6 si 0.238→0.167; the exception is B1 ta,
   0.406→0.433). The single most cost-effective lever in the whole project
   so far: no training, one cohort file.
2. **PLDA: refuted on Sinhala, mildly confirmed on Tamil.** PLDA *hurts* si
   EER for every model (e.g. B6 2.71→3.84) while helping ta for two of
   three models (ECAPA 3.21→2.46, B6 1.48→1.11; B1 worsens slightly,
   1.67→1.77). Plausible mechanism: the PLDA fit is dominated
   by the 478 si speakers of near-single-session crowdsourced audio, so its
   within-speaker covariance underestimates real channel variability
   (session confound, benchmark-design doc §5); ta's studio recordings fit
   the model better. This is a *benchmark artifact hypothesis* — v1
   (SLCeleb, multi-session) is the decisive test, and the arXiv:2204.11403
   "PLDA wins under shift" claim should not be dismissed until then.
3. **RQ4 (do AS-Norm and PLDA compose?): at v0, AS-Norm ≻ PLDA and
   composition does not beat cosine+AS-Norm on si** (it does give the best ta
   minDCF). Answer is condition-dependent — worth keeping both in the v1
   rerun.
4. **Cross-lingual score shift: large and real.** Swapping calibrators across
   languages inflates Cllr up to **5.7×** (si scored with a ta-fit
   calibrator). Deployment implication: **per-language calibration is
   mandatory**. Notably, PLDA scores are the most *transportable* across
   languages (smallest cross-cal penalty, 1.2–2.7×) even where its EER is
   worse — a genuinely publishable nuance for the score-shift story
   (extends arXiv:2110.09150 to Sinhala/Tamil).

## 5. Composition with fine-tuning (P2 × P3) — run 2026-07-03

Backend stack applied to the **fine-tuned** WavLM (P3 `full` arm, seed 42)
embeddings; extraction via `tools/extract_p3.py` (8 s cap, matching the P3
final-eval protocol; `cosine` row reproduces the P3 finals exactly).
Raw: `results/p2xp3_composition.{json,md}`.

| Cond | si EER (minDCF01, Cllr) | ta EER (minDCF01, Cllr) |
|---|---|---|
| cosine | **1.63** (0.180, 0.080) | 1.30 (0.175, 0.079) |
| cosine+asnorm | 1.71 (**0.144**, 0.068) | **1.13** (**0.149**, 0.068) |
| plda | 2.46 (0.258, 0.132) | 1.57 (0.234, 0.084) |
| plda+asnorm | 2.37 (0.197, 0.127) | 1.43 (0.201, 0.075) |

**Findings:**
- After in-language fine-tuning, **AS-Norm's EER gain saturates** (si even
  −0.07 abs; ta still +13% rel) **but it keeps improving the operating-point
  metrics** — minDCF(.01) si 0.180→0.144, ta 0.175→0.149, and Cllr down ~15%
  on both. Backend adaptation remains worth keeping for deployment even when
  the encoder is adapted.
- **PLDA is now clearly redundant/harmful on both languages** — exactly the
  arXiv:2204.11403 prediction: PLDA helps under domain shift, is redundant
  in-domain. Fine-tuning moved us in-domain; the v0 PLDA story is coherent.
- **Best v0 system overall: fine-tuned WavLM (full) + AS-Norm** —
  si 1.71%/0.144, ta 1.13%/0.149 (or plain cosine if EER is the sole
  criterion: 1.63/1.30).

## 5b. Remaining analysis

- [ ] v1 (SLCeleb) rerun once files are placed — open-set + multi-session is
      where PLDA gets its fair test.
- [ ] Repeat composition on the LoRA arm (less in-domain than full FT —
      AS-Norm/PLDA may retain more value there).

## 6. Reproduction

```bash
# 1. Cohort + PLDA lists (from staged train files, seed 42):
#    p2_cohort.txt (1 utt/spk), p2_plda_list.txt (<=20 utts/spk)  — generated
#    from p3_train_list.txt; see git history of this doc for the snippet.

# 2. Embeddings for cohort+PLDA files (GPU node, ~30 min on a T4):
ssh compute-node-2 'cd /home/anuraj/sl_spv_bench && \
  /home/anuraj/anaconda2025/envs/SL_SPV/bin/python code/extract_files.py \
    --root train_audio --lists p2_cohort.txt p2_plda_list.txt \
    --models speechbrain_ecapa redimnet:b1 redimnet:b6 \
    --cache_dir emb_cache_train --device cuda'

# 3. Ablation (head node, CPU, minutes):
python tools/backend_adapt.py \
  --test_cache /home/anuraj/sl_spv_bench/emb_cache \
  --train_cache /home/anuraj/sl_spv_bench/emb_cache_train \
  --trials test_list_si.txt test_list_ta.txt \
  --cohort p2_cohort.txt --plda_list p2_plda_list.txt \
  --models speechbrain_ecapa redimnet:b1 redimnet:b6 \
  --out results/p2_backend.json
```

## 7. Notes / risks

- Closed-set caveat inherited from v0 (eval speakers = train speakers,
  utterance-disjoint): PLDA sees the trial speakers at fit time, which is
  *standard* for domain-adaptation PLDA but must be stated. The v1 SLCeleb
  benchmark provides the open-set condition.
- `plda_dim=150` respects the small-speaker-count constraint
  (2·dim ≤ 527 speakers, cf. `sl_p0_baseline.yaml` comment).
- K=300 of 527 cohort follows the repo default (`as_norm_top_k: 300`).
