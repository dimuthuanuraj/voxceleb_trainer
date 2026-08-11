# P3 — Parameter-Efficient Fine-Tuning of WavLM for Sinhala/Tamil SV

**Date started:** 2026-07-03 · **Status:** COMPLETE — 4 arms (seed 42) + 3-seed replication of lora/full; all final_eval.json in
**Roadmap:** P3 of `2026-07-03-project-audit-sota-roadmap.md` ·
**Benchmark:** v0 (`2026-07-03-sl-benchmark-v0-design.md`) ·
**Baseline to beat:** zero-shot table (`2026-07-03-zeroshot-baseline-table.md`)

---

## 1. Research question (thesis RQ2)

Given a pretrained speech foundation model and a modest labeled Sinhala/Tamil
speaker set (527 speakers, 31k training utterances), **which fine-tuning
policy gives the best verification accuracy per trainable parameter — and
does parameter-efficient tuning beat full fine-tuning under language shift,
as reported for Chinese transfer?**

Literature predictions this experiment tests directly:

| Prediction | Source |
|---|---|
| PEFT (adapters/LoRA, ~2–6% params) ≥ full FT under domain/language shift | UniPET-SPK, TASLP 2025 (arXiv:2501.16542): CN-Celeb 13.12% (PET) vs 14.49% (full FT) |
| Frozen encoder + trained head already captures most of the SSL gain | WavLM paper (arXiv:2110.13900): frozen 0.617% vs full-FT 0.383% Vox1-O |
| Speaker info concentrates in shallow WavLM layers → learnable layer weights matter | WavLM SUPERB layer-weight analysis |
| LLRD (layer-wise LR decay) stabilizes full FT on small data | standard practice; repo FEATURE-005 |

## 2. Method

### 2.1 Architecture (`tools/peft_finetune.py`, new)

```
WavLM-base-plus (94.4M, frozen CNN feature extractor in ALL arms)
  └─ 13 hidden states → learnable softmax layer weights → weighted sum
       └─ Attentive Statistics Pooling (128-d bottleneck) → mean‖std
            └─ Linear → 192-d embedding
                 └─ AAM-softmax head (527 classes, margin 0.2, scale 30)
```

Backbone checkpoint: `microsoft/wavlm-base-plus`, converted once to
safetensors at `/home/anuraj/sl_spv_bench/models/wavlm-base-plus`
(transformers ≥4.56 refuses `.bin` weights on torch 2.5.1 — CVE-2025-32434).

### 2.2 Arms (all seed 42, identical data/schedule; only the tuning policy varies)

| Arm | Trainable | LRs | Node/GPU |
|---|---|---|---|
| `frozen` | head only (0.49M, 0.5%) | head 1e-3 | node1 T4 |
| `lora` | LoRA r=8 α=16 on q/k/v/out projections + head — **1.08M, 1.1%** | LoRA 5e-4, head 1e-3 | node3 A10:0 |
| `full` | encoder + head (90.68M, 95.6%) | encoder 1e-5, head 1e-3 | node4 A40 |
| `llrd` | encoder + head, LR decayed ×0.9 per layer from top | top 1e-5 → bottom ~3e-6, head 1e-3 | node3 A10:1 |

### 2.3 Training

- Data: `p3_train_list.txt` — the v0 train split capped at 60 utts/speaker
  (31,049 utts / 527 speakers = 478 si + 49 ta), staged to
  `/home/anuraj/sl_spv_bench/train_audio` (3.5 GB; compute nodes lack the
  NFS project mounts).
- Random 2.0 s crops, batch 64, AdamW (wd 2e-5), ExponentialLR γ=0.95,
  AMP mixed precision, grad-clip 5.0, 15 epochs.
- **Train/test disjointness:** the trial files come from the held-out 20%
  utterance split (sl_dataprep, seed 42); the train list never contains them.
- Note the closed-set caveat: the 527 eval speakers ARE the training
  speakers (per-utterance split, sl_dataprep convention). This measures
  representation adaptation, not open-set generalization; v1 (SLCeleb, with
  its own dev/test speaker split) will provide the open-set condition.

### 2.4 Evaluation

- Per epoch: cosine EER on stratified 2,000-pair subsamples of
  `test_list_si.txt` / `test_list_ta.txt` (seed-stable); best pooled-EER
  checkpoint kept.
- At end: best checkpoint on the FULL 12,000-pair lists — directly comparable
  to the zero-shot table. EER_avg primary; minDCF at p=0.01 and 0.05.

## 3. Reference points (zero-shot, same trial lists)

| System | si EER | ta EER |
|---|---|---|
| ReDimNet-B6 (best zero-shot) | 2.71% | 1.48% |
| SpeechBrain ECAPA | 4.78% | 3.21% |

The P3 arms must beat **frozen** to justify their extra cost, and the
interesting comparison is `lora` vs `full`: if LoRA ≥ full here, the
UniPET-SPK finding extends to Sinhala/Tamil.

## 4. Results (seed 42) — full 12,000-pair lists, cosine, no score norm

| Arm | Trainable | Best epoch | si EER_avg | ta EER_avg | si minDCF .01 | ta minDCF .01 |
|---|---|---|---|---|---|---|
| frozen | 0.49M (0.5%) | 15 | 3.74% | 3.13% | 0.286 | 0.289 |
| lora | 1.08M (1.1%) | 11 | 3.17% | 2.44% | 0.276 | 0.276 |
| full | 90.68M (95.6%) | 15 | **1.63%** | **1.30%** | 0.180 | 0.175 |
| llrd | 90.68M (95.6%) | 12 | 2.40% | 1.81% | 0.238 | 0.275 |

### 4.1 Seed replication (arms: lora, full; seeds 42/43/44)

| Arm | Seed | si EER | ta EER |
|---|---|---|---|
| lora | 42 | 3.17 | 2.44 |
| lora | 43 | 3.94 | 3.63 |
| lora | 44 | 4.08 | 3.24 |
| **lora mean±std** | | **3.73 ± 0.49** | **3.11 ± 0.61** |
| full | 42 | 1.63 | 1.30 |
| full | 43 | 1.69 | 1.46 |
| full | 44 | 1.75 | 1.48 |
| **full mean±std** | | **1.69 ± 0.06** | **1.41 ± 0.10** |

Full FT is not only better on average (si 1.69 vs 3.73, ta 1.41 vs 3.11 —
a ~2.2× EER gap) but also far more stable across seeds (±0.06 vs ±0.49 si).
The seed-42 LoRA number (3.17) is the *best* of its three seeds, so the
single-seed table above flatters LoRA; the multi-seed gap is larger.
(A safety copy `p3_frozen_s42.orig_backup/` exists next to the frozen run;
its `final_eval.json` is identical to the live one — no discrepancy.)

Zero-shot reference on the same lists: ReDimNet-B6 2.71% / 1.48%;
SpeechBrain ECAPA 4.78% / 3.21%.

Verdicts at seed 42 (all four arms final):
- **Ordering: full (1.63/1.30) ≻ llrd (2.40/1.81) ≻ lora (3.17/2.44) ≻
  frozen (3.74/3.13).** Touching the encoder monotonically helps with the
  amount of encoder freedom — the UniPET-SPK "PEFT beats full FT" result does
  NOT reproduce on this closed-set v0 benchmark (see caveat §2.3: closed-set
  eval removes full-FT's usual overfitting-to-source penalty).
- **LoRA has high seed variance (±0.5–0.6 EER across 3 seeds)** — any
  PEFT-vs-full claim needs the multi-seed table, single-seed comparisons here
  would be unsound.
- **frozen was still improving at epoch 15** (best = last epoch): the probe
  is undertrained at this schedule; its number is a lower bound.
- Even the frozen probe (0.5% params) lands within ~1 EER of zero-shot
  ReDimNet-B6 on si — the WavLM features carry most of the signal;
  everything above frozen is adaptation gain.

Earlier notes (pre-completion):
- **Full fine-tuning of WavLM-base+ on 31k in-language utterances beats the
  best zero-shot supervised model** (si 1.63% vs 2.71% = 40% relative; ta
  1.30% vs 1.48%) and improves minDCF(.01) from 0.24 → 0.18 (si).
- **LoRA (1.1% of params) recovers most of the gap to full FT** (3.17/2.44 vs
  1.63/1.30) and beats zero-shot ECAPA everywhere, but — unlike the
  UniPET-SPK CN-Celeb finding — does NOT beat full FT here at seed 42.
  Plausible reasons to probe: closed-set eval favors full FT (no
  overfitting-to-source penalty since eval speakers are the training
  speakers); r=8 may be undersized; head-only LRs may need retuning for the
  LoRA arm. The llrd/frozen arms + a v1 (SLCeleb, open-set) rerun will
  arbitrate.
- LoRA's best epoch was 11 of 15 (early plateau) vs full FT still improving
  at 15 — full FT may gain from a longer schedule.

Artifacts per arm: `/home/anuraj/sl_spv_bench/runs/p3_<arm>_s42/`
(`history.json` per-epoch curves, `best.pt` checkpoint, `final_eval.json`).
Logs: `runs/p3_<arm>_s42.log`.

## 5. Analysis plan

- [x] PEFT-vs-full verdict under language shift (the headline claim):
      **full FT ≻ LoRA at every seed** (3-seed means si 1.69 vs 3.73,
      ta 1.41 vs 3.11) — the UniPET-SPK direction does NOT reproduce on v0.
      Must be reported with the closed-set caveat (§2.3); v1 open-set rerun
      is the arbiter.
- [x] Marginal value of touching the encoder: monotone in encoder freedom —
      frozen 3.74/3.13 → lora 3.17/2.44 (s42) → llrd 2.40/1.81 →
      full 1.63/1.30 (si/ta EER).
- [x] Fine-tuned WavLM-base+ vs zero-shot ReDimNet-B6: yes — 31k in-language
      utterances take a 95M English-pretrained SSL model past the best
      off-the-shelf supervised model (si 1.69±0.06 vs 2.71; ta 1.41±0.10 vs
      1.48, the latter within noise).
- [x] 3-seed replication of lora and full (§4.1); frozen/llrd left at one
      seed — both are far from the decision boundary between arms.
- [ ] Learned layer-weight distribution per arm (are shallow layers dominant,
      as WavLM reports for speaker tasks?) — read `layer_weights` from best.pt.
- [ ] Per-language asymmetry: si has 10× the speakers of ta; check whether
      fine-tuning helps ta (49 spk) or overfits it (needs per-language
      breakdown beyond the trial-list means above).

## 6. Reproduction

```bash
PY=/home/anuraj/anaconda2025/envs/SL_SPV/bin/python   # has transformers 4.57 + peft 0.19

# 0. One-time: safetensors backbone (see §2.1) and staged data:
#    - train: cap train_list at 60 utts/spk (seed 42) -> p3_train_list.txt,
#      rsync -aL the files to /home/anuraj/sl_spv_bench/train_audio
#    - eval audio + lists: already staged for P1 (SL_ZEROSHOT_RUNBOOK.md §4b)
#    - code: cp tools/peft_finetune.py tuneThreshold.py -> sl_spv_bench/code/

# 1. One arm (repeat with --mode frozen|lora|full|llrd, matching GPU):
ssh compute-node-4 'cd /home/anuraj/sl_spv_bench && \
  nohup /home/anuraj/anaconda2025/envs/SL_SPV/bin/python code/peft_finetune.py \
    --mode lora --backbone models/wavlm-base-plus \
    --train_list p3_train_list.txt --train_root train_audio \
    --trials test_list_si.txt test_list_ta.txt --eval_root audio \
    --epochs 15 --batch_size 64 --num_workers 8 --seed 42 --device cuda \
    --out_dir runs/p3_lora_s42 > runs/p3_lora_s42.log 2>&1 &'

# 2. Smoke test first (validates data/arm/eval path in ~1 min):
#    add: --epochs 1 --limit_batches 3 --eval_subsample 60 --batch_size 16
```

## 7. Risks / notes

- T4 (frozen arm) is ~3–4× slower than the A40; expect multi-hour wall time.
- Closed-set caveat (§2.3) must be repeated in any write-up of v0 numbers.
- WavLM-base-plus pretraining is English-heavy (94k h EN); a multilingual
  backbone (w2v-BERT 2.0) is the natural follow-up if LoRA wins here.
- SLCeleb (v1) rerun: same script, new lists — the protocol and code need no
  changes (user will place SLCeleb files; see runbook §6 for ingest notes).
