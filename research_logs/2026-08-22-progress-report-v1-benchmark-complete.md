---
title: "Sinhala/Tamil Speaker Verification --- Progress Report"
subtitle: "Benchmark v1 complete: 54/54 trained, 54/54 scored on held-out test"
author: "SL-SPV project"
date: "2026-08-22"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
toc: true
toc-depth: 2
---

\newpage

# 1. Where this sits, and what changed since 17 August

This continues the programme recorded in
`2026-08-17-progress-report-architecture-frontend-and-corpus-study.md`, which is
**not superseded and has not been altered** --- it remains the record of what was
known on 17 August. This report states what changed.

| Phase | Date | Established |
|---|---|---|
| P0--P3 | 2026-07-03 | Benchmark **v0**, closed-set, SLR52 + SLR65. Thesis ch. 1--4 and `papers/ieee_spl` written against it. |
| Dataset build + QC | 2026-08-10 | Eight corpora unified; two speaker-identity bugs found; Tamil pool 50 -> 752 speakers. |
| Benchmark v1 (interim) | 2026-08-12--17 | Staged study launched. **50/54 trained, 24/54 scored.** All headline numbers were *validation* EER. |
| **Benchmark v1 (final)** | **2026-08-18--22** | **54/54 trained, 54/54 scored on the held-out test set, every rebuild verified against its checkpoint.** |

The 17 August report carried an explicit caveat: *"All figures are validation EER
on single runs. They become claims only after the held-out evaluation and the
paired bootstrap complete."* That condition is now met.

## 1.1 The four things that had to be fixed first

Completing the benchmark was not a matter of waiting. Four defects stood between
the 17 August state and a trustworthy table, and three of them would have
produced **confident, plausible, wrong numbers** rather than an error.

| # | Defect | Consequence had it stood |
|---|---|---|
| 1 | Validation dataloader deadlock in the two `ssl_wavlm_ecapa` runs | Both had been stuck since 16 August --- 6 attempts, always at the first validation after a resume. Stage H would have been reported as 2/4. |
| 2 | `si_pooled` keyed on condition *name*, not on data | `KeyError`; all seven pooled-Sinhala runs unevaluable, losing the read-vs-wild contrast entirely. |
| 3 | `--ssl_*` arguments absent from `resolved_parameters` | Evaluation rebuilt SSL models from **model defaults**: `ssl_layer` fell back to `-1`. `F_ssl_wavlm_low`, trained on layer 0, would have been *scored on layer 12* --- every tensor matching, no warning. |
| 4 | Trainer argparse defaults not applied at rebuild | `--encoder_type` defaults to `SAP` in the trainer and `ASP` in the model class. SSL models were rebuilt with ASP (1536-dim) against SAP (768-dim) checkpoints: 229 encoder tensors loaded, `bn.*` and `fc.weight` did not, leaving the final projection at **random initialisation**. Nine evaluations reported EERs from partly-random weights. |

Defects 3 and 4 share one root cause and it is worth naming precisely: **the
trainer passes its entire parsed namespace to the model, so a flag nobody
mentions still reaches the model carrying the trainer's default.** Any evaluator
that rebuilds from a manifest instead of from that namespace inherits the
*model's* defaults instead, and the divergence is silent whenever it does not
change a tensor shape.

Fixes, in `experiments/tools/evaluate.py`:

* `argv_parameters()` recovers arguments from `command.json`, the literal argv.
* `trainer_defaults()` AST-parses `trainSpeakerNet.py` for all 82 argparse
  defaults --- closing the whole class rather than the two flags observed.
* A skipped backbone tensor is now a **hard failure**, not a warning. It should
  never have been a warning: a wrong number is worse than no number.
* A prior `test_eval.json` is archived to `eval_history/` before any overwrite.

**All 54 experiments now rebuild with zero backbone mismatch.**

\newpage

# 2. Headline results (held-out test, speaker-disjoint)

## 2.1 Best system per language

| Language | System | Cosine EER | AS-Norm EER | minDCF |
|---|---|---|---|---|
| Sinhala | `ssl_mhubert_lw` (layer-weighted mHuBERT-147) | **2.369** | **2.178** | 0.156 |
| Tamil | `ssl_mhubert_ecapa` (mHuBERT + ECAPA head) | 0.402 | **0.321** | 0.026 |
| Tamil (cosine) | `ssl_wavlm_ecapa` | **0.392** | 0.392 | 0.037 |

Against the strongest mel-filterbank baseline (`ecapa1024`: 4.375 si / 0.884 ta),
the layer-weighted SSL front end is **46 % better on Sinhala** and the SSL+ECAPA
hybrid **55 % better on Tamil**.

## 2.2 The layer result, now on held-out test

The 17 August report predicted a mid-stack peak and was falsified. The held-out
test set confirms the falsification and sharpens it:

| Front end | si EER | ta EER |
|---|---|---|
| layer 0 (lowest) | 4.033 | 1.125 |
| layer 6 (mid) | 5.525 | 1.898 |
| last layer (the conventional choice) | 8.247 | 4.620 |
| **13 learned layer weights** | **3.297** | **0.623** |

Speaker information falls **monotonically with depth**, and the conventional
`--ssl_layer -1` is the worst available choice. Learned weighting beats every
single layer, so the gain is not merely "use a low layer" --- the mixture carries
information no individual layer does.

## 2.3 A new result: the last-layer penalty scales with linguistic distance

Reading the same frozen WavLM at its last layer costs almost nothing in English
and is catastrophic in Tamil:

| Language | SSL last layer | mel ECAPA-1024 | penalty |
|---|---|---|---|
| English (`en_matched`) | 7.880 | 7.580 | **1.04x** |
| Sinhala | 8.247 | 4.375 | **1.88x** |
| Tamil | 4.620 | 0.884 | **5.23x** |

WavLM is pretrained on English. Its upper layers specialise toward
English phonetic content, so discarding the lower layers removes speaker
information that the English task can partly recover from context and the Sri
Lankan languages cannot. **This is the clearest evidence in the study that
last-layer SSL practice is an English-centric default, and that its cost is a
function of the target language.** It also explains why the English control was
worth running: without it this would have looked like an SSL failure rather than
a transfer failure.

\newpage

# 3. Domain, channel and corpus effects

## 3.1 Read speech versus in-the-wild Sinhala

The `si_pooled` condition trains on read + in-the-wild Sinhala and is scored
separately on each:

| Backbone | read | wild | ratio |
|---|---|---|---|
| `ecapa1024` | 3.297 | 16.501 | 5.0x |
| `ecapa512` | 3.357 | 16.978 | 5.1x |
| `resnetse34v2` | 3.811 | 18.887 | 5.0x |
| `mlpmixer` | 4.275 | 17.638 | 4.1x |
| `vggvox` | 5.384 | 23.660 | 4.4x |
| `resnetse34l` | 5.212 | 21.527 | 4.1x |
| `ssl_wavlm` | 7.965 | 27.508 | 3.5x |

Genuine cross-session audio is **4--5x harder** than read speech for every
backbone. Trained on in-the-wild Sinhala alone (`si_celeb`, 82 speakers) the
task is harder still --- 19.7 % to 27.3 % EER. Any deployment claim resting on
read-speech numbers is overstated by roughly a factor of five.

## 3.2 Tamil's advantage is partly, but not wholly, a channel artefact

SLR127 pools three collection sites; a same-batch-only impostor list removes the
channel cue:

| System | pooled list | same-batch | inflation |
|---|---|---|---|
| `ecapa1024` | 0.884 | 1.301 | 1.47x |
| `ssl_mhubert_lw` | 0.422 | 0.562 | 1.33x |
| `ssl_mhubert_ecapa` | 0.402 | 0.505 | 1.26x |
| `ssl_wavlm_ecapa` | 0.392 | 0.563 | 1.44x |

The effect is consistent (1.26--1.47x) and must be reported, but Tamil remains
substantially easier than Sinhala even after control: 1.30 % against 4.38 % for
the same backbone. The asymmetry is real; its *magnitude* was inflated.

## 3.3 English control: the architecture ranking does not transfer

| Backbone | English `en_matched` | Sinhala | Tamil |
|---|---|---|---|
| `ecapa1024` | 7.580 | 4.375 | 0.884 |
| `resnetse34v2` | **7.260** | 4.708 | 1.376 |
| `ssl_wavlm` (last) | 7.880 | 8.247 | 4.620 |
| `vggvox` | 10.420 | 5.847 | 1.949 |

On English, `resnetse34v2` edges `ecapa1024` and last-layer SSL is competitive.
On Sinhala and Tamil, `ecapa1024` leads and last-layer SSL collapses. **An
architecture ranking established on English does not transfer to these
languages**, which is the direct answer to the question that motivated the
English arm.

\newpage

# 4. Status of the proposed architectures (A1--A9)

`proposals/` implements nine architectures from the literature study, as
self-contained code that never modifies `voxceleb_trainer/`. `selftest.py`
reports **13 passed, 0 failed**, including the isolation assertion.

Implementation is not evidence, and the two must not be conflated:

| | Implemented | Runner | Run on real data |
|---|---|---|---|
| A2 CC-NAP | yes | yes | **started 22 Aug** |
| A3 SL-AMEC | yes | yes | **yes** (14 result files) |
| A1, A4, A5, A6, A7, A8, A9 | yes | **no** | **no** |

So **one of nine** has real results, a second is now running, and seven have
never been executed against a trained model. The plan for those is section 6.

\newpage

# 5. What is now safe to claim, and what is not

**Safe.** Every number in sections 2 and 3 is held-out test EER on
speaker-disjoint splits, from a model verified to match its own checkpoint.

**Not yet safe.** Single-seed results. The power analysis in the 17 August report
still binds: effective sample size is capped by *speaker* count, not trial count
(`n_eff -> S/rho`), so a single absolute EER carries roughly +/-5--6 pp at
S = 91. Differences of the size seen between `ecapa1024` and `ecapa512`
(0.09 pp) are **not** resolvable; differences of the size seen between
`ssl_mhubert_lw` and `ecapa1024` (2.0 pp on Sinhala) are. The paired
speaker-clustered bootstrap over identical trials is the correct instrument and
is running.

**Explicitly still open.** The §4.2 arbiter from the 17 August report ---
layer-weighted *and* fine-tuned WavLM on open-set data --- remains unrun. v1
measured layer-weighted+frozen (2.369 si) and single-layer+fine-tuned (5.404 si)
but never the diagonal that P3's headline claim actually rests on.

\newpage

# 6. Next steps, in priority order

1. **The arbiter run.** Layer-weighted + fine-tuned, open-set. One run per
   language. Directly tests P3's "full fine-tuning beats LoRA" claim under v1's
   protocol; that claim currently underpins `papers/ieee_spl`.
2. **A7 PLLW** (per-language layer weighting). The cheapest high-value proposal
   and the direct extension of this report's strongest result: if the optimal
   layer differs between Sinhala and Tamil, one shared weight vector is
   leaving performance on the table. ~39 extra parameters.
3. **A9 ARI-SubCenter.** A loss-only change, so it reuses the existing recipe
   unchanged. Motivated directly by the QC audit: sub-centres absorb the label
   noise the audit proved is present.
4. **Seeds.** Three seeds for the top three systems per language, to convert
   single-run numbers into intervals.
5. **A2/A3 completion**, then the trainable proposals A1/A4/A5/A6/A8.

\newpage

# 7. Reproduction

```bash
# status of everything, any time
python3 experiments/tools/status.py
python3 experiments/tools/status.py --watch 60

# outstanding evaluations spread over every free GPU
python3 experiments/tools/eval_dispatch.py --run

# aggregate + paired bootstrap
python3 experiments/tools/analyze.py --n-boot 2000 --out experiments/analysis/v1-final

# corpus copy to the dedicated ricproject5 spindle (waits for an idle cluster)
python3 experiments/tools/copy_to_ricproject5.py
```

Prior evaluations are preserved in `experiments/results/*/eval_history/` and in
`experiments/results_backup/pre-redo-20260822-094650/`. No previous report has
been modified or removed.
