---
title: "Speaker Verification for Sri Lankan Languages --- Annual Research Progress Report"
subtitle: "May 2025 -- August 2026: every architecture, its justification, its predicted outcome, and its measured outcome"
author: "Dimuthu Anuraj"
date: "2026-09-10"
geometry: margin=2.3cm
fontsize: 10pt
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
toc: true
toc-depth: 3
---

\newpage

# 1. Scope, sources, and how to read this report

## 1.1 What this covers

Sixteen months of work, **May 2025 to August 2026**, across three successive
research programmes that share a subject (speaker verification) but not a
dataset, a protocol, or a standard of evidence:

| Phase | Period | Subject | Data |
|---|---|---|---|
| **I** | May -- Dec 2025 | Custom lightweight architectures + knowledge distillation | English VoxCeleb (mini and full) |
| **II** | Jan -- Jun 2026 | Advanced feature engineering (V3--V6), SSL, deployment | English VoxCeleb (**planned**) |
| **III** | Jul -- Aug 2026 | Sinhala/Tamil speaker verification | 8 corpora, si + ta + English control |

The report gives, for every architecture in all three phases: **why it was
chosen, what it predicted, how it was revised, what it actually measured, and
how the two compare**. Failed results are given the same space as successful
ones --- in this programme they carry more information, and section 8 is devoted
to them.

## 1.2 Evidence standard used here

Every number below is tagged by the strength of its provenance. This matters
more than usual for this project, for a reason section 5 documents in full.

| Tag | Meaning |
|---|---|
| **[M]** | Measured: a result file, log, or evaluation JSON exists in the repository and was read for this report |
| **[R]** | Reported: appears in a written project document, consistent with other records, but the raw artefact was not located |
| **[U]** | Unverified: appears in a project document and is **contradicted or unsupported** by the repository and by the July 2026 audit |
| **[P]** | Predicted / planned: a target stated before the work, used here as the comparison baseline |

Sources read for this report: 17 research logs (2025-10-26 to 2026-08-23), the
two prior period reports, the July 2026 project audit, the v1 analysis output
(`experiments/analysis/v1-final/`, 54 experiments), the proposals tree
(A1--A9), `transformer_sv/` (T1--T6), `feature_level_testing/`, the dataset QC
reports, and the git history of `voxceleb_trainer`.

## 1.3 The single most important thing in this report

Read section 5 before quoting any number from the January--June 2026 period.
**The results reported for that period are not reproducible from the repository
and are contradicted by the project's own audit of 3 July 2026.** Everything
after that audit was rebuilt to a different standard of evidence, and the
difference between the two halves of this year is the programme's main
methodological finding.

\newpage

# 2. Timeline at a glance

```
2025   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
             |     |     |     |     |     |     |
Infrastructure     [=========]
Dataset/VoxCeleb   [==============]
Mini-dataset             [==]
Throughput opt.             [=========]
ResNetSE34L baseline           [=====]
LSTM+AE teacher                   [=====]
MLP-Mixer V1 (MSE)                     [==]      <- FAILED
MLP-Mixer V2 (cosine)                    [==]
V2_Large / V2_Large_LowAlpha              [==]
NestedSpeakerNet                          [=]    <- FAILED (3 attempts, NaN)

2026   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug
       |     |     |     |     |     |     |     |
V3-V6 programme   [============ PLANNED, NOT EVIDENCED ==]
EN baseline       [....... BLOCKED: 564 missing wavs .....]
                                          |
PROJECT AUDIT ---------------------------[*] 2026-07-03
                                          |
Benchmark v0 (P0-P3)                      [=]
  zero-shot table / AS-Norm+PLDA / PEFT    |
Corpus build + QC audit                        [=]
Benchmark v1 (open-set, 54 experiments)        [======]
Evaluation-integrity rebuild                      [=]
Proposals A1-A9                                   [==]
transformer_sv T1-T6                               [=]
feature-level testing                              [=]
```

**Three architectures reached a verified, held-out, open-set number in the
entire sixteen months**, and all three did so in the final six weeks. The rest
of this report explains why, and what the other thirteen months established.

\newpage

# 3. Phase I --- English VoxCeleb (May -- December 2025)

## 3.1 Research question

> Can a lightweight student architecture, distilled from a stronger teacher,
> retain the teacher's accuracy at a fraction of its inference cost?

The framing was deployment-driven: a speaker verification model small enough
for an edge device, trained on English VoxCeleb because that is where the
labelled data and the published baselines are.

## 3.2 Infrastructure and the mini-dataset **[M]**

Before any architecture was compared, the experimental platform was rebuilt for
turnaround time. This was correct sequencing and it paid for itself.

| Measure | Before | After | Factor |
|---|---|---|---|
| Training throughput | 329 samples/s | 809 samples/s | **2.46x** |
| GPU utilisation | 51 % | 87 % | +36 pp |
| Validation (10k pairs) | 38 min | 4 min | **9.5x** |
| 100 epochs, full data | 6.5 days | 2.1 days | -67 % |

Techniques: LRU file cache (1,000 files, 40--45 % hit rate), 8 dataloader
workers with persistent workers and prefetch, FP16 mixed precision,
non-blocking host-to-device transfers, gradient accumulation.

**The mini-VoxCeleb1 dataset (140 speakers, ~28,000 utterances, 3.2 GB)** was
the highest-leverage decision of the phase: **~45x faster per epoch** (8 s vs
6 min), turning a 10-hour training run into 13 minutes. Every architecture
comparison in Phase I was run on it.

> **Scientific justification, and its limit.** Reduced-scale development is
> sound *if* relative rankings are preserved. The Phase I documents assert this
> and offer supporting evidence, but never test it directly --- no architecture
> was ranked on both mini and full data. Section 5 shows this assumption was
> later violated in a way that mattered: V1 and V2_Large ran on full VoxCeleb2
> while V2 and V2_Large_LowAlpha ran on mini, and the four were then compared in
> one table as though the dataset were a controlled factor. It was not.

## 3.3 Architecture 1 --- ResNetSE34L (baseline) **[M]**

**Why chosen.** Established, reproducible speaker-verification backbone already
present in the trainer; squeeze-excitation residual blocks with self-attentive
pooling. Chosen as a *reference*, not a contribution.

**Result: 15.48 % EER** on mini-VoxCeleb1 (140 speakers), stable for 61 epochs,
1.50 M parameters.

**Recorded inconsistency.** The July--December 2025 period report describes the
baseline as "34 layers, 6.8 M parameters" and elsewhere quotes ResNetSE34L at
"~8--10 % EER, ~20 M params". The research logs of 29--30 December give 1.50 M
parameters and 15.48 % EER. The logs are the primary record and are used here.
**The period report's figures for this model should be corrected.**

## 3.4 Architecture 2 --- LSTM + Autoencoder (teacher) **[M/R]**

**Why chosen.** Two hypotheses, both defensible for audio:

1. **Denoising autoencoder** pre-training forces a compact representation robust
   to channel noise --- the standard argument for reconstruction as an auxiliary
   task where labels are scarce relative to acoustic variability.
2. **Bidirectional LSTM** models the temporal sequence explicitly, which a
   frame-independent CNN does not. Speaker identity has slow-varying components
   (speaking rate, prosody) that a recurrent state can carry.

**Architecture.** Denoising AE feature extractor -> 2-layer BiLSTM (256 hidden)
-> attention pooling -> 512-d embedding, AAM-Softmax head. **3.87 M parameters.**

**Result: 9.68 % EER** (best checkpoint, epoch 57), 143--150 samples/s
inference.

**Expected vs actual.** Predicted 20--35 % improvement over the ResNetSE34L
baseline; achieved **37 % relative** (15.48 % -> 9.68 %). **Prediction met.**
This is the one Phase I architecture whose outcome matched its stated
expectation, and it became the teacher for everything that followed.

## 3.5 Architecture 3 --- MLP-Mixer student, four variants

**Why chosen.** From a 2025 paper proposing a modified MLP-Mixer as a student
for transformer compression in speaker verification. Three innovations were
preserved:

| Innovation | Purpose | Scientific rationale |
|---|---|---|
| **ID Convolution** (depthwise 1-D conv + residual, before token mixing) | local temporal dependency | Token-mixing MLPs are permutation-agnostic across time; a depthwise conv restores locality, which phoneme-scale structure needs |
| **Max-Feature-Map activation** (split channels, element-wise max) | discriminative selection | Acts as a learned competitive gate, suppressing redundant channels --- established in face/speaker anti-spoofing literature |
| **Grouped projections** (`groups=4`) | parameter efficiency | 4x fewer multiplications in both mixing MLPs at modest expressiveness cost |

**Adaptation made, and why.** The paper uses raw waveform + a WavLM-Large
(600 M) teacher. This project substituted **mel-spectrogram input** and the
**LSTM+AE (3.87 M) teacher**, so results would be directly comparable with the
existing CNN/LSTM baselines on the same pipeline. That is a legitimate
adaptation, but it changed the compression ratio from ~200x (paper) to
**1.45x** --- a fact that turns out to explain two of the four results below.

**Configuration:** hidden 192, 6 blocks, expansion 3, groups 4, 512-d output.
**2.66 M parameters**, 292 samples/s (**2.04x** the teacher).

### The four variants

| Variant | Params | alpha | Distill loss | Data | Predicted **[P]** | **Measured [M]** | Verdict |
|---|---|---|---|---|---|---|---|
| **V1** | 2.66 M | 0.5 | MSE | full Vox2 | 10--11 % | **16.13 %** | **Failed** |
| **V2** | 2.66 M | 0.7 | cosine | mini | 13--14 % | **10.32 %** | Beat prediction |
| **V2_Large** | 7.84 M | 0.7 | cosine | full Vox2 | 11--12 % | **14.84 %** | **Failed** |
| **V2_Large_LowAlpha** | 7.84 M | 0.4 | cosine | mini | < 10 % | **10.11 %** | Partly met |

### V1 --- the MSE failure, and the finding that came out of it

**What happened.** Distillation loss sat at **0.000244** and fell to 0.000122 by
epoch 100. Against a classification loss of ~3.0 at alpha = 0.5, the distillation
term contributed **0.008 % of the combined loss**. The student was, in effect,
trained without a teacher, and landed at 16.13 % --- *worse than the 15.48 %
undistilled baseline*.

**Root cause, derived not guessed.** Both embeddings are L2-normalised to the
unit hypersphere. For two unit vectors,
$\lVert s - t \rVert^2 = 2(1 - \cos(s,t))$, so at a realistic cosine similarity
of ~0.8 the MSE is ~0.0002 --- three orders of magnitude below the
classification term. The gradient $\partial \text{MSE}/\partial s$ is scaled
down by the same factor.

**The fix and its effect.** Cosine distance $L = 1 - \cos(s,t)$ is bounded in
$[0,2]$ and had measured magnitude 0.235--0.420 --- **~1,360x larger**. Combined
with alpha raised to 0.7, distillation went from 0.008 % to ~77 % of the total
loss, and EER fell **16.13 % -> 10.32 %** (36 % relative).

> **Finding I-1 (holds; the strongest transferable result of Phase I).**
> *For distillation between L2-normalised embeddings, the loss must measure
> angular distance. MSE, L1 and Euclidean distance are unusable: the
> normalisation constraint compresses their dynamic range below the gradient
> scale of the classification objective.* This is a geometric fact, not a
> dataset artefact, and it survives everything else in this report.

### V2_Large --- the capacity failure

**Hypothesis [P].** More student capacity -> better modelling of the teacher's
decision boundary -> 11--12 % EER.

**Measured [M].** 14.84 % --- **4.52 pp worse than the 3x smaller V2.** The
student (7.84 M) exceeded the teacher (3.87 M) by 102 %.

**What the diagnosis found.** V2_Large achieved *better* teacher alignment
(distillation loss 0.259 vs V2's 0.312) while producing *worse* verification.
It could copy the teacher's embeddings and still not match its decision
boundaries. At alpha = 0.7, 97.8 % of the loss pulled the student toward a
teacher with **less capacity than itself** --- the teacher's knowledge acted as
a ceiling rather than a floor.

### V2_Large_LowAlpha --- hypothesis validated, benefit absent

**Hypothesis [P].** Lower alpha (0.7 -> 0.4) frees the over-capacity student to
learn from hard labels and surpass both V2 and the teacher.

**Measured [M].** 10.11 % --- **the capacity-mismatch repair worked** (+4.73 pp
over V2_Large) but the model did **not** beat V2 (10.32 %) in any meaningful
sense, at 195 % more parameters and 25 % slower inference.

> **Finding I-2 (holds, with a caveat).** *The optimal distillation weight
> tracks the student/teacher capacity ratio.* Small student (ratio 0.69):
> alpha ~ 0.7. Over-capacity student (ratio 2.03): alpha ~ 0.4. The proposed
> heuristic table is a reasonable engineering rule but was fitted on two points
> and never tested out of sample.

### A data-integrity problem in this comparison **[M]**

`V2` and `V2_Large_LowAlpha` are reported with **bit-identical validation EER at
every logged epoch** --- 12.90 % at 20, 10.75 % at 60, 10.11 % at 90, 10.32 % at
100 --- despite differing by 195 % in parameters and by 0.3 in alpha. The analysis
document itself notes them as "identical" and treats that as a scientific
result about capacity saturation.

Two independent training runs of differently-sized networks do not produce
identical numbers at four checkpoints. **The far more likely explanation is that
one run's numbers were recorded for both.** This should be resolved by re-running
before either number is used again.

There is a second, separate discrepancy: the implementation log of 30 December
records **V2 at 14.62 % EER**, while the analysis log of 30--31 December records
**V2 at 10.32 %**. Both describe the same configuration. **[U]**

## 3.6 Architecture 4 --- NestedSpeakerNet: the instructive failure **[M]**

**Why chosen.** From a paper arguing that shallow networks whose every level
receives *all* previous levels can match deep sequential networks --- more
gradient paths, better feature reuse, fewer layers. **Predicted [P]: 8--13 % EER
reduction and ~2x faster inference** against the ResNetSE34L baseline.

**Design.** 4 nested levels, 32->64->128->256->512 channels, depthwise separable
convolutions, SE blocks, multi-scale fusion, SAP/ASP pooling. **1.62 M
parameters**, 28.83 ms inference (1.09x faster than baseline --- already well
short of the 2x predicted).

### Three attempts, three failures

| Attempt | Stabilisation applied | Best EER | Outcome |
|---|---|---|---|
| **1** --- full nested | fixed 0.5x scaling, BatchNorm | 21.72 % (ep 6) | **NaN at epoch 11** |
| **2** --- stabilised | learnable sigmoid weights, GroupNorm(8), adaptive pooling, Dropout2d, batch 48, grad-clip 5.0 | **18.71 %** (ep 7) | **NaN at epoch 12** |
| **3** --- simplified | 75 % of nested connections removed (10 -> 3) | 29.03 % (ep 16) | Stable, **87 % worse than baseline** |

Baseline for comparison: **15.48 %, stable for 61 epochs.**

### Why it failed --- the analysis, which is the actual contribution

The investigation went well past "it diverged" and produced a domain-compatibility
argument with five measured criteria:

| Criterion | Vision (works) | Audio (measured here) |
|---|---|---|
| Input variance $\sigma^2$ | ~0.2 | **~5.3** (mel-spectrogram, high dynamic range) |
| Inter-level feature correlation $\rho$ | ~+0.65 | **-0.23 (anti-correlated)** |
| Batch statistic stability | same-size images | variable-length crops, F0 85--220 Hz within one batch |
| Entropy ratio $H(f_{nested})/\max H(f_i)$ | ~0.95 | **~1.47** |
| Hessian condition number | ~100 | ~10,000, **~160,000 with nesting** |

The **negative inter-level correlation** is the crux. Concatenating anti-correlated
features gives
$\lVert [f_1 \Vert f_2] \rVert = \sqrt{\lVert f_1\rVert^2 + \lVert f_2\rVert^2 + 2\langle f_1,f_2\rangle}$,
and when $\langle f_1,f_2\rangle < 0$ the aggregation *amplifies* rather than
regularises. In vision, nested connections add redundant, stabilising
information (mutual information between consecutive levels ~0.85 bits); in audio
the measured figure is ~0.31 bits, so the same connections add **conflicting**
information.

Measured gradients corroborate: max gradient 18.7 (attempt 1, epoch 10) and
**42.3** (attempt 2, epoch 11) against a clip threshold of 5.0. Clipping
truncated the symptom without fixing the ill-conditioned landscape.

### Alternative explanations tested and rejected

| Hypothesis | How it was refuted |
|---|---|
| Implementation bug | Three independent implementations, all fail; architecture test passes; inference works; only training diverges |
| Hyperparameters | 15+ combinations swept (lr 1e-4 -- 1e-2, wd 0 -- 1e-3, batch 16--64). All either collapsed (lr > 5e-4) or failed to converge |
| Longer training | NaN corruption is irreversible --- all parameters become NaN via backpropagation |
| Different loss | Loss function does not address gradient explosion through nested paths |
| More data | Baseline trains fine on the same 140 speakers; dataset size is not the binding constraint |

**Attempt 3 is the clinching argument.** Removing the nested connections
*restored stability and destroyed performance* --- the architecture is only
stable when it is no longer nested. The core innovation and the instability are
the same mechanism.

> **Finding I-3 (holds).** *Nested/dense connectivity is a domain-specific
> technique, not a universal one. It requires low input variance, positive
> inter-level correlation and stable batch statistics --- none of which
> spectrogram-based audio provides.* The architecture was archived; the July 2026
> audit later recommended removing its configs from the repository (item C7).

## 3.7 What Phase I established --- and what it did not

**Established.** The cosine-vs-MSE distillation geometry (I-1); the
capacity-dependent alpha rule (I-2); the nested-learning domain incompatibility
(I-3); a 2.46x faster training platform; a working reduced-scale development
methodology.

**Not established, and this is the honest reckoning:**

1. **Every number is English VoxCeleb.** No Sinhala or Tamil audio was involved.
2. **Every headline number is a single seed**, and the best of them (10.32 %)
   predates the `--deterministic` flag.
3. **The best result, 10.32 % EER, is roughly 12x worse than the contemporary
   state of the art at comparable size.** ReDimNet-B1 (2.2 M parameters, MIT
   licence, released weights) achieves **0.85 %** on VoxCeleb1-O; ReDimNet-B0
   at 1.0 M achieves 1.16 %. The programme was training small models from
   scratch in a race that the field had settled by fine-tuning released
   checkpoints.

That third point is the strategic finding of Phase I, and it was not recognised
until July 2026.

\newpage

# 4. Phase II --- January to June 2026: the plan

The January--June 2026 period was scoped, in the report written on 8 January
2026, as the implementation of four advanced feature-engineering variants plus
self-supervised learning and deployment. The plan itself was scientifically
reasonable and is recorded here in full, because its *design rationale* remains
valid even though its results do not stand.

| ID | Architecture | Why chosen --- scientific rationale | Predicted **[P]** |
|---|---|---|---|
| **V4** | Channel attention (Squeeze-Excitation), reduction 8 | Channel-wise recalibration lets the network suppress uninformative filterbank channels. Cheap: global pool -> 2 FC -> sigmoid gate | 8.5--9.5 % (-15--20 %) |
| **V3** | Multi-resolution STFT (256/512/1024) + attention fusion | Fixed time-frequency resolution cannot serve both transients (stops, bursts) and stable harmonics (vowel formants). Analogous to feature-pyramid networks in vision | 9.0--9.5 % (-10--15 %) |
| **V5** | Res2Net multi-scale temporal blocks (4 sub-groups) | Hierarchical within-block scales span phoneme (10 ms) to word (500 ms) in one residual unit | 8.5--9.0 % (-13 %) |
| **V6** | Complex-valued networks over complex STFT | Magnitude spectrograms discard phase. Group delay (phase derivative) tracks formant motion; literature reports ~9 % relative gain | 9.0--9.5 %, 2x params |
| **P3** | Raw waveform, SincNet learnable bandpass (80 filters) | Mel scale is a fixed psychoacoustic prior; learnable centre frequency and bandwidth can adapt to the task | test: adopt if <= 10.5 % |
| **--** | WavLM-Base+ fine-tuning (94 M, 50 % frozen) | Self-supervised pretraining on 94k h is the mechanism behind every SOTA system; from-scratch training on 2.4k h cannot compete | **2--3 %** (-70--80 %) |
| **--** | INT8 post-training quantisation | Deployment: 4x size reduction at small accuracy cost | 3.1 MB, +3 % EER |

The **WavLM reasoning was correct and important**: the report identified that
the ~9.6 pp gap to state of the art was primarily attributable to the absence of
large-scale pretraining, and that transfer learning --- not architecture
search --- was the path to closing it. That diagnosis is right, and Phase III
confirmed it independently.

\newpage

# 5. The July 2026 audit --- and the reconciliation Phase II requires

## 5.1 What the audit found

On **3 July 2026** a full audit of the codebase and research plan was carried
out against the 2022--2026 literature. Its first substantive statement:

> "**The research phase has not started.** Every EER number to date is English
> VoxCeleb. There is no Sinhala/Tamil audio in the tree and no SL EER number."

And on the state of the one English experiment then in the tree:

> "The current English baseline (`exps/EN_p0_baseline_seed42`) is **blocked**:
> 564 missing VoxCeleb2 wavs (`bad_wavs.txt` / `drop_relpaths.txt`), **8 launch
> attempts on 2026-05-27**, `result/scores.txt` empty, no checkpoints."

## 5.2 The reconciliation

The January--June 2026 report describes work that the audit, conducted five days
after that report's own period closed, states did not exist. The two documents
cannot both be right.

| Claim in the Jan--Jun 2026 report | Status against the repository and the audit |
|---|---|
| V4 SE-blocks: 8.87 % EER | **[U]** No result file, log, config or checkpoint found. Not mentioned in the audit's inventory |
| V3 multi-resolution: 9.15 % EER | **[U]** Same |
| V5 Res2Net: 8.92 % EER | **[U]** Same |
| V6 complex-valued: 9.37 % EER | **[U]** Same |
| P3 raw waveform: 10.67 % EER | **[U]** Same |
| Nested learning: 9.84 % EER, 12 layers, 4.2 M params, "validated for efficiency" | **[U]** **Directly contradicted.** The measured record (Dec 2025) is three NaN collapses, best 18.71 %, architecture abandoned. The audit lists it as "NestedSpeakerNet (failed, NaN, quarantined)" |
| WavLM fine-tuned: **2.34 % EER** on VoxCeleb1-O | **[U]** No checkpoint, no run, no log. The audit records the *only* English baseline as blocked and unrun |
| CN-Celeb cross-lingual: 12.43 % / 4.76 % | **[U]** CN-Celeb is not on the mounts; the audit lists it as a *future* option |
| VoxMovies domain evaluation: 14.23 % / 5.87 % | **[U]** No such dataset or evaluation in the tree |
| WavLM -> V4 distillation: 4.52 % EER | **[U]** Depends on two unevidenced models |
| INT8 quantisation: 3.1 MB, 4.67 % EER | **[U]** No quantised artefact |
| Android app on Galaxy S21, 85 ms latency, <2 % battery / 100 verifications | **[U]** No mobile codebase, no TFLite export, no measurement harness |
| "Interspeech 2026 paper accepted in May" | **[U]** The `papers/` tree holds *scaffolds*; the audit describes them as targets, not acceptances |

**Every substantive experimental claim of the January--June 2026 period is
unsupported by the repository, and several are contradicted by measured records
in it.**

## 5.3 What this means, stated plainly

1. **These numbers must not enter the thesis, any paper, or any subsequent
   progress report.** The nested-learning entry is the clearest case: the report
   presents as a validated efficiency result an architecture the project had
   already measured as unstable across three stabilisation attempts.
2. **The V3--V6 designs remain worth running.** Nothing in the audit says the
   architectures are bad --- only that they were not run. The rationales in
   section 4 are sound, and V4 (SE) and V5 (Res2Net) in particular are cheap.
3. **The plan's central diagnosis was right**: pretraining, not architecture, is
   the lever. Phase III confirmed it with measured data.
4. **The corrective action has already happened**, and it is the reason Phase III
   looks the way it does. Compare the July report's own framing of its purpose:

   > "the failure mode this project has repeatedly paid for is a plausible
   > number produced by a system that was not measuring what it claimed."

   Every Phase III design choice --- pre-registered predictions, held-out
   evaluation scored once, paired bootstrapping, hard-failing evaluators,
   shuffled-metadata controls, `selftest.py` isolation assertions, "a folder
   without a `RESULT.md` has not been evaluated" --- is a direct response to
   this. **The programme diagnosed and fixed its own evidence problem.** That
   is the most valuable thing that happened this year.

\newpage

# 6. Phase III --- Sinhala/Tamil speaker verification (July -- August 2026)

## 6.1 The strategic reset

The audit produced seven research-strategy findings that redirected the entire
programme:

| # | Finding | Consequence |
|---|---|---|
| R1 | Custom MLP-Mixer + LSTM-AE distillation is ~12x off SOTA at matched size | Stop the from-scratch race; keep the *methodology* findings (I-1, I-2) |
| R2 | From-scratch ECAPA on ~100 SL speakers will fail | Frozen/pretrained encoder + light head or PEFT instead |
| R3 | No zero-shot baseline was planned before training anything | Evaluate released checkpoints first --- days, not weeks |
| R4 | `lang_aux` head and DANN/GRL have **opposite** objectives on language | Choose GRL for invariance; demote lang_aux to an ablation |
| R5 | Backend/calibration levers under-planned | AS-Norm, PLDA, language-aware calibration --- all training-free |
| R6 | 100-speaker plan is smaller than available data | SLCeleb (280) + SLR52 (478) + MILE (531) + Kathbath |
| R7 | The 10.32 % headline is unreplicated, single-seed, pre-`--deterministic` | Re-run 3 seeds or retire the claim |

It also surfaced the strategic opportunity the project had been sitting on:
**SLCeleb, the group's own 280-speaker Sinhala/Tamil corpus, has no published
baseline table, and the only published Sinhala SV paper is the group's own.**
A systematic benchmark is an open, citable, low-effort contribution.

## 6.2 Benchmark v0 --- protocol first **[M]**

Built 2026-07-03 on SLR52 (478 si) + SLR65 (49 ta), addressing five evaluation
defects the audit identified:

- **Cross-session targets enforced** where session metadata exists (v0 limitation:
  OpenSLR is flat, so v0 ran single-session --- documented as optimistic).
- **Same-gender impostors** where gender is known (verified: 0 cross-gender pairs
  in Tamil).
- **EER_avg** as primary (literature standard), EER_max as diagnostic.
- **minDCF at both p=0.01 and p=0.05** --- NIST- and VoxCeleb-comparable.
- **Unlabelled trials are a hard error**, replacing a silent
  `random.randint(0,1)` label fallback.

3,000 targets + 9,000 impostors per language, seed-reproducible.

## 6.3 P1 --- the zero-shot baseline table **[M]**

**Question.** How do released VoxCeleb-trained checkpoints perform on Sinhala
and Tamil with **no adaptation**, and how large is the English->SL degradation?

**Prediction [P].** CN-Celeb precedent: 2--4x EER inflation.

| Model | Params | Published Vox1-O | si EER | ta EER | si factor | ta factor |
|---|---|---|---|---|---|---|
| SpeechBrain ECAPA | ~20 M | 0.80 % | 4.78 % | 3.21 % | 6.0x | 4.0x |
| ReDimNet-B1 | 2.2 M | 0.73 % | 3.57 % | 1.67 % | 4.9x | 2.3x |
| ReDimNet-B2 | 4.7 M | 0.52 % | 4.17 % | 1.97 % | 8.0x | 3.8x |
| **ReDimNet-B6** | 15 M | 0.37 % | **2.71 %** | **1.48 %** | 7.3x | 4.0x |

**Expected vs actual.** Tamil (2.3--4.0x) sits inside the predicted band.
**Sinhala (4.9--8.0x) exceeds it** --- and does so on *clean read speech*, which
should have been the kind condition. **Language shift alone costs a multiple,
not a margin.**

**Two secondary findings:**

- **Vox1-O rank does not fully transfer**: B1 (2.2 M) beats B2 (4.7 M) on both
  languages despite B2's better published number. Parameter count does not buy
  cross-lingual robustness monotonically --- only B6 justifies its size.
- **minDCF(0.01) is poor everywhere (0.24--0.48) while EERs look usable.** That
  is the signature of uncalibrated scores in a mismatched domain --- and it is
  what motivated P2.

## 6.4 P2 --- training-free backend adaptation **[M]**

**Question (thesis RQ4).** How much cross-lingual degradation is a fixable
*score-domain* problem rather than an embedding-quality problem?

| Prediction **[P]** | Source | **Outcome [M]** |
|---|---|---|
| AS-Norm with target-domain cohort ~20--30 % relative | Matejka 2017 | **Confirmed: 12--39 %** (5 of 6 cells in band) |
| PLDA beats cosine under domain shift | arXiv:2204.11403 | **Refuted on Sinhala**, mildly confirmed on Tamil |
| Cross-lingual trials suffer systematic score shift | arXiv:2110.09150 | **Confirmed, and larger than expected** |

**Best training-free v0 system: ReDimNet-B6 + AS-Norm --- si 2.07 %, ta 0.90 %**,
from 2.71 % / 1.48 % zero-shot. No training, one cohort file. The single most
cost-effective lever measured in the entire programme.

**The PLDA refutation and its mechanism.** PLDA *hurt* Sinhala for every model
(B6: 2.71 -> 3.84). The proposed mechanism: the PLDA fit is dominated by 478
Sinhala speakers of near-single-session crowdsourced audio, so its within-speaker
covariance underestimates real channel variability. This was flagged as a
**benchmark artefact hypothesis** rather than a refutation of the literature ---
correct scientific caution, and the P2xP3 composition later vindicated the
literature's actual claim (below).

**Cross-lingual score shift, measured.** Scoring Sinhala trials with a
Tamil-fitted calibrator inflates Cllr by up to **5.7x**.

> **Finding III-1.** *Per-language calibration is mandatory for deployment.*
> A subtlety worth publishing: **PLDA scores are the most transportable across
> languages** (cross-calibration penalty only 1.2--2.7x) even where their EER is
> worse --- extending arXiv:2110.09150 to Sinhala/Tamil.

**Composition with fine-tuning (P2 x P3).** After in-language fine-tuning,
AS-Norm's EER gain saturates but it keeps improving operating-point metrics
(minDCF si 0.180 -> 0.144, Cllr down ~15 %), and **PLDA becomes clearly
redundant on both languages** --- exactly the arXiv:2204.11403 prediction that
PLDA helps under shift and is redundant in-domain. Fine-tuning moved the system
in-domain; the v0 PLDA story is coherent after all.

## 6.5 P3 --- parameter-efficient fine-tuning of WavLM **[M]**

**Question (thesis RQ2).** Does PEFT beat full fine-tuning under language shift,
as reported for Chinese transfer?

**Architecture.** WavLM-base-plus (94.4 M, frozen CNN extractor in all arms) ->
**13 hidden states through learnable softmax layer weights** -> Attentive
Statistics Pooling -> 192-d embedding -> AAM-Softmax (527 classes).

Note the layer weighting: this design was **already correct** on the question
that would later prove most consequential (section 6.9).

| Arm | Trainable | si EER | ta EER | 3-seed si | 3-seed ta |
|---|---|---|---|---|---|
| frozen | 0.49 M (0.5 %) | 3.74 | 3.13 | --- | --- |
| LoRA r=8 | 1.08 M (1.1 %) | 3.17 | 2.44 | **3.73 +/- 0.49** | **3.11 +/- 0.61** |
| LLRD | 90.68 M | 2.40 | 1.81 | --- | --- |
| **full FT** | 90.68 M (95.6 %) | **1.63** | **1.30** | **1.69 +/- 0.06** | **1.41 +/- 0.10** |

**Expected vs actual.** Prediction: PEFT >= full FT (UniPET-SPK reports 13.12 %
vs 14.49 % on CN-Celeb transfer). **Measured: full FT beat LoRA at every seed**,
by ~2.2x, and was **8x more stable across seeds** (+/-0.06 vs +/-0.49).
**The prediction was falsified.**

Ordering is **monotone in encoder freedom**: frozen -> LoRA -> LLRD -> full.

**Why the falsification was reported honestly rather than explained away.** The
log records the closed-set caveat as the likely cause --- eval speakers *are* the
training speakers, which removes full FT's usual overfitting-to-source penalty ---
and explicitly names the v1 open-set rerun as the arbiter. It also notes that the
seed-42 LoRA number (3.17) is the *best* of its three seeds, so the single-seed
table **flatters LoRA**, not the other way round.

**Headline v0 result [M]:** 31k in-language utterances take a 95 M
English-pretrained SSL model **past the best off-the-shelf supervised model** ---
si 1.69 +/- 0.06 vs ReDimNet-B6's 2.71.

## 6.6 Corpus expansion --- from 2 corpora to 8 **[M]**

**The binding constraint identified:** at 49 Tamil speakers the EER confidence
interval swamps the effect sizes being chased. A survey of every open
Sinhala/Tamil corpus was carried out, then executed the same day.

| Built corpus | Lang | Speakers | Utts | Hours | Trials |
|---|---|---|---|---|---|
| `slr52_sinhala` | si | 478 | 185,293 | 224.5 | 12,000 |
| `slr127_tamil` | ta | 531 -> **638** | 89,401 | 150.1 | 12,000 |
| `kathbath_tamil` | ta | 60 -> **49** | 7,171 | 13.2 | 150,000 official |
| `nisp_tamil` | ta + en | 65 (all bilingual) | 4,905 | 13.3 | 8,000 + 4,000 cross-lingual |
| `slr65_tamil` | ta | 50 | 4,291 | 7.1 | 12,000 |
| `slceleb2026_sinhala` | si | 123 | 46,960 | 85.1 | real sessions |

291,061 files converted to 16 kHz mono PCM_16, **zero conversion errors**.
**Tamil speaker pool: 50 -> 706 as built** (802 after the QC corrections below).
*Note: the 17 August report quotes "50 -> 752", which reconciles with neither
figure and should be corrected.*

Three structural facts were established and matter for every claim downstream:

1. **Sinhala has almost no open audio.** SLR52 *is* the Sinhala open corpus.
   Sinhala is **absent from Common Voice** (verified against the CV 26.0 locale
   list, 294 locales, no `si`).
2. **Every large Tamil corpus is Indian Tamil, not Sri Lankan Tamil.** Excellent
   training data, useless for claiming Sri Lankan performance. The
   Indian-trained / Sri-Lankan-evaluated **dialect gap is an unpublished number
   this project is uniquely positioned to produce.**
3. **SiTa is CC BY-NC 4.0** --- non-commercial, so unusable in the VoiceID
   product path. The only such restriction in the collection and easy to forget.

## 6.7 The dataset quality audit --- two speaker-identity bugs **[M]**

A seven-layer QC pipeline (integrity, signal quality, label reliability, trial
audit, power, calibration, fairness) was run over all corpora. It found two
identity bugs **in corpora built and documented by this project**, both invisible
to every check that does not use audio.

### Bug 1 --- SLR127: the prefix is a numbering scheme, not a session

The original ingestion took field 2 as the speaker (531 distinct values,
matching the documented speaker count) and the `ISTL`/`MICI`/`MILE` prefix as a
recording session --- yielding "107 multi-session speakers" and the claim of the
only genuine cross-session read-speech trials in the collection.

**Embedding the centroids settled it:**

| Comparison | Cosine |
|---|---|
| same number, same prefix (split-half of one speaker) | **0.961** |
| same number, **different** prefix | **0.269** |
| different numbers (impostor baseline) | 0.319 |

Cross-prefix pairs score **below the impostor baseline**. `ISTL_0000202` and
`MILE_0000202` are two different people; each collection batch numbers its
speakers from scratch.

| | Before | After |
|---|---|---|
| Speakers | 531 | **638** |
| EER | **55.31 %** (worse than chance) | **1.07 %** |
| d' / NMI | 3.14 / 0.963 | **5.23 / 0.984** |
| silhouette | 0.377 | 0.463 |

The old trial list drew 3,000 "targets" across prefixes --- pairs of *different
people labelled as the same speaker* --- which is why it scored worse than
chance.

> **Finding III-2.** *"531 distinct values matches the documented 531 speakers"
> was a red herring. Counting agreement is not identity verification; only audio
> can answer an identity question.* Every label-quality metric moved in the
> direction a correct key predicts without being the quantity optimised --- about
> as close to independent confirmation as is available without ground truth.

### Bug 2 --- Kathbath: recurring split ids are the same person

11 numeric ids occur in both `valid` and `test_known` and had been namespaced
apart. Cross-split cosine: **0.992** (baseline 0.348). They are the same people.
**49 speakers, not 60.** Training labels merged; official trial paths left
untouched to preserve comparability.

### Other findings from the audit

- **Channel is nearly identity in the read corpora.** $\eta^2(\text{SNR}\mid\text{speaker})$
  = **0.91** in SLR127 and 0.68 in SLR52 --- a model can score recording
  conditions rather than voice. Correcting bug 1 *raised* this from 0.74 to 0.91,
  the expected direction (the confound was partly hidden by the bug).
- **The channel probe.** Multinomial logistic regression predicts
  **corpus-of-origin from the speaker embedding at 91.2 % accuracy** against 20 %
  chance. *No single-corpus absolute EER should ever be quoted as a performance
  claim.*
- **Trials do not buy power; speakers do.** Kathbath: 50,000 official trials, 20
  speakers, **MDE 4.5--6.2 pp**. SLR127: 12,000 trials, 638 speakers, **MDE
  0.26 pp** --- 17x better resolution from a quarter of the trials.
- **Kathbath's official lists contain 351 self-pairs** (a file paired with
  itself), which score cosine 1.0 by construction and bias the published EER
  low by ~1.4 % of targets. Left untouched deliberately --- altering them would
  break comparability with every published IndicSUPERB number --- but now quoted
  with the caveat attached automatically.
- **First cross-lingual number in the project.** NISP: same speakers score
  1.10 % within Tamil and **2.90 %** Tamil<->English --- a **2.6x** degradation
  from the language switch alone, on identical speakers and one embedding.
- **Fairness cannot be assessed on the two largest corpora** --- SLR52 and SLR127
  ship no gender labels. Where measurable, FDR >= 0.97 (no material gender
  disparity).

### The same-recording confound, measured directly

SLCeleb 2026 V3 is the only corpus in the collection with genuine sessions
(400 YouTube videos; 100 of 123 speakers span 2+):

| Pair type | Mean cosine |
|---|---|
| Same speaker, **same** session | 0.612 |
| Same speaker, **different** session | 0.474 |
| Different speakers | 0.136 |

Allowing same-session targets gives d' = 3.46; restricting to cross-session
gives d' = 2.44. **The confound inflates apparent separability by 42 %**, on
identical speakers and one embedding. Every other corpus maps one session per
utterance, so their EERs carry optimism of about this size.

## 6.8 Benchmark v1 --- the open-set rebuild **[M]**

Three findings landed **before a single full experiment ran**, and all three
changed how results must be produced.

### 6.8.1 Every shipped corpus list was 100 % closed-set

```
corpus                train spk   test spk   overlap
slr52_sinhala             478        478     478  (100 %)
slr127_tamil              638        638     638  (100 %)
slceleb2026_sinhala       123        123     123  (100 %)
slr65_tamil                50         49      49  (100 %)
```

**Why this is worse than "optimistic".** For a speaker seen in training, $f$ has
been explicitly optimised to collapse that speaker onto its AAM-Softmax centroid
$w_c$, so the margin enforced at training time transfers directly into the trial
score. **The inflation grows with capacity to memorise centroids** --- a 95 M SSL
front end gains more from it than a 1.4 M ResNet. A closed-set ranking of
architectures is therefore partly a ranking of *memorisation capacity*, and the
two orderings are not the same.

Rebuilt: 70/10/20 by speaker, stratified by duration, deterministic, SHA-256 per
list. Four corpora held out entirely as generalisation probes. **The shipped
`lists/` were left untouched** so prior results stay reproducible --- they are
simply not comparable.

### 6.8.2 The power ceiling

With $m$ trials per speaker, $S$ speakers, intra-speaker correlation $\rho$:

$$n_{\text{eff}} = \frac{mS}{1 + (m-1)\rho} \longrightarrow \frac{S}{\rho}$$

At $S=91$, $\rho=0.7$: $n_{\text{eff}} \le 130$, giving **~+/-6 pp on a single
absolute EER**. Two consequences, both acted on:

1. **Trial lists cut 80k -> 20k pairs.** At $m \approx 20$ the design already
   reaches 98 % of the ceiling; generating more pairs is measurement theatre.
2. **Comparison must be paired.** Writing per-speaker error as
   $e_A(s) = \mu(s) + a(s)$, the speaker-difficulty term $\mu(s)$ is common to
   both systems on identical trials and **cancels in the difference**.

Validated against synthetic data with known ground truth: EER matched the repo's
own `tuneThreshold` exactly; a known injected difference was correctly detected;
a system against itself correctly found nothing; **paired intervals were 2.6x
tighter** than absolute EER.

### 6.8.3 The research and production stacks were never the same model

`voiceid` (deployed) learns a **softmax weighting over all 13 WavLM layers**;
the trainer reads only the **last** one --- the layer masked-prediction
pretraining works hardest to strip speaker identity out of. **Nothing had
measured the difference.** This became the study's principal result.

## 6.9 Stage F and the SSL layer probe --- the principal result **[M]**

Stage A's biggest surprise was that `ssl_wavlm` (94.98 M) finished **last** on
both languages despite being 6x larger than the winner. A direct probe --- run
the frozen encoder, mean-pool each hidden state, cosine-score, **no training
at all** --- explains it:

| Layer | si EER % | ta EER % |
|---|---|---|
| **0** (CNN output) | **22.30** | **11.05** |
| 3 | 29.27 | 15.78 |
| 6 | 36.50 | 24.48 |
| 9 | 38.90 | 31.68 |
| **12** (`--ssl_layer -1`, the default) | **40.33** | **32.08** |

**Monotonic degradation with depth in both languages.** The conventional default
is the **worst available choice** --- 1.81x worse (si) and 2.90x worse (ta) than
layer 0.

**Scientific explanation.** Masked-prediction pretraining rewards recovering a
masked frame from context, which drives upper layers toward phonetic content and
treats speaker identity as *nuisance*. The last layer is the one the objective
worked hardest to make speaker-invariant.

### Front-end sweep, held-out test **[M]**

| Front end | si | ta | Family |
|---|---|---|---|
| **`ssl_mhubert_lw`** | **2.369** | **0.422** | SSL, 13 learned layer weights |
| `ssl_mhubert_ecapa` | 2.793 | **0.402** | SSL + full ECAPA backbone |
| `ssl_wavlm_ecapa` | 3.166 | **0.392** | SSL + full ECAPA backbone |
| `ssl_wavlm_lw` | 3.297 | 0.623 | SSL, layer-weighted |
| `ssl_wavlm_low` (layer 0) | 4.033 | 1.125 | SSL, single layer |
| *`ecapa1024` --- mel baseline* | *4.375* | *0.884* | filterbank |
| `mfcc40d` | 4.396 | 1.135 | cepstral |
| `mfcc80` | 4.486 | 1.125 | cepstral |
| `ssl_wavlm_ft` (unfrozen, single layer) | 5.404 | 1.788 | SSL, fine-tuned |
| `ssl_wavlm_mid` (layer 6) | 5.525 | 1.898 | SSL, single layer |
| *`ssl_wavlm` --- layer 12* | *8.247* | *4.620* | SSL, last layer |

> **Finding III-3 (the programme's headline).** *How the SSL layers are read
> matters more than which architecture reads them.* Same encoder, same data,
> same recipe: **8.247 -> 3.297 on Sinhala, a 2.5x improvement from 13 scalar
> parameters.** Against the strongest mel baseline, the layer-weighted SSL front
> end is **46 % better on Sinhala** and the SSL+ECAPA hybrid **55 % better on
> Tamil**. Paired bootstrap: si Delta = -2.006 pp, CI [-2.420, -1.550], p = 0.000.

**Learned weighting beats every fixed depth** (lw > layer 0 > layer 6 > layer 12),
so the gain is not merely "use a low layer" --- **the mixture carries information
no individual layer does.**

**Multilingual pretraining pays at equal capacity.** mHuBERT-147 (94.97 M,
147 languages including si and ta) beats WavLM-base-plus (94.98 M, English) on
both languages: si Delta = -0.927 pp (CI [-1.294, -0.654], p = 0.000), ta
Delta = -0.191 pp (p = 0.005). Because the parameter counts are matched to within
0.01 %, this **isolates pretraining coverage from capacity** --- a clean contrast
that the confounded XLS-R-300m comparison could not provide.

### A pre-registered prediction, falsified and kept

The 12 August design predicted speaker information would peak **mid-stack**, and
`ssl_wavlm_mid` was built at layer 6 on that basis. Layer 6 measures 36.5 % ---
barely better than the last layer, far off layer 0's 22.3 %. **The prediction was
wrong.** The entry was retained as the mid-curve data point and the falsification
recorded in the registry **rather than edited away.** This is what
pre-registration is for.

### A second pre-registered prediction, also falsified

MFCC was predicted to lose badly: the DCT decorrelates filterbank channels, which
is essential for diagonal-covariance GMM-UBM systems but *actively harmful* for a
CNN/TDNN that wants the inter-band correlations where formant structure lives.
**Measured: MFCC lands within ~0.2 pp of log-mel** and the paired contrasts are
not significant on Sinhala (mfcc80 vs ecapa512: Delta = -0.202, p = 0.463). The
prediction was directionally right and practically wrong.

### 6.9.1 A new result: the last-layer penalty scales with linguistic distance

| Language | SSL last layer | mel ECAPA-1024 | penalty |
|---|---|---|---|
| English (`en_matched`) | 7.880 | 7.580 | **1.04x** |
| Sinhala | 8.247 | 4.375 | **1.88x** |
| Tamil | 4.620 | 0.884 | **5.23x** |

> **Finding III-4.** *Last-layer SSL practice is an English-centric default whose
> cost is a function of the target language.* WavLM is pretrained on English, so
> its upper layers specialise toward English phonetic content; discarding the
> lower layers removes speaker information that the English task can partly
> recover from context and the Sri Lankan languages cannot. **Without the English
> control this would have looked like an SSL failure rather than a transfer
> failure** --- the direct justification for running the control arm.

## 6.10 Stage A --- architecture sweep, and the capacity null **[M]**

Seven backbones x {si, ta}, everything else pinned (embedding dim 256, 80 mels,
2.0 s train / 3.0 s eval crops, identical MUSAN+RIR augmentation, Adam 1e-3,
60 epochs, AS-Norm off during training).

| Architecture | si test | ta test | Params |
|---|---|---|---|
| **`ecapa512`** | **4.285** | 1.045 | 5.99 M |
| `ecapa1024` | 4.375 | **0.884** | 14.46 M |
| `resnetse34v2` | 4.708 | 1.376 | 7.37 M |
| `mlpmixer` | 4.930 | 1.256 | 7.71 M |
| `resnetse34l` | 4.990 | 1.708 | 1.40 M |
| `vggvox` | 5.847 | 1.949 | 3.64 M |
| `ssl_wavlm` (layer 12) | 8.247 | 4.620 | 94.98 M |

> **Finding III-5.** *This regime is data-limited, not capacity-limited.*
> ECAPA-512 matches ECAPA-1024 at 40 % of the parameters: si Delta = 0.091 pp,
> CI [-0.352, +0.415], **p = 0.835**; ta Delta = -0.151 pp, p = 0.134. **2.4x the
> parameters buys nothing measurable.** This is a null result with a tight
> interval, not an absence of evidence, and it directly justified not pursuing
> larger from-scratch models.

**MLP-Mixer, revisited.** Phase I's architecture was carried into the v1 sweep on
equal terms and placed **4th of 7 on Sinhala and 4th of 7 on Tamil** --- competent
but beaten by ECAPA at both sizes, and beaten decisively by every layer-weighted
SSL front end. It is a fair, open-set, held-out verdict on the Phase I
architecture: **respectable, not competitive.**

## 6.11 Stage H --- does the backbone still matter under SSL features? **[M]**

Stage A varies the backbone with the front end fixed; Stage F varies the front
end with the head fixed. Stage H measures the interaction neither can see.

| System | si | ta |
|---|---|---|
| `ssl_mhubert_lw` (pooling head) | **2.369** | 0.422 |
| `ssl_mhubert_ecapa` (full ECAPA backbone) | 2.793 | **0.402** |

**The answer is language-dependent, which is itself the finding.** On Sinhala the
heavier backbone *hurts* significantly (Delta = -0.423 pp favouring the pooling
head, p = 0.001); on Tamil the difference is **not significant** (Delta = 0.030,
p = 0.406). At 0.392 % (`ssl_wavlm_ecapa`), Tamil is the best single result in the
programme.

Practical reading: **once features come from a well-read SSL encoder,
representation dominates architecture.** Spend effort on the front end.

**Caveat recorded in the module:** ECAPA's dilations (2, 3, 4) were tuned for
100 Hz mel frames; SSL encoders emit **50 Hz**, so the receptive field spans twice
the intended time. Not necessarily harmful, but it travels with the front-end
change and must not be attributed to it.

## 6.12 Domain, channel and corpus effects **[M]**

### Read speech vs genuine cross-session audio

`si_pooled` trains on both Sinhala domains and scores each separately, so one
model gives a like-for-like comparison:

| Backbone | read (slr52) | wild (cross-session) | ratio |
|---|---|---|---|
| `ecapa1024` | 3.297 | 16.501 | **5.0x** |
| `ecapa512` | 3.357 | 16.978 | 5.1x |
| `resnetse34v2` | 3.811 | 18.887 | 5.0x |
| `mlpmixer` | 4.275 | 17.638 | 4.1x |
| `ssl_wavlm` | 7.965 | 27.508 | 3.5x |

> **Finding III-6 --- the most important caveat in the programme.** *Genuine
> cross-session audio is 4--5x harder than read speech for every backbone.* Any
> deployment claim resting on read-speech numbers is overstated by roughly a
> factor of five --- **including every v0 number, which used SLR52 throughout.**
> The architecture *ranking* is stable across domains, which is reassuring for
> external validity even where absolute numbers do not transfer.

### The Tamil channel artefact

SLR127 pools three collection sites. Splitting impostor trials on whether both
sides come from the same site:

| Impostor type | Count | Mean score | EER |
|---|---|---|---|
| cross-batch | 4,805 | **-0.0433** | **0.475 %** |
| same-batch | 5,151 | **+0.0527** | **1.298 %** |
| all (as reported) | 9,956 | --- | 0.894 % |

Cross-batch impostors score *negative* --- rejected almost for free because the
channel differs, not the speaker. **~48 % of the trial list is that easy kind.**
Reproduced across architectures at 1.26--3.20x inflation.

**Reporting rule adopted: quote the same-batch figure for SLR127.** Tamil remains
substantially easier than Sinhala after control (1.30 % vs 4.38 % for the same
backbone) --- **the asymmetry is real; its magnitude was inflated.**

### English control --- the ranking does not transfer

| Backbone | English `en_matched` | Sinhala | Tamil |
|---|---|---|---|
| `resnetse34v2` | **7.260** | 4.708 | 1.376 |
| `ecapa1024` | 7.580 | **4.375** | **0.884** |
| `ssl_wavlm` (last) | 7.880 | 8.247 | 4.620 |
| `vggvox` | 10.420 | 5.847 | 1.949 |

On English, `resnetse34v2` edges `ecapa1024` and last-layer SSL is competitive.
On Sinhala and Tamil, `ecapa1024` leads and last-layer SSL collapses.
**An architecture ranking established on English does not transfer to these
languages** --- the direct answer to the question that motivated the English arm.

The `en_full` positive control reached **3.496 % validation EER** on
VoxCeleb1-O, close enough to the known reference for this recipe to validate the
harness end to end --- loader, augmentation, scoring and checkpoint selection.

### A documented negative result

Trained alone on 82 speakers, `si_celeb` reached **19.7--27.3 % EER** across
every architecture with minDCF ~0.99. Training worked (loss 7.34 -> 0.66, train
accuracy 6 % -> 85 %): **the models learned their training speakers and
transferred to none.** Two design faults --- 82 speakers is far too few, and the
validation set has 9 speakers ($n_{\text{eff}} \le 13$), which cannot measure
anything yet selected every checkpoint. Retained as a negative result; its EERs
should not be quoted. Its correct use is as an *evaluation probe*, where it is
highly informative.

## 6.13 The evaluation-integrity incident --- 22 August **[M]**

Before the v1 benchmark could be closed, four defects were found between the
17 and 22 August states. **Three of them would have produced confident,
plausible, wrong numbers rather than an error.**

| # | Defect | Consequence had it stood |
|---|---|---|
| 1 | Validation dataloader deadlock in two `ssl_wavlm_ecapa` runs | Stuck since 16 Aug, 6 attempts; Stage H would have been reported as 2/4 |
| 2 | `si_pooled` keyed on condition *name*, not on data | All seven pooled-Sinhala runs unevaluable; the read-vs-wild contrast lost entirely |
| 3 | `--ssl_*` arguments absent from `resolved_parameters` | Evaluation rebuilt SSL models from **model defaults**: `ssl_layer` fell back to -1. `F_ssl_wavlm_low`, **trained on layer 0, would have been scored on layer 12** --- every tensor matching, no warning |
| 4 | Trainer argparse defaults not applied at rebuild | `--encoder_type` defaults to SAP in the trainer, ASP in the model class. **Nine evaluations reported EERs from partly-random weights** |

**The root cause, named precisely:** the trainer passes its entire parsed
namespace to the model, so a flag nobody mentions still reaches the model
carrying the trainer's default. Any evaluator rebuilding from a manifest
inherits the *model's* defaults instead, and **the divergence is silent whenever
it does not change a tensor shape.**

**Fixes:** arguments recovered from `command.json` (the literal argv);
`trainSpeakerNet.py` AST-parsed for **all 82 argparse defaults**, closing the
whole class rather than the two flags observed; **a skipped backbone tensor is
now a hard failure, not a warning** ("a wrong number is worse than no number");
prior evaluations archived before overwrite. **All 54 experiments now rebuild
with zero backbone mismatch.**

\newpage

# 7. The three parallel strands (August 2026)

## 7.1 Proposals A1--A9 --- literature designs, matched comparison **[M]**

Nine architectures were drawn from a 32-paper literature study and implemented
as self-contained code that **never modifies `voxceleb_trainer/`** (asserted by
`selftest.py` via `git status --porcelain`; 13 checks, 0 failures). Each run
copies the **literal argv** of its matched v1 baseline and overrides only what
the proposal changes, so the proposal is the single changed factor.

### A constraint discovered, not designed

`DomainTable` reported that the `si` and `ta` conditions each contain **one
language and one corpus**. A1, A6, A7 and A8 all depend on a domain variable ---
adversarial, disentangling, or per-language. With one class, every discriminator
is constant, every gradient-reversal term is uninformative, per-language vectors
collapse to one, **and each method silently degrades to its own baseline while
still producing a number.**

> **Finding III-7.** ***Four of nine proposals are undefined on the conditions
> the benchmark was built from, and nothing in their papers or READMEs says so.***

**A7 is the cautionary case**, because the guard was added only after it ran.
Trained on `ta` with `n_languages` left at its default of 3, it built three
weight vectors and a three-way language-ID head on data containing one language,
trained to completion, and scored **0.663 % against its baseline's 0.623 %**.
That reads as a small regression by a plausible margin. **It is not a
regression** --- it is shared layer weighting plus two dead vectors and a
degenerate head, measuring nothing. The run was retired with that explanation
rather than deleted.

> *A method that is undefined on the data still trains, still converges, and
> still emits a number in the right range. Only a guard derived from the data
> catches it.*

### Results

| Proposal | Design and rationale | Predicted **[P]** | **Measured [M]** | Verdict |
|---|---|---|---|---|
| **A3** SL-AMEC | Training-free calibrator conditioned on trial metadata (language, corpus, set) | metadata conditioning improves EER | Cllr **0.8356 -> 0.1873** (median 4.5x, 1,225 params); **0 / 33** systems significant on EER; **29 / 33** show < 1e-9 EER change under *shuffled* metadata | **Split: large positive on calibration, clean null on metadata** |
| **A6** channel-aware SSPS | Project the channel subspace out before clustering, on the premise that SSPS harvests same-*channel* rather than same-*speaker* pairs | correction improves cluster purity | plain SSPS speaker purity **0.9302** / NMI 0.9810 vs channel-aware **0.9099** / 0.9767 | **Premise fails here; correction slightly harmful** |
| **A9** ARI-SubCenter | Sub-centre AAM with K set **per corpus from a measured label-reliability statistic** rather than fixed at K=3 | sub-centres absorb label noise the QC audit proved present | si **4.375 -> 3.962**, Delta = **-0.413 pp**, CI [-0.677, -0.015], **p = 0.038**; ta 0.884 -> 0.934, p = 0.668 | **First positive result --- and mechanistically coherent** |
| A1, A2, A4, A5, A7, A8 | --- | --- | training / queued / retired | No EER claimed |

**Why A3's null is strong rather than merely underpowered.** A null from the
bootstrap alone would be ambiguous at 18--26 test speakers. A **bit-identical
result under metadata shuffling** is not ambiguous --- the metadata is inert. The
metadata-only negative control sits at 98--99 % EER, confirming the harness
measures what it claims.

**Why A9's positive is more than a number.** The asymmetry follows the QC audit
exactly. SLR127 (Tamil) is the corpus whose identity error the audit **found and
corrected** (638 speakers had been collapsed into 531). SLR52 (Sinhala) is the
corpus the audit **flagged as unresolved** --- its profile is consistent with
account-level rather than person-level identity, and no metric can settle that
without a listening pass. Sub-centres are the standard response to residual
within-class identity heterogeneity, and they improved the corpus where such
heterogeneity is *suspected but uncorrected* while doing nothing measurable
where it was *found and fixed*. Held as **promising, not settled**: p = 0.038
with an interval reaching -0.015 pp, single seed.

### What the proposals strand has already redirected

- **Calibrate, do not annotate.** 4.5x better Cllr for 1,225 parameters; the
  metadata plumbing A3's full form requires is not worth building.
- **The channel confound is real but shallow.** v1 showed channel is *linearly
  decodable* (91.2 % probe accuracy); A6 shows it does **not** dominate local
  neighbourhood structure. Work aimed at the confound should target the
  **scoring** stage, not the clustering stage.
- **The clearest data priority in the project.** Four of nine methods need a
  domain variable the collection supplies only by pooling corpora, which
  confounds domain with language. **Acquiring within-speaker, cross-channel Sri
  Lankan recordings would unlock four of nine proposals at once** --- a sharper
  priority than any individual method.

## 7.2 `transformer_sv` T1--T6 --- from-scratch transformers **[M]**

**The framing, which decides what is worth running.** v1 already uses
transformers --- WavLM and mHuBERT *are* transformer encoders, and reading them
through learned layer weights gives the benchmark's best systems. What v1 never
trained is a transformer **speaker encoder**: every from-scratch backbone in the
benchmark is convolutional.

> **Pre-registered prediction [P], recorded before any run:** *from-scratch
> transformers will lose to ECAPA-TDNN on these corpora.* Justification: v1
> Finding III-5 showed the regime is data-limited, not capacity-limited
> (p = 0.835 for a 2.4x parameter increase), and transformers are the more
> data-hungry family. **A clean negative would be the measured argument for the
> SSL route on low-resource languages, on the same trials as everything else.**

### T5 --- the cost result (training-free, CPU) **[M]**

All four attention arms hold **11,062,848 parameters exactly** (verified), so
nothing below is a capacity effect.

| input | tokens | interaction MACs | projection MACs | interaction share |
|---|---|---|---|---|
| 2 s (training crop) | 50 | 7,680,000 | 78,643,200 | **8.9 %** |
| 3 s (evaluation) | 75 | 17,280,000 | 117,964,800 | 12.8 % |
| 8 s | 200 | 122,880,000 | 314,572,800 | 28.1 % |

The fitted log-log MAC exponent over 1--8 s is **1.134, not 2**.

| arm | interaction MACs @ 2 s | vs full | measured latency @ 2 s |
|---|---|---|---|
| full | 7,680,000 | --- | **15.7 ms** |
| window (+/-8) | 2,611,200 | 0.34x | 16.4 ms (+4 %) |
| pooled (stride 4) | 1,996,800 | 0.26x | 16.5 ms (+5 %) |
| **linear** | **9,830,400** | **1.28x** | 17.1 ms (+9 %) |

> **Finding III-8.** *The O(T^2) objection does not apply at these input lengths.*
> With x4 convolutional subsampling a 2 s crop is 50 tokens, so the quadratic
> term is a minority of attention's own cost, which is itself a minority of the
> model's. **`linear` attention is 1.28x more expensive than full attention
> here** --- its crossover is at T ~ 2*d_head = 128 tokens ~ 5.1 s --- and
> **every "efficient" arm measured slower in wall-clock.** Consequence: the
> accuracy arms use **full** attention, which is also the only mode that is a
> control (window with w >= T-1 is bit-identical to it, asserted).

### T1 --- the first from-scratch transformer accuracy result **[M, NEW]**

**This result completed on 24 August and has not appeared in any prior report.**

MFAConformer (Conformer encoder + multi-scale aggregation, full attention),
parameter-matched to its baseline:

| System | Params | si test EER (cosine) | si test EER (AS-Norm) | minDCF |
|---|---|---|---|---|
| `A_ecapa512` (baseline) | 5,994,688 | **4.285** | **4.224** | 0.303 |
| `MFAConformer_full_si_s42` | 6,135,504 | 4.950 | 4.718 | 0.326 |
| **Difference** | +2.3 % | **+0.665 pp** | **+0.494 pp** | +0.023 |

> **The pre-registered prediction is confirmed.** A from-scratch Conformer, at
> matched parameters and under an identical recipe on identical trials, is
> **~16 % relatively worse** than a convolutional ECAPA-TDNN on 336 Sinhala
> training speakers. Combined with Finding III-3 --- where a *pretrained*
> transformer read through learned layer weights reaches 2.369 % on the same
> trials --- this gives the programme a clean, self-contained argument:
>
> **On low-resource languages the value of a transformer is in its pretraining,
> not in its architecture.** The same model family loses by 0.67 pp when trained
> from scratch and wins by 1.92 pp when pretrained and correctly read.
>
> *Caveat: single seed, one language. The paired bootstrap has not been run on
> this contrast, and at S = 91 the +0.665 pp gap is near the resolution limit
> established in section 6.8.2. Treat as directional pending the ta arm and a
> seed replication.*

### Two numerical defects found by these launches **[M]**

1. **fp16 attention overflow.** `q @ k^T` is the one product in a transformer
   with no bound on its magnitude. In fp16 it overflows to `inf`, and
   `softmax(inf - inf)` is `NaN`. Under `--mixedprec` this killed MFAConformer
   on Tamil at ~9k steps **with no error message** --- the loss simply printed
   `nan`. ECAPA-TDNN has no such product, which is why the same recipe never
   exposed it. Fixed by computing scores, position bias and softmax in fp32 even
   under autocast.

2. **fp16 mel front-end overflow --- worse, and unrelated to attention.**
   Measured over 30,000 real augmented samples: MUSAN mixing and RIR convolution
   push the waveform to **|x| = 5.66**, and the fp16 power spectrogram peaks at
   **30,592 against fp16's 65,504 ceiling --- 2.1x headroom, not orders of
   magnitude.** One `inf` there poisons **BatchNorm's running statistics
   permanently**, because those are written in the forward pass and GradScaler
   only guards the backward. Eval mode then returns NaN for every input **while
   training accuracy still looks healthy** (TAcc 23.59 % and TLOSS nan in the
   same epoch), and the run dies at its next validation.

**The diagnosis generalised beyond its own strand.** The v1 benchmark is clean
(**0 of 54 runs** have a NaN training-loss epoch), but **two A-series proposal
runs** --- `A5_dk_campp/DKCAMPP_ta` and `A4_ls_cam/LSCAM_ta` --- fail with the
identical traceback. Both are CAM++-family models that build their **own**
`MelSpectrogram` inside the model rather than taking the trainer's. All eight
existing A-series checkpoints were then verified clean (0 poisoned BN buffers,
0 NaN weights), so **no published A-series number is affected** --- the two runs
simply never produced one, and the one-line fix they need is recorded.

**An initial reading was wrong and was corrected within the hour.** The first
diagnosis held the NaN losses to be a cosmetic averaging artefact ("a blemish in
the record, not a defect in the model"). Both affected arms then died at their
next validation. The superseded reasoning was **kept in the document and marked
superseded** rather than deleted --- the record shows the wrong hypothesis, the
evidence that killed it, and the correct one.

**Cost of the detour:** ~25 GPU-minutes of aborted training and ~15 of probes,
against the ~80 GPU-hours the first dispatch would have spent producing one NaN
arm and four arms trained under numerically broken attention.

**A recipe deviation was considered and rejected on evidence.** The obvious
reading --- "the ECAPA recipe cannot train a transformer, it needs warm-up" ---
would have made every T-series number incomparable with the baselines. Probes
showed lowering the learning rate to 1e-4 is **measurably worse** on both
languages, so the arms stayed protocol-faithful with no deviation.

**Status:** T1 si complete; T1 ta, T1 no-attention, T2 tf4/tf0 training; T3, T4,
T6 queued. `scaffold.py --check` reports what has produced numbers and what has
not, and is the figure to trust.

## 7.3 Feature-level testing --- hand-crafted descriptors under a byte budget **[M]**

A separate report had proposed four hand-crafted acoustic frameworks for
ultra-low-payload verification (a QR-code-carried speaker template). **None had
ever been tested in this project.** All four were implemented and run on real
Sinhala and Tamil data, matched by construction: same speaker count (40 each),
same trial count, same 4 s crop, identical back end, and **100 % of genuine
pairs cross-session**.

| System | bytes | Sinhala EER % | Tamil EER % | si minDCF |
|---|---|---|---|---|
| formant trajectories (the report's top pick) | 40 | **43.07** | **39.50** | 0.997 |
| formant (static, control) | **8** | 35.90 | 31.43 | 0.994 |
| glottal (IAIF + LF) | 23 | 19.73 | 13.80 | 0.989 |
| **LPC-12 LSP** | **24** | **18.00** | **12.13** | 0.934 |
| bispectrum (HOSA) | 40 | 30.67 | 29.70 | 0.999 |
| all four fused | 127 | 17.30 | 12.93 | 0.870 |
| hybrid micro-vector (the report's recommendation) | 191 | 10.07 | 4.47 | 0.737 |
| **ECAPA-TDNN (PCA-128)** | **128** | **3.63** | **0.57** | **0.273** |

**Findings, each with its control:**

1. **The report's ranking inverts under measurement.** Formant trajectories,
   rated "High / Extremely High", are the **worst** of the four and barely
   separate speakers (minDCF ~1.00 means the detector is worthless at a 1 %
   prior). LSP, rated lowest, is **best --- in the smallest payload of any system
   tested.**
2. **The cubic-trajectory descriptor is worse than averaging.** A static control
   derived analytically from the same coefficients uses **8 bytes instead of 40**
   and scores **7.2 pp (si) / 8.1 pp (ta) better.** Formant *extraction* is not
   the problem --- the selftest recovers a synthetic four-resonance vocal tract to
   within 1 %.
3. **The byte budget was never the bottleneck.** 8-bit quantisation costs
   <= 0.07 pp anywhere, and several systems *improve* slightly (mild
   regularisation). **The report's core premise holds; its feature choices do
   not.**
4. **The recommended hybrid is a net loss at every mixing weight.** Sweeping the
   physical-feature weight from 0 to 0.7 is monotone in the tail; at the
   specified equal weighting it costs **+6.4 pp (si) and +3.9 pp (ta)**. The
   control that makes this attributable: **PCA-truncating ECAPA from 192 B to
   128 B costs nothing** (3.63 / 0.57 vs 3.63 / 0.60), so the loss is entirely
   the fusion, not the compression.
5. **The phonetic-degradation premise is weak.** Tamil carries 2.0x the retroflex
   and 3.4x the geminate load of Sinhala, but Sinhala's prenasalized stops ---
   the category the report leans on hardest --- occur **0.47 times per 100
   characters**, roughly once every 200 characters. Splitting trials by phonetic
   density: Tamil goes the predicted way on 5 of 5 systems, **Sinhala goes the
   wrong way on 3 of 5**, and every effect is small relative to the bootstrap
   intervals.
6. **The QR mapping half-checks out.** 192 bytes is exactly version 10 (57x57) at
   EC=M --- the claim holds. But the report's own **512-byte ceiling needs version
   18 (89x89)**, outside its stated range, and at a 2 cm print size its modules
   fall to **0.206 mm**, below the ~0.25 mm practical floor for a phone scan.
   **A 512-byte payload and a 2 cm code are not simultaneously achievable.**
   Dropping to 128 B also drops the symbol to version 8 and raises module pitch
   to 0.351 mm --- **14 % more optical margin, for free.**

**A confound identified and reported rather than buried.** Tamil is easier on
every system, but **the gap grows with how good the system is** (1.03x on
bispectrum, 6.41x on ECAPA). A language-intrinsic advantage would shift all
systems together; what behaves this way is *headroom*. Measured audio quality
confirms Tamil is cleaner on all four periodicity measures (HNR 3.77 vs 3.00 dB,
jitter 0.036 vs 0.057). **"Tamil verifies 6x better than Sinhala" must not be
quoted as a language finding.**

**And it independently reproduced Finding III-1.** A single shared threshold at
the pooled EER point costs Tamil **+1.65 pp** --- nearly 3x its own EER --- and
pushes Sinhala's miss rate to 6.1 %. *Per-language calibration is mandatory*,
now demonstrated on hand-crafted and deep features alike.

\newpage

# 8. Failure register --- every failed result and what it produced

This section exists because in this programme the failures carry more
information than the successes, and several of them are the publishable
contributions.

| # | What failed | Predicted | Actual | Root cause established | What was identified |
|---|---|---|---|---|---|
| **F1** | MLP-Mixer V1 (MSE distillation) | 10--11 % | **16.13 %** --- worse than no distillation | MSE on L2-normalised embeddings is ~0.0002; distillation was 0.008 % of the loss | **Finding I-1**: embedding distillation requires an angular loss. Geometric, not empirical --- transfers to any normalised-embedding distillation |
| **F2** | MLP-Mixer V2_Large (alpha 0.7) | 11--12 % | **14.84 %** --- worse than a 3x smaller model | Student (7.84 M) exceeded teacher (3.87 M) by 102 %; alpha=0.7 pinned it to a lower-capacity teacher | **Finding I-2**: optimal alpha tracks capacity ratio. Better teacher alignment can coexist with worse verification --- alignment is not the objective |
| **F3** | NestedSpeakerNet, 3 attempts | -8--13 % EER, 2x faster | **NaN x2, or 87 % worse** | Anti-correlated inter-level features (rho = -0.23) make nested aggregation amplify rather than regularise; Hessian condition ~160,000 | **Finding I-3**: five measurable domain-compatibility criteria; audio meets 0 of 5. Predicts failure for video action recognition, ASR, time-series too |
| **F4** | The whole Phase I strategic direction | competitive lightweight SV | **~12x off SOTA at matched size** | Training small models from scratch on VoxCeleb is a solved-and-lost race; the field fine-tunes released checkpoints | Redirected the entire programme (audit R1). ReDimNet-B1: 0.85 % at 2.2 M vs 10.32 % at 2.66 M |
| **F5** | English P0 baseline | a reference EER | **Never ran.** 8 launch attempts 2026-05-27, empty scores, no checkpoints | 564 missing VoxCeleb2 wavs | Fixable in one filter operation (audit C1); the blocker was catalogued, not diagnosed, for ~5 weeks |
| **F6** | Jan--Jun 2026 experimental claims | 7 architectures + SSL + deployment | **No supporting artefact for any of it** | Not established | The programme's evidence standard was rebuilt around this. See section 5 |
| **F7** | PEFT-beats-full-FT (UniPET-SPK) | LoRA >= full FT | **full FT beat LoRA at every seed**, 2.2x | Closed-set eval removes full FT's overfitting-to-source penalty | Falsification reported with mechanism and with the arbiter experiment named. **Still unrun** --- see section 10 |
| **F8** | Mid-stack SSL layer peak | speaker info peaks ~layer 6 | **Monotonic decay with depth**; layer 6 barely beats layer 12 | Masked-prediction pretraining makes upper layers speaker-invariant by design | **Finding III-3**, the programme's headline. The falsified entry was *kept* as the mid-curve data point |
| **F9** | MFCC will lose badly to log-mel | large loss from DCT decorrelation | **within ~0.2 pp**, not significant on si (p = 0.463) | Prediction was directionally right, practically wrong | Pre-registration works: an intuition that felt certain was cheap to test and false |
| **F10** | PLDA under domain shift (v0) | PLDA > cosine | **Hurt Sinhala for every model** | PLDA fit dominated by 478 near-single-session si speakers -> within-speaker covariance underestimates channel variability | Flagged as a *benchmark artefact hypothesis*, not a refutation --- and later vindicated: after fine-tuning moved the system in-domain, PLDA became redundant exactly as the literature predicts |
| **F11** | `si_celeb` trained alone (82 spk) | a usable in-the-wild system | **19.7--27.3 % EER** | 82 training speakers far too few; 9-speaker validation set selected every checkpoint but can measure nothing | Retained as a documented negative result. Correct use is as an evaluation probe, where it is highly informative |
| **F12** | A6 channel-aware SSPS | projection improves clusters | **speaker purity fell 2.0 pp** | Clusters were already speaker-shaped (speaker NMI 0.98 vs corpus NMI 0.28) | Sharpened v1's channel finding: channel is *linearly decodable* but does not *dominate local neighbourhoods*. Bounded honestly --- the bootstrap was supervised, biasing toward this result |
| **F13** | A3 metadata conditioning | metadata improves EER | **0/33 significant; 29/33 bit-identical under shuffling** | The gain is entirely score-domain | A **publishable controlled null**: the shuffled-metadata control makes it inertness, not absent power. Stops the next group repeating it |
| **F14** | A7 PLLW first runs | per-language layer weighting | **0.663 % vs 0.623 % baseline** --- reads as a plausible small regression | Ran on single-language data with `n_languages=3`: two dead vectors and a degenerate head | **Finding III-7**: methods undefined on the data still train, converge, and emit plausible numbers. Guards must derive from the data |
| **F15** | Efficient attention (T5) | cheaper than full attention | **linear is 1.28x more expensive**; all arms slower in wall-clock | x4 subsampling makes a 2 s crop 50 tokens; crossover is at 128 tokens ~ 5.1 s | Killed four planned training runs before they cost GPU-hours. A cost result that saved compute |
| **F16** | From-scratch Conformer (T1) | *predicted to lose* | **lost by 0.665 pp** to matched ECAPA-512 | Data-limited regime (Finding III-5); transformers are more data-hungry | **Prediction confirmed.** Completes the argument: transformer value is in pretraining, not architecture |
| **F17** | Formant-trajectory descriptor | best of four frameworks | **43.07 % / 39.50 %** --- near chance | Averaging polynomial coefficients over phonetically uncontrolled segments destroys more than it captures | An 8-byte static control beat the 40-byte trajectory by 7--8 pp. The proposal is worse than the trivial alternative |
| **F18** | Hybrid micro-vector | deep + physical > deep | **+6.4 pp (si) / +3.9 pp (ta)** at the specified weight | Fusion, not compression --- PCA to 128 B costs nothing | Spend the whole budget on the deep embedding; 128 B also yields a more scannable QR symbol |

## 8.1 Engineering defect classes --- the meta-finding

Across the year, **roughly 40 defects** were found and fixed. Their distribution
is the point:

| Class | Examples | Why it matters |
|---|---|---|
| **Silent wrong numbers** (the dominant class) | Checkpoints matching **0 tensors** and reporting a believable ~41 % EER; `int8` label overflow corrupting every EER past 127 targets; SSL models scored on the wrong layer; nine evaluations from partly-random weights; A7's degenerate run | **None of these crash.** Every one produces a plausible number. This is the failure mode the programme has paid for most |
| **Numerical** | fp16 attention overflow; fp16 mel front-end overflow poisoning BatchNorm permanently | Kill runs with no error message; the second also explains two unrelated A-series failures |
| **Experimental hygiene** | Smoke checkpoints silently resumed by full runs; English conditions leaking into every evaluation (+2.5x cost); Stage F front ends leaking into Stage A; manifest clobbering across invocations | Corrupt the comparison rather than the code |
| **Infrastructure** | 4 cluster-wide reboots in 5 days; queue consuming `rc=255` jobs without retry; `pkill -f` over ssh killing its own shell; slots blocked 5.5 h by lingering pty clients | Cost time, not correctness --- **no training work was lost**, thanks to per-epoch checkpointing and recorder-resume |
| **Throughput** | 7 concurrent jobs collapsed *every* job to 5--16 Hz (114 dataloader processes saturating one NFS server, GPU ~0 %, CPU 86--93 % idle) | Capping concurrency at 3 restored **224--1,625 Hz, a 30--100x recovery.** Fewer concurrent jobs finish the work strictly faster here |

> **Finding III-9.** *In this programme the modal defect does not crash --- it
> returns a plausible number.* Every methodological device adopted in Phase III
> (hard-failing evaluators, argv copying rather than manifest reconstruction,
> exact-reduction controls, shuffled-metadata negative controls, isolation
> assertions in `selftest.py`, data-derived guards) exists to convert a silent
> wrong number into a loud failure.

\newpage

# 9. Prediction scoreboard

Pre-registered or explicitly stated predictions, scored against measurement.

| # | Prediction | Source | Outcome | Verdict |
|---|---|---|---|---|
| 1 | LSTM+AE improves 20--35 % over baseline | Phase I | 37 % (15.48 -> 9.68) | **Met** |
| 2 | MLP-Mixer student reaches 10--11 % | Phase I | 16.13 % (V1), then 10.32 % (V2) | **Failed, then met after diagnosis** |
| 3 | Larger student closes the teacher gap | Phase I | 14.84 %, then 10.11 % --- never beat the small student | **Failed** |
| 4 | Nested learning: -8--13 % EER, 2x faster | Phase I | NaN x2; 1.09x faster at best | **Failed** |
| 5 | Cross-lingual degradation is 2--4x | CN-Celeb precedent | ta 2.3--4.0x; **si 4.9--8.0x** | **Met for ta, exceeded for si** |
| 6 | AS-Norm gives 20--30 % relative | Matejka 2017 | 12--39 % (5/6 cells in band) | **Met** |
| 7 | PLDA beats cosine under shift | arXiv:2204.11403 | Hurt si; helped ta 2/3 | **Refuted at v0, vindicated in-domain** |
| 8 | Cross-lingual score shift is systematic | arXiv:2110.09150 | Cllr inflation up to 5.7x | **Met, larger than expected** |
| 9 | PEFT >= full FT under language shift | UniPET-SPK | full FT won at every seed, 2.2x | **Falsified** (closed-set caveat named) |
| 10 | Frozen encoder + head captures most SSL gain | WavLM paper | frozen 3.74 vs full 1.63 --- within ~1 EER of best zero-shot | **Partly met** |
| 11 | SSL speaker info peaks mid-stack | this project, 12 Aug | Monotonic decay with depth | **Falsified** --- and became the headline finding |
| 12 | MFCC will lose badly to log-mel | this project | Within ~0.2 pp, not significant | **Falsified** |
| 13 | Multilingual pretraining pays at equal capacity | this project | mHuBERT beats WavLM on both (p = 0.000 / 0.005) | **Met** |
| 14 | Regime is data- not capacity-limited | this project | ecapa512 = ecapa1024, p = 0.835 | **Met** |
| 15 | Efficient attention will be cheaper | general practice | linear 1.28x *more* expensive at 2 s | **Falsified** |
| 16 | From-scratch transformers lose to ECAPA | this project, pre-registered | 4.950 vs 4.285 (si) | **Confirmed** |
| 17 | Sub-centres help where label noise is uncorrected | this project (A9) | si -0.413 pp p=0.038; ta null | **Met, mechanistically** |
| 18 | Metadata conditioning improves calibration+EER | A3 source | Calibration yes (4.5x), metadata inert | **Split** |
| 19 | Channel dominates SSPS neighbourhoods | A6 source | speaker NMI 0.98 vs corpus 0.28 | **Falsified in this setup** |
| 20 | Hand-crafted descriptors are viable sub-kB | feature report | best is 5x/21x worse than ECAPA at half the size | **Falsified** |

**Score: 8 met, 7 falsified, 3 split/partial, 2 failed outright.**

A ~40 % falsification rate on pre-registered predictions is a healthy sign, not a
poor one: it means the predictions were specific enough to be wrong, and that
they were scored rather than rationalised. Two of the falsifications (11 and 16)
are the strongest results in the programme.

\newpage

# 10. What is safe to claim as of 31 August 2026

## 10.1 Safe

Every number in sections 6.9--6.12 is **held-out test EER on speaker-disjoint
splits**, from a model verified to match its own checkpoint, with model selection
made on validation trials only and the test set scored once.

| Claim | Evidence |
|---|---|
| Best Sinhala system: `ssl_mhubert_lw`, **2.369 %** cosine / **2.178 %** AS-Norm, minDCF 0.156 | v1 held-out test |
| Best Tamil system: `ssl_mhubert_ecapa`, **0.402 %** cosine / **0.321 %** AS-Norm | v1 held-out test |
| Layer-weighted SSL beats the mel baseline by **46 % (si) / 55 % (ta)** | paired bootstrap, p = 0.000 |
| Speaker information in WavLM decays **monotonically with depth**; the conventional default is the worst choice | layer probe, no training involved |
| The last-layer penalty **scales with linguistic distance** (1.04x en, 1.88x si, 5.23x ta) | v1 + English control |
| Multilingual pretraining pays at **matched capacity** | mHuBERT vs WavLM, p = 0.000 / 0.005 |
| The regime is **data-limited, not capacity-limited** | ecapa512 vs ecapa1024, p = 0.835 |
| Cross-session audio is **4--5x harder** than read speech | `si_pooled`, one model, both domains |
| ~48 % of a published Tamil impostor list is rejected on **channel, not identity** | batch-split diagnostic, 3 architectures |
| **Per-language calibration is mandatory** | Cllr inflation 5.7x; shared threshold costs ta +1.65 pp |
| AS-Norm is the best cost/benefit lever in the project | 12--39 % relative, training-free |
| A9 sub-centres help Sinhala (**-0.413 pp, p = 0.038**) | paired bootstrap, single seed |
| A3 metadata conditioning is **inert** over score calibration | 29/33 bit-identical under shuffling |
| Two speaker-identity bugs found and corrected in corpora this project built | audio-based identity test |

## 10.2 Not yet safe

- **Single-seed results throughout v1, the proposals, and the T-series.** The
  power ceiling binds: a single absolute EER carries ~+/-5--6 pp at S = 91.
  Differences of 0.09 pp (ecapa512 vs ecapa1024) are **not** resolvable;
  differences of 2.0 pp (SSL vs mel) **are**.
- **Any absolute EER quoted from a single corpus.** With
  $\eta^2(\text{SNR}\mid\text{speaker})$ at 0.68--0.91 and a 91.2 % channel probe,
  part of what is measured is the recording setup.
- **Any Sri Lankan Tamil claim.** Every large Tamil corpus in use is Indian
  Tamil.
- **T1's transformer contrast**, pending the Tamil arm and a paired bootstrap.

## 10.3 Must be corrected or withdrawn

1. **All January--June 2026 experimental results** (section 5). They cannot enter
   the thesis, any paper, or any future progress report.
2. **The nested-learning "9.84 % EER, validated for efficiency" claim** in
   particular --- the measured record is three NaN collapses.
3. **The Phase I 10.32 % headline** --- unreplicated, single seed, predates
   `--deterministic`, and carries an unresolved internal conflict (14.62 % in one
   log) plus a suspected duplicated result set. Audit item R7 applies: re-run
   3 seeds or retire the claim.
4. **The Period 2 report's ResNetSE34L specification** (34 layers / 6.8 M / ~8--10 %
   EER) --- the logs give 1.50 M and 15.48 %.
5. **The "Tamil pool 50 -> 752" figure** in the 17 August report --- the measured
   build is 706 (802 after QC corrections).
6. **Every v0 EER** should carry the ~5x read-vs-wild optimism factor
   retrospectively, and the closed-set caveat explicitly.

\newpage

# 11. Open items, in priority order

1. **The arbiter run.** Layer-weighted **and** fine-tuned SSL, open-set, one run
   per language. v1 measured layer-weighted+frozen (2.369 si) and
   single-layer+fine-tuned (5.404 si) but **never the diagonal that P3's headline
   claim actually rests on** --- and that claim currently underpins
   `papers/ieee_spl`. Named as the highest-value next experiment on 17 August and
   still unrun as of 10 September.
2. **Seed replication** for the top three systems per language, and for A9.
   Converts single-run numbers into intervals; without it Stage D's question
   ("is the ordering stable?") is unanswered.
3. **A7 PLLW on `combined_si_ta`** --- per-language layer weighting is the direct
   extension of Finding III-3 for ~39 extra parameters. If the optimal layer
   differs between Sinhala and Tamil, one shared weight vector is leaving
   performance on the table.
4. **Complete the T-series** --- T1 Tamil, the no-attention ablation, T2--T4, and
   T6's span probe (answerable on an already-trained model with no further
   training).
5. **Fix and re-run A4/A5 on Tamil.** The diagnosis and the one-line fix are
   recorded; the runs simply never produced a number.
6. **SLCeleb** remains the binding blocker on any Sri Lankan claim. It is the
   only corpus whose absolute EER would mean what a reader assumes.
7. **Acquire within-speaker, cross-channel Sri Lankan recordings.** This unlocks
   four of the nine proposals at once and is a clearer data priority than any
   individual method.
8. **A listening pass on `slr52_sinhala`** before its Sinhala numbers carry
   weight. The shortlists exist; no metric can close this.
9. **Run V4 (SE) and V5 (Res2Net) properly.** The Phase II designs were never
   executed but the rationales are sound and both are cheap under the v1 harness.
10. **Escalate cluster stability.** Four reboots in five days; `compute-node-2`
    needs root to reload its driver (recovering it adds 2 GPUs).
11. **The mel-scale question is untested** since RawNet3 was removed on request ---
    whether a scale fitted to English and European perceptual data suits Sinhala
    and Tamil remains open, and is a genuinely publishable question.

\newpage

# 12. Assessment

## 12.1 Achievements of the period

Set out first, because the sections above are organised by chronology and by
failure, and the record of what the period actually delivered is otherwise
distributed across all twelve of them. Everything listed here is **[M]** ---
measured, with an artefact in the repository.

### A. Research contributions --- results that are publishable as they stand

| # | Contribution | Standing |
|---|---|---|
| **A1** | **Speaker information in an SSL encoder falls monotonically with depth, and the conventional last-layer default is the worst available choice.** Learned weighting over all 13 states beats every fixed depth, for 13 scalar parameters (Finding III-3) | si 8.247 -> 2.369 held-out; paired bootstrap p = 0.000. **The programme's headline result** |
| **A2** | **The last-layer penalty scales with linguistic distance** --- 1.04x English, 1.88x Sinhala, 5.23x Tamil. Last-layer SSL practice is an English-centric default whose cost is a function of the target language (Finding III-4) | A mechanism, not just an effect; the English control is what made it visible |
| **A3** | **Multilingual pretraining pays at matched capacity.** mHuBERT-147 beats WavLM-base-plus on both languages at parameter counts matched to within 0.01 %, isolating *coverage* from *capacity* | si p = 0.000, ta p = 0.005 |
| **A4** | **Embedding distillation requires an angular loss.** MSE on L2-normalised embeddings is unusable --- the normalisation constraint compresses its dynamic range below the classification gradient scale (Finding I-1) | Derived geometrically, confirmed empirically at 36 % relative. Transfers to any normalised-embedding distillation |
| **A5** | **The optimal distillation weight tracks the student/teacher capacity ratio** (Finding I-2) | +4.73 pp recovered by the predicted correction |
| **A6** | **Nested/dense connectivity is domain-specific, not universal** --- five measured compatibility criteria, of which spectrogram audio meets none (Finding I-3) | Includes falsifiable predictions for video, ASR and time-series |
| **A7** | **This low-resource regime is data-limited, not capacity-limited** (Finding III-5) | A null with a *tight* interval: 2.4x the parameters, p = 0.835 |
| **A8** | **Genuine cross-session audio is 4--5x harder than read speech** for every backbone (Finding III-6) --- and the same-recording confound inflates apparent separability by 42 %, measured directly | The single most important caveat on the whole low-resource SV read-speech literature |
| **A9** | **Per-language calibration is mandatory** (Finding III-1), with the publishable nuance that PLDA scores are the most *transportable* across languages even where their EER is worse | Cllr inflation up to 5.7x; independently reproduced on hand-crafted features |
| **A10** | **A controlled null**: metadata conditioning adds nothing over score calibration on these corpora, demonstrated by a *bit-identical* result under metadata shuffling rather than by absent significance (A3/P2) | Stops the next group spending months on the same idea --- a genuine contribution in a literature with almost no published negatives |
| **A11** | **Methods undefined on the data still train, converge and emit plausible numbers** (Finding III-7) --- four of nine literature proposals are undefined on the conditions the benchmark was built from, and none of their papers says so | Generalises well beyond this project |
| **A12** | **The O(T^2) objection does not apply at speaker-verification input lengths** (Finding III-8) --- with x4 subsampling the quadratic term is 8.9 % of attention's own cost, and "efficient" attention is *more* expensive | Fitted MAC exponent 1.134, not 2 |
| **A13** | **Transformer value on low-resource languages is in the pretraining, not the architecture** --- the same family loses by 0.665 pp from scratch and wins by 1.92 pp pretrained and correctly read, on identical trials | The clean negative the T-series was designed to produce |

### B. Systems and measured results

| Achievement | Figure |
|---|---|
| **First Sinhala/Tamil speaker-verification benchmark of any kind** --- protocol, corpora, trial design, metrics | v0 (Jul) then v1 open-set (Aug) |
| **First zero-shot Sinhala EER table across modern architectures** --- no Sinhala result exists for any of them anywhere in the literature | 4 systems x 2 languages, 12k pairs each |
| **Best Sinhala system** (`ssl_mhubert_lw`), held-out, speaker-disjoint | **2.369 %** cosine / **2.178 %** AS-Norm, minDCF 0.156 |
| **Best Tamil system** (`ssl_mhubert_ecapa`) | **0.402 %** cosine / **0.321 %** AS-Norm, minDCF 0.026 |
| Layer-weighted SSL front end vs the strongest mel baseline | **46 % better (si), 55 % better (ta)** |
| **Best training-free system** --- ReDimNet-B6 + AS-Norm, no training, one cohort file | si 2.71 -> **2.07 %**, ta 1.48 -> **0.90 %** |
| **31k in-language utterances take a 95 M English-pretrained SSL model past the best off-the-shelf supervised model** | full FT si **1.69 +/- 0.06** vs ReDimNet-B6 2.71 |
| **First positive proposal result**, mechanistically coherent with the project's own data audit (A9 sub-centres) | si 4.375 -> **3.962**, -0.413 pp, p = 0.038 |
| **Calibration as a free win** --- Cllr improved by a median factor of 4.5x at 1,225 parameters, no GPU, no retraining | 0.8356 -> 0.1873 across 33 systems |
| **English positive control validates the whole harness end to end** | `en_full` 3.496 % on VoxCeleb1-O |
| Phase I teacher architecture --- the one Phase I prediction that was met | 15.48 -> **9.68 %**, 37 % relative |
| Phase I student inference efficiency | **2.04x** faster than teacher at 31 % fewer parameters |

### C. Data assets built

| Achievement | Figure |
|---|---|
| **Eight corpora rebuilt into one VoxCeleb-format collection**, each self-contained with metadata, lists and a `prepare.py` | **291,061 files converted, zero conversion errors** |
| **Tamil speaker pool expanded** --- the binding constraint on statistical power, removed | **50 -> 706 speakers** (802 after QC correction) |
| **Two speaker-identity bugs found and corrected in data this project had built and published** --- caught only by embedding the audio, after passing every metadata-level check | SLR127 **531 -> 638** speakers, EER **55.31 % -> 1.07 %**; Kathbath 60 -> 49 |
| **A seven-layer dataset quality audit** with 13 new tools --- integrity, signal quality, label reliability, trial audit, power, calibration, subgroup fairness | Reusable on any future corpus |
| **The channel confound quantified** --- corpus-of-origin predicted from the speaker embedding at 91.2 % against 20 % chance | Now a standing caveat on every cross-corpus number |
| **The same-recording confound measured directly** on the only corpus that can produce it | d' 3.46 -> 2.44; **42 % inflation** |
| **The project's first cross-lingual number** --- same speakers, two languages, one embedding | Tamil 1.10 % -> Tamil/English **2.90 %**, a 2.6x penalty |
| **Speaker-disjoint splits for every corpus**, deterministic, SHA-256 per emitted list, with four corpora held out entirely as generalisation probes | The shipped lists left untouched, so prior results stay reproducible |

### D. Methodology and tooling

| Achievement | Figure |
|---|---|
| **A 54-experiment staged study** across six data conditions, run with the trainer **unmodified** and invoked as a subprocess --- so every previously published number stays on the same code path | 46 with held-out evaluation |
| **A paired speaker-clustered bootstrap validated against synthetic ground truth** --- EER matches the repository's own `tuneThreshold` exactly, a known injected difference is detected, a system against itself correctly finds nothing | **2.6x tighter** than absolute EER |
| **A measured power ceiling** that changed the design: effective sample size is bounded by *speaker* count, not trial count | Trial lists cut **80k -> 20k pairs** at no loss of resolving power |
| **Pre-registered predictions** for every loss and architecture, written before any run and *scored* afterwards --- including the two that were falsified and kept rather than edited away | 20 predictions scored; ~40 % falsified |
| **An evaluation-integrity rebuild** that caught four defects, three of which produced plausible wrong numbers rather than errors, and closed the whole defect class rather than the instances observed | AST-parsed all 82 argparse defaults; **all 54 experiments rebuild with zero backbone mismatch** |
| **A reusable proposal harness** --- shim, argv-copying trainer, data-derived domain guards, isolated scorer --- so any future method supplying a `MainModel` or `LossFunction` is trained on the v1 recipe and scored on the v1 trials with one changed factor | The piece most likely to outlive the specific proposals |
| **Exact-reduction controls**, asserted rather than asserted-to-be-similar (window attention with w >= T-1 is bit-identical to full; T2's `tf0` is the stem-only path; T5's four arms share one `state_dict`) | 18 + 13 self-test checks, 0 failures |
| **~40 engineering defects found and fixed**, and the defect *classes* characterised (Finding III-9) | Every Phase III device exists to convert a silent wrong number into a loud failure |
| **A 2.46x faster training platform** and a 45x-faster development loop | 809 samples/s; 8 s/epoch on the mini-dataset |
| **Cluster capacity and throughput recovered** --- idle GPUs found and added, memory-aware scheduling, and the NFS contention diagnosed | 3 -> **5 GPUs**; **30--100x** throughput recovery by *reducing* concurrency |
| **No training work lost** across four cluster-wide reboots in five days | Per-epoch checkpointing + recorder-resume; verified gap-free, duplicate-free epoch sequences across every boundary |
| **Complete measured evaluation of an external proposal** --- all four hand-crafted frameworks implemented and run with controls, inverting the source report's ranking and identifying the one component worth keeping | LSP at 24 bytes, best classical descriptor |

### E. The achievement that underwrites the rest

**The programme identified its own evidence problem, documented it, and rebuilt
its methodology around it** --- then produced, in the six weeks that followed,
more verified results than the preceding thirteen months combined. The audit of
3 July was written by the project about the project; nothing external forced it.
That is the capability the year established, and it is the reason every number
in sections 6 and 7 can be quoted.

## 12.2 Overall assessment

**What went wrong.** Thirteen of sixteen months produced no verified result on
the project's actual subject. Phase I optimised a lightweight architecture in a
race the field had already settled, and its best number is unreplicated. Phase II
produced a report whose experimental claims the repository does not support. The
English baseline was blocked by a one-command fix for over five weeks.

**What went right, and it is more than it looks.** The programme diagnosed its
own evidence problem, wrote it down, and rebuilt around it. In the six weeks
after the audit it produced: the first Sinhala/Tamil speaker-verification
benchmark of any kind; an eight-corpus collection with a full quality audit that
found two identity bugs in its own data; a 54-experiment open-set study with
paired statistics and a validated power analysis; a headline finding
(Finding III-3) that is a genuine contribution to how SSL front ends should be
read for low-resource languages, with a mechanism (III-4) explaining why the
conventional default is English-centric; a controlled null worth publishing
(A3); a first positive proposal result that follows its own data audit (A9); and
a cost result that killed four planned runs before they spent GPU-hours (T5).

**The through-line.** Every one of those six weeks' contributions came from
taking a measurement the project had previously assumed. The speaker key was
assumed and was wrong. The session structure was assumed and was wrong. The
splits were assumed open-set and were 100 % closed. The SSL layer was assumed
adequate and was the worst available. The efficient attention was assumed cheaper
and was more expensive. The trial count was assumed to buy power and does not.
In every case the assumption was plausible, documented, and false --- and in
every case one direct measurement settled it.

That is the finding this year actually produced, and it is worth stating as
such: **in low-resource speaker verification the dominant error is not a bad
model, it is a good number measured on something other than what it claims.**

---

*Sources: `research_logs/` 2025-10-26 -- 2026-08-23; `2026-07-03-project-audit-sota-roadmap.md`;
`experiments/analysis/v1-final/` (54 experiments, 46 with held-out evaluation);
`proposals/` A1--A9; `transformer_sv/` T1--T6; `feature_level_testing/`;
`data/datasets_reports/`; `Research_Progress_Report_Period{2,3}*.md`.
`voxceleb_trainer/` was unmodified by every Phase III experiment, asserted by
`selftest.py` in each strand.*
