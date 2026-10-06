---
title: "Sinhala/Tamil Speaker Verification --- Research Progress Report"
subtitle: "Benchmark v1: the open-set rebuild, and the architecture / front-end study"
author: "SL-SPV project"
date: "2026-08-17"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
toc: true
toc-depth: 2
---

\newpage

# 1. Where this sits in the research programme

This report covers **12--17 August 2026**. It is not a standalone study: it is
the **v1 open-set benchmark** that the July roadmap named as the arbiter for its
own results.

## 1.1 The arc so far

| Phase | Date | What was established |
|---|---|---|
| **P0--P3** | 2026-07-03 | Benchmark **v0** on SLR52 (478 si) + SLR65 (49 ta). Zero-shot table, backend adaptation, PEFT study. Thesis ch. 1--4 and `papers/ieee_spl` written against these numbers. |
| **Dataset build + QC** | 2026-08-10 | Eight corpora rebuilt into one VoxCeleb-format collection. Full quality audit; **two speaker-identity bugs found** (SLR127 is 638 speakers not 531; Kathbath 49 not 60). Tamil pool went **50 -> 752 speakers**. |
| **Benchmark v1** | **2026-08-12 -- 17** | **This report.** Speaker-disjoint rebuild, staged architecture / loss / front-end / interaction study across six data conditions. |

## 1.2 What v0 concluded, and the caveat it carried

P3 (PEFT fine-tuning) established a clean monotone result --- performance
improved with encoder freedom:

| P3 arm (v0, closed-set) | si EER | ta EER |
|---|---|---|
| frozen encoder | 3.74 | 3.13 |
| LoRA (r=8, 1.1 % params) | 3.17 | 2.44 |
| LLRD | 2.40 | 1.81 |
| **full fine-tune** | **1.63** | **1.30** |

Three-seed means: full **1.69 / 1.41** vs LoRA **3.73 / 3.11**. Full fine-tuning
beat LoRA at every seed, which does *not* reproduce the UniPET-SPK direction.

The P3 log stated its own limit explicitly:

> "the 527 eval speakers ARE the training speakers ... This measures
> representation adaptation, not open-set generalization; **v1 will provide the
> open-set condition**." and "Must be reported with the closed-set caveat;
> **v1 open-set rerun is the arbiter**."

**That rerun is what this report delivers.** Everything below is measured on
speaker-disjoint splits, so it is the condition v0 could not test.

## 1.3 What changed between v0 and v1

- **Protocol.** Every corpus rebuilt speaker-disjoint (§3.1). v0's per-utterance
  split shared speakers between train and test.
- **Tamil corpus.** v0 used SLR65 (49 speakers). v1 uses **SLR127 (638)**, from
  the August dataset build --- a 13x increase, and the reason Tamil results here
  are not directly comparable with v0's.
- **Scope.** v0 studied one architecture family under adaptation. v1 sweeps
  seven backbones, nine front ends and the front-end x backbone interaction,
  across six data conditions.
- **Statistics.** v1 adds the paired speaker-clustered bootstrap and a measured
  power ceiling (§3.2), neither of which v0 had.

## 1.4 Continuity note on the P3 architecture

P3 used **WavLM with learnable softmax layer weights** over all hidden states
(`tools/peft_finetune.py`), which is also what the deployed VoiceID system uses.
That design was already correct on the question §4 turns out to be the most
consequential one --- **which layer the features come from**.

The main trainer's `SSLFrontendSpeaker` does *not* do this: it reads a single
layer, defaulting to the last. Stage A used that default, which is why the
95M-parameter SSL model finished last there (§6) and why the layer investigation
began. So v1 did not discover something P3 missed; it **quantified, on open-set
data, the thing P3's architecture had already assumed** --- and found the effect
is worth 3.3x.

\newpage

# 2. Executive summary

54 experiments across six speaker-disjoint data conditions. **50 trained,
24 evaluated, 4 running.**

| # | Finding | Evidence |
|---|---|---|
| **1** | **Every shipped corpus list is 100 % closed-set.** All test speakers also appear in training. Rank-distorting, not merely optimistic --- inflation scales with capacity to memorise class centroids. | slr52 478/478, slr127 638/638, slceleb 123/123 |
| **2** | **The SSL front end was reading its worst layer.** Speaker information falls *monotonically* with depth; `--ssl_layer -1` is the worst available choice. | layer 0 = 22.30 %, layer 12 = 40.33 %; **1.81x / 2.90x** worse |
| **3** | **Tamil's apparent superiority is largely a channel artifact.** SLR127 pools three collection sites; about 48 % of impostor trials are rejected on channel, not identity. | cross-batch 0.475 % vs same-batch **1.298 %** |
| **4** | **Fixing the front end beat every architecture change.** Best system is 42 % better than the strongest mel baseline. | 2.141 % vs 3.721 % (si) |
| **5** | **Read speech is about 6x easier than genuine cross-session audio.** | read 3.421 % vs wild 21.381 %, same model |

> **Status.** All figures are validation EER on single runs. They become claims
> only after the held-out evaluation and the paired bootstrap complete. A single
> absolute EER here carries roughly ±5--6 pp; only paired contrasts resolve
> architecture-scale differences.

\newpage

# 3. Methodological foundation

## 3.1 The closed-set problem

A trial score is $s = \cos(f(x_1), f(x_2))$. For a speaker seen in training, $f$
has been explicitly optimised to collapse that speaker onto a learned
AAM-softmax centroid $w_c$, so the between-class margin enforced during training
transfers directly into the trial score --- the model is evaluated on the
objective it was trained on.

The inflation therefore **grows with model capacity**: a 95M SSL front end gains
more from the contamination than a 1.4M ResNet. A closed-set ranking of
architectures partly ranks *memorisation capacity*, and the two orderings are
not the same. This is the concrete mechanism behind the caveat P3 recorded.

`tools/build_splits.py` rebuilds every corpus into speaker-disjoint train /
validation / test partitions --- 70/10/20 by speaker, stratified by total
duration, deterministic from `--seed`, SHA-256 per emitted list. The shipped
`lists/` are untouched, so v0 results stay reproducible; they are simply not
comparable with v1.

## 3.2 The power ceiling

Trials sharing a speaker are correlated. With $m$ trials per speaker, $S$
speakers and intra-speaker correlation $\rho$:

$$n_{\text{eff}} = \frac{mS}{1 + (m-1)\rho} \longrightarrow \frac{S}{\rho} \quad \text{as } m \to \infty$$

Effective sample size is bounded by the **speaker count**, not the trial count.
At $S = 91$, $\rho = 0.7$: $n_{\text{eff}} \le 130$, giving roughly ±6 pp on a
single absolute EER.

Two consequences, both acted on:

1. Trial lists cut **80k -> 20k pairs** --- $m \approx 20$ already reaches 98 %
   of the ceiling, so the remainder was pure evaluation cost.
2. Inference uses a **paired speaker-clustered bootstrap** on EER differences
   over identical trials, where the speaker-difficulty term cancels. Validated
   on synthetic ground truth: **2.6x tighter** than absolute EER, correctly
   detects a known injected difference, correctly finds none between a system
   and itself. EER matches the repo's own `tuneThreshold` exactly.

## 3.3 Controlled factors

Pinned across every run: embedding dim 256, 80 mel bins, 2.0 s train / 3.0 s
eval crops, identical MUSAN+RIR augmentation, Adam 1e-3 with 0.95 decay and
wd 2e-5, 60 epochs, **per-epoch validation**, cosine scoring with AS-Norm off
during training.

Four documented deviations, each recorded in the affected runs' manifests:
VGGVox requires `n_mels=40` architecturally; `ssl_wavlm_ft` uses lr 1e-4 + LLRD
(1e-3 destroys a pretrained transformer); `en_full` capped at 10 epochs
(measured 341 h for 60); small corpora cap batch at 0.75x speaker count.

Model selection uses **validation trials only**; the test set is scored once, at
the validation-selected checkpoint.

## 3.4 Data conditions

| Condition | Corpus | Train speakers | Train utts | Role |
|---|---|---|---|---|
| `si` | slr52_sinhala | 336 | 129,042 | Sinhala read speech |
| `ta` | slr127_tamil | 446 | 63,338 | Tamil read speech |
| `si_celeb` | slceleb2026 | 82 | 31,284 | in-the-wild, real sessions |
| `si_pooled` | slr52 + slceleb | 418 | 160,326 | pooled domains |
| `en_matched` | VoxCeleb2 subset | 336 | 47,926 | scale-matched English |
| `en_full` | VoxCeleb2 -> Vox1-O | 5,871 | 1,065,152 | positive control |

\newpage

# 4. Stage F --- the front-end sweep (COMPLETE, 14/14)

The study's principal result. Backbone, loss, splits and schedule fixed; only
the input representation changes.

| Front end | si | ta | Family |
|---|---|---|---|
| **`ssl_mhubert_lw`** | **2.141** | **0.506** | SSL, layer-weighted |
| `ssl_wavlm_lw` | 2.801 | 0.587 | SSL, layer-weighted |
| `ssl_wavlm_low` (layer 0) | 3.701 | 1.113 | SSL, single layer |
| *`ecapa1024` --- mel baseline* | *3.721* | *0.810* | filterbank |
| `mfcc40d` | 3.942 | 1.032 | cepstral |
| `mfcc80` | 4.062 | 1.093 | cepstral |
| `ssl_wavlm_ft` (unfrozen) | 4.522 | 1.821 | SSL, fine-tuned |
| `ssl_wavlm_mid` (layer 6) | 4.742 | 1.579 | SSL, single layer |
| *`ssl_wavlm` --- layer 12* | *7.083* | *4.837* | SSL, last layer |

- **Layer choice was worth 3.3x.** Same encoder, same data: 7.083 -> 2.801,
  purely from how the layers are read.
- **Learned weighting beats every fixed depth** --- `lw` > layer 0 > layer 6 >
  layer 12. Cost: 13 scalar parameters.
- **Multilingual pretraining pays at equal capacity.** mHuBERT-147 and WavLM are
  both 94M; mHuBERT wins on both languages, isolating *coverage* from capacity.
- **MFCC is not the disaster predicted** --- within about 0.2 pp of log-mel. A
  pre-registered prediction, falsified.

## 4.1 The layer probe

Measured directly with **no training**: run the frozen encoder, mean-pool each
hidden state, cosine-score 4,000 validation trials.

| Layer | si EER % | ta EER % |
|---|---|---|
| **0** (CNN output) | **22.30** | **11.05** |
| 3 | 29.27 | 15.78 |
| 6 | 36.50 | 24.48 |
| 9 | 38.90 | 31.68 |
| **12** (the default) | **40.33** | **32.08** |

Monotonic in depth, in both languages. This is what masked-prediction
pretraining predicts: the objective rewards recovering a masked frame from
context, driving upper layers toward phonetic content and treating speaker
identity as nuisance.

> **A pre-registered prediction was falsified.** The 12 August log predicted
> speaker information would peak *mid-stack*, and `ssl_wavlm_mid` was built at
> layer 6 on that basis. Layer 6 measures 36.5 % --- barely better than the last
> layer. The entry was kept as the mid-curve data point and the falsification
> recorded, rather than edited away.

## 4.2 Relation to P3 --- and the experiment that would close the loop

P3 found performance **monotone in encoder freedom** (frozen 3.74 -> LoRA 3.17
-> LLRD 2.40 -> full 1.63 on si), closed-set, with a layer-weighted encoder.

v1 finds a **frozen, layer-weighted** encoder (2.141) beating an **unfrozen,
single-layer** one (4.522). These are not in direct contradiction, because the
two comparisons vary different things:

| | layer handling | encoder | si EER |
|---|---|---|---|
| P3 `frozen` (v0, closed) | layer-weighted | frozen | 3.74 |
| P3 `full` (v0, closed) | layer-weighted | fine-tuned | 1.63 |
| v1 `ssl_mhubert_lw` (open) | layer-weighted | frozen | 2.141 |
| v1 `ssl_wavlm_ft` (open) | **single, last** | fine-tuned | 4.522 |

**The missing cell is layer-weighted + fine-tuned on open-set data** --- exactly
P3's winning configuration, under v1's protocol. That single run is the direct
open-set arbiter for P3's headline claim, and it has not been done. It is the
highest-value next experiment in the programme (§10).

\newpage

# 5. Stage H --- does the backbone still matter under SSL features? (2/4)

Stage A varies the backbone with the front end fixed; Stage F varies the front
end with the head fixed. Stage H measures the interaction neither can see.

| System | si | ta |
|---|---|---|
| `ssl_mhubert_lw` (pooling head) | **2.141** | 0.506 |
| `ssl_mhubert_ecapa` (full ECAPA backbone) | 2.421 | **0.324** |

**The answer is language-dependent, which is itself the finding.** On Tamil the
ECAPA backbone helps substantially (0.506 -> 0.324, -36 %); on Sinhala it
*hurts* (2.141 -> 2.421, +13 %). At **0.324 %**, `ssl_mhubert_ecapa_ta` is the
best single result in the programme.

Two `ssl_wavlm_ecapa` runs are still training and will complete the 2x2.

# 6. Stage A --- architecture sweep (COMPLETE)

| Architecture | si | ta | Params |
|---|---|---|---|
| `ecapa512` | **3.642** | 1.032 | 5.99 M |
| `ecapa1024` | 3.721 | **0.810** | 14.46 M |
| `resnetse34v2` | 4.242 | 1.417 | 7.37 M |
| `resnetse34l` | 4.362 | 1.761 | 1.40 M |
| `mlpmixer` | 4.582 | 1.154 | 7.71 M |
| `vggvox` | 5.162 | 1.902 | 3.64 M |
| `ssl_wavlm` (layer 12) | 7.083 | 4.837 | 94.98 M |

**ECAPA-512 matches ECAPA-1024** at 40 % of the parameters --- the capacity
control suggests these corpora are data-limited rather than capacity-limited.
**The largest model finished last**, which is what triggered §4.1.

\newpage

# 7. Corpus difficulty --- the most important caveat

`si_pooled` trains on both Sinhala domains and evaluates them *separately*, so
read-speech and cross-session numbers are directly comparable for one model.

| Architecture | Read (slr52) | Wild (cross-session) | Ratio |
|---|---|---|---|
| `ecapa512` | **3.421** | **21.381** | 6.2x |
| `ecapa1024` | 3.662 | 21.521 | 5.9x |
| `resnetse34v2` | 4.022 | 22.563 | 5.6x |
| `mlpmixer` | 4.062 | 23.103 | 5.7x |
| `resnetse34l` | 4.522 | 25.726 | 5.7x |
| `vggvox` | 5.202 | 26.867 | 5.2x |

**A about 6x gap between read speech and genuine cross-session audio, for the same
model trained on both.** This is the single most important caveat on every
read-speech number in the programme --- including v0's, which used SLR52
throughout. The architecture *ranking* is nonetheless stable across domains
(`ecapa512` leads both), which is reassuring for external validity even where
absolute numbers do not transfer.

## 7.1 The Tamil channel artifact

SLR127 pools three collection sites (ISTL / MILE / MICI). Splitting its impostor
trials by whether both sides come from the same site, on `ecapa1024_ta`:

| Impostor type | Count | Mean score | EER |
|---|---|---|---|
| cross-batch | 4,805 | **-0.0433** | **0.475 %** |
| same-batch | 5,151 | **+0.0527** | **1.298 %** |
| all (as reported) | 9,956 | --- | 0.894 % |

Cross-batch impostors score *negative* --- rejected almost for free because the
channel differs, not the speaker. about 48 % of the trial list is that easy kind.
Reproduced across architectures at 2.47--3.20x inflation. SLR52 has no such
structure, so comparing the two corpora compares protocols as much as languages.

**Reporting rule: quote the same-batch figure for SLR127.** The like-for-like
Sinhala/Tamil gap is then about 3.4x, not about 4.9x.

## 7.2 English at matched scale

At an identical 336 training speakers, `ecapa1024` scores **7.180 %** on
`en_matched` against **3.721 %** on Sinhala. A domain effect, not a language one
--- VoxCeleb is in-the-wild, SLR52 is clean read speech --- reinforcing that
corpus condition dominates language in every comparison here.

## 7.3 A documented negative result

Trained alone on 82 speakers, `si_celeb` reached 23--28 % EER with
minDCF of about 0.99 across every architecture. Training worked (loss
7.34 -> 0.66, train accuracy 6 % -> 85 %): the models learned their training
speakers and transferred to none. Two design faults --- 82 speakers is far too
few, and the validation set has **9 speakers** ($n_{\text{eff}} \le 13$), which
cannot measure anything yet selected every checkpoint. Retained as a negative
result; its EERs should not be quoted. Its correct use is as an evaluation
probe (§7), where it is highly informative.

\newpage

# 8. Engineering and infrastructure

## 8.1 Fourteen bugs found and fixed

| Bug | Consequence if unfixed |
|---|---|
| Checkpoints loaded into the wrong wrapper class | Matched **0 tensors** --- silently evaluating untrained weights at a believable about 41 % EER |
| `int8` labels overflow the error-rate accumulator | Silent corruption of every EER past 127 target trials |
| GPU memory requirement guessed at 20 GB; **measured 0.9--1.5 GB** | Every SSL job stranded on one A40 while 15 GB T4s idled |
| Duplicate scheduling across queues | Two processes writing one checkpoint directory --- occurred twice |
| Reduced-scale runs shared `save_path` with real runs | Smoke checkpoints silently resumed by full runs |
| Batch size may not exceed the training-speaker count | Zero batches -> crash on the 82-speaker corpus |
| Stage F per-entry overrides not forwarded | `ssl_wavlm_ft` would have trained at lr 1e-3 and wrecked the encoder |
| AS-Norm cohort resolved against the wrong corpus root | Invalid cohort on every transfer and probe set |
| English conditions leaked into every evaluation | +57k trials per experiment, about 2.5x cost |
| ROC plot failed on every run (`1 - list`) | Fixed via `sitecustomize`; trainer untouched |

Plus: Stage F front ends leaking into Stage A, manifest clobbering across
invocations, `pkill -f` over ssh killing its own shell, and deliberate stops
being indistinguishable from node failures.

## 8.2 Cluster reliability

**Four cluster-wide reboots in five days** (13th 12:56, 14th 14:54, 16th 07:44,
17th 15:34), plus two nodes dropping off the network independently. Each reboot
kills every queue driver on the head node --- nothing survives, including tmux.

**No training work was lost.** Per-epoch checkpointing plus recorder-resume
meant every interruption cost at most one epoch. Verified: all five
reboot-interrupted runs have gap-free, duplicate-free epoch sequences across the
boundary.

## 8.3 The throughput collapse

Running 7 concurrent jobs collapsed *every* job to 5--16 Hz --- including
pure-mel runs with no SSL encoder. Diagnosis: 114 dataloader processes across
three nodes saturating a single NFS server, with GPU utilisation about 0 % and CPU
86--93 % idle. Capping concurrency at 3 restored **224--1,625 Hz, a 30--100x
recovery**. Fewer concurrent jobs finish the work strictly faster here.

\newpage

# 9. Current state

| Stage | Progress |
|---|---|
| A --- si | 7/7 complete |
| A --- ta | 7/7 complete |
| A --- si_celeb | 7/7 complete |
| A --- si_pooled | 6/7 |
| **F --- front ends** | **14/14 complete** |
| E --- english | 7/8 |
| H --- hybrids | 2/4 |

**50/54 trained, 24/54 evaluated.** Four running; the A40 is clearing a 28-run
evaluation backlog. One node remains unreachable.

# 10. Not yet done

1. **Held-out evaluation** --- 24 of 54. Everything above is validation EER.
2. **The paired bootstrap** --- no contrast has a confidence interval yet. Until
   then, *no ranking in this report is a claim*.
3. **The P3 arbiter run (§4.2)** --- layer-weighted + fine-tuned, open-set. The
   single experiment that resolves v0's headline claim under v1's protocol.
4. **Stages B (loss), C (combined si+ta), D (seeds), M (margin)** --- deferred.
   Stage D matters most: every result here is a single seed.
5. **The mel-scale question** --- untested since RawNet3 was removed on request.
6. **Channel-controlled Tamil re-scoring** --- the split exists and is applied
   automatically, but the figures above are still the uncontrolled ones.

# 11. Recommendations

1. **Report `ssl_mhubert_lw` as the headline system**, with
   `ssl_mhubert_ecapa` for Tamil, quoting the mel baseline alongside so the
   front-end gain is visible.
2. **Run the §4.2 arbiter next.** It closes the loop with P3 and determines
   whether the thesis's PEFT chapter needs revising for the open-set condition.
3. **Use same-batch Tamil figures** in any Sinhala/Tamil comparison.
4. **Quote the cross-session number** as the realistic performance estimate;
   read-speech EER is about 6x optimistic --- this applies retrospectively to v0.
5. **Run Stage D before publishing.** Single-seed differences of a few tenths of
   a point are not interpretable at this power.
6. **Escalate cluster stability.** Four reboots in five days, two nodes lost.

# 12. Impact on existing write-ups

- **Thesis ch. 1--4** cite v0 numbers throughout. They remain valid *as stated*
  --- v0 measured representation adaptation, not open-set generalisation --- but
  every quoted EER needs the closed-set caveat made explicit, and §7's about 6x
  read-vs-wild factor should temper the real-world framing.
- **`papers/ieee_spl`** is built on the P3 comparison. Its central claim
  (full FT beats LoRA) is still awaiting its open-set test (§4.2).
- **VoiceID** deploys `p3_full_s42`, a layer-weighted fine-tuned WavLM ---
  architecturally the configuration §4 supports. Its published thresholds were
  calibrated on closed-set trials and should be re-derived on v1 splits.

---

*Full detail: `experiments/README.md`, `experiments/FEATURE_EXTRACTION.md`, and
the logs of 2026-08-12 and 2026-08-14. Trainer `trainSpeakerNet.py` unmodified
throughout.*
