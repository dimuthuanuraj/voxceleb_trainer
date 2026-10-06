# Experiment Harness, Front-End Study, and the English Reference

**Date:** 2026-08-12 · **Status:** harness built and validated; 36 experiment scripts generated; no full-scale run executed yet
**Code:** `experiments/` (new) · **Design doc:** `experiments/README.md` · **Front-end doc:** `experiments/FEATURE_EXTRACTION.md`
**Trainer:** `trainSpeakerNet.py` — **unmodified**, invoked as a subprocess

---

## 0. Headline

Built a staged experimental programme to find the best architecture, loss and
front end for Sinhala and Tamil speaker verification, individually and jointly,
with English as a reference language.

Three findings landed before a single full experiment ran, and all three change
how the results must be produced and read.

| | Finding |
|---|---|
| **1** | **Every corpus in `data/` ships 100 % closed-set lists.** All test speakers appear in training — slr52 478/478, slr127 638/638, slceleb2026 123/123. Not merely optimistic: **rank-distorting** for architecture comparison. |
| **2** | **EER resolution is capped by *speaker* count, not trial count.** At 91 held-out Sinhala speakers, a single absolute EER carries ±5–6 pp — coarser than the differences being chased. Ranking must use a **paired** statistic. |
| **3** | **The research and production stacks were never the same model.** `voiceid` learns a softmax weighting over all 13 WavLM layers; the trainer reads only the last one — the layer masked-prediction pretraining works hardest to strip speaker identity out of. Nothing had measured the difference. |

Plus **seven** real bugs found by smoke-testing and review, any one of which
would have produced plausible-looking but wrong numbers, or silently inflated
cost (§5).

---

## 1. The closed-set problem

### 1.1 What was found

```
corpus                train spk   test spk   overlap
slr52_sinhala             478        478     478  (100 %)
slr127_tamil              638        638     638  (100 %)
slceleb2026_sinhala       123        123     123  (100 %)
slr65_tamil                50         49      49  (100 %)
```

### 1.2 Why it is worse than "optimistic"

A trial score is `s = cos(f(x₁), f(x₂))`. For a speaker seen in training, `f`
has been explicitly optimised to collapse that speaker onto a learned class
centroid `w_c` — the AAM-softmax weight column. The between-class margin the
loss enforces at training time therefore **transfers directly into the trial
score**: the model is being evaluated on the objective it was trained on.

The size of that inflation grows with a model's capacity to memorise centroids.
A 95 M-parameter SSL frontend gains more from the contamination than a
1.4 M-parameter ResNet. So a closed-set ranking of architectures is partly a
ranking of *memorisation capacity*, and the two orderings are not the same.
This is the concrete mechanism behind the "closed-set caveat" that
`2026-07-03-*` logs flag without naming.

### 1.3 What was done

`experiments/tools/build_splits.py` rebuilds every corpus into speaker-disjoint
partitions: 70/10/20 by speaker, stratified by total speech duration (gender is
`unk` throughout slr52/slr127, so duration decile is the only usable
stratifier), deterministic from `--seed`, SHA-256 of every emitted list in the
manifest.

```
corpus                train / val / test spk    train utts    test trials
slr52_sinhala              336 / 51 / 91          129,042        19,838
slr127_tamil               446 / 61 / 131          63,338        19,912
combined_si_ta             782                    192,380        per-language
```

Four corpora are **held out entirely** — never trained on — as cross-corpus
generalisation probes: `slceleb2026_sinhala`, `slr65_tamil`, `kathbath_tamil`,
`nisp_tamil` (the last being the only genuine cross-lingual trials, same
speakers in two languages).

**The shipped `lists/` were left untouched**, so results in the thesis and
papers that cite them remain reproducible. They are simply not comparable with
anything produced by this programme, and mixing the two would be an error.

---

## 2. The power ceiling — why the trial lists got *smaller*

Trials sharing a speaker are correlated. With `m` trials per speaker, `S`
speakers and intra-speaker correlation `ρ`:

```
n_eff = m·S / (1 + (m−1)·ρ)   →   S / ρ    as m → ∞
```

**Effective sample size is bounded by the speaker count**, no matter how many
pairs are synthesised. At `S = 91`, `ρ = 0.7`: `n_eff ≤ 130`, and

```
MDE ≈ 2.8 · √(2p(1−p)/n_eff) ≈ 5.9 pp
```

Two consequences, both acted on:

1. **Trial count barely matters.** At `m ≈ 20` the design already reaches 98 %
   of the ceiling. The default test list was cut **80k → 20k pairs**: a 4×
   saving in evaluation cost for no loss of resolving power. Generating more
   pairs is measurement theatre.

2. **The comparison must be paired.** Writing per-speaker error rates as
   `e_A(s) = μ(s) + a(s)` and `e_B(s) = μ(s) + b(s)`, the speaker-difficulty
   term `μ(s)` is common to both systems on identical trials and **cancels in
   the difference**. `analyze.py` resamples *speakers* with replacement and
   reports the distribution of `EER_A − EER_B`.

Validated against synthetic data with known ground truth (100 speakers, 200
trials each, system B made genuinely 0.14 worse on targets):

```
EER vs repo's own tuneThreshold      22.2100  vs  22.2100   (exact match)
d′                                   1.534    (true separation 1.6)
minCllr ≤ Cllr                       0.6648 ≤ 0.7794        ✓
paired A vs B   Δ −2.01 pp, CI [−2.32, −1.81], p 0.000, significant   ✓
paired A vs A   Δ  0.00 pp, CI [ 0.00,  0.00], not significant        ✓
absolute EER CI width  1.340 pp
paired  ΔEER CI width  0.510 pp   →  2.6× tighter
```

> **Reporting rule for this programme: quote the paired contrasts, not absolute
> EER gaps.** An absolute EER difference of 1 pp between two architectures on
> one corpus is inside the noise; the paired contrast is not.

Beyond EER, three diagnostics that say *why* a system wins: `d′` (score
separation in pooled-SD units; divergence from EER exposes non-Gaussian score
behaviour), **minCllr** (information cost over all operating points after
optimal monotonic recalibration by PAV — a system ahead on EER but behind on
minCllr wins at one threshold and loses overall), and **Cllr − minCllr**, the
calibration loss, which is exactly what AS-Norm and per-language thresholds can
recover without retraining.

---

## 3. The harness

`trainSpeakerNet.py` is **not modified**. It is launched as a subprocess and its
per-epoch stdout is parsed into structured JSONL. This keeps every previously
published number on the same code path as the new ones.

```
experiments/
├── registry.py           single source of truth for every experiment
├── common/
│   ├── lossspec.py       all 8 losses: LaTeX objective, geometry, pre-registered prediction
│   ├── modelcard.py      architecture specs + live introspection (params/module, GFLOPs)
│   ├── recorder.py       structured-folder writer + environment capture
│   ├── runner.py         subprocess wrapper; trainer stdout → JSONL
│   └── harness.py        shared entry point, with registry-drift detection
├── tools/
│   ├── build_splits.py       speaker-disjoint splits
│   ├── build_en_splits.py    the two English conditions
│   ├── fetch_ssl_encoder.py  encoders as safetensors
│   ├── gen_scripts.py        registry → one standalone .py per experiment
│   ├── run_queue.py          dispatch across the cluster's GPUs
│   ├── evaluate.py           held-out scoring, retains per-trial scores
│   ├── layer_weights.py      where speaker info lives in an SSL encoder
│   └── analyze.py            paired bootstrap, diagnostics, tables, figures
├── scripts/              generated, self-documenting, one per experiment
└── results/<exp_id>/     manifest · epochs.jsonl · config · command · stdout · final · test_eval · scores/*.npz
```

Per run, `epochs.jsonl` carries **one row per epoch** (train loss, train metric,
LR, val EER, minDCF, threshold, per-language breakdown, AS-Norm diagnostic, wall
clock, epoch duration), and `manifest.json` carries everything knowable before
epoch 1 — resolved config, measured model card, loss spec, SHA-256 of every data
list, git commit + dirty flag, torch/CUDA/driver, GPU model, package versions.

`scores/*.npz` keeps **per-trial scores with the speaker id behind each side** —
without which the paired bootstrap is impossible.

Two design choices worth recording:

* **Deferred stages.** A, F and H are fixed; B, C, D, G, M compute their
  membership from the preceding stage's results at generation time, and the
  selection is written to `scripts/stage_<X>_manifest.json`. No GPU-hour is
  spent on a cell whose outcome is already determined.
* **Pre-registered predictions.** Every loss and architecture carries a
  `predicted_effect` written *before* any run, so the analysis is scored against
  a hypothesis rather than rationalised afterwards.

### Model selection

Epoch selection uses **validation trials only**. The test set is scored exactly
once per experiment, at the checkpoint validation already chose. A test EER that
is the minimum over 30 peeks is not a held-out estimate.

---

## 4. The experimental programme

Scripts exist for A, E, F and H (36 in total); B, C, D, G and M are deferred by
design — see §9.1 for the generation status and what each deferred stage is
waiting on.

| stage | varies | runs | question |
|---|---|---|---|
| **A** | architecture (7) × {si, ta} | 14 | which backbone? |
| **B** | loss (5) × top-2 arch × {si, ta} | 20 | which objective? |
| **C** | best pairs on `combined` | 2–4 | does joint si+ta training help? |
| **D** | winners × 3 seeds × 3 conditions | ~12 | is the ordering stable? |
| **E** | English reference | 2–14 | how does the ranking compare to a known language? |
| **F** | front end (5 new) × {si, ta} | 10 | what should the network be fed? |
| **G** | technique (6) × condition | ~14 | which universal feature flags pay? |
| **H** | SSL front end × full backbone | 4 | does the backbone still matter under SSL features? |
| **M** | AAM margin (4) × 3 conditions | 12 | the hypersphere-packing question |

**Controlled factors**, pinned across Stage A: embedding dim 256 for every
backbone (defaults differ 192/256/512 and width alone moves EER), 80 mel bins,
2.0 s train / 3.0 s eval crops, identical MUSAN+RIR augmentation, Adam 1e-3 with
0.95 step decay and wd 2e-5, 60 epochs / eval every 2 / patience 8, cosine
scoring with **AS-Norm off during training** (it is an inference-time transform;
folding it in would confound the backbone comparison with a calibration effect,
so it is measured as an explicit on/off factor in `evaluate.py` instead).

One documented deviation: **VGGVox requires `n_mels=40`** — its final
`Conv2d(kernel=(4,1))` is sized against a 40-bin axis and raises otherwise. It
is recorded in the affected runs' manifests and quoted with that caveat.

**RawNet3 was removed from the study on request.** The module and its spec
remain, so restoring it is a one-line change. Its removal leaves the
learned-filterbank question — whether the mel scale, fitted to English and
European perceptual data, is right for Sinhala and Tamil — **untested**. Recorded
as a known limitation rather than quietly dropped.

---

## 5. Bugs found by smoke-testing

Every one of these produces a plausible-looking number rather than a crash,
which is what makes them worth recording.

| # | Bug | Consequence if unfixed |
|---|---|---|
| 1 | **Checkpoints load into `SpeakerNet`, not `WrappedModel`.** `saveParameters` writes `self.__model__.module.state_dict()` → bare `__S__.*` keys; a `WrappedModel` looks for `module.__S__.*`. | Matched **0 tensors**, silently evaluating **untrained weights** and reporting a believable ~41 % EER. `evaluate.py` now hard-fails at 0 matched tensors. |
| 2 | **`int8` labels overflow `ComputeErrorRates`.** It accumulates a running sum in the array's own dtype. | Silent corruption of **every EER** past 127 targets. Labels are now int32 and cast to Python ints at the metric boundary. |
| 3 | **Reduced-scale runs shared `save_path` with the real run.** The trainer resumes by globbing `model0*.model`. | A 2-epoch smoke checkpoint would be **silently resumed** by the subsequent full run. |
| 4 | **AS-Norm cohort resolved against the evaluation corpus's root.** | Cohort paths like `<tamil_corpus>/wav/si/<sinhala_speaker>/…` on every transfer and probe set. |
| 5 | **Stage F front ends leaked into Stage A.** Registering them in `ARCHITECTURES` at import time enlarged the backbone sweep. | Stage A silently became a 12-architecture sweep. `STAGE_A_ARCHS` is now frozen before registration. |
| 6 | **The English conditions leaked into every experiment's evaluation.** `evaluate.py` builds its cross-language transfer sets by iterating `CONDITIONS`, so adding `en_full`/`en_matched` silently enrolled them. | Every si/ta model was about to be scored against VoxCeleb1-O (37,720 trials) and `en_matched` (20,000) on top of its own — **39,750 → 97,470 trials per experiment**, ~2.5× the evaluation cost across ~28 experiments, plus pulling in the whole VoxCeleb1 audio tree. The English conditions are *controls*, not transfer targets: they now carry `auto_transfer: False` and are opt-in via `evaluate.py --transfer-en`. |
| 7 | **`gen_scripts.py` clobbered a stage manifest when the stage was generated in more than one invocation.** | Stage E is built as two calls (7 backbones on `en_matched`, then `ecapa1024` on `en_full`); the manifest ended up claiming **1 experiment while 8 scripts sat on disk**. The manifest is the audit record this whole design leans on. It now merges by `exp_id`, keeps a `selection_history`, and reports any script on disk that no manifest entry describes (`scripts_on_disk_not_in_manifest`). |

Three trainer constraints were also discovered and worked around **without
touching the trainer**:

* `--channels` is not an argparse option and the YAML loader *discards unknown
  keys* (`trainSpeakerNet.py:227`) → the ECAPA-512 capacity control needed its
  own module, `models/ECAPA_TDNN_C512.py`. Same for `n_mfcc` / `mfcc_deltas`.
* `--augment` and `--log_input` are `type=bool`, so a bare `--augment` is an
  argparse error and `--augment False` evaluates to `bool("False") == True`.
  Only safe encodings are `--augment True` or omitting the flag.
* transformers ≥ 4.56 refuses `torch.load` on a `.bin` below torch 2.6
  (CVE-2025-32434), and `SL_SPV` runs torch 2.5.1 → SSL encoders are
  materialised as **safetensors** snapshots under `models/weights/`. Upgrading
  torch was rejected: it would put these runs on a different numerical stack
  from every previously published result.

---

## 6. The front-end study (Stage F)

### 6.1 What VoiceID actually uses — and the gap it exposed

`voiceid/backend/app/speaker_model.py` (`SSLSpeakerNet`):

```
waveform → WavLM-base-plus (94 M)
         → learned layer weighting   w = softmax(θ) ∈ R¹³,  h = Σ_l w_l·H_l
         → ASP pooling → Linear → 192-d
         → AAM-Softmax head (training only)
```

Not a two-stage "extract embeddings then train a classifier" — one end-to-end
network in which WavLM *is* the front end. **No Whisper anywhere** (the cached
Whisper models belong to `SLT_Zoom_Project`). Checkpoint `p3_full_s42/best.pt`,
encoder fully fine-tuned (full FT beat LoRA: 1.69/1.41 vs 3.73/3.10).

**The gap:** the trainer's `SSLFrontendSpeaker` reads **one** layer
(`--ssl_layer`, default `-1` = last). This matters because masked-prediction
pretraining drives the upper layers toward phonetic content, for which speaker
identity is *nuisance*; speaker information survives most strongly in the lower
and middle layers. **The default is close to the worst available choice for a
speaker task.** `models/SSLFrontendSpeakerLW.py` ports the production design
into the trainer so the two can finally be compared.

**Early evidence.** After only 2 smoke epochs, the fitted weights already drift
in the predicted direction — mass above uniform on states 0–4, below on 7–12:

```
[0.0801 0.0796 0.0804 0.0797 0.0797 0.0784 0.0760
 0.0749 0.0741 0.0739 0.0746 0.0744 0.0741]     uniform = 0.0769
argmax = state 2 · L1 deviation from uniform = 0.0329
```

`tools/layer_weights.py` extracts this per run and reports peak, centre of mass,
lower/upper mass split and normalised entropy. **The fitted profile is itself a
result**: its argmax states where speaker information lives in that encoder for
Sinhala and Tamil, comparable against the same encoder's English profile.

### 6.2 The four families

| family | why it is / is not the right default |
|---|---|
| **MFCC** | The DCT decorrelates filterbank channels — essential for diagonal-covariance GMM-UBM and i-vector systems, **actively harmful** for a CNN/TDNN, which *wants* the inter-band correlations where formant structure lives. Truncation also discards fine spectral detail carrying speaker-specific glottal information. **Predicted to lose.** Run anyway, twice, to get the number. |
| **log-mel** | The study's baseline and the right input for convolutional backbones. Caveat: the mel scale was fitted to *English and European* perceptual data; whether its warping suits Sinhala and Tamil is unestablished — and now untested, since RawNet3 was removed. |
| **learned filterbank** | Removed with RawNet3. |
| **SSL** | Strongest option, at 6.6× the parameters and 4.3× the compute of ECAPA-1024 (94.98 M / 22.1 GFLOPs vs 14.46 M / 5.17). Two separable questions: *which layer* (§6.1) and *which encoder* (§6.3). |

### 6.3 Coverage vs capacity — the clean contrast

| encoder | params | pretraining languages | role |
|---|---|---|---|
| WavLM-base-plus | 94.98 M | English only | control |
| **mHuBERT-147** | **94.97 M** | **147, incl. si + ta** | **the clean contrast** |
| XLS-R-300m | 300 M | 128, incl. si + ta | confounded — bigger *and* multilingual |

mHuBERT-147 is the informative one: **the same parameter count as WavLM**, so a
difference between them is attributable to *pretraining coverage* rather than
capacity. Fetched and verified (94,371,712 params, hidden 768, 12 layers).

### 6.4 Why not Whisper

ASR training explicitly rewards **speaker invariance** — a good transcriber maps
the same words from different voices to the same output, so speaker identity is
precisely the nuisance variable it is trained to discard. Masked prediction has
no such incentive. Whisper's encoder also consumes fixed 30 s windows against
2–8 s verification segments, and large-v3 is ~16× WavLM-base-plus. Excluded on
priors; the cheapest defensible test (whisper-small in `SSLFrontendSpeakerLW`,
`si` only, one run) is documented in `FEATURE_EXTRACTION.md` §4 if wanted.

---

## 7. Stage H — the interaction nobody had measured

Stage A varies the backbone with the front end fixed; Stage F varies the front
end with the head fixed. **Neither measures the interaction**, which is where
the actionable question lives:

> once the features come from a pretrained SSL encoder, does the backbone still
> matter?

`models/SSL_ECAPA.py` supplies the missing cell — SSL encoder + learned layer
weighting feeding the **full ECAPA-TDNN backbone**, with only `layer1` re-sized
from `n_mels` to the encoder's hidden dim. Every other block is identical to
Stage A's `ecapa1024`, so the contrast isolates the front end. Measured:
**112.37 M params, 25.3 GFLOPs**.

```
log-mel + ECAPA backbone      Stage A  ecapa1024        14.46 M   5.17 GFLOPs
WavLM   + pooling head        Stage F  ssl_wavlm_lw     94.98 M  22.10 GFLOPs
WavLM   + ECAPA backbone      Stage H  ssl_wavlm_ecapa 112.37 M  25.30 GFLOPs
```

If the backbone upgrade buys much less under SSL features than under log-mel,
**representation dominates architecture** in this low-resource regime — spend
the effort on the front end, not on backbone search. This is also the standard
WavLM+ECAPA recipe in current SV systems, so the study measures the known-strong
arrangement rather than an invention of its own.

**Caveat recorded in the module:** ECAPA's dilations (2, 3, 4) were tuned for
100 Hz mel frames; SSL encoders emit **50 Hz**, so the receptive field spans
twice the intended time. Not necessarily bad for speaker modelling, but it
travels with the front-end change and must not be attributed to it.

---

## 8. English, and why there are two conditions

| condition | speakers | train | test | job |
|---|---|---|---|---|
| `en_full` | 5,871 | 1,065,152 utts | VoxCeleb1-O (canonical) | **positive control** |
| `en_matched` | **336** | 47,926 utts / ~107 h | 100 held-out speakers | **fair cross-language comparison** |

English is **already speaker-disjoint by construction** (VoxCeleb2-dev →
VoxCeleb1-O, overlap 0), so `en_full` keeps the canonical trial list untouched.
That is deliberate: its EER is then directly comparable with the published
literature, and if ECAPA lands near the ~1 % this recipe is known to give, the
whole harness — loader, augmentation, scoring, checkpoint selection — is
validated end to end. **If it does not, every Sinhala and Tamil number here is
suspect.** It is also the source checkpoint for Stage G's cross-lingual
fine-tune arms.

`en_full` cannot answer "does this architecture suit Sinhala better than
English": at 5,871 vs 336 speakers, a ranking difference would measure how each
architecture *scales with data*. `en_matched` subsamples VoxCeleb2 to **exactly**
the Sinhala speaker count.

Matching is on **speakers (exact)** and **total speech hours (approximate)**,
not utterance count — VoxCeleb utterances average **8.04 s** (measured on a
400-file sample) against slr52's 4.38 s, so equal counts would mean unequal
audio, and duration is what bounds how much distinct speech a speaker
contributes under random-crop sampling.

> **Honest limitation.** VoxCeleb2 speakers average ~182 utterances, so 336 of
> them top out near 137 h against slr52's 157 h; `en_matched` reaches **107 h,
> 68 %** of the reference. Adding speakers would fix the hours but break the
> exact speaker-count match, which is the more important half of the control.
> Recorded in the manifest as `hours_match_pct` and
> `hours_ceiling_for_this_speaker_count`. **Quote the architecture *ranking*
> across conditions, not the absolute EER gap.**

---

## 9. Validation status

All Stage A backbones and all Stage F front ends smoke-tested end to end on the
cluster (2 epochs, real data, real GPUs, `exit_code 0`). Smoke EERs are
meaningless by design — the check is that every configuration builds, trains,
evaluates and records.

| architecture | params | GFLOPs/2 s | s/epoch |
|---|---|---|---|
| `resnetse34l` | 1.40 M | 1.79 | 22.7 |
| `vggvox` | 3.64 M | 1.06 | 12.9 |
| `ecapa512` | 5.99 M | 1.93 | 29.2 |
| `resnetse34v2` | 7.37 M | 9.25 | 33.2 |
| `mlpmixer` | 7.71 M | 3.03 | 25.7 |
| `ecapa1024` | 14.46 M | 5.17 | 29.2 |
| `ssl_wavlm` | 94.98 M | 22.10 | 101.2 |
| `mfcc80` / `mfcc40d` | 14.46 / 14.67 M | 5.17 / 5.25 | ~29 |
| `ssl_wavlm_mid` / `_lw` / `ssl_mhubert_lw` | ~94.98 M | 22.10 | ~105–110 |
| `ssl_wavlm_ecapa` / `ssl_mhubert_ecapa` | 112.37 M | 25.30 | — |

Statistics separately validated against synthetic ground truth (§2). Flag
validation: every emitted CLI option checked against the trainer's argparse
across all 50 experiments in stages A/F/E/G — **no unknown flags**.

---

### 9.1 Script generation status

| stage | scripts | status |
|---|---|---|
| **A** architecture | 14 | generated |
| **E** English reference | **8** | generated |
| **F** front end | 10 | generated |
| **H** SSL × backbone | 4 | generated |
| **B** loss sweep | 0 | deferred — needs Stage A's top-2 architectures |
| **C** combined-language | 0 | deferred — needs the best (arch, loss) pairs |
| **D** seed replication | 0 | deferred — needs the winners |
| **G** technique ablation | 0 | deferred — arch/loss from A/B, plus the English checkpoint for 2 of 6 techniques |
| **M** margin sweep | 0 | deferred — needs the best classification loss |
| | **36** | one script per stage dry-run verified |

Stage E is **8 runs, deliberately asymmetric**: 7 backbones on `en_matched` (the
cross-language ranking comparison, which requires the scale match) plus
`ecapa1024` alone on `en_full` (the positive control — one run against a known
reference number, and the source checkpoint Stage G's fine-tune arms consume).
Running all 7 backbones on `en_full` would cost 7 × 1.07 M utterances for a
question `en_matched` already answers better.

**Stage G is partially unblocked but deliberately not generated.** Of its 14
runs, 8 (`baseline`, `plda`, `lang_aux`, `dann_lang`) need no English
checkpoint; only `finetune_en` and `finetune_en_llrd` do. But Stage G's
*architecture and loss* come from Stages A/B, and generating now would fall back
to `ecapa1024 + aamsoftmax` and bake an unvalidated choice into the scripts —
exactly what the deferred-stage design exists to prevent.

### 9.2 Cluster GPU inventory and memory-aware placement

Re-probed 2026-08-12. The queue was using only 3 of the 5 usable GPUs.

| node | IP | GPUs | free | usable |
|---|---|---|---|---|
| compute-node-1 | 10.222.1.119 | 2× Tesla T4 | 14,914 MiB each | ✅ **added 2026-08-12 — was idle** |
| compute-node-2 | 10.222.1.118 | 2× Tesla T4 | — | ❌ **driver not loaded** |
| compute-node-3 | 10.222.1.120 | 2× NVIDIA A10 | 22,595 MiB each | ✅ |
| compute-node-4 | 10.222.1.121 | 1× NVIDIA A40 | 45,489 MiB | ✅ |
| compute-node-5 | 10.222.1.125 | none | — | dev node |
| head-node | 10.222.1.116 | none | — | — |

**compute-node-2's GPUs are physically present** — `lspci` reports two
`TU104GL [Tesla T4]` — but there are no `/dev/nvidia*` device nodes and
`nvidia-smi` fails to communicate with the driver. This is the same fault
compute-node-4 had earlier in 2026 and it needs root (`modprobe` or a reboot).
Recovering it would add **2 more T4s**. Re-add it to `CANDIDATE_NODES` in
`tools/gpurun.sh` once `ssh compute-node-2 nvidia-smi` works.

Adding compute-node-1 took the pool from **3 to 5 GPUs (+67 %)**, but the
cluster is now heterogeneous — 15 GB T4s against 23 GB A10s and a 46 GB A40 —
so a first-come-first-served scheduler would hand a 112 M-parameter hybrid to a
T4 and lose the run to an OOM minutes in, while the A40 sat idle.

`run_queue.py` is therefore **memory-aware**: every architecture declares
`min_gpu_mb` (9,000 for mel backbones, 20,000 for the SSL and hybrid models),
each worker scans the pending list for the first job its slot can hold, and jobs
that fit nowhere are reported up front rather than failing late. Verified: a
mel backbone runs to completion on a T4, and a Stage H dry-run correctly leaves
both T4s idle with `4 job(s) need a larger GPU` rather than taking work it
cannot finish.

> A subtlety worth recording: the front-end registration loop copied a
> hand-listed subset of keys into `ARCHITECTURES`, so `min_gpu_mb` was silently
> dropped and `ssl_mhubert_lw` came out eligible for a T4. It now copies the
> whole entry — an explicit key list quietly loses any field added later.

### 9.3 Where the English data is actually used

Three places, only one of which trains on English audio:

| where | uses English how | on by default |
|---|---|---|
| **Stage E** | *trains* on it — `en_matched` and `en_full` | yes, when Stage E runs |
| **Stage G** | consumes the English **checkpoint** as `--initial_model`; training data is still si/ta | only with `--initial-model` |
| **`evaluate.py`** | scores si/ta models *against* English test sets (`transfer_en_*`) | **no — opt-in via `--transfer-en`** (see §5 bug 6) |

**Stages A, F and H never touch English.** They run on `si` and `ta` only.

---

## 10. What has NOT been done

* **No full-scale experiment has run.** Everything above is harness, splits,
  design and validation. Stage A is ~14 runs and is the next step.
* **The mel-scale question is untested** (RawNet3 removed).
* **`n_utts == n_sessions` in both anchor corpora** — every utterance is its own
  recording, so there is no within-speaker session structure and cross-session
  trials cannot be distinguished from cross-utterance ones. With the QC audit's
  `η²(SNR|speaker)` of 0.68–0.74, **channel is partly identity** here. This is
  why the held-out *corpus* probes carry the generalisation claim rather than
  in-corpus test EER.
* **Read speech only** in the anchor pair; the in-the-wild check
  (`slceleb2026_sinhala`) has no Tamil counterpart, so in-the-wild Tamil remains
  unmeasured.
* **Nothing committed to git.** ~40 files staged clean, bulk artefacts ignored.

---

## 11. Next steps

All 36 generated scripts are ready to run as-is:

```bash
# 1. the ranking everything downstream is conditioned on
python experiments/tools/run_queue.py     --stage A            # 14 runs
python experiments/tools/evaluate.py      --all --stage A --probes
python experiments/tools/analyze.py

# 2. the positive control — run early regardless of budget
python experiments/tools/run_queue.py     --only E_ecapa1024_aamsoftmax_en_full_s42

# 3. the front-end question, most likely to pay
python experiments/tools/run_queue.py     --stage F            # 10 runs
python experiments/tools/layer_weights.py --all               # where speaker info lives

# 4. the interaction, and the cross-language ranking
python experiments/tools/run_queue.py     --stage H            # 4 runs
python experiments/tools/run_queue.py     --stage E            # 8 runs

# 5. stages that unlock once results exist
python experiments/tools/gen_scripts.py   --stage B --top 2
python experiments/tools/gen_scripts.py   --stage G \
    --initial-model exps/E_ecapa1024_aamsoftmax_en_full_s42/model/model_best.model
```

**Priority if GPU time is short:**

1. **A** — everything else is conditioned on this ranking.
2. **E `en_full`** — the positive control. Worth running early *regardless of
   budget*: if ECAPA does not land near the ~1 % this recipe is known to give on
   VoxCeleb1-O, the harness has a fault and every Sinhala/Tamil number is
   suspect. Better to discover that before spending days on stages A–H.
3. **F** — the layer-weighting question, the change most likely to pay for
   itself (13 extra scalar parameters).
4. **H** → **E `en_matched`** → **B** → **G** → **C/D/M**.

---

## 12. Files

**New code:** `experiments/` (registry, common/, tools/, scripts/, splits/)
**New models:** `ECAPA_TDNN_C512.py`, `ECAPA_TDNN_MFCC.py`,
`ECAPA_TDNN_MFCC40D.py`, `SSLFrontendSpeakerLW.py`, `SSL_ECAPA.py`
**New weights:** `models/weights/wavlm-base-plus/`, `models/weights/mhubert-147/`
**Docs:** `experiments/README.md`, `experiments/FEATURE_EXTRACTION.md`
**Generated scripts:** 36 under `experiments/scripts/` — A 14, E 8, F 10, H 4,
with `stage_<X>_manifest.json` recording each stage's membership and the
selection that produced it
**Unchanged:** `trainSpeakerNet.py`, `SpeakerNet.py`, `DatasetLoader.py`, all
existing models, all existing `configs/`, all shipped `data/*/lists/`
