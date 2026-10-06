# Sinhala / Tamil speaker verification — experimental programme

A staged, statistically-grounded search for the best architecture and training
objective for Sinhala and Tamil speaker verification, individually and jointly.

Everything here is additive: no existing file in the repository was modified,
and `trainSpeakerNet.py` is invoked as a subprocess exactly as it always was.

---

## 0. Read this first: the closed-set problem

Every corpus under `data/` ships a `lists/train_list.txt` and a
`lists/test_list.txt` **whose speaker sets are identical**:

| corpus | train speakers | test speakers | overlap |
|---|---|---|---|
| `slr52_sinhala` | 478 | 478 | **478 (100%)** |
| `slr127_tamil` | 638 | 638 | **638 (100%)** |
| `slceleb2026_sinhala` | 123 | 123 | **123 (100%)** |
| `slr65_tamil` | 50 | 49 | **49 (100%)** |

An EER measured on speakers seen during training is not a verification result.
Worse for this study's purpose, it is *rank-distorting*: a trial score is
`cos(f(x1), f(x2))`, and for a training speaker `f` has been explicitly
optimised to collapse that speaker onto a learned class centroid. The margin the
loss enforces during training transfers straight into the trial score, so the
inflation grows with a model's capacity to memorise centroids. A 95M-parameter
SSL frontend gains more from the contamination than a 1.4M-parameter ResNet.
Ranking architectures on those lists would partly rank memorisation capacity.

`tools/build_splits.py` therefore rebuilds every corpus into **speaker-disjoint**
train / validation / test partitions, and synthesises fresh trials over held-out
speakers only. All numbers produced by this programme come from those splits.
The shipped `lists/` are left untouched — prior results that cite them remain
reproducible, they simply are not comparable with these.

---

## 1. Design

### Data conditions

| key | corpus | train speakers | train utts | purpose |
|---|---|---|---|---|
| `si` | `slr52_sinhala` | 336 | 129,042 | Sinhala only |
| `ta` | `slr127_tamil` | 446 | 63,338 | Tamil only |
| `combined` | union | 782 | 192,380 | joint bilingual |
| `en_full` | VoxCeleb2 → VoxCeleb1-O | 5,871 | 1,065,152 | positive control + transfer source |
| `en_matched` | VoxCeleb2 subset | **336** | 47,926 | scale-matched cross-language comparison |

The two English conditions do different jobs. `en_full` keeps the canonical
VoxCeleb1-O trial list so its EER is comparable with published numbers — if
ECAPA lands near the ~1% this recipe is known to give, the whole harness is
validated end to end; if it does not, every Sinhala and Tamil number is suspect.
`en_matched` holds data volume constant against `si` (identical speaker count)
so that a *ranking* difference across languages is attributable to the language
rather than to corpus size. See [`FEATURE_EXTRACTION.md`](FEATURE_EXTRACTION.md)
§7, including the honest note that `en_matched` reaches only 68% of `si`'s
training hours because VoxCeleb2 speakers do not carry enough audio.

`slr52` and `slr127` are the anchor pair because they are *matched*: both are
OpenSLR read speech at comparable scale, so the difference between them is the
language rather than the recording style. Pooling in the in-the-wild Sinhala
corpus would have confounded the two.

Four further corpora are **held out entirely** — never trained on by any run —
and used as cross-corpus generalisation probes: `slceleb2026_sinhala`,
`slr65_tamil`, `kathbath_tamil`, `nisp_tamil`. NISP additionally provides the
only genuine cross-lingual trials (the same speakers in two languages).

### Stages

Each stage prunes the next, so no GPU-hour is spent on a cell whose outcome is
already determined.

| stage | what varies | runs | question |
|---|---|---|---|
| **A** | architecture (7) × language (2) | 14 | which backbone? |
| **B** | loss (5) × top-2 arch × language (2) | 20 | which objective? |
| **C** | best pairs trained on `combined` | ~2–4 | does joint training help? |
| **D** | winners at 3 seeds × 3 conditions | ~12 | is the ordering stable? |
| **E** | English reference (`en_matched`, `en_full`) | 2–14 | how does the ranking compare to a known language? |
| **F** | front end / input representation (5 new) × language (2) | 10 | what should the network be fed? |
| **G** | training technique (6) × condition | ~14 | which of the repo's feature flags actually pay? |
| **H** | SSL front end × full ECAPA backbone | 4 | does the backbone still matter under SSL features? |
| **M** | AAM margin sweep (4 values) × 3 conditions | 12 | the geometry question |

Stage A and F are fully determined now. B, C, D, G and M are *deferred*: their
membership is computed from the preceding stage's results at generation time,
and the selection is recorded in `scripts/stage_<X>_manifest.json`.

**Stages E, F and G are documented in detail in
[`FEATURE_EXTRACTION.md`](FEATURE_EXTRACTION.md)**, which also answers what the
VoiceID product uses today, why MFCCs are the wrong default for a convolutional
backbone, why Whisper is not used for embeddings, and why English needs two
conditions rather than one.

> **RawNet3 was removed from the study on request (2026-08-12).** The module
> `models/RawNet3.py` and its architecture spec remain in place, so restoring it
> is a one-line change to `ARCHITECTURES` in `registry.py`. Its removal leaves
> the learned-filterbank question — whether the mel scale, fitted to English and
> European perceptual data, is right for Sinhala and Tamil — untested. That is
> recorded as a known limitation rather than quietly dropped.

### What is held fixed

For an architecture comparison to mean anything, everything else must be pinned:

* **embedding dim 256** for every backbone (defaults differ: 192 ECAPA, 256
  ResNet, 512 Mixer — embedding width alone moves EER)
* **80 mel bins, log input**, 2.0 s train crops / 3.0 s eval crops
* **identical MUSAN + RIR augmentation**
* **Adam, lr 1e-3, decay 0.95, wd 2e-5**, 60 epochs, eval every 2, patience 8
* **cosine scoring, AS-Norm off during training**

One deviation exists and is recorded in the manifest of the affected runs:
**VGGVox requires `n_mels=40`** — its final `Conv2d(kernel=(4,1))` is sized
against a 40-bin axis and raises otherwise. It is quoted with that caveat.

AS-Norm is deliberately excluded from training-time evaluation. It is an
inference-time score transform; folding it in would confound the backbone
comparison with a calibration effect. It is measured as an explicit on/off
factor in `tools/evaluate.py`, where its contribution is attributable alone.

### Model selection

Epoch selection uses **validation trials only**. The test set is scored exactly
once per experiment, at the checkpoint validation already chose. A test EER that
is the minimum of 30 peeks is not a held-out estimate.

---

## 2. Statistical basis

Absolute EER cannot rank these systems, and the reason is structural. Trials
sharing a speaker are correlated, so with `m` trials per speaker and
intra-speaker correlation `rho`:

```
n_eff = m*S / (1 + (m-1)*rho)   ->   S / rho     as m grows
```

The effective sample size is bounded by the **speaker** count, not the trial
count. With 91 held-out Sinhala speakers and `rho = 0.7`, `n_eff <= 130` and a
single absolute EER carries roughly **±5–6 pp** of uncertainty — far coarser
than the differences between architectures. Two consequences:

1. **Trial count barely matters.** `m ≈ 20` already reaches 98% of the ceiling,
   which is why the test lists are 20,000 pairs rather than 80,000 — a 4×
   saving in evaluation cost for no loss of resolving power.
2. **The comparison must be paired.** Scoring two systems on identical trials
   and writing per-speaker error as `e_A(s) = mu(s) + a(s)`,
   `e_B(s) = mu(s) + b(s)`, the speaker-difficulty term `mu(s)` cancels in the
   difference. `analyze.py` resamples **speakers** with replacement and reports
   the distribution of `EER_A − EER_B`. On synthetic data with known ground
   truth this interval is **2.6× tighter** than the absolute-EER interval, and
   it correctly finds no difference between a system and itself.

Report contrasts from §3 of the generated report, not absolute EER gaps.

Three further diagnostics separate *why* a system wins:

| metric | what it adds over EER |
|---|---|
| `d'` | score-distribution separation in pooled-SD units. EER is a function of `d'` only under Gaussian scores; a divergence exposes non-Gaussian behaviour. |
| `minCllr` | information cost over *all* operating points, after optimal monotonic recalibration (PAV). Better EER but worse minCllr = winning at one threshold, losing overall. |
| `Cllr − minCllr` | calibration loss — precisely what AS-Norm and per-language thresholds can recover without retraining. |

---

## 3. Usage

### Which conda environment

**`SL_SPV`** for anything that touches a GPU. It is the same environment on
every compute node (shared `$HOME`): Python 3.11.15, torch 2.5.1+torchaudio
2.5.1, transformers 4.57.6, plus soundfile/scipy/sklearn/matplotlib/tensorboard.

**You normally do not activate it yourself.** `run_queue.py` dispatches through
`tools/gpurun.sh`, which activates `SL_SPV` on the remote node for you — so the
queue is launched from whatever environment you happen to be in on the dev node:

```bash
# from the dev node, base env is fine — the queue activates SL_SPV remotely
python experiments/tools/run_queue.py --stage A
```

Activate it manually only when running a script **directly on a GPU node**:

```bash
ssh compute-node-3
conda activate SL_SPV
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer
python experiments/scripts/A_ecapa512_aamsoftmax_si_s42.py
```

| tool | environment | why |
|---|---|---|
| `run_queue.py` | either — base is fine | dispatches only; `gpurun.sh` activates `SL_SPV` remotely |
| the generated `scripts/*.py` | **`SL_SPV`**, on a GPU node | training |
| `evaluate.py` | **`SL_SPV`**, on a GPU node | embedding extraction |
| `build_splits.py`, `build_en_splits.py` | either | no GPU, no torch (soundfile only) |
| `gen_scripts.py` | either | CPU model-card introspection |
| `analyze.py`, `layer_weights.py` | either | numpy / CPU checkpoint reads |
| `fetch_ssl_encoder.py` | **base, on the head/dev node** | see below |

> **The one real exception.** `fetch_ssl_encoder.py` converts a `.bin`
> checkpoint to safetensors, and transformers refuses to `torch.load` a `.bin`
> below torch 2.6 (CVE-2025-32434). `SL_SPV` has torch **2.5.1**; the dev node's
> **base** env has torch **2.8.0**. So that conversion must run in `base`, which
> is exactly why the tool exists — it produces the safetensors snapshot that
> `SL_SPV` can then load. Encoders that already publish safetensors
> (mHuBERT-147) are unaffected.

The dev node has **no GPU**, so a generated script run there will train on CPU.
Use `--dry-run` to inspect a command locally, and the queue to actually run it.

```bash
# 1. build the speaker-disjoint splits (once)
python experiments/tools/build_splits.py --all

# 2. generate Stage A scripts
python experiments/tools/gen_scripts.py --stage A

# 3. validate the plumbing cheaply (2 tiny epochs per run)
python experiments/tools/run_queue.py --stage A --scale smoke

# 4. run Stage A for real, across every free GPU
python experiments/tools/run_queue.py --stage A

# 5. score the held-out test set at each validation-selected checkpoint
python experiments/tools/evaluate.py --all --stage A --probes

# 6. aggregate, run the paired tests, write the report
python experiments/tools/analyze.py

# 7. let Stage A's results choose Stage B, and repeat
python experiments/tools/gen_scripts.py --stage B --top 2
python experiments/tools/run_queue.py --stage B
```

Front end, technique and English stages:

```bash
# English splits (once) — needed by Stage E and by Stage G's fine-tune arms
python experiments/tools/build_en_splits.py

# Stage F — what should the network be fed?
python experiments/tools/fetch_ssl_encoder.py --all      # encoders as safetensors
python experiments/tools/gen_scripts.py --stage F
python experiments/tools/run_queue.py    --stage F
python experiments/tools/layer_weights.py --all          # where speaker info lives

# Stage H — does the backbone still matter under SSL features?
python experiments/tools/gen_scripts.py --stage H
python experiments/tools/run_queue.py    --stage H

# Stage E — English reference
python experiments/tools/gen_scripts.py --stage E --conditions en_matched,en_full
python experiments/tools/run_queue.py    --stage E

# Stage G — the repo's universal feature flags, ablated on disjoint splits
python experiments/tools/gen_scripts.py --stage G \
    --initial-model exps/E_ecapa1024_aamsoftmax_en_full_s42/model/model_best.model
python experiments/tools/run_queue.py    --stage G
```

Any single experiment is also a standalone script:

```bash
python experiments/scripts/A_ecapa1024_aamsoftmax_si_s42.py            # real run
python experiments/scripts/A_ecapa1024_aamsoftmax_si_s42.py --scale dev
python experiments/scripts/A_ecapa1024_aamsoftmax_si_s42.py --dry-run  # print cmd
```

`run_queue.py` skips experiments whose `final.json` records `exit_code == 0`, so
killing and restarting it resumes rather than redoes.

---

## 4. What each experiment records

```
experiments/results/<exp_id>/
├── manifest.json    everything known before epoch 1:
│                    resolved config, model card (measured parameter counts per
│                    module, embedding dim, GFLOPs), loss spec (objective in
│                    LaTeX, geometry, pre-registered prediction), architecture
│                    spec, data fingerprints (SHA-256 of every list), full
│                    environment (git commit + dirty flag, torch/CUDA/driver,
│                    GPU model, package versions)
├── epochs.jsonl     ONE ROW PER EPOCH: train loss, train metric, LR, val EER,
│                    val minDCF, threshold, per-language breakdown, AS-Norm
│                    diagnostic, wall-clock and per-epoch duration
├── config.yaml      the exact resolved configuration
├── command.txt      the exact argv — re-runnable verbatim
├── script.py        a copy of the script that produced this run
├── stdout.log       complete trainer output
├── final.json       best epoch, full curves, timings, exit status, events
├── test_eval.json   held-out metrics per evaluation set, cosine and AS-Norm
└── scores/*.npz     per-trial scores + labels + the speaker id behind each
                     side — what the paired bootstrap resamples over
```

`manifest.json` + `command.txt` are sufficient to reproduce a run;
`epochs.jsonl` + `final.json` + `scores/` are sufficient to analyse it without
re-reading any trainer log.

### Pre-registered predictions

`common/lossspec.py` and `common/modelcard.py` carry a `predicted_effect` for
every loss and architecture, written **before** any run. The analysis can then
be scored against a prediction rather than rationalised after the fact. The
central ones:

* **`angleproto` should degrade least in the `combined` condition.** It builds
  centroids from the batch and has no `(nOut × nClasses)` weight matrix, so its
  capacity does not grow with speaker count — unlike AAM-softmax, whose margin
  must be satisfiable by 782 centroids packed onto a 256-dimensional
  hypersphere.
* **RawNet3 is the falsifiable one.** If a learned sinc filterbank beats the mel
  frontend, the trained cut-off frequencies should visibly deviate from mel
  spacing — a directly checkable claim, not a black-box win.
* **The SSL encoder comparison isolates pretraining coverage.** WavLM is
  English-only; XLS-R and mHuBERT-147 both cover Sinhala and Tamil. If the
  multilingual encoders gain more on Tamil than on Sinhala, that is coverage,
  not architecture.
* **`ecapa1024` vs `ecapa512` and `resnetse34v2` vs `resnetse34l`** are capacity
  controls. If both gaps are small, these corpora are data-limited rather than
  capacity-limited, and the cheaper models are the right deployment choice.

---

## 5. Layout

```
experiments/
├── registry.py           single source of truth for every experiment
├── common/
│   ├── lossspec.py       mathematical spec of all 8 losses + predictions
│   ├── modelcard.py      architecture specs + live model introspection
│   ├── recorder.py       the structured-folder writer + environment capture
│   ├── runner.py         subprocess wrapper; parses trainer stdout to JSONL
│   └── harness.py        shared entry point for the generated scripts
├── tools/
│   ├── build_splits.py   speaker-disjoint splits (run this first)
│   ├── gen_scripts.py    registry -> one standalone .py per experiment
│   ├── run_queue.py      dispatch across the cluster's GPUs
│   ├── evaluate.py       held-out scoring, retains per-trial scores
│   └── analyze.py        paired bootstrap, diagnostics, tables, figures
├── scripts/              generated experiment scripts (+ stage manifests)
├── splits/               generated splits (+ per-corpus manifests)
├── results/              one directory per run — see §4
└── analysis/             results_table.csv, comparisons.json, report.md, figures/
```

Two things were **added** outside `experiments/`; nothing existing was modified:

* `models/ECAPA_TDNN_C512.py` — a thin wrapper pinning ECAPA to 512 channels.
  The trainer has no `--channels` flag and discards unknown YAML keys, so the
  capacity control needed its own model module rather than an edit to
  `trainSpeakerNet.py`.
* `models/ECAPA_TDNN_MFCC.py`, `models/ECAPA_TDNN_MFCC40D.py` — cepstral front
  ends for Stage F. Separate modules because `n_mfcc` and `mfcc_deltas` are not
  argparse options either.
* `models/SSLFrontendSpeakerLW.py` — SSL front end with learned layer weights,
  porting the design `SL_SPV/voiceid` actually deploys into the trainer so the
  two stacks can finally be compared.
* `models/SSL_ECAPA.py` — SSL encoder feeding the full ECAPA-TDNN backbone
  (Stage H). Supplies the front-end × backbone interaction cell that Stage A and
  Stage F each miss on their own.
* `models/weights/wavlm-base-plus/`, `models/weights/mhubert-147/` — self-contained safetensors snapshots.
  The `SL_SPV` env runs torch 2.5.1 and transformers refuses to `torch.load` a
  `.bin` below torch 2.6 (CVE-2025-32434), while the cached hub revision for
  that id carries only `.bin`. Upgrading torch would have put these runs on a
  different numerical stack from every previously published result. See that
  directory's `README.md`.

### Validation status

All Stage A architectures and all Stage F front ends were smoke-tested
end-to-end on the cluster (2 epochs, real data, real GPUs); every one completes
with `exit_code 0`:

| architecture | params | GFLOPs/2s | s/epoch (smoke) |
|---|---|---|---|
| `resnetse34l` | 1.40M | 1.79 | 22.7 |
| `vggvox` | 3.64M | 1.06 | 12.9 |
| `ecapa512` | 5.99M | 1.93 | 29.2 |
| `resnetse34v2` | 7.37M | 9.25 | 33.2 |
| `mlpmixer` | 7.71M | 3.03 | 25.7 |
| `ecapa1024` | 14.46M | 5.17 | 29.2 |
| `ssl_wavlm` | 94.98M | 22.10 | 101.2 |

| Stage F front end | params | GFLOPs/2s | s/epoch (smoke) |
|---|---|---|---|
| `mfcc80` | 14.46M | 5.17 | ~29 |
| `mfcc40d` | 14.67M | 5.25 | ~29 |
| `ssl_wavlm_mid` | 94.98M | 22.10 | ~110 |
| `ssl_wavlm_lw` | 94.98M | 22.10 | ~105 |
| `ssl_mhubert_lw` | 94.97M | 22.10 | ~105 |

The smoke EERs are meaningless (2 epochs on 4 utterances per speaker, all near
chance) — the check is that every backbone builds, trains, evaluates and
records. The statistics in `analyze.py` were separately validated against
synthetic data with known ground truth: EER matches the repo's own
`tuneThreshold` implementation exactly, `minCllr <= Cllr` holds, the paired
bootstrap recovers a known injected difference and correctly reports no
difference between a system and itself.

---

## 6. Known limitations

* **Power is bounded by held-out speaker count** (91 si / 131 ta). Absolute EERs
  are coarse; only the paired contrasts are well-powered. Escalation path if a
  contrast matters and lands inconclusive: k-fold over the speaker partition,
  which reuses all 478/638 speakers as test across folds at k× the compute.
* **`n_utts == n_sessions` in both anchor corpora** — every utterance is its own
  recording, so there is no within-speaker session structure and cross-session
  trials cannot be distinguished from cross-utterance ones. Combined with the
  QC audit's finding that `eta^2(SNR | speaker)` is 0.68–0.74 on these corpora,
  channel is partly identity here. This is why the held-out *corpus* probes
  carry the generalisation claim rather than the in-corpus test EER.
* **Read speech only** in the anchor pair. The in-the-wild check is
  `slceleb2026_sinhala`, and it has no Tamil counterpart, so "in-the-wild Tamil"
  remains unmeasured.
* **`--scale smoke` and `--scale dev` runs are excluded** from selection and
  from the analysis tables by default, and write to their own `save_path` so
  they cannot be resumed into a real run.
