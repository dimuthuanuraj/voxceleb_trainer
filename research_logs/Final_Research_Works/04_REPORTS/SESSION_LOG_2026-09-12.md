---
title: "Final Research Works --- Session Log"
subtitle: "2026-09-12: plan construction, Wave 0, and the first Wave A/C results"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# What this session did

Built the completion plan for SL_SPV from the annual report of 2026-09-10 plus a
fresh filesystem audit, then executed Wave 0 and the parts of Waves A and C that
the surviving hardware allows.

Everything below is **[M]** --- measured, with an artefact under `03_RESULTS/`.

---

# 1. The audit, and the three things it found

**Nothing had run since 24 August.** Nineteen days idle. Not a research decision:
`transformer_sv/dispatch_state.json` still read `"phase": "running"` with five
jobs marked `training`, and no matching process existed on any node. The
dispatcher died with the cluster and nobody restarted it.

**Ten trained or part-trained runs had checkpoints and no score.** Several
GPU-days already spent, sitting unharvested.

**Three of five compute nodes cannot run CUDA.** Two have an NVIDIA
kernel-module / userspace version skew (NVRM 580.173.02 against
`libnvidia-ml.so.580.178.04`); one does not route. Usable capacity is 2 x T4
(15 GB) --- and the highest-priority experiment needs 30 GB, so it cannot be
placed at all.

---

# 2. A correction to the obvious plan

The tempting move on the five interrupted T-series arms was to declare them
"done enough" and score their best checkpoints. **Measured, that is not safe:**

| Arm | Epochs | Best @ | Since best | Converged? |
|---|---:|---:|---:|---|
| `MFAConformer_full_ta_s42` | 30 | 30 | **0** | no --- still improving |
| `MFAConformer_no_attention_si_s42` | 27 | 19 | 8 | no --- patience 16 not reached |
| `MFAConformer_no_attention_ta_s42` | 7 | 7 | **0** | no --- barely started |
| `Res2Former_tf0_si_s42` | 33 | 32 | 1 | no --- still improving |
| `Res2Former_tf4_si_s42` | 32 | 21 | 11 | no --- patience not reached |

Three of five had their *best* epoch as their *last* epoch when the cluster took
them down. Scoring them where they stand would put artificially pessimistic
numbers in the same table as `MFAConformer_full_si_s42`, which did early-stop
properly at 31 --- exactly the "good number measured on something other than what
it claims" failure the programme exists to avoid.

They must be **resumed**, and resuming is free: `trainSpeakerNet.py:474-486`
globs `model0*.model`, loads the highest, and continues from its index + 1.

---

# 3. Wave 0 --- completed

| Task | Result |
|---|---|
| **W0.1** Preflight | 2 usable GPUs, 3 dead nodes diagnosed, 10 unscored trained runs enumerated |
| **W0.2** `test_normalize` | Delegation added to both A1/A8 wrapper losses. **4 blocked runs unblocked** |
| **W0.3** fp32 mel guard | Applied and **verified on GPU** |
| **W0.4** Driver ticket | Generated with live per-node evidence and the exact root fix |
| **W0.5** Arbiter registry | Two front-ends registered, verified a **single-factor** contrast |
| **W0.6** Scorer `sys.path` | **New defect found and fixed** (see section 4) |

## W0.2 --- why four runs had zero checkpoints

A1 and A8 each wrap the trainer's AAM-Softmax in `self.speaker`. The eval path
reads `self.__model__.module.__L__.test_normalize` (`SpeakerNet.py:694`); the
wrapper never exposed it. All four runs died at their **first validation**, which
is why they left no checkpoints --- the failure was at the start, not the end.
One delegated attribute per file.

## W0.3 --- and a tighter number than expected

The patch computes the mel front end in fp32 with autocast disabled. Verified on
a T4 at |x| = 8.0 (above the 5.66 observed over 30,000 real augmented samples):

* **0 non-finite values** under autocast;
* **`max |amp - fp32| = 0.0`** --- bit-identical, so no published number moves;
* power-spectrogram peak **46,286** against fp16's 65,504 ceiling ---
  **1.42x headroom**, against the 2.1x recorded at the lower input level.

The overflow was never hypothetical. Any model that builds its own mel front end
*inside the model*, rather than taking the trainer's, needs this guard --- a rule,
not an incident.

## W0.5 --- the arbiter is a registry entry, not an implementation

`SSLFrontendSpeakerLW` already accepts `ssl_freeze=False`, and the trainer
already accepts `--no_ssl_freeze --llrd`. Verified the generated argv carries all
three, and that the new entry differs from `ssl_wavlm_ft` on **`model` alone** ---
so the layer-weighting contrast is genuinely single-factor. It needs 30,000 MiB
and stays blocked on W0.4.

---

# 4. Two defects found while building the plan

## D-1 --- the scorer could not score most proposals

`proposals/evaluate_proposal.py` never put `proposals/_trainer_shim/` on
`sys.path`, though training does (`common/train.py:98`). Any proposal supplying
its own `MainModel` **trained fine and could not be scored**:

```
ModuleNotFoundError: No module named 'models.PLLW'
```

Affected A7, A4, A5 and A1. **Unaffected: A9** --- which supplies a *loss*, and
is exactly the one proposal that had been scored. This single missing path is
why three fully-trained runs sat unscored for three weeks.

## D-2 --- the two shims shadow each other, and the first fix caused it

Found *by* fixing D-1. `proposals/` and `transformer_sv/` each ship a
`_trainer_shim/models/`, and both contribute to the same `models` **namespace
package**, whose `__path__` is fixed at first import. `evaluate_transformer.py`
imports `evaluate_proposal` as a library --- the right design, and why T-series
numbers sit on the same code path as A-series ones --- so an **import-time**
`sys.path` insert in the proposals scorer silently hijacks `models` for the
T-series.

| `evaluate_proposal.py` | `models.__path__` | `models.MFAConformer` |
|---|---|---|
| import-time insert (first attempt) | `proposals/_trainer_shim/models` | **ModuleNotFoundError** |
| entry-point-only insert (shipped) | `transformer_sv/_trainer_shim/models` | OK |

The first W0.6 broke T6. The fix: the path change belongs to the **entry point**,
not the import. Run directly, `main()` sets it up; imported as a library, it
touches nothing. Both strands are now verified in **isolated subprocesses** ---
testing them in one interpreter would measure only whichever imported first,
which is the bug itself.

> Finding III-9 recurring one level down. The failure here was *loud* --- an
> ImportError, not a plausible number --- which is why it cost an hour rather
> than a paper.

---

# 5. Results produced

## C5 --- the PEFT publication gate: **CLEARED** **[M]**

`MASTER_SYNTHESIS` section 6 risk 3 warned that if the PEFT and full-FT arms
differ in two-stage-ness, the negative PEFT result is not safe to publish.
Scanning the literal argv of **74 runs**:

* **0 runs warm-started** --- no `--initial_model`, no `--finetune`, in either arm;
* so the staging confound **does not apply**, and prediction 9's *Falsified*
  verdict stands on that axis.

**But** the arms differ in learning rate --- fine-tuned 1e-4, frozen 1e-3. That
is a *declared* registry deviation (1e-3 destroys a pretrained transformer), yet
it is a second uncontrolled factor and **must travel with the PEFT contrast
wherever it is reported.** Prediction P27 is therefore scored **met, with a
caveat it did not anticipate.**

## C4 --- the score-shift figure **[M]**

Per-language score midpoints on matched held-out test sets:

| | midpoint |
|---|---:|
| Sinhala | +0.2809 |
| Tamil | +0.4084 |
| **shift** | **+0.1275** |

**A methodological note that changed the number.** The first run tagged each
result by the language the model was *trained* on, so Sinhala models evaluated on
Tamil probes counted as Sinhala. That gave **+0.0346**. Tagging by the language
of the *trials* --- and restricting to matched in-language held-out sets ---
gives **+0.1275**, 3.7x larger. Cross-lingual probes measure a different
quantity and must not be averaged into a per-language midpoint.

## A1 --- A7 PLLW scored: annual report open item #3 **[M]**

The run had been trained since 24 August --- 53 checkpoints, early-stopped,
never scored. Twenty-eight GPU-minutes recovered it.

| System | train | speakers | si test EER (cos) | AS-Norm | minDCF (AS-Norm) |
|---|---|---:|---:|---:|---:|
| `F_ssl_wavlm_lw_..._si_s42` (reference) | si | 336 | 3.297 | 2.964 | 0.1654 |
| **`A7_pllw_combined_s42`** | combined | 782 | **3.024** | **2.631** | **0.1565** |
| difference | | | **-0.272 pp** | **-0.333 pp** | -0.009 |

Same **19,838 trials**, same order, so the pairing is exact and the
speaker-clustered bootstrap is valid on it.

### And the number is not yet interpretable

The two runs differ in **two** things, not one:

* A7 is trained on `combined_si_ta` --- **782 speakers**;
* its reference baseline is trained on `si` alone --- **336 speakers**.

So the -0.272 pp confounds the **method** (per-language layer weighting) with
**2.3x more training data**, and more speakers alone would be expected to help.
The proposal harness's own rule is that the proposal must be the single changed
factor; here it is not, and it could not have been --- per-language weighting is
undefined on a single-language condition, so A7 *has* to train on `combined`.

**The control that would resolve it does not exist.** Checking
`experiments/analysis/v1-final/results_table.csv`: **zero rows with
`condition == "combined"`.** Nothing in the v1 benchmark was ever trained on the
pooled condition, so there is no shared-layer-weight run to compare A7 against.

Added as task **A6**: `ssl_wavlm_lw` (one shared 13-vector) trained on
`combined_si_ta`, scored on the same si trials. That splits the confounded pair
into two single-factor contrasts:

```
A7 per-language  vs  A6 shared      (both combined)   <- the METHOD
A6 shared        vs  v1 shared      (combined vs si)  <- the DATA
```

SSL encoder frozen, so ~8 GPU-h on a T4 --- runnable today.

> **What is safe to say now, and only this:** A7 PLLW reaches 3.024 % cosine /
> 2.631 % AS-Norm on the held-out Sinhala test set, on the same trials as every
> v1 system. Whether per-language layer weighting caused any of it is
> **unresolved**.

This is the same shape as Finding III-7 one step along: there, methods were
undefined on the data and still emitted plausible numbers; here the method is
well defined, and it is the *comparison* that is undefined. Scoring the run was
necessary and not sufficient.

## G2-G6 --- the errata register **[M]**

`04_REPORTS/ERRATA.md`, plus a machine-readable `forbidden_claims.json` for a
pre-commit or CI check. Scanning the project's documents for the six claims
section 10.3 requires corrected or withdrawn:

| Item | Claim | Occurrences |
|---|---|---:|
| E2 | nested learning "9.84 % EER, validated" | 2 |
| **E3** | **Phase I 10.32 % headline** | **47** |
| E4 | ResNetSE34L 34-layer / 6.8 M spec | 2 |
| E5 | "Tamil pool 50 -> 752" | 4 |

**The 10.32 % figure appears in 47 places.** It is the most widely propagated
number in the repository and it is single-seed, unreplicated, predates
`--deterministic`, and is contradicted by another log in the same record
(14.62 %). G1 must decide: three seeds, or formal retirement. There is no third
option, and the register makes leaving it alone visible rather than silent.

---

# 6. Pre-registered predictions

Seven registered in `02_STATE/predictions.json` **before** their runs, continuing
the annual report's practice. One scored so far (P27, above). The rest are
pending their tasks and will be scored **whichever way they come out** --- the
annual report's ~40 % falsification rate is the standard being held to, and two
of its falsifications became the programme's strongest results.

---

# 7. Where it stands

| | |
|---|---|
| Tasks defined | **37**, plus 4 tracked non-compute items |
| Passed this session | **9** (W0.1, W0.2, W0.3, W0.5, W0.6, **A1**, C4, C5, G2-G6) |
| Running at session end | A3 (T6 span probe) |
| Ready now on 2 x T4 | **A6** (makes A1 interpretable), A2, A4, C1, C2, C3, C6, D1, D2, D3, D4, F3 |
| Blocked on the driver ticket | B1, B2, B4, B5, F1, G1 --- ~760 of 1,100 GPU-hours |

**The single highest-value action available is not a research task.** It is
sending `03_RESULTS/W0.4/DRIVER_TICKET.md` to the cluster administrator.
