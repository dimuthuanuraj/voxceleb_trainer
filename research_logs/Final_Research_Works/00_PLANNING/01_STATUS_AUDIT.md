---
title: "Final Research Works — Status Audit"
subtitle: "Measured state of every strand, 2026-09-12"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# 1. Purpose

The annual report (`2026-09-10-annual-research-progress-report-may2025-aug2026.md`)
states what has been **achieved**. This document states what is **on disk right
now**, measured on 2026-09-12 by reading the filesystem rather than the prose.
It is the input to [`02_PENDING_WORK_REGISTER.md`](02_PENDING_WORK_REGISTER.md).

Method: every row below was produced by listing checkpoints, reading
`*.train.json` status fields, running each strand's own `scaffold.py --check`,
and probing the cluster over SSH. Nothing here is quoted from a previous
document.

**Evidence tags** follow the annual report: **[M]** measured from an artefact
read today, **[R]** reported in a project document, **[P]** planned.

---

# 2. The headline finding of this audit

> **Nothing has been running since 24 August 2026.** Nineteen days of wall-clock
> have passed with zero GPU work, and the cause is not a research decision —
> it is that the dispatcher died with the cluster and nobody restarted it.

Two independent consequences, both measured:

1. **Seven training runs are sitting at 45–100 % completion with checkpoints on
   disk and no score.** Several GPU-days have already been spent on them. **[M]**
2. **Three of five compute nodes cannot run CUDA today.** Two have an NVIDIA
   kernel-module / userspace version mismatch; one does not route. **[M]**

Both are recoverable, and the first is recoverable without a GPU-day of new
training. That is the reason Wave 1 of the execution plan is what it is.

---

# 3. Cluster — probed 2026-09-12 13:25 UTC **[M]**

| Node | Address | GPUs | State today | Detail |
|---|---|---|---|---|
| `compute-node-1` | .119 | 2 × Tesla T4 15 GB | **USABLE, both idle** | 0 MiB used on both cards |
| `compute-node-2` | .118 | 2 × Tesla T4 15 GB | **DEAD — driver mismatch** | kernel module `580.173.02`, userspace `libnvidia-ml 580.178.04` |
| `compute-node-3` | .120 | 2 × NVIDIA A10 23 GB | **DEAD — no route to host** | unchanged since 2026-08-15 |
| `compute-node-4` | .121 | 1 × NVIDIA A40 46 GB | **DEAD — driver mismatch** | same NVRM/userspace skew as node-2 |
| `compute-node-5` | .125 | none | storage/dev | 14.1 TiB free |
| `head-node` | — | none | control | this host |

**Diagnosis, stated precisely.** Both failing nodes report
`Failed to initialize NVML: Driver/library version mismatch`. `/proc/driver/nvidia/version`
gives NVRM **580.173.02** while `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.178.04`
is what userspace loads. Both nodes carry `nvidia-driver-580 580.178.04` in
`dpkg`. So the package was upgraded underneath a *running* kernel module and the
module was never reloaded. `sudo -n` fails on both, so this needs the cluster
administrator.

**The fix is a reboot** (or `rmmod nvidia* && modprobe nvidia` as root). Node-2
has been up 1 d 19 h, node-4 up 1 d 14 h — so both were rebooted *after* the
package upgrade and still mismatch, which means the upgrade landed in between.
A clean reboot now should resolve it; if it does not, the 535 packages still
installed on node-2 are the likely confusion and should be purged.

> **Capacity consequence.** Usable VRAM today is **2 × 15 GB**. The registry
> marks `ssl_wavlm_ft` at `min_gpu_mb = 30000`. **The arbiter run — the report's
> #1 open item — cannot be placed on any currently-live GPU.** It needs node-4's
> A40. This is the single highest-value blocker in the programme and it is an
> infrastructure ticket, not a research task.

---

# 4. `transformer_sv` — T1–T6 **[M]**

`scaffold.py --check` on 2026-09-12: *2/6 have at least one result from a real
run; 1 scored arm in total.*

## 4.1 What the dispatcher was doing when it died

`dispatch_state.json` still reads `"phase": "running"`, last updated
**2026-08-24T10:34:01Z**. Of 14 planned jobs: 1 done, 5 `training`, 8 `queued`.
No process matching the dispatcher or `trainSpeakerNet` exists on any node today.
The state file is **stale, not current** — it must not be read as a live status.

## 4.2 Per-arm measured state

Epoch counts from `out/exps/<tag>/model/model0*.model`; validation EER from the
matching `.eer` files. `max_epoch` is 60 and `test_interval` is 1 for every arm.

| Arm | Epochs done | Best val EER | Epochs since best | Converged? | Scored? |
|---|---:|---:|---:|---|---|
| `MFAConformer_full_si_s42` | 31 | 4.0416 @28 | 3 | **yes — early-stopped** | **yes** |
| `MFAConformer_full_ta_s42` | 30 | 1.3965 @30 | **0** | **no — still improving** | no |
| `MFAConformer_no_attention_si_s42` | 27 | 4.0816 @19 | 8 | no — patience 16 not reached | no |
| `MFAConformer_no_attention_ta_s42` | 7 | 2.3882 @7 | 0 | **no — barely started** | no |
| `Res2Former_tf0_si_s42` | 33 | 4.3417 @32 | 1 | **no — still improving** | no |
| `Res2Former_tf4_si_s42` | 32 | 5.2021 @21 | 11 | no — patience not reached | no |

> **A correction to the obvious plan.** The tempting move is to declare the five
> interrupted arms "done enough" and score their best checkpoints. **That is not
> safe.** `full_ta` and `tf0_si` were *still improving* when they died — their
> best epoch is their last epoch — and `no_attention_ta` had done 7 of 60.
> Scoring them now would put an artificially pessimistic number in the same table
> as `full_si`, which *did* early-stop. They must be **resumed to their own
> stopping criterion first.**

**Resuming is free.** `trainSpeakerNet.py:474-486` globs `model0*.model` in the
save path and loads the highest, setting the start epoch to its index + 1. So
re-issuing the identical `run.py` command continues from epoch 31/28/8/34/33
rather than restarting. **[M]** No new code is needed for this.

## 4.3 T5 and T6

* **T5 — complete.** The training-free CPU cost result is in
  `T5_efficient_attn/out/cost_cpu.json`. It is the strand's one reportable
  result and it already removed four planned runs from the plan.
* **T6 — not run, and it is the cheapest informative thing in the whole
  programme.** `out/self_check.json` exists (machinery verified); no probe has
  been run against a trained model. It needs **no training at all** — it caps
  attention span on an already-trained checkpoint and re-scores.
  `MFAConformer_full_si_s42` has been scored and is available **today**, on a T4.

## 4.4 Verdict

`transformer_sv` is **≈ 20 % complete**: 1 of 6 experiments reportable, 1 of 14
dispatched arms scored. It is not blocked by anything except operator time.

---

# 5. `proposals` — A1–A9 **[M]**

`scaffold.py --check`: *7/9 have results from a real run.* That figure overstates
readiness — "results" counts any output file, including a training log from a run
that crashed.

| Proposal | Runs | Measured state | What it needs |
|---|---|---|---|
| **A1** DA²-LoRA | 2 (`anchored`, `control`) | **FAILED, 0 checkpoints** | one-line defect fix, then full retrain |
| **A2** CC-NAP | 2 | projector + scores present | consolidate into a result |
| **A3** SL-AMEC | 46 | **complete, reported** | — |
| **A4** LS-CAM | `si` **trained, unscored**; `ta` FAILED | 61 ckpts, best 4.3417 | **score si**; fp32 fix then retrain ta |
| **A5** DK-CAM++ | `si` **trained, unscored**; `ta` FAILED | 61 ckpts, best 4.0416 | **score si**; fp32 fix then retrain ta |
| **A6** CA-SSPS | 2 pooled arms trained | cluster JSONs present, reported | — |
| **A7** PLLW | `combined` **trained, unscored**; `si` FAILED | 54 ckpts, best 1.9217, early-stopped | **score it — this is report open item #3** |
| **A8** CD-NDAL | 2 | **FAILED, 0 checkpoints** | same one-line fix as A1, then retrain |
| **A9** ARI-SubCenter | 2 | **complete, reported** (si p=0.038) | seed replication |

## 5.1 Two defects, four-plus runs, both one-line

**Defect 1 — `test_normalize` missing on wrapper losses.** A1 and A8 both
supply a custom `LossFunction` that *wraps* the trainer's AAM-Softmax in
`self.speaker`. `SpeakerNet.py:694` reads `self.__model__.module.__L__.test_normalize`
on the eval path. The wrapper never exposes it, so all four runs die with:

```
AttributeError: 'LossFunction' object has no attribute 'test_normalize'
```

They die at their **first validation**, which is why both have **0 checkpoints**
— the failure is not at the end of training, it is at the start, and the whole
run is lost. Files:
`_trainer_shim/loss/da2_lora_adv.py:40` and `_trainer_shim/loss/cd_ndal_adv.py:49`.
Fix: delegate the attribute to the wrapped AAM (which sets it `True` at
`loss/aamsoftmax.py:15`).

**Defect 2 — fp16 mel overflow in the CAM++ family.** `A5_dk_campp/DK_CAMPP.py:298`
computes `self.torchfb(x)` inside `features()` with no autocast guard. Under
`--mixedprec` the power spectrogram overflows fp16 (measured peak 30,592 against
a 65,504 ceiling — 2.1× headroom), one `inf` reaches `InstanceNorm1d` and then
BatchNorm's running buffers, which are written in the forward pass where
`GradScaler` does not guard. A4 inherits it: `ls_cam.py:223` builds a `DK_CAMPP`
trunk. This is the defect the annual report §7.2 diagnosed from the T-series and
traced back here. Fix: compute the front end in fp32 with autocast disabled —
the same remedy `transformer_sv/common/attn.py:229` already applies to `q @ kᵀ`,
and the identity in fp32, so no existing number changes.

## 5.2 Verdict

`proposals` is **≈ 55 % complete**. Three runs are trained and unscored
(A4 si, A5 si, A7 combined) — **GPU-minutes from being results**. Four more are
blocked behind two one-line fixes.

---

# 6. `feature_level_testing` — **COMPLETE** **[M]**

All four phases ran to completion (`pipeline/out/phase{1,2,3,4}/` all populated),
all eight systems scored on both languages, `RESULTS.md` is written, and the
findings are already carried in annual report §7.3. `selftest.py` passes 24
checks including a synthetic vocal-tract ground truth.

**Nothing is pending inside this folder.** It names exactly one follow-up, in
its own §5.5, and that follow-up belongs to the programme rather than to the
folder:

> *"Tamil's apparent advantage is confounded with recording quality and should
> not be reported as a language effect without a within-corpus bilingual
> control (`nisp_tamil` has 65 bilingual speakers and is the obvious next
> step)."*

`experiments/splits/nisp_tamil/` exists. **[M]** This is a real, cheap, unrun
experiment and it is carried into the register as **C3**.

One documentation defect, inherited and worth fixing while here: the report
build (`report/build.sh`) depends on a `xetex -progname=xelatex` workaround
because no `xelatex` binary exists and `lualatex` lacks `luaotfload`. That is
recorded in the folder's README and needs no action unless the toolchain moves.

---

# 7. `My_refered_papers` — **COMPLETE as a study, and it is the source of the plan** **[M]**

Not an experiment folder. 32 papers in 9 categories, each with a deep-study
document, plus `_deep_study/PROPOSED_ARCHITECTURES.md` (A1–A9, which is where
the `proposals/` tree came from) and `_deep_study/00_MASTER_SYNTHESIS.md`.

**Nothing is pending in the literature review itself.** What *is* pending is the
part of it the project has not executed: `00_MASTER_SYNTHESIS.md §5` lays out a
**four-paper publication roadmap** and a section of "also worth doing" items
that are cheap, unrun, and unclaimed anywhere in the literature:

| Source | Item | Training needed? |
|---|---|---|
| §5 "Also worth doing" | The **privacy/enrolment table** — N = 1, 5, 10, 15 enrolment utterances | none |
| §5 "Also worth doing" | The **short-duration table** — truncate test utterances to 2 / 3 / 5 s | none |
| §5 "Also worth doing" | The **dialect gap** — train Indian Tamil, evaluate Sri Lankan Tamil | none (existing ckpts) |
| §5 P1 | Reproduce **Thienpondt Figure 1** (score-shift histogram) for si/ta | none — scores on disk |
| §6 risk 3 | Check the **two-stage fine-tuning confound** before publishing the PEFT negative | none — read argv |

All five are carried into the register as Wave C. Four of the five require **no
GPU training at all**, which makes them the correct work to do while the cluster
is degraded.

The synthesis also maps each A-proposal to a target paper (P1 = A2+A3,
P2 = A7+A8+A9, P3 = A1+A4, P4 = A5+A6). That mapping is reproduced in
[`05_PUBLICATION_MAP.md`](05_PUBLICATION_MAP.md) and is what makes the ordering
of the execution plan a publication schedule rather than a task list.

---

# 8. `experiments` (benchmark v1) — complete for v1, one gap **[M]**

84 entries under `experiments/results/`, 54 experiments with the superseded runs
set aside, analysis in `analysis/v1-final/`. Stages A, E, F, H are all present
and scored.

**The gap is a cell, not a stage.** The registry defines `ssl_wavlm_lw`
(layer-weighted, frozen) and `ssl_wavlm_ft` (last-layer, fine-tuned) but **no
entry that is both**. That missing diagonal is exactly the arbiter run, and the
code to run it already exists: `SSLFrontendSpeakerLW` accepts `ssl_freeze=False`
and the trainer accepts `--no_ssl_freeze --llrd`. **[M]** So the arbiter needs a
registry entry and a GPU, and no new model code.

---

# 9. Strand completeness, consolidated

| Strand | Complete | Pending compute | Blocked by |
|---|---:|---|---|
| `feature_level_testing` | **100 %** | one named follow-up (C3) | nothing |
| `My_refered_papers` | **100 %** as a study | five unrun cheap analyses | nothing |
| `experiments` (v1) | ~95 % | the arbiter cell | **A40 driver** |
| `proposals` A1–A9 | ~55 % | 3 scores, 4 retrains | two one-line fixes |
| `transformer_sv` T1–T6 | ~20 % | 5 resumes, 8 launches, 1 probe | operator time |
| Seed replication | **0 %** | every top system | GPU capacity |
| Phase II V4/V5 | **0 %** | two architectures | nothing |
| Corrections (§10.3) | 0 % | R7 re-run or retire | nothing |

---

# 9b. Defects found *by this audit*, not inherited from the report

Three defects were discovered while building and running the completion plan.
All three were latent — none is a regression introduced by recent work — and
each blocked results that were otherwise ready.

## D-1 — `evaluate_proposal.py` cannot score any proposal that supplies a model **[M]**

`proposals/evaluate_proposal.py` set up `sys.path` with `proposals/`,
`experiments/tools/` and `voxceleb_trainer/` — but **never
`proposals/_trainer_shim/`**. Training does: `common/train.py:98` builds
`PYTHONPATH = [SHIM, PATCHES, TRAINER]` for the child process. So a proposal
supplying its own `MainModel` **trains fine and cannot be scored**:

```
ModuleNotFoundError: No module named 'models.PLLW'
```

Affected: **A7 (PLLW), A4 (LSCAM), A5 (DKCAMPP), A1 (DA²-LoRA)**.
Unaffected: **A9**, which supplies a *loss* and uses a standard model — and A9
is exactly the one proposal that had been scored. **This single missing path is
why three fully-trained runs sat unscored for three weeks.**

Fixed as task **W0.6**.

## D-2 — the two shims shadow each other through a namespace package **[M]**

Found while fixing D-1, and it is the more interesting defect. `proposals/` and
`transformer_sv/` each ship a `_trainer_shim/models/` directory, and both
contribute to the **same `models` namespace package**. A namespace package's
`__path__` is fixed at first import from whatever `sys.path` held at that
moment.

`transformer_sv/evaluate_transformer.py` imports `evaluate_proposal` **as a
library**, to reuse its scorer rather than copy it — which is the right design
and is why T-series numbers sit on the same code path as A-series numbers. But
it means that an *import-time* `sys.path` insert in `evaluate_proposal.py`
silently hijacks `models` for the T-series:

| `evaluate_proposal.py` | `models.__path__` resolves to | `models.MFAConformer` |
|---|---|---|
| import-time insert | `proposals/_trainer_shim/models` | **ModuleNotFoundError** |
| entry-point-only insert | `transformer_sv/_trainer_shim/models` | OK |

The first attempt at W0.6 used an import-time insert and **broke T6**. The fix
is that the path change belongs to the **entry point**, not the import: run
directly, `evaluate_proposal.py` sets it up in `main()`; imported as a library,
it touches nothing and the importer's own shim keeps priority. Both strands are
now verified in **isolated subprocesses**, because testing them in one
interpreter would measure only whichever imported first — which is the bug.

> **This is Finding III-9 recurring one level down.** Every Phase III device
> exists to convert a silent wrong number into a loud failure. Here the failure
> *was* loud — an ImportError, not a plausible number — which is why it cost an
> hour rather than a paper.

## D-3 — the fp16 mel headroom is tighter than previously measured **[M]**

W0.3's GPU verification drove the patched front end at |x| = 8.0 (above the 5.66
observed over 30,000 real augmented samples) and measured a power-spectrogram
peak of **46,286 against fp16's 65,504 ceiling — 1.42× headroom**, against the
2.1× recorded in the annual report at the lower input level. The fp32 path is
confirmed bit-identical (`max |amp − fp32| = 0.0`), so the fix changes no
published number.

**Consequence:** the overflow was never hypothetical, and any future model that
builds its own mel front end inside the model — rather than taking the
trainer's — needs the same fp32 guard. That is a rule, not an incident.

---

# 10. What this audit changes about the plan

1. **Wave 1 is scoring, not training.** Three proposal runs and one training-free
   probe are hours from being results, on the hardware that works today.
2. **The five interrupted T-arms must be resumed, not scored where they stand.**
   Measured: three of five were still improving when they died.
3. **The arbiter run is blocked on an infrastructure ticket**, not on research.
   Raising that ticket is therefore a Wave 0 task with a named owner.
4. **While the cluster is degraded, do the zero-training work.** Wave C is five
   analyses that need no GPU and three of them are unpublished for any language.
5. **Two one-line fixes unblock four runs.** They cost minutes and they are the
   highest leverage lines of code in the programme.
