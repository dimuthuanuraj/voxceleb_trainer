# Final Research Works

**The completion plan for SL_SPV.** Everything still outstanding as of
2026-09-12, with a script for each item and a runner that executes them one at a
time.

Input: [the annual research progress report, 2026-09-10](../2026-09-10-annual-research-progress-report-may2025-aug2026.md),
plus a fresh audit of what is actually on disk in `feature_level_testing/`,
`transformer_sv/`, `My_refered_papers/`, `proposals/` and `experiments/`.

---

## Start here

> ### -> [`00_PLANNING/06_RUNBOOK.md`](00_PLANNING/06_RUNBOOK.md)
> **Every command you need to run, in order, with what it does, how long it
> takes, and how you know it worked.** Start there.

```bash
cd 01_SCRIPTS
./preflight.sh                 # cluster, env, splits, disk  (run after every reboot)
python3 runner.py --status     # what has passed, what is ready, what is blocked
python3 runner.py --next       # name the next ready task
python3 runner.py --run W0.2   # run one task
```

The runner executes **one task at a time** and refuses any task whose
dependencies have not passed their **exit test** — and the exit test is
evaluated from the filesystem, not from a status flag a crashed process was
supposed to write. That is the same discipline each strand's `scaffold.py
--check` uses, and it is why the answer to *"what happened while I was away"* is
always one command.

---

## Layout

```
Final_Research_Works/
├── README.md                  ← you are here
├── 00_PLANNING/
│   ├── 01_STATUS_AUDIT.md     what is actually on disk, measured 2026-09-12
│   ├── 02_PENDING_WORK_REGISTER.md   every task: source, cost, dependency, exit test
│   ├── 03_EXECUTION_PLAN.md   the order of work, and why it is that order
│   ├── 04_BLOCKERS_AND_RISKS.md      what is stopping what, with the fix
│   ├── 05_PUBLICATION_MAP.md  which task feeds which paper
│   └── 06_RUNBOOK.md          <- EVERY COMMAND TO RUN, IN ORDER
├── 01_SCRIPTS/
│   ├── preflight.sh           W0.1
│   ├── runner.py              the one-at-a-time driver
│   ├── tasks.json             the task graph (machine-readable)
│   ├── lib/common.sh          shared paths, guards, cluster probing
│   └── tasks/                 one script per task
├── 02_STATE/                  ledger.json, preflight.json, predictions.json
├── 03_RESULTS/                one directory per task, each with result.json
└── 04_REPORTS/                ERRATA.md, final_research_report.md
```

---

## What the audit found

Three things, all measured rather than inferred:

1. **Nothing has run since 24 August.** Nineteen days idle. Not a research
   decision — the dispatcher died with the cluster and nobody restarted it.
2. **Ten trained or part-trained runs have checkpoints and no score.** Several
   GPU-days are already spent on them. Wave A recovers those results without a
   GPU-day of new training.
3. **Three of five compute nodes cannot run CUDA.** Two have an NVIDIA
   kernel-module / userspace version skew; one does not route. Usable capacity
   today is 2 × T4 (15 GB), and **the highest-priority experiment needs 30 GB**,
   so it cannot be placed at all.

Full detail: [`00_PLANNING/01_STATUS_AUDIT.md`](00_PLANNING/01_STATUS_AUDIT.md).

---

## Are the three named directories complete?

| Directory | Verdict |
|---|---|
| [`feature_level_testing/`](../../../feature_level_testing) | **COMPLETE.** All four phases ran, all eight systems scored on both languages, `RESULTS.md` written, findings already in the annual report §7.3. It names exactly one follow-up — the `nisp_tamil` bilingual control — which is carried here as task **C3**. |
| [`transformer_sv/`](../../../transformer_sv) | **~20 % complete.** 1 of 6 experiments reportable; 1 of 14 dispatched arms scored. Five arms sit interrupted mid-training with resumable checkpoints. T6 has never run and needs **no training at all**. Tasks **A3, A4, A5, B6, E1–E4**. |
| [`My_refered_papers/`](../../../My_refered_papers) | **COMPLETE as a study** — 32 papers, 9 categories, and the source of the A1–A9 proposals. What is *not* done is the work it prescribes: a four-paper roadmap and five cheap unrun analyses, four of which need no GPU. Tasks **C1–C6**, mapped in [`05_PUBLICATION_MAP.md`](00_PLANNING/05_PUBLICATION_MAP.md). |

---

## The waves

| Wave | What | GPU-h | Gated by |
|---|---|---:|---|
| **W0** | Unblock: two one-line defect fixes, the arbiter registry entry, the driver ticket | 0 | — |
| **A** | Harvest work already paid for: score trained checkpoints, resume interrupted arms | ~90 | nothing — runs today |
| **B** | The arbiter run, and seed replication | ~340 | **the A40 driver** |
| **C** | Zero-training analyses | ~8 | nothing |
| **D** | Retrain what the W0 fixes unblocked | ~210 | W0.2, W0.3, W0.6 |
| **E** | Complete the T-series | ~310 | A4 |
| **F** | Phase II V4/V5, and the mel-scale question | ~110 | W0.4 |
| **G** | Corrections, withdrawals, final report | ~30 | — |
| **H** | Data acquisition — tracked, not scriptable | — | external |

**~1,100 GPU-hours ≈ 46 GPU-days.** On today's 2 × T4 that is 23 days of pure
compute. On the full complement it is about 7. **The schedule is a function of
one infrastructure ticket and almost nothing else** — which is the argument for
raising it today.

---

## Already done in this session

| Task | Outcome |
|---|---|
| **W0.1** | Preflight: 2 usable GPUs, 3 dead nodes diagnosed, 10 unscored trained runs found |
| **W0.2** | `test_normalize` delegation added to the A1/A8 wrapper losses — **4 blocked runs unblocked** |
| **W0.3** | fp32 mel guard in `DK_CAMPP.features()`, **verified on GPU**: 0 non-finite under autocast, bit-identical in fp32, measured fp16 headroom only **1.42×** |
| **W0.4** | Driver ticket generated with live per-node evidence and the exact fix |
| **W0.5** | Arbiter front-ends registered and verified a **single-factor** contrast (only `model` differs from `ssl_wavlm_ft`) |
| **W0.6** | **New defect found and fixed**: `evaluate_proposal.py` never put `_trainer_shim/` on `sys.path`, so no proposal supplying its own model could be scored — which is why A7, A4 and A5 sat trained-but-unscored |
| **C4** | Score-shift figure: per-language midpoint shift **+0.1275** on matched held-out sets |
| **C5** | **Publication gate cleared** — 0 of 74 runs warm-started, so the two-stage confound does not apply; the PEFT negative may enter paper P2 |
| **G2–G6** | `ERRATA.md` generated; the retired 10.32 % claim found in **47 places** across project documents |
| **A1** | **A7 PLLW scored** — report open item #3. 3.024 % cosine / 2.631 % AS-Norm on held-out si. *Not yet interpretable*: confounded with 2.3x more training data, and the control did not exist — added as **A6** |
| **A3** | **Failed correctly.** T6 refused to write a result: its uncapped rung differed from the published number by 0.010082 pp = **exactly 2 trials of 19,838**, because T1 was scored on an A40 and probed on a T4. Script now re-scores on matching hardware first |

---

## The standard

Unchanged from the A-series' six rules and the T-series' two additions, because
results from this plan are meant to sit in the same tables as
`experiments/results/`:

1. Fit on train, apply to eval.
2. **Use `experiments/splits/` only** — the shipped `lists/` are 100 % closed-set.
3. Same metrics code — the trainer's own `tuneThreshold`, borrowed not copied.
4. Paired comparison on identical trials, with the speaker-clustered bootstrap.
5. Report the seed, and run three for any claimed improvement under 1 pp.
6. **Never write inside `voxceleb_trainer/`** — asserted, not assumed.
7. Parameter-match before quoting; label every unmatched arm wherever it appears.
8. A control must be an **exact reduction**, asserted bit-identical.

Rule 2 has bitten this project once already, and rule 8 exists because of A7.
Neither is ceremonial.

---

## Pre-registered predictions

Seven, in [`02_STATE/predictions.json`](02_STATE/predictions.json), recorded
before the runs and to be scored afterwards **whichever way they come out**. The
annual report scores ~40 % of its predictions as falsified and argues that rate
is a healthy sign: it means they were specific enough to be wrong, and that they
were scored rather than rationalised. Two of its falsifications became the
programme's strongest results.

A plan that quietly dropped its own falsified predictions would be the first
regression from that standard.
