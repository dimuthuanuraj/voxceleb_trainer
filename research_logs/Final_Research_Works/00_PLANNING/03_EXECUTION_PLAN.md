---
title: "Final Research Works — Execution Plan"
subtitle: "The order of work, why it is that order, and how to run it one task at a time"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# 1. The operating principle

> **Run one task. Read its exit test. Then run the next.**

Not because sequencing is elegant, but because this project's own record says so.
The annual report's §12.2 through-line is that *"in low-resource speaker
verification the dominant error is not a bad model, it is a good number measured
on something other than what it claims."* Every device in Phase III exists to
convert a silent wrong number into a loud failure. A batch dispatch that walks
away from fourteen jobs is how 24 August produced five interrupted arms, four
NaN runs and nineteen idle days.

So the runner ([`../01_SCRIPTS/runner.py`](../01_SCRIPTS/runner.py)) is
deliberately **one task at a time, with a ledger**. It refuses to start a task
whose dependencies have not passed their exit test, it records what it ran, and
it survives a cluster reboot because the ledger is on disk and every training
task resumes from its own checkpoints.

---

# 2. Ordering rationale

Four rules decide the order. They are applied in this precedence:

1. **Unblock before doing.** A one-line fix that frees four runs outranks any
   single run. (Wave 0.)
2. **Harvest before spending.** Work already paid for in GPU-days and not yet
   scored outranks new training. (Wave A.)
3. **When capacity is scarce, spend operator time instead of GPU time.** While
   only 2 × T4 are alive, the zero-training analyses are strictly better value
   than queueing a run that will not fit. (Wave C, interleaved.)
4. **Highest-value-per-GPU-hour first, among what fits.** The arbiter run decides
   whether an existing paper stands. It outranks completeness. (Wave B.)

---

# 3. The schedule

Three parallel tracks. The **cluster track** is gated on the administrator; the
**GPU track** and **desk track** are not, and must not idle waiting for it.

## Day 0 — today

| Track | Tasks | Why now |
|---|---|---|
| Cluster | **W0.4** raise the driver ticket | Critical path for everything in Wave B, D, E, F |
| GPU | **W0.1** preflight → **A1**, **A2**, **A3** | Three results and a training-free probe, all on the live T4s |
| Desk | **W0.2**, **W0.3**, **W0.5** | Two one-line fixes and the arbiter registry entry |

Day 0 exit: **four results that did not exist this morning**, two defects closed,
and the arbiter ready to launch the moment an A40 appears.

## Days 1–3 — while the ticket is open

| Track | Tasks |
|---|---|
| GPU | **A4** resume the five interrupted T-arms (they fit on T4s), then **A5** |
| Desk | **C4**, **C5** — score-shift histogram and the PEFT two-stage gate, both CPU |
| GPU (spare) | **C1**, **C2**, **C3**, **C6** as slots free |

> **C5 is scheduled here deliberately.** It is a half-hour of reading argv that
> decides whether prediction 9 on the scoreboard survives. Doing it before any
> paper text is written costs nothing; doing it after costs a retraction.

## Days 4–10 — assuming the ticket clears

| Priority | Tasks | Placement |
|---|---|---|
| 1 | **B1**, **B2** arbiter si + ta | **A40 only** — 30 GB |
| 2 | **D1** A4/A5 `ta` retrains under the fp32 fix | T4s |
| 3 | **D2**, **D3** A1/A8 on `combined` | T4s / A10s |
| 4 | **B3** the 2 × 2 analysis | CPU, immediately after B2 |

## Days 11–20

| Priority | Tasks |
|---|---|
| 1 | **B4**, **B5**, **B6** seed replication — the largest single block, ~300 GPU-h |
| 2 | **E1**, **E2** T-series completion |
| 3 | **F1** V4/V5, **F3** the mel-scale question |

## Days 21–28

| Priority | Tasks |
|---|---|
| 1 | **E3**, **E4** T3/T4 |
| 2 | **G1**–**G6** corrections and errata |
| 3 | **G7** consolidated report and paper drafts |

**If the ticket never clears**, the plan degrades to Waves 0, A, C, and the T4-
sized parts of D and E — roughly 40 % of the register — and Wave B is deferred
indefinitely. That is the cost of W0.4, stated in advance.

---

# 4. Running it

```bash
cd research_logs/Final_Research_Works/01_SCRIPTS

./preflight.sh                    # W0.1 — always first, after any reboot
python3 runner.py --status        # the ledger: what passed, what is ready, what is blocked
python3 runner.py --next          # show the next ready task without running it
python3 runner.py --run W0.2      # run one task
python3 runner.py --run-next      # run the next ready task
python3 runner.py --graph         # dependency graph and critical path
```

The runner never runs two tasks at once and never runs a task whose dependencies
have not recorded a pass. To override — which should be rare and deliberate —
`--force` is available and is written into the ledger as an override so the
record shows it happened.

**After any cluster reboot:** re-run `./preflight.sh`, then
`python3 runner.py --status`. Training tasks resume from their own checkpoints
(`trainSpeakerNet.py:474-486` loads the highest `model0*.model` and continues),
so a reboot costs the current epoch and nothing more.

---

# 5. The standard every result must meet

Unchanged from the A-series' six rules (`proposals/EXPERIMENT_PLAN.md §5`) and
the T-series' two additions, because results from this plan are meant to sit in
the same tables as `experiments/results/`:

1. **Fit on train, apply to eval.** Calibrators and cohorts come from
   `train_list.txt` only.
2. **Same splits as v1** — `experiments/splits/`, speaker-disjoint. **Never** the
   shipped `lists/`, which are 100 % closed-set.
3. **Same metrics code** — the trainer's own `tuneThreshold`, borrowed not copied.
4. **Paired comparison on identical trials**, with the speaker-clustered
   bootstrap. Absolute EER at S ≈ 91 carries ± 5–6 pp; only paired contrasts
   resolve architecture-scale differences.
5. **Report the seed, and run three** for any claimed improvement under 1 pp.
6. **Never write inside `voxceleb_trainer/`** — each strand's `selftest.py`
   asserts this via `git status --porcelain`.
7. **Parameter-match before quoting**, and label every unmatched arm as such
   wherever its number appears.
8. **A control must be an exact reduction**, asserted bit-identical — not
   asserted to be similar.

> Rule 2 has bitten this project once already and rule 8 exists because of A7.
> Neither is ceremonial.

---

# 6. Pre-registered predictions for this plan

Written before the runs, to be scored afterwards whichever way they come out.
This continues the practice the annual report §9 scores at ~40 % falsified, and
that falsification rate is the point.

| # | Task | Prediction | Rationale |
|---|---|---|---|
| **P21** | **B1/B2 arbiter** | Layer-weighted **+** fine-tuned will beat layer-weighted **+** frozen on `si`, but by **less than** the 2.369 → 1.69 gap that full fine-tuning showed at last-layer | If layer weighting already recovers most of what depth costs (Finding III-3), fine-tuning has less left to recover. A large gain would mean the two mechanisms are independent; a null would mean layer weighting *is* the cheap substitute for fine-tuning — the more useful result of the two. |
| **P22** | **A3 T6 span probe** | A 41-frame attention cap will cost **< 0.3 pp** on `MFAConformer_full_si` | Finding III-8 showed the quadratic term is a minority of cost at these lengths; if long range were doing work, the from-scratch Conformer would not have lost to ECAPA in the first place. |
| **P23** | **C3 `nisp_tamil` bilingual control** | The Tamil advantage will **shrink by more than half** when measured within one corpus on the same speakers | FLT §5.5 measured Tamil cleaner on all four periodicity measures, and the si/ta gap grows with system quality — the signature of headroom, not of language. |
| **P24** | **B4 seed replication** | The **ordering** of the top 3 systems per language will be stable, but at least one **absolute** EER will move by > 0.5 pp | Finding III-5 says the regime is data-limited; seed variance at S = 91 should exceed the differences between adjacent systems. |
| **P25** | **D3 A8 CD-NDAL** | The corpus adversary will **not** reverse PLDA's sign on `si` | MS §6 risk 2 names this as the falsifiable half of the channel-confound hypothesis, and A6 already showed the confound is real but shallow — it does not dominate neighbourhood structure. |
| **P26** | **G1 Phase I R7** | The 10.32 % headline will **not** replicate within ± 1 pp at 3 seeds | It is single-seed, predates `--deterministic`, and one log in the same record gives 14.62 %. |

Each prediction is recorded in `02_STATE/predictions.json` with a timestamp, and
the runner refuses to mark its task complete until an outcome is written against
it.

---

# 7. What "done" means for this programme

The plan is complete when:

1. Every **[M]** claim in annual report §10.1 still holds under seed replication,
   or has been revised with an interval.
2. Every item in annual report §10.3 has been **corrected or withdrawn** in
   writing — not silently dropped.
3. The arbiter run has either **confirmed or withdrawn** the P3 headline claim
   underpinning `papers/ieee_spl`.
4. The four-paper roadmap in `MS §5` has, for each paper, either its experiments
   done or a stated reason it is deferred.
5. Every pre-registered prediction in §6 above has an outcome written against it,
   including the falsified ones.

Point 5 is not decoration. The annual report's §12.1 E argues that the capability
this programme established is that it *"identified its own evidence problem,
documented it, and rebuilt its methodology around it."* A plan that quietly drops
its own falsified predictions would be the first regression from that standard.
