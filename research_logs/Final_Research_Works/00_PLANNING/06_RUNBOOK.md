---
title: "Final Research Works --- Runbook"
subtitle: "Every command, in order. Start at step 0 and work down."
author: "Dimuthu Anuraj"
date: "2026-09-14"
---

# How to use this

Copy-paste, top to bottom. Each step says **what it does**, **how long**, and
**how you know it worked**. Steps marked **[BLOCKED]** will refuse to run until
their dependency clears — that is intentional, not a bug.

One rule: **run one step, read its output, then run the next.** The runner
enforces it, and the reason is in
[`03_EXECUTION_PLAN.md`](03_EXECUTION_PLAN.md) §1.

```bash
# The only path you need. Everything below assumes you are here.
cd /mnt/ricproject3/2026/SL_SPV/voxceleb_trainer/research_logs/Final_Research_Works/01_SCRIPTS
```

**Two ways to work:** step A runs everything unattended in tmux; steps 0–10 are
the manual sequence. They share the same ledger, so you can switch between them
freely — start the autopilot, stop it, run a task by hand, start it again.

---

# Step A — Unattended mode (the autopilot)

If you want the plan worked down without babysitting it, start here and skip to
step 9 when it stops.

```bash
./start_autopilot.sh              # create the tmux session and attach
./start_autopilot.sh --detach     # create it, leave it running, stay in your shell
./start_autopilot.sh --stop       # stop scheduling (detached training survives)
```

The session `slspv` has three windows:

| Window | Shows |
|---|---|
| **0 autopilot** | the scheduler log, also tee'd to `02_STATE/autopilot.log` |
| **1 status** | the task ledger, refreshed every 60 s |
| **2 gpus** | live GPU usage across every node |

`Ctrl-b` then `0`/`1`/`2` to switch, `Ctrl-b` then `d` to detach,
`tmux attach -t slspv` to come back.

## What it does

Picks every task whose dependencies are satisfied, runs several at once, one per
idle GPU, and keeps going until nothing is left. Each attempt writes
`03_RESULTS/<ID>/autopilot_<n>_<timestamp>.log`, and
`03_RESULTS/<ID>/attempts.json` records the history. **Nothing is overwritten.**

## What it will not do, deliberately

| | Why |
|---|---|
| **W0.4** (driver ticket) | needs the cluster administrator, not a GPU |
| **G1** (Phase I claim) | needs a *recorded human decision* — retire or re-run. It must not be picked by default |
| **B1/B2** (arbiter) | needs 30 GB; the largest live card is 14.9 GB. It checks before dispatching rather than OOM-ing half an hour in |
| Retry forever | a task that runs cleanly but never satisfies its exit test is **parked** after a few attempts. Parking is visible; an infinite loop is not |

## GPU policy

**One training job per card.** These are T4s and two concurrent trainings on one
card thrash rather than share. The scheduler counts cards with *zero compute
processes* — not cards that merely answer `nvidia-smi` — and multi-arm tasks
dispatch only as many arms as there are idle cards, deferring the rest to the
next pass. Every such task is idempotent, so re-running picks up where it
stopped.

> This was learned the hard way on 2026-09-14: `gpurun.sh` picks the emptiest
> card, but a model takes ~30 s to allocate, so five arms dispatched ten seconds
> apart all read the same stale free-memory figure and piled onto one node while
> another sat idle. Slots are now handed out round-robin from a live idle-card
> list.

## Watching it

```bash
python3 autopilot.py --status            # what is running right now
python3 autopilot.py --dry-run           # what would start, without starting it
tail -f ../02_STATE/autopilot.log        # the scheduler log
tail -f ../03_RESULTS/A4/autopilot_*.log # one task's output
```

**Manual steps still needed while it runs:** step 1 (send the driver ticket) and
step 9a (decide the Phase I claim). Everything else it will attempt.

---

# Step 0 — Orient yourself (every session, 1 min)

```bash
./preflight.sh                  # cluster, env, splits, disk, orphan checkpoints
python3 runner.py --status      # what passed / is ready / is blocked
```

**Run `./preflight.sh` again after every cluster reboot.** It is the only thing
that tells you whether the GPUs came back.

Reading `--status`:

| | |
|---|---|
| `PASS` | exit test satisfied — an artefact exists on disk |
| `READY` | dependencies met, not yet done — **you can run this now** |
| `BLOCK` | waiting on something; the line below names it |

Other commands you will want:

```bash
python3 runner.py --next        # name the next ready task, without running it
python3 runner.py --graph       # dependency graph + critical path
python3 runner.py --run-next    # run whatever is next
python3 runner.py --run A2      # run one specific task
python3 runner.py --run A2 --force   # override a block (recorded in the ledger)
```

---

# Already done — no action needed

These passed on 2026-09-12 and their artefacts are under `../03_RESULTS/`. They
appear as `PASS` in `--status`; you do not need to run them.

| Task | What it did |
|---|---|
| **W0.1** | Preflight: found 2 usable GPUs, 3 dead nodes, 10 unscored trained runs |
| **W0.2** | `test_normalize` delegation on the A1/A8 wrapper losses — unblocked 4 dead runs |
| **W0.3** | fp32 mel guard in `DK_CAMPP.features()`, verified on GPU (0 non-finite, bit-identical fp32, fp16 headroom only **1.42x**) |
| **W0.5** | Arbiter front-ends registered; verified a **single-factor** contrast |
| **W0.6** | Fixed `evaluate_proposal.py` — it could not score *any* proposal supplying its own model. This is why A7, A4 and A5 sat unscored |
| **A1** | Scored A7 PLLW (report open item #3) — see step 2b for why it needs A6 |
| **C4** | Score-shift figure: per-language midpoint shift **+0.1275** |
| **C5** | PEFT publication gate **cleared** — 0 of 74 runs warm-started |
| **G2-G6** | `ERRATA.md` + machine-readable forbidden-claims list |

To re-run any of them anyway: `python3 runner.py --run <ID> --force`.

---

# Step 1 — Send the driver ticket (5 min of your time; unblocks ~760 GPU-hours)

**This is the highest-value action available and it is not a research task.**

```bash
cat ../03_RESULTS/W0.4/DRIVER_TICKET.md
```

Send that file to the cluster administrator. It carries live per-node evidence
and the exact fix. You cannot do it yourself — `sudo -n` fails for this account
on both affected nodes.

To refresh the evidence before sending:

```bash
python3 runner.py --run W0.4 --force
```

**Done when:** `python3 runner.py --status` shows 3+ GPUs and `W0.4` flips to
`PASS` on its own.

**Update 2026-09-14: `compute-node-2` has returned** — capacity is now **4 × T4
(15 GB)**, double what it was. But `node-3` (2 × A10 23 GB) and `node-4`
(A40 46 GB) are still down, so **the largest single card is 14,914 MiB and the
arbiter still cannot be placed.** More T4s do not help a 30 GB job. The ticket
still matters, and `W0.4`'s exit test now requires a card with ≥ 20 GB free
rather than just a GPU count — so it will not report success on T4s alone.

---

# Step 2 — Harvest the work already paid for (today, on the T4s)

These are trained checkpoints that were never scored. The GPU-days are already
spent.

## 2a. Score A4 LSCAM_si and A5 DKCAMPP_si — 50 GPU-min

```bash
python3 runner.py --run A2
```

**Done when:** two `*.eval.json` files appear under
`proposals/A4_ls_cam/out/` and `proposals/A5_dk_campp/out/`.

> **Caveat to carry with A4's si number:** LS-CAM segments by *language*, and
> `si` has one. Report it as a **trunk** result (DK-CAM++ with LS-CAM's head),
> not as evidence about language segmentation. The informative arm is the Tamil
> one, which step 4a produces.

## 2b. Make A7's number interpretable — ~8 GPU-h, fits a T4

```bash
python3 runner.py --run A6
```

A1 already scored A7 PLLW: **3.024 % cosine / 2.631 % AS-Norm** on held-out
Sinhala, against its reference baseline's 3.297 / 2.964 — on the same 19,838
trials. **But that difference confounds the method with 2.3× more training
speakers**, and no `combined`-condition control exists anywhere in v1 (verified:
zero rows). A6 builds it.

**Done when:** `03_RESULTS/A6/result.json` has `"ok": true` and prints two
single-factor contrasts (METHOD and DATA) instead of one confounded pair.

## 2c. The T6 span probe — 40 GPU-min, no training

```bash
python3 runner.py --run A3 --force
```

`--force` because the first attempt failed and is recorded as such.

> **What failed, and why it was right to fail.** T6 refused to write a result:
> its uncapped rung gave 4.960177 against the published 4.950096 — a difference
> of **0.010082 pp, which is exactly 2 trials of 19,838**. T1 was scored on
> node-4's **A40**; the probe ran on node-1's **T4**. Same checkpoint, 231
> tensors loaded both times — this is cuDNN kernel selection across
> architectures, not a rebuild error. The guard exists so a capped EER is never
> compared against an uncapped one from a different code path, and it should not
> be disabled.
>
> A3 now handles it: if no large card is free it re-scores T1 on the current
> card first, **backs up and restores the published eval.json** (T1's 4.950 is
> quoted in the annual report), then probes against a same-hardware baseline.

**Done when:** a JSON appears in `transformer_sv/T6_span_probe/out/` naming the
tag, with an EER-vs-span ladder.

**Then score prediction P22** — "a 41-frame cap costs < 0.3 pp" — in
`../02_STATE/predictions.json`, whichever way it came out.

## 2d. Resume the five interrupted T-arms — 60–90 GPU-h, detached

```bash
python3 runner.py --run A4
```

It launches detached jobs and returns. **Re-run the same command** after they
finish to score them; it skips anything already scored.

> These are **not** safe to score where they stand. Three of the five had their
> *best* epoch as their *last* when the cluster died — they were still
> improving. Resuming is free: the trainer auto-resumes from the highest
> checkpoint.

**Watch them:**
```bash
tail -f ../03_RESULTS/A4/*.resume.log
cd ../../../../transformer_sv && python3 scaffold.py --check   # the figure to trust
```

## 2e. Paired bootstrap on the T1 contrast — 15 min CPU

```bash
python3 runner.py --run A5     # needs A4 first
```

Removes the annual report's standing caveat on T1: *"the paired bootstrap has
not been run on this contrast."*

---

# Step 3 — Zero-training analyses (do these while the cluster is degraded)

Four of the six need no GPU training at all. Three are unpublished for Sinhala
or Tamil by anyone.

```bash
python3 runner.py --run C3      # nisp_tamil bilingual control   ~2 GPU-h
python3 runner.py --run C1      # short-duration table 2/3/5 s   ~3 GPU-h
python3 runner.py --run C2      # enrolment / privacy table      ~2 GPU-h
python3 runner.py --run C6      # dialect gap (Indian -> SL Tamil) ~1 GPU-h
python3 runner.py --run D4      # A2 CC-NAP consolidation        ~1 GPU-h
```

**C3 is the one to do first.** It settles whether "Tamil verifies 6× better than
Sinhala" is a language finding or a recording artefact — currently a standing
caveat on every Tamil number in the programme. Score **P23** when it lands.

Already done, no action needed: **C4** (score-shift figure, shift **+0.1275**)
and **C5** (PEFT gate — **cleared**, 0 of 74 runs warm-started).

---

# Step 4 — Retrain what the Wave 0 fixes unblocked

Each launches detached; **re-run the same command afterwards to score**.

## 4a. A4/A5 Tamil, under the fp32 mel fix — ~70 GPU-h

```bash
python3 runner.py --run D1
```

> **Read the first epoch before walking away.** The lesson these very runs
> taught: a mixed-precision NaN is invisible in the harness — no exception, no
> exit code, just `Loss nan` scrolling past.
> ```bash
> grep -c 'nan' ../../../../proposals/A*/out/*_ta_s42.train.log   # expect 0
> ```

## 4b. A1 DA²-LoRA and A8 CD-NDAL on `combined` — ~70 GPU-h each

```bash
python3 runner.py --run D2
python3 runner.py --run D3
```

Both were dead before W0.2 — they died at their *first* validation and left zero
checkpoints. `combined` is mandatory, not convenient: on a single-language
condition both discriminators are constant and the methods silently degrade to
their own baselines. The shim now refuses rather than run.

Score **P25** on D3 — "the corpus adversary will not reverse PLDA's sign on si."

---

# Step 5 — [BLOCKED on step 1] The arbiter run

**The report's #1 open item, and it decides whether a drafted paper stands.**

```bash
python3 runner.py --run B1      # si, ~20 GPU-h, A40 ONLY (30 GB)
python3 runner.py --run B2      # ta, ~20 GPU-h
python3 runner.py --run B3      # the 2x2 analysis, 20 min CPU
```

v1 measured layer-weighted+frozen (2.369 si) and last-layer+fine-tuned (5.404
si) but **never the diagonal** — and P3's headline claim, which underpins
`papers/ieee_spl`, rests on it. B3 either confirms the claim or forces its
withdrawal. Score **P21**.

The registry entry is already in place and verified a **single-factor** contrast
(only `model` differs from `ssl_wavlm_ft`).

---

# Step 6 — Seed replication (runs on the T4s today)

```bash
python3 runner.py --run B4      # top 3 per language x seeds {123,7}  ~180 GPU-h
python3 runner.py --run B5      # A9 sub-centre x seeds               ~60 GPU-h
python3 runner.py --run B6      # T1 full_si x seeds                  ~60 GPU-h
```

Largest block in the plan. Every v1, proposal and T-series number is currently
single-seed, and the power ceiling says an absolute EER at S ≈ 91 carries ±5–6
pp. Score **P24**.

> **These need only 8 GB, so they run on the recovered T4s today.** They were
> originally gated behind the driver ticket, which was over-restrictive: task
> capacity is already enforced per-task by `gpu_mb`, and only the 30 GB arbiter
> (B1/B2) genuinely needs a big card. Rewired 2026-09-14.

**B5 matters most per GPU-hour:** A9's positive result has a CI reaching
−0.015 pp. It is one seed away from crossing zero.

---

# Step 7 — Complete the T-series

Run in this order; it is fixed by `transformer_sv/EXPERIMENT_PLAN.md` §4 and
should not be reshuffled for convenience.

```bash
python3 runner.py --run E1      # T1 no_mfa + m1024      ~70 GPU-h
python3 runner.py --run E2      # T2 Tamil arms          ~70 GPU-h
python3 runner.py --run E3      # T3 VOT                 ~80 GPU-h
python3 runner.py --run E4      # T4 SpecViT             ~90 GPU-h
```

---

# Step 8 — Phase II and the open questions (runs on the T4s today)

```bash
python3 runner.py --run F1      # V4 (SE) and V5 (Res2Net)   ~70 GPU-h
python3 runner.py --run F3      # the mel-scale question     ~40 GPU-h
```

**F3 is scaffolded, not implemented** — it prints an implementation plan and
exits `not-ok` by design. It needs a `--fb_scale` front-end option and an
exactness assertion before it can run. Read its output before committing GPU
time.

**Never present F1 as recovering a Phase II number.** Annual report §10.3 item 1
forbids citing that period at all. These are new open-set runs.

---

# Step 9 — Corrections and withdrawals

## 9a. Decide the Phase I 10.32 % claim — **this needs your decision**

```bash
python3 runner.py --run G1      # prints the two options and exits 1
```

It appears in **47 places** across project documents. It is single-seed,
predates `--deterministic`, and another log in the same record gives 14.62 %.
Audit item R7 gives two options and there is no third:

```bash
G1_MODE=retire python3 runner.py --run G1 --force   # 0 GPU-hours
G1_MODE=rerun  python3 runner.py --run G1 --force   # ~30 GPU-hours
```

**Retirement is legitimate and cheap.** Phase I studied English VoxCeleb with a
question the field has since settled, and the thesis subject is now Sinhala and
Tamil. But it must be a recorded decision, not a drift.

After deciding, regenerate the errata so it carries the outcome:

```bash
python3 runner.py --run G2-G6 --force
cat ../04_REPORTS/ERRATA.md
```

## 9b. Apply the errata to the documents

`../03_RESULTS/G2-G6/forbidden_claims.json` lists every forbidden claim with its
replacement and file:line locations. Work through them, then re-run the scan to
confirm.

---

# Step 10 — The final report

```bash
python3 runner.py --run G7
cat ../04_REPORTS/final_research_report.md
```

It reads the filesystem, so it cannot drift from what exists. **Re-run it after
any task completes.**

Before calling the programme done, check all five in
[`03_EXECUTION_PLAN.md`](03_EXECUTION_PLAN.md) §7 — in particular that **every
pre-registered prediction has an outcome written against it, including the
falsified ones.**

---

# Not scriptable — these need a person

| ID | What | Why it matters |
|---|---|---|
| **H1** | Integrate **SLCeleb** | The binding blocker on *any* Sri Lankan claim. The only corpus whose absolute EER means what a reader assumes |
| **H2** | Acquire **within-speaker, cross-channel Sri Lankan recordings** | **Unlocks 4 of the 9 proposals at once** — a sharper priority than any individual method |
| **H3** | **Listening pass** on `slr52_sinhala` | The shortlists exist. No metric can close this, and every Sinhala number's weight depends on it |
| **H4** | Build a **code-switched SL corpus** | Paper P3 is blocked on it entirely. Decide its fate by day 20 rather than letting it slip silently |

---

# If something goes wrong

**A task fails.** Read `../03_RESULTS/<ID>/*.log`. The scripts tee everything.
Fix, then `python3 runner.py --run <ID> --force`.

**The cluster reboots mid-run.** Nothing is lost — per-epoch checkpointing plus
auto-resume, verified gap-free across four reboots in August. Re-run
`./preflight.sh`, then re-issue the same task command; it resumes.

**A task says `ran-but-incomplete`.** Normal for long detached jobs: the
dispatch returned but training is still going. Re-run it later to score.

**You are not sure what is safe to quote.** `../04_REPORTS/ERRATA.md` for what
is forbidden; annual report §10.1 for what is safe; §10.2 for what is not yet.

**Two jobs fight over a GPU.** The runner runs one task at a time, but detached
jobs persist. Check with:
```bash
ssh compute-node-1 nvidia-smi
```

---

# Quick reference

| Want | Command |
|---|---|
| Where am I? | `python3 runner.py --status` |
| What next? | `python3 runner.py --next` |
| Just do the next thing | `python3 runner.py --run-next` |
| After a reboot | `./preflight.sh && python3 runner.py --status` |
| Cluster GPUs | `ssh compute-node-1 nvidia-smi` |
| T-series truth | `cd ../../../../transformer_sv && python3 scaffold.py --check` |
| Proposals truth | `cd ../../../../proposals && python3 scaffold.py --check` |
| Regenerate the report | `python3 runner.py --run G7 --force` |
