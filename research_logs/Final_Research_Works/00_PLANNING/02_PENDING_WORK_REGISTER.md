---
title: "Final Research Works — Pending Work Register"
subtitle: "Every outstanding task, with its evidence, cost, dependency and exit test"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# 1. How to read this

One row per task. Every task has:

* an **ID** (`W0.1`, `A1` …) used by the runner and by every other document here;
* a **source** — where the requirement comes from, so nothing is invented;
* a **cost** in GPU-hours or CPU-minutes, measured from comparable completed runs;
* a **dependency** — what must finish first;
* an **exit test** — the artefact that proves it is done. *A task without an
  exit test is not a task, it is an intention.*

Sources are abbreviated: **AR** = annual report 2026-09-10 (with section),
**AUD** = [`01_STATUS_AUDIT.md`](01_STATUS_AUDIT.md), **MS** =
`My_refered_papers/_deep_study/00_MASTER_SYNTHESIS.md`, **FLT** =
`feature_level_testing/RESULTS.md`, **TEP** = `transformer_sv/EXPERIMENT_PLAN.md`.

Cost units: **GPU-h** on a Tesla T4 unless stated. A full si/ta backbone run to
early stopping is ≈ 30–36 GPU-h on a T4 (measured: T1 `full_si` reached epoch 31
in ~8 h wall on an A40; T4s are ~2.5× slower).

---

# 2. Wave 0 — Unblock. No GPU. Do this first.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **W0.1** | Preflight: probe cluster, verify conda env, splits SHA-256, disk headroom | AUD §3 | 3 min CPU | — | `02_STATE/preflight.json` written, `ok: true` |
| **W0.2** | Fix `test_normalize` on the A1/A8 wrapper losses | AUD §5.1 | 5 min CPU | — | Both shims expose `test_normalize`; import-time assertion passes |
| **W0.3** | Fix fp16 mel overflow in `DK_CAMPP.features()` (fp32 guard) | AR §7.2, AUD §5.1 | 10 min CPU | — | fp16 forward over 30 k augmented samples yields 0 `inf`; fp32 output bit-identical |
| **W0.4** | **Escalate the A40/T4 driver mismatch to the cluster administrator** | AUD §3 | operator | — | node-2 and node-4 return `nvidia-smi` output; 5 GPUs visible |
| **W0.5** | Register the arbiter front-ends (`ssl_wavlm_lw_ft`, `ssl_mhubert_lw_ft`) | AR §11.1, AUD §8 | 15 min CPU | — | `registry.py` dry-run emits a valid argv; param count within 0.1 % of `ssl_wavlm_lw` |
| **W0.6** | Fix `evaluate_proposal.py`'s `sys.path`: no proposal supplying a model can be scored | AUD §9b D-1/D-2 | 10 min CPU | — | Both strands' shim models import in isolated subprocesses; neither shadows the other |

> **W0.6 was discovered by running A1, not by reading the report.** It is the
> reason three fully-trained proposal runs sat unscored for three weeks, and it
> would have blocked D1/D2/D3 as well. Its first attempted fix then broke the
> T-series by shim shadowing — both halves are documented in AUD §9b, because the
> second is the more instructive.

> **W0.4 is the critical path for the whole programme.** It is not a research
> task and it cannot be done from this account — `sudo -n` fails on both nodes.
> Until it clears, usable capacity is 2 × T4 15 GB and the arbiter cannot run.

---

# 3. Wave A — Harvest work already paid for. Runs on 2 × T4 today.

Everything here has **already consumed its GPU-days**. These are the cheapest
results in the programme and none of them needs the broken nodes.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **A1** | Score `A7_pllw_combined_s42` (54 ckpts, early-stopped, never scored) | **AR §11 item 3**, AUD §5 | 25 min GPU | W0.1, W0.6 | `A7_pllw/out/scores/A7_pllw_combined_s42/` + `eval.json` |
| **A2** | Score `LSCAM_si_s42` (A4) and `DKCAMPP_si_s42` (A5), both trained, unscored | AUD §5 | 50 min GPU | W0.1, W0.6 | Two `*.eval.json` with held-out test EER |
| **A3** | **T6 span probe on `MFAConformer_full_si_s42`** — training-free | **AR §11 item 4**, TEP P3 | 40 min GPU | W0.1 | `T6_span_probe/out/*.json` with EER vs span curve |
| **A4** | Resume 5 interrupted T-arms to their own stopping criterion, then score | AUD §4.2 | 60–90 GPU-h | W0.1 | All 5 early-stop or hit epoch 60; 5 `*.eval.json` |
| **A5** | Paired bootstrap: T1 si contrast vs `A_ecapa512` baseline | AR §7.2 caveat | 15 min CPU | A4 | CI and p-value for the +0.665 pp gap |

> **A3 is the single best cost/benefit task in the register.** It is
> training-free, it runs today on a T4, and `transformer_sv/EXPERIMENT_PLAN.md`
> argues it "reframes the whole strand — including for the SSL systems, whose
> front-ends are also attention-based." A 41-frame cap costing nothing would be
> a publishable result obtained in 40 GPU-minutes.

---

# 4. Wave B — The arbiter and the statistics. Needs the A40.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **B1** | **Arbiter run** — layer-weighted **and** fine-tuned SSL, `si` | **AR §11 item 1** | ~20 GPU-h A40 | W0.4, W0.5 | `results/F_ssl_wavlm_lw_ft_..._si_s42/final.json` |
| **B2** | Arbiter run, `ta` | AR §11 item 1 | ~20 GPU-h A40 | B1 | as above, `ta` |
| **B3** | Analyse the 2 × 2: {last, layer-weighted} × {frozen, fine-tuned} | AR §11 item 1 | 20 min CPU | B2 | A 2×2 table with paired CIs; verdict on the P3 headline claim |
| **B4** | Seed replication — top 3 systems per language, seeds {123, 7} | **AR §11 item 2** | ~180 GPU-h | W0.4 | 12 runs; `aggregate_seeds.py` emits mean ± sd |
| **B5** | Seed replication — A9 sub-centre, seeds {123, 7} | AR §11 item 2 | ~60 GPU-h | W0.4 | A9's −0.413 pp gets an interval across seeds |
| **B6** | Seed replication — T1 `full_si`, seeds {123, 7} | AR §7.2 caveat | ~60 GPU-h | A4 | T1 contrast becomes reportable rather than directional |

> **Why B1 matters more than its cost suggests.** v1 measured
> layer-weighted + frozen (2.369 si) and single-layer + fine-tuned (5.404 si) but
> never both at once — and the P3 headline claim currently underpinning
> `papers/ieee_spl` rests on that untested diagonal. B1/B2 either confirm the
> paper or force its withdrawal. It has been the named #1 open item since
> 17 August and is still unrun.

---

# 5. Wave C — Zero-training analyses. **Do these while the cluster is degraded.**

Every task here runs on CPU or a single T4 against checkpoints and score files
that already exist. Four of the five are unpublished for Sinhala or Tamil by
anyone.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **C1** | **Short-duration table** — truncate test utterances to 2 / 3 / 5 s across Stage A checkpoints | MS §5 | 3 GPU-h | W0.1 | EER × duration × language table |
| **C2** | **Enrolment / privacy table** — N = 1, 5, 10, 15 enrolment utterances | MS §5 | 2 GPU-h | W0.1 | EER × N table for the deployed system |
| **C3** | **`nisp_tamil` bilingual control** — within-corpus, same speakers, two languages | **FLT §5.5**, AR §10.2 | 2 GPU-h | W0.1 | Settles whether "Tamil is easier" is language or channel |
| **C4** | **Score-shift histogram** (Thienpondt Fig. 1) for si/ta | MS §5 P1 | 45 min CPU | — | Figure + the scores behind it; anchors the P1 narrative |
| **C5** | **Two-stage fine-tuning confound check** on the PEFT arms | **MS §6 risk 3** | 30 min CPU | — | Verdict on whether the PEFT negative is safe to publish |
| **C6** | **Dialect gap** — train Indian Tamil, evaluate Sri Lankan Tamil | MS §5 | 1 GPU-h | W0.1 | The number, reported with the channel caveat |

> **C5 is a gate, not an analysis.** `MS §6` risk 3 says plainly: if SL_SPV's
> PEFT arms differ in two-stage-ness, *"the negative PEFT result is not safe to
> publish as-is. Check first."* Prediction 9 in the annual report's scoreboard is
> currently recorded as **Falsified**. C5 decides whether that entry survives.
> It costs half an hour of reading argv.

---

# 6. Wave D — Retrain what the Wave 0 fixes unblocked.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **D1** | A4 `LSCAM_ta` + A5 `DKCAMPP_ta` retrain under the fp32 mel fix | **AR §11 item 5** | ~70 GPU-h | W0.3 | Two runs complete with no NaN epoch; checkpoints verified clean |
| **D2** | A1 DA²-LoRA `anchored` + `control` on `combined` | AUD §5.1 | ~70 GPU-h | W0.2 | Both complete and scored |
| **D3** | A8 CD-NDAL `adv0.3` + `control` on `combined` | AUD §5.1 | ~70 GPU-h | W0.2 | Both complete and scored; the pre-registered PLDA-sign prediction is scored |
| **D4** | A2 CC-NAP consolidation into a reportable result | MS §5 P1 | 1 GPU-h | — | `RESULT.md` with the paired comparison |

> **D2 and D3 carry a caveat that must travel with them.** Annual report §7.1
> Finding III-7: A1 and A8 both depend on a domain variable, and `si`/`ta` each
> contain one language and one corpus. They are being run on `combined`
> specifically so the discriminator has more than one class. **Any A1/A8 number
> from a single-language condition is undefined and must not be quoted** — the
> guards added after A7 now enforce this with a `SystemExit`.

---

# 7. Wave E — Complete the T-series.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **E1** | T1 `no_mfa_si`, `m1024_si` | TEP §4, AR §11 item 4 | ~70 GPU-h | A4 | Both scored; `m1024` reported as capacity-unmatched |
| **E2** | T2 `tf4_ta`, `tf0_ta` | TEP P4 | ~70 GPU-h | A4 | Both scored; exact-reduction control asserted |
| **E3** | T3 VOT `r124_si`, `r1_si` | TEP P5 | ~80 GPU-h | E1 | Branch weights reported — the primary output, EER secondary |
| **E4** | T4 SpecViT `p16x4_si`, `frame_si` | TEP P6 | ~90 GPU-h | E3 | Both scored |

> **Ordering follows `EXPERIMENT_PLAN.md §4` and should not be reshuffled for
> convenience.** P1–P3 change one factor against a trustworthy baseline; P5 and
> P6 change more at once and cost more per run. T4 is last because its own token
> accounting shows a ViT is the most expensive arm here (250 tokens/crop against
> T1's 50).

---

# 8. Wave F — Phase II resurrection and the open architecture questions.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **F1** | **V4 (SE) and V5 (Res2Net)** run properly under the v1 harness | **AR §11 item 9** | ~70 GPU-h | W0.4 | Two scored runs; the Phase II designs finally get a verified number |
| **F2** | **A7 PLLW on `combined_si_ta`** — per-language layer weighting | **AR §11 item 3** | — | A1 | *Already trained.* A1 scores it. Kept here as the analysis: does the optimal layer differ by language? |
| **F3** | **The mel-scale question** — is a scale fitted to English perceptual data right for si/ta? | **AR §11 item 11** | ~40 GPU-h | W0.4 | A measured answer to a genuinely publishable open question |

---

# 9. Wave G — Corrections, withdrawals, and consolidation.

These are obligations, not opportunities. Annual report §10.3 lists six items
that **must be corrected or withdrawn**; they are tracked here so they cannot be
quietly skipped.

| ID | Task | Source | Cost | Dep | Exit test |
|---|---|---|---|---|---|
| **G1** | **R7** — re-run the Phase I 10.32 % headline at 3 seeds, or formally retire the claim | **AR §10.3 item 3** | ~30 GPU-h or 0 | W0.4 | Either 3 seeds with an interval, or a signed retirement note |
| **G2** | Errata: Period-2 ResNetSE34L spec (34-layer/6.8 M/8–10 % → 1.50 M/15.48 %) | AR §10.3 item 4 | 20 min | — | Errata entry published |
| **G3** | Errata: "Tamil pool 50 → 752" → measured 706 (802 after QC) | AR §10.3 item 5 | 10 min | — | Errata entry published |
| **G4** | Apply the ~5× read-vs-wild optimism factor and closed-set caveat to every v0 EER | AR §10.3 item 6 | 1 h | — | Every v0 table carries both caveats |
| **G5** | Formally mark all Jan–Jun 2026 results as non-citable in every downstream document | **AR §10.3 item 1** | 1 h | — | A single authoritative statement referenced from each affected doc |
| **G6** | Withdraw the nested-learning "9.84 % EER, validated" claim | AR §10.3 item 2 | 15 min | — | Claim removed; the three NaN collapses recorded in its place |
| **G7** | Final consolidated report + paper drafts | — | 2 days | all | The deliverable |

---

# 10. Wave H — Data and acquisition. Not scriptable. Tracked so they do not vanish.

| ID | Task | Source | Blocking what | Owner |
|---|---|---|---|---|
| **H1** | **SLCeleb** — the binding blocker on *any* Sri Lankan claim | AR §11 item 6 | every absolute EER's external validity | PI |
| **H2** | **Acquire within-speaker, cross-channel Sri Lankan recordings** | **AR §11 item 7** | unlocks 4 of 9 proposals at once | PI |
| **H3** | **Listening pass on `slr52_sinhala`** — shortlists exist, no metric can close it | AR §11 item 8 | every Sinhala number's weight | PI + annotator |
| **H4** | Code-switched SL corpus (broadcast/parliament/podcast), frame-level language tags | MS §5 P3 | paper P3 entirely | PI |

> **H2 is the sharpest data priority in the programme** and the annual report
> says so: four of nine proposals need a domain variable the collection supplies
> only by pooling corpora, which confounds domain with language. One acquisition
> fixes four methods. **H1 and H3 gate the external validity of results the
> project already has** — no amount of compute substitutes for either.

---

# 11. Totals

| Wave | Tasks | GPU-h | Gated by |
|---|---:|---:|---|
| W0 Unblock | 6 | 0 | — |
| A Harvest | 5 | ~90 | nothing (2 × T4 today) |
| B Arbiter + seeds | 6 | ~340 | **A40 driver** |
| C Zero-training | 6 | ~8 | nothing |
| D Retrains | 4 | ~210 | W0.2, W0.3 |
| E T-series | 4 | ~310 | A4 |
| F Phase II + mel | 3 | ~110 | W0.4 |
| G Corrections | 7 | ~30 | — |
| H Data | 4 | — | external |
| **Total** | **45** | **~1,100 GPU-h** | |

**~1,100 GPU-h is 46 GPU-days.** On today's 2 × T4 that is 23 days of pure
compute with zero idle time — not achievable. On the full complement
(2 × T4 + 2 × T4 + A40 + 2 × A10 = 7 GPUs) it is ~7 days of compute, and the
A40 absorbs the SSL work no T4 can hold.

> **The schedule is therefore a function of W0.4, and of nothing else in this
> register.** That is the argument for raising the ticket today.
