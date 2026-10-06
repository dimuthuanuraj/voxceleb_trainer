---
title: "Final Research Works — Publication Map"
subtitle: "Which task feeds which paper, and what each paper is still missing"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# 1. Why this document exists

The register orders work by cost and dependency. This document orders it by
**what it buys**, so that a task can be dropped on an informed basis rather than
because it was at the bottom of a list.

Paper structure follows `My_refered_papers/_deep_study/00_MASTER_SYNTHESIS.md §5`,
which ordered four papers **by readiness, not ambition**. That ordering is
preserved here and updated with what has since been measured.

---

# 2. P1 — "Training-free cross-lingual adaptation for Sinhala and Tamil SV"

**Contains:** A2 (CC-NAP) + A3 (SL-AMEC). **Status: nearly writable.**

| Needs | Task | State |
|---|---|---|
| A3 calibration result | — | **done** — Cllr 0.8356 → 0.1873, median 4.5×, 1,225 params |
| A3 metadata null | — | **done** — 29/33 bit-identical under shuffling |
| A2 CC-NAP consolidated | **D4** | pending, ~1 GPU-h |
| Score-shift histogram (Thienpondt Fig. 1) | **C4** | pending, 45 min CPU |
| AS-Norm result | — | **done** — 12–39 % relative, training-free |

**Assessment.** This paper is two cheap tasks from complete. Its strongest asset
is the **controlled null** — a metadata-inert result demonstrated by bit-identical
scores under shuffling rather than by absent significance. Published negatives are
almost nonexistent in this literature, and §12.1 A10 argues this one "stops the
next group spending months on the same idea."

**Venue:** ICASSP or Odyssey. *Confirm deadlines against the official call.*

---

# 3. P2 — "Why is Sinhala harder? Decomposing the low-resource SV penalty"

**Contains:** A7 (PLLW) + A8 (CD-NDAL) + A9 (ARI-SubCenter).
**Status: the highest-novelty paper, and the one the field is missing.**

| Needs | Task | State |
|---|---|---|
| A9 sub-centre positive | — | **done** — si −0.413 pp, p = 0.038, mechanistically coherent with the QC audit |
| A9 seed replication | **B5** | pending — single seed, CI reaches −0.015 pp |
| **A7 PLLW scored** | **A1** | **trained, unscored.** 54 ckpts on disk |
| A8 CD-NDAL both arms | **D3** | blocked on the `test_normalize` fix (W0.2) |
| Channel confound quantified | — | **done** — 91.2 % corpus probe, d′ 3.46 → 2.44 |
| Label reliability measured | — | **done** — two identity bugs found and corrected |

**Assessment.** Nobody else has an audited Sinhala corpus with measured channel
*and* label-reliability statistics, so nobody else can decompose the penalty into
channel, label noise and genuine language distance. This is also where the
negative results belong — the PEFT non-reproduction, the PLDA refutation, the
closed-set discovery — and framed as a decomposition paper they become
load-bearing rather than defensive.

**One task away from a major upgrade: A1.** Scoring A7 costs 25 GPU-minutes and
answers whether the optimal SSL layer differs between Sinhala and Tamil. If it
does, one shared weight vector is leaving performance on the table — the direct
extension of the programme's headline finding for ~39 extra parameters.

**Gate:** **C5** must clear before the PEFT negative can be included.

**Venue:** Interspeech. The venue rewards analysis papers with clean mechanisms.

---

# 4. P3 — "Code-switched SV for Sri Lankan Sinhala, Tamil and English"

**Contains:** A4 (LS-CAM) + A1 (DA²-LoRA), plus a released code-switched trial list.
**Status: blocked on a corpus that does not exist.**

| Needs | Task | State |
|---|---|---|
| Code-switched SL corpus with frame-level language tags | **H4** | **not started** |
| A4 LS-CAM `si` scored | **A2** | trained, unscored |
| A4 `ta` retrained | **D1** | blocked on the fp32 mel fix |
| A1 DA²-LoRA | **D2** | blocked on the `test_normalize` fix |

**Assessment.** Very high novelty — no prior work at all — and the only paper
here whose blocker is not compute. Everything else in the programme runs on data
in hand; P3 does not. `MS §6` risk 4 is explicit: *budget the collection and
annotation honestly, or P3 slips indefinitely.*

**Recommendation: decide P3's fate by day 20.** If H4 has no progress, defer it
explicitly and move A4/A1 into P2 as architecture arms. An undeclared slip is
worse than a declared deferral.

---

# 5. P4 — "Label-efficient SV for low-resource languages"

**Contains:** A6 (CA-SSPS) + A5 (DK-CAM++). **Status: partially falsified, and that is publishable.**

| Needs | Task | State |
|---|---|---|
| A6 channel-aware SSPS | — | **done — premise falsified.** Plain SSPS purity 0.9302 vs channel-aware 0.9099 |
| A5 DK-CAM++ `si` scored | **A2** | trained, unscored |
| A5 `ta` retrained | **D1** | blocked on the fp32 fix |
| Unlabelled Sinhala broadcast audio | **H4**-adjacent | not started |

**Assessment.** A6's result inverts the paper's original pitch, and honestly
reported that is the contribution: the channel confound is *real but shallow* —
linearly decodable at 91.2 %, yet not dominant in local neighbourhood structure.
The consequence, already recorded in §7.1, is that work aimed at the confound
should target the **scoring** stage, not the clustering stage. That redirection
is worth more than the original hypothesis would have been.

---

# 6. The transformer strand — where does it publish?

T1–T6 do not map to `MS §5`, because the synthesis predates them. They form a
coherent fifth contribution:

> **"On low-resource languages the value of a transformer is in its pretraining,
> not its architecture."** The same family loses by 0.665 pp trained from scratch
> and wins by 1.92 pp pretrained and correctly read, on identical trials.

| Needs | Task | State |
|---|---|---|
| T1 si contrast | — | **done** — 4.950 vs 4.285 |
| T1 ta arm | **A4** | interrupted at epoch 30, resumable |
| Paired bootstrap on the T1 contrast | **A5** | pending |
| T1 seed replication | **B6** | pending |
| T5 cost result | — | **done** — MAC exponent 1.134, "efficient" attention 1.28× *more* expensive |
| **T6 span probe** | **A3** | **training-free, unrun, runs today** |

**Assessment.** Two strong results in hand (the confirmed pre-registered negative,
and the cost finding that killed four planned runs before they spent GPU-hours).
The caveat blocking publication is named in §7.2: *single seed, one language,
no paired bootstrap.* **A4 + A5 + B6 close it exactly.**

T6 could add a third: if a 41-frame cap costs nothing, the long-range advantage
is not doing work on these corpora — and that reframes the strand *including for
the SSL systems, whose front-ends are also attention-based.*

---

# 7. Cross-cutting: what the zero-training tasks buy

None of these belongs to one paper; each strengthens several and none needs the
broken nodes.

| Task | Buys |
|---|---|
| **C1** short-duration table | An operating-point table nobody has published for si or ta. Strengthens every paper's deployment section |
| **C2** enrolment/privacy table | The analysis a KYC deployment will eventually be asked for. Costs no training |
| **C3** `nisp_tamil` bilingual control | **Settles whether "Tamil is easier" is a language finding or a channel artefact.** Currently a standing caveat on every Tamil number in the programme |
| **C4** score-shift histogram | The figure that anchors P1's narrative |
| **C5** PEFT two-stage check | Decides whether P2 may include the PEFT negative |
| **C6** dialect gap | Converts blocker B-2 from a silent limitation into a measured, reportable one |

> **C3 deserves particular emphasis.** The annual report says twice that "Tamil
> verifies 6× better than Sinhala" **must not** be quoted as a language finding,
> because the gap grows with system quality — the signature of headroom, not
> language. `nisp_tamil` has 65 bilingual speakers and its split already exists.
> Two GPU-hours would replace a caveat with a number.

---

# 8. Priority, if only five tasks could be done

1. **A1** — score A7 PLLW. 25 GPU-min. Upgrades P2 and answers open item #3.
2. **A3** — T6 span probe. 40 GPU-min, no training. Potentially reframes the whole transformer strand.
3. **C5** — the PEFT two-stage gate. 30 CPU-min. Decides whether a published claim survives.
4. **B1** — the arbiter run. Decides whether `papers/ieee_spl` stands.
5. **C3** — `nisp_tamil` bilingual control. Replaces the programme's most-repeated caveat with a measurement.

Four of the five cost under an hour each. Only one needs the A40.
