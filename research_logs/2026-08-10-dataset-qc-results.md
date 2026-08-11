# Dataset Quality Assessment — Results

**Date:** 2026-08-10 · **Status:** Layers 0–6 executed on all five built corpora
**Method:** `2026-08-10-dataset-quality-assessment-methods-and-plan.md`
**Reports:** `data/datasets_reports/<corpus>/` — HTML + Markdown + PDF + every figure as an individual PNG/SVG
**Raw outputs:** `data/_qc/` · **Embedding model:** SpeechBrain ECAPA-TDNN, 192-d, cosine

---

## 0. Headline

The audit found **two speaker-identity bugs**, both in corpora I had built and
documented myself, and both invisible to every check that does not use audio.
Fixing them changed the headline numbers more than any other result here.

| | Before | After |
|---|---|---|
| SLR127 speakers | 531 | **638** |
| SLR127 EER | **55.31 %** (worse than chance) | **1.07 %** |
| SLR127 d′ / NMI | 3.14 / 0.963 | **5.23 / 0.984** |
| Kathbath speakers | 60 | **49** |

And the decision the whole exercise exists to serve:

> **`slr127_tamil` is the only corpus close to usable power — MDE 0.26 pp**,
> just above the 0.2 pp target and well inside the 0.5 pp limit. Every other
> corpus is 0.6–6.2 pp. Narrow the ablation to fewer, bolder conditions on
> SLR127; treat the rest as confirmatory.

---

## 1. The two identity bugs

### 1.1 SLR127 — the prefix is a numbering scheme, not a session

The v0 survey flagged that SLR127's speaker-id convention is undocumented, and
I resolved it by counting distinct values: field 2 had exactly 531 distinct
values, matching the published speaker count, so I took field 2 as the speaker
and the `ISTL`/`MICI`/`MILE` prefix as a recording *session*. That gave "107
multi-session speakers" and, I claimed, the only genuine cross-session
read-speech trials in the collection.

It was wrong. Embedding the centroids settles it:

| Comparison | Cosine |
|---|---|
| same number, same prefix (split-half of one speaker) | **0.961** |
| same number, **different** prefix | **0.269** |
| different numbers (impostor baseline) | 0.319 (p99 0.677) |

Cross-prefix pairs score *below* the impostor baseline. `ISTL_0000202` and
`MILE_0000202` are two different people; each collection batch numbers its own
speakers from scratch. Evidence: `data/_qc/slr127_prefix_identity.json`.

**Consequences.** The speaker key is `<PREFIX>_<number>` → **638 speakers**, more
than the documented 531. The corpus has **no session metadata at all**, so it
falls back to one-session-per-utterance like SLR52 and SLR65, and its trial list
had to be regenerated: the old one drew 3,000 "targets" across prefixes, i.e.
pairs of *different people labelled as the same speaker*, which is why it scored
worse than chance.

**Why the fix is trustworthy.** Every label-quality metric moved in the direction
a correct key predicts, without any of them being the thing that was optimised:
d′ 3.14 → 5.23, NMI 0.963 → 0.984, silhouette 0.377 → 0.463, EER 55.31 % →
1.07 %. That is about as close to independent confirmation as one gets without
ground truth.

**Coincidence worth noting.** The "531 distinct values matches the documented 531
speakers" observation, which is what convinced me originally, was a red herring.
Counting agreement is not identity verification. Only audio can answer an
identity question.

### 1.2 Kathbath — recurring split ids are the same person

The methodology proposed this as a falsifiable test, and it came back positive.
11 numeric ids occur in both the `valid` and `test_known` splits. I had assumed
they might be different people and namespaced them apart (`kbv84`, `kbk84`).

| Comparison | Cosine |
|---|---|
| same number across splits (11 pairs) | **0.992** (max 0.996) |
| different ids (baseline) | 0.348 (p99 0.752) |

They are unambiguously the same person. Evidence:
`data/_qc/kathbath_cross_split_identity.json`.

**Consequence.** 22 speaker directories are 11 people, so the corpus is **49
people, not 60**. The directories keep their split tags — the official trial
lists reference those paths and must not be disturbed — but `train_list.txt`
labels are now merged, so fine-tuning no longer sees one person as two classes.
The unknown-speaker `test` split shares no ids with the others, consistent with
its design, so the official evaluation was never affected.

---

## 2. Results by layer

### L0 — integrity

Clean across all five: zero decode failures, zero near-silent files, zero
cross-speaker duplicates, uniform source sample rates per corpus, and effective
bandwidth at 100 % of Nyquist everywhere (no hidden upsampled narrowband audio).
SLR127 has 60 sampled files with >0.1 % clipping — worth a look, not a blocker.

### L1 — signal quality and the channel confound

Perceptual quality is fine everywhere (SQUIM, 500 files per corpus):

| Corpus | median SNR | STOI | PESQ | SI-SDR dB |
|---|---|---|---|---|
| slr65_tamil | 53.5 dB | 0.990 | 3.47 | 25.2 |
| nisp_tamil | 35.5 dB | 0.997 | 3.67 | 25.6 |
| slr52_sinhala | 27.2 dB | 0.979 | **2.52** | 21.6 |
| kathbath_tamil | 18.4 dB | 0.982 | 3.04 | 20.9 |
| slr127_tamil | 18.3 dB | 0.994 | 3.46 | 23.8 |

All are intelligible (STOI ≥ 0.98). SLR52's PESQ of 2.52 is the outlier, which
is what a volunteer-device corpus should look like; Kathbath's 3.04 partly
reflects lossy AAC.

**The ANOVA is the finding, not the means.** η² is the fraction of variance in a
per-utterance signal metric explained by speaker identity:

| Corpus | η²(SNR) | η²(RMS) | η²(bandwidth) |
|---|---|---|---|
| slr127_tamil | **0.91** | **0.93** | 0.53 |
| slr52_sinhala | 0.68 | **0.91** | 0.00 |
| kathbath_tamil | 0.52 | 0.72 | 0.22 |
| slr65_tamil | 0.44 | 0.71 | 0.16 |
| nisp_tamil | 0.37 | 0.53 | 0.00 |

At 0.91, **SNR is very nearly a speaker label in SLR127**: a model can score
recording conditions rather than voice. Loudness is worse still — η²(RMS) ≥ 0.9
on both large corpora.

Worth noting how this number moved: with the *wrong* SLR127 speaker key it read
0.74, and correcting the key pushed it to 0.91. That is the expected direction —
each collection batch has its own recording setup, so once speakers stop being
scrambled across batches, channel aligns with identity more tightly. The
confound was partly hidden by the bug.

### L3 — label reliability

| Corpus | d′ | NMI | ARI | purity | silhouette | LOO < 0.3 |
|---|---|---|---|---|---|---|
| slr127_tamil | **5.23** | **0.984** | 0.796 | 0.918 | 0.463 | 0.20 % |
| nisp_tamil | 5.61 | 0.918 | 0.655 | 0.555 | 0.278 | 0.02 % |
| slr65_tamil | 4.88 | 0.950 | 0.777 | 0.801 | 0.485 | 0.08 % |
| slr52_sinhala | 3.73 | 0.938 | 0.513 | 0.744 | 0.261 | 0.35 % |
| kathbath_tamil | 3.07 | 0.901 | 0.652 | 0.661 | 0.164 | 0.14 % |

All five clear the "no gross label corruption" bar. **slr52_sinhala is the one
to watch**: NMI 0.938 with ARI 0.513 and threshold-clustering finding ~1.96×
as many groups as labels. For a corpus whose speaker ids are anonymised
crowdsourcing accounts with no guarantee that one account is one person, that is
the predicted failure mode and it has not been ruled out. The mislabel
shortlists (`data/_qc/label_audit/*_shortlist.json`) are the next step, and they
require listening — no metric closes this.

**nisp_tamil's 0.55 clusters-per-label is a feature, not a fault.** Its 130
labels are 65 people × 2 languages, and clustering merges each person's Tamil and
English recordings back together. The embedding is language-robust for these
speakers, which is exactly what the corpus is there to test.

### L4 — trial-list audit

Our own generated lists are clean: zero self-pairs, zero same-session targets,
zero impostor pairs that secretly share a speaker, and same-gender impostor
fractions of 1.00 wherever gender is known.

**Kathbath's official lists are not.** 351 of its 24,984 target trials in the
unknown-speaker split pair a file **with itself** (177 in test_known, 163 in
valid). A self-pair scores cosine 1.0 by construction and can never contribute a
miss, so the published EER is biased slightly low — about 1.4 % of targets are
free.

This is a defect in the AI4Bharat protocol, not in our preparation: the lists are
translated verbatim. They are deliberately left untouched, because altering them
would break comparability with every published IndicSUPERB number, which is the
only reason to use the corpus. The right response is to quote Kathbath EERs with
the caveat attached, which the per-corpus report now does automatically.

Its same-gender impostor fraction is also only 0.47–0.48, i.e. the official lists
are roughly half cross-gender — an easier condition than our own gender-matched
lists. One more reason its absolute EER is not comparable with the others here.

### L5 — difficulty and power

| Corpus / list | EER % | 95 % CI | speakers | MDE (ρ=0.7) |
|---|---|---|---|---|
| slr127_tamil / test_list | **1.07** | 0.83 – 1.29 | **638** | **0.26** |
| slr127_tamil / test_list_ta | 1.37 | 1.08 – 1.66 | 638 | 0.33 |
| slr52_sinhala / test_list | 4.47 | 3.86 – 5.07 | 478 | 0.64 |
| slr52_sinhala / test_list_si | 4.80 | 4.21 – 5.44 | 478 | 0.68 |
| nisp_tamil / test_list_ta | 1.10 | 0.59 – 1.81 | 65 | 0.73 |
| nisp_tamil / test_list | 1.45 | 0.76 – 2.16 | 65 | 0.77 |
| nisp_tamil / **test_list_cs** | **2.90** | 2.03 – 3.84 | 65 | 1.00 |
| nisp_tamil / test_list_en | 1.80 | 0.69 – 3.03 | 65 | 1.31 |
| slr65_tamil / test_list_ta | 2.83 | 1.80 – 3.71 | 49 | 1.04 |
| slr65_tamil / test_list | 2.67 | 1.67 – 3.84 | 49 | 1.19 |
| kathbath_tamil / test_list_test | 6.37 | 3.08 – 10.90 | 20 | 4.52 |
| kathbath_tamil / test_list | 6.37 | 2.88 – 11.14 | 20 | 4.65 |
| kathbath_tamil / test_list_valid | 8.96 | 4.43 – 13.56 | 20 | 5.07 |
| kathbath_tamil / test_list_test_known | 10.51 | 4.72 – 16.21 | 20 | 6.22 |

**Trials do not buy power; speakers do.** Kathbath has 50,000 official trials and
lands at MDE 4.5–6.2 pp on 20 speakers. SLR127 reaches 0.26 pp from 12,000 trials
over 638 speakers — a 17× better resolution from a quarter of the trials.
Re-sampling more pairs from the same voices re-uses information already in the
corpus; it does not add any. This is the concrete demonstration of why the
methodology insisted on a *speaker-clustered* bootstrap.

**Cross-lingual penalty, measured.** NISP's same-language Tamil EER is 1.10 %;
the same speakers across Tamil↔English give 2.90 %. A **2.6× degradation** from
the language switch alone, on identical speakers and one embedding. It is a
small corpus (MDE 1.00 pp), so treat the ratio as directional — but the direction
is unambiguous and it is the first cross-lingual number this project has.

### L5b — calibration (Cllr)

Scores are cosine similarities, so they are
not log-likelihood ratios; a 1-D logistic map is fitted before scoring, and
minCllr comes from a PAV fit.

| Corpus / list | Cllr | minCllr | calibration loss |
|---|---|---|---|
| slr127_tamil / test_list | **0.0512** | 0.0438 | 0.0074 |
| nisp_tamil / test_list_ta | 0.0476 | 0.0407 | 0.0068 |
| nisp_tamil / test_list_cs | 0.1134 | 0.1020 | 0.0114 |
| slr65_tamil / test_list | 0.1105 | 0.1061 | 0.0044 |
| slr52_sinhala / test_list | 0.1893 | 0.1783 | 0.0110 |
| kathbath_tamil / test_list | 0.2573 | 0.2492 | 0.0081 |

Calibration loss is small everywhere (≤ 0.017), so almost all of the cost is
*discrimination* loss, not miscalibration — the ranking of corpora by Cllr
matches their ranking by EER. Both numbers are optimistic: the calibration is
fitted on the very trials it is scored against, so treat them as floors.

### L6 — subgroup fairness and cross-corpus shift

**Fairness, where it can be measured at all.** Only three corpora carry gender
labels. FDR at the pooled EER threshold:

| Corpus / list | FNMR (f / m) | FMR (f / m) | FDR |
|---|---|---|---|
| slr65_tamil / test_list_ta | 0.030 / 0.026 | 0.033 / 0.022 | 0.992 |
| nisp_tamil / test_list_ta | 0.003 / 0.016 | 0.014 / 0.010 | 0.991 |
| nisp_tamil / test_list | 0.004 / 0.021 | 0.025 / 0.010 | 0.984 |
| nisp_tamil / test_list_cs | 0.005 / 0.044 | 0.050 / 0.021 | 0.966 |

FDR ≥ 0.97 everywhere: no gender disparity of the magnitude Hutiri & Ding found
in the challenge data they audited. The worst case is NISP's cross-lingual list,
where female speakers see a much lower miss rate and a much higher false-alarm
rate than male — i.e. the operating point sits differently for the two groups,
which a single global threshold cannot serve equally.

**`slr52_sinhala` and `slr127_tamil` ship no gender labels, so subgroup parity
cannot be measured for them at all** — and those are the two largest corpora,
carrying the most weight in any pooled result. That is a real limitation of the
collection, not of the method.

**Cross-corpus shift.** Fréchet distances between embedding distributions run
0.20–0.35, and centroid cosines 0.69–0.92; the most distant pair is
slr52_sinhala ↔ slr65_tamil (cos 0.689), which is also the only cross-language
pair among the extremes.

**The channel probe is the headline.** A multinomial logistic regression predicts
**corpus-of-origin from the speaker embedding with 91.2 % accuracy, against 20 %
chance.** The embedding space encodes recording channel about as strongly as it
encodes anything corpus-independent. Combined with η²(SNR|speaker) of 0.68–0.74,
this is the strongest evidence in the audit that **a single corpus's absolute EER
should never be quoted as a performance claim**, and that cross-corpus
comparisons are partly comparisons of microphones.

---

## 3. What this changes

1. **The ablation can proceed, narrowly, on `slr127_tamil`.** MDE 0.26 pp means
   effects of ~0.3 pp and up are rankable. A wide grid of 0.1–0.2 pp variations
   is not, on any corpus here.
2. **Never report a single-corpus absolute EER as a performance claim.** With
   η²(SNR|speaker) at 0.68–0.74, part of what is being measured is the recording
   channel.
3. **Kathbath stays as the comparability anchor and nothing more.** Its official
   protocol is worth keeping for cross-paper comparison; its power is not usable
   for ranking.
4. **`slr52_sinhala` needs a listening pass** before its Sinhala numbers carry
   weight. Start with `data/_qc/label_audit/slr52_sinhala_shortlist.json`.
5. **Cross-corpus results need a channel caveat attached.** With 91.2 % probe
   accuracy, any "trained on A, tested on B" number is partly measuring the
   recording setup. The Indian-Tamil → Sri-Lankan-Tamil experiment this project
   is built around must report this alongside the dialect gap, or the gap will be
   over-attributed to dialect.
6. **Two of the five corpora cannot support any fairness claim** — no gender
   labels in slr52_sinhala or slr127_tamil. Say so rather than reporting pooled
   numbers as if parity had been checked.
7. **Any future corpus gets the audio-based identity test before it is trusted.**
   Both bugs here passed every metadata-level check and were only caught by
   embedding the speakers. That test is now `tools/label_audit.py` plus the two
   ad-hoc scripts recorded in `data/_qc/`.

## 4. Tooling added

| Tool | Layer | Notes |
|---|---|---|
| `tools/dataset_qc.py` | L0, L2, L4 | CPU only; writes `metadata/qc_report.json` per corpus |
| `tools/signal_quality.py` | L1 | WADA-style SNR, clipping, VAD, SQUIM; ANOVA confound test |
| `tools/extract_embeddings.py` | L3 input | ECAPA, per-speaker capped subsample, cached `.npz` |
| `tools/label_audit.py` | L3 | LOO centroids, silhouette, AHC vs labels, shortlists |
| `tools/eer_power.py` | L5 | EER/minDCF, speaker-clustered bootstrap, MDE |
| `tools/projections.py` | 2-D | t-SNE + UMAP on a shared speaker subsample |
| `tools/tsne_explorer.py` | GUI | `streamlit run …` — interactive t-SNE |
| `tools/umap_explorer.py` | GUI | `streamlit run …` — interactive UMAP |
| `tools/bias_transfer.py` | L6 | subgroup FNMR/FMR, FDR, Fréchet shift, channel probe |
| `tools/calibration.py` | L5 | Cllr via logistic calibration, minCllr via PAV |
| `tools/build_qc_reports.py` | reports | self-contained HTML, inline SVG figures and formulas |
| `tools/qc_report_model.py` | reports | one block model rendered to HTML, MD and PDF |
| `tools/build_report_package.py` | reports | writes data/datasets_reports/ with figures as files |
| `tools/noderun.sh` | infra | GPU dispatch, see §5 |

## 5. Compute notes

Both GPU nodes were used, matched to what each can actually reach:

| Node | GPU | `/mnt/ricproject3` | Used for |
|---|---|---|---|
| 10.222.1.120 (compute-node-3) | 2 × A10 23 GB | **mounted** | all audio-touching work: embedding extraction, trial scoring |
| 10.222.1.121 (compute-node-4) | 1 × A40 46 GB | **not mounted** | embedding-space compute only, staged via the shared `$HOME` |

`compute-node-4`'s driver fault from earlier in 2026 is **resolved** — its A40 is
the largest GPU available here — but it has no NFS mount for the project and no
passwordless sudo to add one. `/home` *is* shared from the same server, which is
the workaround. `tools/noderun.sh` encodes this split.

Two performance bugs found and fixed while running: `np.load` returns a lazy
`NpzFile`, so `z["emb"][i]` inside a comprehension re-decompresses the entire
array on every lookup (this turned a 10-second bootstrap into a 15-minute one);
and a `pgrep -f` wait-loop matched its own command line and never exited.

---

## 6. Addendum — SLCeleb 2026 V3 (Sinhala), added the same day

`/mnt/ricproject3/node5/SLCeleb_v3/webapp/data/SLCeleb_2026_V3` was put through
the identical Layer 0–6 pipeline. It changes the picture, because it is the only
corpus in the collection with **genuine recording sessions**.

| | |
|---|---|
| Speakers | 123 |
| Utterances / hours | 46,960 / 85.1 |
| Sessions (YouTube videos) | 400; **100 of 123 speakers span 2+** |
| EER | **10.99 %** (CI 9.37–13.10), MDE 2.08 pp |
| d' / NMI / silhouette | 2.91 / 0.852 / 0.304 |
| η²(SNR \| speaker) | **0.38** — the lowest here |
| Integrity | **628 of 47,588 source files (1.32 %) fail to decode** |

It ships already at 16 kHz mono PCM_16 in VoxCeleb layout, so `wav/si` is a
symlink and no audio was copied or converted.

### 6.1 The same-recording confound, measured directly

This is the result the whole methodology was designed around, and this corpus is
the only one that can produce it:

| Pair type | Mean cosine |
|---|---|
| Same speaker, **same** session | 0.612 |
| Same speaker, **different** session | 0.474 |
| Different speakers | 0.136 |

Allowing same-session targets gives **d' = 3.46**; restricting to cross-session
gives **d' = 2.44**. The confound inflates apparent separability by **42 %**, on
identical speakers and one embedding.

Every other corpus here maps one session per utterance — the same-sitting
condition — so their separability and their EERs carry an optimism of about this
size. That is the concrete, measured reason SLCeleb's 10.99 % and SLR127's
1.07 % are not comparable numbers, and why no single-corpus absolute EER should
be quoted as a performance claim.

### 6.2 Why its label metrics look "worse"

d' 2.91 and NMI 0.852 are the lowest in the collection and that is **expected,
not a defect**: the methodology set a separate gate for in-the-wild corpora
(NMI > 0.80) precisely because channel variation across sessions legitimately
lowers within-speaker similarity. The session analysis above confirms the cause
is real acoustic variation rather than label noise — within-session similarity is
0.612, comfortably above the 0.136 impostor floor.

Its η²(SNR | speaker) of 0.38 points the same way: because each speaker spans
several recordings, channel does **not** align with identity, unlike SLR127
(0.91) and SLR52 (0.68). Real sessions break the channel confound.

### 6.3 Actions

- **628 corrupt files should be fixed upstream** in the YouTube→segment
  pipeline. 1.32 % threatens no result, but it signals dropped writes and may
  not be uniform across speakers. They are listed in
  `data/slceleb2026_sinhala/metadata/unreadable_files.json` and excluded from
  every list.
- **Decide train-vs-held-out now.** At 85 h and 123 speakers it is large enough
  to be tempting for both, and using it for both destroys the dialect-gap claim.
- **No gender labels**, so its impostors are not gender-matched and subgroup
  fairness cannot be assessed — the same limitation as SLR52 and SLR127.
- **Sinhala only.** The Sri-Lankan-*Tamil* half of the dialect story still has no
  reference set.
