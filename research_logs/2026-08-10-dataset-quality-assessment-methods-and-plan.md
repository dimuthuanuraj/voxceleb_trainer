# Dataset Quality Assessment for Speaker Verification — Methods and Test Plan

**Date:** 2026-08-10 · **Status:** methodology + plan, nothing executed yet
**Scope:** the eight corpora in `data/` (see `data/README.md`)
**Companion:** `2026-08-10-open-si-ta-sv-corpora.md` (which corpora, and why)

That companion log answered *which* datasets to use. This one answers *whether
they are any good* — and, critically, whether they are good enough to support the
feature ablation we actually want to run.

---

## Part I — What "quality" means for speaker verification

The instinct is to reach for audio-quality metrics. For SV that is the least
important axis. A corpus of clean studio speech can be worthless for
verification, and a corpus of noisy YouTube audio can be excellent. Four things
actually decide whether an SV dataset is sound, roughly in order of how badly
they bite:

| # | Failure mode | What it looks like | Why it is fatal |
|---|---|---|---|
| **1** | **Label noise** | two people sharing one speaker id, or one person split across two ids | Corrupts targets *and* impostors. A mislabelled impostor pair that is really a target pair puts a hard ceiling on measurable EER. VoxCeleb2 label noise is well documented; removing noisy samples alone bought ~5.9 % relative improvement in one study. |
| **2** | **Session / channel confound** | target pairs cut from the same recording | The model scores channel, not speaker. 8–17 % of VoxCeleb1-H target pairs have this problem (Hutiri et al., Interspeech 2022). EER looks great and means nothing. |
| **3** | **Insufficient statistical power** | 20–50 speakers, however many trials | Confidence intervals wider than the effects being ranked. **This is the one that decides whether our ablation is worth running at all.** |
| **4** | **Distribution shift / bias** | trained on Indian Tamil, deployed on Sri Lankan Tamil | Headline EER hides subgroup failure. Female error rates 49 % higher than male in one challenge-wide audit (Hutiri & Ding, FAccT 2022). |

Audio quality matters mainly as a *confound detector*: if corpus A is uniformly
cleaner than corpus B, an embedding can separate them trivially, and any
"cross-corpus generalisation" result is measuring codec, not voice.

So the assessment is organised as six layers, cheapest and most decisive first.
**Layers 0–2 need no GPU and no model.** Layer 3 is where the real answers are.

---

## Part II — The methods, layer by layer

### Layer 0 — Integrity and inventory

*Question: is the data physically what the metadata claims?*

| Check | Method | Red flag |
|---|---|---|
| Decodability | `soundfile.read` every file | any failure |
| Format uniformity | sample rate, channels, bit depth, subtype histogram | more than one value per corpus |
| Zero/near-silent files | RMS below −60 dBFS over whole file | > 0.1 % of files |
| Exact duplicates | MD5 over *decoded* PCM, not the container | any cross-speaker duplicate |
| **Effective bandwidth** | mean power spectrum per file; find the rolloff where energy falls below −50 dB relative to peak | rolloff ≪ Nyquist ⇒ **upsampled narrowband audio masquerading as 16 kHz** |

The bandwidth check is the one people skip and regret. A corpus resampled up
from 8 kHz telephone audio still has a `16000` in its header; the model sees a
brick wall at 4 kHz and learns "this is corpus X". `slr65_tamil` was resampled
48 → 16 kHz here, so its true content ceiling is 8 kHz by construction — that is
expected and fine. What we are hunting is *unexpected* band-limiting, and any
corpus that is internally inconsistent.

Existing tooling: `tools/check_wav_integrity.py` covers decodability. The rest
needs a new script.

### Layer 1 — Signal quality, non-intrusively

*Question: how clean is the audio, and is cleanliness confounded with speaker?*

No clean reference exists for any of these corpora, so all metrics must be
**reference-free**:

| Metric | Tool | Note |
|---|---|---|
| SNR | WADA-SNR, or NIST STNR | classic, cheap, no model |
| Clipping rate | fraction of samples with \|x\| > 0.99 | > 0.1 % is audible damage |
| Speech ratio | energy or Silero VAD | < 0.5 means half the file is silence |
| Loudness / dynamic range | LUFS (ITU-R BS.1770) | detects per-corpus normalisation differences |
| **STOI / PESQ / SI-SDR (estimated)** | **`torchaudio.pipelines.SQUIM_OBJECTIVE`** | verified available in this env |
| **MOS (estimated)** | **`SQUIM_SUBJECTIVE`** (NORESQA-MOS) | needs a non-matching reference clip |

DNSMOS and NISQA are the other standard options; SQUIM is chosen here purely
because it ships inside the already-installed `torchaudio` 2.8, so it adds no
dependency. NORESQA-MOS (what SQUIM_SUBJECTIVE uses) generalises better than
direct-estimation MOS predictors like DNSMOS/NISQA.

**The critical analysis is not the mean — it is the *variance within a
speaker*.** If a speaker's utterances all share one SNR and one loudness, and
different speakers differ, then SNR *is* speaker id, and the model can cheat.
Compute the between-speaker vs within-speaker variance ratio for each signal
metric (a one-way ANOVA F-statistic works). A high F on SNR is a warning that
target trials are partly solvable from channel alone.

### Layer 2 — Distribution and coverage

*Question: is the speaker population shaped so that a meaningful trial list can
even be drawn?*

- Speakers; utterances per speaker (min / median / max); **Gini coefficient** of
  utterances per speaker (imbalance in one number).
- **Duration histogram, and the fraction below 2 s and 4 s.** SV accuracy
  collapses on short utterances; a corpus whose median is 2.5 s will report a
  bad EER for reasons that have nothing to do with the speakers.
- Sessions per speaker, and how many speakers have ≥ 2 (this decides how many
  can supply cross-session targets at all).
- Gender balance, and gender × utterance-count (is one gender systematically
  shorter?).
- Duration × speaker correlation — a confound if a handful of speakers own all
  the long files.

All of this is already computable from `metadata/utterances.csv`, which every
dataset folder carries. **Layer 2 costs minutes and needs no audio.**

### Layer 3 — Label reliability via speaker embeddings

*Question: are the speaker labels actually correct?*

This is the layer that finds real problems, and it is the reason to own a
pretrained model. Procedure:

1. **Extract one embedding per utterance** with a strong pretrained model, ideally
   two architecturally different ones so findings are not model artefacts.
   `tools/zeroshot_eval.py` already does extraction with caching and supports
   `speechbrain_ecapa`, `redimnet:b0…b6` and `wespeaker:*`.
2. **Within- vs between-speaker similarity.** Plot the two cosine distributions.
   Healthy corpora show clear separation; heavy overlap means either hard data or
   broken labels, and layers 4–5 cannot tell you which.
3. **Leave-one-out centroid outliers.** For each utterance, cosine to its own
   speaker's centroid computed *without* it. Rank ascending. The bottom tail is
   your mislabel candidate list. **Then actually listen to the top ~30.** This
   step is not automatable and is the only way to convert a suspicion into a fact.
4. **Per-speaker silhouette score.** Speakers with low silhouette are either
   acoustically unremarkable or contain two people. Sort ascending, inspect.
5. **Clustering vs labels.** Agglomerative clustering (cosine, average linkage)
   over embeddings, then compare the partition against the given labels with
   **NMI, ARI, V-measure and cluster purity**. Two diagnostics fall out:
   - clusters ≫ speakers → one person split across ids, or heavy session effects
   - clusters ≪ speakers → distinct ids collapsing, i.e. probable duplicates
6. **Visual check.** t-SNE or UMAP over a speaker subsample. `sklearn` has
   `TSNE` already; `umap-learn` is not installed and is optional.

Literature to follow for the automated part: inconsistency-ranking noisy-label
detection (arXiv:2212.00239) and CEC (arXiv:2406.13268), both designed exactly
for speaker-recognition label noise.

**Expected shape of the answer, per corpus type.** Read speech recorded in one
sitting should cluster almost perfectly (NMI > 0.95) — anything less is a real
label problem. In-the-wild corpora legitimately score lower, so the same number
means different things and thresholds must be set per corpus type, not globally.

### Layer 4 — Protocol and trial-list audit

*Question: is the benchmark we built out of this corpus honest?*

A trial list can be wrong in ways no model will ever reveal:

| Check | Requirement |
|---|---|
| Target / impostor counts and ratio | matches the design |
| Self-pairs (`a == b`) | zero |
| Duplicate unordered pairs | zero |
| **Same-(speaker, session) target pairs** | zero, wherever sessions are real |
| Same-gender impostor fraction | 1.0 where gender is known; documented otherwise |
| **Speaker overlap between `train_list` and `test_list`** | zero, or explicitly declared |
| Utterance reuse frequency | no small set of files dominating trials |
| Enrol/test duration matching | no systematic asymmetry |

The train/test speaker-overlap check becomes essential the moment we start
*combining* corpora — training on SLR127 + Kathbath and evaluating on a merged
list is exactly where leakage sneaks in unnoticed.

Note the subtlety already found in this repo: comparing session *directory
names* is wrong when session names are shared across speakers (SLR127 uses
`ISTL`/`MICI`/`MILE`). The key must be the `(speaker, session)` pair.

### Layer 5 — Difficulty calibration and statistical power

*Question: how hard is this benchmark, and can it detect the effects we care
about?*

**Difficulty.** Run 2–3 pretrained models zero-shot and report EER_avg, EER_max,
minDCF at p_target 0.01 and 0.05 — all of which `tools/zeroshot_eval.py` already
computes. Express results as a ratio against the same models' VoxCeleb-O numbers
to get a corpus-independent difficulty index. Also report **Cllr and min Cllr**
(BOSARIS-style) — the gap between them is miscalibration, which the earlier
finding about cross-language calibrator swaps (Cllr inflated up to 5.7×) makes
directly relevant here.

**Power — the decisive analysis.** EER confidence intervals must come from a
**speaker-clustered bootstrap**: resample *speakers* with replacement (not
trials), rebuild the trial subset, recompute EER, repeat B = 1000, take the
2.5 / 97.5 percentiles. Resampling trials independently is wrong here because
trials from one speaker are correlated, and doing it that way understates the
interval — often badly, when speakers are few.

From that, derive the **minimum detectable effect (MDE)**: the smallest EER
difference this corpus can resolve at 80 % power, α = 0.05, under a paired
comparison of two systems on identical trials. Then state plainly, per corpus,
whether the ablation deltas we expect (~0.2–0.5 % EER) exceed it.

This turns "50 speakers feels too few" into a number, and it is the single most
useful output of the whole exercise.

### Layer 6 — Bias and cross-corpus transfer

*Question: whose voices does this work for, and does it survive a domain change?*

- **Subgroup EER** by gender, and by dialect where known. Report per-subgroup
  FMR/FNMR at a shared threshold, not just per-subgroup EER — a single global
  operating point is what a deployed system actually uses.
- **Fairness Discrepancy Rate (FDR)** for a one-number summary, following the
  bias-quantification framework of Hutiri & Ding (FAccT 2022).
- **Cross-corpus matrix.** Train on A, evaluate on B, for every ordered pair.
  The Indian-Tamil-train / Sri-Lankan-Tamil-test cell is the number this whole
  project is positioned to publish.
- **Channel probe.** Train a logistic regression on embeddings to predict
  corpus-of-origin. If it succeeds near-perfectly, the embedding space encodes
  channel as much as identity, and cross-corpus numbers must be read with that
  in mind.

### Reporting standard

Findings per corpus should land in its existing `metadata.json` and `README.md`
rather than a separate report — the datasheet lives with the data. The framing
follows **Datasheets for Datasets** (Gebru et al., CACM 2021) and **Data
Statements** (Bender & Friedman, TACL 2018): provenance, composition, collection
process, preprocessing, uses, distribution, maintenance. Our `metadata.json`
schema already covers provenance, licence, session semantics and caveats; what
Layers 0–6 add is the *measured* half.

---

## Part III — Structured test plan

### Phasing

Ordered so that the cheapest checks can kill a corpus before expensive ones run.

| Phase | Layers | Compute | Wall-clock | Gate |
|---|---|---|---|---|
| **P0** | L0 + L2 | CPU, metadata only | ~2 h | integrity clean, distribution documented |
| **P1** | L4 | CPU, lists only | ~2 h | no leakage, no same-session targets |
| **P2** | L1 | CPU (SQUIM on GPU is faster) | ~1 day | quality/speaker confound quantified |
| **P3** | L3 | **GPU** | ~2 days | label noise bounded, listened-to sample |
| **P4** | L5 | GPU (reuses P3 embeddings) | ~1 day | **MDE per corpus → go/no-go for the ablation** |
| **P5** | L6 | GPU | ~2 days | subgroup + cross-corpus gaps reported |

P0 and P1 need nothing that is not already on this node. P3 onward needs the
GPU offload path (`tools/gpurun.sh`; this node has no GPU). Embeddings extracted
in P3 are cached and reused by P4 and P5 — extract once.

### Per-dataset plan

Every corpus gets L0/L2/L4. The rest is targeted at each corpus's specific known
risk, so effort goes where it can actually change a decision.

#### `slr52_sinhala` — 478 spk, 185k utts, 224.5 h
- **Priority: HIGH** — the entire Sinhala side of the project rests on it.
- Specific risks: speaker ids are anonymised crowdsourcing accounts with **no
  verification that one account is one person**; no gender labels; single sitting.
- Run: full L0; L2 with attention to the duration tail; **L3 in full** — this is
  the corpus where a clustering-vs-labels check is most valuable, because nothing
  upstream guarantees account = person. Expect NMI > 0.95; investigate hard if not.
- L1 on a 5 k-utterance random sample (185 k is unnecessary for a distribution).
- Red flag that would change plans: clusters ≫ speakers, implying accounts shared
  or reused.

#### `slr127_tamil` — 531 spk, 89k utts, 150.1 h
- **Priority: HIGH** — largest Tamil pool and the only read-speech corpus with
  real sessions.
- Specific risks: the speaker-id convention was **derived by us, not documented
  upstream**; the prefix-as-session interpretation is an inference.
- Run: full L0 including the **bandwidth check** (three collection batches may
  differ in equipment); L2; **L3 with the clustering diagnostic split by prefix**
  — this directly tests the session hypothesis: if `ISTL` and `MILE` recordings
  of one speaker cluster apart, the prefix is a genuine channel change, which
  both validates the session mapping *and* quantifies the channel effect.
- L1 **stratified by prefix** — a between-prefix SNR/loudness difference is the
  physical evidence for the session claim.
- This corpus deserves the most careful L3 because our confidence in it rests on
  an inference rather than documentation.

#### `kathbath_tamil` — 60 spk, 7k utts, 13.2 h
- **Priority: MEDIUM-HIGH** — the comparability anchor.
- Specific risks: **lossy AAC**; only 20 speakers per eval split; 11 numeric ids
  recur across splits (namespaced apart, but the assumption that they are
  different people is unverified).
- Run: full L0/L2/L4; **L3 with one extra test — check whether same-numeric-id
  speakers across splits are in fact the same person.** Embed `kbv84`, `kbk84`,
  `kbt84`; if cosine similarity is high, they *are* one person and our
  namespacing has silently created three identities for one speaker, which would
  make cross-split analysis wrong. This is a concrete, falsifiable question worth
  answering early.
- L5 is important here: 20 speakers with 50 k trials is the textbook case of
  trial count overstating power.
- Do **not** modify the trial lists as a result of any finding — document instead.

#### `nisp_tamil` — 65 spk (all bilingual), 4.9k utts, 13.3 h
- **Priority: MEDIUM** — small, but the only cross-lingual probe.
- Specific risks: tiny; English is accented, not a native control.
- Run: L0/L2/L4; **L3 focused on the cross-lingual question** — compare
  within-speaker-within-language, within-speaker-cross-language, and
  between-speaker similarity distributions. The gap between the first two *is*
  the language effect on the embedding, measured directly, with no trial list
  needed.
- L6 is unusually feasible here because demographics (age, height, region) ship
  with the corpus — a rare chance to test subgroup effects beyond gender.
- L5 will likely show it is underpowered for ranking systems; that is fine, its
  job is a directional probe.

#### `slr65_tamil` — 50 spk, 4.3k utts, 7.1 h
- **Priority: LOW-MEDIUM** — kept for v0 continuity only.
- Run: L0/L2/L4 for completeness; L3 cheap (small); **L5 to put a number on how
  underpowered v0 was.** That number retroactively frames every v0 result and
  belongs in the thesis.
- Not worth deep L1/L6 investment.

#### `commonvoice_tamil` — not yet fetched
- **Priority: MEDIUM**, contingent on the manual download.
- Specific risks: `client_id` is an *account*, not verifiably a person — the
  single highest label-noise risk in the collection; wildly heterogeneous devices.
- Run when available: L0 with **bandwidth check as a first-class concern**
  (contributors record on anything); L1 expecting high variance — and here high
  within-corpus variance is a *feature*, since device diversity is why we want it;
  **L3 mandatory before any use**, with the account-vs-person question explicit.
- L2 should drive the `--min-utts` choice empirically rather than the current
  default of 4.

#### `sita_sinhala_tamil` — not yet fetched
- **Priority: MEDIUM** for diarization and domain probing; **not** an SV benchmark.
- Run: L0/L1/L2 only. **L4 and L5 do not apply** — there is no trial list, by
  design.
- One bespoke analysis worth doing: **cross-recording identity linking.** Embed
  all segments, cluster across recordings, and see whether the same public figures
  recur. If they do, that is the raw material for the first genuine Sri Lankan SV
  benchmark, and this measurement tells you whether the manual effort is worth it.
  That is a research contribution, not a QA step.

#### `slceleb_sinhala_tamil` — audio not on mounts
- **Priority: HIGHEST once available** — it is the reference set, and the only
  corpus whose absolute EER means what a reader will assume.
- Run: **all six layers, in full.** Nothing else in the collection carries as much
  weight per speaker.
- Specific risks: YouTube-derived pipelines are exactly where label noise lives
  (VoxCeleb's own lineage proves it), and unlike VoxCeleb1 this corpus has **no
  published cleaning pass**. L3 here is not optional.
- L6 cross-corpus matrix against SLR127/Kathbath is the headline experiment.

### Acceptance gates

Proposed thresholds — to be revised once P0/P2 show what the corpora actually
look like, and split by corpus type because read speech and in-the-wild are not
comparable:

| Gate | Read-speech corpora | In-the-wild corpora |
|---|---|---|
| Decode failures | 0 | 0 |
| Cross-speaker exact duplicates | 0 | 0 |
| Utterances < 2 s | < 5 % | < 10 % |
| Label NMI (AHC vs labels) | > 0.95 | > 0.80 |
| Centroid-outlier tail flagged for listening | bottom 0.5 % | bottom 1 % |
| Same-(speaker, session) target pairs | 0 | 0 |
| Train/test speaker overlap | 0 | 0 |
| Same-gender impostor fraction | 1.0 where gender known | same |

A corpus failing a gate is not discarded — it is **documented and
down-weighted**, and the failure goes in its `metadata.json → caveats`.

### Deliverables

1. `tools/dataset_qc.py` — L0/L2/L4, pure CPU, emits `metadata/qc_report.json`
   per dataset plus a cross-corpus summary table.
2. `tools/signal_quality.py` — L1, WADA-SNR + clipping + VAD + SQUIM, sampled.
3. `tools/label_audit.py` — L3, consumes cached embeddings from
   `zeroshot_eval.py`, emits outlier lists, clustering metrics, and a
   listen-to-me shortlist of audio paths.
4. `tools/eer_power.py` — L5, speaker-clustered bootstrap CIs and MDE.
5. A results log, `research_logs/<date>-dataset-qc-results.md`, plus each
   corpus's `metadata.json` updated with a `quality` block.

### What would change the ablation plan

The point of all this is a single decision. Three outcomes are actionable:

- **If MDE on the merged Tamil set is below ~0.2 % EER**, the ablation is
  properly powered and can proceed as designed.
- **If MDE sits at 0.3–0.5 %**, only large feature effects are rankable; the
  ablation should be narrowed to fewer, bolder conditions rather than a wide grid.
- **If MDE exceeds 0.5 %**, no amount of clever features will produce a
  defensible ranking, and the honest move is more speakers before more
  experiments.

Layer 3 can also invalidate a corpus outright. Better to find that in P3 than in
peer review.

---

## Part IV — Environment notes

Verified on this host, 2026-08-10:

- `torch` 2.8.0+cu128, `torchaudio` 2.8.0 — **`SQUIM_OBJECTIVE` and
  `SQUIM_SUBJECTIVE` import cleanly**.
- `sklearn` 1.5.1 (TSNE, AgglomerativeClustering, silhouette, NMI/ARI all present).
- `speechbrain` present; `wespeaker`, `pyannote.audio`, `umap-learn` absent
  (all optional).
- **No GPU on this node** (`torch.cuda.is_available()` is False). L3–L6 must go
  through `tools/gpurun.sh`; NFS is mounted at the same path on the compute
  nodes, so no data movement is needed.
- Existing reusable tooling: `tools/zeroshot_eval.py` (embedding extraction with
  caching, EER/minDCF), `tools/check_wav_integrity.py`, `tools/sl_dataprep.py`.

---

## Part V — References

- Hutiri, Ding (2022). *Bias in Automated Speaker Recognition.* FAccT 2022. arXiv:2201.09486.
- Hutiri et al. (2022). Interspeech — same-recording confound in VoxCeleb trial pairs.
- Kumar et al. (2023). *TorchAudio-Squim: Reference-less Speech Quality and Intelligibility Measures.* arXiv:2304.01448.
- Gebru et al. (2021). *Datasheets for Datasets.* CACM 64(12).
- Bender, Friedman (2018). *Data Statements for NLP.* TACL 6.
- Tong et al. (2022). *Inconsistency Ranking-based Noisy Label Detection.* arXiv:2212.00239.
- *CEC: A Noisy Label Detection Method for Speaker Recognition.* arXiv:2406.13268.
- Fathan et al. (2025). *Automatic Labeling and Correction of Noisy Labels for Robust Self-Supervised Speaker Verification.* Interspeech 2025.
- Bisani, Ney (2004). *Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation.* ICASSP 2004.
- Chung et al. (2019). *VoxSRC 2019.* arXiv:1912.02522 — dev/test overlap analysis across VoxCeleb1/2/SITW.
- NIST SRE evaluation plans — the reference for trial-protocol and DCF reporting.
- BOSARIS toolkit — Cllr / min Cllr calibration metrics.
