# SSPS — Deep Analysis and Applicability to SL_SPV

**Paper:** Théo Lepage, Réda Dehak, *"Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling"*
EPITA Research Laboratory (LRE), France · arXiv:2501.17772v4 [eess.AS], 24 July 2025
**Code:** <https://github.com/theolepage/sslsv>
**Analysis date:** 2026-08-11 · **Analysed against:** `SL_SPV/voxceleb_trainer` @ FEATURE-011, `data/_qc` Layer 0–6 outputs

---

## 0. Verdict up front

**The paper's diagnosis applies to your project almost perfectly. Its prescription does not — not as published.**

SSPS exists to remove *recording/channel* information from speaker embeddings without labels. Your own Layer-5 QC already found that **corpus-of-origin is 91.6% linearly decodable from your speaker embeddings (chance 17%)**. You and Lepage & Dehak are attacking the same disease.

But SSPS's mechanism depends on a structural property of VoxCeleb2 that **SLCeleb does not have at the required strength**: many distinct recordings per speaker. VoxCeleb2 has 24.3 sessions/speaker; SLCeleb has 3.3. That single ratio governs whether the method can work, and it puts you in a viable-but-narrow regime rather than the comfortable one the paper operates in.

Three things here are worth acting on. Only one of them is SSPS itself, and it is the lowest-priority of the three.

| # | Action | Cost | Expected value |
|---|---|---|---|
| 1 | Cross-session positive sampling in `DatasetLoader` (supervised) | ~30 lines | **High** — this is the paper's Table I, 41–73% relative EER reduction |
| 2 | Use SLCeleb session IDs as channel labels for FEATURE-008 DANN | ~0 lines, config only | **High** — unblocks a feature you already built |
| 3 | Speaker/recording NMI ratio as QC Layer 7 | ~50 lines | Medium — cheap, publishable diagnostic |
| 4 | SSPS proper (SSPS-Clustering) | New training loop + GPU budget | **Low right now** — gated on corpus structure |

---

## 1. What the paper actually claims

### 1.1 The mechanism

Every SSL framework for SV builds an anchor–positive pair from **two segments of the same utterance**, then relies on data augmentation (MUSAN noise, RIR reverb) to force invariance. The paper's central claim is that this is a *fundamental* design flaw, not a tuning problem:

> Because anchor and positive come from the same recording, they share channel characteristics. The model can satisfy the SSL objective by encoding the recording, not the speaker.

The formal assumption (§III-B) is a similarity ordering. For representations `y₁` (speaker S, recording R), `y₂` (same S, same R, different utterance), `y₃` (same S, different recording), `y₄` (different speaker):

```
sim(y₁,y₂)  ≥  sim(y₁,y₃)  ≥  sim(y₁,y₄)
same recording   same speaker      different
                 diff. recording   speaker
```

**The latent space organises by recording first, speaker second.** SSPS exploits exactly this: if you sample a pseudo-positive from *slightly further away* than the nearest neighbours, you land on the same speaker but a different recording — which is precisely the pair that teaches channel invariance.

It is a bootstrapped method: sampling quality improves as representations improve, which improves sampling. No labels anywhere.

### 1.2 Machinery

- **Reference representations** `ŷᵢ` — computed from a *longer (4 s), non-augmented* segment. Deliberately not the anchor/positive representations, so sampling decisions are made on a clean view of the data.
- **Two memory queues** — `Q̂` (reference representations, size N) and `Q'` (positive embeddings, size N or K).
- **Activation schedule** — standard SSL for 100 epochs first, *then* SSPS for the last ~20. SSPS cannot bootstrap from noise; it needs a latent space that already groups by recording.

### 1.3 Two instantiations

**SSPS-NN** — sample uniformly from the M nearest neighbours. Tuned optimum M=50. Yields pseudo-positives with ~50% speaker accuracy and ~25% recording accuracy. The paper explicitly calls this **inadequate for SR**: neighbours are dominated by the same recording no matter how large M gets.

**SSPS-Clustering** — k-means over `Q̂` at the start of every epoch, K clusters. Sample from a *neighbouring* cluster (M>0), not the anchor's own cluster.

This is the part most people misread, so state it precisely:

| K | M | VoxCeleb1-O EER | Interpretation |
|---|---|---|---|
| 6,000 (≈ #speakers) | 0 (same cluster) | 2.90% | Clusters ≈ speakers, but positives often same-recording |
| **25,000** | **1** | **2.57%** | **Best.** Clusters are sub-speaker; neighbour = different recording |
| 150,000 (≈ #recordings) | — | worse | Clusters fragment below recording level |

The optimum is **K ≈ 4.2× the number of speakers**, *not* K = number of speakers and *not* K = number of recordings. Using cluster centroids instead of real samples ("SSPS-Clustering (C)") **fails** — the paper notes the embedding space is "dense but not continuous."

### 1.4 Headline results (ECAPA-TDNN, VoxCeleb1-O)

| Framework | SSL | **SSPS** | Supervised pos. sampling | Fully supervised |
|---|---|---|---|---|
| SimCLR | 6.30% | **2.57%** (−58%) | 1.72% | — |
| MoCo | 6.20% | — | 1.75% | — |
| SwAV | 7.97% | **6.50%** | 4.38% | — |
| VICReg | 7.70% | **6.95%** | 4.52% | — |
| DINO | 3.07% | **2.53%** | 2.36% | — |
| *AAM-Softmax baseline* | | | | **1.34%** |

### 1.5 The three findings that matter most

**(a) The "Supervised positive sampling" column is the real headline.** Merely using labels to pick positives *from different recordings of the same speaker* — no other change — cuts EER by 72.6% (SimCLR), 71.8% (MoCo), 45.1% (SwAV), 41.3% (VICReg), 23.1% (DINO). This quantifies the cost of same-utterance sampling independently of SSPS. **You have labels. This column is available to you today.**

**(b) SSPS nearly eliminates dependence on data augmentation.**

| | with aug. | without aug. |
|---|---|---|
| SSL | 6.30% | 15.00% |
| **SSPS** | **2.57%** | **2.77%** |

SSL collapses without augmentation; SSPS barely moves — and its minDCF actually *improves* (0.3033 → 0.2840). Positive sampling and augmentation are substitutes, and sampling is the better substitute.

**(c) The channel information is measurably reduced.** Speaker-to-recording NMI ratio rises 1.01 → 1.09 (~8% relative) while the SSL baseline stays flat. This is a direct, label-based measurement that the representations stopped encoding the microphone.

### 1.6 What the paper admits

From the Fig. 6 t-SNE discussion: SSPS fails to unify same-speaker sub-clusters when the SSL model has **already separated them due to strong acoustic variability** before SSPS activation. Once two recordings of one speaker are far apart, neighbouring-cluster sampling never bridges them.

**This failure mode is more likely on your data, not less** — your corpora are a patchwork of acoustically dissimilar sources (Fréchet distances 0.20–0.39 between corpus embedding distributions).

---

## 2. The structural test: does your data support SSPS?

This is the decisive section. I measured your corpora directly rather than assuming.

### 2.1 Corpus geometry, measured

```
data/SLCeleb_2026_V3  →  123 speakers · 403 sessions · 47,588 wavs · 85.07 h
```

| Quantity | VoxCeleb2 dev (paper) | SLCeleb 2026 V3 (si) | Ratio |
|---|---|---|---|
| Speakers | 5,994 | 123 | 49× fewer |
| Recordings (sessions/videos) | 145,569 | 403 | 361× fewer |
| Utterances | 1,092,009 | 47,588 | 23× fewer |
| **Sessions per speaker** | **24.3** | **3.3** (median 3, max 8) | **7.4× fewer** |
| **Utterances per session** | **7.5** | **118.1** | **15.7× more** |
| Utterances per speaker | 182 | 387 | 2.1× more |
| Single-session speakers | ~0 | **23 (19%)** | — |

Your SLCeleb layout is genuinely VoxCeleb-shaped (`ID00315/z7CLxAJUIj8/00027.wav` — speaker / YouTube video / utterance), which is exactly right. **The structure is correct; the density is wrong.** You have deep sessions and few of them; VoxCeleb has shallow sessions and many.

### 2.2 Why that ratio decides everything

SSPS-Clustering works only when a k-means cluster is **bigger than one recording but smaller than one speaker**. Formally, with cluster size `N/K`:

```
utts_per_recording  <  N/K  <  utts_per_speaker
```

Below the lower bound, clusters sit *inside* a recording and neighbouring-cluster sampling returns the same recording. Above the upper bound, clusters merge speakers and pseudo-positives are the wrong person.

| | utts/recording | cluster size (N/K) | utts/speaker | Recordings spanned per cluster |
|---|---|---|---|---|
| VoxCeleb2, K=25,000 | 7.5 | **43.7** | 182 | **≈ 5.8** ✓ comfortable |
| SLCeleb, K=250 | 118.1 | **190** | 387 | **≈ 1.6** ⚠ marginal |

**The window is not empty — but it is narrow, and the achievable channel diversity inside a cluster is ~3.6× lower than what made SSPS work.**

Valid range for SLCeleb: **K ∈ (123, 403)**, i.e. K ≈ 250 as a starting point.

Note how badly naive scaling of the paper's K fails:

| Scaling heuristic | Implied K | Verdict |
|---|---|---|
| Keep utts/cluster = 43.7 | 1,089 | ✗ clusters land inside recordings |
| Keep K = 4.2 × #speakers | 513 | ✗ still below one recording |
| Keep K = 0.17 × #recordings | 69 | ✗ clusters merge speakers |
| **Window argument** | **≈ 250** | ✓ only defensible choice |

**Do not port K=25,000. It is wrong by two orders of magnitude for your corpus.**

### 2.3 The ceiling, even with an oracle

Probability that a randomly chosen *same-speaker* utterance comes from a different recording:

- VoxCeleb2: (182.2 − 7.5) / (182.2 − 1) = **96.4%**
- SLCeleb: (386.9 − 118.1) / (386.9 − 1) = **69.7%**

That is the ceiling with perfect speaker labels and uniform sampling. SSPS samples by *latent proximity*, which biases hard toward same-recording, so the realised rate will be well below 70%. Meanwhile 19% of your speakers have exactly one session and can contribute **nothing** to this objective.

### 2.4 The other corpora are worse

I checked the on-disk structure of all six:

| Corpus | Speakers | Layout | Genuine sessions? |
|---|---|---|---|
| slceleb2026_sinhala | 123 | `spk/<video_id>/*.wav` | **Yes** — median 3 |
| slr52_sinhala | 478 | `spk/<utt_id>/00001.wav` | **No** — one dir per utterance |
| slr127_tamil | 638 | `spk/<utt_id>/00001.wav` | **No** |
| kathbath_tamil | 49 | `spk/<utt_id>/00001.wav` | **No** |
| nisp_tamil | 65 | `spk/<utt_id>/00001.wav` | **No** (but see below) |
| slr65_tamil | 50 | `spk/<utt_id>/00001.wav` | **No** |

Your OpenSLR ingest flattened everything to one pseudo-session per utterance. For read-speech corpora (SLR52/65/127) this is honest — they are single-sitting studio recordings, so there is no session diversity to lose. **SSPS has nothing to exploit on five of your six corpora.**

> **One recoverable exception: NISP.** NISP is a genuine *multi-device* corpus — the same speaker recorded simultaneously across several microphones. That is textbook channel variation with the speaker held fixed. `tools/ingest_openslr.py` discarded the device grouping. Recovering device IDs from NISP filenames would give you real channel labels on 65 speakers — useful for FEATURE-008 and for *validating* any channel-invariance claim, even though 65 speakers is too small to train on.

---

## 3. Will it improve your Sinhala/Tamil scenario?

Split honestly into what transfers, what does not, and what is better than SSPS.

### 3.1 What will NOT help

**SSPS will not beat your current numbers.** Your P3 full fine-tune achieves **1.63% EER (si) / 1.30% EER (ta)**. SSPS's best result is 2.53–2.57% against a *supervised* baseline of 1.34% on the same benchmark. SSPS closes the gap to supervised training; it does not exceed it. You are already on the supervised side of that gap with labelled data. Adopting SSPS as your main recipe would be a regression.

**SSL-from-scratch is out of compute reach.** The paper trains 100 epochs on 1.09M utterances using 2×V100 16GB (4×V100 32GB for DINO), then runs SSPS experiments on 2–4×A100 80GB. Your dev node has no GPU; node-3 has two free A10s and node-4's driver is broken. Even scaled to your 23× smaller corpus, this is a multi-week campaign on borrowed hardware for a method that starts from a worse baseline.

**Your 91.6% channel probe is a different axis — do not conflate them.** That result measures *cross-corpus* separability (which microphone/dataset). SSPS attacks *within-speaker, cross-session* channel information. Related, not identical. SSPS would not directly fix the 91.6% number, and claiming otherwise in the thesis would be an overreach a reviewer will catch.

### 3.2 What WILL help — ranked

#### ① Cross-session positive sampling (do this first)

This is the paper's "Supervised positive sampling" column, and it is the highest value-per-hour item in the whole analysis.

Your metric-learning losses (`angleproto`, `ge2e`, `proto`, `triplet`, `softmaxproto`) consume `nPerSpeaker > 1` utterances per speaker. `DatasetLoader` currently samples those **at random within the speaker** — which, given 118 utterances per session and only 3.3 sessions, means the grouped positives are usually **from the same session**. You are reproducing the exact defect the paper identifies, inside a supervised trainer.

The fix: constrain the `nPerSpeaker` utterances to come from **distinct sessions** where the speaker has them, falling back to random for the 19% single-session speakers. The session ID is already in your path (`spk/<video_id>/utt.wav`), so no new metadata is needed.

On VoxCeleb this change alone was worth 41–73% relative EER reduction depending on framework. Your ceiling is lower (69.7% of pairs can be cross-session vs 96.4%), so expect attenuated gains — but this is ~30 lines in `DatasetLoader.py`, needs no new compute, composes with everything you already built, and lands squarely on the `expects_grouped_input` plumbing you just fixed in BUGFIX-003.

Add it as a config flag (`positive_sampling: cross_session | random`) so the ablation is a single-variable comparison and slots straight into your `sl_p2_no_*` protocol.

#### ② Session ID as the channel label for FEATURE-008 DANN

You built a DANN channel-adversarial head that requires `dann_channel_label_file` mapping speakers to channel classes (mic/phone/codec). Getting true channel labels for SL corpora is hard, which is presumably why this feature is unexercised.

**SLCeleb's YouTube video IDs are de facto channel labels.** Different video = different room, mic, encoder, upload pipeline. You have 403 of them. This unblocks FEATURE-008 immediately, at config cost only.

Caveat worth stating in the write-up: session ID conflates channel with *time* and *content*, so it is a proxy, not ground truth. That is still far better than not running the experiment.

#### ③ Speaker/recording NMI ratio as QC Layer 7

The paper's Fig. 7 metric is a clean, cheap, label-based measure of how much channel information a representation encodes:

```
NMI(clusters, speaker_labels) / NMI(clusters, recording_labels)
```

You can compute this **today** on your existing embeddings — you have `tools/extract_embeddings.py`, k-means infrastructure in `tools/projections.py`, and recording labels for SLCeleb. It gives you:

- a number quantifying how channel-contaminated your current WavLM embeddings are;
- a before/after metric for recommendations ① and ②;
- a natural Layer 7 for the QC suite, complementing the existing corpus-level channel probe with a *within-speaker* measurement.

This is the fastest route from this paper to a defensible thesis figure.

#### ④ SSPS proper — conditional

Worth attempting **only** if you first widen the corpus geometry. The binding constraint is sessions/speaker (3.3), not corpus size. Since SLCeleb is YouTube-derived, harvesting **more videos per existing speaker** — target ≥8–10 sessions each — widens the operating window far more effectively than any hyperparameter tuning, and it also lifts the 69.7% oracle ceiling and retires the 19% single-session speakers.

If you do run it: `K ≈ 250`, `M = 1`, SSPS activated only in the final ~20% of epochs, and report pseudo-positive **speaker accuracy** and **recording accuracy** (you have the labels for both) so the mechanism is verified rather than assumed.

---

## 4. Integration notes against your codebase

| Paper component | Your status | Gap |
|---|---|---|
| ECAPA-TDNN encoder (22.5M, ASP, 512-d) | ✅ FEATURE-006, same config | None |
| Mel frontend, 25 ms / 10 ms | ✅ shared `models/_frontend.py` | None |
| MUSAN + RIR augmentation, SNR ranges | ✅ `DatasetLoader`, BUGFIX-016 | Paper's SNR bands: speech 13–20, music 5–15, noise 0–15 dB |
| Cosine scoring, EER + minDCF (p=0.01) | ✅ `tuneThreshold.py` | Paper uses EER_avg; you now return both (BUGFIX-020) ✓ |
| Weight averaging over last 10 epochs | ❓ not present | Cheap, reliable win — worth adding independently |
| SSL joint-embedding training loop | ❌ absent | This is the real integration cost |
| Memory queues + per-epoch GPU k-means | ❌ absent | Non-trivial; DDP-aware |
| Reference-representation branch (4 s, no aug.) | ❌ absent | Third forward pass per step, ~+30% compute |

**SSPS is not a flag you can add to `trainSpeakerNet.py`.** Your pipeline is *pretrained WavLM → supervised PEFT fine-tune* (`tools/peft_finetune.py`). SSPS operates at a different stage — training a speaker encoder from scratch with an SSL pretext objective. Adopting it means importing the `sslsv` toolkit as a parallel track, not extending your existing trainer.

Recommendations ① and ② live entirely inside your current code. That asymmetry is why they rank above ④.

Note also: the paper found **the projector should be discarded for contrastive methods** (SimCLR/MoCo) as it degrades downstream SV performance — a useful detail if you ever build a contrastive arm.

---

## 5. Research positioning

The paper is VoxCeleb-only and English-dominant. **No published work applies SSPS to low-resource or non-English SV.** That gap is real and it is yours to fill.

Two framings, both defensible:

**Positive framing.** "Cross-session positive sampling for low-resource SV" — recommendation ①, ablated against random sampling, with the NMI ratio as the mechanism-level evidence. Cheap, self-contained, and it works with the corpus you already have.

**Negative-result framing.** "Why SSPS does not transfer to low-resource corpora: a session-density analysis." The cluster-size window argument in §2.2 is a general result — it predicts SSPS viability from three corpus statistics (utts/recording, utts/speaker, corpus size) without running anything. A reviewer can apply it to their own dataset. You already have precedent for publishing negative results honestly (BUGFIX-010, the NestedSpeakerNet quarantine), and this one comes with a quantitative criterion rather than just a failure report.

The strongest thesis contribution is probably **both**: the window criterion as theory, recommendation ① as the thing that actually improved your numbers.

Slots naturally as RQ5 (front-end / training-time), alongside your existing RQ4 on backend adaptation.

---

## 6. Bottom line

The paper correctly identifies the disease you have independently measured in your own data. Its cure was designed for a corpus with 24 recordings per speaker; you have 3.3, and five of your six corpora have none at all.

**Take the insight, not the implementation.** The insight is that *positives sampled from different recordings are worth more than any amount of data augmentation*. You can act on that with labels you already own, in code you already wrote, without a GPU — and that is where the return is.

Reconsider SSPS itself once SLCeleb reaches ~8 sessions per speaker. Until then it is a well-executed solution to a problem you cannot yet pose.

---

## 7. Citation

```bibtex
@article{lepage2025ssps,
  title   = {Self-Supervised Frameworks for Speaker Verification
             via Bootstrapped Positive Sampling},
  author  = {Lepage, Th\'eo and Dehak, R\'eda},
  journal = {arXiv preprint arXiv:2501.17772},
  year    = {2025},
  note    = {v4, 24 July 2025},
  url     = {https://arxiv.org/abs/2501.17772}
}
```

**Key numbers for citation:** SimCLR-SSPS 2.57% EER / DINO-SSPS 2.53% EER on VoxCeleb1-O; supervised baseline 1.34%; 58% relative EER reduction for SimCLR; SSPS-Clustering K=25,000, M=1; speaker-to-recording NMI ratio 1.01 → 1.09.

---

*Companion documents: `SL_LANGUAGE_SPV_ANALYSIS.md` (project audit), `research_logs/2026-08-10-dataset-qc-results.md` (the 91.6% channel probe), `docs/bugfixes/FEATURE-008-dann-adversarial.md` and `FEATURE-009-ssl-pretraining.md` (the deferred SSL design this paper informs).*
