# Configuration Rationale Guide — SL-SPV

> Why each configuration is what it is, what hypothesis each one tests,
> and what to expect specifically for Sinhala / Tamil.
>
> Companion to [`RUN_GUIDE.md`](RUN_GUIDE.md) (operational steps) and
> [`SL_LANGUAGE_SPV_ANALYSIS.md`](SL_LANGUAGE_SPV_ANALYSIS.md) (the
> shipping-feature audit). This document is the **why**; RUN_GUIDE
> is the **how**.

---

## Contents

- [Part 1 — Linguistic context: why Sinhala+Tamil is hard](#part-1--linguistic-context)
- [Part 2 — The thirteen configurations](#part-2--the-thirteen-configurations)
- [Part 3 — Knob-by-knob justification](#part-3--knob-by-knob-justification)
- [Part 4 — What to expect, language by language](#part-4--what-to-expect-language-by-language)
- [Part 5 — Reading the results: how to interpret each delta](#part-5--reading-the-results)

---

# Part 1 — Linguistic context

Sri Lankan Sinhala and Tamil are linguistically and phonologically
distinct in ways that bear directly on speaker-verification design.

## 1.1 Sinhala (Indo-Aryan family, ~17M speakers)

- **Stop-consonant inventory**: distinguishes aspirated vs unaspirated
  voiceless stops (/p/ vs /pʰ/, /t/ vs /tʰ/) and prenasalised stops
  (/ᵐb/, /ⁿd/, /ᵑɡ/). The latter is rare among the world's languages
  and concentrates speaker-discriminative cues in transient
  formant trajectories that mel-spectrogram preserves well.
- **Vowel system**: 7 short + 7 long vowel qualities, plus diphthongs.
  Long vs short distinction affects utterance duration distributions.
- **Influence on SV**: most speaker-discriminative information lies in
  vowel formants and stop-release acoustic transients — both of which
  are well-captured by 80 mel bins at 16 kHz.

## 1.2 Tamil (Dravidian family, ~5M Sri Lankan speakers)

- **Stop-consonant inventory**: contrasts on place of articulation
  (5 distinct places including retroflex, central to Dravidian) but
  no aspiration distinction in most Sri Lankan dialects. Geminate
  (doubled) consonants are phonemic and add duration cues.
- **Vowel system**: 5 short + 5 long vowels. Simpler than Sinhala but
  with characteristic centralised mid-vowels.
- **Influence on SV**: retroflex consonants produce distinctive
  spectral patterns; geminate vs single-consonant contrast adds
  prosodic / timing cues that augment the speaker signature.

## 1.3 Cross-lingual challenge

The two languages share **no genetic relationship** (Indo-European vs
Dravidian), but Sri Lankan bilinguals — common in our corpus —
unconsciously share **phonetic habits** across both: voice quality,
fundamental-frequency range, breathiness, and prosodic timing remain
relatively stable per speaker across languages. The SV embedding
should capture *those* invariant cues rather than language-specific
phoneme distributions.

The configurations in Part 2 are designed to test how well each
shipped feature contributes to extracting language-invariant speaker
identity from a 100-speaker bilingual corpus.

## 1.4 Deployment context: Sri Lankan telephony

The realistic deployment channel for SL voice biometrics is
**telephony at 8 kHz**, often with codec compression (G.711 μ-law,
AMR-NB, Opus 8 kHz). Our training corpus is 16 kHz mono studio
quality; the live-channel mismatch is a known threat to deployment
EER. Mitigation in the current configs:

- The MUSAN + RIR augmentation chain ([BUGFIX-016](docs/bugfixes/BUGFIX-016-configurable-augment-chain.md))
  with `clean: 0.30` ensures the model sees corrupted audio 70% of
  the time, building general channel robustness.
- **Not yet mitigated**: explicit 8 kHz down/upsampling + μ-law
  companding during training. §3.1 #5 of the analysis doc flags
  this as a Tier-1 future improvement. If your deployment is
  telephony-first, add this to the augmentation pipeline before
  reporting deployment numbers.

---

# Part 2 — The thirteen configurations

The configs partition into three groups by purpose. Each YAML file
inherits NOTHING — every key is explicit so the file is
self-documenting.

## 2.1 Primary configurations (3)

These produce the headline EER numbers in your thesis / paper.
Run each with 3 random seeds (42, 123, 7) for `mean ± std`.

| Config | File | Hypothesis | Expected pooled EER |
|---|---|---|---|
| **P0** | [`configs/sl_p0_baseline.yaml`](configs/sl_p0_baseline.yaml) | Can a modern SV architecture learn anything from 100 SL speakers with NO English prior? | **8 – 20 %** (the floor) |
| **P1** | [`configs/sl_p1_finetune.yaml`](configs/sl_p1_finetune.yaml) | How much does English-pretrained knowledge transfer to SL via fine-tune + LLRD? | **5 – 12 %** (30–50 % relative drop vs P0) |
| **P2** | [`configs/sl_full_stack.yaml`](configs/sl_full_stack.yaml) | What does the full feature stack add over fine-tune alone? | **3 – 8 %** (40–60 % relative drop vs P0) |

## 2.2 Feature-add configurations (4)

Each adds ONE feature to P0 in isolation. Tells you the **standalone
contribution** of each feature, independent of the others.

| Config | File | Feature isolated | Expected delta vs P0 |
|---|---|---|---|
| Fine-tune only | [`configs/sl_feature_finetune_only.yaml`](configs/sl_feature_finetune_only.yaml) | FEATURE-004 (cross-lingual init + LR multiplier, NO LLRD) | 25–45 % relative EER drop |
| Lang-aux only | [`configs/sl_feature_lang_aux_only.yaml`](configs/sl_feature_lang_aux_only.yaml) | FEATURE-007 (multi-task language-ID head) | 5–15 % relative drop |
| AS-Norm only | [`configs/sl_feature_as_norm_only.yaml`](configs/sl_feature_as_norm_only.yaml) | FEATURE-002 (cohort score normalisation) | 5–15 % relative MinDCF drop, smaller EER drop |
| PLDA only | [`configs/sl_feature_plda_only.yaml`](configs/sl_feature_plda_only.yaml) | FEATURE-010 (two-covariance PLDA backend) | 0–10 % relative EER drop, larger on short trials |

> Note: LLRD has no isolated feature-add config because it has no
> meaning without `initial_model` (it adjusts per-layer LRs of a
> loaded checkpoint). Its contribution is measured ONLY by the
> P2-no-LLRD ablation in §2.3.

## 2.3 P2 ablation configurations (5)

Each removes ONE feature from the full stack. Tells you the
**marginal contribution** of each feature given everything else is
on — typically a smaller number than the feature-add delta because
some lift is shared / redundant across features.

| Config | File | Feature removed | Expected delta vs P2 |
|---|---|---|---|
| No fine-tune | [`configs/sl_p2_no_finetune.yaml`](configs/sl_p2_no_finetune.yaml) | FEATURE-004 (and LLRD by extension) | LARGE EER increase (dominant lever) |
| No LLRD | [`configs/sl_p2_no_llrd.yaml`](configs/sl_p2_no_llrd.yaml) | FEATURE-005 (LLRD only; fine-tune kept) | 0–2 % relative (likely small on 4-layer ECAPA) |
| No lang-aux | [`configs/sl_p2_no_lang_aux.yaml`](configs/sl_p2_no_lang_aux.yaml) | FEATURE-007 | 0–5 % relative; LARGER on Cs (cross-lingual) subset |
| No AS-Norm | [`configs/sl_p2_no_as_norm.yaml`](configs/sl_p2_no_as_norm.yaml) | FEATURE-002 | 0–5 % relative EER; 5–15 % relative MinDCF |
| No PLDA | [`configs/sl_p2_no_plda.yaml`](configs/sl_p2_no_plda.yaml) | FEATURE-010 | 0–3 % relative EER on long utterances |

## 2.4 Total experimental count

3 primary × 3 seeds = 9 runs
4 feature-add × 1 seed = 4 runs
5 P2-ablation × 1 seed = 5 runs
**Total: 18 trainer invocations.**

Wall-clock estimate dispatched across your A40 / 2× T4 (DDP) / A10:
**~5–7 days end to end** per RUN_GUIDE.md §12.

---

# Part 3 — Knob-by-knob justification

Why specific parameter values were chosen. Useful to defend in
viva / journal review.

## 3.1 Architecture: `model: ECAPA_TDNN`, `channels: 1024`

- **Why ECAPA-TDNN over ResNetSE34V2?** ECAPA's Res2Net multi-scale
  convolutions capture speaker cues at multiple temporal resolutions
  simultaneously — useful when bilingual speakers may shift speech
  rate between languages.
- **Why `channels: 1024` (Large)?** Standard "ECAPA-Large" from
  Desplanques et al. 2020. With 100 SL speakers the risk is
  overfitting, but our **full-stack** regularisation (lang-aux
  + AS-Norm + PLDA + finetune-freeze + LLRD) more than offsets the
  capacity. Switch to `channels: 512` only if T4 VRAM forces it
  or if P0 EER is unstable across seeds.
- **Why `nOut: 192`?** Paper-default embedding dim. Higher (e.g. 256)
  marginally helps EER on VoxCeleb-scale; on 100 SL speakers the
  PLDA fit becomes the bottleneck (constraint `2·plda_dim ≤
  n_speakers`), so we stay at 192.

## 3.2 Loss: AAM-Softmax `margin: 0.2, scale: 30`

- **Margin 0.2**: standard ArcFace value. For very small corpora
  (<200 speakers) some recent work uses 0.1 to soften the
  classification difficulty; 0.2 is the safe, well-replicated
  default.
- **Scale 30**: scales the cosine-margin softmax to a sharper
  temperature; 30 is industry standard.

## 3.3 Mel frontend: 80 bins, log input, pre-emphasis 0.97

- **80 bins** captures Sinhala's full formant range (F1–F3 fall in
  300–3500 Hz, all covered by 80 bins at 16 kHz). Tamil retroflex
  consonants produce energy concentration around 2–3 kHz, also well
  represented.
- **Pre-emphasis 0.97**: ECAPA paper standard; boosts high-frequency
  formant detail.

## 3.4 Cross-lingual fine-tune: `lr_multiplier: 0.1`, `freeze: [frontend]`

- **Why freeze the mel front-end specifically?** With <10k SL
  utterances the front-end can overfit to corpus-specific channel
  noise. Freezing it preserves the English-pretrained channel
  invariance and lets the encoder layers absorb the SL adaptation.
- **Why `lr_multiplier: 0.1`?** BERT / WavLM fine-tune literature
  consistently uses 0.1×. For SV specifically, 0.1× × `lr=0.001`
  = `1e-4`, which is the empirical sweet spot for cross-lingual SV
  in published low-resource studies.

## 3.5 LLRD: `decay: 0.9`, `pattern: ecapa`

- **Why decay 0.9 (not 0.7 like transformer fine-tune)?** ECAPA has
  only 4 sequential SE-Res2 blocks; 12-layer-transformer decays of
  0.7 would put the input layer at `0.7^11 ≈ 0.02×`, essentially
  freezing it. 0.9 on 4 layers gives `0.9^3 = 0.73×` between top
  and bottom — meaningful but not crushing.

## 3.6 Multi-task lang-aux: `weight: 0.3`, `num_classes: 4`

- **Why λ = 0.3?** SV+aux literature uses 0.1 – 0.5 commonly; 0.3
  balances regularisation against speaker-loss dominance.
  Lower (0.1) may not provide enough signal on 100 speakers;
  higher (0.5) over-regularises and starts degrading speaker EER.
- **Why 4 classes (si/ta/en/mix)?** Matches the per-language test
  list partition (FEATURE-003). The `mix` class captures
  code-switched utterances common in Sri Lankan speech.

## 3.7 AS-Norm: `top_K: 300`, cohort size 500

- **Why top-K 300?** Matejka et al. 2017 literature standard; for a
  500-speaker cohort, the top 300 captures the most discriminative
  impostors without diluting statistics with weakly-impostor
  cohort members.
- **Why 500 cohort speakers?** Drawn from the train set, so does not
  overlap test trials. 500 is well above the
  rule-of-thumb $\geq 2 \times \mathrm{top\_K}$.

## 3.8 PLDA: `dim: 50`

- **Why only 50?** Hard constraint: `2 × plda_dim ≤ n_speakers`.
  With 100 SL speakers, max possible is 50. For VoxCeleb-scale
  corpora the standard is 200; bump after corpus grows beyond
  ~400 speakers.
- **Limitation note**: the small `plda_dim` is the most significant
  capacity bottleneck in the current config. The expected
  PLDA-vs-cosine delta will be smaller than literature values
  precisely because of this. **Acknowledge explicitly in your
  thesis Discussion section.**

## 3.9 Augmentation chain: `clean: 0.30` vs upstream default 0.50

- **Why lower clean fraction?** SL deployment is rarely studio; the
  model needs to be robust to in-the-wild acoustic conditions. We
  see 70 % corrupted audio (reverb / music / speech / noise) during
  training so the model can generalise to noisy SL test conditions.

## 3.10 Reproducibility: `--deterministic`

Sets cuDNN deterministic flag, disables TF32, enables
`torch.use_deterministic_algorithms`. Adds ~15 % wall-clock cost
but is mandatory for paper-grade reporting.

---

# Part 4 — What to expect, language by language

These are **predictions** (not guarantees) based on the linguistics
and the literature. Use them to sanity-check your runs.

## 4.1 Monolingual Sinhala (`test_list_si.txt`)

- **Easier than cross-lingual.** Same-language enrolment-test
  matches the in-domain training distribution.
- **Easier than monolingual Tamil** by a small margin (1–3 %
  absolute EER lower) because Sinhala has 60 % of our training
  speakers vs Tamil's 40 % — more in-class training samples.
- **Hardest case**: short utterances (<2 s) where the rich Sinhala
  consonant inventory is sparse in any single utterance. PLDA
  helps most here.

## 4.2 Monolingual Tamil (`test_list_ta.txt`)

- **Slightly harder than Sinhala** for the imbalance reason above.
- **Cross-lingual fine-tune contributes less than for Sinhala**
  because English (the pretraining language) is closer typologically
  to Sinhala than Tamil; English-pretrained features transfer
  marginally less cleanly to Tamil.
- **Lang-aux helps more here** because the multi-task signal
  amortises the lower train-data quantity per Tamil speaker by
  forcing the encoder to extract language-discriminative features
  that also help speaker discrimination.

## 4.3 Cross-lingual (`test_list_cs.txt`)

- **Hardest of the three** — enrolment in one language, test in
  another. Same-speaker bilingual recordings drive this list.
- **Cross-lingual gap** = `EER(Cs) − ½(EER(Si) + EER(Ta))`. This is
  the **headline number for code-switching robustness**.
- Expected gap under **P0**: 5–10 % absolute (large; the model
  hasn't learned language-invariant speaker features).
- Expected gap under **P2**: 1–4 % absolute (small; the lang-aux
  + AS-Norm + fine-tune stack closes most of the cross-lingual
  penalty).

## 4.4 Pooled (`test_list.txt`)

- Concatenation of si, ta, cs — closest to deployment-time
  performance when the enrolment-test language is unknown.
- Best reported as the **headline single number** in the abstract.

## 4.5 Expected per-language deltas as features are added

A rough mental model of what each lever helps with most:

| Feature | Helps Si | Helps Ta | Helps Cs (cross-lingual) |
|---|---|---|---|
| Cross-lingual fine-tune (FEATURE-004) | YES (large) | YES (moderate) | YES (large) |
| LLRD (FEATURE-005) | YES (small) | YES (small) | YES (small) |
| Lang-aux (FEATURE-007) | YES (small) | YES (moderate) | **YES (large)** ← Cs is where this lever shines |
| AS-Norm (FEATURE-002) | YES (small EER, larger MinDCF) | YES (similar) | YES (similar) |
| PLDA (FEATURE-010) | YES (depends on utterance length) | YES (similar) | YES (similar) |

The strongest single signal in your thesis Discussion will be:
**which feature contributes the most to closing the cross-lingual
gap?** Based on literature, that is most likely the lang-aux head
(FEATURE-007) — but only your runs can confirm.

---

# Part 5 — Reading the results

Once `tools/aggregate_seeds.py --latex` has produced your tables,
the analysis pattern is:

## 5.1 Headline question: did cross-lingual fine-tune dominate?

Compare P0 vs P1 pooled EER:
- If P1 / P0 ≤ 0.7 → fine-tune is the dominant lever. This is the
  expected outcome.
- If P1 / P0 ≥ 0.9 → something's wrong. Either the English checkpoint
  is corrupt, or the SL corpus has channel mismatch the model can't
  bridge. Diagnose before going further.

## 5.2 Compounding question: did the full stack add real value over P1?

Compare P1 vs P2 pooled EER:
- If P2 / P1 ≤ 0.85 → the in-domain regularisation + scoring backends
  compounded productively.
- If P2 ≈ P1 → the small SL corpus didn't have enough signal for
  lang-aux / AS-Norm / PLDA to differentiate themselves. Honest
  finding to report.

## 5.3 Per-lever question: what does each lever actually contribute?

Two perspectives, both worth reporting:

**Feature-add view** (P0 → P0+feature):
- Tells you the lever's effect from a cold start.
- Larger numbers because lift is uncontested.

**P2-ablation view** (P2 → P2-feature):
- Tells you the lever's effect at the margin.
- Smaller numbers because earlier features absorbed some of the lift.

Report **both** in your thesis ablation chapter to distinguish
"feature X is useful" (feature-add) from "feature X is uniquely
useful given everything else" (P2-ablation).

## 5.4 Cross-lingual question: did we close the gap?

For each config, compute `gap = EER(Cs) − ½(EER(Si) + EER(Ta))`.
Plot `gap` vs config from P0 → P1 → P2. A monotonically decreasing
curve is the strongest possible single-figure statement of your
contribution.

## 5.5 Sanity-check red flags

- **P0 < 3 %** → trial-pair leakage. Re-check `tools/sl_dataprep.py
  --seed` and verify enrolment/test utterances don't overlap.
- **P2 ≥ P0** → a feature is broken or the YAML wired wrong. Single-
  feature ablation isolates which one.
- **Si ≈ Cs** → the cross-lingual trials may not actually be
  cross-lingual (check `test_list_cs.txt` is non-empty and contains
  bilingual speakers).
- **EER std across seeds > 1 % absolute** → the corpus is too small
  for stable training; consider corpus expansion before reporting.

---

## Compose your master script

```bash
# All 13 unique configs (excluding multi-seed expansion):
CONFIGS=(
    sl_p0_baseline.yaml
    sl_p1_finetune.yaml
    sl_full_stack.yaml                  # P2
    sl_feature_finetune_only.yaml
    sl_feature_lang_aux_only.yaml
    sl_feature_as_norm_only.yaml
    sl_feature_plda_only.yaml
    sl_p2_no_finetune.yaml
    sl_p2_no_llrd.yaml
    sl_p2_no_lang_aux.yaml
    sl_p2_no_as_norm.yaml
    sl_p2_no_plda.yaml
)
for CFG in "${CONFIGS[@]}"; do
    python trainSpeakerNet.py --config configs/${CFG} --nClasses ${NCLASSES}
done
```

Multi-seed extension for the 3 primary configs:

```bash
for SEED in 42 123 7; do
  for PRIMARY in sl_p0_baseline sl_p1_finetune sl_full_stack; do
    python trainSpeakerNet.py --config configs/${PRIMARY}.yaml \
        --seed ${SEED} --nClasses ${NCLASSES} \
        --save_path exps/${PRIMARY%.yaml}_seed${SEED}
  done
done
```

Aggregate at the end:

```bash
python tools/aggregate_seeds.py --latex --prefixes \
    sl_p0_baseline sl_p1_finetune sl_full_stack
```

— and paste the LaTeX table directly into your thesis §4 / paper §3.
