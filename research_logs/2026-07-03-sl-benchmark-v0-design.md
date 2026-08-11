# SL-SPV Benchmark v0 — Corpus, Trial Protocol, and Metrics (P0)

**Date:** 2026-07-03 · **Status:** built and smoke-tested · **Roadmap:** P0 of
`2026-07-03-project-audit-sota-roadmap.md`

## 1. Purpose

First reproducible Sinhala/Tamil speaker-verification benchmark for this
project. Design follows the audit's protocol requirements (issues E1–E5) and
the evaluation-practice literature (NIST SRE24 plan; Hutiri et al.,
Interspeech 2022 on the same-recording confound; SdSV protocol for
cross-lingual partitions).

## 2. Corpora (v0 — public, on-disk now)

| Corpus | Lang | Speakers kept | Utts | Notes |
|---|---|---|---|---|
| OpenSLR SLR52 (CC BY-SA) | si | 478 (min 10 utts) | ~185k | crowdsourced read speech, FLAC 16 kHz, speaker IDs from `utt_spk_text.tsv`; **no gender, no session metadata** |
| OpenSLR SLR65 (CC BY-SA) | ta | 49 (min 10 utts) | 4,286 | studio read speech, 48 kHz wav (resampled at load), gender from `taf_`/`tag_` filename prefix |

Raw archives: `/mnt/ricproject3/2025/data/sl_corpora/{slr52,slr65}`.
Corpus layout (symlinks, no copies): `/mnt/ricproject3/2025/data/sl_celeb/<lang>/<spk>/<utt>.{flac,wav}`,
built by `tools/ingest_openslr.py` (new, this date). Manifest + speaker
metadata: `ingest_manifest.csv`, `spk_meta.csv` at the corpus root.

**v1 upgrade path (pending):** SLCeleb (our group's 280-speaker in-the-wild
corpus, IEEE DataPort) is NOT on the local mounts — only its video-processing
pipelines are. Retrieval from IEEE DataPort/Google Drive is an open action.
SLCeleb adds: multi-genre audio, multi-session speakers (video ID = session),
and bilingual speakers for true si↔ta cross-lingual trials.

## 3. Trial protocol (implemented in the upgraded `tools/sl_dataprep.py`)

- **Split:** per-speaker 80/20 train/test utterance split, seed 42.
- **Targets:** same speaker, same language; **cross-session enforced** when
  session keys are derivable (`--session_from parentdir|regex`); speakers with
  one session are excluded from target sampling and counted in the log.
  *v0 limitation:* OpenSLR layouts are flat (no session metadata), so v0 runs
  with `--session_from none`. Read-speech single-session targets make absolute
  EERs **optimistic** — v0 numbers are comparable to each other, not to
  in-the-wild benchmarks. SLCeleb (v1) fixes this.
- **Impostors:** same language; **same-gender when both genders known**
  (`--spk_meta`); verified 0 cross-gender impostor pairs in the ta smoke run.
- **Dedup:** unordered pair dedup + `a != b` guaranteed; sampling is
  attempt-bounded with reported shortfalls instead of infinite loops.
- **Cross-lingual list (`test_list_cs.txt`):** same speaker across languages —
  empty for v0 (OpenSLR has no bilingual speakers); becomes meaningful with
  SLCeleb/new collection.
- **v0 volumes:** 3,000 targets + 9,000 impostors per language (1:3 ratio),
  seed-reproducible.

## 4. Metrics

Computed by `tools/zeroshot_eval.py` (and the trainer's eval path):

- **EER_avg** (literature-standard mean of FPR/FNR at crossing) as the primary
  number; EER_max kept as the conservative diagnostic (repo convention,
  BUGFIX-020).
- **minDCF at p_target = 0.01 and 0.05**, C_miss = C_fa = 1 — reported at both
  operating points to be comparable with the NIST and VoxCeleb literatures
  (fixes audit issue E2).
- Unlabeled trials are a **hard error** in the harness (fixes E4's silent
  `random.randint` label fallback for this path).
- Calibration metrics (Cllr/actDCF) arrive with P2.

## 5. Known limitations of v0 (to state in any write-up)

1. Read speech, near single-session speakers → optimistic absolute EERs; valid
   for *relative* model comparison only.
2. SLR52 has no gender metadata → Sinhala impostors are not gender-controlled
   (Tamil ones are). Gender predictor or SLCeleb metadata can fix this later.
3. No telephony/codec condition; no code-switched speech.
4. Tamil is Indian-accented (SLR65), not Sri Lankan Tamil — flag when claiming
   Sri Lankan coverage; SLCeleb provides the Sri Lankan Tamil condition.

## 6. Reproduction

```bash
# 1. Ingest (symlinks)
python tools/ingest_openslr.py \
  --slr52_dir /mnt/ricproject3/2025/data/sl_corpora/slr52/extracted \
  --slr65_dir /mnt/ricproject3/2025/data/sl_corpora/slr65/extracted \
  --out_root /mnt/ricproject3/2025/data/sl_celeb --min_utts 10

# 2. Lists (seed-reproducible)
python tools/sl_dataprep.py \
  --corpus_root /mnt/ricproject3/2025/data/sl_celeb \
  --out_dir /mnt/ricproject3/2025/data/sl_celeb/lists \
  --langs si ta --test_frac 0.2 \
  --target_pairs_per_lang 3000 --impostor_ratio 3 \
  --cohort_size 500 \
  --spk_meta /mnt/ricproject3/2025/data/sl_celeb/spk_meta.csv --seed 42
```
