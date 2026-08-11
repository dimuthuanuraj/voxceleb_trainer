# Openly Available Sinhala / Tamil Corpora for Speaker Verification

**Date:** 2026-08-10 · **Status:** survey **executed** — see §5 for what was actually built
**Context:** benchmark v0 (`2026-07-03-sl-benchmark-v0-design.md`) runs on SLR52 (478 si)
+ SLR65 (49 ta). The Tamil side is too small to resolve the feature-ablation deltas
we now want to measure. This log lists every open corpus that could widen it.

> **Read §5 first if you are acting on this.** §0–§4 are the survey as written
> *before* anything was downloaded. Five corpora were then built into
> `data/` (index: `data/README.md`), and doing so corrected several claims below.
> Corrections are marked **[CORRECTED §5]** inline; nothing has been deleted, so
> the original reasoning stays auditable.

---

## 0. Headline

Three things are true and they shape everything below:

1. **Tamil is the binding constraint, and it is fixable.** 49 speakers is not enough
   to separate a 0.3 % EER improvement from noise. Two corpora take us to ~800 Tamil
   speakers: **SLR127 (531 spk)** and **Kathbath/IndicSUPERB Tamil (226 spk, with
   published ASV trial lists)**. Both are open licences.
2. **Sinhala has almost nothing new.** SLR52 already is the Sinhala open corpus.
   The only genuinely new Sinhala audio is **SiTa** (10 h, in-the-wild) and
   **SLCeleb** (ours). Sinhala is *absent from Common Voice* — verified against the
   CV 26.0 locale list (294 locales, no `si`).
3. **Every large Tamil corpus is Indian Tamil, not Sri Lankan Tamil.** SLR65, SLR127,
   Kathbath, IndicVoices, Vaani — all recorded in India. They are excellent
   *training/enrolment-population* data and useless for claiming Sri Lankan Tamil
   performance. Keep SLCeleb + SiTa Tamil as the held-out SL-dialect eval.

---

## 1. Tier 1 — use these

| Corpus | Lang | Speakers | Size | Licence | Why it matters |
|---|---|---|---|---|---|
| **Kathbath / IndicSUPERB** (AI4Bharat) | ta (+11 Indic) | **226 ta** | 1,684 h total | **CC0** | The only one with a *ready-made ASV protocol*: `valid_data.txt`, `test_known_data.txt`, `test_data.txt` per language; speaker + gender in filenames; ships a **noisy test variant** — a free robustness axis for the ablation. Audio is m4a. **[CORRECTED §5: the 226 counts the 85 GB train split too; the three eval splits hold 20 speakers each (60 built), and the trial count is large while the *speaker* count is not.]** |
| **IISc-MILE Tamil ASR** (SLR127) | ta | **531** | ~150 h | CC BY 2.0 | Biggest open Tamil speaker pool. Clean read speech, 16 kHz/16-bit mono, train/test split already present. ~~⚠️ the OpenSLR page does not document the speaker-ID convention~~ **[CORRECTED §5: resolved — field 2 is the speaker; and 107 of 531 speakers turn out to be genuinely multi-session, the best cross-session read speech we have.]** |
| **SLR52 Large Sinhala ASR** | si | 478 kept | ~185 k utts | CC BY-SA 4.0 | Already ingested. Still the Sinhala backbone. |
| **SLCeleb** (ours, IEEE DataPort) | si + ta | 280 | 34 k utts | CC BY 4.0 | Still the *only* source of multi-session, multi-genre, bilingual, Sri-Lankan-dialect speakers. Remains the v1 blocker (USER_ACTION_ITEMS §1). |

## 2. Tier 2 — worth adding, with caveats

| Corpus | Lang | Speakers | Size | Licence | Caveat |
|---|---|---|---|---|---|
| **Common Voice 26.0** | ta only | **981 distinct voices** | 425 h total / **235 h validated** | CC0 | `client_id` is a pseudo-speaker ID and contributors record across *multiple days* → ~~one of the few open sources of genuine **cross-session** Tamil trials~~ **[CORRECTED §5: overstated. CV ships no session field and no timestamp, so cross-sitting pairs are *likely* but cannot be enforced or verified. Its real distinction is device/channel diversity.]** Gender is missing on ~66 % of clips, which limits same-gender impostor sampling. No Sinhala. |
| **IndicVoices / IndicVoices-R** | ta (+21) | 10,496 total | 1,704 h | CC BY 4.0 | Spontaneous + read, rich per-speaker metadata (pitch, SNR, C50, demographics). Tamil subset also mirrored on Kaggle. Speaker counts per language need checking. |
| **Vaani** (ARTPARK-IISc × Google DeepMind) | ta (+104) | 158,441 total | ~31,255 h raw | CC BY 4.0 | Image-prompted spontaneous speech, downloadable **per district** — so you can pull Tamil Nadu districts only. The authors explicitly pitch it for speaker ID/verification. Huge; treat as a pretraining/AS-Norm-cohort pool, not an eval set. |
| **NISP** (IISc LEAP) | ta (+4 Indic +en) | 345 total | ~4–5 min/spk | open, GitHub | Small, but **each speaker records in both their mother tongue and English** → a real cross-lingual trial list without SLCeleb. Directly relevant to the P4 cross-lingual study. |
| **SiTa** (Univ. of Moratuwa, CHiPSAL 2025) | **si + ta** | si: 1–10 per video (60 videos, 602 min) · ta: 2–6 per video (14 videos, 121 min) | ~12 h | ~~see repo~~ **CC BY-NC 4.0** | The only new **Sri Lankan** in-the-wild audio: YouTube panel shows, debates, quizzes, code-mixed, real overlap. Labels are *diarization* turns — speaker identity is within-recording only, so it gives you conversational eval segments, not cross-recording trials, unless you link identities manually. **[CORRECTED §5: licence is CC BY-NC 4.0 — NON-COMMERCIAL, so unusable in the VoiceID product path. Access is also gated behind a request form.]** |
| **SPRING-INX** (IIT Madras) | ta (+9) | not stated | ~2,000 h | CC BY 4.0 | Large and legally clean; speaker labelling for SV needs verification. |
| **SLR30 Sinhala TTS** (Google) | si | multi-speaker, count unconfirmed | ~699 MB | CC BY-SA 4.0 | UserID is embedded in the FileID, so speaker labels exist. Small; likely only a handful of speakers. Cheap to check, low expected yield. |
| **SLR65** | ta | 49 kept | 4,286 utts | CC BY-SA 4.0 | Already ingested. Keep for continuity with v0 numbers. |

## 3. Checked and rejected (and why)

- **Common Voice Sinhala** — does not exist. Not a size problem; the locale is not in the corpus.
- **VoxLingua107** (si 67 h, ta 51 h, CC BY 4.0) — YouTube segments cut by *automatic* diarization, no reliable speaker identity. Fine for SSL/domain pretraining of the frontend, invalid for trials.
- **FLEURS si_lk / ta_in** — few speakers, no official speaker labels.
- **DISPLACE 2023/2024** (Tamil conversational, IISc) — requires a signed Terms & Conditions submission, so not "openly available"; worth requesting separately since it is the closest thing to conversational Tamil with speaker labels.
- **Microsoft Speech Corpus (Indian languages)** — Tamil/Telugu/Gujarati, but research-only non-commercial licence with mandatory attribution. Usable for a paper, not for the VoiceID product path.
- **IARPA Babel Tamil (LDC2017S13)** — conversational telephone Tamil, the classic SV-style data, but LDC-licensed and paid. Noted for completeness.
- **Sinhala TTS sets** (SafnasKaldeen HF, pnfo/sinhala-tts-dataset, SinhalaVITS) — 1–4 speakers each. Useless for SV.

## 4. What this changes for the ablation plan

The ablation ("which feature/combination improves si/ta performance") is currently
**underpowered on Tamil**. With 49 speakers, the impostor pool is small enough that
the EER confidence interval swamps the effect sizes we are chasing — recall the P3
seed spread was already ±0.10 EER on ta.

Suggested order of work — **steps 1 and 2 are DONE, see §5**:

1. ✅ **Kathbath ta first.** CC0, ASV lists already exist, and the noisy variant doubles
   as a robustness condition. Lowest effort per unit of statistical power.
2. ✅ **SLR127 second** — biggest speaker count, same read-speech domain as v0 so it
   drops into the existing protocol once the speaker-ID convention is confirmed.
   *(Convention confirmed; it also turned out to carry real sessions — §5.)*
3. ⬜ **Common Voice ta third**, ~~specifically to build the first *cross-session* Tamil
   trial list~~ — **[CORRECTED §5: CV cannot provide that; SLR127 already did. Fetch
   it for device/channel diversity instead. Needs a manual browser download.]**
4. ⬜ **SiTa + SLCeleb** as the Sri-Lankan-dialect held-out set. Never train on these.
   *(Both scaffolded; both still gated on external access.)*

Ingestion effort is small: `tools/ingest_openslr.py` is structured as one
`collect_<corpus>()` per source (see `collect_slr52` / `collect_slr65`), so each new
corpus is roughly one 25-line function plus a CLI flag; `tools/sl_dataprep.py` and the
trial-list logic are unchanged.

**Dialect split is a publishable axis, not just a caveat.** Train on Indian Tamil
(SLR127 + Kathbath), evaluate on Sri Lankan Tamil (SLCeleb/SiTa), report the gap.
Nobody has published that number.

## 5. What was actually built (same day, after the survey)

The §4 plan was executed. Five corpora are now prepared as self-contained
VoxCeleb-format datasets under `data/`, one folder each, with
`README.md` + `metadata.json` + `metadata/*.csv` + `lists/` + `wav/` +
`prepare.py`. Shared logic lives in `data/_common/slprep.py`; the index is
`data/README.md`.

| Folder | Lang | Speakers | Utts | Hours | Trials |
|---|---|---|---|---|---|
| `slr52_sinhala` | si | 478 | 185,293 | 224.5 | 12,000 |
| `slr127_tamil` | ta | **531** | 89,401 | 150.1 | 12,000 (true cross-session) |
| `kathbath_tamil` | ta | 60 | 7,171 | 13.2 | **150,000 official** |
| `nisp_tamil` | ta + en | 65 (all bilingual) | 4,905 | 13.3 | 8,000 + **4,000 cross-lingual** |
| `slr65_tamil` | ta | 50 | 4,291 | 7.1 | 12,000 |

291,061 files converted to 16 kHz mono PCM_16, **zero conversion errors**. All
five load through the trainer's own `loadWAV`. Scaffolded but awaiting gated
data: `commonvoice_tamil`, `sita_sinhala_tamil`, `slceleb_sinhala_tamil`.

**The headline claim in §0 held: the Tamil speaker pool went 50 → 706.**

### Corrections the build forced

1. **SLR127's filename convention (was flagged as a risk in §1).** Resolved by
   measurement, not assumption: field 2 has exactly 531 distinct values,
   matching the documented speaker count, so **field 2 is the speaker**. Field 3
   is a *prompt* id reused by up to 312 speakers — using it as an utterance key
   would have collided constantly. The full stem is the utterance key.
2. **SLR127 has genuine sessions — the survey missed this entirely.** Its three
   filename prefixes (`ISTL`, `MICI`, `MILE`) are collection batches, and **107
   of 531 speakers appear under two of them**. Session is therefore mapped to
   the prefix, making its target trials really cross-session (verified: 0
   same-session targets, all 107 speakers contributing, 0 impostor pairs sharing
   a speaker). This is the only *read-speech* corpus here that can do that, and
   it materially raises SLR127's value above what §1 claimed. Cost: the other
   424 speakers are excluded from target sampling (they still serve as impostors
   and training data); `--session_from none` trades that back if wanted.
3. **Common Voice cross-session claim was overstated** — see the inline
   correction. No session field, no timestamp.
4. **SiTa is CC BY-NC 4.0**, not the unspecified "see repo" of §2. Non-commercial
   rules it out of the VoiceID product path — the only such restriction in the
   whole collection, and easy to forget later.
5. **Kathbath's 226 speakers include the 85 GB train split.** The three eval
   splits are 20 speakers each. Its 150,000 trials are statistically powered by
   20 speakers, not by 150,000 pairs — do not read the trial count as power.
6. **NISP has two download traps** the survey didn't anticipate: the Tamil and
   English recordings are in *different* GitHub folders (fetching only the first
   silently yields a monolingual corpus and discards the entire reason to use
   NISP), and the concatenated split archive is **double-gzipped**, so
   `tar -xzf` fails until you decompress twice.

### Decisions worth remembering

- **Kathbath's trials are AI4Bharat's, translated to our paths, not resampled**
  (all 150,000, zero dropped). Regenerating them with `tools/sl_dataprep.py`
  would destroy comparability with published IndicSUPERB numbers, which is the
  main reason to use the corpus.
- **`sita_sinhala_tamil/prepare.py` deliberately emits no trial list.** RTTM
  speaker labels are recording-local, so any list built from it carries the
  same-recording confound. It would look like a benchmark and report flattering
  numbers. Turning SiTa into a real Sri Lankan SV benchmark needs manual
  identity linking across recordings — that remains an open, publishable task.
- **Session semantics differ per corpus and decide whether an absolute EER means
  anything.** Only SLCeleb (and partially SLR127) give real sessions; the
  read-speech corpora use one-session-per-utterance, so their EERs are
  optimistic and valid only for ranking systems on identical trials. The table
  in `data/README.md` records this per dataset.
- **Gender matching is not uniform** — SLR65/Kathbath/NISP have gender labels
  and use same-gender impostors; SLR52/SLR127 do not. Their EERs are not
  directly comparable.
- **Host quirks:** no `ffmpeg`/`sox`/system codecs, so m4a (Kathbath) and mp3
  (Common Voice) decode via pip-installed **PyAV**, which bundles its own ffmpeg
  libraries.

### Still open

`slceleb_sinhala_tamil` remains the binding blocker on any Sri Lankan claim —
only a one-speaker sample is on the mounts. It is the only corpus whose absolute
EER would mean what a reader assumes, and the Indian-trained /
Sri-Lankan-evaluated dialect gap is still the unpublished number this project is
positioned for.

## 6. Sources

- OpenSLR resource index — https://www.openslr.org/resources.php (SLR30, SLR52, SLR65, SLR127)
- IISc-MILE Tamil ASR Corpus — https://www.openslr.org/127/
- IndicSUPERB / Kathbath — https://github.com/AI4Bharat/IndicSUPERB · https://arxiv.org/abs/2208.11761 · https://huggingface.co/datasets/ai4bharat/Kathbath
- IndicVoices-R — https://github.com/AI4Bharat/IndicVoices-R · https://arxiv.org/pdf/2409.05356
- Vaani — https://vaani.iisc.ac.in/ · https://huggingface.co/datasets/ARTPARK-IISc/Vaani · https://arxiv.org/html/2603.28714v1
- Common Voice release stats — https://github.com/common-voice/cv-dataset (`datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json`)
- NISP — https://github.com/iiscleap/NISP-Dataset · https://arxiv.org/abs/2007.06021
- SiTa — https://aclanthology.org/2025.chipsal-1.8.pdf · https://github.com/SiTa-SpeakerDiarization/SiTa
- SPRING-INX — https://arxiv.org/abs/2310.14654
- VoxLingua107 — https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/
- DISPLACE — https://displace2024.github.io/
- Microsoft Speech Corpus (Indian languages) — https://www.microsoft.com/en-us/download/details.aspx?id=105292
