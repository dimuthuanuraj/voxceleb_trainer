---
title: "SL_SPV Errata and Withdrawals"
subtitle: "Corrections required by the annual report, section 10.3"
author: "Dimuthu Anuraj"
date: "2026-09-12"
---

# Why this document exists

The annual research progress report of 2026-09-10 lists six claims that
**must be corrected or withdrawn**. This document is the single
authoritative record of those corrections. Every other project document
should reference this one rather than restating it, so that a correction
made once is a correction made everywhere.

The register exists because section 10.3 was itself written after claims
that should have been withdrawn were carried forward into later documents
in good faith. A grep-able list is the only mechanism that survives an
author who has not read the annual report.

# The register

## E1 --- WITHDRAWN

**Claim:** Any experimental result dated January-June 2026

**Why it is withdrawn or corrected:** Not reproducible from the repository; contradicted by the project audit of 2026-07-03.

**Replacement:** None. This period has no citable experimental record.

*Source: annual report 10.3 item 1, section 5*

No textual occurrence found in the scanned documents.

## E2 --- WITHDRAWN

**Claim:** Nested learning: 9.84 % EER, validated for efficiency

**Why it is withdrawn or corrected:** The measured record is three NaN collapses across three attempts.

**Replacement:** NestedSpeakerNet failed: 3 attempts, NaN x2, 1.09x faster at best.

*Source: annual report 10.3 item 2, section 3.6*

**2 occurrence(s) found in project documents:**

- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:435`
- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:1433`

## E3 --- UNDER REVIEW (G1)

**Claim:** Phase I MLP-Mixer student 10.32 % EER (headline)

**Why it is withdrawn or corrected:** Single seed, unreplicated, predates --deterministic; one log in the same record gives 14.62 %; a duplicated result set is suspected.

**Replacement:** Pending G1: either 3 seeds with an interval, or formal retirement.

*Source: annual report 10.3 item 3; audit item R7*

**47 occurrence(s) found in project documents:**

- `voxceleb_trainer/README_RAW_WAVEFORM_EXPERIMENT.md:74`
- `voxceleb_trainer/README_RAW_WAVEFORM_EXPERIMENT.md:93`
- `voxceleb_trainer/README_RAW_WAVEFORM_EXPERIMENT.md:302`
- `voxceleb_trainer/SAMPLING_RATE_GUIDE.md:276`
- `voxceleb_trainer/SL_LANGUAGE_SPV_ANALYSIS.md:26`
- `voxceleb_trainer/SL_LANGUAGE_SPV_ANALYSIS.md:244`
- `voxceleb_trainer/SL_LANGUAGE_SPV_ANALYSIS.md:537`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:14`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:172`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:173`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:222`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:234`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:239`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:246`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:374`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:400`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:518`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:520`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:527`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:532`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:637`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:684`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:702`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:726`
- `voxceleb_trainer/research_logs/2025-12-30-31-experimental-results-analysis.md:736`
- ... and 22 more

## E4 --- CORRECTED

**Claim:** ResNetSE34L is 34 layers / 6.8 M params / ~8-10 % EER

**Why it is withdrawn or corrected:** The logs give 1.50 M parameters and 15.48 % EER.

**Replacement:** ResNetSE34L: 1.50 M parameters, 15.48 % EER.

*Source: annual report 10.3 item 4*

**2 occurrence(s) found in project documents:**

- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:155`
- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:1439`

## E5 --- CORRECTED

**Claim:** Tamil speaker pool expanded 50 -> 752

**Why it is withdrawn or corrected:** The measured build is 706 speakers, 802 after QC corrections.

**Replacement:** Tamil pool expanded 50 -> 706 (802 after QC correction).

*Source: annual report 10.3 item 5*

**4 occurrence(s) found in project documents:**

- `voxceleb_trainer/research_logs/2026-08-17-progress-report-architecture-frontend-and-corpus-study.md:28`
- `voxceleb_trainer/research_logs/2026-08-22-progress-report-v1-benchmark-complete.md:27`
- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:632`
- `voxceleb_trainer/research_logs/2026-09-10-annual-research-progress-report-may2025-aug2026.md:1441`

## E6 --- CAVEAT REQUIRED

**Claim:** Any v0 EER quoted without caveats

**Why it is withdrawn or corrected:** v0 lists were 100 % closed-set, and read speech is ~5x optimistic against genuine cross-session audio.

**Replacement:** Every v0 EER must carry both the closed-set caveat and the ~5x read-vs-wild optimism factor.

*Source: annual report 10.3 item 6, sections 6.8.1 and 6.12*

No textual occurrence found in the scanned documents.

# The standing rule

> **No number from January to June 2026 may enter the thesis, any paper,
> or any future progress report.**

This is not a judgement about whether the work happened. It is a statement
about evidence: those results are not reproducible from the repository and
are contradicted by the project's own audit. The audit was written by the
project about the project, and nothing external forced it --- which is the
reason every number from July 2026 onward can be quoted.

*Generated 2026-09-12T08:28:58+00:00 by `01_SCRIPTS/tasks/G2_errata.py`. Re-run it after
editing any project document to re-scan for reintroduced claims.*
