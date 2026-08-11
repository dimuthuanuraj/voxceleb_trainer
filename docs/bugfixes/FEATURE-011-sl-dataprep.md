# FEATURE-011 — SL corpus preparation tool (closes §4.2 #12)

**Status:** complete
**Type:** tooling (not a runtime feature)
**Relates to:** §4.2 #12 of SL_LANGUAGE_SPV_ANALYSIS.md (the **last open audit item**); RUN_GUIDE.md (the experimental protocol that consumes the outputs)
**Owner:** SL-SPV
**Touches:** `tools/sl_dataprep.py` (new), `RUN_GUIDE.md` (new)

## 1. Motivation

§4.2 #12 of the analysis doc has been ⬜ Not started for the entire audit:

> "Add `sl_dataprep.py` and per-language `test_list_*.txt`. — Cannot be sensibly written before the pilot corpus exists."

With the user now reporting they have 100 SL speakers (~60 Sinhala + 40 Tamil) assembled, the prerequisite is met. This feature ships the prep tool plus the experimental protocol that exercises every previously-shipped feature against the new corpus.

## 2. What this ships

`tools/sl_dataprep.py` — a single-file script that walks a directory tree shaped as
`<corpus_root>/<lang>/<speaker_id>/<utterance>.wav` and produces nine output files:

| File | Consumer | Format |
|---|---|---|
| `train_list.txt` | trainer | `<spk_int> <relpath>` |
| `test_list.txt` | pooled eval | `<label> <enrol> <test>` |
| `test_list_si.txt` / `_ta.txt` / `_cs.txt` | FEATURE-003 per-lang eval | same |
| `spk_lang_lookup.txt` | FEATURE-007 / FEATURE-008 | `<spk_int> <lang_int>` |
| `asnorm_cohort.txt` | FEATURE-002 | one wav path / line |
| `plda_train_list.txt` | FEATURE-010 | `<spk_int> <relpath>` |
| `speakers.csv` | reproducibility metadata | header + rows |

Per-utterance train/test split is 80/20 within each speaker; trial pairs are sampled
balanced (1 target : 1 impostor by default). Cross-lingual trials are generated automatically
for speakers present in ≥2 of the configured languages. A single `--seed` controls every
random decision (split, sampling, cohort selection).

The script is the missing piece needed to drive the protocol in RUN_GUIDE.md — every
shipped feature has a corresponding output that consumes the prep result.

## 3. Smoke test

Synthetic corpus: 60 Sinhala speakers × 20 utterances + 40 Tamil speakers × 20 utterances =
100 speakers / 2000 utterances total. Running the prep tool with default flags produced
exactly the expected outputs:

```
train_list.txt:     1600 utterances (80% of 2000)
test_list.txt:      2000 trial pairs (1000 si + 1000 ta, 1:1 target:impostor)
test_list_si.txt:   1000 pairs
test_list_ta.txt:   1000 pairs
test_list_cs.txt:   0 pairs    ← no speakers shared between si and ta in this synthetic
spk_lang_lookup.txt: 100 speakers labelled
asnorm_cohort.txt:  50 utterances (configured --cohort_size 50)
plda_train_list.txt: 1600 utterances
speakers.csv:       100 rows of reproducibility metadata
```

The cross-lingual zero-count is correct behaviour for this synthetic corpus where no
speaker exists in both languages. Real SL data with multilingual speakers will produce
non-zero cross-lingual trials automatically.

## 4. Usage

See RUN_GUIDE.md §4 for the full invocation. Minimal form:

```bash
python tools/sl_dataprep.py \
    --corpus_root $SL_SPV_DATA_ROOT/sl_celeb \
    --out_dir     $SL_SPV_DATA_ROOT/sl_celeb/lists \
    --langs       si ta \
    --seed        42
```

## 5. Closing the audit

§4.2 #12 is now ✅ Closed. The §4 audit table is fully complete — 26 of 26 bug-level
items and FEATURE-011 collectively address every line in the original §4 of the
analysis doc.

## 6. Cross-references

- `RUN_GUIDE.md` — the experimental protocol that consumes these outputs.
- All FEATURE-001 .. FEATURE-010 docs — each consumes one or more of the prep outputs.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 #12 — original audit item, now resolved.
