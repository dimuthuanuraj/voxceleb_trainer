# BUGFIX-010 — Quarantine `NestedSpeakerNet` (does not converge)

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-010-quarantine-nestedspeakernet` |
| **Date** | 2026-05-16 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Critical — `model: NestedSpeakerNet` configs train a model that has been empirically shown to either NaN-crash or stabilise at +88% worse EER than the production baseline. |
| **Scope** | One model file move, one new directory README, three config files annotated, one legacy-doc quarantine notice, plus a cross-reference path update in BUGFIX-006. No runtime behaviour change for any other model. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #9 |
| **Status** | ✅ Fixed (model moved to `models/experimental/`; configs updated; reference docs written) |

---

## 1. Problem

`models/NestedSpeakerNet.py` implemented a multi-path "nested" architecture
inspired by the *Nested Learning* paper: each level aggregates features
from **all** previous levels rather than only the immediately preceding
one. Training the model across three independent stabilisation attempts
produced the following empirical record (full details in
[`research_logs/2025-12-29-nested-learning-experiment.md`](../../research_logs/2025-12-29-nested-learning-experiment.md)):

| Attempt | Configuration | Outcome | Best EER before failure |
|---|---|---|---|
| 1 | Full nested, fixed 0.5× scaling, BatchNorm, batch 32 | NaN cascade at epoch 11 | 21.72% |
| 2 | Learnable nested weights + GroupNorm + adaptive pooling, batch 48 | NaN cascade at epoch 12 | 18.71% |
| 3 | Simplified nested (only deep levels connected) | Stable but **+88% worse than baseline** | 29.03% (vs. ResNetSE34L baseline 15.48%) |

### 1.1 Root cause (per the research log)

The research log identifies two compounding factors that hyperparameter
tuning cannot fix:

1. **Gradient-path-count explosion.** A nested aggregation over `N`
   levels creates O(2^N) distinct gradient paths through the network
   (16 paths for the 4-level config, 32 for the 5-level config). For
   variable-length audio features whose per-frame variance is roughly
   an order of magnitude higher than image features
   (σ² ≈ 2.5–8.0 vs. ≈ 0.1–0.3), the sum of those path contributions
   is large enough that gradient clipping at `max_norm=5.0` cannot tame
   it. AAM-Softmax also concentrates gradient mass on the few hardest
   classes, amplifying the effect.
2. **Anti-correlated audio features.** Adjacent levels in audio CNNs
   have an empirical Pearson correlation of r ≈ −0.23 (vs. r ≈ +0.65
   in vision CNNs). The nested concat/add operation therefore
   *amplifies* feature norms instead of *regularising* them, which is
   the opposite of the dynamic that makes nested networks work for
   image classification.

`NESTED_ARCHITECTURE_FIXES.md` documents the seven stabilisation
techniques tried (learnable weights, adaptive pooling, GroupNorm,
dropout, larger batches, gentler LR decay, gradient clipping). All
seven were applied; none of them converted attempts 1/2 into a stable
training run.

### 1.2 Why a code fix isn't appropriate here

This bug is **not the kind of bug a code edit can repair**. The
architecture's failure mode is information-theoretic (gradient-path
combinatorics × audio statistics), not implementation-level. Three
stabilisation passes already failed; a fourth implemented in this
repository would be re-doing the experimental work that the research
log already concluded. The remaining engineering responsibility is
therefore:

- Stop the failing architecture being trained by accident.
- Make the failure history easy to find from every file that touches
  the model.
- Preserve the code and configs as research artifacts so the negative
  result is not lost.

That is the scope of this fix.

---

## 2. Fix

### 2.1 Move the model out of `models/`

```
models/NestedSpeakerNet.py  →  models/experimental/NestedSpeakerNet.py
```

The trainer loads models dynamically via
`importlib.import_module("models." + model)`, so moving the file is
sufficient to make `model: NestedSpeakerNet` (without the
`experimental.` prefix) raise `ModuleNotFoundError` at config-load
time. This is exactly the desired behaviour: a config that asks for
the production-namespace name now fails loudly instead of silently
training a non-convergent model.

Configs that opt in to training the quarantined model use the
qualified module path:

```yaml
model: experimental.NestedSpeakerNet
```

which `importlib.import_module("models.experimental.NestedSpeakerNet")`
resolves correctly because `models/experimental/` is a regular Python
package.

### 2.2 In-file quarantine banner

A header banner was prepended to
[`models/experimental/NestedSpeakerNet.py`](../../models/experimental/NestedSpeakerNet.py)
that:

- Marks the file as quarantined.
- Records all three failure attempts with epoch numbers and EER
  values.
- States the diagnosed root cause (gradient-path count × audio-feature
  anti-correlation).
- Links to the research log and to this BUGFIX-010 doc.
- Directs anyone considering reviving the architecture to a follow-up
  experiment design instead of hyperparameter tuning.

### 2.3 Directory README

[`models/experimental/README.md`](../../models/experimental/README.md)
explains:

- What `models/experimental/` is for.
- The promotion-out and demotion-in criteria (both require empirical
  evidence — promotion requires baseline-matching EER, demotion
  requires root-caused failure).
- Guidance for anyone considering reviving a quarantined model.
- The three independent reasons we don't simply delete the code
  (negative results are valuable; research logs cite the paths; the
  architecture-diagram tooling references the file).

### 2.4 Config annotations

The three nested configs

- [`configs/nested_4level.yaml`](../../configs/nested_4level.yaml)
- [`configs/nested_4level_asp.yaml`](../../configs/nested_4level_asp.yaml)
- [`configs/nested_5level_asp.yaml`](../../configs/nested_5level_asp.yaml)

each received a header banner stating that the config trains a
non-convergent model, plus the `model:` value was updated to
`experimental.NestedSpeakerNet`. The 5-level config additionally
explains why 5 levels is *strictly worse* than 4 (2^5 = 32 gradient
paths vs. 2^4 = 16) so future readers don't see "5 levels" as a
recovery direction.

### 2.5 Legacy doc quarantine

[`NESTED_ARCHITECTURE_FIXES.md`](../../NESTED_ARCHITECTURE_FIXES.md) at
the repo root was written *before* the experimental verdict came in
and presents seven stabilisation techniques as if they were working
fixes. A quarantine notice was added to the top of the document
clarifying that **none** of the listed "fixes" produced a converging
training run, and pointing to this doc and the research log. The body
of the file is preserved as-is below the notice so the audit trail of
*what was tried* remains intact.

### 2.6 Cross-reference path update

The path reference to `models/NestedSpeakerNet.py` in
[`BUGFIX-006`](BUGFIX-006-model-side-sample-rate-threading.md) (model-side
sample-rate threading, which also touched the nested model) was
updated to point at the new `models/experimental/NestedSpeakerNet.py`
path so the documented file list stays accurate.

---

## 3. Verification

This fix is structural rather than algorithmic, so verification is
likewise structural:

- **Import resolution.** `importlib.import_module("models.experimental.NestedSpeakerNet")`
  succeeds (the file exists at the new path; `models/experimental/`
  is importable). `importlib.import_module("models.NestedSpeakerNet")`
  raises `ModuleNotFoundError`, which is the desired behaviour for
  any config that hasn't been migrated.
- **Config validation.** All three nested configs now name
  `experimental.NestedSpeakerNet`, which matches the moved file.
  Searching the repo for the unqualified `model: NestedSpeakerNet`
  string returns zero hits, so no remaining config points at the
  removed location.
- **Reference graph.** Every file that mentions the model
  (`NESTED_ARCHITECTURE_FIXES.md`, the three configs, the model file
  itself, `models/experimental/README.md`, `test_nested_architecture.py`,
  and BUGFIX-006) now either cites BUGFIX-010 by name or carries a
  quarantine banner. There are no dangling references to the old
  `models/NestedSpeakerNet.py` path.

What this fix does **not** verify:

- It does not verify that the architecture is unfixable. That is an
  open research question; the empirical record so far is "three
  attempts failed", not "proof of impossibility". The quarantine is a
  *safety* claim ("don't train this by accident"), not an
  *impossibility* claim.

---

## 4. Backward-compatibility & migration

- **Configs that name `NestedSpeakerNet` without the `experimental.`
  prefix will break.** That is the point — they were training a model
  with documented NaN cascades. Any external config kept outside this
  repository must be migrated to `model: experimental.NestedSpeakerNet`
  to keep working.
- **No other model is affected.** The `models/` directory is otherwise
  unchanged; production models (`ResNetSE34L`, `ResNetSE34V2`,
  `VGGVox`, `MLPMixerSpeaker`, `MLPMixerSpeaker_RawWaveform`,
  `RawNet3`, `LSTMAutoencoder`) all keep their previous import paths.
- **Existing training runs and checkpoints.** Any checkpoint produced
  by a previous run of `NestedSpeakerNet` is still loadable into the
  moved file (the class definition is unchanged; only its module path
  changed). The `state_dict` keys depend on the class layout, not the
  module path. Users who genuinely want to resume such a run can do
  so by setting `model: experimental.NestedSpeakerNet`.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done in this fix:

- **Outright deletion of the file.** Rejected: deleting forces every
  future contributor with the same architectural idea to re-discover
  the failure, breaks research-log citations, and breaks the
  `visualize_nested_architecture.py` tooling. See
  `models/experimental/README.md` §"Why we don't just delete these
  files" for the full reasoning.
- **A fourth stabilisation attempt.** Rejected: the diagnosed root
  causes (gradient-path count, audio-feature anti-correlation) cannot
  be fixed by hyperparameter tuning, and the research log already
  exhausted the obvious architectural mitigations. Any further work
  belongs in a new research branch with an experiment design that
  targets the root causes directly, not in this audit pass.
- **Renaming or refactoring the model class itself.** Out of scope —
  the goal of this fix is to mark the failure, not to rewrite the
  code. The class signature is preserved so the historical
  reproducibility of the negative result remains intact.
- **Deleting `NESTED_ARCHITECTURE_FIXES.md` outright.** Rejected: the
  document records *what was tried*, which is the bulk of the
  experimental value. Quarantining the top of the file with a
  redirect notice was sufficient.

---

## 6. Rollback plan

If a future research result genuinely revives the architecture (i.e.,
the four promotion criteria in `models/experimental/README.md` are
all met):

1. Move `models/experimental/NestedSpeakerNet.py` back to
   `models/NestedSpeakerNet.py`.
2. Remove the quarantine banner from the file header and replace it
   with a brief note pointing at the BUGFIX/research log that
   established the working configuration.
3. Update the three configs to drop the `experimental.` prefix and
   remove their banner headers; replace with a citation of the
   working-configuration doc.
4. Mark the entry in `models/experimental/README.md`'s status table as
   promoted (or remove it if no other quarantined models remain).
5. Update the quarantine notice in `NESTED_ARCHITECTURE_FIXES.md` to
   reflect the new status — but keep the historical body, since it
   still records the path the project actually walked.

Rolling back **without** new evidence (i.e., just because someone
wants to train it again) is explicitly not supported — the
`models/experimental/` README's promotion criteria exist to prevent
exactly that.

---

## 7. Related items in §4.1 of the analysis

This fix closes item **#9** of the §4.1 Critical list. Summary of the
full list as of this document:

| # | Title | Status |
|---|---|---|
| 1 | `mp.spawn` kwarg typo | ✅ BUGFIX-001 |
| 2 | `result[2]` used as a threshold | ✅ BUGFIX-002 |
| 3 | TEER/TAcc displays 0% when `nPerSpeaker > 1` | ✅ BUGFIX-003 |
| 4 | Hard `.cuda()` call in `SpeakerNet.forward` | ✅ BUGFIX-004 |
| 5 | 16 kHz hard-coded in `loadWAV` | ✅ BUGFIX-005 |
| 6 | `numpy.pad(..., 'wrap')` for short audio | ✅ BUGFIX-007 |
| 7 | SincConv buffer placement in `MLPMixerSpeaker_RawWaveform` | ✅ BUGFIX-008 |
| 8 | `MaxPool1d(...) if pool else False` returns a literal `False` | ✅ BUGFIX-009 |
| 9 | NestedSpeakerNet NaN explosions | ✅ **This document** |

Item §4.1 is now fully closed; subsequent fixes draw from §4.2 of the
analysis.

The model-side sample-rate threading work in
[BUGFIX-006](BUGFIX-006-model-side-sample-rate-threading.md) touched
the nested model file under its old path. The path reference in that
doc has been updated; the threading work itself is preserved at the
new location because the model file was moved unchanged.

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit summarised in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #9.
- **Empirical evidence:**
  [`research_logs/2025-12-29-nested-learning-experiment.md`](../../research_logs/2025-12-29-nested-learning-experiment.md)
  records all three training attempts, EER curves, NaN events, and
  diagnosed root causes.
- **Stabilisation attempts (preserved verbatim):**
  [`NESTED_ARCHITECTURE_FIXES.md`](../../NESTED_ARCHITECTURE_FIXES.md)
  enumerates the seven techniques tried in attempt 2.
- **Quarantine policy:**
  [`models/experimental/README.md`](../../models/experimental/README.md)
  defines the promotion / demotion criteria for this directory.
- **Source paper hypothesis (not achieved in practice):** *Nested
  Learning: The Illusion of Deep Learning Architecture*. Adapted in
  attempts 1–3 with audio-specific modifications; the adaptation is
  what failed, not necessarily the underlying paper.
