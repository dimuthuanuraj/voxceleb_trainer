# BUGFIX-016 — Make augmentation probabilities configurable

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-016-configurable-augment-chain` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Medium — augmentation regimen is one of the largest accuracy levers for speaker verification (the MinDCF guide ascribes a 5–10% relative MinDCF reduction to it), and the hard-coded uniform 20/20/20/20/20 split has no a-priori reason to be optimal for Sri Lankan corpora that differ from VoxCeleb in microphone, room, and noise statistics. |
| **Scope** | Three trainers (argparse arg + YAML loader passthrough), two DataLoaders (helper + `train_dataset_loader.__init__` + `__getitem__` dispatch). No model-side, scheduler, or config-file change. Existing configs continue to work with statistically identical behaviour. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #16 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The train-time augmentation dispatcher in both
[`DatasetLoader.py`](../../DatasetLoader.py) and
[`DatasetLoader_performance_updated.py`](../../DatasetLoader_performance_updated.py)
selected one of five outcomes uniformly:

```python
augtype = random.randint(0, 4)
if   augtype == 1: audio = self.augment_wav.reverberate(audio)
elif augtype == 2: audio = self.augment_wav.additive_noise('music', audio)
elif augtype == 3: audio = self.augment_wav.additive_noise('speech', audio)
elif augtype == 4: audio = self.augment_wav.additive_noise('noise', audio)
# augtype == 0  →  clean (no augmentation)
```

`random.randint(0, 4)` is uniform over `{0, 1, 2, 3, 4}`, so the
probabilities are exactly `0.2` each: clean, RIR-reverberation, music
noise, speech (babble) noise, MUSAN noise. The §4.2 prescription was:

> *"`random.randint(0,4)` in `DatasetLoader.py:150` — probabilities
> are not configurable. Add a config block (`augment_chain: {noise: 0.3,
> music: 0.2, ...}`) as the MinDCF guide already recommends."*

The line-number referenced (`DatasetLoader.py:150`) has shifted to
line **226** in the current tree (it was `_pad_short_with_dither` and
`_resample_if_needed` from BUGFIX-005 / BUGFIX-007 that pushed it
down). The bug itself is exactly as described. The
`MINDCF_IMPROVEMENT_GUIDE.md` §3.2 already documented the desired
config-block shape and quotes a 5–10% relative MinDCF reduction from
tuning the probabilities — but no code consumed the block.

### 1.1 Why the uniform default is the wrong default for SL data

The five categories are not symmetric in their effect on the model:

- **`clean`** preserves the source distribution. With `p=0.2` the
  model sees clean audio in only 1 in 5 minibatches.
- **`reverb`** simulates room impulse responses from a generic corpus
  (RIRS_NOISES). Sri Lankan recordings are commonly indoor /
  small-room; the VoxCeleb-derived RIR mix may underweight that.
- **`music`** and **`speech`** are domain-specific contaminants —
  `speech` (babble) helps for cocktail-party scenarios; `music`
  helps for media-source clips.
- **`noise`** is MUSAN's generic background noise, which is the
  closest match to the random environmental noise common in field
  recordings.

A hard uniform 0.2 / 0.2 / 0.2 / 0.2 / 0.2 is a sensible starting
point for VoxCeleb-shaped data, not a defensible default for a Sri
Lankan deployment where (say) babble is rare and ambient noise is the
dominant nuisance. The fix lets the practitioner reweight without
editing source.

---

## 2. Fix

### 2.1 New surface: `augment_chain`

A new argparse argument, accepted by all three trainers, with three
input forms:

```yaml
# YAML, dict form (the user's prescription, the natural shape):
augment_chain:
  noise:  0.3
  music:  0.2
  reverb: 0.2
  speech: 0.2
# implicit: clean = 1.0 - 0.9 = 0.1
```

```yaml
# YAML, list-of-single-key-dicts form (matches MINDCF_IMPROVEMENT_GUIDE.md §3.2):
augment_chain:
  - noise:  0.3
  - reverb: 0.3
  - music:  0.2
  - speech: 0.2
```

```bash
# CLI, JSON-string form (since argparse can't natively carry a dict):
python trainSpeakerNet.py --augment_chain '{"noise": 0.3, "music": 0.2}'
```

All three forms normalise to the same internal representation: a
5-tuple of probabilities aligned with the index ordering used by the
dispatch (`(clean, reverb, music, speech, noise)`). Empty / unset /
`"uniform"` falls back to `(0.2, 0.2, 0.2, 0.2, 0.2)`, which is
statistically identical to the legacy `random.randint(0, 4)`.

### 2.2 Module-level helper `_parse_augment_chain`

Added to both DataLoader files, immediately below `_resample_if_needed`
(the existing module-level audio helpers' neighbourhood). It encodes
the full validation surface:

- Empty / `""` / `None` / `"uniform"` → legacy uniform 0.2 each.
- `dict` of `{label: prob}` → labels must be a subset of
  `_AUGMENT_LABELS = ("clean", "reverb", "music", "speech", "noise")`.
- `list` of single-key `dict` entries → flattened to a dict.
- `str` → parsed as JSON first, then validated as one of the above.

Validation, in order, with descriptive errors:

| Check | Failure mode |
|---|---|
| Spec is `dict` or convertible to one | `ValueError` naming the type received. |
| All keys are in `_AUGMENT_LABELS` | `ValueError` listing the unknown labels and the valid set. |
| All values are non-negative numbers (rejects `True`/`False` despite being numeric) | `ValueError` naming the offending key and value. |
| Sum ≤ 1.0 + 1e-6 (float tolerance) | `ValueError` reporting the actual sum. |
| Final sum > 0 (at least one outcome can fire) | `ValueError` — pathological case where the user sets every label to 0 and overrides the implicit clean. |

The remainder of `1.0 - sum(spec.values())` is automatically assigned
to `"clean"` if `"clean"` was not specified — the typical user
mindset is "what fraction of batches do I want corrupted with X?",
not "what fraction do I want clean?", so the implicit-clean default
matches that mental model.

A final pass divides by the sum to absorb floating-point drift, so
the output 5-tuple is guaranteed to sum to exactly 1.0 at float
precision.

### 2.3 DataLoader dispatch

Inside `train_dataset_loader.__init__`:

```python
self._augment_probs = _parse_augment_chain(kwargs.get("augment_chain"))
```

Inside `train_dataset_loader.__getitem__`:

```python
if self.augment:
    augtype = random.choices(
        range(len(_AUGMENT_LABELS)),
        weights=self._augment_probs,
        k=1,
    )[0]
    if   augtype == 1: audio = self.augment_wav.reverberate(audio)
    elif augtype == 2: audio = self.augment_wav.additive_noise('music',  audio)
    elif augtype == 3: audio = self.augment_wav.additive_noise('speech', audio)
    elif augtype == 4: audio = self.augment_wav.additive_noise('noise',  audio)
    # augtype == 0 → clean
```

`random.choices` is stdlib (Python 3.6+) — no new dependency. The
index ordering is preserved exactly, so the four `elif` branches did
not need to be renumbered.

### 2.4 YAML loader passthrough for structured values

The trainers' YAML merge loop previously did
`args.__dict__[k] = typ(v)` for every key, where `typ` came from
`find_option_type(k, parser)` — for `augment_chain` that is `str`, and
`str({'noise': 0.3, ...})` produces `"{'noise': 0.3, ...}"` (Python
`repr`, not JSON, and unparseable as JSON).

The fix adds a single conditional for dict / list values:

```python
for k, v in yml_config.items():
    if k in args.__dict__:
        v = _expand_env_vars(v, k)
        if isinstance(v, (dict, list)):
            # Structured YAML values (e.g., augment_chain dict per
            # BUGFIX-016) pass through without scalar type coercion.
            args.__dict__[k] = v
        else:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
    else:
        sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")
```

This is a small generalisation of BUGFIX-013's loader change. It
benefits any future config key whose natural YAML representation is
structured — e.g., a hypothetical `loss_weights: {...}` or
`scheduler_milestones: [...]` — without further loader work.

---

## 3. Verification

### 3.1 Unit tests of `_parse_augment_chain`

Eight cases were exercised by extracting the helper from the file and
invoking it directly under the audit-session Python (no `torch`
required for the parser):

| # | Input | Expected | Result |
|---|---|---|---|
| 1 | `None`, `""`, `"uniform"` | `(0.2, 0.2, 0.2, 0.2, 0.2)` | ✅ |
| 2 | `{'noise': 0.3, 'music': 0.2, 'reverb': 0.1}` (sum 0.6) | `(0.4, 0.1, 0.2, 0.0, 0.3)` — remainder to clean | ✅ |
| 3 | `[{'noise': 0.3}, {'reverb': 0.3}, {'music': 0.2}, {'speech': 0.2}]` (sum 1.0) | `(0.0, 0.3, 0.2, 0.2, 0.3)` | ✅ |
| 4 | JSON string `'{"noise": 0.5, "clean": 0.5}'` | `(0.5, 0.0, 0.0, 0.0, 0.5)` | ✅ |
| 5 | `{'reverberation': 0.5}` (typo on label) | `ValueError("unknown labels")` | ✅ |
| 6 | `{'noise': 0.6, 'music': 0.5}` (sum > 1) | `ValueError("sum…> 1.0")` | ✅ |
| 7 | `{'noise': -0.1}` | `ValueError("non-negative")` | ✅ |
| 8 | `"not json"` (CLI typo) | `ValueError("not parseable as JSON")` | ✅ |

### 3.2 Backwards compatibility at the statistical level

Default behaviour (no `augment_chain` set, or set to empty / uniform)
produces the same probability vector `(0.2,)*5` as the original
`random.randint(0, 4)`. The two underlying RNG calls
(`random.randint` vs `random.choices` with uniform weights) consume a
different number of bits from the underlying Mersenne Twister, so the
*sequence* of outcomes for a given seed will differ — but the
*distribution* is identical, so any aggregate training metric is
unaffected. For paper-grade exact-reproduction of a prior run, this
turn's BUGFIX-017 `--deterministic` flag is the appropriate hook.

### 3.3 Static / syntax

`python -m py_compile` returned exit 0 on all five touched files:
`DatasetLoader.py`, `DatasetLoader_performance_updated.py`,
`trainSpeakerNet.py`, `trainSpeakerNet_performance_updated.py`,
`trainSpeakerNet_distillation.py`.

### 3.4 Not verified

- No real training run was exercised; the audit-session Python lacks
  the project's runtime dependencies. The default-equivalence claim
  (§3.2) and the unit tests (§3.1) are the relevant proofs.
- The MinDCF guide's 5–10% relative improvement claim was *not*
  independently reproduced — this fix delivers the configurability
  the guide assumed; whether any specific reweighting improves a
  given downstream benchmark is a follow-up experiment.

---

## 4. Backward-compatibility & migration

- **Existing configs continue to work unchanged.** None of the 19
  configs under [`configs/`](../../configs/) set `augment_chain`
  today, so all of them implicitly receive the legacy uniform
  distribution. No config edit is required for the fix to land safely.
- **Existing CLI invocations work unchanged.** `--augment_chain` is
  optional with default `""`; any script that does not pass the flag
  is unaffected.
- **Existing checkpoints load unchanged.** This is a data-pipeline
  change, not a model change — `state_dict` keys are untouched.
- **The aggregate training distribution is identical under default
  settings**, so no metric drift is expected when running an existing
  config on the new code.
- **MINDCF_IMPROVEMENT_GUIDE.md §3.2** is no longer aspirational — the
  block it documents now has a consumer. No edit to that guide is
  required, but a reader who wants to act on it can now do so.

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Per-augmentation parameter blocks** (e.g., a sub-dict with
  `noise_snr_range` per outcome). Out of scope — the §4.2 report
  asked for *probabilities*, which is exactly what this fix delivers.
  Per-augmentation parameters belong in `AugmentWAV` itself and would
  warrant a separate bugfix.
- **Per-language / per-corpus presets.** A future research log might
  recommend an "SL_deploy" preset for Sri Lankan recordings, but the
  fix's job is to make presets *expressible*, not to ship them.
- **Validation that `musan_path` / `rir_path` actually exist if the
  user assigns non-zero probability to those categories.** The
  existing code already raises at `AugmentWAV.__init__` time when
  the paths are missing, regardless of how the probabilities are set.
- **Augmentation chain composition** (i.e., applying multiple
  categories to the same clip, e.g., reverb + music). The current
  dispatch is *mutually exclusive*, matching the legacy behaviour.
  Composition is a richer change and worth its own bugfix if a
  research result motivates it. The label `augment_chain` is
  unfortunately slightly misleading in this respect — but it matches
  the existing `MINDCF_IMPROVEMENT_GUIDE.md` vocabulary and the §4.2
  prescription, so the naming is preserved.
- **Updating existing configs to use the new feature.** Out of scope
  by design — the fix is the *capability*; populating configs with
  empirically-tuned probabilities is downstream research work.

---

## 6. Rollback plan

If the new dispatch turns out to be problematic (e.g., subtle RNG
sequence change matters for a published baseline):

1. Revert the `__getitem__` dispatch in both DataLoader files to the
   original `random.randint(0, 4)` form.
2. Remove the `self._augment_probs = ...` line from
   `train_dataset_loader.__init__`.
3. Remove `_AUGMENT_LABELS` and `_parse_augment_chain` from both
   DataLoader files.
4. Remove the `--augment_chain` argparse line from all three
   trainers.
5. Leave the YAML-loader dict/list passthrough in place — it is a
   useful generalisation independent of this fix and breaks nothing.

A *partial* rollback (steps 1–4 only, keeping the loader change) is
the most conservative target. The infrastructure stays available for
any future structured-config consumer.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#16** of the §4.2 list. Roadmap state:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) |
| 15 | `RawNet3.py` debug print + in-place mutation | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ **This document** |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |
| 20 | EER definition differs from common `(fpr+fnr)/2` | ✅ [BUGFIX-020](BUGFIX-020-eer-definition-disclosure.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #16.
- **Documented in advance of code:**
  [`MINDCF_IMPROVEMENT_GUIDE.md`](../../MINDCF_IMPROVEMENT_GUIDE.md)
  §3.2 — described the desired config-block shape and the expected
  5–10% relative MinDCF improvement.
- **`random.choices` semantics:**
  https://docs.python.org/3/library/random.html#random.choices —
  weighted sampling with replacement; weights need not sum to 1
  (they are interpreted as relative), but this fix normalises to
  exact 1.0 for clarity.
- **Related fixes:**
  [BUGFIX-013](BUGFIX-013-portable-config-paths.md) — added env-var
  expansion to the YAML loader; this fix extends the same loader with
  dict/list passthrough.
  [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) — adds
  `--deterministic`, the right hook for anyone needing exact RNG
  reproduction now that the augmentation outcome consumes weighted
  rather than uniform random bits.
