# BUGFIX-022 — Accept both camelCase and snake_case for the 4 upstream-inherited argparse names

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-022-camel-snake-case-aliases` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Cosmetic. Inconsistent naming was the audit's framing; the actual user-facing friction is being unable to type snake_case for the four legacy names and getting "unrecognized arguments" from argparse. |
| **Scope** | Three trainers — twelve `add_argument` alias additions and three YAML-key normalization lines. No model, DataLoader, loss, or config-file change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #22 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.3 polish item flagged:

> *"Inconsistent variable casing (`nClasses`, `nDataLoaderThread`,
> `nOut` vs. `lr_decay`)."*

A repo-wide grep over `add_argument` strings categorises every
argparse argument in the project's three trainers:

| Convention | Count | Examples |
|---|---|---|
| **camelCase** | 4 | `nClasses`, `nDataLoaderThread`, `nOut`, `nPerSpeaker` |
| **snake_case** | ~50 | `lr_decay`, `weight_decay`, `max_frames`, `sample_rate`, `eval_streaming`, `augment_chain`, every flag added by BUGFIX-005 through BUGFIX-021, etc. |
| **all-lowercase, no underscore** | 2 | `mixedprec`, `trainfunc` |

The cluster of four camelCase names all share the same `n*` prefix
and clearly trace back to the upstream `voxceleb_trainer` repo where
they were introduced. Every argparse argument added since the project
forked uses snake_case. The two `lowercase-no-underscore` names
(`mixedprec`, `trainfunc`) are also upstream legacy but the §4.3
item does not flag them.

### 1.1 Why a full rename is the wrong fix

The "obvious" cleanup — rename `nClasses` → `n_classes` everywhere —
is much bigger than it looks:

- The four argparse names flow through `**vars(args)` into model and
  loss `__init__` signatures: `def __init__(..., nClasses, nOut,
  nPerSpeaker, ...)`. Renaming touches ~12 model files, 7 loss
  files, the 3 SpeakerNet files, the 3 DataLoader files, the 3
  trainers.
- All 19 config YAML files use the camelCase keys
  (`nOut: 512`, `nClasses: 5994`, etc.). Renaming requires editing
  every one and bumping a config-version field.
- Anyone with a private branch / fork / personal training script
  that does `args.nClasses` or sets `--nClasses 5994` would have
  their workflow break on `git pull`.
- The benefit is purely cosmetic — internal consistency of an
  identifier scheme — for a real cost (every existing reference
  needs an audit).

### 1.2 Why the audit item still matters

The friction the §4.3 item is gesturing at is real, even if
"rename everything" is the wrong response:

- Anyone using the project for the first time (especially someone
  who came in via PEP-8-following Python work) writes
  `--n_classes 5994` and gets the unhelpful argparse error
  `error: unrecognized arguments: --n_classes 5994`.
- Configs hand-written from a snake_case template (anything derived
  from `pytorch-lightning`, `transformers`, etc.) need a manual
  audit and re-keying.

The right fix is **accept both forms** on the user-facing surface
(CLI + YAML), without renaming anything internally. The four
camelCase names remain the canonical Python attribute names; the
snake_case forms are accepted as input aliases.

---

## 2. Fix

### 2.1 CLI: argparse multi-name option

Each of the four camelCase `add_argument` calls in each of the three
trainers gained a snake_case second option string. argparse handles
multi-name options natively — both forms parse into the same `dest`,
which is derived from the *first* option name:

```diff
- parser.add_argument('--nClasses', type=int, default=5991, ...)
+ parser.add_argument('--nClasses', '--n_classes', type=int, default=5991, ...)
```

The snake_case forms used:

| Canonical | Alias |
|---|---|
| `--nClasses` | `--n_classes` |
| `--nDataLoaderThread` | `--n_data_loader_thread` |
| `--nOut` | `--n_out` |
| `--nPerSpeaker` | `--n_per_speaker` |

After the change, both `python trainSpeakerNet.py --nClasses 7000`
and `python trainSpeakerNet.py --n_classes 7000` parse identically.
Either lands in `args.nClasses = 7000` (the original `dest`).

### 2.2 YAML: key normalization at load time

The YAML loader in each trainer gained a one-line normalization
pass *before* the existing merge loop. A module-level dict maps
snake_case YAML keys back to the canonical camelCase form so the
existing `if k in args.__dict__` lookup continues to work:

```python
_CAMEL_YAML_ALIASES = {
    'n_classes': 'nClasses',
    'n_data_loader_thread': 'nDataLoaderThread',
    'n_out': 'nOut',
    'n_per_speaker': 'nPerSpeaker',
}

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    yml_config = {_CAMEL_YAML_ALIASES.get(k, k): v
                  for k, v in yml_config.items()}
    for k, v in yml_config.items():
        ...
```

The `.get(k, k)` fall-through means keys *not* in the alias table
pass through unchanged. The dict is idempotent: applying the
normalization to an already-canonical key dict yields the same dict.

### 2.3 What stays camelCase

Deliberately *not* renamed:

- Python attribute names: `args.nClasses`, `args.nDataLoaderThread`,
  `args.nOut`, `args.nPerSpeaker`.
- Model / loss `__init__` parameter names: `def __init__(..., nClasses=5994, ...)`.
- Config YAML keys that are *currently* camelCase: the existing 19
  configs use `nOut: 512`, `nClasses: 5994`, etc. and continue to
  work.
- `mixedprec` and `trainfunc` (the two lowercase-no-underscore
  legacy names). The §4.3 item explicitly named only the `n*`
  cluster; touching the other two would expand scope without an
  audit-trail mandate.

This is the line between *resolving inconsistency on the input
surface* (done) and *eliminating it from the internal codebase*
(deferred to a future fork-wide rename, if ever).

---

## 3. Verification

### 3.1 Static

All three modified trainers pass `python -m py_compile`.

### 3.2 Behavioural — four end-to-end tests

A standalone test exercised the argparse and dict-normalisation
paths:

| # | Setup | Expectation | Result |
|---|---|---|---|
| 1 | `parser.parse_args(['--nClasses', '7000'])` | `args.nClasses == 7000` | ✅ |
| 2 | `parser.parse_args(['--n_classes', '7000'])` | `args.nClasses == 7000` (same dest) | ✅ |
| 3 | YAML dict `{'n_classes': 7000, 'lr_decay': 0.95, 'n_out': 768}` | Normalises to `{'nClasses': 7000, 'lr_decay': 0.95, 'nOut': 768}` (snake → camel for known aliases; `lr_decay` passes through) | ✅ |
| 4 | YAML dict already in canonical form `{'nClasses': 7000, 'nOut': 768}` | Unchanged | ✅ |

### 3.3 Not verified

- `python trainSpeakerNet.py --help` was not invoked under the
  project's conda env. argparse's help output for a multi-name
  option lists both forms separated by `,` — this is documented
  argparse behaviour and reading the source confirms it; not a
  novel claim.
- A `--nClasses 5994 --n_classes 7000` (both forms on the same
  command line) was not tested. argparse processes them in order,
  so the second `--n_classes 7000` would win — which is the
  expected and correct behaviour, but pathological input.

---

## 4. Backward-compatibility & migration

- **All 19 existing configs work unchanged.** They use the
  canonical camelCase keys and the normalization pass leaves
  unknown keys alone.
- **Every existing CLI invocation works unchanged.**
  `--nClasses 5994` parses exactly as before.
- **Internal Python attribute access is unchanged.** Every
  `args.nClasses`, `args.nOut`, `args.nPerSpeaker`,
  `args.nDataLoaderThread`, and the matching kwargs in model /
  loss / DataLoader signatures continue to work as before.
- **Checkpoints and `state_dict` files** are entirely orthogonal —
  the argparse names never appear in saved checkpoints.
- **New users / new configs can use the snake_case form** without
  any further change. A future config written from scratch may
  prefer `n_classes: 5994` for stylistic reasons; the trainer
  accepts it.

---

## 5. Out-of-scope

- **Renaming the four argparse names** in their original
  declaration. Out of scope for the reasons in §1.1; would be a
  major-version-bump-level change.
- **Aliasing `mixedprec` → `mixed_prec` and `trainfunc` → `train_func`.**
  Not flagged in the §4.3 item. Adding them is a one-line edit each
  but would expand the BUGFIX scope beyond the stated audit
  trigger.
- **`quick_test_validation.py` and `test_validation_phase.py`.**
  Both have their own `parser.add_argument('--nClasses', ...)`
  blocks and could also benefit. Skipped because (a) BUGFIX-020 §1
  noted these scripts still have a stale `result[2]`-as-threshold
  read from before BUGFIX-002 and are likely broken end-to-end,
  and (b) they are test scripts, not user-facing entry points. A
  future BUGFIX-023-style cleanup of those scripts could add the
  same aliases.
- **A schema-generated config validation** that lists *all*
  accepted keys and their aliases. Useful future polish; not
  required to close the §4.3 item.
- **An auto-formatter / lint rule** enforcing snake_case for new
  argparse args. Belongs in a tooling bugfix, not a content
  cleanup.

---

## 6. Rollback plan

If multi-name argparse options cause trouble with a future
argparse variant (highly unlikely — the feature has been stable
since Python 3.0):

1. Revert each `parser.add_argument('--nClasses', '--n_classes', ...)`
   back to `parser.add_argument('--nClasses', ...)` in all three
   trainers. Twelve one-line reverts.
2. Remove the `_CAMEL_YAML_ALIASES` constant and the
   `{_CAMEL_YAML_ALIASES.get(k, k): v for k, v in yml_config.items()}`
   line in each trainer. Three reverts.

A *partial* rollback (drop only the CLI aliases but keep the YAML
normalization) is the more conservative target — the YAML
normalization is purely additive and cannot break any caller.

---

## 7. Related items in §4.3 of the analysis

This fix closes item **#22**. Roadmap state:

| # | Title | Status |
|---|---|---|
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| 22 | Inconsistent variable casing (`nClasses` vs `lr_decay`) | ✅ **This document** |
| 23 | `dataprep.py` MD5 mismatch raises `Warning` as exception | ✅ [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md) |
| 24 | `analyze_performance.py` / `benchmark_performance.py` / `quick_optimize.py` overlap | ✅ [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md) |
| 25 | Each model duplicates `PreEmphasis` / mel / InstanceNorm | ✅ [BUGFIX-025](BUGFIX-025-shared-audio-frontend.md) |
| 26 | `exps/` folders mostly contain `logs/` / `result/` but no checkpoints | ✅ [BUGFIX-026](BUGFIX-026-exps-cleanup-policy.md) |

§4.1 (Critical) is fully closed (BUGFIX-001..010).
§4.2 (Important) is closed except for **#12** (`lists/` empty — Sri Lankan dataprep workstream).
§4.3 (Minor / polish) — two items closed; four remaining.

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #22.
- **argparse multi-name option syntax:**
  https://docs.python.org/3/library/argparse.html#name-or-flags —
  multiple option strings share a `dest` derived from the first
  long option.
- **PEP 8 on naming:**
  https://peps.python.org/pep-0008/#function-and-variable-names —
  `lower_case_with_underscores` is the recommended convention for
  Python identifiers. The four legacy names predate this project's
  adoption of that convention; this fix accepts both forms on the
  user-facing surface without forcing a rename.
- **Upstream `voxceleb_trainer`:**
  https://github.com/clovaai/voxceleb_trainer — origin of the four
  camelCase `n*` names. Any future merge from upstream will continue
  to use them; the alias layer absorbs the impedance mismatch.
- **Related fixes:**
  [BUGFIX-013](BUGFIX-013-portable-config-paths.md) added
  `${VAR}` expansion to the YAML loader; this fix extends the same
  loader with a key-aliasing pass. They are independent — env-var
  expansion runs on values, alias normalization runs on keys.
  [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) generalised
  the loader to accept dict/list values for structured config; this
  fix is its peer for the *key* axis.
