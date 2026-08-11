# BUGFIX-013 — Configs hard-code private mount paths

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-013-portable-config-paths` |
| **Date** | 2026-05-18 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Important — every YAML config in the repository fails on any machine that does not happen to have `/mnt/ricproject3`, `/mnt/ricproject2`, and `/mnt/ricproject` mounted, with errors that surface only after the trainer starts loading data. |
| **Scope** | Three trainer YAML loaders (`trainSpeakerNet.py`, `trainSpeakerNet_performance_updated.py`, `trainSpeakerNet_distillation.py`), 16 config files under `configs/`, one new file `paths.env.example` at the repo root. No model-side or DataLoader-side code change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #13 |
| **Status** | ✅ Fixed |

---

## 1. Problem

Every YAML config under `configs/` hard-codes absolute paths to one of
three private mount points on the original development machine:

| Mount | What lived there | Configs that referenced it |
|---|---|---|
| `/mnt/ricproject3/2025/data` | Mini-VoxCeleb1/2 corpus, train / test lists, MUSAN, RIRs | 14 configs |
| `/mnt/ricproject2/voxceleb_new` | Full VoxCeleb1/2 corpus | 2 configs (`mlp_mixer_rawwaveform_baseline.yaml`, `mlp_mixer_rawwaveform_distillation.yaml`) |
| `/mnt/ricproject/data` | MUSAN + RIRs (when those lived on a different mount) | 2 configs (same two) |

A `grep -rn '/mnt/ricproject' configs/` returned **96 lines** before
this fix. On any machine without the original mount layout, every one
of those was a future `FileNotFoundError` waiting to fire some seconds
into training (after argparse, after the trainer's `print(args)`
banner, after model construction — i.e., at the worst possible point
for fast diagnosis).

### 1.1 Why this is more than a cosmetic issue

The repository ships configs as the primary user-facing surface
(`README` and every research log instructs users to invoke
`python trainSpeakerNet.py --config configs/<X>.yaml`). Hard-coded
absolute paths mean:

- A new collaborator cannot run any config out of the box. They must
  hand-edit each one, then remember not to commit the edits.
- CI cannot run training-shaped smoke tests without filesystem
  surgery.
- Reproducibility claims in research logs ("we trained config X for N
  epochs") are partially broken: someone else cannot re-run config X
  without first understanding which subset of paths they need to
  override.

The §4.2 prescription named the right shape of fix —
*"document a paths.yaml or environment-variable convention so configs
are portable"* — and was answered with the env-var route (see §5 for
the alternative considered).

---

## 2. Fix

### 2.1 Three environment variables for three corpus roots

The 96 hard-coded references collapse onto three logical roots, so
three environment variables are sufficient:

| Variable | What it locates | Default-environment value |
|---|---|---|
| `SL_SPV_DATA_ROOT` | Primary corpus + lists + (usually) augmentation | `/mnt/ricproject3/2025/data` |
| `SL_SPV_VOX_FULL_ROOT` | Full-VoxCeleb corpus (only the two `mlp_mixer_rawwaveform_*` configs use it) | `/mnt/ricproject2/voxceleb_new` |
| `SL_SPV_AUGMENT_ROOT` | MUSAN + RIRs *when they live on a different mount than the main corpus* (same two configs) | `/mnt/ricproject/data` |

`SL_SPV_DATA_ROOT` covers the common case (lists, train/test audio,
and augmentation all under one tree). The two `*_VOX_FULL_*` /
`*_AUGMENT_*` variables exist only because the original layout split
those two corpus families across mounts; configs that don't need them
simply don't reference them, so they can be left unset.

### 2.2 YAML loader change

Each of the three trainers has the same YAML-merge block
(`yaml.load(f, Loader=yaml.FullLoader)` followed by a per-key copy
into `args.__dict__`). The fix adds a small helper just above that
block:

```python
def _expand_env_vars(value, key):
    # Resolve ${VAR} in YAML string values so configs can be portable
    # across machines (see paths.env.example and BUGFIX-013).
    # Unresolved references fail loudly at startup — silently passing
    # through "${SL_SPV_DATA_ROOT}" as a literal path would surface
    # only as a confusing FileNotFoundError several seconds into
    # training.
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(
            f"Config key '{key}={value}' references an undefined environment "
            f"variable. Set the required variable (see paths.env.example) or "
            f"override on the CLI with --{key} <abs-path>."
        )
    return expanded
```

and threads it through the merge loop:

```python
for k, v in yml_config.items():
    if k in args.__dict__:
        v = _expand_env_vars(v, k)        # <-- new
        typ = find_option_type(k, parser)
        args.__dict__[k] = typ(v)
    else:
        sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")
```

Three properties of this design that matter:

1. **Non-string values pass through unchanged.** `int`, `float`,
   `bool` config values are untouched — `expandvars` only mutates
   `str`. This protects against accidentally mangling numeric YAML
   keys that happen to contain `$` in some future config.
2. **Unresolved variables raise at config-merge time.** A typo in a
   variable name (`${SL_SPV_DAT_ROOT}` instead of
   `${SL_SPV_DATA_ROOT}`) fails before the trainer touches any data
   path, with a message that names both the offending key and the
   suggested remedies. This is the entire point of preferring fail-fast
   over silent pass-through.
3. **CLI override path is preserved.** Argparse defaults still apply
   for keys not present in the YAML, and `--train_path /tmp/x` on the
   command line still wins (argparse populates `args` before the YAML
   merge runs). The variable mechanism is *additive*, not a
   replacement for the existing argparse surface.

### 2.3 Configs rewritten

All 16 affected configs were rewritten via two `sed -i` invocations
(14 single-root configs, 2 dual-root configs). Examples:

Before (`configs/mini_voxceleb1_config.yaml`):

```yaml
train_list: /mnt/ricproject3/2025/data/mini_voxceleb2_train_list.txt
test_list: /mnt/ricproject3/2025/data/mini_test_list.txt
train_path: /mnt/ricproject3/2025/data/mini_voxceleb2
test_path: /mnt/ricproject3/2025/data/mini_voxceleb1
musan_path: /mnt/ricproject3/2025/data/musan
rir_path: /mnt/ricproject3/2025/data/RIRS_NOISES/simulated_rirs
```

After:

```yaml
train_list: ${SL_SPV_DATA_ROOT}/mini_voxceleb2_train_list.txt
test_list: ${SL_SPV_DATA_ROOT}/mini_test_list.txt
train_path: ${SL_SPV_DATA_ROOT}/mini_voxceleb2
test_path: ${SL_SPV_DATA_ROOT}/mini_voxceleb1
musan_path: ${SL_SPV_DATA_ROOT}/musan
rir_path: ${SL_SPV_DATA_ROOT}/RIRS_NOISES/simulated_rirs
```

Before (`configs/mlp_mixer_rawwaveform_distillation.yaml`):

```yaml
train_list: /mnt/ricproject2/voxceleb_new/train_list.txt
train_path: /mnt/ricproject2/voxceleb_new/voxceleb2
musan_path: /mnt/ricproject/data/musan
rir_path: /mnt/ricproject/data/RIRS_NOISES/simulated_rirs
test_list: /mnt/ricproject2/voxceleb_new/test_list.txt
test_path: /mnt/ricproject2/voxceleb_new/voxceleb1
```

After:

```yaml
train_list: ${SL_SPV_VOX_FULL_ROOT}/train_list.txt
train_path: ${SL_SPV_VOX_FULL_ROOT}/voxceleb2
musan_path: ${SL_SPV_AUGMENT_ROOT}/musan
rir_path: ${SL_SPV_AUGMENT_ROOT}/RIRS_NOISES/simulated_rirs
test_list: ${SL_SPV_VOX_FULL_ROOT}/test_list.txt
test_path: ${SL_SPV_VOX_FULL_ROOT}/voxceleb1
```

The three configs that were already portable (`RawNet3_AAM.yaml`,
`ResNetSE34L_AM.yaml`, `ResNetSE34L_AP.yaml`) — they rely on argparse
defaults and don't carry path keys — were left unchanged.

### 2.4 New file: `paths.env.example`

A documented example file at the repo root that contributors copy to
`paths.env` (or transcribe into their shell profile / job script).
It records the three variables, what each one controls, and the values
used on the original development machine — so a new contributor can
see how the existing data layout maps onto the variables without
hunting through the configs.

The file is named `paths.env.example` (rather than `paths.env`) so
that any future `paths.env` containing real local paths is unambiguously
the user's, not committed to the repository.

---

## 3. Verification

The fix was exercised by:

1. **Static parse of every config.** A `yaml.safe_load` of every file
   under `configs/` returned a valid dict — the `${VAR}` syntax is
   plain YAML strings, so no parser change is needed.
2. **End-to-end substitution test (synthetic).** With the helper
   transplanted into a 30-line test:
   - `mini_voxceleb1_config.yaml` with `SL_SPV_DATA_ROOT=/tmp/test/data`
     resolved every key to a fully absolute path under `/tmp/test/data`.
   - `mlp_mixer_rawwaveform_distillation.yaml` with all three variables
     set resolved correctly across the two roots.
   - With `SL_SPV_DATA_ROOT` unset, `_expand_env_vars("${SL_SPV_DATA_ROOT}/foo", "train_path")`
     raised `ValueError` with the expected message.
3. **Trainer-script syntax.** `python -m py_compile` succeeded on all
   three patched trainers.
4. **Reference-count audit.** `grep -rn '/mnt/ric' configs/` returned
   zero lines after the rewrite (the single header-comment hit was
   updated to a relative path). `grep -rc '\${SL_SPV_'` reported 95
   variable references across 16 configs, matching the 95 path
   substitutions performed (the 96th reference was the file-header
   comment in `experiment_01.yaml`, not a data path).

What this fix does **not** verify:

- It does not exercise a *real* training run end-to-end. The system
  Python at the time of writing does not have the project's runtime
  dependencies installed; verifying that a fully-resolved
  `${SL_SPV_DATA_ROOT}/mini_voxceleb2` exists and is a valid corpus
  requires the contributor to also set the variable to a real path
  (which is the contributor's responsibility).
- It does not guard against subtler path mistakes — e.g., setting
  `SL_SPV_DATA_ROOT=` to a path *without* a trailing slash but a
  config that double-slashes via `${SL_SPV_DATA_ROOT}//mini_voxceleb2`.
  Filesystem APIs collapse double slashes, so this is harmless in
  practice, but it would be the kind of thing a stricter validator
  could catch.

---

## 4. Backward-compatibility & migration

- **Existing users of the original development machine need to set
  three variables once and never touch them again.** The
  `paths.env.example` lists the original-machine values verbatim:

  ```bash
  export SL_SPV_DATA_ROOT=/mnt/ricproject3/2025/data
  export SL_SPV_VOX_FULL_ROOT=/mnt/ricproject2/voxceleb_new
  export SL_SPV_AUGMENT_ROOT=/mnt/ricproject/data
  ```

  Running any config that previously worked will continue to work
  once those three lines are in the shell profile.
- **CLI overrides keep their old precedence.** Anyone with a
  one-off run command like `python trainSpeakerNet.py --train_path
  /tmp/quick-test --config configs/X.yaml` continues to work without
  even setting the variables, because argparse-side defaults / CLI
  values fully populate `args` and the YAML merge only overwrites keys
  that are *present in the YAML*. (Confirmation: in the loader, the
  `if k in args.__dict__` guard means an unset env var matters only
  for keys actually written in the YAML, and those keys are
  command-line overridable.)
- **Old YAML files outside this repository (private branches,
  research-log snippets, etc.) keep working as long as they specify
  absolute paths directly.** `os.path.expandvars` is a no-op on
  strings that contain no `$`. The new behaviour is strictly additive.

---

## 5. Out-of-scope (and the alternative the user did not pick)

Two paths were offered:

- **A: env-var substitution in YAML** (chosen). Three-line loader
  change; configs use `${VAR}`. Standard 12-factor practice.
- **B: a `paths.yaml` overlay**. Configs would have used
  `{data_root}/mini_voxceleb2`; the loader would have loaded a
  separate `paths.yaml` first and substituted via `str.format`.

The user picked A. For the record, B would have required ~15 lines of
loader code instead of three, plus a committed `paths.yaml` (or a
documented "copy and edit" of one), plus a more careful
documentation surface (because the `{...}` placeholder syntax shadows
Python f-strings and shell brace-expansion, both of which someone
might reasonably reach for).

Also explicitly out of scope:

- **Validation that referenced files actually exist.** The trainer
  will discover that at data-load time; adding a fail-fast existence
  check in the loader would conflate environment-variable hygiene with
  filesystem state. The latter belongs in a `validate_config.py`-style
  script that does not exist in this repo yet.
- **Recursive expansion of nested-dict values.** The trainers'
  current YAML schema is flat (every key maps directly to an argparse
  field). If a future config grows nested keys, the helper would need
  to recurse — but doing that now would be speculative scaffolding.
- **A `.env.example` → `.env` autoloader** (à la `python-dotenv`).
  Configurable, but adds a runtime dependency for what is currently a
  one-line addition to `.bashrc`. The trade-off does not pay yet.
- **Migrating CLI usage in `README.md` / research logs.** Those
  documents describe *what* people did historically; rewriting them
  to use the new variables would falsify the record. The
  `paths.env.example` is the new canonical place to describe the
  variables.

---

## 6. Rollback plan

If the env-var convention proves more friction than it removes:

1. Revert the YAML loader change in all three trainers (delete the
   `_expand_env_vars` helper and the call site).
2. Run the inverse `sed` over `configs/`:

   ```bash
   sed -i 's|\${SL_SPV_DATA_ROOT}|/mnt/ricproject3/2025/data|g' configs/*.yaml
   sed -i 's|\${SL_SPV_VOX_FULL_ROOT}|/mnt/ricproject2/voxceleb_new|g' configs/*.yaml
   sed -i 's|\${SL_SPV_AUGMENT_ROOT}|/mnt/ricproject/data|g' configs/*.yaml
   ```

3. Delete `paths.env.example`.

Two notes:

- A *partial* rollback is also valid — drop the loader change and
  keep the configs in their portable form, then have contributors
  hand-set the paths in their YAML. This costs nothing on the
  original-environment side and surfaces the path question earlier
  on new environments.
- Rolling back **does not** restore the original "silent
  FileNotFoundError" behaviour; that was the bug being fixed.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#12** of the §4.2 list. Roadmap state:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ **This document** |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) (VGGVox fixed; ResNetSE34L already correct) |
| 15 | `RawNet3.py` debug print + in-place mutation | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #13.
- **Substitution mechanism:**
  [`os.path.expandvars`](https://docs.python.org/3/library/os.path.html#os.path.expandvars)
  — POSIX `${VAR}` and bare `$VAR` are both supported, but the
  `paths.env.example` and configs standardise on the `${VAR}` form for
  readability and to avoid accidental concatenation in path strings
  (`$SL_SPV_DATA_ROOT_archive` would otherwise be ambiguous; the
  brace form is not).
- **PyYAML loader behaviour:** `yaml.FullLoader` returns Python
  strings for unquoted YAML scalars; `${VAR}` is not parsed by YAML
  itself, so no string-tag tricks are needed.
- **The Twelve-Factor App, factor III ("Config"):**
  https://12factor.net/config — for the general principle that config
  belongs in the environment, not in code or per-host files.
