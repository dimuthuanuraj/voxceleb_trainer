# `scripts/archive/` — obsolete project scripts kept for historical reference

This directory holds **scripts that were once part of the project's
active surface but no longer reflect the current state of the
codebase**. They are preserved as research artefacts — so anyone
following a research-log reference or a stale README pointer can
still find them — but they are *not* recommended for use.

## Why a separate directory (and not delete)?

Mirroring the policy established for the quarantined `NestedSpeakerNet`
(see [BUGFIX-010](../../docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md)
and [`models/experimental/README.md`](../../models/experimental/README.md)):

- **Citations exist.** The 2025-10-23 research log,
  `README_SL_COLVAI.md`, and `PERFORMANCE_README.md` all reference
  these scripts. Deleting the files would break those citations.
- **The negative-result evidence is real.** Each script encodes a
  set of optimisation recommendations from a specific point in
  time. Most of those recommendations have since been implemented
  in the perf-updated trainer (see BUGFIX-001..023), which is what
  makes the scripts obsolete — but the *list* of what was once
  recommended is part of the project's history.
- **`scripts/archive/` is explicitly not on the user-facing surface.**
  A new contributor following the project's primary docs hits
  `benchmark_performance.py` at the repo root for runtime
  measurement, never these.

## What lives here

| File | Status | Why archived |
|---|---|---|
| `analyze_performance.py` | Static AST-based source inspection. Flags issues like "no `num_workers`", "no `autocast`", "no gradient accumulation". | All flagged issues have been resolved in the perf-updated trainer (see [BUGFIX-001](../../docs/bugfixes/BUGFIX-001-mp-spawn-kwarg.md) through [BUGFIX-023](../../docs/bugfixes/BUGFIX-023-md5-mismatch-proper-error.md)). The script now produces a long list of false positives. |
| `quick_optimize.py` | Destructive in-place YAML editor + wall of suggested manual code changes. | The config mutation (`batch_size = 128`, `test_interval = 3`) and the recommended code changes (`num_workers`, `pin_memory`, `prefetch_factor`, `autocast`, `GradScaler`) are now the perf-updated trainer's defaults. Running this script today would either be a no-op or *reverse* improvements. |

## What to use instead

| If you want to ... | Use ... |
|---|---|
| Measure end-to-end training wall-time, compare two configs | [`benchmark_performance.py`](../../benchmark_performance.py) at the repo root |
| Diagnose a specific NaN incident | [`analyze_nan_debug.py`](../../analyze_nan_debug.py) + [`NaN_DEBUGGING_GUIDE.md`](../../NaN_DEBUGGING_GUIDE.md) |
| Review what optimisations *should* be in place | [`PERFORMANCE_README.md`](../../PERFORMANCE_README.md) + the BUGFIX history under [`docs/bugfixes/`](../../docs/bugfixes/) |
| Tune augmentation probabilities | `--augment_chain` (see [BUGFIX-016](../../docs/bugfixes/BUGFIX-016-configurable-augment-chain.md)) |
| Enforce paper-grade reproducibility | `--deterministic` (see [BUGFIX-017](../../docs/bugfixes/BUGFIX-017-deterministic-mode-toggle.md)) |

## Reviving a script from this archive

If a future need re-emerges (e.g., porting the project to a new
GPU class and wanting fresh "is `autocast` still active?" checks):

1. Move the file back to the repo root.
2. **Audit and update its rules** — the existing checks are stale
   by months / years of BUGFIX work and will need a rewrite.
3. Remove the archive banner at the top of the file.
4. Open a fresh BUGFIX doc documenting the revival rationale.

Do not assume a script in this directory works as advertised in its
own docstring — the docstring describes the script as it was when
moved here, not as the codebase exists today.

## Why we don't just delete

Three reasons (mirroring `models/experimental/README.md`):

1. **The 2025-10-23 research log** documents the design of
   `benchmark_performance.py` and references the others by name.
   Deleting would break that citation chain.
2. **The `PERFORMANCE_README.md` and `README_SL_COLVAI.md`**
   reference these files. Those are user-facing docs; updating them
   to point here is cleaner than rewriting the history.
3. **Diff archaeology**: a future contributor wondering "did we
   ever have a static analyser?" can find a yes/no answer here
   instead of digging through git history.

If a script in this directory becomes truly irrelevant (no doc
cites it, no research log references it, no one would reattempt
its approach), it can be deleted in a separate cleanup pass — not
as part of the archive itself.
