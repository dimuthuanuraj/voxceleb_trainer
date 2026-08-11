# BUGFIX-024 — Consolidate the three performance scripts

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-024-consolidate-performance-scripts` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (no behavioural impact) but high *clarity* dividend — two of the three scripts had become silently misleading because the codebase moved on without them. |
| **Scope** | Two file moves (`analyze_performance.py` and `quick_optimize.py` to `scripts/archive/`), one new directory README, header banners on the moved files, a refreshed header on the surviving `benchmark_performance.py`, and one updated section in `README_SL_COLVAI.md`. No model, trainer, DataLoader, loss, or config change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #24 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.3 polish item flagged:

> *"`analyze_performance.py`, `benchmark_performance.py`,
> `quick_optimize.py` overlap heavily; consolidate."*

The three scripts were created in the October 2025 performance push
(see `research_logs/2025-10-23.md`) and shipped at the repo root as
peer tools. By mid-2026 the picture had shifted: two of them encoded
recommendations that the *rest of the project* had since
implemented, leaving the scripts as documentation of what to do
without any awareness that it had already been done.

### 1.1 What each script actually does

**`analyze_performance.py`** — 377 lines, static AST-based source
inspection. Reads `DatasetLoader.py`, `trainSpeakerNet.py`, etc. and
flags patterns like:

- *"No `num_workers` configuration mentioned"* — but the
  perf-updated DataLoader explicitly configures it.
- *"No mixed precision training (slower, more memory)"* — but
  every trainer now wraps the forward pass in
  `torch.cuda.amp.autocast` and uses `GradScaler` (since long
  before BUGFIX-017's `--deterministic` work).
- *"No gradient accumulation (limits batch size)"* — but
  `--gradient_accumulation_steps` is an argparse flag with a real
  consumer in the trainer.
- *"No caching mechanism detected"* — but BUGFIX-018 added
  streaming-with-LRU evaluation caching, and the perf-updated
  DataLoader has `loadWAV_cached`.

Running the script today produces a long list of false positives.
A new contributor reading them might "fix" things that aren't
broken — at best wasted time, at worst regressing existing
optimisations.

**`benchmark_performance.py`** — 297 lines, genuine runtime
measurement using `pynvml` for GPU monitoring. The
`PerformanceMonitor` class (timing + summary stats) is the only
real performance-measurement infrastructure in the repo and is
generally useful. The script's "compare original vs optimized"
framing is dated (both variants ship side-by-side now and the
perf-updated one is the production path), but the underlying tool
still does what it claims.

**`quick_optimize.py`** — 131 lines, destructive in-place YAML
editor + a wall of suggested manual code changes. It rewrites
`configs/experiment_01.yaml` (sets `batch_size=128`,
`test_interval=3`) and prints a "do this in your trainer:" block
covering `num_workers`, `pin_memory`, `prefetch_factor`,
`autocast`, `GradScaler`. Every one of those code changes has been
the perf-updated trainer's default since at least October 2025.
Running the script today is at best a no-op and at worst *reverses*
later improvements (the YAML rewrite drops anything not in its
schema, including post-Oct-2025 additions like `augment_chain`,
`eval_streaming`, `deterministic`).

### 1.2 What the "overlap" really was

The audit's "overlap heavily" framing is mildly imprecise. The three
scripts do three different things at a *function* level (static
analysis / runtime benchmark / config mutation). They overlap at
the *intent* level — "make the trainer faster" — and at the
*presentation* level (each prints its own `"=" * 80` banner and
calls itself the canonical tool for performance work). They also
all assume the pre-October-2025 trainer state.

The right "consolidation" is not "merge them into one file" (which
would force a contrived shared interface), but "remove the stale
ones from the user-facing surface and keep the one that still
works". Mirroring the quarantine pattern established by BUGFIX-010
for `NestedSpeakerNet`.

---

## 2. Fix

### 2.1 Create `scripts/archive/` and move the obsolete two

```
analyze_performance.py  →  scripts/archive/analyze_performance.py
quick_optimize.py       →  scripts/archive/quick_optimize.py
```

The new directory exists explicitly to hold deprecated tooling, in
the same role that `models/experimental/` plays for deprecated
models. Anyone following a stale README pointer or research-log
reference still finds the files; anyone reading the project's
primary surface does not stumble onto them.

### 2.2 Add `scripts/archive/README.md`

The new README documents:

- The archive policy (why a separate directory, not deletion).
- A status table per archived file with the reason for archiving.
- A "what to use instead" cross-reference table.
- Revival instructions for anyone tempted to bring a file back —
  importantly, "audit and update the rules first; the docstring
  describes how the script was when archived, not how the codebase
  exists today".
- The three reasons the files are not just `rm`'d (mirrors
  `models/experimental/README.md`'s justification: citations exist,
  negative-result evidence is real, diff archaeology).

### 2.3 Archive banners at the top of each moved file

Each moved file got a prepended docstring banner stating clearly
that it is archived, *why*, and where to look for current
information:

```
======================================================================
 ARCHIVED — do not use as-is.  See scripts/archive/README.md and
 docs/bugfixes/BUGFIX-024-consolidate-performance-scripts.md.

 [file-specific obsolescence summary, e.g. for analyze_performance.py:
 "every flagged issue has since been resolved by the perf-updated
 trainer (BUGFIX-001..023). Running this today produces a long
 list of false positives and would mislead a new contributor."]

 Kept under scripts/archive/ as a research artefact only ...
======================================================================
```

A reader who skips the README and opens the file directly still
sees the warning *before* the original docstring.

### 2.4 Refresh `benchmark_performance.py`'s header

The pre-fix header read:

> *"This script compares the original vs optimized versions of
> the VoxCeleb trainer. Run this to measure the actual speedup
> achieved with optimizations."*

That framing is from when "optimized" was a fork-in-progress.
Today, both variants ship and the comparison is no longer the
primary use case. The new header:

- Describes the tool by what it *does today* — generic runtime
  benchmark, `PerformanceMonitor` infrastructure, optional GPU
  monitoring — rather than what it was created to compare.
- Names three concrete uses someone would invoke it for
  (two configs, two trainer variants, one-off flag flip).
- Documents the script's history under "History (see BUGFIX-024)"
  including why the `--mode original|optimized` flag is preserved
  for backward compatibility even though the framing is dated.

The script's *code* is untouched — only the docstring changes.

### 2.5 Update README references

`README_SL_COLVAI.md` listed all three scripts in its
"Debugging Tools" section, two of which would now point at empty
file slots in the repo root. The section was rewritten to:

- Point at the new `scripts/archive/<name>.py` paths for the two
  archived files, with an explicit *Archived* tag.
- Add the BUGFIX-024 reason inline next to each archived entry.
- Add the genuinely-current debugging tools we have today
  (`analyze_nan_debug.py` + `NaN_DEBUGGING_GUIDE.md` from BUGFIX-012).

`PERFORMANCE_README.md` and `PERFORMANCE_CHANGES_SUMMARY.md` only
ever referenced `benchmark_performance.py` (the survivor), so they
need no update.

---

## 3. Verification

### 3.1 Static

| Check | Result |
|---|---|
| `ls analyze_performance.py quick_optimize.py` in the repo root | "No such file" — correctly moved out ✅ |
| `ls scripts/archive/` | `analyze_performance.py`, `quick_optimize.py`, `README.md` ✅ |
| `python -m py_compile scripts/archive/analyze_performance.py` | exit 0 ✅ |
| `python -m py_compile scripts/archive/quick_optimize.py` | exit 0 ✅ |
| `python -m py_compile benchmark_performance.py` | exit 0 ✅ |
| `grep -rn 'analyze_performance\|quick_optimize' --include='*.md' --include='*.py'` | only updated references, no stale path mentions in the user-facing docs |

### 3.2 Documentation graph

Every external reference to the archived files now either:

- Points at the new `scripts/archive/<name>.py` path explicitly
  (`README_SL_COLVAI.md`), **or**
- Refers to them by name only as a historical fact
  (`research_logs/2025-10-23.md`), which the archive's README
  explicitly preserves citation for.

### 3.3 Not verified

- The pre-archive false-positive claim in §1.1 is based on reading
  the rules in the script and cross-referencing against the
  current code; no end-to-end run was performed. (Running the
  script under the audit-session Python would itself fail at
  import time — it has no torch dependency, but its assumptions
  about file existence and the presence/absence of specific lines
  are easy to verify by inspection.)
- `benchmark_performance.py` itself was not exercised — the
  audit session lacks the runtime deps and a CUDA GPU. The header
  refresh is pure documentation and does not affect its behaviour.

---

## 4. Backward-compatibility & migration

- **`benchmark_performance.py` at the repo root continues to work
  exactly as before.** Same `--config`, `--mode`, no flag changes.
- **`python analyze_performance.py` from the repo root now fails
  with `FileNotFoundError` instead of running.** This is intended:
  the previous run-anyway behaviour was a footgun. Anyone who
  needs to invoke the archived script can do so explicitly via
  `python scripts/archive/analyze_performance.py` — but they should
  read the archive banner first.
- **Same for `python quick_optimize.py`.** Hard-failing at the old
  path beats silently running a config-destructive tool whose
  recommendations would regress the project.
- **No checkpoint, model, or trainer-flow impact** — these scripts
  were external CLI tooling and never part of the training-loop
  path.

---

## 5. Out-of-scope

- **Deleting the archived scripts outright.** Rejected for the
  same three reasons that BUGFIX-010 keeps `NestedSpeakerNet`:
  citations exist, the negative-result evidence (what was once
  thought to need fixing) is part of the history, and diff
  archaeology benefits.
- **Rewriting `analyze_performance.py`'s rules to match the
  current codebase.** The static-analyser idea is fine; making it
  *correct* is a meaningful chunk of work and a future bugfix
  candidate (BUGFIX-025+ territory if anyone wants to take it
  on). For now, the archive note is honest.
- **Merging `benchmark_performance.py` and the surviving useful
  infrastructure into a `scripts/perf.py` with subcommands.**
  Option B from the scope question; rejected because the user
  preferred the minimal "archive obsolete, keep one" path.
- **Updating `research_logs/2025-10-23.md` to retroactively note
  the archival.** Research logs are historical record; editing
  them to back-cite a later cleanup falsifies the timeline. The
  archive README handles the forward-pointer.

---

## 6. Rollback plan

If anyone needs to undo this:

1. `mv scripts/archive/analyze_performance.py ./`
2. `mv scripts/archive/quick_optimize.py ./`
3. Remove the BUGFIX-024 banner from each file's top docstring.
4. Remove the archive entries from `README_SL_COLVAI.md`'s
   "Debugging Tools" section; restore the original three-line list.
5. Optionally `rm -r scripts/archive/` if no other archived files
   land there in the meantime.
6. Revert the header refresh on `benchmark_performance.py`.

A *partial* rollback — bring one of the two archived files back
out — is fine and only requires steps 1+3+4 for the specific file.

---

## 7. Related items in §4.3 of the analysis

This fix closes item **#24**. Roadmap state:

| # | Title | Status |
|---|---|---|
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| 22 | Inconsistent variable casing | ✅ [BUGFIX-022](BUGFIX-022-camel-snake-case-aliases.md) |
| 23 | `dataprep.py` MD5 mismatch raises `Warning` | ✅ [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md) |
| 24 | Performance scripts overlap | ✅ **This document** |
| 25 | Each model duplicates `PreEmphasis` / mel / InstanceNorm | ✅ [BUGFIX-025](BUGFIX-025-shared-audio-frontend.md) |
| 26 | `exps/` folders mostly contain `logs/` / `result/` but no checkpoints | ✅ [BUGFIX-026](BUGFIX-026-exps-cleanup-policy.md) |

§4.1 (Critical) is fully closed (BUGFIX-001..010).
§4.2 (Important) is closed except for **#12** (`lists/` empty — Sri Lankan dataprep workstream).
§4.3 (Minor / polish) — four items closed; two remaining.

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #24.
- **Archive-instead-of-delete policy:** mirrors the precedent set
  by [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) for
  `NestedSpeakerNet`. The two situations are different in
  detail — one is a model that fails to converge, the other is a
  tooling script whose advice was absorbed into the trainer — but
  the structural pattern (quarantine directory, README, banner,
  forward-pointer) is the same.
- **The optimisation work the scripts predated:** all twenty-three
  BUGFIX docs from [BUGFIX-001](BUGFIX-001-mp-spawn-kwarg.md)
  through [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md)
  together describe how the trainer reached its current state.
  `analyze_performance.py`'s rules read as a checklist of what
  *had to be done* in October 2025; the BUGFIX trail records that
  it *was* done.
- **The surviving `benchmark_performance.py`:** kept as-is with
  only a header refresh; its `PerformanceMonitor` class is the
  natural building block if anyone later wants the consolidated
  `scripts/perf.py` tool that Option B of the scope question
  described.
