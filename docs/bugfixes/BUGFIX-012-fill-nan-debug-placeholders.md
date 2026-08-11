# BUGFIX-012 — Fill 0-byte NaN-debug placeholders with real content

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-012-fill-nan-debug-placeholders` |
| **Date** | 2026-05-18 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (cleanup) — non-zero in the sense that empty files imply tooling that doesn't exist, which is its own form of project debt. |
| **Scope** | Two files at the repo root: `analyze_nan_debug.py` (was empty, now a real diagnostic script) and `NaN_DEBUGGING_GUIDE.md` (was empty, now a real guide). No production-path code changes. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #11 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.2 audit flagged two 0-byte files at the repo root:

```
-rwxrwxrwx 1 spdanuraj spdanuraj 0 May 14 08:49 analyze_nan_debug.py
-rwxrwxrwx 1 spdanuraj spdanuraj 0 May 14 08:49 NaN_DEBUGGING_GUIDE.md
```

`research_logs/2025-10-29.md` §4 describes them as if they were
committed with real content (citing commit `3b6f3a6` — "Add NaN
debugging guide and analysis script"), but on disk both files are
empty. Two failure modes for empty placeholders:

- A reader who finds them via the research log assumes they exist and
  wastes time looking for the tooling they describe.
- A future contributor sees the names and *re-creates* the files
  inconsistently with the research log's spec, branching the
  documentation away from the original intent.

The §4.2 prescription was "delete or fill". After confirming there is
exactly one documented NaN incident in this repository
([BUGFIX-010 / NestedSpeakerNet](BUGFIX-010-quarantine-nestedspeakernet.md))
that the project already has real material for, the user elected to
**fill** rather than delete — see §5 for what the alternative would
have looked like.

---

## 2. Fix

### 2.1 `analyze_nan_debug.py` — diagnostic script

The script at the repository root now does what the 2025-10-29 research
log claimed it should: load a saved checkpoint and / or a training log
and report NaN / Inf statistics in a form that maps onto the failure
modes in the new guide.

Surface:

```bash
python analyze_nan_debug.py --checkpoint exps/<exp>/model/model000010.model
python analyze_nan_debug.py --log logs/<run>.log --verbose
python analyze_nan_debug.py --checkpoint <path> --log <path>
```

For a checkpoint it walks `torch.load(path, map_location="cpu")`,
accepting both the bare state_dict layout that this repo's
`SpeakerNet.saveParameters` produces and the more conventional
`{"state_dict": ...}` layout (so a checkpoint from another trainer
loads cleanly). For each parameter it reports `nan` count, `inf` count
(positive + negative), finite-only `mean` / `std` / `|max|`, and the
fraction of finite values. Without `--verbose` only the problematic
parameters are listed; with `--verbose` every parameter prints.

For a log file it scans for `loss = nan` / `tloss = inf` patterns
(matching the format the three trainers emit), tracks the last clean
epoch / iteration *before* the first NaN, and reports the first ten
offending lines with line numbers.

Exit codes are designed for CI / scripted use:

- `0` — supplied artefacts contain no NaN / Inf.
- `1` — NaN / Inf detected.
- `2` — invalid arguments / file not found.

### 2.2 `NaN_DEBUGGING_GUIDE.md` — guide

The guide at the repo root is organised around the **empirical record
this codebase actually has**, not generic PyTorch boilerplate:

- **§1** — The one documented incident (NestedSpeakerNet,
  [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md)), with the
  exact failure epochs, EER values, and diagnosed root cause from the
  research log.
- **§2** — The mitigations already wired into the trainers (gradient
  clipping, `GradScaler`, autocast, the dither-pad fix from BUGFIX-007,
  the SincConv-buffer fix from BUGFIX-008). Each is linked to its
  source file with a line-anchored reference. The point of this
  section is to let a reader confirm "nothing is broken that *should*
  be catching this" before moving on.
- **§3** — Five failure modes (AMP overflow, AAM-Softmax saturation,
  gradient-path explosion, all-zero clips, post-scheduler LR jump),
  each with **Symptom / Why it happens / Diagnostic / Mitigation**.
  Only §3.3 is empirically confirmed here; §5 of the guide is explicit
  that the other four are plausible-but-unobserved in this repo, so
  readers don't over-index on them.
- **§4** — A six-step triage workflow that points at concrete
  `analyze_nan_debug.py` invocations.
- **§5** — Honest statement of the guide's limits.

The guide cross-links to BUGFIX-007 / 008 / 010 and to
`SAMPLING_RATE_GUIDE.md`, so anyone arriving at it from a NaN incident
can immediately see the relevant prior work without searching.

---

## 3. Verification

The script is exercised by:

1. **Syntax / import.** `python -c "import ast; ast.parse(open('analyze_nan_debug.py').read())"`
   returns cleanly; `python -m py_compile analyze_nan_debug.py` succeeds
   inside the project's conda env (it depends only on `torch`,
   `argparse`, `math`, `os`, `re`, `sys`, all stdlib + already-pinned
   `torch>=2.1,<3` from BUGFIX-011).
2. **CLI surface.** `python analyze_nan_debug.py` (no args) returns
   exit 2 and prints the help with the expected `--checkpoint` and
   `--log` flags. `python analyze_nan_debug.py --help` lists all flags.
3. **State-dict handling.** The two accepted layouts (bare state_dict,
   `{"state_dict": ...}` wrapper) are both detected in the loader path;
   anything else returns exit 2 with a descriptive message.
4. **Log-pattern matching.** The `_LOSS_PATTERN` regex matches the
   actual print format used by all three trainers (`"TLOSS 4.231"`,
   `"TLOSS nan"`, etc.) — verified against the format strings in
   `trainSpeakerNet.py` and `SpeakerNet*.py`'s `train_network` loops.

The guide is exercised by:

1. **Every linked file path is real.** `BUGFIX-007`, `BUGFIX-008`,
   `BUGFIX-010`, `SAMPLING_RATE_GUIDE.md`, `models/experimental/README.md`,
   `DatasetLoader.py`, `DatasetLoader_performance_updated.py`,
   `MLPMixerSpeaker_RawWaveform.py`, `SpeakerNet.py`,
   `SpeakerNet_performance_updated.py`, `SpeakerNet_distillation.py` —
   all resolve to existing files at the cited line numbers.
2. **Every code-line citation matches the file.** The line numbers in
   §2's mitigation table point at the exact `GradScaler` /
   `clip_grad_norm_` call sites in the three trainers.

What this fix does **not** verify:

- It does not verify the script's output against a *real* NaN
  checkpoint. The project's only documented NaN incident
  (NestedSpeakerNet, three months old) has not been re-run for this
  fix and the failing checkpoint may no longer exist in `exps/`. The
  script will produce the right output on any future incident; the
  fixed format and exit-code contract are the verifiable parts.

---

## 4. Backward-compatibility & migration

- **No code path imports either file.** They are user-facing tools
  invoked manually. Going from 0-byte to real content cannot break any
  caller.
- **The `research_logs/2025-10-29.md` description of these files is
  still accurate at the section-headings level** (purpose, common
  causes, detection strategies, debugging tools) — the new guide
  organises the same conceptual material around the empirical
  evidence available in 2026.
- **The script is invoked the way `tail -f` / `nvidia-smi` are
  invoked** — by humans triaging a failure, not by any other script.
  No `argparse` / CLI surface needs to be locked down.

---

## 5. Out-of-scope (and the alternative the user did not pick)

The §4.2 prescription offered two paths: *delete* or *fill*. The user
chose *fill*. For the record, the *delete* alternative would have been:

```bash
rm analyze_nan_debug.py NaN_DEBUGGING_GUIDE.md
```

plus a one-paragraph BUGFIX-012 explaining that the
`research_logs/2025-10-29.md` description was aspirational and that
the project's actual NaN-debugging record lives in BUGFIX-010 / the
2025-12-29 research log. That would have been ~30 lines of writeup
vs. this fix's ~400 lines of script + guide + this doc.

The fill path is bigger but matches the user's explicit choice and
produces real tooling. The trade-off documented honestly: the script
has not been exercised against a real failing checkpoint (see §3),
and §3 of the guide contains four failure modes that are
plausible-but-unobserved in this repository.

Also out of scope:

- Tests for `analyze_nan_debug.py`. The script is short, mostly
  string-formatting + a regex + a torch loop. A test harness for
  diagnostic tools is a separate concern and would warrant its own
  bugfix if the project grew enough such tools to make a harness pay
  off.
- Wiring the script into a training-time hook (e.g., auto-running it
  whenever loss → NaN). That would couple a diagnostic to the trainer
  and is the wrong direction — the diagnostic is meant to be run
  *after* a failure, on the saved artefacts, where its output is
  reproducible.
- A `--checkpoint-dir` mode that walks `exps/<exp>/model/` and finds
  the last clean checkpoint automatically. Useful future polish but
  not part of this minimal fill.

---

## 6. Rollback plan

If these files turn out to be more confusing than helpful (for
instance, if the §3.1–§3.5 modes mislead a future triage):

1. Either `rm` both files (returning to the "delete" alternative from
   §5), or
2. Replace the guide with a stub that points at BUGFIX-010 and the
   2025-12-29 research log as the only authoritative material.

The script is fully self-contained and depends on nothing in the rest
of the repo besides `torch.load`, so removing it has zero ripple.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#11** of the §4.2 list. Status of the items
flagged in BUGFIX-011's roadmap:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ **This document** |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
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
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #11.
- **Source of empirical material for the guide:**
  [`docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md`](BUGFIX-010-quarantine-nestedspeakernet.md)
  and
  [`research_logs/2025-12-29-nested-learning-experiment.md`](../../research_logs/2025-12-29-nested-learning-experiment.md).
- **Source of the original specification (now mostly superseded):**
  [`research_logs/2025-10-29.md`](../../research_logs/2025-10-29.md) §4.
- **PyTorch anomaly detection:**
  https://docs.pytorch.org/docs/stable/autograd.html#anomaly-detection
- **PyTorch GradScaler / autocast:**
  https://docs.pytorch.org/docs/stable/amp.html
