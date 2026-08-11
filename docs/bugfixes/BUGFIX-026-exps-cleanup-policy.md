# BUGFIX-026 — Document the `exps/` cleanup policy

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-026-exps-cleanup-policy` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Documentation only. No code change, no behaviour change. The audit prescription was explicit: *document* a cleanup policy. |
| **Scope** | One new file (`exps/README.md`). No other file is touched. The 11 existing experiment directories are *not* deleted by this fix — actual cleanup is left to the user as a deliberate decision. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #26 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.3 polish item flagged:

> *"The 12 experiment folders under `exps/` mostly contain only
> `logs/` and `result/`, no model checkpoints — they will be
> useless for resume. Document a cleanup policy."*

The current state of `exps/` was confirmed by direct inspection:

| Directory count | `model/` subdir? | Resumable? |
|---|---|---|
| 11 | none | no — `glob.glob('model0*.model')` returns empty |
| 0 | (none have it) | — |

Total: 11 experiment directories (the audit said 12; one was
presumably removed between the audit and this fix), every one of
them in a half-state that contains `logs/` and `result/` but no
checkpoint files. The trainer's startup logic at
[`trainSpeakerNet.py:276`](../../trainSpeakerNet.py#L276) silently
restarts from epoch 1 if `model/` is missing — there is no warning
about the discarded prior state.

### 1.1 Two distinct problems the audit named

The wording "*they will be useless for resume*" + "*document a
cleanup policy*" splits cleanly:

1. **A current-state observation**: the existing 11 directories
   cannot serve as resume points. That's a fact about today's
   `exps/`, not a bug class. Either someone manually deleted the
   `model/` subdirs to save disk, runs crashed before
   `saveParameters` was ever called, or the directories are stubs
   from runs that were configured but never started.
2. **A policy gap**: there is no documentation anywhere in the
   repo about *what to keep* in `exps/`, *what to delete*, or
   *which artefacts are needed for which downstream use case*
   (resume vs. inference vs. reproducibility-citation). New users
   land in `exps/` and have no signal about what each file is
   for.

The audit's prescription "document a cleanup policy" addresses
the second problem directly. The first problem is implicitly
addressed by §2.4 of the new policy doc, which categorises the
current 11 directories explicitly and gives an action plan.

### 1.2 Why "document, not delete"

Three reasons the audit's framing is right and *delete-as-part-of-the-fix*
would be wrong:

- **Destructive actions on artefacts need explicit consent.** The
  user may have private reasons to keep a `result/scores.txt` —
  it might be cited in `research_logs/` outside this repo's
  visibility, or be the only audit trail for a number quoted in a
  conversation. An automated cleanup pass would risk losing
  exactly the kind of evidence the policy is supposed to
  preserve.
- **The 11 directories are small** (~200-800 KB each, total
  ~5 MB). The disk-pressure case for automated cleanup is weak.
- **A policy that the user enacts manually scales** — they apply
  it once they understand it, and it covers every future run.
  Automating cleanup of just the current set leaves the policy
  gap unfixed for tomorrow's runs.

So this fix is **documentation only**.

---

## 2. Fix

### 2.1 New file: `exps/README.md`

A 5-section README placed at the root of the `exps/` directory. The
README sits where readers actually look for it (a `ls exps/` shows
it alongside the experiment subdirs) rather than being buried in
`docs/`.

Section structure:

| § | Title | Purpose |
|---|---|---|
| 1 | What's in each subdir | Reference table for `model/`, `result/`, `logs/` contents with "Required for" column distinguishing resume from reproducibility |
| 2 | Cleanup policy | Four-tier retention table (Active / Recent / Stable / Archived), ordered list of "what to delete first", a "never delete" list, and explicit handling of the existing 11 unresumable directories |
| 3 | Resume semantics | How `args.save_path` controls fresh-run vs. resume, with the relevant trainer code citation |
| 4 | Naming convention recommendations | Forward-looking — no rename of existing directories; just guidance for new ones |
| 5 | See also | Cross-links to BUGFIX-026, BUGFIX-001 status table, research_logs, the two other directory-README precedents (`models/experimental/`, `scripts/archive/`) |

### 2.2 Key policy decisions encoded in the README

The document encodes several non-obvious policy choices that
needed to be made for the audit close:

**Decision 1 — What's required for resume vs. reproducibility:**

| Use case | Required artefacts |
|---|---|
| Resume training | `model/model0*.model` (most recent suffices; trainer's glob+sort picks the highest-numbered) |
| Re-deploy / fine-tune | `model/model_best.model` + `model/model_best.threshold` |
| Cite a number in a paper | `result/scores.txt` + `result/run<timestamp>.cmd` |
| Full bit-exact reproduction | All of the above + `--deterministic` flag on the original run (see [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md)) |

This distinction is the heart of the policy. Different use cases
need different subsets of the artefacts, and the cleanup tiers
follow accordingly.

**Decision 2 — Four retention tiers, not a binary keep/delete:**

The README's §2.1 defines four tiers (Active / Recent / Stable /
Archived) instead of a "delete after 30 days" cron-job approach.
The tiers map onto research workflow stages (run-in-progress,
recently-completed-need-comparison, baseline-for-future-runs,
historical-citation-only) rather than wall-clock time. This is the
right granularity for a research codebase where some runs become
load-bearing baselines and never expire.

**Decision 3 — Never-delete list:**

Three categories are explicitly listed in §2.3 as "should never be
deleted" — `run<timestamp>.cmd`, `scores.txt` for cited runs, and
`model_best.model` for deployed/cited models. The first two are
trivially small (<1 KB each); the third is the actual reproducibility
asset. Putting them on the never-delete list raises the bar for
anyone running a future automated cleanup pass.

**Decision 4 — The 11 current unresumable directories:**

§2.4 explicitly addresses the project's current state. The
recommendation is to leave them as score-archives (they are
already non-resumable; deleting `logs/` recovers ~500 KB per dir
if disk is tight; the rest should be inspected against
`research_logs/` references before any action). The actual
deletion remains the user's call.

**Decision 5 — Forward-looking naming, not retroactive rename:**

§4 recommends a config-derived naming pattern (model + corpus +
key-hyperparams + date) for *future* experiments but explicitly
does not propose renaming the 11 existing directories. Renaming
would break any external references in research logs and gain
little for sunk-cost archived runs.

### 2.3 Pattern continuity

The new `exps/README.md` follows the same template established by
two prior BUGFIX docs in this audit pass:

- [`models/experimental/README.md`](../../models/experimental/README.md)
  (BUGFIX-010) — directory README explaining quarantine policy.
- [`scripts/archive/README.md`](../../scripts/archive/README.md)
  (BUGFIX-024) — directory README explaining archival policy.

This makes the three "directory-with-policy-README" patterns mutually
recognisable. A reader who has seen one knows what to expect of the
others, and the cross-link in §5 ties them together.

---

## 3. Verification

### 3.1 Static

The new file exists at the expected path:

```
exps/
├── README.md                                                (new)
├── mini_voxceleb1_experiment/
├── mini_voxceleb1_experiment_2/
├── ...
├── my_first_custom_model/
└── performance_optimized_model/
```

The README is well-formed Markdown — verifiable by visual
inspection.

### 3.2 Policy coverage

Every artefact category named in §1 of the README is covered by
exactly one row in the retention tier table (§2.1), one item in
the "delete first" priority list (§2.2), or one item in the
"never delete" list (§2.3). No artefact is left without a policy
verdict.

### 3.3 Not verified

- No actual cleanup was performed. The 11 unresumable directories
  remain in place. The user can apply the §2.4 recommendation at
  their convenience.
- No test of the resume path was performed against an
  intentionally-pruned `model/` directory. The resume semantics
  documented in §3 of the README are by inspection of
  [`trainSpeakerNet.py:276`](../../trainSpeakerNet.py#L276) and
  the surrounding glob+sort logic, not by live test.

---

## 4. Backward-compatibility & migration

- **No code change.** Every trainer / dataloader / model file is
  untouched.
- **No checkpoint impact.** Existing checkpoints, configs, and
  research-log citations continue to work exactly as before.
- **Adoption is opt-in.** The user can apply the policy now (to
  the 11 existing dirs), apply it incrementally as runs complete,
  or ignore it entirely. The policy doc exists; it imposes no
  behaviour.
- **Future runs are unaffected by the policy doc**. The trainer's
  `os.makedirs(args.model_save_path, exist_ok=True)` continues to
  create `model/` on every fresh run, just as before. The policy
  is about what to do with these dirs after the runs complete.

---

## 5. Out-of-scope

- **Actually deleting any of the 11 unresumable directories.**
  Destructive operations on user artefacts require explicit
  user consent — see §1.2 of this doc.
- **Adding a `--cleanup` flag to the trainer** that prunes
  non-best checkpoints automatically. Useful future tooling
  enhancement; not required to close §4.3 #26 (which asked for
  documentation only) and not aligned with the "manual cleanup
  with explicit consent" principle.
- **Renaming the existing experiment directories** to the
  config-derived convention recommended in §4 of the README.
  Out of scope — backward-incompatible with any external
  reference.
- **Migrating `result/run<timestamp>.zip` to a separate
  `provenance/` directory** to make the keep-forever artefacts
  visually distinct from the regenerable ones. Possible future
  organisational improvement; not required.
- **A CI / pre-commit hook that fails when `exps/<dir>/model/`
  is missing after a complete run.** Would catch the
  "manual `rm` of `model/`" footgun, but the README's §3
  warning is sufficient for now.

---

## 6. Rollback plan

If the README turns out to be misleading or its policy
recommendations get refined by experience:

1. Either edit `exps/README.md` in place to refine the policy, or
2. `rm exps/README.md` and revert to no documentation. (Strongly
   not recommended — leaves the §4.3 audit item open again.)

A *partial* refinement is the more likely path: as the project
accumulates more experimental runs, the four-tier retention
boundaries in §2.1 may need adjustment, and the "delete first"
priority list in §2.2 may need additions. Both are local edits to
the README that don't disturb anything else.

---

## 7. Related items in §4.3 of the analysis

This fix closes item **#26**, the final entry of the §4.3 list.

| # | Title | Status |
|---|---|---|
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| 22 | Inconsistent variable casing | ✅ [BUGFIX-022](BUGFIX-022-camel-snake-case-aliases.md) |
| 23 | `dataprep.py` MD5 mismatch raises `Warning` | ✅ [BUGFIX-023](BUGFIX-023-md5-mismatch-proper-error.md) |
| 24 | Performance scripts overlap | ✅ [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md) |
| 25 | Each model duplicates `PreEmphasis` / mel / InstanceNorm | ✅ [BUGFIX-025](BUGFIX-025-shared-audio-frontend.md) |
| 26 | `exps/` cleanup policy | ✅ **This document** |

§4.3 (Minor / polish) is now fully closed.

Overall audit status:

| Section | Items | Status |
|---|---|---|
| §4.1 Critical | #1–#9 | ✅ Fully closed (BUGFIX-001..010) |
| §4.2 Important | #10–#20 | ✅ Closed except #12 (`lists/` empty — a Sri Lankan dataprep workstream, not a bug-class fix) |
| §4.3 Minor / polish | #21–#26 | ✅ Fully closed (BUGFIX-021..026) |

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #26.
- **The trainer's resume logic:**
  [`trainSpeakerNet.py:276-285`](../../trainSpeakerNet.py#L276) —
  the `glob.glob('model0*.model')` + `loadParameters(modelfiles[-1])`
  pattern is the load-bearing reason `model/` is required for
  resume.
- **Pattern precedents (directory-with-policy-README):**
  [`models/experimental/README.md`](../../models/experimental/README.md)
  from [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md), and
  [`scripts/archive/README.md`](../../scripts/archive/README.md)
  from [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md).
  The new `exps/README.md` extends the same template.
- **Related fixes:**
  [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) is the
  reason "full bit-exact reproduction" needs `--deterministic` on
  the *original* run — without it, an `exps/<dir>` with full
  artefacts still cannot be exactly reproduced.
  [BUGFIX-018](BUGFIX-018-streaming-evaluation.md) introduced
  `<save_path>/eval_feats_tmp/` which is mentioned in the
  README's broader save-path discussion (though it's separate
  from the cleanup tiers).
