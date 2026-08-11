# BUGFIX-023 — `dataprep.py` MD5 mismatch: `raise Warning` → `raise ValueError`

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-023-md5-mismatch-proper-error` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Low (corrupt-download path was already fatal, just confusingly named). The fix is a clarity / convention alignment. |
| **Scope** | Two lines in `dataprep.py`, both inside MD5-mismatch branches. No model, trainer, DataLoader, or config change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #23 |
| **Status** | ✅ Fixed |

---

## 1. Problem

The §4.3 polish item flagged:

> *"`dataprep.py` MD5 mismatch raises `Warning` *as exception* —
> non-fatal, but confusing; either `print` or `raise`."*

Both occurrences look like:

```python
if md5ck == md5gt:
    print('Checksum successful %s.' % outfile)
else:
    raise Warning('Checksum failed %s.' % outfile)
```

### 1.1 The bug-report framing is half-right

The report says the construct is "non-fatal". That is **inaccurate**:
`Warning` is in fact a subclass of `Exception` in Python — see
`builtins.Warning(Exception)` in the standard inheritance tree — so
`raise Warning(...)` **does** terminate the script with a stack
trace, exactly like `raise ValueError(...)`. The output looks
something like:

```text
Traceback (most recent call last):
  File "dataprep.py", line N, in <module>
    ...
Warning: Checksum failed voxceleb1_dev_wav_partaa.
```

A reader skimming `dataprep.py` sees `raise Warning(...)`, mentally
parses it as "this is going to warn, not abort", and is wrong. That
is the *real* problem — the confusion about whether execution
continues, not whether the program actually halts.

### 1.2 Why aborting is the right semantics

For a checksum failure on a download or concatenation step, the only
sensible behaviour is to abort:

- Continuing past a corrupted download means later steps (`unzip`,
  `tar -xf`, the model trainer reading the audio files) will fail
  with a far more confusing error several minutes later — at best.
- At worst, the corruption is in a binary-similar region where the
  later steps *succeed* but read garbage data into the training
  set. The whole point of an MD5 check is to catch this before
  downstream tools.

So the audit's "either `print` or `raise`" framing is correct in
form — pick one — but the right choice is `raise`. A demoted `print`
defeats the purpose of the check.

### 1.3 The neighbouring code already establishes the convention

The rest of `dataprep.py` already uses `ValueError` for fail-fast
conditions of the same shape:

- `raise ValueError('Download failed %s. ...' % url)` at line 57.
- `raise ValueError('Conversion failed %s.' % fname)` at line 143.
- `raise ValueError('Target directory does not exist.')` at line 170.

Aligning the MD5 sites with this convention is the smallest possible
edit that resolves the audit item.

---

## 2. Fix

### 2.1 Both MD5 sites converted

Two sites — `download()` at the original line 64, and `concatenate()`
at the original line 84. Each was changed from:

```python
raise Warning('Checksum failed %s.' % outfile)
```

to:

```python
raise ValueError('Checksum failed %s. Expected MD5 %s, got %s.' %
                 (outfile, md5gt, md5ck))
```

Two other things land in the same edit:

1. **The error message now includes both the expected and observed
   MD5 hashes.** A user staring at the failure can immediately tell
   whether the problem is "the file is silently corrupted in
   transit" (hashes are deterministic-looking nonsense), "the file
   downloaded is wrong" (hashes look fine but unrelated), or "the
   md5 list is out of date" (their hash matches the file content,
   the expected hash is the stale one). The original one-line
   message gave no diagnostic signal.
2. **A short BUGFIX-023 comment is added** at each site explaining
   the conversion, so anyone reading the file later understands why
   `Warning` was replaced rather than wondering if the demotion to
   `ValueError` is the "non-fatal" route the audit asked about.

### 2.2 No demotion to `print` / `warnings.warn`

Both alternatives were considered. Both are wrong here:

- **`print(...) + continue`** silently advances past a corrupt
  download, which is the exact failure mode the MD5 check exists to
  prevent.
- **`warnings.warn(...)`** is the same shape as `print` from a
  control-flow perspective (it issues a warning and returns; it
  does *not* halt). The whole point of the audit fix is to make the
  *halt* explicit; demoting to a non-halting form would change
  observable behaviour for the worse.

The right fix is the one that preserves the existing "this aborts"
semantics while renaming the exception type to match.

### 2.3 Concatenation site: avoid losing the source parts

The `concatenate()` site has an additional subtlety. After the
checksum block, the next line is:

```python
out = subprocess.call('rm %s/%s' % (args.save_path, infile), shell=True)
```

That deletes the *parts* (the input pieces of the concatenation)
after the concatenated artefact passes its checksum. With
`raise ValueError(...)` on a failure, the `rm` does NOT execute —
because the exception propagates out of the loop body before
reaching the next statement. This is the *correct* behaviour: if
concatenation produced a bad file, the operator wants the source
parts preserved so they can re-concatenate or inspect them
manually. A comment was added at the call site to make this
property visible:

```python
# BUGFIX-023: ... Crucially, do NOT delete the source parts on a
# checksum failure (the rm below); the operator may want to
# re-concatenate or inspect them.
```

This was already the behaviour with the original `raise Warning(...)`
— the exception class change does not affect *whether* the `rm`
runs — but the comment now records *why* the ordering matters in
case someone later "refactors" the failure path into a `try / except
/ continue` that would defeat it.

---

## 3. Verification

### 3.1 Static

`python -m py_compile dataprep.py` returns exit 0.
`grep -n 'raise Warning' dataprep.py` returns zero hits.
`grep -n 'raise ValueError.*Checksum' dataprep.py` returns two hits,
one for each MD5 site (lines 69 and 92 after the fix).

### 3.2 Behavioural

Three failure shapes verified by inspection of the call flow:

| Scenario | Pre-fix | Post-fix |
|---|---|---|
| Download succeeds, checksum matches | "Checksum successful ..." printed, loop continues | Same |
| Download succeeds, checksum fails | `Warning` exception propagates, script halts with stack trace | `ValueError` exception propagates, script halts with stack trace + expected/observed MD5 in the message |
| Download fails (wget non-zero) | `ValueError('Download failed ...')` halts before the MD5 check | Same |

The "halts" property is unchanged for every input — only the
exception type and the message detail have changed.

### 3.3 Not verified

- No actual corrupted download was constructed to trigger the new
  message. The format-string substitution is straightforward Python
  and was inspected for correctness; running the corruption test
  end-to-end would require downloading ~30 GB of VoxCeleb data and
  manually mangling a file, which is disproportionate to the fix.
- `dataprep.py` has not been run end-to-end in this audit. It is a
  one-shot data-acquisition script invoked once per machine, not
  part of the training loop, so the at-rest verification (syntax +
  shape inspection) is sufficient.

---

## 4. Backward-compatibility & migration

- **No call to `dataprep.py` in the rest of the repository.** This
  is a standalone script invoked by the user with
  `python dataprep.py --download ...`, so there is no caller to
  break.
- **Anyone who has ever caught `Warning` to silence the MD5 failure
  in a wrapper script** will now see a `ValueError` propagate
  instead. That would have been a *bug* in their wrapper script
  (catching `Warning` to ignore corrupted downloads is exactly the
  wrong thing); fixing it forward is preferable to honouring the
  buggy expectation.
- **The Python interpreter's `python -W ...` warning-filter
  machinery** has no effect on this code path. `raise Warning(...)`
  does **not** go through the `warnings` module — `warnings.warn`
  does, `raise Warning(...)` does not. Anyone whose mental model
  was "this is filtered by `python -W ignore`" was wrong before
  the fix too.

---

## 5. Out-of-scope

- **Validating MD5 *before* `wget` (to skip re-downloading an
  already-good file).** A useful idempotence improvement; not
  required for the §4.3 close. A future bugfix could add a
  pre-check that skips the download if the file exists and matches.
- **Replacing `subprocess.call('rm ...', shell=True)` with
  `os.unlink(...)`.** The `shell=True` form is mildly less secure
  (shell-injection-prone if `args.save_path` ever contained user
  input — it doesn't) and the audit-trail comment in §2.3 does not
  change that. A separate hygiene pass could fix it.
- **Catching the `subprocess` return code on the `rm` and the
  `cat` commands** (`out` is assigned but never inspected at
  those sites). Pre-existing pattern, not flagged by §4.3 #23.
- **Switching the entire `dataprep.py` to `pathlib` / `shutil` /
  Python-native operations.** Out of scope; broader refactor.

---

## 6. Rollback plan

If for some unexpected reason `ValueError` here causes a downstream
script to break:

1. Revert both edits to `raise Warning(...)`.
2. Drop the BUGFIX-023 comments at both sites.

The downside of the rollback is that the audit item re-opens. A
*partial* rollback that picks a non-fatal form (e.g.,
`sys.stderr.write(...) + continue`) is **not** a valid target — it
would defeat the integrity check the file is performing.

---

## 7. Related items in §4.3 of the analysis

This fix closes item **#23**. Roadmap state:

| # | Title | Status |
|---|---|---|
| 21 | `pdb` imports left in production | ✅ [BUGFIX-021](BUGFIX-021-remove-pdb-imports.md) |
| 22 | Inconsistent variable casing | ✅ [BUGFIX-022](BUGFIX-022-camel-snake-case-aliases.md) |
| 23 | `dataprep.py` MD5 mismatch raises `Warning` as exception | ✅ **This document** |
| 24 | `analyze_performance.py` / `benchmark_performance.py` / `quick_optimize.py` overlap | ✅ [BUGFIX-024](BUGFIX-024-consolidate-performance-scripts.md) |
| 25 | Each model duplicates `PreEmphasis` / mel / InstanceNorm | ✅ [BUGFIX-025](BUGFIX-025-shared-audio-frontend.md) |
| 26 | `exps/` folders mostly contain `logs/` / `result/` but no checkpoints | ✅ [BUGFIX-026](BUGFIX-026-exps-cleanup-policy.md) |

§4.1 (Critical) is fully closed (BUGFIX-001..010).
§4.2 (Important) is closed except for **#12** (`lists/` empty — Sri Lankan dataprep workstream).
§4.3 (Minor / polish) — three items closed; three remaining.

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.3 item #23.
- **Python's `Warning` class hierarchy:**
  https://docs.python.org/3/library/exceptions.html#warnings — note
  `Warning(Exception)` is exception inheritance, not the
  `warnings` module's `warn()` machinery. The two are easy to
  confuse and the bug report's "non-fatal" framing is exactly the
  confusion the audit item is calling out.
- **`warnings` module vs `raise Warning(...)`:**
  https://docs.python.org/3/library/warnings.html — `warnings.warn`
  is the actual non-fatal warning path; `raise Warning(...)` is a
  fatal exception that happens to use `Warning` as the class. They
  are unrelated despite the name overlap.
- **Surrounding `ValueError` convention in `dataprep.py`:**
  see [`dataprep.py:57`](../../dataprep.py#L57) (download failure),
  [`dataprep.py:143`](../../dataprep.py#L143) (conversion failure),
  [`dataprep.py:170`](../../dataprep.py#L170) (target dir missing).
- **Related fixes:** none directly related; this is a self-contained
  data-pipeline hygiene fix.
