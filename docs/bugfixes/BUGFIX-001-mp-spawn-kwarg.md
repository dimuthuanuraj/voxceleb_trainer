# BUGFIX-001 — `mp.spawn` keyword-argument typo in `trainSpeakerNet.py`

| Field | Value |
|---|---|
| **ID** | BUGFIX-001 |
| **Severity** | Critical (blocks every multi-GPU / `--distributed` run) |
| **Component** | Distributed training launcher |
| **Files touched** | [trainSpeakerNet.py](../../trainSpeakerNet.py) (1 line) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #1 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

`trainSpeakerNet.py` was passing the keyword argument **`n_procs`** to
`torch.multiprocessing.spawn`, but the PyTorch API has always required the
keyword **`nprocs`**. The two strings differ by a single underscore — the
typo is silent under single-GPU runs (the `args.distributed` branch is never
taken) but is fatal the moment anyone launches with `--distributed`.

### 1.1 Offending line (before fix)

[trainSpeakerNet.py:411](../../trainSpeakerNet.py#L411)
```python
if args.distributed:
    mp.spawn(main_worker, n_procs=n_gpus, args=(n_gpus, args))
                          ^^^^^^^
                          wrong kwarg — Python raises TypeError
```

### 1.2 Failure mode that this caused

```text
$ export CUDA_VISIBLE_DEVICES=0,1
$ python trainSpeakerNet.py --config configs/experiment_01.yaml --distributed
…
TypeError: spawn() got an unexpected keyword argument 'n_procs'
```

Because Python validates keyword arguments at the call site, the process
dies *before* any worker is spawned, before `dist.init_process_group` is
ever called, and before any GPU memory is allocated. There is no partial
state to clean up — but no training happens either.

### 1.3 Reference: the documented PyTorch API

[`torch.multiprocessing.spawn`](https://docs.pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn)
signature (PyTorch ≥ 1.0, unchanged through 2.9):

```python
torch.multiprocessing.spawn(
    fn,                    # worker entry point
    args=(),               # positional args forwarded to fn
    nprocs=1,              # ← correct keyword (this is what we needed)
    join=True,
    daemon=False,
    start_method='spawn',
)
```

The kwarg has been spelled `nprocs` (no underscore) since the API was
introduced. There is no version of PyTorch in which `n_procs` works.

---

## 2. Fix applied

A single-character edit: removed the underscore between `n` and `procs`.

### 2.1 Diff

```diff
--- a/trainSpeakerNet.py
+++ b/trainSpeakerNet.py
@@ -408,7 +408,7 @@
     print('Save path:',args.save_path)

     if args.distributed:
-        mp.spawn(main_worker, n_procs=n_gpus, args=(n_gpus, args))
+        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
     else:
         main_worker(0, None, args)
```

### 2.2 Final state

[trainSpeakerNet.py:410-413](../../trainSpeakerNet.py#L410-L413)
```python
if args.distributed:
    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
else:
    main_worker(0, None, args)
```

### 2.3 No semantic change beyond the kwarg
- `main_worker` signature is unchanged: `(gpu, ngpus_per_node, args)`.
- The positional `args=(n_gpus, args)` tuple is unchanged; `mp.spawn`
  prepends the rank as the first positional argument to `main_worker`, so
  `main_worker(rank, n_gpus, args)` continues to receive its arguments in
  the correct order.
- All downstream DDP initialisation in `main_worker` ([trainSpeakerNet.py:140-149](../../trainSpeakerNet.py#L140-L149))
  is untouched: `MASTER_ADDR/MASTER_PORT`, `dist.init_process_group`,
  `torch.cuda.set_device`, and `DistributedDataParallel` all behave as
  intended.

---

## 3. Sibling files — audited, no fix needed

The original `SL_LANGUAGE_SPV_ANALYSIS.md` flagged that
`trainSpeakerNet_performance_updated.py` and `trainSpeakerNet_distillation.py`
"likely share this bug." I verified each. Both already use the correct
keyword:

| File | Line | Status |
|---|---|---|
| [trainSpeakerNet.py](../../trainSpeakerNet.py) | 411 | ❌ Was `n_procs` — **fixed in this changeset** |
| [trainSpeakerNet_performance_updated.py](../../trainSpeakerNet_performance_updated.py) | 503 | ✅ Already `nprocs` |
| [trainSpeakerNet_distillation.py](../../trainSpeakerNet_distillation.py) | 552 | ✅ Already `nprocs` |

This is consistent with the project history: the original VoxCeleb
trainer carried the typo; subsequent forks (`_performance_updated`,
`_distillation`) were written later and got the keyword right. The
original file was simply never exercised in distributed mode, so the bug
was preserved.

Audit command used to confirm:

```bash
grep -n "mp\.spawn\|spawn(" trainSpeakerNet*.py
```

Re-run this any time a fourth training script is added; the cost is
trivial and the failure mode is binary.

---

## 4. Verification

### 4.1 Syntactic check
A `python -m py_compile trainSpeakerNet.py` succeeds without error,
confirming the file still parses. (No new imports or symbols were
introduced, so a static check is sufficient at this layer.)

### 4.2 Functional check — single-GPU regression
Single-GPU training takes the `else` branch and was therefore never
affected; the fix does not alter that path. A short smoke-run on
single GPU should still:

- print `Python Version: …`, `PyTorch Version: …`, `Number of GPUs: …`,
- enter `main_worker(0, None, args)`,
- load the dataset and complete at least one batch.

If this regresses, the cause is unrelated to BUGFIX-001 — investigate
elsewhere.

### 4.3 Functional check — distributed launch
Once a multi-GPU box is available, the canonical smoke test is:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python trainSpeakerNet.py \
       --config configs/experiment_01.yaml \
       --distributed \
       --max_epoch 1 \
       --test_interval 1
```

Expected log markers:

```text
Number of GPUs: 2
Loaded the model on GPU 0
Loaded the model on GPU 1
Epoch 1, TEER/TAcc …, TLOSS …, LR …
```

Failure modes to watch for **after** this fix (i.e., not regressions of
BUGFIX-001 itself):

| Symptom | Likely cause | Where to look |
|---|---|---|
| `Address already in use` | Stale `MASTER_PORT` from a prior crashed run | `trainSpeakerNet.py:142` — change `--port` |
| `find_unused_parameters` warning | Standard DDP warning for losses with unused weights | Harmless; can be set `False` later |
| Process hangs on `dist.init_process_group` | NCCL not installed or wrong CUDA | `nvidia-smi`, `torch.cuda.is_available()` |
| One GPU silently never starts | `CUDA_VISIBLE_DEVICES` not exported in the same shell | Re-export before launch |

None of those are caused by BUGFIX-001; they are normal DDP operational
concerns that previously could not surface because the typo prevented
launch.

### 4.4 No new tests added
The repository does not currently have a unit-test harness for the
training entry points (`tests/` for the dataloader and architectures
exists, but not for the launcher). Adding a regression test for
`mp.spawn` kwargs is impractical because PyTorch's signature validation
is itself the test — any future regression of this exact form would
again fail at call time with `TypeError`. A dedicated test would
duplicate that contract.

A lightweight safeguard that *would* be cheap to add (deferred — out of
scope for this fix): a Python-level assertion before `mp.spawn` that the
function exists and is callable, e.g.:

```python
assert callable(getattr(mp, "spawn", None)), \
    "torch.multiprocessing.spawn missing — wrong PyTorch?"
```

This catches an unrelated class of error (broken install) but not
keyword typos.

---

## 5. Rollback

If for any reason this fix needs to be reverted (it should not — the
"before" state could never have succeeded), the reverse diff is:

```diff
-    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
+    mp.spawn(main_worker, n_procs=n_gpus, args=(n_gpus, args))
```

There is no scenario where this rollback is correct.

---

## 6. Impact on downstream work

This fix unblocks every item in the proposed 12-week SL plan that
requires multi-GPU training, in particular:

- Full-corpus ECAPA-TDNN training (week 5–6 in `SL_LANGUAGE_SPV_ANALYSIS.md` §5).
- WavLM-frontend fine-tuning (week 6) — a single T4 is marginal for SSL
  encoders; two T4s with DDP is the realistic configuration.
- Multilingual teacher re-distillation (week 11).

For the immediate Sinhala/Tamil pilot (≤ 200 speakers), single-GPU
training is sufficient and this fix has no effect. It is preventative —
it removes a tripwire that the project would have hit the first time it
scaled up.

---

## 7. Related items still open in §4.1 of the analysis

This fix closes item **#1** of the §4.1 Critical list. The remaining
items are independent and tracked in separate bugfix documents:

| # | Title | Status |
|---|---|---|
| 1 | `mp.spawn` kwarg typo | ✅ **This document** |
| 2 | `result[2]` used as a threshold ([trainSpeakerNet.py:241,291](../../trainSpeakerNet.py#L241)) | ✅ [BUGFIX-002](BUGFIX-002-eer-threshold-not-returned.md) |
| 3 | TEER/TAcc displays 0% when `nPerSpeaker > 1` | ✅ [BUGFIX-003](BUGFIX-003-nperspeaker-accuracy.md) |
| 4 | Hard `.cuda()` call in `SpeakerNet.forward` | ✅ [BUGFIX-004](BUGFIX-004-hard-cuda-call.md) |
| 5 | 16 kHz hard-coded in `loadWAV` (`max_audio = max_frames * 160 + 240`) | ✅ [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md) + [BUGFIX-006](BUGFIX-006-model-side-sample-rate-threading.md) |
| 6 | `numpy.pad(..., 'wrap')` for short audio | ✅ [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md) |
| 7 | SincConv buffer placement in `MLPMixerSpeaker_RawWaveform` | ✅ [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md) |
| 8 | `MaxPool1d(...) if pool else False` returns a literal `False` | ✅ [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md) |
| 9 | NestedSpeakerNet NaN explosions | ✅ [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) (quarantined to `models/experimental/`) |

§4.1 of the analysis is now fully closed. Subsequent fixes (BUGFIX-011
onward) draw from §4.2.

Each subsequent fix should land as its own `BUGFIX-00X-<slug>.md` in
this directory so the audit trail stays one-fix-per-document.

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit summarised in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #1.
- **PyTorch documentation:** https://docs.pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
- **Upstream provenance:** the typo originated in the Clova AI
  `voxceleb_trainer` parent repository and was inherited by this fork
  intact. A note for the upstream maintainers may be worth filing
  separately if they have not already received this report.
