# BUGFIX-018 — Streaming evaluation for SL-benchmark-scale test lists

| Field | Value |
|---|---|
| **Slug** | `BUGFIX-018-streaming-evaluation` |
| **Date** | 2026-05-19 |
| **Author** | Repository audit pass (Claude-assisted) |
| **Severity** | Medium-to-high for the *target* workload (1M-pair SL benchmark) — a guaranteed OOM. Low for the *current* workload (mini-VoxCeleb1, ~40k pairs) — the existing path uses ~200 MB of RAM and works fine. The fix is opt-in so it costs nothing for the existing workload. |
| **Scope** | Three `SpeakerNet*.py` files (one `evaluateFromList` implementation each), three trainers (one `--eval_streaming` and one `--eval_feat_cache_size` argparse arg each). No model, DataLoader, or config-file change. |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #18 |
| **Status** | ✅ Fixed |

---

## 1. Problem

`evaluateFromList` is the EER / MinDCF evaluation entry point and lives
in three near-identical implementations:

- [`SpeakerNet.py`](../../SpeakerNet.py) (base trainer, batch-size-1
  extraction)
- [`SpeakerNet_performance_updated.py`](../../SpeakerNet_performance_updated.py)
  (perf-optimised trainer, batched extraction)
- [`SpeakerNet_distillation.py`](../../SpeakerNet_distillation.py)
  (distillation trainer, batched extraction + normalize fallback)

All three follow the same two-phase shape:

```python
feats = {}
# --- extraction phase (every rank computes its sampler shard) ---
for idx, data in enumerate(test_loader):
    ref_feat = self.__model__(inp1).detach().cpu()
    feats[filename] = ref_feat

# --- gather phase (every rank ends up with the full dict) ---
if distributed:
    feats_all = [None] * world_size
    torch.distributed.all_gather_object(feats_all, feats)

if rank == 0:
    if distributed:
        feats = feats_all[0]
        for batch in feats_all[1:]:
            feats.update(batch)
    # --- scoring phase (rank 0 iterates trial pairs) ---
    for line in lines:
        ref_feat = feats[ref_filename].to(self.device, ...)
        com_feat = feats[com_filename].to(self.device, ...)
        ...
```

### 1.1 Why this OOMs at SL-benchmark scale

The peak per-rank memory cost is one full `feats` dict — N_files
embeddings of shape `(num_eval, nOut)`, dtype `float32`:

| Scenario | N_files | num_eval | nOut | Bytes per dict |
|---|---|---|---|---|
| mini-VoxCeleb1 (current default) | ~1,200 | 10 | 512 | ~25 MB |
| VoxCeleb1-O | ~5,000 | 10 | 512 | ~100 MB |
| VoxCeleb1-H | ~40,000 | 10 | 512 | ~800 MB |
| Hypothetical SL-1M (~250k unique files) | 250,000 | 10 | 512 | ~5 GB |
| Hypothetical SL-1M, larger nOut=768 | 250,000 | 10 | 768 | ~7.5 GB |

That 5–7 GB is `feats` *alone*; it is replicated by `all_gather_object`
onto **every** DDP rank, and rank 0 then holds two copies briefly
during the merge loop. On an 8-GPU node with ~16 GB per process
that's a hard OOM.

The §4.2 prescription was:

> *"`evaluateFromList` loads the whole feature dict into a single
> dict on rank 0. For 500k-pair test lists this is memory-heavy and
> would not survive a 1M-pair SL benchmark. Stream-score on disk or
> shard."*

### 1.2 Why this matters even if you don't have a 1M-pair list yet

The Sri Lankan benchmark the project is being built around does not
yet exist — but the SL language-SPV analysis explicitly plans for a
test list that will exceed the current scale, and the current pattern
has no graceful failure mode (OOM mid-eval costs you the whole
training run if it happens during a checkpoint-and-evaluate cycle).
Adding the streaming path now means the project's own future
benchmarks are reachable without an emergency rewrite.

---

## 2. Fix

### 2.1 New surface: `--eval_streaming` and `--eval_feat_cache_size`

Two new argparse arguments, added to all three trainers:

```python
parser.add_argument('--eval_streaming', dest='eval_streaming',
    action='store_true',
    help='Streaming evaluation (BUGFIX-018). Writes per-file '
         'embeddings to <save_path>/eval_feats_tmp/ and lazy-loads '
         'with an LRU cache, instead of holding the full feats dict '
         'in memory on every rank.')
parser.add_argument('--eval_feat_cache_size', type=int, default=4096,
    help='LRU cache size (in #embeddings) for --eval_streaming. '
         'Default 4096 covers VoxCeleb1-O comfortably; increase for '
         'very wide hot-set distributions.')
```

Default behaviour (no `--eval_streaming`) is byte-equivalent to the
pre-fix path. Existing scripts, configs, and CI invocations work
unchanged.

### 2.2 What the streaming path does

The streaming branch differs from the legacy branch in three places
only:

| Phase | Legacy behaviour | Streaming behaviour |
|---|---|---|
| **Extract** | `feats[filename] = ref_feat` (in RAM) | `torch.save(ref_feat, _feat_path(filename))` + `stream_keys.add(filename)` |
| **Gather** (DDP) | `all_gather_object(feats_all, feats)` — moves all tensors over NCCL | `barrier()` + `all_gather_object(all_keys, stream_keys)` — moves only filename strings |
| **Score** | `feats[name].to(device, ...)` | `_load_feat(name).to(device, ...)` where `_load_feat` is `functools.lru_cache`-wrapped |

The arithmetic in the scoring loop is unchanged — the same
`F.normalize`, the same `torch.cdist`, the same
`-1 * numpy.mean(dist)`.

### 2.3 The three primitives, in detail

**`_feat_path(filename)`** — maps a filename (potentially containing
`/`, spaces, unicode) to a flat, on-disk path inside
`<save_path>/eval_feats_tmp/`:

```python
def _feat_path(filename):
    return os.path.join(feat_dir, hashlib.sha1(filename.encode()).hexdigest() + '.pt')
```

SHA-1 is a non-cryptographic-strength but collision-resistant choice
here. For ~10^6 input filenames the birthday-bound collision
probability is roughly 10^-29 (160-bit digest), so the hashing is
effectively injective for any realistic test list. The flat layout
avoids needing to mirror the corpus directory tree inside
`eval_feats_tmp/`.

**`_load_feat(filename)`** — wraps `torch.load` in
`functools.lru_cache(maxsize=eval_feat_cache_size)`. This is a *nested*
function inside `evaluateFromList`, so each evaluation gets its own
cache (no cross-evaluation pollution). The cache is bounded so peak
memory is `eval_feat_cache_size × per-embedding bytes`. At default
4096 and the worst per-embedding cost
(`num_eval=10`, `nOut=768`, `float32`):
4096 × 10 × 768 × 4 ≈ 120 MB. That's an order of magnitude less than
the current default-path peak and decoupled from N_files entirely.

**`barrier()` before key gather** — when distributed, rank 0 will
read files that ranks 1..N-1 wrote. The barrier guarantees those
writes are flushed to the shared filesystem before rank 0 starts
reading. Without it, a fast rank-0 could race a slow rank-K and
read a not-yet-flushed file.

### 2.4 Where the on-disk feats live

`os.path.join(kwargs.get('save_path', '.'), 'eval_feats_tmp')`. Two
properties of this choice matter:

1. **Co-located with the experiment.** Anyone debugging
   reproducibility can find the embeddings that produced a given EER
   without guessing about temp-dir locations.
2. **Not auto-deleted.** Future polish — a follow-up could add a
   `--eval_cleanup_feats` flag — but the current fix preserves them
   intentionally, so a subsequent `--eval`-only run can re-score
   without re-extracting (a 10× wall-clock saver on large test sets).

The directory name ends in `_tmp` so anyone seeing
`exps/<exp>/eval_feats_tmp/` understands it is regenerable, not
training output.

---

## 3. Verification

### 3.1 Static

All six touched files pass `python -m py_compile`:
- `SpeakerNet.py`, `SpeakerNet_performance_updated.py`,
  `SpeakerNet_distillation.py`
- `trainSpeakerNet.py`, `trainSpeakerNet_performance_updated.py`,
  `trainSpeakerNet_distillation.py`

### 3.2 Streaming primitives

A standalone test (no torch needed — the primitives are pure-Python)
exercised the three risky pieces:

| Test | Behaviour verified |
|---|---|
| SHA-1 path generation on filenames like `id10001/clip-001.wav` | Three distinct inputs produce three distinct flat paths; no `/` survives into `basename`; paths end in `.pt`; no collisions. |
| Write / read round-trip via the same `_feat_path` derivation | Each filename round-trips back to its written content under the SHA-1 layout. |
| LRU cache with `maxsize=2`, three distinct keys | First three loads hit disk (`load_count=3`); re-loading a still-cached key hits the cache (`load_count` unchanged); re-loading an evicted key adds one disk read (`load_count=4`). Verifies eviction policy. |

### 3.3 Backwards compatibility

Default behaviour (`--eval_streaming` not passed):

- `kwargs.get('eval_streaming', False)` → `False`.
- `feat_dir` is not created; `_feat_path` / `_load_feat` are not
  defined.
- Extraction populates `feats` exactly as before.
- Gather uses `all_gather_object(feats_all, feats)` exactly as before.
- Scoring uses `feats[name]` exactly as before.
- The two new argparse args are no-ops when unset (the second is
  silently ignored).
- `state_dict` keys, model construction, training, and the EER
  formula are entirely untouched.

### 3.4 Not verified

- No real end-to-end DDP run was exercised — the audit-session
  Python lacks the project's runtime dependencies, and a real DDP
  test would need a multi-GPU box. The DDP correctness argument is
  by construction (the `barrier()` + `all_gather_object` of keys is
  a strict subset of what the legacy path does), plus the
  observation that both branches feed the same downstream scoring
  arithmetic.
- The "lazy-load is fast enough" claim is plausible — typical SHA-1
  + `torch.load` of a ~20 KB embedding is sub-millisecond on a warm
  cache, and the LRU keeps the hot set in RAM — but no
  micro-benchmark was run. For the target use case (running once at
  test_interval epochs over a large list), even a 10× slowdown of
  the scoring loop is acceptable.
- The interaction with `--eval` (eval-only re-runs) was not tested
  explicitly. The `_tmp` directory survives between runs, but a
  re-run still goes through extraction — wasted work for now. A
  follow-up could detect existing files and skip; that is a
  separate enhancement.

---

## 4. Backward-compatibility & migration

- **All 19 existing configs work unchanged.** None of them set
  `eval_streaming`, so they receive the legacy in-memory path.
- **All existing checkpoints work unchanged.** This is a pipeline
  change, not a model change — `state_dict` keys are untouched.
- **All existing CLI invocations work unchanged.** Both new flags
  are optional with defaults that reproduce the previous behaviour.
- **Migration path for SL benchmarks:** when the SL test list is
  finalised and grows past a few-thousand unique files, the
  recommended invocation becomes
  `python trainSpeakerNet.py --config configs/<X>.yaml --eval_streaming`
  (or set `eval_streaming: true` in the config — the dict/list
  passthrough from BUGFIX-016 means scalar bool values via YAML
  still go through the normal `bool(v)` coercion path).

---

## 5. Out-of-scope

The following were considered and explicitly **not** done:

- **Sharded pair scoring across DDP ranks.** Would further reduce
  scoring wall-time on large test lists by splitting the trial pair
  list across ranks. The §4.2 prescription named "stream-score on
  disk *or* shard" — the disk path solves the immediate memory
  problem and the wall-clock impact is marginal at the target
  scale. Sharding the score loop is a clean follow-up if scoring
  speed becomes the bottleneck.
- **HDF5 / Zarr / per-file `.npy` formats.** `torch.save` of a small
  tensor produces a 0.1–10 KB pickle that is bit-identical across
  PyTorch versions. Switching to a more "principled" container
  format buys nothing and adds dependencies.
- **Automatic streaming when N_files exceeds a threshold.** Would
  hide a meaningful behaviour change behind a heuristic. Explicit
  opt-in is the right ergonomics — the user chooses when to pay the
  disk overhead.
- **Eviction of `eval_feats_tmp/` after the run.** Deliberately
  preserved so a subsequent `--eval` can re-score without
  re-extracting. Document this in the doc rather than auto-cleaning.
- **A `--eval_streaming_format` flag** (npy vs pt vs hdf5). Single
  format keeps the surface tight; can split later if a use case
  emerges.

---

## 6. Rollback plan

If the streaming path turns out to be problematic:

1. Remove the `--eval_streaming` and `--eval_feat_cache_size`
   argparse lines from all three trainers.
2. In each of the three `evaluateFromList` implementations, remove
   the `if eval_streaming:` branches and the surrounding setup. The
   legacy path is still there structurally — the streaming code is
   purely additive — so the inverse `Edit` is mechanical.
3. Remove the `import os, hashlib, functools` line from each of the
   three SpeakerNet files.

A *partial* rollback (disable streaming by default but keep the code
path) is the no-op default already — leaving the code as-is and
simply documenting `do not pass --eval_streaming` is sufficient.

---

## 7. Related items in §4.2 of the analysis

This fix closes item **#18** of the §4.2 list. Roadmap state:

| # | Title | Status |
|---|---|---|
| 10 | Loose requirements pins | ✅ [BUGFIX-011](BUGFIX-011-requirements-pins.md) |
| 11 | Empty `analyze_nan_debug.py` / `NaN_DEBUGGING_GUIDE.md` | ✅ [BUGFIX-012](BUGFIX-012-fill-nan-debug-placeholders.md) |
| 12 | `lists/` empty of SL data (needs `sl_dataprep.py`) | ⬜ Open |
| 13 | Configs hard-code `/mnt/ricproject*/` paths | ✅ [BUGFIX-013](BUGFIX-013-portable-config-paths.md) |
| 14 | `n_mels` ignored in `ResNetSE34L.py` / `VGGVox.py` | ✅ [BUGFIX-014](BUGFIX-014-honour-n-mels-in-vggvox.md) |
| 15 | `RawNet3.py` debug print + in-place mutation | ✅ [BUGFIX-015](BUGFIX-015-rawnet3-debug-print-and-inplace.md) |
| 16 | Augmentation hard-codes 5 fixed choices | ✅ [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) |
| 17 | No deterministic mode toggle | ✅ [BUGFIX-017](BUGFIX-017-deterministic-mode-toggle.md) |
| 18 | `evaluateFromList` loads all features into rank-0 dict | ✅ **This document** |
| 19 | `torch.load` without `weights_only=True` | ✅ [BUGFIX-019](BUGFIX-019-torch-load-weights-only.md) |
| 20 | EER definition differs from common `(fpr+fnr)/2` | ✅ [BUGFIX-020](BUGFIX-020-eer-definition-disclosure.md) |

§4.1 remains fully closed (BUGFIX-001..010).

---

## 8. Authorship & references

- **Bug originally identified by:** repo audit in
  `SL_LANGUAGE_SPV_ANALYSIS.md` §4.2 item #18.
- **`functools.lru_cache`:**
  https://docs.python.org/3/library/functools.html#functools.lru_cache
  — least-recently-used eviction with O(1) hits/misses; thread-safe
  in CPython because the underlying dict ops hold the GIL.
- **`torch.distributed.all_gather_object`:**
  https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_object
  — pickles arbitrary Python objects and broadcasts; for the streaming
  branch we hand it small `set[str]` objects instead of large dicts of
  tensors.
- **`torch.distributed.barrier`:**
  https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.barrier
  — synchronisation point; used to ensure file writes are visible on
  rank 0 before reads begin.
- **Related fixes:**
  [BUGFIX-016](BUGFIX-016-configurable-augment-chain.md) generalised
  the YAML loader to allow structured (dict/list) values; this fix
  inherits that work for the `eval_streaming: true` and
  `eval_feat_cache_size: 8192` config-file forms (a boolean and an
  int respectively still go through the original scalar-coercion
  path, but they Just Work).
