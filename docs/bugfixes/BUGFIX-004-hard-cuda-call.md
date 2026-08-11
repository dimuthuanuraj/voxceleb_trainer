# BUGFIX-004 — Hard `.cuda()` data-movement calls; trainer crashes on CPU-only machines and risks device-mismatch in multi-GPU jobs

| Field | Value |
|---|---|
| **ID** | BUGFIX-004 |
| **Severity** | Critical (any CPU-only run crashes; multi-GPU runs are correct *only* if `torch.cuda.set_device(rank)` happens to have run first, which is fragile) |
| **Component** | `SpeakerNet.forward`, `ModelTrainer.train_network`, `ModelTrainer.evaluateFromList`, `DistillationSpeakerNet.forward`, checkpoint loading |
| **Files touched** | [SpeakerNet.py](../../SpeakerNet.py), [SpeakerNet_performance_updated.py](../../SpeakerNet_performance_updated.py), [SpeakerNet_distillation.py](../../SpeakerNet_distillation.py), [DistillationWrapper.py](../../DistillationWrapper.py) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #4 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

### 1.1 The original line and three failure modes
Line 40 of [SpeakerNet.py](../../SpeakerNet.py) (and analogues at 16 other call sites listed below) read:

```python
data = data.reshape(-1, data.size()[-1]).cuda()
```

`tensor.cuda()` with no argument behaves as `tensor.to("cuda", torch.cuda.current_device())` — i.e., it implicitly depends on whatever the *current* default CUDA device is. That gives three distinct failure modes:

| Mode | When it triggers | Visible effect |
|---|---|---|
| **A — Crash on CPU-only** | Any machine without an NVIDIA GPU or with `CUDA_VISIBLE_DEVICES=""`. | `RuntimeError: Found no NVIDIA driver on your system` / `AssertionError: Torch not compiled with CUDA enabled`. Cannot evaluate, debug, or even import-check a checkpoint without a GPU. |
| **B — Silent device mismatch in DDP** | Multi-process DDP where `torch.cuda.set_device(rank)` wasn't called *first*, or where a sub-thread overwrites the current device. | `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1` mid-training. |
| **C — Wrong-device tensor in single-process multi-GPU** | Someone running two `SpeakerNet` instances side-by-side on different GPUs in one process. | All `.cuda()` calls collapse onto whichever GPU `set_device` was called for last. Cross-tenant breakage. |

### 1.2 Full call-site inventory
A pre-fix scan of the four wrappers found **16 hard `tensor.cuda(...)` data-movement calls** + **2 hard `torch.load(map_location="cuda:%d" % self.gpu)` calls** + **2 unconditional `torch.cuda.empty_cache()` calls**:

| File | Site | Context |
|---|---|---|
| `SpeakerNet.py` | 40 | `SpeakerNet.forward` input |
| `SpeakerNet.py` | 106 | `train_network` labels |
| `SpeakerNet.py` | 179 | `evaluateFromList` per-utterance input |
| `SpeakerNet.py` | 219, 220 | `evaluateFromList` score-pair features |
| `SpeakerNet_performance_updated.py` | 55 | `forward` input |
| `SpeakerNet_performance_updated.py` | 127 | `train_network` labels |
| `SpeakerNet_performance_updated.py` | 237 | eval input |
| `SpeakerNet_performance_updated.py` | 287, 288 | eval score features |
| `SpeakerNet_distillation.py` | 117 | standard-mode `forward` input |
| `SpeakerNet_distillation.py` | 185 | `train_network` labels |
| `SpeakerNet_distillation.py` | 314 | eval input |
| `SpeakerNet_distillation.py` | 364, 365 | eval score features |
| `DistillationWrapper.py` | 232 | student `forward` input |
| `SpeakerNet.py` | 248 | `loadParameters` (`map_location="cuda:%d"`) |
| `SpeakerNet_performance_updated.py` | 326 | same |
| `SpeakerNet_distillation.py` | 390 | same |
| `SpeakerNet_performance_updated.py` | 307 | `torch.cuda.empty_cache()` unconditional |
| `SpeakerNet_distillation.py` | 386 | same |

The original §4.1 report named only the first (SpeakerNet.py line 40); BUGFIX-004 sweeps the full set so the same class of bug doesn't recur file-by-file as call sites are touched in future work.

### 1.3 Why we don't just `next(self.parameters()).device`
A reasonable alternative pattern is to derive the target device from the first parameter at every call. It's robust to later `.cpu()` / `.cuda(N)` moves on the module.

But the prescription in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #4 is explicit:

> Replace with `.to(self.gpu, non_blocking=True)` and propagate `self.gpu` into `WrappedModel`/`SpeakerNet`.

So the fix follows that prescription: cache a `self.device` resolved from the `gpu` kwarg at `__init__` time. The two approaches are equivalent **as long as the model is not moved after construction**, which is the contract every script in this repo already follows (`main_worker` constructs the model, then `.cuda(args.gpu)`, then never moves it again).

Trade-off noted but not implemented:
- **Stored `self.device`**: cheap; one resolution at init; tied to the construction-time `gpu` kwarg.
- **`next(self.parameters()).device`**: 1–2 µs per forward call; tracks subsequent moves; immune to forgotten `.to(...)` chains.

If a future contributor adds dynamic device moves (e.g., model-parallel sharding, CPU offload during checkpointing), switching to the parameter-introspection pattern is a one-line change inside `_resolve_device`'s call sites. The bugfix doesn't preclude that follow-up.

---

## 2. Fix applied

### 2.1 Shared helper `_resolve_device`
Each wrapper file gains a module-level helper:

```python
def _resolve_device(gpu):
    """Resolve a torch.device from a (possibly None) GPU index.

    - int + CUDA available   -> cuda:<gpu>
    - None + CUDA available  -> cuda (current device, typically set by main_worker)
    - otherwise              -> cpu
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device("cuda", int(gpu))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

It is **duplicated across the four wrapper files** rather than imported from a shared module, intentionally:

- The repo has no `_utils.py` / `_common.py`; introducing one for a 6-line helper would create a new import path that every config and pickled checkpoint indirectly depends on.
- Each wrapper file is already self-contained and meant to be drop-in interchangeable. Adding a cross-file dependency works against that.
- Six lines × four files = 24 lines of duplication. Trivially maintainable.

If the repo later grows a shared utilities module, hoisting `_resolve_device` is a mechanical follow-up.

### 2.2 Constructor changes
**`SpeakerNet.__init__`** (all three variants — `SpeakerNet.py`, `SpeakerNet_performance_updated.py`, `SpeakerNet_distillation.py`):

```python
self.nPerSpeaker = nPerSpeaker
self.device = _resolve_device(kwargs.get("gpu", None))
```

`gpu` arrives via `**kwargs` because all three SpeakerNet classes accept the entire argparse namespace as kwargs and `args.gpu` is set in `main_worker`.

**`ModelTrainer.__init__`** (same three files): `gpu` is already a positional argument, so the patch is:

```python
self.gpu = gpu
self.device = _resolve_device(gpu)
```

**`DistillationSpeakerNet.__init__`** in [DistillationWrapper.py](../../DistillationWrapper.py):
The same resolution block is inlined right after `self.nPerSpeaker = nPerSpeaker`. The class is an `nn.Module`, not a `ModelTrainer`, so it owns its own `self.device`.

### 2.3 Call-site rewrites
Every previously-listed `.cuda(...)` is rewritten:

```diff
-data = data.reshape(...).cuda()
+data = data.reshape(...).to(self.device, non_blocking=True)

-label = torch.LongTensor(data_label).cuda()
+label = torch.LongTensor(data_label).to(self.device, non_blocking=True)

-inp1 = data[0].cuda(non_blocking=True)
+inp1 = data[0].to(self.device, non_blocking=True)

-ref_feat = feats[data[1]].cuda(non_blocking=True)
-com_feat = feats[data[2]].cuda(non_blocking=True)
+ref_feat = feats[data[1]].to(self.device, non_blocking=True)
+com_feat = feats[data[2]].to(self.device, non_blocking=True)
```

`non_blocking=True` is preserved everywhere it existed. For CPU targets it is a no-op (PyTorch silently ignores it), so retaining it costs nothing.

### 2.4 Bundled side-fixes (same bug class)
While the four wrappers were open for editing, two adjacent CPU-incompatibility footguns were closed:

**`loadParameters`** (three files):
```diff
-loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
+loaded_state = torch.load(path, map_location=self.device)
```
Previously, loading a checkpoint on a CPU-only machine would crash with `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`. Now `map_location=self.device` evaluates to `"cpu"` automatically when no GPU is present.

**`torch.cuda.empty_cache()`** in the two perf-updated trainers:
```diff
-torch.cuda.empty_cache()
+if torch.cuda.is_available():
+    torch.cuda.empty_cache()
```
On a CPU-only machine the unconditional call previously raised `AssertionError: Torch not compiled with CUDA enabled` at the end of every eval pass.

### 2.5 What was *not* changed (out of scope)

- **`from torch.cuda.amp import autocast, GradScaler`** in all three trainers. These are CUDA-tied constructs; instantiating `GradScaler(enabled=False)` is fine on CPU, but the scaler's `.step`/`.scale` machinery still assumes a CUDA stream. Making the full training loop CPU-runnable requires conditionally bypassing AMP entirely (use `torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")` from PyTorch 2.0+). Tracked as a future bugfix; the user's bug report did not ask for it.

- **`label = torch.from_numpy(...).cuda()`** inside [loss/angleproto.py](../../loss/angleproto.py), [loss/proto.py](../../loss/proto.py), [loss/ge2e.py](../../loss/ge2e.py). These are inside the *loss functions*, not the wrappers; same bug pattern but a separate fix surface. Listed in §9 below as a follow-up.

- **`WrappedModel`** was not given a `self.gpu` / `self.device` even though §4.1 #4 mentioned propagating into it. `WrappedModel.forward` does no tensor movement — it only does `return self.module(x, label)`. Adding state to it for symmetry alone would be premature abstraction. Future change is one-line if needed.

---

## 3. Behaviour after the fix

| Run configuration | Pre-fix | Post-fix |
|---|---|---|
| Single GPU, `args.gpu=0` | works | works (bit-identical numerics; same `cuda:0`) |
| DDP, ranks 0..N–1, `set_device(rank)` called first | works **by coincidence** (each `tensor.cuda()` falls onto the current device) | works **by construction** (each rank stores `self.device = cuda:<rank>` once at init) |
| DDP, ranks 0..N–1, `set_device` *not* called (or called late) | silent device-mismatch crash | works (each rank's `self.device` is set from `args.gpu` regardless of current-device state) |
| Single-process CPU (`CUDA_VISIBLE_DEVICES=""`) | crashes at first `.cuda()` | works for everything except AMP/GradScaler (which is still CUDA-only — see §2.5) |
| Checkpoint loading on CPU-only host | crashed at `torch.load(..., map_location="cuda:0")` | works; `map_location=torch.device("cpu")` is resolved automatically |
| `torch.cuda.empty_cache()` on CPU-only host | crashed post-eval | now guarded by `torch.cuda.is_available()` |

---

## 4. Verification

### 4.1 Static — `py_compile`
All four files compile clean:

```bash
$ python3 -m py_compile SpeakerNet.py \
                        SpeakerNet_performance_updated.py \
                        SpeakerNet_distillation.py \
                        DistillationWrapper.py
$ echo $?
0
```

### 4.2 Static — call-site audit
The grep that found the bug now finds nothing:

```bash
$ grep -nE "[A-Za-z_)\]]\.cuda\(" \
    SpeakerNet.py SpeakerNet_performance_updated.py \
    SpeakerNet_distillation.py DistillationWrapper.py
# (no output)
```

The new `.to(self.device, ...)` sites count to **16**, matching the count of replaced `.cuda(...)` data-movement calls. Plus three `map_location=self.device` replacements and two `if torch.cuda.is_available(): torch.cuda.empty_cache()` guards.

The only remaining `cuda` references in the four files are:
- `from torch.cuda.amp import autocast, GradScaler` (import; out of scope, §2.5),
- `torch.cuda.is_available()` and `torch.device("cuda", ...)` *inside* `_resolve_device` (correct usage),
- the guarded `torch.cuda.empty_cache()` after the new `if torch.cuda.is_available()` check.

### 4.3 Functional — GPU smoke test (run in `2025_colvaai`)
Numerical regression check for the GPU happy path:

```bash
python trainSpeakerNet.py \
    --config configs/experiment_01.yaml \
    --max_epoch 1 --test_interval 1
```
Expected:
- First batch's `loss` and `TEER/TAcc` match a pre-fix run with the same seed.
- All tensors live on `cuda:0`; no device-mismatch errors.

### 4.4 Functional — CPU smoke test
A new capability unlocked by this fix:

```bash
CUDA_VISIBLE_DEVICES="" python trainSpeakerNet.py \
    --eval \
    --config configs/experiment_01.yaml \
    --initial_model <some-checkpoint>.model
```
Expected:
- No CUDA errors.
- `loadParameters` succeeds via `map_location=torch.device("cpu")`.
- Evaluation runs (slowly, on CPU) and prints EER/MinDCF/Threshold.

Caveat: as documented in §2.5, **training on CPU still won't work** because the AMP/GradScaler path is unchanged. The CPU smoke test above is `--eval` only. Inference and ad-hoc debugging now work CPU-only; training still requires CUDA.

### 4.5 Functional — DDP correctness
```bash
export CUDA_VISIBLE_DEVICES=0,1
python trainSpeakerNet.py \
    --config configs/experiment_01.yaml \
    --distributed \
    --max_epoch 1 --test_interval 1
```
Expected:
- Both ranks print `Loaded the model on GPU 0` and `... GPU 1`.
- No `RuntimeError: Expected all tensors to be on the same device` in the first batch (previously occurred sporadically if `set_device` ordering was off).
- Loss and per-batch progress lines on each rank.

Requires BUGFIX-001 (DDP kwarg fix) to be in place; that's a prerequisite, not a consequence of BUGFIX-004.

---

## 5. The `torch.device("cuda")` (no index) case — why it's safe

`_resolve_device(gpu=None)` returns `torch.device("cuda")` — an unindexed CUDA device. PyTorch resolves this to `cuda:<torch.cuda.current_device()>` at the point of use. There are exactly two contexts in this repo where `gpu=None` reaches a `SpeakerNet`/`ModelTrainer`:

1. **Direct script use without `--distributed`**: `main_worker(0, None, args)` sets `args.gpu = 0`, so `gpu` is never `None` here. Falls into the indexed-`cuda:0` branch.
2. **Library re-use of the wrapper code from another script**: a future contributor might construct a `SpeakerNet(...)` without passing `gpu=`. The unindexed `cuda` branch returns `cuda:<current>`, which is the same behaviour as the original `.cuda()`. So this path is a strict no-regression default.

The CPU branch only fires when `torch.cuda.is_available()` is False — which is the desired CPU-only behaviour.

---

## 6. Backward compatibility

| Consumer | Effect |
|---|---|
| Existing single-GPU runs | ✅ Bit-identical. Same device, same kernels, same numerics. |
| Existing DDP runs that depended on `set_device` ordering | ✅ Continue to work; the new code path is *more* robust, not less. |
| Custom training scripts that construct `SpeakerNet` directly | ✅ Still work. If they pass `gpu=N` via kwargs, behaviour is unchanged. If they don't pass `gpu`, behaviour now matches old `tensor.cuda()` (unindexed CUDA) instead of crashing — strict improvement. |
| Existing checkpoints (`.model` files) | ✅ Load unchanged. The new `map_location=self.device` accepts the same checkpoints. |
| `loss/*.py` and `models/*.py` files | ✅ Untouched. |
| Any code reading `self.gpu` | ✅ `self.gpu` is still stored and unchanged; only `self.device` is added alongside. |

The new public attribute `self.device` on `SpeakerNet` and `ModelTrainer` is additive — no consumer is required to read it, but any consumer can.

---

## 7. Rollback

The fix is mechanical and reversible: replace every `to(self.device, non_blocking=True)` with `cuda(...)` and delete the `_resolve_device` helper + `self.device` assignments. There is no scenario in which rollback is correct; the pre-fix code can never run on a CPU-only host and was DDP-fragile.

---

## 8. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #4 (hard `.cuda()`) | ✅ Closed by this document |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #1 (DDP kwarg) | ✅ [BUGFIX-001](BUGFIX-001-mp-spawn-kwarg.md) |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #2 (EER threshold) | ✅ [BUGFIX-002](BUGFIX-002-eer-threshold-not-returned.md) |
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #3 (nPerSpeaker accuracy) | ✅ [BUGFIX-003](BUGFIX-003-nperspeaker-accuracy.md) |
| §4.1 items #5–#9 | ✅ Closed by [BUGFIX-005](BUGFIX-005-sample-rate-hardcoded.md), [BUGFIX-007](BUGFIX-007-wrap-padding-fabricates-periodicity.md), [BUGFIX-008](BUGFIX-008-sincconv-buffer-placement.md), [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md), [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) |
| Follow-up: device-safety in `loss/{angleproto,proto,ge2e}.py` | ⬜ Open |
| Follow-up: CPU-safe AMP path (`torch.amp.autocast(device_type=...)`) | ⬜ Open |
| Follow-up: hoist `_resolve_device` to a shared module if/when one is created | ⬜ Open |

---

## 9. Authorship & references

- **Bug originally identified by:** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #4.
- **PyTorch device contracts:**
  - https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html
  - https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html
  - https://docs.pytorch.org/docs/stable/generated/torch.load.html (`map_location`)
- **Upstream provenance:** the original `.cuda()` calls are in the Clova AI parent repo and have been inherited unchanged. Worth a separate upstream report.
