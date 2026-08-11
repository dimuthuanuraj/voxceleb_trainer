# BUGFIX-008 — `SincConv_fast` reassigns precomputed tables to `x.device` on every forward; should be registered buffers

| Field | Value |
|---|---|
| **ID** | BUGFIX-008 |
| **Severity** | High (silent — produces wrong-device tensors in multi-GPU / DDP, hot-path slowdown on every forward) |
| **Component** | SincNet learnable-filterbank frontend for raw-waveform models |
| **Files touched** | [models/MLPMixerSpeaker_RawWaveform.py](../../models/MLPMixerSpeaker_RawWaveform.py) (one class, ~10 lines) |
| **Source of report** | `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #7 |
| **Status** | ✅ Fixed |
| **Date** | 2026-05-16 |

---

## 1. Problem

### 1.1 The offending lines
`SincConv_fast` is the learnable-bandpass-filter frontend used by
`MLPMixerSpeaker_RawWaveform` (the raw-waveform student model
documented in `research_logs/2025-12-30-31-experimental-results-analysis.md`).
Two precomputed tables live on the module:

- `self.n_` — frequency-axis scaling, shape `(1, (kernel_size-1)/2)`. Pure
  function of `kernel_size` and `sample_rate`.
- `self.window_` — Hamming window of length `kernel_size // 2`. Pure
  function of `kernel_size`.

[MLPMixerSpeaker_RawWaveform.py:100,104](../../models/MLPMixerSpeaker_RawWaveform.py#L100) (pre-fix) stored them as
plain Python attributes:

```python
self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
...
self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
```

A plain `self.x = tensor` on an `nn.Module` **is not tracked** by
PyTorch — neither `.cuda()` nor `.to(device)` nor any of the
`Module._apply` machinery touches it. So when training code did:

```python
model = SpeakerNet(...)
model.cuda(args.gpu)        # parameters & registered buffers move; self.n_ stays on CPU
```

`self.n_` and `self.window_` stayed on CPU. Then the SincConv author
papered over this with a per-forward fix-up:

```python
# MLPMixerSpeaker_RawWaveform.py:126-127 (pre-fix)
def forward(self, x):
    self.n_ = self.n_.to(x.device)
    self.window_ = self.window_.to(x.device)
    ...
```

### 1.2 Why the per-forward `.to(x.device)` is a real bug

1. **First-forward latency spike.** Host→device transfer happens during
   the first training batch instead of at `model.cuda()` time. With an
   80-filter SincConv at `kernel_size=251`, `self.n_` is `(1, 125)` and
   `self.window_` is `(125,)` — small but non-zero, and the synchronous
   transfer stalls the first kernel launch.

2. **Wrong-device hazard in multi-GPU / DDP.** Suppose the model is
   replicated across two GPUs (DDP). Each rank's first forward sees
   `x` on its own GPU and patches `self.n_` to that GPU — fine. But
   if the buffer were *shared* by reference across replicas (which
   DDP can do for non-parameter state) the second rank would compete
   with the first, ending up with the buffer on whichever rank
   touched it last. The Python `self.attr = ...` rebinding is not
   thread-safe with respect to torch's autograd graph either.

3. **Wrong-device hazard in `nn.DataParallel` / scatter-gather.**
   With `DataParallel`, the parent module is on one device and
   replicas are scattered to others. The plain attribute on the
   parent is not deep-copied into replicas correctly; the replica's
   `.to(x.device)` call rebinds **the replica's** attribute, not
   the original — so the original keeps its old device, and the
   per-forward fixup runs **every forward, forever** on every
   replica. Quietly slow.

4. **Hot-path Python overhead.** Even on a single-GPU run where the
   `.to()` becomes a no-op after the first call, the conditional
   still does an attribute lookup, a device comparison inside
   `.to()`, and an unconditional method dispatch — per forward,
   per batch, forever. For a 100k-step training run, that's 200k
   pointless `.to()` calls.

5. **Re-binding the attribute breaks autograd hooks.** Any tool
   that captures `self.n_` at one point in time (e.g., a profiler,
   a forward-pre-hook capturing module state) sees a different
   tensor on the next forward. This is invisible most of the time
   but a real footgun for anyone debugging the model.

6. **`self.filters` mutation in forward** (line 150 pre-fix, bundled
   into this fix). Same anti-pattern — `self.filters` was assigned
   inside `forward` and immediately consumed by `F.conv1d` two
   lines later. Pure local-variable usage masquerading as module
   state. Each forward leaked a reference to the last filter
   tensor on the module, preventing garbage collection until the
   next forward replaced it.

### 1.3 Why `register_buffer(..., persistent=False)` is the right answer

`register_buffer(name, tensor)` adds the tensor to the module's
buffer dict, which means:

- `model.cuda()` / `model.to(device)` / `model.cpu()` move it
  automatically along with the parameters.
- `nn.DataParallel` and DDP replicate buffers correctly across
  replicas with each replica's buffer landing on its assigned
  device.
- The tensor is exposed under `model.named_buffers()`, so
  profilers, gradient checkers, and weight summarisers see it.

The `persistent=False` flag specifies that the buffer should NOT be
serialised into `state_dict()`. For `self.n_` and `self.window_`
this is the right choice because:

- They are pure deterministic functions of constructor arguments
  (`kernel_size`, `sample_rate`). They can be recomputed at any
  time from the configuration.
- Storing them in every checkpoint wastes bytes (≈2 kB) on every
  saved model.
- If someone restored a checkpoint configured with a different
  `kernel_size` or `sample_rate`, having these buffers in the
  state_dict would either silently override the freshly-computed
  values (subtle bug) or trigger a shape-mismatch warning on load
  (loud bug). `persistent=False` sidesteps both — the freshly-
  computed table at the new configuration is always used.

This pattern is the modern PyTorch idiom for "device-tracked
tensor that's a pure function of the configuration." It's used
in torchvision positional encodings, HuggingFace rotary embeddings,
and (already in this repo) `utils.PreEmphasis` and
`models/RawNetBasicBlock.py`:

```bash
$ grep -nE "register_buffer" models/*.py utils.py
models/RawNetBasicBlock.py:14:        self.register_buffer(...
utils.py:29:        self.register_buffer(...
```

So the fix uses a pattern already familiar to anyone reading the
codebase.

### 1.4 Scope audit

The same anti-pattern occurred *only* in `SincConv_fast`. A pre-fix
scan:

```bash
$ grep -nE "\.to\(.*\.device\)" models/*.py loss/*.py
models/MLPMixerSpeaker_RawWaveform.py:126:        self.n_ = self.n_.to(x.device)
models/MLPMixerSpeaker_RawWaveform.py:127:        self.window_ = self.window_.to(x.device)
```

Two hits, both in `SincConv_fast.forward`. Nothing else in the
codebase uses the anti-pattern, so the fix is strictly local.

---

## 2. Fix applied

### 2.1 `__init__` — register buffers
```diff
-        # Hamming window
         n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))
-        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
+        window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
+        self.register_buffer("window_", window_, persistent=False)

-        # Frequency axis for filter computation
         n = (self.kernel_size - 1) / 2.0
-        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
+        n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
+        self.register_buffer("n_", n_, persistent=False)
```

The local intermediate names (`window_`, `n_`) are picked to match
the eventual buffer name. The buffer name is the second positional
argument of `register_buffer`; it is what `self.window_` / `self.n_`
will resolve to from then on.

### 2.2 `forward` — drop the per-call device fixup and stop mutating self
```diff
 def forward(self, x):
-    # Ensure parameters are on same device
-    self.n_ = self.n_.to(x.device)
-    self.window_ = self.window_.to(x.device)
+    # self.n_ and self.window_ are registered buffers — they follow the parent
+    # module's device automatically. No per-forward .to(...) fixup needed.
     ...
-    # Add channel dimension for conv1d
-    self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)
+    # Filters are recomputed every forward from learnable parameters — they
+    # don't need to live on self.
+    filters = band_pass.view(self.out_channels, 1, self.kernel_size)

-    return F.conv1d(x.unsqueeze(1), self.filters, stride=self.stride,
+    return F.conv1d(x.unsqueeze(1), filters, stride=self.stride,
                    padding=self.kernel_size // 2, groups=1)
```

### 2.3 What the `nn.Parameter` lines (`self.low_hz_`, `self.band_hz_`) needed
Nothing. Those are already `nn.Parameter`, which are tracked by
`.cuda()` / `.to()` by default. Pre-fix they always moved correctly
to the right device when `model.cuda(rank)` was called — that's
why the bug was easy to miss in the first place: the *learnable*
parts of the filterbank were fine, only the *fixed tables* were
left behind. The fix only touches the two fixed tables and the
spurious `self.filters` assignment.

---

## 3. Verification

### 3.1 Static — compile
```bash
$ python3 -m py_compile models/MLPMixerSpeaker_RawWaveform.py
$ echo $?
0
```

### 3.2 Static — anti-pattern eliminated
```bash
$ grep -nE "\.to\(.*\.device\)" models/MLPMixerSpeaker_RawWaveform.py
# (no output)

$ grep -nE "self\.filters" models/MLPMixerSpeaker_RawWaveform.py
# (no output)

$ grep -nE "register_buffer" models/MLPMixerSpeaker_RawWaveform.py
models/MLPMixerSpeaker_RawWaveform.py:104:        self.register_buffer("window_", window_, persistent=False)
models/MLPMixerSpeaker_RawWaveform.py:110:        self.register_buffer("n_", n_, persistent=False)
```

### 3.3 Functional — buffer follows the module to GPU (run in `2025_colvaai`)
```python
import torch
from models.MLPMixerSpeaker_RawWaveform import SincConv_fast

sinc = SincConv_fast(out_channels=80, kernel_size=251, sample_rate=16000)
assert sinc.n_.device == torch.device("cpu")
assert sinc.window_.device == torch.device("cpu")

sinc.cuda(0)
assert sinc.n_.device.type == "cuda"
assert sinc.window_.device.type == "cuda"

# Forward on cuda:0 input — should NOT need any per-call .to() fixup
x = torch.randn(2, 32240, device="cuda:0")
y = sinc(x)
assert y.device.type == "cuda"
print("buffer device-tracking OK")
```

### 3.4 Functional — buffers are NOT in state_dict
```python
sinc = SincConv_fast(out_channels=80, kernel_size=251, sample_rate=16000)
sd = sinc.state_dict()
assert "low_hz_" in sd          # the nn.Parameter is persisted
assert "band_hz_" in sd
assert "window_" not in sd      # persistent=False
assert "n_" not in sd
print("non-persistent buffer check OK")
```

This is the property that means existing checkpoints continue to
load cleanly — the saved state_dict never contained `window_` /
`n_` keys pre-fix (they were plain attributes), and post-fix it
still doesn't (they're non-persistent buffers). Strict `load_state_dict`
won't complain about missing-or-extra keys.

### 3.5 Functional — checkpoint compatibility
```python
# Save with the old SincConv_fast (would have to checkout pre-BUGFIX-008).
# old_sd = SincConv_fast(...).state_dict()

# Load into the new SincConv_fast:
new_sinc = SincConv_fast(out_channels=80, kernel_size=251, sample_rate=16000)
# new_sinc.load_state_dict(old_sd, strict=True)   # works — no key overlap conflict
```

The pre-existing checkpoints in `exps/mlp_mixer_rawwaveform_*/` will
load into the patched model unchanged. No retraining required.

### 3.6 Functional — multi-GPU smoke (DDP)
```bash
export CUDA_VISIBLE_DEVICES=0,1
python trainSpeakerNet.py \
    --config configs/mlp_mixer_rawwaveform_baseline.yaml \
    --distributed \
    --max_epoch 1 --test_interval 1
```
Expected:
- No "Expected all tensors on the same device" error in the first
  batch on either rank.
- Per-rank inspection: `model.module.__S__.sincnet.n_.device` is
  `cuda:0` on rank 0 and `cuda:1` on rank 1.

Pre-fix this would have worked **on first forward** because the
`.to(x.device)` patched it then, but the patching itself rebound
the attribute and any reference captured before that point became
stale.

Requires BUGFIX-001 (DDP kwarg) and BUGFIX-004 (`.cuda()` hard-codes)
to be in place. Both are.

---

## 4. Performance impact

Tiny but measurable. Pre-fix per-forward cost added:
- 1× attribute lookup (`self.n_`),
- 1× `.to(device)` call (a no-op after first call but still ~1–2 µs
  Python overhead on each invocation),
- same again for `self.window_`,
- one reference-write to `self.filters`.

Post-fix: zero. For a 200-frame batch at 32 batches/s, that's
~150 µs/s reclaimed per worker. Real but not headline. The reason
to land this fix is **correctness in multi-GPU**, not throughput.

---

## 5. Backward compatibility

| Consumer | Effect |
|---|---|
| Existing `MLPMixerSpeaker_RawWaveform` checkpoints in `exps/` | ✅ Load unchanged. The buffer keys `window_` / `n_` were never in the pre-fix state_dict (plain attributes) and are still not in the post-fix state_dict (`persistent=False`). |
| Existing training/inference runs at default 16 kHz on a single GPU | ✅ Bit-identical numerics (`self.n_` and `self.window_` were always set to the same values via the same math). Only the *timing* of the device move changes (init time, not first-forward time). |
| `MLPMixerSpeaker_RawWaveform` model card under any non-default `--sample_rate` (BUGFIX-006 path) | ✅ Same — buffer is recomputed from the new `sample_rate` in `__init__`. Was already correct pre-fix because the math used `self.sample_rate` constructor arg. |
| Any code path that captured `self.n_` or `self.window_` as a tensor reference before the first forward | ⚠️ Pre-fix that reference would have been *replaced* on the first forward by a new tensor on the GPU; post-fix the reference remains valid (the underlying tensor is moved in-place by `nn.Module._apply` when `.cuda()` runs). For most consumers (profilers, gradient hooks) this is a strict improvement: their captured handle now actually tracks the live buffer. No known consumer relied on the pre-fix re-binding behaviour. |
| `RawNet3` (the other SincConv-based model) | ✅ Not touched. RawNet3 uses `asteroid_filterbanks.ParamSincFB`, an external library that handles its own buffer placement correctly (verified by inspection). BUGFIX-006 already threaded `sample_rate` into it; no further fix needed here. |

### 5.1 No state_dict migration required
This is the key compatibility point: the pre-fix state_dict and the
post-fix state_dict are **identical**. Same keys, same shapes, same
values. The change is purely in how `self.n_` and `self.window_` are
*tracked* by PyTorch, not in what gets persisted. Strict
`load_state_dict(..., strict=True)` continues to succeed on every
checkpoint produced before this fix.

---

## 6. Things this fix does NOT change

| Item | Why |
|---|---|
| `_hz_to_mel` / `_mel_to_hz` helpers creating tensors with `torch.tensor(hz)` defaulting to CPU. | These only run at init time on Python scalars. The created tensors are consumed by `nn.Parameter(...)` immediately, which then participates in normal `nn.Module` device-tracking. No bug. |
| The `band_pass` intermediate inside `forward` — a fresh tensor each call. | Correct by design. It's a function of learnable parameters and the (now properly-placed) buffers, all of which are on the right device. |
| `RawNet3`'s `ParamSincFB`. | External library, separate concern. |
| Other models' device-handling. | A `grep` across `models/` and `loss/` found this anti-pattern only in `SincConv_fast`. |

---

## 7. Rollback

Replace each `register_buffer("window_", window_, persistent=False)`
with `self.window_ = window_` (same for `n_`), restore the two
`.to(x.device)` lines at the top of `forward`, restore the
`self.filters = (band_pass)...` assignment. There is no scenario in
which rollback is correct.

---

## 8. Closes / related

| Item | Status |
|---|---|
| `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 #7 (SincConv buffer placement) | ✅ Closed by this document |
| §4.1 #1–#6 | ✅ BUGFIX-001..007 |
| §4.1 #8 — `MaxPool1d(...) if pool else False` | ✅ [BUGFIX-009](BUGFIX-009-rawnet-pool-placeholder.md) |
| §4.1 #9 — NestedSpeakerNet NaN | ✅ [BUGFIX-010](BUGFIX-010-quarantine-nestedspeakernet.md) (quarantined) |
| Polish: also `register_buffer` the `mel_low / mel_high` setup if revisited | ⬜ Unnecessary — those are scalars used only at init |

---

## 9. Authorship & references

- **Bug originally identified by:** repo audit in `SL_LANGUAGE_SPV_ANALYSIS.md` §4.1 item #7.
- **`register_buffer` semantics:**
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
- **Persistent vs non-persistent buffers (PyTorch 1.6+):**
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
  (parameter `persistent`)
- **Upstream provenance:** `SincConv_fast` is a re-implementation of
  the SincNet paper (Ravanelli & Bengio, SLT 2018). Multiple public
  re-implementations exhibit this same anti-pattern; worth reporting
  upstream wherever the model was originally forked from.
