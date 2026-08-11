# NaN Debugging Guide

This guide collects the NaN / Inf failure modes that this repository has
actually encountered, the mitigations already wired into the code, and a
recipe for triaging a new incident with [`analyze_nan_debug.py`](analyze_nan_debug.py).
It is meant for someone who has just opened a log and seen `loss = nan`
or whose validation EER suddenly jumped to 100% with no other warning.

---

## 1. The one NaN incident this repository has on record

So far this codebase has produced exactly one documented NaN cascade.
Knowing it is the right starting point because most second incidents
will resemble the first one.

| Field | Value |
|---|---|
| **Model** | `NestedSpeakerNet` (now quarantined in [`models/experimental/`](models/experimental/)) |
| **Configs** | `configs/nested_4level.yaml`, `configs/nested_4level_asp.yaml`, `configs/nested_5level_asp.yaml` |
| **Failure epoch** | 11 (BatchNorm variant) / 12 (GroupNorm + adaptive-pool variant) |
| **Best EER before NaN** | 21.72% / 18.71% (both worse than the ResNetSE34L baseline) |
| **Diagnosed root cause** | Combinatorial gradient-path explosion (O(2^N) paths through the nested aggregation) × anti-correlated audio features (r ≈ −0.23). The two compound: clipped gradients still amplify rather than regularise. |
| **Authoritative writeups** | [`docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md`](docs/bugfixes/BUGFIX-010-quarantine-nestedspeakernet.md), [`research_logs/2025-12-29-nested-learning-experiment.md`](research_logs/2025-12-29-nested-learning-experiment.md) |

The reason for leading with this is that the incident already cost the
project three full stabilisation attempts before the architectural
root-cause was understood. If a new NaN appears, check whether you are
re-running a quarantined config before chasing it as a fresh bug.

---

## 2. Mitigations already in this codebase

Several common NaN sources are already addressed by code that runs on
every training step. Verify these are still in place before diagnosing
a new failure — a NaN with all of these active points at the model
architecture or the data, not at the trainer.

| Mechanism | Where | What it catches |
|---|---|---|
| `torch.nn.utils.clip_grad_norm_(..., max_norm=5.0)` | [`SpeakerNet_performance_updated.py:159`](SpeakerNet_performance_updated.py#L159), [`SpeakerNet_distillation.py:224`](SpeakerNet_distillation.py#L224) | Bounded gradient magnitudes from any single batch. |
| `torch.cuda.amp.GradScaler` | [`SpeakerNet.py:89`](SpeakerNet.py#L89), [`SpeakerNet_performance_updated.py:105`](SpeakerNet_performance_updated.py#L105), [`SpeakerNet_distillation.py:162`](SpeakerNet_distillation.py#L162) | FP16 overflow during the AMP forward — scaler automatically drops the step if non-finite gradients are detected. |
| `torch.cuda.amp.autocast()` context | three trainers | Per-op precision selection — softmax / norm ops stay in FP32. |
| `_pad_short_with_dither` (silence + low-amplitude Gaussian) | [`DatasetLoader.py`](DatasetLoader.py), [`DatasetLoader_performance_updated.py`](DatasetLoader_performance_updated.py) | Replaces the old `numpy.pad(..., 'wrap')` behaviour. Short clips no longer get periodic repetition that can interact pathologically with mel filters. See [BUGFIX-007](docs/bugfixes/BUGFIX-007-wrap-padding-fabricates-periodicity.md). |
| `register_buffer(..., persistent=False)` for SincConv window / lookup | [`models/MLPMixerSpeaker_RawWaveform.py`](models/MLPMixerSpeaker_RawWaveform.py) (via [BUGFIX-008](docs/bugfixes/BUGFIX-008-sincconv-buffer-placement.md)) | Window tensor moves with the model — no stale CPU buffer mixed into a CUDA forward. |

If a NaN still appears with all of these active, the problem is
upstream — either an architectural issue (like NestedSpeakerNet) or a
data-side issue (corrupted audio file, all-zero clip, mislabelled
class index).

---

## 3. Failure modes that have plausibly produced a NaN in this codebase

The five modes below cover most NaN incidents in PyTorch
speaker-verification training. They are ordered by how common they are
*in practice* (not by how plausible they sound on paper).

### 3.1 Mixed-precision overflow

**Symptom:** Loss is finite for many steps, then suddenly NaN. Often
preceded by a brief spike. Usually accompanied by a warning from
`GradScaler` about a non-finite gradient and a step being skipped.

**Why it happens:** FP16 has a max representable magnitude of ~65k.
A single batch with unusual feature norms (e.g., a very loud clip, or
a long pause that the trainer happens to dither in a way that briefly
produces a large mel-magnitude) can overflow during an inner product
or a softmax. `GradScaler` is designed to absorb this — it skips the
step and lowers its scale — but if the underlying signal is
*persistently* too large, the scaler keeps having to skip and eventually
the model destabilises anyway.

**Diagnostic:**
- Run [`analyze_nan_debug.py --log <path>`](analyze_nan_debug.py) on
  the failing run's log. If it shows several `nan`/`inf` loss lines
  clustered near the failure, AMP is implicated.
- Check the scaler-skip warnings: PyTorch emits a stderr message when
  the scale is reduced. A run that does this every few hundred steps
  is on the edge.

**Mitigation:** Disable AMP with `--mixedprec false` on the config /
argparse, *or* clip the input mel-magnitudes (currently no clip — see
the dB-clip discussion in [`SAMPLING_RATE_GUIDE.md`](SAMPLING_RATE_GUIDE.md)).

### 3.2 AAM-Softmax with a tiny class count

**Symptom:** Loss is finite but very small for several epochs, then
NaN. TEER plateaus near 0 falsely. Most common when nClasses is small
(< 200, e.g., mini-VoxCeleb's 140-speaker subset) and margin is large.

**Why it happens:** `loss/aamsoftmax.py` computes
`cos(θ + m)` via the standard hard-margin trick. For embeddings that
are already very close to their class centroid, the post-margin
logit can be at the saturating end of the sigmoid; further gradient
descent pushes it through. With few classes the softmax denominator
is small, so `1 - cos(θ + m)` can become numerically zero, and
the subsequent `log` produces `-inf`.

**Diagnostic:** Run
[`analyze_nan_debug.py --checkpoint <last clean>.model --verbose`](analyze_nan_debug.py).
The `|max|` column for the AAM-Softmax `weight` parameter tells you
whether the centroids have been blown up.

**Mitigation:** Lower the AAM margin (default 0.2 → try 0.1), reduce
the scale (default 30 → try 20), or move to AM-Softmax during the
last finetune epochs.

### 3.3 Gradient-path combinatorial explosion (the NestedSpeakerNet case)

**Symptom:** Stable training for the first ~10 epochs with a *good-looking*
loss curve, then loss → NaN over one or two iterations. Gradient clipping
warnings appear in the log just before the failure.

**Why it happens:** Architectures that route information from each layer
into multiple later layers (nested / densely-connected variants) create
O(2^N) gradient paths. For audio features (high per-frame variance,
anti-correlated adjacent levels) the sum of these contributions grows
faster than `clip_grad_norm_` can contain. See BUGFIX-010 §1 for the
quantitative version.

**Diagnostic:** Look at the architecture topology, not at the data.
If the model has multi-path aggregation and the configs use the
quarantined path, you are reproducing the documented incident — re-read
BUGFIX-010 first.

**Mitigation:** There is no in-code mitigation. The diagnosed root
cause is architectural; hyperparameter tuning has been tried and does
not converge. See [`models/experimental/README.md`](models/experimental/README.md)
for the promotion criteria a revived variant would need to meet.

### 3.4 Data-side: silent / all-zero clips

**Symptom:** NaN in the very first epoch, usually within the first few
hundred iterations. Sometimes accompanied by a `LinAlgError` from
SVD inside the SAP / ASP pooling.

**Why it happens:** An all-zero mel-spectrogram passed through
`log(mel + eps)` is fine — the eps protects it. But the SAP attention
weights, computed as `softmax(W * h + b)`, can be uniform over an
all-zero hidden state, and the subsequent weighted sum has zero
variance. Some encoders divide by that variance.

**Diagnostic:** Reproduce by running a single forward pass with the
problematic batch. If you have the iteration number from
`analyze_nan_debug.py --log`, the `train_list.txt` line matching it
identifies the file.

**Mitigation:** Filter the train list — see the polish item flagged
in BUGFIX-007 §5 (a corpus-prep tool that drops clips below
`max_audio` samples or below an RMS threshold).

### 3.5 Learning rate too high after a scheduler step

**Symptom:** NaN at the exact epoch the scheduler resets / restarts.
Common with cosine restarts or `OneCycleLR`.

**Why it happens:** A scheduler that re-warms or jumps the LR up can
push it above the architecture's stability budget. With AMP this is
usually absorbed by the scaler; without AMP it lands directly in the
weights.

**Diagnostic:** Compare the `--log` output of `analyze_nan_debug.py`
against the trainer's per-epoch LR print line (`"LR <value>"`). A NaN
within one epoch of an LR jump is suggestive.

**Mitigation:** Halve the LR floor or use a gentler schedule
(`StepLR` with the existing default `lr_decay=0.95` is conservative
and has been stable in this repository).

---

## 4. Triage workflow

When a new NaN incident lands, the recommended sequence is:

1. **Confirm the failure is reproducible.** Re-run with the same seed
   for ~30 minutes. Intermittent NaN that does not reproduce is almost
   always FP16 overflow (§3.1) — increase `GradScaler`'s initial scale
   floor or disable AMP.

2. **Run the log analyser.**

   ```bash
   python analyze_nan_debug.py --log logs/<latest>.log --verbose
   ```

   This identifies:
   - The last clean epoch / iteration.
   - The first NaN epoch / iteration.
   - Whether the NaN appeared as a single spike or as a sustained
     cascade.

3. **Run the checkpoint analyser on the *last clean* checkpoint.**

   ```bash
   python analyze_nan_debug.py \
       --checkpoint exps/<exp>/model/model00000N.model \
       --verbose
   ```

   The last clean checkpoint is the most informative — by the time the
   NaN propagates to a saved file, every parameter is non-finite and
   the diagnostic is just "everything is NaN". The checkpoint *before*
   the failure shows which parameter started growing first.

4. **Cross-check against §1.** If the model is `NestedSpeakerNet` or
   uses a quarantined config, stop. The failure is documented in
   BUGFIX-010 and the architectural fix is "use a different model".

5. **Decide which §3 mode is implicated** and apply the matching
   mitigation. The five modes do not require any code change *to
   diagnose* — they only require a code or config change to *fix*.

6. **Enable anomaly detection for the next reproduction.**

   ```python
   import torch
   torch.autograd.set_detect_anomaly(True)
   ```

   This makes the backward pass slow but pinpoints the *operation*
   that first produced a NaN gradient. Use only on a known-failing run;
   never leave it on for full training.

---

## 5. Limits of this guide

- This guide is built from the empirical record of one documented
  incident (BUGFIX-010 / NestedSpeakerNet). The §3 list is what the
  speaker-verification literature and PyTorch documentation collectively
  consider common; only §3.3 has been confirmed in this repository.
  The other four are plausible-but-unobserved here.

- The script [`analyze_nan_debug.py`](analyze_nan_debug.py) reports
  *what* went non-finite. It does not infer *why*. Mapping a
  diagnostic to a §3 mode is human work, informed by the surrounding
  log and the architecture being trained.

- This guide is not a substitute for reading the trainer's source.
  Both [`SpeakerNet_performance_updated.py`](SpeakerNet_performance_updated.py)
  and [`SpeakerNet_distillation.py`](SpeakerNet_distillation.py) have
  inline behaviour around `GradScaler` / `autocast` that is occasionally
  surprising; if a NaN looks AMP-flavoured, read the actual
  `train_network` loop in the relevant trainer file rather than
  reasoning by analogy with stock PyTorch examples.
