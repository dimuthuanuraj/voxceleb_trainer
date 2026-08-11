# FEATURE-004 — Cross-lingual fine-tune flag

**Status:** in progress (this PR)
**Type:** feature
**Relates to:** §3.1 #1 of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-001 (`ssl_freeze` follows the same pattern)
**Owner:** SL-SPV
**Touches:** `SpeakerNet.py`, `trainSpeakerNet.py`

## 1. Motivation

§3.1 #1 of the analysis doc calls out cross-lingual fine-tuning as Tier-1: "with ≤10k SL utterances, fine-tune with low LR; with ≥100k, also unfreeze the SincConv front-end." Today the trainer supports the loose form of this (set `initial_model: path/to/english.model` and a low `lr`), but **selective freezing of the front-end** — the part that actually stops a small SL corpus from washing out the English-pretrained representations — has no config knob.

FEATURE-001 already added `ssl_freeze` for the SSL-encoder variant. This feature generalises the pattern to the non-SSL paths (mel-frontend / SincConv) and adds the two missing knobs: an LR multiplier, and an `initial_model` assertion so a typo doesn't silently train from scratch.

## 2. What this is and is not

**Is:**
- A `--finetune` flag that flips three knobs as a unit: assert `initial_model`, optionally freeze named submodules, optionally scale `lr`.
- A `--finetune_freeze` list (CLI string or YAML list) of either named aliases or dotted module paths.
- A `--finetune_lr_multiplier` (default 1.0; common values 0.1 / 0.01).
- A startup diagnostic: `[finetune] Training N params (M frozen across K modules)`.

**Is not:**
- Layer-wise learning-rate decay (LLRD).
- Warmup schedule.
- Discriminative fine-tuning (different LR per layer).
- A new optimizer or scheduler.

Those are all separate features; this one is the minimum to make cross-lingual fine-tuning a one-line YAML change.

## 3. Design

### 3.1 CLI / config surface

```
--finetune                          # bool, default False
--finetune_freeze "frontend"        # comma-list of aliases or dotted paths
--finetune_lr_multiplier 0.1        # default 1.0
```

YAML form:
```yaml
finetune: true
finetune_freeze: [frontend]          # or ["__S__.encoder", "__L__"]
finetune_lr_multiplier: 0.1
initial_model: exps/<english-run>/model/best.model
```

### 3.2 Freeze aliases

The freeze walker resolves these short names against the model's submodule tree:

| Alias | Resolves to | Use case |
|---|---|---|
| `frontend` | `__S__.torchfb`, `__S__.instancenorm` (whichever exist) | Mel / SincConv path — the literal "front-end" §3.1 #1 calls out |
| `ssl_encoder` | `__S__.encoder` (if exists; FEATURE-001 SSL models only) | Equivalent of the existing `ssl_freeze: true` |
| `backbone` | All of `__S__` | Train only the loss head — very small-corpus regime |
| `loss_head` | `__L__` | Keep the English softmax; rarely useful but supported for ablation |

Direct dotted paths (e.g. `__S__.encoder.encoder.layers.0`) are passed through verbatim — useful for advanced users targeting a specific transformer layer.

Unmatched aliases / paths **fail loudly at startup** with the list of available top-level submodule names. Silent skip would mask a typo for a full training run.

### 3.3 Hook point

Inside `ModelTrainer.__init__` (in `SpeakerNet.py`), between assigning `self.__model__` and constructing the optimizer:

1. If `finetune` is False → no-op.
2. Else:
   1. Resolve `finetune_freeze` entries to a list of submodule objects.
   2. Walk each submodule's parameters, set `requires_grad = False`, set submodule to `eval()` (so BN stats don't drift).
   3. Scale `kwargs['lr']` by `finetune_lr_multiplier`.
   4. Print the diagnostic line.
3. The optimizer is built over `[p for p in self.__model__.parameters() if p.requires_grad]`. Without freezing, this is identical to today's `self.__model__.parameters()` — same param order, same optimizer state shape.

### 3.4 `initial_model` assertion

In `trainSpeakerNet.py` `main_worker`, immediately after argparse + YAML are resolved (before any model construction):

```python
if args.finetune and not args.initial_model:
    raise ValueError(
        "--finetune requires --initial_model to be set; refusing to fine-tune "
        "from random init. Set initial_model: <checkpoint> in YAML or pass "
        "--initial_model <path> on the CLI."
    )
```

Fails before GPU allocation so the typo surfaces in <1 s instead of after model load.

### 3.5 Composition with the FEATURE-001 `ssl_freeze` flag

`ssl_freeze: true` (existing) and `finetune_freeze: [ssl_encoder]` (new) end up doing the same thing for SSL models. The new flag is the more general path; the old one stays for back-compat. When both are set, the union is frozen (idempotent — requires_grad=False is set twice, which is fine).

### 3.6 Saved checkpoint shape

`saveParameters` saves `self.__model__.module.state_dict()` — frozen params are still saved, just with their unchanged values. Loading is unchanged. So a fine-tuned checkpoint is interchangeable with a from-scratch one at evaluation time.

## 4. Risk and rollback

- **Default off** — `finetune: false` (default) is byte-identical to today.
- **Optimizer param-list rebuild** is the riskiest piece. Mitigation: an explicit startup print of `[finetune] Training N params (M frozen)` and a hard assertion that `N > 0` (refuse to optimize zero params).
- **BatchNorm + frozen modules** — frozen modules are set to `.eval()` so BN running stats don't continue updating against the new corpus. This matches HuggingFace's standard pattern and the existing `ssl_freeze` path (FEATURE-001). Without this step, frozen weights are stable but BN stats drift, which is almost always the wrong behaviour.
- **Rollback** — single revert removes the feature; saved checkpoints stay loadable (the format is unchanged).

## 5. Validation plan (not blocking this PR)

1. Run any existing config with `--finetune false` (default) → identical loss / EER trajectory to a baseline pre-feature run. **Mandatory before merging into a research branch.**
2. Run with `--finetune --initial_model <english.model> --finetune_freeze "frontend" --finetune_lr_multiplier 0.1` on a small VoxCeleb1 dev split → confirm:
   - Startup print shows expected (N_train, M_frozen) counts.
   - Loss decreases from a much lower starting point than from-scratch.
   - Frozen module weights are byte-identical to the loaded checkpoint after 1 epoch.
3. Negative test: `--finetune` without `--initial_model` → ValueError before model load.

## 6. Follow-ups (intentionally not in this PR)

- **Layer-wise LR decay (LLRD).** Different LR for different transformer layers. Useful for >100k-utterance SL fine-tuning when the full encoder is unfrozen.
- **Warmup scheduler.** Linear warmup over the first 1–2 epochs is standard for fine-tuning; the existing `steplr` scheduler doesn't have it.
- **Discriminative fine-tuning.** Per-param-group LR overrides via config.
- **Re-init last layer.** Reset the loss head for the new speaker set instead of inheriting the English softmax — required when `nClasses` changes.

## 7. Cross-references

- `docs/bugfixes/FEATURE-001-language-aware-frontend.md` — `ssl_freeze` is the SSL-specific predecessor of this flag.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 #1 — original ask.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §9 action 5 — "fine-tune ... with `initial_model: <best English checkpoint>`, low LR (`1e-4`), 60 epochs." This feature is the config knob that action 5 needs.
