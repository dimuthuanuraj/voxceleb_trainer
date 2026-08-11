# FEATURE-005 — Layer-wise learning-rate decay (LLRD)

**Status:** in progress (this PR)
**Type:** feature
**Relates to:** §3.1 #1 (fine-tuning) and §3.1 #2 (SSL front-ends) of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-004 (composes naturally)
**Owner:** SL-SPV
**Touches:** `SpeakerNet.py`, `trainSpeakerNet.py`

## 1. Motivation

LLRD gives each model layer a different learning rate that decays exponentially from the top of the network to the bottom: deeper layers learn general transferable features (low-level acoustics, phoneme-like patterns) that should change *less* during cross-lingual fine-tuning; upper layers carry task-specific features that should change *more*. It sits between "freeze everything" (the lower bound that FEATURE-004 already supports) and "fine-tune everything at one LR" (the upper bound that today is the default once `initial_model` is set).

For the SL pilot's most promising path — fine-tuning a frozen-encoder SSL model (FEATURE-001) with ≤10k SL utterances — LLRD is the standard recipe: BERT, WavLM, Whisper, and the wav2vec2 family all use it during downstream fine-tuning, with reported gains of **3–10% relative EER** on cross-lingual SV.

For non-transformer encoders (mel/SincConv, ResNet) the benefit is marginal because the layer hierarchy is less semantic. We support those paths but default off; the win is concentrated on the SSL paths.

## 2. What this is and is not

**Is:**
- A `--llrd` flag that switches optimizer construction from a flat trainable-param list to a list of param_groups, each with its own LR.
- A `--llrd_decay` knob (default 0.9; common values 0.8 – 0.95). LR for depth `d` is `base_lr × decay^d`.
- A `--llrd_layer_pattern` knob that picks how parameters are bucketed by depth. Aliases: `ssl` (HuggingFace transformer layers via `\.encoder\.encoder\.layers\.(\d+)\.`), `mlpmixer` (`\.mixer_blocks\.(\d+)\.`). Custom regex strings also accepted.
- A startup diagnostic table: per-depth (#params, lr).

**Is not:**
- Warmup. Separate feature.
- Discriminative cosine schedule. Separate feature.
- Per-parameter custom LR overrides. Use param_groups directly if you need that.

## 3. Design

### 3.1 CLI / config surface

```
--llrd                              # bool, default False
--llrd_decay 0.9                    # float
--llrd_layer_pattern ssl            # alias or raw regex
```

YAML form:
```yaml
llrd: true
llrd_decay: 0.9
llrd_layer_pattern: ssl              # or "mlpmixer", or a raw regex
finetune: true
finetune_freeze: []
finetune_lr_multiplier: 1.0
initial_model: exps/<english-run>/model/best.model
lr: 1e-4
```

### 3.2 Depth assignment

For each parameter `name`:

| Match | Depth | Notes |
|---|---|---|
| `__L__.*` (loss head) | `0` | Always top |
| `__S__.attention.*`, `__S__.bn.*`, `__S__.fc.*` (pooling+projection head) | `0` | Top, above the encoder |
| Matches the layer-pattern regex with index `N` | `(max_N - N) + 1` | Last transformer layer is depth 1; first is depth `max_N + 1` |
| `feature_extractor`, `feature_projection`, `masked_spec_embed`, `pos_conv_embed` (HF SSL CNN front) | `max_N + 2` | Deepest |
| Anything else | `(max_N + 1) // 2` | Middle bucket, safe fallback |

Then `lr_d = base_lr × decay^d`. With `decay=0.9` and a 12-layer transformer:
- Loss head + pooling: `base_lr`
- Last transformer layer: `0.9 × base_lr`
- First transformer layer: `0.31 × base_lr`
- Feature extractor: `0.28 × base_lr`

### 3.3 Hook point

Inside `ModelTrainer.__init__`, immediately after the FEATURE-004 freeze step and before the optimizer is built. If `llrd` is on:

1. Detect the maximum layer index via the layer-pattern regex.
2. If no matches → print warning, fall back to flat trainable list (LLRD is a no-op for this model).
3. Else: build the param_groups list, pass it to the optimizer in place of the flat list.

The Optimizer wrappers in `optimizer/adam.py` and `optimizer/sgd.py` already pass their first arg through to `torch.optim.Adam` / `torch.optim.SGD`, both of which accept a list of param_groups transparently. No changes needed to the optimizer modules. The `lr` kwarg becomes the default for any group that doesn't specify one — every group does specify one, so it's effectively unused (but still required by the wrapper signature).

### 3.4 Composition with FEATURE-004

LLRD runs **after** freezing. Frozen params have `requires_grad=False` and are excluded from every depth bucket. The diagnostic line reports the effective trainable count per depth.

`finetune_lr_multiplier` is applied to `kwargs['lr']` before LLRD reads it, so:
```
final_lr_at_depth_d = base_lr × finetune_lr_multiplier × decay^d
```

This composes cleanly: a research config can say "fine-tune the SSL encoder with 0.1× base LR overall, decaying 0.9 per layer" by setting `finetune_lr_multiplier: 0.1` + `llrd_decay: 0.9`.

### 3.5 Schedulers

The existing schedulers (`steplr`, etc.) operate on `optimizer.param_groups`, scaling each group's LR uniformly each step. So LLRD's per-group ratios are preserved across training — only the absolute scale changes. No scheduler modifications needed.

## 4. Risk and rollback

- **Default off** — `llrd: false` (default) is byte-identical to today.
- **Optimizer state shape** — with LLRD, the optimizer holds N param_groups instead of 1. This affects checkpoint compatibility *only* for optimizer state checkpoints (not model checkpoints, which are unchanged). The trainer doesn't currently save / load optimizer state across runs, so this is moot.
- **Layer-pattern miss** — handled by a warning + fallback to flat list, so a bad regex doesn't silently degrade to depth-0 only.
- **Rollback** — single revert removes the feature; saved model checkpoints stay loadable.

## 5. Validation plan (not blocking this PR)

1. Run any existing config with `llrd: false` (default) → identical loss / EER trajectory to a baseline pre-feature run.
2. Run an SSL config (`configs/language_aware_ssl_wavlm.yaml` augmented with `llrd: true, llrd_layer_pattern: ssl, llrd_decay: 0.9`) → confirm:
   - Startup table shows ~14 depth buckets (12 transformer + 1 CNN front + 1 head) with monotonically decreasing LRs.
   - Loss decreases at least as fast as the flat-LR baseline.
   - Per-language EER on the SL pilot improves vs flat-LR by ~3–10% relative (literature expectation).
3. Negative test: run with `llrd_layer_pattern: bogus.regex.nothing.matches` → warning printed, fallback to flat list, behaviour identical to `llrd: false`.

## 6. Follow-ups (not in this PR)

- **Linear / cosine warmup.** Standard 1–2 epoch warmup before LLRD takes effect — improves the first few epochs of fine-tuning. Separate feature; would touch the scheduler modules.
- **Cosine LR schedule with restarts.** Often paired with LLRD for the longest fine-tunes (>100 epochs). Out of scope here.
- **Per-group weight-decay overrides.** LayerNorm / bias parameters typically get zero weight-decay; this is a separate tuning knob.

## 7. Cross-references

- `docs/bugfixes/FEATURE-004-cross-lingual-finetune.md` — composes naturally; LLRD reads the scaled LR FEATURE-004 produces.
- `docs/bugfixes/FEATURE-001-language-aware-frontend.md` — LLRD's headline benefit is on the SSL paths FEATURE-001 added.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.1 #1 / #2, §9 action 5.
