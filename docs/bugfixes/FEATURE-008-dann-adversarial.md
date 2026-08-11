# FEATURE-008 — Domain-adversarial training (DANN) for language & channel invariance

**Status:** in progress (this PR)
**Type:** feature (training)
**Relates to:** §3.3 #11 of SL_LANGUAGE_SPV_ANALYSIS.md; FEATURE-007 (the *opposite* objective on language)
**Owner:** SL-SPV
**Touches:** `SpeakerNet.py`, `trainSpeakerNet.py`

## 1. Motivation

§3.3 #11 of the analysis doc proposes DANN-style adversarial heads for channel and language invariance. The mechanism is Ganin & Lempitsky's *Unsupervised Domain Adaptation by Backpropagation* (ICML 2015): attach a domain classifier (language or channel) downstream of the encoder, but route its gradient through a **Gradient Reversal Layer (GRL)** before it reaches the encoder. The encoder is then trained to **fool** the domain classifier — producing features that are *uninformative* about the domain.

For cross-lingual SV: a DANN head on language strips language-specific cues from the embedding, so the speaker loss is forced to discriminate using language-invariant features. The same idea on channel labels yields channel-robust embeddings — useful when the deployment channel (telephony, microphone, codec'd audio) differs from the training distribution.

## 2. Relationship to FEATURE-007 (lang-aux head)

**These two features have opposite objectives on the same signal.** A short table:

| Feature | Head | Gradient to encoder | Effect on embedding |
|---|---|---|---|
| FEATURE-007 lang-aux | language classifier | **standard** (positive) | retains language info |
| FEATURE-008 DANN lang | language classifier through GRL | **reversed** (negative × λ) | strips language info |

If you set `lang_aux: true` AND `dann_lang: true` against the same lookup, the two gradients on the encoder roughly cancel — don't do that. Pick one based on your hypothesis:

- **Use lang-aux** when language ID is itself useful context the embedding should preserve (the literature shows 10–15 % EER gain when this hypothesis holds, e.g. when you have language-aware downstream code).
- **Use DANN-lang** when language is a *confound* to strip, especially in code-switched or mixed-language enrolment/test scenarios where the speaker should be matched despite language differences.

DANN-channel doesn't conflict with anything else and is generally useful when train/test channels differ.

## 3. What this is and is not

**Is:**
- A `GradientReversalFn` (a `torch.autograd.Function`) implementing the identity forward / `-λ·grad` backward operation.
- A `DANNHead` module: 2-layer MLP (`Linear → ReLU → Linear`) over the GRL'd embedding + CE loss with `ignore_index=-1`.
- Two independent instances supportable in one model: `dann_lang_head` and `dann_channel_head`. Each has its own lookup file, weight λ, and num-classes.
- Per-speaker lookup files (same `<spk_label_int> <domain_label_int>` format as FEATURE-007).
- Constant λ. The Ganin-style ramp `λ(p) = 2/(1+exp(-γ·p)) − 1` is documented as a follow-up — it requires per-epoch updates and is worth it only after you've validated the constant-λ baseline.

**Is not:**
- A per-utterance lookup. Per-speaker only (extend later).
- A gradient-clipping or stability mechanism for the adversarial term. DANN is known to be brittle; if you see oscillation, drop λ.
- Combined CDAN / MDD style domain adaptation. Out of scope (Tier-3 follow-ups).

## 4. Design

### 4.1 GRL

```python
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)               # identity

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

Forward is identity. Backward returns the negated, scaled gradient. The `None` is the (no-grad) partial w.r.t. the `lambda_` argument.

### 4.2 DANNHead

```python
DANNHead(embedding_dim, num_classes, hidden_dim=256, lambda_=1.0):
    classifier = Sequential(Linear(emb, hid), ReLU(), Linear(hid, num_classes))
    crit = CrossEntropyLoss(ignore_index=-1)
    forward(x, label):
        rev = GradientReversalFn.apply(x, self.lambda_)
        logits = classifier(rev)
        loss = crit(logits, label)
        return loss, prec
```

`self.lambda_` is a plain Python attribute (not a Parameter), mutable by the trainer for schedule support later. The 2-layer MLP is the canonical DANN discriminator size — sufficient capacity that the encoder has to work to fool it.

### 4.3 CLI / config surface

```
--dann_lang                            # bool, default False
--dann_lang_weight 1.0                 # default 1.0 (multiplies the aux loss before adding to total)
--dann_lang_lambda 0.1                 # GRL strength; default 0.1 (smaller than FEATURE-007 because of the sign flip on the encoder)
--dann_lang_num_classes 4
--dann_lang_label_file path.txt        # required when --dann_lang is set

--dann_channel                         # bool, default False
--dann_channel_weight 1.0
--dann_channel_lambda 0.1
--dann_channel_num_classes 3           # e.g. mic / phone / codec
--dann_channel_label_file path.txt
```

The `weight` and `lambda` are conceptually different:
- `weight` scales the contribution of the DANN loss to `total_loss` (also affects the discriminator's own update).
- `lambda` scales the gradient that flows BACK to the encoder via GRL (only affects the encoder).

In the original DANN paper they're identical; we expose them separately so the discriminator can train faster than the adversarial pressure on the encoder, which empirically stabilises training.

YAML form (lookup file format identical to FEATURE-007):
```yaml
dann_lang: true
dann_lang_weight: 1.0
dann_lang_lambda: 0.1
dann_lang_label_file: data/sl_celeb/spk_lang_lookup.txt
dann_lang_num_classes: 4
```

### 4.4 Hook point

`SpeakerNet.forward` — same place as FEATURE-007. After computing `outp = self.__S__.forward(data)` (shape `[nPerSpeaker * B, D]`) and before the `nPerSpeaker` reshape, accumulate the auxiliary losses:

```python
extra_loss = 0
if self.lang_aux:        # FEATURE-007
    extra_loss = extra_loss + self.lang_aux_weight   * aux_loss
if self.dann_lang:       # FEATURE-008
    extra_loss = extra_loss + self.dann_lang_weight  * dann_lang_loss
if self.dann_channel:    # FEATURE-008
    extra_loss = extra_loss + self.dann_channel_weight * dann_channel_loss
```

All three reuse `label.repeat_interleave(self.nPerSpeaker)` for label expansion; lookup buffers are independent.

### 4.5 Composition with other features

| Feature | Composes? | Notes |
|---|---|---|
| FEATURE-001 (SSL frontend) | ✅ | DANN heads are model-agnostic; sit above the embedding. |
| FEATURE-002 (AS-Norm) | ✅ | Eval-only; DANN heads are silent at eval. |
| FEATURE-003 (per-language eval) | ✅ | DANN's *goal* is to strip language; per-language eval verifies whether the per-language EERs equalise. |
| FEATURE-004 (fine-tune freeze) | ⚠️ | Freezing the encoder + running DANN does nothing useful (the adversarial gradient is reversed but can't update frozen weights). Skip DANN in pure freeze-and-fine-tune-head runs. |
| FEATURE-005 (LLRD) | ✅ | DANN heads land in depth bucket 0 alongside the speaker loss head. |
| FEATURE-006 (ECAPA backbone) | ✅ | Pure backbone swap. |
| FEATURE-007 (lang-aux) | ⚠️ | Opposite objective on the same signal — see §2. Don't enable both on `lang`. Channel-DANN + lang-aux is fine. |

### 4.6 Saved checkpoint shape

DANN heads register `dann_lang_head.*` / `dann_channel_head.*` / `dann_lang_spk_to_lang` / `dann_channel_spk_to_lang` keys in the state_dict. Non-strict loading already handles missing/extra keys, so a checkpoint trained with DANN can be loaded for eval without DANN (the keys are ignored) and vice versa. The lookup buffers are saved alongside the head weights — convenient for reproducibility but means a renamed lookup file doesn't invalidate a checkpoint.

## 5. Risk and rollback

- **Default off** — `dann_lang: false` and `dann_channel: false` (defaults) are byte-identical to today. Head modules aren't instantiated.
- **Training instability** — DANN is known to oscillate when λ is too high early. Mitigation: start with `dann_lang_lambda: 0.05–0.1`. If loss oscillates, halve it.
- **Lookup file errors** — strict validation via `_load_lang_lookup` (reused from FEATURE-007) — fails loudly on bad lines / out-of-range labels.
- **Rollback** — single revert; saved checkpoints stay loadable (non-strict).

## 6. Validation plan (not blocking this PR)

1. Smoke test in this PR:
   - Gradient sign: build a model with `dann_lang=true`, run forward + backward, verify that gradients on the encoder parameters point *away* from the DANN-loss minimum (sign matches `-λ`).
   - Default-off path: instantiating with `dann_lang=false` and `dann_channel=false` produces no DANN heads (`hasattr(model, 'dann_lang_head') == False`).
2. On the first SL pilot run with `dann_lang=true, dann_lang_lambda=0.1`:
   - Expect per-language EERs to **equalise** (smaller gap between si / ta / cs) vs the baseline.
   - Pooled EER may improve, stay flat, or worsen — DANN trades per-domain peak for cross-domain consistency. Both outcomes are informative.
3. λ sweep on the pilot: `dann_lang_lambda ∈ {0.05, 0.1, 0.2, 0.5}`. Look for the elbow where pooled EER starts dropping.

## 7. Follow-ups (intentionally not in this PR)

- **λ ramp schedule.** `lambda(p) = 2 / (1 + exp(-γ·p)) − 1` where `p = epoch / max_epoch`. The trainer would call `model.set_dann_lambda(epoch, max_epoch)` at each epoch boundary.
- **Per-utterance domain labels.** Required for channel labels that vary within a speaker (which they usually do — same speaker recorded in studio AND telephony).
- **Gradient-norm balancing** between speaker loss and DANN loss (GradNorm / DTP).
- **CDAN / MDD** — strictly stronger domain adaptation methods.

## 8. Cross-references

- `docs/bugfixes/FEATURE-007-lang-aux-head.md` — the *opposite* objective on language. Don't combine on the same signal.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.3 #11 — the original ask.
- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*, ICML 2015. https://arxiv.org/abs/1409.7495
