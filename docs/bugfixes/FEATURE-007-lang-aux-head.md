# FEATURE-007 — Multi-task language-ID auxiliary head

**Status:** in progress (this PR)
**Type:** feature (training)
**Relates to:** §3.2 #7 of SL_LANGUAGE_SPV_ANALYSIS.md
**Owner:** SL-SPV
**Touches:** `SpeakerNet.py`, `trainSpeakerNet.py`

## 1. Motivation

§3.2 #7 of the analysis doc proposes a multi-task language-ID head: on top of the shared backbone, attach a small classifier predicting language (si / ta / en / mix). The total loss becomes `L_speaker + λ·L_lang`. This forces the embedding to be informative about *both* the speaker and the language it was spoken in, which the literature reports yields **10–15% relative EER reduction** on cross-lingual SV.

The mechanism: speaker discrimination + language discrimination are not orthogonal — a model that can identify which language an utterance is in has learned phonetic / prosodic structure that also helps disambiguate speakers across languages. Adding the aux head is a low-cost regulariser that biases the encoder toward those features.

## 2. What this is and is not

**Is:**
- A `nn.Linear(nOut → num_lang_classes)` head attached to the same embedding consumed by the speaker loss head.
- A per-speaker `lang_label_file` lookup (one line per speaker: `<spk_label_int> <lang_label_int>`) loaded once into a `LongTensor[nClasses]` buffer on the model.
- Training-time only — the aux head is unused at eval time, so AS-Norm / per-language eval / inference paths see no change.
- Cross-entropy with `ignore_index=-1` so speakers missing from the lookup file silently skip the aux loss for their samples (don't poison the gradient).

**Is not:**
- A per-utterance language label. The first version uses per-speaker only (one language per speaker_id). Per-utterance lookup is documented as a follow-up; needed only for corpora where the SAME speaker has both monolingual and code-switched utterances and you want them labelled separately.
- A gradient reversal / adversarial setup. That's §3.3 #11 (DANN-style), a separate Tier-3 feature.
- A change to the data loader. The lang label is derived from the speaker label via the lookup at the model's forward step.
- A change to evaluation. The aux head's outputs are not used downstream.

## 3. Design

### 3.1 CLI / config surface

```
--lang_aux                         # bool, default False
--lang_aux_weight 0.3              # λ; default 0.3
--lang_aux_num_classes 4           # default 4 (si/ta/en/mix)
--lang_aux_label_file path.txt     # required when --lang_aux is set
```

YAML:
```yaml
lang_aux: true
lang_aux_weight: 0.3
lang_aux_num_classes: 4
lang_aux_label_file: data/sl_celeb/spk_lang_lookup.txt
```

`spk_lang_lookup.txt` format (one speaker per line):
```
# spk_label_int lang_label_int    [comment]
0  0    # si
1  0    # si
2  1    # ta
3  2    # en
4  3    # mix
...
```

Speakers absent from the file get `lang_label = -1` and are excluded from the aux loss for their samples (via `ignore_index=-1`).

### 3.2 Architecture

```
audio
  ↓ backbone (mel / SincConv / SSL)
  ↓ pooling head
embedding x  [B*P, nOut]                ← lang-aux head taps here
  ↓
  ├─→ speaker loss head (AAM-Softmax / etc.)  → L_speaker, prec1
  └─→ lang_aux_head: Linear(nOut → num_lang)  → L_lang, lang_prec
                                                 ↑ CE with ignore_index=-1

total_loss = L_speaker + λ·L_lang
```

Embedding `x` is the model's forward output **before** the `nPerSpeaker` reshape — so each utterance contributes one prediction, not one per speaker group.

### 3.3 Hook point

`SpeakerNet.forward(data, label)` — already produces `outp` shaped `[nPerSpeaker * B, D]`. When `lang_aux` is on:

1. Expand `label` (shape `[B]`, dtype long, speaker IDs) by `repeat_interleave(nPerSpeaker)` → `[nPerSpeaker * B]`.
2. Look up `lang_label = self.lang_spk_to_lang[label_expanded]` (the lookup buffer registered at `__init__`).
3. Forward `outp` through `LangAuxHead` to get `(lang_loss, lang_prec)`.
4. Run the existing speaker loss path unchanged, get `(nloss, prec1)`.
5. Return `(nloss + λ · lang_loss, prec1)`.

`prec1` continues to be the speaker classification accuracy (so existing TEER/TAcc display logic is unaffected). The lang accuracy `lang_prec` is computed but not reported in this first pass — adding it to the per-epoch print is a one-line follow-up if you want it.

### 3.4 Lookup buffer

```python
self.register_buffer('lang_spk_to_lang', _load_lang_lookup(path, nClasses))
```

The buffer lives on the model's device and moves automatically with `.to(device)` / DDP. Shape: `[nClasses]`, dtype `long`. Missing speakers carry `-1`.

### 3.5 Composition with existing features

| Feature | Composes? | Notes |
|---|---|---|
| FEATURE-001 (SSL frontend) | ✅ | Lang-aux head sits above the SSL output projection; no SSL-specific code. |
| FEATURE-002 (AS-Norm) | ✅ | Eval-time only; aux head is silent at eval. |
| FEATURE-003 (per-lang eval) | ✅ | Same as above. |
| FEATURE-004 (fine-tune freeze) | ✅ | If you freeze `loss_head` via FEATURE-004, the speaker loss head is frozen — the lang-aux head is a *separate* module, not under `__L__`, so it stays trainable. To freeze it too, add `module.lang_aux_head` to `finetune_freeze`. |
| FEATURE-005 (LLRD) | ✅ | `lang_aux_head` lands in depth bucket 0 (top), same as the speaker loss head. |
| FEATURE-006 (ECAPA-TDNN) | ✅ | Pure backbone swap; lang-aux head is model-agnostic. |

## 4. Risk and rollback

- **Default off** — `lang_aux: false` (default) is byte-identical to today. Loss path is unchanged, the head module isn't even instantiated.
- **Missing lookup entries** — silently excluded via `ignore_index=-1`. The aux loss reports the fraction of valid labels as a sanity check at startup.
- **Wrong num_classes** — if a lookup entry has `lang_label >= num_classes`, `CrossEntropyLoss` will raise a CUDA assertion at the first batch. Add an explicit range check in `_load_lang_lookup`.
- **Rollback** — single revert; saved checkpoints keep loading via the standard `state_dict` interface (the `lang_aux_head.*` keys are unused at eval and absent on pre-feature checkpoints; non-strict loading already handles missing/extra keys per BUGFIX-019's path).

## 5. Validation plan (not blocking this PR)

1. Smoke test in this PR: instantiate `SpeakerNet` with `lang_aux=true`, push a `[B, T]` audio batch + speaker labels through, assert returned loss is finite and includes the aux term. **Done in the implementation step below.**
2. On the first SL pilot run with `lang_aux=true` + `lang_aux_weight=0.3`:
   - Expect speaker EER drop of ~5–15% relative vs `lang_aux=false`, all else equal.
   - Aux-loss-validity diagnostic prints "X of Y speakers have language labels" at startup.
3. λ sweep: `lang_aux_weight ∈ {0.1, 0.3, 0.5, 1.0}` → broad plateau expected; not a knife-edge.

## 6. Follow-ups (intentionally not in this PR)

- **Per-utterance lang labels.** A separate file `<rel_audio_path> <lang_label_int>` consulted by the data loader. Required only for corpora with same-speaker mixed-language utterances.
- **Gradient reversal layer.** §3.3 #11 — make the aux head adversarial (language-invariant embeddings). Different research direction.
- **TensorBoard reporting** of lang-aux accuracy.
- **Per-lang accuracy** in the validation print.

## 7. Cross-references

- `docs/bugfixes/FEATURE-001-language-aware-frontend.md` — the SSL front-end provides multilingual features the aux head exploits.
- `docs/bugfixes/FEATURE-003-per-language-eval.md` — eval-time per-lang reporting; reads from a separate test-list, doesn't touch this feature.
- `SL_LANGUAGE_SPV_ANALYSIS.md` §3.2 #7 — original ask; §9 action 5/6 — the SL pilot run that validates this.
