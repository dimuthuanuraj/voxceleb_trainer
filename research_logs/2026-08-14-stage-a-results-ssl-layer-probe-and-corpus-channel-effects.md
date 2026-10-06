# Stage A Results, the SSL Layer Probe, and Corpus Channel Effects

**Date:** 2026-08-14 · **Status:** Stage A complete (14/14) and under evaluation; Stage F running; two new split sets and one new condition added
**Predecessor:** `2026-08-12-experiment-harness-frontend-and-english-study.md` (harness design)
**Code:** `experiments/` · **Trainer:** `trainSpeakerNet.py` — still unmodified

---

## 0. Headline

Stage A finished. Three findings came out of it, and two of them change how the
results must be reported.

| | Finding |
|---|---|
| **1** | **The Tamil advantage is largely a channel artifact.** `slr127_tamil` pools three collection sites; ~48% of its impostor trials are cross-batch and get rejected on recording channel rather than speaker identity. Measured: EER 0.475% cross-batch vs **1.298% same-batch** — a **2.5–3.2×** inflation, consistent across three architectures. |
| **2** | **The SSL frontend was reading its worst layer.** A direct probe of WavLM shows speaker information decreasing *monotonically* with depth. The trainer's default `--ssl_layer -1` is **1.81× worse (si) / 2.90× worse (ta)** than layer 0. This explains why the 95M SSL model came last in Stage A. |
| **3** | **ECAPA-512 matches ECAPA-1024.** 3.642 vs 3.721 (si val), 1.032 vs 0.810 (ta val) — the capacity control suggests these corpora are data-limited, not capacity-limited. Provisional: the gaps are inside the noise band until the paired bootstrap runs. |

---

## 1. Stage A results

7 architectures × {si, ta}, AAM-softmax, speaker-disjoint splits, per-epoch
validation. All 14 completed.

| architecture | si val EER | ta val EER | params |
|---|---|---|---|
| `ecapa1024` | 3.721 | **0.810** | 14.46 M |
| `ecapa512` | **3.642** | 1.032 | 5.99 M |
| `resnetse34v2` | 4.242 | 1.417 | 7.37 M |
| `resnetse34l` | 4.362 | 1.761 | 1.40 M |
| `mlpmixer` | 4.582 | 1.154 | 7.71 M |
| `vggvox` | 5.162 | 1.902 | 3.64 M |
| `ssl_wavlm` | 7.083 | 4.837 | 94.98 M |

Held-out test evaluation (partial, in progress at time of writing):

| | val EER | val DCF | test EER | test DCF | +AS-Norm |
|---|---|---|---|---|---|
| `ecapa1024_ta` | 0.810 | 0.0702 | 0.894 | 0.0695 | **0.814** |
| `ecapa512_ta` | 1.032 | 0.0684 | 1.035 | 0.0753 | **0.984** |
| `ecapa1024_si` | 3.721 | 0.2423 | 4.386 | 0.3068 | **3.942** |
| `ecapa512_si` | 3.642 | 0.2611 | 4.295 | 0.3028 | **4.204** |

Two things worth recording. The **val→test gap is small** (+0.08 to +0.67 pp),
so per-epoch validation selection is not badly overfitting. And **AS-Norm helps
every time** (−2% to −10% relative), for free at inference — which is the
calibration effect that justified holding it out of training and measuring it as
its own factor.

> On Sinhala the ECAPA-512/1024 ordering **flips between views**: validation
> favours 512, raw-cosine test favours 512, AS-Norm test favours 1024. Three
> orderings from three views of the same two models, all within ~0.4 pp. This is
> exactly why the paired bootstrap is the deciding statistic and no single
> column is a claim.

---

## 2. Why Tamil looked so much better — it mostly did not

Every architecture put Tamil 3–5× below Sinhala. That ordering is *opposite* to
the data volumes (Tamil trains on 63,338 utterances, Sinhala on 129,042) and
opposite to speaker counts (Tamil has more test speakers, 131 vs 91), so neither
explains it. Two candidate causes were tested directly.

### 2.1 Utterance duration — a small effect

| | median | p90 | < 4 s |
|---|---|---|---|
| `slr52_sinhala` | 4.00 s | 6.20 s | 46.8% |
| `slr127_tamil` | 5.16 s | 10.51 s | 29.5% |

Tamil utterances are ~38% longer, and duration is one of the strongest known
effects in SV. But re-probing both corpora at a **matched 2-second** duration
barely moved the gap:

```
probe length     Sinhala    Tamil    ratio
    4.0 s         22.30%   11.05%    2.02x
    2.0 s         21.08%   12.03%    1.75x
```

Removing nearly all of Tamil's duration advantage closes only ~13% of the gap.
Not the main cause.

### 2.2 Collection-batch channel — the main cause

`slr127_tamil` pools **three collection sites**: `ISTL`, `MILE`, `MICI`.
Splitting its impostor trials on whether both sides come from the same site
(measured on the evaluated `ecapa1024_ta` checkpoint):

| impostor type | count | mean score | EER |
|---|---|---|---|
| **cross-batch** | 4,805 | **−0.0433** | **0.475%** |
| **same-batch** | 5,151 | **+0.0527** | **1.298%** |
| all (as reported) | 9,956 | — | 0.894% |

Cross-batch impostors score *negative* — rejected almost for free because the
channel differs, not the speaker. ~48% of the trial list is that easy kind.
Reproduced across architectures:

```
experiment                  all     same    cross   inflation
A_ecapa1024_aamsoftmax_ta  0.894   1.298   0.475     2.73x
A_ecapa512_aamsoftmax_ta   1.035   1.338   0.542     2.47x
A_mlpmixer_aamsoftmax_ta   1.235   1.669   0.521     3.20x
```

`slr52_sinhala` has no comparable structure — one homogeneous collection, so
**none** of its impostors get that free rejection. Comparing the two corpora's
EERs therefore compares protocols as much as languages.

This is the concrete form of the QC audit's `eta^2(SNR | speaker) = 0.68–0.74`:
channel is partly identity in these corpora.

> **Reporting rule.** Quote the **same-batch** figure for `slr127_tamil`
> (1.298% for ecapa1024, not 0.894%). The like-for-like Sinhala/Tamil gap is
> then ~3.4×, not ~4.9×.

### 2.3 What was built

* `experiments/splits/slr127_tamil_samebatch/` — the same trials with impostor
  pairs constrained to one collection batch. Targets untouched. **No retraining
  needed**: only the trial list differs, so existing checkpoints are re-scored.
* `evaluate.py` now scores every Tamil model against that list as
  `test_samebatch`, automatically.
* `analyze.py` now emits a **channel/collection-batch diagnostic** for any
  corpus whose speaker ids carry a batch prefix, reporting all / same / cross
  EER and the inflation factor. Corpora with no prefix scheme are skipped, which
  is itself informative.

---

## 3. The SSL layer probe

Stage A's biggest surprise was `ssl_wavlm` finishing **last** on both languages
(4.84% ta, 7.08% si) despite being 6× larger than the winner. Two structural
causes were suspected and one was measured.

`experiments/tools/ssl_layer_probe.py` measures speaker information per layer
with **no training at all**: run the frozen encoder, mean-pool each hidden
state, L2-normalise, cosine-score the validation trials.

| layer | si EER% | ta EER% |
|---|---|---|
| **0** (CNN output) | **22.30** | **11.05** |
| 3 | 29.27 | 15.78 |
| 6 | 36.50 | 24.48 |
| 9 | 38.90 | 31.68 |
| **12** ← `--ssl_layer -1` | **40.33** | **32.08** |

**Monotonic degradation with depth, in both languages.** The default reads
features **1.81× worse (si), 2.90× worse (ta)** than the best available.

This is what masked-prediction pretraining predicts: the objective rewards
recovering a masked frame from context, which drives upper layers toward
phonetic content and treats speaker identity as nuisance. The last layer is the
one the objective worked hardest to make speaker-invariant.

The second cause is the frozen encoder: only **594,304 of 94,981,240 parameters
(0.63%)** train in the Stage A SSL runs.

### 3.1 A pre-registered prediction was falsified

The 2026-08-12 log predicted speaker information would peak **mid-stack**, and
`ssl_wavlm_mid` was built at layer 6 on that basis. Layer 6 measures 36.5% —
barely better than the last layer, and far off layer 0's 22.3%. **The prediction
was wrong**, which is what pre-registration is for. The entry was kept (as the
mid-curve point measured with a *trained* head rather than mean pooling) and the
falsification recorded in `registry.py` rather than edited away.

---

## 4. New experiments

### 4.1 Stage F additions (running)

| run | what it tests |
|---|---|
| `ssl_wavlm_low` | layer 0 — the probe's measured optimum |
| `ssl_wavlm_ft` | encoder **unfrozen**, lr 1e-4 + LLRD 0.9 — tests the 0.63%-trainable ceiling |

`ssl_wavlm_low` smoke result is already striking: **15.79% EER after 2 epochs on
1,344 utterances**, where the last-layer model needed a full 64k-utterance epoch
to reach 13.59%.

`ssl_wavlm_ft` carries a **documented deviation** from the pinned optimiser: lr
1e-4 with layer-wise decay instead of 1e-3, because 1e-3 destroys a pretrained
transformer in the first epochs. Batch 24, A40-only.

### 4.2 New condition: `si_celeb` (running)

`slceleb2026_sinhala` as a full training condition, not just a probe — the
7-architecture Stage A sweep repeated on it. Experiment ids carry the condition
(`A_ecapa1024_aamsoftmax_si_celeb_s42`), so no result from a different corpus
can be confused with it.

**Why it matters:** `slr52_sinhala` has `n_utts == n_sessions` — every utterance
is its own recording — so it has no within-speaker session structure and a
"cross-session" trial is indistinguishable from a cross-utterance one.
`slceleb2026` is the **only Sinhala corpus here with real sessions** (session =
YouTube video id; 100 of 123 speakers appear in ≥2 videos, median 3). Its target
trials are therefore built **strictly cross-session**, which makes its absolute
EER meaningful rather than optimistic.

**The cost is power, and it is severe.** 123 speakers total → 82 train / 10 val
/ 31 test, of which 24 have the ≥2 sessions a cross-session trial needs. That
gives `n_eff <= 34` and **MDE ~11.5 pp**. The protocol is honest; the precision
is not. Read the paired contrasts, never the absolute EER.

---

## 5. Bugs and infrastructure fixed today

| # | Problem | Fix |
|---|---|---|
| 1 | **All three nodes rebooted at 12:56** mid-Stage-A. Five runs died; five more failed instantly with `rc=255` because sshd answered `pam_nologin` while booting, and the queue *consumed* them without retrying. | `run_queue.py` now retries `rc=255`-with-no-`final.json` (infrastructure) up to `--max-retries 3` with exponential backoff. A run that wrote a result is never retried. |
| 2 | **Ctrl+C did three different things at once** — orphaned a job on one node, killed jobs on another, and kept launching new ones. | Jobs launch with `start_new_session=True` so SIGINT reaches only the queue; a signal handler then stops scheduling and stops each running job deliberately. `--keep-running` preserves detach behaviour. Verified end-to-end. |
| 3 | **`pkill -f <pattern>` over ssh kills its own shell** — the remote `bash -c` carries the pattern in its own command line. | Collect PIDs first, skip `$$`/`$PPID`. |
| 4 | **Three slots sat blocked for 5.5 h** on jobs that had already written `exit_code 0`; their `ssh -t` clients never exited because stray dataloader workers held the pty. | `final.json` is now the authority on completion; a lingering ssh client gets 180 s then is closed and the slot freed. |
| 5 | **Resume truncated its own history** — `ExperimentRecorder` rewrites `epochs.jsonl` from memory, which on resume started empty. | Load existing rows on resume. Verified: all five reboot-interrupted runs have gap-free, duplicate-free epoch sequences across the boundary. |
| 6 | **`gen_scripts.py` clobbered a stage manifest** when a stage was generated in two invocations (Stage E: 1 recorded, 8 on disk). | Merge by `exp_id`, keep `selection_history`, report `scripts_on_disk_not_in_manifest`. |
| 7 | **The ROC plot failed on every run** — `tuneThreshold.ComputeErrorRates` returns lists and the plotting code computes `1 - fnrs`. | A `sitecustomize` on `PYTHONPATH` wraps those in a list *subclass* adding `__rsub__`. `trainSpeakerNet.py` untouched; minDCF verified identical; ROC images now written. |
| 8 | **`batch_size` must not exceed the training-speaker count.** The sampler forbids repeating a speaker within a batch, so with 82 SLCeleb speakers and `batch_size 200`, `round_down(82, 200) = 0` — zero batches, `ZeroDivisionError`. | `params()` caps `batch_size` at `0.75 × n_classes`. si/ta/en_matched unaffected (verified no drift). |
| 9 | **Stage F's per-entry `overrides` were not forwarded**, so `ssl_wavlm_ft` would have silently trained at lr 1e-3 and wrecked the encoder. | `stage_f()` passes them through. |
| 10 | **compute-node-1's two T4s were never used** — `gpurun.sh` polled only nodes 3 and 4. | Added (3→5 GPUs). `run_queue.py` is now memory-aware (`min_gpu_mb` per architecture) so a 112M hybrid is never sent to a 15 GB card, and gained `--nodes` so a second queue can run without double-booking. |
| 11 | **The English conditions leaked into every experiment's evaluation**, adding ~57k trials each. | `auto_transfer: False`; opt in with `--transfer-en`. |

**compute-node-2 (10.222.1.118) remains unusable** — two T4s present per
`lspci`, but no `/dev/nvidia*` and the driver will not load. Needs root.
Recovering it would add 2 GPUs.

---

## 6. Current state

| | status |
|---|---|
| Stage A (si, ta) | ✅ 14/14 trained · evaluation in progress |
| Stage A (`si_celeb`) | 🔄 7 runs, on the A40 |
| Stage F (front ends) | 🔄 14 runs, incl. the 2 new WavLM variants |
| Stage E, H | ⏸ scripts generated (8 + 4) |
| Stage B, C, D, G, M | ⏸ deferred on results |

---

## 7. What to do next

1. Finish the Stage A evaluation, then `analyze.py` — the paired contrasts are
   what turn §1's ordering into a claim.
2. Read Stage F's `ssl_wavlm_low` / `_lw` / `_ft` against `ssl_wavlm`. If the
   probe's story holds, all three should beat it substantially, and the fitted
   layer weights (`tools/layer_weights.py`) should concentrate low.
3. Re-score Tamil against `test_samebatch` everywhere and use those numbers.
4. Treat `si_celeb` as the honest-protocol Sinhala result and `si` as the
   larger-but-channel-confounded one; report both.

## 8. Files added today

```
experiments/tools/ssl_layer_probe.py      layer-wise speaker information, no training
experiments/tools/build_extra_splits.py   same-batch Tamil + SLCeleb primary split
experiments/tools/stop_queue.py           find/stop stray jobs on the cluster
experiments/common/runtime_patches/       sitecustomize ROC fix (trainer untouched)
experiments/splits/slr127_tamil_samebatch/
experiments/splits/slceleb2026_sinhala_split/
experiments/analysis/ssl_layer_probe_wavlm-base-plus_{si,ta}.json
```
