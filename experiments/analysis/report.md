# Sinhala / Tamil speaker verification — experimental analysis

15 experiments; 11 with held-out test evaluation.

All splits are speaker-disjoint (`experiments/tools/build_splits.py`).
Model selection used validation trials only; the test set was scored once,
at the validation-selected checkpoint.

## 1. Headline results (held-out test set)

| arch | loss | cond | val EER % | test EER % | Δ val→test | test EER AS-Norm % | minDCF | d' | minCllr | params | GFLOPs/2s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ecapa512 | aamsoftmax | si | 3.642 | 4.295 | 0.653 | 4.204 | 0.303 | 3.768 | 0.175 | 5994688 | 1.927 |
| ecapa1024 | aamsoftmax | si | 3.721 | 4.396 | 0.674 | 3.942 | 0.309 | 3.752 | 0.172 | 14461056 | 5.167 |
| resnetse34v2 | aamsoftmax | si | 4.242 | 4.688 | 0.446 | 4.638 | 0.304 | 3.708 | 0.178 | 7373388 | 9.255 |
| mlpmixer | aamsoftmax | si | 4.582 | 4.950 | 0.368 | 4.648 | 0.350 | 3.574 | 0.193 | 7713024 | 3.031 |
| resnetse34l | aamsoftmax | si | 4.362 | 5.001 | 0.639 | 4.809 | 0.379 | 3.642 | 0.193 | 1404054 | 1.788 |
| vggvox | aamsoftmax | si | 5.162 | 5.888 | 0.726 | 5.797 | 0.391 | 3.399 | 0.232 | 3637952 | 1.060 |
| ecapa1024 | aamsoftmax | ta | 0.810 | 0.894 | 0.084 | 0.814 | 0.070 | 5.492 | 0.036 | 14461056 | 5.167 |
| ecapa512 | aamsoftmax | ta | 1.032 | 1.035 | 0.002 | 0.984 | 0.075 | 5.337 | 0.041 | 5994688 | 1.927 |
| mlpmixer | aamsoftmax | ta | 1.154 | 1.256 | 0.102 | 1.185 | 0.083 | 5.213 | 0.047 | 7713024 | 3.031 |
| resnetse34v2 | aamsoftmax | ta | 1.417 | 1.386 | -0.031 | 1.266 | 0.099 | 5.387 | 0.051 | 7373388 | 9.255 |
| resnetse34l | aamsoftmax | ta | 1.761 | 1.697 | -0.063 | 1.607 | 0.139 | 4.835 | 0.065 | 1404054 | 1.788 |

## 2. Validation results (all runs)

| experiment | arch | loss | cond | val EER % | minDCF | best ep | epochs | s/epoch |
|---|---|---|---|---|---|---|---|---|
| A_ecapa512_aamsoftmax_si_s42 | ecapa512 | aamsoftmax | si | 3.642 | 0.261 | 27 | 43 | 418.460 |
| A_ecapa1024_aamsoftmax_si_s42 | ecapa1024 | aamsoftmax | si | 3.721 | 0.242 | 41 | 57 | 355.410 |
| A_resnetse34v2_aamsoftmax_si_s42 | resnetse34v2 | aamsoftmax | si | 4.242 | 0.273 | 9 | 25 | 536.760 |
| A_resnetse34l_aamsoftmax_si_s42 | resnetse34l | aamsoftmax | si | 4.362 | 0.335 | 21 | 37 | 343.790 |
| A_mlpmixer_aamsoftmax_si_s42 | mlpmixer | aamsoftmax | si | 4.582 | 0.294 | 32 | 48 | 364.780 |
| A_vggvox_aamsoftmax_si_s42 | vggvox | aamsoftmax | si | 5.162 | 0.344 | 30 | 46 | 255.370 |
| A_ssl_wavlm_aamsoftmax_si_s42 | ssl_wavlm | aamsoftmax | si | 7.083 | 0.459 | 41 | 60 | 646.860 |
| A_ecapa1024_aamsoftmax_ta_s42 | ecapa1024 | aamsoftmax | ta | 0.810 | 0.070 | 46 | 60 | 302.360 |
| A_ecapa512_aamsoftmax_ta_s42 | ecapa512 | aamsoftmax | ta | 1.032 | 0.068 | 38 | 54 | 325.330 |
| A_mlpmixer_aamsoftmax_ta_s42 | mlpmixer | aamsoftmax | ta | 1.154 | 0.084 | 58 | 60 | 307.700 |
| A_resnetse34v2_aamsoftmax_ta_s42 | resnetse34v2 | aamsoftmax | ta | 1.417 | 0.093 | 35 | 51 | 254.450 |
| A_resnetse34l_aamsoftmax_ta_s42 | resnetse34l | aamsoftmax | ta | 1.761 | 0.122 | 54 | 60 | 296.450 |
| A_vggvox_aamsoftmax_ta_s42 | vggvox | aamsoftmax | ta | 1.902 | 0.119 | 56 | 60 | 192.190 |
| A_ssl_wavlm_aamsoftmax_ta_s42 | ssl_wavlm | aamsoftmax | ta | 4.837 | 0.276 | 35 | 51 | 562.100 |

## 3. Paired significance tests

Speaker-clustered paired bootstrap on the EER difference over identical
trials. `significant` means the 95% interval excludes zero.

| system A | system B | EER A | EER B | Δ pp | CI low | CI high | p | sig |
|---|---|---|---|---|---|---|---|---|
| A_ecapa512_aamsoftmax_si_s42 | A_vggvox_aamsoftmax_si_s42 | 4.295 | 5.847 | -1.553 | -2.221 | -0.946 | 0.000 | True |
| A_ecapa1024_aamsoftmax_si_s42 | A_vggvox_aamsoftmax_si_s42 | 4.396 | 5.847 | -1.452 | -2.211 | -0.829 | 0.000 | True |
| A_resnetse34v2_aamsoftmax_si_s42 | A_vggvox_aamsoftmax_si_s42 | 4.688 | 5.847 | -1.159 | -1.863 | -0.494 | 0.000 | True |
| A_ecapa1024_aamsoftmax_ta_s42 | A_vggvox_aamsoftmax_ta_s42 | 0.894 | 1.949 | -1.055 | -1.306 | -0.743 | 0.000 | True |
| A_ecapa512_aamsoftmax_ta_s42 | A_vggvox_aamsoftmax_ta_s42 | 1.035 | 1.949 | -0.914 | -1.149 | -0.624 | 0.000 | True |
| A_mlpmixer_aamsoftmax_si_s42 | A_vggvox_aamsoftmax_si_s42 | 4.950 | 5.847 | -0.897 | -1.542 | -0.362 | 0.000 | True |
| A_resnetse34l_aamsoftmax_si_s42 | A_vggvox_aamsoftmax_si_s42 | 5.000 | 5.847 | -0.847 | -1.519 | -0.262 | 0.002 | True |
| A_ecapa1024_aamsoftmax_ta_s42 | A_resnetse34l_aamsoftmax_ta_s42 | 0.894 | 1.698 | -0.803 | -1.086 | -0.522 | 0.000 | True |
| A_ecapa512_aamsoftmax_si_s42 | A_resnetse34l_aamsoftmax_si_s42 | 4.295 | 5.000 | -0.706 | -1.181 | -0.182 | 0.006 | True |
| A_mlpmixer_aamsoftmax_ta_s42 | A_vggvox_aamsoftmax_ta_s42 | 1.246 | 1.949 | -0.703 | -0.936 | -0.436 | 0.000 | True |
| A_ecapa512_aamsoftmax_ta_s42 | A_resnetse34l_aamsoftmax_ta_s42 | 1.035 | 1.698 | -0.663 | -0.904 | -0.420 | 0.000 | True |
| A_ecapa512_aamsoftmax_si_s42 | A_mlpmixer_aamsoftmax_si_s42 | 4.295 | 4.950 | -0.655 | -1.059 | -0.201 | 0.001 | True |
| A_ecapa1024_aamsoftmax_si_s42 | A_resnetse34l_aamsoftmax_si_s42 | 4.396 | 5.000 | -0.605 | -1.172 | -0.137 | 0.007 | True |
| A_resnetse34v2_aamsoftmax_ta_s42 | A_vggvox_aamsoftmax_ta_s42 | 1.376 | 1.949 | -0.573 | -0.834 | -0.360 | 0.000 | True |
| A_ecapa1024_aamsoftmax_si_s42 | A_mlpmixer_aamsoftmax_si_s42 | 4.396 | 4.950 | -0.554 | -0.979 | -0.177 | 0.004 | True |
| A_ecapa1024_aamsoftmax_ta_s42 | A_resnetse34v2_aamsoftmax_ta_s42 | 0.894 | 1.376 | -0.482 | -0.636 | -0.201 | 0.000 | True |
| A_mlpmixer_aamsoftmax_ta_s42 | A_resnetse34l_aamsoftmax_ta_s42 | 1.246 | 1.698 | -0.452 | -0.680 | -0.240 | 0.000 | True |
| A_ecapa512_aamsoftmax_si_s42 | A_resnetse34v2_aamsoftmax_si_s42 | 4.295 | 4.688 | -0.393 | -0.910 | 0.122 | 0.149 | False |
| A_ecapa1024_aamsoftmax_ta_s42 | A_mlpmixer_aamsoftmax_ta_s42 | 0.894 | 1.246 | -0.351 | -0.560 | -0.134 | 0.001 | True |
| A_ecapa512_aamsoftmax_ta_s42 | A_resnetse34v2_aamsoftmax_ta_s42 | 1.035 | 1.376 | -0.342 | -0.502 | -0.098 | 0.002 | True |
| A_resnetse34l_aamsoftmax_ta_s42 | A_resnetse34v2_aamsoftmax_ta_s42 | 1.698 | 1.376 | 0.321 | 0.153 | 0.585 | 0.000 | True |
| A_resnetse34l_aamsoftmax_si_s42 | A_resnetse34v2_aamsoftmax_si_s42 | 5.000 | 4.688 | 0.312 | -0.210 | 0.780 | 0.229 | False |
| A_ecapa1024_aamsoftmax_si_s42 | A_resnetse34v2_aamsoftmax_si_s42 | 4.396 | 4.688 | -0.292 | -0.894 | 0.199 | 0.227 | False |
| A_mlpmixer_aamsoftmax_si_s42 | A_resnetse34v2_aamsoftmax_si_s42 | 4.950 | 4.688 | 0.262 | -0.172 | 0.694 | 0.243 | False |
| A_resnetse34l_aamsoftmax_ta_s42 | A_vggvox_aamsoftmax_ta_s42 | 1.698 | 1.949 | -0.251 | -0.484 | 0.040 | 0.096 | False |
| A_ecapa512_aamsoftmax_ta_s42 | A_mlpmixer_aamsoftmax_ta_s42 | 1.035 | 1.246 | -0.211 | -0.393 | -0.029 | 0.033 | True |
| A_ecapa1024_aamsoftmax_ta_s42 | A_ecapa512_aamsoftmax_ta_s42 | 0.894 | 1.035 | -0.141 | -0.301 | 0.030 | 0.123 | False |
| A_mlpmixer_aamsoftmax_ta_s42 | A_resnetse34v2_aamsoftmax_ta_s42 | 1.246 | 1.376 | -0.131 | -0.273 | 0.112 | 0.378 | False |
| A_ecapa1024_aamsoftmax_si_s42 | A_ecapa512_aamsoftmax_si_s42 | 4.396 | 4.295 | 0.101 | -0.342 | 0.431 | 0.787 | False |
| A_mlpmixer_aamsoftmax_si_s42 | A_resnetse34l_aamsoftmax_si_s42 | 4.950 | 5.000 | -0.050 | -0.508 | 0.405 | 0.833 | False |

## 4. Figures

![learning_curves.png](figures/learning_curves.png)
![efficiency_frontier.png](figures/efficiency_frontier.png)
![det_curves.png](figures/det_curves.png)

## 5. How to read these numbers

* **EER alone does not rank systems.** With 91 held-out Sinhala and 131 Tamil
  speakers, a single absolute EER carries roughly ±5 pp of uncertainty. Use
  the paired contrasts in §3; they cancel speaker difficulty and resolve
  much finer differences.
* **d' vs EER.** If a system has the better EER but the worse d', its score
  distribution is non-Gaussian and its advantage is confined to one
  operating point.
* **minCllr vs EER.** minCllr integrates over all operating points. A system
  ahead on EER but behind on minCllr is not the better embedding extractor.
* **calibration_loss = Cllr − minCllr** is what score normalisation and
  per-language thresholds can recover without touching the model.

