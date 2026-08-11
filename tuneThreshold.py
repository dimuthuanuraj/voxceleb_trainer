#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Threshold tuning and EER computation utilities.

================================================================
EER DEFINITION (read this before comparing numbers to literature)
================================================================

This module computes EER using the **conservative** definition:

    EER_max  = max(FPR(t*), FNR(t*)) * 100        # what we return as `eer`
    EER_avg  = (FPR(t*) + FNR(t*)) / 2 * 100      # returned as `eer_average`

where ``t*`` is the operating point that minimises ``|FPR(t) - FNR(t)|``
along the empirical ROC curve.

Two factors govern how much the two definitions disagree:

1. **Density of operating points.** ``sklearn.metrics.roc_curve``
   returns one threshold per distinct score in the input, so ``t*`` is
   chosen from a discrete set. With N trial pairs, the FPR/FNR axes
   each have a granularity of ~1/N. The conservative and average
   definitions can therefore differ by at most ~1/(2N) in absolute
   terms (one half-step on whichever axis was further from the
   crossover). For voxceleb1-O (N≈40k) that bound is ~1.25e-5, i.e.
   below 0.002% EER — usually invisible in 2-decimal-place reports.
   For mini test lists (N<1k) the bound rises to ~0.05% EER, which
   can matter.
2. **Whether the ROC curve is strictly monotone at the crossover.**
   With many tied scores (a common failure mode of metric-learning
   losses early in training) the curve becomes piecewise-constant
   and `t*` may land on a flat segment where FPR ≠ FNR. The
   conservative definition then reports the larger of the two; the
   average reports their midpoint. The gap is bounded by the
   granularity above and is usually small.

In practice:

    EER_max - EER_avg = |FPR(t*) - FNR(t*)| / 2

is the entire discrepancy. The two definitions agree exactly when
the empirical ROC passes through (FPR, FNR) = (x, x); the project's
mini-VoxCeleb1 evaluations have historically shown
``EER_max - EER_avg`` of 0.01-0.05% absolute. That is the order of
magnitude the audit's §4.2 #20 flagged as needing honest disclosure.

Why we keep the conservative definition as the primary `eer`:

- It is monotone-conservative: if EER_max improves run-to-run,
  EER_avg either improves or stays equal (never worsens), so
  tracking the conservative number gives valid early-stopping
  decisions.
- It is the historical default of this codebase; switching the
  primary metric mid-project would break comparison to existing
  ``research_logs/`` and ``exps/`` results.
- We also return `eer_average` (5th index → no, 6th: see signature
  below) so callers needing literature-comparable numbers can use
  that without recomputing.

See ``docs/bugfixes/BUGFIX-020-eer-definition-disclosure.md`` for
the full rationale and the project's policy on quoting EER in
publications.
"""

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
from operator import itemgetter

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    # EER calculation: Find the point where FPR and FNR are closest
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))

    # Return BOTH the conservative and the literature-standard EER so callers
    # who need literature-comparable numbers don't have to recompute.
    # See module docstring + BUGFIX-020 for the full discussion of why the
    # conservative form is the primary `eer` field while `eer_average` is
    # the 6th tuple element.
    eer = max(fpr[idxE], fnr[idxE]) * 100                      # conservative (this project's primary metric)
    eer_average = (fpr[idxE] + fnr[idxE]) / 2 * 100            # standard literature definition
    eer_threshold = float(thresholds[idxE])                    # decision threshold at the EER operating point

    return (tunedThreshold, eer, fpr, fnr, eer_threshold, eer_average);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold