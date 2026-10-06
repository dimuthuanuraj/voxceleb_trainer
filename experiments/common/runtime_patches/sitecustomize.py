#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime patches applied to trainer subprocesses, without editing the trainer.

Python imports ``sitecustomize`` automatically at interpreter startup if it is
importable.  The harness puts *this directory* on ``PYTHONPATH`` for the
subprocesses it launches, so these patches apply to exactly those runs and to
nothing else on the machine.  No file in the repository is modified, and the
``trainSpeakerNet.py`` used by previously published results is byte-identical.

PATCH 1 — the ROC curve
-----------------------
``trainSpeakerNet.py`` line 653 does::

    tprs = 1 - fnrs

but ``tuneThreshold.ComputeErrorRates`` builds its outputs with ``.append`` and
returns plain Python **lists**, so this raises::

    unsupported operand type(s) for -: 'int' and 'list'

The exception is caught by the surrounding ``try/except``, so the run is
unaffected -- EER, minDCF, thresholds, checkpoints and early stopping all still
work -- but the per-epoch ``roc_curve_best.png`` is never written.  Every
current run hits it; the only ROC images in ``exps/`` are from
``mini_voxceleb1_experiment_*``, which predate whatever change made these
return lists.

The fix here is deliberately the smallest one that works: wrap ``fnrs`` and
``fprs`` in a **list subclass** that additionally supports ``scalar - self``.
It is still a ``list`` -- ``isinstance(x, list)`` holds, indexing, ``len()``,
slicing and iteration are unchanged, so ``ComputeMinDcf`` and every other
consumer behave exactly as before -- it merely gains the one operation the
plotting code assumed it had.  Changing the return type to a numpy array would
have been the obvious alternative and is riskier: it alters the type seen by
eight call sites across three trainer variants.

Nothing about the numbers changes.  The only observable difference is that the
ROC image now gets written.

PATCH 2 — the validation dataloader deadlock (opt-in)
-----------------------------------------------------
Both ``H_ssl_wavlm_ecapa`` runs hang in **validation**, reproducibly, on the
first epoch after a resume -- H_si at epoch 2 on four separate attempts
(2026-08-16, 08-17 twice, 08-20), H_ta at epoch 4 on two.  Each time the epoch's
training row is written with ``val_eer: null`` and the process then sits
forever: main thread in ``do_poll``, every dataloader worker at 0 % CPU, the
whole node idle, GPU at 0 %.  Not fd exhaustion (126 of 1024) and not shared
memory (54 MB of 63 GB).  It is the dataloader IPC itself.

``SpeakerNet.evaluateFromList`` builds a *fresh* ``DataLoader`` with
``num_workers=nDataLoaderThread`` on every validation, while the training loader
keeps its ``persistent_workers=True`` set alive -- so validation runs 16 worker
processes per job, 32 on a node hosting two.  Setting the validation loader to
in-process loading removes that machinery entirely.

Scope is deliberately narrow: only loaders whose dataset is a
``test_dataset_loader`` are touched, so the training loader keeps its workers and
training throughput is unchanged.  Validation reads a few thousand files
single-threaded, costing on the order of a minute per epoch against epochs that
take 20--40 -- a cost worth paying for a run that otherwise makes no progress at
all.  Nothing about the numbers changes: same files, same order, same crops.

Off unless ``SLSPV_VAL_WORKERS`` is set, so it applies to the runs that need it
and to nothing else.
"""

from __future__ import annotations

import os
import sys


def _patch_compute_error_rates() -> None:
    repo = os.environ.get("SLSPV_REPO_ROOT")
    if repo and repo not in sys.path:
        # sitecustomize runs before the main script's directory is added to
        # sys.path, so the repo root has to be supplied explicitly.
        sys.path.insert(0, repo)

    import numpy
    import tuneThreshold

    original = tuneThreshold.ComputeErrorRates
    if getattr(original, "_slspv_patched", False):
        return

    class _RSubList(list):
        """A list that also answers ``scalar - self`` with an elementwise array."""

        __slots__ = ()

        def __rsub__(self, other):
            return other - numpy.asarray(self, dtype=float)

        def __sub__(self, other):
            return numpy.asarray(self, dtype=float) - other

    def ComputeErrorRates(scores, labels):
        fnrs, fprs, thresholds = original(scores, labels)
        return _RSubList(fnrs), _RSubList(fprs), thresholds

    ComputeErrorRates._slspv_patched = True
    ComputeErrorRates.__doc__ = original.__doc__
    tuneThreshold.ComputeErrorRates = ComputeErrorRates


def _patch_validation_dataloader() -> None:
    """Force in-process loading for the validation/eval DataLoader only."""
    raw = os.environ.get("SLSPV_VAL_WORKERS")
    if raw is None:
        return
    try:
        want = int(raw)
    except ValueError:
        return

    repo = os.environ.get("SLSPV_REPO_ROOT")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)

    import torch.utils.data as tud
    from DatasetLoader import test_dataset_loader

    original = tud.DataLoader.__init__
    if getattr(original, "_slspv_patched", False):
        return

    def __init__(self, dataset, *args, **kwargs):
        if isinstance(dataset, test_dataset_loader):
            kwargs["num_workers"] = want
            # persistent_workers and prefetch_factor are rejected by PyTorch
            # when num_workers == 0, so drop them rather than let the
            # constructor raise inside a patch.
            if want == 0:
                kwargs.pop("persistent_workers", None)
                kwargs.pop("prefetch_factor", None)
            print(f"[sitecustomize] validation DataLoader forced to "
                  f"num_workers={want}", flush=True)
        return original(self, dataset, *args, **kwargs)

    __init__._slspv_patched = True
    tud.DataLoader.__init__ = __init__


try:
    _patch_compute_error_rates()
except Exception:
    pass

try:
    _patch_validation_dataloader()
except Exception:
    # A patch must never be able to take down a training run. If it cannot be
    # applied the trainer simply behaves as it did before: the ROC plot fails,
    # is caught, and everything else proceeds.
    pass
