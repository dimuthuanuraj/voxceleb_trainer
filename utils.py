#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from models._frontend import PreEmphasis as _CanonicalPreEmphasis


class PreEmphasis(_CanonicalPreEmphasis):
    """Backwards-compatibility alias for :class:`models._frontend.PreEmphasis`.

    This class is preserved (rather than removed) so existing imports —
    ``from utils import PreEmphasis`` in ``models/ResNetSE34V2.py`` and
    anywhere else — keep working. The historical behaviour (output
    shape ``(B, T)`` via ``.squeeze(1)``) is pinned via the default
    ``squeeze=True`` on the canonical class.

    New code should import directly from :mod:`models._frontend`.
    See ``docs/bugfixes/BUGFIX-025-shared-audio-frontend.md``.
    """
    pass