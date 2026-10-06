#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ECAPA-TDNN with the *classical* MFCC front end: 40 coefficients + Δ + ΔΔ.

The companion to ``ECAPA_TDNN_MFCC`` (80 coefficients, static only). Two
separate modules rather than one parameterised module because ``n_mfcc`` and
``mfcc_deltas`` are not registered argparse options in ``trainSpeakerNet.py``,
and unknown YAML keys are discarded — the same constraint that produced
``ECAPA_TDNN_C512``.

The pair separates two things that "use MFCCs" normally conflates:

* ``ECAPA_TDNN_MFCC``    80 static coefficients — same channel count as the
                         log-mel baseline, so the contrast against
                         ``ECAPA_TDNN`` is purely the DCT.
* this module            40 coefficients + first and second time derivatives —
                         the configuration i-vector and GMM-UBM systems actually
                         used, testing whether explicit dynamics recover what
                         truncation to 40 throws away.

Both feed an identical backbone, so any difference is attributable to the
representation alone.
"""

from models.ECAPA_TDNN_MFCC import ECAPA_TDNN_MFCC


def MainModel(nOut=192, encoder_type="ASP", n_mels=80, log_input=False,
              sample_rate=16000, **kwargs):
    kwargs.pop("channels", None)
    kwargs.pop("n_mfcc", None)
    kwargs.pop("mfcc_deltas", None)
    return ECAPA_TDNN_MFCC(
        nOut=nOut, encoder_type=encoder_type, channels=1024,
        n_mels=n_mels, n_mfcc=40, mfcc_deltas=True,
        log_input=False, sample_rate=sample_rate, **kwargs,
    )
