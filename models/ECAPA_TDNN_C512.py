#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ECAPA-TDNN at 512 channels — the capacity control for the SL architecture study.

``trainSpeakerNet.py`` builds a model from ``--model <module>`` and forwards
``vars(args)`` as keyword arguments, so a backbone hyperparameter can only reach
the model if it is a registered argparse option.  ``channels`` is not one, and
YAML cannot supply it either (``trainSpeakerNet.py`` line 227 discards unknown
config keys with "Ignored unknown parameter").

Rather than add a flag to the trainer — which would put every previously
published number on a different code path from the new ones — the 512-channel
variant is exposed as its own model module.  This file adds nothing and changes
nothing; it only pins one argument of the existing implementation.

Paired with ``ECAPA_TDNN`` (1024 channels), this isolates width from topology:
the two runs differ in exactly one number, so the EER gap between them measures
whether these corpora can use the extra capacity at all.  See
``experiments/registry.py`` for how the pair is used.
"""

from models.ECAPA_TDNN import ECAPA_TDNN


def MainModel(nOut=192, encoder_type="ASP", n_mels=80, log_input=True,
              sample_rate=16000, **kwargs):
    # channels is fixed here and deliberately not read from kwargs: the whole
    # point of the module is that its width cannot drift from 512.
    kwargs.pop("channels", None)
    return ECAPA_TDNN(
        nOut=nOut,
        encoder_type=encoder_type,
        channels=512,
        n_mels=n_mels,
        log_input=log_input,
        sample_rate=sample_rate,
        **kwargs,
    )
