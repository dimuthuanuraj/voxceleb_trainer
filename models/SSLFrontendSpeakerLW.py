#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSL frontend with LEARNED LAYER WEIGHTS — the design the VoiceID product uses.

Why this module exists
----------------------
``models/SSLFrontendSpeaker.py`` takes features from exactly one transformer
layer (``--ssl_layer``, default -1 = last).  The deployed system in
``SL_SPV/voiceid`` does something different and better: it learns a softmax
weighting over *all* hidden states and sums them
(``voiceid/backend/app/speaker_model.py``, ``SSLSpeakerNet``):

    w = softmax(theta) in R^{L+1},      h = sum_{l=0}^{L} w_l * H_l

with ``theta`` trained jointly with the head.  So the research stack and the
production stack were not running the same model, and no experiment in this
repository measured the difference.  This module closes that gap: it is the
production design, expressed as a trainer model so it can be compared against
single-layer selection on identical splits.

Why layer choice matters so much for speaker verification
---------------------------------------------------------
Self-supervised speech encoders are trained on a masked-prediction objective,
which drives the *upper* layers toward phonetic and lexical content -- the
information needed to predict a masked frame from context.  Speaker identity is
largely nuisance for that objective, so it survives most strongly in the lower
and middle layers, near the convolutional feature extractor.  Taking the last
layer, the default, is therefore close to the worst choice for a speaker task:
it is the layer the pretraining objective worked hardest to make
speaker-invariant.

A learned weighting sidesteps the question entirely -- it lets the data choose
the depth -- and the fitted weights are themselves a result worth reporting: the
argmax layer is a direct, quantitative statement about where speaker information
lives in that encoder for Sinhala and Tamil, and can be compared against the
same encoder's profile on English.

``layer_weights()`` exposes the fitted distribution for exactly that analysis.
"""

import torch
import torch.nn as nn

from models.SSLFrontendSpeaker import SSLFrontendSpeaker


class SSLFrontendSpeakerLW(SSLFrontendSpeaker):
    """SSL frontend that learns a softmax weighting over all hidden states."""

    def __init__(self, nOut: int = 512, encoder_type: str = "ASP",
                 ssl_encoder_name: str = "microsoft/wavlm-base",
                 ssl_freeze: bool = True, sample_rate: int = 16000,
                 layer_weight_init: str = "uniform", **kwargs):
        # ssl_layer is meaningless here -- every layer is used. Pin it to -1 so
        # the parent's single-layer branch is never taken.
        kwargs.pop("ssl_layer", None)
        super().__init__(nOut=nOut, encoder_type=encoder_type,
                         ssl_encoder_name=ssl_encoder_name,
                         ssl_freeze=ssl_freeze, ssl_layer=-1,
                         sample_rate=sample_rate, **kwargs)

        # +1 because hidden_states[0] is the convolutional feature-extractor
        # output, before any transformer block. That entry is often the most
        # speaker-informative of all, so excluding it would bias the result.
        n_layers = int(self.encoder.config.num_hidden_layers) + 1
        self.n_layers = n_layers

        if layer_weight_init == "uniform":
            init = torch.zeros(n_layers)          # softmax -> uniform
        elif layer_weight_init == "lower":
            # Mild prior toward the lower half, where speaker information is
            # expected to concentrate. Only a starting point; it is learned.
            init = torch.linspace(1.0, -1.0, n_layers)
        else:
            raise ValueError(f"unknown layer_weight_init {layer_weight_init!r}")
        self.layer_weights = nn.Parameter(init)

    def _extract_ssl_features(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted sum over all hidden states -> (B, ssl_dim, T')."""
        if self.ssl_freeze:
            with torch.no_grad():
                outputs = self.encoder(x)
            hidden = tuple(h.detach() for h in outputs.hidden_states)
        else:
            outputs = self.encoder(x)
            hidden = outputs.hidden_states

        # (L, B, T, D). The weights stay trainable even when the encoder is
        # frozen -- that is the whole point: no encoder gradient is needed to
        # learn *which depth* to read from.
        stack = torch.stack(hidden, dim=0)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1).to(stack.dtype)
        features = (w * stack).sum(dim=0)
        return features.transpose(1, 2)

    @torch.no_grad()
    def layer_weights_(self):
        """Fitted layer distribution, for the analysis. Index 0 = CNN output."""
        return torch.softmax(self.layer_weights.detach().float(), dim=0).cpu().numpy()


def MainModel(nOut: int = 512, **kwargs) -> SSLFrontendSpeakerLW:
    return SSLFrontendSpeakerLW(nOut=nOut, **kwargs)
