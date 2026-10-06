#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSL encoder feeding the full ECAPA-TDNN backbone.

Closes the last hole in the study's design.  Stage A varies the *backbone* with
the front end fixed to log-mel; Stage F varies the *front end* with the head
fixed to simple attentive pooling.  Neither measures the **interaction**, and the
interaction is where the interesting question lives:

    once the features come from a pretrained SSL encoder, does the backbone
    still matter?

This module supplies the missing cell. Combined with the runs that already
exist it gives:

    log-mel  + ECAPA backbone        Stage A  `ecapa1024`
    WavLM    + attentive-pool head   Stage F  `ssl_wavlm_lw`
    WavLM    + ECAPA backbone        Stage H  this module

If the backbone upgrade buys much less under SSL features than under log-mel,
the conclusion is that **representation dominates architecture** in this
low-resource regime — which is a directly actionable finding: spend the effort
on the front end, not on backbone search. If it buys just as much, the two
factors are separable and both are worth tuning. Either way the answer is a
number rather than an opinion.

Design
------
    waveform
      -> SSL encoder                 (B, T', D)   D = 768 for base-size models
      -> learned layer weighting     w = softmax(theta), h = sum_l w_l H_l
      -> transpose                   (B, D, T')
      -> ECAPA-TDNN backbone         SE-Res2 blocks, MFA, attentive stats pooling
      -> embedding                   (B, nOut)

The backbone is the *same* ``ECAPA_TDNN`` class Stage A uses; only ``layer1`` is
re-sized, from ``n_mels`` input channels to the encoder's hidden size. Every
other block, the multi-layer aggregation and the pooling are untouched, so a
contrast against Stage A's ``ecapa1024`` isolates the front end.

This is the WavLM+ECAPA arrangement that current speaker-verification systems
use, so the study is measuring the standard strong recipe rather than an
invention of its own.

One caveat worth stating, because it is easy to miss
-----------------------------------------------------
ECAPA's dilations (2, 3, 4) were designed for 100 Hz mel frames — a 10 ms hop.
SSL encoders emit **50 Hz** frames (20 ms stride), so at identical dilation
settings the receptive field covers *twice* as much time. That is not
necessarily bad for speaker modelling, which benefits from long context, but it
does mean the backbone is not operating at the temporal resolution it was tuned
for, and the comparison against ``ecapa1024`` carries that difference along with
the front-end change. It is recorded here rather than buried.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ECAPA_TDNN import ECAPA_TDNN, Conv1dReluBn

try:
    from transformers import AutoModel
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover - mirrors SSLFrontendSpeaker's handling
    AutoModel = None
    _HF_AVAILABLE = False


class SSL_ECAPA(ECAPA_TDNN):
    """SSL front end (optionally layer-weighted) + the ECAPA-TDNN backbone."""

    def __init__(self, nOut=256, channels=1024,
                 ssl_encoder_name="microsoft/wavlm-base-plus",
                 ssl_freeze=True, ssl_layer=-1, layer_weighted=True,
                 sample_rate=16000, **kwargs):
        if not _HF_AVAILABLE:
            raise ImportError(
                "SSL_ECAPA requires `transformers`. See requirements.txt "
                "(FEATURE-001 §2.4)."
            )
        # Build the ECAPA backbone at its normal mel width; layer1 is re-sized
        # below once the encoder's hidden size is known. log_input is off
        # because no mel spectrum is ever computed here.
        kwargs.pop("n_mels", None)
        kwargs.pop("log_input", None)
        kwargs.pop("encoder_type", None)
        super().__init__(nOut=nOut, encoder_type="ASP", channels=channels,
                         n_mels=80, log_input=False, sample_rate=sample_rate,
                         **kwargs)

        self.ssl_encoder_name = ssl_encoder_name
        self.ssl_freeze = ssl_freeze
        self.ssl_layer = ssl_layer
        self.layer_weighted = layer_weighted

        self.encoder = AutoModel.from_pretrained(
            ssl_encoder_name, output_hidden_states=True,
        )
        if ssl_freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        ssl_dim = int(self.encoder.config.hidden_size)
        self.ssl_dim = ssl_dim
        n_states = int(self.encoder.config.num_hidden_layers) + 1
        self.n_states = n_states

        if layer_weighted:
            # Uniform init; +1 covers hidden_states[0], the convolutional
            # feature-extractor output, which is often the most
            # speaker-informative state of all.
            self.layer_weights = nn.Parameter(torch.zeros(n_states))
        else:
            self.layer_weights = None

        # The one structural change to the backbone.
        self.layer1 = Conv1dReluBn(ssl_dim, channels, kernel_size=5, padding=2)

        # The inherited mel front end is unreachable in this class's forward.
        # Deleting it keeps it out of the parameter count and the state dict, so
        # a model card for this module reports what actually runs.
        self.torchfb = None
        self.instancenorm = None

        print(f"Embedding size is {nOut}, SSL_ECAPA over {ssl_encoder_name} "
              f"(dim {ssl_dim}, {n_states} states, "
              f"{'layer-weighted' if layer_weighted else f'layer {ssl_layer}'}, "
              f"{'frozen' if ssl_freeze else 'fine-tuned'}).")

    # -- front end -----------------------------------------------------
    def _normalise_waveform(self, x):
        # Matches Wav2Vec2FeatureExtractor's do_normalize=True default, and
        # SSLFrontendSpeaker's handling, so the encoders see what they expect.
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)

    def _ssl_features(self, x):
        """-> (B, ssl_dim, T')"""
        if self.ssl_freeze:
            with torch.no_grad():
                out = self.encoder(x)
            hidden = tuple(h.detach() for h in out.hidden_states)
            last = out.last_hidden_state.detach()
        else:
            out = self.encoder(x)
            hidden = out.hidden_states
            last = out.last_hidden_state

        if self.layer_weighted:
            stack = torch.stack(hidden, dim=0)               # (L+1, B, T, D)
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            feats = (w.to(stack.dtype) * stack).sum(dim=0)   # (B, T, D)
        elif self.ssl_layer != -1:
            feats = hidden[self.ssl_layer]
        else:
            feats = last
        return feats.transpose(1, 2)

    # -- forward -------------------------------------------------------
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        x = self._normalise_waveform(x)
        feats = self._ssl_features(x)

        x1 = self.layer1(feats)
        x2 = self.layer2(x1)
        x3 = self.layer3(x1 + x2)
        x4 = self.layer4(x1 + x2 + x3)

        out = torch.cat([x2, x3, x4], dim=1)
        out = F.relu(self.mfa(out))
        out = self.attention(out)
        out = self.bn(out)
        out = self.fc(out)
        out = self.bn_out(out)
        return out

    @torch.no_grad()
    def layer_weights_(self):
        """Fitted layer distribution, for the analysis. Index 0 = CNN output."""
        if self.layer_weights is None:
            return None
        return torch.softmax(self.layer_weights.detach().float(), dim=0).cpu().numpy()


def MainModel(nOut=256, **kwargs):
    return SSL_ECAPA(nOut=nOut, **kwargs)
