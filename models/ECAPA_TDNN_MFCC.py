#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ECAPA-TDNN with an MFCC front end instead of log-mel.

The frontend axis of the Sinhala/Tamil study (`experiments/`, Stage F) asks what
the network should actually be fed.  This module is the classical end of that
axis: mel-frequency cepstral coefficients.

What changes relative to ``ECAPA_TDNN``
---------------------------------------
Only the front end.  The backbone, pooling, embedding size and every training
hyperparameter are identical, so a Stage F contrast between this and
``ECAPA_TDNN`` isolates the representation and nothing else.

MFCC applies a DCT to the log-mel spectrum:

    c_n = sum_{k=1}^{M} log(E_k) * cos( n (k - 1/2) pi / M ),   n = 0..N-1

Two consequences matter for speaker verification, and they pull in opposite
directions:

* The DCT approximately decorrelates the filterbank channels, which is why MFCCs
  were the right input for diagonal-covariance GMM-UBM and i-vector systems --
  those models could not afford a full covariance.  A CNN or TDNN has no such
  constraint: it *wants* the correlations between adjacent frequency bands,
  because that is where formant structure lives.
* Truncating to N < M coefficients discards the fine spectral detail that
  carries much of the speaker-specific glottal and vocal-tract information,
  keeping mostly the smooth spectral envelope.

So the prediction is that MFCC should *lose* to log-mel with a modern backbone,
and the experiment is worth running precisely because it quantifies how much --
which is the number needed to justify not using MFCCs, rather than asserting it.

``log_input`` is forced off: the log is already inside the MFCC definition, and
taking it again would be a second nonlinearity on an already-signed quantity.
"""

import torch
import torch.nn as nn
import torchaudio

from models.ECAPA_TDNN import ECAPA_TDNN, Conv1dReluBn
from models._frontend import PreEmphasis


class ECAPA_TDNN_MFCC(ECAPA_TDNN):
    """ECAPA-TDNN whose ``torchfb`` produces MFCCs rather than a mel spectrum."""

    def __init__(self, nOut=192, encoder_type="ASP", channels=1024,
                 n_mels=80, n_mfcc=80, log_input=False, sample_rate=16000,
                 mfcc_deltas=False, **kwargs):
        n_out_feats = n_mfcc * (3 if mfcc_deltas else 1)
        # Build the parent at its normal mel width, then re-size layer1 below.
        # Passing n_out_feats straight through would make the parent construct a
        # throwaway mel filterbank at that width, and at 120 bins over a 512-pt
        # FFT torchaudio warns about all-zero filters -- a confusing message
        # about an object this class immediately discards.
        super().__init__(nOut=nOut, encoder_type=encoder_type, channels=channels,
                         n_mels=n_mels, log_input=False,
                         sample_rate=sample_rate, **kwargs)

        # layer1 is the only backbone layer whose width depends on the front end.
        self.layer1 = Conv1dReluBn(n_out_feats, channels, kernel_size=5, padding=2)

        # log_input is pinned False: MFCC already contains the log.
        self.log_input = False
        self.mfcc_deltas = mfcc_deltas

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=512, win_length=400, hop_length=160,
                f_min=0.0, f_max=sample_rate / 2, n_mels=n_mels,
                window_fn=torch.hamming_window,
            ),
        )
        # Same pre-emphasis as the mel path, so the only difference is the DCT.
        self.torchfb = nn.Sequential(PreEmphasis(squeeze=True), mfcc)
        self.instancenorm = nn.InstanceNorm1d(n_out_feats)
        self._deltas = torchaudio.transforms.ComputeDeltas() if mfcc_deltas else None

    def forward(self, x):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                x = self.torchfb(x)
                if self._deltas is not None:
                    # Deltas restore the local temporal dynamics that a purely
                    # static cepstrum drops. Classical systems always used them;
                    # a TDNN sees several frames at once and may not need them,
                    # which is what the mfcc_deltas contrast measures.
                    d1 = self._deltas(x)
                    d2 = self._deltas(d1)
                    x = torch.cat([x, d1, d2], dim=1)
                x = self.instancenorm(x)

        return self._backbone_forward(x)

    def _backbone_forward(self, x):
        import torch.nn.functional as F

        x1 = self.layer1(x)
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


def MainModel(nOut=192, encoder_type="ASP", n_mels=80, log_input=False,
              sample_rate=16000, **kwargs):
    kwargs.pop("channels", None)
    return ECAPA_TDNN_MFCC(
        nOut=nOut, encoder_type=encoder_type, channels=1024,
        n_mels=n_mels, n_mfcc=kwargs.pop("n_mfcc", 80),
        mfcc_deltas=kwargs.pop("mfcc_deltas", False),
        log_input=False, sample_rate=sample_rate, **kwargs,
    )
