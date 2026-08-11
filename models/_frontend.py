#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Shared audio frontend building blocks for the SPV models.

This module is the **single source of truth** for two things previously
duplicated across model files:

  1. :class:`PreEmphasis` — the pre-emphasis filter used as the first
     stage of the spectrogram pipeline. Before BUGFIX-025 this class
     existed in *two* places (``utils.PreEmphasis`` and
     ``models.RawNetBasicBlock.PreEmphasis``) with subtly different
     output shapes (one squeezed the channel dim, the other did not).
     The canonical version here parameterises that difference via the
     ``squeeze`` constructor argument. Both legacy import paths still
     work — they are now thin subclasses of this class that pin the
     historical default.

  2. :func:`make_mel_frontend` — the standard ``MelSpectrogram`` +
     ``InstanceNorm1d`` chain used by the four hamming-windowed
     spectrogram models in this codebase: ``ResNetSE34L``,
     ``ResNetSE34V2``, ``MLPMixerSpeaker``, ``LSTMAutoencoder``. The
     factory honours BUGFIX-006's "16 kHz-derived n_fft / win_length /
     hop_length are kept constant" policy.

What is **deliberately not** consolidated here:

  - ``VGGVox.py`` keeps its own mel construction inline because it
    sets ``f_min=0.0, f_max=sample_rate/2, pad=0`` and omits
    ``window_fn=hamming``. Those are model-specific parameters tied
    to the VGGVox architecture's expectations (see BUGFIX-014 for
    the architectural constraints) and are not duplication.
  - ``RawNet3.py`` uses ``SincConv_fast`` as its waveform-domain
    frontend — different frontend family entirely. Its own
    ``nn.Sequential(PreEmphasis(squeeze=False), InstanceNorm1d(1, ...))``
    pipeline is preserved because the InstanceNorm operates on
    raw-waveform shape, not on mel-spectrogram shape.

The factory returns a *tuple* ``(torchfb, instancenorm)`` rather than
a wrapped ``nn.Sequential``. Each model assigns the two pieces to
``self.torchfb`` and ``self.instancenorm`` as before — this keeps the
saved-checkpoint ``state_dict`` keys (``torchfb.spectrogram.window``
etc.) byte-identical so every pre-BUGFIX-025 checkpoint loads
unchanged.

See ``docs/bugfixes/BUGFIX-025-shared-audio-frontend.md`` for the
full extraction rationale and the rejected alternatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class PreEmphasis(nn.Module):
    """Canonical pre-emphasis high-pass filter.

    Applies the standard ``y[n] = x[n] - coef * x[n-1]`` filter via a
    1D convolution with a fixed flipped kernel. The ``coef`` default
    of ``0.97`` is the speaker-verification literature standard.

    Two output shapes are supported, controlled by ``squeeze``:

    - ``squeeze=True`` (default): output shape is ``(B, T)``. Used
      when the downstream stage is a ``torchaudio.MelSpectrogram``,
      which expects 2D input.
    - ``squeeze=False``: output shape is ``(B, 1, T)``. Used when the
      downstream stage is a 1-channel ``InstanceNorm1d`` (as in the
      RawNet3 raw-waveform pipeline).

    The default matches ``utils.PreEmphasis``'s historical behaviour
    (squeeze on); the raw-waveform variant inherits with ``squeeze=False``
    pinned (see ``models.RawNetBasicBlock.PreEmphasis``).
    """

    def __init__(self, coef: float = 0.97, squeeze: bool = True) -> None:
        super().__init__()
        self.coef = coef
        self.squeeze = squeeze
        # Cross-correlation flip of the [1, -coef] filter kernel so
        # F.conv1d implements convolution rather than correlation.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            len(x.size()) == 2
        ), "PreEmphasis expects a 2D input tensor of shape (batch, samples)."
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), "reflect")
        out = F.conv1d(x, self.flipped_filter)
        return out.squeeze(1) if self.squeeze else out


def make_mel_frontend(
    sample_rate: int = 16000,
    n_mels: int = 40,
    pre_emphasis: bool = False,
):
    """Build the standard mel-spectrogram + instance-norm frontend.

    Parameters
    ----------
    sample_rate :
        Audio sample rate in Hz. Used only to set the analytical
        ``f_max`` and the mel-filter-bank mapping in
        ``MelSpectrogram``. The ``n_fft``, ``win_length`` and
        ``hop_length`` are **not** scaled with this — they are kept at
        the 16 kHz-derived 512 / 400 / 160 values by design (BUGFIX-006).
    n_mels :
        Number of mel filter banks. Drives both ``MelSpectrogram(n_mels=...)``
        and ``InstanceNorm1d(n_mels)``.
    pre_emphasis :
        When ``True``, the returned ``torchfb`` is a
        ``nn.Sequential(PreEmphasis(squeeze=True), MelSpectrogram(...))``
        — the layout used by ``ResNetSE34V2``. When ``False``, it is
        just the ``MelSpectrogram`` directly — the layout used by
        ``ResNetSE34L``, ``MLPMixerSpeaker``, and ``LSTMAutoencoder``.

    Returns
    -------
    (torchfb, instancenorm) : tuple
        Two ``nn.Module``\\ s. Callers assign each to its own
        attribute (``self.torchfb = torchfb``,
        ``self.instancenorm = instancenorm``) so the resulting
        ``state_dict`` layout is identical to the pre-BUGFIX-025
        per-model construction.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window_fn=torch.hamming_window,
        n_mels=n_mels,
    )
    if pre_emphasis:
        torchfb = nn.Sequential(PreEmphasis(squeeze=True), mel)
    else:
        torchfb = mel
    instancenorm = nn.InstanceNorm1d(n_mels)
    return torchfb, instancenorm
