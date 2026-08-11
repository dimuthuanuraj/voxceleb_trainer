#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
SSLFrontendSpeaker — speaker-verification model with a multilingual
self-supervised-learning (SSL) front-end.

Implements the SSL half of FEATURE-001 (language-aware front-end). The
mel-spectrogram block of the standard SPV models is replaced with a
pretrained SSL encoder loaded from the HuggingFace Hub. The default
encoder list (each known to cover Sinhala and Tamil):

  - ``microsoft/wavlm-base``                — WavLM-Base, ~94 languages
  - ``facebook/wav2vec2-xls-r-300m``        — XLS-R-300M, ~128 languages
  - ``utter-project/mHuBERT-147``           — mHuBERT-147, 147 languages

Any HuggingFace ``AutoModel``-compatible audio-encoder name works as
``--ssl_encoder_name``; the three above are the ones the FEATURE-001
doc references as tested defaults.

Frozen vs. fine-tunable encoder is controlled by ``--ssl_freeze`` (a
bool). Default is ``True`` (frozen) — this is the recommended setting
when the downstream training data is small (the typical Sri Lankan
pilot scenario). Set to ``False`` once the corpus is large enough
that the SSL features can be re-tuned without overfitting.

Output of the SSL encoder is a sequence of per-frame embeddings of
dimension ``encoder.config.hidden_size`` (768 for wavlm-base; 1024 for
xls-r-300m; 768 for mhubert-147). The model pools these with the same
SAP / ASP attention-pooling head used by ResNetSE / VGGVox / MLPMixer
in this repo, then projects to ``nOut``.

The companion SincConv variant of FEATURE-001 is
``models.MLPMixerSpeaker_RawWaveform`` and is unchanged by this
addition — see ``configs/language_aware_sincconv.yaml`` for that path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# HuggingFace transformers is an optional dependency (see requirements.txt
# and FEATURE-001 §2.4). Without it, this model cannot be constructed —
# users who don't need the SSL path are unaffected.
try:
    from transformers import AutoModel
    _HF_AVAILABLE = True
except ImportError:
    AutoModel = None  # type: ignore[assignment]
    _HF_AVAILABLE = False


class SSLFrontendSpeaker(nn.Module):
    """Speaker-verification model with a (typically frozen) multilingual
    SSL encoder as the audio front-end + an attention-pooling head."""

    def __init__(
        self,
        nOut: int = 512,
        encoder_type: str = "ASP",
        ssl_encoder_name: str = "microsoft/wavlm-base",
        ssl_freeze: bool = True,
        ssl_layer: int = -1,
        log_input: bool = False,
        sample_rate: int = 16000,
        **kwargs,
    ) -> None:
        super().__init__()

        if not _HF_AVAILABLE:
            raise ImportError(
                "SSLFrontendSpeaker requires the `transformers` package. "
                "Install it with: pip install 'transformers>=4.30,<5'. "
                "See docs/bugfixes/FEATURE-001-language-aware-frontend.md."
            )

        if sample_rate != 16000:
            # WavLM / XLS-R / mHuBERT are pretrained at 16 kHz. Other rates
            # require resampling before the encoder; trainer-side resampling
            # via BUGFIX-005 handles this if --sample_rate=16000 is left
            # at default, but a non-16k rate here is almost certainly a
            # configuration mistake worth surfacing.
            print(
                f"[SSLFrontendSpeaker] WARNING: sample_rate={sample_rate} but "
                f"the SSL encoder was pretrained at 16 kHz. The DataLoader's "
                f"--sample_rate handling (BUGFIX-005) should be set to 16000."
            )

        print(
            f"Embedding size is {nOut}, encoder_type {encoder_type}, "
            f"ssl_encoder_name={ssl_encoder_name}, ssl_freeze={ssl_freeze}."
        )

        self.nOut = nOut
        self.encoder_type = encoder_type
        self.ssl_encoder_name = ssl_encoder_name
        self.ssl_freeze = ssl_freeze
        self.ssl_layer = ssl_layer
        self.sample_rate = sample_rate

        # Load the SSL encoder from the HuggingFace Hub. This downloads to
        # the user's HF cache on first run; subsequent runs hit the cache.
        # output_hidden_states=True so we can pick a specific layer rather
        # than the last one if requested.
        self.encoder = AutoModel.from_pretrained(
            ssl_encoder_name,
            output_hidden_states=True,
        )

        if ssl_freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        ssl_dim = int(self.encoder.config.hidden_size)

        # Attention-pooling head (SAP / ASP), matching the convention used
        # elsewhere in the repo (cf. ResNetSE34V2.attention).
        self.attention = nn.Sequential(
            nn.Conv1d(ssl_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, ssl_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if encoder_type == "SAP":
            out_dim = ssl_dim
        elif encoder_type == "ASP":
            out_dim = ssl_dim * 2
        else:
            raise ValueError(
                f"Undefined encoder_type {encoder_type!r}. Use 'SAP' or 'ASP'."
            )

        self.bn = nn.BatchNorm1d(out_dim)
        self.fc = nn.Linear(out_dim, nOut)

    def _normalise_waveform(self, x: torch.Tensor) -> torch.Tensor:
        # SSL encoders (WavLM, XLS-R, mHuBERT) expect zero-mean / unit-std
        # waveforms per-utterance — matches the HuggingFace
        # Wav2Vec2FeatureExtractor's `do_normalize=True` default.
        return (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True) + 1e-7
        )

    def _extract_ssl_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SSL encoder and return (B, ssl_dim, T') features.

        Respects ``ssl_freeze``: when frozen, runs under ``torch.no_grad``
        and detaches the output so the encoder's activations are not held
        for the backward pass."""
        if self.ssl_freeze:
            with torch.no_grad():
                outputs = self.encoder(x)
            if self.ssl_layer != -1 and outputs.hidden_states is not None:
                features = outputs.hidden_states[self.ssl_layer]
            else:
                features = outputs.last_hidden_state
            features = features.detach()
        else:
            outputs = self.encoder(x)
            if self.ssl_layer != -1 and outputs.hidden_states is not None:
                features = outputs.hidden_states[self.ssl_layer]
            else:
                features = outputs.last_hidden_state

        # HF returns (B, T', D); we want (B, D, T') for Conv1d head.
        return features.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T) or (B, 1, T); SpeakerNet-side wrapping sometimes
        # adds a channel dim before forwarding.
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        x = self._normalise_waveform(x)
        features = self._extract_ssl_features(x)
        # features now (B, ssl_dim, T')

        w = self.attention(features)
        mu = torch.sum(features * w, dim=2)

        if self.encoder_type == "ASP":
            sigma = torch.sqrt(
                (torch.sum((features ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4)
            )
            pooled = torch.cat((mu, sigma), dim=1)
        else:
            pooled = mu

        pooled = self.bn(pooled)
        return self.fc(pooled)


def MainModel(nOut: int = 512, **kwargs) -> SSLFrontendSpeaker:
    """Factory matching the convention used by every other model in this
    package (see ResNetSE34L.MainModel, MLPMixerSpeaker.MainModel, ...).
    The trainer's dynamic import (`importlib.import_module('models.' +
    args.model)`) calls this entry point."""
    return SSLFrontendSpeaker(nOut=nOut, **kwargs)
