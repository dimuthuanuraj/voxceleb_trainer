#! /usr/bin/python
# -*- encoding: utf-8 -*-
"""
FEATURE-006 - ECAPA-TDNN (Desplanques et al., Interspeech 2020).

Strongest practical SV architecture today (~0.8-1.5% EER on VoxCeleb1-O).
Implementation follows the paper's "ECAPA-TDNN-Large" variant: C=1024
channels, 3 SE-Res2Blocks with dilations 2/3/4 and Res2 scale 8,
multi-layer feature aggregation, channel-attentive statistical pooling.

Uses the shared mel front-end from models/_frontend.py (BUGFIX-025).
Compatible with all trainSpeakerNet flags: AAM-Softmax loss, FEATURE-002
AS-Norm, FEATURE-003 per-language eval, FEATURE-004 fine-tune freezing,
FEATURE-005 LLRD (via the new `ecapa` layer-pattern alias).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._frontend import make_mel_frontend  # BUGFIX-025


# ----- Building blocks --------------------------------------------------------

class Conv1dReluBn(nn.Module):
    """Conv1d -> ReLU -> BN (pre-activation ordering used throughout ECAPA)."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    """Multi-scale 1D convolution with hierarchical accumulation (Res2Net 1D)."""

    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True, scale=8):
        super().__init__()
        assert channels % scale == 0, (
            f"Res2 scale={scale} must divide channels={channels}."
        )
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.nums):
            self.convs.append(nn.Conv1d(
                self.width, self.width, kernel_size, stride,
                padding, dilation, bias=bias,
            ))
            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class SE_Connect(nn.Module):
    """Squeeze-Excite over channel dimension, applied multiplicatively."""

    def __init__(self, channels, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, channels)

    def forward(self, x):
        # x: [B, C, T]
        s = x.mean(dim=2)                            # [B, C]
        s = F.relu(self.linear1(s))
        s = torch.sigmoid(self.linear2(s))
        return x * s.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """ECAPA SE-Res2 block: Conv1d -> Res2Conv1d -> Conv1d -> SE, with skip."""

    def __init__(self, channels, kernel_size, dilation, scale=8):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.bottleneck1 = Conv1dReluBn(channels, channels, 1)
        self.res2 = Res2Conv1dReluBn(
            channels, kernel_size=kernel_size, padding=padding,
            dilation=dilation, scale=scale,
        )
        self.bottleneck2 = Conv1dReluBn(channels, channels, 1)
        self.se = SE_Connect(channels)

    def forward(self, x):
        out = self.bottleneck1(x)
        out = self.res2(out)
        out = self.bottleneck2(out)
        out = self.se(out)
        return out + x


class AttentiveStatsPool(nn.Module):
    """Channel-dependent attentive statistics pooling (ECAPA default)."""

    def __init__(self, in_dim, attention_channels=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, attention_channels, 1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, 1)

    def forward(self, x):
        # x: [B, C, T]
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)   # [B, C, T]
        mu = torch.sum(alpha * x, dim=2)
        sig = torch.sqrt((torch.sum(alpha * (x ** 2), dim=2) - mu ** 2).clamp(min=1e-9))
        return torch.cat([mu, sig], dim=1)                  # [B, 2C]


# ----- Main model -------------------------------------------------------------

class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN-Large (Desplanques et al. 2020).

    Args:
        nOut: embedding dim (default 192 per paper).
        channels: TDNN width C (default 1024 = "Large"; 512 = "Small").
        n_mels: number of mel filterbanks (default 80).
        log_input: log-mel input.
        sample_rate: input sample rate; passed to the shared front-end.
        encoder_type: accepted for compat with the trainer args, but ECAPA
            always uses attentive-stat pooling; the kwarg is ignored.
    """

    def __init__(self, nOut=192, encoder_type="ASP", channels=1024,
                 n_mels=80, log_input=True, sample_rate=16000, **kwargs):
        super().__init__()
        self.log_input = log_input

        # BUGFIX-025: shared mel front-end. ECAPA uses pre-emphasis 0.97.
        self.torchfb, self.instancenorm = make_mel_frontend(
            sample_rate=sample_rate, n_mels=n_mels, pre_emphasis=True,
        )

        # Initial frame-level processing
        self.layer1 = Conv1dReluBn(n_mels, channels, kernel_size=5, padding=2)
        # SE-Res2 blocks with hierarchical dilations
        self.layer2 = SE_Res2Block(channels, kernel_size=3, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, dilation=4, scale=8)

        # Multi-layer feature aggregation
        self.mfa = nn.Conv1d(3 * channels, 1536, kernel_size=1)
        # Channel-attentive statistical pooling
        self.attention = AttentiveStatsPool(1536, attention_channels=128)
        # Final projection to embedding
        self.bn = nn.BatchNorm1d(3072)
        self.fc = nn.Linear(3072, nOut)
        self.bn_out = nn.BatchNorm1d(nOut)

    def forward(self, x):
        # Mel front-end. amp.autocast disabled for the FFT (matches the
        # convention in ResNetSE34V2 — fp32 mel keeps numerics stable).
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x = self.torchfb(x) + 1e-6
                if self.log_input:
                    x = x.log()
                x = self.instancenorm(x)                # [B, n_mels, T]

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


def MainModel(nOut=192, encoder_type="ASP", channels=1024, n_mels=80,
              log_input=True, sample_rate=16000, **kwargs):
    return ECAPA_TDNN(
        nOut=nOut, encoder_type=encoder_type, channels=channels,
        n_mels=n_mels, log_input=log_input, sample_rate=sample_rate,
        **kwargs,
    )
