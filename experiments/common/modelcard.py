#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Architecture specifications and live model introspection.

Two halves:

``ARCH_SPECS``   a-priori, hand-written description of each backbone: how it
                 aggregates time, what its inductive bias is, and the
                 pre-registered hypothesis for how it should behave on Sinhala
                 and Tamil.  Written before any run.

``build_card()`` loads the model exactly as ``SpeakerNet`` would, then measures
                 what it actually is: parameter counts per top-level module,
                 embedding dimension, activation shapes, and multiply-accumulate
                 cost for one 2-second input.  Measured numbers beat remembered
                 ones, and the per-module breakdown is what lets the analysis
                 separate "this backbone is better" from "this backbone is
                 simply bigger".
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

# The trainer's model package is imported as ``models.<Name>`` relative to the
# repo root, so the root must be importable no matter where this is called from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ARCH_SPECS = {
    "ECAPA_TDNN": {
        "display": "ECAPA-TDNN",
        "family": "TDNN / 1-D convolutional",
        "temporal_aggregation": "attentive statistics pooling (ASP)",
        "structure": (
            "Res2Net-style dilated 1-D convolutions with Squeeze-Excitation, "
            "multi-layer feature aggregation (the outputs of all three SE-Res2 "
            "blocks are concatenated before pooling), then channel- and "
            "context-dependent attentive statistics pooling that produces both a "
            "weighted mean and a weighted standard deviation."
        ),
        "inductive_bias": (
            "Dilation gives a wide receptive field at low parameter cost, and the "
            "statistics pooling makes the embedding explicitly invariant to "
            "utterance length. The sigma branch of ASP carries the within-"
            "utterance variability, which is informative for speakers whose "
            "phonetic content varies across the read prompts."
        ),
        "params_scale_with": "channels (C=512 or 1024)",
        "predicted_effect": (
            "Expected strongest mel-based backbone on both languages. The "
            "512-channel variant is included to test whether the 1024 width is "
            "actually used at these corpus sizes (129k si / 63k ta utterances) "
            "or is simply overparameterised."
        ),
    },
    "ECAPA_TDNN_C512": {
        "display": "ECAPA-TDNN (512 channels)",
        "family": "TDNN / 1-D convolutional",
        "temporal_aggregation": "attentive statistics pooling (ASP)",
        "structure": (
            "Identical to ECAPA_TDNN in every respect except channel width, "
            "which is pinned to 512 instead of 1024 (models/ECAPA_TDNN_C512.py)."
        ),
        "inductive_bias": (
            "Same as ECAPA-TDNN. The pair exists to separate width from "
            "topology: only one number differs between the two runs."
        ),
        "params_scale_with": "fixed at 512 channels",
        "predicted_effect": (
            "If it matches ECAPA-1024 within the paired-bootstrap interval, "
            "these corpora are data-limited rather than capacity-limited, and "
            "the cheaper model is the correct deployment choice."
        ),
    },
    "ECAPA_TDNN_MFCC": {
        "display": "ECAPA-TDNN + MFCC front end",
        "family": "TDNN / 1-D convolutional",
        "temporal_aggregation": "attentive statistics pooling (ASP)",
        "structure": (
            "Identical to ECAPA_TDNN except the front end, which applies a DCT "
            "to the log-mel spectrum to produce cepstral coefficients."
        ),
        "inductive_bias": (
            "The DCT approximately decorrelates filterbank channels. That was "
            "essential for diagonal-covariance GMM-UBM and i-vector systems, "
            "which could not model correlations between bands. A convolutional "
            "network has no such limitation and actively uses those "
            "correlations, so the decorrelation is at best redundant and at "
            "worst destroys the local spectral structure convolution relies on."
        ),
        "params_scale_with": "n_mfcc",
        "predicted_effect": (
            "Should LOSE to log-mel. The experiment exists to quantify the gap, "
            "which is the number needed to retire MFCCs from this project with "
            "evidence rather than by assertion."
        ),
    },
    "ECAPA_TDNN_MFCC40D": {
        "display": "ECAPA-TDNN + classical MFCC (40 + Δ + ΔΔ)",
        "family": "TDNN / 1-D convolutional",
        "temporal_aggregation": "attentive statistics pooling (ASP)",
        "structure": (
            "As ECAPA_TDNN_MFCC but with the classical configuration: 40 "
            "coefficients plus first and second time derivatives."
        ),
        "inductive_bias": (
            "Explicit delta features hand the network local temporal dynamics "
            "that a static cepstrum discards. A TDNN already sees a wide "
            "temporal context through dilation, so the deltas may be redundant."
        ),
        "params_scale_with": "n_mfcc x 3",
        "predicted_effect": (
            "Better than static MFCC, still below log-mel. If it matches "
            "mfcc80, the deltas are adding nothing a dilated TDNN could not "
            "already compute for itself."
        ),
    },
    "SSLFrontendSpeakerLW": {
        "display": "SSL frontend with learned layer weights",
        "family": "self-supervised transformer frontend",
        "temporal_aggregation": "attentive statistics pooling over a weighted "
                                "sum of all hidden states",
        "structure": (
            "As SSLFrontendSpeaker, but instead of reading one transformer "
            "layer it learns a softmax weighting over all L+1 hidden states "
            "(including the convolutional feature-extractor output) and sums "
            "them. This is the design deployed in SL_SPV/voiceid."
        ),
        "inductive_bias": (
            "Masked-prediction pretraining drives the upper layers toward "
            "phonetic and lexical content, for which speaker identity is "
            "nuisance; speaker information therefore survives most strongly in "
            "the lower and middle layers. Taking the last layer -- the default "
            "-- is close to the worst choice for a speaker task. A learned "
            "weighting lets the data pick the depth instead."
        ),
        "params_scale_with": "encoder choice; the weighting adds only L+1 scalars",
        "predicted_effect": (
            "Should beat last-layer selection clearly, and beat a fixed "
            "mid-layer choice by a smaller margin. The fitted weights are "
            "themselves a reportable result: the argmax layer states where "
            "speaker information lives in that encoder for Sinhala and Tamil, "
            "and can be compared against the same encoder's English profile."
        ),
    },
    "SSL_ECAPA": {
        "display": "SSL encoder (layer-weighted) + ECAPA-TDNN backbone",
        "family": "hybrid: self-supervised frontend + TDNN backbone",
        "temporal_aggregation": "attentive statistics pooling (ECAPA's own)",
        "structure": (
            "A pretrained SSL encoder produces frame-level features, combined "
            "across all hidden states by a learned softmax weighting, and those "
            "features feed the full ECAPA-TDNN backbone (SE-Res2 blocks, "
            "multi-layer aggregation, attentive statistics pooling) in place of "
            "a log-mel spectrogram. Only ECAPA's first convolution is re-sized, "
            "from n_mels to the encoder's hidden dimension; every other block is "
            "identical to the Stage A ecapa1024 run."
        ),
        "inductive_bias": (
            "Combines learned multilingual representations with a backbone that "
            "has a strong temporal-modelling prior. Note that ECAPA's dilations "
            "were tuned for 100 Hz mel frames while SSL encoders emit 50 Hz, so "
            "the receptive field spans twice the time it was designed for -- "
            "helpful for speaker modelling, but a real difference carried along "
            "with the front-end change."
        ),
        "params_scale_with": "encoder choice + ECAPA channels",
        "predicted_effect": (
            "Should be the strongest configuration in absolute EER, and is the "
            "recipe current SV systems use. The scientifically interesting "
            "quantity is not its EER but the INTERACTION: if the ECAPA backbone "
            "buys much less over a plain pooling head under SSL features than "
            "log-mel does, representation dominates architecture in this "
            "low-resource regime, and front-end choice is where the effort "
            "belongs."
        ),
    },
    "ResNetSE34L": {
        "display": "ResNetSE-34 (light, 16-32-64-128)",
        "family": "2-D CNN over the spectrogram",
        "temporal_aggregation": "self-attentive pooling (SAP) / mean",
        "structure": (
            "ResNet-34 topology over the time-frequency plane with "
            "Squeeze-Excitation on each block; filter widths 16/32/64/128."
        ),
        "inductive_bias": (
            "2-D convolution treats the spectrogram as an image, so it imposes "
            "locality in frequency as well as time -- it can learn formant-shaped "
            "local patterns, but its receptive field grows more slowly in time "
            "than a dilated TDNN's."
        ),
        "params_scale_with": "filter widths",
        "predicted_effect": (
            "Solid but below ECAPA. Its frequency-local bias should matter more "
            "for Tamil, whose retroflex consonants carry identity cues in "
            "localised high-frequency structure."
        ),
    },
    "ResNetSE34V2": {
        "display": "ResNetSE-34 V2 (32-64-128-256)",
        "family": "2-D CNN over the spectrogram",
        "temporal_aggregation": "attentive statistics pooling (ASP)",
        "structure": "As ResNetSE34L with doubled filter widths (32/64/128/256).",
        "inductive_bias": (
            "Same bias as the light variant with roughly 4x the capacity; the "
            "pair isolates capacity from topology."
        ),
        "params_scale_with": "filter widths",
        "predicted_effect": (
            "The L-vs-V2 pair is the study's capacity control: if V2 beats L by "
            "much more than ECAPA-1024 beats ECAPA-512, the corpus rewards raw "
            "capacity; if both gaps are small, these corpora are data-limited "
            "rather than capacity-limited."
        ),
    },
    "RawNet3": {
        "display": "RawNet3",
        "family": "raw-waveform",
        "temporal_aggregation": "attentive statistics pooling",
        "structure": (
            "Learnable analytic (sinc) filterbank replacing the mel front end, "
            "followed by Res2Net blocks with SE. The first layer's filters are "
            "parameterised by cut-off frequencies rather than free weights."
        ),
        "inductive_bias": (
            "Learns its own frequency decomposition instead of accepting the mel "
            "scale. The mel scale was fitted to perceptual data on English and "
            "other European languages; whether its warping is right for Sinhala "
            "and Tamil is an open question, and RawNet3 is the experiment that "
            "answers it -- the learned cut-off distribution can be compared "
            "directly against the mel spacing after training."
        ),
        "params_scale_with": "sinc_stride, model_scale",
        "predicted_effect": (
            "The scientifically most interesting backbone here. If it wins, the "
            "learned filterbank is compensating for a mel mismatch, and the "
            "trained cut-offs should visibly deviate from mel spacing -- a "
            "directly checkable, falsifiable claim rather than a black-box win."
        ),
    },
    "SSLFrontendSpeaker": {
        "display": "SSL frontend (WavLM / XLS-R / mHuBERT) + pooling head",
        "family": "self-supervised transformer frontend",
        "temporal_aggregation": "attentive statistics pooling over transformer states",
        "structure": (
            "A pretrained self-supervised transformer produces frame-level "
            "representations; a lightweight head pools them into a speaker "
            "embedding. The encoder is frozen by default (--ssl_freeze)."
        ),
        "inductive_bias": (
            "Imports representations learned from large multilingual unlabelled "
            "audio. The transfer question is whether Sinhala and Tamil are "
            "in-distribution for the pretraining corpus: XLS-R and mHuBERT-147 "
            "both cover them, WavLM does not (English-only pretraining), so the "
            "encoder comparison measures pretraining-language coverage directly."
        ),
        "params_scale_with": "encoder choice (94M base / 300M XLS-R)",
        "predicted_effect": (
            "Strong absolute numbers but at 10-50x the parameter count and "
            "inference cost. The interesting result is the frozen-encoder "
            "comparison: if XLS-R (which saw Tamil) beats WavLM (which did not) "
            "by more on Tamil than on Sinhala, that is direct evidence that "
            "pretraining coverage, not architecture, drives the transfer."
        ),
    },
    "MLPMixerSpeaker": {
        "display": "MLP-Mixer speaker net",
        "family": "attention-free token/channel mixing",
        "temporal_aggregation": "mean/statistics over mixed tokens",
        "structure": (
            "Alternating token-mixing and channel-mixing MLPs over spectrogram "
            "patches; no convolution and no attention."
        ),
        "inductive_bias": (
            "Almost none -- neither locality nor translation equivariance. Mixers "
            "typically need far more data than convolutional models to reach the "
            "same point, so on a 63k-utterance Tamil corpus this is a test of how "
            "much the convolutional prior is worth."
        ),
        "params_scale_with": "depth, hidden dim, patch size",
        "predicted_effect": (
            "Expected to underperform clearly at this data scale. Included "
            "deliberately as the low-inductive-bias end of the axis: its gap to "
            "ECAPA quantifies the value of the convolutional prior in the "
            "low-resource regime, which is the regime both languages are in."
        ),
    },
    "VGGVox": {
        "display": "VGGVox",
        "family": "2-D CNN (classical)",
        "temporal_aggregation": "temporal average pooling",
        "structure": "The original VoxCeleb VGG-M style CNN over spectrograms.",
        "inductive_bias": "Plain stacked convolutions, no residual path, no attention.",
        "params_scale_with": "fixed",
        "predicted_effect": (
            "Historical floor. Its role is to anchor the bottom of the ranking so "
            "improvements are quoted against a known reference rather than in the "
            "abstract."
        ),
    },
}


def get_arch(name: str) -> dict:
    if name not in ARCH_SPECS:
        raise KeyError(f"unknown architecture {name!r}; known: {sorted(ARCH_SPECS)}")
    return dict(ARCH_SPECS[name])


# --------------------------------------------------------------------------
# live introspection
# --------------------------------------------------------------------------
def _human(n: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def build_card(model_name: str, model_kwargs: dict, probe_seconds: float = 2.0,
               sample_rate: int = 16000) -> dict:
    """Instantiate the model and measure what it actually is.

    Returns a dict that is safe to JSON-serialise.  Failures are captured rather
    than raised: a model card is documentation, and a missing card must never
    take down a training run that would otherwise succeed.
    """
    card = {"model": model_name, "kwargs": dict(model_kwargs)}
    try:
        import torch

        module = importlib.import_module("models." + model_name)
        net = module.__getattribute__("MainModel")(**model_kwargs)
        net.eval()

        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        card.update(
            {
                "parameters_total": total,
                "parameters_trainable": trainable,
                "parameters_frozen": total - trainable,
                "parameters_total_human": _human(total),
                "trainable_fraction": round(trainable / total, 6) if total else None,
            }
        )

        # per-top-level-module breakdown: separates backbone cost from frontend
        # cost, which is the difference that matters for the SSL models.
        card["modules"] = {
            name: {
                "parameters": sum(p.numel() for p in child.parameters()),
                "type": type(child).__name__,
            }
            for name, child in net.named_children()
        }

        # parameter count by dimensionality -- a crude but useful proxy for where
        # capacity sits (conv kernels vs linear projections vs norms)
        by_dim = {}
        for _, p in net.named_parameters():
            by_dim[p.dim()] = by_dim.get(p.dim(), 0) + p.numel()
        card["parameters_by_tensor_rank"] = {str(k): v for k, v in sorted(by_dim.items())}

        # forward probe: embedding dim and MAC cost for one utterance
        n_samples = int(probe_seconds * sample_rate)
        with torch.no_grad():
            x = torch.randn(1, n_samples)
            try:
                out = net(x)
            except Exception:
                # some backbones expect an explicit channel axis
                out = net(x.unsqueeze(0))
        card["embedding_dim"] = int(out.shape[-1])
        card["probe_input"] = {"seconds": probe_seconds, "samples": n_samples}

        macs = _count_macs(net, n_samples)
        if macs is not None:
            card["macs_per_2s_utterance"] = macs
            card["macs_human"] = _human(macs)
            card["gflops_per_2s_utterance"] = round(2.0 * macs / 1e9, 4)

    except Exception:
        card["introspection_error"] = traceback.format_exc(limit=4)
    return card


def _count_macs(net, n_samples: int):
    """Multiply-accumulate count via forward hooks on Conv1d/Conv2d/Linear.

    Deliberately a lower bound: it ignores normalisation, activation and
    attention-softmax cost.  That is fine for the purpose it serves here, which
    is comparing backbones on the same measuring stick, not predicting wall-clock
    latency -- for latency the analysis uses the measured per-epoch time instead.
    """
    try:
        import torch
        import torch.nn as nn

        total = [0]
        handles = []

        def conv_hook(mod, inp, out):
            k = 1
            for d in mod.kernel_size:
                k *= d
            out_elems = out.numel()
            total[0] += out_elems * k * (mod.in_channels // mod.groups)

        def lin_hook(mod, inp, out):
            total[0] += out.numel() * mod.in_features

        for m in net.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                handles.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(lin_hook))

        with torch.no_grad():
            x = torch.randn(1, n_samples)
            try:
                net(x)
            except Exception:
                net(x.unsqueeze(0))
        for h in handles:
            h.remove()
        return int(total[0])
    except Exception:
        return None


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ECAPA_TDNN"
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"nOut": 192, "n_mels": 80}
    print(json.dumps(build_card(name, kwargs), indent=2)[:4000])
