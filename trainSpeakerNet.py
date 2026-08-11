#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import random
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")

# Try to import new dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    print("Loaded TensorBoard SummaryWriter.")
except ImportError:
    print("TensorBoard not found. Please run 'pip install tensorboard' to enable logging.")
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
    print("Loaded Matplotlib.")
except ImportError:
    print("Matplotlib not found. Please run 'pip install matplotlib' to enable ROC curve plotting.")
    plt = None


## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--sample_rate',    type=int,   default=16000,  help='Target audio sample rate (Hz). Files at other rates are resampled at load time')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', '--n_data_loader_thread', type=int, default=5,     help='Number of loader threads. snake_case alias accepted (BUGFIX-022).')
parser.add_argument('--prefetch_factor',  type=int,  default=2,     help='Batches each worker prefetches (only used when nDataLoaderThread>0).')
parser.add_argument('--persistent_workers', type=lambda x: str(x).lower() in ('1','true','yes'), default=False, help='Keep DataLoader workers alive between epochs to avoid respawn cost.')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')
parser.add_argument('--patience',       type=int,   default=10,     help='Number of test intervals to wait for EER improvement before early stopping (0 to disable)')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    '--n_per_speaker', type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses. snake_case alias accepted (BUGFIX-022).')
parser.add_argument('--nClasses',       '--n_classes',     type=int,   default=5991,   help='Number of speakers in the softmax layer, only for softmax-based losses. snake_case alias accepted (BUGFIX-022).')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           '--n_out',         type=int,   default=512,    help='Embedding size in the last FC layer. snake_case alias accepted (BUGFIX-022).')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--ddp_find_unused_parameters', type=lambda x: str(x).lower() in ('1','true','yes'), default=True, help='DDP find_unused_parameters. Safe to set False when every param is used in forward (e.g. plain ECAPA baseline) to avoid an extra autograd traversal per step.')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--deterministic',  dest='deterministic', action='store_true', help='Reproducibility mode for paper / ablation runs: seeds Python/NumPy/Torch with --seed, sets cudnn.deterministic, disables cudnn.benchmark and TF32, and enables torch.use_deterministic_algorithms (warn_only). ~10-30%% slower; some ops emit warnings when no deterministic kernel exists. See BUGFIX-017.')
parser.add_argument('--augment_chain',  type=str,   default="",     help='Augmentation probability config (BUGFIX-016). Empty/uniform = legacy 0.2 over {clean, reverb, music, speech, noise}. CLI: JSON like \'{"noise":0.3,"music":0.2}\'. YAML: native dict or list-of-single-key-dicts. Missing labels get 0.0; unspecified mass drains into clean.')
parser.add_argument('--eval_streaming', dest='eval_streaming', action='store_true', help='Streaming evaluation (BUGFIX-018). Writes per-file embeddings to <save_path>/eval_feats_tmp/ and lazy-loads with an LRU cache, instead of holding the full feats dict in memory on every rank. Required for SL-benchmark-scale (~1M-pair) test lists; safe to enable for smaller lists with a small disk cost.')
parser.add_argument('--eval_feat_cache_size', type=int, default=4096, help='LRU cache size (in #embeddings) for --eval_streaming. Default 4096 covers VoxCeleb1-O comfortably; increase for very wide hot-set distributions.')
parser.add_argument('--ssl_encoder_name',   type=str,   default="microsoft/wavlm-base", help='SSL encoder HuggingFace model ID for model=SSLFrontendSpeaker (FEATURE-001). Tested defaults: microsoft/wavlm-base, facebook/wav2vec2-xls-r-300m, utter-project/mHuBERT-147. All three cover Sinhala/Tamil.')
parser.add_argument('--ssl_freeze',         dest='ssl_freeze', action='store_true', default=True, help='Freeze the SSL encoder during training (FEATURE-001). Default True; recommended for small downstream corpora. Use --no_ssl_freeze (or set ssl_freeze: false in YAML) to fine-tune the encoder.')
parser.add_argument('--no_ssl_freeze',      dest='ssl_freeze', action='store_false', help='Allow the SSL encoder to be fine-tuned (FEATURE-001).')
parser.add_argument('--ssl_layer',          type=int,   default=-1, help='Which transformer layer of the SSL encoder to use as features. -1 = last layer (default). Mid-layer features sometimes transfer better for speaker tasks; experiment if accuracy plateaus.')

## AS-Norm score normalisation (FEATURE-002)
parser.add_argument('--as_norm',                dest='as_norm', action='store_true', default=False, help='Enable adaptive symmetric score normalisation (FEATURE-002). Default off; behaviour byte-identical when disabled.')
parser.add_argument('--no_as_norm',             dest='as_norm', action='store_false', help='Explicitly disable AS-Norm (overrides YAML and earlier CLI). Used by ablation runs.')
parser.add_argument('--as_norm_cohort_list',    type=str,   default="",   help='Path to a text file listing cohort utterances (one wav path per line; last whitespace field used). Required when --as_norm is set. ~500 speakers is the literature default.')
parser.add_argument('--as_norm_cohort_path',    type=str,   default="",   help='Root path for files in --as_norm_cohort_list. Falls back to --test_path when empty.')
parser.add_argument('--as_norm_top_k',          type=int,   default=300,  help='Top-K cohort scores to anchor per-side AS-Norm statistics. Clamped to cohort_size. 300 is the standard default.')
parser.add_argument('--as_norm_save_cohort',    dest='as_norm_save_cohort', action='store_true',  default=True,  help='Cache cohort embeddings to <save_path>/asnorm_cohort.pt (default on; auto-invalidates if the cohort list changes).')
parser.add_argument('--no_as_norm_save_cohort', dest='as_norm_save_cohort', action='store_false', help='Disable cohort embedding cache.')
parser.add_argument('--as_norm_chunk_size',     type=int,   default=64,   help='Number of trial files cdist-batched per cohort-stats step. Drop if you OOM on a small GPU.')

## Per-language evaluation (FEATURE-003)
parser.add_argument('--per_lang_test_lists',    type=str,   default="",   help='Comma-separated lang:path pairs, e.g. "si:/data/test_list_si.txt,ta:/data/test_list_ta.txt,cs:/data/test_list_cs.txt". When set, each list is evaluated independently and EER/MinDCF/Threshold are reported per language plus a pooled combined number. Overrides --test_list. YAML accepts a native dict.')

## Cross-lingual fine-tune (FEATURE-004)
parser.add_argument('--finetune',                dest='finetune', action='store_true', default=False, help='Enable cross-lingual fine-tune mode (FEATURE-004). Requires --initial_model. Default off; behaviour byte-identical when disabled.')
parser.add_argument('--no_finetune',             dest='finetune', action='store_false', help='Explicitly disable fine-tune mode (overrides YAML and earlier CLI). Used by ablation runs.')
parser.add_argument('--finetune_freeze',         type=str,   default="",   help='Comma-separated list of submodule aliases or dotted paths to freeze. Aliases: frontend, ssl_encoder, backbone, loss_head. Example: "frontend" (for mel/SincConv), "ssl_encoder" (FEATURE-001 SSL models), "backbone,loss_head" (train nothing — useless, but supported). YAML accepts a native list.')
parser.add_argument('--finetune_lr_multiplier',  type=float, default=1.0,  help='Multiplicative scale on --lr when --finetune is set. Default 1.0 (no change). Common values: 0.1, 0.01 for small-corpus SL fine-tune from English checkpoint.')

## Layer-wise learning-rate decay (FEATURE-005)
parser.add_argument('--llrd',                    dest='llrd', action='store_true', default=False, help='Enable layer-wise learning-rate decay (FEATURE-005). Optimizer is built over per-depth param groups with lr = base_lr * decay^depth. Default off; behaviour byte-identical when disabled. Composes with --finetune.')
parser.add_argument('--no_llrd',                 dest='llrd', action='store_false', help='Explicitly disable LLRD (overrides YAML and earlier CLI). Used by ablation runs.')
parser.add_argument('--llrd_decay',              type=float, default=0.9,  help='Per-layer decay multiplier when --llrd is set. Common values 0.8-0.95. Default 0.9 (= last transformer layer at full lr, first at ~0.31x).')
parser.add_argument('--llrd_layer_pattern',      type=str,   default="ssl",help='Alias name or raw regex selecting the layer index in parameter names. Aliases: "ssl" (HuggingFace transformer: encoder.encoder.layers.N), "mlpmixer" (mixer_blocks.N), "ecapa" (layerN). A raw regex with one numeric capture group is also accepted.')

## Language-ID auxiliary head (FEATURE-007)
parser.add_argument('--lang_aux',                dest='lang_aux', action='store_true', default=False, help='Enable language-ID auxiliary head (FEATURE-007). Adds L_speaker + lambda*L_lang during training only; eval is unaffected. Default off; behaviour byte-identical when disabled.')
parser.add_argument('--no_lang_aux',             dest='lang_aux', action='store_false', help='Explicitly disable lang-aux head (overrides YAML and earlier CLI). Used by ablation runs.')
parser.add_argument('--lang_aux_weight',         type=float, default=0.3,  help='Lambda weighting the lang-aux loss against the speaker loss. Default 0.3; common range 0.1-1.0.')
parser.add_argument('--lang_aux_num_classes',    type=int,   default=4,    help='Number of language classes (default 4: si/ta/en/mix).')
parser.add_argument('--lang_aux_label_file',     type=str,   default="",   help='Path to a <spk_label_int> <lang_label_int> lookup file. Required when --lang_aux is set. Speakers absent from the file are excluded from the aux loss (ignore_index=-1).')

## Domain-adversarial training (FEATURE-008) — opposite objective to FEATURE-007 on language.
parser.add_argument('--dann_lang',                    dest='dann_lang', action='store_true', default=False, help='Enable language DANN adversarial head (FEATURE-008). Strips language info from the embedding via GRL. Do NOT combine with --lang_aux on the same signal.')
parser.add_argument('--dann_lang_weight',             type=float, default=1.0,  help='Multiplicative weight on the DANN language loss before adding to total. Default 1.0.')
parser.add_argument('--dann_lang_lambda',             type=float, default=0.1,  help='Gradient-reversal scale (encoder-side). Default 0.1; common range 0.05-0.5. Halve if loss oscillates.')
parser.add_argument('--dann_lang_num_classes',        type=int,   default=4,    help='Number of language classes for DANN head (default 4: si/ta/en/mix).')
parser.add_argument('--dann_lang_label_file',         type=str,   default="",   help='Path to a <spk_label_int> <lang_label_int> lookup file. Same format as --lang_aux_label_file. Required when --dann_lang is set.')

parser.add_argument('--dann_channel',                 dest='dann_channel', action='store_true', default=False, help='Enable channel DANN adversarial head (FEATURE-008). Strips channel info (mic/phone/codec) from the embedding.')
parser.add_argument('--dann_channel_weight',          type=float, default=1.0,  help='Multiplicative weight on the DANN channel loss. Default 1.0.')
parser.add_argument('--dann_channel_lambda',          type=float, default=0.1,  help='Gradient-reversal scale for the channel head. Default 0.1.')
parser.add_argument('--dann_channel_num_classes',     type=int,   default=3,    help='Number of channel classes (default 3: mic/phone/codec).')
parser.add_argument('--dann_channel_label_file',      type=str,   default="",   help='Path to a <spk_label_int> <channel_label_int> lookup file. Required when --dann_channel is set.')

## PLDA scoring backend (FEATURE-010)
parser.add_argument('--plda',                     dest='plda', action='store_true', default=False, help='Enable PLDA scoring backend (FEATURE-010). Replaces cosine -cdist with two-covariance PLDA. Default off; behaviour byte-identical when disabled. Composes with --as_norm.')
parser.add_argument('--no_plda',                  dest='plda', action='store_false', help='Explicitly disable PLDA (overrides YAML and earlier CLI). Used by ablation runs.')
parser.add_argument('--plda_train_list',          type=str,   default="",   help='Path to a "<spk_label_int> <relative_path>" file used to fit PLDA. Required when --plda is set; typically the SV train list or a held-out PLDA-dev set.')
parser.add_argument('--plda_train_path',          type=str,   default="",   help='Root path for files in --plda_train_list. Falls back to --train_path when empty.')
parser.add_argument('--plda_dim',                 type=int,   default=200,  help='LDA reduction dimension before PLDA fit. Default 200; typical 150-256.')
parser.add_argument('--plda_save',                dest='plda_save',    action='store_true',  default=True,  help='Cache fitted PLDA to <save_path>/plda.pkl (default on; loads automatically on subsequent runs).')
parser.add_argument('--no_plda_save',             dest='plda_save',    action='store_false', help='Disable PLDA cache.')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

def _expand_env_vars(value, key):
    # Resolve ${VAR} in YAML string values so configs can be portable across
    # machines (see paths.env.example and BUGFIX-013). Unresolved references
    # fail loudly at startup — silently passing through "${SL_SPV_DATA_ROOT}"
    # as a literal path would surface only as a confusing FileNotFoundError
    # several seconds into training.
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(
            f"Config key '{key}={value}' references an undefined environment "
            f"variable. Set the required variable (see paths.env.example) or "
            f"override on the CLI with --{key} <abs-path>."
        )
    return expanded

# BUGFIX-022: snake_case YAML aliases for the four camelCase argparse args
# inherited from voxceleb_trainer upstream. Either form is accepted; the
# canonical Python attribute name stays camelCase so downstream model /
# loss signatures (def __init__(..., nClasses, nOut, ...)) are untouched.
_CAMEL_YAML_ALIASES = {
    'n_classes': 'nClasses',
    'n_data_loader_thread': 'nDataLoaderThread',
    'n_out': 'nOut',
    'n_per_speaker': 'nPerSpeaker',
}

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    yml_config = {_CAMEL_YAML_ALIASES.get(k, k): v for k, v in yml_config.items()}
    for k, v in yml_config.items():
        if k in args.__dict__:
            v = _expand_env_vars(v, k)
            if isinstance(v, (dict, list)):
                # Structured YAML values (e.g., augment_chain dict per
                # BUGFIX-016) pass through without scalar type coercion.
                args.__dict__[k] = v
            else:
                typ = find_option_type(k, parser)
                # store_true / store_false flags register with type=None in
                # argparse. YAML already parses `true`/`false` as bool, so
                # pass them through instead of crashing on `None(v)`.
                args.__dict__[k] = v if typ is None else typ(v)
        else:
            sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")


## ===== ===== ===== ===== ===== ===== ===== =====
## FEATURE-003 — per-language evaluation
## ===== ===== ===== ===== ===== ===== ===== =====

def _parse_per_lang_test_lists(value):
    """Accept either a CLI string "lang1:path1,lang2:path2" or a dict
    (from YAML). Returns a list of (lang, path) tuples preserving order,
    or [] if empty. Fails loudly on malformed input — silent fall-through
    to "no per-language eval" would mask a config typo for an entire run.
    """
    if not value:
        return []
    if isinstance(value, dict):
        return [(str(k), str(v)) for k, v in value.items()]
    pairs = []
    for chunk in str(value).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(
                f"--per_lang_test_lists item {chunk!r} missing 'lang:path' colon."
            )
        lang, path = chunk.split(':', 1)
        pairs.append((lang.strip(), path.strip()))
    return pairs


def _validate_per_lang_lists(pairs):
    """Fail loudly if any list path is missing or empty (zero lines).
    Silent failures here would produce empty pooled metrics that look
    plausible but are computed over no trials.
    """
    for lang, path in pairs:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--per_lang_test_lists: {lang} -> {path} does not exist."
            )
        with open(path) as f:
            nlines = sum(1 for ln in f if ln.strip())
        if nlines == 0:
            raise ValueError(
                f"--per_lang_test_lists: {lang} -> {path} is empty."
            )


def _eval_one_list(trainer, args, label, raw_collect):
    """Run evaluateFromList for the current args.test_list and report
    EER / VEER_avg / MinDCF / Threshold (plus AS-Norm diagnostics when
    enabled). Appends (scores, labels) onto raw_collect for pooling.
    Returns (eer, mindcf, threshold).
    """
    sc, lab, _ = trainer.evaluateFromList(**vars(args))
    if args.gpu != 0:
        return None
    result = tuneThresholdfromScore(sc, lab, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, _ = ComputeMinDcf(
        fnrs, fprs, thresholds,
        args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa,
    )
    eer = float(result[1])
    eer_avg = float(result[5])
    threshold = float(result[4])
    print(f'[per-lang {label}] VEER {eer:2.4f}, VEER_avg {eer_avg:2.4f}, '
          f'MinDCF {mindcf:2.5f}, Threshold {threshold:f}')
    raw_sc = getattr(trainer, '_last_raw_scores', None)
    if getattr(args, 'as_norm', False) and raw_sc is not None:
        raw_result = tuneThresholdfromScore(raw_sc, lab, [1, 0.1])
        raw_fnrs, raw_fprs, raw_th = ComputeErrorRates(raw_sc, lab)
        raw_mindcf, _ = ComputeMinDcf(
            raw_fnrs, raw_fprs, raw_th,
            args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa,
        )
        print(f'[per-lang {label}] (AS-Norm) Raw VEER {raw_result[1]:2.4f}, '
              f'Raw MinDCF {raw_mindcf:2.5f}  '
              f'(delta MinDCF {(mindcf - raw_mindcf):+.5f})')
        raw_collect['raw_scores'].extend(raw_sc)
    raw_collect['scores'].extend(sc)
    raw_collect['labels'].extend(lab)
    return eer, mindcf, threshold


def _run_per_lang_eval(trainer, args, per_lang_pairs, epoch_label=""):
    """Loop over (lang, test_list_path); evaluate each; then report
    pooled metrics on the concatenation. Mutates args.test_list during
    the loop and restores it on exit. Rank 0 only does the reporting;
    other ranks still participate in distributed eval.
    """
    saved_test_list = args.test_list
    raw_collect = {'scores': [], 'labels': [], 'raw_scores': []}
    try:
        for lang, path in per_lang_pairs:
            args.test_list = path
            tag = f"{epoch_label}{lang}" if epoch_label else lang
            _eval_one_list(trainer, args, tag, raw_collect)
    finally:
        args.test_list = saved_test_list

    if args.gpu != 0:
        return None

    pooled_sc, pooled_lab = raw_collect['scores'], raw_collect['labels']
    if not pooled_sc:
        print("[per-lang POOLED] no trials collected; skipping pooled metrics.")
        return None
    pooled_result = tuneThresholdfromScore(pooled_sc, pooled_lab, [1, 0.1])
    pooled_fnrs, pooled_fprs, pooled_th = ComputeErrorRates(pooled_sc, pooled_lab)
    pooled_mindcf, _ = ComputeMinDcf(
        pooled_fnrs, pooled_fprs, pooled_th,
        args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa,
    )
    print(f'[per-lang POOLED{("/" + epoch_label) if epoch_label else ""}] '
          f'VEER {pooled_result[1]:2.4f}, VEER_avg {pooled_result[5]:2.4f}, '
          f'MinDCF {pooled_mindcf:2.5f}, Threshold {pooled_result[4]:f}')
    return {
        'sc': pooled_sc, 'lab': pooled_lab,
        'raw_sc': raw_collect['raw_scores'] if raw_collect['raw_scores'] else None,
        'eer': float(pooled_result[1]),
        'mindcf': float(pooled_mindcf),
        'threshold': float(pooled_result[4]),
        'fnrs': pooled_fnrs, 'fprs': pooled_fprs,
    }


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def _configure_determinism(args):
    # Default (training) path: cudnn picks the best kernels per shape; allows
    # nondeterministic CUDA reductions for speed.
    # --deterministic path: force reproducibility for paper / ablation runs.
    # See docs/bugfixes/BUGFIX-017 for the trade-offs.
    if getattr(args, "deterministic", False):
        # CUBLAS workspace config — required by PyTorch for deterministic
        # CUBLAS ops. Must be set BEFORE any CUDA tensor allocation, which
        # is why this runs at the top of main_worker.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only=True lets ops with no deterministic implementation still
        # run with a warning. Flip to warn_only=False for strict paper-grade
        # mode where any nondeterministic op should hard-fail.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    _configure_determinism(args)

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=args.ddp_find_unused_parameters)

        print(f'Loaded the model on GPU {args.gpu}')

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]
    
    # Define variables for early stopping
    best_eer = float('inf')
    epochs_since_improvement = 0
    best_model_path = os.path.join(args.model_save_path, "model_best.model")
    best_eer_path = os.path.join(args.model_save_path, "model_best.eer")
    best_threshold_path = os.path.join(args.model_save_path, "model_best.threshold")
    best_roc_curve_path = os.path.join(args.result_save_path, "roc_curve_best.png")
    
    # Initialize TensorBoard writer
    writer = None
    if args.gpu == 0 and SummaryWriter is not None:
        log_dir = os.path.join(args.save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled. Log directory: {log_dir}")
    

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile_path = os.path.join(args.result_save_path, "scores.txt")
        scorefile   = open(scorefile_path, "a+")
        print(f"Score file opened at: {scorefile_path}")
        
        # Check if a best EER file already exists (for resuming)
        if os.path.exists(best_eer_path):
            try:
                with open(best_eer_path, 'r') as f:
                    best_eer = float(f.readline().strip())
                print(f"Resuming training, best EER so far: {best_eer:2.4f}%")
            except:
                print(f"Could not read {best_eer_path}, starting EER from infinity.")
                best_eer = float('inf')


    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    # prefetch_factor / persistent_workers are only valid when workers > 0.
    _loader_extra = {}
    if args.nDataLoaderThread > 0:
        _loader_extra = dict(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        **_loader_extra,
    )

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    # Find all model0*.model files, sort them, and remove 'model_best.model' if it's caught
    modelfiles = glob.glob(os.path.join(args.model_save_path, 'model0*.model'))
    modelfiles.sort()
    if best_model_path in modelfiles:
        modelfiles.remove(best_model_path)

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print(f"Model {args.initial_model} loaded!")
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print(f"Model {modelfiles[-1]} loaded from previous state!")
        # Get epoch number from filename (e.g., model000000010.model -> 10)
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print(f'Total parameters: {pytorch_total_params}')
        print(f'Test list: {args.test_list}')

        # FEATURE-003: per-language evaluation. When --per_lang_test_lists is
        # set, each list runs independently and per-language + pooled metrics
        # are reported. Returns early; legacy single-list path below is skipped.
        per_lang_pairs = _parse_per_lang_test_lists(getattr(args, 'per_lang_test_lists', ''))
        if per_lang_pairs:
            if args.gpu == 0:
                _validate_per_lang_lists(per_lang_pairs)
                print(f'[per-lang] {len(per_lang_pairs)} lists: '
                      + ', '.join(f"{l}={p}" for l, p in per_lang_pairs))
            _run_per_lang_eval(trainer, args, per_lang_pairs)
            return

        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            # VEER is the conservative max(FPR,FNR) definition; VEER_avg is the
            # standard literature (FPR+FNR)/2 definition (BUGFIX-020).
            print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")}, VEER {result[1]:2.4f}, VEER_avg {result[5]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[4]:f}')

            # FEATURE-002: when AS-Norm is on, also report raw (pre-AS-Norm)
            # EER/MinDCF for diagnostic comparison. Raw scores were stashed on
            # the trainer by SpeakerNet._run_as_norm.
            raw_sc = getattr(trainer, '_last_raw_scores', None)
            if args.as_norm and raw_sc is not None:
                raw_result = tuneThresholdfromScore(raw_sc, lab, [1, 0.1])
                raw_fnrs, raw_fprs, raw_th = ComputeErrorRates(raw_sc, lab)
                raw_mindcf, _ = ComputeMinDcf(raw_fnrs, raw_fprs, raw_th, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                print(f'[AS-Norm] Raw     VEER {raw_result[1]:2.4f}, VEER_avg {raw_result[5]:2.4f}, MinDCF {raw_mindcf:2.5f}')
                print(f'[AS-Norm] AS-Norm VEER {result[1]:2.4f}, VEER_avg {result[5]:2.4f}, MinDCF {mindcf:2.5f}  '
                      f'(delta MinDCF {(mindcf - raw_mindcf):+.5f})')

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(os.path.join(args.result_save_path, f'run{strtime}.zip'), 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(os.path.join(args.result_save_path, f'run{strtime}.cmd'), 'w') as f:
            f.write(f'{args}')

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch {it}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f}')
            scorefile.write(f"Epoch {it}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f} \n")
            
            # Log training stats to TensorBoard
            if writer is not None:
                writer.add_scalar('Train/Loss', loss, it)
                writer.add_scalar('Train/EER', traineer, it)
                writer.add_scalar('Train/LR', max(clr), it)


        if it % args.test_interval == 0:

            # FEATURE-003: per-language evaluation during training. The pooled
            # EER drives best-model tracking / early stopping so the rest of
            # the loop keeps a single scalar to compare against; per-language
            # banners are printed inside _run_per_lang_eval.
            _epoch_pairs = _parse_per_lang_test_lists(getattr(args, 'per_lang_test_lists', ''))

            if _epoch_pairs:
                pooled = _run_per_lang_eval(trainer, args, _epoch_pairs, epoch_label=f"epoch{it}/")
                if args.gpu != 0 or pooled is None:
                    # Non-rank-0 workers participate in eval but skip reporting.
                    # If rank 0 got None (no trials), skip downstream too.
                    continue
                sc, lab = pooled['sc'], pooled['lab']
                fnrs, fprs = pooled['fnrs'], pooled['fprs']
                current_eer = pooled['eer']
                current_threshold = pooled['threshold']
                mindcf = pooled['mindcf']
                raw_sc_for_diag = pooled['raw_sc']
                eers.append(current_eer)
                print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch {it} '
                      f'(POOLED), VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, '
                      f'Threshold {current_threshold:f}')
                scorefile.write(
                    f"Epoch {it} (POOLED), VEER {current_eer:2.4f}, "
                    f"MinDCF {mindcf:2.5f}, Threshold {current_threshold:f}\n"
                )
            else:
                sc, lab, _ = trainer.evaluateFromList(**vars(args))
                if args.gpu != 0:
                    continue
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                current_eer = float(result[1])
                current_threshold = float(result[4])
                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                eers.append(current_eer)
                print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch {it}, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {current_threshold:f}')
                scorefile.write(f"Epoch {it}, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {current_threshold:f}\n")
                raw_sc_for_diag = getattr(trainer, '_last_raw_scores', None)

            if args.gpu == 0:

                # FEATURE-002: per-epoch AS-Norm diagnostic (raw vs normalised).
                if args.as_norm and raw_sc_for_diag is not None:
                    raw_fnrs, raw_fprs, raw_th = ComputeErrorRates(raw_sc_for_diag, lab)
                    raw_mindcf, _ = ComputeMinDcf(raw_fnrs, raw_fprs, raw_th, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                    raw_result = tuneThresholdfromScore(raw_sc_for_diag, lab, [1, 0.1])
                    print(f'[AS-Norm] Epoch {it}, Raw VEER {raw_result[1]:2.4f}, Raw MinDCF {raw_mindcf:2.5f}  '
                          f'(delta MinDCF {(mindcf - raw_mindcf):+.5f})')
                    scorefile.write(
                        f"Epoch {it}, RawVEER {raw_result[1]:2.4f}, RawMinDCF {raw_mindcf:2.5f}\n"
                    )

                # Log validation stats to Tensorboard
                if writer is not None:
                    writer.add_scalar('Val/EER', current_eer, it)
                    writer.add_scalar('Val/MinDCF', mindcf, it)

                # --- NEW BEST MODEL & EARLY STOPPING LOGIC ---
                
                if current_eer < best_eer:
                    print(f'🎉 New best EER: {current_eer:2.4f}% (was {best_eer:2.4f}%)')
                    best_eer = current_eer
                    epochs_since_improvement = 0 # Reset patience
                    
                    # Save the best model
                    trainer.saveParameters(best_model_path)
                    with open(best_eer_path, 'w') as eerfile:
                        eerfile.write(f'{best_eer:2.4f}')
                    
                    # Save the best threshold
                    with open(best_threshold_path, 'w') as f:
                        f.write(f'{current_threshold:f}')

                    print(f'SAVING BEST MODEL (Epoch {it}) to {best_model_path}')
                    print(f'SAVING BEST THRESHOLD ({current_threshold:f}) to {best_threshold_path}')

                    # Plot and save ROC curve
                    if plt is not None:
                        try:
                            tprs = 1 - fnrs
                            fig = plt.figure()
                            plt.plot(fprs, tprs, label=f'ROC (EER = {current_eer:2.2f}%)')
                            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                            
                            # Find and plot the EER point
                            eer_fpr = fprs[numpy.nanargmin(numpy.abs(fnrs - fprs))]
                            eer_tpr = tprs[numpy.nanargmin(numpy.abs(fnrs - fprs))]
                            plt.plot(eer_fpr, eer_tpr, 'ro', label=f'EER Point ({eer_fpr:.2f}, {eer_tpr:.2f})')

                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate (1 - FNR)')
                            plt.title(f'ROC Curve - Epoch {it}')
                            plt.legend()
                            plt.grid(True)
                            
                            # Save to file
                            plt.savefig(best_roc_curve_path)
                            print(f"Saved new best ROC curve to {best_roc_curve_path}")

                            # Add to TensorBoard
                            if writer is not None:
                                writer.add_figure('Val/ROC_Curve', fig, global_step=it)
                            
                            plt.close(fig) # Close figure to free memory
                        
                        except Exception as e:
                            print(f"Failed to plot or save ROC curve: {e}")

                else:
                    epochs_since_improvement += 1
                    print(f'EER did not improve: {current_eer:2.4f}% (Best is {best_eer:2.4f}%)')
                
                # --- ORIGINAL CHECKPOINTING (for resuming) ---
                # Always save the latest interval checkpoint
                latest_model_path = os.path.join(args.model_save_path, f"model{it:09d}.model")
                trainer.saveParameters(latest_model_path)
                
                latest_eer_path = os.path.join(args.model_save_path, f"model{it:09d}.eer")
                with open(latest_eer_path, 'w') as eerfile:
                    eerfile.write(f'{current_eer:2.4f}')
                
                print(f"Saved interval checkpoint to {latest_model_path}")
                
                scorefile.flush()

                # --- CHECK FOR EARLY STOPPING ---
                if args.patience > 0 and epochs_since_improvement >= args.patience:
                    print(f'\nNo EER improvement for {epochs_since_improvement} test intervals (patience={args.patience}). EARLY STOPPING.')
                    scorefile.write(f"\nEarly stopping at epoch {it}.\n")
                    break # Exit the main training loop
        
        # Check if the loop was broken by early stopping
        if args.patience > 0 and epochs_since_improvement >= args.patience:
            break

    if args.gpu == 0:
        scorefile.close()
        if writer is not None:
            writer.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    # FEATURE-004: assert initial_model is set when --finetune is on. Fail
    # before GPU allocation so a typo surfaces in <1s instead of after model
    # load. Refusing to silently fine-tune from random init.
    if getattr(args, 'finetune', False) and not args.initial_model:
        raise ValueError(
            "--finetune requires --initial_model to be set; refusing to "
            "fine-tune from random init. Set initial_model: <checkpoint> "
            "in YAML or pass --initial_model <path> on the CLI."
        )

    args.model_save_path     = os.path.join(args.save_path, "model")
    args.result_save_path    = os.path.join(args.save_path, "result")
    args.feat_save_path      = "" # Not used in this script, but kept for compatibility

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print(f'Number of GPUs: {n_gpus}')
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()