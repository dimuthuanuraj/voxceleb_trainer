#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib
import os, hashlib, functools, re  # BUGFIX-018: streaming evaluation; re: FEATURE-005 LLRD

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler

# FEATURE-002: AS-Norm score normalisation. Pure functions; no Trainer coupling.
from score_norm import (
    extract_cohort_embeddings,
    compute_file_cohort_stats,
    apply_as_norm,
)

# FEATURE-010: PLDA scoring backend. Pure NumPy; no Trainer coupling.
from plda import TwoCovPLDA, extract_plda_train_embeddings


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


def _resolve_device(gpu):
    """Resolve a torch.device from a (possibly None) GPU index.

    - int + CUDA available -> cuda:<gpu>
    - None + CUDA available -> cuda (current device, typically set by main_worker)
    - otherwise              -> cpu
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device("cuda", int(gpu))
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


## FEATURE-008: Gradient Reversal Layer (Ganin & Lempitsky 2015).
## Identity forward, -lambda * grad backward. The lambda kwarg controls how
## strongly the discriminator's loss pushes the encoder AWAY from the
## discriminator's minimum.
class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


## FEATURE-008: DANN head. 2-layer MLP discriminator over the GRL'd embedding.
## Standard CE with ignore_index=-1; speakers missing from the lookup are
## excluded from the adversarial gradient.
class DANNHead(nn.Module):

    def __init__(self, embedding_dim, num_classes, hidden_dim=256, lambda_=1.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        self.crit = nn.CrossEntropyLoss(ignore_index=-1)
        # Plain Python attribute (not a Parameter or buffer); trainer can
        # mutate it directly to implement a lambda schedule later.
        self.lambda_ = float(lambda_)

    def forward(self, x, label):
        rev = GradientReversalFn.apply(x, self.lambda_)
        logits = self.classifier(rev)
        loss = self.crit(logits, label)
        with torch.no_grad():
            valid = label >= 0
            if valid.any():
                prec = (logits[valid].argmax(-1) == label[valid]).float().mean()
            else:
                prec = torch.zeros((), device=x.device)
        return loss, prec


## FEATURE-007: language-ID auxiliary head. Module-level so it stays
## importable / testable independently of the SpeakerNet wrapper.
class LangAuxHead(nn.Module):
    """A small `Linear(embedding_dim -> num_classes)` classifier with CE loss.
    Speakers absent from the lookup get label = -1 and are excluded via
    ignore_index=-1. Returns (loss, fraction-correct-on-valid)."""

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.crit = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, label):
        logits = self.fc(x)
        loss = self.crit(logits, label)
        with torch.no_grad():
            valid = label >= 0
            if valid.any():
                prec = (logits[valid].argmax(-1) == label[valid]).float().mean()
            else:
                prec = torch.zeros((), device=x.device)
        return loss, prec


def _load_lang_lookup(path, n_classes_spk, num_lang_classes):
    """Read a `<spk_label_int> <lang_label_int>` file into a LongTensor[nClasses].
    Lines starting with '#' and blank lines are skipped. Missing speakers
    default to -1 (excluded by ignore_index=-1 in CrossEntropyLoss).
    Raises ValueError on out-of-range lang labels or non-integer columns —
    silent skip would mask data-prep bugs that only manifest as a CUDA
    assertion at the first batch."""
    table = torch.full((int(n_classes_spk),), -1, dtype=torch.long)
    n_valid = 0
    with open(path) as f:
        for ln_no, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            try:
                spk = int(parts[0])
                lang = int(parts[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"lang_aux_label_file {path} line {ln_no}: expected "
                    f"'<spk_int> <lang_int>', got {ln!r}"
                )
            if not (0 <= lang < num_lang_classes):
                raise ValueError(
                    f"lang_aux_label_file {path} line {ln_no}: lang_label "
                    f"{lang} out of range [0, {num_lang_classes})."
                )
            if 0 <= spk < int(n_classes_spk):
                table[spk] = lang
                n_valid += 1
    return table, n_valid


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker
        self.device = _resolve_device(kwargs.get("gpu", None))

        # Cache shared values for FEATURE-007 / FEATURE-008.
        emb_dim = int(kwargs.get('nOut', 512))
        n_classes_spk = int(kwargs.get('nClasses', 0))

        # FEATURE-007: optional language-ID aux head. When enabled, the lookup
        # buffer maps each speaker_label_int -> lang_label_int (or -1 for
        # speakers absent from the lookup file). The head is trained jointly
        # with the speaker loss and unused at eval time.
        self.lang_aux = bool(kwargs.get('lang_aux', False))
        if self.lang_aux:
            n_lang = int(kwargs.get('lang_aux_num_classes', 4))
            self.lang_aux_weight = float(kwargs.get('lang_aux_weight', 0.3))
            lookup_path = kwargs.get('lang_aux_label_file', '') or ''
            if not lookup_path:
                raise ValueError(
                    "lang_aux: true requires lang_aux_label_file to be set."
                )
            table, n_valid = _load_lang_lookup(lookup_path, n_classes_spk, n_lang)
            print(f"[lang_aux] {n_valid}/{n_classes_spk} speakers have lang labels "
                  f"(weight={self.lang_aux_weight}, num_classes={n_lang})")
            self.register_buffer('lang_spk_to_lang', table)
            self.lang_aux_head = LangAuxHead(emb_dim, n_lang)

        # FEATURE-008: DANN adversarial heads (language and/or channel).
        # Same per-speaker lookup format as FEATURE-007. The encoder gradient
        # is reversed via GRL inside DANNHead.forward; the discriminator's own
        # weights still update normally.
        self.dann_lang = bool(kwargs.get('dann_lang', False))
        if self.dann_lang:
            if self.lang_aux:
                print("[dann_lang] WARNING: lang_aux is also enabled; the two "
                      "objectives are OPPOSITE on the same signal. Disable one.")
            n_lang = int(kwargs.get('dann_lang_num_classes', 4))
            self.dann_lang_weight = float(kwargs.get('dann_lang_weight', 1.0))
            lookup_path = kwargs.get('dann_lang_label_file', '') or ''
            if not lookup_path:
                raise ValueError(
                    "dann_lang: true requires dann_lang_label_file to be set."
                )
            table, n_valid = _load_lang_lookup(lookup_path, n_classes_spk, n_lang)
            print(f"[dann_lang] {n_valid}/{n_classes_spk} speakers have lang labels "
                  f"(weight={self.dann_lang_weight}, "
                  f"lambda={float(kwargs.get('dann_lang_lambda', 0.1))}, "
                  f"num_classes={n_lang})")
            self.register_buffer('dann_lang_spk_to_lang', table)
            self.dann_lang_head = DANNHead(
                emb_dim, n_lang,
                lambda_=float(kwargs.get('dann_lang_lambda', 0.1)),
            )

        self.dann_channel = bool(kwargs.get('dann_channel', False))
        if self.dann_channel:
            n_ch = int(kwargs.get('dann_channel_num_classes', 3))
            self.dann_channel_weight = float(kwargs.get('dann_channel_weight', 1.0))
            lookup_path = kwargs.get('dann_channel_label_file', '') or ''
            if not lookup_path:
                raise ValueError(
                    "dann_channel: true requires dann_channel_label_file to be set."
                )
            table, n_valid = _load_lang_lookup(lookup_path, n_classes_spk, n_ch)
            print(f"[dann_channel] {n_valid}/{n_classes_spk} speakers have channel labels "
                  f"(weight={self.dann_channel_weight}, "
                  f"lambda={float(kwargs.get('dann_channel_lambda', 0.1))}, "
                  f"num_classes={n_ch})")
            self.register_buffer('dann_channel_spk_to_channel', table)
            self.dann_channel_head = DANNHead(
                emb_dim, n_ch,
                lambda_=float(kwargs.get('dann_channel_lambda', 0.1)),
            )

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).to(self.device, non_blocking=True)
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        # FEATURE-007 / FEATURE-008: auxiliary heads on per-utterance
        # embeddings (BEFORE the nPerSpeaker reshape, so each utterance
        # contributes one prediction). All three losses share the same
        # label expansion.
        extra_loss = None
        if self.lang_aux or self.dann_lang or self.dann_channel:
            label_expanded = label.repeat_interleave(self.nPerSpeaker)
        if self.lang_aux:                                       # FEATURE-007
            lang_label = self.lang_spk_to_lang[label_expanded]
            aux_loss, _ = self.lang_aux_head(outp, lang_label)
            extra_loss = self.lang_aux_weight * aux_loss
        if self.dann_lang:                                      # FEATURE-008
            lang_label = self.dann_lang_spk_to_lang[label_expanded]
            dann_l, _ = self.dann_lang_head(outp, lang_label)
            extra_loss = (extra_loss if extra_loss is not None else 0) \
                + self.dann_lang_weight * dann_l
        if self.dann_channel:                                   # FEATURE-008
            ch_label = self.dann_channel_spk_to_channel[label_expanded]
            dann_c, _ = self.dann_channel_head(outp, ch_label)
            extra_loss = (extra_loss if extra_loss is not None else 0) \
                + self.dann_channel_weight * dann_c

        # Reshape (nPerSpeaker * B, D) -> (B, nPerSpeaker, D), the grouped form
        # consumed by metric-learning losses.
        outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0)

        if getattr(self.__L__, "expects_grouped_input", False):
            # angleproto / proto / ge2e / softmaxproto / triplet: keep (B, P, D).
            nloss, prec1 = self.__L__.forward(outp, label)
        else:
            # softmax / amsoftmax / aamsoftmax: flatten to (B*P, D) and replicate
            # labels so prec1 measures real classification accuracy when P > 1.
            outp = outp.reshape(-1, outp.size(-1))
            label = label.repeat_interleave(self.nPerSpeaker)
            nloss, prec1 = self.__L__.forward(outp, label)

        if extra_loss is not None:
            nloss = nloss + extra_loss

        return nloss, prec1


## FEATURE-004: cross-lingual fine-tune freeze helper.
## Resolves aliases / dotted paths to submodules of the WrappedModel, freezes
## their parameters, and switches them to eval() so BN stats stop drifting.
_FINETUNE_FREEZE_ALIASES = {
    'frontend':    ['module.__S__.torchfb', 'module.__S__.instancenorm'],
    'ssl_encoder': ['module.__S__.encoder'],
    'backbone':    ['module.__S__'],
    'loss_head':   ['module.__L__'],
}


def _parse_finetune_freeze(value):
    """Accept CLI string ('frontend,loss_head'), YAML list, or empty.
    Returns a flat list of resolved dotted module names."""
    if not value:
        return []
    if isinstance(value, str):
        items = [s.strip() for s in value.split(',') if s.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(s).strip() for s in value if str(s).strip()]
    else:
        raise ValueError(f"finetune_freeze must be str or list, got {type(value).__name__}")
    resolved = []
    for it in items:
        if it in _FINETUNE_FREEZE_ALIASES:
            resolved.extend(_FINETUNE_FREEZE_ALIASES[it])
        else:
            resolved.append(it)
    return resolved


def _apply_finetune_freeze(wrapped_model, freeze_specs):
    """Walk freeze_specs against wrapped_model's named submodules. Freeze
    matching submodules in place. Returns (n_train, n_frozen, hits) for
    the diagnostic print. Fails loudly if any spec matches nothing.
    """
    named = dict(wrapped_model.named_modules())
    hits = []
    misses = []
    for spec in freeze_specs:
        # Aliases resolve to a list; if the alias's expansion includes
        # a path that doesn't exist on this particular model, that's
        # OK (e.g. 'frontend' on an SSL model has no torchfb). Direct
        # user-typed paths still fail loudly.
        if spec not in named:
            if any(spec in v for v in _FINETUNE_FREEZE_ALIASES.values()):
                continue
            misses.append(spec)
            continue
        hits.append(spec)
        mod = named[spec]
        for p in mod.parameters():
            p.requires_grad = False
        mod.eval()
    if misses:
        avail = ', '.join(sorted(named.keys())[:20])
        raise ValueError(
            f"finetune_freeze: no submodule matched {misses!r}. "
            f"First 20 available names: {avail} (...)"
        )
    n_train = sum(1 for p in wrapped_model.parameters() if p.requires_grad)
    n_frozen = sum(1 for p in wrapped_model.parameters() if not p.requires_grad)
    return n_train, n_frozen, hits


## FEATURE-005: layer-wise learning-rate decay (LLRD).
## Bucket each parameter by depth from the loss head (depth 0 = top),
## then assign lr = base_lr * decay^depth. Aliases map common encoder
## families to layer-index regexes.
_LLRD_PATTERN_ALIASES = {
    'ssl':       r'\.encoder\.encoder\.layers\.(\d+)\.',   # WavLM / XLS-R / mHuBERT
    'mlpmixer':  r'\.mixer_blocks\.(\d+)\.',
    'ecapa':     r'\.layer(\d+)\.',                        # FEATURE-006 ECAPA-TDNN
}
_LLRD_HEAD_HINTS = (
    '__L__.',                # loss head
    '__S__.attention.',      # ASP/SAP attention pooling
    '__S__.bn.',             # post-pool batchnorm
    '__S__.fc.',             # final FC projection
    'module.__L__.',
    'module.__S__.attention.',
    'module.__S__.bn.',
    'module.__S__.fc.',
)
_LLRD_DEEPEST_HINTS = (
    'feature_extractor',
    'feature_projection',
    'masked_spec_embed',
    'pos_conv_embed',
)


def _resolve_llrd_pattern(value):
    """Accept an alias name or a raw regex; return a compiled regex.
    Empty / None defaults to the 'ssl' alias."""
    if not value:
        value = 'ssl'
    pat = _LLRD_PATTERN_ALIASES.get(value, value)
    return re.compile(pat)


def _build_llrd_param_groups(wrapped_model, base_lr, decay, layer_re):
    """Walk named_parameters; bucket each trainable param by depth; return
    (param_groups, max_layer_idx). Returns (None, -1) if no layer-pattern
    match exists — caller falls back to a flat list."""
    max_n = -1
    for name, _ in wrapped_model.named_parameters():
        m = layer_re.search(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    if max_n < 0:
        return None, max_n

    buckets = {}  # depth -> list[(name, param)]
    for name, p in wrapped_model.named_parameters():
        if not p.requires_grad:
            continue
        depth = _llrd_depth_for(name, layer_re, max_n)
        buckets.setdefault(depth, []).append((name, p))

    param_groups = []
    for d in sorted(buckets.keys()):
        lr = float(base_lr) * (float(decay) ** d)
        param_groups.append({
            'params': [p for _, p in buckets[d]],
            'lr': lr,
            'name': f'llrd_depth_{d}',
        })
    return param_groups, max_n


def _llrd_depth_for(name, layer_re, max_n):
    """Depth 0 = top (loss head + pooling head). Higher = deeper."""
    if any(h in name for h in _LLRD_HEAD_HINTS):
        return 0
    m = layer_re.search(name)
    if m:
        layer_n = int(m.group(1))
        return (max_n - layer_n) + 1
    if any(h in name for h in _LLRD_DEEPEST_HINTS):
        return max_n + 2
    # Middle bucket fallback for params that match nothing else
    return max(1, (max_n + 1) // 2)


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        # FEATURE-004: cross-lingual fine-tune. Freeze submodules BEFORE
        # optimizer construction so the param list and optimizer state are
        # already shape-correct. LR multiplier is applied to kwargs in-place
        # so the optimizer module sees the scaled value.
        if bool(kwargs.get('finetune', False)):
            freeze_specs = _parse_finetune_freeze(kwargs.get('finetune_freeze', ''))
            if freeze_specs:
                n_train, n_frozen, hits = _apply_finetune_freeze(self.__model__, freeze_specs)
                print(f"[finetune] Froze {len(hits)} submodule(s): {hits}")
                print(f"[finetune] Training {n_train} params ({n_frozen} frozen)")
                if n_train == 0:
                    raise ValueError(
                        "finetune_freeze froze every parameter; nothing left to optimise."
                    )
            mult = float(kwargs.get('finetune_lr_multiplier', 1.0) or 1.0)
            if mult != 1.0:
                kwargs['lr'] = float(kwargs.get('lr', 0.001)) * mult
                print(f"[finetune] Scaled lr by {mult} -> {kwargs['lr']}")

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")

        # FEATURE-005: layer-wise learning-rate decay. When llrd is on, the
        # optimizer is built over param_groups (one per depth bucket) instead
        # of a flat trainable list. The kwarg `lr` becomes the default for
        # any group that lacks one — every group here specifies its own.
        if bool(kwargs.get('llrd', False)):
            layer_re = _resolve_llrd_pattern(kwargs.get('llrd_layer_pattern', 'ssl'))
            decay = float(kwargs.get('llrd_decay', 0.9) or 0.9)
            base_lr = float(kwargs.get('lr', 0.001))
            param_groups, max_n = _build_llrd_param_groups(
                self.__model__, base_lr, decay, layer_re,
            )
            if param_groups is None:
                print(f"[llrd] layer pattern {layer_re.pattern!r} matched no parameters; "
                      f"falling back to flat trainable list.")
                trainable = [p for p in self.__model__.parameters() if p.requires_grad]
                self.__optimizer__ = Optimizer(trainable, **kwargs)
            else:
                print(f"[llrd] {len(param_groups)} depth buckets (max layer index {max_n}, "
                      f"decay {decay}, base lr {base_lr}):")
                for g in param_groups:
                    print(f"  {g['name']:>18}: {len(g['params']):>5} params, lr={g['lr']:.3e}")
                self.__optimizer__ = Optimizer(param_groups, **kwargs)
        else:
            # FEATURE-004: optimise only parameters with requires_grad=True.
            # Without freezing, this is identical to self.__model__.parameters().
            trainable = [p for p in self.__model__.parameters() if p.requires_grad]
            self.__optimizer__ = Optimizer(trainable, **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu
        self.device = _resolve_device(gpu)

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:

            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).to(self.device, non_blocking=True)

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}                  # only populated when not streaming
        stream_keys = set()         # only populated when streaming
        tstart = time.time()

        # BUGFIX-018: streaming evaluation. When --eval_streaming is set, the
        # per-file embeddings are written to <save_path>/eval_feats_tmp/ and
        # lazy-loaded with an LRU cache, instead of being held in the feats
        # dict on every rank. Required for SL-benchmark-scale (~1M-pair)
        # test lists where the dict would not fit in memory.
        eval_streaming = bool(kwargs.get('eval_streaming', False))
        eval_feat_cache_size = int(kwargs.get('eval_feat_cache_size', 4096))
        if eval_streaming:
            feat_dir = os.path.join(kwargs.get('save_path', '.'), 'eval_feats_tmp')
            os.makedirs(feat_dir, exist_ok=True)
            def _feat_path(filename):
                return os.path.join(feat_dir, hashlib.sha1(filename.encode()).hexdigest() + '.pt')
            @functools.lru_cache(maxsize=eval_feat_cache_size)
            def _load_feat(filename):
                # BUGFIX-019: weights_only=True; we wrote these files
                # ourselves moments earlier as pure torch.Tensor pickles,
                # so the strict loader is a no-op for the expected payload.
                return torch.load(_feat_path(filename), map_location='cpu', weights_only=True)

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].to(self.device, non_blocking=True)
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            if eval_streaming:
                torch.save(ref_feat, _feat_path(data[1][0]))
                stream_keys.add(data[1][0])
            else:
                feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            if eval_streaming:
                # Only gather filename keys (cheap); tensors live on the
                # shared filesystem. Barrier first so all writes are visible
                # on rank 0 before it starts reading.
                torch.distributed.barrier()
                all_keys = [None for _ in range(0, torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_keys, stream_keys)
            else:
                ## Gather features from all GPUs
                feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                if eval_streaming:
                    for ks in all_keys[1:]:
                        stream_keys |= ks
                else:
                    feats = feats_all[0]
                    for feats_batch in feats_all[1:]:
                        feats.update(feats_batch)

            # FEATURE-010: PLDA fit / load. Rank 0 only; default None.
            # When set, the trial loop uses PLDA scoring instead of
            # -cdist, and skips the L2-normalisation (PLDA owns
            # preprocessing). Fit is cached to <save_path>/plda.pkl.
            plda_obj = self._setup_plda(kwargs) if bool(kwargs.get('plda', False)) else None

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                if eval_streaming:
                    ref_feat = _load_feat(data[1]).to(self.device, non_blocking=True)
                    com_feat = _load_feat(data[2]).to(self.device, non_blocking=True)
                else:
                    ref_feat = feats[data[1]].to(self.device, non_blocking=True)
                    com_feat = feats[data[2]].to(self.device, non_blocking=True)

                if plda_obj is not None:
                    # FEATURE-010: PLDA scoring. Mean-pool over num_eval crops
                    # to one embedding per side; PLDA's own preprocessing
                    # (centre + length-norm + LDA) handles the rest. Skip the
                    # trial-loop L2-norm — PLDA expects raw embeddings.
                    ref_mean = ref_feat.reshape(num_eval, -1).mean(dim=0).detach().cpu().numpy()
                    com_mean = com_feat.reshape(num_eval, -1).mean(dim=0).detach().cpu().numpy()
                    score = plda_obj.score(ref_mean, com_mean)
                else:
                    if self.__model__.module.__L__.test_normalize:
                        ref_feat = F.normalize(ref_feat, p=2, dim=1)
                        com_feat = F.normalize(com_feat, p=2, dim=1)
                    dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()
                    score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

            # FEATURE-002: AS-Norm post-processing. Rank 0 only; default off.
            self._last_raw_scores = None
            if bool(kwargs.get('as_norm', False)):
                if eval_streaming:
                    def _get_feat(name):
                        return _load_feat(name)
                    file_keys = sorted(stream_keys)
                else:
                    def _get_feat(name):
                        return feats[name]
                    file_keys = sorted(feats.keys())
                all_scores = self._run_as_norm(
                    raw_scores=all_scores,
                    all_trials=all_trials,
                    file_keys=file_keys,
                    get_feat=_get_feat,
                    num_eval=num_eval,
                    kwargs=kwargs,
                )

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## PLDA scoring backend (FEATURE-010)
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def _setup_plda(self, kwargs):
        """Fit or load a PLDA backend. Returns the TwoCovPLDA object, or
        None if --plda was set but the train list is missing (caller falls
        back to cosine scoring with a printed warning)."""
        train_list = kwargs.get('plda_train_list', '') or ''
        if not train_list:
            print("[PLDA] --plda set but plda_train_list is empty; falling back to cosine.")
            return None
        save_path = kwargs.get('save_path', '.')
        cache_file = os.path.join(save_path, 'plda.pkl')
        do_save = bool(kwargs.get('plda_save', True))

        if do_save and os.path.exists(cache_file):
            try:
                plda = TwoCovPLDA.load(cache_file)
                print(f"[PLDA] Loaded cached PLDA from {cache_file} "
                      f"(lda_dim={plda.lda_dim}, {plda.n_speakers} spk, "
                      f"{plda.n_utts} utts)")
                return plda
            except Exception as e:
                print(f"[PLDA] Cache load failed ({e}); re-fitting.")

        train_path = kwargs.get('plda_train_path', '') or kwargs.get('train_path', '')
        print(f"[PLDA] Extracting embeddings over {train_list} "
              f"(root={train_path}) for PLDA fit ...")
        emb, labels = extract_plda_train_embeddings(
            model=self.__model__,
            train_list_path=train_list,
            train_path=train_path,
            nDataLoaderThread=int(kwargs.get('nDataLoaderThread', 4)),
            num_eval=int(kwargs.get('num_eval', 10)),
            eval_frames=int(kwargs.get('eval_frames', 0)),
            sample_rate=int(kwargs.get('sample_rate', 16000)),
            device=self.device,
            rank=0,
        )
        lda_dim = int(kwargs.get('plda_dim', 200))
        print(f"[PLDA] Fitting two-cov PLDA: {emb.shape[0]} embeddings, "
              f"dim {emb.shape[1]} -> lda_dim {lda_dim}")
        plda = TwoCovPLDA(lda_dim=lda_dim).fit(emb, labels)
        print(f"[PLDA] Fit done: {plda.n_speakers} spk, {plda.n_utts} utts.")
        if do_save:
            plda.save(cache_file)
            print(f"[PLDA] Cached to {cache_file}")
        return plda

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## AS-Norm score normalisation (FEATURE-002)
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def _run_as_norm(self, raw_scores, all_trials, file_keys, get_feat,
                     num_eval, kwargs):
        """Compute cohort embeddings, per-file (mu, sigma), and apply
        AS-Norm to raw_scores. Stashes raw_scores on
        self._last_raw_scores so the trainer can still report raw EER.
        Returns the normalised score list.
        """
        cohort_list = kwargs.get('as_norm_cohort_list', '') or ''
        cohort_path = kwargs.get('as_norm_cohort_path', '') or kwargs.get('test_path', '')
        if not cohort_list:
            print("[AS-Norm] --as_norm set but as_norm_cohort_list is empty; skipping.")
            return raw_scores

        save_cohort = bool(kwargs.get('as_norm_save_cohort', True))
        cache_file = None
        if save_cohort:
            cache_file = os.path.join(kwargs.get('save_path', '.'), 'asnorm_cohort.pt')

        # Resolve test_normalize from the loss head (matches trial loop branch).
        try:
            test_normalize = bool(self.__model__.module.__L__.test_normalize)
        except AttributeError:
            test_normalize = True

        print(f"[AS-Norm] cohort_list={cohort_list}  cohort_path={cohort_path}  "
              f"top_k={kwargs.get('as_norm_top_k', 300)}  test_normalize={test_normalize}")

        cohort_feats, _ = extract_cohort_embeddings(
            model=self.__model__,
            cohort_list=cohort_list,
            cohort_path=cohort_path,
            nDataLoaderThread=int(kwargs.get('nDataLoaderThread', 4)),
            num_eval=num_eval,
            eval_frames=int(kwargs.get('eval_frames', 0)),
            sample_rate=int(kwargs.get('sample_rate', 16000)),
            cache_file=cache_file,
            device=self.device,
            rank=0,
        )

        def _iter_files():
            for k in file_keys:
                yield k, get_feat(k)

        file_stats = compute_file_cohort_stats(
            file_iter=_iter_files(),
            cohort_feats=cohort_feats,
            top_k=int(kwargs.get('as_norm_top_k', 300)),
            normalize=test_normalize,
            device=self.device,
            chunk_size=int(kwargs.get('as_norm_chunk_size', 64)),
            rank=0,
        )

        self._last_raw_scores = list(raw_scores)
        return apply_as_norm(raw_scores, all_trials, file_stats)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        # BUGFIX-019: weights_only=True refuses to unpickle arbitrary Python
        # objects from the checkpoint file, blocking the standard
        # pickle-based RCE vector. Saved checkpoints in this repo are pure
        # state_dicts (see saveParameters), so the stricter loader is a
        # no-op for any checkpoint the trainer itself wrote.
        loaded_state = torch.load(path, map_location=self.device, weights_only=True)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
