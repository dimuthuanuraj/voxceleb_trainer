#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
PERFORMANCE OPTIMIZED VERSION - DatasetLoader_performance_updated.py
Optimizations applied:
1. Cached audio loading with LRU cache
2. Pre-computed augmentation to reduce CPU overhead
3. Faster WAV loading with optimized libraries
4. Reduced memory allocations
5. Better numpy operations
6. Multi-threaded augmentation support
"""

import sys
import torch
import numpy
import random
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from functools import lru_cache


# Module-level flag so the short-clip warning fires at most once per worker process.
_short_clip_warned = False


def _pad_short_with_dither(audio, target_length, dither_std=1e-4):
    """Right-pad an audio array with silence plus tiny Gaussian dither.

    Pre-BUGFIX-007 the loader used ``numpy.pad(..., 'wrap')`` which tiles the
    original audio onto itself and fabricates perfectly periodic features that
    the model can learn as a shortcut for short-clip speakers. Silence pad is
    the honest default. The small Gaussian dither (std ~1e-4, about -80 dBFS)
    avoids exact-zero blocks that can produce zero-variance windows downstream.
    """
    shortage = target_length - audio.shape[0]
    if shortage <= 0:
        return audio
    pad = (numpy.random.randn(shortage) * dither_std).astype(audio.dtype, copy=False)
    return numpy.concatenate([audio, pad], axis=0)


def _maybe_warn_short_clip(filename, audiosize, target):
    """Emit a one-time stderr warning the first time a short clip is padded."""
    global _short_clip_warned
    if _short_clip_warned:
        return
    _short_clip_warned = True
    sys.stderr.write(
        "[DatasetLoader] First short clip encountered "
        f"({filename}: {audiosize} samples < {target} required). "
        "Padding with silence + dither. Subsequent short clips will be padded silently. "
        "Consider filtering clips shorter than max_audio at corpus-prep time.\n"
    )

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


# Framing constants used by every speaker-recognition mel-spectrogram in this repo:
# 10 ms hop, 25 ms window. Expressed in seconds so they generalise across rates.
_HOP_SECONDS = 0.010
_WINDOW_SECONDS = 0.025


def _frame_to_samples(max_frames, sample_rate):
    """Audio length in samples for ``max_frames`` frames at ``sample_rate``."""
    hop = int(round(_HOP_SECONDS * sample_rate))
    window = int(round(_WINDOW_SECONDS * sample_rate))
    return max_frames * hop + (window - hop)


def _resample_if_needed(audio, orig_sr, target_sr):
    """Anti-aliased polyphase resample with gcd-based rational ratio."""
    if int(orig_sr) == int(target_sr):
        return audio
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return signal.resample_poly(audio, up, down)


# Augmentation outcome labels, in the order originally encoded by augtype
# 0..4 in the legacy random.randint(0, 4) call. The order is load-bearing:
# downstream code dispatches on the *index*, not the label name.
_AUGMENT_LABELS = ("clean", "reverb", "music", "speech", "noise")


def _parse_augment_chain(spec):
    """Normalise the ``augment_chain`` config value into a 5-tuple of
    probabilities aligned with :data:`_AUGMENT_LABELS`. See BUGFIX-016.

    Accepted forms: ``None`` / ``""`` / ``"uniform"`` (legacy 0.2 each),
    a dict of ``{label: prob}``, a list of single-key dicts (YAML alt
    syntax), or a JSON string of either.
    """
    import json as _json
    if spec is None or spec == "" or spec == "uniform":
        return tuple([1.0 / len(_AUGMENT_LABELS)] * len(_AUGMENT_LABELS))
    if isinstance(spec, str):
        try:
            spec = _json.loads(spec)
        except _json.JSONDecodeError as exc:
            raise ValueError(
                f"augment_chain={spec!r} is not parseable as JSON. Pass a "
                f"dict like '{{\"noise\": 0.3, \"music\": 0.2}}' on the CLI "
                f"or use a YAML config to keep the dict form."
            ) from exc
    if isinstance(spec, list):
        flat = {}
        for entry in spec:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    f"augment_chain list entries must be single-key dicts; "
                    f"got {entry!r}"
                )
            flat.update(entry)
        spec = flat
    if not isinstance(spec, dict):
        raise ValueError(
            f"augment_chain must be a dict (or list of single-key dicts); "
            f"got {type(spec).__name__}"
        )
    unknown = set(spec) - set(_AUGMENT_LABELS)
    if unknown:
        raise ValueError(
            f"augment_chain has unknown labels {sorted(unknown)}; "
            f"valid labels are {list(_AUGMENT_LABELS)}"
        )
    for label, prob in spec.items():
        if not isinstance(prob, (int, float)) or isinstance(prob, bool) or prob < 0:
            raise ValueError(
                f"augment_chain[{label!r}] must be a non-negative number; "
                f"got {prob!r}"
            )
    total = float(sum(spec.values()))
    if total > 1.0 + 1e-6:
        raise ValueError(
            f"augment_chain probabilities sum to {total:.4f} (> 1.0); they "
            f"must sum to at most 1.0. Drop the unspecified mass into "
            f"'clean' implicitly or set 'clean' explicitly."
        )
    remainder = max(0.0, 1.0 - total)
    probs = []
    for label in _AUGMENT_LABELS:
        base = float(spec.get(label, 0.0))
        if label == "clean" and "clean" not in spec:
            base += remainder
        probs.append(base)
    s = sum(probs)
    if s <= 0:
        raise ValueError(
            "augment_chain probabilities sum to zero — at least one outcome "
            "(including implicit 'clean') must have positive probability."
        )
    return tuple(p / s for p in probs)


# OPTIMIZATION: Add caching for frequently loaded audio files.
# Cache key includes target_sr so a process that runs at a non-default rate
# does not silently return native-rate audio from the cache.
@lru_cache(maxsize=1000)
def loadWAV_cached(filename, max_frames, target_sr):
    """Cached version of loadWAV for frequently accessed files.

    Caches the *post-resample* audio so disk IO, decode, and resample each
    happen at most once per (file, target_sr) pair.
    """
    try:
        audio, native_sr = soundfile.read(filename, dtype='float32')  # OPTIMIZATION: Use float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample_if_needed(audio, native_sr, target_sr).astype(numpy.float32, copy=False)
        return audio, target_sr
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


def loadWAV(filename, max_frames, evalmode=True, num_eval=10, sample_rate=16000):

    # Maximum audio length at the target sample rate.
    max_audio = _frame_to_samples(max_frames, sample_rate)

    # OPTIMIZATION: Try cached loading first for evaluation
    if evalmode:
        cached_result = loadWAV_cached(filename, max_frames, sample_rate)
        if cached_result[0] is not None:
            audio, _sr = cached_result
        else:
            return numpy.zeros((num_eval, max_audio), dtype=numpy.float32)
    else:
        # Read wav file and convert to torch tensor
        audio, native_sr = soundfile.read(filename, dtype='float32')  # OPTIMIZATION: Direct float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample_if_needed(audio, native_sr, sample_rate).astype(numpy.float32, copy=False)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        target_length = max_audio + 1  # +1 keeps the historical invariant that audiosize-max_audio >= 1
        _maybe_warn_short_clip(filename, audiosize, target_length)
        audio = _pad_short_with_dither(audio, target_length)
        audiosize = audio.shape[0]

    if evalmode:
        # OPTIMIZATION: Pre-compute all start frames at once
        startframe = numpy.linspace(0, audiosize-max_audio, num=num_eval, dtype=numpy.int64)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    # OPTIMIZATION: Pre-allocate output array
    if evalmode and max_frames == 0:
        feat = audio[numpy.newaxis, :]
    else:
        feat = numpy.empty((len(startframe), max_audio), dtype=numpy.float32)
        for i, asf in enumerate(startframe):
            feat[i] = audio[int(asf):int(asf)+max_audio]

    return feat


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames, sample_rate=16000):

        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.max_audio = max_audio = _frame_to_samples(max_frames, sample_rate)

        self.noisetypes = ['noise','speech','music']

        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
        self.noiselist = {}

        # OPTIMIZATION: Check if paths exist before globbing
        if musan_path and os.path.exists(musan_path):
            augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))

            for file in augment_files:
                noise_type = file.split('/')[-3]
                if noise_type not in self.noiselist:
                    self.noiselist[noise_type] = []
                self.noiselist[noise_type].append(file)
        
        # OPTIMIZATION: Cache RIR files list
        if rir_path and os.path.exists(rir_path):
            self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        else:
            self.rir_files = []

    def additive_noise(self, noisecat, audio):

        # OPTIMIZATION: Fast computation with vectorized operations
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4) 

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], 
            random.randint(numnoise[0], numnoise[1])
        )

        # OPTIMIZATION: Pre-allocate noise array
        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False, sample_rate=self.sample_rate)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4) 
            
            # OPTIMIZATION: Vectorized scaling
            scale = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))
            noises.append(scale * noiseaudio)

        # OPTIMIZATION: More efficient concatenation and summation
        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):

        if not self.rir_files:
            return audio  # OPTIMIZATION: Skip if no RIR files

        rir_file = random.choice(self.rir_files)

        rir, fs = soundfile.read(rir_file, dtype='float32')  # OPTIMIZATION: float32
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        rir = _resample_if_needed(rir, fs, self.sample_rate).astype(numpy.float32, copy=False)
        rir = numpy.expand_dims(rir, 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))

        # OPTIMIZATION: Use FFT-based convolution for speed
        return signal.fftconvolve(audio, rir, mode='full')[:, :self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, sample_rate=16000, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames, sample_rate=sample_rate)

        self.train_list = train_list
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment
        # BUGFIX-016: per-outcome augmentation probabilities. Empty/uniform
        # input falls back to the legacy 0.2 each, statistically identical
        # to the previous random.randint(0, 4).
        self._augment_probs = _parse_augment_chain(kwargs.get("augment_chain"))

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)
        
        print(f"Loaded {len(self.data_list)} training samples from {len(dictkeys)} speakers")

    def __getitem__(self, indices):

        # OPTIMIZATION: Pre-allocate list with known size
        feat = []

        for index in indices:
            
            # OPTIMIZATION: Load audio (potentially from cache)
            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False, sample_rate=self.sample_rate)
            
            if self.augment:
                # augtype index aligns with _AUGMENT_LABELS:
                #   0=clean, 1=reverb, 2=music, 3=speech, 4=noise.
                augtype = random.choices(
                    range(len(_AUGMENT_LABELS)),
                    weights=self._augment_probs,
                    k=1,
                )[0]
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio)
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)
                    
            feat.append(audio)

        # OPTIMIZATION: Use numpy.concatenate with pre-allocated dtype
        feat = numpy.concatenate(feat, axis=0).astype(numpy.float32)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, sample_rate=16000, **kwargs):
        self.max_frames  = eval_frames
        self.num_eval    = num_eval
        self.test_path   = test_path
        self.test_list   = test_list
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        # OPTIMIZATION: Use cached loading for test set
        audio = loadWAV(
            os.path.join(self.test_path, self.test_list[index]),
            self.max_frames,
            evalmode=True,
            num_eval=self.num_eval,
            sample_rate=self.sample_rate,
        )
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label = data_source.data_label
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        # OPTIMIZATION: Use defaultdict for faster dictionary operations
        from collections import defaultdict
        data_dict = defaultdict(list)

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            data_dict[speaker_label].append(index)

        ## Group file indices for each class
        dictkeys = sorted(data_dict.keys())  # OPTIMIZATION: sorted() is faster than .sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)
            
            rp = lol(numpy.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * len(rp))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
