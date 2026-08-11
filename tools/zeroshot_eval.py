#!/usr/bin/env python3
"""
P1 (2026-07-03 roadmap) — zero-shot evaluation of released pretrained
speaker-verification checkpoints on the SL benchmark trial lists.

No training. For each (model, trial list): extract one full-utterance
embedding per unique file, cosine-score the trials, report EER_avg,
EER_max, and minDCF at p_target 0.01 and 0.05.

Backends (imported lazily; install only what you evaluate):
    speechbrain_ecapa   SpeechBrain spkrec-ecapa-voxceleb   (pip install speechbrain)
    redimnet:<size>     IDRnD ReDimNet via torch.hub, e.g. redimnet:b1,
                        redimnet:b2, redimnet:b6 (weights: ft_lm / vox2)
    wespeaker:<lang>    WeSpeaker pretrained (pip install git+wespeaker), e.g.
                        wespeaker:english (ResNet34/ECAPA per install)

Embeddings are cached per model under --cache_dir so multiple trial lists
reuse the same extraction pass.

Usage:
    python tools/zeroshot_eval.py \
        --corpus_root /mnt/ricproject3/2025/data/sl_celeb \
        --trials lists/test_list_si.txt lists/test_list_ta.txt lists/test_list_cs.txt \
        --models speechbrain_ecapa redimnet:b1 redimnet:b2 \
        --cache_dir /mnt/ricproject3/2025/data/sl_celeb/emb_cache \
        --out results/zeroshot_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the repo's metric implementations (BUGFIX-002/020 semantics).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf  # noqa: E402

TARGET_SR = 16000


def load_audio(path, target_sr=TARGET_SR):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from math import gcd
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype('float32')
    return audio


class EmbeddingBackend:
    """Lazy wrapper: name -> callable(np.float32 waveform) -> np.ndarray."""

    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        self._fn = None

    def _build(self):
        import torch
        spec = self.spec
        if spec == 'speechbrain_ecapa':
            from speechbrain.inference.speaker import EncoderClassifier
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device})
            def fn(wav):
                t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = model.encode_batch(t)
                return emb.squeeze().cpu().numpy()
            return fn
        if spec.startswith('redimnet:'):
            size = spec.split(':', 1)[1]
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet',
                                   model_name=size, train_type='ft_lm',
                                   dataset='vox2')
            model.eval().to(self.device)
            def fn(wav):
                t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = model(t)
                return emb.squeeze().cpu().numpy()
            return fn
        if spec.startswith('wespeaker:'):
            lang = spec.split(':', 1)[1]
            import wespeaker
            model = wespeaker.load_model(lang)
            if self.device.startswith('cuda'):
                model.set_device(self.device)
            def fn(wav):
                # wespeaker's extract_embedding takes a path; use its
                # internal API on a tensor instead.
                import torch as _t
                pcm = _t.from_numpy(wav).unsqueeze(0)
                feats = model.compute_fbank(pcm, sample_rate=TARGET_SR,
                                            cmn=True)
                feats = feats.unsqueeze(0).to(model.device)
                with _t.no_grad():
                    outputs = model.model(feats)
                    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
                return outputs.squeeze().cpu().numpy()
            return fn
        raise ValueError(f"unknown model spec: {spec}")

    def __call__(self, wav):
        if self._fn is None:
            self._fn = self._build()
        return self._fn(wav)


def cache_path(cache_dir, spec):
    safe = spec.replace(':', '_').replace('/', '_')
    return Path(cache_dir) / f"{safe}.npz"


def extract_embeddings(spec, files, corpus_root, cache_dir, device):
    """Return dict relpath -> L2-normalized embedding."""
    cpath = cache_path(cache_dir, spec)
    cache = {}
    if cpath.is_file():
        z = np.load(cpath, allow_pickle=False)
        cache = {k: z[k] for k in z.files}
        print(f"[zeroshot] {spec}: loaded {len(cache)} cached embeddings")
    todo = [f for f in files if f not in cache]
    if todo:
        backend = EmbeddingBackend(spec, device)
        t0 = time.time()
        for i, rel in enumerate(todo):
            wav = load_audio(os.path.join(corpus_root, rel))
            emb = backend(wav).astype('float32').reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            cache[rel] = emb
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[zeroshot] {spec}: {i+1}/{len(todo)} "
                      f"({rate:.1f} files/s)", flush=True)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, **cache)
        print(f"[zeroshot] {spec}: extracted {len(todo)} new, cached to {cpath}")
    return cache


def read_trials(path):
    trials = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) != 3:
                sys.exit(f"[zeroshot] malformed trial line in {path}: {ln!r} "
                         f"(unlabeled trials are not accepted — issue E4)")
            trials.append((int(parts[0]), parts[1], parts[2]))
    return trials


def evaluate(embs, trials):
    scores, labels = [], []
    for lab, a, b in trials:
        scores.append(float(np.dot(embs[a], embs[b])))
        labels.append(lab)
    result = tuneThresholdfromScore(scores, labels, [1, 0.1])
    eer_max, eer_avg = result[1], result[5]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    mindcf_01, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
    mindcf_05, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    return {
        'eer_avg': round(eer_avg, 4), 'eer_max': round(eer_max, 4),
        'mindcf_p01': round(mindcf_01, 4), 'mindcf_p05': round(mindcf_05, 4),
        'n_trials': len(trials),
        'n_target': sum(labels), 'n_impostor': len(labels) - sum(labels),
    }


def main():
    p = argparse.ArgumentParser(description="Zero-shot SV baseline table (P1).")
    p.add_argument('--corpus_root', required=True)
    p.add_argument('--trials', nargs='+', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None,
                   help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()

    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[zeroshot] device: {args.device}")

    trial_sets = {t: read_trials(t) for t in args.trials}
    all_files = sorted({f for trials in trial_sets.values()
                        for _l, a, b in trials for f in (a, b)})
    print(f"[zeroshot] {len(all_files)} unique files across "
          f"{len(trial_sets)} trial lists")

    results = {}
    for spec in args.models:
        embs = extract_embeddings(spec, all_files, args.corpus_root,
                                  args.cache_dir, args.device)
        results[spec] = {}
        for tpath, trials in trial_sets.items():
            r = evaluate(embs, trials)
            results[spec][os.path.basename(tpath)] = r
            print(f"[zeroshot] {spec} | {os.path.basename(tpath)}: "
                  f"EER_avg {r['eer_avg']}%  minDCF(.01) {r['mindcf_p01']}  "
                  f"minDCF(.05) {r['mindcf_p05']}  ({r['n_trials']} trials)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'device': args.device, 'corpus_root': args.corpus_root,
                   'results': results}, f, indent=2)
    # Markdown table alongside the JSON.
    md = out.with_suffix('.md')
    with open(md, 'w') as f:
        f.write("| Model | Trial list | EER_avg % | minDCF p=0.01 | minDCF p=0.05 | trials |\n")
        f.write("|---|---|---|---|---|---|\n")
        for spec, per_list in results.items():
            for tname, r in per_list.items():
                f.write(f"| {spec} | {tname} | {r['eer_avg']} | "
                        f"{r['mindcf_p01']} | {r['mindcf_p05']} | {r['n_trials']} |\n")
    print(f"[zeroshot] wrote {out} and {md}")


if __name__ == '__main__':
    main()
