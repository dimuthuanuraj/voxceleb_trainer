#!/usr/bin/env python3
"""
P2×P3 composition helper — extract embeddings with a fine-tuned P3
checkpoint (tools/peft_finetune.py `best.pt`) in the same .npz cache format
as tools/zeroshot_eval.py, so tools/backend_adapt.py can run its AS-Norm /
PLDA / calibration ablation on the fine-tuned model.

Embeddings use the same protocol as the P3 final eval (up to 8 s per file,
L2-normalized), so the `cosine` condition in backend_adapt reproduces the
P3 final numbers.

Usage (on a GPU node, staged layout):
    python extract_p3.py --ckpt runs/p3_full_s42/best.pt \
        --backbone models/wavlm-base-plus \
        --spec p3_full_s42 \
        --root audio       --lists test_list_si.txt test_list_ta.txt \
        --cache_dir emb_cache --device cuda
    python extract_p3.py --ckpt runs/p3_full_s42/best.pt \
        --backbone models/wavlm-base-plus \
        --spec p3_full_s42 \
        --root train_audio --lists p2_cohort.txt p2_plda_list.txt \
        --cache_dir emb_cache_train --device cuda
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peft_finetune import SSLSpeakerNet, load_audio, TARGET_SR  # noqa: E402


def read_paths(path):
    """Accept plain-path lines, '<label> <path>' rows, and 3-column
    '<label> <a> <b>' trial rows (both sides taken)."""
    out = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) == 3:
                out.extend(parts[1:])
            elif parts:
                out.append(parts[-1])
    return out


def build_model(ckpt_path, backbone, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    train_args = ckpt.get('args', {})
    mode = train_args.get('mode', 'full')
    # n_classes from the head weight shape.
    n_classes = ckpt['state_dict']['head.w'].shape[0]
    model = SSLSpeakerNet(backbone, n_classes)
    if mode == 'lora':
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=train_args.get('lora_r', 8),
                         lora_alpha=train_args.get('lora_alpha', 16),
                         lora_dropout=0.05,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        model.encoder = get_peft_model(model.encoder, cfg)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing or unexpected:
        sys.exit(f"[extract_p3] state_dict mismatch: missing={missing[:3]} "
                 f"unexpected={unexpected[:3]} — wrong mode/backbone?")
    print(f"[extract_p3] loaded {ckpt_path} (mode={mode}, epoch "
          f"{ckpt.get('epoch')}, {n_classes} classes)")
    return model.eval().to(device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--spec', required=True,
                   help="cache name, e.g. p3_full_s42")
    p.add_argument('--root', required=True)
    p.add_argument('--lists', nargs='+', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--max_sec', type=float, default=8.0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    files = sorted({f for lst in args.lists for f in read_paths(lst)})
    cpath = Path(args.cache_dir) / f"{args.spec}.npz"
    cache = {}
    if cpath.is_file():
        z = np.load(cpath, allow_pickle=False)
        cache = {k: z[k] for k in z.files}
        print(f"[extract_p3] {len(cache)} cached")
    todo = [f for f in files if f not in cache]
    print(f"[extract_p3] {len(files)} files, {len(todo)} to extract")
    if todo:
        model = build_model(args.ckpt, args.backbone, args.device)
        cap = int(args.max_sec * TARGET_SR)
        t0 = time.time()
        for i, rel in enumerate(todo):
            wav = load_audio(os.path.join(args.root, rel))[:cap]
            if len(wav) < TARGET_SR // 2:
                wav = np.concatenate(
                    [wav, np.zeros(TARGET_SR // 2 - len(wav), dtype='float32')])
            t = torch.from_numpy(wav).unsqueeze(0).to(args.device)
            e = model.embed(t).squeeze(0)
            cache[rel] = F.normalize(e, dim=0).cpu().numpy().astype('float32')
            if (i + 1) % 500 == 0:
                print(f"[extract_p3] {i+1}/{len(todo)} "
                      f"({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, **cache)
    print(f"[extract_p3] cached {len(cache)} embeddings -> {cpath}")


if __name__ == '__main__':
    main()
