#!/usr/bin/env python3
"""
P2 helper — extract embeddings for an arbitrary file list (cohort / PLDA
train utterances) with the same backends and cache format as
tools/zeroshot_eval.py. Run on a GPU node against staged audio.

Usage:
    python extract_files.py --root train_audio \
        --lists p2_cohort.txt p2_plda_list.txt \
        --models speechbrain_ecapa redimnet:b1 redimnet:b6 \
        --cache_dir emb_cache_train --device cuda

Lists may be plain relpaths (one per line) or "<label> <relpath>" rows;
labels are ignored here (they are re-read by the consumer).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_eval import extract_embeddings  # noqa: E402


def read_paths(path):
    out = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if not parts:
                continue
            out.append(parts[-1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--lists', nargs='+', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    files = sorted({f for lst in args.lists for f in read_paths(lst)})
    print(f"[extract] {len(files)} unique files from {len(args.lists)} lists")
    for spec in args.models:
        extract_embeddings(spec, files, args.root, args.cache_dir, args.device)


if __name__ == '__main__':
    main()
