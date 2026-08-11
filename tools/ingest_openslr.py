#!/usr/bin/env python3
"""
P0 (2026-07-03 roadmap) — OpenSLR -> sl_celeb corpus layout.

Builds the `<corpus_root>/<lang>/<speaker>/<utt>` tree that
tools/sl_dataprep.py expects, using SYMLINKS into the extracted OpenSLR
archives (no audio is copied; loadWAV reads FLAC and resamples natively).

Supported sources:
  SLR52 (Sinhala, crowdsourced ASR corpus, ~478 speakers):
      --slr52_dir  <dir containing utt_spk_text.tsv and extracted audio>
      Speaker comes from column 2 of utt_spk_text.tsv; audio files are
      located by utterance ID anywhere under the directory.
  SLR65 (Tamil, crowdsourced multi-speaker TTS corpus, 50 speakers):
      --slr65_dir  <dir containing line_index_*.tsv and extracted audio>
      Speaker is embedded in the filename: ta{f,g}_<spkid>_<hash>.wav
      (taf = female, tag = male) -> speaker ID "taf_<spkid>" / "tag_<spkid>".

Outputs under --out_root:
    si/<spk_id>/<utt_id>.flac -> symlink into SLR52 tree
    ta/<spk_id>/<utt_id>.wav  -> symlink into SLR65 tree
    ingest_manifest.csv        lang, spk_id, utt_id, source_path
    spk_meta.csv               spk_id, lang, gender (m/f/unk), n_utts

Options:
    --min_utts N   drop speakers with fewer than N utterances (default 10)
    --max_utts N   cap utterances per speaker, deterministic sample (default 0 = all)
    --seed         controls the per-speaker cap sampling only

Usage:
    python tools/ingest_openslr.py \
        --slr52_dir /mnt/ricproject3/2025/data/sl_corpora/slr52 \
        --slr65_dir /mnt/ricproject3/2025/data/sl_corpora/slr65 \
        --out_root  /mnt/ricproject3/2025/data/sl_celeb \
        --min_utts 10
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def index_audio_files(root, exts=('.flac', '.wav')):
    """Map file stem -> absolute path for every audio file under root."""
    idx = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            stem, ext = os.path.splitext(name)
            if ext.lower() in exts:
                idx[stem] = os.path.join(dirpath, name)
    return idx


def collect_slr52(slr52_dir):
    """Return dict spk_id -> list[(utt_id, abs_path)], gender map (all unk)."""
    tsv = Path(slr52_dir) / 'utt_spk_text.tsv'
    if not tsv.is_file():
        sys.exit(f"[ingest] SLR52: {tsv} not found")
    print(f"[ingest] SLR52: indexing audio under {slr52_dir} ...")
    audio_idx = index_audio_files(slr52_dir)
    print(f"[ingest] SLR52: {len(audio_idx)} audio files on disk")
    by_spk = defaultdict(list)
    n_missing = 0
    with open(tsv) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            utt_id, spk_id = parts[0], parts[1]
            path = audio_idx.get(utt_id)
            if path is None:
                n_missing += 1
                continue
            by_spk[spk_id].append((utt_id, path))
    if n_missing:
        print(f"[ingest] SLR52: {n_missing} TSV rows had no audio on disk "
              f"(fine if only some archives are extracted)")
    return by_spk, {spk: 'unk' for spk in by_spk}


def collect_slr65(slr65_dir):
    """Return dict spk_id -> list[(utt_id, abs_path)], gender map from ta{f,m}."""
    print(f"[ingest] SLR65: indexing audio under {slr65_dir} ...")
    audio_idx = index_audio_files(slr65_dir)
    print(f"[ingest] SLR65: {len(audio_idx)} audio files on disk")
    by_spk = defaultdict(list)
    gender = {}
    for stem, path in audio_idx.items():
        parts = stem.split('_')
        if len(parts) < 3 or parts[0] not in ('taf', 'tag'):
            continue
        spk_id = f"{parts[0]}_{parts[1]}"
        by_spk[spk_id].append((stem, path))
        gender[spk_id] = 'f' if parts[0] == 'taf' else 'm'
    return by_spk, gender


def link_corpus(by_spk, lang, out_root, min_utts, max_utts, rng, manifest_rows):
    lang_dir = Path(out_root) / lang
    kept_spk = 0
    kept_utt = 0
    for spk_id in sorted(by_spk):
        utts = sorted(by_spk[spk_id])
        if len(utts) < min_utts:
            continue
        if max_utts and len(utts) > max_utts:
            utts = sorted(rng.sample(utts, max_utts))
        spk_dir = lang_dir / spk_id
        spk_dir.mkdir(parents=True, exist_ok=True)
        for utt_id, src in utts:
            dst = spk_dir / (utt_id + os.path.splitext(src)[1].lower())
            if not dst.is_symlink() and not dst.exists():
                os.symlink(src, dst)
            manifest_rows.append((lang, spk_id, utt_id, src))
        kept_spk += 1
        kept_utt += len(utts)
    print(f"[ingest] {lang}: kept {kept_spk} speakers / {kept_utt} utterances "
          f"(min_utts={min_utts}, max_utts={max_utts or 'all'})")
    return kept_spk, kept_utt


def main():
    p = argparse.ArgumentParser(description="OpenSLR -> sl_celeb layout (P0).")
    p.add_argument('--slr52_dir', help="Extracted SLR52 dir (Sinhala)")
    p.add_argument('--slr65_dir', help="Extracted SLR65 dir (Tamil)")
    p.add_argument('--out_root', required=True)
    p.add_argument('--min_utts', type=int, default=10)
    p.add_argument('--max_utts', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not args.slr52_dir and not args.slr65_dir:
        sys.exit("[ingest] nothing to do: pass --slr52_dir and/or --slr65_dir")

    rng = random.Random(args.seed)
    manifest_rows = []
    meta_rows = []

    if args.slr52_dir:
        by_spk, gender = collect_slr52(args.slr52_dir)
        link_corpus(by_spk, 'si', args.out_root, args.min_utts, args.max_utts,
                    rng, manifest_rows)
        for spk, utts in sorted(by_spk.items()):
            if len(utts) >= args.min_utts:
                meta_rows.append((spk, 'si', gender[spk], len(utts)))

    if args.slr65_dir:
        by_spk, gender = collect_slr65(args.slr65_dir)
        link_corpus(by_spk, 'ta', args.out_root, args.min_utts, args.max_utts,
                    rng, manifest_rows)
        for spk, utts in sorted(by_spk.items()):
            if len(utts) >= args.min_utts:
                meta_rows.append((spk, 'ta', gender[spk], len(utts)))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'ingest_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lang', 'spk_id', 'utt_id', 'source_path'])
        w.writerows(manifest_rows)
    with open(out_root / 'spk_meta.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['spk_id', 'lang', 'gender', 'n_utts'])
        w.writerows(meta_rows)
    print(f"[ingest] wrote {out_root/'ingest_manifest.csv'} "
          f"({len(manifest_rows)} rows) and spk_meta.csv ({len(meta_rows)} speakers)")


if __name__ == '__main__':
    main()
