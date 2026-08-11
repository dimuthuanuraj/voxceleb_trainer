#!/usr/bin/env python3
"""Extract speaker embeddings for the Layer-3 label audit and Layer-5 power analysis.

Layer 3 needs an embedding per utterance over a controlled subsample of the whole
corpus (not just the trial files), so this is separate from tools/zeroshot_eval.py,
which is organised around trial lists.

Sampling: up to --per-speaker utterances per speaker, then a global cap of
--max-utts, both seeded. Every speaker is kept -- the cap trims utterances, never
speakers, because speaker count is what the audit is about.

Output (one .npz per dataset under --out):
    emb        float32 [N, D]  L2-normalised
    path       str    [N]      relative to <dataset>/wav
    spk, sess, lang, gender, dur

Run it on a GPU node:
    tools/gpurun.sh -- python tools/extract_embeddings.py --all
    # or directly
    ssh 10.222.1.121 '... python tools/extract_embeddings.py --dataset data/slr52_sinhala'
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
LOCAL_ECAPA = "/mnt/ricproject3/2026/SLT_Zoom_Project/pretrained_models/spkrec-ecapa"
TARGET_SR = 16000
MAX_SECONDS = 8.0          # centre-crop long files; SV embeddings saturate well before this


def load_model(device):
    """SpeechBrain ECAPA-TDNN (VoxCeleb). Prefers the on-disk copy over the hub."""
    from speechbrain.inference.speaker import EncoderClassifier
    src = LOCAL_ECAPA if os.path.isdir(LOCAL_ECAPA) else "speechbrain/spkrec-ecapa-voxceleb"
    print(f"[emb] loading {src} on {device}", flush=True)
    return EncoderClassifier.from_hparams(
        source=src,
        savedir=os.path.join("/tmp", f"sb_ecapa_{os.getuid()}"),
        run_opts={"device": str(device)},
    )


def read_rows(ds):
    with open(os.path.join(ds, "metadata", "utterances.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def subsample(rows, per_speaker, max_utts, seed):
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[(r["lang"], r["spk_id"])].append(r)

    kept = []
    for k in sorted(by_spk):
        items = by_spk[k]
        if len(items) > per_speaker:
            items = rng.sample(items, per_speaker)
        kept.extend(items)

    if max_utts and len(kept) > max_utts:
        # trim proportionally but never drop a speaker entirely
        by_spk2 = defaultdict(list)
        for r in kept:
            by_spk2[(r["lang"], r["spk_id"])].append(r)
        n_spk = len(by_spk2)
        quota = max(2, max_utts // n_spk)
        kept = []
        for k in sorted(by_spk2):
            items = by_spk2[k]
            kept.extend(items if len(items) <= quota else rng.sample(items, quota))
    kept.sort(key=lambda r: r["path"])
    return kept


def load_batch(paths, ds):
    waves, lens, ok = [], [], []
    for p in paths:
        try:
            a, sr = sf.read(os.path.join(ds, "wav", p), dtype="float32")
        except Exception:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        n_max = int(MAX_SECONDS * TARGET_SR)
        if len(a) > n_max:                       # centre crop
            s = (len(a) - n_max) // 2
            a = a[s:s + n_max]
        if len(a) < TARGET_SR // 2:              # pad very short clips to 0.5 s
            a = np.pad(a, (0, TARGET_SR // 2 - len(a)))
        waves.append(torch.from_numpy(a))
        lens.append(len(a))
        ok.append(p)
    if not waves:
        return None, None, []
    n = max(lens)
    batch = torch.zeros(len(waves), n)
    for i, w in enumerate(waves):
        batch[i, :len(w)] = w
    rel = torch.tensor([l / n for l in lens], dtype=torch.float32)
    return batch, rel, ok


def run(ds, model, device, args):
    name = os.path.basename(ds.rstrip("/"))
    out_path = os.path.join(args.out, f"{name}.npz")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[emb] {name}: exists, skipping (use --overwrite)", flush=True)
        return

    rows = read_rows(ds)
    kept = subsample(rows, args.per_speaker, args.max_utts, args.seed)
    n_spk = len({(r['lang'], r['spk_id']) for r in kept})
    print(f"[emb] {name}: {len(kept)}/{len(rows)} utts over {n_spk} speakers",
          flush=True)

    embs, meta = [], []
    t0 = time.time()
    for i in range(0, len(kept), args.batch):
        chunk = kept[i:i + args.batch]
        batch, rel, ok = load_batch([c["path"] for c in chunk], ds)
        if batch is None:
            continue
        with torch.no_grad():
            e = model.encode_batch(batch.to(device), rel.to(device))
        e = e.squeeze(1).float().cpu().numpy()
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        embs.append(e)
        keep = {p: c for p, c in zip([c["path"] for c in chunk], chunk)}
        meta.extend(keep[p] for p in ok)
        if (i // args.batch) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  [{name}] {done}/{len(kept)}  {rate:.0f} utt/s", flush=True)

    emb = np.concatenate(embs).astype(np.float32)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        out_path,
        emb=emb,
        path=np.array([m["path"] for m in meta]),
        spk=np.array([m["spk_id"] for m in meta]),
        sess=np.array([m["session_id"] for m in meta]),
        lang=np.array([m["lang"] for m in meta]),
        gender=np.array([m["gender"] for m in meta]),
        dur=np.array([float(m["duration_s"]) for m in meta], dtype=np.float32),
    )
    print(f"[emb] {name}: wrote {out_path}  emb={emb.shape}  "
          f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA, "_qc", "embeddings"))
    ap.add_argument("--per-speaker", type=int, default=60)
    ap.add_argument("--max-utts", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    targets = args.dataset
    if args.all or not targets:
        targets = [os.path.join(DATA, d) for d in sorted(os.listdir(DATA))
                   if os.path.isfile(os.path.join(DATA, d, "metadata", "utterances.csv"))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[emb] WARNING: no CUDA -- this will be slow. Use tools/gpurun.sh.",
              file=sys.stderr)
    model = load_model(device)
    os.makedirs(args.out, exist_ok=True)
    for ds in targets:
        run(ds if os.path.isabs(ds) else os.path.join(REPO, ds), model, device, args)


if __name__ == "__main__":
    main()
