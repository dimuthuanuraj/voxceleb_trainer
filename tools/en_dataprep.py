#!/usr/bin/env python3
"""
English VoxCeleb data-prep helper.

Companion to `tools/sl_dataprep.py`. The full VoxCeleb2 corpus already
ships with `train_list.txt` (speaker_id + relpath, one per line) and a
trial file (`veri_test2.txt` / `test_list.txt`), so we don't need to
re-walk the tree. What we *do* need to materialise are the two list
files consumed by the universal feature plumbing:

  * `asnorm_cohort.txt`   FEATURE-002 score-normalisation cohort
                          format: one absolute-or-relative wav path per line
  * `plda_train_list.txt` FEATURE-010 back-end training pool
                          format: <speaker_id> <relpath>   (mirrors train_list)

Both are sampled deterministically from the existing train_list.

Usage (run from voxceleb_trainer/ — `data/voxceleb_new` is a symlink
to the real corpus root, see SETUP.md §7):

    python tools/en_dataprep.py \\
        --vox_root      data/voxceleb_new \\
        --train_list    data/voxceleb_new/train_list.txt \\
        --out_dir       data/voxceleb_new/lists \\
        --cohort_size   5000 \\
        --plda_speakers 1000 \\
        --seed          42
"""

import argparse
import os
import random
import sys
from collections import defaultdict


def read_train_list(path):
    """Yields (speaker_id, relpath) tuples, skipping malformed lines."""
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"[en_dataprep] skipping malformed line {ln}: {line!r}",
                      file=sys.stderr)
                continue
            yield parts[0], parts[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vox_root", required=True,
                   help="Root containing voxceleb2/ (passed through to "
                        "the trainer; only used for sanity checks here).")
    p.add_argument("--train_list", required=True,
                   help="Existing VoxCeleb2 train_list.txt "
                        "(<speaker_id> <relpath_under_voxceleb2>).")
    p.add_argument("--out_dir", required=True,
                   help="Where asnorm_cohort.txt / plda_train_list.txt "
                        "will be written.")
    p.add_argument("--cohort_size", type=int, default=5000,
                   help="Number of utterances for the AS-Norm cohort. "
                        "300-5000 is typical; bigger = slower but more "
                        "stable normalisation stats.")
    p.add_argument("--plda_speakers", type=int, default=0,
                   help="If >0, sample this many speakers for the PLDA "
                        "training pool (all their utterances are used). "
                        "0 = use every speaker.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.isdir(args.vox_root):
        print(f"[en_dataprep] ERROR: vox_root not found: {args.vox_root}",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.train_list):
        print(f"[en_dataprep] ERROR: train_list not found: {args.train_list}",
              file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)

    by_spk = defaultdict(list)
    for spk, path in read_train_list(args.train_list):
        by_spk[spk].append(path)

    total_utts = sum(len(v) for v in by_spk.values())
    n_speakers = len(by_spk)
    print(f"[en_dataprep] train_list: {n_speakers} speakers, "
          f"{total_utts} utterances")

    # ---- AS-Norm cohort ------------------------------------------------
    all_paths = [p for paths in by_spk.values() for p in paths]
    if args.cohort_size >= len(all_paths):
        cohort = list(all_paths)
        print(f"[en_dataprep] cohort_size ({args.cohort_size}) >= total "
              f"utterances; using all {len(cohort)}.")
    else:
        cohort = rng.sample(all_paths, args.cohort_size)
    cohort_path = os.path.join(args.out_dir, "asnorm_cohort.txt")
    with open(cohort_path, "w") as f:
        f.write("\n".join(sorted(cohort)) + "\n")
    print(f"[en_dataprep] asnorm_cohort.txt: {len(cohort)} utterances")

    # ---- PLDA training list -------------------------------------------
    if args.plda_speakers and args.plda_speakers < n_speakers:
        plda_spks = set(rng.sample(list(by_spk.keys()), args.plda_speakers))
    else:
        plda_spks = set(by_spk.keys())
    plda_rows = []
    for spk in sorted(plda_spks):
        for path in by_spk[spk]:
            plda_rows.append(f"{spk} {path}")
    plda_path = os.path.join(args.out_dir, "plda_train_list.txt")
    with open(plda_path, "w") as f:
        f.write("\n".join(plda_rows) + "\n")
    print(f"[en_dataprep] plda_train_list.txt: {len(plda_rows)} utterances "
          f"from {len(plda_spks)} speakers")

    print(f"[en_dataprep] done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
