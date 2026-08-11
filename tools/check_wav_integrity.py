#!/usr/bin/env python3
"""Scan WAV files for missing/corrupt entries before a training run.

Two input modes (combine freely):

  --train-list / --train-path
      Read a voxceleb_trainer train list (``speaker_id  relpath`` per line) and
      check every referenced file under ``train_path``.

  --glob PATTERN
      Check every file matching the glob (repeatable). Useful for MUSAN
      (``data/musan/*/*/*.wav``) and RIRs (``data/RIRS_NOISES/simulated_rirs/*/*/*.wav``).

A file is considered bad if it is missing, zero-length, or ``soundfile.read``
raises any exception (libsndfile errors, OSError, etc.).

By default the script only reports. Pass ``--delete`` to remove bad files, or
``--write-bad-list PATH`` to dump the offending paths for manual handling.

Example:
  python tools/check_wav_integrity.py \
      --train-list data/voxceleb_new/train_list.txt \
      --train-path data/voxceleb_new/voxceleb2 \
      --glob 'data/musan/*/*/*.wav' \
      --glob 'data/RIRS_NOISES/simulated_rirs/*/*/*.wav' \
      --write-bad-list bad_wavs.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile


def _check_one(path: str) -> tuple[str, str | None]:
    """Return (path, error) where error is None if the file is healthy."""
    try:
        if not os.path.isfile(path):
            return path, "missing"
        if os.path.getsize(path) == 0:
            return path, "zero-byte"
        with soundfile.SoundFile(path) as f:
            if len(f) <= 0:
                return path, "zero-frames"
        return path, None
    except Exception as exc:  # libsndfile errors may be un-stringifiable
        try:
            msg = str(exc)
        except Exception:
            msg = f"{type(exc).__name__} (unprintable)"
        return path, msg or type(exc).__name__


def _collect_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []

    if args.train_list:
        if not args.train_path:
            sys.exit("--train-list requires --train-path")
        with open(args.train_list) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                paths.append(os.path.join(args.train_path, parts[1]))

    for pattern in args.glob or []:
        matched = glob.glob(pattern, recursive=True)
        if not matched:
            print(f"[warn] glob matched nothing: {pattern}", file=sys.stderr)
        paths.extend(matched)

    # De-duplicate while keeping order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-list", help="voxceleb_trainer-style train list file")
    ap.add_argument("--train-path", help="root that train-list relpaths join onto")
    ap.add_argument("--glob", action="append", help="extra glob pattern (repeatable)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--write-bad-list", help="write offending paths to this file")
    ap.add_argument("--delete", action="store_true", help="delete bad files after scan")
    args = ap.parse_args()

    paths = _collect_paths(args)
    if not paths:
        sys.exit("no paths to check — pass --train-list/--train-path or --glob")

    print(f"[info] checking {len(paths)} files with {args.workers} workers")

    bad: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, err in ex.map(_check_one, paths, chunksize=64):
            done += 1
            if err is not None:
                bad.append((path, err))
            if done % 5000 == 0 or done == len(paths):
                print(f"  scanned {done}/{len(paths)}  bad so far: {len(bad)}", flush=True)

    print(f"\n[result] {len(bad)} bad / {len(paths)} total")
    for path, err in bad[:20]:
        print(f"  {err:14s}  {path}")
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")

    if args.write_bad_list:
        with open(args.write_bad_list, "w") as fh:
            for path, err in bad:
                fh.write(f"{path}\t{err}\n")
        print(f"[info] wrote bad list to {args.write_bad_list}")

    if args.delete and bad:
        removed = 0
        for path, _ in bad:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            except OSError as exc:
                print(f"[warn] could not delete {path}: {exc}", file=sys.stderr)
        print(f"[info] deleted {removed} files")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
