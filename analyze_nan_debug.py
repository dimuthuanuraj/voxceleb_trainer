#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
analyze_nan_debug.py

Diagnostic tool for investigating NaN / Inf incidents in this repository's
speaker-verification training runs. Operates on the two artefacts a failing
run actually leaves behind:

  1. A checkpoint file written by SpeakerNet.saveParameters (a bare
     state_dict via `torch.save(model.state_dict(), path)`).
  2. A training log file (whatever the trainer was redirected to —
     stdout, logs/*.log, etc.) that records per-epoch loss / TEER.

For each artefact provided the script reports the information that is
useful for triaging the failure mode documented in BUGFIX-010 (gradient-path
explosion / feature-anti-correlation in NestedSpeakerNet) and the more
generic NaN modes in NaN_DEBUGGING_GUIDE.md.

Usage:
    python analyze_nan_debug.py --checkpoint exps/<exp>/model/model000010.model
    python analyze_nan_debug.py --log logs/nested_fixed_20251229.log
    python analyze_nan_debug.py --checkpoint <path> --log <path> --verbose

Exit codes:
    0  no NaN / Inf observed
    1  NaN / Inf detected in checkpoint or log
    2  invalid arguments / file not found
"""

import argparse
import math
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------
# Checkpoint analysis
# ---------------------------------------------------------------------

def _summarise_tensor(t: torch.Tensor) -> dict:
    """Per-parameter statistics. NaN / Inf are computed on the full tensor;
    mean / std / abs_max ignore non-finite values so a partially-corrupted
    parameter still gives a usable summary."""
    flat = t.detach().to(torch.float64).reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    nan_count = int(torch.isnan(flat).sum().item())
    pos_inf = int((flat == math.inf).sum().item())
    neg_inf = int((flat == -math.inf).sum().item())
    return {
        "shape": tuple(t.shape),
        "numel": flat.numel(),
        "nan": nan_count,
        "pos_inf": pos_inf,
        "neg_inf": neg_inf,
        "finite_frac": float(finite_mask.sum().item()) / max(1, flat.numel()),
        "mean": float(finite.mean().item()) if finite.numel() else float("nan"),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else float("nan"),
        "abs_max": float(finite.abs().max().item()) if finite.numel() else float("nan"),
    }


def analyse_checkpoint(path: str, verbose: bool = False) -> Tuple[int, int]:
    """Return (num_params_with_nan, num_params_with_inf), or (-1, -1) on error."""
    if not os.path.isfile(path):
        print(f"[error] checkpoint not found: {path}", file=sys.stderr)
        return -1, -1

    # The repo's saved files are bare state_dicts. We pass map_location='cpu'
    # so this works on machines without CUDA — the diagnostic does not need
    # to materialise on GPU.
    #
    # BUGFIX-019: weights_only=True is the right default for a diagnostic
    # that reads externally-supplied checkpoints. If the user has a legacy
    # checkpoint with non-tensor objects (e.g., a serialized custom class),
    # the load will fail with a clear error and they can re-save it as a
    # state_dict — preferable to silently exposing this tool to the
    # standard pickle-based RCE vector.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[error] failed to load {path}: {exc}", file=sys.stderr)
        return -1, -1

    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        state_dict = obj
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    else:
        print(
            f"[error] {path} does not look like a state_dict "
            f"(top-level type={type(obj).__name__})",
            file=sys.stderr,
        )
        return -1, -1

    print(f"=== checkpoint: {path} ===")
    print(f"parameter tensors: {len(state_dict)}")
    print()

    nan_params: List[Tuple[str, dict]] = []
    inf_params: List[Tuple[str, dict]] = []
    rows: List[Tuple[str, dict]] = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        s = _summarise_tensor(tensor)
        rows.append((name, s))
        if s["nan"] > 0:
            nan_params.append((name, s))
        if s["pos_inf"] + s["neg_inf"] > 0:
            inf_params.append((name, s))

    def _fmt_row(name: str, s: dict) -> str:
        return (
            f"  {name:<60} "
            f"shape={str(s['shape']):<20} "
            f"nan={s['nan']:<6} inf={s['pos_inf']+s['neg_inf']:<6} "
            f"mean={s['mean']:+.3e} std={s['std']:.3e} "
            f"|max|={s['abs_max']:.3e}"
        )

    if nan_params or inf_params:
        print("--- non-finite parameters ---")
        for name, s in nan_params + inf_params:
            print(_fmt_row(name, s))
        print()

    if verbose:
        print("--- all parameters ---")
        for name, s in rows:
            print(_fmt_row(name, s))
        print()

    print(f"summary: {len(nan_params)} params with NaN, {len(inf_params)} params with Inf")
    return len(nan_params), len(inf_params)


# ---------------------------------------------------------------------
# Log analysis
# ---------------------------------------------------------------------

# Trainer log lines look like (from trainSpeakerNet.py and its variants):
#   "IT 1234, LR 0.00050000, TLOSS 4.231, TEER/TAcc 12.3"
#   "Epoch 11, VEER 21.72"
# We match conservatively — looking for explicit "nan" / "inf" tokens and
# pulling out the surrounding epoch / iteration for context.
_LOSS_PATTERN = re.compile(r"(?i)\b(t?loss|nloss|loss)\s*[:=]?\s*([-+]?(?:inf|nan|\d+\.?\d*))")
_EPOCH_PATTERN = re.compile(r"(?i)\b(?:epoch|it)\s*[:=]?\s*(\d+)")


def analyse_log(path: str, verbose: bool = False) -> int:
    """Return number of NaN / Inf loss occurrences found in the log.
    Returns -1 if the file cannot be read."""
    if not os.path.isfile(path):
        print(f"[error] log not found: {path}", file=sys.stderr)
        return -1

    nan_hits: List[Tuple[int, str]] = []
    first_nan_epoch: Optional[str] = None
    last_clean_epoch: Optional[str] = None
    last_clean_loss: Optional[float] = None

    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            loss_match = _LOSS_PATTERN.search(line)
            if not loss_match:
                continue
            value = loss_match.group(2).lower()
            if value in ("nan", "+nan", "-nan", "inf", "+inf", "-inf"):
                nan_hits.append((lineno, line.rstrip()))
                if first_nan_epoch is None:
                    epoch_match = _EPOCH_PATTERN.search(line)
                    if epoch_match:
                        first_nan_epoch = epoch_match.group(1)
            else:
                try:
                    loss_val = float(value)
                except ValueError:
                    continue
                if math.isfinite(loss_val):
                    epoch_match = _EPOCH_PATTERN.search(line)
                    if epoch_match:
                        last_clean_epoch = epoch_match.group(1)
                    last_clean_loss = loss_val

    print(f"=== log: {path} ===")
    if last_clean_epoch is not None and last_clean_loss is not None:
        print(
            f"last clean step: epoch/it {last_clean_epoch}, "
            f"loss {last_clean_loss:.4f}"
        )
    if first_nan_epoch is not None:
        print(f"first NaN / Inf observed at epoch/it {first_nan_epoch}")
    print(f"total NaN / Inf hits in log: {len(nan_hits)}")

    if nan_hits and verbose:
        print()
        print("--- offending log lines (first 10) ---")
        for lineno, raw in nan_hits[:10]:
            print(f"  line {lineno}: {raw}")
        if len(nan_hits) > 10:
            print(f"  ... and {len(nan_hits) - 10} more")
    return len(nan_hits)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved state_dict (.model / .pt / .pth).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to a training log file to scan for NaN / Inf loss reports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-parameter table and full log-hit listing.",
    )
    args = parser.parse_args(argv)

    if args.checkpoint is None and args.log is None:
        parser.print_help(sys.stderr)
        print("\n[error] supply --checkpoint and/or --log", file=sys.stderr)
        return 2

    failed = False

    if args.checkpoint is not None:
        nan_n, inf_n = analyse_checkpoint(args.checkpoint, verbose=args.verbose)
        if nan_n < 0:
            return 2
        if nan_n > 0 or inf_n > 0:
            failed = True
        print()

    if args.log is not None:
        log_n = analyse_log(args.log, verbose=args.verbose)
        if log_n < 0:
            return 2
        if log_n > 0:
            failed = True
        print()

    if failed:
        print("[result] NaN / Inf detected. See NaN_DEBUGGING_GUIDE.md for triage steps.")
        return 1
    print("[result] no NaN / Inf detected in the supplied artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
