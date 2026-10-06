#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Materialise a HuggingFace audio encoder as a local safetensors snapshot.

    python experiments/tools/fetch_ssl_encoder.py utter-project/mHuBERT-147
    python experiments/tools/fetch_ssl_encoder.py facebook/wav2vec2-xls-r-300m
    python experiments/tools/fetch_ssl_encoder.py --list

Why this is needed
------------------
The `SL_SPV` conda env on the GPU nodes runs **torch 2.5.1**, and
transformers >= 4.56 refuses to `torch.load` a `pytorch_model.bin` on any torch
below 2.6 (CVE-2025-32434):

    ValueError: Due to a serious vulnerability issue in `torch.load` ... we now
    require users to upgrade torch to at least v2.6 ... This version restriction
    does not apply when loading files with safetensors.

Some encoders publish `model.safetensors` (mHuBERT-147) and some do not
(wav2vec2-xls-r-300m, and the cached revision of wavlm-base-plus). For the
latter this script performs the conversion **on the head node**, which runs
torch 2.8 and is therefore allowed to read the `.bin`, and writes a safetensors
snapshot the GPU nodes can load.

Upgrading torch inside `SL_SPV` was rejected as the fix: it would put these runs
on a different numerical stack from every previously published result in this
repository.

Output: `models/weights/<short-name>/` containing `config.json`,
`model.safetensors`, `preprocessor_config.json` (when published) and a README
recording provenance. Point the trainer at it with
`--ssl_encoder_name models/weights/<short-name>`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "models", "weights")

# The encoders this study can use, and what each is for.
KNOWN = {
    "microsoft/wavlm-base-plus": (
        "wavlm-base-plus",
        "94M. English-only pretraining (LibriLight/GigaSpeech/VoxPopuli-en). "
        "The control for the pretraining-coverage question, and the encoder the "
        "VoiceID product already deploys.",
    ),
    "utter-project/mHuBERT-147": (
        "mhubert-147",
        "95M. 147 languages including Sinhala and Tamil. Same parameter scale as "
        "WavLM-base-plus, so coverage is isolated from capacity.",
    ),
    "facebook/wav2vec2-xls-r-300m": (
        "xls-r-300m",
        "300M. 128 languages including Sinhala and Tamil. Larger, so it confounds "
        "coverage with capacity -- read it alongside mHuBERT-147, not instead.",
    ),
}


def convert(model_id: str, dest_name: str | None = None, force: bool = False):
    from huggingface_hub import snapshot_download

    short = dest_name or KNOWN.get(model_id, (model_id.split("/")[-1].lower(), ""))[0]
    dest = os.path.join(WEIGHTS_DIR, short)
    if os.path.isdir(dest) and not force:
        print(f"[{model_id}] already present at {os.path.relpath(dest, REPO_ROOT)} "
              f"(--force to rebuild)")
        return dest

    print(f"[{model_id}] downloading ...")
    src = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
    )

    os.makedirs(dest, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json"):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy(os.path.realpath(p), os.path.join(dest, name))

    st = os.path.join(src, "model.safetensors")
    if os.path.isfile(st):
        shutil.copy(os.path.realpath(st), os.path.join(dest, "model.safetensors"))
        route = "copied published model.safetensors"
        print(f"[{model_id}] published safetensors copied")
    else:
        import torch
        from safetensors.torch import save_file

        if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
            raise SystemExit(
                f"{model_id} publishes no safetensors, and converting its .bin "
                f"requires torch >= 2.6 (this interpreter has {torch.__version__}). "
                f"Run this script on the head node, which has a newer torch."
            )
        print(f"[{model_id}] no published safetensors -- converting the .bin "
              f"(torch {torch.__version__})")
        state = torch.load(os.path.join(src, "pytorch_model.bin"),
                           map_location="cpu", weights_only=True)
        # safetensors rejects shared storage, which HF checkpoints often carry.
        state = {k: v.clone().contiguous() for k, v in state.items()
                 if hasattr(v, "clone")}
        save_file(state, os.path.join(dest, "model.safetensors"),
                  metadata={"format": "pt"})
        route = f"converted from pytorch_model.bin with torch {torch.__version__}"

    note = KNOWN.get(model_id, ("", "no description recorded"))[1]
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            f"# {short} — local safetensors snapshot\n\n"
            f"Source: `{model_id}`\n\n"
            f"{note}\n\n"
            f"Assembled {datetime.now(timezone.utc).date()} by "
            f"`experiments/tools/fetch_ssl_encoder.py` ({route}).\n\n"
            f"Exists because the `SL_SPV` env runs torch 2.5.1 and transformers "
            f"refuses to `torch.load` a `.bin` below torch 2.6 (CVE-2025-32434); "
            f"the restriction does not apply to safetensors.\n\n"
            f"Use with `--ssl_encoder_name models/weights/{short}`.\n"
        )

    print(f"[{model_id}] -> {os.path.relpath(dest, REPO_ROOT)}")
    return dest


def verify(dest: str):
    """Load it once, so a broken snapshot fails here and not 3 hours into a run."""
    try:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(dest)
        n = sum(p.numel() for p in m.parameters())
        print(f"    verified: {n:,} parameters, hidden_size={m.config.hidden_size}, "
              f"layers={m.config.num_hidden_layers}")
        return True
    except Exception as exc:
        print(f"    VERIFY FAILED: {exc!r}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_id", nargs="?", help="HuggingFace model id")
    ap.add_argument("--all", action="store_true", help="fetch every known encoder")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--name", default=None, help="override the local directory name")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("known encoders:\n")
        for mid, (short, note) in KNOWN.items():
            present = os.path.isdir(os.path.join(WEIGHTS_DIR, short))
            print(f"  [{'x' if present else ' '}] {mid}")
            print(f"      -> models/weights/{short}")
            print(f"      {note}\n")
        return 0

    targets = list(KNOWN) if args.all else ([args.model_id] if args.model_id else [])
    if not targets:
        ap.error("pass a model id, --all, or --list")

    failed = []
    for mid in targets:
        try:
            dest = convert(mid, args.name, args.force)
            if not args.no_verify and not verify(dest):
                failed.append(mid)
        except SystemExit as exc:
            print(f"[{mid}] {exc}")
            failed.append(mid)
        except Exception as exc:
            print(f"[{mid}] FAILED: {exc!r}")
            failed.append(mid)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
