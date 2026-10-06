#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structured recording of a single experiment.

Every experiment writes exactly this tree::

    experiments/results/<exp_id>/
      manifest.json      everything known BEFORE the first epoch:
                         resolved config, model card, loss spec, architecture
                         spec, data fingerprints, environment, hardware, git
      epochs.jsonl       one JSON object per epoch, appended live
      config.yaml        the exact YAML handed to trainSpeakerNet.py
      command.txt        the exact argv, re-runnable verbatim
      script.py          a copy of the experiment script that produced this run
      stdout.log         complete trainer output
      final.json         summary written at exit: best epoch, curves, timings,
                         exit status
      test_eval.json     held-out test-set evaluation at the best checkpoint
                         (written by the separate evaluation pass)

The design rule is that ``manifest.json`` + ``command.txt`` are sufficient to
reproduce the run, and ``epochs.jsonl`` + ``final.json`` are sufficient to
analyse it without ever re-reading the trainer's own logs.

Deliberately: nothing here imports torch at module scope, so the recorder can be
used on a CPU-only head node to inspect or repair results produced elsewhere.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "experiments", "results")


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------
def _run(cmd, cwd=None):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        return None


def capture_environment() -> dict:
    """Snapshot everything that could change a number between two runs."""
    env = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    env["git"] = {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT),
        # A dirty tree means the recorded commit does not fully describe the
        # code that ran; the analysis flags any such run rather than silently
        # comparing it against clean ones.
        "dirty": bool(_run(["git", "status", "--porcelain"], cwd=REPO_ROOT)),
    }

    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                    ),
                    "capability": ".".join(
                        map(str, torch.cuda.get_device_capability(i))
                    ),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        env["torch"] = {"error": repr(exc)}

    for mod in ("numpy", "scipy", "soundfile", "torchaudio", "transformers", "sklearn"):
        try:
            m = __import__(mod)
            env.setdefault("packages", {})[mod] = getattr(m, "__version__", "?")
        except Exception:
            env.setdefault("packages", {})[mod] = None

    nvsmi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                  "--format=csv,noheader"])
    if nvsmi:
        env["nvidia_smi"] = nvsmi.splitlines()
    env["cpu_count"] = os.cpu_count()
    return env


def fingerprint_file(path: str, head_lines: int = 0) -> dict:
    """SHA-256 + line count for a data list, so a silently regenerated split is
    detectable after the fact rather than quietly invalidating a comparison."""
    if not path or not os.path.isfile(path):
        return {"path": path, "exists": False}
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += chunk.count(b"\n")
    out = {
        "path": path,
        "exists": True,
        "sha256": h.hexdigest(),
        "lines": n,
        "bytes": os.path.getsize(path),
    }
    if head_lines:
        with open(path, encoding="utf-8", errors="replace") as fh:
            out["head"] = [next(fh, "").rstrip() for _ in range(head_lines)]
    return out


# --------------------------------------------------------------------------
class ExperimentRecorder:
    """Owns one ``experiments/results/<exp_id>/`` directory."""

    def __init__(self, exp_id: str, root: str = None, resume: bool = False):
        self.exp_id = exp_id
        self.dir = os.path.join(root or RESULTS_ROOT, exp_id)
        if os.path.isdir(self.dir) and not resume:
            # Never silently merge two runs' epoch streams into one file: the
            # analysis would read the concatenation as a single monotonic run.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = f"{self.dir}.superseded-{stamp}"
            shutil.move(self.dir, backup)
            print(f"[recorder] existing results moved aside -> {backup}")
        os.makedirs(self.dir, exist_ok=True)
        self.epochs_path = os.path.join(self.dir, "epochs.jsonl")
        self.t0 = time.time()
        self._epoch_rows = []
        self._last_epoch_time = self.t0

        # On resume, load the epochs already recorded. Without this the history
        # is DESTROYED rather than continued: merge_into_last_epoch rewrites the
        # whole file from _epoch_rows, so the first validation of the resumed run
        # would truncate epochs.jsonl to a single row and silently discard every
        # epoch recorded before the interruption.
        if resume and os.path.isfile(self.epochs_path):
            with open(self.epochs_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._epoch_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if self._epoch_rows:
                print(f"[recorder] resuming {exp_id}: "
                      f"{len(self._epoch_rows)} epoch(s) already recorded")

    # -- pre-run -------------------------------------------------------
    def write_manifest(self, manifest: dict):
        manifest = dict(manifest)
        manifest["environment"] = capture_environment()
        manifest["recorder_started_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_json("manifest.json", manifest)
        self.manifest = manifest
        return manifest

    def write_config(self, yaml_text: str):
        self._write_text("config.yaml", yaml_text)

    def write_command(self, argv):
        self._write_text("command.txt", " \\\n    ".join(argv) + "\n")
        self._write_json("command.json", list(argv))

    def copy_script(self, script_path: str):
        if script_path and os.path.isfile(script_path):
            shutil.copy2(script_path, os.path.join(self.dir, "script.py"))

    # -- during run ----------------------------------------------------
    def log_epoch(self, row: dict):
        """Append one epoch record. Called live so a killed run keeps its
        history up to the last completed epoch."""
        now = time.time()
        row = dict(row)
        row.setdefault("wall_time_utc", datetime.now(timezone.utc).isoformat())
        row.setdefault("elapsed_s", round(now - self.t0, 2))
        row.setdefault("epoch_duration_s", round(now - self._last_epoch_time, 2))
        self._last_epoch_time = now
        with open(self.epochs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self._epoch_rows.append(row)
        return row

    def merge_into_last_epoch(self, epoch: int, extra: dict):
        """Attach late-arriving metrics (validation, per-language, AS-Norm) to
        the row for `epoch`, rewriting the file.

        Validation lands after the training line for the same epoch, so a naive
        append would produce two partial rows per epoch and every downstream
        groupby would have to re-join them.  One row per epoch is the invariant
        worth paying a rewrite for -- the files are a few hundred lines.
        """
        hit = False
        for row in self._epoch_rows:
            if row.get("epoch") == epoch:
                row.update(extra)
                hit = True
        if not hit:
            self.log_epoch(dict(extra, epoch=epoch))
            return
        with open(self.epochs_path, "w", encoding="utf-8") as fh:
            for row in self._epoch_rows:
                fh.write(json.dumps(row) + "\n")

    # -- post-run ------------------------------------------------------
    def write_final(self, extra: dict = None):
        rows = self._epoch_rows
        val = [r for r in rows if r.get("val_eer") is not None]
        best = min(val, key=lambda r: r["val_eer"]) if val else None
        summary = {
            "exp_id": self.exp_id,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "total_wall_s": round(time.time() - self.t0, 2),
            "total_wall_h": round((time.time() - self.t0) / 3600.0, 3),
            "n_epochs_completed": len({r["epoch"] for r in rows if "epoch" in r}),
            "n_validations": len(val),
            "best": (
                {
                    "epoch": best["epoch"],
                    "val_eer": best["val_eer"],
                    "val_mindcf": best.get("val_mindcf"),
                    "val_threshold": best.get("val_threshold"),
                }
                if best
                else None
            ),
            "final_train_loss": rows[-1].get("train_loss") if rows else None,
            "mean_epoch_duration_s": (
                round(sum(r.get("epoch_duration_s", 0) for r in rows) / len(rows), 2)
                if rows
                else None
            ),
            "curves": {
                "epoch": [r.get("epoch") for r in rows],
                "train_loss": [r.get("train_loss") for r in rows],
                "train_metric": [r.get("train_metric") for r in rows],
                "lr": [r.get("lr") for r in rows],
                "val_eer": [r.get("val_eer") for r in rows],
                "val_mindcf": [r.get("val_mindcf") for r in rows],
            },
        }
        if extra:
            summary.update(extra)
        self._write_json("final.json", summary)
        return summary

    # -- helpers -------------------------------------------------------
    def path(self, name):
        return os.path.join(self.dir, name)

    def _write_json(self, name, obj):
        with open(self.path(name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)

    def _write_text(self, name, text):
        with open(self.path(name), "w", encoding="utf-8") as fh:
            fh.write(text)
