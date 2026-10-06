#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch ``trainSpeakerNet.py`` and turn its stdout into structured records.

The trainer is invoked as a subprocess and left completely untouched.  Its
per-epoch console lines have stable, greppable shapes (verified against
trainSpeakerNet.py lines 282-345 and 550-640), and this module converts them to
one JSON row per epoch as they appear.

Why a subprocess rather than importing the trainer:

* ``trainSpeakerNet.py`` parses argv at module scope and calls ``mp.spawn``;
  importing it would fight both of those.
* The exact argv becomes the reproducibility record -- ``command.txt`` can be
  pasted into a shell and re-run.
* A crashed or OOM-killed run cannot corrupt the recorder's own state, and the
  epochs recorded up to that point stay valid.

Parsed line shapes::

    2026-.. Epoch 3, TEER/TAcc 12.34, TLOSS 5.678901, LR 0.001000
    2026-.. Epoch 3, VEER 4.1234, MinDCF 0.31415, Threshold 0.512000
    2026-.. Epoch 3 (POOLED), VEER 4.1234, MinDCF 0.31415, Threshold 0.512000
    [per-lang epoch3/si] VEER 3.2100, VEER_avg 3.3000, MinDCF 0.29000, Threshold 0.4
    [per-lang POOLED/epoch3/] VEER 4.1234, VEER_avg .., MinDCF .., Threshold ..
    [AS-Norm] Epoch 3, Raw VEER 5.0000, Raw MinDCF 0.40000  (delta MinDCF -0.011)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

RE_TRAIN = re.compile(
    r"Epoch\s+(?P<epoch>\d+),\s*TEER/TAcc\s+(?P<metric>[-\d.]+),\s*"
    r"TLOSS\s+(?P<loss>[-\d.eE+]+),\s*LR\s+(?P<lr>[-\d.eE+]+)"
)
RE_VAL = re.compile(
    r"Epoch\s+(?P<epoch>\d+)(?:\s+\(POOLED\))?,\s*VEER\s+(?P<eer>[-\d.]+),\s*"
    r"MinDCF\s+(?P<mindcf>[-\d.]+),\s*Threshold\s+(?P<thr>[-\d.eE+]+)"
)
RE_PERLANG = re.compile(
    r"\[per-lang\s+(?:epoch(?P<epoch>\d+)/)?(?P<lang>[^\]/]+?)\]\s*"
    r"VEER\s+(?P<eer>[-\d.]+),\s*VEER_avg\s+(?P<eer_avg>[-\d.]+),\s*"
    r"MinDCF\s+(?P<mindcf>[-\d.]+),\s*Threshold\s+(?P<thr>[-\d.eE+]+)"
)
RE_PERLANG_POOLED = re.compile(
    r"\[per-lang\s+POOLED(?:/epoch(?P<epoch>\d+)/?)?\]\s*"
    r"VEER\s+(?P<eer>[-\d.]+),\s*VEER_avg\s+(?P<eer_avg>[-\d.]+),\s*"
    r"MinDCF\s+(?P<mindcf>[-\d.]+),\s*Threshold\s+(?P<thr>[-\d.eE+]+)"
)
RE_ASNORM = re.compile(
    r"\[AS-Norm\]\s+Epoch\s+(?P<epoch>\d+),\s*Raw\s+VEER\s+(?P<raw_eer>[-\d.]+),\s*"
    r"Raw\s+MinDCF\s+(?P<raw_mindcf>[-\d.]+)"
)
RE_BEST = re.compile(r"New best EER:\s*(?P<eer>[-\d.]+)")
RE_EARLY = re.compile(r"Early stopping at epoch\s+(?P<epoch>\d+)")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class TrainerRunner:
    """Runs the trainer, streams its output into an ExperimentRecorder."""

    def __init__(self, recorder, argv, env=None, cwd=None, echo=True):
        self.rec = recorder
        self.argv = list(argv)
        self.env = env or os.environ.copy()
        self.cwd = cwd or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.echo = echo
        self.events = {"best_updates": [], "early_stopped_at": None, "warnings": []}
        # Per-language rows arrive before the pooled line for the same epoch, so
        # they are buffered and flushed onto the epoch row once it is known.
        self._pending_lang = {}

    def _handle(self, line: str):
        rec = self.rec

        m = RE_TRAIN.search(line)
        if m:
            rec.log_epoch(
                {
                    "epoch": int(m["epoch"]),
                    "phase": "train",
                    "train_loss": _f(m["loss"]),
                    "train_metric": _f(m["metric"]),
                    "lr": _f(m["lr"]),
                }
            )
            return

        m = RE_PERLANG_POOLED.search(line)
        if m:
            ep = int(m["epoch"]) if m["epoch"] else None
            payload = {
                "val_eer": _f(m["eer"]),
                "val_eer_avg": _f(m["eer_avg"]),
                "val_mindcf": _f(m["mindcf"]),
                "val_threshold": _f(m["thr"]),
                "val_source": "per_lang_pooled",
            }
            if ep is not None:
                payload["per_language"] = self._pending_lang.pop(ep, {})
                rec.merge_into_last_epoch(ep, payload)
            return

        m = RE_PERLANG.search(line)
        if m:
            ep = int(m["epoch"]) if m["epoch"] else None
            entry = {
                "eer": _f(m["eer"]),
                "eer_avg": _f(m["eer_avg"]),
                "mindcf": _f(m["mindcf"]),
                "threshold": _f(m["thr"]),
            }
            self._pending_lang.setdefault(ep, {})[m["lang"].strip()] = entry
            return

        m = RE_VAL.search(line)
        if m:
            ep = int(m["epoch"])
            payload = {
                "val_eer": _f(m["eer"]),
                "val_mindcf": _f(m["mindcf"]),
                "val_threshold": _f(m["thr"]),
                "val_source": "pooled" if "(POOLED)" in line else "single_list",
            }
            if ep in self._pending_lang:
                payload["per_language"] = self._pending_lang.pop(ep)
            rec.merge_into_last_epoch(ep, payload)
            return

        m = RE_ASNORM.search(line)
        if m:
            rec.merge_into_last_epoch(
                int(m["epoch"]),
                {
                    "asnorm_raw_eer": _f(m["raw_eer"]),
                    "asnorm_raw_mindcf": _f(m["raw_mindcf"]),
                },
            )
            return

        m = RE_BEST.search(line)
        if m:
            self.events["best_updates"].append(_f(m["eer"]))
            return

        m = RE_EARLY.search(line)
        if m:
            self.events["early_stopped_at"] = int(m["epoch"])
            return

        low = line.lower()
        if ("out of memory" in low or "nan" in low) and len(self.events["warnings"]) < 50:
            self.events["warnings"].append(line.strip()[:300])

    def run(self) -> int:
        self.rec.write_command(self.argv)
        log_path = self.rec.path("stdout.log")
        with open(log_path, "w", encoding="utf-8", buffering=1) as log:
            proc = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                for line in proc.stdout:
                    log.write(line)
                    if self.echo:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    try:
                        self._handle(line)
                    except Exception as exc:  # never let parsing kill a run
                        log.write(f"[runner] parse error: {exc!r}\n")
            except KeyboardInterrupt:
                proc.terminate()
                raise
            finally:
                proc.wait()
        return proc.returncode
