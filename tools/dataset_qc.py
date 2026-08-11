#!/usr/bin/env python3
"""Layers 0, 2 and 4 of the dataset quality assessment.

See research_logs/2026-08-10-dataset-quality-assessment-methods-and-plan.md.

L0  integrity & inventory   format uniformity, near-silent files, decoded-PCM
                            duplicates, effective bandwidth
L2  distribution & coverage speaker/utterance/session/duration statistics,
                            Gini, short-utterance fractions
L4  protocol audit          trial-list sanity, same-(speaker,session) targets,
                            train/test leakage, utterance reuse

Everything here is CPU-only. Metadata-derived checks are exact (they read
metadata/utterances.csv, which prepare.py wrote from the real decode); checks
that need the audio itself run on a random sample controlled by --audio-sample.

Usage:
    python tools/dataset_qc.py --dataset data/slr52_sinhala
    python tools/dataset_qc.py --all --audio-sample 3000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
SILENCE_DBFS = -60.0
CLIP_THRESH = 0.99


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def gini(x):
    """Gini coefficient of a non-negative array. 0 = perfectly even."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def read_utterances(ds):
    path = os.path.join(ds, "metadata", "utterances.csv")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def effective_bandwidth(audio, sr, drop_db=50.0):
    """Highest frequency whose smoothed power is within drop_db of the peak.

    Detects narrowband audio that has been resampled up: a genuine 16 kHz
    recording carries energy to ~8 kHz, while an upsampled 8 kHz telephone
    recording falls off a cliff at ~4 kHz while still claiming sr=16000.
    """
    if len(audio) < sr // 4:
        return None
    n = 1 << 12
    win = np.hanning(n)
    step = n // 2
    frames = [audio[i:i + n] * win for i in range(0, len(audio) - n, step)]
    if not frames:
        return None
    spec = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames[:200]], axis=0)
    spec = spec / (spec.max() + 1e-20)
    db = 10 * np.log10(spec + 1e-20)
    # smooth over ~10 bins so a single notch does not fool us
    k = 10
    db = np.convolve(db, np.ones(k) / k, mode="same")
    above = np.where(db > -drop_db)[0]
    if len(above) == 0:
        return None
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(freqs[above[-1]])


# --------------------------------------------------------------------------- #
# L0
# --------------------------------------------------------------------------- #
def layer0(ds, rows, sample_n, rng):
    out = {"metadata_derived": {}, "audio_sampled": {}}

    srs = Counter(r["orig_sr"] for r in rows)
    chs = Counter(r["orig_channels"] for r in rows)
    out["metadata_derived"] = {
        "utterances": len(rows),
        "source_sample_rates": dict(srs),
        "source_channels": dict(chs),
        "source_rate_uniform": len(srs) <= 1,
        "output_sample_rate": 16000,
        "output_channels": 1,
        "output_subtype": "PCM_16",
    }

    paths = [os.path.join(ds, "wav", r["path"]) for r in rows]
    sample = paths if len(paths) <= sample_n else rng.sample(paths, sample_n)

    failures, silent, clipped, bw, hashes = [], [], [], [], {}
    dup_pairs = []
    for p in sample:
        try:
            a, sr = sf.read(p, dtype="float32")
        except Exception as exc:
            failures.append(f"{p}: {exc}")
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        if a.size == 0:
            silent.append(p)
            continue
        rms = float(np.sqrt(np.mean(a ** 2)))
        dbfs = 20 * np.log10(rms + 1e-20)
        if dbfs < SILENCE_DBFS:
            silent.append(p)
        clip_rate = float(np.mean(np.abs(a) > CLIP_THRESH))
        if clip_rate > 0.001:
            clipped.append((os.path.relpath(p, ds), round(clip_rate, 5)))
        b = effective_bandwidth(a, sr)
        if b is not None:
            bw.append(b)
        h = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
        if h in hashes:
            dup_pairs.append((os.path.relpath(hashes[h], ds), os.path.relpath(p, ds)))
        else:
            hashes[h] = p

    bw = np.array(bw) if bw else np.array([0.0])
    # a duplicate is only alarming when it crosses speakers
    cross_spk_dups = [d for d in dup_pairs if d[0].split("/")[1] != d[1].split("/")[1]]
    out["audio_sampled"] = {
        "sampled": len(sample),
        "decode_failures": len(failures),
        "decode_failure_examples": failures[:5],
        "near_silent_files": len(silent),
        "near_silent_examples": [os.path.relpath(s, ds) for s in silent[:5]],
        "clipped_files_over_0.1pct": len(clipped),
        "clipped_examples": clipped[:5],
        "duplicate_pairs": len(dup_pairs),
        "cross_speaker_duplicate_pairs": len(cross_spk_dups),
        "cross_speaker_duplicate_examples": cross_spk_dups[:5],
        "effective_bandwidth_hz": {
            "p05": round(float(np.percentile(bw, 5)), 1),
            "median": round(float(np.median(bw)), 1),
            "p95": round(float(np.percentile(bw, 95)), 1),
            "nyquist": 8000.0,
            "median_over_nyquist": round(float(np.median(bw)) / 8000.0, 3),
        },
    }
    return out


# --------------------------------------------------------------------------- #
# L2
# --------------------------------------------------------------------------- #
def layer2(rows):
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    out = {}
    for lang, lr in sorted(by_lang.items()):
        durs = np.array([float(r["duration_s"]) for r in lr])
        per_spk = Counter(r["spk_id"] for r in lr)
        sess_per_spk = defaultdict(set)
        for r in lr:
            sess_per_spk[r["spk_id"]].add(r["session_id"])
        n_sess = np.array([len(v) for v in sess_per_spk.values()])
        counts = np.array(list(per_spk.values()))

        gender = Counter()
        for spk in per_spk:
            g = {r["gender"] for r in lr if r["spk_id"] == spk} if len(per_spk) < 200 else None
            if g is not None:
                gender[g.pop() if len(g) == 1 else "unk"] += 1
        if not gender:  # large corpora: derive per-speaker gender cheaply
            spk_gender = {}
            for r in lr:
                spk_gender.setdefault(r["spk_id"], set()).add(r["gender"])
            gender = Counter(next(iter(v)) if len(v) == 1 else "unk"
                             for v in spk_gender.values())

        # duration vs speaker: does one speaker own the long files?
        spk_mean_dur = defaultdict(list)
        for r in lr:
            spk_mean_dur[r["spk_id"]].append(float(r["duration_s"]))
        means = np.array([np.mean(v) for v in spk_mean_dur.values()])

        out[lang] = {
            "speakers": len(per_spk),
            "utterances": len(lr),
            "hours": round(float(durs.sum()) / 3600, 2),
            "utts_per_speaker": {
                "min": int(counts.min()), "median": float(np.median(counts)),
                "mean": round(float(counts.mean()), 1), "max": int(counts.max()),
                "gini": round(gini(counts), 4),
            },
            "sessions_per_speaker": {
                "min": int(n_sess.min()), "median": float(np.median(n_sess)),
                "max": int(n_sess.max()),
                "speakers_multi_session": int((n_sess > 1).sum()),
                "speakers_single_session": int((n_sess == 1).sum()),
                "frac_multi_session": round(float((n_sess > 1).mean()), 4),
            },
            "duration_s": {
                "min": round(float(durs.min()), 2),
                "p05": round(float(np.percentile(durs, 5)), 2),
                "median": round(float(np.median(durs)), 2),
                "mean": round(float(durs.mean()), 2),
                "p95": round(float(np.percentile(durs, 95)), 2),
                "max": round(float(durs.max()), 2),
                "std": round(float(durs.std()), 2),
            },
            "short_utterance_fraction": {
                "under_2s": round(float((durs < 2).mean()), 4),
                "under_4s": round(float((durs < 4).mean()), 4),
            },
            "gender_speaker_counts": dict(gender),
            "speaker_mean_duration_spread": {
                "min": round(float(means.min()), 2),
                "max": round(float(means.max()), 2),
                "cv": round(float(means.std() / (means.mean() + 1e-12)), 4),
            },
            "duration_histogram": {
                "bin_edges_s": [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 60],
                "counts": np.histogram(
                    durs, bins=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 60])[0].tolist(),
            },
        }
    return out


# --------------------------------------------------------------------------- #
# L4
# --------------------------------------------------------------------------- #
def parse_trials(path):
    trials = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) == 3:
                trials.append((p[0], p[1], p[2]))
    return trials


def key(rel):
    """rel = <lang>/<spk>/<session>/<utt>.wav -> (lang, spk, session)."""
    parts = rel.split("/")
    return (parts[0], parts[1], parts[2]) if len(parts) >= 4 else (None, None, None)


def layer4(ds, rows):
    lists_dir = os.path.join(ds, "lists")
    if not os.path.isdir(lists_dir):
        return {"status": "no lists/ directory"}

    gender_of = {}
    for r in rows:
        gender_of[r["path"]] = r["gender"]

    out = {}
    train_spk = set()
    tl = os.path.join(lists_dir, "train_list.txt")
    if os.path.isfile(tl):
        n = 0
        for line in open(tl, encoding="utf-8"):
            p = line.split()
            if len(p) == 2:
                train_spk.add(key(p[1])[1])
                n += 1
        out["train_list"] = {"utterances": n, "speakers": len(train_spk)}

    for name in sorted(os.listdir(lists_dir)):
        if not (name.startswith("test_list") and name.endswith(".txt")):
            continue
        trials = parse_trials(os.path.join(lists_dir, name))
        if not trials:
            out[name] = {"trials": 0}
            continue

        labels = Counter(t[0] for t in trials)
        self_pairs = sum(1 for t in trials if t[1] == t[2])
        seen, dups = set(), 0
        same_sess_tgt = 0
        tgt_diff_spk = 0
        imp_same_spk = 0
        xgender_imp = 0
        gender_known_imp = 0
        used = Counter()
        tgt_speakers = set()

        for lab, a, b in trials:
            pair = tuple(sorted((a, b)))
            if pair in seen:
                dups += 1
            seen.add(pair)
            used[a] += 1
            used[b] += 1
            ka, kb = key(a), key(b)
            if lab == "1":
                if ka[1] != kb[1]:
                    tgt_diff_spk += 1
                else:
                    tgt_speakers.add(ka[1])
                if ka[:3] == kb[:3]:
                    same_sess_tgt += 1
            else:
                if ka[1] == kb[1]:
                    imp_same_spk += 1
                ga, gb = gender_of.get(a, "unk"), gender_of.get(b, "unk")
                if ga != "unk" and gb != "unk":
                    gender_known_imp += 1
                    if ga != gb:
                        xgender_imp += 1

        test_spk = {key(t[1])[1] for t in trials} | {key(t[2])[1] for t in trials}
        reuse = np.array(list(used.values()))
        out[name] = {
            "trials": len(trials),
            "targets": labels.get("1", 0),
            "impostors": labels.get("0", 0),
            "self_pairs": self_pairs,
            "duplicate_pairs": dups,
            "targets_with_different_speakers": tgt_diff_spk,
            "impostors_sharing_a_speaker": imp_same_spk,
            "same_speaker_session_targets": same_sess_tgt,
            "distinct_speakers_supplying_targets": len(tgt_speakers),
            "distinct_speakers_in_trials": len(test_spk),
            "same_gender_impostor_fraction": (
                round(1 - xgender_imp / gender_known_imp, 4)
                if gender_known_imp else None),
            "impostor_pairs_with_known_gender": gender_known_imp,
            "utterance_reuse": {
                "distinct_utterances": len(used),
                "max_uses": int(reuse.max()),
                "median_uses": float(np.median(reuse)),
                "gini": round(gini(reuse), 4),
            },
            "train_test_speaker_overlap": (
                len(train_spk & test_spk) if train_spk else None),
        }
    return out


# --------------------------------------------------------------------------- #
def run(ds, sample_n, seed):
    rng = random.Random(seed)
    name = os.path.basename(ds.rstrip("/"))
    rows = read_utterances(ds)
    if not rows:
        return {"dataset": name, "status": "no metadata/utterances.csv -- not built"}
    print(f"[qc] {name}: {len(rows)} utterances", flush=True)
    report = {
        "dataset": name,
        "status": "ok",
        "L0_integrity": layer0(ds, rows, sample_n, rng),
        "L2_distribution": layer2(rows),
        "L4_protocol": layer4(ds, rows),
    }
    out = os.path.join(ds, "metadata", "qc_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"[qc] {name}: wrote {out}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--audio-sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary", default=os.path.join(DATA, "qc_summary.json"))
    args = ap.parse_args()

    targets = args.dataset
    if args.all or not targets:
        targets = [os.path.join(DATA, d) for d in sorted(os.listdir(DATA))
                   if os.path.isdir(os.path.join(DATA, d))
                   and os.path.isfile(os.path.join(DATA, d, "prepare.py"))]

    all_reports = {}
    for ds in targets:
        ds = ds if os.path.isabs(ds) else os.path.join(REPO, ds)
        r = run(ds, args.audio_sample, args.seed)
        all_reports[r["dataset"]] = r

    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(all_reports, fh, indent=2)
        fh.write("\n")
    print(f"[qc] summary -> {args.summary}")


if __name__ == "__main__":
    main()
