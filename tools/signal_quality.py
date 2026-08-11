#!/usr/bin/env python3
"""Layer 1: reference-free signal quality, and the speaker/quality confound test.

The headline number here is NOT the mean SNR. It is the one-way ANOVA F for each
metric across speakers:

    F = MS_between / MS_within
    MS_between = sum_g n_g (mean_g - mean)^2 / (k - 1)
    MS_within  = sum_g sum_{i in g} (x_i - mean_g)^2 / (N - k)

If a signal metric (SNR, loudness, bandwidth) separates speakers strongly -- a
large F -- then that metric *is* partial speaker identity, and a verification
model can score channel instead of voice. eta^2 = SS_between / SS_total gives the
same idea as a bounded effect size in [0, 1].

Metrics per utterance:
  snr_db        percentile-split segmental SNR: frame energies split at the 20th
                and 80th percentile, SNR = 10 log10(P_high / P_low)
  clip_rate     fraction of samples with |x| > 0.99
  speech_ratio  fraction of frames above (noise_floor + 6 dB), an energy VAD
  rms_dbfs      20 log10(rms)
  crest_db      20 log10(peak / rms), i.e. dynamic headroom
  bandwidth_hz  highest frequency within 50 dB of the spectral peak
  squim_*       STOI / PESQ / SI-SDR estimates (torchaudio SQUIM, optional)

Usage:
    python tools/signal_quality.py --all --sample 1500
    tools/noderun.sh 3 python -u tools/signal_quality.py --all --sample 1500 --squim
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
OUT_DIR = os.path.join(DATA, "_qc", "signal")
FRAME, HOP = 400, 160          # 25 ms / 10 ms at 16 kHz


def frame_energy(a):
    n = 1 + max(0, (len(a) - FRAME) // HOP)
    if n < 3:
        return np.array([np.mean(a ** 2) + 1e-20])
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return np.mean(a[idx] ** 2, axis=1) + 1e-20


def metrics(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.size < FRAME * 3:
        return None
    e = frame_energy(a)
    lo, hi = np.percentile(e, 20), np.percentile(e, 80)
    snr = 10 * np.log10(hi / max(lo, 1e-20))

    floor_db = 10 * np.log10(np.percentile(e, 10))
    e_db = 10 * np.log10(e)
    speech_ratio = float(np.mean(e_db > floor_db + 6))

    rms = float(np.sqrt(np.mean(a ** 2)))
    peak = float(np.max(np.abs(a)))
    n = 1 << 12
    if len(a) >= n:
        win = np.hanning(n)
        frames = [a[i:i + n] * win for i in range(0, len(a) - n, n // 2)][:100]
        spec = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
        spec /= spec.max() + 1e-20
        db = 10 * np.log10(spec + 1e-20)
        db = np.convolve(db, np.ones(10) / 10, mode="same")
        above = np.where(db > -50)[0]
        bw = float(np.fft.rfftfreq(n, 1 / sr)[above[-1]]) if len(above) else np.nan
    else:
        bw = np.nan

    return {
        "snr_db": float(snr),
        "clip_rate": float(np.mean(np.abs(a) > 0.99)),
        "speech_ratio": speech_ratio,
        "rms_dbfs": float(20 * np.log10(rms + 1e-20)),
        "crest_db": float(20 * np.log10((peak + 1e-20) / (rms + 1e-20))),
        "bandwidth_hz": bw,
    }


def anova_f(values, groups):
    """One-way ANOVA F and eta^2 for values grouped by speaker."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    labs = np.unique(g)
    k, N = len(labs), len(v)
    if k < 2 or N - k < 1:
        return None
    grand = v.mean()
    ss_b = sum(len(v[g == l]) * (v[g == l].mean() - grand) ** 2 for l in labs)
    ss_w = sum(((v[g == l] - v[g == l].mean()) ** 2).sum() for l in labs)
    ss_t = ss_b + ss_w
    ms_b, ms_w = ss_b / (k - 1), ss_w / max(N - k, 1)
    return {
        "F": round(float(ms_b / max(ms_w, 1e-20)), 3),
        "eta_squared": round(float(ss_b / max(ss_t, 1e-20)), 4),
        "groups": int(k), "n": int(N),
    }


def squim_scores(paths, ds, batch=16):
    import torch
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SQUIM_OBJECTIVE.get_model().to(device).eval()
    out = {}
    for i in range(0, len(paths), batch):
        for p in paths[i:i + batch]:
            try:
                w, sr = torchaudio.load(os.path.join(ds, "wav", p))
                w = w[:1]
                if w.shape[1] < 16000:
                    continue
                w = w[:, :16000 * 10]
                with torch.no_grad():
                    stoi, pesq, sisdr = model(w.to(device))
                out[p] = {"stoi": float(stoi[0]), "pesq": float(pesq[0]),
                          "si_sdr": float(sisdr[0])}
            except Exception:
                continue
    return out


def run(ds, args, rng):
    name = os.path.basename(ds.rstrip("/"))
    with open(os.path.join(ds, "metadata", "utterances.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    sample = rows if len(rows) <= args.sample else rng.sample(rows, args.sample)
    print(f"[sig] {name}: {len(sample)} of {len(rows)} utterances", flush=True)

    recs = []
    for r in sample:
        m = metrics(os.path.join(ds, "wav", r["path"]))
        if m:
            m.update(spk=r["spk_id"], lang=r["lang"], path=r["path"])
            recs.append(m)

    if args.squim:
        print(f"[sig] {name}: SQUIM on {min(len(recs), args.squim_n)} files", flush=True)
        sq = squim_scores([r["path"] for r in recs[:args.squim_n]], ds)
        for r in recs:
            if r["path"] in sq:
                r.update({f"squim_{k}": v for k, v in sq[r["path"]].items()})

    keys = ["snr_db", "clip_rate", "speech_ratio", "rms_dbfs", "crest_db",
            "bandwidth_hz", "squim_stoi", "squim_pesq", "squim_si_sdr"]
    dist, conf = {}, {}
    spk = [r["spk"] for r in recs]
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        if not np.isfinite(v).any():
            continue
        f = v[np.isfinite(v)]
        dist[k] = {
            "mean": round(float(f.mean()), 4), "std": round(float(f.std()), 4),
            "p05": round(float(np.percentile(f, 5)), 4),
            "median": round(float(np.median(f)), 4),
            "p95": round(float(np.percentile(f, 95)), 4),
            "n": int(len(f)),
        }
        a = anova_f(v, spk)
        if a:
            conf[k] = a

    report = {
        "dataset": name,
        "sampled": len(recs),
        "distributions": dist,
        "speaker_confound_anova": conf,
        "note": ("High F / eta^2 means the metric separates speakers, i.e. it acts "
                 "as partial speaker identity and a model can exploit channel "
                 "instead of voice."),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    np.savez_compressed(
        os.path.join(args.out, f"{name}_raw.npz"),
        **{k: np.array([r.get(k, np.nan) for r in recs], dtype=float) for k in keys},
        spk=np.array(spk))
    snr = dist.get("snr_db", {})
    print(f"[sig] {name}: SNR {snr.get('median')} dB  "
          f"eta2(SNR|speaker)={conf.get('snr_db', {}).get('eta_squared')}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--squim", action="store_true")
    ap.add_argument("--squim-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    targets = args.dataset
    if args.all or not targets:
        targets = [os.path.join(DATA, d) for d in sorted(os.listdir(DATA))
                   if os.path.isfile(os.path.join(DATA, d, "metadata", "utterances.csv"))]
    rng = random.Random(args.seed)
    summary = {}
    for ds in targets:
        ds = ds if os.path.isabs(ds) else os.path.join(REPO, ds)
        summary[os.path.basename(ds.rstrip("/"))] = run(ds, args, rng)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"[sig] -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
