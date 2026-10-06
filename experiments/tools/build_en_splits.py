#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""English (VoxCeleb) splits — the reference language for the cross-language study.

    python experiments/tools/build_en_splits.py

Produces two English conditions, and the distinction between them is the whole
point of adding English to this study.

`en_full` — the standard protocol
---------------------------------
VoxCeleb2-dev for training (5,991 speakers, 1.09M utterances), VoxCeleb1-O for
trials (40 speakers, 37,720 pairs). Already speaker-disjoint by construction, so
no repartitioning is needed or wanted: keeping the canonical lists means the
number produced here is directly comparable with the published literature.

That comparability is its real job. If ECAPA-TDNN under this harness lands near
the ~1% EER the recipe is known to give on VoxCeleb1-O, the whole pipeline —
loader, augmentation, scoring, checkpoint selection — is validated end to end.
If it does not, every Sinhala and Tamil number is suspect. It is the positive
control.

A validation split is carved out of VoxCeleb2 *training* speakers so that
epoch-level model selection never touches VoxCeleb1-O.

`en_matched` — matched to the Sri Lankan corpora
------------------------------------------------
`en_full` cannot answer "does this architecture suit Sinhala better than
English", because it differs from the SL conditions in speaker count (5,991 vs
336) and utterance count (1.09M vs 129k) by more than an order of magnitude.
Comparing architecture rankings across those conditions would measure how each
architecture scales with data, not how it responds to a language.

`en_matched` therefore subsamples VoxCeleb2 to the Sinhala condition's shape:
the same number of training speakers, and a similar utterances-per-speaker
distribution. With data volume held constant, a difference in the ranking
between `en_matched`, `si` and `ta` is attributable to the language and the
recording conditions rather than to scale.

Both are needed. `en_full` says "is the harness right"; `en_matched` says "does
the language matter".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_ROOT = os.path.join(REPO_ROOT, "data")
SPLIT_ROOT = os.path.join(REPO_ROOT, "experiments", "splits")
EN_ROOT = os.path.join(DATA_ROOT, "voxceleb_new")

from experiments.tools.build_splits import (  # noqa: E402
    make_trials, mde_estimate, n_eff_cap, sha256, write_trials,
)


def load_train_list(path):
    """VoxCeleb train list: `<speaker> <relpath>` per line."""
    by_spk = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # session is the middle component of id/session/utt.wav
            comps = parts[1].split("/")
            session = comps[1] if len(comps) >= 3 else ""
            by_spk[parts[0]].append((parts[1], session))
    return by_spk


def estimate_mean_duration(by_spk, rng, n_sample=400):
    """Mean utterance duration, measured on a random sample of the audio.

    VoxCeleb ships no duration metadata and reading 1.09M headers would be
    wasteful, so this samples a few hundred files. Only the mean is needed, and
    at n=400 its standard error is a few hundredths of a second — far below the
    precision the matching requires.
    """
    import soundfile as sf

    root = os.path.join(EN_ROOT, "voxceleb2")
    all_utts = [(s, p) for s, v in by_spk.items() for p, _ in v]
    picks = rng.sample(all_utts, min(n_sample, len(all_utts)))
    durs = []
    for _, rel in picks:
        try:
            info = sf.info(os.path.join(root, rel))
            durs.append(info.frames / float(info.samplerate))
        except Exception:
            continue
    if not durs:
        # Documented VoxCeleb2 average, used only if no audio is readable here.
        print("[en] WARNING: could not read any audio; falling back to 7.8 s")
        return 7.8
    return sum(durs) / len(durs)


def write_train_list(path, spks, by_spk, cap=None, rng=None):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for s in sorted(spks):
            utts = sorted(by_spk[s])
            if cap is not None and len(utts) > cap:
                utts = sorted(rng.sample(utts, cap))
            for p, _ in utts:
                fh.write(f"{s} {p}\n")
                n += 1
    return n


def build(seed=42, n_trials=20000, match_condition="si"):
    train_list = os.path.join(EN_ROOT, "train_list.clean.txt")
    if not os.path.isfile(train_list):
        train_list = os.path.join(EN_ROOT, "train_list.txt")
    if not os.path.isfile(train_list):
        raise SystemExit(f"no VoxCeleb train list under {EN_ROOT}")
    test_list = os.path.join(EN_ROOT, "test_list.txt")

    by_spk = load_train_list(train_list)
    spks = sorted(by_spk)
    rng = random.Random(seed)
    print(f"[en] source: {os.path.relpath(train_list, REPO_ROOT)} — "
          f"{len(spks)} speakers, {sum(len(v) for v in by_spk.values())} utterances")

    # ---------------- en_full -------------------------------------------
    out_full = os.path.join(SPLIT_ROOT, "en_full")
    os.makedirs(out_full, exist_ok=True)

    # Hold out VoxCeleb2 speakers for validation; VoxCeleb1-O stays the test set.
    val_spk = rng.sample(spks, min(120, len(spks) // 10))
    train_spk = [s for s in spks if s not in set(val_spk)]
    n_train = write_train_list(os.path.join(out_full, "train_list.txt"),
                               train_spk, by_spk)
    val_trials, val_stats = make_trials(val_spk, by_spk, n_trials // 4, seed + 1)
    write_trials(os.path.join(out_full, "val_trials.txt"), val_trials)

    cohort_spk = rng.sample(train_spk, min(500, len(train_spk)))
    n_cohort = 0
    with open(os.path.join(out_full, "asnorm_cohort.txt"), "w", encoding="utf-8") as fh:
        for s in sorted(cohort_spk):
            for p, _ in sorted(by_spk[s])[:3]:
                fh.write(f"{p}\n")
                n_cohort += 1

    man_full = {
        "corpus": "en_full",
        "language": "en",
        "role": "reference_standard_protocol",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "speakers": {"train": len(train_spk), "val": len(val_spk),
                     "test": "VoxCeleb1-O (40, disjoint by construction)"},
        "utterances": {"train": n_train},
        "trials": {"val": val_stats},
        "test_list": test_list,
        "test_list_note": "canonical VoxCeleb1-O; deliberately NOT regenerated, "
                          "so the EER is comparable with published numbers",
        "cohort_utterances": n_cohort,
        "purpose": "positive control — ECAPA here should land near the ~1% EER "
                   "this recipe is known to give on VoxCeleb1-O. A large "
                   "deviation invalidates the SL numbers too.",
        "sha256": {f: sha256(os.path.join(out_full, f))
                   for f in ("train_list.txt", "val_trials.txt", "asnorm_cohort.txt")},
    }
    with open(os.path.join(out_full, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man_full, fh, indent=2)
    print(f"[en_full] train {len(train_spk)} spk / {n_train} utts, "
          f"val {len(val_spk)} spk, test = VoxCeleb1-O")

    # ---------------- en_matched ----------------------------------------
    ref_path = os.path.join(SPLIT_ROOT,
                            {"si": "slr52_sinhala", "ta": "slr127_tamil"}[match_condition],
                            "manifest.json")
    if not os.path.isfile(ref_path):
        raise SystemExit(f"build the SL splits first: missing {ref_path}")
    with open(ref_path, encoding="utf-8") as fh:
        ref = json.load(fh)
    n_ref_spk = ref["speakers"]["train"]
    n_ref_utt = ref["utterances"]["train"]
    ref_hours = ref["duration_hours"]["train"]

    # Match on total SPEECH DURATION, not utterance count.
    #
    # VoxCeleb utterances average ~7.8 s against slr52's 4.4 s, so equal
    # utterance counts would mean ~1.8x more English audio. Duration is also the
    # quantity that actually governs training: the loader draws a random
    # `max_frames` crop per utterance, so what a speaker contributes is bounded
    # by how much distinct audio exists, not by how it was segmented.
    mean_dur = estimate_mean_duration(by_spk, rng, n_sample=400)
    utt_per_spk = max(1, round((ref_hours * 3600.0) / (n_ref_spk * mean_dur)))
    print(f"[en_matched] VoxCeleb mean utterance {mean_dur:.2f}s (measured on a "
          f"sample) -> {utt_per_spk} utts/spk to match {ref_hours:.1f} h")

    # The target may exceed what VoxCeleb2 holds for this many speakers: its
    # speakers average ~182 utterances, so 336 of them cap out near 137 h
    # against slr52's 157 h. Capping cannot manufacture audio that is not there,
    # so the shortfall is measured and recorded rather than papered over by
    # quietly adding speakers -- which would break the exact speaker-count match
    # that is the more important half of the control.
    avail_hours = sum(min(len(by_spk[s]), utt_per_spk) for s in spks) * mean_dur / 3600.0

    out_m = os.path.join(SPLIT_ROOT, "en_matched")
    os.makedirs(out_m, exist_ok=True)

    rng_m = random.Random(seed + 100)
    # Only speakers with enough material to hit the target depth, so the match is
    # on the utterances-per-speaker distribution and not just the speaker count.
    eligible = [s for s in spks if len(by_spk[s]) >= min(utt_per_spk, 40)]
    pool = rng_m.sample(eligible, min(n_ref_spk + 140, len(eligible)))
    m_val = pool[:40]
    m_test = pool[40:140]
    m_train = pool[140:140 + n_ref_spk]

    n_m_train = write_train_list(os.path.join(out_m, "train_list.txt"),
                                 m_train, by_spk, cap=utt_per_spk, rng=rng_m)
    m_val_trials, m_val_stats = make_trials(m_val, by_spk, n_trials // 4, seed + 2)
    m_test_trials, m_test_stats = make_trials(m_test, by_spk, n_trials // 2, seed + 3)
    write_trials(os.path.join(out_m, "val_trials.txt"), m_val_trials)
    write_trials(os.path.join(out_m, "test_trials.txt"), m_test_trials)

    n_cohort_m = 0
    with open(os.path.join(out_m, "asnorm_cohort.txt"), "w", encoding="utf-8") as fh:
        for s in sorted(m_train)[:500]:
            for p, _ in sorted(by_spk[s])[:3]:
                fh.write(f"{p}\n")
                n_cohort_m += 1

    man_m = {
        "corpus": "en_matched",
        "language": "en",
        "role": "scale_matched_control",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "matched_to": {
            "condition": match_condition,
            "corpus": ref["corpus"],
            "matched_on": "speaker count (exact) and total speech hours (approx)",
            "train_speakers": n_ref_spk,
            "reference_train_utterances": n_ref_utt,
            "reference_train_hours": ref_hours,
            "en_mean_utterance_s": round(mean_dur, 3),
            "en_utterances_per_speaker_cap": utt_per_spk,
            "en_estimated_train_hours": round(n_m_train * mean_dur / 3600.0, 2),
            "hours_match_pct": round(100.0 * (n_m_train * mean_dur / 3600.0) / ref_hours, 1),
            "hours_ceiling_for_this_speaker_count": round(avail_hours, 1),
            "note": "utterance COUNTS differ by design — VoxCeleb utterances are "
                    "longer, so equal counts would mean unequal audio. Duration "
                    "is matched instead; it is what bounds how much distinct "
                    "speech a speaker contributes under random-crop sampling.",
        },
        "speakers": {"train": len(m_train), "val": len(m_val), "test": len(m_test),
                     "disjoint": True,
                     "overlap_train_test": len(set(m_train) & set(m_test))},
        "utterances": {"train": n_m_train},
        "trials": {"val": m_val_stats, "test": m_test_stats},
        "cohort_utterances": n_cohort_m,
        "test_mde_pp": mde_estimate(m_test_stats["n_target"], m_test_stats["n_speakers"]),
        "test_n_eff_cap": n_eff_cap(m_test_stats["n_speakers"]),
        "purpose": "holds data volume constant against the SL conditions so a "
                   "difference in architecture RANKING is attributable to the "
                   "language rather than to corpus size",
        "sha256": {f: sha256(os.path.join(out_m, f))
                   for f in ("train_list.txt", "val_trials.txt", "test_trials.txt")},
    }
    with open(os.path.join(out_m, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man_m, fh, indent=2)
    achieved_h = n_m_train * mean_dur / 3600.0
    print(f"[en_matched] train {len(m_train)} spk / {n_m_train} utts "
          f"(~{n_m_train // max(1, len(m_train))}/spk), "
          f"val {len(m_val)} spk, test {len(m_test)} spk")
    print(f"             speakers matched EXACTLY to {match_condition} ({n_ref_spk}); "
          f"hours {achieved_h:.1f} vs reference {ref_hours:.1f} "
          f"({100.0 * achieved_h / ref_hours:.0f}%)")
    if achieved_h < 0.85 * ref_hours:
        print(f"             NOTE: VoxCeleb2 speakers average "
              f"{sum(len(v) for v in by_spk.values()) / len(spks):.0f} utterances, "
              f"so {n_ref_spk} of them cannot reach {ref_hours:.0f} h. The residual "
              f"gap is recorded in the manifest; quote the ranking, not the "
              f"absolute EER, when comparing across conditions.")
    print(f"             test MDE ~{man_m['test_mde_pp']}pp")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=20000)
    ap.add_argument("--match", default="si", choices=["si", "ta"],
                    help="which SL condition en_matched is sized against")
    args = ap.parse_args()
    build(args.seed, args.n_trials, args.match)


if __name__ == "__main__":
    main()
