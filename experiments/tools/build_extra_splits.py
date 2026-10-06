#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two additional split sets, added 2026-08-14 after Stage A.

    python experiments/tools/build_extra_splits.py --all

Both exist because of things Stage A's results revealed, and both are written to
**new directories** so that nothing an in-flight run reads is modified.

------------------------------------------------------------------------------
1. `slr127_tamil_samebatch` -- channel-controlled Tamil trials
------------------------------------------------------------------------------
Stage A reported Tamil EERs 3-5x lower than Sinhala on every architecture. That
is not a language effect, and it is not data volume (Tamil trains on 63k
utterances against Sinhala's 129k). Measured on the evaluated ecapa1024_ta
checkpoint, `slr127_tamil` pools **three collection sites** -- ISTL, MILE, MICI
-- and its impostor trials split cleanly:

    impostor pairs, cross-batch  4,805   mean score -0.0433   EER 0.475%
    impostor pairs, same-batch   5,151   mean score +0.0527   EER 1.298%
    all impostors (as reported)  9,956                        EER 0.894%

Cross-batch impostors score *negative*: the model rejects them almost for free
because the recording channel differs, not because the speakers do. About 48% of
the shipped trial list is that easy kind, so the headline 0.894% is inflated by
roughly a factor of 2.7 relative to the channel-controlled 1.298%.

`slr52_sinhala` has no comparable structure -- one homogeneous collection, so
none of its impostors get the free rejection. Comparing the two corpora's EERs
therefore compares protocols as much as languages.

This builds a trial list whose **impostor pairs are constrained to the same
collection batch**, removing the channel shortcut. Target pairs are unchanged
(a speaker is always within their own batch). It is an *evaluation* fix, so no
retraining is needed: existing checkpoints are simply scored against it.

------------------------------------------------------------------------------
2. `slceleb2026_sinhala_split` -- Sinhala with genuine sessions
------------------------------------------------------------------------------
`slr52_sinhala` has `n_utts == n_sessions`: every utterance is its own recording,
so it has no within-speaker session structure and a "cross-session" trial cannot
be distinguished from a cross-utterance one. Combined with the QC audit's
eta^2(SNR|speaker) of 0.68-0.74, channel is partly identity there too.

`slceleb2026_sinhala` is the only Sinhala corpus in this collection with **real
sessions** -- session = YouTube video id, so different sessions are different
days, studios, microphones and backgrounds. 100 of its 123 speakers appear in
more than one video (median 3, max 8). Cross-session target trials here are
genuine, which makes its absolute EER meaningful rather than optimistic.

Target trials are therefore **strictly cross-session**: a target pair whose two
sides come from the same video would largely measure channel similarity, which
is the artifact this split exists to avoid.

The cost is power. 123 speakers is small, so the test partition holds few
speakers and n_eff is capped hard (see the manifest's `test_n_eff_cap`). Treat
its absolute EER as indicative and rely on the paired contrasts, exactly as for
the other conditions -- the value here is that the *protocol* is honest, not
that the number is precise.
"""

from __future__ import annotations

import argparse
import csv
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

from experiments.tools.build_splits import (  # noqa: E402
    load_corpus, mde_estimate, n_eff_cap, partition_speakers, sha256,
    write_train_list, write_trials,
)


def batch_of(spk_id: str) -> str:
    """Collection batch for an slr127 speaker id, e.g. 'ISTL_0000202' -> 'ISTL'."""
    return spk_id.split("_")[0]


# --------------------------------------------------------------------------
def build_tamil_samebatch(seed=42):
    """Re-emit slr127's test/val trials with impostors constrained to one batch."""
    src = os.path.join(SPLIT_ROOT, "slr127_tamil")
    dst = os.path.join(SPLIT_ROOT, "slr127_tamil_samebatch")
    if not os.path.isdir(src):
        raise SystemExit(f"missing {src}; run build_splits.py first")
    os.makedirs(dst, exist_ok=True)

    def speaker_of(path):
        p = path.replace("\\", "/").split("/")
        return p[-3] if len(p) >= 3 else p[0]

    stats = {}
    for name in ("test_trials.txt", "val_trials.txt"):
        s = os.path.join(src, name)
        if not os.path.isfile(s):
            continue
        kept, dropped, tgt = [], 0, 0
        with open(s, encoding="utf-8") as fh:
            for line in fh:
                p = line.split()
                if len(p) < 3:
                    continue
                lab, a, b = int(p[0]), p[1], p[2]
                if lab == 1:
                    kept.append((lab, a, b))
                    tgt += 1
                    continue
                if batch_of(speaker_of(a)) == batch_of(speaker_of(b)):
                    kept.append((lab, a, b))
                else:
                    dropped += 1
        write_trials(os.path.join(dst, name), kept)
        stats[name] = {
            "targets": tgt,
            "impostors_kept_same_batch": len(kept) - tgt,
            "impostors_dropped_cross_batch": dropped,
            "total": len(kept),
        }
        print(f"  [{name}] kept {len(kept):,} "
              f"({tgt:,} target + {len(kept) - tgt:,} same-batch impostor), "
              f"dropped {dropped:,} cross-batch impostor")

    man = {
        "corpus": "slr127_tamil_samebatch",
        "language": "ta",
        "role": "channel_controlled_evaluation",
        "derived_from": "experiments/splits/slr127_tamil",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "change": "impostor pairs restricted to the same collection batch "
                  "(ISTL / MILE / MICI); target pairs untouched",
        "why": "cross-batch impostors are rejected on channel rather than "
               "speaker identity: measured mean score -0.0433 vs +0.0527 for "
               "same-batch, and EER 0.475% vs 1.298% on ecapa1024_ta. ~48% of "
               "the original impostors were cross-batch, inflating the headline "
               "EER about 2.7x.",
        "no_retraining_required": "this changes only the trial list, so existing "
                                  "checkpoints are re-scored, not retrained",
        "trials": stats,
        "sha256": {n: sha256(os.path.join(dst, n)) for n in stats},
        "wav_root": os.path.join(DATA_ROOT, "slr127_tamil", "wav"),
    }
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print(f"  -> {os.path.relpath(dst, REPO_ROOT)}")


# --------------------------------------------------------------------------
def make_cross_session_trials(spks, utts_by_spk, n_target, seed, max_utt=60):
    """Trials whose TARGET pairs always come from two different sessions."""
    rng = random.Random(seed)
    by_sess = {}
    for s in spks:
        d = defaultdict(list)
        for path, sess in utts_by_spk[s]:
            d[sess].append(path)
        if len(d) >= 2:                      # needs >=2 sessions to be usable
            by_sess[s] = {k: v[:max_utt] for k, v in d.items()}
    usable = sorted(by_sess)
    if len(usable) < 2:
        raise SystemExit("not enough multi-session speakers for cross-session trials")

    per_spk = max(1, n_target // len(usable))
    trials, seen = [], set()
    for s in usable:
        sessions = list(by_sess[s])
        made, attempts = 0, 0
        while made < per_spk and attempts < per_spk * 60:
            attempts += 1
            s1, s2 = rng.sample(sessions, 2)          # ALWAYS different sessions
            pa = rng.choice(by_sess[s][s1])
            pb = rng.choice(by_sess[s][s2])
            key = (pa, pb) if pa < pb else (pb, pa)
            if key in seen:
                continue
            seen.add(key)
            trials.append((1, key[0], key[1]))
            made += 1

    n_tgt = len(trials)
    made, attempts = 0, 0
    while made < n_tgt and attempts < n_tgt * 60:
        attempts += 1
        x, y = rng.sample(usable, 2)
        pa = rng.choice(by_sess[x][rng.choice(list(by_sess[x]))])
        pb = rng.choice(by_sess[y][rng.choice(list(by_sess[y]))])
        key = (pa, pb) if pa < pb else (pb, pa)
        if key in seen:
            continue
        seen.add(key)
        trials.append((0, key[0], key[1]))
        made += 1

    rng.shuffle(trials)
    return trials, {"n_target": n_tgt, "n_nontarget": made,
                    "n_speakers": len(usable),
                    "target_pairs_cross_session": n_tgt,
                    "cross_session_fraction": 1.0}


def build_slceleb_primary(seed=42, n_trials=20000):
    corpus = "slceleb2026_sinhala"
    utts_by_spk, dur_by_spk = load_corpus(corpus)
    spks = sorted(utts_by_spk)
    # Written to its OWN directory: experiments/splits/slceleb2026_sinhala/ is
    # the held-out probe list that in-flight evaluations are reading right now,
    # and must not change underneath them.
    out = os.path.join(SPLIT_ROOT, "slceleb2026_sinhala_split")
    os.makedirs(out, exist_ok=True)

    train, val, test = partition_speakers(spks, dur_by_spk, seed)
    n_utt = write_train_list(os.path.join(out, "train_list.txt"), train, utts_by_spk)

    val_tr, val_st = make_cross_session_trials(val, utts_by_spk, n_trials // 4, seed + 1)
    test_tr, test_st = make_cross_session_trials(test, utts_by_spk, n_trials // 2, seed + 2)
    write_trials(os.path.join(out, "val_trials.txt"), val_tr)
    write_trials(os.path.join(out, "test_trials.txt"), test_tr)

    rng = random.Random(seed + 3)
    n_cohort = 0
    with open(os.path.join(out, "asnorm_cohort.txt"), "w", encoding="utf-8") as fh:
        for s in sorted(train):
            for p, _ in sorted(utts_by_spk[s])[:3]:
                fh.write(f"{p}\n")
                n_cohort += 1

    man = {
        "corpus": "slceleb2026_sinhala_split",
        "language": "si",
        "role": "primary_in_the_wild_sinhala",
        "source": corpus,
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "speakers": {"total": len(spks), "train": len(train), "val": len(val),
                     "test": len(test), "disjoint": True,
                     "overlap_train_test": len(set(train) & set(test))},
        "duration_hours": {
            "train": round(sum(dur_by_spk[s] for s in train) / 3600.0, 2),
            "test": round(sum(dur_by_spk[s] for s in test) / 3600.0, 2),
        },
        "utterances": {"train": n_utt},
        "trials": {"val": val_st, "test": test_st},
        "cohort_utterances": n_cohort,
        "test_mde_pp": mde_estimate(test_st["n_target"], test_st["n_speakers"]),
        "test_n_eff_cap": n_eff_cap(test_st["n_speakers"]),
        "protocol": "target pairs are STRICTLY cross-session (session = YouTube "
                    "video id, so a different session is a different day, studio, "
                    "microphone and background)",
        "why_it_matters": "slr52_sinhala has n_utts == n_sessions, so it has no "
                          "within-speaker session structure and its cross-session "
                          "trials are indistinguishable from cross-utterance ones. "
                          "This corpus is the only Sinhala data here where a "
                          "cross-session target trial is genuine.",
        "caveat": "123 speakers total, so the test partition is small and the "
                  "absolute EER is imprecise (see test_n_eff_cap). The protocol "
                  "is honest; the precision is not. Use the paired contrasts.",
        "sha256": {n: sha256(os.path.join(out, n))
                   for n in ("train_list.txt", "val_trials.txt", "test_trials.txt")},
        "wav_root": os.path.join(DATA_ROOT, corpus, "wav"),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print(f"  speakers {len(train)}/{len(val)}/{len(test)} (train/val/test), "
          f"{n_utt:,} train utts")
    print(f"  test trials {test_st['n_target'] + test_st['n_nontarget']:,} over "
          f"{test_st['n_speakers']} speakers, 100% cross-session targets")
    print(f"  test MDE ~{man['test_mde_pp']}pp  (n_eff cap {man['test_n_eff_cap']})")
    print(f"  -> {os.path.relpath(out, REPO_ROOT)}")


def build_si_pooled(seed=42):
    """slr52 (read) + slceleb2026 (in-the-wild) as one Sinhala training set.

    Why: trained alone, slceleb2026's 82 speakers are far too few -- its runs
    reached 85% train accuracy while validation EER sat at 25%, i.e. it learned
    its training speakers and transferred to none. And slr52 alone, though it has
    336 speakers, is single-condition read speech: models trained on it degrade
    ~4x on slceleb's genuine cross-session trials (4.4% -> 17.7% for ecapa1024).

    Pooling gives 418 speakers AND two recording domains, and the question it
    answers is a real one: does adding 82 in-the-wild speakers buy more
    in-the-wild robustness than it costs on read speech? Evaluation keeps the two
    domains separate so both halves of that trade are visible.

    Labels are namespaced by corpus so a slr52 speaker and an slceleb speaker can
    never collapse into one softmax class; paths are absolute because the two
    corpora live under different roots.
    """
    parts = [("slr52_sinhala", os.path.join(SPLIT_ROOT, "slr52_sinhala")),
             ("slceleb2026_sinhala", os.path.join(SPLIT_ROOT,
                                                  "slceleb2026_sinhala_split"))]
    out = os.path.join(SPLIT_ROOT, "si_pooled")
    os.makedirs(out, exist_ok=True)

    n_utt = 0
    spks = set()
    with open(os.path.join(out, "train_list.txt"), "w", encoding="utf-8") as fh:
        for corpus, sdir in parts:
            wav = os.path.join(DATA_ROOT, corpus, "wav")
            src = os.path.join(sdir, "train_list.txt")
            if not os.path.isfile(src):
                raise SystemExit(f"missing {src}")
            for line in open(src, encoding="utf-8"):
                p = line.split()
                if len(p) < 2:
                    continue
                fh.write(f"{corpus}_{p[0]} {os.path.join(wav, p[1])}\n")
                spks.add(f"{corpus}_{p[0]}")
                n_utt += 1

    # Absolute-path trial lists, so one run can evaluate both domains through
    # --per_lang_test_lists (which resolves everything against a single
    # --test_path, hence "/").
    for corpus, sdir in parts:
        wav = os.path.join(DATA_ROOT, corpus, "wav")
        for name in ("test_trials.txt", "val_trials.txt"):
            src = os.path.join(sdir, name)
            if not os.path.isfile(src):
                continue
            tag = "read" if "slr52" in corpus else "wild"
            dst = os.path.join(out, f"{name[:-4]}_{tag}_abs.txt")
            with open(dst, "w", encoding="utf-8") as fh:
                for line in open(src, encoding="utf-8"):
                    p = line.split()
                    if len(p) >= 3:
                        fh.write(f"{p[0]} {os.path.join(wav, p[1])} "
                                 f"{os.path.join(wav, p[2])}\n")

    cohort = os.path.join(out, "asnorm_cohort.txt")
    n_coh = 0
    with open(cohort, "w", encoding="utf-8") as fh:
        for corpus, sdir in parts:
            wav = os.path.join(DATA_ROOT, corpus, "wav")
            src = os.path.join(sdir, "asnorm_cohort.txt")
            if not os.path.isfile(src):
                continue
            for i, line in enumerate(open(src, encoding="utf-8")):
                if i >= 750:
                    break
                fh.write(f"{os.path.join(wav, line.strip())}\n")
                n_coh += 1

    man = {
        "corpus": "si_pooled", "language": "si", "role": "pooled_domain_sinhala",
        "sources": [c for c, _ in parts], "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "speakers": {"train": len(spks)},
        "utterances": {"train": n_utt},
        "domains": {"read": "slr52_sinhala", "wild": "slceleb2026_sinhala"},
        "evaluation": "kept per-domain: *_read_abs.txt and *_wild_abs.txt, so the "
                      "read-speech cost and the in-the-wild gain are separately "
                      "visible rather than averaged into one number",
        "cohort_utterances": n_coh,
        "why": "slceleb2026 alone has 82 speakers and does not generalise "
               "(85% train acc, 25% val EER); slr52 alone is single-condition "
               "read speech and degrades ~4x on cross-session trials.",
        "sha256": {"train_list.txt": sha256(os.path.join(out, "train_list.txt"))},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    print(f"  speakers {len(spks)} (336 read + 82 wild), {n_utt:,} utterances")
    print(f"  per-domain trial lists: test_trials_read_abs.txt / _wild_abs.txt")
    print(f"  -> {os.path.relpath(out, REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tamil-samebatch", action="store_true")
    ap.add_argument("--slceleb", action="store_true")
    ap.add_argument("--si-pooled", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=20000)
    args = ap.parse_args()
    if not (args.all or args.tamil_samebatch or args.slceleb or args.si_pooled):
        ap.error("pass --all, --tamil-samebatch, --slceleb or --si-pooled")

    if args.all or args.tamil_samebatch:
        print("[slr127_tamil_samebatch] channel-controlled Tamil trials")
        build_tamil_samebatch(args.seed)
    if args.all or args.slceleb:
        print("\n[slceleb2026_sinhala_split] in-the-wild Sinhala, real sessions")
        build_slceleb_primary(args.seed, args.n_trials)
    if args.all or args.si_pooled:
        print("\n[si_pooled] slr52 (read) + slceleb2026 (in-the-wild)")
        build_si_pooled(args.seed)


if __name__ == "__main__":
    main()
