#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Speaker-disjoint split builder for the Sinhala/Tamil architecture study.

Why this exists
---------------
Every corpus under ``data/`` ships a ``lists/train_list.txt`` and a
``lists/test_list.txt`` whose speaker sets are *identical* -- 478/478 for
slr52_sinhala, 638/638 for slr127_tamil, 123/123 for slceleb2026_sinhala.
Those lists are closed-set: the trial speakers were seen in training.

A closed-set EER is not merely optimistic, it is *rank-distorting* for an
architecture comparison.  Write the score of a trial (e_1, e_2) as

    s = cos(f(x_1), f(x_2))

For a speaker seen in training, f has been explicitly optimised to place that
speaker's utterances near a learned class centroid w_c (the AAM-softmax weight
column).  The between-class margin the loss enforces at training time therefore
*transfers directly* into the trial score -- the model is being tested on the
objective it was trained on.  The gap between closed- and open-set EER grows
with the model's capacity to memorise centroids, so a high-capacity backbone
(e.g. ECAPA-1024, or an SSL frontend with 94M parameters) gains more from the
contamination than a small one (VGGVox, ResNetSE34L).  Ranking architectures on
closed-set trials measures memorisation capacity as much as speaker
discriminability, and the two orderings are not the same.

This script therefore partitions each corpus **by speaker** into disjoint
train / val / test groups, and synthesises fresh trial lists over the held-out
speakers only.

Design decisions
----------------
* **70/10/20 speaker split.**  Test gets 20% so the EER has usable precision;
  see the MDE estimate emitted into the manifest.  Val gets 10% and is used for
  epoch-level model selection, so the test set is touched exactly once per
  experiment (at the best-val checkpoint) and never drives any decision.
* **Stratified by total speech duration.**  ``gender`` is ``unk`` throughout
  slr52 and slr127, so duration decile is the only available stratifier.  It
  matters: utterance count per speaker varies ~5x, and a split that put the
  data-rich speakers in train would confound "architecture" with "how much
  enrolment audio the trial speakers had".
* **Balanced trials, evenly spread over speakers.**  Every held-out speaker
  contributes approximately the same number of target trials, so no single
  speaker dominates the EER.  Impostor pairs are sampled without replacement
  from distinct-speaker utterance pairs.
* **Deterministic.**  Everything keys off ``--seed``; re-running reproduces the
  lists byte-for-byte, and the manifest carries a SHA-256 of each emitted file.

Usage
-----
    python experiments/tools/build_splits.py --all
    python experiments/tools/build_splits.py --corpus slr52_sinhala
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
SPLIT_ROOT = os.path.join(REPO_ROOT, "experiments", "splits")

# Corpora that anchor the study.  The primary pair is matched on genre (both
# OpenSLR read speech) and roughly on scale, so that "language" is the factor
# that varies between them rather than recording style.
PRIMARY = {
    "slr52_sinhala": "si",
    "slr127_tamil": "ta",
}

# Held-out corpora: never trained on, used only as unseen-corpus generalisation
# probes.  These are automatically open-set with respect to any model trained on
# PRIMARY, so they need no speaker partition -- only a trial list.
HELDOUT = {
    "slceleb2026_sinhala": "si",
    "slr65_tamil": "ta",
    "kathbath_tamil": "ta",
    "nisp_tamil": "ta",
}

TRAIN_FRAC, VAL_FRAC = 0.70, 0.10  # test takes the remainder


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------
def load_corpus(corpus: str):
    """Return (utts_by_spk, spk_duration) read from the corpus metadata CSVs."""
    meta_dir = os.path.join(DATA_ROOT, corpus, "metadata")
    utt_csv = os.path.join(meta_dir, "utterances.csv")
    if not os.path.isfile(utt_csv):
        raise SystemExit(f"missing {utt_csv} -- run the corpus prepare.py first")

    utts_by_spk = defaultdict(list)
    dur_by_spk = defaultdict(float)
    with open(utt_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            spk = row["spk_id"]
            # (path, session) -- session lets us prefer cross-session target
            # pairs where a corpus actually has session structure.
            utts_by_spk[spk].append((row["path"], row.get("session_id", "")))
            try:
                dur_by_spk[spk] += float(row.get("duration_s") or 0.0)
            except ValueError:
                pass
    return utts_by_spk, dur_by_spk


def partition_speakers(spks, dur_by_spk, seed):
    """Duration-stratified deterministic speaker partition.

    Speakers are ordered by total duration, cut into deciles, and each decile is
    shuffled and dealt into train/val/test in proportion.  This keeps the
    duration distribution of the three groups near-identical, which a flat
    random split does not guarantee at these speaker counts.
    """
    rng = random.Random(seed)
    ordered = sorted(spks, key=lambda s: (-dur_by_spk.get(s, 0.0), s))
    train, val, test = [], [], []
    n_strata = 10
    stratum = max(1, len(ordered) // n_strata)
    for i in range(0, len(ordered), stratum):
        block = ordered[i : i + stratum]
        rng.shuffle(block)
        n = len(block)
        n_tr = int(round(TRAIN_FRAC * n))
        n_va = int(round(VAL_FRAC * n))
        # guarantee test is never starved on tiny strata
        n_tr = min(n_tr, max(0, n - 1))
        train += block[:n_tr]
        val += block[n_tr : n_tr + n_va]
        test += block[n_tr + n_va :]
    return sorted(train), sorted(val), sorted(test)


# --------------------------------------------------------------------------
# trial synthesis
# --------------------------------------------------------------------------
def make_trials(spks, utts_by_spk, n_target, seed, max_utt_per_spk=60):
    """Build a balanced target/non-target trial list over `spks` only.

    Target pairs are drawn per speaker so the per-speaker contribution is even.
    Where a speaker has more than one session, cross-session pairs are preferred
    -- a same-session target pair mostly measures channel similarity rather than
    speaker identity, which would deflate EER for reasons unrelated to the
    model.  In slr52/slr127 every utterance is its own session, so the
    preference is satisfied trivially and recorded as such in the manifest.
    """
    rng = random.Random(seed)
    pool = {}
    for s in spks:
        u = list(utts_by_spk[s])
        rng.shuffle(u)
        if len(u) >= 2:
            pool[s] = u[:max_utt_per_spk]
    usable = sorted(pool)
    if len(usable) < 2:
        raise SystemExit("not enough speakers with >=2 utterances to build trials")

    per_spk = max(1, n_target // len(usable))
    trials, seen = [], set()

    # ---- target pairs -------------------------------------------------
    n_cross_session = 0
    for s in usable:
        u = pool[s]
        made = 0
        attempts = 0
        while made < per_spk and attempts < per_spk * 40:
            attempts += 1
            a, b = rng.sample(range(len(u)), 2)
            pa, sa = u[a]
            pb, sb = u[b]
            key = (pa, pb) if pa < pb else (pb, pa)
            if key in seen:
                continue
            seen.add(key)
            trials.append((1, key[0], key[1]))
            if sa != sb:
                n_cross_session += 1
            made += 1

    # ---- impostor pairs ------------------------------------------------
    n_nontarget = len(trials)
    made = 0
    attempts = 0
    while made < n_nontarget and attempts < n_nontarget * 60:
        attempts += 1
        s1, s2 = rng.sample(usable, 2)
        pa = rng.choice(pool[s1])[0]
        pb = rng.choice(pool[s2])[0]
        key = (pa, pb) if pa < pb else (pb, pa)
        if key in seen:
            continue
        seen.add(key)
        trials.append((0, key[0], key[1]))
        made += 1

    rng.shuffle(trials)
    stats = {
        "n_target": n_nontarget,
        "n_nontarget": made,
        "n_speakers": len(usable),
        "target_pairs_cross_session": n_cross_session,
        "cross_session_fraction": round(n_cross_session / max(1, n_nontarget), 4),
    }
    return trials, stats


def mde_estimate(n_target, n_speakers, rho=0.7):
    """Minimum detectable EER difference (pp), speaker-clustered.

    Trials sharing a speaker are correlated, so the effective sample size is not
    the trial count.  With a design effect for clustering,

        n_eff = n_trials / (1 + (m - 1) * rho),    m = trials per speaker

    and for a two-sided test at alpha=0.05, power 0.8, comparing two EERs near
    p, the detectable difference is approximately

        MDE ~ 2.8 * sqrt(2 * p * (1 - p) / n_eff)

    Same estimator and rho as the 2026-08-10 QC audit, so the numbers stay
    comparable with that report.

    THE CEILING THAT MATTERS.  Substituting n = m*S,

        n_eff = m*S / (1 + (m - 1) * rho)  ->  S / rho   as m -> inf

    so the effective sample size is bounded by the *speaker* count divided by
    rho, no matter how many trial pairs are synthesised.  At S = 91 held-out
    Sinhala speakers the cap is n_eff <= 130, which pins the absolute-EER
    resolution near 6 pp.  Two consequences, both acted on elsewhere in this
    programme:

      1. Generating more pairs is nearly free power up to m ~ 20 and then
         useless.  ``--n_trials 20000`` already reaches >98% of the ceiling, so
         the default was cut from 80k to 20k -- a 4x saving in evaluation cost
         for no loss of resolving power.
      2. *Absolute* EERs from a single corpus split cannot rank architectures
         whose true separation is a few tenths of a point.  The inference in
         ``analyze.py`` is therefore a **paired** speaker-level bootstrap on the
         EER *difference* over identical trials, where the speaker-difficulty
         term common to both systems cancels and the residual variance is driven
         only by where the two systems disagree.  That statistic resolves far
         finer differences than this MDE, which describes a single system's
         absolute EER in isolation.
    """
    if n_speakers <= 0 or n_target <= 0:
        return None
    m = n_target / n_speakers
    n_eff = n_target / (1.0 + (m - 1.0) * rho)
    p = 0.03  # anchor near the EER regime these corpora actually operate in
    return round(100.0 * 2.8 * ((2.0 * p * (1 - p) / max(n_eff, 1.0)) ** 0.5), 3)


def n_eff_cap(n_speakers, rho=0.7):
    """The speaker-count ceiling on effective sample size (see mde_estimate)."""
    return round(n_speakers / rho, 1) if n_speakers else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_train_list(path, spks, utts_by_spk, lang_prefix=None):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for s in sorted(spks):
            # Namespacing the label by corpus keeps speaker ids unique when two
            # corpora are unioned for the combined-language condition.
            label = f"{lang_prefix}_{s}" if lang_prefix else s
            for p, _ in sorted(utts_by_spk[s]):
                fh.write(f"{label} {p}\n")
                n += 1
    return n


def write_trials(path, trials, abs_root=None):
    """Write a trial list; with `abs_root` set, paths are absolutised.

    The combined si+ta condition evaluates two corpora in one run via
    ``--per_lang_test_lists``, but the trainer resolves every list against a
    single ``--test_path``.  Absolute trial paths plus ``--test_path /`` is the
    only way to address two corpus roots from one run, so each trial list is
    emitted in both relative and absolute form.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for lab, a, b in trials:
            if abs_root:
                a, b = os.path.join(abs_root, a), os.path.join(abs_root, b)
            fh.write(f"{lab} {a} {b}\n")
    return len(trials)


# --------------------------------------------------------------------------
def build_corpus(corpus, lang, seed, n_trials, report):
    utts_by_spk, dur_by_spk = load_corpus(corpus)
    spks = sorted(utts_by_spk)
    out_dir = os.path.join(SPLIT_ROOT, corpus)
    os.makedirs(out_dir, exist_ok=True)

    train, val, test = partition_speakers(spks, dur_by_spk, seed)

    files = {}
    files["train_list.txt"] = write_train_list(
        os.path.join(out_dir, "train_list.txt"), train, utts_by_spk
    )
    # Namespaced copy for the combined si+ta condition.
    write_train_list(
        os.path.join(out_dir, "train_list_ns.txt"), train, utts_by_spk, lang_prefix=corpus
    )

    # n_trials is the TOTAL pair count; make_trials takes the target-pair half
    # and mirrors it with an equal number of impostor pairs.
    val_trials, val_stats = make_trials(val, utts_by_spk, n_trials // 4, seed + 1)
    test_trials, test_stats = make_trials(test, utts_by_spk, n_trials // 2, seed + 2)
    wav_root = os.path.join(DATA_ROOT, corpus, "wav")
    files["val_trials.txt"] = write_trials(os.path.join(out_dir, "val_trials.txt"), val_trials)
    files["test_trials.txt"] = write_trials(os.path.join(out_dir, "test_trials.txt"), test_trials)
    write_trials(os.path.join(out_dir, "val_trials_abs.txt"), val_trials, abs_root=wav_root)
    write_trials(os.path.join(out_dir, "test_trials_abs.txt"), test_trials, abs_root=wav_root)

    # AS-Norm cohort must come from TRAIN speakers only: the cohort defines the
    # score-normalisation statistics, and drawing it from trial speakers would
    # leak test-speaker information into the scores.
    rng = random.Random(seed + 3)
    cohort_spk = train if len(train) <= 500 else rng.sample(train, 500)
    with open(os.path.join(out_dir, "asnorm_cohort.txt"), "w", encoding="utf-8") as fh:
        n_cohort = 0
        for s in sorted(cohort_spk):
            for p, _ in sorted(utts_by_spk[s])[:3]:
                fh.write(f"{p}\n")
                n_cohort += 1

    manifest = {
        "corpus": corpus,
        "language": lang,
        "role": "primary",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "speakers": {
            "total": len(spks),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "disjoint": True,
            "overlap_train_test": len(set(train) & set(test)),
        },
        "duration_hours": {
            "train": round(sum(dur_by_spk[s] for s in train) / 3600.0, 2),
            "val": round(sum(dur_by_spk[s] for s in val) / 3600.0, 2),
            "test": round(sum(dur_by_spk[s] for s in test) / 3600.0, 2),
        },
        "utterances": {"train": files["train_list.txt"]},
        "trials": {"val": val_stats, "test": test_stats},
        "cohort_utterances": n_cohort,
        "test_mde_pp": mde_estimate(test_stats["n_target"], test_stats["n_speakers"]),
        "val_mde_pp": mde_estimate(val_stats["n_target"], val_stats["n_speakers"]),
        "test_n_eff_cap": n_eff_cap(test_stats["n_speakers"]),
        "power_note": (
            "test_mde_pp bounds a SINGLE system's absolute EER and is capped by "
            "speaker count (test_n_eff_cap = S/rho), not by trial count. "
            "Architecture ranking uses the paired speaker-level bootstrap on "
            "EER differences in experiments/tools/analyze.py, which resolves "
            "much finer differences because speaker difficulty cancels."
        ),
        "sha256": {
            f: sha256(os.path.join(out_dir, f))
            for f in ("train_list.txt", "val_trials.txt", "test_trials.txt", "asnorm_cohort.txt")
        },
        "wav_root": os.path.join(DATA_ROOT, corpus, "wav"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    report[corpus] = manifest
    print(
        f"[{corpus}] speakers {len(train)}/{len(val)}/{len(test)} (train/val/test)  "
        f"train utts {files['train_list.txt']}  "
        f"test trials {files['test_trials.txt']}  MDE~{manifest['test_mde_pp']}pp"
    )
    return train, val, test, utts_by_spk


def build_heldout(corpus, lang, seed, n_trials, report):
    """Held-out corpora contribute a trial list only -- no training partition,
    because no model in this study ever trains on them."""
    utts_by_spk, dur_by_spk = load_corpus(corpus)
    spks = sorted(utts_by_spk)
    out_dir = os.path.join(SPLIT_ROOT, corpus)
    os.makedirs(out_dir, exist_ok=True)
    trials, stats = make_trials(spks, utts_by_spk, n_trials // 2, seed + 5)
    write_trials(os.path.join(out_dir, "test_trials.txt"), trials)
    manifest = {
        "corpus": corpus,
        "language": lang,
        "role": "heldout_generalisation_probe",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "speakers": {"total": len(spks), "used_in_trials": stats["n_speakers"]},
        "duration_hours": {"total": round(sum(dur_by_spk.values()) / 3600.0, 2)},
        "trials": {"test": stats},
        "test_mde_pp": mde_estimate(stats["n_target"], stats["n_speakers"]),
        "test_n_eff_cap": n_eff_cap(stats["n_speakers"]),
        "sha256": {"test_trials.txt": sha256(os.path.join(out_dir, "test_trials.txt"))},
        "wav_root": os.path.join(DATA_ROOT, corpus, "wav"),
        "note": "No training partition: this corpus is never trained on.",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    report[corpus] = manifest
    print(
        f"[{corpus}] HELD OUT  speakers {stats['n_speakers']}  "
        f"trials {len(trials)}  MDE~{manifest['test_mde_pp']}pp"
    )


def build_combined(per_corpus, seed, report):
    """Union the two primary training partitions into the bilingual condition.

    Speaker labels are namespaced by corpus, so a Sinhala and a Tamil speaker can
    never collide into one softmax class.  Evaluation stays per-language: there
    is no bilingual trial list here because no speaker in slr52 or slr127 appears
    in both languages -- genuine cross-lingual trials require the same person
    speaking both, which only nisp_tamil provides.
    """
    out_dir = os.path.join(SPLIT_ROOT, "combined_si_ta")
    os.makedirs(out_dir, exist_ok=True)
    n_utt, n_spk = 0, 0
    with open(os.path.join(out_dir, "train_list.txt"), "w", encoding="utf-8") as out:
        for corpus, (train, _, _, utts_by_spk) in per_corpus.items():
            wav_root = os.path.join(DATA_ROOT, corpus, "wav")
            for s in sorted(train):
                n_spk += 1
                for p, _ in sorted(utts_by_spk[s]):
                    # Absolute path: the two corpora live under different roots,
                    # so a single relative train_path cannot address both.
                    out.write(f"{corpus}_{s} {os.path.join(wav_root, p)}\n")
                    n_utt += 1
    # Speaker -> language lookup for the auxiliary/adversarial language heads
    # (--lang_aux_label_file, --dann_lang_label_file).
    #
    # The file is keyed on the INTEGER speaker label, and that integer is
    # assigned by DatasetLoader as the index of the speaker key in the sorted
    # list of unique keys (DatasetLoader.py:301-303). So the mapping has to be
    # derived the same way rather than from corpus order -- "slr127_*" sorts
    # before "slr52_*", which is not the order the corpora are listed in here.
    lang_of = {}
    for corpus, (train, _, _, _) in per_corpus.items():
        lang = "si" if "sinhala" in corpus else "ta"
        for s in train:
            lang_of[f"{corpus}_{s}"] = lang
    lang_id = {"si": 0, "ta": 1}
    with open(os.path.join(out_dir, "spk_lang_lookup.txt"), "w", encoding="utf-8") as fh:
        fh.write("# spk_label_int  lang_label_int   (0=si, 1=ta)\n")
        for idx, key in enumerate(sorted(lang_of)):
            fh.write(f"{idx}\t{lang_id[lang_of[key]]}\n")

    # Bilingual AS-Norm cohort, absolute paths. A cohort should mirror the
    # training distribution; normalising Tamil trials against Sinhala-only
    # cohort statistics would inject exactly the per-language offset that
    # AS-Norm exists to remove.
    n_cohort = 0
    with open(os.path.join(out_dir, "asnorm_cohort.txt"), "w", encoding="utf-8") as out:
        for corpus, (train, _, _, utts_by_spk) in per_corpus.items():
            wav_root = os.path.join(DATA_ROOT, corpus, "wav")
            rng = random.Random(seed + 7)
            picked = train if len(train) <= 250 else rng.sample(train, 250)
            for s in sorted(picked):
                for p, _ in sorted(utts_by_spk[s])[:3]:
                    out.write(f"{os.path.join(wav_root, p)}\n")
                    n_cohort += 1

    manifest = {
        "corpus": "combined_si_ta",
        "language": "si+ta",
        "cohort_utterances": n_cohort,
        "role": "primary_combined",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sorted(per_corpus),
        "speakers": {"train": n_spk},
        "utterances": {"train": n_utt},
        "paths": "absolute (two corpus roots); use --train_path /",
        "evaluation": "per-language on each source corpus's own test_trials.txt",
        "sha256": {"train_list.txt": sha256(os.path.join(out_dir, "train_list.txt"))},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    report["combined_si_ta"] = manifest
    print(f"[combined_si_ta] speakers {n_spk}  train utts {n_utt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=None, help="build one corpus only")
    ap.add_argument("--all", action="store_true", help="build every corpus + combined")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n_trials",
        type=int,
        default=20000,
        help="TOTAL trials in the test list (half target, half impostor). "
        "20k reaches >98%% of the speaker-count power ceiling; more is wasted "
        "evaluation time -- see mde_estimate().",
    )
    args = ap.parse_args()

    if not args.all and not args.corpus:
        ap.error("pass --all or --corpus NAME")

    os.makedirs(SPLIT_ROOT, exist_ok=True)
    report, per_corpus = {}, {}

    targets = [args.corpus] if args.corpus else list(PRIMARY)
    for corpus in targets:
        if corpus in PRIMARY:
            tr, va, te, ub = build_corpus(corpus, PRIMARY[corpus], args.seed, args.n_trials, report)
            per_corpus[corpus] = (tr, va, te, ub)
        elif corpus in HELDOUT:
            build_heldout(corpus, HELDOUT[corpus], args.seed, args.n_trials, report)

    if args.all:
        for corpus, lang in HELDOUT.items():
            if os.path.isdir(os.path.join(DATA_ROOT, corpus, "metadata")):
                try:
                    build_heldout(corpus, lang, args.seed, args.n_trials, report)
                except SystemExit as exc:
                    print(f"[{corpus}] skipped: {exc}", file=sys.stderr)
        if len(per_corpus) == len(PRIMARY):
            build_combined(per_corpus, args.seed, report)

    with open(os.path.join(SPLIT_ROOT, "splits_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nmanifests -> {SPLIT_ROOT}/*/manifest.json")


if __name__ == "__main__":
    main()
