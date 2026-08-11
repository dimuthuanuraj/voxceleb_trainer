#!/usr/bin/env python3
"""
FEATURE-011 (closes §4.2 #12) — SL corpus prep.

Walks a corpus tree shaped like:

    <corpus_root>/
        <lang>/                 # one of: si, ta, en, mix
            <speaker_id>/
                <utterance>.wav
                ...

and generates all the list / lookup / cohort files the trainer needs.

Outputs (all in --out_dir):
    train_list.txt              <spk_label_int> <relative_path>
    test_list.txt               pooled test trial pairs (labeled)
    test_list_si.txt            per-language test pairs (FEATURE-003)
    test_list_ta.txt
    test_list_en.txt           (only if --langs includes 'en')
    test_list_cs.txt           (cross-lingual trials)
    spk_lang_lookup.txt         <spk_label_int> <lang_label_int>   (FEATURE-007 / 008)
    asnorm_cohort.txt           one cohort wav path per line       (FEATURE-002)
    plda_train_list.txt         subset of train_list               (FEATURE-010)
    speakers.csv                spk_label_int, original_id, lang, n_train, n_test

Reproducible: a single --seed controls the train/test split and trial sampling.

Trial-protocol controls (2026-07-03 roadmap, issues E1/E5):
- --session_from {none,parentdir,regex}: derive a recording-session key per
  utterance. Target (same-speaker) trials are then constrained to CROSS-SESSION
  pairs, avoiding the same-recording confound (8-17% of VoxCeleb1-H target
  pairs share a recording; Hutiri et al., Interspeech 2022). 'parentdir' uses
  the utterance's parent directory below the speaker dir (VoxCeleb-style
  <lang>/<spk>/<video>/<seg>.wav layouts). Speakers with a single session are
  excluded from target sampling and reported.
- --spk_meta <csv>: spk_id,lang,gender[,...] table (see tools/ingest_openslr.py).
  When given, impostor trials are same-gender wherever both genders are known.
- Trial pairs are deduplicated and a == b pairs are never emitted.

Usage:
    python tools/sl_dataprep.py \\
        --corpus_root /data/sl_celeb \\
        --out_dir     /data/sl_celeb/lists \\
        --langs       si ta \\
        --test_frac   0.2 \\
        --target_pairs_per_lang 1000 \\
        --impostor_ratio 1 \\
        --cohort_size 500 \\
        --seed 42

Conventions:
- Speaker IDs in the corpus are STRINGS (any format); the script assigns
  contiguous integer labels in the order they're first encountered.
- Speakers are NOT split into "enrol" vs "verify" sets (all 100 SL speakers
  go into train AND test pools). Train/test split is per-utterance within
  each speaker: --test_frac of utterances per speaker held out for trials.
- For each held-out test utterance, --target_pairs_per_lang/N target pairs
  (same speaker, same lang) and --impostor_ratio * that many impostor pairs
  (different speaker, same lang) are generated.
- Cross-lingual trials are special: target=same speaker across two languages,
  impostor=different speaker across two languages. Skipped if a speaker
  doesn't appear in both langs (per --langs).
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


# Canonical language label order (FEATURE-007 / 008 use these integers).
LANG_INT = {'si': 0, 'ta': 1, 'en': 2, 'mix': 3}


AUDIO_EXTS = ('.wav', '.flac')


def walk_corpus(root, langs):
    """Return dict: lang -> dict: spk_id -> list[relpath]."""
    out = defaultdict(lambda: defaultdict(list))
    root = Path(root)
    for lang in langs:
        lang_dir = root / lang
        if not lang_dir.is_dir():
            print(f"[sl_dataprep] warning: language dir {lang_dir} missing; skipping.",
                  file=sys.stderr)
            continue
        for spk_dir in sorted(lang_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            wavs = sorted([p for p in spk_dir.rglob('*')
                           if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
            if not wavs:
                continue
            spk_id = spk_dir.name
            rel = [str(w.relative_to(root)) for w in wavs]
            out[lang][spk_id].extend(rel)
    return out


def make_session_fn(mode, regex):
    """Return callable relpath -> session key, or None when sessions are
    not derivable (mode == 'none')."""
    if mode == 'none':
        return None
    if mode == 'parentdir':
        # <lang>/<spk>/.../<parent>/<utt>. Session = path between the speaker
        # dir and the file. Flat layouts (<lang>/<spk>/<utt>) yield '' for
        # every utterance -> single session per speaker (caller warns).
        def key(relpath):
            parts = Path(relpath).parts
            return '/'.join(parts[2:-1])
        return key
    if mode == 'regex':
        import re
        pat = re.compile(regex)
        def key(relpath):
            m = pat.search(relpath)
            return m.group(1) if m else ''
        return key
    raise ValueError(f"unknown session_from mode: {mode}")


def assign_spk_labels(corpus):
    """Stable integer labels: a speaker present in multiple languages gets
    ONE label across languages (so spk_lang_lookup treats the speaker as a
    single identity with one primary language taken from the language with
    more utterances). Returns:
        spk_to_label : dict (lang, spk_id) -> int_label
        primary_lang : dict int_label -> lang
        all_speakers : list[(lang, spk_id, label)]
    """
    # Discover unique speaker IDs across all languages.
    spk_utterance_counts = defaultdict(dict)  # spk_id -> lang -> count
    for lang, by_spk in corpus.items():
        for spk_id, utts in by_spk.items():
            spk_utterance_counts[spk_id][lang] = len(utts)

    label_of = {}     # spk_id -> int_label
    primary  = {}     # int_label -> lang
    for spk_id in sorted(spk_utterance_counts.keys()):
        label = len(label_of)
        label_of[spk_id] = label
        # Primary language = the one with most utterances.
        primary[label] = max(spk_utterance_counts[spk_id].items(),
                             key=lambda kv: kv[1])[0]
    return label_of, primary


def split_train_test(corpus, label_of, test_frac, rng):
    """Per-speaker, per-language: hold out test_frac of utterances for test.
    Returns:
        train_rows : list[(label, lang, relpath)]
        test_pool  : dict (label, lang) -> list[relpath]
    """
    train_rows = []
    test_pool = defaultdict(list)
    for lang, by_spk in corpus.items():
        for spk_id, utts in by_spk.items():
            label = label_of[spk_id]
            utts = list(utts)
            rng.shuffle(utts)
            n_test = max(1, int(round(len(utts) * test_frac)))
            test_utts = utts[:n_test]
            train_utts = utts[n_test:]
            if not train_utts:
                # Speaker had too few utterances; put one back in train.
                train_utts = [test_utts.pop()]
            for u in train_utts:
                train_rows.append((label, lang, u))
            test_pool[(label, lang)].extend(test_utts)
    return train_rows, test_pool


def _dedup_add(pairs, seen, label, a, b):
    """Append (label, a, b) unless a==b or the unordered pair was emitted."""
    if a == b:
        return False
    key = (a, b) if a <= b else (b, a)
    if key in seen:
        return False
    seen.add(key)
    pairs.append((label, a, b))
    return True


def sample_trial_pairs(test_pool, target_n, impostor_ratio, langs, rng,
                       cross_lingual=False, session_fn=None, gender_of=None,
                       max_attempts_factor=20):
    """Build a list of trial pair lines '<label_int> <a_path> <b_path>'.

    For each language in `langs`:
        target_pairs    = same speaker, same lang, two different utts
                          (cross-session when session_fn is given)
        impostor_pairs  = different speakers, same lang
                          (same-gender when gender_of is given and known)
    target count per language: target_n
    impostor count per language: target_n * impostor_ratio

    cross_lingual=True special case: build (lang1, lang2) pairs where target
    = same speaker present in both langs, impostor = different speakers in
    different langs. Returns a single list across all cross-lang pairs.

    Sampling is attempt-bounded: if the pool cannot supply the requested
    count under the constraints (small corpora), the function returns what
    it found and the caller reports the shortfall.
    """
    pairs = []  # list of (label, a, b)
    seen = set()

    def cross_session_sample(pool):
        """Two utts from pool with different session keys; None if impossible."""
        if session_fn is None:
            return rng.sample(pool, 2) if len(pool) >= 2 else None
        by_sess = defaultdict(list)
        for u in pool:
            by_sess[session_fn(u)].append(u)
        if len(by_sess) < 2:
            return None
        s1, s2 = rng.sample(list(by_sess.keys()), 2)
        return rng.choice(by_sess[s1]), rng.choice(by_sess[s2])

    if not cross_lingual:
        for lang in langs:
            spk_in_lang = [(spk, lang) for (spk, l) in test_pool if l == lang
                           and len(test_pool[(spk, l)]) >= 2]
            if len(spk_in_lang) < 2:
                continue
            # Speakers eligible for cross-session targets.
            if session_fn is not None:
                target_spk = [k for k in spk_in_lang
                              if len({session_fn(u) for u in test_pool[k]}) >= 2]
                n_single = len(spk_in_lang) - len(target_spk)
                if n_single:
                    print(f"[sl_dataprep]   {lang}: {n_single}/{len(spk_in_lang)} "
                          f"speakers single-session -> excluded from targets")
            else:
                target_spk = spk_in_lang
            # Target
            attempts = 0
            made = 0
            while made < target_n and attempts < target_n * max_attempts_factor:
                attempts += 1
                if not target_spk:
                    break
                spk_key = rng.choice(target_spk)
                got = cross_session_sample(test_pool[spk_key])
                if got and _dedup_add(pairs, seen, 1, got[0], got[1]):
                    made += 1
            if made < target_n:
                print(f"[sl_dataprep]   {lang}: only {made}/{target_n} target "
                      f"pairs possible under constraints")
            # Impostor (same-gender when known)
            n_imp = target_n * impostor_ratio
            attempts = 0
            made = 0
            while made < n_imp and attempts < n_imp * max_attempts_factor:
                attempts += 1
                spk_a, spk_b = rng.sample(spk_in_lang, 2)
                if gender_of is not None:
                    g_a, g_b = gender_of.get(spk_a[0]), gender_of.get(spk_b[0])
                    if g_a and g_b and g_a != 'unk' and g_b != 'unk' and g_a != g_b:
                        continue
                a = rng.choice(test_pool[spk_a])
                b = rng.choice(test_pool[spk_b])
                if _dedup_add(pairs, seen, 0, a, b):
                    made += 1
            if made < n_imp:
                print(f"[sl_dataprep]   {lang}: only {made}/{n_imp} impostor "
                      f"pairs possible under constraints")
    else:
        # Find speakers present in >=2 of the given langs.
        spk_to_langs = defaultdict(list)
        for (spk, lang) in test_pool:
            if lang in langs and test_pool[(spk, lang)]:
                spk_to_langs[spk].append(lang)
        multi_lang_spk = [s for s, ls in spk_to_langs.items() if len(ls) >= 2]
        if not multi_lang_spk:
            return pairs
        # Target: same speaker, two langs (inherently cross-session in any
        # sane corpus; still deduped and a != b enforced).
        attempts = 0
        made = 0
        while made < target_n and attempts < target_n * max_attempts_factor:
            attempts += 1
            spk = rng.choice(multi_lang_spk)
            l1, l2 = rng.sample(spk_to_langs[spk], 2)
            a = rng.choice(test_pool[(spk, l1)])
            b = rng.choice(test_pool[(spk, l2)])
            if _dedup_add(pairs, seen, 1, a, b):
                made += 1
        if made < target_n:
            print(f"[sl_dataprep]   cs: only {made}/{target_n} cross-lingual "
                  f"target pairs possible")
        # Impostor: different speakers, two langs (same-gender when known)
        n_imp = target_n * impostor_ratio
        attempts = 0
        made = 0
        while made < n_imp and attempts < n_imp * max_attempts_factor:
            attempts += 1
            s_a, s_b = rng.sample(multi_lang_spk, 2)
            if gender_of is not None:
                g_a, g_b = gender_of.get(s_a), gender_of.get(s_b)
                if g_a and g_b and g_a != 'unk' and g_b != 'unk' and g_a != g_b:
                    continue
            l_a = rng.choice(spk_to_langs[s_a])
            l_b = rng.choice(spk_to_langs[s_b])
            a = rng.choice(test_pool[(s_a, l_a)])
            b = rng.choice(test_pool[(s_b, l_b)])
            if _dedup_add(pairs, seen, 0, a, b):
                made += 1
    return pairs


def write_list(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(' '.join(str(x) for x in r) + '\n')


def main():
    p = argparse.ArgumentParser(description="SL corpus prep — FEATURE-011 / §4.2 #12.")
    p.add_argument('--corpus_root', required=True)
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--langs', nargs='+', default=['si', 'ta'],
                   help="Languages to include (default: si ta).")
    p.add_argument('--test_frac',     type=float, default=0.2)
    p.add_argument('--target_pairs_per_lang', type=int, default=1000)
    p.add_argument('--impostor_ratio',  type=int, default=1,
                   help="Impostor pairs per target pair (default 1 = balanced).")
    p.add_argument('--cohort_size',   type=int, default=500,
                   help="AS-Norm cohort size; sampled from train pool.")
    p.add_argument('--plda_train_speakers', type=int, default=0,
                   help="0 = use all train speakers for PLDA fit; "
                        "otherwise sample this many.")
    p.add_argument('--session_from', choices=['none', 'parentdir', 'regex'],
                   default='none',
                   help="Derive per-utterance session keys; target trials are "
                        "then cross-session only (issue E1).")
    p.add_argument('--session_regex', default=None,
                   help="Regex with one capture group applied to the relpath "
                        "(used with --session_from regex).")
    p.add_argument('--spk_meta', default=None,
                   help="spk_meta.csv (spk_id,lang,gender,...) enabling "
                        "same-gender impostor trials.")
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    session_fn = make_session_fn(args.session_from, args.session_regex)
    gender_of = None
    if args.spk_meta:
        gender_of = {}
        with open(args.spk_meta) as f:
            for row in csv.DictReader(f):
                gender_of[row['spk_id']] = row.get('gender', 'unk')
        n_known = sum(1 for g in gender_of.values() if g in ('m', 'f'))
        print(f"[sl_dataprep] spk_meta: gender known for {n_known}/{len(gender_of)} speakers")

    print(f"[sl_dataprep] Walking {args.corpus_root} for langs={args.langs} ...")
    corpus = walk_corpus(args.corpus_root, args.langs)
    n_spk_per_lang = {l: len(v) for l, v in corpus.items()}
    print(f"[sl_dataprep] Speakers per language: {n_spk_per_lang}")
    total_utts = sum(len(u) for by_spk in corpus.values() for u in by_spk.values())
    print(f"[sl_dataprep] Total utterances: {total_utts}")

    label_of, primary_lang = assign_spk_labels(corpus)
    n_speakers = len(label_of)
    print(f"[sl_dataprep] {n_speakers} unique speakers across languages.")

    # Re-key gender map by integer label (trial pools use labels, not IDs).
    gender_by_label = None
    if gender_of is not None:
        gender_by_label = {label_of[spk]: g for spk, g in gender_of.items()
                           if spk in label_of}

    train_rows, test_pool = split_train_test(corpus, label_of, args.test_frac, rng)
    # Train list: just <label> <path>. Strip language column for trainer compatibility.
    train_path = os.path.join(args.out_dir, 'train_list.txt')
    write_list(train_path, [(lbl, p) for (lbl, _lng, p) in train_rows])
    print(f"[sl_dataprep] train_list.txt: {len(train_rows)} utterances")

    # Pooled test list + per-language test lists.
    all_pairs = sample_trial_pairs(
        test_pool, args.target_pairs_per_lang, args.impostor_ratio, args.langs, rng,
        session_fn=session_fn, gender_of=gender_by_label,
    )
    write_list(os.path.join(args.out_dir, 'test_list.txt'), all_pairs)
    print(f"[sl_dataprep] test_list.txt (pooled): {len(all_pairs)} trial pairs")

    for lang in args.langs:
        per_lang = sample_trial_pairs(
            test_pool, args.target_pairs_per_lang, args.impostor_ratio, [lang], rng,
            session_fn=session_fn, gender_of=gender_by_label,
        )
        write_list(os.path.join(args.out_dir, f'test_list_{lang}.txt'), per_lang)
        print(f"[sl_dataprep] test_list_{lang}.txt: {len(per_lang)} pairs")

    # Cross-lingual trials (uses all configured langs).
    if len(args.langs) >= 2:
        cs_pairs = sample_trial_pairs(
            test_pool, args.target_pairs_per_lang, args.impostor_ratio,
            args.langs, rng, cross_lingual=True,
            session_fn=session_fn, gender_of=gender_by_label,
        )
        write_list(os.path.join(args.out_dir, 'test_list_cs.txt'), cs_pairs)
        print(f"[sl_dataprep] test_list_cs.txt (cross-lingual): {len(cs_pairs)} pairs")

    # FEATURE-007 / 008 lang lookup.
    lookup_path = os.path.join(args.out_dir, 'spk_lang_lookup.txt')
    with open(lookup_path, 'w') as f:
        f.write("# spk_label_int  lang_label_int (FEATURE-007 / FEATURE-008)\n")
        for label in range(n_speakers):
            lang = primary_lang[label]
            f.write(f"{label}  {LANG_INT.get(lang, 3)}\n")  # default to 'mix'=3
    print(f"[sl_dataprep] spk_lang_lookup.txt: {n_speakers} speakers")

    # FEATURE-002 AS-Norm cohort: one utterance per cohort speaker, sampled
    # from the TRAIN pool so it never overlaps with test trials.
    train_by_spk = defaultdict(list)
    for lbl, _lng, p in train_rows:
        train_by_spk[lbl].append(p)
    cohort_speakers = list(train_by_spk.keys())
    rng.shuffle(cohort_speakers)
    cohort_speakers = cohort_speakers[:args.cohort_size]
    cohort_lines = [rng.choice(train_by_spk[s]) for s in cohort_speakers]
    write_list(os.path.join(args.out_dir, 'asnorm_cohort.txt'), cohort_lines)
    print(f"[sl_dataprep] asnorm_cohort.txt: {len(cohort_lines)} utterances")

    # FEATURE-010 PLDA train list — subset of train_list (or all).
    if args.plda_train_speakers and args.plda_train_speakers < len(train_by_spk):
        plda_spks = rng.sample(list(train_by_spk.keys()), args.plda_train_speakers)
        plda_rows = [(lbl, p) for lbl, _lng, p in train_rows if lbl in set(plda_spks)]
    else:
        plda_rows = [(lbl, p) for (lbl, _lng, p) in train_rows]
    write_list(os.path.join(args.out_dir, 'plda_train_list.txt'), plda_rows)
    print(f"[sl_dataprep] plda_train_list.txt: {len(plda_rows)} utterances")

    # speakers.csv — reproducibility metadata.
    spk_csv = os.path.join(args.out_dir, 'speakers.csv')
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for lbl, _l, _p in train_rows:
        train_counts[lbl] += 1
    for (lbl, _l), utts in test_pool.items():
        test_counts[lbl] += len(utts)
    inv_label = {lbl: spk_id for spk_id, lbl in label_of.items()}
    with open(spk_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['spk_label_int', 'original_id', 'primary_lang', 'n_train', 'n_test'])
        for lbl in range(n_speakers):
            w.writerow([lbl, inv_label[lbl], primary_lang[lbl],
                        train_counts[lbl], test_counts[lbl]])
    print(f"[sl_dataprep] speakers.csv: {n_speakers} rows")

    # Summary banner.
    print()
    print(f"[sl_dataprep] ===== Done. Outputs in {args.out_dir} =====")
    print(f"  nClasses for trainer: --nClasses {n_speakers}")
    print(f"  paths.env:    export SL_SPV_DATA_ROOT={args.corpus_root}")


if __name__ == '__main__':
    main()
