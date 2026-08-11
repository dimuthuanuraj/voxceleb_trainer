#!/usr/bin/env python3
"""
RUN_GUIDE.md companion — aggregate VEER / MinDCF / RawVEER (FEATURE-002) /
per-language EER (FEATURE-003) across multiple-seed runs.

Walks `exps/<prefix>_seed*/result/scores.txt`, picks the best-EER epoch
for each seed, then reports mean +- std across seeds. Prints a paper-
ready table (markdown by default, --latex switches to LaTeX).

Usage:
    python tools/aggregate_seeds.py
    python tools/aggregate_seeds.py --prefixes P0_ecapa_baseline P1_ecapa_finetune P2_full_stack
    python tools/aggregate_seeds.py --exps_dir my_runs --latex
"""

import argparse
import glob
import os
import re
import statistics
import sys


# Regex covers both standard ("VEER X.XXXX, ...MinDCF Y.YYYYY...") and
# FEATURE-002 dual-line ("Raw VEER ... Raw MinDCF ...") banner formats.
RE_EER       = re.compile(r"VEER\s+([\d.]+),.*?MinDCF\s+([\d.]+)")
RE_RAW       = re.compile(r"RawVEER\s+([\d.]+),\s*RawMinDCF\s+([\d.]+)")
RE_PER_LANG  = re.compile(r"\[per-lang\s+([^\]]+)\]\s+VEER\s+([\d.]+),.*?MinDCF\s+([\d.]+)")


def best_for_run(scores_path):
    """Return dict of best-epoch metrics from a single run's scores.txt."""
    out = {'eer': None, 'mindcf': None, 'raw_eer': None, 'raw_mindcf': None,
           'per_lang': {}}
    try:
        text = open(scores_path).read()
    except FileNotFoundError:
        return out
    pairs = RE_EER.findall(text)
    if pairs:
        best = min(pairs, key=lambda t: float(t[0]))   # lowest EER
        out['eer']    = float(best[0])
        out['mindcf'] = float(best[1])
    raw_pairs = RE_RAW.findall(text)
    if raw_pairs:
        best = min(raw_pairs, key=lambda t: float(t[0]))
        out['raw_eer']    = float(best[0])
        out['raw_mindcf'] = float(best[1])
    # Per-lang: take the best appearance of each language label.
    for lang, eer_s, dcf_s in RE_PER_LANG.findall(text):
        eer, dcf = float(eer_s), float(dcf_s)
        prev = out['per_lang'].get(lang)
        if prev is None or eer < prev[0]:
            out['per_lang'][lang] = (eer, dcf)
    return out


def aggregate_prefix(exps_dir, prefix):
    """Aggregate all `<prefix>_seed*` runs under exps_dir."""
    pattern = os.path.join(exps_dir, f'{prefix}_seed*', 'result', 'scores.txt')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    all_metrics = [best_for_run(p) for p in matches]
    seeds_found = [re.search(r'_seed(\d+)', p).group(1) for p in matches]

    def collect(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if not vals:
            return None
        if len(vals) == 1:
            return (vals[0], 0.0, len(vals))
        return (statistics.mean(vals), statistics.stdev(vals), len(vals))

    per_lang_agg = {}
    all_langs = set()
    for m in all_metrics:
        all_langs.update(m['per_lang'].keys())
    for lang in sorted(all_langs):
        eers = [m['per_lang'][lang][0] for m in all_metrics if lang in m['per_lang']]
        if not eers:
            continue
        if len(eers) == 1:
            per_lang_agg[lang] = (eers[0], 0.0)
        else:
            per_lang_agg[lang] = (statistics.mean(eers), statistics.stdev(eers))

    return {
        'seeds':       seeds_found,
        'n':           len(matches),
        'eer':         collect('eer'),
        'mindcf':      collect('mindcf'),
        'raw_eer':     collect('raw_eer'),
        'raw_mindcf':  collect('raw_mindcf'),
        'per_lang':    per_lang_agg,
    }


def fmt_meanstd(t, fmt='.3f'):
    if t is None:
        return '-'
    m, s, _n = t
    return f'{m:{fmt}} +- {s:{fmt}}'


def print_markdown(rows):
    # Build column set from observed per-lang entries.
    langs = sorted({l for r in rows for l in (r['agg']['per_lang'] or {})})
    print(f'| Config | n seeds | EER % | MinDCF | Raw EER % | Raw MinDCF |'
          + ''.join(f' EER % ({l}) |' for l in langs))
    print('|---' * (5 + len(langs)) + '|')
    for r in rows:
        a = r['agg']
        cells = [
            r['prefix'],
            str(a['n']),
            fmt_meanstd(a['eer']),
            fmt_meanstd(a['mindcf'], '.5f'),
            fmt_meanstd(a['raw_eer']),
            fmt_meanstd(a['raw_mindcf'], '.5f'),
        ]
        for lang in langs:
            cells.append(
                f"{a['per_lang'][lang][0]:.3f} +- {a['per_lang'][lang][1]:.3f}"
                if lang in a['per_lang'] else '-'
            )
        print('| ' + ' | '.join(cells) + ' |')


def print_latex(rows):
    langs = sorted({l for r in rows for l in (r['agg']['per_lang'] or {})})
    cols = 'l' + 'c' * (5 + len(langs))
    print('\\begin{tabular}{' + cols + '}')
    print('\\toprule')
    header = ['Config', '$n$', 'EER \\%', 'MinDCF', 'Raw EER \\%', 'Raw MinDCF']
    header += [f'EER \\% ({lang})' for lang in langs]
    print(' & '.join(header) + ' \\\\')
    print('\\midrule')
    for r in rows:
        a = r['agg']
        cells = [
            r['prefix'].replace('_', r'\_'),
            str(a['n']),
            fmt_meanstd(a['eer']).replace('+-', r'$\pm$'),
            fmt_meanstd(a['mindcf'], '.5f').replace('+-', r'$\pm$'),
            fmt_meanstd(a['raw_eer']).replace('+-', r'$\pm$'),
            fmt_meanstd(a['raw_mindcf'], '.5f').replace('+-', r'$\pm$'),
        ]
        for lang in langs:
            if lang in a['per_lang']:
                m, s = a['per_lang'][lang]
                cells.append(f'{m:.3f} $\\pm$ {s:.3f}')
            else:
                cells.append('-')
        print(' & '.join(cells) + ' \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exps_dir', default='exps')
    p.add_argument('--prefixes', nargs='+',
                   default=['P0_ecapa_baseline', 'P1_ecapa_finetune', 'P2_full_stack'],
                   help="Prefix (without _seedN) of each config's runs.")
    p.add_argument('--latex', action='store_true', help='LaTeX table output.')
    args = p.parse_args()

    rows = []
    for prefix in args.prefixes:
        agg = aggregate_prefix(args.exps_dir, prefix)
        if agg is None:
            print(f"[aggregate_seeds] {prefix}: no runs found at "
                  f"{args.exps_dir}/{prefix}_seed*/result/scores.txt",
                  file=sys.stderr)
            continue
        rows.append({'prefix': prefix, 'agg': agg})

    if not rows:
        print("[aggregate_seeds] No runs aggregated; exiting.", file=sys.stderr)
        sys.exit(1)

    if args.latex:
        print_latex(rows)
    else:
        print_markdown(rows)


if __name__ == '__main__':
    main()
