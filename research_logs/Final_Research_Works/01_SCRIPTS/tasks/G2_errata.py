#!/usr/bin/env python3
"""G2-G6 -- Errata and withdrawals. Annual report section 10.3.

These are OBLIGATIONS, not opportunities. Section 10.3 lists six things that
"must be corrected or withdrawn". This task produces a single authoritative
errata document and, critically, a machine-readable list of forbidden claims, so
that the correction does not depend on each author remembering it.

The six items:

  1. ALL January-June 2026 experimental results. Not reproducible from the
     repository, contradicted by the project's own audit of 2026-07-03. They
     cannot enter the thesis, any paper, or any future progress report.
  2. The nested-learning "9.84 % EER, validated for efficiency" claim -- the
     measured record is three NaN collapses.
  3. The Phase I 10.32 % headline -- unreplicated, single seed, predates
     --deterministic, and carries an unresolved internal conflict (14.62 % in one
     log) plus a suspected duplicated result set. Handled by G1.
  4. The Period 2 report's ResNetSE34L specification (34 layers / 6.8 M /
     ~8-10 % EER). The logs give 1.50 M and 15.48 %.
  5. The "Tamil pool 50 -> 752" figure in the 17 August report. The measured
     build is 706 (802 after QC corrections).
  6. Every v0 EER should carry the ~5x read-vs-wild optimism factor
     retrospectively, and the closed-set caveat explicitly.

Why a machine-readable list: section 10.3 exists because claims that should have
been withdrawn were not. A grep-able registry of forbidden numbers is the only
mechanism that survives an author who has not read the annual report.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRW = os.path.dirname(os.path.dirname(HERE))
SL_SPV = os.path.abspath(os.path.join(FRW, "..", "..", ".."))
OUT = os.path.join(FRW, "03_RESULTS", "G2-G6")
REPORTS = os.path.join(FRW, "04_REPORTS")

FORBIDDEN = [
    {"id": "E1", "claim": "Any experimental result dated January-June 2026",
     "pattern": None, "status": "WITHDRAWN",
     "reason": "Not reproducible from the repository; contradicted by the project audit of 2026-07-03.",
     "source": "annual report 10.3 item 1, section 5",
     "replacement": "None. This period has no citable experimental record."},
    {"id": "E2", "claim": "Nested learning: 9.84 % EER, validated for efficiency",
     "pattern": r"(?<![\d.])9\.84(?![\d])", "status": "WITHDRAWN",
     "reason": "The measured record is three NaN collapses across three attempts.",
     "source": "annual report 10.3 item 2, section 3.6",
     "replacement": "NestedSpeakerNet failed: 3 attempts, NaN x2, 1.09x faster at best."},
    {"id": "E3", "claim": "Phase I MLP-Mixer student 10.32 % EER (headline)",
     "pattern": r"(?<![\d.])10\.32(?![\d])", "status": "UNDER REVIEW (G1)",
     "reason": "Single seed, unreplicated, predates --deterministic; one log in the same "
               "record gives 14.62 %; a duplicated result set is suspected.",
     "source": "annual report 10.3 item 3; audit item R7",
     "replacement": "Pending G1: either 3 seeds with an interval, or formal retirement."},
    {"id": "E4", "claim": "ResNetSE34L is 34 layers / 6.8 M params / ~8-10 % EER",
     "pattern": r"6\.8\s*M|34 layers", "status": "CORRECTED",
     "reason": "The logs give 1.50 M parameters and 15.48 % EER.",
     "source": "annual report 10.3 item 4",
     "replacement": "ResNetSE34L: 1.50 M parameters, 15.48 % EER."},
    {"id": "E5", "claim": "Tamil speaker pool expanded 50 -> 752",
     "pattern": r"(?<![\d.])752\b(?!\s*%)", "status": "CORRECTED",
     "reason": "The measured build is 706 speakers, 802 after QC corrections.",
     "source": "annual report 10.3 item 5",
     "replacement": "Tamil pool expanded 50 -> 706 (802 after QC correction)."},
    {"id": "E6", "claim": "Any v0 EER quoted without caveats",
     "pattern": None, "status": "CAVEAT REQUIRED",
     "reason": "v0 lists were 100 % closed-set, and read speech is ~5x optimistic "
               "against genuine cross-session audio.",
     "source": "annual report 10.3 item 6, sections 6.8.1 and 6.12",
     "replacement": "Every v0 EER must carry both the closed-set caveat and the "
                    "~5x read-vs-wild optimism factor."},
]

SCAN_GLOBS = [
    "voxceleb_trainer/research_logs/*.md",
    "voxceleb_trainer/papers/*/*.tex",
    "voxceleb_trainer/thesis_chapters/*.tex",
    "journal_paper/*.md",
    "*/*.md",
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    print("  scanning project documents for forbidden numbers...\n")
    hits: dict[str, list] = {}
    files: list[str] = []
    for g in SCAN_GLOBS:
        files.extend(glob.glob(os.path.join(SL_SPV, g)))
    files = sorted(set(f for f in files if "Final_Research_Works" not in f))
    print(f"  {len(files)} documents in scope\n")

    for item in FORBIDDEN:
        if not item["pattern"]:
            continue
        rx = re.compile(item["pattern"])
        for f in files:
            try:
                txt = open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            for n, line in enumerate(txt.splitlines(), 1):
                if rx.search(line):
                    hits.setdefault(item["id"], []).append(
                        {"file": os.path.relpath(f, SL_SPV), "line": n,
                         "text": line.strip()[:160]})

    for item in FORBIDDEN:
        h = hits.get(item["id"], [])
        flag = f"{len(h)} occurrence(s)" if h else "no textual occurrences"
        print(f"  {item['id']}  {item['status']:18s} {flag}")
        for x in h[:4]:
            print(f"        {x['file']}:{x['line']}  {x['text'][:96]}")
        if len(h) > 4:
            print(f"        ... and {len(h) - 4} more")

    # ---------------------------------------------------------------- errata doc
    doc = [
        "---",
        'title: "SL_SPV Errata and Withdrawals"',
        'subtitle: "Corrections required by the annual report, section 10.3"',
        'author: "Dimuthu Anuraj"',
        f'date: "{stamp[:10]}"',
        "---",
        "",
        "# Why this document exists",
        "",
        "The annual research progress report of 2026-09-10 lists six claims that",
        "**must be corrected or withdrawn**. This document is the single",
        "authoritative record of those corrections. Every other project document",
        "should reference this one rather than restating it, so that a correction",
        "made once is a correction made everywhere.",
        "",
        "The register exists because section 10.3 was itself written after claims",
        "that should have been withdrawn were carried forward into later documents",
        "in good faith. A grep-able list is the only mechanism that survives an",
        "author who has not read the annual report.",
        "",
        "# The register",
        "",
    ]
    for item in FORBIDDEN:
        h = hits.get(item["id"], [])
        doc += [
            f"## {item['id']} --- {item['status']}", "",
            f"**Claim:** {item['claim']}", "",
            f"**Why it is withdrawn or corrected:** {item['reason']}", "",
            f"**Replacement:** {item['replacement']}", "",
            f"*Source: {item['source']}*", "",
        ]
        if h:
            doc += [f"**{len(h)} occurrence(s) found in project documents:**", ""]
            doc += [f"- `{x['file']}:{x['line']}`" for x in h[:25]]
            if len(h) > 25:
                doc.append(f"- ... and {len(h) - 25} more")
            doc.append("")
        else:
            doc += ["No textual occurrence found in the scanned documents.", ""]

    doc += [
        "# The standing rule",
        "",
        "> **No number from January to June 2026 may enter the thesis, any paper,",
        "> or any future progress report.**",
        "",
        "This is not a judgement about whether the work happened. It is a statement",
        "about evidence: those results are not reproducible from the repository and",
        "are contradicted by the project's own audit. The audit was written by the",
        "project about the project, and nothing external forced it --- which is the",
        "reason every number from July 2026 onward can be quoted.",
        "",
        f"*Generated {stamp} by `01_SCRIPTS/tasks/G2_errata.py`. Re-run it after",
        "editing any project document to re-scan for reintroduced claims.*",
        "",
    ]

    errata_path = os.path.join(REPORTS, "ERRATA.md")
    open(errata_path, "w", encoding="utf-8").write("\n".join(doc))

    json.dump({"task": "G2-G6", "ok": True, "generated": stamp,
               "documents_scanned": len(files),
               "forbidden_claims": FORBIDDEN,
               "occurrences": hits,
               "errata_document": os.path.relpath(errata_path, FRW)},
              open(os.path.join(OUT, "result.json"), "w"), indent=2)
    json.dump(FORBIDDEN, open(os.path.join(OUT, "forbidden_claims.json"), "w"), indent=2)

    print(f"\n  -> {errata_path}")
    print(f"  -> {OUT}/result.json")
    print(f"  -> {OUT}/forbidden_claims.json  (machine-readable, for CI or a pre-commit check)")
    print("\n  G1 (the Phase I R7 re-run) is tracked separately -- it needs GPUs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
