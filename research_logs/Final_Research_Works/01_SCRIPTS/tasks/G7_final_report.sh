#!/usr/bin/env bash
# G7 -- The final consolidated research report.
#
# Assembles every result this plan produced into one document, in the same
# evidence-tagged style as the annual report, and -- crucially -- scores every
# pre-registered prediction whichever way it came out.
#
# Requires A5, B3, C5 and G2-G6 to have passed, because those four are the ones
# that decide what may be CLAIMED rather than merely reported.

. "$(dirname "$0")/../lib/common.sh"

OUT="$REPORTS/final_research_report.md"
mkdir -p "$REPORTS"

step "Gathering every task result"
python3 - "$OUT" <<'PY'
import datetime, glob, json, os, sys

out_path = sys.argv[1]
FRW = os.environ["FRW_ROOT"]
RES = os.environ["RESULTS"]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

tasks = json.load(open(os.path.join(os.environ["SCRIPTS"], "tasks.json")))
ledger_p = os.path.join(os.environ["STATE"], "ledger.json")
ledger = json.load(open(ledger_p)) if os.path.exists(ledger_p) else {}
preds_p = os.path.join(os.environ["STATE"], "predictions.json")
preds = json.load(open(preds_p)) if os.path.exists(preds_p) else {}

results = {}
for f in sorted(glob.glob(os.path.join(RES, "*", "result.json"))):
    tid = os.path.basename(os.path.dirname(f))
    try:
        results[tid] = json.load(open(f))
    except Exception:
        pass

L = []
A = L.append
A("---")
A('title: "SL_SPV Final Research Works --- Consolidated Report"')
A('subtitle: "Every task in the completion plan, its evidence, and its outcome"')
A('author: "Dimuthu Anuraj"')
A(f'date: "{stamp[:10]}"')
A("toc: true"); A("toc-depth: 3"); A("---"); A("")

A("# 1. What this report is")
A("")
A("The annual report of 2026-09-10 stated what had been achieved and listed what")
A("remained open. This report states what the completion plan then did about it.")
A("")
A("It uses the same evidence standard: **[M]** measured with an artefact in the")
A("repository, **[R]** reported, **[U]** unverified, **[P]** predicted. Anything")
A("without an artefact under `03_RESULTS/` is not tagged **[M]** here.")
A("")

A("# 2. Task ledger")
A("")
A("| ID | Wave | Task | Status | Evidence |")
A("|---|---|---|---|---|")
for t in tasks["tasks"]:
    tid = t["id"]
    st = ledger.get(tid, {}).get("status", "not started")
    r = results.get(tid, {})
    ev = "`03_RESULTS/%s/result.json`" % tid if r else "---"
    ok = r.get("ok")
    mark = "**[M]**" if ok else ("[P]" if st == "not started" else "[R]")
    A(f"| {tid} | {t['wave']} | {t['title'][:58]} | {st} {mark} | {ev} |")
A("")

A("# 3. Results by wave")
A("")
for wave, title in [("W0", "Unblocking"), ("A", "Harvesting work already paid for"),
                    ("B", "The arbiter and the statistics"),
                    ("C", "Zero-training analyses"), ("D", "Retrains"),
                    ("E", "The T-series"), ("F", "Phase II and open questions"),
                    ("G", "Corrections and consolidation")]:
    ids = [t["id"] for t in tasks["tasks"] if t["wave"] == wave]
    got = [i for i in ids if results.get(i, {}).get("ok")]
    if not got:
        continue
    A(f"## Wave {wave} --- {title}")
    A("")
    for i in got:
        r = results[i]
        A(f"### {i}")
        A("")
        A("```json")
        A(json.dumps({k: v for k, v in r.items() if k not in ("probe", "verification")},
                     indent=2)[:2600])
        A("```")
        A("")

A("# 4. Pre-registered predictions, scored")
A("")
if preds:
    A("| # | Prediction | Outcome | Verdict |")
    A("|---|---|---|---|")
    for k, v in sorted(preds.items()):
        A(f"| {k} | {v.get('prediction','')[:70]} | {v.get('outcome','pending')[:50]} "
          f"| {v.get('verdict','---')} |")
else:
    A("`02_STATE/predictions.json` is empty. **Every prediction in")
    A("`00_PLANNING/03_EXECUTION_PLAN.md` section 6 must be scored here, including")
    A("the falsified ones.** The annual report scores ~40 % of its predictions as")
    A("falsified and argues that rate is a healthy sign rather than a poor one:")
    A("it means the predictions were specific enough to be wrong, and that they")
    A("were scored rather than rationalised. A plan that drops its own falsified")
    A("predictions would be the first regression from that standard.")
A("")

A("# 5. What is now safe to claim")
A("")
A("Fill this from section 3, applying the same rule the annual report used: a")
A("claim is safe only if it is held-out test EER on speaker-disjoint splits,")
A("from a model verified to match its own checkpoint, with model selection made")
A("on validation trials only and the test set scored once.")
A("")
A("Carry forward unchanged from the annual report section 10.2 unless a task in")
A("this plan has since resolved it:")
A("")
A("- single-seed results (resolved by B4/B5/B6 where those have passed);")
A("- any absolute EER from a single corpus;")
A("- **any Sri Lankan Tamil claim** --- unresolved until H1 lands;")
A("- T1's transformer contrast (resolved by A4 + A5 + B6 where passed).")
A("")

A("# 6. Outstanding")
A("")
A("| ID | Task | Blocked by |")
A("|---|---|---|")
for t in tasks["tasks"]:
    if not results.get(t["id"], {}).get("ok"):
        dep = ", ".join(t["deps"]) or "---"
        A(f"| {t['id']} | {t['title'][:58]} | {dep} |")
A("")
A("## Not scriptable")
A("")
A("| ID | Task | Owner |")
A("|---|---|---|")
for m in tasks.get("manual_tracking", []):
    A(f"| {m['id']} | {m['title'][:64]} | {m['owner']} |")
A("")
A(f"*Generated {stamp} by `01_SCRIPTS/tasks/G7_final_report.sh`. Re-run after any")
A("task completes; it reads the filesystem, so it cannot drift from what exists.*")

open(out_path, "w", encoding="utf-8").write("\n".join(L))
print(f"  tasks in ledger : {len(ledger)}")
print(f"  results found   : {len(results)}")
print(f"  predictions     : {len(preds)}")
print(f"\n  -> {out_path}")
PY

step "Building the PDF"
if command -v pandoc >/dev/null 2>&1; then
    # The voiceid doc pipeline's hard-won lesson: there is no xelatex binary here,
    # but the format file survives, so xetex must be driven as xelatex.
    if pandoc "$OUT" -o "${OUT%.md}.pdf" --pdf-engine=xelatex 2>"$REPORTS/.pandoc.log"; then
        log "PDF built: ${OUT%.md}.pdf"
    else
        warn "xelatex failed; trying the xetex format-file workaround"
        pandoc "$OUT" -o "${OUT%.md}.pdf" \
            --pdf-engine=xetex --pdf-engine-opt=-progname=xelatex \
            2>>"$REPORTS/.pandoc.log" && log "PDF built via xetex" || \
            warn "PDF build failed -- the markdown is the deliverable; see .pandoc.log"
    fi
else
    warn "pandoc not found -- markdown only"
fi

log "final report: $OUT"
