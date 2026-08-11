# Thesis chapter templates

LaTeX chapter scaffolds for the SL-SPV MPhil thesis / journal paper.
Designed to be paired with [`../RUN_GUIDE.md`](../RUN_GUIDE.md), which
contains the experimental protocol that produces the numbers you'll
paste into these chapters.

## Files

| File | Status | Notes |
|---|---|---|
| `01_introduction.tex` | template | Motivation, problem statement, research questions, contributions, structure. Bracketed `[N=100]` etc. placeholders for your corpus stats. |
| `02_related_work.tex` | template | Five sub-fields (architectures, losses, cross-lingual transfer, scoring, SSL) + positioning of this work. `[CITE_OR_EXPAND: ...]` markers flag spots where you should add specific citations during the lit-review pass. |
| `03_methods.tex` | embedded in `../RUN_GUIDE.md §13` | Methods chapter template (corpus, architecture, training recipe, scoring backends, reproducibility). Already paper-ready. |
| `04_results.tex` | embedded in `../RUN_GUIDE.md §14` | Headline + ablation LaTeX table templates. Run `python tools/aggregate_seeds.py --latex` to populate. |
| `05_discussion.tex` | **TODO** (not scaffolded yet) | Per-feature contribution analysis, RQ answers, limitations. |
| `06_conclusion.tex` | **TODO** (not scaffolded yet) | Summary, future work pointing at FEATURE-009 SSL pretraining. |

## How to use these templates

1. **Replace every `[BRACKETED]` placeholder.** Each one marks a value
   you should fill in from your actual data, run results, or
   literature review. They're intentionally ugly so they're hard to
   miss during proof-reading.

2. **Replace every `[CITE_OR_EXPAND: ...]` marker.** These flag spots
   in Related Work where you should add specific citations or
   expand the prose. Each marker contains a hint about what should
   go there.

3. **Add to the `.bib` block** in `../RUN_GUIDE.md §16` as you cite
   new papers. Use the same key format (`author_year_keyword`).

4. **Compile order**: combine these chapters in a master `.tex` like:
   ```latex
   \documentclass[12pt,a4paper]{report}
   \usepackage{...}
   \begin{document}
   \include{thesis_chapters/01_introduction}
   \include{thesis_chapters/02_related_work}
   \include{thesis_chapters/03_methods}       % paste from RUN_GUIDE.md §13
   \include{thesis_chapters/04_results}       % paste from RUN_GUIDE.md §14
   \include{thesis_chapters/05_discussion}    % TODO
   \include{thesis_chapters/06_conclusion}    % TODO
   \bibliography{references}                  % bib from RUN_GUIDE.md §16
   \end{document}
   ```

5. **For the journal-paper version**, collapse chapters into sections.
   The chapter structure maps 1:1 to a standard journal-paper
   layout (Introduction, Related Work, Methods, Results,
   Discussion, Conclusion).

## What's *not* in these templates

- Empirical numbers — every result placeholder is bracketed. You fill
  them in after running the protocol in `RUN_GUIDE.md`.
- Figures — `[INSERT_FIGURE_X]` markers are not yet inserted (DET
  curves, per-language EER bar charts, t-SNE of embeddings will
  go in once you have results).
- A specific thesis class file — pick your institution's required
  LaTeX class (`report`, `book`, or a custom thesis class).
- The bibliography itself — extract from `../RUN_GUIDE.md §16` into
  `references.bib`.

## Related files

- `../RUN_GUIDE.md` — the experimental protocol that produces the
  numbers, plus the Methods/Results chapter templates and the
  `.bib` citation block.
- `../SL_LANGUAGE_SPV_ANALYSIS.md` — the audit document; useful as a
  source of historical context (what was broken, what was fixed,
  why each feature was added).
- `../docs/bugfixes/FEATURE-*.md` — per-feature design documents;
  cite from these as you describe each lever in Methods chapter.
