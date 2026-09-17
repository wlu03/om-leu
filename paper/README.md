# Paper source

Build:

```bash
tectonic -X compile main.tex        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Regenerate every table from the saved per-run artifacts (nothing in `tables/` is typed by hand):

```bash
../venv/bin/python make_tables.py          # reads ablation/*/results and the consequence artifact root
../venv/bin/python figures/make_fig1.py    # redraws Figure 1
```

`iclr2025_conference.{sty,bst}` is the most recent template the ICLR Master-Template repository
serves. Replace both files and the `\usepackage` line in `main.tex` with the current year's
template before submitting; nothing else changes.

Layout: main text ends on page 9, references begin on page 10, appendix follows.

Framing: the contribution is the identification procedure (control ladder, weight-matched
uninformative control, erasure of the attribute span, out-of-fold weight) and the finding it
returns, not the reader architecture, which an ordinary two-layer network matches. Regenerate
`tables/identification.tex` after any new `id_*` run.

Sources for the numbers:

| table | produced from |
|---|---|
| 1 main, 3 floor, 4 structural, 5 knock-outs, 6 protocol | `ablation/*/results/*.json` and the consequence artifact root |
| 7 improvements, 8 transfer, 9 curves, 10 weights | same |
| audit rates in Appendix A | `omleu_experiments/e1/validators.py` scan, reported in the text |
