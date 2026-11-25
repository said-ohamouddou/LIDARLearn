# Automated reports


Reports turn LIDARLearn from a training library into a **full research pipeline**. Run a sweep, call one report script, and get a **typeset LaTeX document + CSV + Markdown**, ready for a paper.

No hand-built tables, no copying numbers into Overleaf, no formatting drift when you rerun.

**See the actual output:** [**`docs/model_comparison_tables.pdf`**](https://github.com/said-ohamouddou/LIDARLearn/blob/main/docs/model_comparison_tables.pdf) — generated end-to-end by this library from raw training logs.

All report generators live in `scripts/reports/` and emit a standalone,
`pdflatex`-compilable `.tex` (plus CSV + Markdown) into `<exp_dir>/latex/`:

| Script | Produces |
|---|---|
| `classification_comparison_report.py` | CV classification table (one combined table) |
| `classification_comparison_split_report.py` | CV classification table split by SSL init source |
| `partseg_comparison_report.py` | ShapeNetParts part-seg comparison table |
| `semseg_comparison_report.py` | S3DIS semantic-seg summary + optional per-class IoU table (`--per_class`) |
| `fewshot_comparison_report.py` | Point-BERT-style few-shot table (5/10-way × 10/20-shot) |
| `friedman_significance_report.py` | Friedman + Nemenyi + Wilcoxon + CD-diagram report (written to `<exp_dir>/friedman/`) |
| `model_citations_report.py` | Standalone citation paragraph for every supported method |

Every generator accepts `--use_citation_in_tables` (inline `\citep{}` in each
cell). When omitted (default) a single `\paragraph{Methods.}` is prepended
above the tables. Bib keys are resolved from `scripts/reports/references.bib`.

**Best-in-column highlighting**

- In every numeric column, the best value is bold: `\textbf{}` in LaTeX, `**…**` in Markdown.
- "Best" is the highest accuracy, IoU, F1, recall or precision, and the lowest parameter count or epoch time.
- Ties are all bolded.

Each report folder also gets a copy of `references.bib` (from `scripts/reports/references.bib`), so the `.tex` files compile standalone with `pdflatex` + `bibtex` — drop them straight into a paper submission.
