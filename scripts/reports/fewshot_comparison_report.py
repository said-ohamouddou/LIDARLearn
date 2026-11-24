#!/usr/bin/env python3
"""
Generate Few-Shot Classification Table from CV Summary Files

Scans an experiment directory for few-shot runs (each produced by
`python main.py --run_all_folds`) and builds a Point-BERT-style table:

    +-----------+---------------+---------------+
    |           |    5-way      |    10-way     |
    |           |  10-shot 20   |  10-shot 20   |
    +-----------+---------------+---------------+
    | Model A   | 94.6±3.1 ...  | 91.0±5.4 ...  |
    | Model B   | ...           | ...           |
    +-----------+---------------+---------------+

For each experiment, the (way, shot) combo is parsed from the saved
`config.yaml` (dataset.train.others.way / .shot). The cell value is
the `{metric}_mean` ± `{metric}_std` from `cv_summary.csv`.

Usage
-----
    # Default (metric=best_accuracy, scans experiments/ModelNetFewShot/)
    python scripts/reports/fewshot_comparison_report.py \\
        --exp_dir experiments/ModelNetFewShot

    # Custom metric + output dir
    python scripts/reports/fewshot_comparison_report.py \\
        --exp_dir experiments/ModelNetFewShot \\
        --metric f1_macro \\
        --output_dir tables/fewshot

Output (under --output_dir):
    fewshot_table_<metric>.tex
    fewshot_table_<metric>.csv
    fewshot_table_<metric>.md
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _citations import (
    cite_inline,
    build_citation_paragraph,
    render_grouped_table,
)


# -----------------------------------------------------------------------------
# Model ordering / citations — reused from generate_comparison_table_one_table
# -----------------------------------------------------------------------------
MODEL_ORDER = [
    # Supervised
    'PointNet', 'PointNet2', 'DGCNN', 'PCT', 'PointMLP', 'CurveNet', 'DeepGCN',
    'DELA', 'RSCNN', 'PointConv', 'PointWeb', 'SONet', 'RepSurf', 'PointCNN',
    'PointSCNet', 'GDAN', 'PPFNet', 'PVT', 'PointKAN', 'KANDGCNN',
    'MSDGCNN', 'MSDGCNN2',
    'PointTransformer', 'PointTransformerV2', 'PointTransformerV3',
    'P2P', 'PointTNT', 'GlobalTransformer',
    # SSL
    'PointMAE', 'PointBERT', 'PointGPT', 'ACT', 'RECON', 'PCP', 'PointM2AE', 'PointMamba',
]

MODEL_CITATIONS = {
    'PointNet': 'qi2017pointnet', 'PointNet2': 'qi2017pointnet++',
    'DGCNN': 'wang2019dynamic', 'PCT': 'guo2021pct', 'PointMLP': 'pointmlp',
    'CurveNet': 'curvenet', 'DeepGCN': 'deepgcn', 'DELA': 'dela',
    'RSCNN': 'rscnn', 'PointConv': 'pointconv', 'PointWeb': 'pointweb',
    'SONet': 'sonet', 'RepSurf': 'repsurf', 'PointCNN': 'pointcnn',
    'PointSCNet': 'pointscnet', 'GDAN': 'gdanet', 'GDANet': 'gdanet',
    'PPFNet': 'ppfnet', 'PVT': 'pvt', 'PointKAN': 'pointkan',
    'KANDGCNN': 'kandgcnn', 'MSDGCNN': 'msdgcnn', 'MSDGCNN2': 'msdgcnn2',
    'PointTransformer': 'zhao2021point',
    'PointTransformerV2': 'wu2022point', 'PointTransformerV3': 'wu2024point',
    'P2P': 'p2p', 'PointTNT': 'pointtnt', 'GlobalTransformer': 'pointtnt',
    'PointMAE': 'pang2022masked', 'PointBERT': 'yu2022point',
    'PointGPT': 'chen2024pointgpt', 'ACT': 'dong2023act',
    'RECON': 'qi2023contrast', 'PCP': 'pcpmae', 'PointMamba': 'liang2024pointmamba', 'PointM2AE': 'zhang2022point',
}

# Way/shot combos in the order they appear in the Point-BERT table
COMBOS = [(5, 10), (5, 20), (10, 10), (10, 20)]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def strip_init_source(name: str) -> str:
    return re.sub(r'\s*\[.*?\]\s*$', '', str(name)).strip()


def get_base_model_name(name: str) -> str:
    clean = strip_init_source(name)
    if ':' in clean:
        return clean.split(':', 1)[0].strip()
    return clean


def normalize(name: str) -> str:
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def get_sort_key(base_model: str) -> int:
    if base_model in MODEL_ORDER:
        return MODEL_ORDER.index(base_model)
    norm = normalize(base_model)
    for i, m in enumerate(MODEL_ORDER):
        if normalize(m) == norm:
            return i
    return len(MODEL_ORDER)


def get_citation(base_model: str) -> str:
    if base_model in MODEL_CITATIONS:
        return MODEL_CITATIONS[base_model]
    norm = normalize(base_model)
    for m, cite in MODEL_CITATIONS.items():
        if normalize(m) == norm:
            return cite
    return norm


def parse_mean_std(value):
    """Parse 'mean ± std' (or '±' / '+/-' / '$\\pm$') into (mean, std) floats."""
    if pd.isna(value):
        return None, None
    try:
        m = re.match(
            r'([\d.]+)\s*(?:±|\+/-|\$\\pm\$|\\pm)\s*([\d.]+)',
            str(value).strip(),
        )
        if m:
            return float(m.group(1)), float(m.group(2))
        return float(value), 0.0
    except (ValueError, AttributeError):
        return None, None


def load_way_shot_from_config(exp_dir: Path):
    """Return (way, shot) from config.yaml in a fold subdir or exp root."""
    candidates = [
        exp_dir / 'config.yaml',
        exp_dir / 'fold_0' / 'config.yaml',
    ]
    # Also glob for any config.yaml inside the exp subtree (first match wins)
    candidates.extend(sorted(exp_dir.glob('**/config.yaml')))

    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            others = cfg['dataset']['train']['others']
            return int(others['way']), int(others['shot'])
        except (KeyError, TypeError, ValueError):
            continue
    return None, None


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------
def find_fewshot_summaries(exp_dir: Path):
    """Yield (exp_subdir, cv_summary_path) for every few-shot run under exp_dir."""
    patterns = [
        exp_dir / '*' / 'cv_summary.csv',
        exp_dir / '*' / '*' / 'cv_summary.csv',
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(str(pat)):
            path = Path(p).resolve()
            if path in seen:
                continue
            seen.add(path)
            yield path.parent, path


def collect_results(exp_dir: Path, metric: str):
    """Return dict {(base_model, strategy): {(way, shot): 'mean$\\pm$std'}}."""
    results = {}
    per_run_rows = []  # for csv dump

    for run_dir, cv_path in find_fewshot_summaries(exp_dir):
        try:
            df = pd.read_csv(cv_path)
        except Exception as e:
            print(f"  [WARN] failed to read {cv_path}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue

        model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else 'Unknown'
        clean = strip_init_source(model_name)
        if ':' in clean:
            base, strat = clean.split(':', 1)
            strat = strat.strip()
            if strat in ('Full Finetuning', 'Full'):
                strat = 'FF'
        else:
            base = clean
            strat = '-'

        way, shot = load_way_shot_from_config(run_dir)
        if way is None or shot is None:
            print(
                f"  [SKIP] {run_dir.name}: could not find way/shot in config.yaml",
                file=sys.stderr,
            )
            continue

        if (way, shot) not in COMBOS:
            print(
                f"  [SKIP] {run_dir.name}: unexpected combo ({way}, {shot})",
                file=sys.stderr,
            )
            continue

        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col in df.columns and std_col in df.columns:
            mean = float(df[mean_col].iloc[0])
            std = float(df[std_col].iloc[0])
        elif metric in df.columns:
            mean, std = parse_mean_std(df[metric].iloc[0])
            if mean is None:
                print(
                    f"  [WARN] {run_dir.name}: metric {metric!r} unparseable",
                    file=sys.stderr,
                )
                continue
        else:
            print(
                f"  [WARN] {run_dir.name}: metric {metric!r} not in cv_summary.csv",
                file=sys.stderr,
            )
            continue

        key = (base, strat)
        results.setdefault(key, {})[(way, shot)] = (mean, std)
        per_run_rows.append({
            'model': base,
            'strategy': strat,
            'way': way,
            'shot': shot,
            f'{metric}_mean': round(mean, 2),
            f'{metric}_std': round(std, 2),
            'cv_summary': str(cv_path),
        })
        print(f"  [OK] {base}:{strat} {way}w{shot}s → {mean:.2f}$\\pm${std:.2f}")

    return results, per_run_rows


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------
def build_ordered_rows(results):
    """Sort (base_model, strategy) keys using MODEL_ORDER and return list."""
    strat_order = {'-': 0, 'FF': 0, 'Full Finetuning': 0,
                   'DAPT': 1, 'IDPT': 2, 'PPT': 3, 'GST': 4}

    def key_fn(k):
        base, strat = k
        return (get_sort_key(base), strat_order.get(strat, 9), base, strat)

    return sorted(results.keys(), key=key_fn)


def best_per_combo(results, rows):
    """Return {(way, shot): (base, strat)} for the max-mean cell in each column."""
    best = {}
    for combo in COMBOS:
        best_mean = -float('inf')
        best_key = None
        for key in rows:
            cell = results[key].get(combo)
            if cell is None:
                continue
            mean, _ = cell
            if mean > best_mean:
                best_mean = mean
                best_key = key
        if best_key is not None:
            best[combo] = best_key
    return best


def _latex_escape(text):
    """Escape LaTeX special characters in free-text strings."""
    if not isinstance(text, str):
        return text
    for ch in ('_', '&', '%', '#'):
        text = text.replace(f'\\{ch}', f'\x00{ch}')
        text = text.replace(ch, f'\\{ch}')
        text = text.replace(f'\x00{ch}', f'\\{ch}')
    return text


def format_cell(cell, bold=False):
    if cell is None:
        return 'N/A'
    mean, std = cell
    if bold:
        return f"\\textbf{{{mean:.1f}}}$\\pm$\\textbf{{{std:.1f}}}"
    return f"{mean:.1f}$\\pm${std:.1f}"


def generate_latex_table(results, metric, caption, label, cite_in_tables=False):
    rows = build_ordered_rows(results)

    # Build the list of row-dicts consumed by render_grouped_table.
    combo_keys = [f"{w}w{s}s" for (w, s) in COMBOS]
    data_rows = []
    for key in rows:
        base, strat = key
        row = {'base': base, 'strategy': strat if strat not in ('Full Finetuning', 'Full') else 'FF'}
        for combo, combo_key in zip(COMBOS, combo_keys):
            cell = results[key].get(combo)
            if cell is None:
                row[combo_key] = None
            else:
                mean, std = cell
                row[combo_key] = f"{mean:.1f}$\\pm${std:.1f}"
        data_rows.append(row)

    # Two-level header: top row covers 5-way / 10-way groups (cols 4-5 & 6-7)
    extra_header = (
        r" &  &  & \multicolumn{2}{c}{5-way} & \multicolumn{2}{c}{10-way} \\"
        r" \cmidrule(lr){4-5}\cmidrule(lr){6-7}"
    )
    table = render_grouped_table(
        data_rows,
        metric_cols=combo_keys,
        metric_headers=['10-shot', '20-shot', '10-shot', '20-shot'],
        caption=caption,
        label=label,
        cite_in_tables=cite_in_tables,
        extra_header_row=extra_header,
        metric_width='1.6cm',
    )

    lines = []
    lines.append("% Auto-generated LaTeX document — few-shot classification")
    lines.append("% Compile with: pdflatex <filename>.tex  (then bibtex if using \\bibliography)")
    lines.append("\\documentclass[a4paper,11pt]{article}")
    lines.append("\\usepackage[utf8]{inputenc}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{multirow}")
    lines.append("\\usepackage{adjustbox}")
    lines.append("\\usepackage[numbers,sort&compress]{natbib}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\bibliographystyle{unsrtnat}")
    lines.append("\\begin{document}")
    lines.append("")
    if not cite_in_tables:
        bases = [base for base, _ in rows]
        strats = [strat for _, strat in rows if strat not in ('-', 'FF', 'Full Finetuning', 'Full')]
        paragraph = build_citation_paragraph(bases + strats)
        if paragraph:
            lines.append(paragraph)
            lines.append("")
    lines.append(table)
    lines.append("")
    lines.append("\\bibliography{references}")
    lines.append("\\end{document}")
    return "\n".join(lines)


def generate_markdown_table(results, metric):
    rows = build_ordered_rows(results)
    best = best_per_combo(results, rows)

    lines = []
    lines.append("|       | 5-way |       | 10-way |       |")
    lines.append("| Model | 10-shot | 20-shot | 10-shot | 20-shot |")
    lines.append("|---|---|---|---|---|")

    for key in rows:
        base, strat = key
        model_label = base if strat in ('-', 'FF', 'Full Finetuning', 'Full') else f"{base}-{strat}"
        cols = [model_label]
        for combo in COMBOS:
            cell = results[key].get(combo)
            if cell is None:
                cols.append("N/A")
                continue
            mean, std = cell
            text = f"{mean:.1f} ± {std:.1f}"
            if best.get(combo) == key:
                text = f"**{text}**"
            cols.append(text)
        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines)


def generate_wide_csv(results, metric):
    """Flat CSV: one row per (model, strategy), one column per combo."""
    rows = build_ordered_rows(results)
    data = []
    for key in rows:
        base, strat = key
        entry = {'model': base, 'strategy': strat}
        for way, shot in COMBOS:
            cell = results[key].get((way, shot))
            if cell is None:
                entry[f'{way}w{shot}s_mean'] = None
                entry[f'{way}w{shot}s_std'] = None
                entry[f'{way}w{shot}s'] = 'N/A'
            else:
                mean, std = cell
                entry[f'{way}w{shot}s_mean'] = round(mean, 2)
                entry[f'{way}w{shot}s_std'] = round(std, 2)
                entry[f'{way}w{shot}s'] = f"{mean:.2f} +/- {std:.2f}"
        data.append(entry)
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate few-shot classification table from cv_summary.csv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--exp_dir',
        type=str,
        default='experiments/ModelNetFewShot',
        help='Root directory containing few-shot experiment subdirs',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for .tex / .csv / .md files (default: <exp_dir>/latex)',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='best_accuracy',
        help='Which metric to report (column in cv_summary.csv, e.g. '
             'best_accuracy, f1_macro, balanced_acc)',
    )
    parser.add_argument(
        '--caption',
        type=str,
        default=None,
        help='LaTeX table caption (default depends on metric)',
    )
    parser.add_argument(
        '--label',
        type=str,
        default='tab:fewshot_modelnet40',
        help='LaTeX table label',
    )
    parser.add_argument(
        '--use_citation_in_tables',
        action='store_true',
        help='Put \\citep{} next to each model name inside the table. '
             'Default: cite methods once in a paragraph above the table.',
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        print(f"[ERROR] exp_dir not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {exp_dir} for few-shot cv_summary.csv files ...")
    results, per_run_rows = collect_results(exp_dir, args.metric)
    if not results:
        print("[ERROR] no few-shot results found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'latex'
    output_dir.mkdir(parents=True, exist_ok=True)

    caption = args.caption or (
        f"Few-shot classification results on ModelNet40. "
        f"We report the average {args.metric.replace('_', ' ')} (\\%) "
        f"as well as the standard deviation over 10 independent experiments."
    )

    latex = generate_latex_table(results, args.metric, caption, args.label,
                                 cite_in_tables=args.use_citation_in_tables)
    md = generate_markdown_table(results, args.metric)
    wide_df = generate_wide_csv(results, args.metric)

    tex_path = output_dir / f"fewshot_table_{args.metric}.tex"
    md_path = output_dir / f"fewshot_table_{args.metric}.md"
    csv_path = output_dir / f"fewshot_table_{args.metric}.csv"
    raw_csv_path = output_dir / f"fewshot_per_run_{args.metric}.csv"

    tex_path.write_text(latex + "\n")
    md_path.write_text(md + "\n")
    wide_df.to_csv(csv_path, index=False)
    pd.DataFrame(per_run_rows).to_csv(raw_csv_path, index=False)

    from _citations import copy_references_bib
    copy_references_bib(output_dir)

    print("\n=== Few-shot table (markdown preview) ===")
    print(md)
    print(f"\nSaved: {tex_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {raw_csv_path}")


if __name__ == '__main__':
    main()
