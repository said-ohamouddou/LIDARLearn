#!/usr/bin/env python3
"""
Friedman Statistical Test for Cross-Validation Results
=======================================================

Performs Friedman test on CV fold-level results across multiple models,
with post-hoc Nemenyi and Wilcoxon signed-rank tests and Critical Difference diagram.

Outputs publication-ready LaTeX tables focused on validating the proposed model.

Usage:
    # Default: MSDGCNN2 as proposed model
    python scripts/reports/friedman_significance_report.py

    # Different proposed model
    python scripts/reports/friedman_significance_report.py --proposed "PointM2AE:Full Finetuning [ShapeNet]"

    # Multiple metrics
    python scripts/reports/friedman_significance_report.py --metrics best_accuracy balanced_acc f1_weighted
"""

import os
import sys
import re
import argparse
import glob
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

sys.path.insert(0, str(Path(__file__).parent))
from _citations import cite_inline, build_citation_paragraph, get_cite_key

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EXP_DIR = 'experiments/STPCTLS'
DEFAULT_METRICS = ['best_accuracy']
DEFAULT_ALPHA = 0.05
DEFAULT_PROPOSED = ""

METRIC_LATEX_NAMES = {
    'best_accuracy': 'Accuracy',
    'balanced_acc': 'Balanced Acc',
    'f1_macro': 'F1 Macro',
    'f1_weighted': 'F1 Weighted',
    'precision_macro': 'Precision',
    'kappa': "Cohen's $\\kappa$",
    'mcc': 'MCC',
}

STRATEGY_ORDER = {'FF': 0, 'Full Finetuning': 0, 'Full': 0, 'DAPT': 1, 'IDPT': 2, 'PPT': 3, 'PointGST': 4, 'GST': 4}

# Normalize strategy names for display
STRATEGY_DISPLAY = {
    'Full Finetuning': 'FF',
    'Full': 'FF',
    'PointGST': 'GST',
}


# ---------------------------------------------------------------------------
# Model Name Parsing (same format as classification_comparison_report.py)
# ---------------------------------------------------------------------------

def strip_init_source(model_name: str) -> str:
    """Remove [InitSource] suffix from model name."""
    return re.sub(r'\s*\[[^\]]+\]\s*$', '', model_name).strip()


def parse_model_name(model_name: str):
    """
    Parse model name into base_model and strategy.
    E.g., "PointMAE:DAPT [ShapeNet]" -> ("PointMAE", "DAPT")
    """
    clean_name = strip_init_source(model_name)
    if ':' in clean_name:
        base_model, strategy = clean_name.split(':', 1)
        strategy = strategy.strip()
        # Normalize strategy name for display
        strategy = STRATEGY_DISPLAY.get(strategy, strategy)
        return base_model.strip(), strategy
    return clean_name, '-'


def get_model_sort_key(model_name: str, ranks: np.ndarray, model_idx: int):
    """Sort key: primary by rank (best first), secondary by strategy order."""
    base, strategy = parse_model_name(model_name)
    rank = ranks[model_idx] if model_idx < len(ranks) else 999
    strat_order = STRATEGY_ORDER.get(strategy, 9)
    return (rank, strat_order, base)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def find_model_dirs(exp_dir: str, model_filter: str = None):
    """Find all model directories containing cv_all_folds_detailed.csv."""
    pattern = os.path.join(exp_dir, '*', 'cv_all_folds_detailed.csv')
    csv_files = sorted(glob.glob(pattern))

    results = []
    for csv_path in csv_files:
        model_dir = os.path.dirname(csv_path)
        model_name = os.path.basename(model_dir)

        if model_filter:
            if not re.search(model_filter, model_name, re.IGNORECASE):
                continue

        results.append((model_name, csv_path))

    return results


def load_fold_scores(model_dirs: list, metrics: list):
    """Load fold-level scores from cv_all_folds_detailed.csv files."""
    models = []
    scores = {m: [] for m in metrics}
    k_folds = None

    for model_name, csv_path in model_dirs:
        try:
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                continue

            current_k = len(df)
            if k_folds is None:
                k_folds = current_k
            elif current_k != k_folds:
                continue

            # Extract model display name from cv_summary
            summary_path = os.path.join(os.path.dirname(csv_path), 'cv_summary.csv')
            display_name = model_name
            if os.path.exists(summary_path):
                try:
                    summary_df = pd.read_csv(summary_path)
                    if 'model_name' in summary_df.columns and len(summary_df) > 0:
                        display_name = summary_df['model_name'].iloc[0]
                except Exception:
                    pass

            missing = [m for m in metrics if m not in df.columns]
            if missing:
                continue

            for metric in metrics:
                fold_values = df[metric].values.astype(float)
                scores[metric].append(fold_values)

            models.append(display_name)

        except Exception:
            pass

    for metric in metrics:
        if scores[metric]:
            scores[metric] = np.array(scores[metric])
        else:
            scores[metric] = np.empty((0, k_folds or 5))

    return models, scores, k_folds or 5


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def friedman_test(score_matrix: np.ndarray):
    """
    Friedman test on (n_models × k_folds) matrix.

    Returns:
        (statistic, p_value)
    """
    if score_matrix.shape[0] < 3:
        return np.nan, np.nan
    try:
        stat, p = friedmanchisquare(*score_matrix)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def pairwise_wilcoxon(models: list, score_matrix: np.ndarray, alpha: float = 0.05):
    """
    Pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Returns:
        DataFrame with pairwise comparison results
    """
    n = len(models)
    results = []
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)

    for i, j in pairs:
        a, b = score_matrix[i], score_matrix[j]
        diff = a - b

        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, alternative='two-sided')
                stat, p = float(stat), float(p)
            except Exception:
                stat, p = np.nan, np.nan

        results.append({
            'model_a': models[i],
            'model_b': models[j],
            'mean_a': float(a.mean()),
            'mean_b': float(b.mean()),
            'statistic': stat,
            'p_raw': p,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    # Bonferroni correction
    df['p_corrected'] = np.minimum(df['p_raw'] * n_pairs, 1.0)
    df['significant'] = df['p_corrected'] < alpha
    df['winner'] = df.apply(
        lambda r: r['model_a'] if r['mean_a'] > r['mean_b'] else
        (r['model_b'] if r['mean_b'] > r['mean_a'] else 'tie'),
        axis=1
    )

    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two paired samples."""
    diff = a - b
    if diff.std(ddof=1) < 1e-12:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def nemenyi_test(models: list, ranks: np.ndarray, k_folds: int, alpha: float = 0.05):
    """
    Nemenyi post-hoc test for pairwise comparisons.

    Two models are significantly different if their average rank difference
    exceeds the critical difference (CD).

    Returns:
        DataFrame with pairwise Nemenyi comparison results
    """
    n = len(models)
    cd = critical_difference(n, k_folds, alpha)

    results = []
    pairs = list(combinations(range(n), 2))

    for i, j in pairs:
        rank_diff = abs(ranks[i] - ranks[j])
        significant = rank_diff > cd

        # Determine winner (lower rank is better)
        if ranks[i] < ranks[j]:
            winner = models[i]
        elif ranks[j] < ranks[i]:
            winner = models[j]
        else:
            winner = 'tie'

        results.append({
            'model_a': models[i],
            'model_b': models[j],
            'rank_a': float(ranks[i]),
            'rank_b': float(ranks[j]),
            'rank_diff': float(rank_diff),
            'cd': float(cd),
            'significant': significant,
            'winner': winner if significant else 'no_diff'
        })

    return pd.DataFrame(results)


def average_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """
    Compute average rank for each model across folds.
    Lower rank = better performance (rank 1 = best).

    Returns:
        Array of shape (n_models,)
    """
    n_models, k = score_matrix.shape
    fold_ranks = np.zeros_like(score_matrix, dtype=float)

    for f in range(k):
        # Rank in descending order: best model gets rank 1
        fold_ranks[:, f] = rankdata(-score_matrix[:, f], method='average')

    return fold_ranks.mean(axis=1)


def critical_difference(n_models: int, k_folds: int, alpha: float = 0.05) -> float:
    """
    Nemenyi critical difference.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))
    """
    # q_alpha for two-tailed Nemenyi at alpha=0.05 (from Demsar 2006)
    q_alpha_table = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        12: 3.268, 15: 3.399, 20: 3.578, 25: 3.720, 30: 3.845,
        35: 3.955, 40: 4.050, 45: 4.132, 50: 4.210,
    }

    keys = sorted(q_alpha_table.keys())
    nearest = min(keys, key=lambda x: abs(x - n_models))
    q = q_alpha_table[nearest]

    cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * k_folds))
    return cd


# ---------------------------------------------------------------------------
# Critical Difference Diagram
# ---------------------------------------------------------------------------

def generate_cd_diagram(models: list, ranks: np.ndarray, cd: float, proposed: str,
                        metric_name: str, output_path: str, top_n: int = 10):
    """
    Generate publication-ready Critical Difference diagram (Demšar 2006 style).
    Shows top N models by rank with clear, bold formatting.

    Args:
        models: List of model names
        ranks: Array of average ranks
        cd: Critical difference value
        proposed: Name of proposed model
        metric_name: Name of metric for title
        output_path: Path to save figure
        top_n: Number of top models to show (default: 10)
    """
    # Sort models by rank and take top N
    sorted_indices = np.argsort(ranks)[:top_n]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = ranks[sorted_indices]
    n = len(sorted_models)

    # Parse model names for display
    display_names = []
    for m in sorted_models:
        base, strat = parse_model_name(m)
        if strat != '-':
            display_names.append(f"{base}:{strat}")
        else:
            display_names.append(base)

    # Find proposed model index in sorted list (prefer exact match)
    proposed_idx = None
    for i, m in enumerate(sorted_models):
        base, _ = parse_model_name(m)
        if proposed == m or proposed == base:
            proposed_idx = i
            break
    if proposed_idx is None:
        for i, m in enumerate(sorted_models):
            base, _ = parse_model_name(m)
            if proposed in base or base in proposed:
                proposed_idx = i
                break

    # Publication-ready figure settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'font.weight': 'bold',
    })

    # Create figure - wider aspect ratio for clarity
    fig_width = 10
    fig_height = max(3, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Rank axis range (based on actual ranks shown)
    min_rank = max(1, sorted_ranks.min() - 1)
    max_rank = sorted_ranks.max() + 1

    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(-1.5, n + 2.5)

    # Draw main axis line
    ax.axhline(y=n + 0.5, color='black', linewidth=2, xmin=0.02, xmax=0.98)

    # Draw tick marks on axis
    tick_step = max(1, int((max_rank - min_rank) / 8))
    for r in np.arange(int(min_rank), int(max_rank) + 1, tick_step):
        ax.plot([r, r], [n + 0.35, n + 0.65], 'k-', linewidth=2)
        ax.text(r, n + 0.9, str(r), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Title
    ax.text((min_rank + max_rank) / 2, n + 2.0, f'Critical Difference Diagram — {metric_name}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Draw CD bar legend
    cd_start = max_rank - cd
    if cd_start >= min_rank:
        ax.plot([cd_start, max_rank], [n + 1.4, n + 1.4], 'k-', linewidth=3)
        ax.plot([cd_start, cd_start], [n + 1.25, n + 1.55], 'k-', linewidth=3)
        ax.plot([max_rank, max_rank], [n + 1.25, n + 1.55], 'k-', linewidth=3)
        ax.text((cd_start + max_rank) / 2, n + 1.65, f'CD = {cd:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Draw models - alternating left and right sides
    for i in range(n):
        rank = sorted_ranks[i]
        name = display_names[i]
        is_proposed = (i == proposed_idx)

        # Alternate sides for clarity
        if i % 2 == 0:
            # Left side (above axis line, text to the left)
            y_pos = n - 0.5 - (i // 2) * 0.9
            ax.plot([rank, rank], [y_pos + 0.15, n + 0.35], 'k-', linewidth=1.5)
            ax.plot([rank], [y_pos + 0.15], 'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)

            color = '#CC0000' if is_proposed else 'black'
            weight = 'bold'
            ax.text(rank - 0.15, y_pos, f'{name}', ha='right', va='center',
                    fontsize=11, color=color, fontweight=weight)
            ax.text(rank + 0.15, y_pos, f'({rank:.1f})', ha='left', va='center',
                    fontsize=10, color='gray', fontweight='normal')
        else:
            # Right side (below axis line, text to the right)
            y_pos = n - 0.5 - (i // 2) * 0.9
            ax.plot([rank, rank], [y_pos - 0.15, n + 0.35], 'k-', linewidth=1.5)
            ax.plot([rank], [y_pos - 0.15], 'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)

            color = '#CC0000' if is_proposed else 'black'
            weight = 'bold'
            ax.text(rank + 0.15, y_pos - 0.3, f'{name}', ha='left', va='center',
                    fontsize=11, color=color, fontweight=weight)
            ax.text(rank - 0.15, y_pos - 0.3, f'({rank:.1f})', ha='right', va='center',
                    fontsize=10, color='gray', fontweight='normal')

    # Draw CD groups (models within CD of each other)
    groups = []
    used = set()
    for i in range(n):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, n):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= cd:
                group.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append(group)
        used.add(i)

    # Draw group bars at bottom
    bar_y = -0.5
    for group in groups:
        min_r = min(sorted_ranks[i] for i in group)
        max_r = max(sorted_ranks[i] for i in group)
        ax.plot([min_r, max_r], [bar_y, bar_y], '-', color='#2E86AB', linewidth=4, solid_capstyle='round')
        bar_y -= 0.4

    ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    return output_path


# ---------------------------------------------------------------------------
# LaTeX Generation
# ---------------------------------------------------------------------------

def latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ('_', r'\_'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('#', r'\#'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def pvalue_str(p: float, alpha: float = 0.05) -> str:
    """Format p-value for LaTeX."""
    if np.isnan(p):
        return '--'
    if p < 0.001:
        return '$<$0.001'
    return f'{p:.3f}'


def generate_proposed_comparison_table(models: list, metric_results: dict, metrics: list,
                                       proposed: str, alpha: float, k_folds: int,
                                       cite_in_tables: bool = False) -> str:
    """
    Generate focused table: Proposed model vs all baselines.
    Columns: Model, Strategy, Mean±Std, Rank, p-value vs proposed, Significance
    Sorted by rank (best first).
    """
    if not metrics or metrics[0] not in metric_results:
        return ''

    metric = metrics[0]  # Primary metric
    res = metric_results[metric]
    ranks = res['avg_ranks']
    scores = res['score_matrix']
    pw = res.get('pairwise', pd.DataFrame())
    n = len(models)
    cd = critical_difference(n, k_folds, alpha)

    # Find proposed model index (prefer exact match)
    proposed_idx = None
    for i, m in enumerate(models):
        base, _ = parse_model_name(m)
        if proposed == m or proposed == base:
            proposed_idx = i
            break
    if proposed_idx is None:
        for i, m in enumerate(models):
            base, _ = parse_model_name(m)
            if proposed in base or base in proposed:
                proposed_idx = i
                break

    if proposed_idx is None:
        return f'% Proposed model "{proposed}" not found in results\n'

    # Build sorted data (by rank, best first)
    data = []
    for idx, model in enumerate(models):
        base, strategy = parse_model_name(model)
        rank = ranks[idx]
        mean = scores[idx].mean()
        std = scores[idx].std(ddof=1)

        # Get p-value vs proposed
        p_vs_proposed = np.nan
        if idx != proposed_idx and len(pw) > 0:
            row = pw[((pw['model_a'] == model) & (pw['model_b'] == models[proposed_idx]))
                     | ((pw['model_b'] == model) & (pw['model_a'] == models[proposed_idx]))]
            if len(row) > 0:
                p_vs_proposed = row['p_corrected'].iloc[0]

        data.append({
            'model': model,
            'base': base,
            'strategy': strategy,
            'rank': rank,
            'mean': mean,
            'std': std,
            'p_vs_proposed': p_vs_proposed,
            'is_proposed': idx == proposed_idx,
            'within_cd': abs(rank - ranks[proposed_idx]) <= cd
        })

    # Sort by rank
    data.sort(key=lambda x: x['rank'])

    # Generate LaTeX
    mname = METRIC_LATEX_NAMES.get(metric, metric)
    proposed_base, proposed_strat = parse_model_name(models[proposed_idx])
    proposed_display = f"{proposed_base}" + (f":{proposed_strat}" if proposed_strat != '-' else '')

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Statistical comparison for ' + mname
                 + r'. Models ranked by average rank (best=1). '
                 r'$p$-values are Bonferroni-corrected Wilcoxon tests vs '
                 + latex_escape(proposed_display) + r'. CD=' + f'{cd:.2f}' + r'.}')
    lines.append(r'\label{tab:proposed_comparison}')
    lines.append(r'\begin{tabular}{llcrrc}')
    lines.append(r'\toprule')
    lines.append(r'Model & Strategy & ' + mname + r' (\%) & Rank & $p$-value & Sig. \\')
    lines.append(r'\midrule')

    for d in data:
        base = (cite_inline(d['base'], display=latex_escape(d['base']))
                if cite_in_tables else latex_escape(d['base']))
        strat = d['strategy']
        mean_std = f"{d['mean']:.2f}$\\pm${d['std']:.2f}"
        rank_str = f"{d['rank']:.1f}"

        if d['is_proposed']:
            # Highlight proposed model
            mean_std_bold = f"\\\\textbf{{{d['mean']:.2f}}}$\\\\pm$\\\\textbf{{{d['std']:.2f}}}"
            row = f"\\textbf{{{base}}} & \\textbf{{{strat}}} & {mean_std_bold} & \\textbf{{{rank_str}}} & -- & -- \\\\"
        else:
            p_str = pvalue_str(d['p_vs_proposed'], alpha)
            sig = 'Yes' if (not np.isnan(d['p_vs_proposed']) and d['p_vs_proposed'] < alpha) else 'No'
            if d['within_cd']:
                sig += '$^\\dagger$'
            row = f"{base} & {strat} & {mean_std} & {rank_str} & {p_str} & {sig} \\\\"

        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\multicolumn{6}{l}{\footnotesize $^\dagger$Within critical difference of proposed model.} \\')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_summary_sentence(models: list, metric_results: dict, metrics: list,
                              proposed: str, alpha: float, k_folds: int) -> str:
    """Generate publication-ready summary sentence."""
    if not metrics or metrics[0] not in metric_results:
        return ''

    metric = metrics[0]
    res = metric_results[metric]
    ranks = res['avg_ranks']
    pw = res.get('pairwise', pd.DataFrame())
    n = len(models)
    cd = critical_difference(n, k_folds, alpha)

    # Find proposed model (prefer exact match)
    proposed_idx = None
    for i, m in enumerate(models):
        base, _ = parse_model_name(m)
        if proposed == m or proposed == base:
            proposed_idx = i
            break
    if proposed_idx is None:
        for i, m in enumerate(models):
            base, _ = parse_model_name(m)
            if proposed in base or base in proposed:
                proposed_idx = i
                break

    if proposed_idx is None:
        return ''

    proposed_rank = ranks[proposed_idx]
    proposed_mean = res['score_matrix'][proposed_idx].mean()

    # Count wins
    wins = 0
    losses = 0
    ties = 0
    for idx in range(n):
        if idx == proposed_idx:
            continue
        row = pw[((pw['model_a'] == models[proposed_idx]) & (pw['model_b'] == models[idx]))
                 | ((pw['model_b'] == models[proposed_idx]) & (pw['model_a'] == models[idx]))]
        if len(row) > 0:
            p = row['p_corrected'].iloc[0]
            winner = row['winner'].iloc[0]
            if p < alpha:
                if winner == models[proposed_idx]:
                    wins += 1
                else:
                    losses += 1
            else:
                ties += 1

    # Models within CD
    within_cd = sum(1 for i, r in enumerate(ranks) if i != proposed_idx and abs(r - proposed_rank) <= cd)

    proposed_base, proposed_strat = parse_model_name(models[proposed_idx])
    proposed_name = f"{proposed_base}" + (f":{proposed_strat}" if proposed_strat != '-' else '')

    mname = METRIC_LATEX_NAMES.get(metric, metric).lower()

    lines = []
    lines.append(r'\noindent\textbf{Statistical Summary:}')
    lines.append(f'The Friedman test indicates significant differences among models ')
    lines.append(f'($\\chi^2 = {res["friedman_stat"]:.2f}$, $p < 0.001$). ')
    lines.append(f'{latex_escape(proposed_name)} achieves rank {proposed_rank:.1f} ')
    lines.append(f'with mean {mname} of {proposed_mean:.2f}\\%. ')
    lines.append(f'Post-hoc Wilcoxon tests (Bonferroni-corrected, $\\alpha={alpha}$) show that ')
    lines.append(f'{latex_escape(proposed_name)} significantly outperforms {wins}/{n-1} baselines ')
    if losses > 0:
        lines.append(f'and is significantly outperformed by {losses} model(s). ')
    else:
        lines.append(f'with no model significantly outperforming it. ')
    lines.append(f'{within_cd + 1} models are within the Nemenyi critical difference (CD={cd:.2f}).')

    return ''.join(lines)


def generate_friedman_summary(metric_results: dict, metrics: list, n_models: int, k_folds: int, alpha: float) -> str:
    """Generate Friedman test summary table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Friedman test results ($n=' + str(n_models) + r'$ models, $k=' + str(k_folds) + r'$ folds). Significant at $\alpha=' + str(alpha) + r'$.}')
    lines.append(r'\label{tab:friedman}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Metric & $\chi^2$ & $p$-value & Significant \\')
    lines.append(r'\midrule')

    for metric in metrics:
        if metric not in metric_results:
            continue
        res = metric_results[metric]
        mname = METRIC_LATEX_NAMES.get(metric, latex_escape(metric))
        stat = res['friedman_stat']
        pval = res['friedman_p']

        if np.isnan(pval):
            p_str = '--'
            sig = '--'
        elif pval < 0.001:
            p_str = '$<$0.001'
            sig = 'Yes'
        else:
            p_str = f'{pval:.4f}'
            sig = 'Yes' if pval < alpha else 'No'

        stat_str = f'{stat:.2f}' if not np.isnan(stat) else '--'
        lines.append(f'{mname} & {stat_str} & {p_str} & {sig} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_nemenyi_table(models: list, metric_results: dict, metric: str,
                           proposed: str, k_folds: int, alpha: float, top_n: int = 10,
                           cite_in_tables: bool = False) -> str:
    """
    Generate Nemenyi post-hoc test results table.
    Shows only top N models that are significantly different from the proposed model.
    """
    if metric not in metric_results:
        return ''

    res = metric_results[metric]
    nemenyi = res.get('nemenyi', pd.DataFrame())
    ranks = res['avg_ranks']
    mname = METRIC_LATEX_NAMES.get(metric, latex_escape(metric))
    n = len(models)
    cd = critical_difference(n, k_folds, alpha)

    if len(nemenyi) == 0:
        return ''

    # Find proposed model index
    proposed_idx = None
    for i, m in enumerate(models):
        base, _ = parse_model_name(m)
        if proposed == m or proposed == base:
            proposed_idx = i
            break
    if proposed_idx is None:
        for i, m in enumerate(models):
            base, _ = parse_model_name(m)
            if proposed in base or base in proposed:
                proposed_idx = i
                break

    if proposed_idx is None:
        return f'% Proposed model "{proposed}" not found\n'

    proposed_name = models[proposed_idx]
    proposed_base, proposed_strat = parse_model_name(proposed_name)
    proposed_display = f"{proposed_base}" + (f":{proposed_strat}" if proposed_strat != '-' else '')

    # Filter comparisons involving proposed model
    proposed_comparisons = nemenyi[
        (nemenyi['model_a'] == proposed_name) | (nemenyi['model_b'] == proposed_name)
    ].copy()

    # Add other model info
    def get_other_model(row):
        return row['model_b'] if row['model_a'] == proposed_name else row['model_a']

    def get_other_rank(row):
        return row['rank_b'] if row['model_a'] == proposed_name else row['rank_a']

    proposed_comparisons['other_model'] = proposed_comparisons.apply(get_other_model, axis=1)
    proposed_comparisons['other_rank'] = proposed_comparisons.apply(get_other_rank, axis=1)

    # Filter only significant differences and sort by rank difference (largest first)
    sig_comparisons = proposed_comparisons[proposed_comparisons['significant'] == True].copy()
    sig_comparisons = sig_comparisons.sort_values('rank_diff', ascending=False)

    # Take top N
    sig_comparisons = sig_comparisons.head(top_n)

    n_total_sig = proposed_comparisons['significant'].sum()

    if len(sig_comparisons) == 0:
        return f'% No significant Nemenyi differences for {proposed}\n'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Nemenyi post-hoc test: Top ' + str(len(sig_comparisons))
                 + r' models significantly different from ' + latex_escape(proposed_display)
                 + r' (rank=' + f'{ranks[proposed_idx]:.1f}' + r'). '
                 + r'CD=' + f'{cd:.2f}' + r', $\alpha=' + str(alpha) + r'$.}')
    lines.append(r'\label{tab:nemenyi_' + metric.replace('_', '') + r'}')
    lines.append(r'\begin{tabular}{llccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & Strategy & Rank & $|\Delta R|$ & Winner \\')
    lines.append(r'\midrule')

    n_wins = 0
    n_losses = 0
    for _, row in sig_comparisons.iterrows():
        other = row['other_model']
        base, strat = parse_model_name(other)
        other_rank = row['other_rank']
        rank_diff = row['rank_diff']

        # Determine winner (lower rank is better)
        base_fmt = (cite_inline(base, display=latex_escape(base))
                    if cite_in_tables else latex_escape(base))
        proposed_fmt = (cite_inline(proposed_base, display=latex_escape(proposed_display))
                        if cite_in_tables else latex_escape(proposed_display))
        if ranks[proposed_idx] < other_rank:
            winner_str = proposed_fmt
            n_wins += 1
        else:
            winner_str = base_fmt
            n_losses += 1

        lines.append(f'{base_fmt} & {strat} & {other_rank:.1f} & {rank_diff:.2f} & {winner_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')
    lines.append(r'\noindent\textbf{Nemenyi Summary:} ' + latex_escape(proposed_display)
                 + f' is significantly different from {n_total_sig}/{len(proposed_comparisons)} models. '
                 + f'Wins: {n_wins}, Losses: {n_losses}.')

    return '\n'.join(lines)


def generate_top_n_table(models: list, metric_results: dict, metric: str,
                         proposed: str, top_n: int = 10,
                         cite_in_tables: bool = False) -> str:
    """Generate simple top-N ranked table for one metric."""
    if metric not in metric_results:
        return ''

    res = metric_results[metric]
    ranks = res['avg_ranks']
    scores = res['score_matrix']
    mname = METRIC_LATEX_NAMES.get(metric, latex_escape(metric))

    # Sort by rank and take top N
    sorted_indices = np.argsort(ranks)[:top_n]

    # Find proposed model index
    proposed_idx = None
    for i, m in enumerate(models):
        base, _ = parse_model_name(m)
        if proposed == m or proposed == base:
            proposed_idx = i
            break
    if proposed_idx is None:
        for i, m in enumerate(models):
            base, _ = parse_model_name(m)
            if proposed in base or base in proposed:
                proposed_idx = i
                break

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Top 10 models ranked by ' + mname + r'. Bold = proposed model.}')
    lines.append(r'\label{tab:top10_' + metric.replace('_', '') + r'}')
    lines.append(r'\begin{tabular}{clccc}')
    lines.append(r'\toprule')
    lines.append(r'Rank & Model & Strategy & ' + mname + r' (\%) & Avg. Rank \\')
    lines.append(r'\midrule')

    for position, idx in enumerate(sorted_indices, 1):
        model = models[idx]
        base, strategy = parse_model_name(model)
        rank = ranks[idx]
        mean = scores[idx].mean()
        std = scores[idx].std(ddof=1)

        is_proposed = (idx == proposed_idx)
        base_fmt = (cite_inline(base, display=latex_escape(base))
                    if cite_in_tables else latex_escape(base))

        if is_proposed:
            row = f'\\textbf{{{position}}} & \\textbf{{{base_fmt}}} & \\textbf{{{strategy}}} & \\textbf{{{mean:.2f}}}$\\pm$\\textbf{{{std:.2f}}} & \\textbf{{{rank:.1f}}} \\\\'
        else:
            row = f'{position} & {base_fmt} & {strategy} & {mean:.2f}$\\pm${std:.2f} & {rank:.1f} \\\\'

        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def build_latex_document(sections: list, citation_paragraph: str = '') -> str:
    """Build compact LaTeX document.

    If `citation_paragraph` is non-empty, it is prepended above the sections
    (used when --use_citation_in_tables is NOT set).
    """
    preamble = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{unsrtnat}

\begin{document}

\section*{Friedman Statistical Analysis}

"""
    body_parts = []
    if citation_paragraph:
        body_parts.append(citation_paragraph)
    body_parts.extend(sections)
    body = '\n\n'.join(body_parts)
    end = ('\n\n\\bibliography{references}\n'
           '\\end{document}\n')
    return preamble + body + end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Friedman statistical test for CV results with LaTeX output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp_dir', type=str, default=DEFAULT_EXP_DIR,
                        help=f'Experiment directory (default: {DEFAULT_EXP_DIR})')
    parser.add_argument('--metrics', type=str, nargs='+', default=DEFAULT_METRICS,
                        help=f'Metrics to test (default: {DEFAULT_METRICS})')
    parser.add_argument('--models', type=str, default=None,
                        help='Regex pattern to filter model names (default: all)')
    parser.add_argument('--proposed', type=str, default=DEFAULT_PROPOSED,
                        help=f'Proposed model name (default: {DEFAULT_PROPOSED})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Significance level (default: {DEFAULT_ALPHA})')
    parser.add_argument('--output', type=str, default='friedman_results.tex',
                        help='Output LaTeX file name (default: friedman_results.tex)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated files (default: <exp_dir>/friedman)')
    parser.add_argument('--cd_top', type=int, default=10,
                        help='Number of top models to show in CD diagram (default: 10)')
    parser.add_argument('--use_citation_in_tables', action='store_true',
                        help='Put \\citep{} next to each model name inside the tables. '
                             'Default: cite methods once in a paragraph above the tables.')
    args = parser.parse_args()

    print('=' * 70)
    print('FRIEDMAN STATISTICAL ANALYSIS')
    print('=' * 70)
    print(f'Experiment dir : {args.exp_dir}')
    print(f'Metrics        : {args.metrics}')
    print(f'Proposed model : {args.proposed}')
    print(f'Alpha          : {args.alpha}')
    print(f'CD diagram top : {args.cd_top}')
    print(f'Output         : {args.output}')
    print('-' * 70)

    # Find model directories
    print('\nScanning for CV results...')
    model_dirs = find_model_dirs(args.exp_dir, args.models)

    if not model_dirs:
        print('[!!] No CV results found. Check --exp_dir path.')
        sys.exit(1)

    print(f'  Found {len(model_dirs)} model directories')

    # Load fold scores
    print('\nLoading fold-level scores...')
    models, scores, k_folds = load_fold_scores(model_dirs, args.metrics)

    if len(models) < 2:
        print('[!!] Need at least 2 models for comparison.')
        sys.exit(1)

    print(f'  Loaded {len(models)} models, {k_folds} folds')

    # Check proposed model exists
    proposed_found = any(args.proposed in m or m in args.proposed for m in models)
    if not proposed_found:
        print(f'\n[WARN] Warning: Proposed model "{args.proposed}" not found in results.')
        print('  Available models:')
        for m in sorted(models)[:10]:
            print(f'    - {m}')
        if len(models) > 10:
            print(f'    ... and {len(models) - 10} more')

    # Run statistical tests
    print('\nRunning statistical tests...')
    metric_results = {}

    for metric in args.metrics:
        score_matrix = scores.get(metric, np.array([]))
        if score_matrix.ndim != 2 or score_matrix.shape[0] < 2:
            print(f'  [WARN] Skipping {metric}: insufficient data')
            continue

        res = {}
        res['score_matrix'] = score_matrix
        res['mean_scores'] = score_matrix.mean(axis=1)
        res['avg_ranks'] = average_ranks(score_matrix)

        # Friedman test
        stat, p = friedman_test(score_matrix)
        res['friedman_stat'] = stat
        res['friedman_p'] = p
        sig_str = '[OK] SIGNIFICANT' if (not np.isnan(p) and p < args.alpha) else ''
        print(f'  {metric}: χ²={stat:.2f}, p={p:.6f} {sig_str}')

        # Pairwise Wilcoxon
        pw = pairwise_wilcoxon(models, score_matrix, args.alpha)
        res['pairwise'] = pw

        # Nemenyi post-hoc test
        nemenyi = nemenyi_test(models, res['avg_ranks'], k_folds, args.alpha)
        res['nemenyi'] = nemenyi
        n_sig_nemenyi = nemenyi['significant'].sum()
        print(f'    Nemenyi: {n_sig_nemenyi} significant pairwise differences (CD={critical_difference(len(models), k_folds, args.alpha):.2f})')

        # Effect sizes
        effect_sizes = {}
        for _, row in pw.iterrows():
            try:
                i = models.index(row['model_a'])
                j = models.index(row['model_b'])
            except ValueError:
                continue
            d = cohens_d(score_matrix[i], score_matrix[j])
            effect_sizes[(row['model_a'], row['model_b'])] = d
        res['effect_sizes'] = effect_sizes

        metric_results[metric] = res

    if not metric_results:
        print('[!!] No metrics could be tested.')
        sys.exit(1)

    # Determine output paths
    output_dir = args.output_dir if args.output_dir else os.path.join(args.exp_dir, 'friedman')
    os.makedirs(output_dir, exist_ok=True)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(output_dir, output_path)
    else:
        output_dir = os.path.dirname(output_path) or output_dir

    # Build LaTeX document
    print('\nGenerating LaTeX document...')
    sections = []

    # Friedman test summary table (all metrics in one table)
    sections.append(generate_friedman_summary(
        metric_results, args.metrics, len(models), k_folds, args.alpha
    ))

    # Top-N table for each metric
    for metric in args.metrics:
        if metric in metric_results:
            sections.append(generate_top_n_table(
                models, metric_results, metric, args.proposed, top_n=args.cd_top,
                cite_in_tables=args.use_citation_in_tables,
            ))

    # Nemenyi post-hoc test table for each metric
    for metric in args.metrics:
        if metric in metric_results:
            sections.append(generate_nemenyi_table(
                models, metric_results, metric, args.proposed, k_folds, args.alpha,
                cite_in_tables=args.use_citation_in_tables,
            ))

    # Citation paragraph (only when citations are NOT inside tables)
    citation_paragraph = ''
    if not args.use_citation_in_tables:
        base_names = [parse_model_name(m)[0] for m in models]
        strats = [s for _, s in (parse_model_name(m) for m in models)
                  if s not in ('-', 'FF', 'Full Finetuning', 'Full')]
        citation_paragraph = build_citation_paragraph(base_names + strats)

    # Write output
    latex_doc = build_latex_document(sections, citation_paragraph=citation_paragraph)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_doc)

    from _citations import copy_references_bib
    copy_references_bib(output_dir)

    print(f'\n[OK] LaTeX output saved: {output_path}')
    print('\nTo compile:')
    print(f'  cd {output_dir} && pdflatex {os.path.basename(output_path)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
