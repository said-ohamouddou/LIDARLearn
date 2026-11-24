#!/usr/bin/env python3
"""
Generate Model Comparison Table from CV Summary Files

This script collects cv_summary.csv files from all trained models and generates:
1. A combined CSV comparison table
2. A LaTeX table with best values highlighted in bold (with units in column headers)
3. A Markdown table

Default metrics (in order):
- Accuracy, Balanced Accuracy, F1 Weighted, Recall, Precision
- Per-class Recall for all classes
- Total Parameters (M), Trainable Parameters (M), Epoch Time (s)

Usage:
    # Use defaults (recommended)
    python scripts/reports/classification_comparison_report.py --exp_dir experiments/tree_dataset_cv

    # Custom metrics
    python scripts/reports/classification_comparison_report.py --exp_dir experiments/tree_dataset_cv \\
        --metrics best_accuracy balanced_acc f1_weighted kappa mcc

    # Disable per-class metrics
    python scripts/reports/classification_comparison_report.py --exp_dir experiments/tree_dataset_cv --no_per_class

    # Include per-class precision and recall
    python scripts/reports/classification_comparison_report.py --exp_dir experiments/tree_dataset_cv \\
        --per_class precision recall f1
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re


# Default metrics to include (the combined "mean ± std" columns)
DEFAULT_METRICS = [
    'best_accuracy',
    'balanced_acc',
    'f1_macro',
    'recall_macro',
    'precision_macro',
    'total_params_M',
    'trainable_params_M',
    'avg_epoch_time_s',
]

# Default per-class metrics (recall for all classes)
DEFAULT_PER_CLASS = ['recall']

# Mapping for nicer display names (console/markdown)
METRIC_DISPLAY_NAMES = {
    'best_accuracy': 'Accuracy',
    'balanced_acc': 'Balanced Acc',
    'f1_macro': 'F1 Macro',
    'f1_weighted': 'F1 Weighted',
    'precision_macro': 'Precision',
    'precision_weighted': 'Precision (W)',
    'recall_macro': 'Recall',
    'recall_weighted': 'Recall (W)',
    'kappa': 'Kappa',
    'mcc': 'MCC',
    'model_name': 'Model',
    'k_folds': 'K-Folds',
    'total_params_M': 'Total Params',
    'trainable_params_M': 'Train Params',
    'avg_epoch_time_s': 'Epoch Time',
}

# LaTeX display names with units
LATEX_DISPLAY_NAMES = {
    'best_accuracy': 'Accuracy (\\%)',
    'balanced_acc': 'Balanced Acc (\\%)',
    'f1_macro': 'F1 Macro (\\%)',
    'f1_weighted': 'F1 Weighted (\\%)',
    'precision_macro': 'Precision (\\%)',
    'precision_weighted': 'Precision W (\\%)',
    'recall_macro': 'Recall (\\%)',
    'recall_weighted': 'Recall W (\\%)',
    'kappa': 'Kappa (\\%)',
    'mcc': 'MCC (\\%)',
    'model_name': 'Model',
    'k_folds': 'K-Folds',
    'total_params_M': 'Total Params (M)',
    'trainable_params_M': 'Train Params (M)',
    'avg_epoch_time_s': 'Epoch Time (s)',
}

# Per-class metric prefixes
PER_CLASS_PREFIXES = ['precision', 'recall', 'f1']

# =============================================================================
# MODEL ORDERING AND CATEGORIES (from TRAINING_COMMANDS_tree_n_epochs.sh)
# =============================================================================

# Model ordering based on training script - defines exact row order in table
MODEL_ORDER = [
    # Point-based (lines 21-36)
    'PointNet', 'PointNet2_SSG', 'PointNet2_MSG', 'SONet', 'PPFNet', 'PointCNN',
    'PointWeb', 'PointConv', 'RSCNN', 'PointMLP', 'PointMLPLite',
    'PointSCNet', 'RepSurf', 'PointKAN', 'DELA',
    # Attention-based
    'PCT', 'P2P', 'PointTNT', 'GlobalTransformer', 'PVT',
    'PointTransformer', 'PointTransformerV2', 'PointTransformerV3',
    # Graph-based
    'DGCNN', 'DeepGCN', 'CurveNet', 'GDAN', 'GDANet', 'MSDGCNN', 'KANDGCNN', 'MSDGCNN2',
    # Self-supervised (lines 55-84)
    'Point-MAE', 'ACT', 'RECON', 'PointGPT', 'Point-M2AE', 'PointBERT', 'PCP',
    'PointMamba',
]

# Category assignments for each model
MODEL_CATEGORIES = {
    # Point-based
    'PointNet': 'Point-based', 'PointNet2_SSG': 'Point-based', 'PointNet2_MSG': 'Point-based',
    'SONet': 'Point-based', 'PPFNet': 'Point-based', 'PointCNN': 'Point-based',
    'PointWeb': 'Point-based', 'PointConv': 'Point-based', 'RSCNN': 'Point-based',
    'PointMLP': 'Point-based', 'PointMLPLite': 'Point-based',
    'PointSCNet': 'Point-based', 'RepSurf': 'Point-based', 'PointKAN': 'Point-based', 'DELA': 'Point-based',
    # Attention-based
    'PCT': 'Attention-based', 'P2P': 'Attention-based', 'PointTNT': 'Attention-based',
    'GlobalTransformer': 'Attention-based', 'PVT': 'Attention-based', 'PointTransformer': 'Attention-based',
    'PointTransformerV2': 'Attention-based', 'PointTransformerV3': 'Attention-based',
    # Graph-based
    'DGCNN': 'Graph-based', 'DeepGCN': 'Graph-based', 'CurveNet': 'Graph-based',
    'GDAN': 'Graph-based', 'GDANet': 'Graph-based', 'MSDGCNN': 'Graph-based', 'KANDGCNN': 'Graph-based', 'MSDGCNN2': 'Graph-based',
    # Self-supervised
    'Point-MAE': 'Self-supervised', 'ACT': 'Self-supervised', 'RECON': 'Self-supervised',
    'PointGPT': 'Self-supervised', 'PCP': 'Self-supervised', 'PointMamba': 'Self-supervised', 'Point-M2AE': 'Self-supervised',
    'PointBERT': 'Self-supervised',
}

# Citation keys for LaTeX (model name -> BibTeX key from references.bib)
MODEL_CITATIONS = {
    # Supervised — point-based
    'PointNet': 'qi2017pointnet', 'PointNet2_SSG': 'qi2017pointnet++', 'PointNet2_MSG': 'qi2017pointnet++',
    'SONet': 'sonet', 'PPFNet': 'ppfnet', 'PointCNN': 'pointcnn',
    'PointWeb': 'pointweb', 'PointConv': 'pointconv', 'RSCNN': 'rscnn',
    'PointMLP': 'pointmlp', 'PointMLPLite': 'pointmlp',
    'PointSCNet': 'pointscnet', 'RepSurf': 'repsurf', 'PointKAN': 'pointkan', 'DELA': 'dela',
    'RandLANet': 'randlanet',
    # Supervised — attention-based
    'PCT': 'guo2021pct', 'P2P': 'p2p', 'PointTNT': 'pointtnt',
    'GlobalTransformer': 'pointtnt', 'PVT': 'pvt',
    'PointTransformer': 'zhao2021point', 'PointTransformerV2': 'wu2022point', 'PointTransformerV3': 'wu2024point',
    # Supervised — graph-based
    'DGCNN': 'wang2019dynamic', 'DeepGCN': 'deepgcn', 'CurveNet': 'curvenet',
    'GDAN': 'gdanet', 'GDANet': 'gdanet', 'MSDGCNN': 'msdgcnn', 'KANDGCNN': 'kandgcnn', 'MSDGCNN2': 'msdgcnn2',
    # Self-supervised
    'Point-MAE': 'pang2022masked', 'ACT': 'dong2023act', 'RECON': 'qi2023contrast',
    'PointGPT': 'chen2024pointgpt', 'PCP': 'pcpmae', 'PointMamba': 'liang2024pointmamba', 'Point-M2AE': 'zhang2022point', 'PointBERT': 'yu2022point',
    # PEFT strategies (when they appear as base_model in SSL:Strategy format)
    'IDPT': 'zha2023instance', 'DAPT': 'zhou2024dynamic', 'PPT': 'zhang2025positional',
    'PointGST': 'pointgst', 'VPT_Deep': 'jia2022visual', 'VPT-Deep': 'jia2022visual',
}

# Category display order
CATEGORY_ORDER = ['Point-based', 'Attention-based', 'Graph-based', 'Self-supervised']


def strip_init_source(model_name):
    """Remove [init_source] suffix from model name if present.

    Example: 'Point-MAE:DAPT [HELIAS]' -> 'Point-MAE:DAPT'
    """
    import re
    return re.sub(r'\s*\[.*?\]\s*$', '', model_name).strip()


def get_init_source(model_name):
    """Extract init_source from model name if present.

    Example: 'Point-MAE:DAPT [HELIAS]' -> 'HELIAS'
    Returns None if no init_source suffix.
    """
    import re
    match = re.search(r'\[([^\]]+)\]\s*$', model_name)
    return match.group(1) if match else None


def get_base_model_name(model_name):
    """Extract base model name from full model name (handles SSL variants like 'Point-MAE:DAPT').

    Also strips [init_source] suffix if present.
    """
    # First strip init_source suffix
    clean_name = strip_init_source(model_name)
    if ':' in clean_name:
        return clean_name.split(':')[0].strip()
    return clean_name


def normalize_model_name(name):
    """Normalize model name for matching (lowercase, no hyphens/underscores/spaces)."""
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def get_model_sort_key(model_name):
    """Get sort key for a model based on MODEL_ORDER."""
    # Strip init_source suffix for sorting
    clean_name = strip_init_source(model_name)
    base_name = get_base_model_name(model_name)

    # Try exact match
    if base_name in MODEL_ORDER:
        base_idx = MODEL_ORDER.index(base_name)
    else:
        # Try normalized match
        base_idx = len(MODEL_ORDER)
        norm_base = normalize_model_name(base_name)
        for i, m in enumerate(MODEL_ORDER):
            if normalize_model_name(m) == norm_base:
                base_idx = i
                break

    # For SSL models with strategies, add sub-ordering
    if ':' in clean_name:
        strategy = clean_name.split(':')[1].strip()
        strategy_order = {'Full Finetuning': 0, 'Full': 0, 'FF': 0, 'DAPT': 1, 'IDPT': 2, 'PPT': 3, 'GST': 4}
        sub_idx = strategy_order.get(strategy, 9)
        # Add init_source as tertiary sort key (alphabetical)
        init_src = get_init_source(model_name) or ''
        return (base_idx, sub_idx, init_src)

    init_src = get_init_source(model_name) or ''
    return (base_idx, 0, init_src)


def get_model_category(model_name):
    """Get category for a model."""
    base_name = get_base_model_name(model_name)

    if base_name in MODEL_CATEGORIES:
        return MODEL_CATEGORIES[base_name]

    # Try normalized match
    norm_base = normalize_model_name(base_name)
    for m, cat in MODEL_CATEGORIES.items():
        if normalize_model_name(m) == norm_base:
            return cat

    return 'Other'


def get_model_citation(model_name):
    """Get citation key for a model."""
    base_name = get_base_model_name(model_name)

    if base_name in MODEL_CITATIONS:
        return MODEL_CITATIONS[base_name]

    # Try normalized match
    norm_base = normalize_model_name(base_name)
    for m, cite in MODEL_CITATIONS.items():
        if normalize_model_name(m) == norm_base:
            return cite

    # Default: lowercase model name without special chars
    return norm_base


def find_cv_summaries(exp_dir):
    """Find all cv_summary.csv files in the experiment directory."""
    cv_files = []

    # Search pattern: exp_dir/*_config/*/cv_summary.csv
    pattern = os.path.join(exp_dir, '*', '*', 'cv_summary.csv')
    cv_files.extend(glob.glob(pattern))

    # Also check direct subdirectories
    pattern2 = os.path.join(exp_dir, '*', 'cv_summary.csv')
    cv_files.extend(glob.glob(pattern2))

    # Remove duplicates and sort
    cv_files = sorted(list(set(cv_files)))

    return cv_files


def extract_class_names(df):
    """Extract class names from per-class metric columns."""
    class_names = set()
    for col in df.columns:
        for prefix in PER_CLASS_PREFIXES:
            if col.startswith(f'{prefix}_') and not col.endswith('_mean') and not col.endswith('_std'):
                # Check it's not a macro/weighted aggregate
                suffix = col[len(prefix) + 1:]
                if suffix not in ['macro', 'weighted']:
                    class_names.add(suffix)
    return sorted(class_names)


def load_cv_summaries(cv_files, metrics, per_class_metrics=None, mean_only=False):
    """Load all CV summary files and extract specified metrics.

    Args:
        cv_files: List of paths to cv_summary.csv files
        metrics: List of global metrics to include
        per_class_metrics: List of per-class metric types ('precision', 'recall', 'f1')
        mean_only: If True, show only mean values without ± std

    Returns:
        DataFrame with model data, list of class names, list of per-class columns, dict of class support counts
    """
    all_data = []
    all_class_names = set()
    class_support = {}  # Will store support values from any file that has them

    # First pass: discover all class names and support values
    for cv_file in cv_files:
        try:
            df = pd.read_csv(cv_file)
            class_names = extract_class_names(df)
            all_class_names.update(class_names)

            # Extract support values if available (take from first file that has them)
            if not class_support:
                for class_name in class_names:
                    sup_col = f'support_{class_name}'
                    if sup_col in df.columns:
                        class_support[class_name] = int(df[sup_col].iloc[0])
        except Exception:
            pass

    all_class_names = sorted(all_class_names)

    # Build list of per-class metric columns to extract
    per_class_cols = []
    if per_class_metrics:
        for metric_type in per_class_metrics:
            for class_name in all_class_names:
                per_class_cols.append(f"{metric_type}_{class_name}")

    # Second pass: load data
    for cv_file in cv_files:
        try:
            df = pd.read_csv(cv_file)

            if len(df) == 0:
                print(f"  [WARN] Empty file: {cv_file}")
                continue

            # Get model name
            model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else 'Unknown'

            # Extract specified metrics (the combined "mean ± std" columns)
            row_data = {'model_name': model_name}

            # Global metrics
            for metric in metrics:
                if metric in df.columns:
                    val = df[metric].iloc[0]
                    # Format parameter columns with 2 decimal precision
                    if metric in ['total_params_M', 'trainable_params_M']:
                        try:
                            row_data[metric] = f"{float(val):.2f}"
                        except (ValueError, TypeError):
                            row_data[metric] = str(val)
                    elif metric == 'avg_epoch_time_s':
                        try:
                            row_data[metric] = f"{float(val):.2f}"
                        except (ValueError, TypeError):
                            row_data[metric] = str(val)
                    else:
                        # If value contains ± and mean_only, extract just the mean
                        if mean_only and isinstance(val, str) and '±' in val:
                            mean_val, _ = parse_mean_std(val)
                            row_data[metric] = f"{mean_val:.2f}" if mean_val is not None else val
                        else:
                            row_data[metric] = sanitize_pm(val)
                elif f'{metric}_mean' in df.columns:
                    # If only _mean/_std columns exist, create combined string
                    mean_val = df[f'{metric}_mean'].iloc[0]
                    std_val = df[f'{metric}_std'].iloc[0]
                    if mean_only:
                        row_data[metric] = f"{mean_val:.2f}"
                    else:
                        row_data[metric] = f"{mean_val:.2f}$\\pm${std_val:.2f}"
                else:
                    row_data[metric] = 'N/A'

            # Per-class metrics
            for col in per_class_cols:
                if col in df.columns:
                    val = df[col].iloc[0]
                    # If value contains ± and mean_only, extract just the mean
                    if mean_only and isinstance(val, str) and '±' in val:
                        mean_val, _ = parse_mean_std(val)
                        row_data[col] = f"{mean_val:.2f}" if mean_val is not None else val
                    else:
                        row_data[col] = sanitize_pm(val)
                elif f'{col}_mean' in df.columns:
                    mean_val = df[f'{col}_mean'].iloc[0]
                    std_val = df[f'{col}_std'].iloc[0]
                    if mean_only:
                        row_data[col] = f"{mean_val:.2f}"
                    else:
                        row_data[col] = f"{mean_val:.2f}$\\pm${std_val:.2f}"
                else:
                    row_data[col] = 'N/A'

            # Add k_folds if available
            if 'k_folds' in df.columns:
                row_data['k_folds'] = int(df['k_folds'].iloc[0])

            all_data.append(row_data)
            print(f"  [OK] Loaded: {model_name} from {cv_file}")

        except Exception as e:
            print(f"  [!!] Error loading {cv_file}: {e}")

    return pd.DataFrame(all_data), all_class_names, per_class_cols, class_support


def parse_mean_std(value_str):
    """Parse 'mean ± std' string and return (mean, std) as floats."""
    if pd.isna(value_str) or value_str == 'N/A':
        return None, None

    try:
        # Handle "XX.XX ± YY.YY" format
        match = re.match(r'([\d.]+)\s*(?:±|\$\\\\pm\$|\$\\pm\$|\\pm)\s*([\d.]+)', str(value_str))
        if match:
            return float(match.group(1)), float(match.group(2))
        # Handle plain number
        return float(value_str), 0.0
    except (ValueError, AttributeError):
        return None, None


def sanitize_pm(val):
    """Normalise Unicode `±` and ASCII `+/-` to the tight LaTeX form `mean$\\pm$std`."""
    if isinstance(val, str):
        v = val.replace('\u00b1', '$\\pm$').replace('+/-', '$\\pm$').strip()
        v = v.replace(' $\\pm$', '$\\pm$').replace('$\\pm$ ', '$\\pm$')
        return v
    return val


def _latex_escape(text):
    """Escape LaTeX special characters in free-text strings (model names,
    strategy labels, category names) so they don't break compilation.

    Preserves intentional LaTeX commands (\\cite, \\textbf, \\multirow, $\\pm$)
    by only escaping bare special chars that are NOT preceded by a backslash.
    """
    if not isinstance(text, str):
        return text
    # Order matters: & before _ to avoid double-escaping
    for ch in ('_', '&', '%', '#'):
        # Don't re-escape if already escaped
        text = text.replace(f'\\{ch}', f'\x00{ch}')  # protect existing escapes
        text = text.replace(ch, f'\\{ch}')
        text = text.replace(f'\x00{ch}', f'\\{ch}')  # restore
    return text


def _build_citation_paragraph(df):
    """Build a LaTeX paragraph that cites every model grouped by category.

    Inserted at the top of the generated document so all citations are
    resolved by BibTeX, and the tables themselves stay clean (no \\cite).
    """
    # Collect unique (base_model, category, citation_key) triples
    seen_keys = set()
    by_cat = {}  # {category: [(display_name, bib_key), ...]}

    for model_name in df['model_name'].unique():
        base = get_base_model_name(model_name)
        cat = get_model_category(model_name)
        key = get_model_citation(base)
        if key and key not in seen_keys:
            seen_keys.add(key)
            display = _latex_escape(base.replace('_', '-'))
            by_cat.setdefault(cat, []).append((display, key))

        # If the model name includes a PEFT strategy (e.g. "Point-MAE:DAPT"),
        # also cite the strategy itself.
        clean = strip_init_source(model_name)
        if ':' in clean:
            strat = clean.split(':', 1)[1].strip()
            if strat not in ('Full Finetuning', 'Full', 'FF'):
                skey = MODEL_CITATIONS.get(strat)
                if skey and skey not in seen_keys:
                    seen_keys.add(skey)
                    by_cat.setdefault('PEFT', []).append(
                        (_latex_escape(strat.replace('_', '-')), skey))

    cat_order = CATEGORY_ORDER + [c for c in by_cat if c not in CATEGORY_ORDER]
    cat_desc = {
        'Point-based': 'point-based methods',
        'Attention-based': 'attention and transformer architectures',
        'Graph-based': 'graph neural networks',
        'Self-supervised': 'self-supervised pre-training methods',
        'PEFT': 'parameter-efficient fine-tuning (PEFT) strategies',
    }

    def _join(items):
        if len(items) <= 1:
            return items[0] if items else ''
        return ', '.join(items[:-1]) + ', and ' + items[-1]

    lines = []
    lines.append('\\section*{Models and References}')
    for cat in cat_order:
        entries = by_cat.get(cat, [])
        if not entries:
            continue
        desc = cat_desc.get(cat, cat.lower())
        cited = [f'{name}~\\citep{{{key}}}' for name, key in sorted(entries)]
        if cat == 'Self-supervised':
            lines.append(
                f'Self-supervised pre-training: {_join(cited)}.')
        elif cat == 'PEFT':
            lines.append(
                f'Parameter-efficient fine-tuning: {_join(cited)}.')
        else:
            lines.append(
                f'{cat} ({desc}): {_join(cited)}.')
        lines.append('')

    return '\n'.join(lines)


def find_best_values(df, metrics, per_class_cols=None):
    """Find the best value for each metric.

    For most metrics, best = highest (accuracy, F1, etc.)
    For params and time metrics, best = lowest (smaller/faster is better)
    """
    best_indices = {}

    # Metrics where lower is better
    LOWER_IS_BETTER = {'trainable_params_M', 'total_params_M', 'avg_epoch_time_s'}

    all_metrics = list(metrics)
    if per_class_cols:
        all_metrics.extend(per_class_cols)

    for metric in all_metrics:
        if metric not in df.columns or metric == 'model_name':
            continue

        lower_better = metric in LOWER_IS_BETTER
        best_mean = float('inf') if lower_better else -float('inf')
        best_idx = None

        for idx, value in df[metric].items():
            mean, _ = parse_mean_std(value)
            if mean is not None:
                if lower_better:
                    if mean < best_mean:
                        best_mean = mean
                        best_idx = idx
                else:
                    if mean > best_mean:
                        best_mean = mean
                        best_idx = idx

        if best_idx is not None:
            best_indices[metric] = best_idx

    return best_indices


def get_display_name(metric, latex=False):
    """Get display name for a metric (handles per-class metrics).

    Args:
        metric: Metric name
        latex: If True, use LaTeX formatting with units
    """
    name_dict = LATEX_DISPLAY_NAMES if latex else METRIC_DISPLAY_NAMES

    if metric in name_dict:
        return name_dict[metric]

    # Handle per-class metrics like precision_Oak -> Prec(Oak) or Rec(Oak) (%)
    for prefix in PER_CLASS_PREFIXES:
        if metric.startswith(f'{prefix}_'):
            class_name = metric[len(prefix) + 1:]
            prefix_short = {'precision': 'Prec', 'recall': 'Rec', 'f1': 'F1'}
            if latex:
                return f"{prefix_short.get(prefix, prefix.title())}({class_name}) (\\%)"
            else:
                return f"{prefix_short.get(prefix, prefix.title())}({class_name})"

    return metric.replace('_', ' ').title()


def generate_latex_table(df, metrics, best_indices, per_class_cols=None,
                         caption="Cross-Validation Results Comparison", label="tab:cv_comparison",
                         col_widths=None, landscape=False, class_counts=None,
                         cite_in_tables=False):
    """Generate a LaTeX table grouped by category with citations, following training script order.

    Args:
        col_widths: dict with column width settings. Keys:
            - 'category': width for Category column (default: '1.8cm')
            - 'model': width for Model column (default: '2.5cm')
            - 'strategy': width for Strategy column (default: '1.5cm')
            - 'metric': width for metric columns (default: '1.2cm')
            Set to None to use auto-width (l/c format).
        landscape: If True, wrap table in landscape environment.
        class_counts: dict mapping class names to sample counts (for per-class table header).
    """
    # Default column widths for a clear table that fits the page
    default_widths = {
        'category': '1.8cm',
        'model': '2.5cm',
        'strategy': '1.5cm',
        'metric': '1.2cm',
    }
    if col_widths:
        default_widths.update(col_widths)
    widths = default_widths

    all_metrics = list(metrics)
    if per_class_cols:
        all_metrics.extend(per_class_cols)

    # Add category, sort key, and prepare data
    df = df.copy()
    df['category'] = df['model_name'].apply(get_model_category)
    df['sort_key'] = df['model_name'].apply(get_model_sort_key)
    df['original_idx'] = df.index

    # Parse SSL model names to extract base_model and strategy (handles [init_source] suffix)
    # Abbreviate 'Full Finetuning' and 'Full' to 'FF'
    # Note: init_source (e.g., [HELIAS]) is stripped and not shown in tables
    def parse_model(name):
        clean_name = strip_init_source(name)
        if ':' in clean_name:
            base, strat = clean_name.split(':', 1)
            strat = strat.strip()
            # Abbreviate Full Finetuning to FF
            if strat in ('Full Finetuning', 'Full'):
                strat = 'FF'
            return base.strip(), strat
        return clean_name, '-'

    df['base_model'] = df['model_name'].apply(lambda x: parse_model(x)[0])
    df['strategy'] = df['model_name'].apply(lambda x: parse_model(x)[1])

    # Sort by category order then by model order within category
    def category_sort_key(row):
        cat_idx = CATEGORY_ORDER.index(row['category']) if row['category'] in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (cat_idx, row['sort_key'])

    df['full_sort_key'] = df.apply(category_sort_key, axis=1)
    df = df.sort_values('full_sort_key').reset_index(drop=True)

    # Start LaTeX table
    latex_lines = []
    if landscape:
        latex_lines.append("% Requires: \\usepackage{multirow}, \\usepackage{booktabs}, \\usepackage{adjustbox}, \\usepackage{lscape}")
        latex_lines.append("\\begin{landscape}")
    else:
        latex_lines.append("% Requires: \\usepackage{multirow}, \\usepackage{booktabs}, \\usepackage{adjustbox}")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    # Use adjustbox to center table even when it exceeds page margins
    latex_lines.append("\\begin{adjustbox}{center}")

    # Column format: Category, Model, Strategy, then metrics
    # Use p{width} for fixed-width columns to control table width
    cat_col = f"p{{{widths['category']}}}" if widths.get('category') else "l"
    model_col = f"p{{{widths['model']}}}" if widths.get('model') else "l"
    strat_col = f"p{{{widths['strategy']}}}" if widths.get('strategy') else "l"
    metric_col = f"p{{{widths['metric']}}}" if widths.get('metric') else "c"
    col_format = cat_col + model_col + strat_col + metric_col * len(all_metrics)
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\toprule")

    # Header row
    header_cols = ["Category", "Model", "Strategy"]
    for metric in all_metrics:
        display_name = get_display_name(metric, latex=True)
        header_cols.append(display_name)
    latex_lines.append(" & ".join(header_cols) + " \\\\")

    # Add class counts row for per-class tables (arrow header showing n=count)
    if class_counts and per_class_cols:
        count_cols = ["", "", ""]  # Empty for Category, Model, Strategy
        for metric in all_metrics:
            # Extract class name from metric (e.g., recall_Buche -> Buche)
            class_name = None
            for prefix in PER_CLASS_PREFIXES:
                if metric.startswith(f'{prefix}_'):
                    class_name = metric[len(prefix) + 1:]
                    break
            if class_name and class_name in class_counts:
                count_cols.append(f"$\\downarrow$ n={class_counts[class_name]}")
            else:
                count_cols.append("")
        latex_lines.append(" & ".join(count_cols) + " \\\\")

    latex_lines.append("\\midrule")

    # Process rows grouped by category, then by base_model for SSL
    current_category = None
    current_base_model = None
    category_row_idx = 0
    base_model_row_idx = 0

    # Pre-calculate counts for multirow
    category_counts = df.groupby('category').size().to_dict()

    # For SSL category, we need base_model counts within category
    ssl_base_counts = {}
    ssl_df = df[df['category'] == 'Self-supervised']
    if not ssl_df.empty:
        ssl_base_counts = ssl_df.groupby('base_model').size().to_dict()

    for idx, row in df.iterrows():
        category = row['category']
        base_model = row['base_model']
        strategy = row['strategy']
        original_idx = row['original_idx']

        row_cols = []

        # Category column with multirow
        if category != current_category:
            if current_category is not None:
                latex_lines.append("\\midrule")
            current_category = category
            current_base_model = None
            category_row_idx = 0
            cat_total = category_counts.get(category, 1)
            row_cols.append(f"\\multirow{{{cat_total}}}{{*}}{{{category}}}")
        else:
            row_cols.append("")

        category_row_idx += 1

        # Model column — citation only when --use_citation_in_tables is set.
        # Otherwise, citations are in the paragraph above the table.
        display_model = _latex_escape(base_model.replace('_', '-'))
        if cite_in_tables:
            base_key = get_model_citation(base_model)
            model_with_cite = (f"{display_model}~\\citep{{{base_key}}}"
                               if base_key else display_model)
        else:
            model_with_cite = display_model

        if category == 'Self-supervised':
            # Use multirow for SSL base models with multiple strategies
            if base_model != current_base_model:
                # Add horizontal line between SSL base models only if previous had 2+ strategies
                # Use cline to exclude category column (columns 2 to end)
                if current_base_model is not None:
                    prev_count = ssl_base_counts.get(current_base_model, 1)
                    if prev_count >= 2:
                        total_cols = 3 + len(all_metrics)  # Category + Model + Strategy + metrics
                        latex_lines.append(f"\\cmidrule(lr){{2-{total_cols}}}")
                current_base_model = base_model
                base_model_row_idx = 0
                base_total = ssl_base_counts.get(base_model, 1)
                if base_total > 1:
                    row_cols.append(f"\\multirow{{{base_total}}}{{*}}{{{model_with_cite}}}")
                else:
                    row_cols.append(model_with_cite)
            else:
                row_cols.append("")
            base_model_row_idx += 1
        else:
            # Non-SSL: just model name (with citation if enabled)
            row_cols.append(model_with_cite)

        # Strategy column — also cite strategy when flag is on.
        # Normalise the VPT Deep label to plain "VPT" (citation key unchanged).
        strat_label = 'VPT' if strategy in ('VPT_Deep', 'VPT-Deep', 'VPT Deep') else strategy
        strat_display = _latex_escape(strat_label)
        if cite_in_tables and strategy not in ('-', 'FF', 'Full Finetuning', 'Full'):
            strat_key = MODEL_CITATIONS.get(strategy)
            if strat_key:
                strat_display = f"{strat_display}~\\citep{{{strat_key}}}"
        row_cols.append(strat_display)

        # Metric columns
        skip_cols = {'category', 'sort_key', 'original_idx', 'base_model', 'strategy', 'full_sort_key'}
        for metric in all_metrics:
            if metric not in row.index or metric in skip_cols:
                row_cols.append("N/A")
                continue

            value = row[metric]
            is_best = best_indices.get(metric) == original_idx

            if is_best and value != 'N/A':
                pm_match = re.match(r'([\d.]+)\$\\pm\$([\d.]+)', str(value))
                if pm_match:
                    mean_part = pm_match.group(1)
                    std_part = pm_match.group(2)
                    row_cols.append(f"\\textbf{{{mean_part}}}$\\pm$\\textbf{{{std_part}}}")
                else:
                    row_cols.append(f"\\textbf{{{value}}}")
            else:
                row_cols.append(str(value))

        latex_lines.append(" & ".join(row_cols) + " \\\\")

    # End table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{adjustbox}")
    latex_lines.append("\\end{table}")
    if landscape:
        latex_lines.append("\\end{landscape}")

    return "\n".join(latex_lines)


def generate_markdown_table(df, metrics, best_indices, per_class_cols=None):
    """Generate a Markdown table with best values in bold, ordered by category."""

    all_metrics = list(metrics)
    if per_class_cols:
        all_metrics.extend(per_class_cols)

    # Add category and sort key, then sort
    df = df.copy()
    df['category'] = df['model_name'].apply(get_model_category)
    df['sort_key'] = df['model_name'].apply(get_model_sort_key)
    df['original_idx'] = df.index

    def category_sort_key(row):
        cat_idx = CATEGORY_ORDER.index(row['category']) if row['category'] in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (cat_idx, row['sort_key'])

    df['full_sort_key'] = df.apply(category_sort_key, axis=1)
    df = df.sort_values('full_sort_key').reset_index(drop=True)

    lines = []

    # Header
    header_cols = ["Category", "Model", "Strategy"]
    for metric in all_metrics:
        display_name = get_display_name(metric)
        header_cols.append(display_name)
    lines.append("| " + " | ".join(header_cols) + " |")

    # Separator
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    # Data rows
    current_category = None
    for idx, row in df.iterrows():
        model_name = row['model_name']
        category = row['category']
        original_idx = row['original_idx']

        # Parse model name (strip init_source suffix first)
        # Abbreviate 'Full Finetuning' and 'Full' to 'FF'
        # Note: init_source (e.g., [HELIAS]) is stripped and not shown in tables
        clean_name = strip_init_source(model_name)
        if ':' in clean_name:
            base_model, strategy = clean_name.split(':', 1)
            strategy = strategy.strip()
            if strategy in ('Full Finetuning', 'Full'):
                strategy = 'FF'
        else:
            base_model = clean_name
            strategy = "-"

        # Category (show only on first row of group)
        if category != current_category:
            current_category = category
            cat_display = f"**{category}**"
        else:
            cat_display = ""

        strat_label = 'VPT' if strategy in ('VPT_Deep', 'VPT-Deep', 'VPT Deep') else strategy
        row_cols = [cat_display, base_model, strat_label]

        skip_cols = {'category', 'sort_key', 'original_idx', 'full_sort_key'}
        for metric in all_metrics:
            if metric not in df.columns or metric in skip_cols:
                row_cols.append("N/A")
                continue

            value = row[metric]
            is_best = best_indices.get(metric) == original_idx

            if is_best and value != 'N/A':
                row_cols.append(f"**{value}**")
            else:
                row_cols.append(str(value))

        lines.append("| " + " | ".join(row_cols) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate model comparison table from CV summary files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (accuracy, balanced_acc, f1_weighted, recall, precision,
  # per-class recall, params, epoch time)
  python scripts/reports/classification_comparison_report.py \\
      --exp_dir experiments/tree_dataset_cv

  # Custom metrics
  python scripts/reports/classification_comparison_report.py \\
      --exp_dir experiments/tree_dataset_cv \\
      --metrics best_accuracy balanced_acc f1_weighted kappa mcc

  # Disable per-class metrics
  python scripts/reports/classification_comparison_report.py \\
      --exp_dir experiments/tree_dataset_cv \\
      --no_per_class

  # Include per-class precision and recall (in addition to defaults)
  python scripts/reports/classification_comparison_report.py \\
      --exp_dir experiments/tree_dataset_cv \\
      --per_class precision recall f1

  # Custom output directory
  python scripts/reports/classification_comparison_report.py \\
      --exp_dir experiments/tree_dataset_cv \\
      --output_dir results/
        """
    )

    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Directory containing experiment results (e.g., experiments/tree_dataset_cv)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help=f'Global metrics to include. Default: {DEFAULT_METRICS}')
    parser.add_argument('--per_class', type=str, nargs='*', default=None,
                        choices=['precision', 'recall', 'f1'],
                        help=f'Per-class metrics to include. Default: {DEFAULT_PER_CLASS}. Use --per_class without args to disable.')
    parser.add_argument('--no_per_class', action='store_true',
                        help='Disable per-class metrics')
    parser.add_argument('--all_metrics', action='store_true',
                        help='Include all available global metrics')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated files (default: <exp_dir>/latex)')
    parser.add_argument('--output_name', type=str, default='model_comparison',
                        help='Base name for output files (default: model_comparison)')
    parser.add_argument('--no_cv', action='store_true',
                        help='Show only mean values without ± std (for non-CV single run results)')
    parser.add_argument('--principal_pretraining_data', type=str, default=None,
                        help='Principal pretraining dataset (e.g., ShapeNet, HELIAS). SSL models with this init_source stay with supervised models. Other SSL models get separate tables.')
    parser.add_argument('--class_counts', type=str, nargs='+', default=None,
                        help='Class sample counts in format "ClassName:count" (e.g., "Buche:150 Douglasie:80"). Displayed as header row in per-class tables.')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Filter classes to include in per-class tables. Can be class names or 0-based indices (e.g., "Buche Eiche" or "0 2 4"). Default: all classes.')
    parser.add_argument('--use_citation_in_tables', action='store_true',
                        help='Put \\citep{} next to each model name inside the tables. '
                             'Default: cite methods once in a paragraph above the tables.')

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.exp_dir, 'latex')
    os.makedirs(output_dir, exist_ok=True)

    # Dataset name derived from the experiment directory — used as a caption prefix
    # so readers immediately know which benchmark a table belongs to.
    dataset_name = os.path.basename(os.path.normpath(args.exp_dir))

    # Set metrics
    if args.all_metrics:
        metrics = DEFAULT_METRICS.copy()
    elif args.metrics:
        metrics = args.metrics
    else:
        metrics = DEFAULT_METRICS.copy()

    # Set per-class metrics
    if args.no_per_class:
        per_class = None
    elif args.per_class is not None:
        # User explicitly provided --per_class (possibly empty list)
        per_class = args.per_class if args.per_class else None
    else:
        # Use default
        per_class = DEFAULT_PER_CLASS.copy()

    print("=" * 70)
    print("MODEL COMPARISON TABLE GENERATOR")
    print("=" * 70)
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Global metrics: {', '.join(metrics)}")
    if per_class:
        print(f"Per-class metrics: {', '.join(per_class)}")
    print("-" * 70)

    # Find CV summary files
    print("\nSearching for CV summary files...")
    cv_files = find_cv_summaries(args.exp_dir)

    if not cv_files:
        print("  [!!] No cv_summary.csv files found!")
        print(f"  Searched in: {args.exp_dir}")
        return

    print(f"  Found {len(cv_files)} CV summary files")

    # Load data
    print("\nLoading CV summaries...")
    df, class_names, per_class_cols, class_support_from_data = load_cv_summaries(cv_files, metrics, per_class, mean_only=args.no_cv)

    if len(df) == 0:
        print("  [!!] No valid data loaded!")
        return

    print(f"\n  Loaded {len(df)} models")
    if class_names:
        print(f"  Detected classes: {', '.join(class_names)}")
    if per_class_cols:
        print(f"  Per-class columns: {len(per_class_cols)}")

    # Get class counts: prefer from results, fallback to manual --class_counts argument
    class_counts_dict = None
    if class_support_from_data:
        class_counts_dict = class_support_from_data
        print(f"  Class counts (from results): {class_counts_dict}")
    elif args.class_counts:
        class_counts_dict = {}
        for item in args.class_counts:
            if ':' in item:
                name, count = item.split(':', 1)
                class_counts_dict[name.strip()] = int(count.strip())
        print(f"  Class counts (manual): {class_counts_dict}")

    # Filter classes if --classes is specified
    if args.classes and per_class_cols and class_names:
        # Determine which classes to keep
        selected_classes = []
        for cls_spec in args.classes:
            if cls_spec.isdigit():
                # Index-based selection
                idx = int(cls_spec)
                if 0 <= idx < len(class_names):
                    selected_classes.append(class_names[idx])
            else:
                # Name-based selection
                if cls_spec in class_names:
                    selected_classes.append(cls_spec)

        if selected_classes:
            # Filter per_class_cols to only include selected classes
            filtered_cols = []
            for col in per_class_cols:
                for prefix in PER_CLASS_PREFIXES:
                    if col.startswith(f'{prefix}_'):
                        class_name = col[len(prefix) + 1:]
                        if class_name in selected_classes:
                            filtered_cols.append(col)
                        break
            per_class_cols = filtered_cols
            print(f"  Filtered to classes: {', '.join(selected_classes)}")
            print(f"  Filtered per-class columns: {len(per_class_cols)}")

    # Sort by model name
    df = df.sort_values('model_name').reset_index(drop=True)

    # Find best values
    best_indices = find_best_values(df, metrics, per_class_cols)

    # Generate outputs
    print("\nGenerating output files...")

    # 1. CSV file
    csv_path = os.path.join(output_dir, f"{args.output_name}.csv")
    # Select model_name, global metrics, and per-class metrics for CSV
    csv_cols = ['model_name'] + [m for m in metrics if m in df.columns]
    if per_class_cols:
        csv_cols.extend([c for c in per_class_cols if c in df.columns])
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"  [OK] CSV saved: {csv_path}")

    # 2. LaTeX tables
    # Combine all metrics in one table: performance metrics first, then params/time
    # Order: performance metrics, total_params_M, trainable_params_M, avg_epoch_time_s
    perf_metrics = [m for m in metrics if m not in ['total_params_M', 'trainable_params_M', 'avg_epoch_time_s']]
    info_metrics = [m for m in ['total_params_M', 'trainable_params_M', 'avg_epoch_time_s'] if m in metrics]
    all_metrics_ordered = perf_metrics + info_metrics

    # Start building the complete LaTeX document
    latex_doc_lines = []
    latex_doc_lines.append("% Auto-generated LaTeX document with model comparison tables")
    latex_doc_lines.append("% Compile with: pdflatex <filename>.tex")
    latex_doc_lines.append("")
    latex_doc_lines.append("\\documentclass{article}")
    latex_doc_lines.append("\\usepackage[utf8]{inputenc}")
    latex_doc_lines.append("\\usepackage{booktabs}")
    latex_doc_lines.append("\\usepackage{multirow}")
    latex_doc_lines.append("\\usepackage{adjustbox}")
    latex_doc_lines.append("\\usepackage{lscape}")
    latex_doc_lines.append("\\usepackage[numbers,sort&compress]{natbib}")
    latex_doc_lines.append("\\usepackage[margin=1in]{geometry}")
    latex_doc_lines.append("\\bibliographystyle{unsrtnat}")
    latex_doc_lines.append("")
    latex_doc_lines.append("\\begin{document}")
    latex_doc_lines.append("")

    # ── Citation paragraph: cite every model ABOVE the tables ──
    # Skipped when --use_citation_in_tables is set (cells carry citations instead).
    if not args.use_citation_in_tables:
        latex_doc_lines.append(_build_citation_paragraph(df))
        latex_doc_lines.append("")
        latex_doc_lines.append("\\clearpage")
        latex_doc_lines.append("")

    # If --principal_pretraining_data is specified:
    # - SSL models with that init_source stay with supervised models in main tables
    # - SSL models with OTHER init_sources get separate tables (3 tables per unique init_source)
    if args.principal_pretraining_data:
        principal_data = args.principal_pretraining_data.upper()

        # Add helper columns
        df['_category'] = df['model_name'].apply(get_model_category)
        df['_init_source'] = df['model_name'].apply(lambda x: (get_init_source(x) or '').upper())

        # Main tables: supervised models + SSL models with principal pretraining data
        def is_main_model(row):
            if row['_category'] != 'Self-supervised':
                return True  # Supervised model
            return row['_init_source'] == principal_data  # SSL with principal data

        df_main = df[df.apply(is_main_model, axis=1)].copy().reset_index(drop=True)

        # Other SSL models: SSL with init_source != principal_data
        df_other_ssl = df[
            (df['_category'] == 'Self-supervised')
            & (df['_init_source'] != principal_data)
            & (df['_init_source'] != '')
        ].copy().reset_index(drop=True)

        # Get unique other init_sources
        other_init_sources = df_other_ssl['_init_source'].unique().tolist()

        # Find best values for main
        best_main = find_best_values(df_main, metrics, per_class_cols) if len(df_main) > 0 else {}

        # Column widths for combined table
        combined_col_widths = {'category': '1.8cm', 'model': '2.3cm', 'strategy': '1cm', 'metric': '1.5cm'}
        perclass_col_widths = {'category': '1.8cm', 'model': '2.3cm', 'strategy': '1cm', 'metric': '1.5cm'}

        # === MAIN TABLES (supervised + SSL with principal data) ===
        # Table 1: Combined Performance + Parameters table (portrait)
        if len(df_main) > 0:
            latex_doc_lines.append(generate_latex_table(
                df_main, all_metrics_ordered, best_main, per_class_cols=None,
                caption=f"{dataset_name} — Model Comparison: Performance Metrics and Parameters (Principal: {principal_data})",
                label="tab:comparison_main",
                col_widths=combined_col_widths,
                landscape=False,
                cite_in_tables=args.use_citation_in_tables
            ))
            latex_doc_lines.append("")
            latex_doc_lines.append("\\clearpage")
            latex_doc_lines.append("")

        # Table 2: Main Per-class
        if per_class_cols and len(df_main) > 0:
            latex_doc_lines.append(generate_latex_table(
                df_main, [], best_main, per_class_cols=per_class_cols,
                caption=f"{dataset_name} — Per-Class Recall (Principal: {principal_data})",
                label="tab:perclass_main",
                col_widths=perclass_col_widths,
                landscape=True,
                class_counts=class_counts_dict,
                cite_in_tables=args.use_citation_in_tables
            ))
            latex_doc_lines.append("")
            latex_doc_lines.append("\\clearpage")
            latex_doc_lines.append("")

        # === SEPARATE TABLES FOR EACH OTHER INIT_SOURCE ===
        for idx, init_src in enumerate(sorted(other_init_sources)):
            df_src = df_other_ssl[df_other_ssl['_init_source'] == init_src].copy().reset_index(drop=True)
            if len(df_src) == 0:
                continue

            best_src = find_best_values(df_src, metrics, per_class_cols)
            label_suffix = init_src.lower().replace(' ', '_')

            # Combined Performance + Parameters table for this init_source (portrait)
            latex_doc_lines.append(generate_latex_table(
                df_src, all_metrics_ordered, best_src, per_class_cols=None,
                caption=f"{dataset_name} — Model Comparison: Performance Metrics and Parameters (Self-Supervised, pretrained on {init_src})",
                label=f"tab:comparison_{label_suffix}",
                col_widths=combined_col_widths,
                landscape=False,
                cite_in_tables=args.use_citation_in_tables
            ))
            latex_doc_lines.append("")
            latex_doc_lines.append("\\clearpage")
            latex_doc_lines.append("")

            # Per-class table for this init_source
            if per_class_cols:
                latex_doc_lines.append(generate_latex_table(
                    df_src, [], best_src, per_class_cols=per_class_cols,
                    caption=f"{dataset_name} — Per-Class Recall (Self-Supervised, pretrained on {init_src})",
                    label=f"tab:perclass_{label_suffix}",
                    col_widths=perclass_col_widths,
                    landscape=True,
                    class_counts=class_counts_dict,
                    cite_in_tables=args.use_citation_in_tables
                ))
                latex_doc_lines.append("")
                latex_doc_lines.append("\\clearpage")
                latex_doc_lines.append("")

        print(f"  Generated tables: {len(df_main)} main models (supervised + SSL[{principal_data}])")
        if other_init_sources:
            print(f"  Separate SSL tables for: {', '.join(other_init_sources)}")
            for init_src in sorted(other_init_sources):
                count = len(df_other_ssl[df_other_ssl['_init_source'] == init_src])
                print(f"    - {init_src}: {count} models")

    else:
        # Single combined table with all metrics: performance + params + time (portrait)
        combined_col_widths = {
            'category': '1.8cm',
            'model': '2.3cm',
            'strategy': '1cm',
            'metric': '1.5cm',
        }
        latex_combined_content = generate_latex_table(
            df, all_metrics_ordered, best_indices, per_class_cols=None,
            caption=f"{dataset_name} — Cross-Validation Results - Performance Metrics and Model Parameters",
            label="tab:cv_comparison",
            col_widths=combined_col_widths,
            landscape=False,
            cite_in_tables=args.use_citation_in_tables
        )
        latex_doc_lines.append(latex_combined_content)
        latex_doc_lines.append("")
        latex_doc_lines.append("\\clearpage")
        latex_doc_lines.append("")

        if per_class_cols:
            perclass_col_widths = {
                'category': '1.8cm',
                'model': '2.3cm',
                'strategy': '1cm',
                'metric': '1.5cm',
            }
            latex_perclass_content = generate_latex_table(
                df, [], best_indices, per_class_cols=per_class_cols,
                caption=f"{dataset_name} — Cross-Validation Results - Per-Class Recall",
                label="tab:cv_perclass",
                col_widths=perclass_col_widths,
                landscape=True,
                class_counts=class_counts_dict,
                cite_in_tables=args.use_citation_in_tables
            )
            latex_doc_lines.append(latex_perclass_content)
            latex_doc_lines.append("")

    # Bibliography — resolve \cite{} commands from references.bib
    # The .bib file must be in the same directory or on BIBINPUTS path
    # when compiling: pdflatex <file> && bibtex <file> && pdflatex <file> x2
    latex_doc_lines.append("")
    latex_doc_lines.append("\\bibliography{references}")
    latex_doc_lines.append("")

    # End document
    latex_doc_lines.append("\\end{document}")

    # Write single compiled .tex file
    latex_full_path = os.path.join(output_dir, f"{args.output_name}_tables.tex")
    with open(latex_full_path, 'w') as f:
        f.write("\n".join(latex_doc_lines))
    print(f"  [OK] LaTeX (complete document) saved: {latex_full_path}")

    from _citations import copy_references_bib
    copy_references_bib(output_dir)

    # 3. Markdown table
    md_path = os.path.join(output_dir, f"{args.output_name}.md")
    md_content = generate_markdown_table(df, metrics, best_indices, per_class_cols)
    with open(md_path, 'w') as f:
        f.write(f"# Model Comparison\n\n{md_content}\n")
    print(f"  [OK] Markdown saved: {md_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(md_content)
    print("=" * 70)

    # Print best model for each metric
    print("\nBest Models (Global Metrics):")
    for metric in metrics:
        if metric in best_indices:
            best_idx = best_indices[metric]
            best_model = df.loc[best_idx, 'model_name']
            best_value = df.loc[best_idx, metric]
            display_name = get_display_name(metric)
            print(f"  {display_name}: {best_model} ({best_value})")

    if per_class_cols:
        print("\nBest Models (Per-Class Metrics):")
        for col in per_class_cols:
            if col in best_indices:
                best_idx = best_indices[col]
                best_model = df.loc[best_idx, 'model_name']
                best_value = df.loc[best_idx, col]
                display_name = get_display_name(col)
                print(f"  {display_name}: {best_model} ({best_value})")

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
