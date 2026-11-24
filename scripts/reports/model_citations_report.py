#!/usr/bin/env python3
"""
Generate a LaTeX paragraph that cites every model from the experiment
results, grouped by category — ready to paste into a paper.

Usage:
    # From experiment results (reads model_name from cv_summary.csv)
    python scripts/reports/model_citations_report.py --exp_dir experiments/STPCTLS

    # All library models (no experiment dir needed)
    python scripts/reports/model_citations_report.py --all

    # Custom output
    python scripts/reports/model_citations_report.py --all --output_dir docs/

Output:
    model_citations.tex  — a single \\input-able .tex snippet with one
                           paragraph per category, each model cited inline.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Model metadata — same source of truth as the table generators
# ─────────────────────────────────────────────────────────────

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
    'Point-MAE': 'pang2022masked', 'PointMAE': 'pang2022masked',
    'ACT': 'dong2023act', 'RECON': 'qi2023contrast', 'ReCon': 'qi2023contrast',
    'PointGPT': 'chen2024pointgpt', 'PCP': 'pcpmae', 'PCP-MAE': 'pcpmae',
    'Point-M2AE': 'zhang2022point', 'PointM2AE': 'zhang2022point',
    'PointBERT': 'yu2022point', 'Point-BERT': 'yu2022point',
    # PEFT
    'IDPT': 'zha2023instance', 'DAPT': 'zhou2024dynamic', 'PPT': 'zhang2025positional',
    'PointGST': 'pointgst', 'VPT_Deep': 'jia2022visual', 'VPT-Deep': 'jia2022visual',
}

MODEL_CATEGORIES = {
    # Point-based
    'PointNet': 'Point-based', 'PointNet2_SSG': 'Point-based', 'PointNet2_MSG': 'Point-based',
    'SONet': 'Point-based', 'PPFNet': 'Point-based', 'PointCNN': 'Point-based',
    'PointWeb': 'Point-based', 'PointConv': 'Point-based', 'RSCNN': 'Point-based',
    'PointMLP': 'Point-based', 'PointMLPLite': 'Point-based',
    'PointSCNet': 'Point-based', 'RepSurf': 'Point-based', 'PointKAN': 'Point-based', 'DELA': 'Point-based',
    'RandLANet': 'Point-based',
    # Attention-based
    'PCT': 'Attention-based', 'P2P': 'Attention-based', 'PointTNT': 'Attention-based',
    'GlobalTransformer': 'Attention-based', 'PVT': 'Attention-based',
    'PointTransformer': 'Attention-based', 'PointTransformerV2': 'Attention-based', 'PointTransformerV3': 'Attention-based',
    # Graph-based
    'DGCNN': 'Graph-based', 'DeepGCN': 'Graph-based', 'CurveNet': 'Graph-based',
    'GDAN': 'Graph-based', 'GDANet': 'Graph-based', 'MSDGCNN': 'Graph-based', 'KANDGCNN': 'Graph-based', 'MSDGCNN2': 'Graph-based',
    # Self-supervised
    'Point-MAE': 'Self-supervised', 'PointMAE': 'Self-supervised',
    'ACT': 'Self-supervised', 'RECON': 'Self-supervised', 'ReCon': 'Self-supervised',
    'PointGPT': 'Self-supervised', 'PCP': 'Self-supervised', 'PCP-MAE': 'Self-supervised',
    'Point-M2AE': 'Self-supervised', 'PointM2AE': 'Self-supervised',
    'PointBERT': 'Self-supervised', 'Point-BERT': 'Self-supervised',
    # PEFT
    'IDPT': 'PEFT', 'DAPT': 'PEFT', 'PPT': 'PEFT',
    'PointGST': 'PEFT', 'VPT_Deep': 'PEFT', 'VPT-Deep': 'PEFT',
}

# Display names for LaTeX (escape underscores)
DISPLAY_NAMES = {
    'PointNet2_SSG': 'PointNet++ (SSG)',
    'PointNet2_MSG': 'PointNet++ (MSG)',
    'PointMLPLite': 'PointMLP-Elite',
    'VPT_Deep': 'VPT-Deep',
    'Point-MAE': 'Point-MAE',
    'Point-BERT': 'Point-BERT',
    'Point-M2AE': 'Point-M2AE',
    'PCP-MAE': 'PCP-MAE',
    'MSDGCNN2': 'MS-DGCNN++',
}

CATEGORY_ORDER = ['Point-based', 'Attention-based', 'Graph-based', 'Self-supervised', 'PEFT']

CATEGORY_DESCRIPTIONS = {
    'Point-based': 'point-based methods that operate directly on raw coordinates',
    'Attention-based': 'attention and transformer architectures',
    'Graph-based': 'graph neural networks that construct local neighbourhood graphs',
    'Self-supervised': 'self-supervised pre-training methods',
    'PEFT': 'parameter-efficient fine-tuning strategies',
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _latex_escape(text):
    if not isinstance(text, str):
        return text
    for ch in ('_', '&', '%', '#'):
        text = text.replace(f'\\{ch}', f'\x00{ch}')
        text = text.replace(ch, f'\\{ch}')
        text = text.replace(f'\x00{ch}', f'\\{ch}')
    return text


def _display(name):
    """Return a LaTeX-safe display name for a model."""
    if name in DISPLAY_NAMES:
        return _latex_escape(DISPLAY_NAMES[name])
    return _latex_escape(name.replace('_', '-'))


def _cite_key(name):
    """Return the BibTeX key for a model, or None if unknown."""
    return MODEL_CITATIONS.get(name)


def _category(name):
    return MODEL_CATEGORIES.get(name, 'Other')


def _strip_strategy(model_name):
    """Extract (base_model, strategy) from 'Model:Strategy' or 'Model' format."""
    import re
    # Strip [init_source] suffix
    clean = re.sub(r'\s*\[.*?\]\s*$', '', model_name).strip()
    if ':' in clean:
        base, strat = clean.split(':', 1)
        strat = strat.strip()
        if strat in ('Full Finetuning', 'Full', 'FF'):
            return base.strip(), None
        return base.strip(), strat.strip()
    return clean, None


def _join_names(items):
    """Join a list of 'Name~\\citep{key}' strings with commas and 'and'."""
    if len(items) == 0:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ', and ' + items[-1]


# ─────────────────────────────────────────────────────────────
# Model discovery
# ─────────────────────────────────────────────────────────────

def discover_models_from_experiments(exp_dir):
    """Find all model names from cv_summary.csv files."""
    patterns = [
        os.path.join(exp_dir, '*', 'cv_summary.csv'),
        os.path.join(exp_dir, '*', '*', 'cv_summary.csv'),
    ]
    names = set()
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                df = pd.read_csv(path)
                if 'model_name' in df.columns:
                    names.add(df['model_name'].iloc[0])
            except Exception:
                pass
    return names


def get_all_library_models():
    """Return the full set of model names in the library."""
    return set(MODEL_CITATIONS.keys())


# ─────────────────────────────────────────────────────────────
# Paragraph generation
# ─────────────────────────────────────────────────────────────

def build_paragraph(model_names):
    """Build a LaTeX paragraph citing all models grouped by category.

    Returns a string ready to \\input{} into a paper.
    """
    # Deduplicate: extract base model (ignore strategy suffix)
    seen_keys = set()   # deduplicate by citation key
    models_by_cat = {}  # {category: [(display, cite_key), ...]}

    for raw_name in sorted(model_names):
        base, strategy = _strip_strategy(raw_name)

        # Add the base model
        key = _cite_key(base)
        if key and key not in seen_keys:
            seen_keys.add(key)
            cat = _category(base)
            models_by_cat.setdefault(cat, []).append((_display(base), key))

        # Add the PEFT strategy itself (if present)
        if strategy:
            skey = _cite_key(strategy)
            if skey and skey not in seen_keys:
                seen_keys.add(skey)
                scat = _category(strategy)
                models_by_cat.setdefault(scat, []).append((_display(strategy), skey))

    # Build paragraph — wrapped in a compilable standalone document
    lines = []
    lines.append('% Auto-generated LaTeX document — model citations paragraph')
    lines.append('% Compile with: pdflatex <filename>.tex  (then bibtex if using \\bibliography)')
    lines.append('\\documentclass[a4paper,11pt]{article}')
    lines.append('\\usepackage[utf8]{inputenc}')
    lines.append('\\usepackage[numbers,sort&compress]{natbib}')
    lines.append('\\usepackage[margin=1in]{geometry}')
    lines.append('\\bibliographystyle{unsrtnat}')
    lines.append('\\begin{document}')
    lines.append('')

    for cat in CATEGORY_ORDER:
        entries = models_by_cat.get(cat, [])
        if not entries:
            continue

        desc = CATEGORY_DESCRIPTIONS.get(cat, cat.lower())
        cited = [f'{name}~\\citep{{{key}}}' for name, key in entries]
        joined = _join_names(cited)

        n = len(entries)
        count_word = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                      6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}.get(n, str(n))

        if cat == 'Self-supervised':
            lines.append(
                f'For self-supervised pre-training, \\textsc{{LIDARLearn}} integrates '
                f'{count_word} methods: {joined}.'
            )
        elif cat == 'PEFT':
            lines.append(
                f'Five parameter-efficient fine-tuning (PEFT) strategies are supported: '
                f'{joined}.'
            )
        else:
            lines.append(
                f'The library includes {count_word} {desc}: {joined}.'
            )

        lines.append('')

    # Total count
    total = len(seen_keys)
    lines.append(
        f'In total, \\textsc{{LIDARLearn}} provides {total} distinct methods '
        f'spanning supervised learning, self-supervised pre-training, and '
        f'parameter-efficient transfer, all accessible through a unified '
        f'configuration and training interface.'
    )

    lines.append('')
    lines.append('\\bibliography{references}')
    lines.append('\\end{document}')

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate a LaTeX paragraph citing every supported model.',
    )
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory to discover models from cv_summary.csv files')
    parser.add_argument('--all', action='store_true',
                        help='Cite all library models (ignores --exp_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <exp_dir>/latex, or docs/ if --all)')
    args = parser.parse_args()

    if args.all:
        model_names = get_all_library_models()
        print(f"Using all {len(model_names)} library models")
    elif args.exp_dir:
        model_names = discover_models_from_experiments(args.exp_dir)
        print(f"Discovered {len(model_names)} models from {args.exp_dir}")
        if not model_names:
            print("[ERROR] No cv_summary.csv files found", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    paragraph = build_paragraph(model_names)

    if args.output_dir:
        output_dir = args.output_dir
    elif args.exp_dir:
        output_dir = os.path.join(args.exp_dir, 'latex')
    else:
        output_dir = 'docs'
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'model_citations.tex')
    with open(out_path, 'w') as f:
        f.write(paragraph + '\n')

    sys.path.insert(0, str(Path(__file__).parent))
    from _citations import copy_references_bib
    copy_references_bib(output_dir)

    print(f"\nSaved to: {out_path}")
    print(f"\nPreview:\n{'=' * 70}")
    print(paragraph)
    print('=' * 70)
    print(f"\nUsage in your paper:")
    print(f"  \\input{{model_citations}}")


if __name__ == '__main__':
    main()
