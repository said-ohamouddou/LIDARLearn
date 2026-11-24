"""
Shared citation helpers for the report generators.

All table generators support a `--use_citation_in_tables` flag. When set, each
model name in a table cell gets a `\\citep{<bib_key>}` inline citation. When
unset (default), no citations appear inside tables — instead a paragraph is
prepended above the tables that cites every method once.

The bib keys here must exist in `scripts/reports/references.bib`.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

# Canonical location of the bibliography used by every report generator.
REFERENCES_BIB = Path(__file__).resolve().parent / "references.bib"


def copy_references_bib(output_dir) -> Path | None:
    """Copy the canonical references.bib next to a generated .tex file.

    Returns the destination path on success, or None if the source is missing.
    Overwrites any existing file at the destination so the bib stays in sync
    with scripts/reports/references.bib.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "references.bib"
    if not REFERENCES_BIB.is_file():
        print(f"  [WARN] {REFERENCES_BIB} not found — \\cite{{}} will show [?].")
        return None
    shutil.copy2(REFERENCES_BIB, dst)
    print(f"  [OK] Copied references.bib to {output_dir}{os.sep}")
    return dst


# ---------------------------------------------------------------------------
# Canonical model-name → bib-key mapping
# (covers the display names used throughout the repo)
# ---------------------------------------------------------------------------
MODEL_CITATIONS: dict[str, str] = {
    # Supervised — point-based
    'PointNet':            'qi2017pointnet',
    'PointNet2_SSG':       'qi2017pointnet++',
    'PointNet2_MSG':       'qi2017pointnet++',
    'PointNet2':           'qi2017pointnet++',
    'SONet':               'sonet',
    'SO-Net':              'sonet',
    'PPFNet':              'ppfnet',
    'PointCNN':            'pointcnn',
    'PointWeb':            'pointweb',
    'PointConv':           'pointconv',
    'RSCNN':               'rscnn',
    'PointMLP':            'pointmlp',
    'PointMLPElite':       'pointmlp',
    'PointSCNet':          'pointscnet',
    'RepSurf':             'repsurf',
    'RepSurf2x':           'repsurf',
    'RapSurf':             'repsurf',
    'PointKAN':            'pointkan',
    'DELA':                'dela',
    'RandLANet':           'randlanet',
    'RandLA-Net':          'randlanet',
    # Supervised — attention-based
    'PCT':                 'guo2021pct',
    'P2P':                 'p2p',
    'PointTNT':            'pointtnt',
    'Point-TnT':           'pointtnt',
    'GlobalTransformer':   'pointtnt',
    'PVT':                 'pvt',
    'PointTransformer':    'zhao2021point',
    'PointTransformerV2':  'wu2022point',
    'PointTransformerV3':  'wu2024point',
    # Supervised — graph-based
    'DGCNN':               'wang2019dynamic',
    'DeepGCN':             'deepgcn',
    'CurveNet':            'curvenet',
    'GDAN':                'gdanet',
    'GDANet':              'gdanet',
    'MSDGCNN':             'msdgcnn',
    'MS-DGCNN':            'msdgcnn',
    'KANDGCNN':            'kandgcnn',
    'MSDGCNN2':            'msdgcnn2',
    'MS-DGCNN++':          'msdgcnn2',
    # Self-supervised
    'Point-MAE':           'pang2022masked',
    'PointMAE':            'pang2022masked',
    'ACT':                 'dong2023act',
    'RECON':               'qi2023contrast',
    'ReCon':               'qi2023contrast',
    'PointGPT':            'chen2024pointgpt',
    'PCP':                 'pcpmae',
    'PCP-MAE':             'pcpmae',
    'PCPMAE':              'pcpmae',
    'PointMamba':          'liang2024pointmamba',
    'Point-M2AE':          'zhang2022point',
    'PointM2AE':           'zhang2022point',
    'PointBERT':           'yu2022point',
    'Point-BERT':          'yu2022point',
    # PEFT strategies
    'IDPT':                'zha2023instance',
    'DAPT':                'zhou2024dynamic',
    'PPT':                 'zhang2025positional',
    'PointGST':            'pointgst',
    'GST':                 'pointgst',
    'VPT_Deep':            'jia2022visual',
    'VPT-Deep':            'jia2022visual',
}


def _normalize(name: str) -> str:
    """Strip common separators so lookups are forgiving."""
    return name.replace('-', '').replace('_', '').replace(' ', '').lower()


_NORMALIZED = {_normalize(k): v for k, v in MODEL_CITATIONS.items()}


def get_cite_key(name: str) -> str | None:
    """Return bib-key for a model identifier, or None if not found.

    For composite labels like "pointmae_dapt" (backbone + PEFT), returns the
    backbone key — the PEFT strategy is cited separately via `get_peft_key`.
    """
    if not name:
        return None
    if name in MODEL_CITATIONS:
        return MODEL_CITATIONS[name]
    key = _NORMALIZED.get(_normalize(name))
    if key:
        return key
    # composite "<backbone>_<strategy>"
    parts = name.split('_')
    if len(parts) >= 2:
        head = _NORMALIZED.get(_normalize(parts[0]))
        if head:
            return head
    return None


def cite_inline(name: str, display: str | None = None,
                escape_underscore: bool = True) -> str:
    """Return `{display}~\\citep{{key}}` — or just `{display}` if no key.

    `escape_underscore` replaces '_' with '\\_' in the display text.
    """
    text = display if display is not None else name
    if escape_underscore:
        text = text.replace('_', r'\_')
    key = get_cite_key(name)
    if key:
        return f"{text}~\\citep{{{key}}}"
    return text


# ---------------------------------------------------------------------------
# Shared table style helpers — matches the classification_comparison_report
# layout: Category | Model | Strategy | Metric columns, with \multirow for the
# category (and for SSL base models that have several PEFT strategies), best
# values bolded per column, booktabs, scriptsize, adjustbox.
# ---------------------------------------------------------------------------

MODEL_CATEGORIES: dict[str, str] = {
    # Point-based
    'PointNet':            'Point-based',
    'PointNet2':           'Point-based',
    'PointNet2_SSG':       'Point-based',
    'PointNet2_MSG':       'Point-based',
    'SONet':               'Point-based',
    'SO-Net':              'Point-based',
    'PPFNet':              'Point-based',
    'PointCNN':            'Point-based',
    'PointWeb':            'Point-based',
    'PointConv':           'Point-based',
    'RSCNN':               'Point-based',
    'PointMLP':            'Point-based',
    'PointMLPElite':       'Point-based',
    'PointSCNet':          'Point-based',
    'RepSurf':             'Point-based',
    'RepSurf2x':           'Point-based',
    'RapSurf':             'Point-based',
    'PointKAN':            'Point-based',
    'DELA':                'Point-based',
    'RandLANet':           'Point-based',
    'RandLA-Net':          'Point-based',
    # Attention-based
    'PCT':                 'Attention-based',
    'P2P':                 'Attention-based',
    'PointTNT':            'Attention-based',
    'Point-TnT':           'Attention-based',
    'GlobalTransformer':   'Attention-based',
    'PVT':                 'Attention-based',
    'PointTransformer':    'Attention-based',
    'PointTransformerV2':  'Attention-based',
    'PointTransformerV3':  'Attention-based',
    # Graph-based
    'DGCNN':               'Graph-based',
    'DeepGCN':             'Graph-based',
    'CurveNet':            'Graph-based',
    'GDAN':                'Graph-based',
    'GDANet':              'Graph-based',
    'MSDGCNN':             'Graph-based',
    'MS-DGCNN':            'Graph-based',
    'KANDGCNN':            'Graph-based',
    'MSDGCNN2':            'Graph-based',
    'MS-DGCNN++':          'Graph-based',
    # Self-supervised
    'Point-MAE':           'Self-supervised',
    'PointMAE':            'Self-supervised',
    'ACT':                 'Self-supervised',
    'RECON':               'Self-supervised',
    'ReCon':               'Self-supervised',
    'PointGPT':            'Self-supervised',
    'PCP':                 'Self-supervised',
    'PCP-MAE':             'Self-supervised',
    'PCPMAE':              'Self-supervised',
    'PointMamba':          'Self-supervised',
    'Point-M2AE':          'Self-supervised',
    'PointM2AE':           'Self-supervised',
    'PointBERT':           'Self-supervised',
    'Point-BERT':          'Self-supervised',
}

CATEGORY_ORDER = ['Point-based', 'Attention-based', 'Graph-based', 'Self-supervised']

# Preferred row order inside a category
MODEL_ORDER = [
    # Point-based
    'PointNet', 'PointNet2', 'PointNet2_SSG', 'PointNet2_MSG',
    'SONet', 'PPFNet', 'PointCNN', 'PointWeb', 'PointConv', 'RSCNN',
    'PointMLP', 'PointMLPElite', 'PointSCNet', 'RepSurf', 'RepSurf2x', 'PointKAN', 'DELA',
    'RandLANet',
    # Attention-based
    'PCT', 'P2P', 'PointTNT', 'GlobalTransformer', 'PVT',
    'PointTransformer', 'PointTransformerV2', 'PointTransformerV3',
    # Graph-based
    'DGCNN', 'DeepGCN', 'CurveNet', 'GDAN', 'GDANet',
    'MSDGCNN', 'KANDGCNN', 'MSDGCNN2',
    # Self-supervised
    'Point-MAE', 'ACT', 'RECON', 'PointGPT', 'Point-M2AE', 'PointBERT', 'PCP',
    'PointMamba',
]

# PEFT strategies — order inside each SSL base model
STRATEGY_ORDER = ['FF', 'Full', 'Full Finetuning', '-', 'DAPT', 'IDPT', 'PPT', 'GST', 'PointGST']


# Lowercase labels (as used by part-seg folder names) → canonical names
_LOWERCASE_ALIASES = {
    'pointnet':            'PointNet',
    'pointnet2':           'PointNet2',
    'pointnet2_ssg':       'PointNet2_SSG',
    'pointnet2_msg':       'PointNet2_MSG',
    'sonet':               'SONet',
    'ppfnet':              'PPFNet',
    'pointcnn':            'PointCNN',
    'pointweb':            'PointWeb',
    'pointconv':           'PointConv',
    'rscnn':               'RSCNN',
    'pointmlp':            'PointMLP',
    'pointmlpelite':       'PointMLPElite',
    'pointmlp_elite':      'PointMLPElite',
    'pointscnet':          'PointSCNet',
    'repsurf':             'RepSurf',
    'rapsurf':             'RepSurf',
    'repsurf2x':           'RepSurf2x',
    'repsurf_2x':          'RepSurf2x',
    'pointkan':            'PointKAN',
    'dela':                'DELA',
    'randlanet':           'RandLANet',
    'randlenet':           'RandLANet',
    'pct':                 'PCT',
    'p2p':                 'P2P',
    'pointtnt':            'PointTNT',
    'globaltransformer':   'GlobalTransformer',
    'pvt':                 'PVT',
    'pointtransformer':    'PointTransformer',
    'pointtransformerv2':  'PointTransformerV2',
    'pointtransformerv3':  'PointTransformerV3',
    'dgcnn':               'DGCNN',
    'deepgcn':             'DeepGCN',
    'curvenet':            'CurveNet',
    'gdan':                'GDAN',
    'gdanet':              'GDANet',
    'msdgcnn':             'MSDGCNN',
    'kandgcnn':            'KANDGCNN',
    'msdgcnn2':            'MSDGCNN2',
    'pointmae':            'Point-MAE',
    'act':                 'ACT',
    'recon':               'RECON',
    'pointgpt':            'PointGPT',
    'pcpmae':              'PCP',
    'pointm2ae':           'Point-M2AE',
    'pointbert':           'PointBERT',
    'pointmamba':          'PointMamba',
}


def canonicalize(label: str) -> tuple[str, str]:
    """Map a raw label to (canonical_base, strategy).

    Accepts both lowercase folder labels ('pointmae_dapt') and canonical
    names ('Point-MAE:DAPT'). Returns strategy='FF' for bare base models
    (Full Finetuning is the default for SSL when no strategy is given).
    """
    if not label:
        return label, 'FF'
    # canonical form "Base:Strategy"
    if ':' in label:
        base, strat = label.split(':', 1)
        return base.strip(), strat.strip() or 'FF'
    low = label.lower()
    # composite "<base>_<strategy>" (part-seg folder style)
    parts = low.split('_')
    if len(parts) >= 2 and parts[-1] in {'ppt', 'dapt', 'idpt', 'gst', 'ff'}:
        base_part = '_'.join(parts[:-1])
        strat = parts[-1].upper()
        return _LOWERCASE_ALIASES.get(base_part, label), strat
    return _LOWERCASE_ALIASES.get(low, label), 'FF'


def get_category(name: str) -> str:
    """Return the category of a canonical model name; fall back to 'Other'."""
    if name in MODEL_CATEGORIES:
        return MODEL_CATEGORIES[name]
    key = _NORMALIZED.get(_normalize(name))
    # _NORMALIZED is keyed on MODEL_CITATIONS; rebuild categories lookup once
    for canonical, category in MODEL_CATEGORIES.items():
        if _normalize(canonical) == _normalize(name):
            return category
    return 'Other'


def _model_sort_key(name: str) -> int:
    try:
        return MODEL_ORDER.index(name)
    except ValueError:
        return len(MODEL_ORDER) + hash(name) % 1000


def _strategy_sort_key(strat: str) -> int:
    try:
        return STRATEGY_ORDER.index(strat)
    except ValueError:
        return len(STRATEGY_ORDER)


def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    out = []
    for ch in s:
        if ch in ('_', '&', '%', '#', '$'):
            out.append('\\' + ch)
        else:
            out.append(ch)
    return ''.join(out)


def render_grouped_table(rows: list[dict],
                         metric_cols: list[str],
                         metric_headers: list[str],
                         *,
                         caption: str,
                         label: str,
                         cite_in_tables: bool = False,
                         lower_is_better: list[str] | None = None,
                         extra_header_row: str | None = None,
                         metric_width: str = '1.4cm') -> str:
    """Render a classification-style grouped table.

    Each `rows` entry must provide:
        - `base` : canonical model name (used for category + citation)
        - `strategy` : e.g. 'FF', 'DAPT', 'IDPT', 'PPT', 'GST', or '-'
        - one key per metric_col (already formatted as str, or a number)

    The table is wrapped in \\begin{table}[htbp] ... \\end{table}
    with scriptsize, adjustbox, booktabs, multirow.
    """
    if not rows:
        return ''
    lower_is_better = set(lower_is_better or [])

    # Sort by (category, base, strategy) using the canonical orders
    def row_key(r):
        base = r['base']
        return (
            CATEGORY_ORDER.index(get_category(base))
            if get_category(base) in CATEGORY_ORDER
            else len(CATEGORY_ORDER),
            _model_sort_key(base),
            _strategy_sort_key(r.get('strategy', 'FF')),
        )
    rows = sorted(rows, key=row_key)

    # Pre-compute counts for multirow
    cat_counts: dict[str, int] = {}
    base_counts: dict[tuple[str, str], int] = {}
    for r in rows:
        c = get_category(r['base'])
        cat_counts[c] = cat_counts.get(c, 0) + 1
        k = (c, r['base'])
        base_counts[k] = base_counts.get(k, 0) + 1

    # Best values per metric
    def to_float(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            import re as _re
            m = _re.match(r'\s*\$?([\d.]+)', v)
            return float(m.group(1)) if m else None
        return None

    best_values: dict[str, float] = {}
    for m in metric_cols:
        vals = [to_float(r.get(m)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        best_values[m] = min(vals) if m in lower_is_better else max(vals)

    # Column format
    col_fmt = 'p{1.9cm}p{2.4cm}p{1.3cm}' + f"p{{{metric_width}}}" * len(metric_cols)
    n_metric = len(metric_cols)

    out = []
    out.append(r'\begin{table}[htbp]')
    out.append(r'\centering')
    out.append(r'\scriptsize')
    out.append(rf'\caption{{{caption}}}')
    out.append(rf'\label{{{label}}}')
    out.append(r'\begin{adjustbox}{center}')
    out.append(rf'\begin{{tabular}}{{{col_fmt}}}')
    out.append(r'\toprule')
    if extra_header_row:
        out.append(extra_header_row)
    headers = ['Category', 'Model', 'Strategy'] + metric_headers
    out.append(' & '.join(headers) + r' \\')
    out.append(r'\midrule')

    current_cat = None
    current_base = None
    for r in rows:
        base = r['base']
        cat = get_category(base)
        strategy = r.get('strategy', 'FF') or 'FF'

        # Category separator
        if cat != current_cat:
            if current_cat is not None:
                out.append(r'\midrule')
            current_cat = cat
            current_base = None
            cat_total = cat_counts.get(cat, 1)
            cat_cell = rf'\multirow{{{cat_total}}}{{*}}{{{cat}}}'
        else:
            cat_cell = ''

        # SSL grouping: multirow for the base when multiple strategies
        base_total = base_counts.get((cat, base), 1)
        if base != current_base:
            # Add a subtle rule between SSL base models with multi strategies
            if current_base is not None and cat == 'Self-supervised':
                prev_total = base_counts.get((cat, current_base), 1)
                if prev_total >= 2:
                    out.append(rf'\cmidrule(lr){{2-{3 + n_metric}}}')
            current_base = base
            disp = _latex_escape(base)
            if cite_in_tables:
                key = get_cite_key(base)
                if key:
                    disp = f'{disp}~\\citep{{{key}}}'
            if base_total > 1:
                model_cell = rf'\multirow{{{base_total}}}{{*}}{{{disp}}}'
            else:
                model_cell = disp
        else:
            model_cell = ''

        # Strategy cell (+ cite when PEFT citations are requested)
        strat_disp = strategy if strategy not in ('-', 'FF') else strategy
        strat_disp = _latex_escape(strat_disp)
        if cite_in_tables and strategy not in ('-', 'FF', 'Full', 'Full Finetuning'):
            k = get_cite_key(strategy)
            if k:
                strat_disp = f'{strat_disp}~\\citep{{{k}}}'

        # Metric cells with best-value bold
        metric_cells = []
        for m in metric_cols:
            v = r.get(m)
            if v is None:
                metric_cells.append('--')
                continue
            if isinstance(v, (int, float)):
                txt = f'{v:.2f}'
                is_best = m in best_values and abs(v - best_values[m]) < 1e-9
                metric_cells.append(f'\\textbf{{{txt}}}' if is_best else txt)
            else:
                f = to_float(v)
                if f is None:
                    metric_cells.append(str(v))
                    continue
                is_best = m in best_values and abs(f - best_values[m]) < 1e-9
                if is_best:
                    # Bold both mean and std in "a\pm b" forms, else the whole string
                    import re as _re
                    mb = _re.match(r'([\d.]+)\$\\pm\$([\d.]+)(.*)', str(v))
                    if mb:
                        mean_s, std_s, tail = mb.group(1), mb.group(2), mb.group(3)
                        metric_cells.append(
                            f'\\textbf{{{mean_s}}}$\\pm$\\textbf{{{std_s}}}{tail}')
                    else:
                        metric_cells.append(f'\\textbf{{{v}}}')
                else:
                    metric_cells.append(str(v))

        row_cells = [cat_cell, model_cell, strat_disp] + metric_cells
        out.append(' & '.join(row_cells) + r' \\')

    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{adjustbox}')
    out.append(r'\end{table}')
    return '\n'.join(out)


def build_citation_paragraph(model_names: Iterable[str],
                             heading: str = "Methods") -> str:
    """Build a paragraph that cites every given model once.

    Returns an empty string if no citable names were given.
    """
    seen: dict[str, str] = {}
    for name in model_names:
        key = get_cite_key(name)
        if key and key not in seen.values():
            seen[name] = key
    if not seen:
        return ""
    parts = [f"{n.replace('_', chr(92) + '_')}~\\citep{{{k}}}"
             for n, k in seen.items()]
    if len(parts) == 1:
        joined = parts[0]
    elif len(parts) == 2:
        joined = f"{parts[0]} and {parts[1]}"
    else:
        joined = ', '.join(parts[:-1]) + ', and ' + parts[-1]
    return (
        f"\\paragraph{{{heading}.}} This comparison includes the following "
        f"methods: {joined}."
    )
