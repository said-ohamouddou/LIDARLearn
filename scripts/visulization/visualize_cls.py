"""
Visualize classification predictions on point clouds.

Loads a trained classification model, runs inference on the validation set,
and saves per-sample interactive 3D visualizations colored by predicted vs
ground-truth class, plus a confusion matrix and summary.

Usage:
    python scripts/visulization/visualize_cls.py \
        --config experiments/HELIALS/ms_dgcnn2_norm_both_helials/config.yaml \
        --ckpt experiments/HELIALS/ms_dgcnn2_norm_both_helials/ckpt-best.pth \
        --num_vis 30

Outputs:
    <out_dir>/
        000_Pine_pred-Pine.html       — interactive 3D point cloud
        confusion_matrix.html         — interactive confusion matrix
        summary.txt                   — per-class accuracy + overall metrics
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    f1_score, cohen_kappa_score, matthews_corrcoef,
)
from utils.checkpoint import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ── Distinct colors for up to 20 classes ─────────────────────────────────────
CLASS_PALETTE = np.array([
    [255, 50, 50],   # 0  red
    [50, 200, 50],   # 1  green
    [50, 100, 255],   # 2  blue
    [255, 200, 0],   # 3  gold
    [220, 40, 200],   # 4  magenta
    [0, 210, 210],   # 5  cyan
    [255, 140, 30],   # 6  orange
    [140, 70, 230],   # 7  purple
    [50, 220, 130],   # 8  spring green
    [255, 80, 140],   # 9  hot pink
    [128, 128, 0],   # 10 olive
    [0, 128, 128],   # 11 teal
    [200, 130, 80],   # 12 tan
    [100, 100, 255],   # 13 periwinkle
    [200, 200, 50],   # 14 yellow-green
    [180, 50, 50],   # 15 dark red
    [50, 150, 200],   # 16 steel blue
    [230, 180, 255],   # 17 lavender
    [0, 160, 80],   # 18 forest green
    [255, 180, 180],   # 19 light pink
], dtype=np.uint8)


def _display_names(config):
    m = config.model
    mname = getattr(m, 'base_model', None) or m.NAME
    bb = getattr(m, 'backbone', None)
    if bb is not None and getattr(bb, 'NAME', None):
        mname = f'{bb.NAME} (via {mname})'
    try:
        dname = config.dataset.val._base_.NAME
    except AttributeError:
        dname = getattr(getattr(config.dataset, 'train', None), '_base_', None)
        dname = getattr(dname, 'NAME', '?') if dname is not None else '?'
    return mname, dname


def make_cls_html(xyz, label_color, title="", point_size=3, model_name="", dataset_name=""):
    """Create interactive 3D HTML for a single point cloud colored by class."""
    def _rgb(c):
        return [f'rgb({r},{g},{b})' for r, g, b in c]

    subtitle = f'Model: {model_name} &nbsp;·&nbsp; Dataset: {dataset_name}' if (model_name or dataset_name) else ''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{{margin:0;font-family:sans-serif;background:#1a1a2e;display:flex;flex-direction:column;align-items:center}}
h3{{color:#eee;margin:10px 10px 2px 10px}}
.subtitle{{color:#aaa;font-size:13px;margin:0 10px 8px 10px}}
#plot{{width:90vw;height:85vh}}
</style></head>
<body><h3>{title}</h3>
<div class="subtitle">{subtitle}</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot',[{{
  x:{xyz[:,0].tolist()}, y:{xyz[:,1].tolist()}, z:{xyz[:,2].tolist()},
  mode:'markers', type:'scatter3d',
  marker:{{size:{point_size}, color:{_rgb(label_color)}}},
  hoverinfo:'skip'
}}],{{
  scene:{{aspectmode:'data', bgcolor:'#0f0f23'}},
  margin:{{l:0,r:0,t:10,b:0}},
  paper_bgcolor:'#1a1a2e'
}});
</script></body></html>"""
    return html


def make_confusion_html(cm, class_names, title="Confusion Matrix", model_name="", dataset_name=""):
    """Create interactive heatmap HTML for confusion matrix."""
    n = len(class_names)
    # Normalize per row (true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm_norm / row_sums * 100

    # Build text annotations: "count\n(pct%)"
    text = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(f'{cm[i,j]}<br>({cm_pct[i,j]:.0f}%)')
        text.append(row)

    subtitle = f'Model: {model_name} &nbsp;·&nbsp; Dataset: {dataset_name}' if (model_name or dataset_name) else ''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{{margin:0;background:#1a1a2e;color:#eee;display:flex;flex-direction:column;align-items:center}}
.subtitle{{color:#aaa;font-size:13px;margin:6px 10px}}
#plot{{width:80vw;height:80vh}}</style></head>
<body><div class="subtitle">{subtitle}</div><div id="plot"></div>
<script>
Plotly.newPlot('plot',[{{
  z:{cm_pct.tolist()},
  x:{list(class_names)},
  y:{list(class_names)},
  type:'heatmap',
  colorscale:'YlOrRd',
  text:{text},
  texttemplate:'%{{text}}',
  hoverinfo:'skip'
}}],{{
  title:{{text:'{title}',font:{{color:'#eee',size:18}}}},
  xaxis:{{title:'Predicted',color:'#eee',tickfont:{{size:11}}}},
  yaxis:{{title:'True',color:'#eee',autorange:'reversed',tickfont:{{size:11}}}},
  paper_bgcolor:'#1a1a2e',
  plot_bgcolor:'#1a1a2e',
  margin:{{l:100,r:30,t:60,b:80}}
}});
</script></body></html>"""
    return html


def make_gallery_html(samples, class_names, out_dir, title="Classification Gallery", model_name="", dataset_name=""):
    """Create an index HTML linking to all individual sample HTMLs."""
    rows = []
    for s in samples:
        color = 'lime' if s['correct'] else 'red'
        rows.append(
            f'<tr style="color:{color}">'
            f'<td>{s["idx"]}</td>'
            f'<td><a href="{s["filename"]}" style="color:{color}">{s["gt_name"]}</a></td>'
            f'<td>{s["pred_name"]}</td>'
            f'<td>{"correct" if s["correct"] else "WRONG"}</td></tr>'
        )
    table = '\n'.join(rows)

    subtitle = f'Model: {model_name} &nbsp;·&nbsp; Dataset: {dataset_name}' if (model_name or dataset_name) else ''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:sans-serif;background:#1a1a2e;color:#eee;padding:20px}}
.subtitle{{color:#aaa;font-size:13px;margin:-10px 0 15px 0}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #333;padding:6px 12px;text-align:left}}
th{{background:#2a2a4e}} a{{text-decoration:none}}
</style></head>
<body>
<h2>{title}</h2>
<div class="subtitle">{subtitle}</div>
<p><a href="confusion_matrix.html" style="color:cyan">View Confusion Matrix</a></p>
<table><tr><th>#</th><th>True</th><th>Predicted</th><th>Status</th></tr>
{table}
</table></body></html>"""

    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Visualize classification predictions")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_vis', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.ckpt), 'vis_cls')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────
    from utils.config import cfg_from_yaml_file
    config = cfg_from_yaml_file(args.config)
    model_name, dataset_name = _display_names(config)

    # ── Dataset ──────────────────────────────────────────────────────────
    from tools.builder import dataset_builder
    import datasets

    split_cfg = config.dataset.get(args.split, config.dataset.val)
    _, data_loader = dataset_builder(
        argparse.Namespace(distributed=False, num_workers=4, local_rank=0),
        split_cfg,
    )
    ds = data_loader.dataset
    class_names = ds.classes if hasattr(ds, 'classes') else [str(i) for i in range(config.model.get('num_classes', 10))]
    num_classes = len(class_names)
    print(f"Dataset: {len(ds)} samples, {num_classes} classes: {class_names}")

    # ── Model ────────────────────────────────────────────────────────────
    import models
    from models.build import build_model_from_cfg

    model = build_model_from_cfg(config.model)
    ckpt = load_checkpoint(args.ckpt, map_location='cpu')
    sd = ckpt.get('base_model', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    print(f"Loaded: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    # ── Inference ────────────────────────────────────────────────────────
    all_gt, all_pred = [], []
    vis_samples = []
    count = 0

    with torch.no_grad():
        for taxonomy_ids, model_ids, data in data_loader:
            points = data[0].float().cuda()
            labels = data[1].long().cuda()

            logits = model(points)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)

            all_gt.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

            B = points.shape[0]
            for i in range(B):
                if count >= args.num_vis:
                    continue

                gt_idx = labels[i].item()
                pred_idx = preds[i].item()
                gt_name = class_names[gt_idx] if gt_idx < len(class_names) else str(gt_idx)
                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                correct = gt_idx == pred_idx

                pts_np = points[i].cpu().numpy()[:, :3]

                # Color by GT class
                color = np.tile(CLASS_PALETTE[gt_idx % len(CLASS_PALETTE)], (pts_np.shape[0], 1))

                # Filename
                status = 'ok' if correct else 'WRONG'
                fname = f'{count:03d}_{gt_name}_pred-{pred_name}_{status}'
                fname_safe = fname.replace(' ', '_').replace('/', '_')

                # Save HTML
                border = '' if correct else ' | MISCLASSIFIED'
                title = f'GT: {gt_name} | Pred: {pred_name}{border}'
                html = make_cls_html(pts_np, color, title=title,
                                     model_name=model_name, dataset_name=dataset_name)
                html_path = os.path.join(args.out_dir, f'{fname_safe}.html')
                with open(html_path, 'w') as f:
                    f.write(html)

                # Save NPY
                npy_data = np.column_stack([pts_np, np.full(pts_np.shape[0], gt_idx),
                                            np.full(pts_np.shape[0], pred_idx)]).astype(np.float32)
                np.save(os.path.join(args.out_dir, f'{fname_safe}.npy'), npy_data)

                vis_samples.append({
                    'idx': count, 'gt_name': gt_name, 'pred_name': pred_name,
                    'correct': correct, 'filename': f'{fname_safe}.html',
                })
                count += 1

    # ── Metrics ──────────────────────────────────────────────────────────
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    acc = accuracy_score(all_gt, all_pred) * 100
    bal_acc = balanced_accuracy_score(all_gt, all_pred) * 100
    f1 = f1_score(all_gt, all_pred, average='macro', zero_division=0) * 100
    kappa = cohen_kappa_score(all_gt, all_pred) * 100
    mcc = matthews_corrcoef(all_gt, all_pred) * 100
    cm = confusion_matrix(all_gt, all_pred, labels=list(range(num_classes)))

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_gt == c
        if mask.sum() > 0:
            per_class_acc[class_names[c]] = (all_pred[mask] == c).mean() * 100

    # ── Confusion matrix HTML ────────────────────────────────────────────
    cm_html = make_confusion_html(cm, class_names,
                                   model_name=model_name, dataset_name=dataset_name)
    with open(os.path.join(args.out_dir, 'confusion_matrix.html'), 'w') as f:
        f.write(cm_html)

    # ── Gallery index ────────────────────────────────────────────────────
    make_gallery_html(vis_samples, class_names, args.out_dir,
                      title=f'Classification: {model_name}',
                      model_name=model_name, dataset_name=dataset_name)

    # ── Summary ──────────────────────────────────────────────────────────
    lines = []
    lines.append(f"Model         : {config.model.NAME}")
    lines.append(f"Checkpoint    : {args.ckpt}")
    lines.append(f"Split         : {args.split}")
    lines.append(f"Samples       : {len(all_gt)}")
    lines.append(f"")
    lines.append(f"Accuracy      : {acc:.2f}%")
    lines.append(f"Balanced Acc  : {bal_acc:.2f}%")
    lines.append(f"F1 Macro      : {f1:.2f}%")
    lines.append(f"Cohen Kappa   : {kappa:.2f}%")
    lines.append(f"MCC           : {mcc:.2f}%")
    lines.append(f"")
    lines.append(f"Per-class accuracy:")
    for name, a in sorted(per_class_acc.items()):
        n = int((all_gt == class_names.index(name)).sum())
        lines.append(f"  {name:<14}: {a:6.2f}%  (n={n})")

    summary = '\n'.join(lines)
    print(f"\n{summary}")

    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')

    n_correct = sum(1 for s in vis_samples if s['correct'])
    n_wrong = sum(1 for s in vis_samples if not s['correct'])
    print(f"\nSaved to {args.out_dir}/")
    print(f"  {count} .html point clouds ({n_correct} correct, {n_wrong} wrong)")
    print(f"  confusion_matrix.html")
    print(f"  index.html (gallery)")
    print(f"  summary.txt")


if __name__ == '__main__':
    main()
