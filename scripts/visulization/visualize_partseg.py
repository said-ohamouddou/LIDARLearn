"""
Visualize ShapeNet Parts segmentation predictions.

Loads a trained part-segmentation model, runs inference on the test set,
and saves per-shape interactive 3D visualizations (GT vs Pred).

Usage:
    python scripts/visulization/visualize_partseg.py \
        --config cfgs/segmentation/PointNet/ShapeNetParts/pointnet_partseg.yaml \
        --ckpt experiments/ShapeNetParts/pointnet_partseg/ckpt-last-seg.pth \
        --num_vis 30

Outputs:
    <out_dir>/
        000_Airplane_acc95.3.html   — interactive 3D (GT left, Pred right)
        000_Airplane_acc95.3.npy    — [N, 5] = x y z gt pred
        summary.txt                 — per-category mIoU + instance mIoU
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.ShapeNet55DatasetPretrain import SHAPENET_SEG_CLASSES, SHAPENET_SEG_LABEL_TO_CAT
from utils.checkpoint import load_checkpoint

# ── 16 category colors (one per object type) ────────────────────────────────
CATEGORY_COLORS = {
    'Airplane': (70, 130, 180),
    'Bag': (220, 20, 60),
    'Cap': (255, 165, 0),
    'Car': (0, 128, 0),
    'Chair': (138, 43, 226),
    'Earphone': (255, 215, 0),
    'Guitar': (0, 206, 209),
    'Knife': (178, 34, 34),
    'Lamp': (60, 179, 113),
    'Laptop': (100, 149, 237),
    'Motorbike': (255, 99, 71),
    'Mug': (147, 112, 219),
    'Pistol': (244, 164, 96),
    'Rocket': (72, 61, 139),
    'Skateboard': (32, 178, 170),
    'Table': (210, 105, 30),
}

# ── 50 part-label colors — maximally distinct, hot/vivid ─────────────────────
# Each part within a category gets a completely different hue so parts
# are immediately distinguishable in the 3D visualization.
PART_PALETTE = np.array([
    # Airplane 0-3: red, green, blue, yellow
    [255, 50, 50], [50, 220, 50], [50, 100, 255], [255, 220, 30],
    # Bag 4-5: magenta, cyan
    [255, 0, 200], [0, 230, 230],
    # Cap 6-7: orange, lime
    [255, 140, 0], [180, 255, 30],
    # Car 8-11: red, teal, gold, purple
    [230, 30, 30], [0, 190, 180], [255, 200, 0], [160, 50, 240],
    # Chair 12-15: hot pink, sky blue, orange, green
    [255, 80, 150], [60, 180, 255], [255, 160, 40], [40, 200, 80],
    # Earphone 16-18: yellow, violet, spring green
    [255, 255, 0], [180, 60, 255], [0, 255, 140],
    # Guitar 19-21: coral, aqua, lime
    [255, 100, 80], [0, 220, 210], [190, 255, 50],
    # Knife 22-23: red, blue
    [240, 40, 40], [40, 120, 255],
    # Lamp 24-27: gold, magenta, cyan, chartreuse
    [255, 190, 0], [230, 30, 180], [0, 210, 255], [170, 240, 30],
    # Laptop 28-29: blue, orange
    [50, 80, 255], [255, 150, 30],
    # Motorbike 30-35: red, green, blue, yellow, magenta, cyan
    [240, 40, 40], [40, 200, 40], [40, 80, 240],
    [240, 220, 30], [220, 30, 180], [0, 200, 200],
    # Mug 36-37: hot pink, teal
    [255, 50, 140], [0, 200, 170],
    # Pistol 38-40: orange, purple, lime
    [255, 130, 0], [140, 40, 220], [160, 255, 40],
    # Rocket 41-43: red, cyan, gold
    [230, 50, 50], [30, 210, 220], [255, 200, 30],
    # Skateboard 44-46: magenta, green, blue
    [230, 40, 200], [50, 220, 80], [60, 100, 255],
    # Table 47-49: coral, teal, purple
    [255, 100, 70], [0, 190, 170], [170, 60, 230],
], dtype=np.uint8)


def _to_one_hot(y, num_classes):
    return torch.eye(num_classes, device=y.device)[y.long()]


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


def make_partseg_html(xyz, gt, pred, seg_classes_map, category, title="", model_name="", dataset_name=""):
    """Create interactive HTML with GT vs Pred colored by part labels."""
    # Get valid parts for this category
    valid_parts = seg_classes_map.get(category, list(range(50)))

    # Assign colors per part label
    gt_colors = PART_PALETTE[gt.astype(int) % 50]
    pred_colors = PART_PALETTE[pred.astype(int) % 50]

    # Build part name labels
    part_names_gt = [f'{category}_part{int(g)-valid_parts[0]}' for g in gt]
    part_names_pred = [f'{category}_part{int(p)-valid_parts[0]}' for p in pred]

    def _rgb(c):
        return [f'rgb({r},{g},{b})' for r, g, b in c]

    subtitle = f'Model: {model_name} &nbsp;·&nbsp; Dataset: {dataset_name}' if (model_name or dataset_name) else ''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{{margin:0;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;background:#1a1a2e}}
h3{{color:#eee;margin:10px 10px 2px 10px}}
.subtitle{{color:#aaa;font-size:13px;margin:0 10px 8px 10px}}
.row{{display:flex;gap:10px}}
.panel{{width:48vw;height:80vh}}
</style></head>
<body><h3>{title}</h3>
<div class="subtitle">{subtitle}</div>
<div class="row"><div id="gt" class="panel"></div><div id="pred" class="panel"></div></div>
<script>
var x={xyz[:,0].tolist()}, y={xyz[:,1].tolist()}, z={xyz[:,2].tolist()};
var gt_c={_rgb(gt_colors)}, pred_c={_rgb(pred_colors)};
var gt_t={part_names_gt}, pred_t={part_names_pred};
var layout={{scene:{{aspectmode:'data',bgcolor:'#0f0f23'}},
  margin:{{l:0,r:0,t:40,b:0}},paper_bgcolor:'#1a1a2e',font:{{color:'#eee'}}}};
var mk={{size:3}};
Plotly.newPlot('gt',[{{x:x,y:y,z:z,mode:'markers',type:'scatter3d',
  marker:Object.assign({{color:gt_c}},mk),text:gt_t,hoverinfo:'text'}}],
  Object.assign({{title:'Ground Truth'}},layout));
Plotly.newPlot('pred',[{{x:x,y:y,z:z,mode:'markers',type:'scatter3d',
  marker:Object.assign({{color:pred_c}},mk),text:pred_t,hoverinfo:'text'}}],
  Object.assign({{title:'Prediction'}},layout));
</script></body></html>"""
    return html


def compute_shape_iou(gt, pred, valid_parts):
    """Compute IoU for a single shape, restricted to its category's valid parts."""
    ious = []
    for part in valid_parts:
        inter = np.sum((gt == part) & (pred == part))
        union = np.sum((gt == part) | (pred == part))
        if union == 0:
            ious.append(1.0)  # part not present → perfect IoU
        else:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser(description="Visualize ShapeNet Parts segmentation")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num_vis', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.ckpt), 'vis_partseg')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────
    from utils.config import cfg_from_yaml_file
    config = cfg_from_yaml_file(args.config)
    model_name, dataset_name = _display_names(config)

    seg_classes = int(config.model.get('seg_classes', 50))
    num_obj_classes = int(config.model.get('num_obj_classes', 16))
    use_cls_label = config.model.get('use_cls_label', True)

    # ── Dataset ──────────────────────────────────────────────────────────
    from tools.builder import dataset_builder
    import datasets

    val_cfg = config.dataset.val
    _, val_loader = dataset_builder(
        argparse.Namespace(distributed=False, num_workers=4, local_rank=0),
        val_cfg,
    )
    ds = val_loader.dataset
    seg_classes_map = getattr(ds, 'seg_classes', SHAPENET_SEG_CLASSES)
    category_names = getattr(ds, 'category_names', sorted(seg_classes_map.keys()))
    print(f"Val set: {len(ds)} samples, {len(category_names)} categories, {seg_classes} part classes")

    # ── Model ────────────────────────────────────────────────────────────
    import models
    from models.build import build_model_from_cfg

    model = build_model_from_cfg(config.model)
    ckpt = load_checkpoint(args.ckpt, map_location='cpu')
    sd = ckpt.get('base_model', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    print(f"Loaded: {args.ckpt}")

    # ── Inference ────────────────────────────────────────────────────────
    per_cat_ious = {cat: [] for cat in category_names}
    all_shape_ious = []
    count = 0

    with torch.no_grad():
        for tax, cat_name, data in val_loader:
            if count >= args.num_vis:
                break

            points, cls_label, target = data
            points = points.float().cuda()
            target = target.long().cuda()
            cls_label = cls_label.long().cuda().view(-1)

            pts_bcn = points.transpose(1, 2).contiguous()
            cls_onehot = _to_one_hot(cls_label, num_obj_classes)
            logits = model(pts_bcn, cls_onehot)  # [B, 50, N]

            B = points.shape[0]
            for i in range(B):
                if count >= args.num_vis:
                    break

                # Get category for this shape
                if isinstance(cat_name, (list, tuple)):
                    cat = cat_name[i]
                else:
                    cat = str(cat_name)

                gt_np = target[i].cpu().numpy()
                logits_np = logits[i].cpu().numpy()  # [50, N]
                pts_np = points[i].cpu().numpy()[:, :3]

                # Restrict prediction to valid parts for this category
                valid_parts = seg_classes_map.get(cat, list(range(50)))
                pred_np = np.argmax(logits_np[valid_parts, :], axis=0) + valid_parts[0]

                # Compute IoU
                shape_iou = compute_shape_iou(gt_np, pred_np, valid_parts)
                acc = (gt_np == pred_np).mean() * 100

                all_shape_ious.append(shape_iou)
                per_cat_ious.setdefault(cat, []).append(shape_iou)

                # Save .npy
                npy_data = np.column_stack([pts_np, gt_np, pred_np]).astype(np.float32)
                fname = f'{count:03d}_{cat}_acc{acc:.0f}'
                np.save(os.path.join(args.out_dir, f'{fname}.npy'), npy_data)

                # Save .html
                title = f'{cat} — Acc: {acc:.1f}% | IoU: {shape_iou*100:.1f}%'
                html = make_partseg_html(pts_np, gt_np, pred_np, seg_classes_map, cat, title,
                                          model_name=model_name, dataset_name=dataset_name)
                html_path = os.path.join(args.out_dir, f'{fname}.html')
                with open(html_path, 'w') as f:
                    f.write(html)

                count += 1
                print(f"  [{count:3d}/{args.num_vis}] {cat:<12} acc={acc:5.1f}%  IoU={shape_iou*100:5.1f}%")

    # ── Summary ──────────────────────────────────────────────────────────
    lines = []
    instance_miou = np.mean(all_shape_ious) * 100 if all_shape_ious else 0.0
    cat_mious = {}
    for cat in sorted(per_cat_ious.keys()):
        ious = per_cat_ious[cat]
        if ious:
            cat_mious[cat] = np.mean(ious) * 100
    class_miou = np.mean(list(cat_mious.values())) if cat_mious else 0.0

    lines.append(f"Instance mIoU : {instance_miou:.2f}%")
    lines.append(f"Class mIoU    : {class_miou:.2f}%")
    lines.append(f"Num samples   : {count}")
    lines.append("")
    lines.append("Per-category mIoU:")
    for cat in sorted(cat_mious.keys()):
        n = len(per_cat_ious[cat])
        lines.append(f"  {cat:<14}: {cat_mious[cat]:6.2f}%  (n={n})")

    summary = '\n'.join(lines)
    print(f"\n{summary}")

    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')

    print(f"\nSaved to {args.out_dir}/")
    print(f"  {count} .html files (open in browser)")
    print(f"  {count} .npy files (x y z gt pred)")
    print(f"  summary.txt")


if __name__ == '__main__':
    main()
