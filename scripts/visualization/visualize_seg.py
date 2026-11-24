"""
Visualize semantic segmentation predictions.

Loads a trained segmentation model, runs inference on the validation set,
and saves per-block visualizations as HTML files (interactive 3D via plotly)
and per-block .npy arrays for downstream analysis.

Usage:
    python scripts/visualization/visualize_seg.py \
        --config cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis.yaml \
        --ckpt experiments/S3DIS/pointmae_s3dis/ckpt-best-seg.pth \
        --num_vis 20 \
        --out_dir experiments/S3DIS/pointmae_s3dis/vis

Outputs per sample:
    <out_dir>/
        sample_000_<room>.html   — interactive 3D scatter (GT vs Pred)
        sample_000_<room>.npy    — [N, 8] = x y z r g b gt_label pred_label
        summary.txt              — per-class IoU + overall mIoU
"""

import argparse
import os
import sys

import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.checkpoint import load_checkpoint


# ── S3DIS class names + colors ──────────────────────────────────────────────
S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter',
]

# 13 visually distinct colors (RGB 0–255)
S3DIS_COLORS = [
    (189, 198, 255),  # ceiling   — light blue
    (255, 198, 138),  # floor     — light orange
    (230, 230, 230),  # wall      — light gray
    (255, 128, 128),  # beam      — salmon
    (128, 0, 128),    # column    — purple
    (0, 204, 255),    # window    — cyan
    (204, 102, 0),    # door      — brown
    (0, 153, 0),      # table     — green
    (255, 255, 0),    # chair     — yellow
    (255, 0, 255),    # sofa      — magenta
    (0, 0, 204),      # bookcase  — dark blue
    (102, 204, 102),  # board     — light green
    (128, 128, 128),  # clutter   — gray
]


def label_to_color(labels, palette):
    """Map integer labels → RGB colors. Returns [N, 3] uint8."""
    palette = np.array(palette, dtype=np.uint8)
    return palette[labels.astype(int) % len(palette)]


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


def make_html(xyz, gt, pred, class_names, palette, title="", model_name="", dataset_name=""):
    """Create a self-contained HTML file with two side-by-side 3D scatter plots."""
    gt_colors = label_to_color(gt, palette)
    pred_colors = label_to_color(pred, palette)

    def _rgb_str(c):
        return [f'rgb({r},{g},{b})' for r, g, b in c]

    subtitle = f'Model: {model_name} &nbsp;·&nbsp; Dataset: {dataset_name}' if (model_name or dataset_name) else ''

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{{margin:0;display:flex;flex-direction:column;align-items:center}}
h3{{margin:10px 10px 2px 10px}}
.subtitle{{color:#666;font-size:13px;margin:0 10px 8px 10px}}
.row{{display:flex;gap:10px}} .panel{{width:48vw;height:80vh}}</style></head>
<body><h3>{title}</h3>
<div class="subtitle">{subtitle}</div>
<div class="row"><div id="gt" class="panel"></div><div id="pred" class="panel"></div></div>
<script>
var x={xyz[:,0].tolist()}, y={xyz[:,1].tolist()}, z={xyz[:,2].tolist()};
var gt_c={_rgb_str(gt_colors)}, pred_c={_rgb_str(pred_colors)};
var gt_text={[class_names[int(g)] for g in gt]};
var pred_text={[class_names[int(p)] for p in pred]};
var layout={{scene:{{aspectmode:'data'}},margin:{{l:0,r:0,t:30,b:0}}}};
var mk={{size:2}};
Plotly.newPlot('gt',[{{x:x,y:y,z:z,mode:'markers',type:'scatter3d',
  marker:Object.assign({{color:gt_c}},mk),text:gt_text,hoverinfo:'text'}}],
  Object.assign({{title:'Ground Truth'}},layout));
Plotly.newPlot('pred',[{{x:x,y:y,z:z,mode:'markers',type:'scatter3d',
  marker:Object.assign({{color:pred_c}},mk),text:pred_text,hoverinfo:'text'}}],
  Object.assign({{title:'Prediction'}},layout));
</script></body></html>"""
    return html


def compute_per_class_iou(gt, pred, num_classes):
    """Returns per-class IoU dict and overall mIoU."""
    ious = {}
    for c in range(num_classes):
        inter = np.sum((gt == c) & (pred == c))
        union = np.sum((gt == c) | (pred == c))
        if union > 0:
            ious[c] = inter / union
    miou = np.mean(list(ious.values())) if ious else 0.0
    return ious, miou


def _to_one_hot(y, num_classes):
    return torch.eye(num_classes, device=y.device)[y.long()]


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions")
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to ckpt-best-seg.pth')
    parser.add_argument('--num_vis', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: next to ckpt)')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.ckpt), 'vis')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────
    from utils.config import cfg_from_yaml_file
    config = cfg_from_yaml_file(args.config)
    model_name, dataset_name = _display_names(config)

    use_cls_label = config.model.get('use_cls_label', True)
    seg_classes = int(config.model.get('seg_classes', 50))
    num_obj_classes = int(config.model.get('num_obj_classes', 16))

    # Resolve class names from dataset or fallback
    class_names = S3DIS_CLASSES if seg_classes == 13 else [str(i) for i in range(seg_classes)]
    palette = S3DIS_COLORS if seg_classes == 13 else [
        tuple(np.random.RandomState(i).randint(0, 255, 3)) for i in range(seg_classes)
    ]

    # ── Load dataset (val split) ─────────────────────────────────────────
    from tools.builder import dataset_builder
    import datasets  # register datasets

    val_cfg = config.dataset.val
    _, val_loader = dataset_builder(
        argparse.Namespace(
            distributed=False, num_workers=4, local_rank=0
        ),
        val_cfg,
    )
    ds = val_loader.dataset
    if hasattr(ds, 'category_names') and ds.category_names:
        class_names = ds.category_names
    print(f"Val set: {len(ds)} samples, {seg_classes} classes")

    # ── Load model ───────────────────────────────────────────────────────
    import models  # register models
    from models.build import build_model_from_cfg

    model = build_model_from_cfg(config.model)

    ckpt = load_checkpoint(args.ckpt, map_location='cpu')
    sd = ckpt.get('base_model', ckpt)
    # Handle DataParallel key prefix
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── Inference + visualization ────────────────────────────────────────
    all_gt, all_pred = [], []
    count = 0

    with torch.no_grad():
        for tax, room_name, data in val_loader:
            if count >= args.num_vis:
                break

            # Unpack — handle both (pts, seg) and (pts, cls, seg)
            if len(data) == 3:
                points, cls_label, target = data
            else:
                points, target = data
                cls_label = None

            points = points.float().cuda()
            target = target.long().cuda()
            pts_bcn = points.transpose(1, 2).contiguous()

            if use_cls_label and cls_label is not None:
                cls_label = cls_label.long().cuda().view(-1)
                cls_onehot = _to_one_hot(cls_label, num_obj_classes)
                logits = model(pts_bcn, cls_onehot)
            else:
                logits = model(pts_bcn, None)

            pred = logits.argmax(dim=1)   # [B, N]

            # Process each sample in the batch
            B = points.shape[0]
            for i in range(B):
                if count >= args.num_vis:
                    break

                pts_np = points[i].cpu().numpy()             # [N, C]
                gt_np = target[i].cpu().numpy()              # [N]
                pred_np = pred[i].cpu().numpy()              # [N]
                xyz = pts_np[:, :3]

                # RGB for .npy (if available)
                if pts_np.shape[1] >= 6:
                    rgb = pts_np[:, 3:6]
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = np.zeros((pts_np.shape[0], 3), dtype=np.uint8)

                # Room name
                if isinstance(room_name, (list, tuple)):
                    rname = room_name[i] if i < len(room_name) else room_name[0]
                else:
                    rname = str(room_name)
                rname_safe = rname.replace('/', '_').replace('\\', '_')

                # Save .npy: [N, 8] = x y z r g b gt pred
                npy_data = np.column_stack([xyz, rgb, gt_np, pred_np]).astype(np.float32)
                npy_path = os.path.join(args.out_dir, f'sample_{count:03d}_{rname_safe}.npy')
                np.save(npy_path, npy_data)

                # Save .html
                title = f'{rname} — Acc: {(gt_np == pred_np).mean()*100:.1f}%'
                html = make_html(xyz, gt_np, pred_np, class_names, palette, title=title,
                                 model_name=model_name, dataset_name=dataset_name)
                html_path = os.path.join(args.out_dir, f'sample_{count:03d}_{rname_safe}.html')
                with open(html_path, 'w') as f:
                    f.write(html)

                all_gt.append(gt_np)
                all_pred.append(pred_np)
                count += 1
                print(f"  [{count}/{args.num_vis}] {rname}: "
                      f"acc={100*(gt_np==pred_np).mean():.1f}%  → {html_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    ious, miou = compute_per_class_iou(all_gt, all_pred, seg_classes)

    lines = []
    lines.append(f"Overall accuracy: {100*(all_gt == all_pred).mean():.2f}%")
    lines.append(f"Overall mIoU:     {100*miou:.2f}%")
    lines.append(f"Num samples:      {count}")
    lines.append("")
    lines.append("Per-class IoU:")
    for c, iou in sorted(ious.items()):
        name = class_names[c] if c < len(class_names) else str(c)
        lines.append(f"  {name:<14}: {100*iou:.2f}%")

    summary = '\n'.join(lines)
    print(f"\n{summary}")

    summary_path = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(f"\nSaved to {args.out_dir}/")
    print(f"  {count} .html files (open in browser for interactive 3D)")
    print(f"  {count} .npy files (columns: x y z r g b gt pred)")
    print(f"  summary.txt")


if __name__ == '__main__':
    main()
