"""
Segmentation Metrics Tracker.

Tracks per-point accuracy and mean Intersection-over-Union (mIoU) for
point cloud segmentation. Computes two mIoU variants standard for
ShapeNet Parts evaluation:

- Instance-average mIoU: per-shape IoU averaged over all shapes.
- Class-average mIoU: per-shape IoU averaged within each category,
  then averaged across categories.

The best model is selected by instance-average mIoU (standard convention).
"""

import csv
import os
import numpy as np

from utils.logger import print_log


class SegMetricsTracker:
    """Tracks segmentation metrics across epochs."""

    def __init__(
        self,
        num_seg_classes: int,
        seg_classes_map: dict = None,
        category_names: list = None,
    ):
        """
        Args:
            num_seg_classes: total number of part/semantic classes (50 for ShapeNet Parts).
            seg_classes_map: dict mapping category name → list of part label ids,
                e.g. {'Airplane': [0,1,2,3], 'Chair': [12,13,14,15], ...}
                Required for ShapeNet Parts-style mIoU. If None, only semantic
                mIoU (single category, all classes) is computed.
            category_names: optional list of category names for logging.
        """
        self.num_seg_classes = num_seg_classes
        self.seg_classes_map = seg_classes_map
        self.category_names = category_names or (
            list(seg_classes_map.keys()) if seg_classes_map else ['all']
        )

        # Build reverse map: part_label → category name
        self.seg_label_to_cat = {}
        if seg_classes_map:
            for cat, labels in seg_classes_map.items():
                for lab in labels:
                    self.seg_label_to_cat[lab] = cat

        # Best metrics
        self.best_instance_miou = 0.0
        self.best_class_miou = 0.0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.per_category_iou_at_best = {}

        # History
        self.history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_acc': [], 'val_class_miou': [], 'val_instance_miou': [],
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(self, pred_logits: np.ndarray, target: np.ndarray,
                       shape_categories: np.ndarray, accumulator: dict):
        """Accumulate per-shape IoUs for one batch.

        Args:
            pred_logits: [B, N, num_seg_classes] numpy array (or B,C,N transposed)
            target:      [B, N] int labels
            shape_categories: [B] int indicating each shape's object category
                              (only used to restrict valid parts for part seg).
            accumulator: dict with keys 'total_correct', 'total_seen',
                         'shape_ious' (list of per-shape ious by category).
        """
        B, N = target.shape
        # Accept [B, N, C] or [B, C, N]
        if pred_logits.shape[1] == N and pred_logits.shape[2] == self.num_seg_classes:
            # [B, N, C] already
            logits_bnc = pred_logits
        else:
            logits_bnc = pred_logits.transpose(0, 2, 1)  # [B, C, N] -> [B, N, C]

        cur_pred = np.zeros((B, N), dtype=np.int32)
        for i in range(B):
            if self.seg_classes_map is not None:
                # Part seg: restrict argmax to the valid parts of this shape's category.
                cat_idx = shape_categories[i]
                # Resolve category name: either directly from category_names[cat_idx],
                # or via target's first label through seg_label_to_cat.
                if 0 <= cat_idx < len(self.category_names):
                    cat_name = self.category_names[cat_idx]
                else:
                    cat_name = self.seg_label_to_cat.get(int(target[i, 0]), None)
                if cat_name is not None and cat_name in self.seg_classes_map:
                    valid = np.array(self.seg_classes_map[cat_name])
                    local_pred = np.argmax(logits_bnc[i][:, valid], axis=1)
                    cur_pred[i, :] = valid[local_pred]   # correct for non-contiguous labels
                    continue
            # Fallback: full argmax (semantic seg or unknown category)
            cur_pred[i, :] = np.argmax(logits_bnc[i], axis=1)

        accumulator['total_correct'] += int(np.sum(cur_pred == target))
        accumulator['total_seen'] += int(B * N)

        # Per-shape IoU
        for i in range(B):
            segp = cur_pred[i]
            segl = target[i]
            if self.seg_classes_map is not None:
                cat_name = self.seg_label_to_cat.get(int(segl[0]), None)
                if cat_name is None:
                    continue
                parts = self.seg_classes_map[cat_name]
                part_ious = []
                for l in parts:
                    inter = np.sum((segl == l) & (segp == l))
                    union = np.sum((segl == l) | (segp == l))
                    if union == 0:
                        part_ious.append(1.0)
                    else:
                        part_ious.append(inter / float(union))
                accumulator['shape_ious'].setdefault(cat_name, []).append(
                    float(np.mean(part_ious))
                )
            else:
                # Semantic seg: accumulate global TP/FP/FN per class.
                # This matches the PointNet/PointNet++ S3DIS evaluation protocol
                # and avoids the per-block averaging bias.
                for l in range(self.num_seg_classes):
                    pred_l = segp == l
                    true_l = segl == l
                    accumulator['tp'][l] += int(np.sum(pred_l & true_l))
                    accumulator['fp'][l] += int(np.sum(pred_l & ~true_l))
                    accumulator['fn'][l] += int(np.sum(~pred_l & true_l))

    def new_accumulator(self):
        acc = {'total_correct': 0, 'total_seen': 0, 'shape_ious': {}}
        if self.seg_classes_map is None:
            # Semantic seg: global TP/FP/FN per class (correct S3DIS protocol).
            acc['tp'] = np.zeros(self.num_seg_classes, dtype=np.int64)
            acc['fp'] = np.zeros(self.num_seg_classes, dtype=np.int64)
            acc['fn'] = np.zeros(self.num_seg_classes, dtype=np.int64)
        return acc

    def finalize(self, accumulator: dict):
        """Compute mIoU from an accumulator.

        Part segmentation (seg_classes_map is not None):
            - instance_miou: mean over all shapes of per-shape mean-IoU.
            - class_miou: mean over categories of per-category mean-IoU.

            Both follow the standard ShapeNet Parts convention.

        Semantic segmentation (seg_classes_map is None):
            - Uses GLOBAL TP/FP/FN accumulated across ALL points (PointNet++
              S3DIS protocol).  Per-block averaging inflates mIoU by giving
              equal weight to small and large blocks, which can shift results
              by 1–4 pp versus published baselines.
            - instance_miou and class_miou are both the global mIoU
              (they are identical in the single-category semantic case).
        """
        accuracy = (
            accumulator['total_correct'] / float(accumulator['total_seen'])
            if accumulator['total_seen'] > 0 else 0.0
        )

        if self.seg_classes_map is None:
            # ── Semantic seg: global TP/FP/FN ──────────────────────────────
            tp = accumulator['tp']
            fp = accumulator['fp']
            fn = accumulator['fn']
            denom = tp + fp + fn
            iou_per_class = np.where(denom > 0, tp / denom.astype(np.float64), np.nan)
            # Only count classes that appear in the evaluation set
            valid = ~np.isnan(iou_per_class)
            miou = float(np.mean(iou_per_class[valid])) if valid.any() else 0.0
            per_cat = {
                str(c): float(iou_per_class[c])
                for c in range(self.num_seg_classes)
                if not np.isnan(iou_per_class[c])
            }
            return {
                'accuracy': accuracy * 100.0,
                'instance_miou': miou * 100.0,
                'class_miou': miou * 100.0,
                'per_category_iou': {k: v * 100.0 for k, v in per_cat.items()},
            }
        else:
            # ── Part seg: per-shape IoU average ────────────────────────────
            all_ious = []
            per_cat = {}
            for cat, ious in accumulator['shape_ious'].items():
                all_ious.extend(ious)
                per_cat[cat] = float(np.mean(ious)) if ious else 0.0
            instance_miou = float(np.mean(all_ious)) if all_ious else 0.0
            class_miou = float(np.mean(list(per_cat.values()))) if per_cat else 0.0
            return {
                'accuracy': accuracy * 100.0,
                'instance_miou': instance_miou * 100.0,
                'class_miou': class_miou * 100.0,
                'per_category_iou': {k: v * 100.0 for k, v in per_cat.items()},
            }

    # ------------------------------------------------------------------
    # Update best
    # ------------------------------------------------------------------

    def update(self, epoch: int, metrics: dict, selection_metric: str = 'instance_miou') -> bool:
        """Update best-metrics tracking. Selection criterion is configurable."""
        current = metrics.get(selection_metric, metrics['instance_miou'])
        best_so_far = getattr(self, f'_best_{selection_metric}', 0.0)
        is_best = current > best_so_far
        if is_best:
            setattr(self, f'_best_{selection_metric}', current)
        if is_best:
            self.best_instance_miou = metrics['instance_miou']
            self.best_class_miou = metrics['class_miou']
            self.best_accuracy = metrics['accuracy']
            self.best_epoch = epoch
            self.per_category_iou_at_best = dict(metrics['per_category_iou'])
        return is_best

    def update_history(self, epoch, train_loss=0.0, train_acc=0.0,
                       val_acc=0.0, val_class_miou=0.0, val_instance_miou=0.0):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['val_class_miou'].append(val_class_miou)
        self.history['val_instance_miou'].append(val_instance_miou)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def print_best(self, logger=None):
        print_log(
            f"[Best Seg Model] Epoch {self.best_epoch}: "
            f"Acc={self.best_accuracy:.4f}%, "
            f"Class mIoU={self.best_class_miou:.4f}%, "
            f"Instance mIoU={self.best_instance_miou:.4f}%",
            logger=logger,
        )
        if self.per_category_iou_at_best:
            print_log("  Per-category IoU:", logger=logger)
            for cat, iou in sorted(self.per_category_iou_at_best.items()):
                print_log(f"    {cat:<14}: {iou:.4f}%", logger=logger)

    def save_history_csv(self, save_path: str):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            writer.writerows(zip(*self.history.values()))

    def save_summary_csv(self, save_path: str, model_name: str, dataset: str = ''):
        row = {
            'model_name': model_name,
            'dataset': dataset,
            'best_epoch': self.best_epoch,
            'best_accuracy': f'{self.best_accuracy:.4f}',
            'best_class_miou': f'{self.best_class_miou:.4f}',
            'best_instance_miou': f'{self.best_instance_miou:.4f}',
        }
        for cat, iou in self.per_category_iou_at_best.items():
            row[f'iou_{cat}'] = f'{iou:.4f}'
        file_exists = os.path.exists(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
