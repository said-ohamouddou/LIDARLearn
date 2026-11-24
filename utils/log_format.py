"""
Unified log formatting for LIDARLearn.

All training output styling lives here — banners, epoch logs, summaries,
metrics tables. Keeps runner code clean and output consistent.
"""

from utils.logger import print_log


# ── Colors (ANSI) ──

BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RED = '\033[91m'


# ── Training Banner ──

def print_banner(args, config, logger=None):
    """Print the startup banner with training configuration."""
    mode = getattr(args, 'mode', 'finetune')
    task_map = {
        'finetune': 'Classification',
        'pretrain': 'Pretraining',
        'pretrain_recon': 'Pretraining (ReCon)',
        'seg': 'Segmentation',
    }
    task = task_map.get(mode, mode)

    model_cfg = config.model
    model_name = getattr(model_cfg, 'base_model', None) or model_cfg.NAME
    backbone_cfg = getattr(model_cfg, 'backbone', None)
    if backbone_cfg is not None and getattr(backbone_cfg, 'NAME', None):
        model_name = f'{backbone_cfg.NAME} (via {model_name})'
    strategy = getattr(model_cfg, 'finetuning_strategy', None)
    init_source = getattr(model_cfg, 'init_source', None)
    dataset_name = config.dataset.train._base_.NAME
    k_folds = config.dataset.train.others.get('K_FOLDS', 1)
    fold = config.dataset.train.others.get('fold', None)
    max_epoch = getattr(config, 'max_epoch', '?')
    lr = config.optimizer.kwargs.get('lr', '?')
    bs = config.dataset.train.others.get('bs', '?')
    selection = config.get('selection_metric', 'acc' if mode != 'seg' else 'instance_miou')
    patience = config.get('early_stopping_patience', 0)

    rows = [('Task', task), ('Model', model_name)]
    if strategy and strategy != 'Full Finetuning':
        rows.append(('Strategy', strategy))
    if init_source:
        rows.append(('Pretrained', init_source))
    rows.append(('Dataset', dataset_name))
    if k_folds > 1 and fold is not None:
        rows.append(('Fold', f'{fold + 1}/{k_folds}'))
    rows.append(('Epochs', str(max_epoch)))
    rows.append(('LR', str(lr)))
    rows.append(('Batch', str(bs)))
    rows.append(('Metric', selection))
    if patience > 0:
        rows.append(('EarlySto', f'{patience} epochs'))
    if mode == 'seg':
        seg_cls = getattr(model_cfg, 'seg_classes', None)
        use_cls = getattr(model_cfg, 'use_cls_label', None)
        if seg_cls is not None:
            rows.append(('SegClasses', str(seg_cls)))
        if use_cls is not None:
            rows.append(('SegType', 'Part' if use_cls else 'Semantic'))

    # Render
    W = 52
    bar = f'{DIM}{"─" * W}{RESET}'
    print(f'\n{bar}')
    print(f'  {BOLD}{CYAN}LIDARLearn{RESET}')
    print(bar)
    for label, value in rows:
        print(f'  {DIM}{label:<12}{RESET} {value}')
    print(f'{bar}\n')

    # Log compact one-liner (no ANSI in log file)
    summary = ' | '.join(f'{k}={v}' for k, v in rows)
    print_log(f'[Config] {summary}', logger=logger)


# ── Training Start ──

def print_training_start(config, selection_metric, early_stopping_patience, logger=None):
    """Print training start info."""
    msg = f'{BOLD}Training{RESET}  {config.max_epoch} epochs  LR={config.optimizer.kwargs.lr}  metric={selection_metric}'
    if early_stopping_patience > 0:
        msg += f'  early_stop={early_stopping_patience}'
    print(f'\n  {msg}\n')
    print_log(
        f'Start training: {config.max_epoch} epochs, LR={config.optimizer.kwargs.lr}, '
        f'metric={selection_metric}',
        logger=logger,
    )


# ── Augmentation ──

def print_augmentation(transforms, logger=None):
    """Print augmentation info."""
    name = type(transforms).__name__
    print(f'  {DIM}Augmentation{RESET}  {name}')
    print_log(f'Augmentation: {name}', logger=logger)


# ── Model Info (params + checkpoint) ──

def print_model_info(trainable, total, ckpt_info=None, logger=None):
    """Print model parameters and checkpoint loading info in a compact block."""
    W = 52
    bar = f'{DIM}{"─" * W}{RESET}'
    pct = 100 * trainable / total if total > 0 else 0

    print(bar)
    print(f'  {DIM}{"Parameters":<14}{RESET} {trainable/1e6:.2f}M / {total/1e6:.2f}M ({pct:.1f}%)')

    if ckpt_info:
        path = ckpt_info['path']
        loaded = ckpt_info['loaded']
        total_keys = ckpt_info['total']
        load_pct = 100 * loaded / total_keys if total_keys > 0 else 0
        print(f'  {DIM}{"Checkpoint":<14}{RESET} {path}')
        print(f'  {DIM}{"Loaded":<14}{RESET} {loaded}/{total_keys} layers ({load_pct:.1f}%)')
    else:
        print(f'  {DIM}{"Checkpoint":<14}{RESET} from scratch')

    print(bar)

    # Log to file
    print_log(f'Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total ({pct:.1f}%)', logger=logger)
    if ckpt_info:
        print_log(f'Checkpoint: {ckpt_info["path"]} ({ckpt_info["loaded"]}/{ckpt_info["total"]} layers)', logger=logger)


# ── Epoch Summary ──

def format_epoch_line(epoch, loss, train_acc, val_acc, f1, kappa, best_acc, lr, time_s, is_best):
    """Format a single epoch summary line for tqdm.write."""
    marker = f' {GREEN}*{RESET}' if is_best else ''
    return (
        f'  {DIM}E{epoch:>3d}{RESET}  '
        f'loss={loss:.4f}  '
        f'train={train_acc:.1f}%  '
        f'{BOLD}val={val_acc:.1f}%{RESET}  '
        f'f1={f1:.1f}%  '
        f'k={kappa:.1f}%  '
        f'{DIM}best={best_acc:.1f}%  '
        f'lr={lr:.6f}  '
        f'{time_s:.1f}s{RESET}'
        f'{marker}'
    )


# ── Final Summary ──

def print_final_summary(metrics_tracker, class_names, epoch, logger=None):
    """Print the final training results."""
    W = 52
    bar = f'{DIM}{"─" * W}{RESET}'

    print(f'\n{bar}')
    print(f'  {BOLD}Results{RESET}  best @ epoch {epoch}')
    print(bar)

    metrics = [
        ('Accuracy',    f'{metrics_tracker.best_acc:.2f}%'),
        ('Balanced',    f'{metrics_tracker.balanced_acc_at_best_acc:.2f}%'),
        ('F1 macro',    f'{metrics_tracker.f1_macro_at_best_acc:.2f}%'),
        ('F1 weighted', f'{metrics_tracker.f1_weighted_at_best_acc:.2f}%'),
        ('Precision',   f'{metrics_tracker.precision_macro_at_best_acc:.2f}%'),
        ('Kappa',       f'{metrics_tracker.kappa_at_best_acc:.2f}%'),
        ('MCC',         f'{metrics_tracker.mcc_at_best_acc:.2f}%'),
    ]

    for label, value in metrics:
        print(f'  {DIM}{label:<14}{RESET} {value}')

    # Per-class
    if class_names:
        print(bar)
        print(f'  {DIM}{"Class":<14} {"Prec":>7} {"Rec":>7} {"F1":>7}{RESET}')
        for cls in class_names:
            p = metrics_tracker.per_class_precision_at_best_acc.get(f'precision_{cls}', 0.0)
            r = metrics_tracker.per_class_recall_at_best_acc.get(f'recall_{cls}', 0.0)
            f = metrics_tracker.per_class_f1_at_best_acc.get(f'f1_{cls}', 0.0)
            print(f'  {cls:<14} {p:>6.1f}% {r:>6.1f}% {f:>6.1f}%')

    print(f'{bar}\n')

    # Log to file (plain text, no ANSI)
    print_log(f'Best @ epoch {epoch}: '
              f'acc={metrics_tracker.best_acc:.2f}% '
              f'f1={metrics_tracker.f1_macro_at_best_acc:.2f}% '
              f'kappa={metrics_tracker.kappa_at_best_acc:.2f}%',
              logger=logger)


# ── Saved Outputs ──

def print_saved_outputs(experiment_path, filenames, logger=None):
    """Print list of saved output files."""
    print(f'  {DIM}Outputs{RESET}  {experiment_path}/')
    for f in filenames:
        print(f'           {f}')
    for f in filenames:
        print_log(f'Saved: {f}', logger=logger)


# ── Early Stopping ──

def print_early_stopping(epoch, metric, patience, logger=None):
    """Print early stopping message."""
    msg = f'  {YELLOW}Early stop{RESET} at epoch {epoch} (no {metric} improvement for {patience} epochs)'
    print(msg)
    print_log(f'Early stopping at epoch {epoch} ({metric}, patience={patience})', logger=logger)


# ── Segmentation Epoch Line ──

def format_seg_epoch_line(epoch, loss, train_acc, val_acc, ins_miou, cls_miou,
                          best_ins_miou, lr, time_s, is_best):
    """Format a segmentation epoch summary line."""
    marker = f' {GREEN}*{RESET}' if is_best else ''
    return (
        f'  {DIM}E{epoch:>3d}{RESET}  '
        f'loss={loss:.4f}  '
        f'train={train_acc:.1f}%  '
        f'{BOLD}val={val_acc:.1f}%{RESET}  '
        f'ins_mIoU={ins_miou:.2f}%  '
        f'cls_mIoU={cls_miou:.2f}%  '
        f'{DIM}best={best_ins_miou:.2f}%  '
        f'lr={lr:.6f}  '
        f'{time_s:.1f}s{RESET}'
        f'{marker}'
    )


# ── Segmentation Summary ──

def print_seg_summary(metrics_tracker, logger=None):
    """Print segmentation final results."""
    W = 52
    bar = f'{DIM}{"─" * W}{RESET}'

    print(f'\n{bar}')
    print(f'  {BOLD}Results{RESET}  best @ epoch {metrics_tracker.best_epoch}')
    print(bar)

    metrics = [
        ('Accuracy',      f'{metrics_tracker.best_accuracy:.2f}%'),
        ('Instance mIoU', f'{metrics_tracker.best_instance_miou:.2f}%'),
        ('Class mIoU',    f'{metrics_tracker.best_class_miou:.2f}%'),
    ]
    for label, value in metrics:
        print(f'  {DIM}{label:<16}{RESET} {value}')

    # Per-category IoU if available
    if hasattr(metrics_tracker, 'best_per_category_iou') and metrics_tracker.best_per_category_iou:
        print(bar)
        print(f'  {DIM}{"Category":<16} {"IoU":>7}{RESET}')
        for cat, iou in sorted(metrics_tracker.best_per_category_iou.items()):
            print(f'  {cat:<16} {iou:>6.2f}%')

    print(f'{bar}\n')

    print_log(
        f'Best @ epoch {metrics_tracker.best_epoch}: '
        f'ins_mIoU={metrics_tracker.best_instance_miou:.2f}% '
        f'cls_mIoU={metrics_tracker.best_class_miou:.2f}%',
        logger=logger,
    )


# ── Training Complete ──

def print_training_complete(best_val, logger=None):
    """Print training complete message."""
    print(f'  {GREEN}{BOLD}Done{RESET}  best: {best_val:.2f}%\n')
    print_log(f'Training complete. Best: {best_val:.2f}%', logger=logger)
