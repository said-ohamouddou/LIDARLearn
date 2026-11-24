import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--ckpts', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency (epochs)')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident)')
    parser.add_argument(
        '--fold', type=int, default=-1)
    parser.add_argument(
        '--tsne',
        action='store_true',
        default=False,
        help='tsne test mode')
    parser.add_argument(
        '--run_all_folds',
        action='store_true',
        default=False,
        help='run training on all CV folds and aggregate results')
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['none', 'rotate', 'scale_translate', 'jitter', 'scale', 'translate',
                 'dropout', 'flip', 'z_rotate_tree', 'y_rotate'],
        default='none',
        help='Data augmentation technique to use during training. Options: '
             'none (identity), rotate (3D rotation), scale_translate, jitter, '
             'scale, translate, dropout, flip, z_rotate_tree (tree-specific Z-axis rotation), '
             'y_rotate (Y-axis rotation as in Point-BERT/PointMamba; ScanObjectNN)')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=None,
        help='Override max_epoch from config file. If not specified, uses value from YAML config.')
    parser.add_argument(
        '--data_fraction',
        type=float,
        default=None,
        help='Fraction of each split to actually use (0 < f <= 1). Useful for '
             'smoke-testing: --data_fraction 0.05 keeps only the first 5%% of '
             'samples per dataloader. Applied via torch.utils.data.Subset, so '
             'no dataset code changes. Default: full data.')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['finetune', 'pretrain', 'pretrain_recon', 'seg'],
        default='finetune',
        help='Training mode: finetune (classification), pretrain (SSL pretraining), '
             'pretrain_recon (ReCon pretraining), or seg (per-point segmentation)')
    parser.add_argument(
        '--init_source',
        type=str,
        default=None,
        help='Override init_source in model config (e.g., "HELIALS" for HELIALS-pretrained weights). '
             'If not specified, uses value from YAML config (default: ShapeNet).')
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=None,
        help='Override early_stopping_patience from config. 0=disabled.')
    parser.add_argument(
        '--selection_metric',
        type=str,
        default=None,
        help='Override selection_metric from config (e.g., acc, f1_macro, balanced_acc).')
    parser.add_argument(
        '--runner',
        type=str,
        choices=['runner_finetune', 'runner_finetune_test'],
        default='runner_finetune',
        help='Training runner to use: runner_finetune (val metrics), '
             'runner_finetune_test (test set final metrics)')

    args = parser.parse_args()

    if args.resume and args.ckpts is not None:
        raise ValueError(
            '--resume and --ckpts cannot be both activated')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem,
                                        args.exp_name)
    args.log_name = Path(args.config).stem
    # Auto-suffix if the target experiment path already exists, unless the
    # user is resuming an interrupted run (which must reuse the path).
    args.experiment_path = _resolve_experiment_path(args)
    create_experiment_dir(args)
    return args


def _resolve_experiment_path(args):
    """Return a non-colliding experiment path.

    * --resume → keep the existing path so we can restore ckpt/optimizer.
    * --run_all_folds → keep the base path (fold_{N}/ subdirs live under it,
      so re-running is an explicit overwrite the user has to accept).
    * Otherwise, if the directory exists, append _run2, _run3, ... so two
      sweeps with the same --exp_name at different seeds don't silently
      overwrite each other.
    """
    path = args.experiment_path
    if args.resume:
        return path
    if getattr(args, 'run_all_folds', False):
        return path
    if not os.path.exists(path):
        return path
    # Already exists — pick the next free suffix
    i = 2
    while os.path.exists(f"{path}_run{i}"):
        i += 1
    new_path = f"{path}_run{i}"
    print(f"[parser] experiment path {path} exists; using {new_path} instead")
    return new_path


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
