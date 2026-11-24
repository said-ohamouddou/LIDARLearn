import os
import sys
import copy
import time
import warnings

warnings.filterwarnings('ignore', message='.*adaptive_max_pool2d_backward_cuda.*')
warnings.filterwarnings('ignore', message='.*TORCH_CUDA_ARCH_LIST.*')

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from tools import finetune_run_net as finetune
from tools.runner_finetune_test import run_net as finetune_test
from tools.runner_pretrain import run_net as pretrain
from tools.runner_pretrain_recon import run_net as pretrain_recon
from tools.runner_seg import run_net as seg_runner
from utils import parser, dist_utils, misc
from utils.logger import get_root_logger, print_log
from utils.config import get_config, log_args_to_file, log_config_to_file


def aggregate_cv_results(all_fold_results, exp_path, logger):
    """Aggregate cross-validation results from all folds and save CSVs."""
    if not all_fold_results:
        print_log("No fold results to aggregate!", logger=logger)
        return None

    model_name = all_fold_results[0]['model_type']
    dataset_name = all_fold_results[0]['dataset_name']
    k_folds = len(all_fold_results)
    trainable_params = all_fold_results[0].get('trainable_params', 0)
    total_params = all_fold_results[0].get('total_params', 0)

    epoch_times = [fr.get('avg_epoch_time', 0.0) for fr in all_fold_results]
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    first_mt = all_fold_results[0]['metrics_tracker']
    class_names = getattr(first_mt, 'class_names', [])

    # Detect "episodic" CV where each fold holds a different class set
    # (e.g. ModelNetFewShot, where fold i is an independently sampled
    # N-way task). In that case per-class aggregation across folds is
    # meaningless — "class_0" in fold 0 and fold 1 refer to different real
    # categories — so we skip it entirely and only aggregate scalar metrics.
    fold_class_names = [
        tuple(getattr(fr['metrics_tracker'], 'class_names', []) or [])
        for fr in all_fold_results
    ]
    episodic_cv = len(set(fold_class_names)) > 1
    if episodic_cv:
        class_names = []
        print_log(
            "[CV] Detected episodic cross-validation (class set varies per "
            "fold, e.g. ModelNetFewShot). Skipping per-class aggregation — "
            "reporting scalar metrics only.",
            logger=logger,
        )

    fold_data = []
    for fold_result in all_fold_results:
        mt = fold_result['metrics_tracker']
        fold_entry = {
            'fold': fold_result['fold'],
            'best_accuracy': mt.best_acc,
            'balanced_acc': mt.balanced_acc_at_best_acc,
            'f1_macro': mt.f1_macro_at_best_acc,
            'f1_weighted': mt.f1_weighted_at_best_acc,
            'precision_macro': mt.precision_macro_at_best_acc,
            'precision_weighted': mt.precision_weighted_at_best_acc,
            'recall_macro': mt.recall_macro_at_best_acc,
            'recall_weighted': mt.recall_weighted_at_best_acc,
            'kappa': mt.kappa_at_best_acc,
            'mcc': mt.mcc_at_best_acc,
            'best_epoch': mt.best_acc_epoch,
        }

        for class_name in class_names:
            for metric_type in ('precision', 'recall', 'f1'):
                key = f"{metric_type}_{class_name}"
                fold_entry[key] = getattr(mt, f'per_class_{metric_type}_at_best_acc', {}).get(key, 0.0)

        fold_data.append(fold_entry)

    df = pd.DataFrame(fold_data)

    metric_cols = [
        'best_accuracy', 'balanced_acc', 'f1_macro', 'f1_weighted',
        'precision_macro', 'precision_weighted', 'recall_macro',
        'recall_weighted', 'kappa', 'mcc',
    ]
    per_class_metric_cols = [
        f"{metric}_{cls}" for cls in class_names
        for metric in ('precision', 'recall', 'f1')
    ]

    class_support = {}
    for class_name in class_names:
        sup_key = f"support_{class_name}"
        class_support[class_name] = sum(
            fr['metrics_tracker'].per_class_support_at_best_acc.get(sup_key, 0)
            for fr in all_fold_results
        )

    summary = {
        'model_name': model_name,
        'dataset': dataset_name,
        'k_folds': k_folds,
        'trainable_params_M': trainable_params / 1e6,
        'total_params_M': total_params / 1e6,
        'avg_epoch_time_s': avg_epoch_time,
    }

    for col in metric_cols + per_class_metric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            summary[f'{col}_mean'] = mean_val
            summary[f'{col}_std'] = std_val
            summary[col] = f"{mean_val:.2f} +/- {std_val:.2f}"

    for class_name, support in class_support.items():
        summary[f'support_{class_name}'] = int(support)

    # Print summary
    print_log("\n" + "=" * 60, logger=logger)
    print_log("CROSS-VALIDATION SUMMARY", logger=logger)
    print_log("=" * 60, logger=logger)
    print_log(f"Model: {model_name}", logger=logger)
    print_log(f"Dataset: {dataset_name}", logger=logger)
    print_log(f"Number of Folds: {k_folds}", logger=logger)
    print_log(f"Trainable Parameters: {trainable_params / 1e6:.2f}M", logger=logger)
    print_log(f"Total Parameters: {total_params / 1e6:.2f}M", logger=logger)
    print_log(f"Avg Epoch Time: {avg_epoch_time:.2f}s", logger=logger)
    print_log("-" * 60, logger=logger)
    print_log(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'Result':<20}", logger=logger)
    print_log("-" * 60, logger=logger)

    for col in metric_cols:
        if col in df.columns:
            print_log(
                f"{col:<25} {summary[f'{col}_mean']:<12.2f} "
                f"{summary[f'{col}_std']:<12.2f} {summary[col]:<20}",
                logger=logger,
            )

    print_log("-" * 60, logger=logger)
    print_log("Per-Fold Results:", logger=logger)
    for _, row in df.iterrows():
        print_log(
            f"  Fold {int(row['fold'])}: Acc={row['best_accuracy']:.2f}%, "
            f"F1={row['f1_macro']:.2f}%, Kappa={row['kappa']:.2f}%",
            logger=logger,
        )
    print_log("=" * 60 + "\n", logger=logger)

    df.to_csv(os.path.join(exp_path, 'cv_all_folds_detailed.csv'), index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(exp_path, 'cv_summary.csv'), index=False)

    print_log(f"Results saved to: {exp_path}", logger=logger)
    return summary, df


def run_all_folds(args, config, logger):
    """Run training on all CV folds and aggregate results."""
    k_folds = config.dataset.train.others.K_FOLDS
    model_name = config.model.NAME
    dataset_name = config.dataset.train._base_.NAME

    print_log(f"\n{'=' * 60}", logger=logger)
    print_log(f"CROSS-VALIDATION: {k_folds} folds | {model_name} | {dataset_name}", logger=logger)
    print_log(f"{'=' * 60}\n", logger=logger)

    all_fold_results = []
    base_exp_path = args.experiment_path

    for fold in range(k_folds):
        print_log(f"FOLD {fold + 1} / {k_folds}", logger=logger)

        if args.seed is not None:
            fold_seed = args.seed * 10000 + fold * 100 + args.local_rank
            misc.set_random_seed(fold_seed, deterministic=args.deterministic)
            print_log(
                f"[SEED] Fold {fold}: seed={fold_seed} "
                f"(base={args.seed}, fold={fold}, rank={args.local_rank})",
                logger=logger,
            )

        fold_config = copy.deepcopy(config)
        fold_config.dataset.train.others.fold = fold
        fold_config.dataset.val.others.fold = fold

        fold_args = copy.deepcopy(args)
        fold_args.experiment_path = os.path.join(base_exp_path, f"fold_{fold}")
        os.makedirs(fold_args.experiment_path, exist_ok=True)

        fold_result = finetune(fold_args, fold_config)
        fold_result['fold'] = fold
        all_fold_results.append(fold_result)

        best_acc = fold_result['metrics_tracker'].best_acc
        print_log(f"\nFold {fold} completed. Best Accuracy: {best_acc:.2f}%", logger=logger)

    print_log(f"\n{'=' * 60}", logger=logger)
    print_log("AGGREGATING CROSS-VALIDATION RESULTS", logger=logger)
    print_log(f"{'=' * 60}\n", logger=logger)

    summary, df = aggregate_cv_results(all_fold_results, base_exp_path, logger)

    print_log(f"\nCross-validation completed. Results saved to: {base_exp_path}", logger=logger)
    return summary, df


def _setup_distributed(args):
    """Initialize distributed training and return world_size."""
    if args.launcher == 'none':
        args.distributed = False
        args.world_size = 1
        return

    args.distributed = True
    dist_utils.init_dist(args.launcher)
    _, world_size = dist_utils.get_dist_info()
    args.world_size = world_size


def _configure_batch_sizes(config, args):
    """Resolve per-loader batch sizes for single-GPU and distributed runs.

    Priority:
      1. Explicit ``others.bs`` already set in the yaml — kept as-is.
      2. ``config.total_bs`` — split across ``world_size`` for distributed,
         used verbatim for single-GPU.
      3. Fallback default (16) so any yaml that forgot to declare ``bs`` still
         runs on a single GPU instead of crashing inside ``dataset_builder``.
    """
    DEFAULT_BS = 16

    def _ensure_bs(split_cfg, value):
        others = split_cfg.others
        if getattr(others, 'bs', None) is None:
            others.bs = value

    if args.distributed:
        world_size = args.world_size
        if hasattr(config, 'total_bs'):
            if config.total_bs % world_size != 0:
                raise ValueError(
                    f"total_bs ({config.total_bs}) must be divisible by "
                    f"world_size ({world_size})"
                )
            per_gpu_bs = config.total_bs // world_size
            config.dataset.train.others.bs = per_gpu_bs
            config.dataset.val.others.bs = per_gpu_bs * 2
            if config.dataset.get('extra_train'):
                config.dataset.extra_train.others.bs = per_gpu_bs * 2
            if config.dataset.get('extra_val'):
                config.dataset.extra_val.others.bs = per_gpu_bs * 2
            if config.dataset.get('test'):
                config.dataset.test.others.bs = per_gpu_bs * 2
        else:
            train_bs = getattr(config.dataset.train.others, 'bs', DEFAULT_BS)
            config.dataset.train.others.bs = max(1, train_bs // world_size)
            config.dataset.val.others.bs = max(1, (train_bs // world_size) * 2)
        return

    # Single-GPU / non-distributed path — inject sensible defaults so configs
    # that don't ship an explicit ``bs`` (e.g. the SSL pretrain yamls) still run.
    train_bs = (
        getattr(config.dataset.train.others, 'bs', None)
        or getattr(config, 'total_bs', None)
        or DEFAULT_BS
    )
    _ensure_bs(config.dataset.train, train_bs)
    _ensure_bs(config.dataset.val, train_bs)
    if config.dataset.get('extra_train'):
        _ensure_bs(config.dataset.extra_train, train_bs)
    if config.dataset.get('extra_val'):
        _ensure_bs(config.dataset.extra_val, train_bs)
    if config.dataset.get('test'):
        _ensure_bs(config.dataset.test, train_bs)


def _apply_cli_overrides(args, config, logger):
    """Apply command-line overrides to config."""
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
        logger.info(f'early_stopping_patience={args.early_stopping_patience} (CLI override)')

    if args.selection_metric is not None:
        config.selection_metric = args.selection_metric
        logger.info(f'selection_metric={args.selection_metric} (CLI override)')

    if args.init_source is not None:
        config.model.init_source = args.init_source
        logger.info(f'init_source={args.init_source} (CLI override)')

    if args.seed is not None:
        config.dataset.train.others.seed = args.seed
        config.dataset.val.others.seed = args.seed
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.seed = args.seed


def main():
    args = parser.get_args()

    # CUDA setup
    args.use_gpu = torch.cuda.is_available()

    # Distributed init (must happen before logger, which depends on rank)
    _setup_distributed(args)

    # Seed (before any data/config loading for reproducibility)
    # Determinism is strictly opt-in via --deterministic. Without it, seeds
    # still produce repeatable runs at the framework level (python/np/torch
    # RNGs are fixed), but cuDNN may pick different kernels across runs and
    # a few CUDA ops (e.g. adaptive_max_pool backward) have no deterministic
    # impl — turning on --deterministic forces warn-and-fallback for those,
    # see utils/misc.set_random_seed.
    if args.seed is not None:
        deterministic = args.deterministic
        misc.set_random_seed(args.seed + args.local_rank, deterministic=deterministic)
    else:
        deterministic = args.deterministic

    # Logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    if args.seed is not None:
        logger.info(
            f'Seed: {args.seed} + rank={args.local_rank}, '
            f'deterministic={deterministic} '
            f'(cuDNN benchmark={"ON (non-reproducible)" if not deterministic else "OFF"})'
        )
    else:
        logger.warning('No random seed specified. Results will NOT be reproducible.')

    # Config
    config = get_config(args, logger=logger)
    _apply_cli_overrides(args, config, logger)
    _configure_batch_sizes(config, args)

    # Log args and config to file only (not terminal)
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    if args.distributed:
        if args.local_rank != torch.distributed.get_rank():
            raise RuntimeError(
                f"local_rank mismatch: args.local_rank={args.local_rank}, "
                f"dist.get_rank()={torch.distributed.get_rank()}"
            )

    # Dispatch to runner
    mode = args.mode
    if mode == 'pretrain':
        pretrain(args, config)
    elif mode == 'pretrain_recon':
        pretrain_recon(args, config)
    elif mode == 'seg':
        seg_runner(args, config)
    elif args.run_all_folds:
        run_all_folds(args, config, logger)
    elif args.runner == 'runner_finetune_test':
        finetune_test(args, config)
    else:
        finetune(args, config)


if __name__ == '__main__':
    main()
