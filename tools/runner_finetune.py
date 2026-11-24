"""
Main Training Runner
Handles the training loop and coordination
Supports all models with their original fine-tuner behavior
"""

import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import get_logger, print_log
from utils.AverageMeter import AverageMeter
from utils import log_format as fmt
import os
from tqdm import tqdm

from .metrics_tracker import Acc_Metric, ValidationMetricsTracker
from .validation import validate

from datasets.augmentation import get_train_transforms


def apply_model_specific_freezing(base_model, config, logger=None):
    """
    Apply model-specific parameter freezing based on original fine-tuner behavior.

    Freezing strategies by model (from original fine-tuners):
    - PPT/PointGPT_PPT: If type=="pos", only train cls, prompt, first_conv, pos_embed
    - IDPT/PointGPT_IDPT: If type=="idpt", only train cls, prompt, pos_embed
    - PCP-MAE: If type!="full", only train cls head
    - RECON: If type!="full", only train cls head
    - All others: Train all parameters (no freezing)
    """
    model_name = config.model.NAME
    model_type = config.model.get('type', None)
    model_upper = model_name.upper().replace('-', '_')

    # PPT / PointGPT_PPT: Freeze all except cls, prompt, first_conv, pos_embed when type == "pos"
    if model_upper in ['PPT', 'POINTGPT_PPT'] and model_type == "pos":
        for name, param in base_model.named_parameters():
            if 'cls' in name or 'prompt' in name or 'first_conv' in name or 'pos_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Prompt tuning: only training cls, prompt, first_conv, pos_embed", logger=logger)
        return base_model

    # IDPT / PointGPT_IDPT: Freeze all except cls, prompt, pos_embed when type == "idpt"
    if model_upper in ['IDPT', 'POINTGPT_IDPT'] and model_type == "idpt":
        for name, param in base_model.named_parameters():
            if 'cls' in name or 'prompt' in name or 'pos_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] IDPT tuning: only training cls, prompt, pos_embed", logger=logger)
        return base_model

    # PCP-MAE: Freeze all except cls when type != "full"
    if model_upper == 'PCP' and model_type != "full":
        for name, param in base_model.named_parameters():
            if 'cls' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Linear probing: only training cls head", logger=logger)
        return base_model

    # RECON: Freeze all except cls when type != "full"
    if model_upper == 'RECON' and model_type != "full":
        for name, param in base_model.named_parameters():
            if 'cls' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Linear probing: only training cls head", logger=logger)
        return base_model

    # All other models: train all parameters (no freezing)
    # PointBERT, Point-MAE, PointGPT, Point-M2AE, DAPT, ACT, IDPT, PointGST, PointGPT-DAPT, ACT-GST
    for param in base_model.parameters():
        param.requires_grad = True
    if logger:
        print_log(f"[{model_name}] Full fine-tuning: training all parameters", logger=logger)

    return base_model




def run_net(args, config):
    logger = get_logger(args.log_name)

    # Create experiment name from config
    # Combine base_model and finetuning_strategy if available
    base_model = config.model.get('base_model', None)
    finetuning_strategy = config.model.get('finetuning_strategy', None)
    init_source = config.model.get('init_source', None)  # e.g., "ShapeNet", "HELIAS", "authors"

    if base_model and finetuning_strategy:
        # Format: "RECON:DAPT" or "RECON:Full Finetuning"
        model_name = f"{base_model}:{finetuning_strategy}"
    else:
        # Fallback to NAME if base_model/finetuning_strategy not available
        model_name = config.model.NAME

    # Append init_source to model_name if specified (differentiates pretrain sources)
    if init_source:
        model_name = f"{model_name} [{init_source}]"
    dataset_name = config.dataset.train._base_.NAME

    # Get CV parameters from config
    k_folds = config.dataset.train.others.K_FOLDS
    fold = config.dataset.train.others.get('fold', None)
    is_cv_mode = k_folds > 1 and fold is not None

    fmt.print_banner(args, config, logger=logger)

    # build dataset
    #
    # NOTE: the second loader is the VALIDATION split (config.dataset.val),
    # and is used both for per-epoch validation and for "best checkpoint"
    # selection. If you have a genuine 3-way train/val/test split and want
    # the best ckpt chosen on val and the final metric reported on test,
    # use --runner runner_finetune_test instead — that runner loads a
    # separate config.dataset.test loader for final-only evaluation.
    (train_sampler, train_dataloader), (_, val_dataloader) = (
        builder.dataset_builder(args, config.dataset.train),
        builder.dataset_builder(args, config.dataset.val)
    )

    # build model
    base_model = builder.model_builder(config.model)

    # Apply model-specific parameter freezing
    base_model = apply_model_specific_freezing(base_model, config, logger=logger)

    # Get data transforms based on augmentation argument
    augmentation_type = args.augmentation if hasattr(args, 'augmentation') else 'none'
    train_transforms = get_train_transforms(augmentation_type)

    fmt.print_augmentation(train_transforms, logger=logger)

    # --- Training control from config ---
    selection_metric = config.get('selection_metric', 'acc')
    early_stopping_patience = config.get('early_stopping_patience', 0)  # 0 = disabled
    aux_loss_weight = float(config.get('aux_loss_weight', 3.0))  # weight for auxiliary loss (e.g. RECON)

    start_epoch = 0
    best_metrics = Acc_Metric(0., selection_metric=selection_metric)
    metrics = Acc_Metric(0., selection_metric=selection_metric)
    epochs_without_improvement = 0

    # Get class names from dataset if available
    train_dataset = train_dataloader.dataset
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
    num_classes = len(class_names) if class_names else config.model.get('num_classes', 7)

    # Initialize metrics tracker with class names
    metrics_tracker = ValidationMetricsTracker(num_classes, class_names=class_names)

    # Load checkpoint
    ckpt_info = None
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric, selection_metric=selection_metric)
    else:
        if args.ckpts is not None:
            # Snapshot params before loading to measure what changed
            pre_state = {k: v.clone() for k, v in base_model.state_dict().items()}
            base_model.load_model_from_ckpt(args.ckpts)
            post_state = base_model.state_dict()
            loaded_count = sum(
                1 for k in pre_state
                if k in post_state and not torch.equal(pre_state[k], post_state[k])
            )
            total_count = len(post_state)
            ckpt_info = {
                'path': args.ckpts,
                'loaded': loaded_count,
                'total': total_count,
            }
            del pre_state

    if args.use_gpu:
        base_model.to(args.local_rank)

    # DDP
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()]
        )
    else:
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler (PEFT freezing happens here for IDPT/DAPT/GST)
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    total_params = sum(p.numel() for p in base_model.module.parameters())
    trainable_params = sum(p.numel() for p in base_model.module.parameters() if p.requires_grad)
    fmt.print_model_info(trainable_params, total_params, ckpt_info, logger=logger)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
        # RNG must be restored LAST — after all weight/optimizer allocations
        builder.resume_rng_state(args, logger=logger)

    fmt.print_training_start(config, selection_metric, early_stopping_patience, logger=logger)

    # Track epoch times for averaging
    epoch_times = []

    # Training loop with tqdm
    epoch_pbar = tqdm(
        range(start_epoch, config.max_epoch + 1),
        desc="Training",
        unit="epoch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    base_model.zero_grad()
    for epoch in epoch_pbar:
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0

        # Batch progress bar
        batch_pbar = tqdm(
            train_dataloader,
            desc=f"  Epoch {epoch:3d}",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for idx, (taxonomy_ids, model_ids, data) in enumerate(batch_pbar):
            num_iter += 1

            points = data[0].cuda()
            label = data[1].cuda()

            # Apply training data augmentation
            points = train_transforms(points)

            ret = base_model(points)

            # Handle tuple/list returns from model
            if isinstance(ret, (tuple, list)):
                loss1 = ret[1] if len(ret) > 1 else None
                ret = ret[0]
            else:
                loss1 = None

            loss, acc = base_model.module.get_loss_acc(ret, label)
            if loss1 is not None:
                _loss = loss + aux_loss_weight * loss1
            else:
                _loss = loss
            _loss.backward()

            # Gradient update step (with optional accumulation)
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(),
                        config.grad_norm_clip,
                        norm_type=2
                    )
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{losses.avg(0):.4f}',
                'acc': f'{losses.avg(1):.1f}%'
            })

        batch_pbar.close()

        # Apply any remaining accumulated gradients at end of epoch
        if num_iter > 0:
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    config.grad_norm_clip,
                    norm_type=2
                )
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        elif scheduler is not None:
            scheduler.step(epoch)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        lr = optimizer.param_groups[0]['lr']

        # Validation
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics, val_metrics_dict = validate(
                base_model, val_dataloader, epoch, args, config,
                metrics_tracker, logger=None  # Suppress verbose validation logging
            )
            # Populate metrics with full dict for flexible selection
            metrics.selection_metric = selection_metric
            metrics.set_metrics({
                'acc': val_metrics_dict['val_acc'],
                'balanced_acc': val_metrics_dict['val_balanced_acc'],
                'f1_macro': val_metrics_dict['val_f1_macro'],
                'kappa': val_metrics_dict['val_kappa'],
                'mcc': val_metrics_dict['val_mcc'],
            })

            # Update history for plotting
            metrics_tracker.update_history(
                epoch=epoch,
                train_loss=losses.avg(0),
                train_acc=losses.avg(1),
                val_acc=val_metrics_dict['val_acc'],
                val_balanced_acc=val_metrics_dict['val_balanced_acc'],
                val_f1_macro=val_metrics_dict['val_f1_macro'],
                val_kappa=val_metrics_dict['val_kappa'],
                val_mcc=val_metrics_dict['val_mcc'],
            )

            better = metrics.better_than(best_metrics)

            # Compact epoch summary
            val_acc = val_metrics_dict['val_acc']
            val_f1 = val_metrics_dict['val_f1_macro']
            val_kappa = val_metrics_dict['val_kappa']
            best_acc = metrics_tracker.best_acc

            # Update epoch progress bar description
            epoch_pbar.set_postfix({
                'loss': f'{losses.avg(0):.3f}',
                'val_acc': f'{val_acc:.1f}%',
                'best': f'{best_acc:.1f}%'
            })

            tqdm.write(fmt.format_epoch_line(
                epoch, losses.avg(0), losses.avg(1), val_acc, val_f1,
                val_kappa, best_acc, lr, epoch_time, better,
            ))

            # Save checkpoints (silently)
            if better:
                best_metrics = metrics
                epochs_without_improvement = 0
                builder.save_checkpoint(
                    base_model, optimizer, epoch, metrics, best_metrics,
                    'ckpt-best', args, logger=None
                )
            else:
                epochs_without_improvement += 1

            builder.save_checkpoint(
                base_model, optimizer, epoch, metrics, best_metrics,
                'ckpt-last', args, logger=None
            )

            # Early stopping check
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                fmt.print_early_stopping(epoch, selection_metric, early_stopping_patience, logger=logger)
                break
        else:
            # Non-validation epoch - just show training stats
            epoch_pbar.set_postfix({
                'loss': f'{losses.avg(0):.3f}',
                'acc': f'{losses.avg(1):.1f}%'
            })

    epoch_pbar.close()

    # Final summary
    fmt.print_final_summary(metrics_tracker, class_names, metrics_tracker.best_acc_epoch, logger=logger)

    saved_files = []

    # Save confusion matrix as PDF
    if is_cv_mode:
        cm_filename = f"confusion_matrix_fold{fold}_{model_name}.pdf"
    else:
        cm_filename = f"confusion_matrix_{model_name}.pdf"
    cm_path = os.path.join(args.experiment_path, cm_filename)
    metrics_tracker.save_confusion_matrix_pdf(cm_path, title=f"Confusion Matrix - {model_name}")

    # Save confusion matrix as LaTeX
    cm_latex_filename = cm_filename.replace('.pdf', '.tex')
    cm_latex_path = os.path.join(args.experiment_path, cm_latex_filename)
    metrics_tracker.save_confusion_matrix_latex(cm_latex_path, title=f"Confusion Matrix - {model_name}")

    # Save training history plots as separate PDFs
    if is_cv_mode:
        history_filename = f"training_history_fold{fold}_{model_name}.pdf"
    else:
        history_filename = f"training_history_{model_name}.pdf"
    history_path = os.path.join(args.experiment_path, history_filename)
    metrics_tracker.save_training_history_plot(history_path, title=f"{model_name}")

    # Save training history as CSV
    if is_cv_mode:
        history_csv_filename = f"training_history_fold{fold}_{model_name}.csv"
    else:
        history_csv_filename = f"training_history_{model_name}.csv"
    history_csv_path = os.path.join(args.experiment_path, history_csv_filename)
    metrics_tracker.save_history_csv(history_csv_path)

    # Save CV results to CSV (for cross-validation tracking across folds)
    if is_cv_mode:
        cv_results_filename = f"fold_{fold}_{model_name}.csv"
        cv_results_path = os.path.join(args.experiment_path, cv_results_filename)
        metrics_tracker.save_cv_results_csv(
            cv_results_path,
            model_name=model_name,
            fold=fold,
            k_folds=k_folds,
            trainable_params=trainable_params,
            total_params=total_params,
            extra_info={
                'dataset': dataset_name,
                'max_epoch': config.max_epoch,
                'npoints': config.npoints,
                'batch_size': config.dataset.train.others.bs,
                'learning_rate': config.optimizer.kwargs.lr,
            }
        )
    else:
        # Save summary CSV for simple (non-CV) training - compatible with table generation
        # Uses same format as CV summary: metric_mean, metric_std, and combined "mean ± std" string
        import pandas as pd
        best_metrics_dict = metrics_tracker.get_best_metrics_dict(trainable_params=trainable_params)

        # Core metrics to include (matching CV summary format)
        # Note: key names must match get_best_metrics_dict() output
        core_metrics = ['best_accuracy', 'balanced_acc', 'f1_macro', 'f1_weighted',
                        'precision_macro', 'precision_weighted', 'kappa', 'mcc']

        summary_data = {
            'model_name': model_name,
            'dataset': dataset_name,
            'k_folds': 1,  # Single run = 1 fold
            'trainable_params_M': trainable_params / 1e6,
            'total_params_M': total_params / 1e6,
            'avg_epoch_time_s': sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
        }

        # Add core metrics with _mean, _std, and combined format
        for metric in core_metrics:
            val = best_metrics_dict.get(metric, 0.0)
            summary_data[f'{metric}_mean'] = val
            summary_data[f'{metric}_std'] = 0.0  # No std for single run
            summary_data[metric] = f"{val:.2f} ± 0.00"

        # Add per-class metrics if available
        for key, value in best_metrics_dict.items():
            if key.startswith(('precision_', 'recall_', 'f1_')) and key not in core_metrics:
                summary_data[f'{key}_mean'] = value
                summary_data[f'{key}_std'] = 0.0
                summary_data[key] = f"{value:.2f} ± 0.00"
            elif key.startswith('support_'):
                # Support values are integers, no mean/std needed
                summary_data[key] = int(value)

        summary_df = pd.DataFrame([summary_data])
        summary_csv_path = os.path.join(args.experiment_path, 'cv_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        saved_files.append('cv_summary.csv')

    saved_files.extend([cm_filename, history_filename, history_csv_filename])
    if is_cv_mode:
        saved_files.append(cv_results_filename)

    fmt.print_saved_outputs(args.experiment_path, saved_files, logger=logger)
    fmt.print_training_complete(metrics_tracker.best_acc, logger=logger)

    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    # Return metrics tracker and training info for CV aggregation
    return {
        'metrics_tracker': metrics_tracker,
        'model_type': model_name,
        'dataset_name': dataset_name,
        'fold': fold,
        'k_folds': k_folds,
        'is_cv_mode': is_cv_mode,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'avg_epoch_time': avg_epoch_time,
        'best_metrics': best_metrics,
    }
