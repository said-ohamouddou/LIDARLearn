"""
Training Runner with Test Set Evaluation
Trains on train set, validates on val set to select best model,
then evaluates final results on test set (like CV fold format).
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
import numpy as np

from .metrics_tracker import Acc_Metric, ValidationMetricsTracker
from .validation import validate
from datasets.augmentation import get_train_transforms
from utils.checkpoint import load_checkpoint


def apply_model_specific_freezing(base_model, config, logger=None):
    """Apply model-specific parameter freezing based on original fine-tuner behavior."""
    model_name = config.model.NAME
    model_type = config.model.get('type', None)
    model_upper = model_name.upper().replace('-', '_')

    if model_upper in ['PPT', 'POINTGPT_PPT'] and model_type == "pos":
        for name, param in base_model.named_parameters():
            if 'cls' in name or 'prompt' in name or 'first_conv' in name or 'pos_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Prompt tuning: only training cls, prompt, first_conv, pos_embed", logger=logger)
        return base_model

    if model_upper in ['IDPT', 'POINTGPT_IDPT'] and model_type == "idpt":
        for name, param in base_model.named_parameters():
            if 'cls' in name or 'prompt' in name or 'pos_embed' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] IDPT tuning: only training cls, prompt, pos_embed", logger=logger)
        return base_model

    if model_upper == 'PCP' and model_type != "full":
        for name, param in base_model.named_parameters():
            if 'cls' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Linear probing: only training cls head", logger=logger)
        return base_model

    if model_upper == 'RECON' and model_type != "full":
        for name, param in base_model.named_parameters():
            if 'cls' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if logger:
            print_log(f"[{model_name}] Linear probing: only training cls head", logger=logger)
        return base_model

    for param in base_model.parameters():
        param.requires_grad = True
    if logger:
        print_log(f"[{model_name}] Full fine-tuning: training all parameters", logger=logger)

    return base_model




def evaluate_on_test_set(base_model, test_dataloader, args, config, class_names, logger=None):
    """Evaluate model on test set and return comprehensive metrics."""
    from sklearn.metrics import (
        balanced_accuracy_score, cohen_kappa_score, f1_score,
        precision_score, matthews_corrcoef, accuracy_score,
        classification_report, confusion_matrix
    )

    base_model.eval()
    test_pred = []
    test_label = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader, desc="Testing", leave=False)):
            points = data[0].cuda()
            label = data[1].cuda()
            logits = base_model(points)

            if isinstance(logits, (tuple, list)) and len(logits) > 1:
                logits = logits[0]

            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        test_pred_np = test_pred.cpu().numpy()
        test_label_np = test_label.cpu().numpy()

        acc = accuracy_score(test_label_np, test_pred_np) * 100.
        balanced_acc = balanced_accuracy_score(test_label_np, test_pred_np) * 100.
        f1_macro = f1_score(test_label_np, test_pred_np, average='macro', zero_division=0) * 100.
        f1_weighted = f1_score(test_label_np, test_pred_np, average='weighted', zero_division=0) * 100.
        precision_macro = precision_score(test_label_np, test_pred_np, average='macro', zero_division=0) * 100.
        precision_weighted = precision_score(test_label_np, test_pred_np, average='weighted', zero_division=0) * 100.
        kappa = cohen_kappa_score(test_label_np, test_pred_np) * 100.
        mcc = matthews_corrcoef(test_label_np, test_pred_np) * 100.

        report = classification_report(
            test_label_np, test_pred_np,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(test_label_np, test_pred_np)

        if args.distributed:
            torch.cuda.synchronize()

    return {
        'accuracy': acc,
        'balanced_acc': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'kappa': kappa,
        'mcc': mcc,
        'per_class_report': report,
        'confusion_matrix': cm,
        'predictions': test_pred_np,
        'labels': test_label_np,
    }


def run_net(args, config):
    logger = get_logger(args.log_name)

    base_model_name = config.model.get('base_model', None)
    finetuning_strategy = config.model.get('finetuning_strategy', None)
    init_source = config.model.get('init_source', None)

    if base_model_name and finetuning_strategy:
        model_name = f"{base_model_name}:{finetuning_strategy}"
    else:
        model_name = config.model.NAME

    if init_source:
        model_name = f"{model_name} [{init_source}]"

    dataset_name = config.dataset.train._base_.NAME

    fmt.print_banner(args, config, logger=logger)

    # Build datasets: train, val, test
    (train_sampler, train_dataloader), (_, val_dataloader) = (
        builder.dataset_builder(args, config.dataset.train),
        builder.dataset_builder(args, config.dataset.val)
    )
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    base_model = apply_model_specific_freezing(base_model, config, logger=logger)

    augmentation_type = args.augmentation if hasattr(args, 'augmentation') else 'none'
    train_transforms = get_train_transforms(augmentation_type)

    fmt.print_augmentation(train_transforms, logger=logger)

    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    train_dataset = train_dataloader.dataset
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
    num_classes = len(class_names) if class_names else config.model.get('num_classes', 9)

    val_metrics_tracker = ValidationMetricsTracker(num_classes, class_names=class_names)

    ckpt_info = None
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    else:
        if args.ckpts is not None:
            pre_state = {k: v.clone() for k, v in base_model.state_dict().items()}
            base_model.load_model_from_ckpt(args.ckpts)
            post_state = base_model.state_dict()
            loaded_count = sum(
                1 for k in pre_state
                if k in post_state and not torch.equal(pre_state[k], post_state[k])
            )
            ckpt_info = {'path': args.ckpts, 'loaded': loaded_count, 'total': len(post_state)}
            del pre_state

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()]
        )
    else:
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    total_params = sum(p.numel() for p in base_model.module.parameters())
    trainable_params = sum(p.numel() for p in base_model.module.parameters() if p.requires_grad)
    fmt.print_model_info(trainable_params, total_params, ckpt_info, logger=logger)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
        # RNG must be restored LAST — after all weight/optimizer allocations
        builder.resume_rng_state(args, logger=logger)

    selection_metric = config.get('selection_metric', 'acc')
    early_stopping_patience = config.get('early_stopping_patience', 0)
    aux_loss_weight = float(config.get('aux_loss_weight', 3.0))
    fmt.print_training_start(config, selection_metric, early_stopping_patience, logger=logger)

    epoch_times = []
    best_val_epoch = 0

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
            points = train_transforms(points)
            ret = base_model(points)

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

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
            losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            batch_pbar.set_postfix({'loss': f'{losses.avg(0):.4f}', 'acc': f'{losses.avg(1):.1f}%'})

        batch_pbar.close()

        if num_iter > 0:
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        lr = optimizer.param_groups[0]['lr']

        # Validation on val set
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics, val_metrics_dict = validate(
                base_model, val_dataloader, epoch, args, config,
                val_metrics_tracker, logger=None
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

            val_metrics_tracker.update_history(
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
            val_acc = val_metrics_dict['val_acc']
            best_val_acc = val_metrics_tracker.best_acc

            epoch_pbar.set_postfix({'loss': f'{losses.avg(0):.3f}', 'val_acc': f'{val_acc:.1f}%', 'best': f'{best_val_acc:.1f}%'})

            val_f1 = val_metrics_dict.get('val_f1_macro', 0)
            val_kappa = val_metrics_dict.get('val_kappa', 0)
            tqdm.write(fmt.format_epoch_line(
                epoch, losses.avg(0), losses.avg(1), val_acc, val_f1,
                val_kappa, best_val_acc, lr, epoch_time, better,
            ))

            if better:
                best_metrics = metrics
                best_val_epoch = epoch
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=None)

            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=None)
        else:
            epoch_pbar.set_postfix({'loss': f'{losses.avg(0):.3f}', 'acc': f'{losses.avg(1):.1f}%'})

    epoch_pbar.close()

    # Final evaluation on test set
    best_ckpt_path = os.path.join(args.experiment_path, 'ckpt-best.pth')
    if os.path.exists(best_ckpt_path):
        checkpoint = load_checkpoint(best_ckpt_path, map_location='cpu')
        state_dict = checkpoint['base_model']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        base_model.module.load_state_dict(state_dict)

    test_results = evaluate_on_test_set(base_model, test_dataloader, args, config, class_names, logger=logger)

    # Create test metrics tracker for saving
    test_metrics_tracker = ValidationMetricsTracker(num_classes, class_names=class_names)
    test_metrics_tracker.best_acc = test_results['accuracy']
    test_metrics_tracker.best_acc_epoch = best_val_epoch
    test_metrics_tracker.balanced_acc_at_best_acc = test_results['balanced_acc']
    test_metrics_tracker.f1_macro_at_best_acc = test_results['f1_macro']
    test_metrics_tracker.f1_weighted_at_best_acc = test_results['f1_weighted']
    test_metrics_tracker.precision_macro_at_best_acc = test_results['precision_macro']
    test_metrics_tracker.precision_weighted_at_best_acc = test_results['precision_weighted']
    test_metrics_tracker.kappa_at_best_acc = test_results['kappa']
    test_metrics_tracker.mcc_at_best_acc = test_results['mcc']
    test_metrics_tracker.confusion_matrix_at_best_acc = test_results['confusion_matrix']
    test_metrics_tracker.predictions_at_best_acc = test_results['predictions']
    test_metrics_tracker.labels_at_best_acc = test_results['labels']

    for class_name in class_names:
        if class_name in test_results['per_class_report']:
            report = test_results['per_class_report'][class_name]
            test_metrics_tracker.per_class_precision_at_best_acc[f"precision_{class_name}"] = report['precision'] * 100
            test_metrics_tracker.per_class_recall_at_best_acc[f"recall_{class_name}"] = report['recall'] * 100
            test_metrics_tracker.per_class_f1_at_best_acc[f"f1_{class_name}"] = report['f1-score'] * 100
            test_metrics_tracker.per_class_support_at_best_acc[f"support_{class_name}"] = report['support']

    fmt.print_final_summary(test_metrics_tracker, class_names, best_val_epoch, logger=logger)

    saved_files = []

    cm_filename = f"confusion_matrix_test_{model_name}.pdf"
    cm_path = os.path.join(args.experiment_path, cm_filename)
    test_metrics_tracker.save_confusion_matrix_pdf(cm_path, title=f"Test Confusion Matrix - {model_name}")

    cm_latex_path = cm_path.replace('.pdf', '.tex')
    test_metrics_tracker.save_confusion_matrix_latex(cm_latex_path, title=f"Test Confusion Matrix - {model_name}")

    history_filename = f"training_history_{model_name}.pdf"
    history_path = os.path.join(args.experiment_path, history_filename)
    val_metrics_tracker.save_training_history_plot(history_path, title=f"{model_name}")

    history_csv_path = os.path.join(args.experiment_path, f"training_history_{model_name}.csv")
    val_metrics_tracker.save_history_csv(history_csv_path)

    # Save test results CSV (same format as fold_0_*.csv)
    test_results_filename = f"test_results_{model_name}.csv"
    test_results_path = os.path.join(args.experiment_path, test_results_filename)
    test_metrics_tracker.save_cv_results_csv(
        test_results_path,
        model_name=model_name,
        fold=0,
        k_folds=1,
        trainable_params=trainable_params,
        total_params=total_params,
        extra_info={
            'dataset': dataset_name,
            'max_epoch': config.max_epoch,
            'npoints': config.npoints,
            'batch_size': config.dataset.train.others.bs,
            'learning_rate': config.optimizer.kwargs.lr,
            'best_val_epoch': best_val_epoch,
            'best_val_acc': val_metrics_tracker.best_acc,
        }
    )

    # Save cv_summary.csv for compatibility with generate_comparison_table.py
    # Uses TEST SET metrics (not validation) - same format as runner_finetune.py
    import pandas as pd
    best_metrics_dict = test_metrics_tracker.get_best_metrics_dict(trainable_params=trainable_params)

    core_metrics = ['best_accuracy', 'balanced_acc', 'f1_macro', 'f1_weighted',
                    'precision_macro', 'precision_weighted', 'kappa', 'mcc']

    summary_data = {
        'model_name': model_name,
        'dataset': dataset_name,
        'k_folds': 1,  # Single test set evaluation
        'trainable_params_M': trainable_params / 1e6,
        'total_params_M': total_params / 1e6,
        'avg_epoch_time_s': sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
        'best_val_epoch': best_val_epoch,
        'best_val_acc': val_metrics_tracker.best_acc,
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
            summary_data[key] = int(value)

    summary_df = pd.DataFrame([summary_data])
    summary_csv_path = os.path.join(args.experiment_path, 'cv_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    saved_files.extend([cm_filename, history_filename, test_results_filename, 'cv_summary.csv'])
    fmt.print_saved_outputs(args.experiment_path, saved_files, logger=logger)
    fmt.print_training_complete(test_results['accuracy'], logger=logger)

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    return {
        'val_metrics_tracker': val_metrics_tracker,
        'test_metrics_tracker': test_metrics_tracker,
        'test_results': test_results,
        'model_type': model_name,
        'dataset_name': dataset_name,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'avg_epoch_time': avg_epoch_time,
        'best_val_epoch': best_val_epoch,
        'best_val_acc': val_metrics_tracker.best_acc,
        'test_acc': test_results['accuracy'],
    }
