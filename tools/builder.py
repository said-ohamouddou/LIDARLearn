"""
General Builder for All Point Cloud Models
Automatically detects model type and applies the correct optimizer/scheduler logic
Based on 11 original builders: BERT, ACT, GPT, PCP_MAE, M2AE, RECON, PCP, PPT, IDPT, DAPT, GST
"""

import os
import random

import numpy as np
import torch
import torch.optim as optim
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import print_log
from utils import misc
from utils.misc import worker_init_fn, build_lambda_sche
from utils.checkpoint import load_checkpoint
# Try importing timm scheduler, fall back to custom implementation
try:
    from timm.scheduler import CosineLRScheduler
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    CosineLRScheduler = None


def dataset_builder(args, config):
    """Build dataset with optional distributed sampling."""
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'

    # Smoke-test / CI fast path: subset every loader to the first N*fraction
    # samples so entire sweeps can run in minutes without touching dataset code.
    data_fraction = getattr(args, 'data_fraction', None)
    if data_fraction is not None and 0.0 < data_fraction < 1.0:
        full_n = len(dataset)
        keep = max(1, int(full_n * data_fraction))
        if keep < full_n:
            dataset = torch.utils.data.Subset(dataset, list(range(keep)))
            print(
                f"[dataset_builder] data_fraction={data_fraction:.3f} -> "
                f"using {keep}/{full_n} samples from "
                f"{config._base_.NAME}[{config.others.subset}]"
            )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.others.bs,
            num_workers=int(args.num_workers),
            drop_last=config.others.subset == 'train',
            worker_init_fn=worker_init_fn,
            sampler=sampler
        )
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.others.bs,
            shuffle=shuffle,
            drop_last=config.others.subset == 'train',
            num_workers=int(args.num_workers),
            worker_init_fn=worker_init_fn
        )
    return sampler, dataloader


def model_builder(config):
    """Build model from config."""
    model = build_model_from_cfg(config)
    return model


def _build_peft_optimizer(base_model, config):
    """
    Build optimizer for Parameter-Efficient Fine-Tuning (PEFT) models.
    Used by: IDPT, DAPT, GST

    Supports part-based parameter filtering:
    - 'only_new': Train only cls head
    - 'dapt': Train cls + tfts + Adapter
    - 'adapt': Train adapt + cls
    - 'decoder': Train decoder_pos_embed + MAE_decoder + increase_dim
    """
    opti_config = config.optimizer
    opti_config.kwargs.lr = float(opti_config.kwargs.lr)
    opti_config.kwargs.weight_decay = float(opti_config.kwargs.weight_decay)

    part = opti_config.part if hasattr(opti_config, 'part') else 'all'

    # Segmentation-head parameter substrings. These are always trainable
    # under any PEFT regime because the dense-prediction head is built
    # from scratch and has no pretrained counterpart. Classification
    # models do not use any of these names, so adding them here is a
    # no-op for the classification path.
    SEG_HEAD_KEYS = ('convs', 'bns', 'propagation', 'label_conv', 'seg_head')

    def _is_seg_head(name):
        return any(k in name for k in SEG_HEAD_KEYS)

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        print_log(f'[Optimizer] PEFT mode - part: {part}', logger='builder')
        decay = []
        no_decay = []
        trained_params = []

        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue

            should_train = False

            # IDPT cls: only train classification head + prompt generator
            if part == 'only_new':
                should_train = ('cls' in name) or ('prompt' in name) or _is_seg_head(name)

            # IDPT seg: train prompt generators + prompt pos params + seg head
            elif part == 'idpt':
                should_train = (
                    ('prompt' in name)
                    or ('cls' in name)
                    or _is_seg_head(name)
                )

            # DAPT: train cls + tfts + Adapter
            # The DAPT class names its adapter `Adapter_MLP` but the
            # PointGPT_DAPT variant names it `adapt_mlp` (lowercase). We
            # match both by checking for the case-insensitive substring
            # `adapter` and the lowercase `adapt_mlp`.
            elif part == 'dapt':
                should_train = (
                    ('cls' in name)
                    or ('tfts' in name)
                    or ('Adapter' in name)
                    or ('adapter' in name.lower())
                    or ('adapt_mlp' in name)
                    or _is_seg_head(name)
                )

            # GST: train adapt + cls
            elif part == 'adapt':
                should_train = (
                    ('adapt' in name)
                    or ('cls' in name)
                    or _is_seg_head(name)
                )

            # PPT: train prompt MLPs + cls head + first_conv + pos_embed
            # Mirrors original_finetuners/PPT.py
            elif part == 'ppt':
                should_train = (
                    ('prompt' in name)
                    or ('cls' in name)
                    or ('first_conv' in name)
                    or ('pos_embed' in name)
                    or _is_seg_head(name)
                )

            # Decoder: train decoder components
            elif part == 'decoder':
                should_train = ('decoder_pos_embed' in name) or ('MAE_decoder' in name) or ('increase_dim' in name)

            # Full fine-tuning
            else:
                should_train = True

            if should_train:
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
                trained_params.append(name)
            else:
                param.requires_grad = False

        if part != 'all' and part is not None:
            print_log(f'[Optimizer] Training {len(trained_params)} parameter groups', logger='builder')

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]

    param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)

    # Build optimizer
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(param_groups, nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError(f"Optimizer type {opti_config.type} not supported for PEFT models")

    return optimizer


def _build_cls_boosted_optimizer(base_model, config):
    """
    Build optimizer with 10x learning rate for classification head.
    Used by: RECON, PCP

    Special handling for PointTransformer cls head with boosted LR.
    """
    opti_config = config.optimizer
    opti_config.kwargs.lr = float(opti_config.kwargs.lr)
    lr = opti_config.kwargs.lr
    weight_decay = opti_config.kwargs.weight_decay
    skip_list = ()

    print_log('[Optimizer] CLS-Boosted mode (10x LR for cls head)', logger='builder')

    decay = []
    no_decay = []
    finetune_head = []

    for name, param in base_model.module.named_parameters():
        if not param.requires_grad:
            continue

        # 10x LR for cls head in PointTransformer
        if 'cls' in name and config.model.NAME == 'PointTransformer':
            print_log(f"10x LR: {name}", logger='builder')
            finetune_head.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {'params': finetune_head, 'lr': lr * 10},
        {'params': no_decay, 'weight_decay': 0., 'lr': lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': lr}
    ]

    # Build optimizer
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'RAdam':
        optimizer = optim.RAdam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(param_groups, nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError(f"Optimizer type {opti_config.type} not supported for CLS-Boosted models")

    return optimizer


def _build_standard_optimizer(base_model, config):
    """
    Build standard optimizer with weight decay grouping.
    Used by: PointBERT, ACT, PointGPT, PCP-MAE, Point-M2AE, PPT, and others

    Supports AdamW, Adam, RAdam (ACT only), SGD.
    """
    opti_config = config.optimizer
    opti_config.kwargs.lr = float(opti_config.kwargs.lr)
    opti_config.kwargs.weight_decay = float(opti_config.kwargs.weight_decay)

    model_name = config.model.NAME.upper()
    print_log(f'[Optimizer] Standard mode for {model_name}', logger='builder')

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []

        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]

    param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)

    # Build optimizer
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'RAdam':
        # RAdam only supported for ACT
        if model_name == 'ACT':
            optimizer = optim.RAdam(param_groups, **opti_config.kwargs)
        else:
            raise NotImplementedError(f"RAdam optimizer only supported for ACT model, not {model_name}")
    elif opti_config.type == 'SGD':
        # ACT uses momentum=0.9
        if model_name == 'ACT':
            optimizer = optim.SGD(param_groups, nesterov=True, momentum=0.9, **opti_config.kwargs)
        else:
            optimizer = optim.SGD(param_groups, nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError(f"Optimizer type {opti_config.type} not supported")

    return optimizer


def _build_cosine_scheduler(optimizer, sche_config, model_name, max_epoch=None):
    """
    Build CosineLR scheduler with compatibility for different timm versions.
    Falls back to PyTorch's CosineAnnealingLR if timm is unavailable.

    If scheduler.kwargs.epochs is not set, falls back to max_epoch from
    the top-level config to avoid duplication.
    """
    epochs = getattr(sche_config.kwargs, 'epochs', None) or max_epoch
    if epochs is None:
        raise ValueError(
            "scheduler.kwargs.epochs not set and max_epoch not provided. "
            "Set at least one."
        )
    warmup_epochs = getattr(sche_config.kwargs, 'initial_epochs', 10)
    lr_min = 1e-7 if model_name == 'ACT' else 1e-6

    if TIMM_AVAILABLE and CosineLRScheduler is not None:
        # Try newer timm API first (0.9+), then fall back to older API
        try:
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=epochs,
                lr_min=lr_min,
                warmup_lr_init=1e-6,
                warmup_t=warmup_epochs,
                cycle_limit=1,
                t_in_epochs=True
            )
        except TypeError:
            # Older timm versions with different parameters
            try:
                scheduler = CosineLRScheduler(
                    optimizer,
                    t_initial=epochs,
                    cycle_mul=1.,
                    lr_min=lr_min,
                    cycle_decay=0.1,
                    warmup_lr_init=1e-6,
                    warmup_t=warmup_epochs,
                    cycle_limit=1,
                    t_in_epochs=True
                )
            except TypeError:
                # Fall back to PyTorch scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr_min
                )
    else:
        # Use PyTorch's built-in CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )

    return scheduler


def _build_scheduler(optimizer, config):
    """
    Build learning rate scheduler with model-specific configurations.
    """
    sche_config = config.scheduler
    model_name = config.model.NAME.upper()

    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)

    elif sche_config.type == 'CosLR':
        max_epoch = getattr(config, 'max_epoch', None)
        scheduler = _build_cosine_scheduler(optimizer, sche_config, model_name, max_epoch=max_epoch)

    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)

    elif sche_config.type == 'function':
        scheduler = None

    else:
        raise NotImplementedError(f"Scheduler type {sche_config.type} not supported")

    return scheduler


def build_opti_sche(base_model, config):
    """
    General optimizer and scheduler builder that automatically detects model type
    and applies the correct building strategy.

    Strategies:
    1. PEFT (Parameter-Efficient Fine-Tuning): IDPT, DAPT, GST
       - Uses part-based parameter filtering

    2. CLS-Boosted: RECON, PCP
       - 10x learning rate for classification head

    3. Standard: PointBERT, ACT, PointGPT, PCP-MAE, Point-M2AE, PPT, etc.
       - Standard weight decay grouping
    """
    model_name = config.model.NAME.upper()

    # An explicit `optimizer.part` field always forces the PEFT dispatcher,
    # regardless of the model class name. This lets segmentation models
    # (PPT_Seg, DAPT_Seg, ...) opt into part-based parameter freezing
    # without being hardcoded into the class-name allowlist below.
    opti_cfg = config.optimizer
    has_peft_part = hasattr(opti_cfg, 'part') and opti_cfg.part not in (None, 'all')

    # Detect model type and route to appropriate builder
    peft_classes = [
        'IDPT', 'VPT_DEEP', 'POINTGPT_VPT_DEEP', 'DAPT', 'POINTGST',
        'POINTGPT_DAPT', 'POINTGPT_GST',
    ]
    if has_peft_part or model_name in peft_classes:
        optimizer = _build_peft_optimizer(base_model, config)
    elif model_name in ['RECON', 'PCP']:
        optimizer = _build_cls_boosted_optimizer(base_model, config)
    else:
        optimizer = _build_standard_optimizer(base_model, config)

    # Build scheduler (common for all models)
    scheduler = _build_scheduler(optimizer, config)

    return optimizer, scheduler


def resume_model(base_model, args, logger=None):
    """Resume model weights and epoch/metric info from checkpoint.

    NOTE: RNG state is NOT restored here. Call resume_rng_state() AFTER
    resume_optimizer() so no allocations happen between RNG restore and
    the first training step.
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger=logger)

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = load_checkpoint(ckpt_path, map_location=map_location)

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})',
              logger=logger)
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    """Resume optimizer from checkpoint."""
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger=logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger=logger)

    # load state dict
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    optimizer.load_state_dict(state_dict['optimizer'])


def resume_rng_state(args, logger=None):
    """Restore all RNG states from the last checkpoint.

    Must be called AFTER resume_model() AND resume_optimizer() so that
    all CUDA/CPU allocations triggered by weight and optimizer loading
    are complete before the RNG state is locked in. Any allocation after
    this point would consume RNG and cause divergence from the saved run.
    """
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        return
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')

    if 'rng_torch' in state_dict:
        torch.set_rng_state(state_dict['rng_torch'])
        print_log('[RESUME INFO] Restored PyTorch RNG state', logger=logger)
    if 'rng_cuda' in state_dict and state_dict['rng_cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state_dict['rng_cuda'])
        print_log('[RESUME INFO] Restored CUDA RNG state', logger=logger)
    if 'rng_np' in state_dict:
        np.random.set_state(state_dict['rng_np'])
        print_log('[RESUME INFO] Restored NumPy RNG state', logger=logger)
    if 'rng_py' in state_dict:
        random.setstate(state_dict['rng_py'])
        print_log('[RESUME INFO] Restored Python RNG state', logger=logger)


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, skip=False, logger=None):
    """
    Save model checkpoint including full RNG state for exact resume.

    Args:
        skip: If True, skip saving (used by ACT for certain conditions)
    """
    if skip:
        print_log(f"Skipped saving checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)
        return

    if args.local_rank == 0:
        def _to_dict(m):
            if m is None:
                return {}
            if isinstance(m, dict):
                return m
            return m.state_dict()

        torch.save({
            'base_model': base_model.module.state_dict() if args.distributed else base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': _to_dict(metrics),
            'best_metrics': _to_dict(best_metrics),
            # RNG states — required for exact reproducibility on resume
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'rng_np': np.random.get_state(),
            'rng_py': random.getstate(),
        }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)


def save_pretrain_model(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    """
    Save pretrained model checkpoint (RECON specific).
    Filters out img_encoder and text_encoder parameters.
    """
    if args.local_rank == 0:
        def _to_dict(m):
            if m is None:
                return {}
            if isinstance(m, dict):
                return m
            return m.state_dict()

        model = base_model.module.state_dict() if args.distributed else base_model.state_dict()
        select_model = {}
        for k, v in model.items():
            if 'img_encoder' in k or 'text_encoder' in k:
                continue
            select_model[k] = v

        torch.save({
            'base_model': select_model,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': _to_dict(metrics),
            'best_metrics': _to_dict(best_metrics),
            # RNG states — required for exact reproducibility on resume
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'rng_np': np.random.get_state(),
            'rng_py': random.getstate(),
        }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger=logger)


def load_model(base_model, ckpt_path, logger=None, strict=True):
    """
    Load model from checkpoint.

    Args:
        strict: If False, allows mismatched keys (used by PCP)
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger=logger)

    # load state dict
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')

    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')

    base_model.load_state_dict(base_ckpt, strict=strict)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger=logger)
    return
