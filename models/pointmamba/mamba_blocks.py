# Copyright (c) 2023, Tri Dao, Albert Gu.
"""
Mamba residual blocks and the stacked Mamba encoder used by PointMamba.

Paper: NeurIPS 2024
Authors: Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, Xiang Bai
Source: Implementation adapted from: https://github.com/LMD0311/PointMamba
        (models/block_scan.py, models/point_mamba_scan.py); the block design
        follows Mamba (Tri Dao, Albert Gu, https://github.com/state-spaces/mamba)
License: Apache-2.0
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    try:
        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Block(nn.Module):
    """Pre-norm residual block: x + DropPath(Mixer(Norm(x)))."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        return hidden_states + self.drop_path(
            self.mixer(self.norm(hidden_states), inference_params=inference_params))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  n_residuals_per_layer=1):
    """GPT-2 style initialisation: residual output projections are scaled by
    1/sqrt(n_residuals_per_layer * n_layer)."""
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False,
                 fused_add_norm=False, layer_idx=None, drop_path=0., device=None, dtype=None):
    ssm_cfg = ssm_cfg or {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm,
                  residual_in_fp32=residual_in_fp32, drop_path=drop_path)
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    """Stack of Mamba blocks applied to (tokens + positional embedding)."""

    def __init__(self, d_model: int, n_layer: int, ssm_cfg=None, norm_epsilon: float = 1e-5,
                 rms_norm: bool = False, initializer_cfg=None, fused_add_norm=False,
                 residual_in_fp32=False, drop_out: float = 0., drop_path=0., device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList([
            create_block(
                d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer, **(initializer_cfg or {})))
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                for i, layer in enumerate(self.layers)}

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
            hidden_states = self.drop_out(hidden_states)
        return self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
