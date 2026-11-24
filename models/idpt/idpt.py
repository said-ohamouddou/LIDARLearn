"""
IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models

Paper: Zha et al., ICCV 2023 — https://arxiv.org/abs/2304.07221
Source: https://github.com/zyh16143998882/ICCV23-IDPT (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    grouping_operation,
)

from .dgcnn import (
    PromptGeneratorDeep,
    PromptGeneratorMedium,
    PromptGeneratorShallow,
    PromptGeneratorMLP,
    PromptGeneratorTransformer,
)

from ..build import MODELS
from ..base_model import BasePointCloudModel
from utils.logger import print_log


# Building blocks — taken verbatim from MIT-licensed Point-BERT
# Yu et al., CVPR 2022. https://github.com/Julie-tang00/Point-BERT

from ..ssl_blocks import Mlp, Attention, Block, TransformerEncoder, Encoder
from utils.checkpoint import load_checkpoint


# Group — FPS + KNN grouping via pointnet2_ops (MIT) + torch.cdist


class Group(nn.Module):
    """
    Farthest-point-sample then k-nearest-neighbor grouping.

    Uses pointnet2_ops for FPS and gather (MIT License, Qi et al.) and
    torch.cdist for KNN index computation.
    """

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        # xyz: [B, N, 3]
        B, N, _ = xyz.shape
        xyz_t = xyz.transpose(1, 2).contiguous()          # [B, 3, N]

        # Farthest point sampling
        center_idx = furthest_point_sample(xyz, self.num_group)    # [B, G]
        center = gather_operation(xyz_t, center_idx)           # [B, 3, G]
        center = center.transpose(1, 2).contiguous()           # [B, G, 3]

        # KNN grouping: torch.cdist → topk
        dists = torch.cdist(center, xyz)                         # [B, G, N]
        knn_idx = dists.topk(self.group_size, dim=-1, largest=False)[1]  # [B, G, k]
        knn_idx = knn_idx.int()

        # Gather neighbors: grouping_operation expects [B, C, N], [B, G, k] → [B, C, G, k]
        neighborhood = grouping_operation(xyz_t, knn_idx)          # [B, 3, G, k]
        neighborhood = neighborhood.permute(0, 2, 3, 1)            # [B, G, k, 3]
        neighborhood = neighborhood - center.unsqueeze(2)          # relative coords

        return neighborhood, center


# BlockWithAdapter — standard Block + bottleneck adapter (Houlsby et al. 2019)
# Paper: "Parameter-Efficient Transfer Learning for NLP", ICML 2019
#        https://arxiv.org/abs/1902.00751

class BlockWithAdapter(nn.Module):
    """
    Transformer block augmented with a bottleneck adapter inserted after the
    feed-forward sublayer.  Adapter weights are zero-initialized so the block
    starts as the identity, preserving any pre-trained weights loaded into the
    attention and MLP sub-modules.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # Bottleneck adapter — down → act → up with inner residual
        self.adapter_down = nn.Linear(dim, dim // 2)
        self.adapter_act = act_layer()
        self.adapter_up = nn.Linear(dim // 2, dim)
        # Zero-initialize so adapter starts as identity
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        h = x                                             # residual before MLP
        mlp_out = self.mlp(self.norm2(x))
        # Adapter with inner skip: makes adapter start as pass-through
        adapted = self.adapter_up(self.adapter_act(self.adapter_down(mlp_out))) + mlp_out
        x = h + self.drop_path(adapted)
        return x


# TransformerEncoderWithAdapter — same as TransformerEncoder but uses
# BlockWithAdapter blocks

class TransformerEncoderWithAdapter(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockWithAdapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


# Prompt generator registry

_PROMPT_MODULES = {
    "dgcnn": PromptGeneratorDeep,
    "dgcnn_light2": PromptGeneratorMedium,
    "dgcnn_light": PromptGeneratorShallow,
    "mlp": PromptGeneratorMLP,
    "transformer": PromptGeneratorTransformer,
}


# DynamicPromptTransformerEncoder — the core IDPT mechanism.
# At a designated layer, a DGCNN-based prompt generator dynamically produces
# an instance-adaptive prompt token from the current patch-token features.
# Zha et al., "IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained
# Point Cloud Models", ICCV 2023.

class DynamicPromptTransformerEncoder(nn.Module):
    """
    Transformer encoder that dynamically generates and injects a prompt
    at a designated layer using a learned prompt generator (e.g., DGCNN).

    Sequence layout before injection (layers 0..prompt_layer-1):
        [cls_token | patch_tokens]                length = 1 + num_group

    Sequence layout after injection (layers prompt_layer..depth-1):
        [cls_token | prompt | patch_tokens]       length = 2 + num_group

    The prompt is generated by passing the current patch-token features
    through the prompt generator (e.g., 3-stage EdgeConv), producing a
    single token that encodes instance-specific geometric structure.
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_group=None, prompt_layer=0, prompt_module="dgcnn",
                 prompt_k=20):
        super().__init__()
        self.num_group = num_group
        self.prompt_layer = prompt_layer

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

        # Build the instance-aware prompt generator
        gen_cls = _PROMPT_MODULES[prompt_module]
        if prompt_module == "transformer":
            self.prompt_gen = gen_cls(embed_dim, num_heads=num_heads)
        elif prompt_module == "mlp":
            self.prompt_gen = gen_cls(embed_dim)
        else:
            self.prompt_gen = gen_cls(embed_dim, k=prompt_k)

    def forward(self, x, pos, prompt_pos):
        """
        x          : [B, 1 + num_group, C]  — [cls_token | patch_tokens]
        pos        : [B, 1 + num_group, C]  — [cls_pos | patch_pos]
        prompt_pos : [B, 1, C]              — learnable prompt position embedding

        Returns x of shape [B, 2 + num_group, C] after prompt injection.
        """
        # Split pos into cls_pos and patch_pos
        cls_pos = pos[:, :1, :]                                # [B, 1, C]
        patch_pos = pos[:, 1:, :]                                # [B, G, C]
        pos_work = pos                                          # [B, 1+G, C]

        for i, block in enumerate(self.blocks):
            if i == self.prompt_layer:
                # Generate instance-adaptive prompt from current patch tokens
                patch_tokens = x[:, -self.num_group:]            # [B, G, C]
                prompt = self.prompt_gen(
                    patch_tokens.transpose(1, 2),                 # [B, C, G]
                    patch_pos,                                    # [B, G, C]
                )                                                 # [B, 1, C]

                # Insert prompt between cls and patch tokens
                x = torch.cat([x[:, :1, :], prompt, x[:, -self.num_group:, :]], dim=1)
                pos_work = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)

            x = block(x + pos_work)

        return x


# DeepPromptTransformerEncoder — VPT-Deep baseline.
# Static learnable prompt tokens appended at every transformer layer.
# Jia et al., "Visual Prompt Tuning", ECCV 2022.

class DeepPromptTransformerEncoder(nn.Module):
    """
    Transformer encoder with per-layer static learnable prompt injection.

    At each transformer layer:
      1. Strip any previous prompt tokens → keep only [cls | patch] tokens.
      2. Append this layer's learnable prompt tokens + positional embeddings.
      3. Run the transformer block on [cls | patch | prompt_i].

    This matches the "Deep Prompt" strategy described in Section 3.2 of:
        Zha et al., "IDPT: Instance-aware Dynamic Prompt Tuning for
        Pre-trained Point Cloud Models", ICCV 2023.

    The prompts are static nn.Parameters (not dynamically generated).
    Each layer has its own set of `prompt_nums` prompt tokens.

    Sequence layout inside each block:
        [cls_token | patch_tokens | prompt_tokens_i]
        length = 1 + num_group + prompt_nums

    Only [cls | patch] tokens (indices 0..num_group) are carried forward
    to the next layer; prompt tokens are discarded and replaced.
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_group=None, prompt_nums=10, **kwargs):
        super().__init__()
        self.num_group = num_group
        self.depth = depth
        self.prompt_nums = prompt_nums

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

        # Per-layer learnable prompt tokens and positional embeddings
        self.prompt_list = nn.ParameterList([
            nn.Parameter(torch.zeros(1, prompt_nums, embed_dim))
            for _ in range(depth)
        ])
        self.prompt_pos_list = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_nums, embed_dim))
            for _ in range(depth)
        ])

    def forward(self, x, pos):
        """
        x   : [B, 1 + num_group, C]   — [cls_token | patch_tokens]
        pos : [B, 1 + num_group, C]   — [cls_pos   | patch_pos]

        Returns:
            x : [B, 1 + num_group + prompt_nums, C] — after the last block
        """
        B = x.shape[0]

        for i, block in enumerate(self.blocks):
            # Strip previous prompt tokens — keep only cls + patch
            x = x[:, :self.num_group + 1, :]       # [B, 1+G, C]
            pos = pos[:, :self.num_group + 1, :]      # [B, 1+G, C]

            # Append this layer's learnable prompt tokens
            prompt = self.prompt_list[i].expand(B, -1, -1)      # [B, P, C]
            prompt_pos = self.prompt_pos_list[i].expand(B, -1, -1)  # [B, P, C]

            x = torch.cat([x, prompt], dim=1)   # [B, 1+G+P, C]
            pos = torch.cat([pos, prompt_pos], dim=1)    # [B, 1+G+P, C]

            x = block(x + pos)

        return x


# Shared checkpoint loading logic

def _load_pretrained(model, ckpt_path: str, strict: bool = False, tag: str = ""):
    """Load weights from a PointMAE / PointBERT / ACT / PCP-MAE checkpoint."""
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = {k.replace("module.", ""): v
             for k, v in ckpt.get("base_model", ckpt).items()}
    prefix_map = {
        "MAE_encoder.": "", "ACT_encoder.": "", "transformer_k.": "",
        "GPT_Transformer.": "", "base_model.": "",
    }
    remapped = {}
    for k, v in state.items():
        new_k = k
        for prefix, repl in prefix_map.items():
            if k.startswith(prefix):
                new_k = repl + k[len(prefix):]
                break
        if "cls_head" not in new_k:
            remapped[new_k] = v
    result = model.load_state_dict(remapped, strict=strict)
    loaded = len(remapped) - len(result.missing_keys)
    print_log(f"[{tag}] Loaded {loaded}/{len(remapped)} parameter groups from {ckpt_path}", logger=tag)
    if result.missing_keys:
        print_log(f"[{tag}] Missing keys: {result.missing_keys[:10]} ...", logger=tag)
    return result


def _init_weights_all(model):
    """Standard weight init for transformer + conv modules."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# IDPT — Instance-aware Dynamic Prompt Tuning (the paper's main contribution).
# A DGCNN-based prompt generator dynamically creates an instance-specific
# prompt token from patch features at a designated layer. The prompt adapts
# to each input point cloud (instance-aware).
# Frozen backbone + trainable prompt generator + cls head.
# Optimizer config: part: only_new

@MODELS.register_module()
class IDPT(BasePointCloudModel):
    """
    Instance-aware Dynamic Prompt Tuning for pre-trained point cloud models.

    At layer `prompt_layer`, a DGCNN-based prompt generator produces a
    single instance-adaptive prompt token from the current patch-token
    features. This prompt is inserted between the cls and patch tokens
    for all subsequent layers.

    The prompt is INSTANCE-AWARE: different inputs produce different prompts.

    Args:
        trans_dim      : transformer embedding dimension
        depth          : number of transformer layers
        num_heads      : attention heads
        drop_path_rate : stochastic depth rate
        cls_dim        : number of output classes
        group_size     : points per group (neighbourhood size)
        num_group      : number of groups (FPS centroids)
        encoder_dims   : mini-PointNet output channels (should equal trans_dim)
        prompt_layer   : layer index at which to inject the dynamic prompt
        prompt_module  : prompt generator type ('dgcnn', 'dgcnn_light',
                         'dgcnn_light2', 'mlp', 'transformer')
        prompt_k       : k for graph-conv based prompt generators
        cls_type       : feature aggregation for classification:
                         'cls'       → cls token only
                         'promptcls' → cls + prompt
                         'pointcls'  → cls + max(patch)
                         'all'       → cls + prompt + max(patch)
    """

    _CLS_TYPE_CHANNELS = {
        "cls": 1, "promptcls": 2, "pointcls": 2, "all": 3,
    }

    def __init__(self, config, **kwargs):
        super().__init__(config, config.cls_dim)

        cls_dim = config.cls_dim
        trans_dim = config.trans_dim
        depth = config.depth
        num_heads = config.num_heads
        drop_path_rate = config.drop_path_rate
        group_size = config.group_size
        num_group = config.num_group
        encoder_dims = config.encoder_dims
        prompt_layer = config.prompt_layer
        prompt_module = config.prompt_module
        prompt_k = config.get('prompt_k', 20)
        cls_type = config.cls_type

        assert prompt_module in _PROMPT_MODULES
        assert cls_type in self._CLS_TYPE_CHANNELS

        self.trans_dim = trans_dim
        self.num_group = num_group
        self.cls_type = cls_type
        channel_mult = self._CLS_TYPE_CHANNELS[cls_type]

        # --- point cloud tokenizer ---
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # --- learnable tokens and positional embeddings ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.prompt_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim),
        )

        # --- transformer with dynamic prompt injection ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = DynamicPromptTransformerEncoder(
            embed_dim=trans_dim, depth=depth, drop_path_rate=dpr,
            num_heads=num_heads, num_group=num_group,
            prompt_layer=prompt_layer, prompt_module=prompt_module,
            prompt_k=prompt_k,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # --- classification head ---
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * channel_mult, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, cls_dim),
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        trunc_normal_(self.prompt_pos, std=0.02)
        _init_weights_all(self)

    def load_pretrained(self, ckpt_path, strict=False):
        return _load_pretrained(self, ckpt_path, strict=strict, tag="IDPT")

    # Alias for runner compatibility
    load_model_from_ckpt = load_pretrained

    def forward(self, pts):
        """pts: [B, N, 3] → logits [B, cls_dim]."""
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)       # [B, G, C]

        B = group_input_tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        prompt_pos = self.prompt_pos.expand(B, -1, -1)

        pos = self.pos_embed(center)                           # [B, G, C]

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)  # [B, 1+G, C]
        pos = torch.cat([cls_pos, pos], dim=1)                     # [B, 1+G, C]

        x = self.blocks(x, pos, prompt_pos)                    # [B, 2+G, C]
        x = self.norm(x)

        # Feature aggregation (prompt at index 1, patch at 2..1+G)
        if self.cls_type == "cls":
            feat = x[:, 0]
        elif self.cls_type == "promptcls":
            feat = torch.cat([x[:, 0], x[:, 1]], dim=-1)
        elif self.cls_type == "pointcls":
            feat = torch.cat([x[:, 0], x[:, 2:].max(1)[0]], dim=-1)
        else:  # "all"
            feat = torch.cat([x[:, 0], x[:, 1], x[:, 2:].max(1)[0]], dim=-1)

        return self.cls_head_finetune(feat)


# VPT_Deep — Visual Prompt Tuning (Deep variant).
# Static learnable prompt tokens appended at EVERY transformer layer.
# The prompts are the SAME regardless of input (not instance-aware).
# Jia et al., "Visual Prompt Tuning", ECCV 2022 — adapted for point clouds in the IDPT paper.
# Frozen backbone + trainable prompt params + cls head.
# Optimizer config: part: only_new

@MODELS.register_module()
class VPT_Deep(BasePointCloudModel):
    """
    Visual Prompt Tuning (Deep) for pre-trained point cloud models.

    Static learnable prompt tokens are appended to the sequence at every
    transformer layer. Prompts are input-independent.

    Args:
        trans_dim      : transformer embedding dimension
        depth          : number of transformer layers
        num_heads      : attention heads
        drop_path_rate : stochastic depth rate
        cls_dim        : number of output classes
        group_size     : points per group
        num_group      : number of groups (FPS centroids)
        encoder_dims   : mini-PointNet output channels (should equal trans_dim)
        prompt_nums    : number of learnable prompt tokens per layer
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, config.cls_dim)

        cls_dim = config.cls_dim
        trans_dim = config.trans_dim
        depth = config.depth
        num_heads = config.num_heads
        drop_path_rate = config.drop_path_rate
        group_size = config.group_size
        num_group = config.num_group
        encoder_dims = config.encoder_dims
        prompt_nums = config.get('prompt_nums', 10)

        self.trans_dim = trans_dim
        self.num_group = num_group

        # --- point cloud tokenizer ---
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # --- learnable tokens and positional embeddings ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim),
        )

        # --- transformer with per-layer static prompt injection ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = DeepPromptTransformerEncoder(
            embed_dim=trans_dim, depth=depth, drop_path_rate=dpr,
            num_heads=num_heads, num_group=num_group,
            prompt_nums=prompt_nums,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # --- classification head ---
        # Feature: cls + max(patch) = 2 * trans_dim (prompts excluded)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * 2, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, cls_dim),
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        for pp in self.blocks.prompt_list:
            trunc_normal_(pp, std=0.02)
        for pp in self.blocks.prompt_pos_list:
            trunc_normal_(pp, std=0.02)
        _init_weights_all(self)

    def load_pretrained(self, ckpt_path, strict=False):
        return _load_pretrained(self, ckpt_path, strict=strict, tag="VPT_Deep")

    load_model_from_ckpt = load_pretrained

    def forward(self, pts):
        """pts: [B, N, 3] → logits [B, cls_dim]."""
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)       # [B, G, C]

        B = group_input_tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        pos = self.pos_embed(center)                           # [B, G, C]

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)  # [B, 1+G, C]
        pos = torch.cat([cls_pos, pos], dim=1)                     # [B, 1+G, C]

        x = self.blocks(x, pos)                                # [B, 1+G+P, C]
        x = self.norm(x)

        # Feature: cls + max(patch), exclude prompt tokens
        concat_f = torch.cat([
            x[:, 0],
            x[:, 1:self.num_group + 1].max(dim=1)[0],
        ], dim=-1)

        return self.cls_head_finetune(concat_f)
