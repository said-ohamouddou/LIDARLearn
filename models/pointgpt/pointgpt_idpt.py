"""
PointGPT: Auto-regressively Generative Pre-training from Point Clouds

Paper: NeurIPS 2023
Authors: Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue
Source: Implementation adapted from: https://github.com/CGuangyan-BIT/PointGPT
License: MIT
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from models.knn import KNN

from .z_order import *
from ..idpt.dgcnn import (
    PromptGeneratorDeep,
    PromptGeneratorShallow,
    PromptGeneratorMedium,
    PromptGeneratorMLP,
)
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
from utils.logger import print_log

from ..gpt_blocks import Encoder_large, Encoder_small, PositionEmbeddingCoordsSine, GPTGroup as Group


class Block(nn.Module):
    """GPT-style block with causal attention."""

    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, attn_mask):
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT_extractor_IDPT(nn.Module):
    """
    GPT extractor with IDPT instance-aware dynamic prompt injection.
    Injects prompts at a specific layer based on instance features.
    """

    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim, group_size,
        num_group=64, prompt_layer=11, prompt_module="dgcnn", cls_type="all"
    ):
        super(GPT_extractor_IDPT, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.num_group = num_group
        self.prompt_layer = prompt_layer
        self.cls_type = cls_type
        self.num_layers = num_layers

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # Standard GPT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)

        # IDPT prompt module
        if prompt_module == "transformer":
            self.prompt_generator = self._build_transformer_prompt(embed_dim, num_heads)
        elif prompt_module == "dgcnn":
            self.prompt_generator = self._build_dgcnn_prompt(embed_dim)
        elif prompt_module == "dgcnn_light":
            self.prompt_generator = self._build_dgcnn_light_prompt(embed_dim)
        elif prompt_module == "dgcnn_light2":
            self.prompt_generator = self._build_dgcnn_light2_prompt(embed_dim)
        elif prompt_module == "mlp":
            self.prompt_generator = self._build_mlp_prompt(embed_dim)
        else:
            raise ValueError(f"Unknown prompt_module: {prompt_module}")

        # Determine channel multiplier for cls head
        self.CHANNEL_NUM = 1
        if cls_type == "all":
            self.CHANNEL_NUM = 3
        elif cls_type in ["promptcls", "pointcls"]:
            self.CHANNEL_NUM = 2

        # Classification head
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * self.CHANNEL_NUM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.cls_norm = nn.LayerNorm(trans_dim)

    def _build_transformer_prompt(self, embed_dim, num_heads):
        """Build transformer-based prompt generator."""
        return nn.ModuleDict({
            'blocks': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=False),
                num_layers=1
            ),
        })

    def _build_dgcnn_prompt(self, embed_dim):
        """Build 3-stage EdgeConv prompt generator."""
        return PromptGeneratorDeep(embed_dim, k=20)

    def _build_dgcnn_light_prompt(self, embed_dim):
        """Build 1-stage EdgeConv prompt generator."""
        return PromptGeneratorShallow(embed_dim, k=20)

    def _build_dgcnn_light2_prompt(self, embed_dim):
        """Build 2-stage EdgeConv prompt generator."""
        return PromptGeneratorMedium(embed_dim, k=20)

    def _build_mlp_prompt(self, embed_dim):
        """Build MLP-only prompt generator."""
        return PromptGeneratorMLP(embed_dim)

    def forward(self, h, pos, prompt_pos, classify=True):
        """
        Forward pass with IDPT prompt injection.
        h: [batch, length, C] - input tokens
        pos: [batch, length, C] - position embeddings
        prompt_pos: [batch, 1, C] - prompt position embedding
        """
        batch, length, C = h.shape

        h = h.transpose(0, 1)  # [length, batch, C]
        pos = pos.transpose(0, 1)
        prompt_pos_t = prompt_pos.transpose(0, 1)  # [1, batch, C]

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        h = torch.cat([sos, h], axis=0)

        # Store original pos for prompt injection
        pos_for_prompt = pos.clone()

        # Track if prompt has been injected
        prompt_injected = False

        for layer_idx, layer in enumerate(self.layers):
            # Build causal attention mask
            seq_len = h.shape[0]
            attn_mask = torch.full(
                (seq_len, seq_len), -float("Inf"), device=h.device, dtype=h.dtype
            ).to(torch.bool)
            attn_mask = torch.triu(attn_mask, diagonal=1)

            # Inject prompt at specified layer
            if layer_idx == self.prompt_layer and not prompt_injected:
                # Get point features for prompt generation (exclude sos)
                point_features = h[1:, :, :]  # [length, batch, C]

                # Generate instance-aware prompt
                if isinstance(self.prompt_generator, nn.ModuleDict):
                    # Transformer-based prompt
                    prompt = self.prompt_generator['blocks'](point_features)
                    prompt = torch.max(prompt, dim=0, keepdim=True)[0]  # [1, batch, C]
                else:
                    # DGCNN/MLP-based prompt
                    prompt = self.prompt_generator(
                        point_features.permute(1, 2, 0),  # [batch, C, length]
                        pos_for_prompt.permute(1, 0, 2)[:, :, :]  # [batch, length, C]
                    )  # [batch, 1, C]
                    prompt = prompt.transpose(0, 1)  # [1, batch, C]

                # Insert prompt after sos token
                h = torch.cat([h[:1, :, :], prompt, h[1:, :, :]], dim=0)

                # Update position embeddings to include prompt position
                pos = torch.cat([pos[:1, :, :], prompt_pos_t, pos[1:, :, :]], dim=0)

                prompt_injected = True

                # Rebuild attention mask for new sequence length
                seq_len = h.shape[0]
                attn_mask = torch.full(
                    (seq_len, seq_len), -float("Inf"), device=h.device, dtype=h.dtype
                ).to(torch.bool)
                attn_mask = torch.triu(attn_mask, diagonal=1)

            h = layer(h + pos, attn_mask)

        h = self.ln_f(h)
        h = h.transpose(0, 1)  # [batch, seq_len, C]
        h = self.cls_norm(h)

        # Feature aggregation based on cls_type
        # After prompt injection: [sos, prompt, point_tokens...]
        if self.cls_type == "promptonly":
            concat_f = h[:, 1]  # prompt token only
        elif self.cls_type == "promptcls":
            concat_f = torch.cat([h[:, 0], h[:, 1]], dim=-1)  # sos + prompt
        elif self.cls_type == "pointcls":
            concat_f = torch.cat([h[:, 0], h[:, 2:].max(1)[0]], dim=-1)  # sos + max(points)
        else:  # "all"
            concat_f = torch.cat([h[:, 0], h[:, 1], h[:, 2:].max(1)[0]], dim=-1)  # sos + prompt + max(points)

        ret = self.cls_head_finetune(concat_f)
        return ret


@MODELS.register_module()
class PointGPT_IDPT(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super(PointGPT_IDPT, self).__init__(config, num_classes)

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.prompt_layer = config.prompt_layer
        self.prompt_module = config.prompt_module
        self.cls_type = config.cls_type

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        # GPT extractor with IDPT prompt injection
        self.blocks = GPT_extractor_IDPT(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            num_group=self.num_group,
            prompt_layer=self.prompt_layer,
            prompt_module=self.prompt_module,
            cls_type=self.cls_type
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.prompt_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        # Loss function
        self.loss_ce = nn.CrossEntropyLoss()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.prompt_pos, std=.02)

    def get_loss_acc(self, ret, gt):
        if isinstance(ret, tuple):
            logits = ret[0]
        else:
            logits = ret
        loss = self.loss_ce(logits, gt.long())
        pred = logits.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = load_checkpoint(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            total_ckpt_params = sum(p.numel() for p in base_ckpt.values())

            for k in list(base_ckpt.keys()):
                # For PointGPT checkpoints
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # Remove classification head from checkpoint
                if 'cls_head_finetune' in k:
                    del base_ckpt[k]

            model_state = self.state_dict()
            matched_params = 0
            matched_keys = []
            for k, v in base_ckpt.items():
                if k in model_state and v.shape == model_state[k].shape:
                    matched_params += v.numel()
                    matched_keys.append(k)

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            used_percentage = (matched_params / total_ckpt_params * 100) if total_ckpt_params > 0 else 0.0
            print_log(f'[PointGPT_IDPT] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(
                f'[PointGPT_IDPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)

        B, L, _ = group_input_tokens.shape

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        prompt_pos = self.prompt_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)

        # Concatenate cls token with group tokens
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((sos_pos, cls_pos, pos), dim=1)

        ret = self.blocks(x, pos, prompt_pos, classify=True)

        return ret
