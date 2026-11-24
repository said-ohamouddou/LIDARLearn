"""
PointGPT: Auto-regressively Generative Pre-training from Point Clouds

Paper: NeurIPS 2023
Authors: Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue
Source: Implementation adapted from: https://github.com/CGuangyan-BIT/PointGPT
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.knn import KNN

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
from utils.logger import print_log

# Reuse shared building blocks
from ..gpt_blocks import Encoder_large, Encoder_small, PositionEmbeddingCoordsSine, GPTGroup as Group
from .pointgpt_idpt import Block


class GPT_extractor_VPT_Deep(nn.Module):
    """
    GPT extractor with per-layer static prompt injection (VPT-Deep).

    At each layer:
      1. Strip previous layer's prompt tokens (keep SOS + patch only).
      2. Append this layer's static learnable prompt tokens.
      3. Build causal attention mask for the new sequence length.
      4. Run the GPT block.

    Prompts are the SAME for all inputs (not instance-aware).
    """

    def __init__(
        self, embed_dim, num_heads, num_layers, num_classes, trans_dim,
        group_size, num_group=64, prompt_nums=10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.num_group = num_group
        self.num_layers = num_layers
        self.prompt_nums = prompt_nums

        # Start-of-sequence token
        self.sos = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # GPT transformer layers
        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # Per-layer static prompt tokens and positional embeddings
        self.prompt_list = nn.ParameterList([
            nn.Parameter(torch.zeros(1, prompt_nums, embed_dim))
            for _ in range(num_layers)
        ])
        self.prompt_pos_list = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_nums, embed_dim))
            for _ in range(num_layers)
        ])

        # Classification head: SOS + max(patch) = 2 * trans_dim
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(trans_dim * 2, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.cls_norm = nn.LayerNorm(trans_dim)

    def forward(self, h, pos):
        """
        h   : [B, L, C]   — patch tokens
        pos : [B, L+1, C]  — [sos_pos | cls_pos | patch_pos]
        """
        batch, length, C = h.shape

        # Transpose to [seq, batch, C] for nn.MultiheadAttention
        h = h.transpose(0, 1)       # [L, B, C]
        pos = pos.transpose(0, 1)   # [L+1, B, C]  (sos_pos at index 0)

        # Prepend SOS token
        sos = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        h = torch.cat([sos, h], dim=0)   # [1+L, B, C]

        # The base sequence length (SOS + patch tokens), without prompts
        base_len = h.shape[0]  # 1 + L

        for i, layer in enumerate(self.layers):
            # Strip previous layer's prompts — keep only SOS + patch
            h = h[:base_len, :, :]

            # Append this layer's static prompt tokens
            prompt = self.prompt_list[i].expand(batch, -1, -1).transpose(0, 1)      # [P, B, C]
            prompt_pos = self.prompt_pos_list[i].expand(batch, -1, -1).transpose(0, 1)  # [P, B, C]

            h_with_prompt = torch.cat([h, prompt], dim=0)             # [1+L+P, B, C]
            pos_with_prompt = torch.cat([pos, prompt_pos], dim=0)       # [L+2+P, B, C]

            # Causal attention mask
            seq_len = h_with_prompt.shape[0]
            attn_mask = torch.full(
                (seq_len, seq_len), -float("Inf"), device=h.device, dtype=h.dtype
            ).to(torch.bool)
            attn_mask = torch.triu(attn_mask, diagonal=1)

            h_with_prompt = layer(h_with_prompt + pos_with_prompt, attn_mask)

            # Keep the full output (including prompts) for this layer's computation,
            # but next layer will strip prompts again
            h = h_with_prompt

        # Final layer norm (strip prompts for feature extraction)
        h = h[:base_len, :, :]
        h = self.ln_f(h)
        h = h.transpose(0, 1)  # [B, 1+L, C]
        h = self.cls_norm(h)

        # Feature: SOS token (index 0) + max of patch tokens (index 1..L)
        concat_f = torch.cat([
            h[:, 0],
            h[:, 1:self.num_group + 1].max(dim=1)[0],
        ], dim=-1)

        return self.cls_head_finetune(concat_f)


@MODELS.register_module()
class PointGPT_VPT_Deep(BasePointCloudModel):
    """
    Visual Prompt Tuning (Deep) for PointGPT.

    Static learnable prompt tokens at every GPT layer. The prompts are
    input-independent. PointGPT backbone is frozen; only prompt params
    and classification head are trained.

    Config:
        model:
          NAME: PointGPT_VPT_Deep
          trans_dim: 384
          depth: 12
          num_heads: 6
          encoder_dims: 384
          num_group: 64
          group_size: 32
          cls_dim: 40
          prompt_nums: 10
          drop_path_rate: 0.1
    """

    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super().__init__(config, num_classes)

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.prompt_nums = config.prompt_nums

        # PointGPT backbone components
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size
        )

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        # GPT extractor with VPT-Deep static prompts
        self.blocks = GPT_extractor_VPT_Deep(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=self.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            num_group=self.num_group,
            prompt_nums=self.prompt_nums,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.loss_ce = nn.CrossEntropyLoss()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        for pp in self.blocks.prompt_list:
            trunc_normal_(pp, std=0.02)
        for pp in self.blocks.prompt_pos_list:
            trunc_normal_(pp, std=0.02)

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
            base_ckpt = {k.replace("module.", ""): v
                         for k, v in ckpt['base_model'].items()}

            total_ckpt_params = sum(p.numel() for p in base_ckpt.values())

            for k in list(base_ckpt.keys()):
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
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

            used_pct = (matched_params / total_ckpt_params * 100) if total_ckpt_params > 0 else 0.0
            print_log(
                f'[PointGPT_VPT_Deep] Loaded {len(matched_keys)}/{len(base_ckpt)} '
                f'parameter groups ({used_pct:.2f}% of checkpoint parameters)',
                logger='Transformer',
            )
            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger='Transformer')
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)

        B = group_input_tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        sos_pos = self.sos_pos.expand(B, -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)
        pos = torch.cat([sos_pos, cls_pos, pos], dim=1)

        return self.blocks(x, pos)
