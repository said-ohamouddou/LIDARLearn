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
from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from .gpt_modules_ppt import GPT_extractor_PPT
from .z_order import *
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
from utils.logger import print_log

from ..gpt_blocks import Encoder_large, Encoder_small, PositionEmbeddingCoordsSine


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def simplied_morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.view(
            batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, 3).contiguous()

        return neighborhood, center


@MODELS.register_module()
class PointGPT_PPT(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super(PointGPT_PPT, self).__init__(config, num_classes)

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        # PPT-specific parameters
        self.group_size_extra = config.prompt_group_size
        self.num_group_extra = config.prompt_num_group

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)
        self.center_divider = Group(
            num_group=self.num_group_extra, group_size=self.group_size_extra)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        # GPT extractor with PPT prompt tuning
        self.blocks = GPT_extractor_PPT(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            num_group=self.num_group,
            num_group_extra=self.num_group_extra
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        # Loss function
        self.loss_ce = nn.CrossEntropyLoss()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

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

            keys_to_remove = []
            keys_to_add = {}
            for k in list(base_ckpt.keys()):
                # For PointGPT checkpoints
                if k.startswith('GPT_Transformer'):
                    new_key = k[len('GPT_Transformer.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)
                elif k.startswith('base_model'):
                    new_key = k[len('base_model.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)
                # Remove classification head from checkpoint
                if 'cls_head_finetune' in k:
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                if k in base_ckpt:
                    del base_ckpt[k]
            base_ckpt.update(keys_to_add)

            model_state = self.state_dict()
            matched_params = 0
            matched_keys = []
            for k, v in base_ckpt.items():
                if k in model_state and v.shape == model_state[k].shape:
                    matched_params += v.numel()
                    matched_keys.append(k)

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            used_percentage = (matched_params / total_ckpt_params * 100) if total_ckpt_params > 0 else 0.0
            print_log(f'[PointGPT_PPT] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

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
                f'[PointGPT_PPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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
        # Dual group processing (PPT style)
        neighborhood, center = self.group_divider(pts)
        neighbor_center, center_center = self.center_divider(pts)

        group_input_tokens = self.encoder(neighborhood)
        center_input_tokens = self.encoder(neighbor_center)

        B, L, _ = group_input_tokens.shape

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        center_pos = self.pos_embed(center_center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        # 3-way token concatenation: cls + points + centers
        x = torch.cat((cls_tokens, group_input_tokens, center_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos, center_pos), dim=1)

        total_len = L + 1 + self.num_group_extra + 1  # points + cls + centers + sos
        attn_mask = torch.full(
            (total_len, total_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        ret, _ = self.blocks(x, pos, attn_mask, classify=True)

        return ret
