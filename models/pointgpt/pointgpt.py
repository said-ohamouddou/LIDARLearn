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

from .gpt_modules import GPT_extractor, GPT_generator
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
        self.knn_2 = KNN(k=1, transpose_mode=True)

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

    def morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        all_indices = []
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            idxs = np.zeros((self.num_group), dtype=np.int32)
            temp = np.arange(self.num_group)
            z_ind = np.argsort(z[temp])
            idxs = temp[z_ind]
            all_indices.append(idxs)
        all_indices = torch.tensor(all_indices, device=xyz.device)

        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        sorted_indices = all_indices + idx_base
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C
                                  // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GPT_Transformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.decoder_depth = config.transformer_config.decoder_depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.group_size = config.group_size
        print_log(f'[args] {config.transformer_config}', logger='Transformer')

        self.encoder_dims = config.transformer_config.encoder_dims

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            pretrained=True,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        # do not perform additional mask on the first (self.keep_attend) tokens
        self.keep_attend = 10
        self.num_groups = config.num_group
        self.num_mask = int(
            (self.num_groups - self.keep_attend) * self.mask_ratio)

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.norm = nn.LayerNorm(self.trans_dim)
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

    def forward(self, neighborhood, center, noaug=False, classify=False):
        # generate mask

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos_absolute = self.pos_embed(center[:, :-1, :])
        pos_absolute = torch.cat([sos_pos, pos_absolute], dim=1)

        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        with torch.no_grad():
            attn_mask = torch.triu(attn_mask, diagonal=1)

            # point wise
            # overall_mask = np.zeros([self.num_groups, self.num_groups])
            # for i in range(self.num_groups):
            #     mask = np.hstack([
            #         np.zeros(self.num_groups-self.num_mask),
            #         np.ones(self.num_mask),
            #     ])
            #     np.random.shuffle(mask)
            #     overall_mask[i, :] = mask
            # overall_mask = torch.from_numpy(
            #     overall_mask).to(torch.bool).to('cuda')

            # column wise
            overall_mask = np.hstack([
                np.zeros(self.num_groups - self.keep_attend - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(overall_mask)
            overall_mask = np.hstack([
                np.zeros(self.keep_attend),
                overall_mask,
            ])
            overall_mask = torch.from_numpy(
                overall_mask).to(torch.bool).to('cuda')

            eye_mask = torch.eye(self.num_groups).to(torch.bool).to('cuda')

            attn_mask = attn_mask | overall_mask.unsqueeze(0) & ~eye_mask

        # transformer
        if classify == False:
            encoded_features = self.blocks(
                group_input_tokens, pos_absolute, attn_mask, classify=classify)
            generated_points = self.generator_blocks(
                encoded_features, pos_relative, attn_mask)
            return generated_points
        else:
            print_log('----error---- This code is detached ----error----')
            logits, generated_points = self.blocks(
                group_input_tokens, pos_absolute, classify=classify)
            return logits, generated_points

@MODELS.register_module()
class PointGPT_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointGPT] ', logger='PointGPT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.GPT_Transformer = GPT_Transformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.weight_center = config.weight_center

        print_log(
            f'[PointGPT] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='PointGPT')
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        self.loss = config.loss

        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p1 = ChamferDistanceL2().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
        self.loss_func_c = nn.MSELoss().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        B = neighborhood.shape[0]

        generated_points = self.GPT_Transformer(
            neighborhood, center)

        gt_points = neighborhood.reshape(
            B * (self.num_group), self.group_size, 3)
        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)

        if vis:  # visualization
            gt_points = gt_points.reshape(
                B, self.num_group, self.group_size, 3)
            gt_points = (gt_points + center.unsqueeze(-2)
                         ).reshape(-1, 3).unsqueeze(0)
            generated_points = generated_points.reshape(
                B, self.num_group, self.group_size, 3) + center.unsqueeze(-2)
            generated_points = generated_points.reshape(-1, 3).unsqueeze(0)

            return generated_points, gt_points, center

        return loss1 + loss2

@MODELS.register_module()
class PointGPT(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super(PointGPT, self).__init__(config, num_classes)

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        loss_type = config.loss
        self.loss_ce = nn.CrossEntropyLoss()
        if loss_type == "cdl1":
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p1 = ChamferDistanceL2().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

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
            for k in base_ckpt.keys():
                if k.startswith('GPT_Transformer'):
                    new_key = k[len('GPT_Transformer.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)
                elif k.startswith('base_model'):
                    new_key = k[len('base_model.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)
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
            print_log(f'[PointGPT] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

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
                f'[PointGPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat(
            [center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        attn_mask = torch.full(
            (L + 2, L + 2), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        ret, encoded_features = self.blocks(x, pos, attn_mask, classify=True)

        encoded_features = torch.cat(
            [encoded_features[:, 0, :].unsqueeze(1), encoded_features[:, 2:-1, :]], dim=1)

        attn_mask = torch.full(
            (L, L), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        attn_mask = torch.triu(attn_mask, diagonal=1)

        generated_points = self.generator_blocks(
            encoded_features, pos_relative, attn_mask)

        neighborhood = neighborhood + center.unsqueeze(2)

        gt_points = neighborhood.reshape(
            B * (self.num_group), self.group_size, 3)

        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)

        return ret, loss1 + loss2
