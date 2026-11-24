"""
PointMamba: A Simple State Space Model for Point Cloud Analysis

Paper: NeurIPS 2024
Authors: Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, Xiang Bai
Source: Implementation adapted from: https://github.com/LMD0311/PointMamba (models/point_mamba_scan.py)
License: Apache-2.0

Point patches (FPS + kNN) are embedded, ordered along two Hilbert curves
('hilbert' and 'hilbert-trans'), scaled per order and processed as one
sequence by a stack of Mamba blocks. Requires the optional ``mamba-ssm``
package (NVIDIA GPU).
"""

import random

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from models.knn import KNN
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
from utils.logger import print_log

from ..base_model import BasePointCloudModel
from ..build import MODELS
from .mamba_blocks import MixerModel
from .serialization import Point


class Encoder(nn.Module):
    """Mini-PointNet embedding of each point patch: B G N 3 -> B G C."""

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG C n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG C
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS centres + kNN neighbourhoods: B N 3 -> (B G M 3 local patches, B G 3 centres)."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)  # B G 3
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def serialization(pos, feat=None, x_res=None, order="z", layers_outputs=None, grid_size=0.02):
    """Reorder patch centres (and matching features) along a space-filling curve."""
    layers_outputs = layers_outputs if layers_outputs is not None else []
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    grid_coord = torch.floor(pos / grid_size).to(torch.int64)
    grid_coord = grid_coord - grid_coord.min(dim=1, keepdim=True)[0]
    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = Point(batch=batch_idx.flatten(), grid_coord=grid_coord.flatten(0, 1))
    point_dict.serialization(order=order)
    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res


def serialization_func(p, x, x_res, order, layers_outputs=None):
    return serialization(p, x, x_res=x_res, order=order, layers_outputs=layers_outputs, grid_size=0.02)


def init_OrderScale(dim):
    """Per-order learnable scale (gamma) and shift (beta)."""
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    if x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def _trunc_normal_init(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------
# Classification (fine-tuning)
# --------------------------------------------------------------------------
@MODELS.register_module()
class PointMamba(BasePointCloudModel):
    """PointMamba classifier (upstream ``PointMambaScan``)."""

    def __init__(self, config, **kwargs):
        super().__init__(config, num_classes=config.cls_dim)
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.use_cls_token = config.get('use_cls_token', False)
        drop_path = config.get('drop_path', 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]
        self.rms_norm = config.get('rms_norm', False)
        self.drop_out = config.get('drop_out', 0.)
        self.max_head = config.get('max_head', False)
        self.avg_head = config.get('avg_head', False)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModel(d_model=self.trans_dim, n_layer=self.depth, rms_norm=self.rms_norm,
                                 drop_out=self.drop_out, drop_path=dpr)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        if self.max_head and self.avg_head:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        """Load an upstream PointMamba pretraining or fine-tuning checkpoint.

        Returns the ``(missing_keys, unexpected_keys)`` of the non-strict load,
        or None when training from scratch.
        """
        if bert_ckpt_path is None:
            print_log('Training from scratch!!!', logger='PointMamba')
            self.apply(_trunc_normal_init)
            return None

        ckpt = load_checkpoint(bert_ckpt_path, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt.pop(k)
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt.pop(k)

        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='PointMamba')
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger='PointMamba')
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='PointMamba')
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='PointMamba')
        print_log(f'[PointMamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='PointMamba')
        return incompatible

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G C
        pos = self.pos_embed(center)  # B G C

        # Two orderings of the same tokens: Hilbert and transposed Hilbert.
        _, _, _, tokens_forward, pos_forward = serialization_func(center, group_input_tokens, pos, 'hilbert')
        _, _, _, tokens_backward, pos_backward = serialization_func(center, group_input_tokens, pos, 'hilbert-trans')
        tokens_forward = apply_OrderScale(tokens_forward, self.OrderScale_gamma_1, self.OrderScale_beta_1)
        tokens_backward = apply_OrderScale(tokens_backward, self.OrderScale_gamma_2, self.OrderScale_beta_2)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(tokens_forward.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(tokens_forward.size(0), -1, -1)
            pos = torch.cat([pos_forward, pos_backward, cls_token], dim=1)
            x = torch.cat([tokens_forward, tokens_backward, cls_pos], dim=1)
        else:
            pos = torch.cat([pos_forward, pos_backward], dim=1)
            x = torch.cat([tokens_forward, tokens_backward], dim=1)

        x = self.blocks(x, pos)

        if self.use_cls_token:
            concat_f = torch.cat([x[:, -1, :], x.mean(1)], dim=1)
        else:
            feats = []
            if self.avg_head:
                feats.append(x.mean(1))
            if self.max_head:
                feats.append(x.max(1)[0])
            assert feats, "PointMamba needs avg_head and/or max_head when use_cls_token is False"
            concat_f = torch.cat(feats, dim=1)

        return self.cls_head_finetune(concat_f)


# --------------------------------------------------------------------------
# Self-supervised pretraining (masked point modelling)
# --------------------------------------------------------------------------
class MaskMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mamba_config.mask_ratio
        self.trans_dim = config.mamba_config.trans_dim
        self.depth = config.mamba_config.depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        print_log(f'[args] {config.mamba_config}', logger='PointMamba')
        self.encoder_dims = config.mamba_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.mamba_config.mask_type
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = MixerModel(d_model=self.trans_dim, n_layer=self.depth,
                                 rms_norm=self.config.rms_norm, drop_path=dpr)

        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(self.trans_dim)
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(self.trans_dim)

        self.apply(_trunc_normal_init)

    def _mask_center_block(self, center, noaug=False):
        """Mask a contiguous region around a random centre: B G 3 -> B G (bool)."""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)  # 1 G
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        return torch.stack(mask_idx).to(center.device)  # B G

    def _mask_center_rand(self, center, noaug=False):
        """Mask a random subset of patches: B G 3 -> B G (bool)."""
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)
        self.num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - self.num_mask), np.ones(self.num_mask)])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        return torch.from_numpy(overall_mask).to(torch.bool).to(center.device)  # B G

    def forward(self, neighborhood, center, order, order_index, noaug=False):
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        if order_index == 0:
            group_input_tokens = apply_OrderScale(group_input_tokens, self.OrderScale_gamma_1, self.OrderScale_beta_1)
        elif order_index == 1:
            group_input_tokens = apply_OrderScale(group_input_tokens, self.OrderScale_gamma_2, self.OrderScale_beta_2)
        else:
            raise ValueError(f"order_index must be 0 or 1, got {order_index}")

        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        x_vis = self.blocks(x_vis, pos)
        return x_vis, bool_masked_pos, group_input_tokens


class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, drop_path=0, config=None):
        super().__init__()
        self.blocks = MixerModel(d_model=embed_dim, n_layer=depth, rms_norm=config.rms_norm, drop_path=drop_path)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, mask_pos):
        B, _, C = x.shape
        x = self.blocks(x, pos)
        return self.head(x[mask_pos, :].view(B, -1, C))  # predictions for the masked tokens only


@MODELS.register_module()
class PointMamba_Pretrain(nn.Module):
    """PointMamba masked-autoencoder pretraining (upstream ``Point_MAE_Mamba_serializationV2``)."""

    def __init__(self, config, **kwargs):
        super().__init__()
        print_log('[PointMamba] ', logger='PointMamba')
        self.config = config
        self.trans_dim = config.mamba_config.trans_dim
        self.MAE_encoder = MaskMamba(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.mamba_config.decoder_depth
        self.drop_path_rate = config.mamba_config.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = MambaDecoder(embed_dim=self.trans_dim, depth=self.decoder_depth,
                                        drop_path=dpr, config=config)

        print_log(f'[PointMamba] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='PointMamba')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 3 * self.group_size, 1))

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2()
        else:
            raise NotImplementedError(f"unknown loss '{loss_type}' (expected cdl1 or cdl2)")

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        B, G, S, _ = neighborhood.shape

        # Each step uses one of the two orderings at random.
        order_list = ['hilbert', 'hilbert-trans']
        order_index = np.random.choice([i for i in range(len(order_list))], 1, replace=False)
        center, order, _, _, _ = serialization_func(center, None, None, order_list[order_index[0]])
        neighborhood = neighborhood.flatten(0, 1)[order].reshape(B, G, S, -1).contiguous()

        x_vis, mask, group_input_tokens = self.MAE_encoder(neighborhood, center, order=order,
                                                           order_index=order_index[0])
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        x_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        x_full[~mask] = x_vis.reshape(-1, C)
        x_full[mask] = self.mask_token.reshape(-1, C)
        pos_full = torch.zeros_like(group_input_tokens, device=group_input_tokens.device)
        pos_full[~mask] = pos_emd_vis.reshape(-1, C)
        pos_full[mask] = pos_emd_mask.reshape(-1, C)

        x_rec = self.MAE_decoder(x_full, pos_full, mask)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center
        return loss1
