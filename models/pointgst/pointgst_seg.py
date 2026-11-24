"""
Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning

Paper: IEEE TPAMI 2025 (arXiv 2410.08114)
Authors: Dingkang Liang, Tianrui Feng, Xin Zhou, Yumeng Zhang, Zhikang Zou, Xiang Bai
Source: Implementation adapted from: https://github.com/jerryfeng2003/PointGST
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .pointgst import Block, Group, Encoder, TransformerEncoder
from .pgst import get_basis, sort
from .z_order import xyz2key
from ..build import MODELS
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


class PointGSTTransformerEncoderSeg(nn.Module):
    """GST transformer encoder that returns intermediate feature lists."""

    def __init__(self, cfg, embed_dim, depth, num_heads, drop_path_rate):
        super().__init__()
        dpr = drop_path_rate if isinstance(drop_path_rate, list) else [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                cfg=cfg,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

    def forward(self, x, pos, U, sub_U, idx, fetch_idx=(3, 7, 11)):
        """
        x:   [B, 1+G, C]  (cls token prepended)
        pos: [B, 1+G, C]
        Returns list of patch-token features at fetch_idx layers, each [B, G, C].
        """
        feature_list = []
        G = x.shape[1] - 1  # exclude cls token
        for i, block in enumerate(self.blocks):
            x = block(x + pos, U, sub_U, idx)
            if i in fetch_idx:
                # take only the G patch tokens (skip cls at position 0)
                feature_list.append(x[:, 1:, :])  # [B, G, C]
        return x, feature_list


@MODELS.register_module()
class PointGST_Seg(BaseSegModel):
    """PointGST part / semantic segmentation model (our extension)."""

    FETCH_IDX = (3, 7, 11)
    PROP_DIM = 1024

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', 384))
        self.depth = int(config.get('depth', 12))
        self.num_heads = int(config.get('num_heads', 6))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        encoder_dims = int(config.get('encoder_dims', self.trans_dim))
        drop_path_rate = float(config.get('drop_path_rate', 0.1))
        dropout = float(config.get('dropout_head', 0.5))
        self.local = int(config.get('local', 32))

        # Group divider + token encoder
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        # cls token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = PointGSTTransformerEncoderSeg(
            cfg=config,
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=dpr,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # --- Segmentation head -------------------------------------------
        feat_dim = 3 * self.trans_dim  # 1152 for trans_dim=384

        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64
        else:
            self.label_conv = None
            label_dim = 0

        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, self.PROP_DIM],
        )

        # prop(1024) + max(1152) + avg(1152) + label(64 or 0)
        in_dim = self.PROP_DIM + 2 * feat_dim + label_dim

        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.seg_classes, 1)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
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

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        """Load pretrained SSL weights (PointMAE, ACT, ReCon, PCP-MAE)."""
        if ckpt_path is None:
            return
        ckpt = load_checkpoint(ckpt_path, map_location='cpu')
        sd = ckpt.get('base_model', ckpt.get('model', ckpt))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        remapped = {}
        for k, v in sd.items():
            nk = k
            for prefix in ('MAE_encoder.', 'ACT_encoder.', 'base_model.',
                           'transformer_q.', 'GPT_Transformer.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            if any(s in nk for s in ('cls_head_finetune', 'cls_token', 'cls_pos')):
                continue
            remapped[nk] = v
        return self.load_state_dict(remapped, strict=strict)

    def forward(self, pts, cls_label=None):
        # Accept [B, 3, N] or [B, N, 3]
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B, 3, N] or [B, N, 3], got {pts.shape}")

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()   # [B, N, 3]
        B, N, _ = pts_bnc.shape

        # --- Group divider -----------------------------------------------
        neighborhood, center = self.group_divider(pts_bnc)
        group_input_tokens = self.encoder(neighborhood)   # [B, G, C]

        # --- Spectral bases (computed once, shared across all blocks) -----
        U = get_basis(center)                             # [B, G, G]
        G = center.shape[1]
        c = center * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center = sort(center, idx0)
        group_size = self.local
        group_num = G // group_size
        sub_U = get_basis(
            sub_center.reshape(B * group_num, group_size, 3)
        ).reshape(B, group_num, group_size, group_size)

        # --- Prepend cls token -------------------------------------------
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(center)                      # [B, G, C]

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)   # [B, 1+G, C]
        pos_full = torch.cat([cls_pos, pos], dim=1)               # [B, 1+G, C]

        # --- Transformer with intermediate fetches -----------------------
        x, feature_list = self.blocks(
            x, pos_full, U, sub_U, [idx0, idx1], fetch_idx=self.FETCH_IDX
        )

        # LayerNorm + to channel-first: list of [B, C, G]
        feature_list = [
            self.norm(f).transpose(-1, -2).contiguous() for f in feature_list
        ]
        x_cat = torch.cat(feature_list, dim=1)            # [B, 3C, G]

        # --- Global descriptor -------------------------------------------
        x_max = x_cat.max(dim=2)[0].unsqueeze(-1).expand(-1, -1, N)
        x_avg = x_cat.mean(dim=2).unsqueeze(-1).expand(-1, -1, N)

        global_parts = [x_max, x_avg]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))   # [B, 64, 1]
            global_parts.append(cls_feat.expand(-1, -1, N))
        x_global = torch.cat(global_parts, dim=1)

        # --- Feature propagation G → N -----------------------------------
        f_level_0 = self.propagation_0(
            xyz1=pts_bnc,      # [B, N, 3]
            xyz2=center,       # [B, G, 3]
            points1=pts_bcn,   # [B, 3, N]
            points2=x_cat,     # [B, 3C, G]
        )                                                  # [B, prop_dim, N]

        # --- Head --------------------------------------------------------
        h = torch.cat([f_level_0, x_global], dim=1)       # [B, in_dim, N]
        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = self.convs3(h)                                 # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
