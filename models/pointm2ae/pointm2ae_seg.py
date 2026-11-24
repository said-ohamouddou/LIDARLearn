"""
Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training

Paper: NeurIPS 2022
Authors: Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li
Source: Implementation adapted from: https://github.com/ZrrSkywalker/Point-M2AE
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .modules import Group, Token_Embed, Encoder_Block
from ..build import MODELS
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


class H_Encoder_Seg(nn.Module):
    """Hierarchical encoder returning the per-stage feature list.

    Mirrors the reference `H_Encoder_seg`: three Token_Embed stages with
    their own positional embeddings, encoder blocks with local-radius
    attention masks, and a per-stage LayerNorm applied after each stage so
    every level can be propagated independently.
    """

    def __init__(self, encoder_depths, encoder_dims, num_heads,
                 local_radius, drop_path_rate=0.1):
        super().__init__()
        self.encoder_depths = encoder_depths
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(encoder_dims)):
            in_c = 3 if i == 0 else encoder_dims[i - 1]
            self.token_embed.append(Token_Embed(in_c=in_c, out_c=encoder_dims[i]))
            self.encoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, encoder_dims[i]),
                nn.GELU(),
                nn.Linear(encoder_dims[i], encoder_dims[i]),
            ))

        self.encoder_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]
        depth_count = 0
        for i in range(len(encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                embed_dim=encoder_dims[i],
                depth=encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + encoder_depths[i]],
                num_heads=num_heads,
            ))
            depth_count += encoder_depths[i]

        self.encoder_norms = nn.ModuleList([nn.LayerNorm(d) for d in encoder_dims])
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def _local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs):
        x_vis_list = []
        xyz_dist = None
        x_vis = None
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neigh = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neigh)

            if self.local_radius[i] > 0:
                mask_vis_att, xyz_dist = self._local_att_mask(
                    centers[i], self.local_radius[i], xyz_dist)
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list


@MODELS.register_module()
class PointM2AE_Seg(BaseSegModel):
    """Point-M2AE hierarchical segmentation model."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = config.get('trans_dim', 384)
        self.group_sizes = config.get('group_sizes', [16, 8, 8])
        self.num_groups = config.get('num_groups', [512, 256, 64])
        self.encoder_dims = config.get('encoder_dims', [96, 192, 384])
        encoder_depths = config.get('encoder_depths', [5, 5, 5])
        num_heads = config.get('num_heads', 6)
        local_radius = config.get('local_radius', [0.32, 0.64, 1.28])
        drop_path_rate = config.get('drop_path_rate', 0.1)
        dropout = config.get('dropout_head', 0.5)

        assert len(self.group_sizes) == 3 and len(self.encoder_dims) == 3, \
            "Point-M2AE seg expects exactly 3 hierarchical stages"

        self.group_dividers = nn.ModuleList([
            Group(num_group=g, group_size=k)
            for g, k in zip(self.num_groups, self.group_sizes)
        ])

        self.h_encoder = H_Encoder_Seg(
            encoder_depths=encoder_depths,
            encoder_dims=self.encoder_dims,
            num_heads=num_heads,
            local_radius=local_radius,
            drop_path_rate=drop_path_rate,
        )

        # Per-stage PointNetFeaturePropagation: encoder_dim[i] -> 1024
        # Matches reference mlp=[trans_dim*4, 1024]. The +3 in in_channel is
        # the raw xyz that PointNetFeaturePropagation concatenates via points1.
        prop_out = 1024
        self.propagations = nn.ModuleList([
            PointNetFeaturePropagation(
                in_channel=self.encoder_dims[i] + 3,
                mlp=[self.trans_dim * 4, prop_out],
            )
            for i in range(3)
        ])

        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64
        else:
            self.label_conv = None
            label_dim = 0

        # per-point concat (3 * 1024) + global max+avg broadcast (3 * 1024) + label
        per_point_dim = 3 * prop_out   # 3072
        global_dim = 3 * prop_out      # 3072 (max+avg element-wise)
        in_dim = per_point_dim + global_dim + label_dim   # 6208 / 6144

        self.convs1 = nn.Conv1d(in_dim, 1024, 1)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.convs4 = nn.Conv1d(256, self.seg_classes, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        """Load pretrained Point-M2AE encoder weights into this seg model.

        Pretrained checkpoints store keys like `h_encoder.token_embed.*`,
        which line up with `self.h_encoder.*` here. Head params
        (`propagations.*`, `convs*`, `bns*`) are initialized from scratch.
        """
        state = load_checkpoint(ckpt_path, map_location='cpu')
        sd = state.get('base_model', state.get('model', state))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.load_state_dict(sd, strict=strict)

    def forward(self, pts, cls_label=None):
        # Normalise to [B, N, 3] for grouping; keep a [B, 3, N] copy for FP.
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bnc = pts.transpose(1, 2).contiguous()
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bnc = pts.contiguous()
        else:
            raise ValueError(f"pts must be [B, 3, N] or [B, N, 3], got {pts.shape}")

        B, N, _ = pts_bnc.shape
        pts_bcn = pts_bnc.transpose(1, 2).contiguous()

        # Hierarchical grouping: stage i groups on stage (i-1)'s centers.
        neighborhoods, centers, idxs = [], [], []
        for i, gd in enumerate(self.group_dividers):
            if i == 0:
                nb, ct, idx = gd(pts_bnc)
            else:
                nb, ct, idx = gd(centers[-1])
            neighborhoods.append(nb)
            centers.append(ct)
            idxs.append(idx)

        # Per-stage encoder outputs: list of [B, C_i, G_i]
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs)

        # Propagate each stage back to N points → [B, 1024, N].
        prop_feats = []
        for i in range(3):
            f = self.propagations[i](
                xyz1=pts_bnc,
                xyz2=centers[i],
                points1=pts_bcn,
                points2=x_vis_list[i],
            )
            prop_feats.append(f)

        x_per = torch.cat(prop_feats, dim=1)                   # [B, 3072, N]
        x_max = x_per.max(dim=2)[0]                            # [B, 3072]
        x_avg = x_per.mean(dim=2)                              # [B, 3072]
        x_global = (x_max + x_avg).unsqueeze(-1).expand(-1, -1, N)  # [B, 3072, N]

        parts = [x_per, x_global]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))  # [B, 64, 1]
            parts.append(cls_feat.expand(-1, -1, N))

        h = torch.cat(parts, dim=1)                             # [B, in_dim, N]

        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = F.relu(self.bns3(self.convs3(h)))
        h = self.convs4(h)                                      # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
