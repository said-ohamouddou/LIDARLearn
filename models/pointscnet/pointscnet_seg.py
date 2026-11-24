"""
PointSCNet: Point Cloud Structure and Correlation Learning Based on
            Space-Filling Curve-Guided Sampling

Paper: Chen et al., Symmetry 2022 — https://doi.org/10.3390/sym14010008
Source: https://github.com/Chenguoz/PointSCNet

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule

from .feature_layers import MultiScaleLocalEncoder
from .attention import DualPathAttention, CrossPointInteraction
from .curve_sampling import HilbertSampler
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class PointSCNet_Seg(BaseSegModel):
    """PointSCNet segmentation — curve-guided encoder + FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        input_channels = config.get('input_channels', 3)
        first_stage_anchors = config.get('first_stage_anchors', 256)
        interaction_refs = config.get('interaction_refs', 64)
        dropout = config.get('dropout', 0.5)

        self.interaction_refs = interaction_refs

        # Initial per-point projection (for skip connection)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )

        # Stage 1: Multi-scale local encoding (N → anchors)
        self.stage1_encoder = MultiScaleLocalEncoder(
            num_anchors=first_stage_anchors,
            radii=[0.1, 0.4],
            neighbors_list=[16, 128],
            input_features=0,
            mlp_configs=[[32, 32, 64], [64, 96, 128]],
        )
        stage1_out_dim = 64 + 128  # 192

        # Curve-based sampler
        self.curve_sampler = HilbertSampler(precision=10)

        # Cross-point interaction
        self.cross_interaction = CrossPointInteraction(
            coord_dim=3, feature_dim=stage1_out_dim,
            num_references=interaction_refs,
        )
        interaction_out_dim = stage1_out_dim + 3  # 195

        # Dual attention refinement
        self.attention_module = DualPathAttention(
            num_channels=interaction_out_dim,
            num_points=first_stage_anchors,
            channel_reduction=2,
        )

        # FP: upsample from anchors to N
        self.fp = PointnetFPModule(mlp=[interaction_out_dim + 64, 256, 128])

        # Seg head
        label_dim = 0
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + label_dim, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        B, C, N = pts.shape

        # Per-point skip features
        l0_feat = self.input_proj(pts[:, :3, :])  # [B, 64, N]
        l0_xyz = pts[:, :3, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

        coords = l0_xyz  # [B, N, 3]

        # Stage 1: Multi-scale local features
        anchor_coords, local_feats = self.stage1_encoder(coords, None)
        # anchor_coords: [B, M, 3], local_feats: [B, 192, M]

        anchor_xyz = anchor_coords.contiguous()  # [B, M, 3]

        # Curve sampling + cross-point interaction
        ref_indices = self.curve_sampler(anchor_coords, self.interaction_refs)
        anchor_coords_t = anchor_coords.transpose(1, 2)
        enhanced_feats = self.cross_interaction(anchor_coords_t, local_feats, ref_indices)
        # [B, 195, M]

        # Dual attention
        refined_feats = self.attention_module(enhanced_feats)  # [B, 195, M]

        # FP upsample: M → N
        up = self.fp(l0_xyz, anchor_xyz, l0_feat, refined_feats)  # [B, 128, N]

        # Seg head
        parts = [up]
        if self.use_cls_label and cls_label is not None:
            cls_feat = self.label_conv(cls_label.unsqueeze(-1)).expand(-1, -1, N)
            parts.append(cls_feat)

        out = torch.cat(parts, dim=1)
        logits = self.seg_head(out)
        return F.log_softmax(logits, dim=1)
