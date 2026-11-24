"""
Surface Representation for Point Clouds

Paper: CVPR 2022
Authors: Haoxi Ran, Jun Liu, Chengjie Wang
Source: Implementation adapted from: https://github.com/hancyran/RepSurf
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .repsurface_utils import (
    SurfaceAbstractionCD,
    SurfaceFeaturePropagationCD,
    UmbrellaSurfaceConstructor,
)
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class RepSurf_Seg(BaseSegModel):
    """RepSurf segmentation — 4-stage SA encoder + 4-stage FP decoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        return_polar = config.get('return_polar', False)
        return_center = config.get('return_center', True)
        group_size = config.get('group_size', 8)
        return_dist = config.get('return_dist', True)
        umb_pool = config.get('umb_pool', 'sum')
        cuda_ops = config.get('cuda_ops', True)
        in_channel = config.get('in_channel', 0)
        dropout = config.get('dropout', 0.5)

        center_channel = 6 if return_polar else 3
        repsurf_in_channel = 10
        repsurf_out_channel = 10

        self.surface_constructor = UmbrellaSurfaceConstructor(
            group_size + 1, repsurf_in_channel,
            return_dist=return_dist, aggr_type=umb_pool, cuda=cuda_ops,
        )

        # Encoder (4 SA stages)
        self.sa1 = SurfaceAbstractionCD(
            npoint=512, radius=0.2, nsample=32,
            feat_channel=in_channel + repsurf_out_channel,
            pos_channel=center_channel, mlp=[32, 32, 64],
            group_all=False, return_polar=return_polar, cuda=cuda_ops,
        )
        self.sa2 = SurfaceAbstractionCD(
            npoint=128, radius=0.4, nsample=32,
            feat_channel=64 + repsurf_out_channel,
            pos_channel=center_channel, mlp=[64, 64, 128],
            group_all=False, return_polar=return_polar, cuda=cuda_ops,
        )
        self.sa3 = SurfaceAbstractionCD(
            npoint=32, radius=0.8, nsample=32,
            feat_channel=128 + repsurf_out_channel,
            pos_channel=center_channel, mlp=[128, 128, 256],
            group_all=False, return_polar=return_polar, cuda=cuda_ops,
        )
        self.sa4 = SurfaceAbstractionCD(
            npoint=8, radius=1.6, nsample=32,
            feat_channel=256 + repsurf_out_channel,
            pos_channel=center_channel, mlp=[256, 256, 512],
            group_all=False, return_polar=return_polar, cuda=cuda_ops,
        )

        # Decoder (4 FP stages)
        self.fp4 = SurfaceFeaturePropagationCD(512, 256, [256, 256])
        self.fp3 = SurfaceFeaturePropagationCD(256, 128, [256, 256])
        self.fp2 = SurfaceFeaturePropagationCD(256, 64, [256, 128])
        self.fp1 = SurfaceFeaturePropagationCD(128, None, [128, 128, 128])

        # Seg head
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.seg_classes, 1),
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] != 3 and pts.shape[2] == 3:
            pts = pts.permute(0, 2, 1).contiguous()

        center0 = pts[:, :3, :]  # [B, 3, N]
        normal0 = self.surface_constructor(center0)

        # Encoder
        center1, normal1, feat1 = self.sa1(center0, normal0, None)
        center2, normal2, feat2 = self.sa2(center1, normal1, feat1)
        center3, normal3, feat3 = self.sa3(center2, normal2, feat2)
        center4, normal4, feat4 = self.sa4(center3, normal3, feat3)

        # Decoder
        up3 = self.fp4([center3, feat3], [center4, feat4])
        up2 = self.fp3([center2, feat2], [center3, up3])
        up1 = self.fp2([center1, feat1], [center2, up2])
        up0 = self.fp1([center0, None], [center1, up1])

        # Seg head
        logits = self.classifier(up0)
        return F.log_softmax(logits, dim=1)
