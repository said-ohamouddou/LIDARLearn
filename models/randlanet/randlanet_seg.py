"""
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

Paper: Hu et al., CVPR 2020 — https://arxiv.org/abs/1911.11236
Source: https://github.com/aRI0U/RandLA-Net-pytorch (no license)

Clean implementation.
"""

import torch
import torch.nn.functional as F

from .randlanet import RandLANet
from ..build import MODELS
from ..base_seg_model import BaseSegModel


@MODELS.register_module()
class RandLANet_Seg(BaseSegModel):
    """RandLA-Net segmentation — native encoder-decoder with random sampling."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

        channels = config.get('channels', 3)
        k = config.get('k', 16)
        decimation = config.get('decimation', 4)
        d_encoder = config.get('d_encoder', 8)
        num_stages = config.get('num_stages', 4)
        dropout = config.get('dropout', 0.5)

        self.backbone = RandLANet(
            d_in=channels,
            num_classes=self.seg_classes,
            k=k,
            decimation=decimation,
            d_encoder=d_encoder,
            num_stages=num_stages,
            dropout=dropout,
        )

    def forward(self, pts, cls_label=None):
        # pts: [B, 3, N] or [B, N, 3]
        if pts.shape[1] == 3 or pts.shape[1] < pts.shape[2]:
            pts = pts.permute(0, 2, 1).contiguous()
        # pts: [B, N, C]

        logits = self.backbone(pts)  # [B, N, seg_classes]
        logits = logits.permute(0, 2, 1).contiguous()  # [B, seg_classes, N]
        return F.log_softmax(logits, dim=1)
