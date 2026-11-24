"""
TransformerSegBase — shared segmentation scaffold for SSL backbones.

Matches the original segmentation implementations:
- PointMAE, ACT, PCP-MAE, PointGPT: fetch_idx=[3,7,11], 1152-dim concat
- ReCon: same + cls token pooling (3776-dim head input)
- Point-M2AE: hierarchical 3-stage (separate implementation)

Architecture (for 384-dim transformers):
  1. Extract features from layers [3, 7, 11] → concat → 1152 channels
  2. PointNetFeaturePropagation(1152+3 → [1536, 1024]) — upsample G→N
  3. Global: max(1152) || avg(1152) → 2304
  4. Label: Conv1d(16→64) → 64 (part seg only)
  5. Concat: propagated(1024) + global(2304) + label(64) = 3392
  6. Seg head: Conv1d(3392→512→256→C)
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_utils import three_nn, three_interpolate

from ..base_seg_model import BaseSegModel
from utils.checkpoint import load_checkpoint


class PointNetFeaturePropagation(nn.Module):
    """3-NN inverse-distance interpolation + MLP."""

    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last = in_channel
        for c in mlp:
            layers.extend([
                nn.Conv1d(last, c, 1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ])
            last = c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N, 3] dense query
        xyz2: [B, S, 3] sparse source (centers)
        points1: [B, D, N] or None
        points2: [B, D', S]
        Returns: [B, mlp[-1], N]
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interp = points2.expand(-1, -1, N)
        else:
            dist, idx = three_nn(xyz1.contiguous(), xyz2.contiguous())
            dist_recip = 1.0 / (dist + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)
            interp = three_interpolate(points2.contiguous(), idx, weight)

        if points1 is not None:
            out = torch.cat([points1, interp], dim=1)
        else:
            out = interp
        return self.mlp(out)


class TransformerSegBase(BaseSegModel):
    """Base class for SSL transformer segmentation models.

    Subclasses must implement _encode(pts_bcn) returning:
        multi_layer_feats: [B, feat_dim, G]  (concatenated multi-layer features)
        centers:           [B, G, 3]
        extra_global:      [B, extra_dim] or None (e.g., ReCon cls token)
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_classes = config.seg_classes
        self.num_obj_classes = config.num_obj_classes
        self.use_cls_label = config.use_cls_label
        self.trans_dim = 0  # set by subclass

    def _build_seg_head(self, feat_dim, extra_global_dim=0, dropout=0.5):
        """Build propagation + seg head matching original implementations.

        Args:
            feat_dim: concatenated multi-layer feature dim (e.g., 1152 for 3x384)
            extra_global_dim: extra channels in global feature (e.g., 384 for ReCon cls)
            dropout: dropout rate
        """
        # Label projection
        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64
        else:
            label_dim = 0

        # Propagation: upsample from G centers to N points
        self.propagation = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, 1024],
        )

        # Seg head: propagated(1024) + max(feat_dim) + avg(feat_dim) + extra + label
        global_dim = feat_dim * 2 + extra_global_dim
        in_dim = 1024 + global_dim + label_dim
        self.seg_head = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.seg_classes, 1),
        )

        self._feat_dim = feat_dim
        self._extra_global_dim = extra_global_dim

    def _encode(self, pts_bcn):
        """Return (multi_layer_feats [B, feat_dim, G], centers [B, G, 3], extra_global or None)."""
        raise NotImplementedError

    def forward(self, pts, cls_label=None):
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B, 3, N] or [B, N, 3], got {pts.shape}")

        B, _, N = pts_bcn.shape
        pts_bnc = pts_bcn.transpose(1, 2).contiguous()

        # 1. Encode: multi-layer features + centers
        feats_bcg, centers, extra_global = self._encode(pts_bcn)

        # 2. Global pooling
        g_max = feats_bcg.max(dim=-1)[0]   # [B, feat_dim]
        g_avg = feats_bcg.mean(dim=-1)     # [B, feat_dim]

        global_parts = [g_max, g_avg]
        if extra_global is not None:
            global_parts.append(extra_global)
        g_broadcast = torch.cat(global_parts, dim=1).unsqueeze(-1).expand(-1, -1, N)

        # 3. Propagation: G centers → N points
        prop = self.propagation(
            xyz1=pts_bnc,        # [B, N, 3]
            xyz2=centers,        # [B, G, 3]
            points1=pts_bcn,     # [B, 3, N]
            points2=feats_bcg,   # [B, feat_dim, G]
        )  # [B, 1024, N]

        # 4. Concat + optional label
        parts = [prop, g_broadcast]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))  # [B, 64, 1]
            parts.append(cls_feat.expand(-1, -1, N))

        x = torch.cat(parts, dim=1)
        logits = self.seg_head(x)
        return F.log_softmax(logits, dim=1)

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'load_model_from_ckpt'):
            return self.backbone.load_model_from_ckpt(ckpt_path)
        state = load_checkpoint(ckpt_path, map_location='cpu')
        sd = state.get('base_model', state.get('model', state))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.backbone.load_state_dict(sd, strict=strict)
