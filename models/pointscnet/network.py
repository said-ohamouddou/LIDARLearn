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
from typing import Optional, Dict, Any

from .curve_sampling import HilbertSampler
from .feature_layers import MultiScaleLocalEncoder, LocalRegionEncoder
from .attention import DualPathAttention, CrossPointInteraction

from ..build import MODELS
from ..base_model import BasePointCloudModel


class ClassificationHead(nn.Module):
    """
    Final classification layers with dropout regularization.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple = (512, 256),
        dropout_rates: tuple = (0.4, 0.4)
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@MODELS.register_module()
class PointSCNet(BasePointCloudModel):
    """
    Spatial Curve Point Cloud Classifier.

    Architecture:
    1. Multi-scale local feature extraction at sampled anchors
    2. Cross-point interaction using curve-sampled references
    3. Dual-path attention refinement
    4. Global aggregation and classification

    Inherits the library's ``BasePointCloudModel`` so it gets
    ``get_loss_acc`` and related helpers used by the training runners.
    """

    def __init__(self, config, num_classes: int = 40, **kwargs):
        """
        Args:
            config: EasyDict config passed by ``build_model_from_cfg``.
                Recognised keys:
                  - num_classes (int)
                  - use_normals (bool, default False)
                  - dropout1, dropout2 (float, default 0.4)
                  - first_stage_anchors (int, default 256)
                  - interaction_refs (int, default 64)
                  - input_channels (int, default 3)
            num_classes: Fallback class count when the config doesn't declare
                one. Kept for backwards compatibility with kwargs callers.
        """
        getter = config.get if hasattr(config, 'get') else \
            (lambda k, d=None: getattr(config, k, d))
        num_classes = int(getter('num_classes', num_classes))
        use_normals = bool(getter('use_normals', False))
        drop_rate_1 = float(getter('dropout1', 0.4))
        drop_rate_2 = float(getter('dropout2', 0.4))
        first_stage_anchors = int(getter('first_stage_anchors', 256))
        interaction_refs = int(getter('interaction_refs', 64))
        input_channels = int(getter('input_channels', 3))

        super().__init__(config, num_classes=num_classes)

        self.use_normals = use_normals
        self.interaction_refs = interaction_refs

        # Additional features beyond xyz
        extra_features = 3 if use_normals else 0

        # Stage 1: Multi-scale local encoding
        # Different radii capture different neighborhood sizes
        self.stage1_encoder = MultiScaleLocalEncoder(
            num_anchors=first_stage_anchors,
            radii=[0.1, 0.4],
            neighbors_list=[16, 128],
            input_features=extra_features,
            mlp_configs=[
                [32, 32, 64],
                [64, 96, 128]
            ]
        )

        # Feature dimension after stage 1: 64 + 128 = 192
        stage1_out_dim = 64 + 128

        # Curve-based sampler for reference points
        self.curve_sampler = HilbertSampler(precision=10)

        # Cross-point interaction module
        self.cross_interaction = CrossPointInteraction(
            coord_dim=3,
            feature_dim=stage1_out_dim,
            num_references=interaction_refs
        )

        # Feature dimension after interaction: stage1_out_dim + 3
        interaction_out_dim = stage1_out_dim + 3

        # Dual attention refinement
        self.attention_module = DualPathAttention(
            num_channels=interaction_out_dim,
            num_points=first_stage_anchors,
            channel_reduction=2
        )

        # Stage 2: Global aggregation
        self.stage2_encoder = LocalRegionEncoder(
            num_anchors=1,
            search_radius=float('inf'),
            neighbors_per_anchor=first_stage_anchors,
            input_features=interaction_out_dim,
            mlp_channels=[256, 512, 1024],
            aggregate_all=True
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=1024,
            num_classes=num_classes,
            hidden_dims=(512, 256),
            dropout_rates=(drop_rate_1, drop_rate_2)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for point cloud classification.

        Args:
            points: [B, N, C] or [B, C, N] point cloud
                   C=3 for xyz, C=6 for xyz+normals

        Returns:
            [B, num_classes] classification logits
        """
        # Handle both input formats
        if points.dim() == 3 and points.shape[1] > points.shape[2]:
            points = points.transpose(1, 2)

        batch_size = points.shape[0]

        # Split coordinates and features
        if self.use_normals and points.shape[1] >= 6:
            coords = points[:, :3, :].transpose(1, 2)  # [B, N, 3]
            normals = points[:, 3:6, :].transpose(1, 2)  # [B, N, 3]
        else:
            coords = points[:, :3, :].transpose(1, 2)  # [B, N, 3]
            normals = None

        # Stage 1: Multi-scale local features
        anchor_coords, local_feats = self.stage1_encoder(coords, normals)
        # anchor_coords: [B, M, 3], local_feats: [B, C1, M]

        # Sample reference points using space-filling curve
        ref_indices = self.curve_sampler(
            anchor_coords,
            self.interaction_refs
        )  # [B, K]

        # Cross-point interaction
        anchor_coords_t = anchor_coords.transpose(1, 2)  # [B, 3, M]
        enhanced_feats = self.cross_interaction(
            anchor_coords_t,
            local_feats,
            ref_indices
        )  # [B, C1+3, M]

        # Attention refinement
        refined_feats = self.attention_module(enhanced_feats)

        # Stage 2: Global aggregation
        refined_coords = anchor_coords
        refined_feats_t = refined_feats.transpose(1, 2)  # [B, M, C]

        _, global_feats = self.stage2_encoder(refined_coords, refined_feats_t)
        # global_feats: [B, 1024, 1]

        # Flatten and classify
        global_vec = global_feats.view(batch_size, -1)
        logits = self.classifier(global_vec)

        return logits


def build_scpcloud_classifier(config: Optional[Dict[str, Any]] = None) -> PointSCNet:
    """
    Factory function to build SCPCloud classifier from a plain dict.

    Wraps the dict in an ``EasyDict``-like object so ``PointSCNet.__init__``
    (which expects a config as its first positional arg) can read fields
    via ``config.get(...)``.

    Args:
        config: Configuration dictionary with optional keys:
            - num_classes, use_normals, dropout1, dropout2,
              first_stage_anchors, interaction_refs, input_channels

    Returns:
        Configured PointSCNet instance
    """
    from easydict import EasyDict
    return PointSCNet(EasyDict(config or {}))
