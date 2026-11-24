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
from typing import Tuple

from pointnet2_ops.pointnet2_utils import gather_operation


class ChannelGate(nn.Module):
    """
    Channel-wise attention using global statistics.

    Computes importance weights for each feature channel
    based on global average and max pooling statistics.
    """

    def __init__(self, num_channels: int, reduction: int = 2):
        """
        Args:
            num_channels: Number of input channels
            reduction: Channel reduction factor for bottleneck
        """
        super().__init__()

        mid_channels = max(num_channels // reduction, 1)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

        # Separate pathways for avg and max
        self.avg_pathway = nn.Sequential(
            nn.Linear(num_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_channels, bias=False)
        )

        self.max_pathway = nn.Sequential(
            nn.Linear(num_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_channels, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N] input features

        Returns:
            [B, C, 1] channel attention weights
        """
        batch_size, num_ch, _ = x.shape

        # Global statistics
        avg_stat = self.global_avg(x).view(batch_size, num_ch)
        max_stat = self.global_max(x).view(batch_size, num_ch)

        # Compute attention from both pathways
        avg_att = self.avg_pathway(avg_stat)
        max_att = self.max_pathway(max_stat)

        # Combine and normalize
        combined = torch.sigmoid(avg_att + max_att)

        return combined.unsqueeze(-1)


class SpatialGate(nn.Module):
    """
    Spatial attention for point-wise importance weighting.

    Learns which points are more important based on
    their feature statistics.
    """

    def __init__(self, input_channels: int, num_points: int):
        """
        Args:
            input_channels: Number of input feature channels
            num_points: Number of points (for output dimension)
        """
        super().__init__()

        mid_ch = max(input_channels // 2, 1)

        self.compress_avg = nn.Sequential(
            nn.Linear(input_channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, num_points, bias=False)
        )

        self.compress_max = nn.Sequential(
            nn.Linear(input_channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, num_points, bias=False)
        )

        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N] input features

        Returns:
            [B, 1, N] spatial attention weights
        """
        batch_size, num_ch, num_pts = x.shape

        # Pool across points to get channel statistics
        avg_stat = self.pool_avg(x).view(batch_size, num_ch)
        max_stat = self.pool_max(x).view(batch_size, num_ch)

        # Project to spatial weights
        avg_spatial = self.compress_avg(avg_stat)
        max_spatial = self.compress_max(max_stat)

        # Combine
        spatial_weights = torch.sigmoid(avg_spatial + max_spatial)

        return spatial_weights.unsqueeze(1)


class DualPathAttention(nn.Module):
    """
    Combined channel and spatial attention module.

    Applies both channel-wise and spatial attention to
    refine point cloud features.
    """

    def __init__(
        self,
        num_channels: int,
        num_points: int,
        channel_reduction: int = 2
    ):
        """
        Args:
            num_channels: Number of feature channels
            num_points: Number of points
            channel_reduction: Reduction factor for channel attention
        """
        super().__init__()

        self.channel_gate = ChannelGate(num_channels, channel_reduction)
        self.spatial_gate = SpatialGate(num_channels, num_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N] input features

        Returns:
            [B, C, N] attention-refined features
        """
        ch_weights = self.channel_gate(x)  # [B, C, 1]
        sp_weights = self.spatial_gate(x)  # [B, 1, N]

        # Apply both attention mechanisms
        refined = x * ch_weights * sp_weights

        return refined


class CrossPointInteraction(nn.Module):
    """
    Learn interactions between sampled reference points and all points.

    Uses curve-sampled reference points to capture global context
    and propagate information across the point cloud.
    """

    def __init__(
        self,
        coord_dim: int,
        feature_dim: int,
        num_references: int = 64
    ):
        """
        Args:
            coord_dim: Coordinate dimension (typically 3)
            feature_dim: Feature dimension
            num_references: Number of reference points to sample
        """
        super().__init__()

        self.num_references = num_references

        # Process coordinate pairs
        self.coord_encoder = nn.Sequential(
            nn.Conv2d(coord_dim * 2, coord_dim * 2, 1, bias=False),
            nn.BatchNorm2d(coord_dim * 2),
            nn.ReLU(inplace=True)
        )

        # Process feature pairs
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True)
        )

        # Combine coordinate and feature information
        combined_dim = coord_dim * 2 + feature_dim * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(combined_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        reference_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coords: [B, 3, N] point coordinates
            features: [B, D, N] point features
            reference_indices: [B, K] indices of reference points

        Returns:
            [B, D+3, N] enhanced features with coordinate info
        """
        batch_size, coord_ch, num_points = coords.shape
        _, feat_ch, _ = features.shape
        num_refs = reference_indices.shape[1]

        # Gather reference points using pointnet2_ops (expects int32 indices, contiguous inputs)
        ref_idx = reference_indices.int()
        ref_coords = gather_operation(coords.contiguous(), ref_idx).transpose(1, 2)   # [B, K, 3]
        ref_feats = gather_operation(features.contiguous(), ref_idx).transpose(1, 2)  # [B, K, D]

        coords_t = coords.transpose(1, 2)   # [B, N, 3]
        feats_t = features.transpose(1, 2)  # [B, N, D]

        # Build pairwise relationships: [B, N, K, C]
        # Expand all points: [B, N, 1, C] and references: [B, 1, K, C]
        all_coords_exp = coords_t.unsqueeze(2)  # [B, N, 1, 3]
        ref_coords_exp = ref_coords.unsqueeze(1)  # [B, 1, K, 3]

        all_feats_exp = feats_t.unsqueeze(2)  # [B, N, 1, D]
        ref_feats_exp = ref_feats.unsqueeze(1)  # [B, 1, K, D]

        # Concatenate pairs
        coord_pairs = torch.cat([
            all_coords_exp.expand(-1, -1, num_refs, -1),
            ref_coords_exp.expand(-1, num_points, -1, -1)
        ], dim=-1)  # [B, N, K, 6]

        feat_pairs = torch.cat([
            all_feats_exp.expand(-1, -1, num_refs, -1),
            ref_feats_exp.expand(-1, num_points, -1, -1)
        ], dim=-1)  # [B, N, K, 2D]

        # Reshape for conv2d: [B, C, K, N]
        coord_pairs = coord_pairs.permute(0, 3, 2, 1)
        feat_pairs = feat_pairs.permute(0, 3, 2, 1)

        # Encode pairs
        coord_encoded = self.coord_encoder(coord_pairs)
        feat_encoded = self.feat_encoder(feat_pairs)

        # Fuse
        combined = torch.cat([coord_encoded, feat_encoded], dim=1)
        fused = self.fusion(combined)  # [B, D, K, N]

        # Aggregate over references
        aggregated = fused.max(dim=2)[0]  # [B, D, N]

        # Project and combine with input
        output = self.output_proj(aggregated)
        enhanced = torch.cat([features + output, coords], dim=1)

        return enhanced
