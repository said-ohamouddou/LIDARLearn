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
from typing import List, Tuple, Optional

from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    ball_query,
    grouping_operation,
    three_nn,
    three_interpolate,
)


class SharedMLP(nn.Module):
    """
    Shared MLP applied point-wise with batch normalization.

    Processes features independently for each point while sharing
    weights across all points.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Conv2d(prev_dim, dim, kernel_size=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, npoint, nsample] input features
        Returns:
            [B, C', npoint, nsample] transformed features
        """
        return self.net(x)


class SharedMLP1D(nn.Module):
    """1D version of SharedMLP for sequence data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        use_bn: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Conv1d(prev_dim, dim, kernel_size=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalRegionEncoder(nn.Module):
    """
    Encode local regions around sampled anchor points.

    For each anchor point, gathers neighbors within a radius,
    computes relative coordinates, and applies shared MLP
    followed by max pooling to get a local descriptor.
    """

    def __init__(
        self,
        num_anchors: int,
        search_radius: float,
        neighbors_per_anchor: int,
        input_features: int,
        mlp_channels: List[int],
        aggregate_all: bool = False
    ):
        """
        Args:
            num_anchors: Number of anchor points to sample
            search_radius: Radius for neighbor search
            neighbors_per_anchor: Max neighbors per anchor
            input_features: Input feature dimension (excluding xyz)
            mlp_channels: Output channels for each MLP layer
            aggregate_all: If True, use all points as single group
        """
        super().__init__()

        self.num_anchors = num_anchors
        self.search_radius = search_radius
        self.neighbors_per_anchor = neighbors_per_anchor
        self.aggregate_all = aggregate_all

        # +3 for relative xyz coordinates
        self.encoder = SharedMLP(input_features + 3, mlp_channels)

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, N, 3] point coordinates
            features: [B, N, D] point features (optional)

        Returns:
            anchor_coords: [B, M, 3] anchor point coordinates
            anchor_features: [B, C, M] aggregated features at anchors
        """
        batch_size, num_points, _ = coords.shape
        coords = coords.contiguous()
        coords_t = coords.transpose(1, 2).contiguous()  # [B, 3, N]

        if self.aggregate_all:
            # Global aggregation: single anchor at centroid
            anchor_coords = torch.zeros(batch_size, 1, 3, device=coords.device)
            centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
            relative_xyz = (coords - centroid).transpose(1, 2).unsqueeze(2)  # [B, 3, 1, N]

            if features is not None:
                feats_t = features.transpose(1, 2).contiguous().unsqueeze(2)  # [B, D, 1, N]
                local_input = torch.cat([relative_xyz, feats_t], dim=1)  # [B, 3+D, 1, N]
            else:
                local_input = relative_xyz
        else:
            # Sample anchor points with FPS
            anchor_idx = furthest_point_sample(coords, self.num_anchors)  # [B, M]
            anchor_coords_t = gather_operation(coords_t, anchor_idx)  # [B, 3, M]
            anchor_coords = anchor_coords_t.transpose(1, 2).contiguous()  # [B, M, 3]

            # Ball query for neighbors
            neighbor_idx = ball_query(
                self.search_radius, self.neighbors_per_anchor,
                coords, anchor_coords
            )  # [B, M, K]

            # Group coordinates and compute relative positions
            grouped_xyz = grouping_operation(coords_t, neighbor_idx)  # [B, 3, M, K]
            grouped_xyz = grouped_xyz - anchor_coords_t.unsqueeze(-1)

            if features is not None:
                feats_t = features.transpose(1, 2).contiguous()  # [B, D, N]
                grouped_feats = grouping_operation(feats_t, neighbor_idx)  # [B, D, M, K]
                local_input = torch.cat([grouped_xyz, grouped_feats], dim=1)  # [B, 3+D, M, K]
            else:
                local_input = grouped_xyz  # [B, 3, M, K]

        # Apply shared MLP: [B, C, npoint, nsample]
        encoded = self.encoder(local_input)

        # Max pool over neighbors (last dim = nsample)
        aggregated = encoded.max(dim=3)[0]  # [B, C', M]

        return anchor_coords, aggregated


class MultiScaleLocalEncoder(nn.Module):
    """
    Extract features at multiple scales around anchor points.

    Uses different radii and neighbor counts to capture
    both fine and coarse local structure.
    """

    def __init__(
        self,
        num_anchors: int,
        radii: List[float],
        neighbors_list: List[int],
        input_features: int,
        mlp_configs: List[List[int]]
    ):
        """
        Args:
            num_anchors: Number of anchor points
            radii: List of search radii for each scale
            neighbors_list: List of neighbor counts for each scale
            input_features: Input feature dimension
            mlp_configs: MLP channel configs for each scale
        """
        super().__init__()

        assert len(radii) == len(neighbors_list) == len(mlp_configs)

        self.num_anchors = num_anchors
        self.radii = radii
        self.neighbors_list = neighbors_list

        self.scale_encoders = nn.ModuleList()
        for mlp_channels in mlp_configs:
            encoder = SharedMLP(input_features + 3, mlp_channels)
            self.scale_encoders.append(encoder)

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, N, 3] point coordinates
            features: [B, N, D] point features

        Returns:
            anchor_coords: [B, M, 3] anchor coordinates
            multi_scale_features: [B, C_total, M] concatenated features
        """
        coords = coords.contiguous()
        coords_t = coords.transpose(1, 2).contiguous()  # [B, 3, N]

        # Sample anchors once with FPS
        anchor_idx = furthest_point_sample(coords, self.num_anchors)  # [B, M]
        anchor_coords_t = gather_operation(coords_t, anchor_idx)  # [B, 3, M]
        anchor_coords = anchor_coords_t.transpose(1, 2).contiguous()  # [B, M, 3]

        feats_t = features.transpose(1, 2).contiguous() if features is not None else None

        scale_outputs = []

        for radius, num_neighbors, encoder in zip(
            self.radii, self.neighbors_list, self.scale_encoders
        ):
            # Ball query at this scale
            neighbor_idx = ball_query(radius, num_neighbors, coords, anchor_coords)  # [B, M, K]

            # Group and compute relative positions
            grouped_xyz = grouping_operation(coords_t, neighbor_idx)  # [B, 3, M, K]
            grouped_xyz = grouped_xyz - anchor_coords_t.unsqueeze(-1)

            if feats_t is not None:
                grouped_feats = grouping_operation(feats_t, neighbor_idx)  # [B, D, M, K]
                local_input = torch.cat([grouped_xyz, grouped_feats], dim=1)  # [B, 3+D, M, K]
            else:
                local_input = grouped_xyz

            # Encode and pool over neighbors (last dim = nsample)
            encoded = encoder(local_input)  # [B, C', M, K]
            pooled = encoded.max(dim=3)[0]  # [B, C', M]
            scale_outputs.append(pooled)

        multi_scale = torch.cat(scale_outputs, dim=1)
        return anchor_coords, multi_scale


class FeatureUpsampler(nn.Module):
    """
    Upsample features from sparse to dense point set.

    Uses inverse distance weighting to interpolate features
    from sampled points back to original resolution.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        num_neighbors: int = 3
    ):
        """
        Args:
            input_dim: Combined input feature dimension
            output_dims: MLP output dimensions
            num_neighbors: Number of neighbors for interpolation
        """
        super().__init__()

        self.num_neighbors = num_neighbors
        self.refine_mlp = SharedMLP1D(input_dim, output_dims)

    def forward(
        self,
        dense_coords: torch.Tensor,
        sparse_coords: torch.Tensor,
        dense_features: Optional[torch.Tensor],
        sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_coords: [B, N, 3] target point coordinates
            sparse_coords: [B, M, 3] source point coordinates
            dense_features: [B, D1, N] features at target points (optional)
            sparse_features: [B, D2, M] features to upsample

        Returns:
            [B, C, N] upsampled and refined features
        """
        batch_size, num_dense, _ = dense_coords.shape
        num_sparse = sparse_coords.shape[1]

        if num_sparse == 1:
            # Single source point: broadcast to all targets
            interpolated = sparse_features.expand(-1, -1, num_dense)  # [B, D2, N]
        else:
            # Three nearest neighbours + inverse-distance weighting via pointnet2_ops
            dist, idx = three_nn(dense_coords, sparse_coords)  # [B, N, 3], [B, N, 3]
            dist_recip = 1.0 / (dist + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)  # [B, N, 3]
            interpolated = three_interpolate(sparse_features, idx, weight)  # [B, D2, N]

        if dense_features is not None:
            combined = torch.cat([dense_features, interpolated], dim=1)
        else:
            combined = interpolated

        return self.refine_mlp(combined)
