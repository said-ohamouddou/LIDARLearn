"""
PointSCNet: Point Cloud Structure and Correlation Learning Based on
            Space-Filling Curve-Guided Sampling

Paper: Chen et al., Symmetry 2022 — https://doi.org/10.3390/sym14010008
Source: https://github.com/Chenguoz/PointSCNet

Clean implementation.
"""

import torch
import numpy as np
from typing import Tuple


def _quantize_coordinates(coords: np.ndarray, bits: int = 16) -> np.ndarray:
    """
    Map floating point coordinates to discrete grid positions.

    Args:
        coords: Nx3 array of normalized coordinates in [-1, 1]
        bits: Number of bits for quantization precision

    Returns:
        Nx3 array of integer grid positions
    """
    grid_size = (1 << bits) - 1
    normalized = (coords + 1.0) * 0.5  # Map to [0, 1]
    quantized = np.clip(normalized * grid_size, 0, grid_size).astype(np.uint32)
    return quantized


def _rotate_quadrant(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """Rotate/flip quadrant for Hilbert curve computation."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_distance_2d(x: int, y: int, order: int) -> int:
    """
    Compute 2D Hilbert curve distance for a point.

    Args:
        x, y: Grid coordinates
        order: Hilbert curve order (determines grid size 2^order)

    Returns:
        Distance along Hilbert curve
    """
    d = 0
    n = 1 << order
    s = n >> 1

    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rotate_quadrant(s, x, y, rx, ry)
        s >>= 1

    return d


def _compute_hilbert_index_3d(x: int, y: int, z: int, order: int) -> int:
    """
    Compute 3D Hilbert-like curve index by combining 2D projections.

    This uses a simpler approach: compute Hilbert distance on XY plane,
    then interleave with Z coordinate for 3D ordering.

    Args:
        x, y, z: Grid coordinates
        order: Curve order

    Returns:
        Combined spatial index
    """
    # Compute 2D Hilbert on XY
    h_xy = _hilbert_distance_2d(x, y, order)
    # Compute 2D Hilbert on XZ
    h_xz = _hilbert_distance_2d(x, z, order)
    # Combine indices with weighting
    combined = (h_xy << order) + h_xz + (z << (2 * order))
    return combined


def compute_spatial_ordering(points: np.ndarray, precision: int = 10) -> np.ndarray:
    """
    Compute space-filling curve indices for 3D points.

    Args:
        points: Nx3 array of 3D coordinates
        precision: Quantization precision in bits

    Returns:
        N array of curve indices for sorting
    """
    quantized = _quantize_coordinates(points, bits=precision)

    n_points = points.shape[0]
    indices = np.zeros(n_points, dtype=np.int64)

    for i in range(n_points):
        indices[i] = _compute_hilbert_index_3d(
            int(quantized[i, 0]),
            int(quantized[i, 1]),
            int(quantized[i, 2]),
            precision
        )

    return indices


class HilbertSampler:
    """
    Sample points from point cloud using Hilbert curve ordering.

    This preserves spatial locality better than random sampling,
    ensuring sampled points are well-distributed in 3D space.
    """

    def __init__(self, precision: int = 10):
        """
        Args:
            precision: Quantization precision for curve computation
        """
        self.precision = precision

    def sample_indices(
        self,
        point_cloud: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        Sample point indices using Hilbert curve ordering.

        Args:
            point_cloud: [B, N, 3] tensor of 3D coordinates
            num_samples: Number of points to sample

        Returns:
            [B, num_samples] tensor of sampled indices
        """
        device = point_cloud.device
        batch_size, num_points, _ = point_cloud.shape

        if num_samples >= num_points:
            # Return all indices if requesting more than available
            all_idx = torch.arange(num_points, device=device)
            return all_idx.unsqueeze(0).expand(batch_size, -1)

        sampled_indices = torch.zeros(
            batch_size, num_samples,
            dtype=torch.long, device=device
        )

        # Process each batch item
        points_np = point_cloud.detach().cpu().numpy()

        for b in range(batch_size):
            # Compute curve ordering
            curve_indices = compute_spatial_ordering(
                points_np[b],
                precision=self.precision
            )
            # Sort by curve index
            sorted_order = np.argsort(curve_indices)

            # Sample uniformly along the curve
            sample_positions = np.linspace(
                0, num_points - 1, num_samples, dtype=np.int64
            )
            selected = sorted_order[sample_positions]

            sampled_indices[b] = torch.from_numpy(selected)

        return sampled_indices.to(device)

    def __call__(
        self,
        point_cloud: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """Convenience method for sampling."""
        return self.sample_indices(point_cloud, num_samples)
