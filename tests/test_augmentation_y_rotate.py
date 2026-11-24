"""
`y_rotate` must reproduce upstream PointMamba / Point-BERT `PointcloudRotate`:
per sample, angle ~ U[0, 2*pi) from numpy, points @ R_y(angle).
"""

from __future__ import annotations

import numpy as np
import torch

from datasets.augmentation import AUGMENTATION_REGISTRY, get_train_transforms


def _upstream_pointcloud_rotate(pc):
    # Verbatim logic of upstream datasets/data_transforms.py::PointcloudRotate.
    pc = pc.clone()
    for i in range(pc.size()[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval, sinval = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
        pc[i, :, :] = torch.matmul(pc[i], R)
    return pc


def test_y_rotate_is_registered():
    assert "y_rotate" in AUGMENTATION_REGISTRY


def test_y_rotate_matches_upstream_pointcloud_rotate():
    pts = torch.rand(4, 256, 3) * 2 - 1
    np.random.seed(123)
    expected = _upstream_pointcloud_rotate(pts)
    np.random.seed(123)
    got = get_train_transforms("y_rotate")(pts.clone())
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got[..., 1], pts[..., 1])  # y is the rotation axis


def test_y_rotate_accepts_channels_first():
    pts = torch.rand(2, 3, 128)
    np.random.seed(7)
    got = get_train_transforms("y_rotate")(pts.clone())
    np.random.seed(7)
    expected = _upstream_pointcloud_rotate(pts.permute(0, 2, 1)).permute(0, 2, 1)
    assert got.shape == pts.shape and torch.allclose(got, expected, atol=1e-6)
