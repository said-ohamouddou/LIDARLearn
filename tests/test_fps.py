"""utils.fps: the farthest point sampling shared by loaders and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from utils.fps import cuda_fps_available, fps, fps_indices, resample


def _cloud(n=2000, d=3, seed=0):
    return np.random.default_rng(seed).random((n, d)).astype(np.float32)


def test_first_pick_is_point_zero_and_picks_are_distinct():
    idx = fps_indices(_cloud(), 256, device="cpu")
    assert idx[0] == 0
    assert len(set(idx.tolist())) == 256


def test_each_pick_is_the_farthest_from_those_before():
    pts = _cloud(300).astype(np.float64)
    idx = fps_indices(pts, 20, device="cpu")
    for i in range(1, len(idx)):
        d = ((pts[:, None] - pts[idx[:i]][None]) ** 2).sum(-1).min(1)
        assert d[idx[i]] == pytest.approx(d.max(), rel=1e-5)


def test_prefix_of_a_larger_sample_is_the_smaller_sample():
    pts = _cloud()
    assert np.array_equal(fps(pts, 512, device="cpu")[:128], fps(pts, 128, device="cpu"))


def test_extra_columns_travel_with_their_point():
    pts = _cloud(500, d=6)
    idx = fps_indices(pts, 64, device="cpu")
    assert np.array_equal(fps(pts, 64, device="cpu"), pts[idx])
    assert np.array_equal(idx, fps_indices(pts[:, :3], 64, device="cpu"))


def test_point_at_the_origin_can_be_selected():
    # pointnet2_ops on its own never selects points with |p|^2 <= 1e-3.
    pts = np.array([[5, 5, 5], [0, 0, 0], [5, 5, 4.9], [4.9, 5, 5]], np.float32)
    assert fps_indices(pts, 2, device="cpu").tolist() == [0, 1]
    if cuda_fps_available():
        assert fps_indices(pts, 2, device="cuda").tolist() == [0, 1]


def test_large_coordinates_keep_their_precision():
    # UTM-sized offsets would collapse nearby points in float32 without the shift.
    pts = _cloud(1000).astype(np.float64) + np.array([5.0e5, 5.0e6, 300.0])
    assert np.array_equal(fps_indices(pts, 100, device="cpu"),
                          fps_indices(pts - pts.min(0), 100, device="cpu"))


def test_matches_open3d_selection():
    o3d = pytest.importorskip("open3d")
    pts = _cloud(3000).astype(np.float64)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    ref = np.asarray(pc.farthest_point_down_sample(256).points)
    ours = fps(pts, 256, device="cpu")
    assert {tuple(p) for p in ours} == {tuple(p) for p in ref}


@pytest.mark.skipif(not cuda_fps_available(), reason="needs CUDA and pointnet2_ops")
def test_gpu_and_cpu_select_the_same_points():
    pts = _cloud(10000, seed=1)
    assert np.array_equal(fps_indices(pts, 1024, device="cpu"),
                          fps_indices(pts, 1024, device="cuda"))


def test_resample_pads_cyclically_and_keeps_exact_size_clouds():
    pts = _cloud(5)
    out = resample(pts, 12)
    assert np.array_equal(out, np.concatenate([pts, pts, pts[:2]]))
    assert resample(pts, 5) is pts
    assert resample(_cloud(50), 10, device="cpu").shape == (10, 3)


@pytest.mark.parametrize("bad", [0, 6])
def test_rejects_impossible_sample_sizes(bad):
    with pytest.raises(ValueError):
        fps_indices(_cloud(5), bad)


def test_rejects_unknown_device():
    with pytest.raises(ValueError):
        fps_indices(_cloud(5), 2, device="gpu")
