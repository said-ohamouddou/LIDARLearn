"""
ModelNet40 dataset: the shipped configs must build their datasets.

Covers the three ways the configs used to fail: the dataset name used by
cfgs/dataset/ModelNet40Dataset.yaml, the `val` split, and N_POINTS smaller
than the distributed 8192-point FPS cache.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from easydict import EasyDict

from datasets import build_dataset_from_cfg
from datasets.build import DATASETS
from utils.config import cfg_from_yaml_file

from .conftest import REPO_ROOT

N_TRAIN, N_TEST = 6, 4


def _fake_modelnet(root, cached_points=8192):
    """Official layout with only the 8192-point FPS caches (no raw .txt shapes)."""
    names = [f"class{i:02d}" for i in range(40)]
    (root / "modelnet40_shape_names.txt").write_text("\n".join(names) + "\n")
    rng = np.random.default_rng(0)
    data = {}
    for split, n in (("train", N_TRAIN), ("test", N_TEST)):
        ids = [f"class{i % 40:02d}_{i:04d}" for i in range(n)]
        (root / f"modelnet40_{split}.txt").write_text("\n".join(ids) + "\n")
        points = [rng.random((cached_points, 6), dtype=np.float32) for _ in range(n)]
        labels = [np.array([i % 40], dtype=np.int32) for i in range(n)]
        with open(root / f"modelnet40_{split}_{cached_points}pts_fps.dat", "wb") as f:
            pickle.dump([points, labels], f)
        data[split] = points
    return data


def _cfg(root, subset, n_points=1024):
    return EasyDict(NAME="ModelNet40Dataset", DATA_PATH=str(root), N_POINTS=n_points,
                    NUM_CATEGORY=40, USE_NORMALS=False, subset=subset)


def test_config_dataset_name_is_registered():
    name = cfg_from_yaml_file(str(REPO_ROOT / "cfgs/dataset/ModelNet40Dataset.yaml")).NAME
    assert name in DATASETS


def test_val_subset_uses_the_test_split(tmp_path):
    _fake_modelnet(tmp_path)
    assert len(build_dataset_from_cfg(_cfg(tmp_path, "val"))) == N_TEST


def test_fewer_points_are_taken_from_the_larger_fps_cache(tmp_path):
    """First N points of an FPS-ordered cache are an FPS sample of N points."""
    data = _fake_modelnet(tmp_path)
    ds = build_dataset_from_cfg(_cfg(tmp_path, "test", n_points=1024))
    _, _, (points, label) = ds[2]
    expected = data["test"][2][:1024, :3]
    expected = expected - expected.mean(axis=0)
    expected = expected / np.max(np.sqrt((expected ** 2).sum(axis=1)))
    assert tuple(points.shape) == (1024, 3)
    assert np.allclose(points.numpy(), expected, atol=1e-6)
    assert int(label) == 2
    assert not (tmp_path / "modelnet40_test_1024pts_fps.dat").exists()


REAL_ROOT = REPO_ROOT / "data/ModelNet/modelnet40_normal_resampled"


@pytest.mark.skipif(not (REAL_ROOT / "modelnet40_test_8192pts_fps.dat").exists(),
                    reason="ModelNet40 data not present")
def test_shipped_modelnet40_config_builds_its_datasets():
    """Builds train/val exactly like tools/runner_finetune.py does."""
    from tools import builder
    cfg = cfg_from_yaml_file(str(REPO_ROOT / "cfgs/classification/PointMamba/ModelNet40/modelnet40.yaml"))
    args = SimpleNamespace(distributed=False, num_workers=0, data_fraction=None)
    sizes = {}
    for split in ("train", "val"):
        _, loader = builder.dataset_builder(args, cfg.dataset[split])
        _, _, (points, _) = loader.dataset[0]
        assert tuple(points.shape) == (1024, 3)
        sizes[split] = len(loader.dataset)
    assert sizes == {"train": 9843, "val": 2468}
