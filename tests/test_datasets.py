"""
Dataset loaders used by the shipped configs (besides ModelNet40 and
ScanObjectNN, see their own test files), tested on small generated data in
each official on-disk layout:

  ModelNetFewShot, ShapeNet (ShapeNet-55 pretraining), ShapeNetPartsH5,
  S3DIS, TreeSpeciesDataset and TreeSpeciesDatasetCV (STPCTLS).

Checks split sizes and membership, label ranges, point shapes and
normalisation, and that no sample leaks between train and evaluation splits.
"""

from __future__ import annotations

import collections
import hashlib
import pickle

import h5py
import numpy as np
import pytest
from easydict import EasyDict

from datasets import build_dataset_from_cfg

RNG = np.random.default_rng(0)


def _build(**cfg):
    return build_dataset_from_cfg(EasyDict(cfg))


def _digest(points):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(points, np.float32)).tobytes()).hexdigest()


def _assert_normalised(points, atol=1e-4):
    points = np.asarray(points)
    assert np.isfinite(points).all()
    assert np.abs(points.mean(0)).max() < 1e-3
    assert abs(np.linalg.norm(points, axis=1).max() - 1.0) < atol


# ----------------------------------------------------------- ModelNetFewShot
@pytest.fixture
def fewshot_root(tmp_path):
    split_dir = tmp_path / "5way_10shot"
    split_dir.mkdir()
    data = {
        "train": [(RNG.random((8192, 6), dtype=np.float32), c, None) for c in range(5) for _ in range(10)],
        "test": [(RNG.random((8192, 6), dtype=np.float32), c, None) for c in range(5) for _ in range(20)],
    }
    with open(split_dir / "0.pkl", "wb") as f:
        pickle.dump(data, f)
    return tmp_path


def _fewshot(root, subset):
    return _build(NAME="ModelNetFewShot", DATA_PATH=str(root), N_POINTS=1024, NUM_CATEGORY=40,
                  USE_NORMALS=False, subset=subset, way=5, shot=10, fold=0)


def test_fewshot_episode_structure(fewshot_root):
    train, test = _fewshot(fewshot_root, "train"), _fewshot(fewshot_root, "test")
    assert (len(train), len(test)) == (50, 100)
    assert collections.Counter(int(train[i][2][1]) for i in range(len(train))) == {c: 10 for c in range(5)}
    assert {int(test[i][2][1]) for i in range(len(test))} == set(range(5))
    assert train.classes == [f"class_{i}" for i in range(5)]
    _, _, (points, _) = test[0]
    assert points.shape[1] == 3
    _assert_normalised(points.numpy())


@pytest.mark.xfail(strict=True, reason="known bug: ModelNetFewShot ignores N_POINTS and returns all "
                                       "8192 stored points (configs and protocol use 1024)")
def test_fewshot_returns_n_points(fewshot_root):
    _, _, (points, _) = _fewshot(fewshot_root, "test")[0]
    assert tuple(points.shape) == (1024, 3)


# ------------------------------------------------- ShapeNet-55 (pretraining)
@pytest.fixture
def shapenet_root(tmp_path):
    data_root, pc_root = tmp_path / "ShapeNet-55", tmp_path / "shapenet_pc"
    data_root.mkdir(); pc_root.mkdir()
    for split, n in (("train", 6), ("test", 3)):
        names = [f"0269115{i}-{split}{i:03d}.npy" for i in range(n)]
        (data_root / f"{split}.txt").write_text("\n".join(names) + "\n")
        for name in names:
            np.save(pc_root / name, RNG.random((8192, 3), dtype=np.float32) * 3 + 5)
    return data_root, pc_root


@pytest.mark.parametrize("whole, expected", [(True, 9), (False, 6)])
def test_shapenet55_pretrain_split_and_samples(shapenet_root, whole, expected):
    data_root, pc_root = shapenet_root
    ds = _build(NAME="ShapeNet", DATA_PATH=str(data_root), PC_PATH=str(pc_root), N_POINTS=8192,
                subset="train", npoints=1024, whole=whole)
    assert len(ds) == expected
    taxonomy, model_id, points = ds[0]
    assert taxonomy.startswith("0269115") and model_id
    assert tuple(points.shape) == (1024, 3)
    _assert_normalised(points.numpy())


# ---------------------------------------------------------- ShapeNetPartsH5
@pytest.fixture
def shapenetparts_root(tmp_path):
    for tag, n in (("train0", 4), ("val0", 2), ("test0", 3)):
        xyz = RNG.random((n, 2048, 3), dtype=np.float32)
        xyz[:, :, 0] = np.arange(2048) / 2048.0  # unique x per point -> recover point identity
        with h5py.File(tmp_path / f"ply_data_{tag}.h5", "w") as f:
            f["data"] = xyz
            f["label"] = (np.arange(n) % 16).reshape(n, 1)
            f["pid"] = np.tile(np.arange(2048) % 50, (n, 1))
    return tmp_path


@pytest.mark.parametrize("subset, expected", [("trainval", 6), ("test", 3)])
def test_shapenetparts_h5_splits(shapenetparts_root, subset, expected):
    ds = _build(NAME="ShapeNetPartsH5", DATA_PATH=str(shapenetparts_root), N_POINTS=2048, subset=subset)
    assert len(ds) == expected and len(ds.classes) == 16


def test_shapenetparts_h5_part_labels_follow_resampled_points(shapenetparts_root):
    ds = _build(NAME="ShapeNetPartsH5", DATA_PATH=str(shapenetparts_root), N_POINTS=1024, subset="test")
    _, category, (points, cls, seg) = ds[1]
    assert points.shape == (1024, 3) and seg.shape == (1024,) and category == ds.classes[int(cls[0])]
    with h5py.File(shapenetparts_root / "ply_data_test0.h5") as f:
        raw = f["data"][1][:, :3]
    centred = raw - raw.mean(0)
    full = centred / np.linalg.norm(centred, axis=1).max()
    original_idx = np.abs(full[None, :, 0] - points[:, :1]).argmin(1)  # x is unique per point
    assert np.array_equal(seg, original_idx % 50)


# ------------------------------------------------------------------- S3DIS
@pytest.fixture
def s3dis_root(tmp_path):
    for room in ("Area_1_office_1", "Area_2_hallway_1", "Area_5_office_1"):
        n = 6000
        label = RNG.integers(0, 12, (n, 1)).astype(np.float32)
        xyz = RNG.random((n, 3), dtype=np.float32) * np.array([2.0, 2.0, 0.0], np.float32)
        xyz[:, 2] = 0.25 * label[:, 0] + 0.1  # height encodes the label -> alignment check
        rgb = RNG.integers(0, 256, (n, 3)).astype(np.float32)
        np.save(tmp_path / f"{room}.npy", np.hstack([xyz, rgb, label]))
    return tmp_path


def _label_from_height(points):
    return np.floor((np.asarray(points)[:, 2] + 0.01) / 0.25).astype(int)


def _s3dis(root, subset):
    return _build(NAME="S3DIS", DATA_PATH=str(root), N_POINTS=512, block_size=1.0, sample_rate=1.0,
                  test_area=5, feature_mode="xyz", speed_test_fraction=1.0, subset=subset)


def test_s3dis_test_area_is_held_out(s3dis_root):
    train, test = _s3dis(s3dis_root, "train"), _s3dis(s3dis_root, "test")
    assert all(not name.startswith("Area_5_") for name in train.room_names)
    assert test.room_names == ["Area_5_office_1"]
    for ds in (train, test):
        _, room, (points, seg) = ds[0]
        assert np.asarray(points).shape == (512, 3) and np.asarray(seg).shape == (512,)
        assert 0 <= int(np.min(seg)) and int(np.max(seg)) < 13
        assert np.array_equal(np.asarray(seg), _label_from_height(points))  # labels stay with their points
        assert room in ds.room_names


# ---------------------------------------------------- STPCTLS (tree species)
SPECIES = [f"species{i}" for i in range(7)]


@pytest.fixture
def stpctls_root(tmp_path):
    for species in SPECIES:
        (tmp_path / species).mkdir()
        for k in range(10):
            np.savetxt(tmp_path / species / f"tree_{k}.xyz", RNG.random((600, 3)) * 10, fmt="%.4f")
    return tmp_path


def test_stpctls_stratified_split_without_leakage(stpctls_root):
    common = dict(NAME="TreeSpeciesDataset", DATA_PATH=str(stpctls_root), N_POINTS=256, NUM_CATEGORY=7,
                  USE_NORMALS=False, noise_std=0.0)
    train, val = _build(subset="train", **common), _build(subset="val", **common)
    assert (len(train), len(val)) == (56, 14)
    assert train.classes == SPECIES
    assert collections.Counter(train.label.tolist()) == {c: 8 for c in range(7)}
    assert collections.Counter(val.label.tolist()) == {c: 2 for c in range(7)}
    assert not {_digest(p) for p in train.data} & {_digest(p) for p in val.data}
    assert train.data.shape[1:] == (256, 3)


def test_stpctls_kfold_validates_every_tree_exactly_once(stpctls_root):
    common = dict(NAME="TreeSpeciesDatasetCV", DATA_PATH=str(stpctls_root), N_POINTS=256, NUM_CATEGORY=7,
                  USE_NORMALS=False, K_FOLDS=5, seed=42)
    seen = collections.Counter()
    for fold in range(5):
        train = _build(subset="train", fold=fold, **common)
        val = _build(subset="val", fold=fold, **common)
        assert (len(train), len(val)) == (56, 14)
        assert collections.Counter(val.label.tolist()) == {c: 2 for c in range(7)}
        train_ids = {_digest(p) for p in train.data}
        val_ids = {_digest(p) for p in val.data}
        assert not train_ids & val_ids
        seen.update(val_ids)
    assert len(seen) == 70 and set(seen.values()) == {1}
