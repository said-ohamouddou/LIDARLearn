"""
ScanObjectNN loader (used by PointMamba's ScanObjectNN configs), tested on
small generated h5 files with the official file names.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from easydict import EasyDict

from datasets import build_dataset_from_cfg

FILES = {
    "ScanObjectNN": ("training_objectdataset.h5", "test_objectdataset.h5"),
    "ScanObjectNN_hardest": ("training_objectdataset_augmentedrot_scale75.h5",
                             "test_objectdataset_augmentedrot_scale75.h5"),
}


def _write_split(root, name, n=8, points=2048):
    rng = np.random.default_rng(0)
    with h5py.File(root / name, "w") as f:
        f["data"] = rng.random((n, points, 3), dtype=np.float32)
        f["label"] = np.arange(n) % 15


@pytest.mark.parametrize("dataset_name", sorted(FILES))
@pytest.mark.parametrize("subset", ["train", "test"])
def test_scanobjectnn_loads_official_layout(tmp_path, dataset_name, subset):
    train_file, test_file = FILES[dataset_name]
    _write_split(tmp_path, train_file)
    _write_split(tmp_path, test_file)

    ds = build_dataset_from_cfg(EasyDict(NAME=dataset_name, ROOT=str(tmp_path), subset=subset))

    assert len(ds) == 8
    taxonomy, model_id, (points, label) = ds[3]
    assert taxonomy == "ScanObjectNN" and model_id == "sample"
    assert tuple(points.shape) == (2048, 3)
    assert 0 <= int(label) < 15 and int(label) == 3


def test_scanobjectnn_missing_files_name_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="training_objectdataset.h5"):
        build_dataset_from_cfg(EasyDict(NAME="ScanObjectNN", ROOT=str(tmp_path), subset="train"))


OFFICIAL_CLASSES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
                    "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]


def test_scanobjectnn_exposes_official_class_names(tmp_path):
    """Per-class metrics are sized from dataset.classes (official label order,
    hkust-vgd/scanobjectnn training_data/shape_names_ext.txt)."""
    _write_split(tmp_path, FILES["ScanObjectNN"][0])
    ds = build_dataset_from_cfg(EasyDict(NAME="ScanObjectNN", ROOT=str(tmp_path), subset="train"))
    assert ds.classes == OFFICIAL_CLASSES
