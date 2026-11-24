"""ShapeNet Parts dataset loader for part segmentation."""

import glob
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from .build import DATASETS


# ShapeNet Parts: 16 object categories, 50 total part classes
SHAPENET_SEG_CLASSES = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35],
    'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
    'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
    'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
    'Chair': [12, 13, 14, 15], 'Knife': [22, 23],
}

SHAPENET_SEG_LABEL_TO_CAT = {
    lab: cat for cat, labs in SHAPENET_SEG_CLASSES.items() for lab in labs
}


def _pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


# Canonical object-label ordering used by the H5 release
# (verified against ply_data_*.h5 obj-label → pid-range mapping).
_H5_CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
    'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table',
]


@DATASETS.register_module()
class ShapeNetPartsH5(Dataset):
    """ShapeNet Parts dataset loaded from the canonical HDF5 release.

    Expects the official HDF5 layout under DATA_PATH:
        <DATA_PATH>/
            ply_data_train0.h5 ... ply_data_train{K}.h5
            ply_data_val0.h5   ... ply_data_val{K}.h5
            ply_data_test0.h5  ... ply_data_test{K}.h5

    Each H5 file contains:
        data:  [M, 2048, 3] float32 — normalized xyz point clouds
        label: [M, 1] uint8         — object category id (0..15)
        pid:   [M, 2048] uint8      — per-point part label (0..49)

    The object-label ordering matches sorted category names (see
    _H5_CATEGORY_NAMES), consistent with ShapeNet Parts convention.

    The 'subset' field accepts:
        'train', 'val', 'trainval' (train + val), 'test'.
    """

    seg_classes = SHAPENET_SEG_CLASSES
    seg_label_to_cat = SHAPENET_SEG_LABEL_TO_CAT
    num_seg_classes = 50
    num_obj_classes = 16

    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = getattr(config, 'N_POINTS', 2048)
        self.partition = config.subset
        self.normal_channel = getattr(config, 'normal_channel', False)
        if self.normal_channel:
            # The standard ply_data_*.h5 release has no normals.
            raise ValueError("ShapeNetPartsH5 does not provide normals.")

        self.category_names = list(_H5_CATEGORY_NAMES)
        self.classes = self.category_names  # runner-friendly alias

        # Resolve split → list of H5 files
        if self.partition == 'trainval':
            split_tags = ['train', 'val']
        else:
            split_tags = [self.partition]
        files = []
        for tag in split_tags:
            pattern = os.path.join(self.root, f'ply_data_{tag}*.h5')
            files.extend(sorted(glob.glob(pattern)))
        if not files:
            raise FileNotFoundError(
                f"No H5 files matching ply_data_{split_tags}*.h5 under {self.root}"
            )

        data_list, label_list, pid_list = [], [], []
        for fp in files:
            with h5py.File(fp, 'r') as f:
                data_list.append(f['data'][:].astype(np.float32))
                label_list.append(f['label'][:].astype(np.int64).squeeze(-1))
                pid_list.append(f['pid'][:].astype(np.int64))
        self.data = np.concatenate(data_list, axis=0)     # [M, 2048, 3]
        self.label = np.concatenate(label_list, axis=0)   # [M]
        self.pid = np.concatenate(pid_list, axis=0)       # [M, 2048]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        point_set = self.data[index].copy()
        seg = self.pid[index].copy().astype(np.int32)
        cls = np.array([self.label[index]], dtype=np.int32)

        # Normalize (already normalized in the release, but redo for safety)
        point_set[:, :3] = _pc_normalize(point_set[:, :3])

        # Resample to requested npoints
        n_avail = point_set.shape[0]
        if self.npoints == n_avail:
            choice = np.arange(n_avail)
        else:
            choice = np.random.choice(n_avail, self.npoints, replace=(self.npoints > n_avail))
        point_set = point_set[choice, :]
        seg = seg[choice]

        return 'ShapeNetPartsH5', self.category_names[int(cls[0])], (point_set, cls, seg)
