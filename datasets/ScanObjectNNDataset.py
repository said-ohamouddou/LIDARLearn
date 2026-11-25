"""
ScanObjectNN dataset (real-world scanned objects, 15 classes, 2048 points).

Source: Implementation adapted from https://github.com/LMD0311/PointMamba
(datasets/ScanObjectNNDataset.py), following Point-BERT / Point-MAE.
License: Apache-2.0

Variants, selected by the dataset config (see DATASET.md):
  * ScanObjectNN, ROOT .../main_split        -> OBJ-BG
  * ScanObjectNN, ROOT .../main_split_nobg   -> OBJ-ONLY
  * ScanObjectNN_hardest, ROOT .../main_split -> PB-T50-RS
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.logger import print_log
from .build import DATASETS


class _ScanObjectNNBase(Dataset):
    """Loads one split from the official h5 files; points are shuffled for training."""

    FILES = {}
    # Official label order (hkust-vgd/scanobjectnn, training_data/shape_names_ext.txt).
    classes = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
               "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        if self.subset not in self.FILES:
            raise ValueError(f"subset must be one of {sorted(self.FILES)}, got '{self.subset}'")

        path = os.path.join(self.root, self.FILES[self.subset])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ScanObjectNN file not found: {path}. "
                f"See DATASET.md for how to download and place ScanObjectNN.")
        with h5py.File(path, 'r') as h5:
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
        print_log(f'[DATASET] {type(self).__name__} {self.subset}: {self.points.shape}', logger='ScanObjectNN')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).float()
        return 'ScanObjectNN', 'sample', (current_points, self.labels[idx])

    def __len__(self):
        return self.points.shape[0]


@DATASETS.register_module()
class ScanObjectNN(_ScanObjectNNBase):
    """OBJ-BG (ROOT .../main_split) or OBJ-ONLY (ROOT .../main_split_nobg)."""

    FILES = {'train': 'training_objectdataset.h5', 'test': 'test_objectdataset.h5'}


@DATASETS.register_module()
class ScanObjectNN_hardest(_ScanObjectNNBase):
    """PB-T50-RS, the hardest perturbed variant (ROOT .../main_split)."""

    FILES = {'train': 'training_objectdataset_augmentedrot_scale75.h5',
             'test': 'test_objectdataset_augmentedrot_scale75.h5'}
