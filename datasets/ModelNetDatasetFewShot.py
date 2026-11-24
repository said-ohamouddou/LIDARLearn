import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch
import random

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

@DATASETS.register_module()
class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')


        print_log('Load processed data from %s...' % self.pickle_path, logger = 'ModelNetFewShot')

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print_log('The size of %s data is %d' % (split, len(self.dataset)), logger = 'ModelNetFewShot')

        # Expose class list so runner_finetune / ValidationMetricsTracker
        # pick up the correct num_classes (= way) instead of falling back
        # to config.model.num_classes (default 7).
        self.classes = [f'class_{i}' for i in range(self.way)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        # Stored points are in FPS order (8192 per shape), so the first
        # N_POINTS are an FPS sample of N_POINTS, as in ModelNetDataset.
        points = points[:self.npoints].copy()
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)