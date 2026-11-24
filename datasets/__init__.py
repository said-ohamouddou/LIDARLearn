from .build import build_dataset_from_cfg
import datasets.TreeSpeciesDataset
import datasets.TreeSpeciesDatasetCV
import datasets.ModelNetDataset
import datasets.ModelNetDatasetFewShot
import datasets.ShapeNet55Dataset
import datasets.ShapeNet55DatasetPretrain
import datasets.S3DISDataset
import datasets.ScanObjectNNDataset
# Import augmentation module
from .augmentation import (
    get_train_transforms,
    get_train_transform,
    get_augmentation,
    list_augmentations,
    AUGMENTATION_REGISTRY,
)
