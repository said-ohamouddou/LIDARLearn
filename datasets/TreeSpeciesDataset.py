import os
import sys
import glob
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from tqdm import tqdm
from .build import DATASETS
from utils.fps import resample
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_point_cloud(file_path):
    """Load point cloud data from various formats (xyz, pts, txt).

    Tolerates an optional header row (e.g. ``x y z`` or ``x y z R G B``)
    by retrying with ``skiprows=1`` when the first parse fails.
    """
    file_extension = Path(file_path).suffix.lower()
    if file_extension not in ('.xyz', '.pts', '.txt'):
        raise ValueError(f"Unsupported file format: {file_extension}")
    try:
        try:
            points = np.loadtxt(file_path)
        except ValueError:
            points = np.loadtxt(file_path, skiprows=1)
        return points[:, :3]
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def point_selection(point_cloud_path, target_point_count):
    """Load a point cloud and resample it to ``target_point_count`` points.

    Larger clouds are downsampled by farthest point sampling, in selection
    order; smaller ones are repeated cyclically (see ``utils.fps.resample``).
    """
    try:
        points = load_point_cloud(point_cloud_path)
        if len(points) < target_point_count:
            logger.warning(f"Point cloud {point_cloud_path} has fewer points ({len(points)}) than target ({target_point_count})")
        return resample(points, target_point_count)
    except Exception as e:
        logger.error(f"Error in point selection for {point_cloud_path}: {str(e)}")
        raise


def load_data(num_points, data_path):
    """Load and process point cloud data, saving to H5 format.

    Args:
        num_points: Number of points to sample from each point cloud
        data_path: Path to the data directory containing class subdirectories.
    """
    folder_path = data_path
    h5_path = os.path.join(data_path, "point_cloud_data.h5")

    try:
        logger.info("=" * 80)
        logger.info(f"Starting data loading from: {folder_path}")
        logger.info(f"Target points per cloud: {num_points}")
        logger.info("=" * 80)

        classes = sorted([d for d in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, d))])

        if not classes:
            raise ValueError(f"No class directories found in {folder_path}")

        logger.info(f"Found {len(classes)} classes: {', '.join(classes)}")
        print(f"\nFound {len(classes)} classes: {', '.join(classes)}\n")

        point_clouds = []
        labels = []
        total_files = 0
        skipped_files = 0

        for class_idx, class_name in enumerate(tqdm(classes, desc="Processing classes", unit="class")):
            class_path = os.path.join(folder_path, class_name)
            files = [f for f in os.listdir(class_path)
                     if f.endswith(('.xyz', '.pts', '.txt'))]

            if not files:
                logger.warning(f"No valid files found in class {class_name}")
                print(f"⚠️  Warning: No valid files in class '{class_name}'")
                continue

            total_files += len(files)
            class_samples = 0

            for file_name in tqdm(files, desc=f"  Class '{class_name}'", leave=False, unit="file"):
                try:
                    file_path = os.path.join(class_path, file_name)
                    points = point_selection(file_path, num_points)
                    point_clouds.append(points)
                    labels.append(np.array([class_idx]))
                    class_samples += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    skipped_files += 1
                    continue

            logger.info(f"Class '{class_name}': {class_samples}/{len(files)} files processed successfully")
            print(f"  ✓ Class '{class_name}': {class_samples}/{len(files)} files loaded")

        if not point_clouds:
            raise ValueError("No valid point cloud data was loaded")

        logger.info(f"\nTotal files processed: {total_files - skipped_files}/{total_files}")
        logger.info(f"Successfully loaded: {len(point_clouds)} point clouds")
        logger.info(f"Skipped: {skipped_files} files")
        print(f"\n✓ Total loaded: {len(point_clouds)} point clouds ({total_files - skipped_files}/{total_files} files)")

        point_clouds = np.array(point_clouds)
        labels = np.array(labels)

        if labels.ndim > 1:
            labels = labels.flatten()

        logger.info(f"Point clouds shape: {point_clouds.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Label range: {labels.min()} to {labels.max()}")
        print(f"  Point clouds: {point_clouds.shape}")
        print(f"  Labels: {labels.shape}, range: [{labels.min()}, {labels.max()}]")

        # Save to H5 file
        logger.info(f"Saving to H5 file: {h5_path}")
        print(f"\n💾 Saving to {h5_path}...")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('point_clouds', data=point_clouds)
            f.create_dataset('labels', data=labels)
            # Store classes as a dataset instead of attributes
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

        file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
        logger.info(f"Data loading completed successfully. H5 file size: {file_size_mb:.2f} MB")
        print(f"✓ Data saved successfully ({file_size_mb:.2f} MB)")
        logger.info("=" * 80)
        print("=" * 80)

        return h5_path
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        raise


def data_split(h5_path, data_path, test_ratio=0.2, random_state=42):
    """Split data into train and test sets.

    Args:
        h5_path: Path to the input h5 file with point cloud data
        data_path: Directory where the split h5 file should be stored
        test_ratio: Ratio of test data (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Starting data splitting")
        logger.info(f"Test ratio: {test_ratio:.1%} ({test_ratio * 100:.1f}%)")
        logger.info(f"Train ratio: {1 - test_ratio:.1%} ({(1 - test_ratio) * 100:.1f}%)")
        print(f"\n{'='*80}")
        print(f"🔄 Splitting data (train: {(1 - test_ratio) * 100:.1f}%, test: {test_ratio * 100:.1f}%)")
        print(f"{'='*80}")

        logger.info(f"Loading data from: {h5_path}")
        print(f"📂 Loading data from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            point_clouds = f['point_clouds'][:]
            labels = f['labels'][:]
            classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

        total_samples = len(point_clouds)
        logger.info(f"Total samples: {total_samples}")
        print(f"  Total samples: {total_samples}")

        if labels.ndim > 1:
            labels_flat = labels.flatten()
            logger.info(f"Flattened labels from shape {labels.shape} to {labels_flat.shape}")
        else:
            labels_flat = labels

        # Check class distribution before split
        unique_labels, counts = np.unique(labels_flat, return_counts=True)
        logger.info(f"Class distribution:")
        print(f"\n📊 Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = classes[int(label)] if int(label) < len(classes) else f"Class_{label}"
            pct = (count / total_samples) * 100
            logger.info(f"  {class_name}: {count} samples ({pct:.2f}%)")
            print(f"  {class_name}: {count} samples ({pct:.2f}%)")

        logger.info("Performing stratified train/test split...")
        print(f"\n✂️  Performing stratified split...")

        # Split data into train and test sets
        train_point_clouds, test_point_clouds, train_labels, test_labels = train_test_split(
            point_clouds, labels_flat, test_size=test_ratio, stratify=labels_flat, random_state=random_state
        )

        n_train = len(train_point_clouds)
        n_test = len(test_point_clouds)

        # Save split data to H5 file in the DATA_PATH directory
        split_h5_path = os.path.join(data_path, 'data_split_simple.h5')
        logger.info(f"Saving split data to: {split_h5_path}")
        print(f"\n💾 Saving split data to {split_h5_path}...")

        with h5py.File(split_h5_path, 'w') as f:
            # Store class names as a dataset
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

            # Store metadata
            f.attrs['test_ratio'] = test_ratio
            f.attrs['random_state'] = random_state

            # Train data
            train_group = f.create_group('train')
            train_group.create_dataset('point_clouds', data=train_point_clouds)
            train_group.create_dataset('labels', data=train_labels)

            # Validation data
            val_group = f.create_group('val')
            val_group.create_dataset('point_clouds', data=test_point_clouds)
            val_group.create_dataset('labels', data=test_labels)

        logger.info(f"Simple split: Train={n_train}, Val={n_test}")
        print(f"✓ Split completed:")
        print(f"  Train: {n_train} samples ({(n_train/total_samples)*100:.2f}%)")
        print(f"  Val:   {n_test} samples ({(n_test/total_samples)*100:.2f}%)")

        # Verify stratification
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)

        logger.info("Verifying stratification:")
        print(f"\n🔍 Verifying stratification:")
        for label in unique_labels:
            class_name = classes[int(label)] if int(label) < len(classes) else f"Class_{label}"
            train_count = train_counts[train_unique == label][0] if label in train_unique else 0
            test_count = test_counts[test_unique == label][0] if label in test_unique else 0
            train_pct = (train_count / n_train) * 100 if n_train > 0 else 0
            test_pct = (test_count / n_test) * 100 if n_test > 0 else 0
            logger.info(f"  {class_name}: Train={train_count} ({train_pct:.2f}%), Val={test_count} ({test_pct:.2f}%)")
            print(f"  {class_name}: Train={train_count} ({train_pct:.2f}%), Val={test_count} ({test_pct:.2f}%)")

        file_size_mb = os.path.getsize(split_h5_path) / (1024 * 1024)
        logger.info(f"Data splitting completed successfully. Split file size: {file_size_mb:.2f} MB")
        print(f"✓ Split data saved successfully ({file_size_mb:.2f} MB)")
        logger.info("=" * 80)
        print("=" * 80 + "\n")
    except Exception as e:
        logger.error(f"Error in data_split: {str(e)}")
        raise


@DATASETS.register_module()
class TreeSpeciesDataset(Dataset):
    def __init__(self, config):
        self.num_points = config.N_POINTS
        self.partition = config.subset

        # Get DATA_PATH from config
        data_path = config.DATA_PATH

        # Get random seed for reproducibility (default: 42)
        self.seed = getattr(config, 'seed', 42)

        # ------------------------------------------------------------------
        # Gaussian-noise difficulty knob (applied to BOTH train and val).
        #
        #   noise_std : float, default 0.0
        #       Standard deviation of the isotropic Gaussian noise added to
        #       every xyz coordinate AFTER unit-sphere normalisation. Points
        #       live in roughly [-1, 1]³ at that stage, so:
        #         * 0.00 → no noise (original dataset)
        #         * 0.01 → barely perceptible
        #         * 0.05 → mild corruption
        #         * 0.10 → strong corruption
        #         * 0.20 → heavy — baseline models start collapsing here
        #         * 0.50 → signal almost destroyed
        #
        #   noise_deterministic_val : bool, default True
        #       When True, the val-split noise is drawn from a seed derived
        #       from (base seed, sample index), so every epoch sees the
        #       SAME noisy val cloud for a given sample → val metrics are
        #       comparable across epochs. Set False to re-draw per epoch.
        #
        #   noise_apply_to : 'both' | 'train' | 'val' | 'none', default 'both'
        #       Coarse control over which split receives noise. Useful for
        #       noise-robustness studies (train clean, eval on noisy).
        #
        # All three fields are optional — legacy configs without them load
        # exactly as before (noise_std=0).
        # ------------------------------------------------------------------
        self.noise_std = float(getattr(config, 'noise_std', 0.0))
        self.noise_deterministic_val = bool(
            getattr(config, 'noise_deterministic_val', True)
        )
        apply_to = str(getattr(config, 'noise_apply_to', 'both')).lower()
        if apply_to not in ('both', 'train', 'val', 'none'):
            raise ValueError(
                f"noise_apply_to must be one of 'both'/'train'/'val'/'none', "
                f"got {apply_to!r}"
            )
        # Resolve whether THIS dataset instance should actually add noise.
        if apply_to == 'none' or self.noise_std <= 0.0:
            self._add_noise = False
        elif apply_to == 'both':
            self._add_noise = True
        elif apply_to == 'train':
            self._add_noise = self.partition == 'train'
        else:  # apply_to == 'val'
            self._add_noise = self.partition == 'val'

        try:
            # Path to the split h5 file in the DATA_PATH directory (simple train-val split)
            split_h5_path = os.path.join(data_path, 'data_split_simple.h5')

            if not os.path.exists(split_h5_path):
                logger.info(f"Split file not found. Creating data split...")
                print(f"\n⚠️  Split file not found at {split_h5_path}")
                print("Creating data split (this may take a while)...")
                h5_path = load_data(self.num_points, data_path=data_path)
                data_split(h5_path, data_path=data_path, random_state=self.seed)
            else:
                # Check if existing split has matching seed
                with h5py.File(split_h5_path, 'r') as f:
                    existing_seed = f.attrs.get('random_state', None)

                # Recreate if seed mismatch
                if existing_seed != self.seed:
                    logger.info(f"Existing split has seed={existing_seed}, but seed={self.seed} requested. Recreating split...")
                    print(f"\n⚠️  Existing split has seed={existing_seed}, but seed={self.seed} requested.")
                    print("Recreating data split with correct seed...")
                    h5_path = os.path.join(data_path, "point_cloud_data.h5")
                    if not os.path.exists(h5_path):
                        h5_path = load_data(self.num_points, data_path=data_path)
                    data_split(h5_path, data_path=data_path, random_state=self.seed)
                else:
                    logger.info(f"Loading existing split from: {split_h5_path}")
                    print(f"✓ Using existing split file: {split_h5_path}")

            with h5py.File(split_h5_path, 'r') as f:
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

                if self.partition not in f:
                    raise ValueError(f"Partition '{self.partition}' not found in split file. Available: {list(f.keys())}")

                self.data = f[self.partition]['point_clouds'][:]
                self.label = f[self.partition]['labels'][:]
                print(f"✓ Loaded {len(self.data)} samples from '{self.partition}' partition ({len(self.classes)} classes)")

            # Ensure labels are properly shaped
            if self.label.ndim > 1:
                self.label = self.label.flatten()

            logger.info(f"Dataset loaded: {len(self.data)} samples in '{self.partition}' partition")
            logger.info(f"Number of classes: {len(self.classes)}")
            if self._add_noise:
                det = "deterministic" if (
                    self.partition == 'val' and self.noise_deterministic_val
                ) else "stochastic"
                logger.info(
                    f"[TreeSpeciesDataset] Gaussian-noise difficulty knob "
                    f"active on '{self.partition}': std={self.noise_std} ({det})"
                )
                print(
                    f"✓ Gaussian noise enabled on '{self.partition}' split: "
                    f"std={self.noise_std} ({det})"
                )
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def _sample_noise(self, item: int, shape: tuple) -> np.ndarray:
        """Draw Gaussian noise of the given shape.

        For val with ``noise_deterministic_val=True`` we seed a local RNG
        from ``(base seed, item)`` so the same sample gets the same
        corruption every epoch — makes val metrics comparable across
        checkpoints. For train (or val with the flag off), we use the
        global numpy RNG for fresh noise each iteration.
        """
        if self.partition == 'val' and self.noise_deterministic_val:
            rng = np.random.default_rng(self.seed * 1_000_003 + int(item))
            return rng.normal(loc=0.0, scale=self.noise_std, size=shape).astype(np.float32)
        return np.random.normal(loc=0.0, scale=self.noise_std, size=shape).astype(np.float32)

    def __getitem__(self, item):
        try:
            pointcloud = self.data[item][:self.num_points].copy()
            label = self.label[item]

            # Extract label value robustly (handle both scalar and array cases)
            if isinstance(label, (np.ndarray, list, tuple)):
                label_value = int(label[0] if len(label) > 0 else label)
            else:
                label_value = int(label)

            # Always normalize the point cloud (this is not augmentation)
            pointcloud = normalize_pc(pointcloud)

            # Optional Gaussian-noise difficulty knob. Applied AFTER
            # normalisation so ``noise_std`` is in the same unit-sphere
            # coordinate system regardless of original scene scale.
            if self._add_noise:
                pointcloud = pointcloud + self._sample_noise(item, pointcloud.shape)

            return 'TreeSpecies', 'sample', (pointcloud.astype(np.float32), label_value)
        except Exception as e:
            logger.error(f"Error getting item {item}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)


def normalize_pc(points):
    """Normalize point cloud to unit sphere (this is not augmentation)."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points


def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and display class distribution in a dataset."""
    class_counts = {}
    total_samples = len(dataset)

    # Count samples per class
    for i in range(len(dataset)):
        label = dataset.label[i].item()
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Display distribution
    logger.info(f"\n{title}:")
    logger.info("-" * 50)
    logger.info(f"{'Class':<20} {'Count':<10} {'Percentage':<10}")
    logger.info("-" * 50)

    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"{class_name:<20} {count:<10} {percentage:>6.2f}%")

    logger.info("-" * 50)
    logger.info(f"Total samples: {total_samples}\n")

    return class_counts


if __name__ == '__main__':
    try:
        train_cfg = SimpleNamespace(N_POINTS=2048, subset='train', DATA_PATH='data/STPCTLS')
        val_cfg = SimpleNamespace(N_POINTS=2048, subset='val', DATA_PATH='data/STPCTLS')
        train_dataset = TreeSpeciesDataset(train_cfg)
        val_dataset = TreeSpeciesDataset(val_cfg)
        print(train_dataset.classes)
        # Analyze class distributions
        train_dist = analyze_class_distribution(train_dataset, "Training Set Distribution")
        test_dist = analyze_class_distribution(val_dataset, "Validation Set Distribution")

        # Compare ratios between train and test
        logger.info("Train/Test Ratio Comparison:")
        logger.info("-" * 50)
        logger.info(f"{'Class':<20} {'Train %':<10} {'Test %':<10} {'Ratio':<10}")
        logger.info("-" * 50)

        total_train = len(train_dataset)
        total_test = len(val_dataset)

        for class_name in train_dist.keys():
            train_pct = (train_dist[class_name] / total_train) * 100
            test_pct = (test_dist[class_name] / total_test) * 100
            ratio = train_pct / test_pct if test_pct > 0 else float('inf')

            logger.info(f"{class_name:<20} {train_pct:>6.2f}%    {test_pct:>6.2f}%    {ratio:>6.2f}")

        logger.info("-" * 50)

        # Verify data loading
        _, _, (sample_data, sample_label) = train_dataset[0]
        logger.info(f"\nSample point cloud shape: {sample_data.shape}")
        logger.info(f"Sample label: {sample_label}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
