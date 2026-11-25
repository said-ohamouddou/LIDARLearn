import os
import sys
import glob
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    # Store h5 file in the same directory as DATA_PATH
    h5_path = os.path.join(data_path, "point_cloud_data.h5")

    try:
        logger.info("=" * 80)
        logger.info(f"Starting data loading from: {folder_path}")
        logger.info(f"Target points per cloud: {num_points}")
        logger.info("=" * 80)

        # Discover classes
        logger.info("Discovering class directories...")
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

        # Process each class with progress bar
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

            # Process files in this class with progress bar
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

        # Convert to numpy arrays
        logger.info("Converting to numpy arrays...")
        point_clouds = np.array(point_clouds)
        labels = np.array(labels)

        # Ensure labels are 1D for proper handling
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


def data_split(h5_path, data_path, k_folds, random_state=42):
    """Split data for k-fold cross-validation (CV-only mode).

    Args:
        h5_path: Path to the input h5 file with point cloud data
        data_path: Directory where the split h5 file should be stored
        k_folds: Number of folds for cross-validation (must be > 1)
        random_state: Random seed for reproducibility (default: 42)

    Cross-Validation Mode:
        - Uses 100% of data for k-fold splits (no held-out test set)
        - Each fold rotates as validation set
        - Stores fold indices in H5 structure:
            data_split.h5
            ├── all_data/          # All data (100%)
            ├── fold_0/
            │   ├── train_indices
            │   └── val_indices
            ├── fold_1/
            │   ├── train_indices
            │   └── val_indices
            └── ...
    """
    if k_folds <= 1:
        raise ValueError(f"TreeSpeciesDatasetCV requires k_folds > 1 for cross-validation. Got k_folds={k_folds}. Use TreeSpeciesDataset for simple train/test split.")
    try:
        logger.info("=" * 80)
        logger.info(f"Starting data splitting")
        logger.info(f"K-Fold Cross-Validation: {k_folds} folds (using 100% of data)")
        print(f"\n{'=' * 80}")
        print(f"🔄 Splitting data with {k_folds}-fold CV (100% data used)")
        print(f"{'=' * 80}")

        # Load data
        logger.info(f"Loading data from: {h5_path}")
        print(f"📂 Loading data from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            point_clouds = f['point_clouds'][:]
            labels = f['labels'][:]
            classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]

        total_samples = len(point_clouds)
        logger.info(f"Total samples: {total_samples}")
        print(f"  Total samples: {total_samples}")

        # Ensure labels are 1D for stratification
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

        # Save split data to H5 file in the DATA_PATH directory (cross-validation)
        split_h5_path = os.path.join(data_path, 'data_split_cv.h5')
        logger.info(f"Saving split data to: {split_h5_path}")
        print(f"\n💾 Saving split data to {split_h5_path}...")

        with h5py.File(split_h5_path, 'w') as f:
            # Store class names as a dataset
            f.create_dataset('classes', data=np.array(classes, dtype='S'))

            # Store metadata
            f.attrs['k_folds'] = k_folds
            f.attrs['random_state'] = random_state

            # K-fold cross-validation mode - use 100% of data
            logger.info(f"Creating {k_folds}-fold cross-validation splits (100% data)...")
            print(f"\n🔀 Creating {k_folds}-fold cross-validation splits...")
            print(f"   Using 100% of data ({total_samples} samples)")

            # Store all data for fold-based access
            all_data_group = f.create_group('all_data')
            all_data_group.create_dataset('point_clouds', data=point_clouds)
            all_data_group.create_dataset('labels', data=labels_flat)

            # Create stratified k-fold splits
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

            # Create folds with progress bar
            for fold_idx, (train_indices, val_indices) in enumerate(tqdm(
                skf.split(point_clouds, labels_flat),
                total=k_folds,
                desc="Creating folds",
                unit="fold"
            )):
                fold_group = f.create_group(f'fold_{fold_idx}')
                fold_group.create_dataset('train_indices', data=train_indices)
                fold_group.create_dataset('val_indices', data=val_indices)

                # Log fold information
                train_fold_labels = labels_flat[train_indices]
                val_fold_labels = labels_flat[val_indices]

                train_pct = (len(train_indices) / total_samples) * 100
                val_pct = (len(val_indices) / total_samples) * 100

                logger.info(f"  Fold {fold_idx}: Train={len(train_indices)} ({train_pct:.1f}%), Val={len(val_indices)} ({val_pct:.1f}%)")
                print(f"  ✓ Fold {fold_idx}: Train={len(train_indices)} ({train_pct:.1f}%), Val={len(val_indices)} ({val_pct:.1f}%)")

                # Verify stratification per fold
                train_unique, train_counts = np.unique(train_fold_labels, return_counts=True)
                val_unique, val_counts = np.unique(val_fold_labels, return_counts=True)

                logger.info(f"    Class distribution for Fold {fold_idx}:")
                for label in unique_labels:
                    class_name = classes[int(label)] if int(label) < len(classes) else f"Class_{label}"
                    train_count = train_counts[train_unique == label][0] if label in train_unique else 0
                    val_count = val_counts[val_unique == label][0] if label in val_unique else 0
                    train_pct_class = (train_count / len(train_indices)) * 100 if len(train_indices) > 0 else 0
                    val_pct_class = (val_count / len(val_indices)) * 100 if len(val_indices) > 0 else 0
                    logger.info(f"      {class_name}: Train={train_count} ({train_pct_class:.1f}%), Val={val_count} ({val_pct_class:.1f}%)")

            print(f"\n✓ Created {k_folds} cross-validation folds (100% data used)")

        file_size_mb = os.path.getsize(split_h5_path) / (1024 * 1024)
        logger.info(f"Data splitting completed successfully. Split file size: {file_size_mb:.2f} MB")
        print(f"✓ Split data saved successfully ({file_size_mb:.2f} MB)")
        logger.info("=" * 80)
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Error in data_split: {str(e)}")
        raise


@DATASETS.register_module()
class TreeSpeciesDatasetCV(Dataset):
    def __init__(self, config):
        self.num_points = config.N_POINTS
        self.partition = config.subset

        # Get DATA_PATH from config
        data_path = config.DATA_PATH

        # Get cross-validation parameters from config
        # K_FOLDS can be in config directly (merged from others)
        k_folds = config.K_FOLDS
        # fold can be in config directly (merged from others)
        fold = config.fold
        # Get random seed for reproducibility (default: 42)
        self.seed = config.seed

        # Enforce CV-only mode: k_folds must be > 1 and fold must be specified
        if k_folds is None or k_folds <= 1:
            raise ValueError(
                f"TreeSpeciesDatasetCV requires K_FOLDS > 1 for cross-validation. "
                f"Got K_FOLDS={k_folds}. Use TreeSpeciesDataset for simple train/test split."
            )

        if fold is None:
            raise ValueError(
                f"TreeSpeciesDatasetCV requires 'fold' parameter to be specified. "
                f"Got fold={fold}. Set fold in config or use --fold argument."
            )

        # Debug logging for CV parameters
        logger.info(f"CV Parameters: k_folds={k_folds}, fold={fold}, partition={self.partition}, seed={self.seed}")

        try:
            # Path to the split h5 file in the DATA_PATH directory (cross-validation)
            split_h5_path = os.path.join(data_path, 'data_split_cv.h5')

            if not os.path.exists(split_h5_path):
                logger.info(f"Split file not found. Creating CV data split...")
                print(f"\n⚠️  Split file not found at {split_h5_path}")
                print("Creating CV data split (this may take a while)...")
                h5_path = load_data(self.num_points, data_path=data_path)
                data_split(h5_path, data_path=data_path, k_folds=k_folds, random_state=self.seed)
            else:
                # Check if existing split has the right structure for CV and matching seed
                with h5py.File(split_h5_path, 'r') as f:
                    existing_k_folds = f.attrs.get('k_folds', None)
                    existing_seed = f.attrs.get('random_state', None)
                    has_all_data = 'all_data' in f
                    has_fold_0 = 'fold_0' in f

                # Recreate if file doesn't have proper CV structure, wrong number of folds, or different seed
                if existing_k_folds != k_folds or existing_seed != self.seed or not has_all_data or not has_fold_0:
                    if existing_seed != self.seed:
                        logger.info(f"Existing split has seed={existing_seed}, but seed={self.seed} requested. Recreating split...")
                        print(f"\n⚠️  Existing split has seed={existing_seed}, but seed={self.seed} requested.")
                    else:
                        logger.info(f"Existing split has {existing_k_folds} folds, but {k_folds} requested. Recreating split...")
                        print(f"\n⚠️  Existing split has {existing_k_folds} folds, but {k_folds} requested.")
                    print("Recreating data split with correct parameters...")
                    h5_path = os.path.join(data_path, "point_cloud_data.h5")
                    if not os.path.exists(h5_path):
                        h5_path = load_data(self.num_points, data_path=data_path)
                    data_split(h5_path, data_path=data_path, k_folds=k_folds, random_state=self.seed)
                else:
                    logger.info(f"Loading existing split from: {split_h5_path}")
                    print(f"✓ Using existing split file: {split_h5_path}")

            # Load data for CV mode
            with h5py.File(split_h5_path, 'r') as f:
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
                file_k_folds = f.attrs.get('k_folds', None)

                # Verify CV structure exists
                if file_k_folds is None or file_k_folds <= 1 or 'all_data' not in f:
                    raise ValueError(
                        f"Split file at {split_h5_path} does not have proper CV structure. "
                        f"Expected k_folds > 1 and 'all_data' group. Got k_folds={file_k_folds}. "
                        f"Please delete the file and let it be recreated."
                    )

                # K-fold cross-validation mode (100% data in folds, no separate test set)
                logger.info(f"Loading fold {fold} for {self.partition} partition (CV mode - 100% data)")

                # Print clear CV fold information message
                if self.partition == 'train':
                    # Generate list of training folds (all except current fold)
                    train_folds = [str(i) for i in range(file_k_folds) if i != fold]
                    train_folds_str = ', '.join(train_folds)

                    print("\n" + "=" * 70)
                    print("║" + " " * 68 + "║")
                    print("║" + f"  CROSS-VALIDATION FOLD CONFIGURATION".center(68) + "║")
                    print("║" + " " * 68 + "║")
                    print("║" + "-" * 68 + "║")
                    print("║" + f"  Validation: Fold {fold}".ljust(68) + "║")
                    print("║" + f"  Training:   Folds {train_folds_str}".ljust(68) + "║")
                    print("║" + " " * 68 + "║")
                    print("║" + f"  Total Folds: {file_k_folds}".ljust(68) + "║")
                    print("║" + " " * 68 + "║")
                    print("=" * 70 + "\n")

                    logger.info(f"CV Configuration: Validation=Fold {fold}, Training=Folds {train_folds_str}")

                # Load all data and apply fold indices
                all_point_clouds = f['all_data']['point_clouds'][:]
                all_labels = f['all_data']['labels'][:]

                fold_group = f[f'fold_{fold}']
                train_indices = fold_group['train_indices'][:]
                val_indices = fold_group['val_indices'][:]

                if self.partition == 'train':
                    self.data = all_point_clouds[train_indices]
                    self.label = all_labels[train_indices]
                    print(f"✓ Loaded {len(self.data)} train samples from fold {fold} ({len(self.classes)} classes)")
                elif self.partition in ['val', 'test']:
                    # In CV mode, both 'val' and 'test' use the validation fold
                    self.data = all_point_clouds[val_indices]
                    self.label = all_labels[val_indices]
                    print(f"✓ Loaded {len(self.data)} val samples from fold {fold} ({len(self.classes)} classes)")
                else:
                    raise ValueError(f"Unknown partition '{self.partition}' for CV mode. Use 'train' or 'val'.")

                logger.info(f"CV mode: Fold {fold}, Partition={self.partition}, Samples={len(self.data)}")

            # Ensure labels are properly shaped
            if self.label.ndim > 1:
                self.label = self.label.flatten()

            logger.info(f"Dataset loaded: {len(self.data)} samples in '{self.partition}' partition")
            logger.info(f"Number of classes: {len(self.classes)}")
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __getitem__(self, item):
        try:
            pointcloud = self.data[item][:self.num_points].copy()  # Make a copy to avoid modifying original data
            label = self.label[item]

            # Extract label value robustly (handle both scalar and array cases)
            if isinstance(label, (np.ndarray, list, tuple)):
                label_value = int(label[0] if len(label) > 0 else label)
            else:
                label_value = int(label)
            # Always normalize the point cloud (this is not augmentation)
            pointcloud = normalize_pc(pointcloud)

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
        train_cfg = SimpleNamespace(N_POINTS=2048, subset='train', DATA_PATH='data/STPCTLS', K_FOLDS=5, fold=0)
        val_cfg = SimpleNamespace(N_POINTS=2048, subset='val', DATA_PATH='data/STPCTLS', K_FOLDS=5, fold=0)
        train_dataset = TreeSpeciesDatasetCV(train_cfg)
        val_dataset = TreeSpeciesDatasetCV(val_cfg)
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
