"""GPU-side point cloud augmentation transforms."""

import math

import numpy as np
import torch
from typing import List, Optional, Union


class PointCloudAugmentation:
    """Base class for point cloud augmentations."""

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to point cloud.

        Args:
            points: Point cloud tensor of shape (B, C, N) or (B, N, C)
                    where B=batch, C=channels (usually 3 for xyz), N=num_points

        Returns:
            Augmented point cloud with same shape as input
        """
        raise NotImplementedError


class NoAugmentation(PointCloudAugmentation):
    """Identity transform - no augmentation applied."""

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return points

    def __repr__(self) -> str:
        return "NoAugmentation()"


class RandomRotate(PointCloudAugmentation):
    """
    Random 3D rotation augmentation.
    Rotates point cloud randomly around all three axes.
    """

    def __init__(self, rotation_range: float = math.pi):
        """
        Args:
            rotation_range: Maximum rotation angle in radians (default: pi)
        """
        self.rotation_range = rotation_range

    def __repr__(self) -> str:
        return f"RandomRotate(range={self.rotation_range:.2f}rad)"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, C, N) or (B, N, C) point cloud
        """
        device = points.device
        batch_size = points.shape[0]

        # Determine if points are (B, C, N) or (B, N, C)
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3

        if is_channels_first:
            # Convert (B, C, N) to (B, N, C) for rotation
            points = points.permute(0, 2, 1)

        result = points.clone()

        for i in range(batch_size):
            # Random rotation angles
            angles = (torch.rand(3, device=device) * 2 - 1) * self.rotation_range

            # Rotation matrices for each axis
            cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
            cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
            cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

            # Rotation matrix around X axis
            Rx = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], device=device, dtype=points.dtype)

            # Rotation matrix around Y axis
            Ry = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], device=device, dtype=points.dtype)

            # Rotation matrix around Z axis
            Rz = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], device=device, dtype=points.dtype)

            # Combined rotation matrix
            R = torch.mm(Rz, torch.mm(Ry, Rx))

            # Apply rotation: (N, 3) @ (3, 3).T = (N, 3)
            result[i] = torch.mm(result[i, :, :3], R.T)

        if is_channels_first:
            result = result.permute(0, 2, 1)

        return result


class RandomYRotate(PointCloudAugmentation):
    """
    Random rotation about the y axis by an angle drawn from U[0, 2*pi).

    Same as ``PointcloudRotate`` in Point-BERT / Point-MAE / PointMamba (numpy
    RNG, points multiplied on the right by R_y). PointMamba uses it for
    ShapeNet-55 pretraining and ScanObjectNN fine-tuning.
    """

    def __repr__(self) -> str:
        return "RandomYRotate(axis=y, range=2pi)"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, C, N) or (B, N, C) point cloud; the first 3 channels are xyz
        """
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3
        if is_channels_first:
            points = points.permute(0, 2, 1)
        result = points.clone()
        for i in range(result.shape[0]):
            angle = np.random.uniform() * 2 * np.pi
            cosval, sinval = np.cos(angle), np.sin(angle)
            rotation = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], dtype=np.float32)
            R = torch.from_numpy(rotation).to(device=result.device, dtype=result.dtype)
            result[i, :, :3] = result[i, :, :3] @ R
        if is_channels_first:
            result = result.permute(0, 2, 1).contiguous()
        return result


class RandomZRotateTree(PointCloudAugmentation):
    """
    Tree-specific rotation around vertical (Z) axis.
    Trees grow upward, so rotation around Z-axis preserves natural orientation.
    """

    def __repr__(self) -> str:
        return "RandomZRotateTree()"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, C, N) point cloud where C=3 (xyz)
        """
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Determine if points are (B, C, N) or (B, N, C)
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3

        result = points.clone()

        for i in range(batch_size):
            # Random rotation angle around Z axis
            theta = torch.rand(1, device=device) * 2 * math.pi

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # Rotation matrix around Z axis
            rot_matrix = torch.tensor([
                [cos_theta.item(), -sin_theta.item(), 0],
                [sin_theta.item(), cos_theta.item(), 0],
                [0, 0, 1]
            ], device=device, dtype=dtype)

            if is_channels_first:
                # (C, N) format: rot_matrix @ points
                result[i, :3, :] = torch.matmul(rot_matrix, result[i, :3, :])
            else:
                # (N, C) format: points @ rot_matrix.T
                result[i, :, :3] = torch.matmul(result[i, :, :3], rot_matrix.T)

        return result


class RandomScale(PointCloudAugmentation):
    """
    Random uniform scaling augmentation.
    Scales the point cloud uniformly in all directions.
    """

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2):
        """
        Args:
            scale_min: Minimum scale factor
            scale_max: Maximum scale factor
        """
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __repr__(self) -> str:
        return f"RandomScale(min={self.scale_min}, max={self.scale_max})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        device = points.device

        # Random scale factor for each sample in batch
        scales = torch.rand(batch_size, device=device) * (self.scale_max - self.scale_min) + self.scale_min

        # Determine shape and apply scaling
        if points.shape[1] == 3 and points.shape[2] != 3:
            # (B, C, N) format
            scales = scales.view(batch_size, 1, 1)
        else:
            # (B, N, C) format
            scales = scales.view(batch_size, 1, 1)

        return points * scales


class RandomTranslate(PointCloudAugmentation):
    """
    Random translation augmentation.
    Translates the point cloud by a random offset.
    """

    def __init__(self, translate_range: float = 0.2):
        """
        Args:
            translate_range: Maximum translation distance in each axis
        """
        self.translate_range = translate_range

    def __repr__(self) -> str:
        return f"RandomTranslate(range={self.translate_range})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Random translation for each sample
        translations = (torch.rand(batch_size, 3, device=device, dtype=dtype) * 2 - 1) * self.translate_range

        result = points.clone()

        # Determine if points are (B, C, N) or (B, N, C)
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3

        if is_channels_first:
            # (B, C, N) format
            for i in range(batch_size):
                result[i, :3, :] = result[i, :3, :] + translations[i].view(3, 1)
        else:
            # (B, N, C) format
            for i in range(batch_size):
                result[i, :, :3] = result[i, :, :3] + translations[i].view(1, 3)

        return result


class RandomScaleTranslate(PointCloudAugmentation):
    """
    Combined random scaling and translation augmentation.
    First scales, then translates the point cloud.
    """

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2, translate_range: float = 0.2):
        """
        Args:
            scale_min: Minimum scale factor
            scale_max: Maximum scale factor
            translate_range: Maximum translation distance in each axis
        """
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.translate_range = translate_range
        self.scale = RandomScale(scale_min, scale_max)
        self.translate = RandomTranslate(translate_range)

    def __repr__(self) -> str:
        return f"RandomScaleTranslate(scale=[{self.scale_min}, {self.scale_max}], translate={self.translate_range})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        points = self.scale(points)
        points = self.translate(points)
        return points


class RandomJitter(PointCloudAugmentation):
    """
    Random jitter (noise) augmentation.
    Adds Gaussian noise to point coordinates.
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            clip: Maximum absolute value of noise (clips to [-clip, clip])
        """
        self.sigma = sigma
        self.clip = clip

    def __repr__(self) -> str:
        return f"RandomJitter(sigma={self.sigma}, clip={self.clip})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        dtype = points.dtype

        # Generate Gaussian noise
        noise = torch.randn_like(points) * self.sigma

        # Clip noise to specified range
        noise = torch.clamp(noise, -self.clip, self.clip)

        return points + noise


class RandomDropout(PointCloudAugmentation):
    """
    Random point dropout augmentation.
    Randomly removes points from the point cloud by setting them to zero or
    duplicating other points.
    """

    def __init__(self, dropout_ratio: float = 0.1, duplicate: bool = True):
        """
        Args:
            dropout_ratio: Fraction of points to drop (0 to 1)
            duplicate: If True, replace dropped points with duplicates of remaining points
                      If False, set dropped points to zero
        """
        self.dropout_ratio = dropout_ratio
        self.duplicate = duplicate

    def __repr__(self) -> str:
        return f"RandomDropout(ratio={self.dropout_ratio}, duplicate={self.duplicate})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        batch_size = points.shape[0]

        # Determine if points are (B, C, N) or (B, N, C)
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3

        if is_channels_first:
            num_points = points.shape[2]
        else:
            num_points = points.shape[1]

        num_drop = int(num_points * self.dropout_ratio)

        if num_drop == 0:
            return points

        result = points.clone()

        for i in range(batch_size):
            # Random indices to drop
            drop_indices = torch.randperm(num_points, device=device)[:num_drop]

            if self.duplicate:
                # Get indices to keep
                keep_indices = torch.randperm(num_points, device=device)[num_drop:]
                # Randomly select replacement indices from kept points
                replace_indices = keep_indices[torch.randint(len(keep_indices), (num_drop,), device=device)]

                if is_channels_first:
                    result[i, :, drop_indices] = result[i, :, replace_indices]
                else:
                    result[i, drop_indices, :] = result[i, replace_indices, :]
            else:
                if is_channels_first:
                    result[i, :, drop_indices] = 0
                else:
                    result[i, drop_indices, :] = 0

        return result


class RandomFlip(PointCloudAugmentation):
    """
    Random axis flipping augmentation.
    Randomly flips the point cloud along specified axes.
    """

    def __init__(self, flip_x: bool = True, flip_y: bool = True, flip_z: bool = False, prob: float = 0.5):
        """
        Args:
            flip_x: Whether to potentially flip along X axis
            flip_y: Whether to potentially flip along Y axis
            flip_z: Whether to potentially flip along Z axis (usually False for trees)
            prob: Probability of flipping each axis
        """
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.prob = prob

    def __repr__(self) -> str:
        axes = []
        if self.flip_x:
            axes.append('x')
        if self.flip_y:
            axes.append('y')
        if self.flip_z:
            axes.append('z')
        return f"RandomFlip(axes=[{','.join(axes)}], prob={self.prob})"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        batch_size = points.shape[0]

        # Determine if points are (B, C, N) or (B, N, C)
        is_channels_first = points.shape[1] == 3 and points.shape[2] != 3

        result = points.clone()

        for i in range(batch_size):
            # Random flip decisions
            flip_mask = torch.rand(3, device=device) < self.prob

            if self.flip_x and flip_mask[0]:
                if is_channels_first:
                    result[i, 0, :] = -result[i, 0, :]
                else:
                    result[i, :, 0] = -result[i, :, 0]

            if self.flip_y and flip_mask[1]:
                if is_channels_first:
                    result[i, 1, :] = -result[i, 1, :]
                else:
                    result[i, :, 1] = -result[i, :, 1]

            if self.flip_z and flip_mask[2]:
                if is_channels_first:
                    result[i, 2, :] = -result[i, 2, :]
                else:
                    result[i, :, 2] = -result[i, :, 2]

        return result


class Compose(PointCloudAugmentation):
    """
    Composes several augmentations together.
    """

    def __init__(self, transforms: List[PointCloudAugmentation]):
        """
        Args:
            transforms: List of augmentation transforms to compose
        """
        self.transforms = transforms

    def __repr__(self) -> str:
        transform_names = [repr(t) for t in self.transforms]
        return f"Compose([{', '.join(transform_names)}])"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            points = t(points)
        return points


class RandomChoice(PointCloudAugmentation):
    """
    Randomly selects one augmentation from a list to apply.
    """

    def __init__(self, transforms: List[PointCloudAugmentation]):
        """
        Args:
            transforms: List of augmentation transforms to choose from
        """
        self.transforms = transforms

    def __repr__(self) -> str:
        transform_names = [repr(t) for t in self.transforms]
        return f"RandomChoice([{', '.join(transform_names)}])"

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(len(self.transforms), (1,)).item()
        return self.transforms[idx](points)


# Factory Functions

AUGMENTATION_REGISTRY = {
    'none': NoAugmentation,
    'rotate': RandomRotate,
    'scale_translate': RandomScaleTranslate,
    'jitter': RandomJitter,
    'scale': RandomScale,
    'translate': RandomTranslate,
    'dropout': RandomDropout,
    'flip': RandomFlip,
    'z_rotate_tree': RandomZRotateTree,
    'y_rotate': RandomYRotate,
}


def get_augmentation(name: str, **kwargs) -> PointCloudAugmentation:
    """
    Get an augmentation by name.

    Args:
        name: Name of the augmentation (see AUGMENTATION_REGISTRY)
        **kwargs: Additional arguments to pass to the augmentation constructor

    Returns:
        PointCloudAugmentation instance

    Raises:
        ValueError: If augmentation name is not recognized
    """
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unknown augmentation: {name}. "
            f"Available: {list(AUGMENTATION_REGISTRY.keys())}"
        )
    return AUGMENTATION_REGISTRY[name](**kwargs)


def get_train_transforms(config_or_name: Union[str, list, object] = 'none', **kwargs) -> PointCloudAugmentation:
    """
    Get training transforms based on config or augmentation name.

    Args:
        config_or_name: Can be one of:
            - str: Single augmentation name (e.g., 'none', 'z_rotate_tree', 'jitter')
            - list: List of augmentation names to compose
            - config object: Configuration with augmentation settings
        **kwargs: Additional parameters passed to augmentation constructors

    Returns:
        PointCloudAugmentation instance (Compose if multiple augmentations)

    Examples:
        # Single augmentation by name
        transform = get_train_transforms('z_rotate_tree')

        # Multiple augmentations
        transform = get_train_transforms(['z_rotate_tree', 'jitter', 'scale'])

        # With config object
        transform = get_train_transforms(config)

    Example YAML config:
        augmentation:
          type: ['z_rotate_tree', 'jitter', 'scale']
          params:
            jitter:
              sigma: 0.01
              clip: 0.05
            scale:
              scale_min: 0.9
              scale_max: 1.1
    """
    # Handle string input (single augmentation name)
    if isinstance(config_or_name, str):
        if config_or_name == 'none' or config_or_name == '':
            return NoAugmentation()
        return get_augmentation(config_or_name, **kwargs)

    # Handle list input (multiple augmentation names)
    if isinstance(config_or_name, (list, tuple)):
        if len(config_or_name) == 0 or config_or_name == ['none']:
            return NoAugmentation()

        transforms = []
        for name in config_or_name:
            if name == 'none':
                continue
            transforms.append(get_augmentation(name, **kwargs))

        if len(transforms) == 0:
            return NoAugmentation()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            return Compose(transforms)

    # Handle config object input
    config = config_or_name

    # Check if augmentation is configured
    if not hasattr(config, 'augmentation'):
        return NoAugmentation()

    aug_config = config.augmentation

    # Get augmentation type(s)
    aug_type = aug_config.type if hasattr(aug_config, 'type') else 'none'

    # Handle single augmentation name from config
    if isinstance(aug_type, str):
        if aug_type == 'none':
            return NoAugmentation()

        # Get parameters for this augmentation
        params = {}
        if hasattr(aug_config, 'params'):
            aug_params = aug_config.params
            if hasattr(aug_params, aug_type):
                params = dict(aug_params[aug_type])
            elif isinstance(aug_params, dict) and aug_type in aug_params:
                params = dict(aug_params[aug_type])

        return get_augmentation(aug_type, **params)

    # Handle list of augmentations from config
    if isinstance(aug_type, (list, tuple)):
        if len(aug_type) == 0 or aug_type == ['none']:
            return NoAugmentation()

        transforms = []
        for name in aug_type:
            if name == 'none':
                continue

            # Get parameters for this augmentation
            params = {}
            if hasattr(aug_config, 'params'):
                aug_params = aug_config.params
                if hasattr(aug_params, name):
                    params = dict(aug_params[name])
                elif isinstance(aug_params, dict) and name in aug_params:
                    params = dict(aug_params[name])

            transforms.append(get_augmentation(name, **params))

        if len(transforms) == 0:
            return NoAugmentation()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            return Compose(transforms)

    return NoAugmentation()


# Alias for backwards compatibility
get_train_transform = get_train_transforms


# Utility Functions

def list_augmentations() -> List[str]:
    """Return list of available augmentation names."""
    return list(AUGMENTATION_REGISTRY.keys())


__all__ = [
    'PointCloudAugmentation',
    'NoAugmentation',
    'RandomRotate',
    'RandomZRotateTree',
    'RandomScale',
    'RandomTranslate',
    'RandomScaleTranslate',
    'RandomJitter',
    'RandomDropout',
    'RandomFlip',
    'Compose',
    'RandomChoice',
    'get_augmentation',
    'get_train_transforms',
    'get_train_transform',
    'list_augmentations',
    'AUGMENTATION_REGISTRY',
]
