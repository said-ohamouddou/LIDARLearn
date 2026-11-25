import numpy as np
import random
import torch
import os
from collections import abc


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    # Imported here so utils.misc (used by every runner) loads without the
    # pointnet2_ops CUDA extension; only callers of fps() need it.
    try:
        from pointnet2_ops import pointnet2_utils
    except ImportError as e:
        raise ImportError(
            "fps() needs the pointnet2_ops CUDA extension, which is not "
            "available. Build it with `bash extensions/install_extensions.sh`. "
            f"(original error: {e})"
        ) from e
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed based on PyTorch's initial seed.

    This ensures reproducibility across DataLoader workers by deriving the seed
    from PyTorch's seed (set via torch.manual_seed) rather than the potentially
    modified global NumPy RNG state.  All three RNG sources are seeded so that
    any __getitem__ using torch.rand / np.random / random is fully deterministic.
    """
    import torch
    # Use PyTorch's seed as base (set by set_random_seed) to avoid shared RNG state issues
    seed = torch.initial_seed() % 2**32
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        def lr_lbmd(e): return max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler



def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility across all random sources.

    Args:
        seed (int): Seed to be used for all random number generators.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    Note:
        - Speed-reproducibility tradeoff: https://pytorch.org/docs/stable/notes/randomness.html
        - When deterministic=True: slower, but fully reproducible
        - When deterministic=False: faster, but may have slight variations
        - For cross-validation, call this before each fold with a unique seed
    """
    # Set Python's built-in random
    random.seed(seed)

    # Set NumPy's random
    np.random.seed(seed)

    # Set PyTorch's random (CPU)
    torch.manual_seed(seed)

    # Set PyTorch's random (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set hash seed for Python (affects dict/set ordering in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # Enable deterministic mode for CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ has this for even stricter determinism. Use warn_only
        # so ops without a deterministic CUDA impl (e.g. adaptive_max_pool2d
        # backward, used by PointNet / PointNet++ / DGCNN) fall back to the
        # non-deterministic path with a warning instead of raising.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            # Older PyTorch versions don't have this
            pass
    else:
        # Allow CUDNN to find optimal algorithms (faster but less reproducible)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True
