
"""
Shared fixtures and helpers for the LIDARLearn test suite.

Two things to keep in mind when adding tests here:

* Tests must be runnable from the repo root via ``pytest tests/`` without any
  real dataset on disk. Everything is driven by synthetic tensors.
* ``conftest.py`` is auto-loaded by pytest — anything at module level also
  runs as import side-effects, so keep it cheap (no dataset loading).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Make the repo root importable so tests can do `from tools import ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure all model/dataset registries are populated.
# These imports trigger @MODELS.register_module() side effects.
import models   # noqa: F401,E402
import datasets  # noqa: F401,E402


# --------------------------------------------------------------------------
# CI without CUDA extensions
# --------------------------------------------------------------------------
# Import names of the CUDA extensions built by extensions/install_extensions.sh.
# These are the only dependencies CI mode is allowed to find missing.
CUDA_EXTENSION_MODULES = frozenset({
    "pointnet2_ops", "chamfer", "chamfer_dist", "dela_cutils",
    "pointops", "ptv_modules", "clip", "index_max",
})


def allow_missing_extensions() -> bool:
    """True in CI (no GPU / nvcc), where models needing a CUDA extension are
    skipped instead of failed. Set LIDARLEARN_ALLOW_MISSING_EXTENSIONS=1."""
    return os.environ.get("LIDARLEARN_ALLOW_MISSING_EXTENSIONS") == "1"


def build_model_or_skip(model_cfg):
    """build_model_from_cfg, but skip (in CI mode) when the model or a nested
    backbone lives in a module that was skipped for a missing dependency."""
    from models.build import MODELS, build_model_from_cfg
    try:
        return build_model_from_cfg(model_cfg)
    except KeyError as e:
        # Only a registry miss may be skipped; any other KeyError is a bug.
        # tests/test_ci_mode.py checks that only CUDA extensions are missing.
        registry_miss = "is not in the models registry" in str(e)
        if registry_miss and allow_missing_extensions() and MODELS.import_failures:
            missing = sorted({dep for dep, _ in MODELS.import_failures.values()})
            pytest.skip(f"{model_cfg.NAME} unavailable: missing {', '.join(missing)}")
        raise


# --------------------------------------------------------------------------
# Global test configuration
# --------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test needs CUDA; skipped when no GPU is available")


def pytest_collection_modifyitems(config, items):
    """Skip GPU-only tests when CUDA isn't available."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# Config discovery
# --------------------------------------------------------------------------
def _glob(pattern: str):
    return sorted(REPO_ROOT.glob(pattern))


CLASSIFICATION_CONFIGS = _glob("cfgs/classification/**/*.yaml")
S3DIS_CONFIGS = _glob("cfgs/segmentation/**/S3DIS/*.yaml")
SHAPENETPARTS_CONFIGS = _glob("cfgs/segmentation/**/ShapeNetParts/*.yaml")
ALL_CONFIGS = _glob("cfgs/**/*.yaml")

# Pretrain configs live under cfgs/classification/ for historical reasons but
# their model NAME points to an SSL pretraining class (``*_Pretrain``) whose
# forward returns a reconstruction/contrastive loss, not per-sample class
# logits. The classification-shape test can't assert a (B, num_classes) shape
# on them, so we split them off into a dedicated list used by the forward-
# shape suite to filter them out.
PRETRAIN_CONFIGS = [p for p in CLASSIFICATION_CONFIGS if "_pretrain" in p.name]
CLASSIFICATION_NONPRETRAIN = [p for p in CLASSIFICATION_CONFIGS if p not in PRETRAIN_CONFIGS]


def config_id(path: Path) -> str:
    """Human-readable test id from a config path."""
    return str(path.relative_to(REPO_ROOT))


# --------------------------------------------------------------------------
# Dummy batch builders
# --------------------------------------------------------------------------
# Keep the batch size small so the test suite stays cheap even on CPU.
BATCH_SIZE = 2
NUM_POINTS = 1024


def _ensure_cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def make_cls_batch(channels: int = 3, n: int = NUM_POINTS, device: str = "cpu"):
    """Classification input: ``(B, N, C)`` point cloud.

    Classification models in this repo expect the canonical ``(B, N, 3)``
    layout (see ``runner_finetune.py`` ``points = data[0].cuda()``).
    """
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, n, channels, device=device)


def make_seg_batch(channels: int = 3, n: int = NUM_POINTS, device: str = "cpu"):
    """Segmentation input: ``(B, C, N)`` channel-first.

    ``runner_seg.py`` transposes ``(B, N, C)`` → ``(B, C, N)`` before calling
    the model: ``points_bcn = points.transpose(1, 2).contiguous()``.
    """
    torch.manual_seed(0)
    return torch.randn(BATCH_SIZE, channels, n, device=device)


def make_cls_label(num_obj_classes: int, device: str = "cpu"):
    """One-hot object-category tensor used by part-segmentation models."""
    ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    return torch.eye(num_obj_classes, device=device)[ids]


# --------------------------------------------------------------------------
# Expected-channel / expected-num_classes resolution
# --------------------------------------------------------------------------
FEATURE_MODE_TO_CHANNELS = {
    "xyz": 3,
    "xyz_rgb": 6,
    "xyz_rgb_norm": 9,
}


def infer_seg_channels(cfg) -> int:
    """Resolve input channel count from config dataset.others.feature_mode.

    S3DIS yamls set ``feature_mode`` under ``dataset.{train,val}.others``.
    ShapeNetParts doesn't set it (always 3-channel xyz). Default 3.
    """
    try:
        fm = cfg.dataset.train.others.feature_mode
    except (AttributeError, KeyError):
        fm = "xyz"
    return FEATURE_MODE_TO_CHANNELS.get(fm, 3)


def expected_cls_classes(cfg) -> int:
    """Pull the classification head size from whichever field the yaml uses."""
    m = cfg.model
    for key in ("cls_dim", "num_classes", "num_category", "cls_num"):
        v = m.get(key) if hasattr(m, "get") else getattr(m, key, None)
        if v is not None:
            return int(v)
    raise ValueError(
        f"Could not find cls_dim/num_classes in model config {getattr(m, 'NAME', '?')}"
    )


def expected_seg_classes(cfg) -> int:
    """Pull the per-point head size from whichever field the yaml uses."""
    m = cfg.model
    for key in ("seg_classes", "num_seg_classes", "num_classes", "cls_dim"):
        v = m.get(key) if hasattr(m, "get") else getattr(m, key, None)
        if v is not None:
            return int(v)
    raise ValueError(
        f"Could not find seg_classes in model config {getattr(m, 'NAME', '?')}"
    )


def num_obj_classes(cfg) -> int:
    return int(cfg.model.get("num_obj_classes", 16))


# --------------------------------------------------------------------------
# Output normalisation
# --------------------------------------------------------------------------
def unwrap(out):
    """Some models return (logits, aux_loss). Keep only logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out
