"""
PointMamba (Liang et al., NeurIPS 2024) — port checks.

PointMamba needs the optional ``mamba-ssm`` package (NVIDIA GPU only); the
model tests are skipped where it is not installed.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from models.build import build_model_from_cfg
from utils.config import cfg_from_yaml_file

from .conftest import REPO_ROOT

CFG_DIR = REPO_ROOT / "cfgs/classification/PointMamba"
MODELNET_CFG = CFG_DIR / "ModelNet40/modelnet40.yaml"
PRETRAIN_CFG = CFG_DIR / "ShapeNet55/shapenet55_pretrain.yaml"
PRETRAIN_CKPT = REPO_ROOT / "pretrained/pointmamba_pretrain.pth"
FINETUNE_CKPT = REPO_ROOT / "pretrained/pointmamba_finetune_modelnet.pth"

def _mamba_installed() -> bool:
    try:
        return importlib.util.find_spec("mamba_ssm") is not None
    except ImportError:  # an import hook may refuse the lookup outright
        return False


# Skip only when mamba-ssm is absent: with it installed, an unregistered
# PointMamba is a bug and must fail.
needs_mamba = pytest.mark.skipif(not _mamba_installed(),
                                 reason="PointMamba needs the optional mamba-ssm package")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Keys of the pretraining checkpoint that belong to the MAE decoder, which
# the classifier does not have.
DECODER_PREFIXES = ("MAE_decoder.", "decoder_pos_embed.", "mask_token", "increase_dim.")


@needs_mamba
@needs_gpu
def test_pointmamba_forward_shape():
    model = build_model_from_cfg(cfg_from_yaml_file(str(MODELNET_CFG)).model).cuda().eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 1024, 3, device="cuda"))
    assert logits.shape == (2, 40)


@needs_mamba
@needs_gpu
def test_pointmamba_pretrain_returns_scalar_loss():
    model = build_model_from_cfg(cfg_from_yaml_file(str(PRETRAIN_CFG)).model).cuda().train()
    loss = model(torch.rand(2, 1024, 3, device="cuda"))
    assert loss.dim() == 0 and math.isfinite(loss.item())


@needs_mamba
@pytest.mark.parametrize("ckpt, is_pretrain", [(PRETRAIN_CKPT, True), (FINETUNE_CKPT, False)],
                         ids=["pretrain", "finetune_modelnet"])
def test_official_checkpoints_load_exactly(ckpt, is_pretrain):
    if not ckpt.exists():
        pytest.skip(f"checkpoint not present: {ckpt.relative_to(REPO_ROOT)}")
    model = build_model_from_cfg(cfg_from_yaml_file(str(MODELNET_CFG)).model)
    incompatible = model.load_model_from_ckpt(str(ckpt))
    if is_pretrain:
        assert all(k.startswith("cls_head_finetune.") for k in incompatible.missing_keys)
        assert all(k.startswith(DECODER_PREFIXES) for k in incompatible.unexpected_keys)
    else:
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []


# Upstream values (github.com/LMD0311/PointMamba, cfgs/*.yaml):
# lr, max_epoch, drop_path, drop_out, num_group, npoints, batch size, cls_dim.
UPSTREAM = {
    "ShapeNet55/shapenet55_pretrain.yaml": dict(lr=0.001, max_epoch=300, num_group=64, npoints=1024, bs=128),
    "ModelNet40/modelnet40.yaml": dict(lr=0.0003, max_epoch=300, drop_path=0.1, drop_out=0.0,
                                       num_group=64, npoints=1024, bs=32, cls_dim=40),
    **{f"ModelNetFewShot/modelnet_fewshot_{w}w{s}s.yaml": dict(lr=0.0005, max_epoch=150, drop_path=0.5,
                                                               drop_out=0.0, num_group=64, npoints=1024,
                                                               bs=32, cls_dim=w, way=w, shot=s)
       for w in (5, 10) for s in (10, 20)},
    "ScanObjectNN/scanobjectnn_objbg.yaml": dict(lr=0.0005, max_epoch=300, drop_path=0.5, drop_out=0.0,
                                                 num_group=128, npoints=2048, bs=32, cls_dim=15),
    "ScanObjectNN/scanobjectnn_objonly.yaml": dict(lr=0.0003, max_epoch=300, drop_path=0.5, drop_out=0.1,
                                                   num_group=128, npoints=2048, bs=32, cls_dim=15),
    "ScanObjectNN/scanobjectnn_hardest.yaml": dict(lr=0.0005, max_epoch=300, drop_path=0.5, drop_out=0.0,
                                                   num_group=128, npoints=2048, bs=32, cls_dim=15),
}


def test_only_upstream_configs_exist():
    """PointMamba ships exactly the combinations of the upstream repository."""
    found = sorted(str(p.relative_to(CFG_DIR)) for p in CFG_DIR.rglob("*.yaml"))
    assert found == sorted(UPSTREAM)


@pytest.mark.parametrize("rel", sorted(UPSTREAM))
def test_pointmamba_configs_match_upstream(rel):
    cfg = cfg_from_yaml_file(str(CFG_DIR / rel))
    want = UPSTREAM[rel]
    model = cfg.model
    got = dict(lr=cfg.optimizer.kwargs.lr, max_epoch=cfg.max_epoch,
               num_group=model.num_group, npoints=cfg.npoints, bs=cfg.dataset.train.others.bs)
    if "drop_path" in want:
        got.update(drop_path=model.drop_path, drop_out=model.drop_out, cls_dim=model.cls_dim)
    if "way" in want:
        got.update(way=cfg.dataset.train.others.way, shot=cfg.dataset.train.others.shot)
        assert cfg.dataset.val.others.way == want["way"] and cfg.dataset.val.others.shot == want["shot"]
    assert got == want


def test_pretrain_uses_upstream_augmentation():
    """Upstream runner_pretrain.py applies PointcloudRotate (random y-axis rotation)."""
    assert cfg_from_yaml_file(str(PRETRAIN_CFG)).pretrain_augmentation == "y_rotate"


def test_scanobjectnn_sweep_defaults_to_upstream_augmentation():
    """Upstream runner_finetune.py rotates ScanObjectNN batches about the y axis."""
    script = (REPO_ROOT / "scripts/train_scanobjectnn.sh").read_text()
    assert "AUGMENTATION=${AUGMENTATION:-y_rotate}" in script
