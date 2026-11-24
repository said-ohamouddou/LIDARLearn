"""
Checkpoint loading must behave the same on torch 2.4 and torch >= 2.6.

torch 2.6 switched ``torch.load`` to ``weights_only=True`` by default, which
rejects LIDARLearn's own checkpoints (they store EasyDict configs and numpy
scalars next to the weights). All loading goes through
``utils.checkpoint.load_checkpoint``; these tests pin that behaviour.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from easydict import EasyDict
from packaging.version import Version

from utils.checkpoint import load_checkpoint

from .conftest import REPO_ROOT


def _training_style_checkpoint():
    return {
        "base_model": {"encoder.weight": torch.randn(4, 3)},
        "epoch": 7,
        "metrics": {"acc": np.float64(91.25)},
        "config": EasyDict({"model": {"NAME": "PointMAE", "trans_dim": 384}}),
    }


def test_load_checkpoint_reads_training_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pth"
    ckpt = _training_style_checkpoint()
    torch.save(ckpt, path)

    loaded = load_checkpoint(path, map_location="cpu")

    assert torch.equal(loaded["base_model"]["encoder.weight"], ckpt["base_model"]["encoder.weight"])
    assert loaded["epoch"] == 7
    assert loaded["metrics"]["acc"] == pytest.approx(91.25)
    assert loaded["config"].model.NAME == "PointMAE"


@pytest.mark.skipif(Version(torch.__version__.split("+")[0]) < Version("2.6"),
                    reason="weights_only=True became the default in torch 2.6")
def test_plain_torch_load_rejects_it_on_new_torch(tmp_path):
    """Documents why the helper exists: the default loader refuses this file."""
    path = tmp_path / "ckpt.pth"
    torch.save(_training_style_checkpoint(), path)
    with pytest.raises(Exception, match="[Ww]eights only load failed"):
        torch.load(path, map_location="cpu")


def test_no_bare_torch_load_in_library():
    """Every load in the repo must go through load_checkpoint (or pass
    weights_only explicitly), otherwise it breaks on torch >= 2.6."""
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in {"tests", "build", "site"} or "build" in rel.parts:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"\btorch\.load\(", line) and "weights_only" not in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "use utils.checkpoint.load_checkpoint:\n" + "\n".join(offenders)
