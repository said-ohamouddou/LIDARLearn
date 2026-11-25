"""
The documentation's model catalogue is generated from cfgs/ and
docs/point_cloud_methods.csv. It must list every model that has configs,
with its paper, and never drop a model silently.
"""

from __future__ import annotations

import shutil
import sys

import pytest

from .conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "docs" / "_ext"))
from model_catalogue import build_catalogue, load_strategies, render_markdown, render_page  # noqa: E402


def _config_dirs(root):
    return {p.name for t in ("classification", "segmentation")
            for p in (root / "cfgs" / t).iterdir() if p.is_dir()}


def test_every_config_model_has_a_paper_row():
    entries = build_catalogue(REPO_ROOT)
    assert {e.config_dir for e in entries} == _config_dirs(REPO_ROOT)
    assert all(e.paper and e.venue for e in entries)


def test_msdgcnn_variants_are_distinct():
    by_dir = {e.config_dir: e for e in build_catalogue(REPO_ROOT)}
    assert by_dir["MSDGCNN"].paper != by_dir["MSDGCNN2"].paper


def test_tasks_and_peft_follow_the_configs():
    by_dir = {e.config_dir: e for e in build_catalogue(REPO_ROOT)}
    assert by_dir["PointMamba"].cls_datasets == ("ModelNet40", "ModelNetFewShot", "ScanObjectNN")
    assert not by_dir["PointMamba"].partseg
    assert by_dir["RandLANet"].semseg and not by_dir["RandLANet"].cls_datasets
    assert set(by_dir["PointMAE"].peft) == {"DAPT", "IDPT", "PPT", "GST", "VPT-Deep"}
    assert by_dir["PointMAE"].pretrain and not by_dir["PointNet"].pretrain
    assert "Other" not in {e.category for e in by_dir.values()}


def test_every_peft_strategy_has_a_paper():
    names = {s["strategy"] for s in load_strategies(REPO_ROOT)}
    assert names == {"DAPT", "IDPT", "PPT", "GST", "VPT-Deep"}


def test_missing_csv_row_raises(tmp_path):
    (tmp_path / "docs").mkdir()
    shutil.copy(REPO_ROOT / "docs" / "point_cloud_methods.csv", tmp_path / "docs")
    (tmp_path / "cfgs" / "classification" / "NotAModel" / "ModelNet40").mkdir(parents=True)
    (tmp_path / "cfgs" / "segmentation").mkdir(parents=True)
    with pytest.raises(KeyError, match="NotAModel"):
        build_catalogue(tmp_path)


def test_render_lists_every_model_and_strategy():
    entries = build_catalogue(REPO_ROOT)
    strategies = load_strategies(REPO_ROOT)
    md = render_markdown(entries, strategies)
    assert all(e.name in md for e in entries)
    assert all(s["paper"] in md for s in strategies)


def test_committed_models_page_is_up_to_date():
    """docs/models.md is committed so it renders on GitHub too; it must match the configs."""
    committed = (REPO_ROOT / "docs" / "models.md").read_text()
    assert committed == render_page(REPO_ROOT), (
        "docs/models.md is out of date with cfgs/. Regenerate it with: "
        "python docs/_ext/model_catalogue.py")


def test_models_page_renders_on_github():
    page = render_page(REPO_ROOT)
    assert "{include}" not in page and "PointMamba" in page and "## Fine-tuning strategies" in page
