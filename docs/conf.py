"""Sphinx configuration for the LIDARLearn documentation."""

import os
import sys

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DOCS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(DOCS_DIR, "_ext"))

# The API pages import the library on a CPU-only machine (Read the Docs, CI):
# models that need a missing CUDA extension are skipped, not fatal.
os.environ.setdefault("LIDARLEARN_ALLOW_MISSING_EXTENSIONS", "1")

project = "LIDARLearn"
author = "Said Ohamouddou, Hanaa El Afia, Abdellatif El Afia, Raddouane Chiheb"
copyright = "2026, " + author

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "_ext", "paper"]
myst_heading_anchors = 3

autodoc_mock_imports = [
    "pointnet2_ops", "chamfer", "chamfer_dist", "dela_cutils", "pointops", "ptv_modules",
    "clip", "index_max", "mamba_ssm", "causal_conv1d", "torch_scatter",
    "torch_cluster", "lightly", "flash_attn", "spconv", "transformers",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"

html_theme = "furo"
html_title = "LIDARLearn"
html_logo = "../media/logo.png"
html_theme_options = {
    "source_repository": "https://github.com/said-ohamouddou/LIDARLearn/",
    "source_branch": "main",
    "source_directory": "docs/",
}


def _write_model_catalogue(app):
    """Regenerate docs/models.md from cfgs/ before reading the sources."""
    from pathlib import Path

    from model_catalogue import write_page

    write_page(Path(REPO_ROOT))


def setup(app):
    app.connect("builder-inited", _write_model_catalogue)
