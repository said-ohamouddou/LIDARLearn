"""Sphinx configuration for the LIDARLearn documentation."""

import os
import sys

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DOCS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(DOCS_DIR, "_ext"))

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
exclude_patterns = ["_build", "paper", "superpowers", "models/catalogue.md"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "LIDARLearn"
html_logo = "../media/logo.png"
html_theme_options = {
    "source_repository": "https://github.com/said-ohamouddou/LIDARLearn/",
    "source_branch": "main",
    "source_directory": "docs/",
}


def _write_model_catalogue(app):
    """Regenerate models/catalogue.md from cfgs/ before reading the sources."""
    from pathlib import Path

    from model_catalogue import build_catalogue, load_strategies, render_markdown

    root = Path(REPO_ROOT)
    text = render_markdown(build_catalogue(root), load_strategies(root))
    out = Path(DOCS_DIR) / "models" / "catalogue.md"
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.read_text() != text:
        out.write_text(text)


def setup(app):
    app.connect("builder-inited", _write_model_catalogue)
