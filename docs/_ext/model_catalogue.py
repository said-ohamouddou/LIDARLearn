"""Generate the documentation's model catalogue from ``cfgs/`` and the methods CSV.

The catalogue lists one entry per model folder in ``cfgs/classification`` and
``cfgs/segmentation``: its paper, venue, category, the tasks and datasets it has
configs for, and the fine-tuning strategies it supports. Nothing is written by
hand, so the page always matches the shipped configs.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

GITHUB_TREE = "https://github.com/said-ohamouddou/LIDARLearn/tree/main"

# Method names in docs/point_cloud_methods.csv that differ from the config folder.
CSV_TO_CONFIG_DIR = {
    "Point-MAE": "PointMAE", "Point-BERT": "PointBERT", "Point-M2AE": "PointM2AE",
    "PCP-MAE": "PCPMAE", "ReCon": "RECON", "DeepGCNs": "DeepGCN",
    "MS-DGCNN": "MSDGCNN", "MS-DGCNN++": "MSDGCNN2", "KAN-DGCNN": "KANDGCNN",
    "KAN-PointNet": "PointKAN", "DeLA": "DELA", "GDANet": "GDAN", "RS-CNN": "RSCNN",
    "SO-Net": "SONet", "Point-TNT": "PointTNT", "PointNet++": "PointNet2",
    "Point Transformer": "PointTransformer", "Global Transformer": "GlobalTransformer",
    "RandLA-Net": "RandLANet", "pointtransformerv2": "PointTransformerV2",
    "pointtransformerv3": "PointTransformerV3",
}
# Display names for CSV entries whose CSV spelling is lowercase.
DISPLAY_NAME = {"pointtransformerv2": "Point Transformer v2", "pointtransformerv3": "Point Transformer v3"}

# PEFT strategies: config-file suffix -> name, and CSV rows describing them.
PEFT_SUFFIXES = {"dapt": "DAPT", "idpt": "IDPT", "ppt": "PPT", "gst": "GST", "vpt_deep": "VPT-Deep"}
CSV_TO_STRATEGY = {"DAPT": "DAPT", "GPT-DAPT": "DAPT", "IDPT": "IDPT", "PPT": "PPT",
                   "PointGST": "GST", "VPT": "VPT-Deep"}

CATEGORY_ORDER = ["Point-based", "Attention-based", "Graph-based", "Self-supervised", "Other"]
PRETRAIN_DIR = "ShapeNet55"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    category: str
    paper: str
    authors: str
    venue: str
    cls_datasets: tuple[str, ...]
    partseg: bool
    semseg: bool
    peft: tuple[str, ...]
    pretrain: bool
    config_dir: str


def _read_csv(repo_root: Path) -> list[dict]:
    with open(repo_root / "docs" / "point_cloud_methods.csv", newline="") as f:
        return list(csv.DictReader(f))


def _category(repo_root: Path, config_dir: str) -> str:
    sys.path.insert(0, str(repo_root / "scripts" / "reports"))
    try:
        import _citations
    except ImportError:
        return "Other"
    finally:
        sys.path.pop(0)
    base, _ = _citations.canonicalize(config_dir.lower())
    return _citations.get_category(base)


def _subdirs(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def _peft(cls_dir: Path) -> tuple[str, ...]:
    found = set()
    for cfg in cls_dir.rglob("*.yaml"):
        m = re.search(r"_(vpt_deep|dapt|idpt|ppt|gst)\.yaml$", cfg.name)
        if m:
            found.add(PEFT_SUFFIXES[m.group(1)])
    return tuple(s for s in PEFT_SUFFIXES.values() if s in found)


def build_catalogue(repo_root: Path) -> list[ModelEntry]:
    """One entry per model folder in cfgs/; raises KeyError for a folder with no CSV row."""
    repo_root = Path(repo_root)
    rows = {CSV_TO_CONFIG_DIR.get(r["Method Name"], r["Method Name"]): r for r in _read_csv(repo_root)}
    cls_root, seg_root = repo_root / "cfgs" / "classification", repo_root / "cfgs" / "segmentation"
    entries = []
    for config_dir in sorted(set(_subdirs(cls_root)) | set(_subdirs(seg_root))):
        if config_dir not in rows:
            raise KeyError(f"cfgs model folder '{config_dir}' has no row in docs/point_cloud_methods.csv")
        row = rows[config_dir]
        cls_dir, seg_dir = cls_root / config_dir, seg_root / config_dir
        datasets = _subdirs(cls_dir)
        entries.append(ModelEntry(
            name=DISPLAY_NAME.get(row["Method Name"], row["Method Name"]),
            category=_category(repo_root, config_dir),
            paper=row["Paper Title"],
            authors=row["Authors"],
            venue=row["Venue/Year"],
            cls_datasets=tuple(d for d in datasets if d != PRETRAIN_DIR),
            partseg=(seg_dir / "ShapeNetParts").is_dir(),
            semseg=(seg_dir / "S3DIS").is_dir(),
            peft=_peft(cls_dir) if cls_dir.is_dir() else (),
            pretrain=PRETRAIN_DIR in datasets,
            config_dir=config_dir,
        ))
    return entries


def load_strategies(repo_root: Path) -> list[dict]:
    """Papers of the PEFT strategies, one per strategy, in PEFT_SUFFIXES order."""
    by_strategy = {}
    for row in _read_csv(Path(repo_root)):
        strategy = CSV_TO_STRATEGY.get(row["Method Name"])
        if strategy and strategy not in by_strategy:
            by_strategy[strategy] = {"strategy": strategy, "paper": row["Paper Title"],
                                     "authors": row["Authors"], "venue": row["Venue/Year"]}
    return [by_strategy[s] for s in PEFT_SUFFIXES.values() if s in by_strategy]


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(entries: list[ModelEntry], strategies: list[dict]) -> str:
    """MyST Markdown: one table per category, then the fine-tuning strategies."""
    out = []
    for category in CATEGORY_ORDER:
        group = sorted((e for e in entries if e.category == category), key=lambda e: e.name.lower())
        if not group:
            continue
        out += [f"## {category}", "",
                "| Model | Paper | Venue | Classification | Part seg | Sem seg | Pre-training | PEFT | Configs |",
                "|---|---|---|---|:-:|:-:|:-:|---|---|"]
        for e in group:
            kind = "classification" if e.cls_datasets or e.pretrain else "segmentation"
            out.append("| " + " | ".join([
                f"**{_cell(e.name)}**", _cell(e.paper), _cell(e.venue),
                ", ".join(e.cls_datasets) or "—",
                "✓" if e.partseg else "—", "✓" if e.semseg else "—",
                "✓" if e.pretrain else "—",
                ", ".join(e.peft) or "—",
                f"[`{e.config_dir}`]({GITHUB_TREE}/cfgs/{kind}/{e.config_dir})",
            ]) + " |")
        out.append("")
    out += ["## Fine-tuning strategies", "",
            "| Strategy | Paper | Venue |", "|---|---|---|"]
    for s in strategies:
        label = "GST (PointGST)" if s["strategy"] == "GST" else s["strategy"]
        out.append(f"| **{label}** | {_cell(s['paper'])} | {_cell(s['venue'])} |")
    out.append("")
    return "\n".join(out)
