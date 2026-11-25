#!/usr/bin/env python3
"""
STPCTLS — full preprocessing pipeline.

Reads the raw per-species tree point clouds from ``preprocessing/STPCTLS/``
(one folder per species, one ``.pts``/``.txt`` file per tree) and writes
downsampled 1024-point ``.xyz`` files into ``data/STPCTLS/`` with the same
per-species folder layout. The label for each tree is the name of its
parent directory.

Output layout:
    data/STPCTLS/
    ├── Buche/
    │   ├── 102.xyz
    │   └── ...
    ├── Douglasie/
    ├── Eiche/
    ├── Esche/
    ├── Fichte/
    ├── Kiefer/
    └── Roteiche/

Pipeline, per tree:
    1. Load the first three columns (XYZ) from the text file; ignore any
       trailing columns (some Roteiche files ship RGB + intensity).
    2. Resample to exactly 1024 points with ``utils.fps.resample``: farthest
       point sampling when the cloud is larger (on the GPU when
       ``pointnet2_ops`` and CUDA are available, in NumPy otherwise), cyclic
       duplication when it is smaller (no tree is dropped).
    3. Write ``.xyz`` with a ``"x y z"`` header + 1024 data rows (``.6f``
       floats, space-separated). Raw coordinates are preserved — no
       centering/scaling, matching the original preprocessed release.

Usage:
    python preprocessing/preprocess_stpctls.py \\
        --input_dir preprocessing/STPCTLS \\
        --output_dir data/STPCTLS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Run as a script from the repo root: make `utils` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.fps import cuda_fps_available, resample  # noqa: E402


TARGET_POINTS = 1024
RAW_EXTENSIONS = (".xyz", ".pts", ".txt")


def load_xyz(path: Path) -> np.ndarray | None:
    """Load XYZ (first 3 columns) from a whitespace-separated text file.

    Tolerates optional header rows and trailing columns (RGB, intensity).
    Returns an (N, 3) float64 array or None on failure.
    """
    try:
        arr = np.loadtxt(path)
    except ValueError:
        arr = np.loadtxt(path, skiprows=1)
    except Exception as e:
        print(f"  [SKIP] {path.name}: load error — {e}")
        return None
    if arr.ndim != 2 or arr.shape[1] < 3:
        print(f"  [SKIP] {path.name}: expected >=3 columns, got shape {arr.shape}")
        return None
    return arr[:, :3].astype(np.float64)


def write_xyz(out_path: Path, points: np.ndarray) -> None:
    """Write (target, 3) points with a ``x y z`` header, 6-decimal floats."""
    lines = ["x y z\n"]
    for p in points:
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    out_path.write_text("".join(lines))


def safe_stem(path: Path) -> str:
    """Replace spaces in file stems so downstream shell pipes don't choke."""
    return path.stem.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Preprocess STPCTLS raw per-species point clouds into "
            f"{TARGET_POINTS}-point .xyz files organised by class folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("preprocessing/STPCTLS"),
        help="Directory containing one folder per species with raw .pts/.txt files.",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/STPCTLS"),
        help="Output directory; one subfolder per species will be created.",
    )
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    species_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if not species_dirs:
        sys.exit(f"No species subdirectories found in {args.input_dir}")

    print(
        f"STPCTLS preprocessing:\n"
        f"  input : {args.input_dir}\n"
        f"  output: {args.output_dir}\n"
        f"  target: {TARGET_POINTS} points per tree "
        f"({'GPU' if cuda_fps_available() else 'CPU'} FPS; cyclic dup < {TARGET_POINTS})\n"
        f"  species ({len(species_dirs)}): {[d.name for d in species_dirs]}\n"
    )

    totals: dict[str, int] = {}
    skipped: list[tuple[str, str]] = []

    for sp_dir in species_dirs:
        trees = sorted(
            p for p in sp_dir.iterdir() if p.suffix.lower() in RAW_EXTENSIONS
        )
        out_sp = args.output_dir / sp_dir.name
        out_sp.mkdir(parents=True, exist_ok=True)

        ok = 0
        for tree_path in tqdm(trees, desc=f"{sp_dir.name:10s}", unit="tree"):
            xyz = load_xyz(tree_path)
            if xyz is None:
                skipped.append((str(tree_path), "load failure"))
                continue
            if xyz.shape[0] == 0:
                skipped.append((str(tree_path), "empty"))
                continue

            resampled = resample(xyz, TARGET_POINTS)
            out_path = out_sp / f"{safe_stem(tree_path)}.xyz"
            write_xyz(out_path, resampled)
            ok += 1

        totals[sp_dir.name] = ok

    print("\nPer-species tree count:")
    for sp, n in totals.items():
        print(f"  {sp:10s} {n:>4d}")
    print(f"  {'TOTAL':10s} {sum(totals.values()):>4d}")

    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for name, reason in skipped[:20]:
            print(f"  {name}  — {reason}")
        if len(skipped) > 20:
            print(f"  ... ({len(skipped) - 20} more)")


if __name__ == "__main__":
    main()
