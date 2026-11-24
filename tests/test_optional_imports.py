"""
Optional-dependency import tests.

LIDARLearn must stay importable when CUDA extensions or optional packages are
missing (CPU-only machines, a failed extension build, a torch/CUDA mismatch).
Each test runs in a fresh interpreter with selected packages blocked, so it
works the same whether or not the extensions are installed here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from .conftest import REPO_ROOT

ALL_OPTIONAL = [
    "pointnet2_ops", "chamfer", "chamfer_dist", "dela_cutils", "pointops",
    "ptv_modules", "clip", "index_max", "torch_scatter", "torch_cluster",
    "torch_geometric", "spconv", "flash_attn",
]

# Models whose modules need no CUDA extension and no PyG package.
PURE_TORCH_CONFIG = "cfgs/classification/DGCNN/ModelNet40/modelnet40_k20.yaml"
# PCT imports pointnet2_ops at module level, so it is skipped without it.
EXTENSION_CONFIG = "cfgs/classification/PCT/ModelNet40/modelnet40.yaml"

_PRELUDE = textwrap.dedent("""
    import importlib.abc, json, sys, warnings
    BLOCKED = set({blocked!r})

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

    sys.meta_path.insert(0, _Block())
    sys.path.insert(0, {root!r})
    warnings.simplefilter("ignore")
""")


def _run(blocked, body):
    code = _PRELUDE.format(blocked=list(blocked), root=str(REPO_ROOT)) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_library_imports_without_any_extension():
    """`import models` and the training entry points load with every optional
    dependency missing, and the skipped modules are recorded."""
    out = _run(ALL_OPTIONAL, """
        import models, datasets, tools
        from utils import misc
        import main  # noqa: F401  (training entry point)
        from models.build import MODELS
        print(json.dumps({
            "registered": len(MODELS.module_dict),
            "deps": sorted({d for d, _ in MODELS.import_failures.values()}),
        }))
    """)
    assert out["registered"] > 0
    assert "pointnet2_ops" in out["deps"]


def test_pure_torch_model_builds_without_extensions():
    out = _run(ALL_OPTIONAL, f"""
        import models
        from models.build import build_model_from_cfg
        from utils.config import cfg_from_yaml_file
        cfg = cfg_from_yaml_file({PURE_TORCH_CONFIG!r})
        model = build_model_from_cfg(cfg.model)
        print(json.dumps({{"params": sum(p.numel() for p in model.parameters())}}))
    """)
    assert out["params"] > 0


def test_missing_extension_gives_actionable_error():
    """Building a model whose module was skipped names the missing dependency."""
    out = _run(["pointnet2_ops"], f"""
        import models
        from models.build import build_model_from_cfg
        from utils.config import cfg_from_yaml_file
        cfg = cfg_from_yaml_file({EXTENSION_CONFIG!r})
        try:
            build_model_from_cfg(cfg.model)
            msg = None
        except KeyError as e:
            msg = str(e)
        print(json.dumps({{"msg": msg}}))
    """)
    assert out["msg"] is not None, "expected a KeyError for a skipped model"
    assert "pointnet2_ops" in out["msg"]
    assert "install_extensions.sh" in out["msg"]


def test_fps_without_extension_gives_actionable_error():
    """utils.misc imports without pointnet2_ops; fps() itself explains the fix."""
    out = _run(["pointnet2_ops"], """
        import torch
        from utils import misc
        try:
            misc.fps(torch.zeros(1, 16, 3), 4)
            msg = None
        except ImportError as e:
            msg = str(e)
        print(json.dumps({"msg": msg}))
    """)
    assert out["msg"] and "install_extensions.sh" in out["msg"]


def test_only_absent_packages_are_skipped():
    """With nothing blocked, a module may be skipped only if its dependency is
    truly not installed. An installed extension that fails to load (e.g. built
    against another torch/CUDA) must surface here instead of hiding."""
    out = _run([], """
        import importlib.util
        import models
        from models.build import MODELS
        bad = {}
        for mod, (dep, err) in MODELS.import_failures.items():
            try:
                installed = importlib.util.find_spec(dep) is not None
            except (ImportError, ValueError):
                installed = False
            if installed:
                bad[mod] = err
        print(json.dumps({"bad": bad}))
    """)
    assert not out["bad"], out["bad"]
