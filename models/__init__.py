"""
LIDARLearn Models Package

Authors: LIDARLearn contributors
License: See repository LICENSE

Aggregate import module that registers every model class with the
MODELS registry.

Every model module is imported through ``_safe_import``: if a module needs a
third-party package or compiled CUDA extension that is not installed (or a
compiled ``.so`` that fails to load, e.g. built against another torch/CUDA),
the module is skipped and recorded in ``MODELS.import_failures``. The rest of
the library keeps working, a single summary warning is emitted, and asking for
a skipped model raises an error naming the missing dependency.

Import errors that come from LIDARLearn's own packages (``models``, ``utils``,
``datasets``, ``tools``) are real bugs and are re-raised, never hidden.
"""
import warnings
from collections import defaultdict

from .build import MODELS, build_model_from_cfg  # noqa: F401

# Top-level packages that belong to LIDARLearn itself.
_INTERNAL_PACKAGES = {"models", "utils", "datasets", "tools"}


def _missing_dependency(exc):
    """Return the name of the missing external dependency, or None if the
    error comes from LIDARLearn's own code."""
    name = getattr(exc, "name", None)
    if isinstance(exc, ModuleNotFoundError) and name:
        top = name.split(".")[0]
        return None if top in _INTERNAL_PACKAGES else top
    # Shared-library load failures (undefined symbol, missing libc10.so, ...)
    # surface as ImportError/OSError without a module name.
    return name.split(".")[0] if name else "compiled extension"


def _safe_import(module_path):
    """Import a model module; record and skip it if a dependency is missing."""
    try:
        __import__(module_path)
    except (ImportError, OSError) as e:
        dep = _missing_dependency(e)
        if dep is None:
            raise
        MODELS.import_failures[module_path] = (dep, f"{type(e).__name__}: {e}")


_MODEL_MODULES = [
    # SSL backbones
    "models.pointmae.pointmae",                       # pointnet2_ops, chamfer_dist
    "models.pointbert.dvae",                          # pointnet2_ops, chamfer_dist
    "models.pointbert.pointbert",
    "models.pointgpt.pointgpt",
    "models.pointgpt.pointgpt_seg",
    "models.act.act",                                 # + clip
    "models.recon.recon",                             # + clip
    "models.pcpmae.pcpmae",
    "models.pcpmae.pcpmae_seg",
    "models.pointm2ae.pointm2ae",
    "models.pointm2ae.pointm2ae_pretrain",
    "models.pointm2ae.pointm2ae_seg",

    # PEFT strategies
    "models.ppt.ppt",
    "models.ppt.ppt_seg",
    "models.idpt.idpt",
    "models.idpt.idpt_seg",
    "models.dapt.dapt",
    "models.dapt.dapt_seg",
    "models.dapt.pointgpt_dapt",
    "models.pointgst.pointgst",
    "models.pointgst.pointgst_seg",
    "models.pointgst.pointgpt_gst",
    "models.pointgpt.pointgpt_ppt",
    "models.pointgpt.pointgpt_idpt",
    "models.pointgpt.pointgpt_vpt_deep",

    # Supervised models
    "models.dgcnn.dgcnn",
    "models.pointnet.pointnet",
    "models.pointnet2.pointnet2",                     # pointnet2_ops
    "models.pointnet2.pointnet2_seg",
    "models.pct.pct",                                 # pointnet2_ops
    "models.pct.pct_seg",
    "models.pointmlp.pointmlp",                       # pointnet2_ops
    "models.pointmlp.pointmlp_seg",
    "models.pointconv.pointconv",
    "models.pointconv.pointconv_seg",                 # pointnet2_ops
    "models.pointweb.pointweb",                       # pointnet2_ops
    "models.pointweb.pointweb_seg",
    "models.curvenet.curvenet",
    "models.curvenet.curvenet_seg",
    "models.deepgcn.deepgcn",                         # torch_cluster
    "models.deepgcn.deepgcn_seg",
    "models.sonet.sonet",                             # index_max
    "models.pointcnn.pointcnn",
    "models.pointkan.pointkan",
    "models.pointkan.pointkan_seg",
    "models.kandgcnn.kan_dgcnn",
    "models.gdan.gdan",                               # pointnet2_ops
    "models.ppfnet.ppfnet",                           # pointnet2_ops
    "models.pointscnet.network",                      # pointnet2_ops
    "models.pointscnet.pointscnet_seg",
    "models.randlenet.randlenet",                     # pointnet2_ops
    "models.randlenet.randlenet_seg",
    "models.rscnn.rscnn",                             # pointnet2_ops
    "models.rscnn.rscnn_seg",
    "models.msdgcnn.msdgcnn",                         # pointnet2_ops
    "models.msdgcnn.msdgcnn_seg",
    "models.msdgcnn2.msdgcnn2",
    "models.globaltransformer.globalltransformer",
    "models.globaltransformer.globaltransformer_seg",
    "models.pointtnt.pointtnt",
    "models.pointtnt.pointtnt_seg",                   # pointnet2_ops
    "models.p2p.p2p",                                 # torch_scatter, pointnet2_ops
    "models.p2p.p2p_seg",
    "models.rapsurf.rapsurf",                         # pointnet2_ops
    "models.rapsurf.rapsurf_seg",
    "models.dela.dela",                               # dela_cutils, pointnet2_ops
    "models.dela.dela_seg",
    "models.pvt.pvt",                                 # ptv_modules
    "models.pvt.pvt_seg",
    "models.pointtransformer.pointtransformer",       # pointops
    "models.pointtransformer.pointtransformer_seg",
    "models.pointtransformerv2.pointtransformerv2",   # pointops, torch_scatter, torch_geometric
    "models.pointtransformerv2.pointtransformerv2_seg",
    "models.pointtransformerv3.pointtransformerv3",   # torch_scatter (spconv optional)
    "models.pointtransformerv3.pointtransformerv3_seg",

    # Segmentation models
    "models.seg.registered_seg",
    "models.seg.transformer_seg",                     # pointnet2_ops
]

for _module in _MODEL_MODULES:
    _safe_import(_module)


def _warn_import_failures():
    if not MODELS.import_failures:
        return
    by_dep = defaultdict(list)
    for module_path, (dep, _) in MODELS.import_failures.items():
        by_dep[dep].append(module_path.split(".")[1])
    lines = [
        f"  - {dep}: {len(mods)} module(s) skipped ({', '.join(sorted(set(mods)))})"
        for dep, mods in sorted(by_dep.items())
    ]
    warnings.warn(
        "Some LIDARLearn models are unavailable because a dependency is missing:\n"
        + "\n".join(lines)
        + "\nAll other models work normally. To enable the skipped ones, install the "
        "package or build the CUDA extensions: bash extensions/install_extensions.sh",
        stacklevel=2,
    )


_warn_import_failures()
