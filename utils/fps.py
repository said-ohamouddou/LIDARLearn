"""Farthest point sampling (FPS) for data preparation.

Every dataset loader and preprocessing script that downsamples a single point
cloud goes through this module, so they all select the same points:

* FPS starts at point 0 (deterministic, no RNG).
* Distances use the first three columns; any extra columns (normals, colour)
  are carried along with their point.
* The result is in selection order, so the first ``k`` rows of an
  ``N``-point FPS sample are themselves a ``k``-point FPS sample. Caches built
  for a large ``N`` can therefore be truncated for a smaller ``N_POINTS``.

The work runs on the GPU through ``pointnet2_ops`` when the extension is
built and CUDA is available, and in NumPy otherwise. Both paths see the same
float32 input and give the same points, except where several candidates are
exactly equally far away.

Batched FPS on tensors already on the GPU, as used inside models and runners,
is ``utils.misc.fps``.
"""

from __future__ import annotations

import numpy as np

DEVICES = ("auto", "cpu", "cuda")


def _cuda_kernel():
    """``pointnet2_ops`` FPS kernel if it can run here, else None."""
    try:
        import torch
        from pointnet2_ops.pointnet2_utils import furthest_point_sample
    except ImportError:
        return None
    return furthest_point_sample if torch.cuda.is_available() else None


def cuda_fps_available() -> bool:
    """True when ``device="auto"`` will run FPS on the GPU."""
    return _cuda_kernel() is not None


def _fps_numpy(xyz: np.ndarray, npoint: int) -> np.ndarray:
    idx = np.empty(npoint, dtype=np.int64)
    dist = np.full(len(xyz), np.inf, dtype=np.float32)
    farthest = 0
    for i in range(npoint):
        idx[i] = farthest
        d = xyz - xyz[farthest]
        np.minimum(dist, np.einsum("ij,ij->i", d, d), out=dist)
        farthest = int(dist.argmax())
    return idx


def _fps_cuda(kernel, xyz: np.ndarray, npoint: int) -> np.ndarray:
    import torch

    pts = torch.from_numpy(xyz).cuda().unsqueeze(0).contiguous()
    return kernel(pts, npoint)[0].long().cpu().numpy()


def fps_indices(points: np.ndarray, npoint: int, device: str = "auto") -> np.ndarray:
    """Indices of ``npoint`` rows of ``points`` (N, >=3) chosen by FPS, in selection order.

    ``device`` is ``"auto"`` (GPU when available), ``"cpu"`` or ``"cuda"``.
    Use ``"cpu"`` inside DataLoader workers, where CUDA cannot be initialised.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected an (N, >=3) array, got shape {points.shape}")
    n = points.shape[0]
    if not 0 < npoint <= n:
        raise ValueError(f"cannot sample {npoint} points from a cloud of {n}")

    # Shift so the bounding box starts at (1, 1, 1). Distances are unchanged,
    # the float32 cast keeps precision for large (e.g. UTM) coordinates, and
    # no point lies near the origin, which the pointnet2_ops kernel would
    # otherwise never select (it skips points with |p|^2 <= 1e-3).
    xyz = points[:, :3].astype(np.float64)
    xyz = np.ascontiguousarray(xyz - xyz.min(axis=0) + 1.0, dtype=np.float32)

    kernel = None if device == "cpu" else _cuda_kernel()
    if kernel is None:
        if device == "cuda":
            raise RuntimeError(
                "device='cuda' needs CUDA and the pointnet2_ops extension "
                "(bash extensions/install_extensions.sh)."
            )
        return _fps_numpy(xyz, npoint)
    return _fps_cuda(kernel, xyz, npoint)


def fps(points: np.ndarray, npoint: int, device: str = "auto") -> np.ndarray:
    """The ``npoint`` rows of ``points`` chosen by FPS, in selection order."""
    return np.asarray(points)[fps_indices(points, npoint, device)]


def resample(points: np.ndarray, npoint: int, device: str = "auto") -> np.ndarray:
    """Exactly ``npoint`` rows: FPS when there are more, the cloud unchanged when
    equal, and cyclic repetition (rows 0..N-1, 0..N-1, ...) when there are fewer."""
    points = np.asarray(points)
    n = points.shape[0]
    if n == 0:
        raise ValueError("cannot resample an empty point cloud")
    if n > npoint:
        return fps(points, npoint, device)
    if n == npoint:
        return points
    reps = -(-npoint // n)
    return np.tile(points, (reps, 1))[:npoint]
