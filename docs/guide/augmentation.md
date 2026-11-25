# Online Augmentation

Point-cloud augmentations are implemented in [`datasets/augmentation.py`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/augmentation.py) and applied **online** — on the GPU, per batch, inside the training loop — not baked into disk files. That means:

- **Every epoch sees freshly-sampled augmentations** (different rotations, jitter noise, dropout masks) even though the underlying `.xyz` / `.h5` data on disk is fixed.
- **Applied to training only** — validation and test passes skip augmentation (only the always-on unit-sphere normalisation runs there). Guarded at the call site in [`tools/runner_finetune.py:255`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/tools/runner_finetune.py#L255), so every model/config gets the same guarantee without extra code.
- **GPU-side** — transforms take and return a `(B, C, N)` or `(B, N, C)` `torch.Tensor` already on CUDA, so there's no CPU→GPU round-trip in the data path.

**Enable via `--augmentation <name>`** (default: `none`). Choices are defined in [`utils/parser.py`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/utils/parser.py) and map 1:1 to classes in the augmentation module:

| Flag | What it does |
|---|---|
| `none` | Identity — no augmentation (default) |
| `rotate` | Random 3D rotation around all three axes |
| `scale_translate` | Combined random uniform scaling + translation |
| `jitter` | Additive Gaussian noise on every point |
| `scale` | Random uniform scaling only |
| `translate` | Random translation only |
| `dropout` | Random point dropout (mask out a fraction of points) |
| `flip` | Random axis flipping (x or y) |
| `z_rotate_tree` | Rotation around Z-axis only — right default for upright tree scans (TLS / ALS) where up is meaningful |
| `y_rotate` | Random rotation around the Y-axis (as `PointcloudRotate` in Point-BERT / PointMamba) — used for PointMamba pretraining and ScanObjectNN |

**Example — full-sweep smoke test with Z-axis rotation for tree data:**

```bash
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv.yaml \
    --augmentation z_rotate_tree --run_all_folds --seed 42 \
    --exp_name pointmae_stpctls_cv_zrot
```

**Composing multiple transforms**

- The CLI takes a single name, but `get_train_transforms()` accepts a list, such as `['z_rotate_tree', 'jitter', 'scale']`.
- Call it from a custom training script, or edit the runner.
- Every transform in the chain is re-randomised per batch.

**Reproducibility**

- Augmentations use the global torch/numpy random generators, so `--seed 42` makes a training run repeatable (given the same CUDA kernels).
- The selected transform is printed at the start of training, so the log shows what ran.
