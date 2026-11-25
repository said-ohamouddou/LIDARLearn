# Changelog

## Unreleased

**STPCTLS (80/20 split) now loads clean point clouds.**

- `cfgs/dataset/TreeDataset.yaml` no longer adds Gaussian noise (`noise_std: 0.20`) to every training and validation sample.
- Results on this split will differ from v0.6.0 and earlier, including the paper. The 5-fold CV split was never affected.
- Noise can still be enabled explicitly with `noise_std`, for robustness studies.

## v0.6.0

**Highlights**

- **PointMamba** (NeurIPS 2024): ShapeNet-55 pre-training, ModelNet40, few-shot and ScanObjectNN (OBJ-BG, OBJ-ONLY, PB-T50-RS). The official checkpoints load unchanged and give the published 93.5% on ModelNet40.
- **PyTorch 2.8 + CUDA 12.8** support, alongside PyTorch 2.4.1 + CUDA 11.8.
- **New configs:** PointMLP-elite and RepSurf-U (2×) on ModelNet40 and STPCTLS.
- **CI:** the test suite runs on every push and pull request (torch 2.4.1 and 2.8.0).

**Fixes that change results — please re-run**

- **ReCon** and **PointGPT + DAPT**: their transformer blocks did not match the original implementations, so pretrained weights ran through a different network. Both now match the original code exactly.
- **Few-shot**: the loader used all 8192 stored points instead of `N_POINTS` (1024).
- **ModelNet40**: configs failed to load the dataset; fixed for every model.

**Verified against the original repositories**

Point-MAE, Point-BERT, PointGPT, ACT, ReCon, PCP-MAE, Point-M2AE and PointMamba, plus the IDPT, VPT-Deep, PPT, DAPT and PointGST strategies. Fine-tuning, pre-training and checkpoint loading all match the original code.

**Other changes**

- The library stays importable without the CUDA extensions, with clear error messages.
- Checkpoints load on torch ≥ 2.6 (`weights_only` default).
- Removed duplicated metrics: `recall_macro` (equal to balanced accuracy) and `recall_weighted` (equal to accuracy).
- Folder names fixed: `scripts/visualization/`, `models/randlanet/`, `models/repsurf/`. Config names are unchanged.
- Fixes to the visualization scripts, PointTransformerV2, PointCNN and `install_extensions.sh`.
- Removed the HELIALS dataset and the unused raw-format ShapeNetParts loader.

**Breaking changes**

- `recall_macro` and `recall_weighted` are no longer in the metric CSVs.
- Import paths `models.randlenet` → `models.randlanet` and `models.rapsurf` → `models.repsurf`.
- The HELIALS dataset and the raw-format `ShapeNetParts` loader were removed.

## v0.5.0

The version used for the results in the [paper](https://arxiv.org/abs/2604.10780). To reproduce them exactly, check out tag [`v0.5.0`](https://github.com/said-ohamouddou/LIDARLearn/tree/v0.5.0).
