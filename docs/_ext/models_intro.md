# Supported models

Every model below ships with ready-to-run YAML configs. This page is generated from `cfgs/` at each documentation build, so it always matches the code.

**Columns**

- **Classification**: the datasets with classification configs (ModelNet40, ModelNetFewShot, STPCTLS, ScanObjectNN).
- **Part seg**: part segmentation (ShapeNetParts).
- **Sem seg**: semantic segmentation (S3DIS).
- **Pre-training**: a ShapeNet-55 self-supervised pre-training config.
- **PEFT**: parameter-efficient fine-tuning strategies available on top of the pretrained backbone (full fine-tuning is always available).

**Variants.** Some config folders hold several variants:

- `PointNet2`: PointNet++ SSG and MSG.
- `PointMLP`: PointMLP and PointMLP-elite.
- `RepSurf`: RepSurf-U and RepSurf-U (2×).
- `DGCNN`, `MSDGCNN`: neighbourhood-size (*k*) ablations on ModelNet40.

**PointMamba**

- Ships exactly the tasks of its [official repository](https://github.com/LMD0311/PointMamba): classification on ModelNet40, ModelNetFewShot and ScanObjectNN (OBJ-BG, OBJ-ONLY, PB-T50-RS).
- Needs the optional `mamba-ssm` package (see [Installation](getting-started/installation.md#5b-optional-install-mamba-for-pointmamba)).
- Its pretraining config and `scripts/train_scanobjectnn.sh` use the upstream `y_rotate` augmentation.
- The shared ModelNet40 and few-shot sweeps keep `--augmentation none` for all models. Upstream PointMamba uses random scaling (2/3–3/2) and translation (±0.2) there.
