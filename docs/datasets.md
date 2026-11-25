# Datasets

We adopt the dataset setup from [Point-BERT](https://github.com/Julie-tang00/Point-BERT).

LIDARLearn expects every dataset under a single top-level `data/` directory.
The overall layout is:

```
LIDARLearn/
├── cfgs/
├── datasets/
├── data/
│   ├── ModelNet/
│   ├── ModelNetFewshot/
│   ├── ScanObjectNN/
│   ├── ShapeNet55-34/
│   ├── ShapeNetPart/
│   ├── S3DIS/
│   └── STPCTLS/
└── ...
```

Each subsection below lists the download source and the exact on-disk
layout the dataloaders in [datasets/](https://github.com/said-ohamouddou/LIDARLearn/tree/main/datasets) and the YAML configs in
[cfgs/dataset/](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/dataset) expect.

## Which datasets need preprocessing?

Most benchmarks ship ready-to-use — only two need a one-time preprocessing
pass before you can train on them:

| Dataset | Preprocessing | Script |
|---|:-:|---|
| ModelNet40 / ModelNet-FewShot / ScanObjectNN / ShapeNet-55 / ShapeNetPart | — | none |
| **S3DIS** | ✓ | [preprocessing/preprocess_s3dis.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/preprocessing/preprocess_s3dis.py) |
| **STPCTLS** | ✓ | [preprocessing/preprocess_stpctls.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/preprocessing/preprocess_stpctls.py) |

Both scripts are idempotent (skip already-processed files) and write
their outputs under `data/`, where the loaders expect them.

---

## ModelNet40

- **Download:** [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/)
- **Config:** [cfgs/dataset/ModelNet40Dataset.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/ModelNet40Dataset.yaml)
- **Loader:** [datasets/ModelNetDataset.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/ModelNetDataset.py)

Unpack the archive under `data/ModelNet/` so that the final layout is:

```
data/ModelNet/
└── modelnet40_normal_resampled/
    ├── modelnet40_shape_names.txt
    ├── modelnet40_train.txt
    ├── modelnet40_test.txt
    ├── modelnet40_train_8192pts_fps.dat
    └── modelnet40_test_8192pts_fps.dat
```

- The `.dat` files hold 8192 farthest-point-sampled (FPS) points per shape, in sampling order.
- Configs with `N_POINTS: 1024` use the first 1024 of them, which is an FPS sample of 1024 points. No raw per-shape files are needed.
- ModelNet40 has no validation split: `subset: val` uses the official test split.

**Alternative source (raw files).** If the Tsinghua Cloud link is unavailable, use the Pointcept mirror on Hugging Face: [`Pointcept/modelnet40_normal_resampled-compressed`](https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed). It holds the raw `modelnet40_normal_resampled` files (10,000 points per shape, with normals).

The archive has no top-level folder, so extract it *inside* the target directory:

```bash
mkdir -p data/ModelNet/modelnet40_normal_resampled
tar -xzf modelnet40_normal_resampled.tar.gz -C data/ModelNet/modelnet40_normal_resampled
```

- On the first run, the loader samples each shape with FPS and caches the result as `modelnet40_{train,test}_<N_POINTS>pts_fps.dat` next to the raw files. This one-time step takes about an hour for 1024 points.
- The resulting data reproduces published accuracy: the official PointMamba checkpoint scores 93.5% on it.
- The dataset comes from the [Princeton ModelNet project](https://modelnet.cs.princeton.edu/); please cite it and follow its terms of use.

---

## ModelNet Few-Shot

- **Download:** [Google Drive](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW)
- **Loader:** [datasets/ModelNetDatasetFewShot.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/ModelNetDatasetFewShot.py)

Put the `.pkl` split files under `data/ModelNetFewshot/` grouped by
`{way}way_{shot}shot`:

```
data/ModelNetFewshot/
├── 5way_10shot/
│   ├── 0.pkl
│   ├── ...
│   └── 9.pkl
├── 5way_20shot/
├── 10way_10shot/
└── 10way_20shot/
```

---

## ScanObjectNN

- **Download:** [official website](https://hkust-vgd.github.io/scanobjectnn/) (subject to the dataset's terms of use)
- **Configs:** [ScanObjectNN_objectbg.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/ScanObjectNN_objectbg.yaml) (OBJ-BG), [ScanObjectNN_objectonly.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/ScanObjectNN_objectonly.yaml) (OBJ-ONLY), [ScanObjectNN_hardest.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/ScanObjectNN_hardest.yaml) (PB-T50-RS)
- **Loader:** [datasets/ScanObjectNNDataset.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/ScanObjectNNDataset.py)
- **Used by:** PointMamba ([cfgs/classification/PointMamba/ScanObjectNN/](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointMamba/ScanObjectNN))

Unpack the official h5 files under `data/ScanObjectNN/`:

```
data/ScanObjectNN/
├── main_split/
│   ├── training_objectdataset.h5
│   ├── test_objectdataset.h5
│   ├── training_objectdataset_augmentedrot_scale75.h5
│   └── test_objectdataset_augmentedrot_scale75.h5
└── main_split_nobg/
    ├── training_objectdataset.h5
    └── test_objectdataset.h5
```

---

## ShapeNet55 / ShapeNet34

- **Download:** [Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view)
- **Loader:** [datasets/ShapeNet55Dataset.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/ShapeNet55Dataset.py), [datasets/ShapeNet55DatasetPretrain.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/ShapeNet55DatasetPretrain.py)

Unzip under `data/ShapeNet55-34/`:

```
data/ShapeNet55-34/
├── ShapeNet-55/
│   ├── train.txt
│   └── test.txt
└── shapenet_pc/
    ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
    ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
    └── ...
```

---

## ShapeNetPart

- **Download:** [Google Drive](https://drive.google.com/drive/folders/1S4TXe0GNOJBWoWEjMVltap29GE5N4b5_?usp=sharing)
- **Config:** [cfgs/dataset/ShapeNetPartsH5.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/ShapeNetPartsH5.yaml)

Place the HDF5 release so the final layout is:

```
data/ShapeNetPart/
└── shapenet_part_seg_hdf5_data/
    └── hdf5_data/
        ├── ply_data_train0.h5 ... ply_data_train5.h5
        ├── ply_data_val0.h5
        └── ply_data_test0.h5, ply_data_test1.h5
```

---

## S3DIS (Stanford3D)

- **Download (request form):** [goo.gl/forms](https://goo.gl/forms/4SoGp4KtH1jfRqEj2)
- **Config:** [cfgs/dataset/S3DIS.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/S3DIS.yaml)
- **Loader:** [datasets/S3DISDataset.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/S3DISDataset.py)
- **Preprocessor:** [preprocessing/preprocess_s3dis.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/preprocessing/preprocess_s3dis.py)

**Step 1.** Unpack the raw release under `data/Stanford3dDataset_v1.2_Aligned_Version/`:

```
data/Stanford3dDataset_v1.2_Aligned_Version/
├── Area_1/
├── Area_2/
├── Area_3/
├── Area_4/
├── Area_5/
└── Area_6/
```

**Step 2.** Run the preprocessor (one-time, ~5 min, produces ~5 GB of `.npy`):

```bash
python preprocessing/preprocess_s3dis.py
```

This merges every per-object `.txt` in each room's `Annotations/` into one
`[N, 7]` array (`x y z r g b label`, 13-class standard mapping with
`stairs → clutter`) and writes it to `data/s3dis_npy/`:

```
data/s3dis_npy/
├── Area_1_conferenceRoom_1.npy
├── Area_1_conferenceRoom_2.npy
└── ...   (~272 rooms total)
```

The loader reads exclusively from `s3dis_npy/` — you don't need to keep the
raw `.txt` tree around after preprocessing if disk is tight. Re-running the
script skips rooms whose `.npy` already exists, so it's safe to resume.

---

## STPCTLS

- **Download:** [Göttingen Research Online](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/FOHUJM) (DOI: 10.25625/FOHUJM)
- **Configs:** [cfgs/dataset/TreeDataset.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/TreeDataset.yaml), [cfgs/dataset/TreeDatasetCV.yaml](https://github.com/said-ohamouddou/LIDARLearn/blob/main/cfgs/dataset/TreeDatasetCV.yaml)
- **Loaders:** [datasets/TreeSpeciesDataset.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/TreeSpeciesDataset.py), [datasets/TreeSpeciesDatasetCV.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/datasets/TreeSpeciesDatasetCV.py)
- **Preprocessor:** [preprocessing/preprocess_stpctls.py](https://github.com/said-ohamouddou/LIDARLearn/blob/main/preprocessing/preprocess_stpctls.py)

**Step 1.** Place the raw per-species TLS tree clouds under `preprocessing/STPCTLS/`,
one folder per species, one `.pts`/`.txt`/`.xyz` file per tree:

```
preprocessing/STPCTLS/
├── Buche/      (164 .pts/.txt files)
├── Douglasie/  (183 files)
├── Eiche/      (22 files)
├── Esche/      (40 files)
├── Fichte/     (158 files)
├── Kiefer/     (25 files)
└── Roteiche/   (100 files)
```

**Step 2.** Run the preprocessor (GPU FPS via `pointnet2_ops`, ~1 min):

```bash
python preprocessing/preprocess_stpctls.py \
    --input_dir preprocessing/STPCTLS \
    --output_dir data/STPCTLS
```

Per tree: if `n ≥ 1024` → GPU FPS to 1024 points; if `n < 1024` → cyclic
duplication up to 1024 (no tree is dropped). Output layout:

```
data/STPCTLS/
├── Buche/*.xyz      # each file: "x y z" header + 1024 rows
├── Douglasie/*.xyz
├── Eiche/*.xyz
├── Esche/*.xyz
├── Fichte/*.xyz
├── Kiefer/*.xyz
└── Roteiche/*.xyz
```

**Step 3.** On the **first training run**, each loader auto-builds its own
h5 cache (shared `point_cloud_data.h5` + per-loader split file). You don't
need to do this manually — just launch training:

```bash
# TreeSpeciesDataset writes data_split_simple.h5 (80/20 stratified)
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls.yaml \
    --seed 42 --mode finetune --exp_name stpctls_run1

# TreeSpeciesDatasetCV writes data_split_cv.h5 (5-fold stratified indices)
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv.yaml \
    --seed 42 --mode finetune --run_all_folds --exp_name stpctls_cv_run1
```

After both have run once, `data/STPCTLS/` also contains:

```
data/STPCTLS/
├── point_cloud_data.h5       # shared (N, 1024, 3) cache
├── data_split_simple.h5      # train/val split (TreeSpeciesDataset)
└── data_split_cv.h5          # k-fold indices (TreeSpeciesDatasetCV)
```

Both loaders auto-rebuild their split file if you change `--seed` or
`K_FOLDS` between runs.
