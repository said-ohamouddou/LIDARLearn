# Quick Start

```bash
# 1. Train PointNet on STPCTLS (simple 80/20 split)
python main.py --config cfgs/classification/PointNet/STPCTLS/stpctls.yaml \
    --mode finetune --seed 42 --exp_name pointnet_stpctls

# 2. Train DGCNN on STPCTLS with 5-fold cross-validation
python main.py --config cfgs/classification/DGCNN/STPCTLS/stpctls_cv.yaml \
    --mode finetune --seed 42 --run_all_folds --exp_name dgcnn_stpctls_cv

# 3. Pre-train Point-MAE on ShapeNet-55
python main.py --config cfgs/classification/PointMAE/ShapeNet55/shapenet55_pretrain.yaml \
    --mode pretrain --exp_name mae_pretrain

# 4. Fine-tune with DAPT on STPCTLS (5-fold CV)
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_dapt.yaml \
    --ckpts pretrained/pretrained_mae.pth --run_all_folds --seed 42

# 5. Generate publication-ready LaTeX tables (outputs land in <exp_dir>/latex/)
python scripts/reports/classification_comparison_report.py --exp_dir experiments/STPCTLS

# 6. Friedman / Nemenyi statistical analysis (outputs land in <exp_dir>/friedman/)
python scripts/reports/friedman_significance_report.py --exp_dir experiments/STPCTLS
```

Sweep the entire benchmark with a single command:

```bash
bash scripts/train_stpctls.sh      # all 58 classification configs
bash scripts/train_fewshot.sh      # few-shot evaluation
bash scripts/train_s3dis.sh        # semantic segmentation
bash scripts/train_shapenetparts.sh # part segmentation
bash scripts/train_scanobjectnn.sh  # ScanObjectNN (PointMamba)
```

## Training a subset of models

Each `scripts/train_*.sh` file is plain bash, with one `python main.py ...` line per model under an `echo "[N/M] <Name>"` header. To run only some models:

- comment out (`#`) the models you want to skip, or
- copy the `python main.py ...` lines you want into your shell.

The report scripts use whichever `cv_summary.csv` files exist under `experiments/<NAME>/`, so partial runs still produce valid tables.
