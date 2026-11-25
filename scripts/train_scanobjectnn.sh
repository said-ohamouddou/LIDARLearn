#!/bin/bash
# ============================================================
# Train on ScanObjectNN (classification): OBJ-BG, OBJ-ONLY, PB-T50-RS
# Models: PointMamba (the only model with ScanObjectNN configs, as in its
#         official repository; needs the optional mamba-ssm package)
# Usage: bash scripts/train_scanobjectnn.sh [--seed 42] [--max_epoch 300]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-300}
# Upstream runner_finetune.py rotates ScanObjectNN batches about the y axis.
AUGMENTATION=${AUGMENTATION:-y_rotate}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)        SEED="$2";        shift 2 ;;
        --max_epoch)   MAX_EPOCH="$2";   shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " ScanObjectNN - PointMamba"
echo " Seed: $SEED | Epochs: $MAX_EPOCH | Aug: $AUGMENTATION"
echo "============================================================"

COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation $AUGMENTATION"
CKPT=pretrained/pointmamba_pretrain.pth

echo "[1/3] PointMamba OBJ-BG"
python main.py --config cfgs/classification/PointMamba/ScanObjectNN/scanobjectnn_objbg.yaml $COMMON --ckpts $CKPT --exp_name pointmamba_scanobjectnn_objbg

echo "[2/3] PointMamba OBJ-ONLY"
python main.py --config cfgs/classification/PointMamba/ScanObjectNN/scanobjectnn_objonly.yaml $COMMON --ckpts $CKPT --exp_name pointmamba_scanobjectnn_objonly

echo "[3/3] PointMamba PB-T50-RS (hardest)"
python main.py --config cfgs/classification/PointMamba/ScanObjectNN/scanobjectnn_hardest.yaml $COMMON --ckpts $CKPT --exp_name pointmamba_scanobjectnn_hardest
