#!/bin/bash
# ============================================================
# Train all models on STPCTLS with cross-validation
# Includes: supervised, SSL (full FT), and PEFT strategies
# Usage: bash scripts/train_stpctls_cv.sh [--seed 42] [--max_epoch 250]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-10}
AUGMENTATION=${AUGMENTATION:-none}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --max_epoch) MAX_EPOCH="$2"; shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " STPCTLS Cross-Validation - All Models"
echo " Seed: $SEED | Epochs: $MAX_EPOCH | Aug: $AUGMENTATION"
echo "============================================================"

COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation $AUGMENTATION --run_all_folds"

# ────────────────────────────────────────────────
#  SUPERVISED MODELS
# ────────────────────────────────────────────────

echo ""; echo "=== Supervised Models ==="

echo "[1/31] PointNet"
python main.py --config cfgs/classification/PointNet/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointnet_stpctls_cv

echo "[2/31] PointNet++ (SSG)"
python main.py --config cfgs/classification/PointNet2/STPCTLS/stpctls_cv_ssg.yaml $COMMON --exp_name pointnet2_ssg_stpctls_cv

echo "[3/31] PointNet++ (MSG)"
python main.py --config cfgs/classification/PointNet2/STPCTLS/stpctls_cv_msg.yaml $COMMON --exp_name pointnet2_msg_stpctls_cv

echo "[4/31] DGCNN"
python main.py --config cfgs/classification/DGCNN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name dgcnn_stpctls_cv

echo "[5/31] PCT"
python main.py --config cfgs/classification/PCT/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pct_stpctls_cv

echo "[6/31] PointMLP"
python main.py --config cfgs/classification/PointMLP/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointmlp_stpctls_cv

echo "[7/31] PointMLP-elite"
python main.py --config cfgs/classification/PointMLP/STPCTLS/stpctls_cv_elite.yaml $COMMON --exp_name pointmlp_elite_stpctls_cv

echo "[8/31] CurveNet"
python main.py --config cfgs/classification/CurveNet/STPCTLS/stpctls_cv.yaml $COMMON --exp_name curvenet_stpctls_cv

echo "[9/31] DeepGCN"
python main.py --config cfgs/classification/DeepGCN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name deepgcn_stpctls_cv

echo "[10/31] DELA"
python main.py --config cfgs/classification/DELA/STPCTLS/stpctls_cv.yaml $COMMON --exp_name dela_stpctls_cv

echo "[11/31] RSCNN"
python main.py --config cfgs/classification/RSCNN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name rscnn_stpctls_cv

echo "[12/31] PointConv"
python main.py --config cfgs/classification/PointConv/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointconv_stpctls_cv

echo "[13/31] PointWeb"
python main.py --config cfgs/classification/PointWeb/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointweb_stpctls_cv

echo "[14/31] SO-Net"
python main.py --config cfgs/classification/SONet/STPCTLS/stpctls_cv.yaml $COMMON --exp_name sonet_stpctls_cv

echo "[15/31] RepSurf"
python main.py --config cfgs/classification/RepSurf/STPCTLS/stpctls_cv.yaml $COMMON --exp_name repsurf_stpctls_cv

echo "[16/31] RepSurf-U (2x)"
python main.py --config cfgs/classification/RepSurf/STPCTLS/stpctls_cv_2x.yaml $COMMON --exp_name repsurf_2x_stpctls_cv

echo "[17/31] PointCNN"
python main.py --config cfgs/classification/PointCNN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointcnn_stpctls_cv

echo "[18/31] PointSCNet"
python main.py --config cfgs/classification/PointSCNet/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointscnet_stpctls_cv

echo "[19/31] GDANet"
python main.py --config cfgs/classification/GDAN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name gdan_stpctls_cv

echo "[20/31] PPFNet"
python main.py --config cfgs/classification/PPFNet/STPCTLS/stpctls_cv.yaml $COMMON --exp_name ppfnet_stpctls_cv

echo "[21/31] PVT"
python main.py --config cfgs/classification/PVT/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pvt_stpctls_cv

echo "[22/31] PointTransformer"
python main.py --config cfgs/classification/PointTransformer/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointtransformer_stpctls_cv

echo "[23/31] PointTransformerV2"
python main.py --config cfgs/classification/PointTransformerV2/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointtransformerv2_stpctls_cv

echo "[24/31] PointTransformerV3"
python main.py --config cfgs/classification/PointTransformerV3/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointtransformerv3_stpctls_cv

echo "[25/31] P2P"
python main.py --config cfgs/classification/P2P/STPCTLS/stpctls_cv.yaml $COMMON --exp_name p2p_stpctls_cv

echo "[26/31] PointTNT"
python main.py --config cfgs/classification/PointTNT/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointtnt_stpctls_cv

echo "[27/31] GlobalTransformer"
python main.py --config cfgs/classification/GlobalTransformer/STPCTLS/stpctls_cv.yaml $COMMON --exp_name globaltransformer_stpctls_cv

echo "[28/31] PointKAN"
python main.py --config cfgs/classification/PointKAN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name pointkan_stpctls_cv

echo "[29/31] MSDGCNN"
python main.py --config cfgs/classification/MSDGCNN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name msdgcnn_stpctls_cv

echo "[30/31] MSDGCNN2"
python main.py --config cfgs/classification/MSDGCNN2/STPCTLS/stpctls_cv.yaml $COMMON --exp_name msdgcnn2_stpctls_cv

echo "[31/31] KAN-DGCNN"
python main.py --config cfgs/classification/KANDGCNN/STPCTLS/stpctls_cv.yaml $COMMON --exp_name kandgcnn_stpctls_cv

# ────────────────────────────────────────────────
#  SSL MODELS — Full Fine-tuning (CV)
# ────────────────────────────────────────────────

echo ""; echo "=== SSL Models (Full Fine-tuning) ==="

echo "[SSL 1/7] PointMAE"
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_stpctls_cv

echo "[SSL 2/7] PointBERT"
python main.py --config cfgs/classification/PointBERT/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_bert.pth --exp_name pointbert_stpctls_cv

echo "[SSL 3/7] PointGPT"
python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_stpctls_cv

echo "[SSL 4/7] ACT"
python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_stpctls_cv

echo "[SSL 5/7] ReCon"
python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_stpctls_cv

echo "[SSL 6/7] PCP-MAE"
python main.py --config cfgs/classification/PCPMAE/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_pcp.pth --exp_name pcpmae_stpctls_cv

echo "[SSL 7/7] Point-M2AE"
python main.py --config cfgs/classification/PointM2AE/STPCTLS/stpctls_cv.yaml $COMMON --ckpts pretrained/pretrained_m2ae.pth --exp_name pointm2ae_stpctls_cv

# ────────────────────────────────────────────────
#  PEFT STRATEGIES (PointMAE, PointGPT, ACT, ReCon) — CV
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: PointMAE ==="

python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_idpt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_idpt_stpctls_cv
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_vpt_deep_stpctls_cv
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_dapt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_dapt_stpctls_cv
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_ppt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_ppt_stpctls_cv
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_gst.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_gst_stpctls_cv

echo ""; echo "=== PEFT: PointGPT ==="

python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv_idpt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_idpt_stpctls_cv
python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_vpt_deep_stpctls_cv
python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv_dapt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_dapt_stpctls_cv
python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv_ppt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_ppt_stpctls_cv
python main.py --config cfgs/classification/PointGPT/STPCTLS/stpctls_cv_gst.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_gst_stpctls_cv

echo ""; echo "=== PEFT: ACT ==="

python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv_idpt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_idpt_stpctls_cv
python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_vpt_deep_stpctls_cv
python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv_dapt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_dapt_stpctls_cv
python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv_ppt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_ppt_stpctls_cv
python main.py --config cfgs/classification/ACT/STPCTLS/stpctls_cv_gst.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_gst_stpctls_cv

echo ""; echo "=== PEFT: ReCon ==="

python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv_idpt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_idpt_stpctls_cv
python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_vpt_deep_stpctls_cv
python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv_dapt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_dapt_stpctls_cv
python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv_ppt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_ppt_stpctls_cv
python main.py --config cfgs/classification/RECON/STPCTLS/stpctls_cv_gst.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_gst_stpctls_cv

echo ""
echo "============================================================"
echo " STPCTLS Cross-Validation Complete"
echo "============================================================"
