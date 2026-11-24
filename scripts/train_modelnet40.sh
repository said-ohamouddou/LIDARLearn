#!/bin/bash
# ============================================================
# Train all models on ModelNet40 (classification)
# Includes: supervised, SSL (full FT), and PEFT strategies
# Usage: bash scripts/train_modelnet40.sh [--seed 42] [--max_epoch 250]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-250}
AUGMENTATION=${AUGMENTATION:-none}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)        SEED="$2";        shift 2 ;;
        --max_epoch)   MAX_EPOCH="$2";   shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " ModelNet40 - All Models"
echo " Seed: $SEED | Epochs: $MAX_EPOCH | Aug: $AUGMENTATION"
echo "============================================================"

COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation $AUGMENTATION"

# ────────────────────────────────────────────────
#  SUPERVISED MODELS
# ────────────────────────────────────────────────

echo ""; echo "=== Supervised Models ==="

echo "[1/33] PointNet"
python main.py --config cfgs/classification/PointNet/ModelNet40/modelnet40.yaml $COMMON --exp_name pointnet_modelnet40

echo "[2/33] PointNet++ (SSG)"
python main.py --config cfgs/classification/PointNet2/ModelNet40/modelnet40_ssg.yaml $COMMON --exp_name pointnet2_ssg_modelnet40

echo "[3/33] PointNet++ (MSG)"
python main.py --config cfgs/classification/PointNet2/ModelNet40/modelnet40_msg.yaml $COMMON --exp_name pointnet2_msg_modelnet40

echo "[4/33] DGCNN (k=5)"
python main.py --config cfgs/classification/DGCNN/ModelNet40/modelnet40_k5.yaml $COMMON --exp_name dgcnn_k5_modelnet40

echo "[5/33] DGCNN (k=20)"
python main.py --config cfgs/classification/DGCNN/ModelNet40/modelnet40_k20.yaml $COMMON --exp_name dgcnn_k20_modelnet40

echo "[6/33] DGCNN (k=30)"
python main.py --config cfgs/classification/DGCNN/ModelNet40/modelnet40_k30.yaml $COMMON --exp_name dgcnn_k30_modelnet40

echo "[7/33] MSDGCNN (k=20,30,40)"
python main.py --config cfgs/classification/MSDGCNN/ModelNet40/modelnet40.yaml $COMMON --exp_name msdgcnn_modelnet40

echo "[8/33] MSDGCNN (k=5,20,30)"
python main.py --config cfgs/classification/MSDGCNN/ModelNet40/modelnet40_k5_20_30.yaml $COMMON --exp_name msdgcnn_k5_20_30_modelnet40

echo "[9/33] MSDGCNN2"
python main.py --config cfgs/classification/MSDGCNN2/ModelNet40/modelnet40.yaml $COMMON --exp_name msdgcnn2_modelnet40

echo "[10/33] PCT"
python main.py --config cfgs/classification/PCT/ModelNet40/modelnet40.yaml $COMMON --exp_name pct_modelnet40

echo "[11/33] PointMLP"
python main.py --config cfgs/classification/PointMLP/ModelNet40/modelnet40.yaml $COMMON --exp_name pointmlp_modelnet40

echo "[12/33] CurveNet"
python main.py --config cfgs/classification/CurveNet/ModelNet40/modelnet40.yaml $COMMON --exp_name curvenet_modelnet40

echo "[13/33] DeepGCN"
python main.py --config cfgs/classification/DeepGCN/ModelNet40/modelnet40.yaml $COMMON --exp_name deepgcn_modelnet40

echo "[14/33] DELA"
python main.py --config cfgs/classification/DELA/ModelNet40/modelnet40.yaml $COMMON --exp_name dela_modelnet40

echo "[15/33] RSCNN"
python main.py --config cfgs/classification/RSCNN/ModelNet40/modelnet40.yaml $COMMON --exp_name rscnn_modelnet40

echo "[16/33] PointConv"
python main.py --config cfgs/classification/PointConv/ModelNet40/modelnet40.yaml $COMMON --exp_name pointconv_modelnet40

echo "[17/33] PointWeb"
python main.py --config cfgs/classification/PointWeb/ModelNet40/modelnet40.yaml $COMMON --exp_name pointweb_modelnet40

echo "[18/33] SO-Net"
python main.py --config cfgs/classification/SONet/ModelNet40/modelnet40.yaml $COMMON --exp_name sonet_modelnet40

echo "[19/33] RepSurf"
python main.py --config cfgs/classification/RepSurf/ModelNet40/modelnet40.yaml $COMMON --exp_name repsurf_modelnet40

echo "[20/33] PointCNN"
python main.py --config cfgs/classification/PointCNN/ModelNet40/modelnet40.yaml $COMMON --exp_name pointcnn_modelnet40

echo "[21/33] PointSCNet"
python main.py --config cfgs/classification/PointSCNet/ModelNet40/modelnet40.yaml $COMMON --exp_name pointscnet_modelnet40

echo "[22/33] GDANet"
python main.py --config cfgs/classification/GDAN/ModelNet40/modelnet40.yaml $COMMON --exp_name gdan_modelnet40

echo "[24/33] PPFNet"
python main.py --config cfgs/classification/PPFNet/ModelNet40/modelnet40.yaml $COMMON --exp_name ppfnet_modelnet40

echo "[25/33] PVT"
python main.py --config cfgs/classification/PVT/ModelNet40/modelnet40.yaml $COMMON --exp_name pvt_modelnet40

echo "[26/33] PointTransformer"
python main.py --config cfgs/classification/PointTransformer/ModelNet40/modelnet40.yaml $COMMON --exp_name pointtransformer_modelnet40

echo "[27/33] PointTransformerV2"
python main.py --config cfgs/classification/PointTransformerV2/ModelNet40/modelnet40.yaml $COMMON --exp_name pointtransformerv2_modelnet40

echo "[28/33] PointTransformerV3"
python main.py --config cfgs/classification/PointTransformerV3/ModelNet40/modelnet40.yaml $COMMON --exp_name pointtransformerv3_modelnet40

echo "[29/33] P2P"
python main.py --config cfgs/classification/P2P/ModelNet40/modelnet40.yaml $COMMON --exp_name p2p_modelnet40

echo "[30/33] PointTNT"
python main.py --config cfgs/classification/PointTNT/ModelNet40/modelnet40.yaml $COMMON --exp_name pointtnt_modelnet40

echo "[31/33] GlobalTransformer"
python main.py --config cfgs/classification/GlobalTransformer/ModelNet40/modelnet40.yaml $COMMON --exp_name globaltransformer_modelnet40

echo "[32/33] PointKAN"
python main.py --config cfgs/classification/PointKAN/ModelNet40/modelnet40.yaml $COMMON --exp_name pointkan_modelnet40

echo "[33/33] KAN-DGCNN"
python main.py --config cfgs/classification/KANDGCNN/ModelNet40/modelnet40.yaml $COMMON --exp_name kandgcnn_modelnet40

# ────────────────────────────────────────────────
#  SSL MODELS — Full Fine-tuning
# ────────────────────────────────────────────────

echo ""; echo "=== SSL Models (Full Fine-tuning) ==="

echo "[SSL 1/8] PointMAE"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_modelnet40

echo "[SSL 2/8] ReCon"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_modelnet40

echo "[SSL 3/8] Point-M2AE"
python main.py --config cfgs/classification/PointM2AE/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_m2ae.pth --exp_name pointm2ae_modelnet40

echo "[SSL 4/8] PointBERT"
python main.py --config cfgs/classification/PointBERT/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_bert.pth --exp_name pointbert_modelnet40

echo "[SSL 5/8] PointGPT"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_modelnet40

echo "[SSL 6/8] ACT"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_modelnet40

echo "[SSL 7/8] PCP-MAE"
python main.py --config cfgs/classification/PCPMAE/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pretrained_pcp.pth --exp_name pcpmae_modelnet40

echo "[SSL 8/8] PointMamba (needs the optional mamba-ssm package)"
python main.py --config cfgs/classification/PointMamba/ModelNet40/modelnet40.yaml $COMMON --ckpts pretrained/pointmamba_pretrain.pth --exp_name pointmamba_modelnet40

# ────────────────────────────────────────────────
#  PEFT STRATEGIES — PointMAE backbone
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: PointMAE ==="

echo "[PEFT] PointMAE-IDPT"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40_idpt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_idpt_modelnet40

echo "[PEFT] PointMAE-VPT_Deep"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_vpt_deep_modelnet40

echo "[PEFT] PointMAE-DAPT"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40_dapt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_dapt_modelnet40

echo "[PEFT] PointMAE-PPT"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40_ppt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_ppt_modelnet40

echo "[PEFT] PointMAE-GST"
python main.py --config cfgs/classification/PointMAE/ModelNet40/modelnet40_gst.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_gst_modelnet40

# ────────────────────────────────────────────────
#  PEFT STRATEGIES — ReCon backbone
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: ReCon ==="

echo "[PEFT] ReCon-IDPT"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40_idpt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_idpt_modelnet40

echo "[PEFT] ReCon-VPT_Deep"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_vpt_deep_modelnet40

echo "[PEFT] ReCon-DAPT"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40_dapt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_dapt_modelnet40

echo "[PEFT] ReCon-PPT"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40_ppt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_ppt_modelnet40

echo "[PEFT] ReCon-GST"
python main.py --config cfgs/classification/RECON/ModelNet40/modelnet40_gst.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_gst_modelnet40

# ────────────────────────────────────────────────
#  PEFT STRATEGIES — ACT backbone
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: ACT ==="

echo "[PEFT] ACT-IDPT"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40_idpt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_idpt_modelnet40

echo "[PEFT] ACT-VPT_Deep"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_vpt_deep_modelnet40

echo "[PEFT] ACT-DAPT"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40_dapt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_dapt_modelnet40

echo "[PEFT] ACT-PPT"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40_ppt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_ppt_modelnet40

echo "[PEFT] ACT-GST"
python main.py --config cfgs/classification/ACT/ModelNet40/modelnet40_gst.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_gst_modelnet40

# ────────────────────────────────────────────────
#  PEFT STRATEGIES — PointGPT backbone
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: PointGPT ==="

echo "[PEFT] PointGPT-IDPT"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40_idpt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_idpt_modelnet40

echo "[PEFT] PointGPT-VPT_Deep"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_vpt_deep_modelnet40

echo "[PEFT] PointGPT-DAPT"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40_dapt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_dapt_modelnet40

echo "[PEFT] PointGPT-PPT"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40_ppt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_ppt_modelnet40

echo "[PEFT] PointGPT-GST"
python main.py --config cfgs/classification/PointGPT/ModelNet40/modelnet40_gst.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_gst_modelnet40

echo ""
echo "============================================================"
echo " ModelNet40 Training Complete"
echo "============================================================"
