#!/bin/bash
# ============================================================
# Few-Shot Classification on ModelNet40 — ALL MODELS + PEFT
# Runs every classification model on all 4 (way, shot) combos
# using --run_all_folds (10 folds per combo).
#
# Also sweeps PEFT fine-tuning strategies (DAPT / IDPT / PPT /
# GST / VPT-Deep) on the SSL backbones that support them:
#   PointMAE, PointGPT, ACT, RECON
#
# Usage:
#   bash scripts/train_fewshot.sh [--seed 42] [--max_epoch 150] [--combos "5w10s ..."] \
#                                 [--models "pointnet dgcnn ..."] \
#                                 [--strategies "ff dapt idpt ppt gst vpt_deep"]
#
# Examples:
#   # Full sweep: 36 models (FF) + 4 SSL × 5 PEFT = 56 (model,strategy) pairs
#   # × 4 combos = 224 runs, each over 10 folds
#   bash scripts/train_fewshot.sh
#
#   # FF only (no PEFT)
#   bash scripts/train_fewshot.sh --strategies ff
#
#   # PEFT only on PointMAE
#   bash scripts/train_fewshot.sh --models pointmae --strategies "dapt idpt ppt gst vpt_deep"
#
#   # SSL models, 5-way only
#   bash scripts/train_fewshot.sh --models "pointmae pointbert pointgpt act recon pcpmae pointm2ae pointmamba" \
#                                 --combos "5w10s 5w20s"
#
# After all runs finish, aggregate into a LaTeX/Markdown/CSV table:
#   python scripts/generate_fewshot_table.py --exp_dir experiments/ModelNetFewShot
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-2}
COMBOS_DEFAULT="5w10s 5w20s 10w10s 10w20s"

# Every classification model that has cfgs/classification/<Dir>/ModelNetFewShot/
# Ordered to roughly match the library's canonical grouping
# (point-based -> attention -> graph -> SSL).
MODELS_DEFAULT="\
pointnet pointnet2 sonet ppfnet pointcnn pointweb pointconv rscnn \
pointmlp pointscnet repsurf pointkan dela \
pct p2p pointtnt globaltransformer pvt pointtransformer pointtransformerv2 pointtransformerv3 \
dgcnn deepgcn curvenet gdan msdgcnn kandgcnn msdgcnn2 \
pointmae act recon pointgpt pointm2ae pointbert pcpmae pointmamba"

# Fine-tuning strategies. 'ff' = Full Finetuning (the plain modelnet_fewshot_*w*s.yaml).
# PEFT strategies only run on SSL models that have matching few-shot configs.
STRATEGIES_DEFAULT="ff dapt idpt ppt gst vpt_deep"

COMBOS="$COMBOS_DEFAULT"
MODELS="$MODELS_DEFAULT"
STRATEGIES="$STRATEGIES_DEFAULT"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)       SEED="$2";       shift 2 ;;
        --max_epoch)  MAX_EPOCH="$2";  shift 2 ;;
        --combos)     COMBOS="$2";     shift 2 ;;
        --models)     MODELS="$2";     shift 2 ;;
        --strategies) STRATEGIES="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# model key -> config subdirectory under cfgs/classification/
declare -A CFG_DIR=(
    # point-based
    [pointnet]=PointNet
    [pointnet2]=PointNet2
    [sonet]=SONet
    [ppfnet]=PPFNet
    [pointcnn]=PointCNN
    [pointweb]=PointWeb
    [pointconv]=PointConv
    [rscnn]=RSCNN
    [pointmlp]=PointMLP
    [pointscnet]=PointSCNet
    [repsurf]=RepSurf
    [pointkan]=PointKAN
    [dela]=DELA
    # RandLANet intentionally omitted — it is a segmentation-only backbone
    # attention-based
    [pct]=PCT
    [p2p]=P2P
    [pointtnt]=PointTNT
    [globaltransformer]=GlobalTransformer
    [pvt]=PVT
    [pointtransformer]=PointTransformer
    [pointtransformerv2]=PointTransformerV2
    [pointtransformerv3]=PointTransformerV3
    # graph-based
    [dgcnn]=DGCNN
    [deepgcn]=DeepGCN
    [curvenet]=CurveNet
    [gdan]=GDAN
    [msdgcnn]=MSDGCNN
    [kandgcnn]=KANDGCNN
    [msdgcnn2]=MSDGCNN2
    # self-supervised (need pretrained ckpts)
    [pointmae]=PointMAE
    [act]=ACT
    [recon]=RECON
    [pointgpt]=PointGPT
    [pointm2ae]=PointM2AE
    [pointbert]=PointBERT
    [pcpmae]=PCPMAE
    [pointmamba]=PointMamba
)

# Pretrained checkpoint per SSL model (none => train from scratch)
declare -A CKPT=(
    [pointmae]=pretrained/pretrained_mae.pth
    [act]=pretrained/pretrained_act.pth
    [recon]=pretrained/pretrained_recon.pth
    [pointgpt]=pretrained/pretrained_gpt.pth
    [pointm2ae]=pretrained/pretrained_m2ae.pth
    [pointbert]=pretrained/pretrained_bert.pth
    [pcpmae]=pretrained/pretrained_pcp.pth
    [pointmamba]=pretrained/pointmamba_pretrain.pth
)

echo "============================================================"
echo " Few-Shot Classification (ModelNet40) — ALL MODELS + PEFT"
echo " Seed:       $SEED"
echo " Max epoch:  $MAX_EPOCH"
echo " Models:     $(echo $MODELS | wc -w) models"
echo " Strategies: $STRATEGIES"
echo " Combos:     $COMBOS"
echo "============================================================"

COMMON="--run_all_folds --seed $SEED --max_epoch $MAX_EPOCH"

run_count=0
fail_count=0
skip_count=0
failed_runs=()

# Resolve the few-shot config path for a (model, strategy, combo) triple.
# Returns empty string if not found. 'ff' strategy uses the plain config.
resolve_cfg() {
    local cfg_dir="$1"
    local combo="$2"
    local strategy="$3"

    local base="cfgs/classification/${cfg_dir}/ModelNetFewShot/modelnet_fewshot_${combo}"
    if [[ "$strategy" == "ff" ]]; then
        echo "${base}.yaml"
    else
        echo "${base}_${strategy}.yaml"
    fi
}

for model in $MODELS; do
    cfg_dir="${CFG_DIR[$model]:-}"
    if [[ -z "$cfg_dir" ]]; then
        echo "[WARN] Unknown model '$model' — skipping"
        skip_count=$((skip_count + 1))
        continue
    fi

    for strategy in $STRATEGIES; do
        for combo in $COMBOS; do
            cfg_file=$(resolve_cfg "$cfg_dir" "$combo" "$strategy")
            if [[ ! -f "$cfg_file" ]]; then
                # PEFT on a non-SSL model (or missing template) is expected to
                # be absent — skip silently for non-ff strategies.
                if [[ "$strategy" == "ff" ]]; then
                    echo "[WARN] Missing config: $cfg_file — skipping"
                fi
                skip_count=$((skip_count + 1))
                continue
            fi

            if [[ "$strategy" == "ff" ]]; then
                exp_name="${model}_${combo}"
            else
                exp_name="${model}_${strategy}_${combo}"
            fi
            run_count=$((run_count + 1))

            # Optional pretrain checkpoint (SSL models only)
            ckpt_arg=""
            ckpt_info="(from scratch)"
            ckpt="${CKPT[$model]:-}"
            if [[ -n "$ckpt" ]]; then
                if [[ -f "$ckpt" ]]; then
                    ckpt_arg="--ckpts $ckpt"
                    ckpt_info="ckpt=$ckpt"
                else
                    ckpt_info="(ckpt missing: $ckpt — running from scratch)"
                fi
            fi

            echo ""
            echo "------------------------------------------------------------"
            echo "[$run_count] $model / $strategy / $combo"
            echo " config:  $cfg_file"
            echo " exp:     experiments/ModelNetFewShot/$exp_name"
            echo " $ckpt_info"
            echo "------------------------------------------------------------"

            python main.py \
                --config "$cfg_file" \
                $ckpt_arg \
                $COMMON \
                --exp_name "$exp_name"

            status=$?
            if [[ $status -ne 0 ]]; then
                fail_count=$((fail_count + 1))
                failed_runs+=("$exp_name (exit=$status)")
                echo "[FAIL] $exp_name exited with status $status — continuing"
            fi
        done
    done
done

echo ""
echo "============================================================"
echo " Few-Shot Runs Finished"
echo "   total runs:   $run_count"
echo "   failed runs:  $fail_count"
echo "   skipped:      $skip_count"
if [[ $fail_count -gt 0 ]]; then
    echo "   failures:"
    for fr in "${failed_runs[@]}"; do
        echo "     - $fr"
    done
fi
echo "============================================================"
echo ""
echo "Next: aggregate into a LaTeX table"
echo "  python scripts/generate_fewshot_table.py --exp_dir experiments/ModelNetFewShot"
