# Pretrained SSL Weights

The 8 self-supervised backbones (Point-MAE, Point-BERT, PointGPT, ACT, ReCon, PCP-MAE, Point-M2AE, PointMamba) need a pretrained checkpoint to fine-tune from. The first seven are published together on Google Drive:

**Download:** [Google Drive — LIDARLearn pretrained weights](https://drive.google.com/drive/folders/1u0EbvwVji_fAMD2GghoNviyzyXxxaUMa?usp=sharing)

Place every downloaded `.pth` into `pretrained/` at the repo root:

```
pretrained/
├── pretrained_mae.pth     # Point-MAE    — used by cfgs/classification/PointMAE/**
├── pretrained_bert.pth    # Point-BERT   — cfgs/classification/PointBERT/**
├── pretrained_gpt.pth     # PointGPT     — cfgs/classification/PointGPT/**
├── pretrained_act.pth     # ACT          — cfgs/classification/ACT/**
├── pretrained_recon.pth   # ReCon        — cfgs/classification/RECON/**
├── pretrained_pcp.pth     # PCP-MAE      — cfgs/classification/PCPMAE/**
├── pretrained_m2ae.pth    # Point-M2AE   — cfgs/classification/PointM2AE/**
├── act_dvae.pth           # ACT dVAE tokeniser (needed only for ACT pre-training)
├── dVAE.pth               # Point-BERT dVAE tokeniser (needed only for Point-BERT pre-training)
└── pointmamba_pretrain.pth  # PointMamba — cfgs/classification/PointMamba/**
```

PointMamba's checkpoints come from its [official release](https://github.com/LMD0311/PointMamba/releases/tag/ckpts-2) and load unchanged: [`pretrain.pth`](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/pretrain.pth) (save as `pretrained/pointmamba_pretrain.pth`) to fine-tune; the authors' ModelNet40 fine-tuned weights [`finetune_modelnet.pth`](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_modelnet.pth) (save as `pretrained/pointmamba_finetune_modelnet.pth`) also load unchanged, e.g. for inference with `load_model_from_ckpt`.

Pass the relevant checkpoint via `--ckpts pretrained/<name>.pth` when fine-tuning, e.g.:

```bash
python main.py --config cfgs/classification/PointMAE/STPCTLS/stpctls_cv_dapt.yaml \
    --ckpts pretrained/pretrained_mae.pth --run_all_folds --seed 42
```

Each loader strips its checkpoint's state-dict prefix (`MAE_encoder.`, `ACT_encoder.`, `transformer_q.`, `GPT_Transformer.`, …), so checkpoints from different SSL methods map cleanly onto the fine-tuning model.

[`tests/test_pretrained_load.py`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/tests/test_pretrained_load.py) checks that every loader actually loads its checkpoint, which catches silent prefix mismatches.

You can skip this section entirely if you only plan to train the 31 supervised backbones — they don't need any pretrained weights.
