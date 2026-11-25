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

## Point-based

| Model | Paper | Venue | Classification | Part seg | Sem seg | Pre-training | PEFT | Configs |
|---|---|---|---|:-:|:-:|:-:|---|---|
| **DeLA** | Decoupled Local Aggregation for Point Cloud Learning | arXiv 2023 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`DELA`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/DELA) |
| **KAN-PointNet** | PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets | Computers & Graphics 2025 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointKAN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointKAN) |
| **PointCNN** | PointCNN: Convolution On X-Transformed Points | NeurIPS 2018 | ModelNet40, ModelNetFewShot, STPCTLS | — | — | — | — | [`PointCNN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointCNN) |
| **PointConv** | PointConv: Deep Convolutional Networks on 3D Point Clouds | CVPR 2019 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointConv`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointConv) |
| **PointMLP** | Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework | ICLR 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointMLP`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointMLP) |
| **PointNet** | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | CVPR 2017 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointNet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointNet) |
| **PointNet++** | PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | NeurIPS 2017 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointNet2`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointNet2) |
| **PointSCNet** | PointSCNet: Point Cloud Structure and Correlation Learning Based on Space-Filling Curve-Guided Sampling | Symmetry 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointSCNet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointSCNet) |
| **PointWeb** | PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing | CVPR 2019 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointWeb`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointWeb) |
| **PPFNet** | PPFNet: Global Context Aware Local Features for Robust 3D Point Matching | CVPR 2018 | ModelNet40, ModelNetFewShot, STPCTLS | — | — | — | — | [`PPFNet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PPFNet) |
| **RandLA-Net** | RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds | CVPR 2020 | — | ✓ | ✓ | — | — | [`RandLANet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/segmentation/RandLANet) |
| **RepSurf** | Surface Representation for Point Clouds | CVPR 2022 (Oral) | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`RepSurf`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/RepSurf) |
| **RS-CNN** | Relation-Shape Convolutional Neural Network for Point Cloud Analysis | CVPR 2019 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`RSCNN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/RSCNN) |
| **SO-Net** | SO-Net: Self-Organizing Network for Point Cloud Analysis | CVPR 2018 | ModelNet40, ModelNetFewShot, STPCTLS | — | — | — | — | [`SONet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/SONet) |

## Attention-based

| Model | Paper | Venue | Classification | Part seg | Sem seg | Pre-training | PEFT | Configs |
|---|---|---|---|:-:|:-:|:-:|---|---|
| **Global Transformer** | Points to Patches: Enabling the Use of Self-Attention for 3D Shape Recognition | ICPR 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`GlobalTransformer`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/GlobalTransformer) |
| **P2P** | P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting | NeurIPS 2022 Spotlight | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`P2P`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/P2P) |
| **PCT** | PCT: Point Cloud Transformer | CVM 2021 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PCT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PCT) |
| **Point Transformer** | Point Transformer | ICCV 2021 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointTransformer`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointTransformer) |
| **Point Transformer v2** | Point Transformer V2: Grouped Vector Attention and Partition-based Pooling | NeurIPS 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointTransformerV2`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointTransformerV2) |
| **Point Transformer v3** | Point Transformer V3: Simpler, Faster, Stronger | CVPR 2024 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointTransformerV3`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointTransformerV3) |
| **Point-TNT** | Points to Patches: Enabling the Use of Self-Attention for 3D Shape Recognition | ICPR 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PointTNT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointTNT) |
| **PVT** | PVT: Point-Voxel Transformer for Point Cloud Learning | International Journal of Intelligent Systems 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`PVT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PVT) |

## Graph-based

| Model | Paper | Venue | Classification | Part seg | Sem seg | Pre-training | PEFT | Configs |
|---|---|---|---|:-:|:-:|:-:|---|---|
| **CurveNet** | Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis | ICCV 2021 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`CurveNet`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/CurveNet) |
| **DeepGCNs** | DeepGCNs: Can GCNs Go as Deep as CNNs? | ICCV 2019 (Oral) | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`DeepGCN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/DeepGCN) |
| **DGCNN** | Dynamic Graph CNN for Learning on Point Clouds | ACM TOG 2019 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`DGCNN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/DGCNN) |
| **GDANet** | Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud | AAAI 2021 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`GDAN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/GDAN) |
| **KAN-DGCNN** | Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds | Pattern Recognition 2026 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`KANDGCNN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/KANDGCNN) |
| **MS-DGCNN** | Multi-Scale Dynamic Graph Convolution Network for Point Clouds Classification | IEEE Access 2020 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`MSDGCNN`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/MSDGCNN) |
| **MS-DGCNN++** | MS-DGCNN++: Multi-Scale Dynamic Graph Convolution with Scale-Dependent Normalization for Robust LiDAR Tree Species Classification | arXiv 2025 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | — | — | [`MSDGCNN2`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/MSDGCNN2) |

## Self-supervised

| Model | Paper | Venue | Classification | Part seg | Sem seg | Pre-training | PEFT | Configs |
|---|---|---|---|:-:|:-:|:-:|---|---|
| **ACT** | Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning? | ICLR 2023 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | DAPT, IDPT, PPT, GST, VPT-Deep | [`ACT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/ACT) |
| **PCP-MAE** | PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders | NeurIPS 2024 Spotlight | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | — | [`PCPMAE`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PCPMAE) |
| **Point-BERT** | Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling | CVPR 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | — | [`PointBERT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointBERT) |
| **Point-M2AE** | Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training | NeurIPS 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | — | [`PointM2AE`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointM2AE) |
| **Point-MAE** | Masked Autoencoders for Point Cloud Self-supervised Learning | ECCV 2022 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | DAPT, IDPT, PPT, GST, VPT-Deep | [`PointMAE`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointMAE) |
| **PointGPT** | PointGPT: Auto-regressively Generative Pre-training from Point Clouds | NeurIPS 2023 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | DAPT, IDPT, PPT, GST, VPT-Deep | [`PointGPT`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointGPT) |
| **PointMamba** | PointMamba: A Simple State Space Model for Point Cloud Analysis | NeurIPS 2024 | ModelNet40, ModelNetFewShot, ScanObjectNN | — | — | ✓ | — | [`PointMamba`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/PointMamba) |
| **ReCon** | Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining | ICML 2023 | ModelNet40, ModelNetFewShot, STPCTLS | ✓ | ✓ | ✓ | DAPT, IDPT, PPT, GST, VPT-Deep | [`RECON`](https://github.com/said-ohamouddou/LIDARLearn/tree/main/cfgs/classification/RECON) |

## Fine-tuning strategies

| Strategy | Paper | Venue |
|---|---|---|
| **DAPT** | Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis | CVPR 2024 |
| **IDPT** | Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models | ICCV 2023 |
| **PPT** | Positional Prompt Tuning for Efficient 3D Representation Learning | ACM MM 2025 |
| **GST (PointGST)** | Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning | IEEE TPAMI 2025 |
| **VPT-Deep** | Visual Prompt Tuning | ECCV 2022 |
