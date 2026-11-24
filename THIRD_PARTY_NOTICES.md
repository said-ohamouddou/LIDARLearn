# Third-Party Notices

LIDARLearn includes clean-room or adapted implementations of the following
point cloud deep learning models. Each model directory under `models/`
contains a `LICENSE` file with the full license text. Repository URLs below
are the upstream sources referenced in [`docs/point_cloud_methods.csv`](docs/point_cloud_methods.csv) and used as
the basis for adaptation.

## Supervised Models

| Model | Directory | License | Original Repository |
|---|---|---|---|
| PointNet | `models/pointnet/` | MIT | https://github.com/yanx27/Pointnet_Pointnet2_pytorch |
| PointNet++ | `models/pointnet2/` | Unlicense | https://github.com/erikwijmans/Pointnet2_PyTorch |
| DGCNN | `models/dgcnn/` | MIT | https://github.com/WangYueFt/dgcnn |
| PCT | `models/pct/` | MIT | https://github.com/Strawberry-Eat-Mango/PCT_Pytorch |
| PointConv | `models/pointconv/` | MIT | https://github.com/DylanWusee/pointconv |
| PointMLP | `models/pointmlp/` | Apache-2.0 | https://github.com/ma-xu/pointMLP-pytorch |
| CurveNet | `models/curvenet/` | MIT | https://github.com/tiangexiang/CurveNet |
| DeepGCN | `models/deepgcn/` | MIT | https://github.com/lightaime/deep_gcns_torch |
| DELA | `models/dela/` | MIT | https://github.com/Matrix-ASC/DeLA |
| GlobalTransformer | `models/globaltransformer/` | MIT | https://github.com/axeber01/point-tnt |
| GDAN | `models/gdan/` | No license | https://github.com/mutianxu/GDANet |
| KANDGCNN | `models/kandgcnn/` | MIT | https://github.com/said-ohamouddou/STFT-KAN-liteDGCNN |
| MSDGCNN | `models/msdgcnn/` | MIT | https://github.com/said-ohamouddou/MS-DGCNN2 |
| MSDGCNN2 | `models/msdgcnn2/` | MIT | https://github.com/said-ohamouddou/MS-DGCNN2 |
| PointCNN | `models/pointcnn/` | MIT | https://github.com/hxdengBerkeley/PointCNN.Pytorch |
| PointKAN | `models/pointkan/` | MIT | https://github.com/Ali-Stanford/PointNet_KAN_Graphic |
| PointSCNet | `models/pointscnet/` | No license | https://github.com/Chenguoz/PointSCNet |
| PointTNT | `models/pointtnt/` | MIT | https://github.com/axeber01/point-tnt |
| PointTransformer | `models/pointtransformer/` | MIT | https://github.com/Pointcept/Pointcept |
| PointTransformerV2 | `models/pointtransformerv2/` | MIT | https://github.com/Pointcept/Pointcept |
| PointTransformerV3 | `models/pointtransformerv3/` | MIT | https://github.com/Pointcept/Pointcept |
| PointWeb | `models/pointweb/` | MIT | https://github.com/hszhao/PointWeb |
| PPFNet | `models/ppfnet/` | MIT | https://github.com/vinits5/learning3d |
| PVT | `models/pvt/` | MIT | https://github.com/HaochengWan/PVT |
| RandLANet | `models/randlanet/` | No license | https://github.com/aRI0U/RandLA-Net-pytorch |
| RepSurf | `models/repsurf/` | Apache-2.0 | https://github.com/hancyran/RepSurf |
| RSCNN | `models/rscnn/` | MIT | https://github.com/Yochengliu/Relation-Shape-CNN |
| SONet | `models/sonet/` | MIT | https://github.com/lijx10/SO-Net |
| P2P | `models/p2p/` | MIT | https://github.com/wangzy22/P2P |

## Self-Supervised / Pretrained Models

| Model | Directory | License | Original Repository |
|---|---|---|---|
| PointMAE | `models/pointmae/` | MIT | https://github.com/Pang-Yatian/Point-MAE |
| PointBERT | `models/pointbert/` | MIT | https://github.com/Julie-tang00/Point-BERT |
| PointGPT | `models/pointgpt/` | MIT | https://github.com/CGuangyan-BIT/PointGPT |
| ACT | `models/act/` | MIT | https://github.com/RunpeiDong/ACT |
| RECON | `models/recon/` | MIT | https://github.com/qizekun/ReCon |
| PCP-MAE | `models/pcpmae/` | MIT | https://github.com/aHapBean/PCP-MAE |
| Point-M2AE | `models/pointm2ae/` | MIT | https://github.com/ZrrSkywalker/Point-M2AE |
| PointMamba | `models/pointmamba/` | Apache-2.0 | https://github.com/LMD0311/PointMamba |

## Parameter-Efficient Fine-Tuning Methods

| Method | Directory | License | Original Repository |
|---|---|---|---|
| IDPT | `models/idpt/` | No license | https://github.com/zyh16143998882/ICCV23-IDPT |
| VPT-Deep | `models/idpt/` | MIT | Adapted from IDPT (Jia et al., ECCV 2022) |
| PPT | `models/ppt/` | MIT | https://github.com/zsc000722/PPT |
| DAPT | `models/dapt/` | Apache-2.0 | https://github.com/LMD0311/DAPT |
| DAPT-PointGST | `models/dapt/` | Apache-2.0 | https://github.com/jerryfeng2003/PointGST |
| PointGST | `models/pointgst/` | Apache-2.0 | https://github.com/jerryfeng2003/PointGST |

## Apache-2.0 Compliance

The following models are licensed under Apache-2.0 and include a `NOTICE` file
as required by Section 4(d) of the license. Each NOTICE documents the original
copyright holder and the changes made for LIDARLearn integration.

| Model | Directory | NOTICE |
|---|---|---|
| PointMLP | `models/pointmlp/` | `models/pointmlp/NOTICE` |
| RepSurf | `models/repsurf/` | `models/repsurf/NOTICE` |
| PointGST | `models/pointgst/` | `models/pointgst/NOTICE` |
| DAPT | `models/dapt/` | `models/dapt/NOTICE` |
| PointMamba | `models/pointmamba/` | `models/pointmamba/NOTICE` |

## Unlicensed Upstream Sources

A few upstream repositories do not include a LICENSE file. The LIDARLearn
implementations for these methods are **clean-room re-implementations** based
on the corresponding papers, not copies of the upstream code.

| Model | Upstream repository |
|---|---|
| GDAN | https://github.com/mutianxu/GDANet |
| PointSCNet | https://github.com/Chenguoz/PointSCNet |
| RandLANet | https://github.com/aRI0U/RandLA-Net-pytorch |
| IDPT | https://github.com/zyh16143998882/ICCV23-IDPT |

## Shared Dependencies

| Component | License | Source |
|---|---|---|
| pointnet2_ops | MIT | https://github.com/erikwijmans/Pointnet2_PyTorch |
| KNN-CUDA | MIT | https://github.com/unlimblue/KNN_CUDA |
| ChamferDistance | MIT | https://github.com/ThibaultGROUEIX/ChamferDistancePytorch |
| timm | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| efficient-kan | MIT | https://github.com/Blealtan/efficient-kan |
| mamba-ssm (optional, PointMamba) | Apache-2.0 | https://github.com/state-spaces/mamba |
| causal-conv1d (optional, PointMamba) | BSD-3-Clause | https://github.com/Dao-AILab/causal-conv1d |
| Point serialization / Hilbert order (`models/pointmamba/serialization.py`, `hilbert.py`) | MIT | https://github.com/Pointcept/Pointcept |
| Hilbert curve encoding (`models/pointmamba/hilbert.py`, via Pointcept) | MIT | https://github.com/PrincetonLIPS/numpy-hilbert-curve |
