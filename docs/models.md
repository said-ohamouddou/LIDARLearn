# Supported models

Every (model × fine-tuning strategy) pair below ships with ready-to-run YAML configs.

**Columns**

- **Cls**: object classification (ModelNet40, STPCTLS, ModelNetFewShot).
- **PartSeg**: part segmentation (ShapeNetParts).
- **SemSeg**: semantic segmentation (S3DIS).
- **Strategy**: the fine-tuning method on top of a self-supervised (SSL) backbone. **FF** is full fine-tuning; the 5 PEFT methods are **DAPT**, **IDPT**, **PPT**, **GST** (PointGST) and **VPT-Deep**.

The 8 SSL backbones also support pre-training on ShapeNet-55.

**PointMamba**

- Ships exactly the tasks of its [official repository](https://github.com/LMD0311/PointMamba): classification on ModelNet40, ModelNetFewShot and ScanObjectNN (OBJ-BG, OBJ-ONLY, PB-T50-RS).
- Needs the optional `mamba-ssm` package (see [Installation](getting-started/installation.md#5b-optional-install-mamba-for-pointmamba)).
- Its pretraining config and `scripts/train_scanobjectnn.sh` use the upstream `y_rotate` augmentation.
- The shared ModelNet40 and few-shot sweeps keep `--augmentation none` for all models. Upstream PointMamba uses random scaling (2/3–3/2) and translation (±0.2) there.

<table border="1" cellpadding="6" cellspacing="0">
<thead>
<tr><th>Category</th><th>Model</th><th>Strategy</th><th align="center">Cls</th><th align="center">PartSeg</th><th align="center">SemSeg</th></tr>
</thead>
<tbody>
<tr><td rowspan="17"><b>Point-based</b></td><td>PointNet</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointNet2-SSG</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointNet2-MSG</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>SONet</td><td>—</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PPFNet</td><td>—</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PointCNN</td><td>—</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PointWeb</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointConv</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>RSCNN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointMLP</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointMLP-elite</td><td>—</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PointSCNet</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>RepSurf</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>RepSurf-U (2×)</td><td>—</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PointKAN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DELA</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>RandLA-Net</td><td>—</td><td align="center">—</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td rowspan="8"><b>Attention-based</b></td><td>PCT</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>P2P</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointTNT</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>GlobalTransformer</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PVT</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointTransformer</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointTransformerV2</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointTransformerV3</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td rowspan="7"><b>Graph-based</b></td><td>DGCNN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DeepGCN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>CurveNet</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>GDANet</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>MS-DGCNN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>KAN-DGCNN</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>MS-DGCNN++</td><td>—</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td rowspan="28"><b>Self-supervised</b></td><td rowspan="6">Point-MAE</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DAPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>IDPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>GST</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>VPT-Deep</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td rowspan="6">ACT</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DAPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>IDPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>GST</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>VPT-Deep</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td rowspan="6">ReCon</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DAPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>IDPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PPT</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>GST</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>VPT-Deep</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td rowspan="6">PointGPT</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>DAPT</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>IDPT</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>PPT</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>GST</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>VPT-Deep</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td>Point-M2AE</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>Point-BERT</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PCP-MAE</td><td>FF</td><td align="center">✓</td><td align="center">✓</td><td align="center">✓</td></tr>
<tr><td>PointMamba</td><td>FF</td><td align="center">✓</td><td align="center">—</td><td align="center">—</td></tr>
</tbody>
</table>

**Total: 59 configurations** (31 supervised × FF, 8 SSL × FF, 4 SSL × 5 PEFT = 20). See [`THIRD_PARTY_NOTICES.md`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/THIRD_PARTY_NOTICES.md) for per-model licences and upstream repositories, and [`docs/point_cloud_methods.csv`](https://github.com/said-ohamouddou/LIDARLearn/blob/main/docs/point_cloud_methods.csv) for paper titles and venues.
