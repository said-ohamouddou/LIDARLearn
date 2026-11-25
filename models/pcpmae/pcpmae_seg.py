"""
PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders

Paper: NeurIPS 2024
Authors: Xiangdong Zhang, Shaofeng Zhang, Junchi Yan
Source: Implementation adapted from: https://github.com/aHapBean/PCP-MAE
License: MIT
"""

from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos import get_pos_embed
from ..build import MODELS, build_model_from_cfg
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


@MODELS.register_module()
class PCPMAE_Seg(BaseSegModel):
    """PCP-MAE part / semantic segmentation model."""

    FETCH_IDX = [3, 7, 11]
    PROP_DIM = 1024

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', 384))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        dropout = float(config.get('dropout_head', 0.5))

        # Build the registered PCP classification backbone; reuse its
        # group divider, encoder, pos MLP, transformer blocks, and norm.
        backbone_cfg = edict({
            'NAME': 'PCP',
            'cls_dim': config.get('cls_dim', 16),
            'trans_dim': self.trans_dim,
            'depth': int(config.get('depth', 12)),
            'num_heads': int(config.get('num_heads', 6)),
            'encoder_dims': int(config.get('encoder_dims', 384)),
            'num_group': self.num_group,
            'group_size': self.group_size,
            'drop_path_rate': float(config.get('drop_path_rate', 0.1)),
            'type': config.get('type', 'full'),
            'label_smoothing': config.get('label_smoothing', 0.0),
            'init_source': config.get('init_source', 'ShapeNet'),
            'base_model': 'PCP',
            'finetuning_strategy': config.get('finetuning_strategy', 'Full Finetuning'),
        })
        self.backbone = build_model_from_cfg(backbone_cfg)

        # Freeze the backbone's classification head (unused in seg).
        for attr in ('cls_head_finetune', 'cls_head'):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters():
                    p.requires_grad = False

        # --- Segmentation head (matches reference line-for-line) ---------
        feat_dim = 3 * self.trans_dim  # 1152 for trans_dim=384

        if self.use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(self.num_obj_classes, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64
        else:
            self.label_conv = None
            label_dim = 0

        # 3-NN FP upsampling from G centres back to N points.
        # in_channel = feat_dim (points2) + 3 (xyz via points1)
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, self.PROP_DIM],
        )

        # Head input: prop(1024) + max(1152) + avg(1152) + label(64 / 0)
        in_dim = self.PROP_DIM + 2 * feat_dim + label_dim   # 3392 or 3328

        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.seg_classes, 1)

    # ------------------------------------------------------------------

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        """Load pretrained PCP-MAE encoder weights into the backbone."""
        if hasattr(self.backbone, 'load_model_from_ckpt'):
            return self.backbone.load_model_from_ckpt(ckpt_path)
        state = load_checkpoint(ckpt_path, map_location='cpu')
        sd = state.get('base_model', state.get('model', state))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.backbone.load_state_dict(sd, strict=strict)

    # ------------------------------------------------------------------

    def forward(self, pts, cls_label=None):
        # Accept [B, 3, N] or [B, N, 3]; keep both representations.
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B, 3, N] or [B, N, 3], got {pts.shape}")

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()   # [B, N, 3]
        B, N, _ = pts_bnc.shape

        # --- Group divider -----------------------------------------------
        neighborhood, center = self.backbone.group_divider(pts_bnc)

        # Token embedding: [B, G, trans_dim]
        group_input_tokens = self.backbone.encoder(neighborhood)

        # --- Sinusoidal position encoding --------------------------------
        # Reference: pos = pos_embed(get_pos_embed(trans_dim, center))
        pos = self.backbone.pos_embed(
            get_pos_embed(self.trans_dim, center)
        )

        # --- Run transformer manually, fetching [3, 7, 11] ---------------
        # Reference seg path does NOT prepend cls / img / text tokens;
        # only the group tokens are fed to the transformer.
        x = group_input_tokens
        feature_list = []
        for i, block in enumerate(self.backbone.blocks.blocks):
            x = block(x + pos)
            if i in self.FETCH_IDX:
                normed = self.backbone.norm(x)                # [B, G, C]
                feature_list.append(normed.transpose(-1, -2).contiguous())  # [B, C, G]

        x_cat = torch.cat(feature_list, dim=1)                # [B, 3*C, G]

        # --- Global max + avg broadcast + optional label -----------------
        x_max = x_cat.max(dim=2)[0].unsqueeze(-1).expand(-1, -1, N)
        x_avg = x_cat.mean(dim=2).unsqueeze(-1).expand(-1, -1, N)

        global_parts = [x_max, x_avg]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))  # [B, 64, 1]
            global_parts.append(cls_feat.expand(-1, -1, N))
        x_global = torch.cat(global_parts, dim=1)

        # --- Upsample group features to N via 3-NN FP --------------------
        f_level_0 = self.propagation_0(
            xyz1=pts_bnc,      # [B, N, 3]
            xyz2=center,       # [B, G, 3]
            points1=pts_bcn,   # [B, 3, N]
            points2=x_cat,     # [B, 3*C, G]
        )                                                     # [B, prop_dim, N]

        # --- Head ---------------------------------------------------------
        h = torch.cat([f_level_0, x_global], dim=1)            # [B, in_dim, N]
        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = self.convs3(h)                                     # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
