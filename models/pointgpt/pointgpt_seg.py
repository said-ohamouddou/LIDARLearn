"""
PointGPT: Auto-regressively Generative Pre-training from Point Clouds

Paper: NeurIPS 2023
Authors: Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue
Source: Implementation adapted from: https://github.com/CGuangyan-BIT/PointGPT
License: MIT
"""

from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS, build_model_from_cfg
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


@MODELS.register_module()
class PointGPT_Seg(BaseSegModel):
    """PointGPT part / semantic segmentation model.

    Reuses the registered ``PointGPT`` classification backbone for its
    group divider, token encoder, position embedding, cls/SOS tokens, and
    GPT transformer blocks. Only the segmentation head (propagation + MLP)
    is new.
    """

    FETCH_IDX = [3, 7, 11]
    PROP_DIM = 1024

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', 384))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        encoder_dims = int(config.get('encoder_dims', 384))
        depth = int(config.get('depth', 12))
        dropout = float(config.get('dropout_head', 0.5))

        self._encoder_dims = encoder_dims
        self._depth = depth

        # Build the PointGPT classification backbone and grab its blocks.
        backbone_cfg = edict({
            'NAME': 'PointGPT',
            'cls_dim': config.get('cls_dim', 16),
            'trans_dim': self.trans_dim,
            'depth': depth,
            'decoder_depth': config.get('decoder_depth', 4),
            'num_heads': config.get('num_heads', 6),
            'encoder_dims': encoder_dims,
            'num_group': self.num_group,
            'group_size': self.group_size,
            'drop_path_rate': config.get('drop_path_rate', 0.1),
            'loss': config.get('loss', 'cdl12'),
            'type': config.get('type', 'full'),
            'weight_center': config.get('weight_center', 1),
            'label_smoothing': config.get('label_smoothing', 0.0),
            'init_source': config.get('init_source', 'ShapeNet'),
            'base_model': 'PointGPT',
            'finetuning_strategy': config.get('finetuning_strategy', 'Full Finetuning'),
        })
        self.backbone = build_model_from_cfg(backbone_cfg)

        # Disable backbone classification head (not used by seg).
        if hasattr(self.backbone.blocks, 'cls_head_finetune'):
            for p in self.backbone.blocks.cls_head_finetune.parameters():
                p.requires_grad = False
        if hasattr(self.backbone.blocks, 'cls_norm'):
            for p in self.backbone.blocks.cls_norm.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, 'generator_blocks'):
            for p in self.backbone.generator_blocks.parameters():
                p.requires_grad = False

        # --- Segmentation head (exactly as reference) ---------------------
        feat_dim = 3 * encoder_dims  # 1152 = concat of three fetched layers

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

        # 3-NN FP upsampling from G centers to N points.
        # in_channel = feat_dim (points2) + 3 (xyz from points1) = 1155
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, self.PROP_DIM],
        )

        # Head input: prop(1024) + max(1152) + avg(1152) + label(64 or 0)
        in_dim = self.PROP_DIM + 2 * feat_dim + label_dim   # 3392 or 3328

        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.seg_classes, 1)

    # ------------------------------------------------------------------

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        """Load pretrained PointGPT weights into the backbone."""
        if hasattr(self.backbone, 'load_model_from_ckpt'):
            return self.backbone.load_model_from_ckpt(ckpt_path)
        state = load_checkpoint(ckpt_path, map_location='cpu')
        sd = state.get('base_model', state.get('model', state))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.backbone.load_state_dict(sd, strict=strict)

    # ------------------------------------------------------------------

    def _run_gpt_encoder(self, x, pos, batch_size):
        """Re-implementation of PointGPT's causal GPT encoder that also
        returns intermediate features at ``self.FETCH_IDX``.

        Mirrors ``GPT_extractor.forward(..., classify=True)``:
        prepend SOS, run layers with an upper-triangular attention mask,
        and fetch block outputs at layers 3/7/11 (already with the SOS
        and cls tokens removed).
        """
        gpt = self.backbone.blocks  # GPT_extractor

        # Sequence-first: [L, B, C]
        h = x.transpose(0, 1).contiguous()
        pos = pos.transpose(0, 1).contiguous()

        # Prepend the learned SOS token along the sequence axis.
        sos = torch.ones(1, batch_size, self._encoder_dims,
                         device=h.device, dtype=h.dtype) * gpt.sos
        h = torch.cat([sos, h], dim=0)              # [2+G, B, C]

        seq_len = h.shape[0]
        attn_mask = torch.full(
            (seq_len, seq_len), float('-inf'),
            device=h.device, dtype=h.dtype,
        )
        attn_mask = torch.triu(attn_mask, diagonal=1).to(torch.bool)

        feature_list = []
        for i, layer in enumerate(gpt.layers):
            h = layer(h + pos, attn_mask)
            if i in self.FETCH_IDX:
                # [L, B, C] → [B, L, C], drop SOS (idx 0) and cls (idx 1)
                feat = h.transpose(0, 1)[:, 2:, :].contiguous()  # [B, G, C]
                feature_list.append(feat)
        return feature_list

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

        # --- Group divider (FPS + k-NN + Morton sort) ---------------------
        neighborhood, center = self.backbone.group_divider(pts_bnc)

        # Token embedding: [B, G, encoder_dims]
        group_input_tokens = self.backbone.encoder(neighborhood)

        # --- Position encodings (sinusoidal) -----------------------------
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        cls_pos = self.backbone.cls_pos.expand(B, -1, -1)
        sos_pos = self.backbone.sos_pos.expand(B, -1, -1)

        pos = self.backbone.pos_embed(center)            # [B, G, C]
        pos = torch.cat([sos_pos, pos], dim=1)           # [B, 1+G, C]
        pos = torch.cat([cls_pos, pos], dim=1)           # [B, 2+G, C]

        x = torch.cat([cls_tokens, group_input_tokens], dim=1)  # [B, 1+G, C]

        # --- Transformer with intermediate fetches -----------------------
        feature_list = self._run_gpt_encoder(x, pos, B)

        # LayerNorm + transpose to channel-first: list of [B, C, G]
        feature_list = [
            self.backbone.norm(f).transpose(-1, -2).contiguous()
            for f in feature_list
        ]
        x_cat = torch.cat(feature_list, dim=1)           # [B, 3*C, G]

        # --- Global max + avg broadcast + optional label -----------------
        x_max = x_cat.max(dim=2)[0].unsqueeze(-1).expand(-1, -1, N)  # [B, 3*C, N]
        x_avg = x_cat.mean(dim=2).unsqueeze(-1).expand(-1, -1, N)    # [B, 3*C, N]

        global_parts = [x_max, x_avg]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))       # [B, 64, 1]
            global_parts.append(cls_feat.expand(-1, -1, N))
        x_global = torch.cat(global_parts, dim=1)

        # --- Upsample group features to N via 3-NN FP --------------------
        f_level_0 = self.propagation_0(
            xyz1=pts_bnc,      # [B, N, 3]
            xyz2=center,       # [B, G, 3]
            points1=pts_bcn,   # [B, 3, N]
            points2=x_cat,     # [B, 3*C, G]
        )                                                # [B, prop_dim, N]

        # --- Head ---------------------------------------------------------
        h = torch.cat([f_level_0, x_global], dim=1)      # [B, in_dim, N]
        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = self.convs3(h)                               # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
