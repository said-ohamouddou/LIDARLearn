"""
Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis

Paper: CVPR 2024
Authors: Xin Zhou, Dingkang Liang, Wei Xu, Xingkui Zhu, Yihan Xu, Zhikang Zou, Xiang Bai
Source: Implementation adapted from: https://github.com/LMD0311/DAPT
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .dapt import Block, Group, init_tfts, apply_tfts
from ..ssl_blocks import Encoder
from ..build import MODELS
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


@MODELS.register_module()
class DAPT_Seg(BaseSegModel):
    """DAPT part / semantic segmentation model (our extension)."""

    FETCH_IDX = (3, 7, 11)
    PROP_DIM = 1024

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', 384))
        self.depth = int(config.get('depth', 12))
        self.num_heads = int(config.get('num_heads', 6))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        encoder_dims = int(config.get('encoder_dims', self.trans_dim))
        drop_path_rate = float(config.get('drop_path_rate', 0.1))
        drop_adapter_rate = float(config.get('drop_adapter_rate', 0.0))
        rank = int(config.get('rank', 64))
        dropout = float(config.get('dropout_head', 0.5))

        # --- Backbone: exactly the DAPT classification tokeniser -----
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        # Model-level TFTS pairs (match the DAPT cls backbone).
        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(self.trans_dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(self.trans_dim)

        # --- Transformer blocks: reuse DAPT Block unchanged ----------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.trans_dim,
                num_heads=self.num_heads,
                drop=0.0,
                attn_drop=0.0,
                adapter_drop=drop_adapter_rate,
                drop_path=dpr[i],
                rank=rank,
                depth=i,
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.trans_dim)

        # --- Segmentation head (identical to Point-MAE seg) ----------
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

        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, self.PROP_DIM],
        )

        # Head input: prop(1024) + max(1152) + avg(1152) + label(64 / 0)
        in_dim = self.PROP_DIM + 2 * feat_dim + label_dim  # 3392 or 3328

        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.seg_classes, 1)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        """Load pretrained SSL encoder weights (PointMAE / ACT / ReCon /
        Point-BERT) and previously-trained DAPT cls weights into the
        segmentation backbone. Prefixes are stripped so the key names
        line up with this model.

        Seg-head parameters (``convs*``, ``bns*``, ``propagation_0``,
        ``label_conv``) are always randomly initialised.
        """
        if ckpt_path is None:
            return
        ckpt = load_checkpoint(ckpt_path, map_location='cpu')
        sd = ckpt.get('base_model', ckpt.get('model', ckpt))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        remapped = {}
        for k, v in sd.items():
            nk = k
            for prefix in ('MAE_encoder.', 'ACT_encoder.', 'base_model.',
                           'transformer_k.', 'transformer_q.',
                           'GPT_Transformer.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            # Drop the classification head, keep everything else.
            if 'cls_head_finetune' in nk:
                continue
            remapped[nk] = v
        return self.load_state_dict(remapped, strict=strict)

    # ------------------------------------------------------------------

    def forward(self, pts, cls_label=None):
        # Accept [B, 3, N] or [B, N, 3]; keep both representations.
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B, 3, N] or [B, N, 3], got {pts.shape}")

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()  # [B, N, 3]
        B, N, _ = pts_bnc.shape
        G = self.num_group

        # --- Tokenise ------------------------------------------------
        neighborhood, center = self.group_divider(pts_bnc)
        group_input_tokens = self.encoder(neighborhood)               # [B, G, C]
        group_input_tokens = apply_tfts(
            group_input_tokens, self.tfts_gamma_1, self.tfts_beta_1,
        )

        # --- Sequence assembly (cls + G patch tokens) ----------------
        cls_tokens = self.cls_token.expand(B, -1, -1)                  # [B, 1, C]
        cls_pos = self.cls_pos.expand(B, -1, -1)                       # [B, 1, C]

        pos = self.pos_embed(center)                                   # [B, G, C]

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)         # [B, 1+G, C]
        pos_full = torch.cat((cls_pos, pos), dim=1)                    # [B, 1+G, C]
        x = x + pos_full                                               # DAPT adds pos once

        # --- Run blocks manually, fetching patch tokens at FETCH_IDX -
        # After block i the sequence length is 1 + (i+1) + G (cls,
        # i+1 adapter prompts, G patch tokens). The final G tokens are
        # always the patch tokens.
        feature_list = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.FETCH_IDX:
                patch_tokens = x[:, -G:, :]                            # [B, G, C]
                feature_list.append(patch_tokens)

        # LayerNorm + model-level TFTS-2 are applied to the fetched
        # patch tokens to match the DAPT cls backbone post-processing.
        feature_list = [
            apply_tfts(self.norm(f), self.tfts_gamma_2, self.tfts_beta_2)
            .transpose(-1, -2).contiguous()
            for f in feature_list
        ]                                                              # 3 × [B, C, G]
        x_cat = torch.cat(feature_list, dim=1)                         # [B, 3C, G]

        # --- Global descriptor broadcast + optional label ------------
        x_max = x_cat.max(dim=2)[0].unsqueeze(-1).expand(-1, -1, N)
        x_avg = x_cat.mean(dim=2).unsqueeze(-1).expand(-1, -1, N)

        global_parts = [x_max, x_avg]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))        # [B, 64, 1]
            global_parts.append(cls_feat.expand(-1, -1, N))
        x_global = torch.cat(global_parts, dim=1)

        # --- Feature propagation G → N -------------------------------
        f_level_0 = self.propagation_0(
            xyz1=pts_bnc,
            xyz2=center,
            points1=pts_bcn,
            points2=x_cat,
        )                                                              # [B, prop_dim, N]

        # --- Seg head -------------------------------------------------
        h = torch.cat([f_level_0, x_global], dim=1)                    # [B, in_dim, N]
        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = self.convs3(h)                                             # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
