"""
Positional Prompt Tuning for Efficient 3D Representation Learning

Paper: ACM MM 2025 (arXiv 2408.11567)
Authors: Shaochen Zhang, Zekun Qi, Runpei Dong, Xiuxiu Bai, Xing Wei
Source: Implementation adapted from: https://github.com/zsc000722/PPT
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .ppt import Block, Group
from ..ssl_blocks import Encoder
from ..build import MODELS
from ..base_seg_model import BaseSegModel
from ..seg.transformer_seg_base import PointNetFeaturePropagation


class PPTTransformerEncoderSeg(nn.Module):
    """Doubled-sequence PPT transformer encoder (reference reproduction)."""

    def __init__(self, embed_dim, depth, num_heads, drop_path_rate, num_group):
        super().__init__()
        self.num_group = num_group
        dpr = drop_path_rate
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                drop_path=dpr[i] if isinstance(dpr, list) else dpr,
            )
            for i in range(depth)
        ])

        self.prompt_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prompt_pos_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prompt_all_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.prompt_mlp.apply(self._init_weights)
        self.prompt_pos_mlp.apply(self._init_weights)
        self.prompt_all_mlp.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pos, fetch_idx=(3, 7, 11)):
        """
        x:   [B, 2G, C] — duplicated group tokens
        pos: [B, 2G, C] — duplicated positional embedding
        Returns: list of fetched features (len 3), each [B, 2G, C].
        """
        feature_list = []
        G = self.num_group
        for i, block in enumerate(self.blocks):
            x = x + pos
            # In-place additive residuals: first G tokens via prompt_mlp,
            # last G tokens via prompt_pos_mlp. Clone the slices so the
            # autograd graph does not see an in-place overwrite.
            first = x[:, :G] + self.prompt_mlp(x[:, :G])
            last = x[:, -G:] + self.prompt_pos_mlp(x[:, -G:])
            x = torch.cat([first, last], dim=1)
            x = block(x, self.prompt_all_mlp)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


@MODELS.register_module()
class PPT_Seg(BaseSegModel):
    """PPT part / semantic segmentation model."""

    FETCH_IDX = (3, 7, 11)
    PROP_DIM = 1024

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.get('trans_dim', config.get('embed_dim', 384)))
        self.depth = int(config.get('depth', 12))
        self.num_heads = int(config.get('num_heads', 6))
        self.num_group = int(config.get('num_group', 128))
        self.group_size = int(config.get('group_size', 32))
        drop_path_rate = float(config.get('drop_path_rate', 0.1))
        dropout = float(config.get('dropout_head', 0.5))

        # Shared tokeniser + group divider.
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.trans_dim)

        # Positional embedding (same as Point-MAE's Linear(3→128)→GELU→Linear(128→C)).
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = PPTTransformerEncoderSeg(
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=dpr,
            num_group=self.num_group,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

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

        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=feat_dim + 3,
            mlp=[self.trans_dim * 4, self.PROP_DIM],
        )

        # Head input: prop(1024) + max(1152) + avg(1152) + label(64 or 0)
        in_dim = self.PROP_DIM + 2 * feat_dim + label_dim  # 3392 / 3328

        self.convs1 = nn.Conv1d(in_dim, 512, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.bns2 = nn.BatchNorm1d(256)
        self.convs3 = nn.Conv1d(256, self.seg_classes, 1)

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
        """Load pretrained SSL encoder weights into the shared backbone.

        Accepts checkpoints from PointMAE, ACT, ReCon, PCP-MAE and
        Point-BERT by stripping the well-known MAE_encoder / ACT_encoder /
        base_model / transformer_q prefixes. Prompt MLPs and the seg head
        are randomly initialised.
        """
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('base_model', ckpt.get('model', ckpt))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        remapped = {}
        for k, v in sd.items():
            nk = k
            for prefix in ('MAE_encoder.', 'ACT_encoder.', 'base_model.',
                           'transformer_q.', 'GPT_Transformer.'):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            if 'cls_head_finetune' in nk or 'cls_token' in nk or 'cls_pos' in nk:
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

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()   # [B, N, 3]
        B, N, _ = pts_bnc.shape

        # --- Group divider -----------------------------------------------
        neighborhood, center = self.group_divider(pts_bnc)
        group_input_tokens = self.encoder(neighborhood)    # [B, G, C]

        pos = self.pos_embed(center)                       # [B, G, C]

        # --- Doubled sequence (reference) --------------------------------
        x = torch.cat([group_input_tokens, group_input_tokens], dim=1)  # [B, 2G, C]
        pos_d = torch.cat([pos, pos], dim=1)                              # [B, 2G, C]

        # --- Transformer with intermediate fetches -----------------------
        feature_list = self.blocks(x, pos_d, fetch_idx=self.FETCH_IDX)

        # LayerNorm + transpose to channel-first: list of [B, C, 2G]
        feature_list = [
            self.norm(f).transpose(-1, -2).contiguous() for f in feature_list
        ]
        x_cat = torch.cat(feature_list, dim=1)             # [B, 3C, 2G]

        # --- Global descriptor over the full doubled sequence ------------
        x_max = x_cat.max(dim=2)[0].unsqueeze(-1).expand(-1, -1, N)
        x_avg = x_cat.mean(dim=2).unsqueeze(-1).expand(-1, -1, N)

        global_parts = [x_max, x_avg]
        if self.use_cls_label:
            if cls_label is None:
                raise ValueError("use_cls_label=True but cls_label is None")
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))    # [B, 64, 1]
            global_parts.append(cls_feat.expand(-1, -1, N))
        x_global = torch.cat(global_parts, dim=1)

        # --- Feature propagation from G centers -------------------------
        # The reference's PointNetFeaturePropagation silently uses only the
        # first G columns of points2 (via index_points). We slice the first
        # G tokens explicitly so the compiled three_interpolate kernel sees
        # matching shapes [B, 3C, G].
        f_level_0 = self.propagation_0(
            xyz1=pts_bnc,                          # [B, N, 3]
            xyz2=center,                           # [B, G, 3]
            points1=pts_bcn,                       # [B, 3, N]
            points2=x_cat[:, :, :self.num_group],  # [B, 3C, G]
        )                                                   # [B, prop_dim, N]

        # --- Head ---------------------------------------------------------
        h = torch.cat([f_level_0, x_global], dim=1)         # [B, in_dim, N]
        h = F.relu(self.bns1(self.convs1(h)))
        h = self.dp1(h)
        h = F.relu(self.bns2(self.convs2(h)))
        h = self.convs3(h)                                  # [B, seg_classes, N]
        return F.log_softmax(h, dim=1)
