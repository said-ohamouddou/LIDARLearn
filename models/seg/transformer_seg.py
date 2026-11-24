"""
Registered TransformerSeg wrappers — one class per SSL backbone.

Each class implements _encode() matching the original segmentation code:
- PointMAE, ACT: fetch layers [3,7,11] → 1152-dim, single 3-NN FP G→N
- ReCon: same + extra cls token pooling
- PointBERT: 4-level FPS + 3-NN init + DGCNN bottom-up propagation
  (faithful to ssl/Point-BERT/segmentation/models/PointTransformer.py)
- PointGPT, PCP-MAE, Point-M2AE, IDPT: separate dedicated files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from pointnet2_ops import pointnet2_utils

from ..build import MODELS, build_model_from_cfg
from ..base_seg_model import BaseSegModel
from .transformer_seg_base import TransformerSegBase, PointNetFeaturePropagation
from utils.checkpoint import load_checkpoint


def _fps(data, number):
    """FPS: data [B, N, 3] → [B, number, 3]."""
    fps_idx = pointnet2_utils.furthest_point_sample(data.contiguous(), number)
    return pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()


def _make_backbone_cfg(config, backbone_name):
    """Turn a seg config into a classification config for the backbone."""
    cfg = edict({
        'NAME': backbone_name,
        'cls_dim': config.get('cls_dim', 16),
        'trans_dim': config.trans_dim,
        'embed_dim': config.get('embed_dim', config.trans_dim),
        'depth': config.depth,
        'num_heads': config.num_heads,
        'encoder_dims': config.encoder_dims,
        'num_group': config.num_group,
        'group_size': config.group_size,
        'drop_path_rate': config.get('drop_path_rate', 0.1),
        'label_smoothing': config.get('label_smoothing', 0.0),
        'finetuning_strategy': config.get('finetuning_strategy', 'Full Finetuning'),
        'init_source': config.get('init_source', 'ShapeNet'),
        'base_model': backbone_name,
        # ACT backbone uses transfer_type to decide which params to freeze;
        # for the seg wrapper we always want full backbone exposed.
        'transfer_type': config.get('transfer_type', 'full'),
    })
    for fld in ('decoder_depth', 'type', 'loss', 'weight_center'):
        if fld in config:
            cfg[fld] = config[fld]
    return cfg


# ─────────────────────────────────────────────────────────────
#  Shared base: PointMAE / ACT
#  fetch_idx = [3, 7, 11]  →  feat_dim = trans_dim * 3
# ─────────────────────────────────────────────────────────────

class _StandardTransformerSeg(TransformerSegBase):
    """Shared for PointMAE and ACT.

    Extracts features from transformer layers [3, 7, 11],
    concatenates to get 3 * trans_dim channels.
    """

    BACKBONE_NAME = ""
    FETCH_IDX = [3, 7, 11]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        dropout = config.get('dropout_head', 0.5)

        backbone_cfg = _make_backbone_cfg(config, self.BACKBONE_NAME)
        self.backbone = build_model_from_cfg(backbone_cfg)

        self.trans_dim = int(config.trans_dim)
        feat_dim = self.trans_dim * len(self.FETCH_IDX)

        # Disable classification head
        for attr in ('cls_head_finetune', 'cls_head'):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters():
                    p.requires_grad = False

        self._build_seg_head(feat_dim=feat_dim, dropout=dropout)

    def _encode(self, pts_bcn):
        pts_bnc = pts_bcn.transpose(1, 2).contiguous()
        B = pts_bnc.shape[0]

        # Group + encode
        neighborhood, center = self.backbone.group_divider(pts_bnc)
        group_tokens = self.backbone.encoder(neighborhood)

        if hasattr(self.backbone, 'reduce_dim'):
            group_tokens = self.backbone.reduce_dim(group_tokens)

        # Positional embedding
        pos = self.backbone.pos_embed(center)

        # Prepend cls_token if present
        has_cls = (
            hasattr(self.backbone, 'cls_token')
            and hasattr(self.backbone, 'cls_pos')
            and self.backbone.cls_token is not None
        )
        if has_cls:
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            cls_pos = self.backbone.cls_pos.expand(B, -1, -1)
            x = torch.cat([cls_tokens, group_tokens], dim=1)
            pos_full = torch.cat([cls_pos, pos], dim=1)
            n_cls = 1
        else:
            x = group_tokens
            pos_full = pos
            n_cls = 0

        # Run transformer blocks one by one, fetch intermediate outputs
        feature_list = []
        for i, block in enumerate(self.backbone.blocks.blocks):
            x = block(x + pos_full)
            if i in self.FETCH_IDX:
                normed = self.backbone.norm(x)
                # Drop cls token(s), transpose to [B, C, G]
                tokens = normed[:, n_cls:]
                feature_list.append(tokens.transpose(-1, -2).contiguous())

        # Concat multi-layer features: [B, 3*trans_dim, G]
        feats = torch.cat(feature_list, dim=1)
        return feats, center, None


# ─────────────────────────────────────────────────────────────
#  ReCon: same + extra cls token pooling (adds trans_dim to global)
# ─────────────────────────────────────────────────────────────

class _ReconTransformerSeg(_StandardTransformerSeg):
    """ReCon uses 3 cls tokens (cls, img, text). Their max-pooled
    feature is added to the global feature (extra +384 channels)."""

    BACKBONE_NAME = "RECON"

    def __init__(self, config, **kwargs):
        # Override: extra_global_dim = trans_dim for the cls token pool
        super(TransformerSegBase, self).__init__(config, **kwargs)
        dropout = config.get('dropout_head', 0.5)

        backbone_cfg = _make_backbone_cfg(config, self.BACKBONE_NAME)
        self.backbone = build_model_from_cfg(backbone_cfg)

        self.trans_dim = int(config.trans_dim)
        feat_dim = self.trans_dim * len(self.FETCH_IDX)

        for attr in ('cls_head_finetune', 'cls_head'):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters():
                    p.requires_grad = False

        self._build_seg_head(
            feat_dim=feat_dim,
            extra_global_dim=self.trans_dim,  # for cls token pooling
            dropout=dropout,
        )

    def _encode(self, pts_bcn):
        pts_bnc = pts_bcn.transpose(1, 2).contiguous()
        B = pts_bnc.shape[0]

        neighborhood, center = self.backbone.group_divider(pts_bnc)
        group_tokens = self.backbone.encoder(neighborhood)

        if hasattr(self.backbone, 'reduce_dim'):
            group_tokens = self.backbone.reduce_dim(group_tokens)

        pos = self.backbone.pos_embed(center)

        # ReCon uses 3 contrastive cls tokens (cls + img + text) and each
        # ReConBlock's cross-attention hardcodes x[:, :3] as those tokens.
        # The sequence MUST be [cls, img, text, group_tokens], otherwise the
        # first two patch tokens get treated as img/text tokens by the block.
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        img_token = self.backbone.img_token.expand(B, -1, -1)
        text_token = self.backbone.text_token.expand(B, -1, -1)
        cls_pos = self.backbone.cls_pos.expand(B, -1, -1)
        img_pos = self.backbone.img_pos.expand(B, -1, -1)
        text_pos = self.backbone.text_pos.expand(B, -1, -1)
        n_cls = 3

        x = torch.cat([cls_token, img_token, text_token, group_tokens], dim=1)
        pos_full = torch.cat([cls_pos, img_pos, text_pos, pos], dim=1)

        # Run blocks, fetch intermediates
        feature_list = []
        for i, block in enumerate(self.backbone.blocks.blocks):
            x = block(x + pos_full)
            if i in self.FETCH_IDX:
                normed = self.backbone.norm(x)
                # Patch tokens only (drop the 3 cls/img/text tokens)
                tokens = normed[:, n_cls:]
                feature_list.append(tokens.transpose(-1, -2).contiguous())

        feats = torch.cat(feature_list, dim=1)  # [B, 3*trans_dim, G]

        # Extra global: max-pool over the 3 contrastive cls tokens
        final_normed = self.backbone.norm(x)
        cls_feats = final_normed[:, :n_cls, :]  # [B, 3, trans_dim]
        extra_global = cls_feats.max(dim=1)[0]  # [B, trans_dim]

        return feats, center, extra_global


# ─────────────────────────────────────────────────────────────
#  PointBERT: 4-level FPS + 3-NN init + DGCNN bottom-up propagation
#  Faithful port of ssl/Point-BERT/segmentation/models/PointTransformer.py
# ─────────────────────────────────────────────────────────────

class DGCNN_Propagation(nn.Module):
    """DGCNN-style graph feature upsampling between two point sets.

    Matches ssl/Point-BERT/segmentation/models/PointTransformer.py
    DGCNN_Propagation (k=4).

    coor, f   : [B, 3, G_src], [B, C_src, G_src]  — source (fine)
    coor_q, f_q: [B, 3, G_dst], [B, 3, G_dst]     — query (coarse, xyz only at init)
    returns   : [B, trans_dim, G_dst]
    """

    def __init__(self, k=4, trans_dim=384):
        super().__init__()
        self.k = k
        self.layer1 = nn.Sequential(
            nn.Conv2d(trans_dim * 2, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, trans_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, trans_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def _get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        """Build edge features [x_k_neighbor - x_q | x_q] for each query point."""
        B, C, G_k = x_k.shape
        G_q = x_q.shape[2]

        # k-NN in coordinate space: [B, k, G_q]
        with torch.no_grad():
            # pairwise squared distances: [B, G_q, G_k]
            coor_q_t = coor_q.transpose(1, 2)   # [B, G_q, 3]
            coor_k_t = coor_k.transpose(1, 2)   # [B, G_k, 3]
            diff = coor_q_t.unsqueeze(2) - coor_k_t.unsqueeze(1)  # [B, G_q, G_k, 3]
            dist = (diff ** 2).sum(-1)           # [B, G_q, G_k]
            idx = dist.topk(self.k, dim=-1, largest=False)[1]  # [B, G_q, k]
            idx = idx.permute(0, 2, 1).contiguous()  # [B, k, G_q]

            idx_base = torch.arange(B, device=x_k.device).view(-1, 1, 1) * G_k
            flat_idx = (idx + idx_base).reshape(-1)

        x_k_t = x_k.transpose(1, 2).contiguous()               # [B, G_k, C]
        feature = x_k_t.view(B * G_k, C)[flat_idx]             # [B*k*G_q, C]
        feature = feature.view(B, self.k, G_q, C).permute(0, 3, 2, 1)  # [B, C, G_q, k]
        x_q_exp = x_q.unsqueeze(-1).expand(-1, -1, -1, self.k) # [B, C, G_q, k]
        return torch.cat([feature - x_q_exp, x_q_exp], dim=1)  # [B, 2C, G_q, k]

    def forward(self, coor, f, coor_q, f_q):
        """
        coor  : [B, 3, G_src]  — fine source coords
        f     : [B, C, G_src]  — fine source features
        coor_q: [B, 3, G_dst]  — coarse query coords
        f_q   : [B, 3, G_dst]  — coarse query (xyz used as initial features)
        """
        f_q = self._get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q).max(dim=-1)[0]

        f_q = self._get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q).max(dim=-1)[0]
        return f_q


@MODELS.register_module()
class PointBERT_Seg(BaseSegModel):
    """PointBERT part/semantic segmentation.

    Faithful port of ssl/Point-BERT/segmentation/models/PointTransformer.py.

    Architecture:
      1. Group divider (FPS+kNN) → tokens, center  [B, G, C]
      2. reduce_dim (encoder_dims → trans_dim)
      3. Prepend cls_token → [B, 1+G, C]
      4. 12-layer transformer, fetch [3,7,11] → 3×[B, G, C]
      5. 4-level FPS hierarchy: level0=N, level1=512, level2=256, level3=G
      6. 3-NN init at levels 2 and 1, DGCNN bottom-up 3→2→1
      7. Final 3-NN FP from level-1 to N (includes label+xyz init)
      8. Conv1d: trans_dim → 128 → seg_classes
    """

    FETCH_IDX = [3, 7, 11]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.trans_dim = int(config.trans_dim)
        self.num_group = int(config.num_group)
        self.group_size = int(config.group_size)
        dropout = float(config.get('dropout_head', 0.5))

        # Build PointBERT classification backbone
        backbone_cfg = _make_backbone_cfg(config, 'PointBERT')
        self.backbone = build_model_from_cfg(backbone_cfg)

        # Disable classification head
        for attr in ('cls_head_finetune', 'cls_head'):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters():
                    p.requires_grad = False

        C = self.trans_dim

        # 3-NN FP modules: init levels 2 and 1 from level-3 features
        # propagation_2: center_level_2 ← center_level_3, f=feature_list[1]
        self.propagation_2 = PointNetFeaturePropagation(
            in_channel=C + 3, mlp=[C * 4, C])
        # propagation_1: center_level_1 ← center_level_3, f=feature_list[0]
        self.propagation_1 = PointNetFeaturePropagation(
            in_channel=C + 3, mlp=[C * 4, C])
        # propagation_0: center_level_0 (N pts) ← center_level_1
        # in_channel = C + 3 + 16 (xyz+label init at level_0)
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=C + 3 + self.num_obj_classes, mlp=[C * 4, C])

        # DGCNN bottom-up: level3→2, level2→1
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4, trans_dim=C)
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4, trans_dim=C)

        # Head: trans_dim → 128 → seg_classes
        self.conv1 = nn.Conv1d(C, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, self.seg_classes, 1)

    def load_backbone_ckpt(self, ckpt_path, strict=False):
        if hasattr(self.backbone, 'load_model_from_ckpt'):
            return self.backbone.load_model_from_ckpt(ckpt_path)
        state = load_checkpoint(ckpt_path, map_location='cpu')
        sd = state.get('base_model', state.get('model', state))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return self.backbone.load_state_dict(sd, strict=strict)

    def forward(self, pts, cls_label=None):
        # normalise input
        if pts.dim() == 3 and pts.shape[1] == 3 and pts.shape[2] != 3:
            pts_bcn = pts
        elif pts.dim() == 3 and pts.shape[2] == 3:
            pts_bcn = pts.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"pts must be [B,3,N] or [B,N,3], got {pts.shape}")

        pts_bnc = pts_bcn.transpose(1, 2).contiguous()  # [B, N, 3]
        B, N, _ = pts_bnc.shape

        # --- Tokenise ----------------------------------------------------
        neighborhood, center = self.backbone.group_divider(pts_bnc)
        group_tokens = self.backbone.encoder(neighborhood)              # [B, G, encoder_dims]
        group_tokens = self.backbone.reduce_dim(group_tokens)           # [B, G, C]

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)         # [B, 1, C]
        cls_pos = self.backbone.cls_pos.expand(B, -1, -1)
        pos = self.backbone.pos_embed(center)                           # [B, G, C]

        x = torch.cat([cls_tokens, group_tokens], dim=1)               # [B, 1+G, C]
        pos_full = torch.cat([cls_pos, pos], dim=1)

        # --- Transformer: fetch [3, 7, 11] --------------------------------
        feature_list = []
        for i, block in enumerate(self.backbone.blocks.blocks):
            x = block(x + pos_full)
            if i in self.FETCH_IDX:
                normed = self.backbone.norm(x)
                # drop cls token, transpose → [B, C, G]
                feature_list.append(normed[:, 1:].transpose(-1, -2).contiguous())

        # --- 4-level FPS hierarchy ----------------------------------------
        # Two coord formats are used:
        #   *_bcn = [B, 3, K]  channel-first, consumed by DGCNN_Propagation
        #   *_bnc = [B, K, 3]  channel-last,  consumed by PointNetFeaturePropagation
        # level-3: group centers (already available as `center`)
        center_level_3_bcn = center.transpose(-1, -2).contiguous()       # [B, 3, G]
        center_level_3_bnc = center.contiguous()                          # [B, G, 3]

        # level-2: 256 FPS points from all N points
        level_2_bnc = _fps(pts_bnc, 256)                                 # [B, 256, 3]
        center_level_2_bnc = level_2_bnc.contiguous()
        center_level_2_bcn = level_2_bnc.transpose(-1, -2).contiguous()  # [B, 3, 256]

        # level-1: 512 FPS points from all N points
        level_1_bnc = _fps(pts_bnc, 512)                                 # [B, 512, 3]
        center_level_1_bnc = level_1_bnc.contiguous()
        center_level_1_bcn = level_1_bnc.transpose(-1, -2).contiguous()  # [B, 3, 512]

        # level-0: all N raw points + label
        center_level_0_bcn = pts_bcn                                     # [B, 3, N]
        center_level_0_bnc = pts_bnc                                     # [B, N, 3]

        # --- Init level features via 3-NN FP ------------------------------
        # f_level_3 = deepest transformer features (layer 11 output)
        f_level_3 = feature_list[2]                                      # [B, C, G]

        # f_level_2 ← propagate layer-7 features from level-3 centers to level-2.
        # PNFP takes channel-last xyz + channel-first features.
        f_level_2 = self.propagation_2(
            center_level_2_bnc, center_level_3_bnc,
            center_level_2_bcn,          # initial xyz-only feature at level-2 (B, 3, 256)
            feature_list[1],             # [B, C, G]
        )                                                                 # [B, C, 256]

        # f_level_1 ← propagate layer-3 features from level-3 centers to level-1
        f_level_1 = self.propagation_1(
            center_level_1_bnc, center_level_3_bnc,
            center_level_1_bcn,          # initial xyz-only feature at level-1 (B, 3, 512)
            feature_list[0],             # [B, C, G]
        )                                                                 # [B, C, 512]

        # level-0 initial feature: [cls_label_one_hot (16→N) | xyz (3, N)]
        if cls_label is not None:
            cls_label_one_hot = cls_label.view(B, self.num_obj_classes, 1).expand(-1, -1, N)
        else:
            cls_label_one_hot = torch.zeros(B, self.num_obj_classes, N, device=pts_bcn.device)
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0_bcn], dim=1)  # [B, 16+3, N]

        # --- DGCNN bottom-up (channel-first coords) -----------------------
        # level3 → level2
        f_level_2 = self.dgcnn_pro_2(center_level_3_bcn, f_level_3, center_level_2_bcn, f_level_2)
        # level2 → level1
        f_level_1 = self.dgcnn_pro_1(center_level_2_bcn, f_level_2, center_level_1_bcn, f_level_1)
        # level1 → level0 (N points) via 3-NN FP (channel-last xyz)
        f_level_0 = self.propagation_0(center_level_0_bnc, center_level_1_bnc, f_level_0, f_level_1)

        # --- Head ---------------------------------------------------------
        feat = F.relu(self.bn1(self.conv1(f_level_0)))   # [B, 128, N]
        x = self.drop1(feat)
        x = self.conv2(x)                                # [B, seg_classes, N]
        return F.log_softmax(x, dim=1)


# ─────────────────────────────────────────────────────────────
#  Registered model classes
# ─────────────────────────────────────────────────────────────

@MODELS.register_module()
class PointMAE_Seg(_StandardTransformerSeg):
    BACKBONE_NAME = "PointMAE"


# PointGPT_Seg is implemented as a dedicated class in models/pointgpt/pointgpt_seg.py
# (it needs manual iteration through GPT_extractor.layers with a causal mask,
# plus SOS-token handling, which does not fit the shared _StandardTransformerSeg).


@MODELS.register_module()
class ACT_Seg(_StandardTransformerSeg):
    BACKBONE_NAME = "ACT"


@MODELS.register_module()
class RECON_Seg(_ReconTransformerSeg):
    BACKBONE_NAME = "RECON"


# PCPMAE_Seg is implemented as a dedicated class in
# models/pcpmae/pcpmae_seg.py. PCP-MAE uses a sinusoidal position
# embedding (get_pos_embed) instead of a linear projection from xyz and
# does not prepend a cls token in the seg forward, so it cannot share
# the _StandardTransformerSeg path.
