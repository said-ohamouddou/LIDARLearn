"""
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

Paper: Hu et al., CVPR 2020 — https://arxiv.org/abs/1911.11236
Source: https://github.com/aRI0U/RandLA-Net-pytorch (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_utils import (
    grouping_operation,    # MIT — Qi / Wijmans et al.
    three_nn,              # MIT
    three_interpolate,     # MIT
    knn_query,             # MIT
)

from ..build import MODELS


# Utilities

def _knn_idx(xyz: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    """
    K nearest neighbors of `query` points in `xyz`.
    Uses pointnet2_ops.knn_query (MIT License — Qi / Wijmans et al.).

    Args:
        xyz   : [B, N, 3]  contiguous float32
        query : [B, M, 3]  contiguous float32
        k     : number of neighbors

    Returns:
        idx : [B, M, k]  int32 — indices into xyz
    """
    _, idx = knn_query(k, xyz.contiguous(), query.contiguous())   # [B, M, k]
    return idx


def _conv2d_bn_act(in_ch: int, out_ch: int, act=nn.LeakyReLU(0.2)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch, eps=1e-6, momentum=0.01),
        act,
    )


def _conv1d_bn_act(in_ch: int, out_ch: int, act=nn.ReLU(inplace=True)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_ch, eps=1e-6, momentum=0.01),
        act,
    )


# Local Spatial Encoding (LocSE).
# For each point and its k-NN, encodes the relative position:
#   [p_i || p_j || (p_i - p_j) || ||p_i - p_j||]  →  10-dim  →  MLP  →  d
# then concatenates the result with the input feature of neighbor j.
# Reference: Section 3 of Hu et al. 2020.

class LocalSpatialEncoding(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """
        d_in  : channel dimension of the incoming feature map
        d_out : output channels of the position MLP
        """
        super().__init__()
        # 10 = 3 (center xyz) + 3 (neighbor xyz) + 3 (relative xyz) + 1 (distance)
        self.pos_mlp = _conv2d_bn_act(10, d_out, act=nn.ReLU(inplace=True))

    def forward(
        self,
        xyz_t: torch.Tensor,   # [B, 3, N]  — point coordinates
        feat: torch.Tensor,   # [B, C, N, 1] — current features
        idx: torch.Tensor,   # [B, N, k]  int32 — knn indices
    ) -> torch.Tensor:
        """Returns [B, d_out + C, N, k]."""
        B, _, N = xyz_t.shape
        k = idx.shape[-1]

        # Gather neighbor coordinates: [B, 3, N, k]
        neighbor_xyz = grouping_operation(xyz_t.contiguous(), idx)

        # Center coordinates broadcast over k: [B, 3, N, k]
        center_xyz = xyz_t.unsqueeze(-1).expand_as(neighbor_xyz)

        # Relative displacement and L2 distance
        diff = center_xyz - neighbor_xyz                               # [B, 3, N, k]
        dist = diff.norm(dim=1, keepdim=True)                          # [B, 1, N, k]

        # Relative position encoding: [B, 10, N, k]
        rel_pos = torch.cat([center_xyz, neighbor_xyz, diff, dist], dim=1)

        # MLP on position encoding: [B, d_out, N, k]
        encoded = self.pos_mlp(rel_pos)

        # Broadcast input feature over k neighbors: [B, C, N, k]
        feat_expand = feat.expand(B, -1, N, k)

        return torch.cat([encoded, feat_expand], dim=1)                # [B, d_out+C, N, k]


# Attentive Pooling.
# Learns per-neighbor attention scores, performs weighted sum over k,
# then applies a pointwise MLP.
# Reference: Section 3 of Hu et al. 2020.

class AttentivePooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Score MLP: [B, in_ch, N, k] → [B, in_ch, N, k] → softmax over k
        self.score_fc = nn.Linear(in_ch, in_ch, bias=False)
        self.out_mlp = _conv2d_bn_act(in_ch, out_ch, act=nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, in_ch, N, k]
        Returns [B, out_ch, N, 1]
        """
        # Compute attention scores over k dimension
        # Transpose to [B, N, k, in_ch] for Linear, then back
        scores = self.score_fc(x.permute(0, 2, 3, 1))                 # [B, N, k, in_ch]
        scores = scores.softmax(dim=2)                                  # softmax over k
        scores = scores.permute(0, 3, 1, 2)                            # [B, in_ch, N, k]

        # Weighted sum over k → [B, in_ch, N, 1]
        aggregated = (scores * x).sum(dim=-1, keepdim=True)

        return self.out_mlp(aggregated)                                 # [B, out_ch, N, 1]


# Local Feature Aggregation (Dilated Residual Block).
# Two stacked LocSE + AttentivePooling blocks with a residual shortcut.
# Output channels = 2 * d_out (to match the skip-connection channel sizes
# used in the encoder/decoder).
# Reference: Section 3 / Figure 4 of Hu et al. 2020.

class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int):
        """
        d_in  : input feature channels
        d_out : intermediate channels  (output will be 2 * d_out)
        k     : number of nearest neighbors
        """
        super().__init__()
        self.k = k

        # Initial pointwise projection
        self.proj = nn.Sequential(
            nn.Conv2d(d_in, d_out // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # First LocSE + pool: in_ch = (d_out//2) + (d_out//2) = d_out
        self.lse1 = LocalSpatialEncoding(d_in=d_out // 2, d_out=d_out // 2)
        self.pool1 = AttentivePooling(in_ch=d_out, out_ch=d_out // 2)

        # Second LocSE + pool
        self.lse2 = LocalSpatialEncoding(d_in=d_out // 2, d_out=d_out // 2)
        self.pool2 = AttentivePooling(in_ch=d_out, out_ch=d_out)

        # Final projection and residual shortcut
        self.out_proj = nn.Conv2d(d_out, 2 * d_out, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(d_in, 2 * d_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * d_out, eps=1e-6, momentum=0.01),
        )
        self.out_bn = nn.BatchNorm2d(2 * d_out, eps=1e-6, momentum=0.01)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        xyz_t: torch.Tensor,   # [B, 3, N]
        feat: torch.Tensor,   # [B, d_in, N, 1]
    ) -> torch.Tensor:
        """Returns [B, 2*d_out, N, 1]."""
        # Compute knn once for both LocSE steps (shared neighborhood)
        xyz = xyz_t.transpose(1, 2).contiguous()                       # [B, N, 3]
        idx = _knn_idx(xyz, xyz, self.k)                               # [B, N, k]

        x = self.proj(feat)                                            # [B, d_out//2, N, 1]

        x = self.lse1(xyz_t, x, idx)                                   # [B, d_out, N, k]
        x = self.pool1(x)                                              # [B, d_out//2, N, 1]

        x = self.lse2(xyz_t, x, idx)                                   # [B, d_out, N, k]
        x = self.pool2(x)                                              # [B, d_out, N, 1]

        main = self.out_bn(self.out_proj(x))
        residual = self.shortcut(feat)
        return self.act(main + residual)                                # [B, 2*d_out, N, 1]


# RandLA-Net.
# Encoder-decoder segmentation network with random point sampling and
# Local Feature Aggregation at each scale.

# NOTE: RandLANet itself is not registered in the MODELS registry — it is
# a pure segmentation backbone with per-point outputs. Use ``RandLANet_Seg``
# (in randlanet_seg.py) which wraps this class and exposes the right I/O
# contract for the seg runner. There are no RandLANet classification
# configs on purpose.
class RandLANet(nn.Module):
    """
    RandLA-Net semantic segmentation model.

    Args:
        d_in         : input feature dimension (e.g., 3 for xyz, 6 for xyz+rgb)
        num_classes  : number of semantic classes
        k            : number of nearest neighbors for LFA
        decimation   : subsampling ratio per encoder stage
        d_encoder    : base channel count (doubles at each stage)
        num_stages   : number of encoder / decoder stages (default 4)
        dropout      : dropout probability in the segmentation head
    """

    def __init__(
        self,
        d_in: int = 6,
        num_classes: int = 13,
        k: int = 16,
        decimation: int = 4,
        d_encoder: int = 8,
        num_stages: int = 4,
        dropout: float = 0.5,
    ):
        # RandLANet is a pure segmentation backbone — build it through
        # ``RandLANet_Seg`` (which wraps this class with the right I/O
        # contract). The kwargs form below is the only supported
        # construction path; there is no classification yaml for RandLANet.
        super().__init__()
        self.decimation = decimation
        self.num_stages = num_stages

        # --- input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_encoder),
            nn.BatchNorm1d(d_encoder, eps=1e-6, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # enc_dims[i] = d_encoder * 2^i  (channels at scale i)
        # LFA at stage i: input enc_dims[i] → output 2*enc_dims[i] = enc_dims[i+1]
        enc_dims = [d_encoder * (2 ** i) for i in range(num_stages + 1)]
        self.enc_dims = enc_dims
        self.num_stages = num_stages

        # --- encoder ---
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(enc_dims[i], enc_dims[i], k)
            for i in range(num_stages)
        ])

        # --- bottleneck ---
        self.bottleneck = _conv1d_bn_act(enc_dims[-1], enc_dims[-1])

        # --- decoder ---
        # At decoder stage i (0 = coarsest → num_stages-1 = finest):
        #   upsampled ch = enc_dims[num_stages - i]   (output of prev stage / bottleneck)
        #   skip ch      = enc_dims[num_stages - i]   (matching encoder skip)
        #   concat ch    = 2 * enc_dims[num_stages - i]
        #   output ch    = enc_dims[num_stages - 1 - i]
        self.decoder = nn.ModuleList([
            _conv1d_bn_act(
                2 * enc_dims[num_stages - i],
                enc_dims[num_stages - 1 - i],
            )
            for i in range(num_stages)
        ])

        # --- segmentation head ---
        head_in = enc_dims[0]   # = d_encoder  (finest decoder output)
        self.seg_head = nn.Sequential(
            _conv1d_bn_act(head_in, 64),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_sample(N: int, n: int, device) -> torch.Tensor:
        """Return `n` random indices in [0, N) without replacement."""
        return torch.randperm(N, device=device)[:n]

    @staticmethod
    def _upsample(
        xyz_sparse: torch.Tensor,   # [B, M, 3]
        xyz_dense: torch.Tensor,   # [B, N, 3]
        feat_sparse: torch.Tensor,   # [B, C, M]
    ) -> torch.Tensor:
        """
        3-NN inverse-distance-weighted feature interpolation from M → N points.
        Uses pointnet2_ops.three_nn + three_interpolate  (MIT License).
        Returns [B, C, N].
        """
        # three_nn expects contiguous float tensors
        xyz_dense_c = xyz_dense.contiguous()
        xyz_sparse_c = xyz_sparse.contiguous()

        dist, idx = three_nn(xyz_dense_c, xyz_sparse_c)                # [B, N, 3]
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)     # [B, N, 3]

        return three_interpolate(feat_sparse.contiguous(), idx, weight)  # [B, C, N]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts : [B, N, d_in]  — point coordinates (+ optional per-point features)

        Returns per-point logits [B, N, num_classes].
        """
        B, N, _ = pts.shape
        device = pts.device

        # Apply random permutation during training for stochastic sampling
        if self.training:
            perm = self._random_sample(N, N, device)
            pts = pts[:, perm]
            inv_perm = torch.argsort(perm)

        # Project input features: [B, N, d_in] → [B, d_encoder, N]
        x = self.input_proj(pts.reshape(B * N, -1)).reshape(B, N, -1)
        x = x.transpose(1, 2).unsqueeze(-1)                            # [B, d_encoder, N, 1]

        xyz = pts[..., :3]                                             # [B, N, 3]

        # ---- Encoder ----
        xyz_stack = []   # downsampled xyz at each stage
        feat_stack = []   # encoded features at each stage

        n_pts = N
        for stage, lfa in enumerate(self.encoder):
            xyz_t = xyz.transpose(1, 2).contiguous()                   # [B, 3, n_pts]

            x = lfa(xyz_t, x)                                          # [B, 2*d, n_pts, 1]

            xyz_stack.append(xyz)
            feat_stack.append(x.squeeze(-1))                           # [B, C, n_pts]

            # Random downsample
            n_pts = n_pts // self.decimation
            idx = self._random_sample(xyz.shape[1], n_pts, device)
            xyz = xyz[:, idx]
            x = x[:, :, idx]

        # ---- Bottleneck ----
        x = self.bottleneck(x.squeeze(-1))                             # [B, C, n_pts]

        # ---- Decoder ----
        for stage, mlp in enumerate(self.decoder):
            # Pop the corresponding encoder scale
            enc_xyz = xyz_stack.pop()                                 # [B, N_prev, 3]
            enc_feat = feat_stack.pop()                                # [B, C_enc, N_prev]

            # Upsample features from current scale to encoder scale
            x = self._upsample(xyz, enc_xyz, x)                       # [B, C, N_prev]

            # Skip connection
            x = torch.cat([x, enc_feat], dim=1)                       # [B, C+C_enc, N_prev]
            x = mlp(x)

            xyz = enc_xyz                                              # move up in resolution

        # ---- Segmentation head ----
        logits = self.seg_head(x)                                      # [B, num_classes, N]

        # Undo permutation
        if self.training:
            logits = logits[:, :, inv_perm]

        return logits.transpose(1, 2)                                  # [B, N, num_classes]
