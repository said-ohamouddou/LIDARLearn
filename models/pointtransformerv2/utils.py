"""
Point Transformer V2: Grouped Vector Attention and Partition-based Pooling

Paper: NeurIPS 2022
Authors: Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao
Source: Implementation adapted from: https://github.com/Pointcept/PointTransformerV2
License: MIT
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_cluster import grid_cluster
from torch_scatter import segment_csr

import einops
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
import pointops


def voxel_grid(pos, size, batch=None, start=None, end=None):
    """Assign each point of a batched point cloud to a voxel of edge ``size``.

    Same semantics as ``torch_geometric.nn.pool.voxel_grid`` in PyG <= 2.6
    (MIT licence), computed directly with ``torch_cluster.grid_cluster``.
    PyG >= 2.7 routes this call through the extra ``pyg-lib`` package, so
    keeping it here lets PTv2 run with any PyG version, or without PyG.
    """
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    dim = pos.size(1)
    if batch is None:
        batch = pos.new_zeros(pos.size(0), dtype=torch.long)
    # The batch index becomes an extra coordinate so samples never share voxels.
    pos = torch.cat([pos, batch.view(-1, 1).to(pos.dtype)], dim=-1)

    def per_dim(value):
        value = torch.as_tensor(value, dtype=pos.dtype, device=pos.device).flatten()
        if value.numel() == 1:
            return value.repeat(dim)
        if value.numel() != dim:
            raise ValueError(f"expected a scalar or {dim} values, got {value.numel()}")
        return value

    size = torch.cat([per_dim(size), pos.new_ones(1)])
    if start is not None:
        start = torch.cat([per_dim(start), pos.new_zeros(1)])
    if end is not None:
        end = torch.cat([per_dim(end), batch.max().unsqueeze(0).to(pos.dtype)])
    return grid_cluster(pos, size, start, end)


def offset2batch(offset):
    """Convert offset to batch indices."""
    return torch.cat([
        torch.tensor([i] * (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o)
        for i, o in enumerate(offset)
    ], dim=0).long().to(offset.device)


def batch2offset(batch):
    """Convert batch indices to offset."""
    return torch.cumsum(batch.bincount(), dim=0).long()


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedLinear(nn.Module):
    """Element-wise multiplication with grouped sum reduction.

    Unlike nn.Linear, this has a single shared weight vector [1, in_features]
    and performs: (input * weight).reshape(..., groups, in_features//groups).sum(-1)
    """

    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        assert in_features % groups == 0
        assert out_features == groups
        self.weight = nn.Parameter(torch.empty(1, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return (
            (input * self.weight)
            .reshape(list(input.shape[:-1]) + [self.groups, input.shape[-1] // self.groups])
            .sum(-1)
        )


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            GroupedLinear(embed_channels, groups, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class UnpoolWithSkip(nn.Module):
    """Map Unpooling with skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels,
                 bias=True, skip=True, backend="map"):
        super().__init__()
        self.skip = skip
        self.backend = backend
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            )
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Decoder(nn.Module):
    """Point Transformer V2 decoder stage."""

    def __init__(self, in_channels, skip_channels, embed_channels, groups,
                 depth, neighbours=16, qkv_bias=True, pe_multiplier=False,
                 pe_bias=True, attn_drop_rate=None, drop_path_rate=None,
                 enable_checkpoint=False, unpool_backend="map"):
        super().__init__()
        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)
