"""
Surface Representation for Point Clouds

Paper: CVPR 2022
Authors: Haoxi Ran, Jun Liu, Chengjie Wang
Source: Implementation adapted from: https://github.com/hancyran/RepSurf
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample, knn_query, ball_query, gather_operation, grouping_operation, three_nn
from .polar_utils import xyz2sphere
from .recons_utils import cal_const, cal_normal, cal_center, check_nan_umb


def index_points(points, idx, cuda=False, is_group=False):
    """
    Index points using either CUDA operations or CPU fallback.

    Parameters
    ----------
    points : torch.Tensor
        (B, N, C) tensor of point features
    idx : torch.Tensor
        (B, npoint) or (B, npoint, nsample) tensor of indices
    cuda : bool
        Whether to use CUDA operations
    is_group : bool
        Whether this is a grouping operation (3D idx) or gathering (2D idx)

    Returns
    -------
    torch.Tensor
        Indexed points: (B, npoint, nsample, C) if is_group, else (B, npoint, C)
    """
    if cuda:
        # Convert idx to int32 for CUDA operations
        idx = idx.int()  # Add this line

        if is_group:
            points_grouped = grouping_operation(points.transpose(1, 2).contiguous(), idx)
            return points_grouped.permute(0, 2, 3, 1).contiguous()
        else:
            points_gathered = gather_operation(points.transpose(1, 2).contiguous(), idx)
            return points_gathered.permute(0, 2, 1).contiguous()

    # CPU fallback
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, center, normal, feature, return_normal=True, return_polar=False, cuda=False):
    """
    Input:
        center: input points position data
        normal: input points normal data
        feature: input points feature
    Return:
        new_center: sampled points position data
        new_normal: sampled points normal data
        new_feature: sampled points feature
    """
    # sample
    center = center.contiguous()
    fps_idx = furthest_point_sample(center, npoint)  # [B, npoint, A]
    torch.cuda.empty_cache()
    # sample center
    new_center = index_points(center, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    # sample normal
    new_normal = index_points(normal, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()

    # group
    idx = ball_query(radius, nsample, center, new_center)
    torch.cuda.empty_cache()
    # group normal
    group_normal = index_points(normal, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, B]
    torch.cuda.empty_cache()
    # group center
    group_center = index_points(center, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, A]
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)
    if feature is not None:
        group_feature = index_points(feature, idx, cuda=cuda, is_group=True)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature


def sample_and_group_all(center, normal, feature, return_normal=True, return_polar=False):
    """
    Input:
        center: input centroid position data
        normal: input normal data
        feature: input feature data
    Return:
        new_center: sampled points position data
        new_normal: sampled points position data
        new_feature: sampled points data
    """
    device = center.device
    B, N, C = normal.shape

    new_center = torch.zeros(B, 1, 3).to(device)
    new_normal = new_center

    group_normal = normal.view(B, 1, N, C)
    group_center = center.view(B, 1, N, 3)
    if return_polar:
        group_polar = xyz2sphere(group_center)
        group_center = torch.cat([group_center, group_polar], dim=-1)

    new_feature = torch.cat([group_center, group_normal, feature.view(B, 1, N, -1)], dim=-1) if return_normal \
        else torch.cat([group_center, feature.view(B, 1, N, -1)], dim=-1)

    return new_center, new_normal, new_feature


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points


def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    # Ensure tensors are contiguous before calling knn_query
    xyz = xyz.contiguous()
    new_xyz = new_xyz.contiguous()

    idx = knn_query(k, xyz, new_xyz)[0]
    idx = idx.int()
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, return_polar=True, return_normal=True, cuda=False):
        super(SurfaceAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_polar=self.return_polar,
                                                                       return_normal=self.return_normal)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_polar=self.return_polar,
                                                                   return_normal=self.return_normal, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class SurfaceFeaturePropagationCD(nn.Module):
    """Surface Feature Propagation with Channel De-differentiation (batched)."""

    def __init__(self, prev_channel, skip_channel, mlp):
        super().__init__()
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if self.skip:
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat1, pos_feat2):
        """
        pos_feat1: [center1 [B,3,N1], feat1 [B,C1,N1] or None]
        pos_feat2: [center2 [B,3,N2], feat2 [B,C2,N2]]
        Returns: upsampled features [B, out_C, N1]
        """
        xyz1 = pos_feat1[0].permute(0, 2, 1).contiguous()  # [B, N1, 3]
        xyz2 = pos_feat2[0].permute(0, 2, 1).contiguous()  # [B, N2, 3]
        points2 = pos_feat2[1].permute(0, 2, 1).contiguous()  # [B, N2, C2]

        B, N1, _ = xyz1.shape

        # 3-NN interpolation
        dist, idx = three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)  # [B, N1, 3]

        # Project + BN source features BEFORE interpolation (matches original)
        N2 = points2.shape[1]
        points2_proj = self.norm_f0(self.mlp_f0(points2).reshape(B * N2, -1)).reshape(B, N2, -1)
        # Weighted interpolation from N2 source to N1 target
        interpolated = torch.zeros(B, N1, points2_proj.shape[-1], device=xyz1.device)
        for i in range(3):
            interpolated += points2_proj[torch.arange(B).unsqueeze(1), idx[:, :, i].long()] * weight[:, :, i:i+1]

        # Skip connection
        if self.skip and pos_feat1[1] is not None:
            points1 = pos_feat1[1].permute(0, 2, 1).contiguous()  # [B, N1, C1]
            skip = self.norm_s0(self.mlp_s0(points1).reshape(B * N1, -1)).reshape(B, N1, -1)
            new_points = F.relu(interpolated + skip)
        else:
            new_points = F.relu(interpolated)

        # MLP layers
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points).reshape(B * N1, -1)).reshape(B, N1, -1))

        return new_points.permute(0, 2, 1).contiguous()  # [B, out_C, N1]


class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False, cuda=False):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.group_all = group_all

        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv2d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True, cuda=False):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        center = center.permute(0, 2, 1)
        # surface construction
        group_xyz = group_by_umbrella(center, center, k=self.k, cuda=self.cuda)  # [B, N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature
