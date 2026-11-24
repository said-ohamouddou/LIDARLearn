"""
Decoupled Local Aggregation for Point Cloud Learning

Paper: arXiv 2308.16532, 2023
Authors: Binjie Chen, Yunzhou Xia, Yu Zang, Cheng Wang, Jonathan Li
Source: Implementation adapted from: https://github.com/Matrix-ASC/DeLA
License: MIT
"""

# Standard library
from easydict import EasyDict

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.amp import autocast
from pointnet2_ops import pointnet2_utils
from dela_cutils import knn_edge_maxpooling

# Local (relative imports)
from timm.models.layers import DropPath
from ..build import MODELS
from ..base_model import BasePointCloudModel

@autocast('cuda', enabled=False)
def calc_pwd(x):
    x2 = x.square().sum(dim=2, keepdim=True)
    return x2 + x2.transpose(1, 2) + torch.bmm(x, x.transpose(1, 2).mul(-2))

def get_graph_feature(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N * k, 1).expand(-1, -1, C)).view(B * N, k, C)
    x = x.view(B * N, 1, C).expand(-1, k, -1)
    return nbr - x

def get_nbr_feature(x, idx):
    B, N, k = idx.shape
    C = x.shape[-1]
    nbr = torch.gather(x, 1, idx.view(B, N * k, 1).expand(-1, -1, C)).view(B * N * k, C)
    return nbr

class LFP(nn.Module):
    r"""
    Local Feature Propagation Layer
    f = linear(f)
    f_i = bn(max{f_j | j in knn_i} - f_i)
    """

    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.proj(x)
        x = knn_edge_maxpooling(x, knn, self.training)
        x = self.bn(x.view(B * N, -1)).view(B, N, -1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = self.mlp(x.view(B * N, -1)).view(B, N, -1)
        return x

class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.lfps = nn.ModuleList([
            LFP(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])

    def forward(self, x, knn):
        x = x + self.drop_paths[0](self.mlp(x))
        for i in range(self.depth):
            x = x + self.drop_paths[i](self.lfps[i](x, knn))
            if i % 2 == 1:
                x = x + self.drop_paths[i](self.mlps[i // 2](x))
        return x

class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth

        self.first = first = depth == 0
        self.last = last = depth == len(args.depths) - 1

        self.n = args.ns[depth]
        self.k = args.ks[depth]

        dim = args.dims[depth]
        nbr_in_dim = 4 if self.first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim // 2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim // 2, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False)
        )
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Identity() if first else nn.Linear(nbr_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, args.bn_momentum, args.act)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    def forward(self, x, xyz, prev_knn, pwd):
        if not self.first:
            xyz = xyz[:, :self.n].contiguous()
            B, N, C = x.shape
            x = self.skip_proj(x.view(B * N, C)).view(B, N, -1)[:, :self.n] + self.lfp(x, prev_knn)[:, :self.n]

        _, knn = pwd[:, :self.n, :self.n].topk(k=self.k, dim=-1, largest=False, sorted=False)

        B, N, k = knn.shape
        nbr = get_graph_feature(xyz, knn).view(-1, 3)
        if self.first:
            height = xyz[..., 1:2] / 10
            height -= height.min(dim=1, keepdim=True)[0]
            if self.training:
                height += torch.empty((B, 1, 1), device=xyz.device).uniform_(-0.2, 0.2) * 4
            nbr = torch.cat([nbr, get_nbr_feature(height, knn)], dim=1)

        nbr = self.nbr_embed(nbr).view(B * N, k, -1).max(dim=1)[0]
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr).view(B, N, -1)
        x = nbr if self.first else nbr + x

        x = self.blk(x, knn)

        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, pwd)
        else:
            sub_x = x
            sub_c = None

        if self.training:
            rel_k = torch.randint(self.k, (B, N, 1), device=x.device)
            rel_k = torch.gather(knn, 2, rel_k)
            rel_cor = get_graph_feature(xyz, rel_k).flatten(1).mul_(self.cor_std)
            rel_p = get_graph_feature(x, rel_k).flatten(1)
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        return sub_x, sub_c

@MODELS.register_module()
class DELA(BasePointCloudModel):
    r"""
    DELA for Classification

    Args:
        config: Configuration object with model parameters

    Config parameters:
        depths:         [4, 4, 4]
        dims:           [96, 192, 384]
        ns:             [1024, 256, 64]
        ks:             [20, 20, 20]
        nbr_dims:       [48, 48]
        bottleneck:     2048
        num_classes:    40
        drop_path:      0.15
        bn_momentum:    0.1
        mlp_ratio:      2
        cor_std:        [2.8, 5.3, 10]
    """

    def __init__(self, config, num_classes=40, **kwargs):
        super(DELA, self).__init__(config, num_classes)

        args = EasyDict()
        args.depths = config.depths
        args.ns = config.ns
        args.ks = config.ks
        args.dims = config.dims
        args.nbr_dims = config.nbr_dims
        args.bottleneck = config.bottleneck
        args.num_classes = self.num_classes

        drop_path = config.drop_path
        drop_rates = torch.linspace(0., drop_path, sum(args.depths)).split(args.depths)
        args.drop_paths = [dpr.tolist() for dpr in drop_rates]

        args.bn_momentum = config.bn_momentum
        args.act = nn.GELU
        args.mlp_ratio = config.mlp_ratio
        args.cor_std = config.cor_std

        self.stage = Stage(args)

        in_dim = args.dims[-1]
        bottleneck = args.bottleneck

        self.proj = nn.Sequential(
            nn.BatchNorm1d(in_dim, momentum=args.bn_momentum),
            nn.Linear(in_dim, bottleneck),
            args.act()
        )

        self.head = nn.Sequential(
            nn.Linear(bottleneck, 512, bias=False),
            nn.BatchNorm1d(512, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(.5),
            nn.Linear(256, self.num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        if xyz.dim() == 3 and xyz.shape[1] == 3:
            xyz = xyz.permute(0, 2, 1)

        if not self.training:
            idx = pointnet2_utils.furthest_point_sample(xyz, 1024).long()
            xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        pwd = calc_pwd(xyz)
        x, closs = self.stage(None, xyz, None, pwd)
        B, N, _ = x.shape
        x = self.proj(x.view(B * N, -1)).view(B, N, -1).max(dim=1)[0]
        if self.training:
            return self.head(x), closs
        else:
            return self.head(x)
