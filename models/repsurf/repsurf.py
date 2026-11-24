"""
Surface Representation for Point Clouds

Paper: CVPR 2022
Authors: Haoxi Ran, Jun Liu, Chengjie Wang
Source: Implementation adapted from: https://github.com/hancyran/RepSurf
License: Apache-2.0
"""

import torch
import torch.nn as nn

from .repsurface_utils import SurfaceAbstractionCD, UmbrellaSurfaceConstructor
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class RepSurf(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.num_classes
        super(RepSurf, self).__init__(config, num_classes)

        self.num_point = config.num_point
        self.return_dist = config.return_dist
        self.return_center = config.return_center
        self.return_polar = config.return_polar
        self.group_size = config.group_size
        self.umb_pool = config.umb_pool
        self.cuda_ops = config.cuda_ops
        self.dropout1 = config.dropout1
        self.dropout2 = config.dropout2

        center_channel = 0 if not self.return_center else (6 if self.return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = self.num_point
        self.surface_constructor = UmbrellaSurfaceConstructor(self.group_size + 1, repsurf_channel,
                                                              return_dist=self.return_dist, aggr_type=self.umb_pool,
                                                              cuda=self.cuda_ops)
        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.2, nsample=32, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[64, 64, 128], group_all=False,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.4, nsample=64, feat_channel=128 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[128, 128, 256], group_all=False,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.sa3 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[256, 512, 1024], group_all=True,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.classfier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(self.dropout1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(self.dropout2),
            nn.Linear(256, self.num_classes))

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            points = pointcloud
        else:
            points = pointcloud.permute(0, 2, 1)

        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)

        feature = feature.view(-1, 1024)
        feature = self.classfier(feature)

        return feature

@MODELS.register_module()
class RAPSurfx2(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.num_classes
        super(RAPSurfx2, self).__init__(config, num_classes)

        self.num_point = config.num_point
        self.return_dist = config.return_dist
        self.return_center = config.return_center
        self.return_polar = config.return_polar
        self.group_size = config.group_size
        self.umb_pool = config.umb_pool
        self.cuda_ops = config.cuda_ops
        self.dropout1 = config.dropout1
        self.dropout2 = config.dropout2

        center_channel = 0 if not self.return_center else (6 if self.return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = self.num_point
        self.surface_constructor = UmbrellaSurfaceConstructor(self.group_size + 1, repsurf_channel,
                                                              return_dist=self.return_dist, aggr_type=self.umb_pool,
                                                              cuda=self.cuda_ops)
        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.1, nsample=24, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[128, 128, 256], group_all=False,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.2, nsample=24, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[256, 256, 512], group_all=False,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.sa3 = SurfaceAbstractionCD(npoint=32, radius=0.4, nsample=24, feat_channel=512 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[512, 512, 1024], group_all=False,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.sa4 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=1024 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[1024, 1024, 2048], group_all=True,
                                        return_polar=self.return_polar, cuda=self.cuda_ops)
        self.classfier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(self.dropout1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(self.dropout2),
            nn.Linear(256, self.num_classes))

    def forward(self, pointcloud):
        if pointcloud.shape[1] == 3 or (pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]):
            points = pointcloud
        else:
            points = pointcloud.permute(0, 2, 1)

        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)
        center, normal, feature = self.sa4(center, normal, feature)

        feature = feature.view(-1, 2048)
        feature = self.classfier(feature)

        return feature
