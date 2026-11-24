"""
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

Paper: Hu et al., CVPR 2020 — https://arxiv.org/abs/1911.11236
Source: https://github.com/aRI0U/RandLA-Net-pytorch (no license)

Clean implementation.
"""

from .randlanet import (
    LocalSpatialEncoding,
    AttentivePooling,
    LocalFeatureAggregation,
    RandLANet,
)

__all__ = [
    "LocalSpatialEncoding",
    "AttentivePooling",
    "LocalFeatureAggregation",
    "RandLANet",
]
