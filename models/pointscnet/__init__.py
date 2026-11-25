"""
PointSCNet: Point Cloud Structure and Correlation Learning Based on
            Space-Filling Curve-Guided Sampling

Paper: Chen et al., Symmetry 2022 — https://doi.org/10.3390/sym14010008
Source: https://github.com/Chenguoz/PointSCNet

Clean implementation.
"""

from .network import PointSCNet
from .curve_sampling import HilbertSampler
from .feature_layers import (
    LocalRegionEncoder,
    MultiScaleLocalEncoder,
    FeatureUpsampler
)
from .attention import DualPathAttention

__all__ = [
    'PointSCNet',
    'HilbertSampler',
    'LocalRegionEncoder',
    'MultiScaleLocalEncoder',
    'FeatureUpsampler',
    'DualPathAttention'
]
