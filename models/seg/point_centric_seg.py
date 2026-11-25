"""
Point-centric segmentation wrapper.

Automatically converts any point-centric classification model (DGCNN, PointNet,
MSDGCNN2, etc.) into a segmentation model by capturing per-point intermediate
features via forward hooks and attaching a per-point convolutional head.

Conversion pattern (identical to dgcnn_segementation.py's DGCNN_partseg):
    1. Run the backbone, capturing per-point features at specified layers.
    2. Aggregate the final per-point feature map into a global descriptor
       (max+avg pool) and broadcast it back to N points.
    3. (Optional) Project a one-hot object class label to a feature vector
       and broadcast to N points.
    4. Concatenate [global_broadcast, *intermediates, (cls_label_broadcast)]
       and apply a per-point 1D-conv head to produce [B, seg_classes, N].

Usage:
    backbone = build_model_from_cfg(cls_config)   # existing classification model
    seg_model = PointCentricSegWrapper(
        backbone=backbone,
        seg_classes=50,
        adapter_name='DGCNN',        # looks up SEG_ADAPTER_CONFIG['DGCNN']
        use_cls_label=True,          # True for part seg, False for semantic seg
    )
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_seg_model import BaseSegModel
from utils.checkpoint import load_checkpoint


# Adapter registry: maps backbone class name → where to hook for per-point feats.
# Each entry specifies:
#   'hook_modules': list of module attribute paths (dot-notation) on the
#                   backbone whose OUTPUT gives per-point features [B, C, N].
#   'inter_channels': list of channel counts matching hook_modules (for dim check).
#   'final_feature_module': attribute path of the module whose output is the
#                   final per-point feature map [B, emb_dims, N] to be pooled.
#   'emb_dims_attr': attribute on the backbone giving emb_dims (e.g. 'emb_dims').
# The wrapper registers forward hooks on these modules; it does NOT require
# modifying the backbone's forward() at all.
SEG_ADAPTER_CONFIG: Dict[str, dict] = {
    # DGCNN: EdgeConv layers produce per-point feats at [64, 64, 128, 256]
    # after max-over-k aggregation, then conv5 produces [emb_dims, N].
    'DGCNN': {
        'hook_modules': ['conv1', 'conv2', 'conv3', 'conv4'],
        'inter_channels': [64, 64, 128, 256],
        'final_feature_module': 'conv5',
        'emb_dims_attr': 'emb_dims',
    },
    # MSDGCNN2: multi-scale fusion produces x1, x2, x3, x4 per-point feats.
    # Uses forward_with_intermediates() to get them — see custom_forward below.
    'MSDGCNN2': {
        'hook_modules': None,  # handled via forward_with_intermediates
        'custom_forward': 'msdgcnn2',
        'inter_channels': [64, 64, 128, 256],  # if fusion=='concat_conv' else [128,64,128,256]
        'emb_dims_attr': 'emb_dims',
    },
    # PointNet: per-point features at conv3 (64), conv4 (128), conv5 (emb_dims).
    # conv5's output IS the final per-point feature map.
    'PointNet': {
        'hook_modules': ['conv3', 'conv4'],
        'inter_channels': [64, 128],
        'final_feature_module': 'conv5',
        'emb_dims_attr': 'emb_dims',
    },
    # GDAN: 3 stages produce per-point features, all preserving N points.
    # stage1_geo(64), stage2_geo(64), stage3_proj(128), global_conv(512).
    'GDAN': {
        'hook_modules': ['stage1_geo', 'stage2_geo', 'stage3_proj'],
        'inter_channels': [64, 64, 128],
        'final_feature_module': 'global_conv',
        'emb_dims_attr': 'emb_dims',
    },
    # KANDGCNN: KAN-based DGCNN. conv1(128) after max-pool over K, conv5(emb_dims).
    'KANDGCNN': {
        'hook_modules': ['conv1'],
        'inter_channels': [128],
        'final_feature_module': 'conv5',
        'emb_dims_attr': 'emb_dims',
    },
}


class _FeatureCapture:
    """Helper that registers forward hooks and captures per-point features."""

    def __init__(self, backbone: nn.Module, module_paths: List[str]):
        self.features: Dict[str, torch.Tensor] = {}
        self._handles = []
        for path in module_paths:
            module = _get_module_by_path(backbone, path)
            handle = module.register_forward_hook(self._make_hook(path))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            # If the layer output is a tuple, take the first element.
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.features[name] = out
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    m = root
    for p in path.split('.'):
        m = getattr(m, p)
    return m


def _as_BCN(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, C, N]. Accepts [B,C,N] or [B,C,N,K]."""
    if t.dim() == 4:
        # EdgeConv raw output is [B, C, N, K] — reduce over K via max.
        t = t.max(dim=-1)[0]
    return t


class PointCentricSegWrapper(BaseSegModel):
    """Generic segmentation wrapper for point-centric classification backbones.

    The wrapper runs the backbone's forward (or a custom extractor) purely to
    trigger the hooks that capture intermediate per-point features. The
    backbone's own classification head output is discarded.

    Args:
        backbone: a classification nn.Module (point-centric architecture).
        seg_classes: number of segmentation classes (e.g. 50 for ShapeNet Parts).
        adapter_name: key into SEG_ADAPTER_CONFIG.
        use_cls_label: if True, expect a one-hot [B, num_obj_classes] label and
                       project it to a 64-dim per-point feature.
        num_obj_classes: number of object categories (16 for ShapeNet Parts).
        dropout: dropout rate in the seg head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        seg_classes: int,
        adapter_name: str,
        use_cls_label: bool = True,
        num_obj_classes: int = 16,
        dropout: float = 0.5,
        config=None,
    ):
        # Build a lightweight config dict for BaseSegModel if not provided.
        if config is None:
            from easydict import EasyDict
            config = EasyDict(
                seg_classes=seg_classes,
                num_obj_classes=num_obj_classes,
                use_cls_label=use_cls_label,
            )
        super().__init__(config, seg_classes=seg_classes, num_obj_classes=num_obj_classes)

        if adapter_name not in SEG_ADAPTER_CONFIG:
            raise KeyError(
                f"No seg adapter registered for '{adapter_name}'. "
                f"Available: {list(SEG_ADAPTER_CONFIG.keys())}"
            )
        self.adapter = SEG_ADAPTER_CONFIG[adapter_name]
        self.adapter_name = adapter_name
        self.backbone = backbone
        self.use_cls_label = use_cls_label

        # Resolve emb_dims from the backbone
        emb_dims_attr = self.adapter['emb_dims_attr']
        self.emb_dims = getattr(backbone, emb_dims_attr)

        # Hook setup (None if using custom_forward path)
        self._capture: Optional[_FeatureCapture] = None
        if self.adapter.get('hook_modules') is not None:
            self._capture = _FeatureCapture(backbone, self.adapter['hook_modules'])

        inter_total = sum(self.adapter['inter_channels'])
        global_dim = self.emb_dims * 2  # max + avg pool

        label_dim = 0
        if use_cls_label:
            self.label_conv = nn.Sequential(
                nn.Conv1d(num_obj_classes, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
            )
            label_dim = 64

        in_dim = global_dim + inter_total + label_dim
        self.seg_head = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, seg_classes, kernel_size=1, bias=False),
        )

    # --- Backbone-specific feature extraction routines ---------------------

    def _extract_dgcnn_like(self, pts: torch.Tensor):
        """For DGCNN / PointNet style backbones using hooks."""
        self._capture.clear()
        # Run backbone forward solely to trigger hooks; discard logits.
        _ = self.backbone(pts)
        inters = [
            _as_BCN(self._capture.features[name])
            for name in self.adapter['hook_modules']
        ]
        # Final feature map: hook on final_feature_module
        # (not registered separately; we call it explicitly below)
        # Since it's not in hook_modules, we reconstruct it: use last inter
        # feature's downstream path. Simpler: register a separate hook once.
        final_path = self.adapter['final_feature_module']
        if final_path not in self._capture.features:
            # Register the final hook lazily
            mod = _get_module_by_path(self.backbone, final_path)
            handle = mod.register_forward_hook(
                self._capture._make_hook(final_path)
            )
            self._capture._handles.append(handle)
            # Run forward again with the new hook in place
            _ = self.backbone(pts)
        final_feat = _as_BCN(self._capture.features[final_path])  # [B, emb, N]
        return inters, final_feat

    def _extract_msdgcnn2(self, pts: torch.Tensor):
        """MSDGCNN2 already has forward_with_intermediates()."""
        # forward_with_intermediates returns (logits, {'x1','x2','x3','x4', 'features'})
        out = self.backbone.forward_with_intermediates(pts)
        if isinstance(out, tuple):
            _, intermediates = out
        else:
            intermediates = out
        # Expect keys x1..x4 (per-point) and a final per-point feature map.
        inters = [intermediates[k] for k in ('x1', 'x2', 'x3', 'x4')]
        final_feat = intermediates.get('features', None)
        if final_feat is None:
            # Fall back: concat intermediates as the "final" feature for pooling
            final_feat = torch.cat(inters, dim=1)
        return inters, final_feat

    # --- Forward ------------------------------------------------------------

    def forward(self, pts: torch.Tensor, cls_label: Optional[torch.Tensor] = None):
        # Normalize input to [B, 3, N] (backbones handle their own reshaping
        # internally, but we need N for broadcasting).
        if pts.dim() == 3 and pts.shape[1] != 3 and pts.shape[2] == 3:
            N = pts.shape[1]
        else:
            N = pts.shape[-1]

        custom = self.adapter.get('custom_forward')
        if custom == 'msdgcnn2':
            inters, final_feat = self._extract_msdgcnn2(pts)
        else:
            inters, final_feat = self._extract_dgcnn_like(pts)

        B = final_feat.shape[0]
        # Global max + avg pool of final per-point features
        g_max = final_feat.max(dim=-1)[0]            # [B, emb_dims]
        g_avg = final_feat.mean(dim=-1)               # [B, emb_dims]
        g = torch.cat([g_max, g_avg], dim=1)          # [B, emb_dims*2]
        g_broadcast = g.unsqueeze(-1).expand(-1, -1, N)  # [B, emb_dims*2, N]

        feats = [g_broadcast] + inters

        if self.use_cls_label:
            if cls_label is None:
                raise ValueError(
                    "use_cls_label=True but no cls_label was passed to forward()."
                )
            # cls_label expected [B, num_obj_classes] one-hot
            cls_feat = self.label_conv(cls_label.unsqueeze(-1))  # [B, 64, 1]
            feats.append(cls_feat.expand(-1, -1, N))

        x = torch.cat(feats, dim=1)
        x = self.seg_head(x)  # [B, seg_classes, N]
        x = F.log_softmax(x, dim=1)
        return x

    def load_backbone_ckpt(self, ckpt_path: str, strict: bool = False):
        """Load pretrained classification weights into the backbone."""
        state = load_checkpoint(ckpt_path, map_location='cpu')
        if 'base_model' in state:
            sd = {k.replace('module.', ''): v for k, v in state['base_model'].items()}
        elif 'model' in state:
            sd = {k.replace('module.', ''): v for k, v in state['model'].items()}
        else:
            sd = state
        incompatible = self.backbone.load_state_dict(sd, strict=strict)
        return incompatible
