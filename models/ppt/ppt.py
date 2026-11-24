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
from timm.models.layers import DropPath, trunc_normal_

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log
from utils.knn import knn_point

from ..ssl_blocks import Mlp, Attention, Encoder

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, prompt_mlp):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        h_x = prompt_mlp(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) + h_x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_gruop=None, num_group_extra=None):
        super().__init__()
        self.num_gruop = num_gruop
        self.num_gruop_center = num_group_extra
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = x + pos
            x[:, 1:self.num_gruop + 1] = x[:, 1:self.num_gruop + 1] + self.prompt_mlp(x[:, 1:self.num_gruop + 1])
            x[:, -self.num_gruop_center:] = x[:, -self.num_gruop_center:] + self.prompt_pos_mlp(x[:, -self.num_gruop_center:])
            x = block(x, self.prompt_all_mlp)
        return x

@MODELS.register_module()
class PPT(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super(PPT, self).__init__(config, num_classes)

        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.group_size_extra = config.prompt_group_size
        self.num_group_extra = config.prompt_num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.center_divider = Group(num_group=self.num_group_extra, group_size=self.group_size_extra)
        self.encoder = Encoder(encoder_channel=self.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            num_gruop=self.num_group,
            num_group_extra=self.num_group_extra
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.embed_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            # Accept a few common checkpoint wrappers: {'base_model': sd},
            # {'model': sd}, or a bare state_dict.
            if isinstance(ckpt, dict) and 'base_model' in ckpt:
                raw = ckpt['base_model']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                raw = ckpt['model']
            else:
                raw = ckpt
            base_ckpt = {k.replace("module.", ""): v for k, v in raw.items()}

            total_ckpt_params = sum(p.numel() for p in base_ckpt.values())

            for k in list(base_ckpt.keys()):
                # For PointMAE, ReCon, PCP-MAE
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # For ACT
                if k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # For Point-BERT
                if k.startswith('transformer_k'):
                    base_ckpt[k[len('transformer_k.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # For PointGPT
                if k.startswith('GPT_Transformer'):
                    base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                # Remove classification head from checkpoint
                if 'cls_head_finetune' in k:
                    del base_ckpt[k]

            model_state = self.state_dict()
            matched_params = 0
            matched_keys = []
            for k, v in base_ckpt.items():
                if k in model_state and v.shape == model_state[k].shape:
                    matched_params += v.numel()
                    matched_keys.append(k)

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            used_percentage = (matched_params / total_ckpt_params * 100) if total_ckpt_params > 0 else 0.0
            print_log(f'[PPT] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[PPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        neighbor_center, center_center = self.center_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        center_input_tokens = self.encoder(neighbor_center)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        center_pos = self.pos_embed(center_center)

        x = torch.cat((cls_tokens, group_input_tokens, center_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos, center_pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:1 + self.num_group].mean(1), x[:, -self.num_group_extra:].mean(1)], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
