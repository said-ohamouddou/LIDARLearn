"""
Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis

Paper: CVPR 2024
Authors: Xin Zhou, Dingkang Liang, Wei Xu, Xingkui Zhu, Yihan Xu, Zhikang Zou, Xiang Bai
Source: Implementation adapted from: https://github.com/LMD0311/DAPT
License: Apache-2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from models.knn import KNN

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
from utils.logger import print_log

from ..ssl_blocks import Encoder


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class DAPTMlp(nn.Module):
    """DAPT-specific Mlp with TFTS feature transformations post-fc1 and post-fc2."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(hidden_features)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = apply_tfts(x, self.tfts_gamma_1, self.tfts_beta_1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        x = self.drop(x)
        return x


class DAPTAttention(nn.Module):
    """DAPT-specific Attention with TFTS post-qkv and post-proj."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(dim * 3)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = apply_tfts(self.qkv(x), self.tfts_gamma_1, self.tfts_beta_1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 adapter_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None, rank=64,
                 depth=0):
        super().__init__()
        self.depth = depth
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DAPTMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = DAPTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Adapter_MLP = Adapter(d_model=dim, bottleneck=rank, dropout=adapter_drop)
        self.non_linear_func = nn.GELU()

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(dim)
        self.tfts_gamma_3, self.tfts_beta_3 = init_tfts(dim)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(apply_tfts(self.norm1(x), self.tfts_gamma_1, self.tfts_beta_1)))
        h = x
        x = self.norm2(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        adapter_mlp = self.Adapter_MLP(x)
        x = self.mlp(x) + adapter_mlp
        x = h + self.drop_path2(x)
        adapter_prompt = self.non_linear_func(adapter_mlp).mean(dim=1, keepdim=True)
        adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma_3, self.tfts_beta_3)
        x = torch.cat([x[:, 0:1, :], adapter_prompt, x[:, 1:, :]], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_dim=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 use_square=False, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.use_square = use_square
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        self.scale = nn.Linear(self.n_embd, 1)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        if out_dim is None:
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        else:
            self.up_proj = nn.Linear(self.down_size, out_dim)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale.bias)
            nn.init.constant_(nn.LayerNorm(self.n_embd).weight, 1.0)
            nn.init.constant_(nn.LayerNorm(self.n_embd).bias, 0.0)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        scale = F.relu(self.scale(x))
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output


@MODELS.register_module()
class DAPT(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super(DAPT, self).__init__(config, num_classes)

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.drop_adapter_rate = config.drop_adapter_rate
        self.rank = config.rank

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            drop_adapter_rate=self.drop_adapter_rate,
            rank=self.rank,
        )

        self.class_token = config.class_token
        self.prompt_token = config.prompt_token
        self.patch_token = config.patch_token

        self.HEAD_CHANEL = 0
        if self.class_token != 'None':
            self.HEAD_CHANEL += 1
        if self.prompt_token != 'None':
            self.HEAD_CHANEL += 1
        if self.patch_token != 'None':
            self.HEAD_CHANEL += 1

        assert self.HEAD_CHANEL != 0

        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(self.trans_dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = load_checkpoint(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

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
            print_log(f'[DAPT] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

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

            print_log(f'[DAPT] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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
        group_input_tokens = self.encoder(neighborhood)

        group_input_tokens = apply_tfts(group_input_tokens, self.tfts_gamma_1, self.tfts_beta_1)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = x + pos
        x = self.blocks(x)

        x = apply_tfts(self.norm(x), self.tfts_gamma_2, self.tfts_beta_2)

        cls_tokens = None
        prompt_tokens = None
        patch_token = None
        if self.class_token != 'None':
            cls_tokens = x[:, 0]
        else:
            cls_tokens = None
        if self.prompt_token == 'None':
            prompt_tokens = None
        else:
            if self.prompt_token == 'max':
                prompt_tokens = x[:, 1:-self.num_group].max(dim=1)[0]
            elif self.prompt_token == 'mean':
                prompt_tokens = x[:, 1:-self.num_group].mean(dim=1)
        if self.patch_token == 'None':
            patch_token = None
        else:
            if self.patch_token == 'max':
                patch_token = x[:, -self.num_group:].max(dim=1)[0]
            elif self.patch_token == 'mean':
                patch_token = x[:, -self.num_group:].mean(dim=1)

        if cls_tokens is None:
            concat_f = None
        else:
            concat_f = cls_tokens
        if prompt_tokens is not None:
            if concat_f is None:
                concat_f = prompt_tokens
            else:
                concat_f = torch.cat([concat_f, prompt_tokens], dim=-1)
        if patch_token is not None:
            if concat_f is None:
                concat_f = patch_token
            else:
                concat_f = torch.cat([concat_f, patch_token], dim=-1)

        ret = self.cls_head_finetune(concat_f)
        return ret


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., drop_adapter_rate=0., rank=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                adapter_drop=drop_adapter_rate, rank=rank,
                depth=i
            )
            for i in range(depth)])

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x


def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
