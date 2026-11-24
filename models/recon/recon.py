"""
Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining

Paper: ICML 2023
Authors: Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, Li Yi
Source: Implementation adapted from: https://github.com/qizekun/ReCon
License: MIT
"""

import random
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .CrossModal import VisionTransformer as ImageEncoder
from .CrossModal import TextTransformer as TextEncoder
from timm.models.layers import trunc_normal_
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, load_checkpoint
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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ReConBlock(nn.Module):
    """
    ReCon transformer block with an optional contrastive cross-attention path.

    When with_contrast=True, the first 3 tokens are treated as the
    (cls, img, text) contrastive cls-tokens and are updated via cross-attention
    to the remaining visible tokens, while the visible tokens self-attend only
    among themselves — this matches upstream ReCon's "Contrast with Reconstruct,
    Guided by Generative Pretraining" design.

    During pretraining (finetune=False) the cross-attention uses detached
    visible tokens, so gradients from the contrastive objective do not flow
    into the MAE encoder. During finetuning (finetune=True) gradients flow.

    Attribute names (norm1, attn, norm2, mlp, norm1_contrast, norm2_contrast,
    attn_contrast, mlp_contrast) are chosen to match upstream ReCon's
    models/transformer.py::Block so that upstream checkpoints load cleanly.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., with_contrast=False,
                 finetune=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.with_contrast = with_contrast
        self.finetune = finetune

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        if self.with_contrast:
            self.norm1_contrast = norm_layer(dim)
            self.norm2_contrast = norm_layer(dim)
            self.mlp_contrast = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                    act_layer=act_layer, drop=drop)
            self.attn_contrast = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        if not self.with_contrast:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        cls = x[:, :3]
        vis_x = x[:, 3:]

        cls = self.norm1_contrast(cls)
        vis_x_norm = self.norm1(vis_x)
        if self.finetune:
            cls = cls + self.drop_path(self.attn_contrast(cls, vis_x_norm))
        else:
            cls = cls + self.drop_path(self.attn_contrast(cls, vis_x_norm.detach()))
        vis_x = vis_x + self.drop_path(self.attn(vis_x_norm))

        cls = cls + self.drop_path(self.mlp_contrast(self.norm2_contrast(cls)))
        vis_x = vis_x + self.drop_path(self.mlp(self.norm2(vis_x)))

        return torch.cat((cls, vis_x), dim=1)

class ReConTransformerEncoder(nn.Module):
    """ReCon-specific transformer encoder stacking ReConBlock."""

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., with_contrast=True, finetune=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReConBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                with_contrast=with_contrast, finetune=finetune,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReConBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, with_contrast=False,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))
        return x
# Pretrain model

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = ReConTransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            with_contrast=True,
            finetune=False,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

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

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, points, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
        pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3:], bool_masked_pos

class ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity, select):
        B = similarity.size(0)
        losses = torch.zeros(B).to(similarity.device)
        for i in range(B):
            pos = torch.masked_select(similarity[i], select[i] == 1)
            neg = torch.masked_select(similarity[i], select[i] == 0)
            pos = torch.mean(pos, dim=0, keepdim=True)
            logits = torch.cat((pos, neg)).reshape(1, -1)
            label = torch.zeros(1, dtype=torch.long).to(similarity.device)
            losses[i] = self.criterion(logits / self.temperature, label)
        loss = losses.mean()
        return loss

@MODELS.register_module()
class RECON_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[ReCon] ', logger='ReCon')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[ReCon] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='ReCon')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        # cross model contrastive
        self.csc_loss = torch.nn.SmoothL1Loss()
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False

        if self.csc_img:
            self.img_encoder = ImageEncoder(config)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
            self.img_proj = nn.Linear(self.trans_dim, self.img_encoder.output_dim)
            self.img_proj.apply(self._init_weights)

        if self.csc_text:
            self.text_encoder = TextEncoder(config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_proj = nn.Linear(self.trans_dim, self.text_encoder.embed_dim)
            self.text_proj.apply(self._init_weights)

        # single modal contrastive
        self.smc = config.self_contrastive
        if self.smc:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.trans_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128)
            )
            self.cls_proj.apply(self._init_weights)
            self.contrastive_head = ContrastiveHead(temperature=0.1)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts, img=None, text=None, noaug=False, **kwargs):
        """
        Forward pass for ReCon pretraining.
        Args:
            pts: Point cloud tensor (B, N, 3)
            img: Image input for cross-modal contrastive (required when csc_img is enabled)
            text: Text input for cross-modal contrastive (required when csc_text or smc is enabled)
            noaug: If True, return cls_token features for SVM validation (no masking)
        """
        if not noaug:
            if self.csc_img and img is None:
                raise ValueError("ReCon pretraining has csc_img enabled but img=None was passed.")
            if self.csc_text and text is None:
                raise ValueError("ReCon pretraining has csc_text enabled but text=None was passed.")
            if self.smc and text is None:
                raise ValueError("ReCon pretraining has self_contrastive enabled but text=None was passed.")

        neighborhood, center = self.group_divider(pts)

        # For feature extraction (SVM validation), use noaug=True to disable masking
        cls_token, img_token, text_token, x_vis, mask = self.MAE_encoder(pts, neighborhood, center, noaug=noaug)

        # If noaug=True, return cls_token as features for SVM
        if noaug:
            return cls_token  # (B, trans_dim)

        losses = {}

        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        losses['mdm'] = self.loss_func(rebuild_points, gt_points)

        if self.csc_img:
            img_feature = self.img_encoder(img)
            img_token = self.img_proj(img_token)
            losses['csc_img'] = self.csc_loss(img_feature, img_token).mean()

        if self.csc_text:
            text_feature = self.text_encoder(text)
            text_token = self.text_proj(text_token)
            losses['csc_text'] = self.csc_loss(text_feature, text_token).mean()

        if self.smc:
            cls_proj = self.cls_proj(cls_token)
            cls_proj = nn.functional.normalize(cls_proj, dim=1)
            similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))

            select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
            for i in range(B):
                for j in range(B):
                    if text[i] == text[j]:
                        select[i, j] = 1
            losses['smc'] = self.contrastive_head(similarity, select)

        loss = sum(losses.values())
        return loss

@MODELS.register_module()
class RECON(BasePointCloudModel):
    def __init__(self, config, **kwargs):
        num_classes = config.cls_dim
        super().__init__(config, num_classes)

        self.type = config.type
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = ReConTransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            with_contrast=True,
            finetune=True,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        if self.type == "linear":
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 4, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

    def load_model_from_ckpt(self, bert_ckpt_path, log=True):
        if bert_ckpt_path is not None:
            ckpt = load_checkpoint(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            total_ckpt_params = sum(p.numel() for p in base_ckpt.values())

            keys_to_remove = []
            keys_to_add = {}
            for k in base_ckpt.keys():
                if k.startswith('MAE_encoder'):
                    new_key = k[len('MAE_encoder.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)
                elif k.startswith('base_model'):
                    new_key = k[len('base_model.'):]
                    keys_to_add[new_key] = base_ckpt[k]
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                if k in base_ckpt:
                    del base_ckpt[k]
            base_ckpt.update(keys_to_add)

            model_state = self.state_dict()
            matched_params = 0
            matched_keys = []
            for k, v in base_ckpt.items():
                if k in model_state and v.shape == model_state[k].shape:
                    matched_params += v.numel()
                    matched_keys.append(k)

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if log:
                used_percentage = (matched_params / total_ckpt_params * 100) if total_ckpt_params > 0 else 0.0
                print_log(f'[RECON_PointTransformer] Loaded {len(matched_keys)}/{len(base_ckpt)} parameter groups ({used_percentage:.2f}% of checkpoint parameters)', logger='Transformer')

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

                print_log(f'[RECON_PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
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

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_token, img_token, text_token, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1], x[:, 2], x[:, 3:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
