"""
Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining

Paper: ICML 2023
Authors: Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, Li Yi
Source: Implementation adapted from: https://github.com/qizekun/ReCon
License: MIT
"""
import torch.nn as nn
import torch
from clip import clip
import timm
import os
import random
from utils.checkpoint import load_checkpoint


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = load_checkpoint(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone_name = self.config.img_encoder
        self.trans_dim = config.transformer_config.trans_dim

        if self.backbone_name in clip._MODELS.keys():
            self.clip = True
            model = load_clip_to_cpu(self.backbone_name)
            image_model = model.visual
        else:
            self.clip = False
            image_model = timm.create_model(self.backbone_name, True)

        for p in image_model.parameters():
            p.requires_grad = False

        if self.clip:
            self.ln_pre = image_model.ln_pre
            self.blocks = image_model.transformer.resblocks
            self.ln_post = image_model.ln_post
            self.visual_embed_depth = image_model.transformer.layers
            self.cls_token = image_model.class_embedding
            self.pos_embed = image_model.positional_embedding
            self.conv1 = image_model.conv1
            self.embed_dim = model.vision_width
            self.output_dim = image_model.output_dim
            self.proj = image_model.proj
            self.patch_size = model.vision_patch_size
        else:
            self.patch_embed = image_model.patch_embed
            self.blocks = image_model.blocks
            self.norm = image_model.norm
            self.visual_embed_depth = len(image_model.blocks)
            self.cls_token = image_model.cls_token
            self.pos_embed = image_model.pos_embed
            self.embed_dim = image_model.embed_dim
            self.output_dim = self.embed_dim
            self.pos_drop = image_model.pos_drop
            self.pre_logits = image_model.pre_logits
            self.patch_size = self.patch_embed.patch_size

    def embeddings(self, x):
        if self.clip:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.pos_embed.to(x.dtype)
            x = self.ln_pre(x)
        else:
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
        return x

    def encoder(self, x):
        if self.clip:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.blocks(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            for blk in self.blocks:
                x = blk(x)
        return x

    def post_process(self, x):
        if self.clip:
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
        else:
            x = self.norm(x)[:, 0]
            x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.post_process(x)
        return x


class TextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        clip = load_clip_to_cpu(self.cfg.text_encoder)

        for p in clip.parameters():
            p.requires_grad = False

        self.transformer = clip.transformer
        self.positional_embedding = clip.positional_embedding
        self.token_embedding = clip.token_embedding
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.dtype = clip.dtype
        self.trans_dim = config.transformer_config.trans_dim
        self.embed_dim = self.transformer.width

        self.template = ['',
                         'A ',
                         'A model of ',
                         'A model of a ',
                         'A image of ',
                         'A image of a ',
                         'A 3D model of ',
                         'A 3D model of a ',
                         'A rendering model of ',
                         'A rendering model of a ',
                         'A point cloud of ',
                         'A point cloud of a ',
                         'A point cloud model of ',
                         'A point cloud model of a ',
                         'A 3D rendering model of ',
                         'A 3D rendering model of a ',
                         'A rendering image of ',
                         'A rendering image of a ',
                         'A 3D rendering image of ',
                         'A 3D rendering image of a '
                         ]
        self.template_last = ['.', ' with white background.', ' with black context.']

    def forward(self, text, index=0):
        if isinstance(text, str):
            prompt_text = self.template[index % 20] + text + self.template_last[index // 20]
        else:
            prompt_text = [random.choice(self.template) + t + random.choice(self.template_last) for t in text]
        # Follow the encoder's current device instead of hardcoding .cuda(),
        # so CPU inference and multi-GPU both work.
        device = next(self.parameters()).device
        prompt_text = clip.tokenize(prompt_text, context_length=77).to(device)

        b, _ = prompt_text.shape
        x = self.token_embedding(prompt_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        text_feature = x[torch.arange(x.shape[0]), prompt_text.argmax(dim=-1)] @ self.text_projection
        return text_feature
