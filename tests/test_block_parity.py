"""
Transformer blocks that pretrained checkpoints depend on must compute the same
function as the block in the original repository; otherwise the loaded
weights run through a different network. Each reference below is a verbatim
copy of the original forward pass.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from .conftest import CUDA_EXTENSION_MODULES, allow_missing_extensions


def _import_or_skip(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        if allow_missing_extensions() and getattr(e, "name", None) in CUDA_EXTENSION_MODULES:
            pytest.skip(f"{module} unavailable: missing {e.name}")
        raise


def _reference_recon_block(b, x):
    # ReCon models/transformer.py::Block.forward (with_contrast=True, finetune=True)
    cls = x[:, :3]
    vis_x = x[:, 3:]
    cls = b.norm1_contrast(cls)
    vis_x = b.norm1(vis_x)
    cls = cls + b.drop_path(b.attn_contrast(cls, vis_x))
    vis_x = vis_x + b.drop_path(b.attn(vis_x))
    cls = b.norm2_contrast(cls)
    vis_x = b.norm2(vis_x)
    cls = cls + b.drop_path(b.mlp_contrast(cls))
    vis_x = vis_x + b.drop_path(b.mlp(vis_x))
    return torch.cat((cls, vis_x), dim=1)


def _reference_gpt_dapt_block(b, x, attn_mask, apply_tfts):
    # PointGST models/GPT_DAPT.py::Block.forward
    x = apply_tfts(b.ln_1(x), b.tfts_gamma_1, b.tfts_beta_1)
    a, _ = b.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
    x = x + a
    m = b.ln_2(x)
    m = apply_tfts(m, b.tfts_gamma_2, b.tfts_beta_2)
    a_mlp = b.adapt_mlp(m)
    m = b.mlp[2](b.mlp[1](apply_tfts(b.mlp[0](x), b.tfts_gamma_4, b.tfts_beta_4)))
    m = apply_tfts(m, b.tfts_gamma_5, b.tfts_beta_5) + a_mlp
    x = x + m
    adapter_prompt = b.non_linear_func(a_mlp).mean(dim=0, keepdim=True)
    adapter_prompt = apply_tfts(adapter_prompt, b.tfts_gamma_3, b.tfts_beta_3)
    return torch.cat([x[0:2, :], adapter_prompt, x[2:, :]], dim=0)


def _randomize(module):
    # Non-trivial values everywhere, so that unused or misplaced
    # parameters (e.g. identity-initialised scale/shift) change the output.
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5 + (1.0 if p.dim() == 1 else 0.0))
    return module.eval()


@pytest.mark.parametrize("finetune", [True, False])
def test_recon_block_matches_original(finetune):
    recon = _import_or_skip("models.recon.recon")
    block = _randomize(recon.ReConBlock(dim=32, num_heads=4, with_contrast=True, finetune=finetune))
    x = torch.randn(2, 3 + 10, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(block(x), _reference_recon_block(block, x))


def test_gpt_dapt_block_matches_original():
    gpt_dapt = _import_or_skip("models.dapt.gpt_dapt")
    block = _randomize(gpt_dapt.Block(embed_dim=32, num_heads=4))
    x = torch.randn(12, 2, 32, generator=torch.Generator().manual_seed(1))  # [seq, batch, C]
    mask = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        torch.testing.assert_close(block(x, mask), _reference_gpt_dapt_block(block, x, mask, gpt_dapt.apply_tfts))
