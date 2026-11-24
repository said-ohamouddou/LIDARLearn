"""
Pretrained SSL checkpoint-loading test.

For every SSL model (PointMAE, Point-BERT, PointGPT, ACT, ReCon, PCP-MAE,
Point-M2AE) this test:

1. Builds the classification finetune model from its canonical yaml.
2. Takes a byte-level snapshot of the model's state_dict BEFORE loading.
3. Calls the model's own ``load_model_from_ckpt(...)`` — same entry point
   used by ``runner_finetune.py``.
4. Takes a snapshot AFTER and counts how many parameters actually changed.
5. Asserts the loaded fraction is > 0 (i.e. at least one param was replaced
   by a tensor from the checkpoint).

Why this matters
----------------
Silent checkpoint-loading failures are the #1 source of "my SSL numbers
look like from-scratch training" bugs. If a prefix-stripping rule breaks
(e.g. the ReCon ckpt uses ``MAE_encoder.`` but the loader strips
``transformer_q.``), ``load_state_dict(strict=False)`` will succeed with
an ``unexpected_keys`` warning and every param will stay at its random
init. The runner sees a valid model and training proceeds — but with no
benefit from pretraining. Comparing state_dict before/after is the only
reliable way to detect this.

Missing checkpoint files are treated as skips so the test works on
machines without the large ``pretrained/`` downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from utils.config import cfg_from_yaml_file

from .conftest import REPO_ROOT, build_model_or_skip  # type: ignore


# ---------------------------------------------------------------------------
# SSL model matrix: (label, config path, pretrained checkpoint)
# ---------------------------------------------------------------------------
# Every entry points to a finetune classification config (the one used by
# the training scripts) and the ckpt file passed via `--ckpts`. Paths are
# relative to the repo root.
SSL_MODELS = [
    ("pointmae",  "cfgs/classification/PointMAE/ModelNet40/modelnet40.yaml",  "pretrained/pretrained_mae.pth"),
    ("pointbert", "cfgs/classification/PointBERT/ModelNet40/modelnet40.yaml", "pretrained/pretrained_bert.pth"),
    ("pointgpt",  "cfgs/classification/PointGPT/ModelNet40/modelnet40.yaml",  "pretrained/pretrained_gpt.pth"),
    ("act",       "cfgs/classification/ACT/ModelNet40/modelnet40.yaml",       "pretrained/pretrained_act.pth"),
    ("recon",     "cfgs/classification/RECON/ModelNet40/modelnet40.yaml",     "pretrained/pretrained_recon.pth"),
    ("pcpmae",    "cfgs/classification/PCPMAE/ModelNet40/modelnet40.yaml",    "pretrained/pretrained_pcp.pth"),
    ("pointm2ae", "cfgs/classification/PointM2AE/ModelNet40/modelnet40.yaml", "pretrained/pretrained_m2ae.pth"),
]


def _compare_state_dicts(before, after):
    """Return (changed_params, total_params) counts and the list of
    parameter names whose tensor contents differ."""
    changed_names = []
    changed_count = 0
    total_count = 0
    for name, tensor_before in before.items():
        total_count += tensor_before.numel()
        tensor_after = after.get(name)
        if tensor_after is None:
            continue
        if tensor_before.shape != tensor_after.shape:
            # Shape mismatches don't happen on a pure load — but if they do,
            # count as "changed" so we don't hide the event.
            changed_count += tensor_before.numel()
            changed_names.append(name)
            continue
        if not torch.equal(tensor_before, tensor_after):
            changed_count += tensor_before.numel()
            changed_names.append(name)
    return changed_count, total_count, changed_names


def _load_ckpt_into_model(model, ckpt_path):
    """Call the model's own checkpoint loader — the same one used by the
    training runners. Every SSL model in this library implements
    ``load_model_from_ckpt``; if that's missing for some reason, the test
    will fail loudly."""
    assert hasattr(model, "load_model_from_ckpt"), (
        f"{type(model).__name__} does not implement load_model_from_ckpt — "
        f"the runner's checkpoint path won't be exercised."
    )
    model.load_model_from_ckpt(str(ckpt_path))


@pytest.mark.parametrize(
    "label,cfg_rel,ckpt_rel",
    SSL_MODELS,
    ids=[m[0] for m in SSL_MODELS],
)
def test_pretrained_checkpoint_loads_nontrivially(label, cfg_rel, ckpt_rel):
    """Building the model and calling load_model_from_ckpt must replace
    at least one parameter tensor — otherwise the checkpoint is being
    silently ignored."""

    cfg_path = REPO_ROOT / cfg_rel
    ckpt_path = REPO_ROOT / ckpt_rel

    if not cfg_path.is_file():
        pytest.skip(f"config not present: {cfg_rel}")
    if not ckpt_path.is_file():
        pytest.skip(f"pretrained checkpoint not present: {ckpt_rel}")

    cfg = cfg_from_yaml_file(cfg_path)
    model = build_model_or_skip(cfg.model)

    # Byte-level snapshot BEFORE loading — deep-copy each tensor so nothing
    # can alias through after load_state_dict replaces the storage.
    before = {name: p.detach().clone() for name, p in model.state_dict().items()}

    _load_ckpt_into_model(model, ckpt_path)

    after = {name: p.detach() for name, p in model.state_dict().items()}
    changed_params, total_params, changed_names = _compare_state_dicts(before, after)

    # Hard gate: at least one parameter must have been replaced.
    assert changed_params > 0, (
        f"[{label}] load_model_from_ckpt({ckpt_rel}) did not modify any "
        f"parameter tensor. The checkpoint is being silently ignored — "
        f"likely a prefix-stripping mismatch in load_model_from_ckpt."
    )

    # Informational: percentage of total params that were replaced.
    loaded_pct = 100.0 * changed_params / max(total_params, 1)
    print(
        f"[{label}] loaded {changed_params:,}/{total_params:,} parameters "
        f"({loaded_pct:.1f}%) from {ckpt_rel}; "
        f"{len(changed_names)} tensors changed"
    )

    # Soft sanity check: the backbone is typically 95%+ of an SSL model's
    # parameter count, so anything under 10% is almost certainly a silent
    # partial-load bug (e.g. the loader stripped the wrong prefix and only
    # the positional embedding made it through).
    assert loaded_pct > 10.0, (
        f"[{label}] only {loaded_pct:.1f}% of parameters were replaced by "
        f"the checkpoint — expected the backbone (≥50%) to load. Check "
        f"the prefix-stripping rules in {type(model).__name__}."
        f".load_model_from_ckpt."
    )

    # Sanity check: at least one recognisable backbone submodule name
    # should appear in the changed list — catches cases where random
    # scalars happen to be > 0 but the actual backbone didn't load.
    backbone_keywords = (
        "encoder", "blocks", "pos_embed", "norm",
        "reduce_dim", "cls_token", "group_divider",
    )
    has_backbone_change = any(
        any(kw in name for kw in backbone_keywords) for name in changed_names
    )
    assert has_backbone_change, (
        f"[{label}] checkpoint changed {len(changed_names)} tensors but "
        f"none of them matched a known backbone name. Something loaded, "
        f"but it isn't the encoder/transformer."
    )
