"""
CI mode (LIDARLEARN_ALLOW_MISSING_EXTENSIONS=1) may only hide what CI cannot
build: the CUDA extensions. A broken pip dependency or a real bug inside a
model constructor must still fail.
"""

from __future__ import annotations

import pytest
from easydict import EasyDict

from models.build import MODELS

from .conftest import CUDA_EXTENSION_MODULES, build_model_or_skip


def test_only_cuda_extensions_may_be_missing():
    """Every skipped model module must be missing a CUDA extension, never a
    pip-installable dependency (e.g. a torch-scatter wheel for the wrong
    torch, or an API change in timm)."""
    missing = {dep for dep, _ in MODELS.import_failures.values()}
    unexpected = missing - CUDA_EXTENSION_MODULES
    details = {mod: err for mod, (dep, err) in MODELS.import_failures.items() if dep in unexpected}
    assert not unexpected, f"non-extension dependencies failed to import: {details}"


def test_constructor_keyerror_is_not_skipped(monkeypatch):
    """A KeyError raised inside a registered model's __init__ is a bug and
    must surface, even in CI mode with extension modules skipped."""
    name = "_CIModeKeyErrorProbe"

    class _Probe:
        def __init__(self, cfg):
            raise KeyError("missing_config_field")

    MODELS.register_module(name=name, module=_Probe)
    monkeypatch.setenv("LIDARLEARN_ALLOW_MISSING_EXTENSIONS", "1")
    monkeypatch.setitem(MODELS.import_failures, "models.fake.module", ("pointnet2_ops", "simulated"))
    try:
        build_model_or_skip(EasyDict(NAME=name))
    except KeyError:
        pass
    except pytest.skip.Exception:
        pytest.fail("a constructor KeyError was reported as a skip")
    else:
        pytest.fail("expected the constructor KeyError to propagate")
    finally:
        MODELS._module_dict.pop(name, None)
