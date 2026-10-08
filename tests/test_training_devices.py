"""Backend selection must not silently turn an explicit GPU run into CPU training."""

import pytest
import torch

from common.training.utils import resolve_device


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_device_selection(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device() == expected


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_explicit_unavailable_gpu_fails(monkeypatch, backend):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=backend.upper()):
        resolve_device(backend)
    assert resolve_device("cpu") == "cpu"


def test_available_mps_request(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == "mps"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_device("other")
