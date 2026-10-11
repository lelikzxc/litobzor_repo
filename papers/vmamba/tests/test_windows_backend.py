"""Portable CUDA scan selection and recurrence-preserving Windows fallback."""

import builtins
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from papers.vmamba.kernels import csms6s


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    monkeypatch.setattr(csms6s, "_SCAN_BACKEND", "auto")


def _extensions(monkeypatch, *available):
    for name, flag in (
        ("oflex", "WITH_SELECTIVESCAN_OFLEX"),
        ("core", "WITH_SELECTIVESCAN_CORE"),
        ("mamba", "WITH_SELECTIVESCAN_MAMBA"),
    ):
        monkeypatch.setattr(csms6s, flag, name in available)


def _inputs(length=257, device="cpu"):
    generator = torch.Generator().manual_seed(19)
    shapes = [(2, 8, length), (2, 8, length), (8, 3),
              (2, 4, 3, length), (2, 4, 3, length), (8,), (8,)]
    values = [torch.randn(shape, generator=generator) * 0.25 for shape in shapes]
    values[2] = -values[2].exp()
    return [value.to(device).requires_grad_() for value in values]


@pytest.mark.parametrize("available,preferred,expected", [
    ((), None, "chunked"),
    ((), "core", "chunked"),
    (("oflex", "core", "mamba"), None, "oflex"),
    (("oflex", "core"), "core", "core"),
    (("oflex",), "core", "oflex"),
    (("mamba",), "oflex", "mamba"),
])
def test_cuda_selection_handles_absent_or_partial_extension_installs(
    monkeypatch, available, preferred, expected
):
    _extensions(monkeypatch, *available)
    assert csms6s.get_scan_backend("cuda", preferred) == expected


def test_backend_override_preserves_explicit_reference(monkeypatch):
    _extensions(monkeypatch, "oflex", "core")
    csms6s.set_scan_backend("chunked")
    assert csms6s.get_scan_backend("cuda", "core") == "chunked"
    assert csms6s.get_scan_backend("cuda", "torch") == "torch"
    csms6s.set_scan_backend("torch")
    assert csms6s.get_scan_backend("cuda", "core") == "torch"


def test_compiled_requirement_fails_clearly_before_training(monkeypatch):
    _extensions(monkeypatch)
    csms6s.set_scan_backend("compiled")
    with pytest.raises(RuntimeError, match="native Windows CUDA"):
        csms6s.get_scan_backend("cuda")
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        csms6s.get_scan_backend("cpu")


def test_non_cuda_never_dispatches_to_native_extension(monkeypatch):
    _extensions(monkeypatch, "oflex", "core", "mamba")
    assert csms6s.get_scan_backend("cpu", "oflex") == "torch"
    assert csms6s.get_scan_backend("mps", "core") == "chunked"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown scan backend"):
        csms6s.set_scan_backend("invalid")
    with pytest.raises(ValueError, match="Unknown scan backend"):
        csms6s.get_scan_backend("cpu", "invalid")


def test_dll_failure_is_reported_and_does_not_disable_portable_backend(monkeypatch):
    def fail_import(name):
        raise OSError("WinError 126: DLL load failed")

    monkeypatch.setattr(csms6s.importlib, "import_module", fail_import)
    monkeypatch.setattr(csms6s, "SCAN_IMPORT_ERRORS", {})
    _extensions(monkeypatch)
    assert csms6s._load_scan_extension("bad_windows_extension") is None
    diagnostics = csms6s.scan_backend_info("cuda")
    assert diagnostics["backend"] == "chunked"
    assert diagnostics["compiled_backends"] == []
    assert "WinError 126" in diagnostics["import_errors"]["bad_windows_extension"]


def test_cross_scan_import_and_backward_never_load_optional_triton_dll(monkeypatch):
    from papers.vmamba.kernels import csm_triton

    original_import = builtins.__import__
    triton_attempts = []

    def guarded_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            triton_attempts.append(name)
            raise OSError("WinError 126: incompatible Triton DLL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    # Execute the module again rather than relying on its cached initial import.
    namespace = runpy.run_path(str(Path(csm_triton.__file__)))
    assert namespace["WITH_TRITON"] is False
    inputs = torch.randn(2, 3, 3, 5, requires_grad=True)
    scanned = namespace["cross_scan_fn"](inputs)
    merged = namespace["cross_merge_fn"](scanned.reshape(2, 4, 3, 3, 5))
    torch.testing.assert_close(merged, inputs.flatten(2) * 4)
    merged.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 4))
    assert triton_attempts == []


def test_cuda_without_extensions_dispatches_outside_custom_autograd_function(monkeypatch):
    _extensions(monkeypatch)
    captured = []
    marker = object()

    def prefix(*args):
        captured.append(args)
        return marker

    monkeypatch.setattr(csms6s, "selective_scan_chunked", prefix)
    fake_cuda = SimpleNamespace(device=torch.device("cuda"), is_cuda=True)
    assert csms6s.selective_scan_fn(fake_cuda, None, None, None, None) is marker
    assert len(captured) == 1
    assert captured[0][0] is fake_cuda


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
)])
def test_chunked_dispatch_matches_3136_token_recurrence_and_every_gradient(device):
    csms6s.set_scan_backend("chunked")
    values = _inputs(3136, device)
    reference_values = [value.detach().cpu().requires_grad_() for value in values]
    actual = csms6s.selective_scan_fn(*values)
    expected = csms6s.selective_scan_torch(*reference_values)
    torch.testing.assert_close(actual.cpu(), expected, rtol=4e-5, atol=4e-5)
    weights = torch.randn_like(expected)
    (actual * weights.to(device)).mean().backward()
    (expected * weights).mean().backward()
    for value, reference in zip(values, reference_values, strict=True):
        assert value.grad is not None and torch.isfinite(value.grad).all().item()
        torch.testing.assert_close(value.grad.cpu(), reference.grad, rtol=5e-4, atol=4e-5)


def test_chunked_public_dispatch_retains_gradients_without_optional_parameters():
    values = _inputs(259)[:5]
    result = csms6s.selective_scan_fn(*values, backend="chunked", oflex=False)
    result.square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_full_fcs_model_trains_without_extensions(monkeypatch):
    from papers.vmamba.models.vmamba import FCSVMamba

    _extensions(monkeypatch)
    csms6s.set_scan_backend("chunked")
    model = FCSVMamba(embed_dim=16, depths=[1] * 4, clca_num_heads=2, d_state=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(torch.rand(2, 3, 32, 32, device="cuda"))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
