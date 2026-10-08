"""MPS scan uses the same forward equation and gradients as the CPU recurrence."""

import pytest
import torch

from papers.vmamba.kernels.csms6s import selective_scan_chunked, selective_scan_torch


def _inputs(length, extreme=False):
    generator = torch.Generator().manual_seed(37)
    shapes = [
        (2, 8, length),
        (2, 8, length),
        (8, 3),
        (2, 2, 3, length),
        (2, 2, 3, length),
        (8,),
        (8,),
    ]
    tensors = [torch.randn(shape, generator=generator) * 0.25 for shape in shapes]
    tensors[2] = -torch.exp(tensors[2])
    if extreme:
        tensors[1][..., ::3] = 200.0  # exp(delta*A) underflows, safely.
        tensors[1][..., 1::3] = -200.0
    return [tensor.requires_grad_() for tensor in tensors]


@pytest.mark.parametrize(
    "length,chunk_size,extreme",
    [(1, 128, False), (35, 16, False), (3136, 128, False), (3136, 512, False), (257, 32, True)],
)
@pytest.mark.parametrize("softplus", [False, True])
def test_prefix_scan_matches_recurrence_forward_and_all_gradients(
    length, chunk_size, extreme, softplus
):
    # Negative raw deltas are invalid decay steps without softplus.
    inputs = _inputs(length, extreme)
    if not softplus:
        # Keep delta+bias positive even for the randomly generated bias.
        inputs[1] = (inputs[1].detach().abs() + 2.0).requires_grad_()
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    actual = selective_scan_chunked(*inputs, delta_softplus=softplus, chunk_size=chunk_size)
    expected = selective_scan_torch(*reference_inputs, delta_softplus=softplus)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    weights = torch.randn_like(actual)
    (actual * weights).mean().backward()
    (expected * weights).mean().backward()
    for current, reference in zip(inputs, reference_inputs, strict=True):
        assert torch.isfinite(current.grad).all()
        torch.testing.assert_close(current.grad, reference.grad, rtol=3e-4, atol=3e-5)


def test_checkpointed_scan_never_saves_full_sequence_state_history():
    length, chunk_size = 1027, 64
    inputs = _inputs(length)
    saved_shapes = []
    boundary_storages = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        if tensor.shape == (2, 2, 4, 3) and tensor.is_contiguous():
            boundary_storages.append(tensor.untyped_storage().nbytes())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = selective_scan_chunked(*inputs, chunk_size=chunk_size)
    # A history would be [batch,groups,channels,length,state_dim].
    assert not any(len(shape) == 5 and shape[-2] > chunk_size for shape in saved_shapes)
    # Boundary states must not be views holding a whole block's storage alive.
    assert len(boundary_storages) == (length + chunk_size - 1) // chunk_size
    assert max(boundary_storages) == 2 * 2 * 4 * 3 * 4
    result.square().mean().backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in inputs)


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        selective_scan_chunked(*_inputs(8), chunk_size=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
@pytest.mark.parametrize("chunk_size", [64, 512])
def test_mps_prefix_scan_matches_cpu_recurrence_and_all_gradients(chunk_size):
    cpu_inputs = _inputs(259 if chunk_size == 64 else 1037, extreme=True)
    mps_inputs = [tensor.detach().to("mps").requires_grad_() for tensor in cpu_inputs]
    actual = selective_scan_chunked(*mps_inputs, chunk_size=chunk_size)
    expected = selective_scan_torch(*cpu_inputs)
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)
    weights = torch.randn_like(expected)
    (actual * weights.to("mps")).mean().backward()
    (expected * weights).mean().backward()
    for mps_value, cpu_value in zip(mps_inputs, cpu_inputs, strict=True):
        assert torch.isfinite(mps_value.grad).all().item()
        torch.testing.assert_close(mps_value.grad.cpu(), cpu_value.grad, rtol=1e-3, atol=5e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_frequency_attention_mps_preserves_gradients():
    from papers.vmamba.modules.fcs_modules import FrequencyAttention

    reference = FrequencyAttention(8)
    mps_model = FrequencyAttention(8).to("mps")
    mps_model.load_state_dict(reference.state_dict())
    cpu_x = torch.randn(2, 8, 8, 8, requires_grad=True)
    mps_x = cpu_x.detach().to("mps").requires_grad_()
    cpu_y, mps_y = reference(cpu_x), mps_model(mps_x)
    torch.testing.assert_close(mps_y.cpu(), cpu_y, atol=2e-5, rtol=2e-5)
    cpu_y.square().mean().backward()
    mps_y.square().mean().backward()
    torch.testing.assert_close(mps_x.grad.cpu(), cpu_x.grad, atol=2e-5, rtol=2e-4)
