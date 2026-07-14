"""Tests for common.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from common.utils.seed import set_seed


def test_set_seed_makes_random_deterministic() -> None:
    set_seed(42)
    first = [random.random() for _ in range(5)]

    set_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_set_seed_makes_numpy_deterministic() -> None:
    set_seed(7)
    first = np.random.rand(4)

    set_seed(7)
    second = np.random.rand(4)

    np.testing.assert_array_equal(first, second)


def test_set_seed_makes_torch_deterministic() -> None:
    set_seed(123)
    first = torch.rand(3)

    set_seed(123)
    second = torch.rand(3)

    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_makes_cuda_deterministic() -> None:
    set_seed(99)
    first = torch.rand(2, device="cuda")

    set_seed(99)
    second = torch.rand(2, device="cuda")

    torch.testing.assert_close(first, second)
