# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent CPU proof tests for the block triangular action route."""

import pytest
import torch


def _make_lower(size, seed):
    generator = torch.Generator().manual_seed(seed)
    lower = torch.tril(torch.randn(size, size, generator=generator, dtype=torch.float64) * 0.02)
    return lower + torch.eye(size, dtype=torch.float64)


@pytest.mark.parametrize('size', [1, 16, 32, 64, 128])
def test_block_action_matches_materialized_inverse(size):
    z = _make_lower(size, 4100 + size).requires_grad_()
    generator = torch.Generator().manual_seed(8000 + size)
    rhs = torch.randn(size, generator=generator, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(size, generator=generator, dtype=torch.float64)
    action = torch.linalg.solve_triangular(z, rhs, upper=False)
    materialized = torch.linalg.solve(z, rhs)
    torch.testing.assert_close(action, materialized, atol=2e-13, rtol=2e-13)
    action_grads = torch.autograd.grad((action * cotangent).sum(), (z, rhs), retain_graph=True)
    materialized_grads = torch.autograd.grad((materialized * cotangent).sum(), (z, rhs))
    for actual, expected in zip(action_grads, materialized_grads):
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('size', [1, 16, 32, 64, 128])
def test_block_transpose_action_matches_inverse_transpose(size):
    z = _make_lower(size, 9100 + size).requires_grad_()
    generator = torch.Generator().manual_seed(12000 + size)
    rhs = torch.randn(size, generator=generator, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn(size, generator=generator, dtype=torch.float64)
    action = torch.linalg.solve_triangular(z.transpose(0, 1), rhs, upper=True)
    materialized = torch.linalg.solve(z.transpose(0, 1), rhs)
    torch.testing.assert_close(action, materialized, atol=2e-13, rtol=2e-13)
    action_grads = torch.autograd.grad((action * cotangent).sum(), (z, rhs), retain_graph=True)
    materialized_grads = torch.autograd.grad((materialized * cotangent).sum(), (z, rhs))
    for actual, expected in zip(action_grads, materialized_grads):
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
