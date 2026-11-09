# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent CPU proof tests for the exact Group RHS fusion.

The production kernel computes the same vector in two pieces, ``Z^{-1}a_f``
and ``-kappa Z^{-1}r``.  The fused route combines their right-hand sides
before the one triangular action.  These tests intentionally use an
independent double-precision construction instead of importing Triton code.
"""

import pytest
import torch


def _exclusive_prefix(x):
    return torch.cat((torch.zeros_like(x[:1]), torch.cumsum(x[:-1], dim=0)))


def _legacy_and_fused(z, diagonal_inverse, nu, beta, h, kappa):
    prefix_h = _exclusive_prefix(h)
    feedback = diagonal_inverse * nu * beta
    state_rhs = diagonal_inverse * nu * (h + beta * prefix_h)
    legacy = torch.linalg.solve_triangular(z, feedback, upper=False)
    legacy = legacy - kappa * torch.linalg.solve_triangular(z, state_rhs, upper=False)
    fused_rhs = diagonal_inverse * nu * (beta - kappa * (h + beta * prefix_h))
    fused = torch.linalg.solve_triangular(z, fused_rhs, upper=False)
    return legacy, fused


@pytest.mark.parametrize('size', [1, 16, 32, 64, 128])
@pytest.mark.parametrize('kappa', [0.0, 0.37, 1.0])
def test_group_rhs_fusion_is_exact_and_has_one_action(size, kappa):
    generator = torch.Generator().manual_seed(1701 + size)
    dtype = torch.float64
    lower = torch.tril(torch.randn(size, size, generator=generator, dtype=dtype) * 0.03)
    z = lower + torch.eye(size, dtype=dtype)
    diagonal_inverse = (0.8 + 0.4 * torch.rand(size, generator=generator, dtype=dtype)).requires_grad_()
    nu = (0.7 + 0.5 * torch.rand(size, generator=generator, dtype=dtype)).requires_grad_()
    beta = torch.rand(size, generator=generator, dtype=dtype).requires_grad_()
    h = torch.randn(size, generator=generator, dtype=dtype).requires_grad_()
    kappa = torch.tensor(kappa, dtype=dtype, requires_grad=True)

    legacy, fused = _legacy_and_fused(z, diagonal_inverse, nu, beta, h, kappa)
    torch.testing.assert_close(fused, legacy, atol=2e-13, rtol=2e-13)

    cotangent = torch.randn(size, generator=generator, dtype=dtype)
    legacy_grads = torch.autograd.grad((legacy * cotangent).sum(),
                                       (diagonal_inverse, nu, beta, h, kappa), retain_graph=True)
    fused_grads = torch.autograd.grad((fused * cotangent).sum(),
                                      (diagonal_inverse, nu, beta, h, kappa))
    for actual, expected in zip(fused_grads, legacy_grads):
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('size', [16, 32, 64])
def test_group_rhs_fusion_endpoint_gates(size):
    generator = torch.Generator().manual_seed(9000 + size)
    dtype = torch.float64
    z = torch.eye(size, dtype=dtype)
    d = torch.rand(size, generator=generator, dtype=dtype)
    nu = torch.rand(size, generator=generator, dtype=dtype)
    h = torch.randn(size, generator=generator, dtype=dtype)
    for beta_value in (0.0, 1.0):
        beta = torch.full((size,), beta_value, dtype=dtype)
        for kappa in (0.0, 0.5):
            legacy, fused = _legacy_and_fused(z, d, nu, beta, h, torch.tensor(kappa, dtype=dtype))
            torch.testing.assert_close(fused, legacy, atol=2e-13, rtol=2e-13)
