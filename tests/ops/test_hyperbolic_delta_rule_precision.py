# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU mathematical evidence for the proposed selective-GEMM precision RFC.

No production precision policy or Triton kernel is selected by these tests.
Rounded QK/Gram operands are independent differentiable inputs so the tests
check the bilinear-product VJP without assuming a derivative of quantization.
"""

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _inputs(length=16, key_dim=8, value_dim=3, curvature=0.5, gate=None):
    torch.manual_seed(883)
    q = torch.nn.functional.normalize(torch.randn(length, key_dim, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = 0.2 * torch.randn(length, value_dim, dtype=torch.float64).tanh()
    beta = torch.rand_like(v) if gate is None else torch.full_like(v, gate)
    state = 0.1 * torch.nn.functional.normalize(torch.randn(key_dim, value_dim, dtype=q.dtype), dim=0)
    c = torch.tensor(curvature, dtype=q.dtype)
    return tuple(x.requires_grad_() for x in (q, k, v, beta, state, c))


def _wy_from_edges(q, k, v, beta, state, curvature, q_read, k_read, k_gram):
    """Use separate operands for the direct, readout-product and Gram edges."""
    length = q.shape[0]
    a = k.square().sum(-1, keepdim=True) - curvature * v.square()
    alpha = beta / a
    gram = k_gram @ k_gram.T
    system = torch.eye(length, dtype=q.dtype)[None] + (
        alpha.T[:, :, None] * (gram[None] - curvature * v.T[:, :, None] * v.T[:, None, :])
    ).tril(-1)
    residual = v - k @ state
    weights = torch.linalg.solve_triangular(
        system, (alpha * residual).T[:, :, None], upper=False, unitriangular=True,
    )[:, :, 0].T
    denominator = 1 + curvature * (v * weights).cumsum(0)
    readout = (q_read @ k_read.T).tril()
    numerator = q @ state + readout @ weights
    output = numerator / denominator
    final = (state + k.T @ weights) / denominator[-1]
    return output, final, (a, alpha, gram, system, residual, weights, denominator, readout)


def _wy_edge_vjp(inputs, d_output, d_final):
    q, k, v, beta, state, c, q_read, k_read, k_gram = inputs
    output, final, saved = _wy_from_edges(*inputs)
    a, alpha, gram, system, residual, weights, denominator, readout = saved
    d_num = d_output / denominator
    d_readout = (d_num @ weights.T).tril()
    dq = d_num @ state.T
    dq_read = d_readout @ k_read
    dk_read = d_readout.T @ q_read
    d_state = q.T @ d_num + d_final / denominator[-1]
    dk = weights @ (d_final / denominator[-1]).T
    dz = -d_output * output / denominator
    dz_suffix = dz.flip(0).cumsum(0).flip(0) - (d_final * final).sum(0) / denominator[-1]
    dw = readout.T @ d_num + k @ (d_final / denominator[-1]) + c * v * dz_suffix
    dv = c * weights * dz_suffix
    dc = (v * weights * dz_suffix).sum()
    z = torch.linalg.solve_triangular(
        system.transpose(-1, -2), dw.T[:, :, None], upper=True, unitriangular=True,
    )[:, :, 0].T
    u = alpha * z
    d_gram = -(u @ weights.T).tril(-1)
    dk_gram = (d_gram + d_gram.T) @ k_gram
    value_prefix = (v * weights).cumsum(0) - v * weights
    adjoint_suffix = (u * v).flip(0).cumsum(0).flip(0) - u * v
    dv = dv + c * (u * value_prefix + weights * adjoint_suffix)
    dc = dc + (u * v * value_prefix).sum()
    d_alpha = z * (residual - gram.tril(-1) @ weights + c * v * value_prefix)
    da = -d_alpha * beta / a.square()
    d_state = d_state - k.T @ u
    dk = dk - u @ state.T + 2 * k * da.sum(-1, keepdim=True)
    dv = dv + u - 2 * c * v * da
    dc = dc - (da * v.square()).sum()
    d_beta = d_alpha / a
    return dq, dk, dv, d_beta, d_state, dc, dq_read, dk_read, dk_gram


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize('length,key_dim,value_dim', [(1, 7, 3), (16, 8, 3), (32, 16, 5)])
@pytest.mark.parametrize('curvature', [0.0, 0.5, 2.0])
@pytest.mark.parametrize('gate', [None, 0.0, 1.0])
def test_selective_gemm_operand_vjp(dtype, length, key_dim, value_dim, curvature, gate):
    args = _inputs(length, key_dim, value_dim, curvature, gate)
    operands = tuple(x.detach().to(dtype).double().requires_grad_() for x in (args[0], args[1], args[1]))
    inputs = (*args, *operands)
    output, final, _ = _wy_from_edges(*inputs)
    do, ds = torch.randn_like(output), torch.randn_like(final)
    expected = torch.autograd.grad((output, final), inputs, (do, ds))
    with torch.no_grad():
        actual = _wy_edge_vjp(inputs, do, ds)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_selective_gemm_edge_gradcheck():
    args = _inputs(length=3, key_dim=4, value_dim=2)
    operands = tuple(x.detach().clone().requires_grad_() for x in (args[0], args[1], args[1]))

    def forward(*inputs):
        return _wy_from_edges(*inputs)[:2]

    assert torch.autograd.gradcheck(forward, (*args, *operands), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('curvature', [0.0, 0.5, 2.0])
@pytest.mark.parametrize('gate', [None, 0.0, 1.0])
def test_selective_qk_does_not_change_forward_state(dtype, curvature, gate):
    args = _inputs(length=32, key_dim=16, value_dim=5, curvature=curvature, gate=gate)
    q, k = args[:2]
    _, baseline, _ = _wy_from_edges(*args, q, k, k)
    _, candidate, _ = _wy_from_edges(*args, q.to(dtype).double(), k.to(dtype).double(), k)
    torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)


def test_cast_graph_cotangent_is_not_fp32_bilinear_vjp():
    x = torch.tensor([[0.1279, -0.3711]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0.4127], [0.8349]], dtype=torch.float64, requires_grad=True)
    dout = torch.tensor([[0.12793]], dtype=torch.float64)
    xr, yr = x.half().double(), y.half().double()
    dx, dy = torch.autograd.grad(xr @ yr, (x, y), dout)
    dx_fp32_vjp, dy_fp32_vjp = dout @ yr.detach().T, xr.detach().T @ dout
    torch.testing.assert_close(dx, dx_fp32_vjp.half().double(), atol=0, rtol=0)
    torch.testing.assert_close(dy, dy_fp32_vjp.half().double(), atol=0, rtol=0)
    assert not torch.equal(dx, dx_fp32_vjp)
    assert not torch.equal(dy, dy_fp32_vjp)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_reduced_gram_can_leave_the_feasible_domain(dtype):
    torch.manual_seed(961)
    k = torch.nn.functional.normalize(torch.randn(16, dtype=torch.float64), dim=0).repeat(32, 1)
    q = k.clone()
    v = torch.full((32, 1), 0.9999, dtype=q.dtype)
    beta = torch.ones_like(v)
    state = torch.zeros(16, 1, dtype=q.dtype)
    c = torch.tensor(1.0, dtype=q.dtype)
    assert (k.square().sum(-1, keepdim=True) - c * v.square() > 0).all()
    _, strict_state, _ = _wy_from_edges(q, k, v, beta, state, c, q, k, k)
    qr, kr = q.to(dtype).double(), k.to(dtype).double()
    _, rounded_state, _ = _wy_from_edges(q, k, v, beta, state, c, qr, kr, kr)
    assert strict_state.square().sum() < 1
    assert rounded_state.square().sum() > 1
