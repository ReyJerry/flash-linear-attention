# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared protection inverse algebra, VJP, and optional accelerator coverage."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule'
_SPEC = importlib.util.spec_from_file_location('shared_inverse_reference', _PATH / 'protected.py')
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _case(rank=3, occupancy='full', curvature=0.3):
    torch.manual_seed(267)
    B, T, H, K, V = 2, 5, 2, max(rank + 2, 7), 4
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1)
    v = 0.03 * torch.randn(B, T, H, V, dtype=torch.float64)
    bank_k = torch.nn.functional.normalize(torch.randn(B, H, rank, K, dtype=torch.float64), dim=-1)
    bank_v = 0.02 * torch.randn(B, H, rank, V, dtype=torch.float64)
    scores = torch.zeros(B, H, rank, dtype=torch.float64)
    if occupancy == 'empty':
        scores.fill_(-1e6)
    elif occupancy == 'partial':
        scores[..., -1] = -1e6
    elif occupancy == 'infeasible':
        bank_v.fill_(10)
    c = torch.tensor(curvature, dtype=torch.float64)
    return tuple(x.requires_grad_() for x in (k, v, bank_k, bank_v, c)), scores


def _projection(args, scores, shared, eps=1e-5):
    k, v, bank_k, bank_v, c = args
    rank, channels = bank_v.shape[-2:]
    active = scores > -5e5
    pk, y = bank_k * active[..., None], bank_v * active[..., None]
    eye = torch.eye(rank, dtype=k.dtype, device=k.device)
    gram = pk @ pk.transpose(-1, -2)
    gram = torch.where(active[..., :, None] & active[..., None, :], gram + eps * eye, eye)
    inverse = torch.linalg.inv(gram) if shared else None
    keys, values, tape = [], [], []
    for channel in range(channels):
        ainv = inverse if shared else torch.linalg.inv(gram)
        yc = y[..., channel]
        u = (ainv @ yc[..., None]).squeeze(-1)
        denom = 1 - c * (yc * u).sum(-1)
        feasible = denom > eps
        ginv = ainv + (c / torch.where(feasible, denom, 1))[..., None, None] * u[..., :, None] * u[..., None, :]
        ginv = torch.where(feasible[..., None, None], ginv, ainv)
        rhs = torch.einsum('bhrk,bthk->bthr', pk, k) - c * yc[:, None] * v[..., channel, None]
        alpha = torch.einsum('bhrj,bthj->bthr', ginv, rhs)
        kp = k - torch.einsum('bhrk,bthr->bthk', pk, alpha)
        vp = v[..., channel] - (yc[:, None] * alpha).sum(-1)
        valid = feasible[:, None] & ((kp.square().sum(-1) - c * vp.square()) > eps)
        keys.append(torch.where(valid[..., None], kp, k))
        values.append(torch.where(valid, vp, v[..., channel]))
        tape.append((ginv, torch.where(valid[..., None], alpha, 0), valid))
    return torch.stack(keys, -2), torch.stack(values, -1), (pk, y, active, tape)


def _projection_vjp(args, tape, dkeys, dvalues):
    k, v, bank_k, bank_v, c = args
    pk, y, active, channels = tape
    dk, dv, dpk, dy, dc = (torch.zeros_like(x) for x in args)
    for channel, (ginv, alpha, valid) in enumerate(channels):
        gk, gv, yc, vc = dkeys[..., channel, :], dvalues[..., channel], y[..., channel], v[..., channel]
        galpha = -torch.einsum('bhrk,bthk->bthr', pk, gk) - yc[:, None] * gv[..., None]
        z = torch.einsum('bhjr,bthj->bthr', ginv, galpha)
        z = torch.where(valid[..., None], z, 0)
        alpha_pk = torch.einsum('bhrk,bthr->bthk', pk, alpha)
        z_pk = torch.einsum('bhrk,bthr->bthk', pk, z)
        dk += gk + z_pk
        dv[..., channel] = gv - c * (yc[:, None] * z).sum(-1)
        gpk = -alpha[..., None] * gk[..., None, :]
        gpk += z[..., None] * (k - alpha_pk)[..., None, :]
        gpk -= alpha[..., None] * z_pk[..., None, :]
        dpk += gpk.sum(1) * active[..., None]
        y_alpha, y_z = (yc[:, None] * alpha).sum(-1), (yc[:, None] * z).sum(-1)
        gy = -alpha * gv[..., None] - c * z * vc[..., None]
        gy += c * z * y_alpha[..., None] + c * alpha * y_z[..., None]
        dy[..., channel] = gy.sum(1) * active
        dc += (y_z * (y_alpha - vc)).sum()
    return dk, dv, dpk, dy, dc


@pytest.mark.parametrize('rank', [1, 3, 16])
@pytest.mark.parametrize('occupancy', ['empty', 'partial', 'full', 'infeasible'])
@pytest.mark.parametrize('curvature', [0.0, 0.3, 1.0])
def test_shared_inverse_math_and_all_projection_gradients(rank, occupancy, curvature):
    args, scores = _case(rank, occupancy, curvature)
    shared_k, shared_v, tape = _projection(args, scores, shared=True)
    repeated_k, repeated_v, _ = _projection(args, scores, shared=False)
    oracle_k, oracle_v = reference._project_chunk_torch(*args[:2], args[2], args[3], scores, args[4])
    for actual, expected in ((shared_k, repeated_k), (shared_v, repeated_v), (shared_k, oracle_k), (shared_v, oracle_v)):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    dkeys, dvalues = torch.randn_like(shared_k), torch.randn_like(shared_v)
    gradients = torch.autograd.grad((shared_k, shared_v), args, (dkeys, dvalues))
    repeated = torch.autograd.grad((repeated_k, repeated_v), args, (dkeys, dvalues))
    oracle = torch.autograd.grad((oracle_k, oracle_v), args, (dkeys, dvalues))
    with torch.no_grad():
        manual = _projection_vjp(args, tape, dkeys, dvalues)
    for actual, repeated_grad, oracle_grad, manual_grad in zip(gradients, repeated, oracle, manual):
        for expected in (repeated_grad, oracle_grad, manual_grad):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


def test_shared_inverse_gradcheck_and_nearly_dependent_bank():
    args, scores = _case(rank=2)
    k, v, bank_k, bank_v, c = args
    k, v = (x[:1, :2, :1].detach().requires_grad_() for x in (k, v))
    bank_k, bank_v = (x[:1, :1].detach().requires_grad_() for x in (bank_k, bank_v))
    scores = scores[:1, :1]
    args = k, v, bank_k, bank_v, c
    assert torch.autograd.gradcheck(lambda *xs: _projection(xs, scores, True)[:2], args, eps=1e-6, atol=1e-5, rtol=1e-4)
    near_k = bank_k.detach().clone()
    near_k[..., 1, :] = near_k[..., 0, :] + 1e-7
    near = k, v, near_k, torch.zeros_like(bank_v), c
    shared = _projection(near, scores, True)[:2]
    repeated = _projection(near, scores, False)[:2]
    for actual, expected in zip(shared, repeated):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


def _run_kernel_pair(monkeypatch, device, dtype, size, rank, occupancy, beta_value):
    from fla.ops.hyperbolic_delta_rule.protected_triton import _ValueParallelChunk

    base, scores = _case(rank=rank, occupancy=occupancy)
    k, v, pk, pv, c = (x.detach().to(device=device, dtype=torch.float32) for x in base)
    k = k[:, :1].expand(-1, size, -1, -1).contiguous()
    v = v[:, :1].expand(-1, size, -1, -1).contiguous()
    q = torch.randn_like(k)
    beta = torch.rand_like(v) if beta_value is None else torch.full_like(v, beta_value)
    state = torch.randn(k.shape[0], k.shape[2], k.shape[-1], v.shape[-1], device=device)
    state = 0.05 * torch.nn.functional.normalize(state, dim=-2)
    inputs = (q.to(dtype), k, v.to(dtype), beta, state, pk, pv, c)
    inputs = tuple(x.detach().requires_grad_() for x in inputs)
    q, k, v, beta, state, pk, pv, c = inputs
    scores = scores.to(device=device, dtype=torch.float32)
    actuals, gradients = [], []
    upstream = None
    for shared in (False, True):
        monkeypatch.setenv('FLA_HYPERBOLIC_SHARED_INVERSE', str(int(shared)))
        output = _ValueParallelChunk.apply(q, k, v, beta, state, pk, pv, scores, c, 0.37, shared)
        assert output[0].grad_fn.shared_inverse is shared
        monkeypatch.setenv('FLA_HYPERBOLIC_SHARED_INVERSE', str(int(not shared)))
        if upstream is None:
            upstream = tuple(torch.randn_like(x) for x in output)
        gradients.append(torch.autograd.grad(output, inputs, upstream))
        actuals.append(output)
    for group in (actuals, gradients):
        for actual, expected in zip(*group):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    oracle_inputs = tuple(x.detach().double().requires_grad_() for x in inputs)
    qr, kr, vr, br, sr, pkr, pvr, cr = oracle_inputs
    kp, vp = reference._project_chunk_torch(kr, vr, pkr, pvr, scores.double(), cr)
    expected = reference._recurrent_chunk_torch(qr, kp, vp, br, sr, cr, 0.37, kr)
    expected_grads = torch.autograd.grad(expected, oracle_inputs, tuple(x.double() for x in upstream))
    for actual, oracle in zip(actuals[1], expected):
        torch.testing.assert_close(actual.float(), oracle.to(actual.dtype).float(), atol=5e-3, rtol=5e-2)
    for actual, oracle in zip(gradients[1], expected_grads):
        torch.testing.assert_close(actual.float(), oracle.to(actual.dtype).float(), atol=8e-3, rtol=8e-2)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 (CPU execution, not CUDA)')
@pytest.mark.parametrize('rank,occupancy,size,beta', [(1, 'empty', 1, 0.0), (3, 'partial', 17, 1.0), (16, 'full', 3, None)])
def test_shared_inverse_interpreter_actual_forward_backward(monkeypatch, rank, occupancy, size, beta):
    _run_kernel_pair(monkeypatch, 'cpu', torch.float32, size, rank, occupancy, beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA is unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rank,occupancy,size,beta', [(1, 'empty', 1, 0.0), (3, 'partial', 17, 1.0), (16, 'full', 32, None)])
def test_shared_inverse_cuda_actual_forward_backward(monkeypatch, dtype, rank, occupancy, size, beta):
    _run_kernel_pair(monkeypatch, 'cuda', dtype, size, rank, occupancy, beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA is unavailable')
@pytest.mark.parametrize('length,chunk,checkpoint', [(0, 16, False), (35, 16, False), (65, 32, True), (1025, 16, True)])
def test_shared_inverse_cuda_multichunk_bank_and_checkpoint(monkeypatch, length, chunk, checkpoint):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    torch.manual_seed(512)
    B, H, K, V, R = 2, 2, 32, 16, 3
    base = (
        torch.randn(B, length, H, K, device='cuda'),
        torch.nn.functional.normalize(torch.randn(B, length, H, K, device='cuda'), dim=-1),
        0.02 * torch.randn(B, length, H, V, device='cuda'),
        torch.rand(B, length, H, V, device='cuda'),
        0.02 * torch.randn(B, H, K, V, device='cuda'),
        torch.nn.functional.normalize(torch.randn(B, H, R, K, device='cuda'), dim=-1),
        0.02 * torch.randn(B, H, R, V, device='cuda'),
        torch.zeros(B, H, R, device='cuda'),
        torch.randn(B, length, H, device='cuda'),
        torch.tensor(0.3, device='cuda'),
    )
    results = []
    upstream = None
    for shared in (False, True):
        monkeypatch.setenv('FLA_HYPERBOLIC_SHARED_INVERSE', str(int(shared)))
        inputs = tuple(x.detach().requires_grad_() for x in base)
        q, k, v, beta, state, pk, pv, ps, scores, c = inputs
        out = triton_protected_chunkwise_hyperbolic_delta_rule(
            q, k, v, beta, initial_state=state, protected_keys=pk, protected_values=pv,
            protected_scores=ps, selection_scores=scores, curvature=c, protection_rank=R,
            checkpoint_size=chunk, checkpoint=checkpoint, kernel_strategy='value_parallel',
        )
        if upstream is None:
            upstream = tuple(torch.randn_like(x) for x in out[:5])
        differentiable = [(x, g) for x, g in zip(out[:5], upstream) if x.requires_grad]
        grads = torch.autograd.grad(
            tuple(x for x, _ in differentiable), inputs,
            tuple(g for _, g in differentiable), allow_unused=True,
        )
        grads = tuple(torch.zeros_like(x) if g is None else g for x, g in zip(inputs, grads))
        results.append((out, grads))
    for actual, expected in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    for actual, expected in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-2)
