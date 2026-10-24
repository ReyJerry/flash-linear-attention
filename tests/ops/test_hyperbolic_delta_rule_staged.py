# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Independent staged algebra/VJP tests and explicitly selected CUDA coverage."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('hyperbolic_staged_reference', _ROOT / 'fla/ops/hyperbolic_delta_rule/naive.py')
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _case(length, *, curvature=0.4, batch=2, heads=2, key_dim=7, value_dim=5, dtype=torch.float64):
    torch.manual_seed(781)
    q = torch.randn(batch, length, heads, key_dim, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(batch, length, heads, value_dim, dtype=dtype).tanh() * 0.2 / max(curvature, 1) ** 0.5
    beta = torch.rand_like(v)
    state = torch.nn.functional.normalize(torch.randn(batch, heads, key_dim, value_dim, dtype=dtype), dim=-2)
    state *= 0.1 / max(curvature, 1) ** 0.5
    c = torch.tensor(curvature, dtype=dtype)
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, state, c))


def _staged_forward(q, k, v, beta, h0, c, chunk_size=16, bounds=None, scale=0.37):
    """Use independent batched triangular solves and a separate readout pass."""
    batch, length, heads, key_dim = q.shape
    bounds = bounds if bounds is not None else [i * length for i in range(batch + 1)]
    vector_beta = beta if beta.ndim == 4 else beta.unsqueeze(-1).expand_as(v)
    qf, kf, vf, bf = (x.reshape(batch * length, heads, x.shape[-1]) for x in (q, k, v, vector_beta))
    tape, finals = {}, []
    # public empty calls retain zero input gradients and identity state gradients
    anchor = (q.sum() + k.sum() + v.sum() + beta.sum() + c) * 0
    output = torch.zeros_like(vf) + anchor
    for sequence, (bos, eos) in enumerate(zip(bounds, bounds[1:])):
        head_states = []
        for head in range(heads):
            state = h0[sequence, head]
            blocks = []
            for start in range(bos, eos, chunk_size):
                end = min(start + chunk_size, eos)
                kk, vv, bb = kf[start:end, head], vf[start:end, head], bf[start:end, head]
                gram = kk @ kk.T
                a = kk.square().sum(-1, keepdim=True) - c * vv.square()
                alpha = bb / a
                per_value = gram.unsqueeze(0) - c * vv.T.unsqueeze(-1) * vv.T.unsqueeze(-2)
                lower = torch.eye(end - start, dtype=q.dtype, device=q.device)
                lower = lower + torch.tril(alpha.T.unsqueeze(-1) * per_value, diagonal=-1)
                residual = vv - kk @ state
                weight = torch.linalg.solve_triangular(lower, (alpha * residual).T.unsqueeze(-1), upper=False)
                weight = weight.squeeze(-1).T
                denominator = 1 + c * (vv * weight).cumsum(0)
                next_state = (state + kk.T @ weight) / denominator[-1]
                blocks.append((start, end, state, next_state, gram, lower, weight, denominator, a, alpha))
                state = next_state
            tape[sequence, head] = blocks
            head_states.append(state)
        finals.append(torch.stack(head_states))
    for (_, head), blocks in tape.items():
        for start, end, state, _, _, _, weight, denominator, _, _ in blocks:
            qq, kk = qf[start:end, head], kf[start:end, head]
            causal = torch.tril(qq @ kk.T)
            output[start:end, head] = scale * (qq @ state + causal @ weight) / denominator
    return output.reshape_as(v), torch.stack(finals) + anchor, tape


def _staged_vjp(args, d_output, d_final, chunk_size=16, bounds=None, scale=0.37):
    q, k, v, beta, h0, c = args
    output, _, tape = _staged_forward(*args, chunk_size=chunk_size, bounds=bounds, scale=scale)
    batch, length, heads, key_dim = q.shape
    vector_beta = beta if beta.ndim == 4 else beta.unsqueeze(-1).expand_as(v)
    qf, kf, vf, bf = (x.reshape(batch * length, heads, x.shape[-1]) for x in (q, k, v, vector_beta))
    do = d_output.reshape_as(vf)
    of = output.reshape_as(vf)
    dq, dk, dv, db = (torch.zeros_like(x) for x in (qf, kf, vf, bf))
    dh0, dc = torch.zeros_like(h0), torch.zeros_like(c)
    local, state_adjoints = {}, {}
    for (sequence, head), blocks in tape.items():
        for block_index, block in enumerate(blocks):
            start, end, state, _, _, _, weight, denominator, _, _ = block
            qq, kk = qf[start:end, head], kf[start:end, head]
            causal = torch.tril(qq @ kk.T)
            d = scale * do[start:end, head] / denominator
            dz = -do[start:end, head] * of[start:end, head] / denominator
            suffix = dz.flip(0).cumsum(0).flip(0)
            local[sequence, head, block_index] = (d, causal.T @ d, suffix)
    for (sequence, head), blocks in tape.items():
        ds = d_final[sequence, head].clone()
        for block_index in range(len(blocks) - 1, -1, -1):
            start, end, state, send, _, lower, weight, denominator, _, alpha = blocks[block_index]
            qq, kk, vv = qf[start:end, head], kf[start:end, head], vf[start:end, head]
            d, dw_readout, suffix_readout = local[sequence, head, block_index]
            normalized_ds = ds / denominator[-1]
            suffix = suffix_readout - (ds * send).sum(0) / denominator[-1]
            dw = dw_readout + kk @ normalized_ds + c * vv * suffix
            z = torch.linalg.solve_triangular(lower.transpose(-1, -2), dw.T.unsqueeze(-1), upper=True)
            z = z.squeeze(-1).T
            u = alpha * z
            ds = qq.T @ d + normalized_ds - kk.T @ u
            state_adjoints[sequence, head, block_index] = (normalized_ds, suffix, z, u)
        dh0[sequence, head] = ds
    for (sequence, head), blocks in tape.items():
        for block_index, block in enumerate(blocks):
            start, end, state, _, gram, _, weight, denominator, a, _ = block
            qq, kk, vv, bb = qf[start:end, head], kf[start:end, head], vf[start:end, head], bf[start:end, head]
            d = local[sequence, head, block_index][0]
            normalized_ds, suffix, z, u = state_adjoints[sequence, head, block_index]
            dp = torch.tril(d @ weight.T)
            dg = -torch.tril(u @ weight.T, diagonal=-1)
            prefix_vw = (vv * weight).cumsum(0) - vv * weight
            uv = u * vv
            suffix_uv = uv.sum(0) - uv.cumsum(0)
            d_alpha = z * (vv - kk @ state - torch.tril(gram, diagonal=-1) @ weight + c * vv * prefix_vw)
            da = -d_alpha * bb / a.square()
            dq[start:end, head] = d @ state.T + dp @ kk
            dk[start:end, head] = (
                dp.T @ qq + weight @ normalized_ds.T - u @ state.T
                + (dg + dg.T) @ kk + 2 * kk * da.sum(-1, keepdim=True)
            )
            dv[start:end, head] = c * weight * suffix + c * (u * prefix_vw + weight * suffix_uv) + u - 2 * c * vv * da
            db[start:end, head] = d_alpha / a
            dc += (vv * weight * suffix + u * vv * prefix_vw - da * vv.square()).sum()
    db = db.reshape_as(vector_beta)
    if beta.ndim == 3:
        db = db.sum(-1)
    return dq.reshape_as(q), dk.reshape_as(k), dv.reshape_as(v), db, dh0, dc


@pytest.mark.parametrize('length', [0, 1, 17, 35])
@pytest.mark.parametrize('curvature', [0.0, 0.4, 2.5])
@pytest.mark.parametrize('gate', ['random', 'zero', 'one'])
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_mathematical_forward_and_vjp(length, curvature, gate, chunk_size):
    args = list(_case(length, curvature=curvature))
    if gate != 'random':
        args[3] = torch.full_like(args[3], int(gate == 'one'), requires_grad=True)
    actual, final, _ = _staged_forward(*args, chunk_size=chunk_size)
    expected, expected_final = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, output_final_state=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final, expected_final, atol=1e-11, rtol=1e-10)
    do, dh = torch.randn_like(actual), torch.randn_like(final)
    reference_vjp = torch.autograd.grad((expected, expected_final), args, (do, dh), allow_unused=True)
    staged_autograd = torch.autograd.grad((actual, final), args, (do, dh))
    with torch.no_grad():
        manual = _staged_vjp(args, do, dh, chunk_size=chunk_size)
    for arg, expected_grad, automatic_grad, manual_grad in zip(args, reference_vjp, staged_autograd, manual):
        expected_grad = torch.zeros_like(arg) if expected_grad is None else expected_grad
        torch.testing.assert_close(automatic_grad, expected_grad, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(manual_grad, expected_grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_packed_empty_sequences_and_scalar_gate(chunk_size):
    args = list(_case(38, batch=1))
    args[3] = args[3][..., 0].detach().requires_grad_()
    bounds = [0, 0, 1, 18, 18, 38]
    args[4] = args[4].detach().expand(5, -1, -1, -1).contiguous().requires_grad_()
    actual, final, _ = _staged_forward(*args, chunk_size=chunk_size, bounds=bounds)
    expected, expected_final = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], cu_seqlens=torch.tensor(bounds), scale=0.37,
        output_final_state=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final, expected_final, atol=1e-11, rtol=1e-10)
    do, dh = torch.randn_like(actual), torch.randn_like(final)
    gradients = torch.autograd.grad((expected, expected_final), args, (do, dh))
    with torch.no_grad():
        manual = _staged_vjp(args, do, dh, chunk_size=chunk_size, bounds=bounds)
    for result, oracle in zip(manual, gradients):
        torch.testing.assert_close(result, oracle, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(manual[4][[0, 3]], dh[[0, 3]])


def test_staged_math_gradcheck():
    args = _case(3, batch=1, heads=1, key_dim=3, value_dim=2)
    assert torch.autograd.gradcheck(lambda *inputs: _staged_forward(*inputs)[:2], args, eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('key_scale', [2**-10, 1.0])
def test_staged_math_near_feasible_boundary(key_scale):
    args = list(_case(3, curvature=1.0, batch=1, heads=1, key_dim=3, value_dim=2))
    args[1] = (args[1].detach() * key_scale).requires_grad_()
    args[2] = torch.tensor([0.95, -0.94, -0.93, 0.92, 0.96, -0.91], dtype=torch.float64).reshape(1, 3, 1, 2)
    args[2] = (args[2] * key_scale).requires_grad_()
    args[3] = torch.ones_like(args[3], requires_grad=True)
    out, final, _ = _staged_forward(*args)
    reference, reference_final = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, output_final_state=True,
    )
    torch.testing.assert_close(out, reference, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final, reference_final, atol=1e-11, rtol=1e-10)
    do, dh = torch.randn_like(out), torch.randn_like(final)
    expected = torch.autograd.grad((reference, reference_final), args, (do, dh))
    with torch.no_grad():
        actual = _staged_vjp(args, do, dh)
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle, atol=1e-10, rtol=1e-9)


def _kernel_case(length, batch, heads, key_dim, value_dim, chunk_size, curvature, gate, packed, device, dtype):
    args = list(_case(length, batch=batch, heads=heads, key_dim=key_dim, value_dim=value_dim, curvature=curvature))
    args = [x.detach().to(device=device, dtype=dtype if index < 4 else torch.float32).requires_grad_()
            for index, x in enumerate(args)]
    if gate != 'random':
        args[3] = torch.full_like(args[3], int(gate == 'one'), requires_grad=True)
    if packed:
        bounds = [0, 0, min(length, 1), min(length, 18), min(length, 18), length]
        args[4] = args[4].detach().expand(len(bounds) - 1, -1, -1, -1).contiguous().requires_grad_()
    else:
        bounds = [index * length for index in range(batch + 1)]
    return tuple(args), torch.tensor(bounds, device=device, dtype=torch.int32 if packed else torch.int64)


def _assert_kernel_reference(args, cu, output, final, *, packed, final_only=False):
    q, k, v, beta, h0, c = args
    oracle_args = tuple(x.detach().clone().requires_grad_() for x in args)
    expected, expected_final = _reference.naive_recurrent_hyperbolic_delta_rule(
        *oracle_args[:4], initial_state=oracle_args[4], curvature=oracle_args[5], scale=0.37,
        cu_seqlens=cu if packed else None, output_final_state=True,
    )
    # use the existing TC test's dtype-specific tolerances
    atol, rtol = {torch.float32: (8e-4, 8e-3), torch.float16: (1e-2, 3e-2), torch.bfloat16: (3e-2, 8e-2)}[q.dtype]
    torch.testing.assert_close(output.float(), expected.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final, atol=atol, rtol=rtol)
    do, dh = torch.randn_like(output), torch.randn_like(final)
    if final_only:
        actual_gradients = torch.autograd.grad(final, args, dh, allow_unused=True)
        expected_gradients = torch.autograd.grad(expected_final, oracle_args, dh, allow_unused=True)
    else:
        actual_gradients = torch.autograd.grad((output, final), args, (do, dh), allow_unused=True)
        expected_gradients = torch.autograd.grad((expected, expected_final), oracle_args, (do, dh), allow_unused=True)
    for arg, actual, expected in zip(args, actual_gradients, expected_gradients):
        actual = torch.zeros_like(arg) if actual is None else actual
        expected = torch.zeros_like(arg) if expected is None else expected
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol * 3, rtol=rtol * 3)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 before importing Triton')
@pytest.mark.parametrize('length,batch,heads,key_dim,value_dim,chunk_size,curvature,gate,packed', [
    (0, 2, 1, 7, 5, 16, 0.0, 'random', False),
    (1, 2, 2, 17, 19, 16, 0.4, 'zero', False),
    (17, 1, 1, 7, 5, 16, 0.4, 'one', False),
    (35, 1, 2, 17, 19, 32, 2.5, 'random', True),
    (33, 1, 1, 64, 48, 16, 0.4, 'random', False),
    (65, 1, 1, 128, 128, 32, 0.4, 'random', False),
])
@pytest.mark.parametrize('final_only', [False, True])
def test_staged_triton_interpreter(length, batch, heads, key_dim, value_dim, chunk_size, curvature, gate, packed, final_only):
    from fla.ops.hyperbolic_delta_rule.staged import StagedHyperbolicDeltaRuleFunction

    args, cu = _kernel_case(
        length, batch, heads, key_dim, value_dim, chunk_size, curvature, gate, packed, 'cpu', torch.float32,
    )
    output, final = StagedHyperbolicDeltaRuleFunction.apply(*args[:5], cu, args[5], 0.37, chunk_size)
    _assert_kernel_reference(args, cu, output, final, packed=packed, final_only=final_only)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 before importing Triton')
@pytest.mark.parametrize('curvature_only', [False, True])
def test_staged_triton_interpreter_partial_autograd(curvature_only):
    from fla.ops.hyperbolic_delta_rule.staged import StagedHyperbolicDeltaRuleFunction

    args, cu = _kernel_case(17, 1, 1, 17, 19, 16, 0.0, 'random', False, 'cpu', torch.float32)
    args = tuple(x.detach().requires_grad_(index == 5 if curvature_only else index < 5) for index, x in enumerate(args))
    output, _ = StagedHyperbolicDeltaRuleFunction.apply(*args[:5], cu, args[5], 0.37, 16)
    expected, _ = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37,
    )
    active_args = tuple(x for x in args if x.requires_grad)
    do = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(output, active_args, do)
    expected_gradients = torch.autograd.grad(expected, active_args, do)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=2.4e-3, rtol=2.4e-2)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 before importing Triton')
def test_staged_triton_interpreter_noncontiguous_cache():
    from fla.ops.hyperbolic_delta_rule.staged import StagedHyperbolicDeltaRuleFunction

    args, cu = _kernel_case(35, 1, 1, 17, 19, 16, 0.4, 'random', False, 'cpu', torch.float32)
    q, k, v, beta, h0, c = args
    scalar_beta = beta[..., 0].detach().requires_grad_()
    beta = scalar_beta.unsqueeze(-1).expand_as(v)
    output, final = StagedHyperbolicDeltaRuleFunction.apply(q, k, v, beta, h0, cu, c, 0.37, 16)
    state, outputs = h0, []
    for start, end in [(0, 13), (13, 13), (13, 21), (21, 35)]:
        segment = [torch.stack((x[:, start:end], x[:, start:end]), -1)[..., 0] for x in (q, k, v, beta)]
        value, state = StagedHyperbolicDeltaRuleFunction.apply(
            *segment, state, torch.tensor([0, end - start]), c, 0.37, 16,
        )
        outputs.append(value)
    partial = torch.cat(outputs, dim=1)
    torch.testing.assert_close(partial, output, atol=8e-4, rtol=8e-3)
    torch.testing.assert_close(state, final, atol=8e-4, rtol=8e-3)
    do, dh = torch.randn_like(output), torch.randn_like(final)
    inputs = (q, k, v, scalar_beta, h0, c)
    full_gradients = torch.autograd.grad((output, final), inputs, (do, dh))
    partial_gradients = torch.autograd.grad((partial, state), inputs, (do, dh))
    for actual, expected in zip(partial_gradients, full_gradients):
        torch.testing.assert_close(actual, expected, atol=2.4e-3, rtol=2.4e-2)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 before importing Triton')
@pytest.mark.parametrize('dtypes', [
    (torch.float16, torch.float32, torch.float32),
    (torch.float32, torch.float16, torch.float16),
    (torch.float32, torch.bfloat16, torch.bfloat16),
])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('need_dc', [False, True])
def test_staged_interpreter_public_host_dispatch(monkeypatch, dtypes, packed, need_dc):
    """Exercise host routing with mocked device labels and real CPU-interpreted kernels, not CUDA execution."""
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    class LogicalCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device('cuda')

        def contiguous(self, *args, **kwargs):
            # Validation sees the logical label; normalization returns the real CPU tensor.
            return self.as_subclass(torch.Tensor).contiguous(*args, **kwargs)

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, cu = _kernel_case(19, 1, 1, 17, 19, 16, 0.4, 'random', packed, 'cpu', torch.float32)
    args = list(args)
    for index, dtype in enumerate(dtypes):
        args[index] = args[index].detach().to(dtype).requires_grad_()
    args[3] = args[3][..., 0].detach().requires_grad_()
    args[5] = args[5].detach().requires_grad_(need_dc)
    logical = [x.as_subclass(LogicalCudaTensor) for x in args]
    output, final = chunk_hyperbolic_delta_rule(
        *logical[:4], initial_state=logical[4], curvature=logical[5], scale=0.37, chunk_size=16,
        cu_seqlens=cu.as_subclass(LogicalCudaTensor) if packed else None,
        output_final_state=True, tensor_core=True, check_inputs=True,
    )
    assert output.device.type == final.device.type == 'cpu'
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    assert output.dtype == dtypes[0] and final.dtype == torch.float32
    reference, reference_final = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37,
        cu_seqlens=cu if packed else None, output_final_state=True,
    )
    atol, rtol = {torch.float32: (8e-4, 8e-3), torch.float16: (1e-2, 3e-2)}[dtypes[0]]
    torch.testing.assert_close(output.float(), reference.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, reference_final, atol=atol, rtol=rtol)
    do, dh = torch.randn_like(output), torch.randn_like(final)
    active_args = tuple(x for x in args if x.requires_grad)
    # A changed process environment must not change the already-selected backward implementation.
    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '0')
    actual_gradients = torch.autograd.grad((output, final), active_args, (do, dh))
    reference_gradients = torch.autograd.grad((reference, reference_final), active_args, (do, dh))
    for actual, expected in zip(actual_gradients, reference_gradients):
        assert actual.device.type == 'cpu' and torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol * 3, rtol=rtol * 3)


def _assert_near_feasible_existing_tc_contract(device, chunk_size):
    from fla.ops.hyperbolic_delta_rule.fused_recurrent import HyperbolicDeltaRuleFunction
    from fla.ops.hyperbolic_delta_rule.staged import StagedHyperbolicDeltaRuleFunction

    key = torch.full((1, 1, 1, 128), 0.0003064170596189797, dtype=torch.float32)
    key[..., 0] = 1.0
    value = torch.full((1, 1, 1, 17), 1.0000059604644775, dtype=torch.float32)
    # This point is feasible in real arithmetic, but is intentionally ill-conditioned.
    exact_a = key.double().square().sum(-1, keepdim=True) - value.double().square()
    assert (exact_a > 0).all()
    if device == 'cpu':
        full_a = key.square().sum(-1, keepdim=True) - value.square()
        partial_norm = sum(key[..., start:start + 32].square().sum(-1, keepdim=True) for start in range(0, 128, 32))
        assert (full_a > 0).all() and (partial_norm - value.square() <= 0).all()
    query = torch.zeros_like(key)
    query[..., 0] = 1.0
    tensors = (query, key, value, torch.zeros_like(value), torch.zeros(1, 1, 128, 17), torch.tensor(1.0))
    args = tuple(x.to(device).requires_grad_() for x in tensors)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    actual = StagedHyperbolicDeltaRuleFunction.apply(*args[:5], cu, args[5], 0.37, chunk_size)
    expected = HyperbolicDeltaRuleFunction.apply(
        *args[:5], cu, args[5], 0.37, chunk_size, 'chunk_tc', True, None,
    )[:2]
    for result, baseline in zip(actual, expected):
        torch.testing.assert_close(result, baseline, atol=8e-4, rtol=8e-3)
    cotangents = tuple(torch.ones_like(x) for x in actual)
    actual_gradients = torch.autograd.grad(actual, args, cotangents)
    expected_gradients = torch.autograd.grad(expected, args, cotangents)
    for result, baseline in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, baseline, atol=2.4e-3, rtol=2.4e-2)
    if device == 'cpu':
        assert (expected_gradients[3].abs() > 0).all()


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires TRITON_INTERPRET=1 before importing Triton')
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_interpreter_near_feasible_full_key_norm(monkeypatch, chunk_size):
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    _assert_near_feasible_existing_tc_contract('cpu', chunk_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_cuda_near_feasible_full_key_norm(monkeypatch, chunk_size):
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    _assert_near_feasible_existing_tc_contract('cuda', chunk_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('length,batch,heads,key_dim,value_dim,chunk_size,curvature,gate,packed', [
    (0, 2, 1, 16, 17, 16, 0.0, 'random', False),
    (1, 2, 2, 17, 19, 16, 0.4, 'zero', False),
    (17, 1, 1, 16, 16, 16, 0.4, 'one', False),
    (35, 1, 2, 17, 19, 32, 2.5, 'random', True),
    (33, 2, 2, 64, 48, 16, 0.4, 'random', False),
    (65, 1, 2, 128, 128, 32, 0.4, 'random', False),
])
@pytest.mark.parametrize('final_only', [False, True])
def test_staged_cuda_reference_and_all_gradients(monkeypatch, dtype, length, batch, heads, key_dim, value_dim,
                                                 chunk_size, curvature, gate, packed, final_only):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, cu = _kernel_case(length, batch, heads, key_dim, value_dim, chunk_size, curvature, gate, packed, 'cuda', dtype)
    output, final = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, chunk_size=chunk_size,
        cu_seqlens=cu if packed else None, output_final_state=True, tensor_core=True,
    )
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    _assert_kernel_reference(args, cu, output, final, packed=packed, final_only=final_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('dtypes', [
    (torch.float16, torch.float32, torch.float32),
    (torch.float32, torch.float16, torch.float16),
    (torch.float32, torch.bfloat16, torch.bfloat16),
])
@pytest.mark.parametrize('packed', [False, True])
def test_staged_cuda_mixed_dtype_public_dispatch(monkeypatch, dtypes, packed):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, cu = _kernel_case(35, 1 if packed else 2, 1, 65, 33, 32, 0.4, 'random', packed, 'cuda', torch.float32)
    args = list(args)
    for index, dtype in enumerate(dtypes):
        args[index] = args[index].detach().to(dtype).requires_grad_()
    output, final = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, chunk_size=32,
        cu_seqlens=cu if packed else None, output_final_state=True, tensor_core=True, check_inputs=True,
    )
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    _assert_kernel_reference(args, cu, output, final, packed=packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('curvature_only', [False, True])
def test_staged_cuda_partial_autograd(monkeypatch, curvature_only):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, _ = _kernel_case(17, 1, 1, 17, 19, 16, 0.0, 'random', False, 'cuda', torch.float32)
    args = tuple(x.detach().requires_grad_(index == 5 if curvature_only else index < 5) for index, x in enumerate(args))
    output, _ = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, tensor_core=True,
    )
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    expected, _ = _reference.naive_recurrent_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37,
    )
    active_args = tuple(x for x in args if x.requires_grad)
    do = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(output, active_args, do)
    expected_gradients = torch.autograd.grad(expected, active_args, do)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=2.4e-3, rtol=2.4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_cuda_noncontiguous_cached_segments(monkeypatch, chunk_size):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, _ = _kernel_case(35, 1, 1, 17, 19, chunk_size, 0.4, 'random', False, 'cuda', torch.float32)
    q, k, v, beta, h0, c = args
    beta = beta[..., 0].detach().requires_grad_()
    options = dict(curvature=c, scale=0.37, tensor_core=True, output_final_state=True, chunk_size=chunk_size)
    output, final = chunk_hyperbolic_delta_rule(q, k, v, beta, initial_state=h0, **options)
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    state, outputs = h0, []
    for start, end in [(0, 13), (13, 13), (13, 21), (21, 35)]:
        segment = [torch.stack((x[:, start:end], x[:, start:end]), -1)[..., 0] for x in (q, k, v, beta)]
        value, state = chunk_hyperbolic_delta_rule(*segment, initial_state=state, **options)
        assert type(value.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
        outputs.append(value)
    partial = torch.cat(outputs, dim=1)
    torch.testing.assert_close(partial, output, atol=8e-4, rtol=8e-3)
    torch.testing.assert_close(state, final, atol=8e-4, rtol=8e-3)
    do, dh = torch.randn_like(output), torch.randn_like(final)
    inputs = (q, k, v, beta, h0, c)
    full_gradients = torch.autograd.grad((output, final), inputs, (do, dh))
    partial_gradients = torch.autograd.grad((partial, state), inputs, (do, dh))
    for actual, expected in zip(partial_gradients, full_gradients):
        torch.testing.assert_close(actual, expected, atol=2.4e-3, rtol=2.4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('key_scale', [2**-10, 1.0])
def test_staged_cuda_small_positive_denominator(monkeypatch, key_scale):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, cu = _kernel_case(17, 1, 1, 17, 19, 16, 0.4, 'random', False, 'cuda', torch.float32)
    args = list(args)
    args[1] = (args[1].detach() * key_scale).requires_grad_()
    args[2] = (args[2].detach() * key_scale).requires_grad_()
    output, final = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, tensor_core=True,
        output_final_state=True, check_inputs=True,
    )
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    _assert_kernel_reference(args, cu, output, final, packed=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('length', [1025, 4097, 16384])
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_staged_cuda_long_sequence_against_existing_tc(monkeypatch, length, chunk_size):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '0')
    args, _ = _kernel_case(length, 1, 1, 128, 128, chunk_size, 0.4, 'random', False, 'cuda', torch.float32)
    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    output, final = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, chunk_size=chunk_size,
        output_final_state=True, tensor_core=True,
    )
    assert type(output.grad_fn).__name__ == 'StagedHyperbolicDeltaRuleFunctionBackward'
    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '0')
    expected, expected_final = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], scale=0.37, chunk_size=chunk_size,
        output_final_state=True, tensor_core=True,
    )
    assert type(expected.grad_fn).__name__ == 'HyperbolicDeltaRuleFunctionBackward'
    torch.testing.assert_close(output, expected, atol=8e-4, rtol=8e-3)
    torch.testing.assert_close(final, expected_final, atol=8e-4, rtol=8e-3)
    do, dh = torch.randn_like(output), torch.randn_like(final)
    actual_gradients = torch.autograd.grad((output, final), args, (do, dh))
    expected_gradients = torch.autograd.grad((expected, expected_final), args, (do, dh))
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2.4e-3, rtol=2.4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA NOT AVAILABLE: GPU validation not run')
@pytest.mark.parametrize('fallback', ['readout', 'large_key', 'legacy_fast'])
def test_staged_cuda_compatible_fallback(monkeypatch, fallback):
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule

    key_dim = 160 if fallback == 'large_key' else 16
    dtype = torch.float16 if fallback == 'legacy_fast' else torch.float32
    args, _ = _kernel_case(17, 1, 1, key_dim, 17, 16, 0.4, 'random', False, 'cuda', dtype)
    options = {'readout_q': args[0]} if fallback == 'readout' else {}
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', '1' if fallback == 'legacy_fast' else '0')
    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '1')
    actual = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], output_final_state=True, tensor_core=True, **options,
    )
    assert type(actual[0].grad_fn).__name__ == 'HyperbolicDeltaRuleFunctionBackward'
    monkeypatch.setenv('FLA_HYPERBOLIC_STAGED', '0')
    expected = chunk_hyperbolic_delta_rule(
        *args[:4], initial_state=args[4], curvature=args[5], output_final_state=True, tensor_core=True, **options,
    )
    cotangents = tuple(torch.randn_like(x) for x in actual)
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle)
    actual_gradients = torch.autograd.grad(actual, args, cotangents)
    expected_gradients = torch.autograd.grad(expected, args, cotangents)
    for result, oracle in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(result, oracle)
