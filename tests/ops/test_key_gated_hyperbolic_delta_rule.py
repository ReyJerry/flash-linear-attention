# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Actual CUDA KHDN entry points against independent same-rounded-source truth."""

# ruff: noqa: E402
import math

import pytest
import torch

pytest.importorskip('triton', reason='Actual KHDN GPU tests require the Triton runtime.')

import fla.ops.key_gated_hyperbolic_delta_rule as ops  # noqa: E402
from fla.utils import IS_NVIDIA, device  # noqa: E402

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='This first production gate targets NVIDIA GPUs.')


def make_case(length, key_dim, value_dim, group_size, dtype, packed=False):
    torch.manual_seed(42)
    batch, heads, sequences = (1, 2, 4) if packed else (2, 2, 2)
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    k = torch.nn.functional.normalize(k, dim=-1) * (.7 + .6 * torch.rand_like(k[..., :1]))
    v = torch.randn(batch, length, heads, value_dim // group_size, group_size, device=device)
    v = torch.nn.functional.normalize(v, dim=-1) * (.25 * math.sqrt(group_size)) * k.norm(dim=-1)[..., None, None]
    beta = .1 + .8 * torch.rand(batch, length, heads, device=device)
    if length:
        beta[:, 0] = 0
    if length > 1:
        beta[:, 1] = 1
    ell = -.01 - .2 * torch.rand_like(q)
    state = torch.randn(sequences, heads, key_dim, value_dim // group_size, group_size, device=device)
    state = state / state.square().sum((-3, -1), keepdim=True).sqrt() * (.1 * math.sqrt(group_size))
    sources = (q.to(dtype), k.to(dtype), v.flatten(-2).to(dtype), beta, ell,
               state.reshape(sequences, heads, key_dim, value_dim), q.new_tensor(.6))
    cu = torch.tensor([0, 0, length // 2, length, length], device=device, dtype=torch.int64) if packed else None
    return tuple(x.detach().requires_grad_() for x in sources), cu


def compare_tensor(name, actual, expected, backward=False):
    if actual.dtype == torch.float32:
        atol, rtol, ratio = (5e-5, 5e-4, 2e-4) if backward else (2e-5, 2e-4, 1e-4)
    elif actual.dtype == torch.bfloat16:
        atol, rtol, ratio = (.005, .03, .015) if backward else (.002, .02, .01)
    elif actual.dtype == torch.float16:
        atol, rtol, ratio = (.001, .005, .0025) if backward else (.0003, .003, .0015)
    else:
        raise AssertionError(f'Unexpected public dtype for {name}: {actual.dtype}')
    assert torch.isfinite(actual).all(), name
    actual64, expected64 = actual.double(), expected.double()
    torch.testing.assert_close(actual64, expected64, atol=atol, rtol=rtol, msg=name)
    if actual.numel():
        error_rms = (actual64 - expected64).square().mean().sqrt()
        reference_rms = expected64.square().mean().sqrt()
        assert error_rms <= ratio * reference_rms.clamp_min(atol), (
            name, error_rms.item(), reference_rms.item(), atol * ratio,
        )


def call(entry, sources, cu, group_size, chunk_size, carry_mode='no_carry', telemetry=None, output_final_state=True):
    function = getattr(ops, entry, None)
    assert callable(function), f'Production operator is missing: {entry}'
    kwargs = dict(checkpoint_size=chunk_size)
    if entry.startswith('chunk_'):
        kwargs = dict(chunk_size=chunk_size, carry_mode=carry_mode)
    return function(
        *sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=output_final_state,
        cu_seqlens=cu, group_size=group_size, check_inputs=True, telemetry=telemetry, **kwargs,
    )


def compare_case(entry, sources, cu, group_size, chunk_size, carry_mode='no_carry', cotangent_mode='joint'):
    reference_sources = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference_sources[:5], curvature=reference_sources[6], initial_state=reference_sources[5],
        output_final_state=True, cu_seqlens=cu, group_size=group_size,
    )
    info = {}
    actual = call(entry, sources, cu, group_size, chunk_size, carry_mode, info)
    assert actual[0].dtype == sources[0].dtype and actual[1].dtype == torch.float32
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    cotangents = tuple(torch.randn_like(x) for x in actual)
    if cotangent_mode == 'output':
        cotangents = (cotangents[0], torch.zeros_like(cotangents[1]))
    elif cotangent_mode == 'state':
        cotangents = (torch.zeros_like(cotangents[0]), cotangents[1])
    actual_grads = torch.autograd.grad(actual, sources, cotangents)
    reference_grads = torch.autograd.grad(expected, reference_sources, tuple(x.double() for x in cotangents))
    for name, a, r, source in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'),
                                  actual_grads, reference_grads, sources):
        assert a.dtype == source.dtype, name
        compare_tensor(name, a, r, backward=True)
    if sources[0].shape[1]:
        backend = 'triton_chunk' if entry.startswith('chunk_') else 'triton_recurrent'
        assert info['backend'] == backend
        assert info['backward'] == f'manual_{backend}_source_vjp'
    else:
        assert info['backend'] == info['backward'] == 'empty_identity'
    assert info['fallback_subblocks'] == 0
    if carry_mode == 'y_carry' and sources[0].shape[1] > chunk_size and cu is None:
        assert info['carry_edges'] > 0


@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('chunk_size', [16, 32, 64])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_chunk(carry_mode, chunk_size, dtype):
    sources, cu = make_case(2 * chunk_size + 3, 128, 128, 32, dtype)
    compare_case('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, chunk_size, carry_mode)


@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
@pytest.mark.parametrize('length,packed,group_size', [(0, False, 32), (0, True, 32), (1, False, 32),
                                                      (7, True, 32), (35, True, 16)])
@pytest.mark.parametrize('cotangent_mode', ['output', 'state', 'joint'])
def test_chunk_varlen(entry, length, packed, group_size, cotangent_mode):
    sources, cu = make_case(length, 32, 64, group_size, torch.float32, packed)
    compare_case(entry, sources, cu, group_size, 16, cotangent_mode=cotangent_mode)


@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
@pytest.mark.parametrize('group_size', [1, 32])
def test_chunk_strong_decay(entry, group_size):
    q = torch.ones(1, 2, 1, 1, device=device)
    k = q.clone()
    v = torch.zeros(1, 2, 1, group_size, device=device)
    v[0, :, 0, 0] = q.new_tensor([.5, .1])
    beta = q.new_tensor([1., 0.]).reshape(1, 2, 1)
    ell = q.new_tensor([-1e8, -1.]).reshape_as(q).requires_grad_()
    state = torch.zeros(1, 1, 1, group_size, device=device)
    c = q.new_tensor(.6)
    info = {}
    output, final = call(entry, (q, k, v, beta, ell, state, c), None, group_size, 16, telemetry=info)
    dell, = torch.autograd.grad(final[0, 0, 0, 0], ell)
    target = .5 * math.exp(-1)
    torch.testing.assert_close(final[0, 0, 0, 0], q.new_tensor(target), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dell[0, 1, 0, 0], q.new_tensor(target), atol=2e-5, rtol=2e-5)
    assert torch.isfinite(output).all() and info['fallback_subblocks'] == 0


@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
def test_chunk_zero_curvature_source(entry):
    q = torch.ones(1, 1, 1, 1, device=device)
    k, v = q.clone(), q.clone() * .5
    beta, ell = q.new_full((1, 1, 1), .5), torch.zeros_like(q)
    state, c = q.new_full((1, 1, 1, 1), .1), q.new_zeros((), requires_grad=True)
    _, final = call(entry, (q, k, v, beta, ell, state, c), None, 1, 16)
    dc, = torch.autograd.grad(final.sum(), c)
    torch.testing.assert_close(dc, q.new_tensor(.02), atol=1e-8, rtol=2e-5)
    assert dc.item() != 0


@pytest.mark.parametrize('length', [2048, 8192, 16384])
@pytest.mark.parametrize('chunk_size', [16, 32, 64])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_long(length, chunk_size, carry_mode):
    sources, cu = make_case(length, 128, 128, 32, torch.float32)
    compare_case('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, chunk_size, carry_mode)


def test_chunk_first_device_gate():
    sources, cu = make_case(7, 32, 64, 32, torch.float32)
    compare_case('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, 16)


@pytest.mark.parametrize('entry,carry', [('chunk_key_gated_hyperbolic_delta_rule', 'no_carry'),
                                         ('chunk_key_gated_hyperbolic_delta_rule', 'y_carry'),
                                         ('fused_recurrent_key_gated_hyperbolic_delta_rule', 'no_carry')])
def test_chunk_streaming_causal_noncontiguous(entry, carry):
    sources, cu = make_case(35, 32, 64, 32, torch.float32)
    sources = tuple((x.transpose(-1, -2).contiguous().transpose(-1, -2) if x.ndim >= 2 else x)
                    .detach().requires_grad_() for x in sources)
    truth = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *truth[:5], curvature=truth[6], initial_state=truth[5], output_final_state=True, group_size=32,
    )
    split = 19
    left = tuple(x[:, :split] for x in sources[:5]) + sources[5:]
    out1, state1 = call(entry, left, cu, 32, 16, carry)
    right = tuple(x[:, split:] for x in sources[:5]) + (state1, sources[6])
    out2, state2 = call(entry, right, cu, 32, 16, carry)
    actual = (torch.cat((out1, out2), dim=1), state2)
    cot = tuple(torch.randn_like(x) for x in actual)
    for name, a, r in zip(('stream_output', 'stream_state'), actual, expected):
        compare_tensor(name, a, r)
    ag = torch.autograd.grad(actual, sources, cot)
    rg = torch.autograd.grad(expected, truth, tuple(x.double() for x in cot))
    for name, a, r in zip(('dq', 'dk', 'dv', 'db', 'dell', 'dS0', 'dc'), ag, rg):
        compare_tensor(name, a, r, True)
    full, _ = call(entry, sources, cu, 32, 16, carry)
    grads = torch.autograd.grad(full[:, :split].sum(), sources[:5])
    for grad in grads:
        assert torch.count_nonzero(grad[:, split:]).item() == 0
    changed = tuple(x.detach().clone() for x in sources)
    changed[0][:, split:] *= 2
    changed[2][:, split:] *= .5
    changed[4][:, split:] -= .3
    altered, _ = call(entry, changed, cu, 32, 16, carry)
    torch.testing.assert_close(full[:, :split], altered[:, :split], atol=0, rtol=0)


@pytest.mark.parametrize('carry', ['no_carry', 'y_carry'])
def test_chunk_first_order_only(carry):
    sources, cu = make_case(3, 8, 32, 32, torch.float32)
    output, _ = call('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, 16, carry)
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(output.sum(), sources, create_graph=True)


@pytest.mark.parametrize('requested_state', [False, True])
def test_chunk_exception_optional_state(requested_state):
    q = torch.full((1, 3, 1, 1), 1e-40, device=device)
    k = q.clone()
    v = torch.ones_like(q, requires_grad=True)
    beta = q.new_tensor([1., 0., 0.]).reshape(1, 3, 1)
    ell = q.new_tensor([0., -math.log(2), -math.log(2)]).reshape_as(q)
    state, c = q.new_zeros((1, 1, 1, 1)), q.new_zeros(())
    sources, info = (q, k, v, beta, ell, state, c), {}
    if requested_state:
        with pytest.raises(RuntimeError, match='representable'):
            call('chunk_key_gated_hyperbolic_delta_rule', sources, None, 1, 16,
                 telemetry=info, output_final_state=True)
    else:
        output, final = call('chunk_key_gated_hyperbolic_delta_rule', sources, None, 1, 16,
                             telemetry=info, output_final_state=False)
        assert final is None and info['backend'] == 'host_reviewed_chunk_exception'
        compare_tensor('optional_output', output.flatten(), q.new_tensor([1., .5, .25]))
        dv, = torch.autograd.grad(output.sum(), v)
        compare_tensor('optional_dv', dv.flatten(), q.new_tensor([1.75, 0., 0.]), True)


@pytest.mark.parametrize('large_state', [False, True])
def test_chunk_exception_cotangent_span(large_state):
    q = torch.zeros((1, 1, 2, 1), dtype=torch.float64, device=device, requires_grad=True)
    k, v, ell = torch.ones_like(q), torch.zeros_like(q), torch.zeros_like(q)
    beta = q.new_zeros((1, 1, 2))
    state = q.new_tensor([0., 2.**600 if large_state else 1.]).reshape(1, 2, 1, 1)
    c, info = q.new_zeros(()), {}
    output, _ = call('chunk_key_gated_hyperbolic_delta_rule', (q, k, v, beta, ell, state, c),
                     None, 1, 16, telemetry=info)
    cot = q.new_tensor([2.**600, 2.**-600]).reshape_as(q)
    dq, = torch.autograd.grad(output, q, cot)
    expected = q.new_tensor([0., 1. if large_state else 2.**-600]).reshape_as(q)
    torch.testing.assert_close(dq, expected, atol=0, rtol=0)
    assert info['backward'] == 'host_reviewed_manual_chunk_source_vjp'


@pytest.mark.parametrize('case', ['margin', 'feedback', 'full_write', 'beta0', 'c0', 'nearest_margin', 'large_query'])
@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
def test_chunk_exception_high_precision(case, entry):
    import mpmath

    values = [1., 1., math.sqrt((1-1e-8)/.6), .5, -.01, .2, .6]
    if case == 'feedback':
        values[4], values[5] = 0., (1-1e-8)/math.sqrt(.6)
    if case == 'full_write':
        values[3] = 1.
    if case == 'beta0':
        values[3] = 0.
    if case == 'c0':
        values[1], values[2], values[6] = 2.**-101, .4*2.**-101, 0.
    if case == 'nearest_margin':
        values[2] = math.nextafter(math.sqrt(1/.6), 0.)
    if case == 'large_query':
        values[0], values[2], values[3] = 2.**201, .4, 1.
    shapes = [(1, 1, 1, 1)]*3 + [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), ()]
    sources = tuple(torch.tensor(x, dtype=torch.float64, device=device).reshape(shape).requires_grad_()
                    for x, shape in zip(values, shapes))
    truths = []
    for digits in (160, 240):
        mp = mpmath.mp.clone()
        mp.dps = digits
        inputs = tuple(mp.mpf(x.item()) for x in sources)

        def canonical(*xs):
            q, k, v, beta, ell, s, c = xs
            x = mp.exp(ell) * s
            margin = k*k-c*v*v
            residual = v-k*x
            sn = (x*margin+k*beta*residual)/(margin+c*v*beta*residual)
            return q*sn, sn

        out, sn = canonical(*inputs)
        derivatives = []
        for index in range(7):
            def loss(x):
                xs = inputs[:index] + (x,) + inputs[index+1:]
                y, s = canonical(*xs)
                return mp.mpf(.7)*y-mp.mpf(.3)*s
            derivatives.append(mp.diff(loss, inputs[index]))
        truths.append(torch.tensor([float(x) for x in (out, sn, *derivatives)], dtype=torch.float64, device=device))
    torch.testing.assert_close(truths[0], truths[1], atol=1e-70, rtol=1e-14)
    info = {}
    output, final = call(entry, sources, None, 1, 16, telemetry=info)
    gradients = torch.autograd.grad((output, final), sources, (torch.full_like(output, .7), torch.full_like(final, -.3)))
    actual = torch.stack([x.reshape(()) for x in (output, final, *gradients)])
    torch.testing.assert_close(actual, truths[1], atol=1e-9, rtol=1e-9)
    assert info['backend'] == 'host_reviewed_chunk_exception'
    assert info['exception_reason'] in ('source_exponent_range', 'raw_key_norm_range', 'feedback_margin', 'entry_ball_margin')


@pytest.mark.parametrize('carry', ['no_carry', 'y_carry'])
def test_chunk_nondefault_stream(carry):
    sources, cu = make_case(19, 32, 64, 32, torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compare_case('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, 16, carry)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.fixture(autouse=True)
def record_device_metrics(monkeypatch, request):
    import json
    import os
    from pathlib import Path

    destination = os.environ.get('KHDN_DEVICE_METRICS')
    if not destination:
        return
    original = compare_tensor

    def recorded(name, actual, expected, backward=False):
        a, r = actual.detach().double(), expected.detach().double()
        error = a - r
        row = dict(test=request.node.nodeid, tensor=name, backward=backward, dtype=str(actual.dtype),
                   elements=a.numel(), max_abs=error.abs().max().item() if a.numel() else 0.,
                   error_rms=error.square().mean().sqrt().item() if a.numel() else 0.,
                   reference_rms=r.square().mean().sqrt().item() if a.numel() else 0.)
        with Path(destination).open('a') as file:
            file.write(json.dumps(row) + '\n')
        original(name, actual, expected, backward)

    monkeypatch.setitem(globals(), 'compare_tensor', recorded)
