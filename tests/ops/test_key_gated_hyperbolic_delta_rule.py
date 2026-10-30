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


def record_conditioning(row):
    import json
    import os
    from pathlib import Path

    destination = os.environ.get('KHDN_CONDITIONING_DEVICE_EVIDENCE')
    if destination:
        with Path(destination).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


@pytest.mark.parametrize('family', ['query', 'state'])
@pytest.mark.parametrize('key', [1.1, 1.3, 3., 7.])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
def test_chunk_endpoint_conditioning(family, key, mode):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    xs = tuple(x.detach().to(device) for x in conditioning_fixture(family, key))
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs, mode, chunk_size=16)
    record_conditioning(row)
    assert info['backend'] == 'host_reviewed_chunk_exception'
    assert info['host_details']['extended_subblocks'] > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('family', ['near_c0', 'target', 'beta0', 'near_beta1'])
@pytest.mark.parametrize('streaming', [False, True])
def test_chunk_endpoint_conditioning_neighbors(dtype, family, streaming):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    xs = list(conditioning_fixture('state', key=1.7, dtype=dtype, length=3))
    with torch.no_grad():
        if family == 'near_c0':
            xs[5].fill_(2.**60)
            xs[6].fill_(2.**-124)
        elif family == 'target':
            xs[1].fill_(2.**-20)
            xs[2].fill_(.2)
            xs[5].fill_(.1)
        else:
            xs[5].fill_(2.**40)
            xs[3].fill_(0. if family == 'beta0' else 1-torch.finfo(dtype).eps)
    xs = tuple(x.detach().to(device) for x in xs)
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs, streaming=streaming, carry_mode='y_carry')
    record_conditioning(row)
    assert info.get('streaming_backends', [info['backend']])[0] == 'host_reviewed_chunk_exception'
    if streaming and family == 'near_beta1':
        assert info['streaming_backends'][-1] == 'triton_chunk'


@pytest.mark.parametrize('source', ['query', 'state'])
@pytest.mark.parametrize('outside', [False, True])
def test_chunk_endpoint_conditioning_boundary(source, outside):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    limit = (1e-10 / 8) / torch.finfo(torch.float64).eps
    xs = list(conditioning_fixture('state', dtype=torch.float64))
    with torch.no_grad():
        xs[5].fill_(.1)
        xs[0 if source == 'query' else 5].fill_((limit if source == 'query' else math.sqrt(limit))
                                                * (1 + (1e-6 if outside else -1e-6)))
    xs = tuple(x.detach().to(device) for x in xs)
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs)
    record_conditioning(row)
    assert (info['backend'] == 'host_reviewed_chunk_exception') == outside


@pytest.mark.parametrize('requested', [(0,), (3,), (4, 5, 6), tuple(range(7))])
@pytest.mark.parametrize('double_source', [0, 5, 6])
def test_chunk_endpoint_conditioning_selective(requested, double_source):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    xs = tuple(x.detach().to(device=device, dtype=torch.float64 if i == double_source else x.dtype)
               for i, x in enumerate(conditioning_fixture('state', length=3)))
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs, requested=requested, carry_mode='y_carry')
    record_conditioning(row)
    assert info['backend'] == 'host_reviewed_chunk_exception'


@pytest.mark.parametrize('group_size', [1, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_chunk_endpoint_conditioning_retained_witness(group_size, dtype):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import TOL32, TOL64, oracle, run, witness

    host = witness(group_size, dtype)
    reference_sources = tuple(x.detach().double().requires_grad_() for x in host)
    reference = run(oracle, reference_sources, group_size, scale=1.)
    xs = tuple(x.detach().to(device).requires_grad_() for x in host)
    info = {}
    result = run(ops.chunk_key_gated_hyperbolic_delta_rule, xs, group_size, chunk_size=32,
                 carry_mode='y_carry', scale=1., telemetry=info)
    cot = tuple(torch.full_like(x, .7) for x in result)
    got = (*result, *torch.autograd.grad(result, xs, cot))
    want = (*reference, *torch.autograd.grad(reference, reference_sources, tuple(x.cpu().double() for x in cot)))
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual.cpu().double(), expected, **(TOL32 if dtype == torch.float32 else TOL64))
    assert info['backend'] == 'host_reviewed_chunk_exception'
    parts = info['host_details']['partition']
    assert [p['stop']-p['start'] for p in parts] == [4]*8
    assert all(p['precision'] == 'torch.float64' for p in parts)
    record_conditioning(dict(label='retained_witness', group_size=group_size, dtype=str(dtype), telemetry=info))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_chunk_endpoint_conditioning_output_scale(dtype):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    xs = list(conditioning_fixture('query', dtype=dtype, length=3))
    with torch.no_grad():
        xs[0].fill_(1.)
    xs = tuple(x.detach().to(device) for x in xs)
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs, scale=2.**80, carry_mode='y_carry')
    record_conditioning(row)
    assert info['backend'] == 'host_reviewed_chunk_exception'


@pytest.mark.parametrize('late_query', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_chunk_endpoint_conditioning_late_readout(late_query, dtype):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_conditioning, conditioning_fixture

    xs = list(conditioning_fixture('state', key=1.3, dtype=dtype, length=5))
    with torch.no_grad():
        xs[3][:, 1:] = 0
        if late_query:
            xs[5].fill_(.1)
            xs[0][:, -1] = 2.**80
    xs = tuple(x.detach().to(device) for x in xs)
    info, row = check_conditioning(ops.chunk_key_gated_hyperbolic_delta_rule, xs, chunk_size=2, carry_mode='y_carry')
    record_conditioning(row)
    assert all(p['precision'] == 'mpmath' for p in info['host_details']['partition'])


@pytest.mark.parametrize('route', ['device', 'host'])
@pytest.mark.parametrize('power', [0, 40, 80])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
def test_chunk_cotangent_accuracy(route, power, mode):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_cotangent_accuracy, cotangent_fixture

    from fla.ops.key_gated_hyperbolic_delta_rule.chunk import _host_call

    xs = tuple(x.detach().to(device).requires_grad_() for x in cotangent_fixture())
    fn = ops.chunk_key_gated_hyperbolic_delta_rule
    if route == 'host':
        def fn(q, k, v, beta, ell, curvature, initial_state, telemetry, **kwargs):
            return _host_call((q, k, v, beta, ell, initial_state, curvature),
                              dict(bounds=[0, 1], scale=1., g=1, c=16, final=True, carry='no_carry'), telemetry)
    info = check_cotangent_accuracy(fn, xs, power, mode)
    record_conditioning(dict(label='cotangent_accuracy', route=route, power=power, mode=mode, telemetry=info))


@pytest.mark.parametrize('double_source', [None, 0, 6])
@pytest.mark.parametrize('requested', [(0,), (3,), (4, 5, 6), tuple(range(7))])
@pytest.mark.parametrize('final', [False, True])
def test_chunk_cotangent_requested(double_source, requested, final):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import cotangent_fixture, cotangent_general_compare

    xs = tuple(x.detach().to(device=device, dtype=torch.float64 if i == double_source else x.dtype)
               for i, x in enumerate(cotangent_fixture()))
    info = cotangent_general_compare(ops.chunk_key_gated_hyperbolic_delta_rule, xs, requested=requested, final=final)
    assert info['backward_host_details']['backward_reprepared']
    record_conditioning(dict(label='cotangent_requested', requested=requested, final=final, telemetry=info))


@pytest.mark.parametrize('power', [40, 80])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_chunk_cotangent_rebalance(power, dtype):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import check_cotangent_accuracy, cotangent_fixture

    for query_power, cot_power in [(0, power), (power, 0)]:
        xs = tuple(x.detach().to(device).requires_grad_() for x in cotangent_fixture(dtype, query_power))
        info = check_cotangent_accuracy(ops.chunk_key_gated_hyperbolic_delta_rule, xs, cot_power, 'output')
        if cot_power:
            assert info['backend'] == 'triton_chunk'
            assert info['backward_exception_reason'] == 'cotangent_accuracy'
            assert info['backward_host_details']['backward_reprepared']
        else:
            assert info['backend'] == 'host_reviewed_chunk_exception'
        record_conditioning(dict(label='cotangent_rebalance', query_power=query_power, cot_power=cot_power, telemetry=info))


@pytest.mark.parametrize('streaming', [False, True])
def test_chunk_cotangent_multichunk(streaming):
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import conditioning_fixture, cotangent_general_compare

    xs = list(conditioning_fixture('query', key=1.3, length=5))
    with torch.no_grad():
        xs[0].fill_(1.)
    xs = tuple(x.detach().to(device) for x in xs)
    info = cotangent_general_compare(ops.chunk_key_gated_hyperbolic_delta_rule, xs, streaming=streaming)
    assert info['backward_host_details']['backward_reprepared']
    record_conditioning(dict(label='cotangent_multichunk', streaming=streaming, telemetry=info))


def test_chunk_cotangent_grouped():
    from test_key_gated_hyperbolic_delta_rule_chunk_reference import cotangent_fixture, cotangent_general_compare

    xs = list(cotangent_fixture())
    xs[2] = torch.cat((xs[2], torch.zeros(1, 1, 1, 31)), -1)
    xs[5] = torch.cat((xs[5], torch.zeros(1, 1, 1, 31)), -1)
    xs[6] = xs[6] * 32
    xs = tuple(x.detach().to(device) for x in xs)
    info = cotangent_general_compare(ops.chunk_key_gated_hyperbolic_delta_rule, xs, group_size=32)
    assert info['backward_host_details']['backward_reprepared']
    record_conditioning(dict(label='cotangent_grouped', telemetry=info))


def test_chunk_cotangent_normal_long():
    sources, cu = make_case(2048, 128, 128, 32, torch.float32)
    compare_case('chunk_key_gated_hyperbolic_delta_rule', sources, cu, 32, 32, 'y_carry')


@pytest.mark.parametrize('carry', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('length,exponent,magnitude,sign,decay', [
    (16, 13, 1., 1., 0.), (16, 10, .25, 1., 0.), (16, 12, 1., -1., 0.),
    (16, 13, 4., 1., 0.), (16, 14, -4., -1., 0.), (16, 15, 1., 1., 0.),
    (16, 13, 1., 1., -.02), (35, 13, 1., 1., -.02),
])
def test_chunk_mixed_readout_cancellation(carry, length, exponent, magnitude, sign, decay):
    q = torch.zeros((1, length, 1, 16), device=device)
    q[..., 0], q[..., 1] = magnitude * (1 + 2.**-exponent), magnitude
    k = torch.zeros_like(q)
    k[..., 15] = 1.
    v = torch.zeros_like(q)
    beta = torch.zeros((1, length, 1), device=device)
    ell = torch.full_like(q, decay)
    state = torch.zeros((1, 1, 16, 16), device=device)
    state[..., 0, :], state[..., 1, :] = sign * (1 - 2.**-exponent), -sign
    sources = tuple(x.requires_grad_() for x in (q, k, v, beta, ell, state, q.new_zeros(())))
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    options = dict(group_size=16, scale=1., output_final_state=True)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6], **options)
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], initial_state=sources[5], curvature=sources[6], chunk_size=16,
        carry_mode=carry, precision_policy='mixed_readout', telemetry=info, **options)
    assert info['precision_route'] == 'experimental_mixed_readout'
    cot = (torch.ones_like(actual[0]), torch.ones_like(actual[1]))
    grads = torch.autograd.grad(actual, sources, cot)
    expected_grads = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, e in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, e)
    for name, a, e in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), grads, expected_grads):
        compare_tensor(name, a, e, backward=True)


@pytest.mark.parametrize('chunk', [16, 32, 64])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('carry', ['no_carry', 'y_carry'])
def test_chunk_mixed_readout_sources(chunk, dtype, carry):
    sources, cu = make_case(2 * chunk + 3, 128, 128, 32, dtype)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6], group_size=32, output_final_state=True)
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], initial_state=sources[5], curvature=sources[6], group_size=32,
        chunk_size=chunk, carry_mode=carry, output_final_state=True, precision_policy='mixed_readout', telemetry=info)
    assert info['precision_route'] == 'experimental_mixed_readout'
    torch.manual_seed(20261026)
    cot = (torch.randn_like(sources[2]), torch.randn_like(sources[5]))
    grads = torch.autograd.grad(actual, sources, cot)
    expected_grads = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, e in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, e)
    for name, a, e in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), grads, expected_grads):
        compare_tensor(name, a, e, backward=True)


@pytest.mark.parametrize('policy,dtype,chunk,magnitude,reason', [
    ('strict', torch.float32, 16, 1., 'strict_requested'),
    ('mixed_readout', torch.float32, 8, 1., 'unqualified_geometry'),
    ('mixed_readout', torch.float16, 16, 1., 'unqualified_source_dtype'),
    ('mixed_readout', torch.float32, 16, 9., 'unqualified_source_magnitude'),
])
def test_chunk_mixed_readout_strict_routes(policy, dtype, chunk, magnitude, reason):
    sources, cu = make_case(19, 128, 128, 32, dtype)
    if magnitude != 1.:
        with torch.no_grad():
            sources[0][..., 0] = magnitude
    info = {}
    options = dict(curvature=sources[6], initial_state=sources[5], group_size=32,
                   output_final_state=True, chunk_size=chunk)
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **options, precision_policy=policy, telemetry=info)
    strict = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **options)
    assert info['precision_route'] == 'strict' and info['qualification_reason'] == reason
    for a, e in zip(actual, strict):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    cot = tuple(torch.ones_like(x) for x in actual)
    actual_grads = torch.autograd.grad(actual, sources, cot)
    strict_grads = torch.autograd.grad(strict, sources, cot)
    for a, e in zip(actual_grads, strict_grads):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize('scale,reason', [
    (2., None), (-2., None),
    (math.nextafter(2., math.inf), 'unqualified_effective_query_magnitude'),
    (-math.nextafter(2., math.inf), 'unqualified_effective_query_magnitude'),
])
def test_chunk_mixed_readout_scaled_query_profile(scale, reason):
    import importlib
    chunk_module = importlib.import_module('fla.ops.key_gated_hyperbolic_delta_rule.chunk')
    sources, cu = make_case(19, 128, 128, 32, torch.float32)
    with torch.no_grad():
        sources[0].fill_(4.)
    options = dict(c=16, g=32, bounds=[0, 19, 38], scale=scale)
    assert chunk_module._qualification_reason(sources, options) == reason


def test_chunk_mixed_readout_reused_telemetry_reports_executed_routes():
    sources, _ = make_case(19, 128, 128, 32, torch.float32)
    info = {'caller_tag': 'preserved'}
    options = dict(curvature=sources[6], initial_state=sources[5], group_size=32,
                   output_final_state=True, chunk_size=16, telemetry=info)
    for policy, route in [('strict', 'strict'), ('mixed_readout', 'experimental_mixed_readout'), ('strict', 'strict')]:
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **options, precision_policy=policy)
        assert info['precision_requested'] == policy and info['precision_route'] == route
        assert info['caller_tag'] == 'preserved'
        assert 'backward_precision_route' not in info and 'exception_reason' not in info
        if policy == 'strict':
            assert info['qualification_reason'] == 'strict_requested'
            assert 'qualification' not in info and 'contraction_precision' not in info
        else:
            assert 'qualification_reason' not in info and info['precision_eligible']
        torch.autograd.grad(actual, sources, tuple(torch.ones_like(x) for x in actual))
        assert info['backward_precision_route'] == route


def test_chunk_mixed_readout_host_exception_overrides_candidate_telemetry():
    q = torch.ones(1, 1, 1, 16, device=device) * .1
    k = torch.zeros_like(q)
    k[..., 0] = 1.
    v = torch.zeros_like(q)
    v[..., 0] = math.sqrt(4.8)
    beta = torch.full((1, 1, 1), .5, device=device)
    ell = torch.zeros_like(q)
    state = torch.zeros(1, 1, 16, 16, device=device)
    curve = q.new_tensor(1.)
    sources = tuple(x.requires_grad_() for x in (q, k, v, beta, ell, state, curve))
    info = {'qualification': 'old', 'contraction_precision': 'old', 'backward_precision_route': 'old'}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=curve, initial_state=state, group_size=16, chunk_size=16,
        output_final_state=True, precision_policy='mixed_readout', telemetry=info,
    )
    assert info['precision_requested'] == 'mixed_readout' and info['precision_eligible']
    assert info['precision_route'] == 'host_exception' and info['exception_reason'] == 'feedback_margin'
    assert info['tensor_cores'] is False and info['backend'] == 'host_reviewed_chunk_exception'
    assert 'qualification' not in info and 'contraction_precision' not in info
    assert 'backward_precision_route' not in info
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=16, output_final_state=True,
    )
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    cot = tuple(torch.ones_like(x) for x in actual)
    gradients = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, r in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), gradients, truth):
        compare_tensor(name, a, r, backward=True)
    assert info['host_details'].get('backward_precision')
    assert info['precision_route'] == 'host_exception'


def test_recurrent_telemetry_adapter_clears_owned_fields_on_reuse():
    sources, cu = make_case(3, 32, 32, 32, torch.float32)
    info = {'qualification': 'old', 'exception_reason': 'old', 'backward_precision_route': 'old'}
    actual = call('fused_recurrent_key_gated_hyperbolic_delta_rule', sources, cu, 32, 16, telemetry=info)
    assert info['precision_route'] == 'strict' and info['backend'] == 'triton_recurrent'
    assert 'qualification' not in info and 'exception_reason' not in info
    assert 'backward_precision_route' not in info
    torch.autograd.grad(actual, sources, tuple(torch.ones_like(x) for x in actual))
    assert info['backward_precision_route'] == 'strict'


@pytest.mark.parametrize('length,chunk,carry,amplitude,route,profile', [
    (35, 16, 'no_carry', 8., 'experimental_mixed_readout', 'empirical_profile_v1'),
    (35, 16, 'no_carry', 128., 'strict_device_replay', 'unqualified'),
    (16, 16, 'no_carry', 128., 'experimental_mixed_readout', 'model_adjoint_v1'),
    (32, 16, 'no_carry', 32., 'experimental_mixed_readout', 'model_joint_adjoint_v1'),
    (32, 16, 'no_carry', 32.000003814697266, 'strict_device_replay', 'unqualified'),
    (35, 16, 'no_carry', 21.333332061767578, 'experimental_mixed_readout', 'model_joint_adjoint_v1'),
    (35, 16, 'no_carry', 21.33333396911621, 'strict_device_replay', 'unqualified'),
    (35, 16, 'no_carry', 128.00001525878906, 'strict_device_replay', 'unqualified'),
    (36, 16, 'no_carry', 128., 'strict_device_replay', 'unqualified'),
    (35, 32, 'no_carry', 128., 'strict_device_replay', 'unqualified'),
    (35, 16, 'y_carry', 128., 'strict_device_replay', 'unqualified'),
])
def test_chunk_mixed_readout_adjoint_profile(length, chunk, carry, amplitude, route, profile):
    sources, cu = make_case(length, 128, 128, 32, torch.float32)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32, output_final_state=True,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32, output_final_state=True,
        chunk_size=chunk, carry_mode=carry, precision_policy='mixed_readout', telemetry=info,
    )
    cot = tuple(torch.ones_like(x) * amplitude for x in actual)
    gradients = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    for name, a, r in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), gradients, truth):
        compare_tensor(name, a, r, backward=True)
    assert info['precision_route'] == 'experimental_mixed_readout'
    assert info['backward_precision_route'] == route and info['backward_adjoint_profile'] == profile
    assert info['backward_output_cotangent_max'] == info['backward_state_cotangent_max'] == amplitude


@pytest.mark.parametrize('curve,chunk,carry,packed,dtype,route', [
    (1.0000001192092896, 16, 'no_carry', False, torch.float32, 'experimental_mixed_readout'),
    (1.5, 16, 'no_carry', True, torch.float32, 'experimental_mixed_readout'),
    (2., 16, 'no_carry', False, torch.float32, 'experimental_mixed_readout'),
    (2., 16, 'no_carry', True, torch.float32, 'experimental_mixed_readout'),
    (2., 16, 'no_carry', False, torch.bfloat16, 'experimental_mixed_readout'),
    (2., 16, 'no_carry', True, torch.bfloat16, 'experimental_mixed_readout'),
    (2.000000238418579, 16, 'no_carry', False, torch.float32, 'strict'),
    (2., 32, 'no_carry', False, torch.float32, 'strict'),
    (2., 64, 'no_carry', False, torch.float32, 'strict'),
    (2., 16, 'y_carry', False, torch.float32, 'strict'),
])
def test_chunk_mixed_readout_model_curvature(curve, chunk, carry, packed, dtype, route):
    sources, cu = make_case(32 if packed else 16, 128, 128, 32, dtype, packed)
    with torch.no_grad():
        sources[6].fill_(curve)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32,
        cu_seqlens=cu, output_final_state=True,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32,
        cu_seqlens=cu, chunk_size=chunk, carry_mode=carry, output_final_state=True,
        precision_policy='mixed_readout', telemetry=info,
    )
    cot = tuple(torch.ones_like(x) * 128. for x in actual)
    gradients = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    for name, a, r in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), gradients, truth):
        compare_tensor(name, a, r, backward=True)
    assert info['precision_route'] == info['backward_precision_route'] == route
    if route == 'experimental_mixed_readout':
        assert info['qualification'] == 'model_curvature_v1'
        assert info['backward_adjoint_profile'] == 'model_adjoint_v1'
    else:
        assert info['qualification_reason'] == (
            'unqualified_curvature' if curve > 2. else 'unqualified_curvature_geometry')
        assert 'qualification' not in info


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_chunk_mixed_readout_joint_adjoint_counterexample(dtype):
    sources, _ = make_case(35, 128, 128, 32, dtype)
    with torch.no_grad():
        sources[6].fill_(1.)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32,
        output_final_state=True, scale=1.,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32,
        chunk_size=16, carry_mode='no_carry', output_final_state=True, scale=1.,
        precision_policy='mixed_readout', telemetry=info,
    )
    cot = tuple((torch.cos(torch.arange(x.numel(), dtype=torch.float64, device=x.device)).reshape(x.shape) * 128.).to(x)
                for x in actual)
    gradients = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    for name, a, r in zip(('dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), gradients, truth):
        compare_tensor(name, a, r, backward=True)
    assert info['precision_route'] == 'experimental_mixed_readout'
    assert info['backward_precision_route'] == 'strict_device_replay'
    assert info['backward_qualification_reason'] == 'unqualified_joint_chunk_adjoint'
    assert info['backward_adjoint_chunks'] == 3 and info['backward_adjoint_load'] == 768.
    assert info['strict_replay_details']['precision_route'] == 'strict'
    assert not info['strict_replay_details']['tensor_cores']


@pytest.mark.parametrize('amplitude', [128., 128.00001525878906])
@pytest.mark.parametrize('case', ['state_source', 'curvature_source', 'alias', 'state_only_q', 'no_final_curvature'])
def test_chunk_mixed_readout_single_chunk_masks(case, amplitude):
    sources, _ = make_case(16, 128, 128, 32, torch.float32)
    sources = list(sources)
    with torch.no_grad():
        sources[6].fill_(2.)
    if case == 'alias':
        sources[0] = sources[1]
    requested = [0] if case in ('alias', 'state_only_q') else [5] if case == 'state_source' else [6]
    for source in sources:
        source.requires_grad_(False)
    for i in requested:
        sources[i].requires_grad_(True)
    reference = [x.detach().double().requires_grad_(i in requested) for i, x in enumerate(sources)]
    if case == 'alias':
        reference[1] = reference[0]
    final = case != 'no_final_curvature'
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32,
        output_final_state=final, scale=1.,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32,
        chunk_size=16, output_final_state=final, scale=1., precision_policy='mixed_readout', telemetry=info,
    )
    which = (1,) if case == 'state_only_q' else (0,) if not final else (0, 1)
    cot = tuple(torch.ones_like(actual[i]) * amplitude for i in which)
    gradients = torch.autograd.grad(tuple(actual[i] for i in which), [sources[i] for i in requested], cot)
    truth = torch.autograd.grad(tuple(expected[i] for i in which), [reference[i] for i in requested],
                                tuple(x.double() for x in cot))
    for name, a, r in zip(('output', 'state'), actual, expected):
        if a is not None:
            compare_tensor(name, a, r)
    for i, a, r in zip(requested, gradients, truth):
        compare_tensor(str(i), a, r, backward=True)
    if case == 'state_only_q':
        assert torch.count_nonzero(gradients[0]) == 0
    assert info['precision_route'] == 'experimental_mixed_readout'
    assert info['backward_precision_route'] == ('experimental_mixed_readout' if amplitude <= 128. else 'strict_device_replay')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('amplitude', [128., 129.])
@pytest.mark.parametrize('kind', ['same', 'nonleaf', 'noncontiguous', 'independent'])
def test_chunk_source_alias_gradient(dtype, amplitude, kind):
    sources, _ = make_case(16, 128, 128, 32, dtype)
    sources = [x.detach() for x in sources]
    sources[6].fill_(2.)
    leaf = sources[1].requires_grad_()
    shared = leaf * 1. if kind == 'nonleaf' else leaf
    if kind == 'noncontiguous':
        shared = leaf.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not shared.is_contiguous()
    sources[0] = sources[1] = shared
    targets = [leaf]
    if kind == 'independent':
        sources[0] = shared.detach().requires_grad_()
        sources[1] = shared.detach().requires_grad_()
        assert sources[0] is not sources[1] and sources[0].data_ptr() == sources[1].data_ptr()
        targets = sources[:2]
    reference = [x.detach().double().requires_grad_(i in (0, 1)) for i, x in enumerate(sources)]
    if kind != 'independent':
        reference[1] = reference[0]
    ref_targets = reference[:2] if kind == 'independent' else reference[:1]
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32,
        scale=1., output_final_state=True,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32,
        scale=1., chunk_size=16, output_final_state=True, precision_policy='mixed_readout', telemetry=info,
    )
    cot = tuple((torch.cos(torch.arange(x.numel(), dtype=torch.float64)).reshape(x.shape) * amplitude).to(x) for x in actual)
    gradients = torch.autograd.grad(actual, targets, cot)
    truth = torch.autograd.grad(expected, ref_targets, tuple(x.double() for x in cot))
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    for a, r in zip(gradients, truth):
        compare_tensor('input_edge', a, r, backward=True)
    assert info['source_alias_groups'] == ([] if kind == 'independent' else [[0, 1]])
    if kind != 'independent':
        assert info['source_gradient_aggregation'] == 'fp64_before_public_cast'
    assert info['precision_route'] == 'experimental_mixed_readout'
    assert info['backward_precision_route'] == ('experimental_mixed_readout' if amplitude <= 128. else 'strict_device_replay')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
@pytest.mark.parametrize('nonleaf', [False, True])
def test_source_alias_host_cotangent_cancellation(dtype, entry, nonleaf):
    leaf = torch.ones(1, 1, 1, 1, dtype=dtype, device=device, requires_grad=True)
    shared = leaf * 1. if nonleaf else leaf
    value = torch.full_like(shared, .2)
    beta = torch.full((1, 1, 1), .5, dtype=torch.float32, device=device)
    ell = torch.zeros_like(shared, dtype=torch.float32)
    state = torch.full_like(shared, 2.**-100, dtype=torch.float32)
    curve = torch.zeros((), dtype=torch.float32, device=device)
    info = {}
    extra = dict(chunk_size=1) if entry.startswith('chunk') else dict(checkpoint_size=1)
    output, final = getattr(ops, entry)(
        shared, shared, value, beta, ell, initial_state=state, curvature=curve, scale=1., group_size=1,
        output_final_state=False, telemetry=info, **extra,
    )
    gradient, = torch.autograd.grad(output, leaf, torch.full_like(output, 2.**100))
    assert final is None
    torch.testing.assert_close(gradient, torch.full_like(gradient, .5), atol=0, rtol=0)
    assert info['backward_precision_route'] == 'host_exception'
    assert info['backward_exception_reason'] == 'cotangent_accuracy'
    assert info['backward_host_details']['source_gradient_aggregation'] == 'mpmath_before_public_cast'
    assert info['backward_host_details']['source_alias_groups'] == [[0, 1]]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('policy,amplitude,route', [
    ('mixed_readout', 128., 'experimental_mixed_readout'),
    ('mixed_readout', 129., 'strict_device_replay'),
    ('strict', 129., 'strict'),
])
def test_source_alias_repeated_grad_request(dtype, policy, amplitude, route):
    sources, _ = make_case(16, 128, 128, 32, dtype)
    sources = list(sources)
    with torch.no_grad():
        sources[6].fill_(2.)
    sources[0] = sources[1]
    reference = [x.detach().double().requires_grad_() for x in sources]
    reference[0] = reference[1]
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], group_size=32,
        scale=1., output_final_state=True,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], group_size=32,
        scale=1., chunk_size=16, output_final_state=True, precision_policy=policy, telemetry=info,
    )
    cot = tuple((torch.cos(torch.arange(x.numel(), dtype=torch.float64)).reshape(x.shape) * amplitude).to(x)
                for x in actual)
    gradients = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cot))
    assert torch.equal(gradients[0], gradients[1])
    for name, a, r in zip(('output', 'state'), actual, expected):
        compare_tensor(name, a, r)
    for name, a, r, source in zip(('dshared0', 'dshared1', 'dv', 'dbeta', 'dell', 'dS0', 'dc'), gradients, truth, sources):
        assert a.dtype == source.dtype
        compare_tensor(name, a, r, backward=True)
    assert info['source_alias_groups'] == [[0, 1]]
    assert info['source_gradient_aggregation'] == 'fp64_before_public_cast'
    assert info['precision_route'] == ('strict' if policy == 'strict' else 'experimental_mixed_readout')
    assert info['backward_precision_route'] == route
