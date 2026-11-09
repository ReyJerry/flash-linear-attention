# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Public no-replay BF16 policy against independently rounded-source dense truth."""

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

pytest.importorskip('triton')

from test_key_gated_hyperbolic_delta_rule import make_case, ops  # noqa: E402

from fla.ops.key_gated_hyperbolic_delta_rule import bf16  # noqa: E402
from fla.utils import IS_NVIDIA, device  # noqa: E402

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='BF16 KHDN currently requires the qualified NVIDIA stack.')


class NoFP64(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        assert not any(isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in tree_leaves((args, kwargs)))
        result = function(*args, **kwargs)
        assert not any(isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in tree_leaves(result))
        return result


@pytest.mark.parametrize('bounds', [(0, 257), (0, 0, 129, 257, 257), (0, 0, 0), tuple(range(0, 16385, 2048))])
@pytest.mark.usefixtures('qualified_bf16')
def test_c128_chunk_metadata_cache_inference_stream_and_eviction(bounds):
    from fla.ops.key_gated_hyperbolic_delta_rule import chunk128

    target = torch.device('cuda', torch.cuda.current_device())
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    chunk128._make_chunk_metadata.cache_clear()
    with torch.cuda.stream(producer), torch.inference_mode():
        metadata, count = chunk128.chunk_metadata(target, bounds, 128)
    assert all(not torch.is_inference(value) for value in metadata)
    with torch.cuda.stream(consumer):
        reused, repeated_count = chunk128.chunk_metadata(target, bounds, 128)
        assert count == repeated_count and all(a is b for a, b in zip(metadata, reused))
        values = torch.ones_like(reused[0], dtype=torch.float32, requires_grad=True)
        (values*reused[0]).sum().backward()
        snapshots = [value.clone() for value in reused]
    with torch.cuda.stream(producer):
        for length in range(1, 21):
            chunk128.chunk_metadata(target, (0, length), 128)
    consumer.synchronize()
    starts = [position for start, end in zip(bounds, bounds[1:]) for position in range(start, end, 128)]
    assert snapshots[0].cpu().tolist() == starts
    assert values.grad.cpu().tolist() == starts
    assert count == len(starts) and chunk128._make_chunk_metadata.cache_info().currsize == 16
    for previous, following in zip(snapshots[0].cpu().tolist(), snapshots[1].cpu().tolist()):
        assert 0 < following-previous <= 128
    torch.cuda.synchronize()


@pytest.fixture(scope='module')
def qualified_bf16():
    try:
        bf16._require_runtime(torch.zeros((), device=device))
    except ValueError as error:
        pytest.skip(str(error))


@pytest.mark.usefixtures('qualified_bf16')
def test_fixed_curvature_cache_lifetime_and_tensor_passthrough():
    from fla.ops.key_gated_hyperbolic_delta_rule.fixed_curvature import _constant, curvature_tensor

    _constant.cache_clear()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer), torch.inference_mode():
        initial = curvature_tensor(.6, device)
    assert not torch.is_inference(initial) and initial.dtype == torch.float32
    with torch.cuda.stream(consumer):
        reused = curvature_tensor(.6, device)
        assert reused is initial
        source = torch.ones((), device=device, requires_grad=True)
        (source * reused).backward()
        snapshot = source.grad.clone()
    for index in range(20):
        curvature_tensor(.01 * index, device)
    consumer.synchronize()
    torch.testing.assert_close(snapshot, torch.tensor(.6, dtype=torch.float32, device=device), atol=0, rtol=0)
    assert _constant.cache_info().currsize == 16
    current = curvature_tensor(.6, device)
    current.fill_(9.)
    restored = curvature_tensor(.6, device)
    assert restored is not current
    torch.testing.assert_close(restored, torch.tensor(.6, dtype=torch.float32, device=device), atol=0, rtol=0)
    trainable = restored.detach().clone().requires_grad_()
    assert curvature_tensor(trainable, device) is trainable
    restored.requires_grad_(True)
    assert curvature_tensor(.6, device) is not restored


def compare_bf16(actual, expected, backward=False):
    atol, rtol, rms = (.005, .03, .015) if backward else (.002, .02, .01)
    assert torch.isfinite(actual).all()
    a, b = actual.double(), expected.double()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    if a.numel():
        assert (a-b).square().mean().sqrt() <= rms * b.square().mean().sqrt().clamp_min(atol)


@pytest.mark.parametrize('chunk_size', [64, 128])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_public_policy_source_vjp(chunk_size, packed, dtype):
    source, cu = make_case(129, 128, 128, 32, packed=packed, dtype=dtype)
    reference = tuple(value.detach().double().requires_grad_() for value in source)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6],
        group_size=32, cu_seqlens=cu, output_final_state=True,
    )
    cotangents = tuple(torch.sin(torch.arange(x.numel(), device=x.device)).reshape(x.shape).float()
                       for x in expected)
    reference_gradients = torch.autograd.grad(expected, reference, tuple(x.double() for x in cotangents))
    telemetry = {}
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *source[:5], initial_state=source[5], curvature=source[6], group_size=32,
            cu_seqlens=cu, output_final_state=True, chunk_size=chunk_size,
            precision_policy='bf16_selective', telemetry=telemetry,
        )
        gradients = torch.autograd.grad(actual, source, cotangents, retain_graph=True)
        repeated = torch.autograd.grad(actual, source, cotangents)
    assert telemetry['precision_requested'] == 'bf16_selective'
    assert telemetry['arithmetic'] == 'selective_bf16_adjoint_queryless'
    assert telemetry['host_fallback'] is False and telemetry['native_kernel_calls'] > 0
    assert all(torch.equal(a, b) for a, b in zip(gradients, repeated))
    for index, (actual_value, expected_value) in enumerate(zip(
            (*actual, *gradients), (*expected, *reference_gradients))):
        compare_bf16(actual_value, expected_value, backward=index >= 2)


@pytest.mark.parametrize('source_vjp', ['coupled', 'coupled_state', 'coupled_fused'])
@pytest.mark.parametrize('case,dtype,packed', [
    ('random', torch.float32, False), ('random', torch.bfloat16, False),
    ('random', torch.float32, True), ('random', torch.bfloat16, True),
    ('fixed_point', torch.float32, False), ('repeated_beta1', torch.float32, False),
    ('strong_gate', torch.float32, False), ('raw_margin', torch.float32, False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_coupled_source_vjp(case, dtype, packed, source_vjp):
    if case == 'random':
        source, cu = make_case(129, 128, 128, 32, packed=packed, dtype=dtype)
        source = tuple(x.detach().requires_grad_(i < 6) for i, x in enumerate(source))
    else:
        length = 2 if case == 'strong_gate' else 1 if case == 'raw_margin' else 65
        q = torch.zeros(1, length, 1, 128, device=device)
        q[..., 0] = 1
        k = q.clone()
        v = torch.zeros(1, length, 1, 32, device=device)
        v[..., 0] = .5
        if case == 'raw_margin':
            v.fill_(.9998779296875)
        beta = q.new_full((1, length, 1), 1. if case == 'repeated_beta1' else .5)
        ell = torch.zeros_like(q)
        state = q.new_zeros(1, 1, 128, 32)
        if case == 'fixed_point':
            state[..., 0, 0] = .5
        if case == 'strong_gate':
            beta[:, 0], beta[:, 1] = 1, 0
            ell[:, 0], ell[:, 1] = -1e8, -1.
        source = tuple(x.requires_grad_() for x in (q, k, v, beta, ell, state)) + (q.new_tensor(1.),)
        cu = None
    reference = tuple(x.detach().double().requires_grad_(i < 6) for i, x in enumerate(source))
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6], group_size=32,
        cu_seqlens=cu, output_final_state=True,
    )
    cot = tuple(torch.sin(torch.arange(x.numel(), device=x.device)).reshape(x.shape).float() for x in expected)
    reference_gradients = torch.autograd.grad(expected, reference[:6], tuple(x.double() for x in cot))
    info = {}
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *source[:5], curvature=source[6], initial_state=source[5], cu_seqlens=cu,
            output_final_state=True, chunk_size=64, source_vjp=source_vjp, telemetry=info, precision_policy='bf16_selective',
        )
        gradients = torch.autograd.grad(actual, source[:6], cot, retain_graph=True)
        repeated = torch.autograd.grad(actual, source[:6], cot)
    assert info['source_vjp'] == source_vjp and info['host_fallback'] is False
    assert all(torch.equal(a, b) for a, b in zip(gradients, repeated))
    for index, (a, b) in enumerate(zip((*actual, *gradients), (*expected, *reference_gradients))):
        compare_bf16(a, b, backward=index >= 2)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.usefixtures('qualified_bf16')
def test_c128_interval_preparation_tail_sources(packed):
    torch.manual_seed(702 + int(packed))
    chunk, length = 128, 257
    shape = (1, length, 1, 128)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1)
    v = .025 * torch.randn(shape)
    beta = torch.sigmoid(torch.randn(1, length, 1))
    beta[:, 0], beta[:, 1] = 0, 1
    ell = -.02 - .08 * torch.rand(shape)
    bounds = torch.tensor([0, 0, chunk+1, length, length], dtype=torch.int64) if packed else None
    state = .001 * torch.randn(4 if packed else 1, 1, 128, 128)
    curvature = torch.tensor(.5)
    cpu = (q, k, v, beta, ell, state, curvature)
    reference = tuple(x.double().requires_grad_() for x in cpu)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6], group_size=32,
        output_final_state=True, cu_seqlens=bounds,
    )
    sources = tuple(x.to(device).requires_grad_() for x in cpu)
    cu = None if bounds is None else bounds.to(device)
    cot = [.03 * torch.randn(x.shape, device=device, dtype=torch.float32) for x in expected]
    truth = torch.autograd.grad(expected, reference, [x.cpu().double() for x in cot])
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], initial_state=sources[5], curvature=sources[6], group_size=32,
            chunk_size=chunk, output_final_state=True, cu_seqlens=cu, precision_policy='bf16',
        )
        gradients = torch.autograd.grad(actual, sources, cot)
    for i, (a, b) in enumerate(zip((*actual, *gradients), (*expected, *truth))):
        compare_bf16(a.cpu(), b, backward=i >= 2)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('chunk,carry,length', [
    (16, 'no_carry', 35), (32, 'no_carry', 35), (16, 'y_carry', 35), (128, 'no_carry', 129),
])
@pytest.mark.parametrize('cotangent', ['output', 'state', 'joint'])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_chunk_sources(dtype, packed, chunk, carry, length, cotangent):
    sources, cu = make_case(length, 128, 128, 32, dtype, packed=packed)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], output_final_state=True,
        cu_seqlens=cu, group_size=32,
    )
    info = {}
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=True,
            cu_seqlens=cu, group_size=32, chunk_size=chunk, carry_mode=carry, precision_policy='bf16', telemetry=info,
        )
        cot = [torch.randn_like(x) for x in actual]
        if cotangent == 'output':
            cot[1].zero_()
        if cotangent == 'state':
            cot[0].zero_()
        observed = torch.autograd.grad(actual, sources, cot, retain_graph=True)
        repeated = torch.autograd.grad(actual, sources, cot)
    for first, second in zip(observed, repeated):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    truth = torch.autograd.grad(expected, reference, [x.double() for x in cot])
    for a, b in zip(actual, expected):
        compare_bf16(a, b)
    for a, b in zip(observed, truth):
        compare_bf16(a, b, backward=True)
    assert actual[1].dtype == torch.float32
    assert info['precision_route'] == info['backward_precision_route'] == 'bf16'
    assert info['fallback_subblocks'] == info['backward_fallback_subblocks'] == 0
    assert info['backward_precision'] == 'device_float32' and not info['host_fallback']


@pytest.mark.parametrize('carry', ['no_carry', 'y_carry'])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_longer_chunk_source_gradient_precision(carry):
    # this prospective seed exposes the C32 key-gradient error from single-product output VJPs
    sources, cu = make_case(67, 128, 128, 32, torch.float32, seed=1009)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], output_final_state=True,
        cu_seqlens=cu, group_size=32,
    )
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=True,
            cu_seqlens=cu, group_size=32, chunk_size=32, carry_mode=carry, precision_policy='bf16',
        )
        torch.manual_seed(901009)
        cotangents = tuple(torch.randn_like(x) for x in actual)
        gradients = torch.autograd.grad(actual, sources, cotangents)
    truth = torch.autograd.grad(expected, reference, tuple(x.double() for x in cotangents))
    for a, b in zip(actual, expected):
        compare_bf16(a, b)
    for a, b in zip(gradients, truth):
        compare_bf16(a, b, backward=True)


@pytest.mark.parametrize('length', [0, 1, 17])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_recurrent_sources(length, packed):
    sources, cu = make_case(length, 128, 128, 32, torch.float32, packed=packed)
    reference = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], curvature=reference[6], initial_state=reference[5], output_final_state=True,
        cu_seqlens=cu, group_size=32,
    )
    info = {}
    with NoFP64():
        actual = ops.fused_recurrent_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=True,
            cu_seqlens=cu, group_size=32, precision_policy='bf16', telemetry=info,
        )
        cot = [torch.randn_like(x) for x in actual]
        observed = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, [x.double() for x in cot])
    for a, b in zip((*actual, *observed), (*expected, *truth)):
        compare_bf16(a, b, backward=True)
    assert info['backend'] == 'triton_recurrent' and info['backward_precision_route'] == 'bf16'


@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_rejects_fp64_without_strict_replay(monkeypatch):
    from fla.ops.key_gated_hyperbolic_delta_rule import chunk
    sources, _ = make_case(17, 128, 128, 32, torch.float64)

    def forbidden(*args, **kwargs):
        raise AssertionError('strict replay or host fallback invoked')

    monkeypatch.setattr(chunk, '_host_call', forbidden)
    monkeypatch.setattr(chunk, '_strict_device_replay', forbidden)
    with pytest.raises(ValueError, match='FP64'):
        ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], precision_policy='bf16', chunk_size=16,
        )


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['same', 'nonleaf', 'noncontiguous', 'independent'])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_alias_retained_graph(dtype, kind):
    sources, _ = make_case(17, 128, 128, 32, dtype)
    sources = [x.detach() for x in sources]
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
        *reference[:5], curvature=reference[6], initial_state=reference[5], output_final_state=True, group_size=32,
    )
    info = {}
    with NoFP64():
        actual = ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=True,
            group_size=32, chunk_size=16, precision_policy='bf16', telemetry=info,
        )
    for a, b in zip(actual, expected):
        compare_bf16(a, b)
    for iteration in range(3):
        with NoFP64():
            cot = [torch.randn_like(x) for x in actual]
            observed = torch.autograd.grad(actual, targets, cot, retain_graph=iteration < 2)
        truth = torch.autograd.grad(expected, ref_targets, [x.double() for x in cot], retain_graph=iteration < 2)
        for a, b in zip(observed, truth):
            compare_bf16(a, b, backward=True)
    assert info['source_alias_groups'] == ([] if kind == 'independent' else [[0, 1]])
    assert info['backward_precision_route'] == 'bf16' and info['backward_fallback_subblocks'] == 0


@pytest.mark.parametrize('key_dim,group_size', [(1, 1), (128, 32)])
@pytest.mark.parametrize('chunk,carry', [(1, 'no_carry'), (2, 'no_carry'), (16, 'no_carry'), (2, 'y_carry')])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_strong_decay(key_dim, group_size, chunk, carry):
    q = torch.full((1, 2, 1, key_dim), key_dim**-.5, device=device)
    k = q.clone()
    v = torch.zeros(1, 2, 1, group_size, device=device)
    v[0, :, 0, 0] = q.new_tensor([.5, .1])
    beta = q.new_tensor([1., 0.]).reshape(1, 2, 1)
    ell = q.new_tensor([-1e8, -1.]).reshape(1, 2, 1, 1).expand_as(q).clone().requires_grad_()
    state = torch.zeros(1, 1, key_dim, group_size, device=device)
    info = {}
    with NoFP64():
        output, final = ops.chunk_key_gated_hyperbolic_delta_rule(
            q, k, v, beta, ell, curvature=.6, initial_state=state, output_final_state=True,
            group_size=group_size, chunk_size=chunk, carry_mode=carry, precision_policy='bf16', telemetry=info,
        )
        dell, = torch.autograd.grad(final[0, 0, 0, 0], ell)
    target = q.new_tensor(.5 * math.exp(-1) / math.sqrt(key_dim))
    torch.testing.assert_close(final[0, 0, 0, 0], target, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dell[0, 1, 0, 0], target, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(output).all() and torch.isfinite(dell).all()
    assert info['precision_route'] == info['backward_precision_route'] == 'bf16'


@pytest.mark.parametrize('entry', ['chunk_key_gated_hyperbolic_delta_rule', 'fused_recurrent_key_gated_hyperbolic_delta_rule'])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_zero_curvature_source(entry):
    q = torch.ones(1, 1, 1, 1, device=device)
    curve = q.new_zeros((), requires_grad=True)
    with NoFP64():
        _, final = getattr(ops, entry)(
            q, q.clone(), q * .5, q.new_full((1, 1, 1), .5), torch.zeros_like(q),
            curvature=curve, initial_state=q.new_full((1, 1, 1, 1), .1), output_final_state=True,
            group_size=1, precision_policy='bf16',
        )
        dc, = torch.autograd.grad(final.sum(), curve)
    torch.testing.assert_close(dc, q.new_tensor(.02), atol=1e-8, rtol=2e-5)
    assert dc.item() != 0


@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_rejects_double_backward():
    sources, _ = make_case(1, 128, 128, 32, torch.float32)
    with NoFP64():
        output, _ = ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], precision_policy='bf16', chunk_size=16,
        )
        with pytest.raises(RuntimeError, match='first-order gradients only'):
            torch.autograd.grad(output.sum(), sources, create_graph=True)


@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_rejects_unqualified_compiler_before_launch(monkeypatch):
    sources, _ = make_case(1, 128, 128, 32, torch.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError('kernel launched with an unqualified compiler')

    monkeypatch.setattr(bf16, '_compiler_digest', lambda: 'unqualified')
    monkeypatch.setattr(bf16, 'launch', forbidden)
    with pytest.raises(ValueError, match='qualified compiler binary'):
        ops.chunk_key_gated_hyperbolic_delta_rule(
            *sources[:5], curvature=sources[6], initial_state=sources[5], precision_policy='bf16', chunk_size=16,
        )


@pytest.mark.parametrize('value_tile', [64, 128])
@pytest.mark.parametrize('case,dtype,packed', [
    ('random', torch.float32, False), ('random', torch.bfloat16, False),
    ('random', torch.float32, True), ('random', torch.bfloat16, True),
    ('fixed_point', torch.float32, False), ('repeated_beta1', torch.float32, False),
    ('strong_gate', torch.float32, False), ('zero_curvature', torch.float32, False),
    ('raw_margin', torch.float32, False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_grouped_output(value_tile, case, dtype, packed, state_value_tile=32, query_bf16=False,
                                  state_residual_mode='stable', source_vjp='coupled_fused'):
    from fla.ops.key_gated_hyperbolic_delta_rule.bf16_selective import chunk_bf16

    length = 2 if case == 'strong_gate' else 1 if case == 'raw_margin' else 65 if case in (
        'fixed_point', 'repeated_beta1') else 129
    source, cu = make_case(length, 128, 128, 32, dtype=dtype, packed=packed)
    q, k, v, beta, ell, state, curve = (x.detach() for x in source)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(dtype)
    if case in ('fixed_point', 'repeated_beta1', 'strong_gate', 'raw_margin'):
        q.zero_()
        q[..., 0] = 1
        k = q.clone()
        v.zero_()
        v[..., ::32] = .5
        beta.fill_(1. if case == 'repeated_beta1' else .5)
        ell.zero_()
        state.zero_()
        curve.fill_(1.)
        if case == 'fixed_point':
            state[:, :, 0, ::32] = .5
        if case == 'strong_gate':
            beta[:, 0], beta[:, 1] = 1, 0
            ell[:, 0], ell[:, 1] = -1e8, -1.
        if case == 'raw_margin':
            v.fill_(.9998779296875)
    if case == 'zero_curvature':
        curve.zero_()
    if query_bf16:
        q = q.to(torch.bfloat16)
    source = tuple(x.requires_grad_(i < 6) for i, x in enumerate((q, k, v, beta, ell, state, curve)))
    reference = tuple(x.detach().double().requires_grad_(i < 6) for i, x in enumerate(source))
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *reference[:5], initial_state=reference[5], curvature=reference[6], group_size=32,
        cu_seqlens=cu, output_final_state=True,
    )
    cot = tuple(torch.sin(torch.arange(x.numel(), device=x.device)).reshape(x.shape).float() for x in expected)
    truth = torch.autograd.grad(expected, reference[:6], tuple(x.double() for x in cot))
    info = {}
    with NoFP64():
        actual = chunk_bf16(
            *source[:5], curvature=curve, initial_state=state, cu_seqlens=cu, output_final_state=True,
            chunk_size=64, source_vjp=source_vjp, value_tile=value_tile,
            state_value_tile=state_value_tile, state_residual_mode=state_residual_mode, telemetry=info,
        )
        gradients = torch.autograd.grad(actual, source[:6], cot, retain_graph=True)
        repeated = torch.autograd.grad(actual, source[:6], cot)
    assert info['value_tile_requested'] == value_tile
    eligible = q.dtype == torch.bfloat16 and case != 'raw_margin'
    assert info['value_tile_effective'] == (value_tile if eligible else 32)
    assert info['state_value_tile_effective'] == (state_value_tile if eligible else 32)
    batch_requested = max(value_tile, state_value_tile) > 32
    assert info['group_batch_fallback'] == (
        None if eligible or not batch_requested else 'materialized_rho' if case == 'raw_margin' else 'fp32_query')
    assert info['state_residual_mode_effective'] == ('direct' if state_residual_mode == 'direct' and eligible else 'stable')
    assert all(torch.equal(a, b) for a, b in zip(gradients, repeated))
    for index, (a, b) in enumerate(zip((*actual, *gradients), (*expected, *truth))):
        compare_bf16(a, b, backward=index >= 2)


@pytest.mark.parametrize('value_tile', [64, 128])
@pytest.mark.parametrize('case,dtype,packed', [
    ('random', torch.float32, False), ('random', torch.bfloat16, True),
    ('fixed_point', torch.float32, False), ('repeated_beta1', torch.float32, False),
    ('strong_gate', torch.float32, False), ('zero_curvature', torch.float32, False),
    ('raw_margin', torch.float32, False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_grouped_state(value_tile, case, dtype, packed):
    test_selective_grouped_output(value_tile, case, dtype, packed, state_value_tile=value_tile)


@pytest.mark.parametrize('value_tile', [64, 128])
@pytest.mark.parametrize('state_wide', [False, True])
@pytest.mark.parametrize('case,packed', [
    ('random', False), ('random', True), ('fixed_point', False), ('repeated_beta1', False),
    ('strong_gate', False), ('zero_curvature', False), ('raw_margin', False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_grouped_bf16_production(value_tile, state_wide, case, packed):
    test_selective_grouped_output(
        value_tile, case, torch.float32, packed,
        state_value_tile=value_tile if state_wide else 32, query_bf16=True,
    )


@pytest.mark.parametrize('source_vjp', ['coupled_state', 'coupled_fused'])
@pytest.mark.parametrize('case,packed', [
    ('random', False), ('random', True), ('fixed_point', False), ('repeated_beta1', False),
    ('strong_gate', False), ('zero_curvature', False), ('raw_margin', False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_direct_state_residual(source_vjp, case, packed):
    test_selective_grouped_output(
        32, case, torch.float32, packed, query_bf16=True,
        state_residual_mode='direct', source_vjp=source_vjp,
    )


@pytest.mark.parametrize('source_vjp', ['coupled_state', 'coupled_fused'])
@pytest.mark.parametrize('case,packed', [
    ('random', False), ('random', True), ('fixed_point', False), ('repeated_beta1', False),
    ('strong_gate', False), ('zero_curvature', False), ('raw_margin', False),
])
@pytest.mark.usefixtures('qualified_bf16')
def test_selective_direct_grouped_state_residual(source_vjp, case, packed):
    test_selective_grouped_output(
        32, case, torch.float32, packed, state_value_tile=64, query_bf16=True,
        state_residual_mode='direct', source_vjp=source_vjp,
    )


@pytest.mark.usefixtures('qualified_bf16')
def test_selective_validation_mode_requires_explicit_training_scope():
    source, cu = make_case(65, 128, 128, 32, dtype=torch.bfloat16)
    with torch.no_grad(), pytest.raises(ValueError, match='training-only research mode'):
        ops.chunk_key_gated_hyperbolic_delta_rule(
            *source[:5], curvature=source[6].detach(), initial_state=source[5], cu_seqlens=cu,
            chunk_size=64, source_vjp='coupled_fused', precision_policy='bf16_selective',
            validation_mode='deferred', telemetry={},
        )


@pytest.mark.usefixtures('qualified_bf16')
def test_selective_deferred_validation_matches_strict_and_repeat_outside_scope():
    from fla.ops.key_gated_hyperbolic_delta_rule.validation import FiniteValidationScope

    source, cu = make_case(65, 128, 128, 32, dtype=torch.bfloat16, packed=True)
    source = tuple(x.detach().requires_grad_(i < 6) for i, x in enumerate(source))
    common = dict(curvature=source[6], initial_state=source[5], cu_seqlens=cu, chunk_size=64,
                  source_vjp='coupled_fused', precision_policy='bf16_selective', output_final_state=True)
    with NoFP64():
        strict = ops.chunk_key_gated_hyperbolic_delta_rule(*source[:5], **common)
        cot = tuple(torch.sin(torch.arange(x.numel(), device=x.device)).reshape(x.shape).float() for x in strict)
        expected = torch.autograd.grad(strict, source[:6], cot)
        with FiniteValidationScope() as checks:
            actual = ops.chunk_key_gated_hyperbolic_delta_rule(
                *source[:5], **common, validation_mode='deferred', validation_scope=checks,
            )
            gradients = torch.autograd.grad(actual, source[:6], cot, retain_graph=True)
            assert checks.checks == 2
        repeated = torch.autograd.grad(actual, source[:6], cot)
    assert checks.committed and not checks.open
    assert all(torch.equal(a, b) for a, b in zip((*actual, *gradients), (*strict, *expected)))
    assert all(torch.equal(a, b) for a, b in zip(gradients, repeated))


@pytest.mark.usefixtures('qualified_bf16')
def test_selective_deferred_validation_blocks_nonfinite_update():
    from fla.ops.key_gated_hyperbolic_delta_rule.validation import FiniteValidationScope

    source, cu = make_case(1, 128, 128, 32, dtype=torch.bfloat16)
    source = tuple(x.detach().requires_grad_(i < 6) for i, x in enumerate(source))
    optimizer = torch.optim.SGD(source[:6], lr=.001, momentum=.9)
    snapshots = [x.detach().clone() for x in source[:6]]
    with pytest.raises(RuntimeError, match='nonfinite KHDN training'), NoFP64(), FiniteValidationScope(optimizer) as checks:
        output, _ = ops.chunk_key_gated_hyperbolic_delta_rule(
            *source[:5], curvature=source[6], initial_state=source[5], cu_seqlens=cu, chunk_size=64,
            source_vjp='coupled_fused', precision_policy='bf16_selective',
            validation_mode='deferred', validation_scope=checks,
        )
        (output.float() * float('nan')).sum().backward()
        optimizer.step()
    assert not optimizer.state and all(torch.equal(x, y) for x, y in zip(source[:6], snapshots))
