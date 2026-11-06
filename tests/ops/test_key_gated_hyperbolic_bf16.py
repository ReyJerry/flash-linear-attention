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


@pytest.fixture(scope='module')
def qualified_bf16():
    try:
        bf16._require_runtime(torch.zeros((), device=device))
    except ValueError as error:
        pytest.skip(str(error))


def compare_bf16(actual, expected, backward=False):
    atol, rtol, rms = (.005, .03, .015) if backward else (.002, .02, .01)
    assert torch.isfinite(actual).all()
    a, b = actual.double(), expected.double()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    if a.numel():
        assert (a-b).square().mean().sqrt() <= rms * b.square().mean().sqrt().clamp_min(atol)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('chunk,carry', [(16, 'no_carry'), (32, 'no_carry'), (16, 'y_carry')])
@pytest.mark.parametrize('cotangent', ['output', 'state', 'joint'])
@pytest.mark.usefixtures('qualified_bf16')
def test_bf16_chunk_sources(dtype, packed, chunk, carry, cotangent):
    sources, cu = make_case(35, 128, 128, 32, dtype, packed=packed)
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
        observed = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, reference, [x.double() for x in cot])
    for a, b in zip(actual, expected):
        compare_bf16(a, b)
    for a, b in zip(observed, truth):
        compare_bf16(a, b, backward=True)
    assert actual[1].dtype == torch.float32
    assert info['precision_route'] == info['backward_precision_route'] == 'bf16'
    assert info['fallback_subblocks'] == info['backward_fallback_subblocks'] == 0
    assert info['backward_precision'] == 'device_float32' and not info['host_fallback']


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
