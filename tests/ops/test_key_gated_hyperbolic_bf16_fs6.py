# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import hashlib
from pathlib import Path

import pytest
import torch
import triton
from test_key_gated_hyperbolic_delta_rule import compare_tensor, make_case, ops


_NVIDIA = torch.cuda.is_available() and torch.version.cuda is not None
pytestmark = pytest.mark.skipif(not _NVIDIA, reason='The serial FS6 GPU tests require NVIDIA CUDA.')
_COMPILER = '1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32'


def _runtime_reason():
    if torch.cuda.get_device_capability() != (9, 0):
        return 'unqualified_device'
    if torch.__version__ != '2.9.0+cu128' or triton.__version__ != '3.5.0':
        return 'unqualified_runtime'
    return None


@pytest.fixture(scope='module')
def qualified_compiler():
    if _runtime_reason():
        pytest.skip('FS6 native assertions require the qualified SM90 software stack.')
    from triton._C import libtriton
    digest = hashlib.sha256()
    with Path(libtriton.__file__).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != _COMPILER:
        pytest.skip('FS6 native assertions require the tested compiler binary.')


def test_radix_projection_unqualified_stack():
    reason = _runtime_reason()
    if reason is None:
        pytest.skip('This case tests the public strict fallback on other NVIDIA stacks.')
    sources, _ = make_case(17, 128, 128, 32, torch.float32)
    opts = dict(curvature=sources[6], initial_state=sources[5], output_final_state=True, group_size=32, chunk_size=16)
    info = {}
    out = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **opts, precision_policy='bf16_fs6', telemetry=info)
    strict = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **opts, precision_policy='strict')
    cot = [torch.randn_like(v) for v in out]
    grad = torch.autograd.grad(out, sources, cot)
    expected = torch.autograd.grad(strict, sources, cot)
    assert all(torch.equal(a, b) for a, b in zip((*out, *grad), (*strict, *expected)))
    assert info['qualification_reason'] == reason
    assert info['precision_route'] == info['backward_precision_route'] == 'strict'


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('loss_mode', ['output', 'state', 'joint'])
@pytest.mark.usefixtures('qualified_compiler')
def test_radix_projection_sources(dtype, packed, loss_mode):
    sources, cu = make_case(35, 128, 128, 32, dtype, packed=packed)
    ref = tuple(x.detach().double().requires_grad_() for x in sources)
    shared = dict(group_size=32, output_final_state=True, cu_seqlens=cu)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(*ref[:5], curvature=ref[6], initial_state=ref[5], **shared)
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=sources[6], initial_state=sources[5], **shared,
        chunk_size=16, precision_policy='bf16_fs6', telemetry=info,
    )
    cot = [torch.randn_like(y) for y in actual]
    if loss_mode == 'output':
        cot[1].zero_()
    elif loss_mode == 'state':
        cot[0].zero_()
    grad = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, ref, [g.double() for g in cot])
    for name, observed, target in zip(('output', 'state'), actual, expected):
        compare_tensor(name, observed, target)
    for name, observed, target in zip(('q', 'k', 'v', 'beta', 'ell', 'S0', 'c'), grad, truth):
        compare_tensor(name, observed, target, backward=True)
    assert info['precision_requested'] == 'bf16_fs6'
    assert info['precision_route'] == info['backward_precision_route'] == 'experimental_bf16_fs6'
    assert info['arithmetic'] == 'bf16_fs6_projection' and info['backward_precision'] == 'device_float64'
    assert info['readout_precision'] == info['state_control_precision'] == 'fp64'
    assert info['tensor_cores'] and info['fallback_subblocks'] == 0 and not info.get('strict_replay_details')


@pytest.mark.parametrize('chunk,carry,k,v,g', [(32, 'no_carry', 128, 128, 32), (64, 'no_carry', 128, 128, 32),
                                              (16, 'y_carry', 128, 128, 32), (16, 'no_carry', 16, 16, 16),
                                              (16, 'no_carry', 128, 128, 16)])
@pytest.mark.usefixtures('qualified_compiler')
def test_radix_projection_rejects_other_geometries(chunk, carry, k, v, g):
    sources, _ = make_case(33, k, v, g, torch.float32)
    opts = dict(curvature=sources[6], initial_state=sources[5], output_final_state=True, group_size=g, chunk_size=chunk,
                carry_mode=carry)
    info = {}
    out = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **opts, precision_policy='bf16_fs6', telemetry=info)
    strict = ops.chunk_key_gated_hyperbolic_delta_rule(*sources[:5], **opts, precision_policy='strict')
    cot = [torch.randn_like(t) for t in out]
    grads = torch.autograd.grad(out, sources, cot)
    reference = torch.autograd.grad(strict, sources, cot)
    assert all(torch.equal(a, b) for a, b in zip((*out, *grads), (*strict, *reference)))
    assert not info['precision_eligible'] and info['qualification_reason'] == 'unqualified_bf16_fs6_geometry'
    assert info['precision_route'] == info['backward_precision_route'] == 'strict' and not info['tensor_cores']


@pytest.mark.parametrize('case,length,curve,amplitude,fwd,bwd', [
    ('zero_scale', 17, .6, 1., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('cancellation', 17, .6, 1., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('normal', 16, 0., 8., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('normal', 16, 1.0000598430633545, 128., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('normal', 16, 2., 128., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('normal', 16, 2.000000238418579, 1., 'strict', 'strict'),
    ('normal', 17, 1.0000598430633545, 1., 'strict', 'strict'),
    ('normal', 32, 1., 32., 'experimental_bf16_fs6', 'experimental_bf16_fs6'),
    ('normal', 32, 1., 32.000003814697266, 'experimental_bf16_fs6', 'strict_device_replay'),
    ('normal', 16, 1., 128.00001525878906, 'experimental_bf16_fs6', 'strict_device_replay'),
])
@pytest.mark.usefixtures('qualified_compiler')
def test_radix_projection_boundaries(case, length, curve, amplitude, fwd, bwd):
    sources, _ = make_case(length, 128, 128, 32, torch.float32)
    q, k, v, beta, ell, state, c = sources
    with torch.no_grad():
        c.fill_(curve)
        if case == 'zero_scale':
            beta[:, ::2] = 0
            state[..., ::2] = 0
        if case == 'cancellation':
            k.zero_()
            k[..., 0], k[..., 1] = 1 + 2.**-13, 1.
            state.zero_()
            state[:, :, 0, :], state[:, :, 1, :] = .25 * (1 - 2.**-13), -.25
            ell.zero_()
    refs = tuple(x.detach().double().requires_grad_() for x in sources)
    expected = ops.naive_recurrent_key_gated_hyperbolic_delta_rule(
        *refs[:5], curvature=refs[6], initial_state=refs[5], group_size=32, output_final_state=True,
    )
    info = {}
    actual = ops.chunk_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, ell, curvature=c, initial_state=state, chunk_size=16, group_size=32,
        output_final_state=True, precision_policy='bf16_fs6', telemetry=info,
    )
    cot = tuple(torch.full_like(y, amplitude) for y in actual)
    grads = torch.autograd.grad(actual, sources, cot)
    truth = torch.autograd.grad(expected, refs, tuple(y.double() for y in cot))
    for name, x, y in zip(('output', 'state'), actual, expected):
        compare_tensor(name, x, y)
    for name, x, y in zip(('q', 'k', 'v', 'beta', 'ell', 'S0', 'c'), grads, truth):
        compare_tensor(name, x, y, backward=True)
    assert info['precision_route'] == fwd and info['backward_precision_route'] == bwd
    if bwd == 'strict_device_replay':
        assert info['strict_replay_details']['precision_route'] == 'strict'
