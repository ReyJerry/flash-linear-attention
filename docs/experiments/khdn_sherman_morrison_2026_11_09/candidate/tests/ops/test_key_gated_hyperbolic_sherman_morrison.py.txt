# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULES = {}
for name in ('naive', 'sherman_morrison_reference'):
    fullname = 'fla.ops.key_gated_hyperbolic_delta_rule.' + name
    spec = importlib.util.spec_from_file_location(fullname, ROOT / 'fla/ops/key_gated_hyperbolic_delta_rule' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    MODULES[name] = module
naive_recurrent_key_gated_hyperbolic_delta_rule = MODULES['naive'].naive_recurrent_key_gated_hyperbolic_delta_rule
sherman_morrison_recurrent_key_gated_hyperbolic_delta_rule = (
    MODULES['sherman_morrison_reference'].sherman_morrison_recurrent_key_gated_hyperbolic_delta_rule
)


def _compare(case, *, batch=1, length=7, packed=False, curvature=0.7, group_size=4):
    torch.manual_seed(case)
    heads, key_dim, value_dim = 2, 8, 8
    q = torch.randn(batch, length, heads, key_dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(batch, length, heads, value_dim, dtype=torch.float64, requires_grad=True) * .1
    beta = torch.sigmoid(torch.randn(batch, length, heads, dtype=torch.float64)).requires_grad_()
    log_decay = (-torch.rand_like(q) * .2).requires_grad_()
    state = torch.randn(2 if packed else batch, heads, key_dim, value_dim, dtype=torch.float64) * .01
    curve = torch.tensor(curvature, dtype=torch.float64)
    cu = torch.tensor([0, 0, 3, length, length], dtype=torch.int64) if packed else None
    if packed:
        state = torch.cat((state[:1], state, state[:1]), dim=0)
    q, k, v, beta, log_decay, state, curve = (x.detach().clone().requires_grad_() for x in
                                              (q, k, v, beta, log_decay, state, curve))
    sources = (q, k, v, beta, log_decay, state, curve)
    reference = naive_recurrent_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=curve, initial_state=state, output_final_state=True,
        cu_seqlens=cu, group_size=group_size,
    )
    sm_info = {}
    candidate = sherman_morrison_recurrent_key_gated_hyperbolic_delta_rule(
        *sources[:5], curvature=curve, initial_state=state, output_final_state=True,
        cu_seqlens=cu, group_size=group_size, telemetry=sm_info,
    )
    torch.testing.assert_close(candidate[0], reference[0], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(candidate[1], reference[1], atol=1e-10, rtol=1e-9)
    cotangent = torch.randn_like(reference[0])
    final_cotangent = torch.randn_like(reference[1])
    ref_grads = torch.autograd.grad((reference[0] * cotangent).sum() + (reference[1] * final_cotangent).sum(), sources)
    sm_grads = torch.autograd.grad((candidate[0] * cotangent).sum() + (candidate[1] * final_cotangent).sum(), sources)
    for actual, expected in zip(sm_grads, ref_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    assert sm_info['sm_fast_solves'] > 0 and sm_info['sm_dense_fallbacks'] == 0


@pytest.mark.parametrize('curvature', [0., .7])
def test_sherman_morrison_dense_and_initial_state_seven_sources(curvature):
    _compare(42, batch=2, length=7, curvature=curvature)


def test_sherman_morrison_packed_tail_and_curvature_gradient():
    _compare(51, batch=1, length=7, packed=True, curvature=.3)


def test_sherman_morrison_dense_fallback_near_zero_d():
    torch.manual_seed(9)
    q = torch.randn(1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
    k = torch.zeros_like(q)
    k[..., 0] = 1
    v = torch.zeros(1, 1, 1, 4, dtype=torch.float64)
    v[..., 0] = 1
    beta = torch.ones(1, 1, 1, dtype=torch.float64, requires_grad=True)
    ell = torch.zeros_like(q, requires_grad=True)
    state = torch.zeros(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    curve = torch.tensor(4. - 2e-12, dtype=torch.float64, requires_grad=True)
    k.requires_grad_()
    v.requires_grad_()
    info = {}
    output, final = sherman_morrison_recurrent_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, ell, curvature=curve, initial_state=state, output_final_state=True,
        group_size=4, telemetry=info,
    )
    assert info['sm_dense_fallbacks'] == 1 and info['sm_fast_solves'] == 0
    assert torch.isfinite(output).all() and torch.isfinite(final).all()

    reference = naive_recurrent_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, ell, curvature=curve, initial_state=state, output_final_state=True, group_size=4,
    )
    torch.testing.assert_close(output, reference[0], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(final, reference[1], atol=1e-10, rtol=1e-9)
    cot = (torch.randn_like(output), torch.randn_like(final))
    sources = (q, k, v, beta, ell, state, curve)
    actual_grads = torch.autograd.grad((output, final), sources, cot)
    expected_grads = torch.autograd.grad(reference, sources, cot)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('curvature', [0., .7])
@pytest.mark.parametrize('beta_value', [0., .5, 1.])
def test_sherman_morrison_manual_token_vjp_endpoints_and_curvature(curvature, beta_value):
    torch.manual_seed(42)
    K, G = 8, 4
    sources = (torch.randn(K, dtype=torch.float64), torch.randn(K, dtype=torch.float64),
               torch.randn(G, dtype=torch.float64) * .1, torch.tensor(beta_value, dtype=torch.float64),
               -torch.rand(K, dtype=torch.float64) * .2, torch.randn(K, G, dtype=torch.float64) * .01,
               torch.tensor(curvature, dtype=torch.float64))
    q, k, v, beta, ell, state, curve = tuple(x.requires_grad_() for x in sources)
    output, final = naive_recurrent_key_gated_hyperbolic_delta_rule(
        q.reshape(1, 1, 1, K), k.reshape(1, 1, 1, K), v.reshape(1, 1, 1, G), beta.reshape(1, 1, 1),
        ell.reshape(1, 1, 1, K), curvature=curve, initial_state=state.reshape(1, 1, K, G),
        output_final_state=True, group_size=G, scale=1.,
    )
    do, ds = torch.randn(G, dtype=torch.float64), torch.randn(K, G, dtype=torch.float64)
    expected = torch.autograd.grad((output, final), (q, k, v, beta, ell, state, curve),
                                   (do.reshape_as(output), ds.reshape_as(final)))
    actual = MODULES['sherman_morrison_reference'].sherman_morrison_token_vjp(
        q, k, v, beta, ell, state, curve, do, ds,
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
