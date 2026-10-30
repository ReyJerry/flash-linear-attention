# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU canonical KHDN contract tests, without Triton or package initialization.

Run with FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q <this file>.
The independent expectations use homogeneous coordinates, closed-form algebra,
and geometric identities; they do not import production helpers or the design smoke.
The FP64 budgets below are frozen before execution and apply to all seven VJPs.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'key_gated_hyperbolic_reference', _root / 'fla/ops/key_gated_hyperbolic_delta_rule/naive.py',
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
recurrent = ref.naive_recurrent_key_gated_hyperbolic_delta_rule
_old_spec = importlib.util.spec_from_file_location(
    'old_hyperbolic_reference', _root / 'fla/ops/hyperbolic_delta_rule/naive.py',
)
old_ref = importlib.util.module_from_spec(_old_spec)
_old_spec.loader.exec_module(old_ref)

ATOL, RTOL = 1e-10, 1e-9
GRADCHECK = dict(eps=1e-6, atol=1e-6, rtol=1e-4)
NAMES = ('q', 'k', 'v', 'beta', 'log_decay', 'initial_state', 'curvature')


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def leaves(xs):
    return tuple(x.detach().clone().requires_grad_() for x in xs)


def make_case(B=2, T=5, H=2, K=5, V=8, g=4, c=0.6):
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn_like(q)
    k = k / k.norm(dim=-1, keepdim=True) * (0.8 + torch.rand(B, T, H, 1, dtype=q.dtype))
    v = 0.2 * torch.randn(B, T, H, V, dtype=q.dtype).tanh() / max(c, 1) ** 0.5
    beta = 0.15 + 0.7 * torch.rand(B, T, H, dtype=q.dtype)
    ell = -0.05 - 0.4 * torch.rand_like(q)
    state = torch.randn(B, H, K, V // g, g, dtype=q.dtype)
    state = 0.2 * g**0.5 * state / state.square().sum((-3, -1), keepdim=True).sqrt() / max(c, 1)**0.5
    return leaves((q, k, v, beta, ell, state.reshape(B, H, K, V), q.new_tensor(c)))


def run(xs, g=4, **kwargs):
    q, k, v, beta, ell, state, c = xs
    return recurrent(
        q, k, v, beta, ell, curvature=c, initial_state=state, output_final_state=True, group_size=g, **kwargs,
    )


def homogeneous_equations(xs, g, scale):
    """Accumulate unnormalized M,z using an analytic rank-one inverse identity."""
    q, k, v, beta, ell, s0, c = xs
    B, T, H, K = q.shape
    V = v.shape[-1]
    output_batches, final_batches = [], []
    for b in range(B):
        output_heads, final_heads = [], []
        for h in range(H):
            output_groups, final_groups = [], []
            for j in range(V // g):
                m = s0[b, h, :, j*g:(j+1)*g]
                z = q.new_tensor(1.0)
                output = []
                for t in range(T):
                    key, value = k[b, t, h], v[b, t, h, j*g:(j+1)*g]
                    m = ell[b, t, h].exp()[:, None] * m
                    n = key @ key
                    r = z * value - m.T @ key
                    a = n - (c/g) * (value @ value)
                    write = beta[b, t, h] * (r / n + (c/g) * value * (value @ r) / (n * a))
                    m = m + torch.outer(key, write)
                    z = z + (c/g) * (value @ write)
                    output.append(scale * (q[b, t, h] @ m) / z)
                output_groups.append(torch.stack(output))
                final_groups.append(m / z)
            output_heads.append(torch.cat(output_groups, dim=-1))
            final_heads.append(torch.cat(final_groups, dim=-1))
        output_batches.append(torch.stack(output_heads, dim=1))
        final_batches.append(torch.stack(final_heads))
    return torch.stack(output_batches), torch.stack(final_batches)


@pytest.mark.parametrize('g', [1, 2, 16, 32], ids=lambda g: f'g{g}')
@pytest.mark.parametrize('c,gate', [(0., 'random'), (.6, 'random'), (2., 'zero'), (.6, 'one'), (.6, 'near_one')])
@pytest.mark.parametrize('cotangent', ['output', 'state', 'joint', 'zero'])
def test_recurrent_homogeneous_values_and_seven_vjps(g, c, gate, cotangent):
    xs = list(make_case(B=2, T=3, H=2, K=5, V=2*g, g=g, c=c))
    if gate != 'random':
        xs[3] = torch.full_like(xs[3], {'zero': 0., 'one': 1., 'near_one': 1-1e-8}[gate], requires_grad=True)
    xs = tuple(xs)
    independent = leaves(xs)
    actual = run(xs, g=g, scale=0.7)
    expected = homogeneous_equations(independent, g=g, scale=0.7)
    for a, e in zip(actual, expected):
        close(a, e)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    if cotangent in ('state', 'zero'):
        do.zero_()
    if cotangent in ('output', 'zero'):
        ds.zero_()
    actual_grads = torch.autograd.grad(actual, xs, (do, ds))
    expected_grads = torch.autograd.grad(expected, independent, (do, ds))
    for name, a, e in zip(NAMES, actual_grads, expected_grads):
        assert torch.isfinite(a).all(), name
        close(a, e)
        if cotangent == 'zero' or (cotangent == 'state' and name == 'q'):
            assert torch.count_nonzero(a) == 0, name


@pytest.mark.parametrize('g', [1, 2])
def test_recurrent_seven_input_gradcheck(g):
    xs = make_case(B=1, T=2, H=1, K=2, V=2*g, g=g)
    assert torch.autograd.gradcheck(lambda *args: run(args, g=g), xs, **GRADCHECK)


@pytest.mark.parametrize('g', [16, 32])
def test_recurrent_representative_multihead_groups(g):
    xs = make_case(B=2, T=3, H=2, K=128, V=128, g=g)
    independent = leaves(xs)
    actual, expected = run(xs, g=g), homogeneous_equations(independent, g=g, scale=128**-0.5)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    for a, e in zip(actual, expected):
        close(a, e)
    for a, e in zip(torch.autograd.grad(actual, xs, (do, ds)), torch.autograd.grad(expected, independent, (do, ds))):
        close(a, e)


@pytest.mark.parametrize('g', [16, 32])
def test_recurrent_longer_odd_length(g):
    xs = make_case(B=1, T=129, H=1, K=5, V=2*g, g=g)
    independent = leaves(xs)
    actual = run(xs, g=g)
    expected = homogeneous_equations(independent, g=g, scale=5**-0.5)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    for a, e in zip(actual, expected):
        close(a, e)
    for a, e in zip(torch.autograd.grad(actual, xs, (do, ds)), torch.autograd.grad(expected, independent, (do, ds))):
        close(a, e)


@pytest.mark.parametrize('g', [1, 16, 32])
@pytest.mark.parametrize('beta_value', [0., .45, 1.])
def test_recurrent_geometry(g, beta_value):
    q, k, v, beta, ell, state, c = make_case(B=1, T=7, H=2, K=5, V=2*g, g=g)
    beta = torch.full_like(beta, beta_value)
    eye = torch.eye(5, dtype=q.dtype)
    for t in range(q.shape[1]):
        xs = (q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], beta[:, t:t+1], ell[:, t:t+1], state, c)
        out, new_state = run(xs, g=g, scale=1)
        for h in range(2):
            key = k[0, t, h]
            projection = eye - torch.outer(key, key) / (key @ key)
            for j in range(2):
                old = state[0, h, :, j*g:(j+1)*g]
                new = new_state[0, h, :, j*g:(j+1)*g]
                value = v[0, t, h, j*g:(j+1)*g]
                x = ell[0, t, h].exp()[:, None] * old
                p = x.T @ key
                n = key @ key
                d = (1-beta_value) + beta_value * (n-(c/g)*(value @ p)) / (n-(c/g)*(value @ value))
                assert d > 0
                assert (c/g) * new.square().sum() < 1
                close(value - new.T @ key, (1-beta_value) * (value-p) / d)
                close(projection @ new, (projection @ x) / d)
                if beta_value == 1:
                    close(new.T @ key, value)
                if beta_value == 0:
                    close(new, x)
        close(out[:, 0], torch.einsum('bhk,bhkv->bhv', q[:, t], new_state))
        state = new_state


@pytest.mark.parametrize('g', [1, 16, 32])
def test_recurrent_post_decay_fixed_point(g):
    q, k, _, beta, ell, state, c = make_case(B=2, T=1, H=2, V=2*g, g=g)
    decayed = ell[:, 0].exp().unsqueeze(-1) * state
    v = torch.einsum('bhk,bhkv->bhv', k[:, 0], decayed).unsqueeze(1)
    _, final = run((q, k, v, beta, ell, state, c), g=g)
    close(final, decayed)
    assert not torch.allclose(final, state)


def test_recurrent_zero_beta_derivative_and_identity_gate_derivative():
    xs = list(make_case(B=1, T=1, H=1, K=3, V=4))
    xs[3] = torch.zeros_like(xs[3], requires_grad=True)
    xs[4] = torch.zeros_like(xs[4], requires_grad=True)
    _, state = run(xs)
    close(state, xs[5])
    key, value, x = xs[1][0, 0, 0], xs[2][0, 0, 0], xs[5][0, 0]
    c, n = xs[6], key @ key
    r = value - x.T @ key
    a = r/n + (c/4)*value*(value @ r)/(n*(n-(c/4)*(value @ value)))
    expected_db = torch.outer(key, a) - (c/4)*(value @ a)*x
    db = torch.autograd.functional.jacobian(lambda b: run((*xs[:3], b, *xs[4:]))[1], xs[3])
    close(db.reshape(3, 4), expected_db)
    assert db.abs().max() > 1e-4
    dell = torch.autograd.grad(state.sum(), xs[4])[0]
    close(dell[0, 0, 0], x.sum(-1))
    assert dell.abs().max() > 1e-4


@pytest.mark.parametrize('unit_key', [False, True])
def test_recurrent_flat_normalized_kda_and_analytic_dc(unit_key):
    xs = make_case(B=2, T=1, H=2, c=0.)
    q, k, v, beta, ell, state, c = xs
    if unit_key:
        k = k / k.norm(dim=-1, keepdim=True)
    key, value, b = k[:, 0], v[:, 0], beta[:, 0, :, None]
    x = ell[:, 0].exp().unsqueeze(-1) * state
    n = key.square().sum(-1, keepdim=True)
    residual = value - torch.einsum('bhk,bhkv->bhv', key, x)
    flat = x + key.unsqueeze(-1) * (b*residual/n).unsqueeze(-2)
    out, final = run((q, k, v, beta, ell, state, c), scale=1.)
    close(final, flat)
    close(out[:, 0], torch.einsum('bhk,bhkv->bhv', q[:, 0], flat))
    expected_dc = torch.zeros_like(flat)
    for j in range(2):
        vg, rg = value[..., j*4:(j+1)*4], residual[..., j*4:(j+1)*4]
        coefficient = b * (vg*rg).sum(-1, keepdim=True) / (4*n)
        expected_dc[..., j*4:(j+1)*4] = coefficient.unsqueeze(-2) * (
            key.unsqueeze(-1) * vg.unsqueeze(-2) / n.unsqueeze(-1) - flat[..., j*4:(j+1)*4]
        )
    dc = torch.autograd.functional.jacobian(lambda c: run((q, k, v, beta, ell, state, c))[1], c)
    close(dc, expected_dc)
    assert dc.abs().max() > 1e-5


def test_recurrent_flat_exact_curvature_counterexample():
    q = k = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
    v = q.new_tensor([[[[.5]]]])
    beta = q.new_tensor([[[.5]]])
    ell = torch.zeros_like(q)
    state = q.new_tensor([[[[0.], [.2]]]])
    c = q.new_tensor(0., requires_grad=True)
    dc = torch.autograd.functional.jacobian(lambda c: run((q, k, v, beta, ell, state, c), g=1)[1], c)
    close(dc.flatten(), q.new_tensor([1/32, -1/40]))


@pytest.mark.parametrize('c', [0., .6, 2.])
def test_recurrent_g1_no_decay_old_hdn_parity(c):
    xs = list(make_case(T=7, V=5, g=1, c=c))
    xs[4] = torch.zeros_like(xs[4], requires_grad=True)
    q, k, v, beta, _, state, curvature = xs
    actual = run(xs, g=1, scale=.3)
    expected = old_ref.naive_recurrent_hyperbolic_delta_rule(
        q, k, v, beta, curvature=curvature, scale=.3, initial_state=state, output_final_state=True,
    )
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    for a, e in zip(actual, expected):
        close(a, e)
    comparable = (q, k, v, beta, state, curvature)
    ga = torch.autograd.grad(actual, comparable, (do, ds), retain_graph=True)
    ge = torch.autograd.grad(expected, comparable, (do, ds))
    for a, e in zip(ga, ge):
        close(a, e)
    assert torch.autograd.grad(actual, xs[4], (do, ds))[0].abs().max() > 1e-6


@pytest.mark.parametrize('K,g', [(1, 1), (3, 32)])
def test_recurrent_exact_strong_decay_and_second_decay_gradient(K, g):
    q = torch.ones(1, 2, 1, K, dtype=torch.float64)
    k = torch.full_like(q, 1/math.sqrt(K))
    v = q.new_tensor([.5, .1]).reshape(1, 2, 1, 1).expand(1, 2, 1, g)
    beta = q.new_tensor([[[1.], [0.]]])
    ell = q.new_tensor([-1e8, -1.]).reshape(1, 2, 1, 1).expand_as(q).clone().requires_grad_()
    state = q.new_zeros(1, 1, K, g)
    out, final = run((q, k, v, beta, ell, state, q.new_tensor(.6)), g=g, scale=1.)
    expected = torch.outer(k[0, 0, 0], v[0, 0, 0]) * math.exp(-1)
    close(final[0, 0], expected)
    close(out[0, 0, 0], q[0, 0, 0] @ (expected * math.e))
    close(out[0, 1, 0], q[0, 1, 0] @ expected)
    dell = torch.autograd.grad(final.sum(), ell)[0]
    close(dell[0, 1, 0], expected.sum(-1))
    close(dell[0, 0, 0], torch.zeros(K, dtype=q.dtype))
    # FP32 cumulative subtraction loses the second local decay in this legal case
    prefix = ell.detach().float().cumsum(dim=1)
    assert torch.count_nonzero(prefix[:, 1] - prefix[:, 0]) == 0


@pytest.mark.parametrize('c', [0., .6])
@pytest.mark.parametrize('cut', [0, 1, 3, 8])
def test_recurrent_streaming_all_seven_gradients(c, cut):
    xs = make_case(T=8, c=c)
    split_xs = leaves(xs)
    actual = run(xs)
    first, mid = run(tuple(x[:, :cut] for x in split_xs[:5]) + split_xs[5:])
    second, final = run(tuple(x[:, cut:] for x in split_xs[:5]) + (mid, split_xs[6]))
    joined = torch.cat((first, second), dim=1), final
    for a, e in zip(joined, actual):
        close(a, e)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    for a, e in zip(torch.autograd.grad(joined, split_xs, (do, ds)), torch.autograd.grad(actual, xs, (do, ds))):
        close(a, e)


@pytest.mark.parametrize('integer_dtype', [torch.int32, torch.int64])
def test_recurrent_packed_empty_and_tail_gradients(integer_dtype):
    q, k, v, beta, ell, _, c = make_case(B=1, T=11)
    cu = torch.tensor([0, 0, 1, 6, 6, 11, 11], dtype=integer_dtype)
    state = make_case(B=6)[5]
    xs = leaves((q, k, v, beta, ell, state, c))
    separate = leaves(xs)
    actual = run(xs, cu_seqlens=cu)
    outputs, finals = [], []
    for i, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        args = tuple(x[:, start:end] for x in separate[:5]) + (separate[5][i:i+1], separate[6])
        out, final = run(args)
        outputs.append(out)
        finals.append(final)
    expected = torch.cat(outputs, dim=1), torch.cat(finals)
    for a, e in zip(actual, expected):
        close(a, e)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    ga = torch.autograd.grad(actual, xs, (do, ds))
    ge = torch.autograd.grad(expected, separate, (do, ds))
    for a, e in zip(ga, ge):
        close(a, e)
    for i in (0, 3, 5):
        close(actual[1][i], xs[5][i])
        close(ga[5][i], ds[i])


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('cotangent', ['output', 'state', 'joint', 'zero'])
@pytest.mark.parametrize('c,large_state', [(0.6, False), (0., True)], ids=['positive-ball', 'flat-large-finite'])
def test_recurrent_empty_connected_zero_gradients(packed, cotangent, c, large_state):
    xs = make_case(B=1 if packed else 2, T=0, c=c)
    cu = torch.tensor([0, 0, 0, 0]) if packed else None
    if packed:
        xs = (*xs[:5], make_case(B=3, c=c)[5], xs[6])
    if large_state:
        xs = (*xs[:5], torch.full_like(xs[5], 1e200, requires_grad=True), xs[6])
    out, state = run(xs, cu_seqlens=cu)
    assert out.shape == (xs[0].shape[0], 0, 2, 8)
    close(state, xs[5])
    ds = torch.randn_like(state) if cotangent in ('state', 'joint') else torch.zeros_like(state)
    grads = torch.autograd.grad((out, state), xs, (torch.zeros_like(out), ds))
    for i, grad in enumerate(grads):
        close(grad, ds if i == 5 else torch.zeros_like(grad))


def test_recurrent_future_perturbations_and_gradients():
    xs = make_case(B=1, T=7)
    actual, _ = run(xs)
    grads = torch.autograd.grad(actual[:, :3].square().sum(), xs)
    for grad in grads[:5]:
        assert torch.count_nonzero(grad[:, 3:]) == 0
    changed = [x.detach().clone() for x in xs]
    for x in changed[:3]:
        x[:, 3:] *= -1
    changed[3][:, 3:] = 1-changed[3][:, 3:]
    changed[4][:, 3:] *= 2
    close(run(changed)[0][:, :3], actual[:, :3])
    cu = torch.tensor([0, 3, 7])
    packed_xs = (*xs[:5], make_case(B=2)[5], xs[6])
    out, final = run(packed_xs, cu_seqlens=cu)
    grads = torch.autograd.grad(out[:, 3:].sum() + final[1].sum(), packed_xs)
    for grad in grads[:5]:
        assert torch.count_nonzero(grad[:, :3]) == 0
    assert torch.count_nonzero(grads[5][0]) == 0


def test_recurrent_noncontiguous_values_and_gradients():
    xs = make_case(B=1)
    noncontiguous = tuple(torch.stack((x, torch.zeros_like(x)), dim=-1)[..., 0].detach().requires_grad_() for x in xs)
    assert all(not x.is_contiguous() for x in noncontiguous[:-1])
    actual, expected = run(noncontiguous), run(xs)
    do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    for a, e in zip(actual, expected):
        close(a, e)
    for a, e in zip(torch.autograd.grad(actual, noncontiguous, (do, ds)), torch.autograd.grad(expected, xs, (do, ds))):
        close(a, e)
    cu = torch.tensor([[0, 91], [2, 91], [5, 91]], dtype=torch.int64)[:, 0]
    packed = (*noncontiguous[:5], make_case(B=2)[5], noncontiguous[6])
    close(run(packed, cu_seqlens=cu)[0], run(packed, cu_seqlens=cu.contiguous())[0])


def test_recurrent_valid_noop_tail_and_active_decay_distinction():
    xs = make_case(B=1, T=5)
    out, final = run(xs)
    tail = 3
    q, k, v, beta, ell = (torch.zeros_like(x[:, :tail]) for x in xs[:5])
    k[..., 0] = 1
    padded = tuple(torch.cat((x, t), dim=1) for x, t in zip(xs[:5], (q, k, v, beta, ell))) + xs[5:]
    padded_out, padded_final = run(padded)
    close(padded_out[:, :5], out)
    close(padded_out[:, 5:], torch.zeros_like(padded_out[:, 5:]))
    close(padded_final, final)
    # a beta-zero valid row with nonidentity decay still changes the state
    active = (*padded[:4], torch.cat((xs[4], ell-.2), dim=1), *padded[5:])
    close(run(active)[1], final * math.exp(-.2*tail))
    zero_key = list(padded)
    zero_key[1] = zero_key[1].clone()
    zero_key[1][:, 5:] = 0
    with pytest.raises(ValueError, match='nonzero Key'):
        run(zero_key)


@pytest.mark.parametrize('margin', [1e-3, 1e-5])
def test_recurrent_near_boundary_correlated_keys(margin):
    q = k = torch.tensor([[[[1., 0.]], [[1., 0.]], [[1., 0.]]]], dtype=torch.float64)
    v = torch.zeros(1, 3, 1, 4, dtype=q.dtype)
    v[..., 0] = 2 * math.sqrt(1-margin)
    v[:, 1] *= -1
    beta = q.new_tensor([[[1.], [.9], [1.]]])
    ell = torch.full_like(q, -0.01)
    state = q.new_zeros(1, 1, 2, 4)
    state[0, 0, 1, 0] = 2 * math.sqrt(1-margin)
    xs = leaves((q, k, v, beta, ell, state, q.new_tensor(1.)))
    actual = run(xs)
    expected = homogeneous_equations(xs, g=4, scale=2**-0.5)
    for a, e in zip(actual, expected):
        close(a, e)
        assert torch.isfinite(a).all()
    assert (actual[1].square().sum()/4 < 1).item()
    for grad in torch.autograd.grad(tuple(x.sum() for x in actual), xs):
        assert torch.isfinite(grad).all()


def test_recurrent_defaults_optional_state_scale_and_checks():
    xs = make_case(B=1, V=64, g=32)
    q, k, v, beta, ell, state, c = xs
    default, none_state = recurrent(q, k, v, beta, ell)
    assert none_state is None
    explicit = recurrent(
        q, k, v, beta, ell, curvature=1., initial_state=torch.zeros_like(state),
        output_final_state=True, scale=5**-0.5, group_size=32,
    )
    close(default, explicit[0])
    close(run(xs, g=32)[0], run(xs, g=32, check_inputs=False)[0])
    zero_out, final = run(xs, g=32, scale=0.)
    close(zero_out, torch.zeros_like(zero_out))
    close(final, run(xs, g=32)[1])
    assert default.dtype == torch.float64 and explicit[1].dtype == torch.float64


@pytest.mark.parametrize('source', [*NAMES, 'cu_seqlens'])
def test_recurrent_device_mismatch_without_accelerator(source):
    xs = list(make_case(B=1))
    kwargs = {}
    if source == 'cu_seqlens':
        kwargs['cu_seqlens'] = torch.empty(2, dtype=torch.int64, device='meta')
    else:
        i = NAMES.index(source)
        xs[i] = torch.empty_like(xs[i], device='meta')
    with pytest.raises(ValueError, match='device'):
        run(xs, **kwargs)


@pytest.mark.parametrize('bad', [
    'beta_negative', 'beta_large', 'beta_vector', 'beta_nan', 'g_zero', 'g_negative', 'g_float', 'g_bool', 'g_remainder',
    'c_negative', 'c_nan', 'c_inf', 'c_vector', 'c_integer_tensor', 'c_timevarying', 'shape_k', 'shape_ell', 'shape_v',
    'shape_state', 'zero_key', 'zero_key_beta_zero_c_zero', 'ell_positive', 'ell_nan', 'ell_inf', 'ell_neg_inf',
    'state_boundary', 'state_outside', 'state_nan', 'value_boundary', 'value_outside', 'q_nan', 'k_inf', 'v_nan',
    'integer_q', 'scale_nan', 'scale_tensor', 'empty_key_dimension', 'empty_value_dimension', 'empty_head', 'empty_batch',
])
def test_recurrent_invalid_domain(bad):
    xs = [x.detach().clone() for x in make_case(B=1, T=2)]
    kwargs = {}
    if bad.startswith('beta_'):
        if bad == 'beta_vector':
            xs[3] = xs[3].unsqueeze(-1).expand_as(xs[2])
        else:
            xs[3].fill_({'beta_negative': -.01, 'beta_large': 1.01, 'beta_nan': math.nan}[bad])
    elif bad.startswith('g_'):
        kwargs['g'] = {'g_zero': 0, 'g_negative': -1, 'g_float': 4., 'g_bool': True, 'g_remainder': 3}[bad]
    elif bad.startswith('c_'):
        xs[6] = {
            'c_negative': torch.tensor(-1.), 'c_nan': torch.tensor(math.nan), 'c_inf': torch.tensor(math.inf),
            'c_vector': torch.ones(1), 'c_integer_tensor': torch.tensor(1), 'c_timevarying': torch.ones(1, 2, 1),
        }[bad]
    elif bad.startswith('shape_'):
        i = {'shape_k': 1, 'shape_ell': 4, 'shape_v': 2, 'shape_state': 5}[bad]
        xs[i] = xs[i][..., :-1]
    elif bad.startswith('zero_key'):
        xs[1][:, 0] = 0
        if bad.endswith('c_zero'):
            xs[3].zero_()
            xs[6].zero_()
    elif bad.startswith('ell_'):
        xs[4].fill_({'ell_positive': .01, 'ell_nan': math.nan, 'ell_inf': math.inf, 'ell_neg_inf': -math.inf}[bad])
    elif bad.startswith('state_'):
        xs[5].zero_()
        xs[6].fill_(1.)
        xs[5][0, 0, 0, 0] = {'state_boundary': 2., 'state_outside': 2.01, 'state_nan': math.nan}[bad]
    elif bad.startswith('value_'):
        xs[1].zero_()
        xs[1][..., 0] = 1
        xs[6].fill_(1.)
        xs[2].zero_()
        xs[2][..., 0] = 2. if bad == 'value_boundary' else 2.01
    elif bad in ('q_nan', 'k_inf', 'v_nan'):
        xs[{'q_nan': 0, 'k_inf': 1, 'v_nan': 2}[bad]].fill_(math.inf if bad == 'k_inf' else math.nan)
    elif bad == 'integer_q':
        xs[0] = xs[0].long()
    elif bad.startswith('scale_'):
        kwargs['scale'] = math.nan if bad == 'scale_nan' else torch.tensor(1.)
    elif bad.startswith('empty_'):
        if bad == 'empty_key_dimension':
            for i in (0, 1, 4):
                xs[i] = xs[i][..., :0]
        elif bad == 'empty_value_dimension':
            xs[2] = xs[2][..., :0]
        elif bad == 'empty_head':
            for i in range(5):
                xs[i] = xs[i][:, :, :0]
        else:
            for i in range(5):
                xs[i] = xs[i][:0]
    with pytest.raises(ValueError):
        run(xs, **kwargs)


@pytest.mark.parametrize('bad', ['batch', 'rank', 'short', 'float', 'bool', 'start', 'end', 'decreasing', 'negative', 'state'])
@pytest.mark.parametrize('check_inputs', [False, True])
def test_recurrent_invalid_packed_boundaries(bad, check_inputs):
    xs = make_case(B=2 if bad == 'batch' else 1, T=3)
    cu = torch.tensor([0, 3])
    variants = {
        'rank': cu.reshape(1, 2), 'short': torch.tensor([0]), 'float': cu.double(), 'bool': cu.bool(),
        'start': torch.tensor([1, 3]), 'end': torch.tensor([0, 2]), 'decreasing': torch.tensor([0, 2, 1, 3]),
        'negative': torch.tensor([0, -1, 3]), 'state': torch.tensor([0, 1, 3]),
    }
    with pytest.raises(ValueError):
        run(xs, cu_seqlens=variants.get(bad, cu), check_inputs=check_inputs)


@pytest.mark.parametrize('layout', ['dense', 'packed', 'packed_empty'])
@pytest.mark.parametrize('dtype,fp64_source', [
    pytest.param(torch.float32, None, id='fp32'),
    pytest.param(torch.bfloat16, None, id='bf16'),
    pytest.param(torch.float16, None, id='fp16'),
    pytest.param(torch.float64, None, id='fp64'),
    *(pytest.param(torch.float32, name, id=f'fp64_{name}') for name in NAMES),
])
def test_recurrent_cpu_autocast_values_and_seven_gradients(layout, dtype, fp64_source):
    packed = layout != 'dense'
    xs = make_case(B=1 if packed else 2, T=0 if layout == 'packed_empty' else 7, K=32, V=64, g=32)
    cu = None
    if packed:
        cu = torch.tensor([0, 0, 0, 0, 0] if layout == 'packed_empty' else [0, 0, 3, 7, 7])
        xs = (*xs[:5], make_case(B=4, K=32, V=64, g=32)[5], xs[6])
    xs = tuple(x.detach().to(torch.float64 if name == fp64_source else dtype).requires_grad_() for name, x in zip(NAMES, xs))
    amp_xs = leaves(xs)
    math_dtype = torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32
    promoted = tuple(x.detach().to(math_dtype).requires_grad_() for x in xs)
    expected = run(xs, g=32, cu_seqlens=cu)
    explicit = run(promoted, g=32, cu_seqlens=cu)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = run(amp_xs, g=32, cu_seqlens=cu)
        assert torch.is_autocast_enabled('cpu')
    assert actual[0].dtype == xs[0].dtype
    assert actual[1].dtype == math_dtype
    for a, e, p in zip(actual, expected, explicit):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
        torch.testing.assert_close(e, p.to(e.dtype), atol=0, rtol=0)
    cotangents = tuple(torch.randn_like(x) for x in expected)
    expected_grads = torch.autograd.grad(expected, xs, cotangents)
    actual_grads = torch.autograd.grad(actual, amp_xs, cotangents)
    explicit_grads = torch.autograd.grad(explicit, promoted, tuple(x.to(math_dtype) for x in cotangents))
    for name, a, e, p in zip(NAMES, actual_grads, expected_grads, explicit_grads):
        assert torch.isfinite(a).all(), name
        torch.testing.assert_close(a, e, atol=0, rtol=0)
        torch.testing.assert_close(e, p.to(e.dtype), atol=0, rtol=0)


@pytest.mark.parametrize('fail', [False, True], ids=['return', 'exception'])
def test_recurrent_cpu_autocast_restoration(monkeypatch, fail):
    xs = tuple(x.detach().float().requires_grad_() for x in make_case(B=1))
    einsum = torch.einsum
    calls = []

    def checked_einsum(equation, *operands):
        calls.append(equation)
        assert not torch.is_autocast_enabled('cpu')
        assert all(x.dtype == torch.float32 for x in operands)
        if fail:
            raise RuntimeError('autocast restoration probe')
        return einsum(equation, *operands)

    monkeypatch.setattr(torch, 'einsum', checked_einsum)
    before = torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu'), torch.is_autocast_cache_enabled()
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
        if fail:
            with pytest.raises(RuntimeError, match='autocast restoration probe'):
                run(xs)
        else:
            run(xs)
        assert torch.is_autocast_enabled('cpu')
        assert torch.get_autocast_dtype('cpu') == torch.bfloat16
        assert not torch.is_autocast_cache_enabled()
        assert torch.mm(torch.ones(2, 2), torch.ones(2, 2)).dtype == torch.bfloat16
    assert calls
    assert before == (torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu'), torch.is_autocast_cache_enabled())
