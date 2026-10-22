# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU-only algebra and derivative tests, independent of the FLA GPU imports.

Run without the repository's device-specific conftest:
    pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_reference.py
These tests do NOT claim to execute or compile Triton kernels.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/naive.py'
_spec = importlib.util.spec_from_file_location('hyperbolic_reference', _path.resolve())
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
recurrent = ref.naive_recurrent_hyperbolic_delta_rule
chunk = ref.naive_chunk_hyperbolic_delta_rule


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_case(T=19, K=7, V=5, B=2, H=2, c=1.0, vector=False, dtype=torch.float64):
    torch.manual_seed(73)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = 0.25 * torch.randn(B, T, H, V, dtype=dtype).tanh() / max(c, 1) ** 0.5
    beta = torch.rand((B, T, H, V) if vector else (B, T, H), dtype=dtype)
    s = torch.randn(B, H, K, V, dtype=dtype)
    s = 0.1 * torch.nn.functional.normalize(s, dim=-2) / max(c, 1) ** 0.5
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, s))


def eval_op(op, args, **kwargs):
    q, k, v, beta, s = args
    return op(q, k, v, beta, initial_state=s, output_final_state=True, **kwargs)


@pytest.mark.parametrize('T', [0, 1, 17, 35])
@pytest.mark.parametrize('c', [0.0, 0.25, 1.0, 2.5])
@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('size', [1, 16, 32])
def test_chunk_recurrent_values_and_gradients(T, c, vector, size):
    args = make_case(T=T, c=c, vector=vector)
    a, sa = eval_op(recurrent, args, curvature=c)
    b, sb = eval_op(chunk, args, curvature=c, chunk_size=size)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(sa, sb, atol=1e-11, rtol=1e-10)
    do, ds = torch.randn_like(a), torch.randn_like(sa)
    ga = torch.autograd.grad((a, sa), args, (do, ds))
    gb = torch.autograd.grad((b, sb), args, (do, ds))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('c', [0.0, 0.3, 1.0])
def test_original_homogeneous_equations(c):
    args = make_case(T=31, B=1, c=c)
    q, k, v, beta, state = args
    z = 1.0 + torch.rand_like(state[0, :, 0])
    m = state[0] * z.unsqueeze(-2)
    out = []
    for t in range(q.shape[1]):
        kt, vt = k[0, t], v[0, t]
        a = kt.square().sum(-1, keepdim=True) - c * vt.square()
        b = beta[0, t, :, None] * (z * vt - torch.einsum('hk,hkv->hv', kt, m)) / a
        m = m + kt.unsqueeze(-1) * b.unsqueeze(-2)
        z = z + c * vt * b
        out.append(torch.einsum('hk,hkv->hv', q[0, t], m) / z)
    actual, last = eval_op(recurrent, args, curvature=c, scale=1)
    torch.testing.assert_close(actual[0], torch.stack(out), atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(last[0], m / z.unsqueeze(-2), atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('op', [recurrent, chunk])
def test_projection_zero_gate_and_flat_limit(op):
    q, k, v, beta, s = make_case(T=1)
    o, _ = op(k, k, v, torch.ones_like(beta), initial_state=s, output_final_state=True, scale=1)
    torch.testing.assert_close(o, v, atol=1e-11, rtol=1e-10)
    o, ht = op(q, k, v, torch.zeros_like(beta), initial_state=s, output_final_state=True, scale=1)
    torch.testing.assert_close(ht, s)
    torch.testing.assert_close(o[:, 0], torch.einsum('bhk,bhkv->bhv', q[:, 0], s))
    kk = 2.7 * k
    r = v[:, 0] - torch.einsum('bhk,bhkv->bhv', kk[:, 0], s)
    delta = beta[:, 0, :, None] * r / kk[:, 0].square().sum(-1, keepdim=True)
    expected = s + kk[:, 0, :, :, None] * delta.unsqueeze(-2)
    _, ht = op(q, kk, v, beta, curvature=0, initial_state=s, output_final_state=True)
    torch.testing.assert_close(ht, expected)


@pytest.mark.parametrize('op', [recurrent, chunk])
def test_packed_empty_sequences_and_initial_state_gradient(op):
    q, k, v, beta, _ = make_case(T=38, B=1)
    cu = torch.tensor([0, 0, 1, 18, 18, 38], dtype=torch.int64)
    s = torch.zeros(5, 2, 7, 5, dtype=q.dtype, requires_grad=True)
    args = q, k, v, beta, s
    out, ht = eval_op(op, args, cu_seqlens=cu)
    outputs, finals = [], []
    for n, (a, b) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        y, h = op(q[:, a:b], k[:, a:b], v[:, a:b], beta[:, a:b], initial_state=s[n:n+1], output_final_state=True)
        outputs.append(y)
        finals.append(h)
    torch.testing.assert_close(out, torch.cat(outputs, dim=1))
    torch.testing.assert_close(ht, torch.cat(finals))
    grad = torch.autograd.grad(ht.sum() + out.sum(), s)[0]
    torch.testing.assert_close(grad[0], torch.ones_like(grad[0]))
    torch.testing.assert_close(grad[3], torch.ones_like(grad[3]))


@pytest.mark.parametrize('op', [recurrent, chunk])
def test_cache_causality_and_noncontiguous(op):
    args = make_case(T=33, vector=True)
    q, k, v, beta, s = args
    full, final = eval_op(op, args)
    a, mid = op(q[:, :13], k[:, :13], v[:, :13], beta[:, :13], initial_state=s, output_final_state=True)
    b, ht = op(q[:, 13:], k[:, 13:], v[:, 13:], beta[:, 13:], initial_state=mid, output_final_state=True)
    torch.testing.assert_close(torch.cat((a, b), 1), full, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(ht, final, atol=1e-10, rtol=1e-9)
    noncontig = tuple(torch.stack((x, torch.zeros_like(x)), -1)[..., 0] for x in args)
    y, sy = eval_op(op, noncontig)
    torch.testing.assert_close(y, full)
    torch.testing.assert_close(sy, final)
    v_changed = v.detach().clone()
    v_changed[:, 20:] *= -1
    yc, _ = op(q, k, v_changed, beta, initial_state=s)
    torch.testing.assert_close(yc[:, :20], full[:, :20])


def test_interference_cancellation_and_counterexample():
    dtype = torch.float64
    old = torch.tensor([0.8, 0.0], dtype=dtype).reshape(1, 1, 2, 1)
    k = torch.tensor([0.6, 0.8], dtype=dtype).reshape(1, 1, 1, 2)
    q = torch.tensor([1.0, 0.0], dtype=dtype).reshape_as(k)
    v = torch.full((1, 1, 1, 1), 0.75, dtype=dtype)
    o, ht = recurrent(q, k, v, torch.ones(1, 1, 1, dtype=dtype), scale=1, initial_state=old, output_final_state=True)
    torch.testing.assert_close(o, torch.full_like(o, 0.8))
    torch.testing.assert_close(ht.flatten(), torch.tensor([0.8, 0.3375], dtype=dtype))
    old = torch.tensor([0.5, 0.0], dtype=dtype).reshape_as(old)
    k = torch.tensor([0.0, 1.0], dtype=dtype).reshape_as(k)
    v.fill_(0.5)
    o, _ = recurrent(q, k, v, torch.ones(1, 1, 1, dtype=dtype), scale=1, initial_state=old)
    torch.testing.assert_close(o, torch.full_like(o, 0.375))


def manual_vjp(args, do, ds, c):
    """Independent PyTorch transcription of the Triton VJP equations, not a GPU test."""
    q, k, v, beta, h0 = args
    T = q.shape[1]
    bv = beta if beta.ndim == 4 else beta.unsqueeze(-1).expand_as(v)
    states = [h0]
    for t in range(T):
        st, kt, vt = states[-1], k[:, t], v[:, t]
        a = kt.square().sum(-1, keepdim=True) - c * vt.square()
        r = vt - torch.einsum('bhk,bhkv->bhv', kt, st)
        u = bv[:, t] * r / a
        states.append((st + kt.unsqueeze(-1) * u.unsqueeze(-2)) / (1 + c * vt * u).unsqueeze(-2))
    dq, dk, dv, db = (torch.zeros_like(x) for x in (q, k, v, bv))
    ds = ds.clone()
    for t in range(T - 1, -1, -1):
        st, cur, kt, vt = states[t], states[t + 1], k[:, t], v[:, t]
        a = kt.square().sum(-1, keepdim=True) - c * vt.square()
        r = vt - torch.einsum('bhk,bhkv->bhv', kt, st)
        u = bv[:, t] * r / a
        d = 1 + c * vt * u
        dq[:, t] = (cur * do[:, t].unsqueeze(-2)).sum(-1)
        ds = ds + q[:, t].unsqueeze(-1) * do[:, t].unsqueeze(-2)
        gnum = ds / d.unsqueeze(-2)
        gden = -(ds * cur).sum(-2) / d
        gu = (gnum * kt.unsqueeze(-1)).sum(-2) + c * vt * gden
        gr, ga = gu * bv[:, t] / a, -gu * u / a
        dv[:, t] = c * u * gden + gr - 2 * c * vt * ga
        db[:, t] = gu * r / a
        dk[:, t] = (gnum * u.unsqueeze(-2) - st * gr.unsqueeze(-2)).sum(-1) + 2 * kt * ga.sum(-1, keepdim=True)
        ds = gnum - kt.unsqueeze(-1) * gr.unsqueeze(-2)
    return dq, dk, dv, db if beta.ndim == 4 else db.sum(-1), ds


@pytest.mark.parametrize('c', [0.0, 0.4, 1.0, 2.0])
@pytest.mark.parametrize('gate', ['random', 'zero', 'one'])
@pytest.mark.parametrize('vector', [False, True])
def test_analytic_backward_against_autograd(c, gate, vector):
    args = list(make_case(T=19, c=c, vector=vector))
    if gate != 'random':
        args[3] = torch.full_like(args[3], 0.0 if gate == 'zero' else 1.0, requires_grad=True)
    o, ht = eval_op(recurrent, args, curvature=c, scale=1)
    do, ds = torch.randn_like(o), torch.randn_like(ht)
    expected = torch.autograd.grad((o, ht), args, (do, ds))
    with torch.no_grad():
        actual = manual_vjp(args, do, ds, c)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize('op', [recurrent, chunk])
def test_finite_difference_gradcheck(op):
    args = make_case(T=3, K=3, V=2, B=1, H=1, vector=True)
    assert torch.autograd.gradcheck(lambda *xs: eval_op(op, xs, scale=1), args, eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('c', [0.0, 0.1, 1.0, 4.0])
def test_safe_layer_preprocessing(dtype, c):
    raw_k = torch.zeros(1, 3, 2, 7, dtype=dtype)
    raw_k[:, 1:] = torch.randn_like(raw_k[:, 1:])
    raw_v = torch.randn(1, 3, 2, 5, dtype=dtype) * 1000
    k, v = ref.prepare_key_value(raw_k, raw_v, c)
    torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k[..., 0]))
    assert (k.square().sum(-1, keepdim=True) - c * v.square() > 0.7).all()


def test_long_sequence_ball_invariant():
    q, k, v, beta, s = make_case(T=2049, B=1, H=1, K=4, V=3, dtype=torch.float32)
    a, ha = recurrent(q, k, v, beta, initial_state=s, output_final_state=True)
    b, hb = chunk(q, k, v, beta, initial_state=s, output_final_state=True, chunk_size=16)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (ha.square().sum(-2) < 1).all() and (hb.square().sum(-2) < 1).all()
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(ha, hb, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('bad', ['curvature', 'gate', 'key', 'value', 'nan', 'state', 'shape', 'cu'])
def test_invalid_inputs_fail_explicitly(bad):
    q, k, v, beta, s = [x.detach().clone() for x in make_case(T=3, B=1)]
    kwargs = dict(initial_state=s, curvature=1.0)
    if bad == 'curvature':
        kwargs['curvature'] = -1.0
    elif bad == 'gate':
        beta.fill_(1.1)
    elif bad == 'key':
        k.zero_()
    elif bad == 'value':
        v.fill_(2.0)
    elif bad == 'nan':
        q.fill_(float('nan'))
    elif bad == 'state':
        s.fill_(2.0)
    elif bad == 'shape':
        k = k[..., :3]
    elif bad == 'cu':
        kwargs['cu_seqlens'] = torch.tensor([0, 4])
    with pytest.raises(ValueError):
        recurrent(q, k, v, beta, **kwargs)


@pytest.mark.parametrize('size', [16, 32])
def test_packed_checkpoint_addresses_do_not_overlap(size):
    torch.manual_seed(29)
    for _ in range(100):
        lengths = torch.randint(0, 101, (31,)).tolist()
        total, bos, used = sum(lengths), 0, set()
        allocation = (total + size - 1) // size + len(lengths)
        for n, length in enumerate(lengths):
            for block in range((length + size - 1) // size):
                index = bos // size + n + block
                assert index not in used and index < allocation
                used.add(index)
            bos += length


def test_learnable_curvature_reference_gradient():
    q, k, v, beta, state = make_case(T=4, K=4, V=2, B=1, H=1, dtype=torch.float64)
    raw = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    curvature = torch.nn.functional.softplus(raw)
    out, final = recurrent(q, k, v, beta, curvature=curvature, initial_state=state, output_final_state=True)
    grad = torch.autograd.grad(out.square().sum() + final.square().sum(), raw)[0]
    assert torch.isfinite(grad) and grad.abs() > 0


def test_protected_reference_replaces_bank_and_is_differentiable():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    q, k, v, beta, state = make_case(T=33, K=5, V=3, B=1, H=1, c=0.5)
    k = k.detach().requires_grad_()
    v = v.detach().requires_grad_()
    selection_scores = torch.linspace(-1, 1, 33, dtype=k.dtype).reshape(1, 33, 1).requires_grad_()
    out, final, pk, pv, scores, count = protected.naive_protected_chunkwise_hyperbolic_delta_rule(
        q, k, v, beta, curvature=0.5, initial_state=state,
        protection_rank=3, chunk_size=16, checkpoint=False, selection_scores=selection_scores,
    )
    assert out.shape == v.shape and final.shape == state.shape
    assert pk.shape == (1, 1, 3, 5) and pv.shape == (1, 1, 3, 3)
    assert scores.shape == (1, 1, 3) and count.shape == (1, 1)
    loss = out.square().mean() + final.square().mean() + pk.square().mean() + pv.square().mean()
    gk, gv, gs = torch.autograd.grad(loss, (k, v, selection_scores))
    assert torch.isfinite(gk).all() and torch.isfinite(gv).all() and torch.isfinite(gs).all()
    assert gk.abs().sum() > 0 and gv.abs().sum() > 0
    assert gs.abs().sum() > 0


def test_protected_checkpoint_matches_uncheckpointed_forward_and_gradients():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    q, k, v, beta, state = make_case(T=19, K=5, V=2, B=1, H=1, c=0.5)
    def run(checkpoint):
        q1, k1, v1, b1, s1 = [x.detach().clone().requires_grad_() for x in (q, k, v, beta, state)]
        c = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        result = protected.naive_protected_chunkwise_hyperbolic_delta_rule(
            q1, k1, v1, b1, curvature=c, initial_state=s1, protection_rank=2,
            chunk_size=16, checkpoint=checkpoint,
        )
        loss = result[0].square().sum() + result[1].square().sum() + result[3].square().sum()
        grads = torch.autograd.grad(loss, (q1, k1, v1, b1, s1, c))
        return result[:2], grads
    (out_a, state_a), grad_a = run(False)
    (out_b, state_b), grad_b = run(True)
    torch.testing.assert_close(out_a, out_b, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(state_a, state_b, atol=1e-8, rtol=1e-7)
    for a, b in zip(grad_a, grad_b):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


def test_protected_reference_forward_and_gradients():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    q, k, v, beta, state = make_case(T=7, K=5, V=3, B=1, H=1, c=0.5)
    raw_c = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    c = torch.nn.functional.softplus(raw_c)
    out, final, pk, pv, scores, pc = protected.protected_recurrent_hyperbolic_delta_rule(
        q, k, v, beta, curvature=c, initial_state=state, protection_rank=2, mode='naive',
    )
    assert out.shape == v.shape
    assert final.shape == state.shape
    assert pk.shape == (1, 1, 2, 5)
    assert pv.shape == (1, 1, 2, 3)
    assert scores.shape == (1, 1, 2)
    assert pc.shape == (1, 1)
    grad = torch.autograd.grad(out.square().sum() + final.square().sum(), (q, k, v, beta, state, raw_c))
    assert all(torch.isfinite(g).all() for g in grad)
    assert grad[-1].abs() > 0




def test_shared_key_gram_projection_matches_per_value_solve():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    torch.manual_seed(101)
    B, T, H, K, V, R = 1, 5, 1, 7, 3, 2
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1)
    v = 0.1 * torch.randn(B, T, H, V, dtype=torch.float64)
    bank_k = torch.nn.functional.normalize(torch.randn(B, H, R, K, dtype=torch.float64), dim=-1)
    bank_v = 0.1 * torch.randn(B, H, R, V, dtype=torch.float64)
    scores = torch.zeros(B, H, R, dtype=torch.float64)
    c = torch.tensor(0.4, dtype=torch.float64)
    kb, vb = protected._project_chunk_torch(k, v, bank_k, bank_v, scores, c, eps=1e-8)
    for t in range(T):
        for vi in range(V):
            gram = bank_k[0, 0] @ bank_k[0, 0].T - c * bank_v[0, 0, :, vi, None] * bank_v[0, 0, None, :, vi]
            rhs = bank_k[0, 0] @ k[0, t, 0] - c * bank_v[0, 0, :, vi] * v[0, t, 0, vi]
            alpha = torch.linalg.solve(gram + 1e-8 * torch.eye(R, dtype=torch.float64), rhs)
            ref_k = k[0, t, 0] - bank_k[0, 0].T @ alpha
            ref_v = v[0, t, 0, vi] - bank_v[0, 0, :, vi] @ alpha
            torch.testing.assert_close(kb[0, t, 0, vi], ref_k, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(vb[0, t, 0, vi], ref_v, atol=1e-7, rtol=1e-7)


def test_shared_key_projection_vjp_is_finite_and_differentiable():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    torch.manual_seed(211)
    B, T, H, K, V, R = 2, 4, 2, 6, 3, 2
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1).requires_grad_()
    v = (0.03 * torch.randn(B, T, H, V, dtype=torch.float64)).requires_grad_()
    bank_k = torch.nn.functional.normalize(torch.randn(B, H, R, K, dtype=torch.float64), dim=-1).requires_grad_()
    bank_v = (0.03 * torch.randn(B, H, R, V, dtype=torch.float64)).requires_grad_()
    scores = torch.zeros(B, H, R, dtype=torch.float64)
    curvature = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    kb, vb = protected._project_chunk_torch(k, v, bank_k, bank_v, scores, curvature, eps=1e-8)
    gk = torch.randn_like(kb)
    gv = torch.randn_like(vb)
    grads = torch.autograd.grad((kb, vb), (k, v, bank_k, bank_v, curvature), (gk, gv))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[2].abs().sum() > 0 and grads[3].abs().sum() > 0
    assert grads[4].abs() > 0


def test_smooth_bank_update_exact_autograd_and_gradcheck():
    protected_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_reference', protected_path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    torch.manual_seed(307)
    B, H, R, K, V = 1, 2, 2, 3, 2
    bank_k = torch.randn(B, H, R, K, dtype=torch.float64, requires_grad=True)
    bank_v = torch.randn(B, H, R, V, dtype=torch.float64, requires_grad=True)
    bank_scores = torch.randn(B, H, R, dtype=torch.float64, requires_grad=True)
    cand_k = torch.randn(B, 3, H, K, dtype=torch.float64, requires_grad=True)
    cand_v = torch.randn(B, 3, H, V, dtype=torch.float64, requires_grad=True)
    cand_scores = torch.randn(B, 3, H, dtype=torch.float64, requires_grad=True)
    def fn(bk, bv, bs, ck, cv, cs):
        out = protected._select_bank_smooth(bk, bv, bs, ck, cv, cs, temperature=0.7)
        return out[0], out[1], out[2]
    assert torch.autograd.gradcheck(fn, (bank_k, bank_v, bank_scores, cand_k, cand_v, cand_scores), eps=1e-6, atol=1e-4, rtol=1e-3)
    result = fn(bank_k, bank_v, bank_scores, cand_k, cand_v, cand_scores)
    grads = torch.autograd.grad(sum(x.square().sum() for x in result),
        (bank_k, bank_v, bank_scores, cand_k, cand_v, cand_scores))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[2].abs().sum() > 0 and grads[5].abs().sum() > 0


def test_protected_wy_realized_readout_algebra_cpu():
    """Prove the fused WY readout formula against the PyTorch recurrence."""
    import torch
    path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_protected_wy', path.resolve())
    protected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protected)
    torch.manual_seed(555)
    B, T, H, K, V, R = 1, 4, 1, 6, 2, 2
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1)
    v = 0.03 * torch.randn(B, T, H, V, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B, T, H, V, dtype=torch.float64))
    state = torch.zeros(B, H, K, V, dtype=torch.float64)
    bank_k = torch.zeros(B, H, R, K, dtype=torch.float64)
    bank_v = torch.zeros(B, H, R, V, dtype=torch.float64)
    bank_s = torch.full((B, H, R), -1e6, dtype=torch.float64)
    curvature = torch.tensor(0.2, dtype=torch.float64)
    kp, vp = protected._project_chunk_torch(k, v, bank_k, bank_v, bank_s, curvature)
    ref_out, ref_state, ref_realized = protected._recurrent_chunk_torch(
        q, kp, vp, beta, state, curvature, K ** -0.5, readout_k=k,
    )
    out_by_channel, state_by_channel, realized_by_channel = [], [], []
    for vi in range(V):
        kk = kp[0, :, 0, vi, :]
        vv = vp[0, :, 0, vi]
        bb = beta[0, :, 0, vi]
        qv = q[0, :, 0, :]
        a = kk.square().sum(-1) - curvature * vv.square()
        alpha = bb / a
        gram = kk @ kk.T - curvature * vv[:, None] * vv[None, :]
        mat = torch.eye(T, dtype=kk.dtype) + torch.tril(alpha[:, None] * gram, -1)
        rhs = alpha * (vv - kk @ state[0, 0, :, vi])
        coef = torch.linalg.solve_triangular(mat, rhs[:, None], upper=False).squeeze(-1)
        den = 1 + curvature * torch.cumsum(vv * coef, dim=0)
        rows = (state[0, 0, :, vi] + torch.cumsum(kk * coef[:, None], dim=0)) / den[:, None]
        out_by_channel.append(K ** -0.5 * (qv * rows).sum(-1))
        realized_by_channel.append((k[0, :, 0, :] * rows).sum(-1))
        state_by_channel.append(rows[-1])
    torch.testing.assert_close(ref_out[0, :, 0], torch.stack(out_by_channel, dim=-1))
    torch.testing.assert_close(ref_realized[0, :, 0], torch.stack(realized_by_channel, dim=-1))
    torch.testing.assert_close(ref_state[0, 0], torch.stack(state_by_channel, dim=-1))
