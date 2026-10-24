# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Independent CPU mathematics and CUDA tests for the protected trajectory split."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule/protected.py'
_spec = importlib.util.spec_from_file_location('protected_staged_reference', _path)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _gate(scores, candidate, temperature):
    rank = scores.shape[-1]
    count = (scores > -5e5).sum(-1)
    insert = F.one_hot(count.clamp(max=rank - 1), rank).to(scores.dtype)
    probability = torch.softmax(-scores / temperature, -1)
    sigma = torch.sigmoid((candidate[..., None] - scores) / temperature)
    return torch.where((count < rank)[..., None], insert, probability * sigma)


def _shared_forward(keys, selection, bank_keys, scores, chunk, temperature):
    snapshots, gates, tape = [], [], []
    for t in range(keys.shape[1]):
        if t % chunk == 0:
            snapshots.append((bank_keys, scores))
        gate = _gate(scores, selection[:, t], temperature)
        tape.append((bank_keys, scores))
        gates.append(gate)
        bank_keys = (1 - gate[..., None]) * bank_keys + gate[..., None] * keys[:, t, :, None]
        scores = (1 - gate) * scores + gate * selection[:, t, :, None]
    return snapshots, gates, tape, bank_keys, scores


def _separated(args, chunk, temperature):
    q, k, v, beta, state, pk, y, scores, selection, c = args
    snapshots, gates, _, pk_final, scores_final = _shared_forward(k, selection, pk, scores, chunk, temperature)
    outputs = []
    for index, start in enumerate(range(0, k.shape[1], chunk)):
        stop = min(start + chunk, k.shape[1])
        bank_k, bank_scores = snapshots[index]
        kp, vp = reference._project_chunk_torch(k[:, start:stop], v[:, start:stop], bank_k, y, bank_scores, c)
        output, state, realized = reference._recurrent_chunk_torch(
            q[:, start:stop], kp, vp, beta[:, start:stop], state, c, k.shape[-1] ** -.5, k[:, start:stop],
        )
        for t in range(start, stop):
            gate = gates[t]
            y = (1 - gate[..., None]) * y + gate[..., None] * realized[:, t - start, :, None]
        outputs.append(output)
    return torch.cat(outputs, 1), state, pk_final, y, scores_final


def _coupled(args, chunk, temperature):
    q, k, v, beta, state, pk, y, scores, selection, c = args
    outputs = []
    for start in range(0, k.shape[1], chunk):
        stop = min(start + chunk, k.shape[1])
        kc, vc = k[:, start:stop], v[:, start:stop]
        kp, vp = reference._project_chunk_torch(kc, vc, pk, y, scores, c)
        output, state, realized = reference._recurrent_chunk_torch(
            q[:, start:stop], kp, vp, beta[:, start:stop], state, c, k.shape[-1] ** -.5, kc,
        )
        pk, y, scores, _ = reference._select_bank_smooth(pk, y, scores, kc, realized, selection[:, start:stop], temperature)
        outputs.append(output)
    return torch.cat(outputs, 1), state, pk, y, scores


def _shared_vjp(keys, selection, snapshots, gates, tape, dpk, dgates, dfinal_k, dfinal_s, chunk, temperature):
    dk, dselection = torch.zeros_like(keys), torch.zeros_like(selection)
    dkey, dscore = dfinal_k.clone(), dfinal_s.clone()
    for t in reversed(range(keys.shape[1])):
        old_k, old_s = tape[t]
        gate = gates[t]
        candidate = selection[:, t]
        dg = dgates[t] + (dkey * (keys[:, t, :, None] - old_k)).sum(-1) + dscore * (candidate[..., None] - old_s)
        dk[:, t] = (gate[..., None] * dkey).sum(-2)
        dselection[:, t] = (gate * dscore).sum(-1)
        p = torch.softmax(-old_s / temperature, -1)
        sigma = torch.sigmoid((candidate[..., None] - old_s) / temperature)
        dsigma = dg * p * sigma * (1 - sigma) / temperature
        softmax_term = p * (dg * sigma - (p * dg * sigma).sum(-1, keepdim=True)) / temperature
        full = ((old_s > -5e5).sum(-1) == old_s.shape[-1])[..., None]
        dselection[:, t] += torch.where(full, dsigma, 0.).sum(-1)
        dscore = (1 - gate) * dscore - torch.where(full, softmax_term + dsigma, 0.)
        dkey = (1 - gate[..., None]) * dkey
        if t % chunk == 0:
            dkey += dpk[t // chunk]
    return dk, dselection, dkey, dscore


def _case(length=35, chunk=16, rank=2, occupied=2, curvature=.3, gate=None):
    torch.manual_seed(2003 + length + rank)
    b, h, key_dim, values = 1, 2, 7, 3
    q = torch.randn(b, length, h, key_dim, dtype=torch.float64)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(b, length, h, values, dtype=q.dtype).tanh() * .05
    beta = torch.rand_like(v) if gate is None else torch.full_like(v, gate)
    state = F.normalize(torch.randn(b, h, key_dim, values, dtype=q.dtype), dim=-2) * .05
    pk = F.normalize(torch.randn(b, h, rank, key_dim, dtype=q.dtype), dim=-1)
    y = torch.randn(b, h, rank, values, dtype=q.dtype) * .02
    scores = torch.randn(b, h, rank, dtype=q.dtype) * .1
    scores[..., occupied:] = -1e6
    selection = torch.randn(b, length, h, dtype=q.dtype) * .1
    c = torch.tensor(curvature, dtype=q.dtype)
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, state, pk, y, scores, selection, c))


@pytest.mark.parametrize('length,chunk,rank,occupied', [(1, 16, 1, 0), (17, 16, 2, 1), (35, 16, 3, 3), (35, 32, 2, 2)])
@pytest.mark.parametrize('curvature,gate', [(0., 0.), (.3, 1.), (1., None)])
def test_protected_trajectory_math_and_vjp(length, chunk, rank, occupied, curvature, gate):
    args = _case(length, chunk, rank, occupied, curvature, gate)
    expected = _coupled(args, chunk, .7)
    actual = _separated(args, chunk, .7)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-11, rtol=1e-10)
    seeds = tuple(torch.randn_like(x) for x in actual)
    expected_grad = torch.autograd.grad(expected, args, seeds, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, args, seeds)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-8)
        assert torch.isfinite(got).all()


@pytest.mark.parametrize('occupied', [0, 1, 3])
def test_protected_shared_trajectory_analytic_vjp(occupied):
    args = _case(rank=3, occupied=occupied)
    _, k, _, _, _, pk, _, scores, selection, _ = args
    snapshots, gates, tape, final_k, final_s = _shared_forward(k, selection, pk, scores, 16, .7)
    dpk = [torch.randn_like(p) for p, s in snapshots]
    dg = [torch.randn_like(g) for g in gates]
    dlast_k, dlast_s = torch.randn_like(final_k), torch.randn_like(final_s)
    expected = torch.autograd.grad(
        tuple(p for p, s in snapshots) + tuple(gates) + (final_k, final_s),
        (k, selection, pk, scores), tuple(dpk) + tuple(dg) + (dlast_k, dlast_s),
    )
    with torch.no_grad():
        actual = _shared_vjp(k, selection, snapshots, gates, tape, dpk, dg, dlast_k, dlast_s, 16, .7)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-9)


def test_protected_trajectory_gradcheck():
    args = _case(length=2, rank=1, occupied=1)
    assert torch.autograd.gradcheck(lambda *xs: _separated(xs, 1, .7), args, eps=1e-6, atol=1e-4, rtol=1e-3)


def _value_track(q, k, v, beta, state, bank_values, keys, active, gates, c, chunk):
    outputs, tapes = [], []
    rank = bank_values.shape[0]
    for block, start in enumerate(range(0, q.shape[0], chunk)):
        pk = keys[block] * active[block][:, None]
        y = bank_values * active[block]
        gram = pk @ pk.T + torch.diag(torch.where(active[block], 1e-5, 1.))
        ainv = torch.linalg.inv(gram)
        u = ainv @ y
        denominator = 1 - c * (y @ u)
        feasible = denominator > 1e-5
        ginv = ainv + torch.where(feasible, c / denominator, 0.) * u[:, None] * u[None, :]
        tokens = []
        for t in range(start, min(start + chunk, q.shape[0])):
            alpha = ginv @ (pk @ k[t] - c * y * v[t])
            kp, vp = k[t] - alpha @ pk, v[t] - alpha @ y
            projected = feasible & ((kp @ kp - c * vp * vp) > 1e-5)
            kp, vp, alpha = torch.where(projected, kp, k[t]), torch.where(
                projected, vp, v[t]), torch.where(projected, alpha, 0.)
            a = kp @ kp - c * vp * vp
            valid = a > 1e-5
            a = torch.where(valid, a, 1.)
            residual = vp - kp @ state
            weight = torch.where(valid, beta[t] * residual / a, 0.)
            den = 1 + c * vp * weight
            next_state = (state + kp * weight) / den
            realized = k[t] @ next_state
            tokens.append((state, bank_values, kp, vp, alpha, projected, a,
                          valid, residual, weight, den, next_state, realized))
            outputs.append(q.shape[-1] ** -.5 * (q[t] @ next_state))
            state = next_state
            bank_values = (1 - gates[t]) * bank_values + gates[t] * realized
        assert y.shape == (rank,)
        tapes.append((pk, y, ginv, tokens))
    return torch.stack(outputs), state, bank_values, tapes


def _value_track_vjp(q, k, v, beta, keys, active, gates, c, tapes, do, ds, dy, chunk):
    dq, dk, dv, db = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(beta)
    dkeys, dg, dc = torch.zeros_like(keys), torch.zeros_like(gates), torch.zeros_like(c)
    ds, dy = ds.clone(), dy.clone()
    for block in reversed(range(len(tapes))):
        pk, y, ginv, tokens = tapes[block]
        dpk, dyp = torch.zeros_like(pk), torch.zeros_like(y)
        for local in reversed(range(len(tokens))):
            t = block * chunk + local
            prev, old_y, kp, vp, alpha, projected, a, valid, residual, weight, den, now, realized = tokens[local]
            dread = (gates[t] * dy).sum()
            dg[t] = dy * (realized - old_y)
            dy = (1 - gates[t]) * dy
            dq[t] = q.shape[-1] ** -.5 * now * do[t]
            direct = now * dread
            ds = ds + q.shape[-1] ** -.5 * q[t] * do[t] + k[t] * dread
            gnum = ds / den
            gden = -(ds @ now) / den
            gw = gnum @ kp + c * vp * gden
            gr = torch.where(valid, gw * beta[t] / a, 0.)
            ga = torch.where(valid, -gw * weight / a, 0.)
            gkp = gnum * weight - prev * gr + 2 * kp * ga
            gvp = c * weight * gden + gr - 2 * c * vp * ga
            db[t] = torch.where(valid, gw * residual / a, 0.)
            dc += vp * weight * gden - vp * vp * ga
            ds = gnum - kp * gr
            z = torch.where(projected, ginv @ (-pk @ gkp - y * gvp), 0.)
            dk[t] = gkp + z @ pk + direct
            dv[t] = gvp - c * (y @ z)
            dpk += active[block][:, None] * (
                -alpha[:, None] * gkp[None, :] + z[:, None] * (k[t] - alpha @ pk)[None, :]
                - alpha[:, None] * (z @ pk)[None, :]
            )
            ya, yz = y @ alpha, y @ z
            dyp += active[block] * (-alpha * gvp - c * z * v[t] + c * z * ya + c * alpha * yz)
            dc += yz * (ya - v[t])
        dy += dyp
        dkeys[block] = dpk
    return dq, dk, dv, db, ds, dy, dkeys, dg, dc


@pytest.mark.parametrize('curvature,gate,occupied,bank_scale', [(0., 0., 0, 1.), (.3, 1., 1, 1.), (1., None, 2, 1.),
                                                                (1., None, 2, 1000.)])
def test_protected_value_trajectory_analytic_vjp(curvature, gate, occupied, bank_scale):
    args = _case(length=35, rank=2, occupied=occupied, curvature=curvature, gate=gate)
    q, k, v, beta, state, pk, y, scores, selection, c = args
    snapshots, gates, _, _, _ = _shared_forward(k, selection, pk, scores, 16, .7)
    active = torch.stack([s[0, 0] > -5e5 for p, s in snapshots])
    inputs = tuple(x.detach().requires_grad_() for x in (
        q[0, :, 0], k[0, :, 0], v[0, :, 0, 0], beta[0, :, 0, 0], state[0, 0, :, 0], y[0, 0, :, 0] * bank_scale,
        torch.stack([p[0, 0] for p, s in snapshots]), torch.stack([g[0, 0] for g in gates]), c,
    ))
    qv, kv, vv, bv, sv, yv, pvs, gs, cv = inputs
    out, final, bank_final, tapes = _value_track(qv, kv, vv, bv, sv, yv, pvs, active, gs, cv, 16)
    do, ds, dy = torch.randn_like(out), torch.randn_like(final), torch.randn_like(bank_final)
    expected = torch.autograd.grad((out, final, bank_final), inputs, (do, ds, dy))
    with torch.no_grad():
        actual = _value_track_vjp(qv, kv, vv, bv, pvs, active, gs, cv, tapes, do, ds, dy, 16)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-8)


def _legacy_kernels(args, chunk, temperature):
    from fla.ops.hyperbolic_delta_rule.protected_triton import _run_value_parallel_chunk

    q, k, v, beta, state, pk, y, scores, selection, c = args
    outputs = []
    for start in range(0, q.shape[1], chunk):
        stop = min(start + chunk, q.shape[1])
        qc, kc, vc, bc, sc = (x[:, start:stop].contiguous() for x in (q, k, v, beta, selection))
        out, state, pk, y, scores = _run_value_parallel_chunk(
            qc, kc, vc, bc, state, pk, y, scores, sc, c, k.shape[-1] ** -.5, temperature,
        )
        outputs.append(out)
    return torch.cat(outputs, 1), state, pk, y, scores


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires a separate TRITON_INTERPRET=1 process')
@pytest.mark.parametrize('length,chunk,window,occupied,curvature,gate', [
    (1, 16, 1, 0, .3, None), (35, 16, 2, 1, .3, None), (35, 32, 1, 2, .3, None),
    (17, 16, 1, 2, 0., 0.), (17, 16, 2, 2, .3, 1.),
])
def test_protected_trajectory_interpreter(length, chunk, window, occupied, curvature, gate):
    from fla.ops.hyperbolic_delta_rule.protected_staged import _ProtectedStaged

    rank = 1 if length == 1 else 3 if length == 35 and chunk == 16 else 2
    args = _case(length=length, rank=rank, occupied=occupied, curvature=curvature, gate=gate)
    if length == 35 and chunk == 16:
        args = tuple(torch.cat((x, -x if i in (0, 1, 2, 4, 5, 6) else x), 0) if x.ndim else x
                     for i, x in enumerate(args))
    args = tuple(x.detach().float().requires_grad_() for x in args)
    q, k, v, beta, state, pk, y, scores, selection, c = args
    actual = _ProtectedStaged.apply(q, k, v, beta, state, pk, y, scores, selection, c, k.shape[-1] ** -.5, .7, chunk, window)
    expected = _legacy_kernels(args, chunk, .7)
    seeds = tuple(torch.randn_like(x) for x in actual)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-3)
    actual_grads = torch.autograd.grad(actual, args, seeds)
    expected_grads = torch.autograd.grad(expected, args, seeds)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires a separate TRITON_INTERPRET=1 process')
def test_protected_trajectory_interpreter_checkpoint():
    from torch.utils.checkpoint import checkpoint

    from fla.ops.hyperbolic_delta_rule.protected_staged import _run_staged_protected

    args = tuple(x.detach().float().requires_grad_() for x in _case(length=17, occupied=1))
    q, k, v, beta, state, pk, y, scores, selection, c = args
    call_args = (q, k, v, beta, state, pk, y, scores, selection, c, k.shape[-1] ** -.5, .7, 16)
    direct = _run_staged_protected(*call_args)
    recomputed = checkpoint(_run_staged_protected, *call_args, use_reentrant=False)
    for got, want in zip(recomputed, direct):
        torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-3)
    seeds = tuple(torch.randn_like(x) for x in direct)
    ga = torch.autograd.grad(recomputed, args, seeds)
    gb = torch.autograd.grad(direct, args, seeds)
    for got, want in zip(ga, gb):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)


def test_protected_staged_window_memory_bound():
    from fla.ops.hyperbolic_delta_rule.protected_staged import _reverse_window_chunks

    for batch, length in [(8, 2048), (4, 4096), (2, 8192), (1, 16384)]:
        window = _reverse_window_chunks(batch, 16, 128, 128, 4, 16, length // 16)
        per_chunk = 4 * batch * 16 * 128 * (16 * (2 * 128 + 4) + 4 * 128)
        assert 1 <= window <= min(length // 16, 16)
        assert window * per_chunk <= max(256 * 1024 ** 2, per_chunk)


def test_protected_staged_empty_cpu():
    from fla.ops.hyperbolic_delta_rule.protected_staged import _run_staged_protected

    q, k, v, beta, state, pk, y, scores, selection, c = _case(length=0)
    result = _run_staged_protected(q, k, v, beta, state, pk, y, scores, selection, c, 1., .7, 16)
    assert result[0].shape == v.shape
    assert not result[0].requires_grad
    for got, want in zip(result[1:], (state, pk, y, scores)):
        assert got is want
    grads = torch.autograd.grad(sum(x.sum() for x in result[1:]), (state, pk, y, scores))
    assert all(torch.equal(g, torch.ones_like(g)) for g in grads)


def test_protected_staged_cpu_public_guard(monkeypatch):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    args = _case(length=1)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '1')
    with pytest.raises(ValueError, match='requires CUDA'):
        triton_protected_chunkwise_hyperbolic_delta_rule(*args[:4])
    automatic = reference.protected_recurrent_hyperbolic_delta_rule(*args[:4], mode='auto', protection_rank=2)
    expected = reference.protected_recurrent_hyperbolic_delta_rule(*args[:4], mode='naive', protection_rank=2)
    for got, want in zip(automatic, expected):
        torch.testing.assert_close(got, want)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires a separate TRITON_INTERPRET=1 process')
@pytest.mark.parametrize('query_dtype,key_dtype,value_dtype,temperature', [
    (torch.float32, torch.float32, torch.float16, .7),
    (torch.bfloat16, torch.float32, torch.float32, .7),
    (torch.float16, torch.float16, torch.float16, 5e-4),
])
def test_protected_trajectory_interpreter_storage(query_dtype, key_dtype, value_dtype, temperature):
    from fla.ops.hyperbolic_delta_rule.protected_staged import _run_staged_protected

    args = [x.detach().float() for x in _case(length=17, occupied=1)]
    args[0], args[1], args[2] = args[0].to(query_dtype), args[1].to(key_dtype), args[2].to(value_dtype)
    args = tuple(x.requires_grad_() for x in args)
    q, k, v, beta, state, pk, y, scores, selection, c = args
    staged = _run_staged_protected(q, k, v, beta, state, pk, y, scores, selection, c, k.shape[-1] ** -.5, temperature, 16)
    legacy = _legacy_kernels(args, 16, temperature)
    assert staged[0].dtype == legacy[0].dtype == v.dtype
    actual = (staged[0].to(q.dtype), *staged[1:])
    expected = (legacy[0].to(q.dtype), *legacy[1:])
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-2)
    seeds = tuple(torch.randn_like(x) for x in actual)
    ga = torch.autograd.grad(actual, args, seeds)
    gb = torch.autograd.grad(expected, args, seeds)
    for got, want in zip(ga, gb):
        torch.testing.assert_close(got, want, atol=8e-3, rtol=8e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='requires a separate TRITON_INTERPRET=1 process')
@pytest.mark.parametrize('outputs', [(2, 4), (3,)])
def test_protected_trajectory_interpreter_bank_loss(outputs):
    from fla.ops.hyperbolic_delta_rule.protected_staged import _run_staged_protected

    args = tuple(x.detach().float().requires_grad_() for x in _case(length=17, occupied=1))
    q, k, v, beta, state, pk, y, scores, selection, c = args
    actual = _run_staged_protected(q, k, v, beta, state, pk, y, scores, selection, c, k.shape[-1] ** -.5, .7, 16)
    expected = _legacy_kernels(args, 16, .7)
    ga = torch.autograd.grad(sum(actual[i].square().sum() for i in outputs), args)
    gb = torch.autograd.grad(sum(expected[i].square().sum() for i in outputs), args)
    for got, want in zip(ga, gb):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('batch,length,heads,key_dim,value_dim,rank,chunk', [
    (2, 1, 2, 16, 7, 1, 16), (2, 35, 2, 32, 16, 3, 16), (1, 65, 1, 64, 48, 4, 32),
    (1, 129, 2, 128, 128, 4, 16), (1, 2049, 1, 128, 128, 4, 32), (1, 16385, 1, 128, 128, 4, 16),
    (1, 35, 2, 64, 32, 16, 16),
])
def test_protected_trajectory_cuda(dtype, batch, length, heads, key_dim, value_dim, rank, chunk, monkeypatch):
    from fla.ops.hyperbolic_delta_rule import protected_staged
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    torch.manual_seed(2007)
    q = torch.randn(batch, length, heads, key_dim, device='cuda', dtype=dtype)
    k = F.normalize(torch.randn_like(q).float(), dim=-1)
    v = torch.randn(batch, length, heads, value_dim, device='cuda').tanh() * .03
    beta = torch.rand_like(v)
    beta[:, 0] = 0.
    if length > 1:
        beta[:, 1] = 1.
    state = F.normalize(torch.randn(batch, heads, key_dim, value_dim, device='cuda'), dim=-2) * .03
    pk = F.normalize(torch.randn(batch, heads, rank, key_dim, device='cuda'), dim=-1)
    y = torch.randn(batch, heads, rank, value_dim, device='cuda') * .01
    scores = torch.zeros(batch, heads, rank, device='cuda')
    scores[..., -1] = -1e6
    selection = torch.randn(batch, length, heads, device='cuda') * .1
    c = torch.tensor(.2, device='cuda')
    args = tuple(x.detach().requires_grad_() for x in (q, k, v, beta, state, pk, y, scores, selection, c))
    calls = []
    execute = protected_staged._run_staged_protected

    def observed(*xs, **kwargs):
        calls.append(True)
        return execute(*xs, **kwargs)

    monkeypatch.setattr(protected_staged, '_run_staged_protected', observed)

    def run(enabled):
        monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', str(int(enabled)))
        return triton_protected_chunkwise_hyperbolic_delta_rule(
            *args[:4], curvature=args[9], initial_state=args[4], protected_keys=args[5], protected_values=args[6],
            protected_scores=args[7], selection_scores=args[8], protection_rank=rank, checkpoint_size=chunk,
            temperature=.7, kernel_strategy='value_parallel', checkpoint=False,
        )

    expected, actual = run(False), run(True)
    assert len(calls) == 1, 'the staged route must actually execute'
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.float(), want.float(), atol=5e-3, rtol=5e-2)
        assert torch.isfinite(got).all()
    seeds = tuple(torch.randn_like(x) for x in actual[:5])
    actual_grads = torch.autograd.grad(actual[:5], args, seeds)
    expected_grads = torch.autograd.grad(expected[:5], args, seeds)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got.float(), want.float(), atol=8e-3, rtol=8e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('fallback', [False, True])
def test_protected_trajectory_cuda_fp64_reference(fallback, monkeypatch):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    original = list(_case(length=17, occupied=2))
    if fallback:
        original[6] = original[6] * 1000.
    args = tuple(x.detach().cuda().float().requires_grad_() for x in original)
    high_precision = tuple(x.detach().cpu().double().requires_grad_() for x in args)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '1')
    actual = triton_protected_chunkwise_hyperbolic_delta_rule(
        *args[:4], curvature=args[9], initial_state=args[4], protected_keys=args[5], protected_values=args[6],
        protected_scores=args[7], selection_scores=args[8], protection_rank=2, checkpoint_size=16,
        temperature=.7, kernel_strategy='value_parallel', checkpoint=False,
    )[:5]
    expected = _coupled(high_precision, 16, .7)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want.float(), atol=5e-3, rtol=5e-2)
        assert torch.isfinite(got).all()
    seeds = tuple(torch.randn_like(x) for x in actual)
    actual_grads = torch.autograd.grad(actual, args, seeds)
    expected_grads = torch.autograd.grad(expected, high_precision, tuple(x.cpu().double() for x in seeds))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got.cpu(), want.float(), atol=8e-3, rtol=8e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('chunk', [16, 32])
def test_protected_trajectory_cuda_streaming(chunk, monkeypatch):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule
    from fla.ops.hyperbolic_delta_rule.streaming import protected_streaming

    q, k, v, beta, state, pk, y, scores, selection, c = (x.detach().cuda().float() for x in _case(length=37))
    beta = beta[..., 0].contiguous()
    q, k, v, beta, state, pk, y, scores, c = (x.requires_grad_() for x in (q, k, v, beta, state, pk, y, scores, c))
    targets = (q, k, v, beta, state, pk, y, scores, c)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '1')
    kwargs = dict(curvature=c, protection_rank=2, checkpoint_size=chunk, temperature=.7, checkpoint=False,
                  kernel_strategy='value_parallel')
    expected = triton_protected_chunkwise_hyperbolic_delta_rule(
        q, k, v, beta, initial_state=state, protected_keys=pk, protected_values=y, protected_scores=scores, **kwargs,
    )
    cache = dict(state=state, protected_keys=pk, protected_values=y, protected_scores=scores)
    pieces, position = [], 0
    for size in (0, 5, 1, 0, 11, 20, 0):
        piece, cache = protected_streaming(
            q[:, position:position + size], k[:, position:position + size], v[:, position:position + size],
            beta[:, position:position + size], cache=cache, mode='triton', **kwargs,
        )
        pieces.append(piece)
        position += size
    actual = (torch.cat(pieces, 1), cache['state'], cache['protected_keys'],
              cache['protected_values'], cache['protected_scores'])
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-3)
    seeds = tuple(torch.randn_like(x) for x in actual)
    actual_grad = torch.autograd.grad(actual, targets, seeds)
    expected_grad = torch.autograd.grad(expected[:5], targets, seeds)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
def test_protected_trajectory_cuda_checkpoint(monkeypatch):
    from fla.ops.hyperbolic_delta_rule import protected_staged
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    args = tuple(x.detach().cuda().float().requires_grad_() for x in _case(length=35, occupied=1))
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '1')
    calls = []
    execute = protected_staged._run_staged_protected

    def observed(*args, **kwargs):
        calls.append(True)
        return execute(*args, **kwargs)

    monkeypatch.setattr(protected_staged, '_run_staged_protected', observed)
    outputs = [triton_protected_chunkwise_hyperbolic_delta_rule(
        *args[:4], curvature=args[9], initial_state=args[4], protected_keys=args[5], protected_values=args[6],
        protected_scores=args[7], selection_scores=args[8], protection_rank=2, checkpoint_size=16,
        temperature=.7, kernel_strategy='value_parallel', checkpoint=enabled,
    ) for enabled in (False, True)]
    for got, want in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(got.float(), want.float(), atol=2e-4, rtol=2e-3)
    seeds = tuple(torch.randn_like(x) for x in outputs[0][:5])
    before_replay = len(calls)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '0')
    actual = torch.autograd.grad(outputs[1][:5], args, seeds)
    assert len(calls) > before_replay, 'checkpoint replay must retain the selected staged implementation'
    expected = torch.autograd.grad(outputs[0][:5], args, seeds)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('query_dtype,key_dtype,value_dtype,temperature', [
    (torch.float32, torch.float32, torch.float16, .7),
    (torch.bfloat16, torch.float32, torch.float32, .7),
    (torch.float16, torch.float16, torch.float16, 5e-4),
])
def test_protected_trajectory_cuda_storage(query_dtype, key_dtype, value_dtype, temperature, monkeypatch):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    args = [x.detach().cuda().float() for x in _case(length=17, occupied=1)]
    args[0], args[1], args[2] = args[0].to(query_dtype), args[1].to(key_dtype), args[2].to(value_dtype)
    args = tuple(x.requires_grad_() for x in args)

    def run(enabled):
        monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', str(int(enabled)))
        return triton_protected_chunkwise_hyperbolic_delta_rule(
            *args[:4], curvature=args[9], initial_state=args[4], protected_keys=args[5], protected_values=args[6],
            protected_scores=args[7], selection_scores=args[8], protection_rank=2, checkpoint_size=16,
            temperature=temperature, kernel_strategy='value_parallel', checkpoint=False,
        )[:5]

    expected, actual = run(False), run(True)
    assert actual[0].dtype == expected[0].dtype == query_dtype
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=5e-3, rtol=5e-2)
    seeds = tuple(torch.randn_like(x) for x in actual)
    ga, gb = torch.autograd.grad(actual, args, seeds), torch.autograd.grad(expected, args, seeds)
    for got, want in zip(ga, gb):
        torch.testing.assert_close(got, want, atol=8e-3, rtol=8e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('strategy', ['auto', 'fused'])
def test_protected_trajectory_cuda_strategy_fallback(strategy, monkeypatch):
    from fla.ops.hyperbolic_delta_rule import protected_staged
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    q, k, v, beta, state, pk, y, scores, selection, c = (x.detach().cuda().float() for x in _case(length=1))

    def unexpected(*args, **kwargs):
        raise AssertionError('explicit fused and small-Value auto must retain the original route')

    monkeypatch.setattr(protected_staged, '_run_staged_protected', unexpected)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '1')
    actual = triton_protected_chunkwise_hyperbolic_delta_rule(q, k, v, beta, protection_rank=2, kernel_strategy=strategy)
    monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', '0')
    expected = triton_protected_chunkwise_hyperbolic_delta_rule(q, k, v, beta, protection_rank=2, kernel_strategy=strategy)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.float(), want.float(), atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
def test_protected_trajectory_cuda_model_checkpoint(monkeypatch):
    from fla.models import HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM
    from fla.ops.hyperbolic_delta_rule import protected_staged

    torch.manual_seed(2011)
    config = HyperbolicDeltaNetConfig(
        hidden_size=32, head_dim=16, num_heads=2, num_hidden_layers=2, intermediate_size=64,
        vocab_size=97, use_short_conv=False, fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        protection_rank=2, protected_checkpoint=False, learnable_curvature=True, chunk_size=16,
    )
    model = HyperbolicDeltaNetForCausalLM(config).cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(1, config.vocab_size, (2, 35), device='cuda')
    calls = []
    execute = protected_staged._run_staged_protected

    def observed(*args, **kwargs):
        calls.append(True)
        return execute(*args, **kwargs)

    monkeypatch.setattr(protected_staged, '_run_staged_protected', observed)
    results = []
    for enabled in (False, True):
        monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', str(int(enabled)))
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(ids, labels=ids, use_cache=False).loss
        loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        results.append((loss.detach(), gradients))
    assert len(calls) >= 2
    torch.testing.assert_close(results[1][0], results[0][0], atol=2e-4, rtol=2e-3)
    assert results[1][1].keys() == results[0][1].keys()
    for name in results[0][1]:
        got, want = results[1][1][name], results[0][1][name]
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)
        assert torch.isfinite(got).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU NOT RUN: CUDA device unavailable')
@pytest.mark.parametrize('outputs', [(2, 4), (3,)])
@pytest.mark.parametrize('curvature', [0., .3])
def test_protected_trajectory_cuda_bank_loss(outputs, curvature, monkeypatch):
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule

    args = tuple(x.detach().cuda().float().requires_grad_() for x in _case(length=17, occupied=1, curvature=curvature))

    def run(enabled):
        monkeypatch.setenv('FLA_HYPERBOLIC_PROTECTED_STAGED', str(int(enabled)))
        result = triton_protected_chunkwise_hyperbolic_delta_rule(
            *args[:4], curvature=args[9], initial_state=args[4], protected_keys=args[5], protected_values=args[6],
            protected_scores=args[7], selection_scores=args[8], protection_rank=2, checkpoint_size=16,
            temperature=.7, kernel_strategy='value_parallel', checkpoint=False,
        )
        return torch.autograd.grad(sum(result[i].square().sum() for i in outputs), args)

    expected, actual = run(False), run(True)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=3e-3, rtol=3e-2)
        assert torch.isfinite(got).all()
