# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Three complete CPU paths: dense oracle, ordinary chunk AD, handwritten chunk VJP.

Run with FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q <this file>.
The optional KHDN_MANUAL_EVIDENCE path records per-source metrics and dispatch decisions.
No author/reviewer implementation is imported; tolerances are fixed before candidate execution.
"""

import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
MODULES = {}
for _name in ('reference_arithmetic', 'naive', 'reference_chunk', 'reference_backward'):
    _fullname = f'fla.ops.key_gated_hyperbolic_delta_rule.{_name}'
    _spec = importlib.util.spec_from_file_location(_fullname, ROOT / 'fla/ops/key_gated_hyperbolic_delta_rule' / f'{_name}.py')
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_fullname] = _module
    _spec.loader.exec_module(_module)
    MODULES[_name] = _module

oracle = MODULES['naive'].naive_recurrent_key_gated_hyperbolic_delta_rule
ordinary = MODULES['reference_chunk'].naive_chunk_key_gated_hyperbolic_delta_rule
manual = MODULES['reference_backward'].manual_chunk_key_gated_hyperbolic_delta_rule
NAMES = ('output', 'state', 'dq', 'dk', 'dv', 'dbeta', 'dell', 'dS0', 'dc')
TOL64 = dict(atol=1e-10, rtol=1e-9)
AUTHOR64 = dict(atol=1e-9, rtol=1e-9)
TOL32 = dict(atol=2e-5, rtol=2e-5)
GRADCHECK = dict(eps=1e-6, atol=1e-6, rtol=1e-4)
ROWS = []


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture(scope='session', autouse=True)
def evidence():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    dest = os.environ.get('KHDN_MANUAL_EVIDENCE')
    if dest:
        files = [Path(__file__), *(ROOT / 'fla/ops/key_gated_hyperbolic_delta_rule' / f'{n}.py' for n in MODULES)]
        report = dict(work_package='KHDN-CPU-CHUNK', gpu='NOT RUN',
                      head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                      python=sys.version, torch=str(torch.__version__), platform=platform.platform(), threads=1,
                      tolerances=dict(reference64=TOL64, author64=AUTHOR64, author32=TOL32, gradcheck=GRADCHECK), rows=ROWS)
        Path(dest).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def leaves(xs, dtype=None):
    return tuple(x.detach().to(dtype or x.dtype).clone().requires_grad_() for x in xs)


def make(B=1, T=7, H=1, K=5, V=8, g=4, c=.6, dtype=torch.float64):
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn_like(q)
    k = k / k.norm(dim=-1, keepdim=True) * (.7 + .8 * torch.rand(B, T, H, 1, dtype=q.dtype))
    v = torch.randn(B, T, H, V // g, g, dtype=q.dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (.22 * math.sqrt(g)) * k.norm(dim=-1)[..., None, None]
    beta = .07 + .85 * torch.rand(B, T, H, dtype=q.dtype)
    ell = -.01 - .16 * torch.rand_like(q)
    state = torch.randn(B, H, K, V // g, g, dtype=q.dtype)
    state = state / state.square().sum((-3, -1), keepdim=True).sqrt() * (.2 * math.sqrt(g))
    return leaves((q, k, v.reshape(B, T, H, V), beta, ell, state.reshape(B, H, K, V), q.new_tensor(c)), dtype)


def run(fn, xs, g, **kwargs):
    return fn(*xs[:5], curvature=xs[6], initial_state=xs[5], output_final_state=True, group_size=g, **kwargs)


def vjp(result, xs, cot):
    return torch.autograd.grad(result, xs, grad_outputs=cot)


def metrics(got, ref):
    err = got.detach().double() - ref.detach().double()
    return dict(max_abs=err.abs().max().item() if err.numel() else 0.,
                reference_rms=ref.detach().double().square().mean().sqrt().item() if ref.numel() else 0.,
                error_rms=err.square().mean().sqrt().item() if err.numel() else 0.)


def compare(xs, g, mode='joint', label='', **kwargs):
    rx = leaves(xs, torch.float64)
    oracle_options = {key: value for key, value in kwargs.items() if key in ('scale', 'cu_seqlens', 'check_inputs')}
    ref = run(oracle, rx, g, **oracle_options)
    cot = tuple(torch.randn_like(x).to(xs[0].dtype).double() for x in ref)
    if mode == 'output':
        cot = cot[0], torch.zeros_like(cot[1])
    elif mode == 'state':
        cot = torch.zeros_like(cot[0]), cot[1]
    elif mode == 'zero':
        cot = tuple(torch.zeros_like(x) for x in cot)
    rg = vjp(ref, rx, cot)
    reference = (*ref, *rg)
    records = []
    for fn in (ordinary, manual):
        xx, info = leaves(xs), {}
        result = run(fn, xx, g, telemetry=info, **kwargs)
        gg = vjp(result, xx, tuple(z.to(r.dtype) for z, r in zip(cot, result)))
        values = (*result, *gg)
        row = dict(label=label, implementation=fn.__name__, dtype=str(xs[0].dtype), shape=list(xs[0].shape),
                   group_size=g, cotangent=mode, telemetry=info,
                   metrics={n: metrics(a, b) for n, a, b in zip(NAMES, values, reference)})
        ROWS.append(row)
        records.append(row)
        for name, got, want in zip(NAMES, values, reference):
            assert torch.isfinite(got).all(), name
            torch.testing.assert_close(got.double(), want.double(), **(TOL64 if xs[0].dtype == torch.float64 else TOL32),
                                       msg=lambda msg, name=name: f'{name}: {msg}')
        assert result[0].dtype == xs[0].dtype
        assert result[1].dtype == (torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32)
        if fn is manual:
            assert info['backward'] == 'complete_manual_chunk_source_vjp'
            assert info['shared_A_vjp'] == len(info['partition'])
            assert info['backward_direct_FS'] == info['direct_FS']
            assert len(info['carry_Fnext_owners']) == info['carry_edges']
    return records


@pytest.mark.parametrize('g,C', [(1, 1), (1, 16), (16, 1), (16, 32), (32, 1), (32, 16), (32, 32), (32, 64)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
def test_chunk_three_paths(g, C, dtype, carry_mode, mode):
    main = g == 32
    xs = list(make(B=1 if main else 2, T=C + 3, H=2, K=128 if main else 5, V=128 if main else 2*g, g=g, dtype=dtype))
    with torch.no_grad():
        xs[3][:, ::3] = 0
        xs[3][:, 1::3] = 1
    rows = compare(xs, g, mode, label='main_matrix', chunk_size=C, carry_mode=carry_mode, scale=.37)
    assert all(r['telemetry']['fallback_chunks'] == 0 for r in rows)


@pytest.mark.parametrize('gauge', ['none', 'fixed', 'zero_live', 'zero_detached'])
@pytest.mark.parametrize('feedback', [False, True])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('kind', ['zero', 'one', 'near_one', 'random', 'c0_ell0'])
def test_chunk_gauge_feedback_endpoints(gauge, feedback, carry_mode, kind):
    xs = list(make(T=8, H=2, c=0. if kind == 'c0_ell0' else .6))
    with torch.no_grad():
        if kind == 'zero':
            xs[3].zero_()
        elif kind == 'one':
            xs[3].fill_(1)
        elif kind == 'near_one':
            xs[3].fill_(1 - 1e-12)
        elif kind == 'c0_ell0':
            xs[4].zero_()
            xs[3][:, ::2] = 1
            xs[3][:, 1::2] = 0
    fixed = (2 * torch.randn(1, 8, 2, 2, dtype=torch.float64)).exp() if gauge == 'fixed' else None
    compare(xs, 4, label=kind, chunk_size=3, carry_mode=carry_mode, gauge=gauge, gauge_values=fixed,
            feedback_precomposition=feedback)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_gradcheck(fn, carry_mode):
    xs = make(T=4, K=2, V=2, g=1)
    assert torch.autograd.gradcheck(lambda *args: run(fn, args, 1, chunk_size=3, carry_mode=carry_mode,
                                                      gauge='zero_live'), xs, **GRADCHECK)


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_c0_nonzero_derivative(fn):
    q = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
    xs = leaves((q, q, torch.tensor([[[[.5]]]], dtype=q.dtype), torch.tensor([[[.5]]], dtype=q.dtype),
                 torch.zeros_like(q), torch.tensor([[[[0.], [.2]]]], dtype=q.dtype), q.new_tensor(0.)))
    state = run(fn, xs, 1, chunk_size=1)[1]
    dc = torch.stack([torch.autograd.grad(state[0, 0, i, 0], xs[6], retain_graph=True)[0] for i in range(2)])
    torch.testing.assert_close(dc, q.new_tensor([1 / 32, -1 / 40]), atol=1e-14, rtol=1e-13)
    dell = torch.autograd.grad(state.sum(), xs[4])[0]
    assert dell.abs().sum() > 0


@pytest.mark.parametrize('g,K', [(1, 1), (32, 1), (32, 4)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_strong_decay(g, K, dtype, fn, carry_mode):
    direction = torch.ones(K, dtype=dtype) / math.sqrt(K)
    q = direction.expand(1, 2, 1, K).clone()
    xs = leaves((q, q.clone(), torch.tensor([.5, .1], dtype=dtype)[None, :, None, None].expand(1, 2, 1, g),
                 torch.tensor([[[1.], [0.]]], dtype=dtype),
                 torch.tensor([-1e8, -1], dtype=dtype)[None, :, None, None].expand_as(q),
                 torch.zeros(1, 1, K, g, dtype=dtype), torch.tensor(.6, dtype=dtype)))
    result = run(fn, xs, g, chunk_size=2, carry_mode=carry_mode, scale=1.)
    projected = direction @ result[1][0, 0, :, 0]
    derivative = torch.autograd.grad(projected, xs[4])[0][0, 1, 0]
    expected = .5 * math.exp(-1)
    tolerance = dict(atol=2e-15, rtol=2e-14) if dtype == torch.float64 else dict(atol=2e-8, rtol=3e-7)
    torch.testing.assert_close(projected, projected.new_tensor(expected), **tolerance)
    torch.testing.assert_close(derivative, derivative.new_full((K,), expected / K), **tolerance)
    assert (xs[4].detach().float().cumsum(1)[:, 1] - xs[4].detach().float().cumsum(1)[:, 0]).eq(0).all()
    if fn is manual:
        compare(xs, g, label='strong_decay_all_sources', chunk_size=2, carry_mode=carry_mode, scale=1.)


def witness(g, dtype, nonscalar=False):
    xs = list(make(T=32, K=3 if nonscalar else 1, V=g, g=g, dtype=dtype))
    with torch.no_grad():
        xs[1].zero_()
        xs[1][..., 0] = 1
        xs[0].copy_(xs[1])
        vals = torch.full((32, g), math.sqrt(.99 / float(xs[6])), dtype=dtype)
        vals[1::2].neg_()
        if nonscalar:
            angles = torch.arange(32, dtype=dtype) * .03
            xs[1][0, :, 0, 0] = angles.cos()
            xs[1][0, :, 0, 1] = angles.sin()
            direction = torch.randn(32, g, dtype=dtype)
            direction /= direction.norm(dim=-1, keepdim=True)
            vals = direction * math.sqrt(.99 * g / float(xs[6]))
        xs[2][0, :, 0] = vals
        xs[3].fill_(1)
        xs[4].zero_()
        xs[5].zero_()
        xs[5][..., 0, :] = .11
    return xs


@pytest.mark.parametrize('g', [1, 32])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
def test_chunk_range_witness(g, dtype, carry_mode, mode):
    rows = compare(witness(g, dtype), g, mode, label='retained_inverse_overflow', chunk_size=32, carry_mode=carry_mode)
    for row in rows:
        info = row['telemetry']
        assert [p['stop'] - p['start'] for p in info['partition']] == [4] * 8
        assert info['fallback_chunks'] == 1 and info['fallback_subblocks'] == 8
        assert info['carry_edges'] == 0 and info['direct_FS'] == 8
        assert all(p['precision'] == 'torch.float64' and 'amplitude_budget_32' in p['reasons'] for p in info['partition'])
        assert all(p['amplitude_bits'] < 32 for p in info['partition'])


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_nonscalar_near_boundary(dtype, carry_mode):
    xs = witness(32, dtype, nonscalar=True)
    with torch.no_grad():
        xs[5].normal_()
        xs[5] *= (.999 * math.sqrt(32 / float(xs[6]))) / xs[5].norm()
    compare(xs, 32, label='nonscalar_near_boundary', chunk_size=32, carry_mode=carry_mode)


@pytest.mark.parametrize('amplitude', [1e-8, 1e8])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_key_scale_fallback(amplitude, carry_mode):
    xs = list(make(T=9, dtype=torch.float32))
    with torch.no_grad():
        xs[1] *= amplitude
        xs[2] *= amplitude
    rows = compare(xs, 4, label='key_scale', chunk_size=4, carry_mode=carry_mode)
    assert all(row['telemetry']['fallback_chunks'] > 0 for row in rows)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
@pytest.mark.parametrize('cuts', [[0, 0, 4, 4, 9, 9], [0, 0, 0]])
def test_chunk_packed_empty(fn, carry_mode, cuts):
    xs = list(make(T=cuts[-1], H=2))
    xs[5] = leaves((xs[5].expand(len(cuts) - 1, -1, -1, -1).clone(),))[0]
    cu = torch.tensor(cuts, dtype=torch.int32)
    compare(xs, 4, label='packed_empty', chunk_size=3, carry_mode=carry_mode, cu_seqlens=cu)
    result = run(fn, xs, 4, chunk_size=3, carry_mode=carry_mode, cu_seqlens=cu)
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if a == b:
            torch.testing.assert_close(result[1][i], xs[5][i], atol=0, rtol=0)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_streaming_causal_noncontiguous(fn, carry_mode):
    xs = make(B=2, T=13, H=2, K=7, V=8)
    xs = tuple(x.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_() if x.ndim >= 3 else x for x in xs)
    assert not xs[0].is_contiguous()
    compare(xs, 4, label='noncontiguous', chunk_size=5, carry_mode=carry_mode)
    full = run(fn, xs, 4, chunk_size=5, carry_mode=carry_mode)
    cot = tuple(torch.randn_like(x) for x in full)
    gf = vjp(full, xs, cot)
    yy = leaves(xs)
    first = run(fn, tuple(x[:, :6] for x in yy[:5]) + yy[5:], 4, chunk_size=5, carry_mode=carry_mode)
    second = run(fn, tuple(x[:, 6:] for x in yy[:5]) + (first[1], yy[6]), 4, chunk_size=5, carry_mode=carry_mode)
    joined = torch.cat((first[0], second[0]), dim=1), second[1]
    gj = vjp(joined, yy, cot)
    for a, b in zip((*joined, *gj), (*full, *gf)):
        torch.testing.assert_close(a, b, **TOL64)
    zz = leaves(xs)
    out = run(fn, zz, 4, chunk_size=5, carry_mode=carry_mode)[0]
    future = torch.autograd.grad(out[:, :5].sum(), zz[:5])
    assert all(x[:, 5:].eq(0).all() for x in future)
    changed = leaves(xs)
    with torch.no_grad():
        changed[0][:, 5:] *= 7
        changed[1][:, 5:] *= 1.2
        changed[2][:, 5:] *= .3
        changed[3][:, 5:] *= .4
        changed[4][:, 5:] *= 9
    oo = run(fn, changed, 4, chunk_size=5, carry_mode=carry_mode)[0]
    torch.testing.assert_close(out[:, :5], oo[:, :5], atol=0, rtol=0)


@pytest.mark.parametrize('reanchor', [0, 1, 2])
def test_chunk_carry_ownership_and_checkpoint(reanchor):
    xs = make(T=11, H=2)
    info = {}
    result = run(manual, xs, 4, chunk_size=4, carry_mode='y_carry', reanchor_interval=reanchor, telemetry=info)
    cot = tuple(torch.randn_like(x) for x in result)
    gradients = vjp(result, xs, cot)
    for seq, head, source, owner in info['carry_Fnext_owners']:
        assert seq == 0 and head in (0, 1) and owner == source + 1
    xx = leaves(xs)
    replay = checkpoint(lambda *z: run(manual, z, 4, chunk_size=4, carry_mode='y_carry',
                                       reanchor_interval=reanchor, execution_plan=info['execution_plan']),
                        *xx, use_reentrant=False)
    replay_grad = vjp(replay, xx, cot)
    for a, b in zip((*replay, *replay_grad), (*result, *gradients)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    compare(xs, 4, label='carry_Fnext_owner', chunk_size=4, carry_mode='y_carry', reanchor_interval=reanchor)


def test_chunk_manual_has_no_ad_backward(monkeypatch):
    xs = make(T=7)
    out = run(manual, xs, 4, chunk_size=3, carry_mode='y_carry')

    def forbidden(*args, **kwargs):
        raise AssertionError('Generic AD was called inside handwritten backward')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(torch.autograd.functional, 'vjp', forbidden)
    (out[0].sum() + out[1].sum()).backward()
    assert all(x.grad is not None for x in xs)


def test_chunk_double_backward_rejection():
    xs = make(T=3)
    out = run(manual, xs, 4, chunk_size=2, carry_mode='y_carry')
    with pytest.raises(RuntimeError, match='double backward is not supported'):
        torch.autograd.grad(out[0].sum() + out[1].sum(), xs, create_graph=True)


@pytest.mark.parametrize('fraction', [0., .4, 1.])
@pytest.mark.parametrize('amplitude', [1e-25, 1., 1e25])
def test_chunk_normalized_mass_tiny_derivatives(fraction, amplitude):
    r, t = [torch.tensor(x * amplitude, dtype=torch.float32) for x in (fraction, 1 - fraction)]
    residual = torch.tensor([.2 * amplitude, -.3 * amplitude], dtype=torch.float32)
    target = torch.tensor([.3, -.4], dtype=torch.float32)
    cot = torch.tensor([.7, -.2], dtype=torch.float32)
    got = MODULES['reference_backward']._mass_vjp(residual, target, r, t, cot)
    xx = leaves((residual, target, r, t), torch.float64)
    z = xx[2] + xx[3]
    ref = torch.autograd.grad((cot.double() * (xx[0] / z + xx[3] / z * xx[1])).sum(), xx)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a.double(), b, atol=0, rtol=2e-6)


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_empty_large_c0(fn):
    xs = list(make(T=0, c=0.))
    with torch.no_grad():
        xs[5].fill_(1e200)
    result = run(fn, xs, 4)
    torch.testing.assert_close(result[1], xs[5], atol=0, rtol=0)
    grads = vjp(result, xs, (torch.zeros_like(result[0]), torch.ones_like(result[1])))
    torch.testing.assert_close(grads[5], torch.ones_like(xs[5]), atol=0, rtol=0)
    assert grads[6] == 0


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_optional_state_dtype_autocast(fn):
    xs = make(T=5, dtype=torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out, state = fn(*xs[:5], curvature=xs[6], group_size=4, chunk_size=3, carry_mode='y_carry')
    assert state is None and out.dtype == torch.float32
    out.sum().backward()
    assert all(x.grad is not None for x in (*xs[:5], xs[6]))


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_external_parameter_chains(fn):
    base = make(T=5, K=5, V=8)
    params = leaves((torch.randn(1, 5, 1, 3, dtype=torch.float64), torch.randn(3, 4, dtype=torch.float64) * .1,
                     torch.randn(4, 5, dtype=torch.float64) * .1, torch.tensor([-.2], dtype=torch.float64),
                     torch.randn(5, dtype=torch.float64) * .1, torch.tensor(-.1, dtype=torch.float64)))

    def outer(which, weights):
        x, w1, w2, alog, bias, raw_c = weights
        c = torch.nn.functional.softplus(raw_c)
        ell = -alog.exp() * torch.nn.functional.softplus(x @ w1 @ w2 + bias)
        v = base[2].detach() / (1 + c).sqrt()
        beta = (base[3].detach() + x.sum(-1) * .1).sigmoid()
        src = (base[0].detach(), base[1].detach(), v, beta, ell, base[5].detach(), c)
        return run(which, src, 4, **({} if which is oracle else dict(chunk_size=3, carry_mode='y_carry')))
    got = outer(fn, params)
    cot = tuple(torch.randn_like(x) for x in got)
    gg = torch.autograd.grad(got, params, cot)
    reference_params = leaves(params)
    ref = outer(oracle, reference_params)
    rg = torch.autograd.grad(ref, reference_params, cot)
    for a, b in zip((*got, *gg), (*ref, *rg)):
        torch.testing.assert_close(a, b, **TOL64)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_fallback_entry_exit_and_replay(dtype, carry_mode):
    xs = list(make(T=35, K=3, V=64, g=32, dtype=dtype))
    with torch.no_grad():
        xs[1][0, 8:24, 0] = xs[1].new_tensor([1., 0., 0.])
        vals = torch.full((16, 64), math.sqrt(.99 / float(xs[6])), dtype=dtype)
        vals[1::2].neg_()
        xs[2][0, 8:24, 0] = vals
        xs[3][0, 8:24, 0] = 1
        xs[4][0, 8:24, 0] = 0
    rows = compare(xs, 32, label='fallback_entry_exit', chunk_size=8, carry_mode=carry_mode)
    for row in rows:
        assert row['telemetry']['fallback_chunks'] == 2
        assert row['telemetry']['fallback_subblocks'] == 4
        assert row['telemetry']['carry_edges'] == (1 if carry_mode == 'y_carry' else 0)
    reference = run(manual, xs, 32, chunk_size=8, carry_mode=carry_mode)
    xx = leaves(xs)
    info = rows[-1]['telemetry']
    replay = checkpoint(lambda *z: run(manual, z, 32, chunk_size=8, carry_mode=carry_mode,
                                       execution_plan=info['execution_plan']), *xx, use_reentrant=False)
    cot = tuple(torch.randn_like(z) for z in reference)
    for a, b in zip((*reference, *vjp(reference, xs, cot)), (*replay, *vjp(replay, xx, cot))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('g', [1, 32])
@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_range_before_unsafe_inverse(monkeypatch, g, carry_mode):
    xs = witness(g, torch.float32)
    original = torch.linalg.solve_triangular
    calls = []

    def checked(matrix, rhs, **kwargs):
        calls.append((matrix.shape[-1], str(matrix.dtype)))
        assert matrix.shape[-1] <= 4
        assert matrix.dtype == torch.float64
        return original(matrix, rhs, **kwargs)
    monkeypatch.setattr(torch.linalg, 'solve_triangular', checked)
    info = {}
    out = run(manual, xs, g, chunk_size=32, carry_mode=carry_mode, telemetry=info)
    torch.autograd.grad(out[0].sum() + out[1].sum(), xs)
    assert calls and info['partition'][0]['requested_amplitude_bits'] > 240


@pytest.mark.parametrize('C', [16, 32, 64])
def test_chunk_amplitude_guard_boundary(C):
    threshold = (1 - 2 ** (-32 / C)) ** 2
    k = torch.ones(C, 1, dtype=torch.float64)
    b = torch.ones(C, dtype=torch.float64)
    c = torch.tensor(.6, dtype=torch.float64)
    reasons = []
    for delta in (threshold - 1e-7, threshold + 1e-7):
        v = k * math.sqrt(delta / c.item())
        result = MODULES['reference_chunk']._input_bound(k, v, b, c, 1, torch.float64, None)
        reasons.append(result['reasons'])
    assert 'amplitude_budget_32' not in reasons[0]
    assert 'amplitude_budget_32' in reasons[1]


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_mixed_source_dtypes(fn):
    xs = list(make(T=5, H=2, dtype=torch.float32))
    xs[5] = leaves((xs[5],), torch.float64)[0]
    xs[6] = leaves((xs[6],), torch.float64)[0]
    result = run(fn, xs, 4, chunk_size=3, carry_mode='y_carry')
    assert result[0].dtype == torch.float32 and result[1].dtype == torch.float64
    compare(xs, 4, label='mixed_source_dtypes', chunk_size=3, carry_mode='y_carry')


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_scalar_gate_and_zero_cotangents(fn):
    xs = list(make(T=6))
    with torch.no_grad():
        xs[4].copy_(xs[4][..., :1].clone().expand_as(xs[4]))
    compare(xs, 4, mode='zero', label='scalar_gate_zero_cotangent', chunk_size=4, carry_mode='y_carry')
    output = run(fn, xs, 4, chunk_size=4, carry_mode='y_carry')
    gg = vjp(output, xs, tuple(torch.zeros_like(x) for x in output))
    assert all(x.eq(0).all() for x in gg)


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_packed_causal_isolation(fn):
    xs = list(make(T=10))
    xs[5] = leaves((xs[5].expand(3, -1, -1, -1),))[0]
    result = run(fn, xs, 4, chunk_size=3, carry_mode='y_carry', cu_seqlens=torch.tensor([0, 4, 4, 10]))
    grads = torch.autograd.grad(result[0][:, :4].sum() + result[1][0].sum(), xs)
    assert all(x[:, 4:].eq(0).all() for x in grads[:5])
    assert grads[5][1:].eq(0).all()


def test_chunk_cotangent_range_scaling():
    xs = make(T=7)
    original = run(manual, xs, 4, chunk_size=3, carry_mode='y_carry')
    cot = tuple(torch.randn_like(x) for x in original)
    expected = vjp(original, xs, cot)
    xx, info = leaves(xs), {}
    result = run(manual, xx, 4, chunk_size=3, carry_mode='y_carry', telemetry=info)
    factor = 2. ** 500
    actual = vjp(result, xx, tuple(x * factor for x in cot))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a / factor, b, **TOL64)
    assert info['cotangent_scale_log2'] > 400


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('length', [1, 3])
@pytest.mark.parametrize('output_final_state', [False, True])
def test_chunk_optional_state_representability(fn, length, output_final_state):
    q = torch.full((1, length, 1, 1), 1e-40, dtype=torch.float32)
    k, v = q.clone(), torch.ones_like(q, requires_grad=True)
    beta, ell = torch.ones(q.shape[:-1]), torch.zeros_like(q)
    options = dict(curvature=0., scale=1., group_size=1, chunk_size=1, output_final_state=output_final_state)
    if output_final_state:
        with pytest.raises(MODULES['reference_chunk'].ReferenceRangeError, match='final state'):
            fn(q, k, v, beta, ell, **options)
    else:
        output, state = fn(q, k, v, beta, ell, **options)
        assert state is None
        torch.testing.assert_close(output, torch.ones_like(v), atol=0, rtol=0)
        dv, = torch.autograd.grad(output.sum(), v)
        torch.testing.assert_close(dv, torch.ones_like(v), atol=0, rtol=0)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('power', [500, 600])
@pytest.mark.parametrize('layout', ['heads', 'groups'])
@pytest.mark.parametrize('large_state', [False, True])
@pytest.mark.parametrize('all_sources', [False, True])
def test_chunk_cotangent_span(fn, power, layout, large_state, all_sources):
    heads, values = (2, 1) if layout == 'heads' else (1, 2)
    q = torch.zeros((1, 1, heads, 1), dtype=torch.float64, requires_grad=True)
    k = torch.ones_like(q)
    v = torch.zeros((1, 1, heads, values), dtype=q.dtype)
    beta, ell = torch.zeros(q.shape[:-1], dtype=q.dtype), torch.zeros_like(q)
    state = torch.tensor([0., 2. ** power if large_state else 1.], dtype=q.dtype).reshape(1, heads, 1, values)
    c = q.new_zeros(())
    xs = (q, k, v, beta, ell, state, c)
    if all_sources:
        xs = leaves(xs)
    info = {}
    output, final = run(fn, xs, 1, scale=1., chunk_size=1, telemetry=info)
    cot = q.new_tensor([2. ** power, 2. ** -power]).reshape_as(output)
    selected = xs if all_sources else xs[:1]
    grads = torch.autograd.grad((output, final), selected, (cot, torch.zeros_like(final)))
    target = q.new_tensor([0., 1. if large_state else 2. ** -power])
    if layout == 'groups':
        target = target.sum().reshape_as(q)
    else:
        target = target.reshape_as(q)
    torch.testing.assert_close(grads[0], target, atol=0, rtol=1e-12)
    for grad in grads[1:]:
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('option,value', [('chunk_size', 0), ('carry_mode', 'other'), ('gauge', 'other'),
                                          ('reanchor_interval', -1), ('feedback_precomposition', 1)])
def test_chunk_option_validation(fn, option, value):
    with pytest.raises(ValueError):
        run(fn, make(), 4, **{option: value})


@pytest.mark.parametrize('carry_mode', ['no_carry', 'y_carry'])
def test_chunk_condition_guard(carry_mode):
    xs = list(make(T=6, dtype=torch.float32))
    with torch.no_grad():
        norms = xs[1].new_tensor([1., 100., 1., 100., 1., 100.])
        old_norm = xs[1].norm(dim=-1)
        xs[2] *= norms[None, :, None, None] / old_norm[..., None]
        xs[1].zero_()
        xs[1][0, :, 0, 0] = norms
        xs[3].fill_(1)
        xs[4].zero_()
    rows = compare(xs, 4, label='condition_guard', chunk_size=6, carry_mode=carry_mode)
    for row in rows:
        info = row['telemetry']
        assert info['fallback_chunks'] == 1
        assert len(info['partition']) == 1 and info['partition'][0]['stop'] == 6
        assert any(reason.startswith('L_condition=') for reason in info['partition'][0]['reasons'])


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_output_scale_range_guard(fn):
    xs = make(T=4, dtype=torch.float32)
    info = {}
    result = run(fn, xs, 4, chunk_size=4, scale=2. ** 30, telemetry=info)
    reference = run(oracle, leaves(xs, torch.float64), 4, scale=2. ** 30)
    for a, b in zip(result, reference):
        torch.testing.assert_close(a.double(), b.double(), **TOL32)
    assert info['fallback_chunks'] == 1
    assert 'source_amplitude_range' in info['partition'][0]['reasons']


def mp_canonical(xs, cot, g, digits):
    """Independent canonical scalar/rank-one expression, differentiated only in tests."""
    import mpmath

    ctx = mpmath.mp.clone()
    ctx.dps = digits
    shapes = [x.shape for x in xs]
    sizes = [x.numel() for x in xs]
    values = tuple(ctx.mpf(x) for source in xs for x in source.detach().double().flatten().tolist())
    B, T, H, K = shapes[0]
    V = shapes[2][-1]
    assert B == H == 1
    co, cs = [tuple(ctx.mpf(v) for v in x.detach().double().flatten().tolist()) for x in cot]

    def canonical(args, outputs=False):
        parts, pos = [], 0
        for size in sizes:
            parts.append(args[pos:pos + size])
            pos += size
        q, k, v, beta, ell, state, c = parts
        curvature = c[0] / g
        state, out = list(state), []
        for t in range(T):
            key = k[t*K:(t+1)*K]
            val = v[t*V:(t+1)*V]
            x = [ctx.exp(ell[t*K+a]) * state[a*V+j] for a in range(K) for j in range(V)]
            if K == V == g == 1:
                a = key[0]**2 - c[0] * val[0]**2
                residual = val[0] - key[0] * x[0]
                state = [(x[0]*a + key[0]*beta[t]*residual) / (a + c[0]*val[0]*beta[t]*residual)]
            else:
                n = ctx.fsum(ka*ka for ka in key)
                state = [ctx.zero] * (K*V)
                for start in range(0, V, g):
                    vg = val[start:start+g]
                    residual = [vg[j] - ctx.fsum(key[a]*x[a*V+start+j] for a in range(K)) for j in range(g)]
                    margin = n - curvature * ctx.fsum(vj*vj for vj in vg)
                    dot = ctx.fsum(vj*rj for vj, rj in zip(vg, residual))
                    u = [beta[t]/n * (rj + curvature*vj*dot/margin) for vj, rj in zip(vg, residual)]
                    denominator = 1 + curvature * ctx.fsum(vj*uj for vj, uj in zip(vg, u))
                    for a in range(K):
                        for j in range(g):
                            state[a*V+start+j] = (x[a*V+start+j] + key[a]*u[j]) / denominator
            out.extend(ctx.fsum(q[t*K+a]*state[a*V+j] for a in range(K)) for j in range(V))
        if outputs:
            return out, state
        return ctx.fsum(a*b for a, b in zip(out, co)) + ctx.fsum(a*b for a, b in zip(state, cs))

    output, state = canonical(values, outputs=True)
    derivatives = []
    for i, value in enumerate(values):
        step_scale = abs(value) if value else ctx.one
        derivative = ctx.diff(lambda z: canonical(values[:i] + (value + step_scale*z,) + values[i+1:]), ctx.zero)
        derivatives.append(derivative / step_scale)
    raw = [output, state]
    pos = 0
    for size in sizes:
        raw.append(derivatives[pos:pos+size])
        pos += size
    all_shapes = [xs[2].shape, xs[5].shape, *shapes]
    rounded = tuple(torch.tensor([float(v) for v in row], dtype=torch.float64).reshape(shape)
                    for row, shape in zip(raw, all_shapes))
    return rounded, [[ctx.nstr(v, digits) for v in row] for row in raw]


def numerical_fixture(case, dtype=torch.float64):
    values = [1., 1., .4, .7, 0., .2, .6]
    options = {}
    if case.startswith('fp32_margin'):
        margin = float(case.split(':')[1])
        values[2], values[3], values[5] = math.sqrt((1-margin)/.6), .01, 0.
    elif case == 'feedback_cancellation' or case.startswith('near_beta'):
        values[2], values[3], values[5] = math.sqrt((1-1e-8)/.6), .5, (1-1e-8)/math.sqrt(.6)
        if case.startswith('near_beta'):
            values[3] = float(case[-1])
    elif case.startswith('key_scale'):
        power = int(case.split(':')[1])
        values[1], values[2] = 2.**power, .4*2.**power
    elif case == 'query_scale' or case.startswith('query_full_write'):
        values[0] = 2.**201 if case == 'query_scale' else 2.**int(case.split(':')[1])
        if case.startswith('query_full_write'):
            values[3] = 1.
    elif case == 'nearest_margin':
        values[2] = math.nextafter(math.sqrt(1/.6), 0.)
    elif case == 'input_margin':
        values[2] = math.sqrt((1-2.**-34)/.6)
    elif case.startswith('gauge'):
        power = int(case.split(':')[1])
        options = dict(gauge='fixed', gauge_values=torch.full((1, 1, 1, 1), 2.**power, dtype=dtype))
    elif case.startswith('endpoint'):
        values[1], values[2], values[3], values[6] = 2.**-101, .4*2.**-101, float(case[-1]), 0.
    shapes = [(1, 1, 1, 1)]*3 + [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), ()]
    return leaves(tuple(torch.tensor(value, dtype=dtype).reshape(shape) for value, shape in zip(values, shapes))), options


def compare_mp(xs, g, mode, label, **options):
    cot = (torch.full_like(xs[2], .7), torch.full_like(xs[5], -.3))
    if g > 1 or xs[0].shape[-1] > 1:
        cot = tuple(torch.randn_like(x) for x in cot)
    if mode == 'output':
        cot = cot[0], torch.zeros_like(cot[1])
    elif mode == 'state':
        cot = torch.zeros_like(cot[0]), cot[1]
    ref90, raw90 = mp_canonical(xs, cot, g, 90)
    reference, raw160 = mp_canonical(xs, cot, g, 160)
    for a, b in zip(ref90, reference):
        torch.testing.assert_close(a, b, atol=1e-70, rtol=1e-14)
    for fn in (ordinary, manual):
        xx, info = leaves(xs), {}
        result = run(fn, xx, g, scale=1., telemetry=info, **options)
        got = (*result, *vjp(result, xx, cot))
        row = dict(label=label, implementation=fn.__name__, dtype=str(xs[0].dtype), shape=list(xs[0].shape),
                   group_size=g, cotangent=mode, telemetry=info,
                   metrics={n: metrics(a, b) for n, a, b in zip(NAMES, got, reference)},
                   inputs_hex=[[float(x).hex() for x in source.detach().double().flatten().tolist()] for source in xs],
                   independent_truth='canonical expression at exact binary inputs; test-only mpmath.diff',
                   truth90=dict(zip(NAMES, raw90)), truth160=dict(zip(NAMES, raw160)))
        ROWS.append(row)
        for name, actual, expected in zip(NAMES, got, reference):
            budget = TOL32 if xs[0].dtype == torch.float32 else TOL64
            torch.testing.assert_close(actual.double(), expected, **budget, msg=lambda msg: f'{label}/{name}: {msg}')
            if xs[0].dtype == torch.float64 and name in ('dk', 'dv', 'dbeta'):
                # the absolute budget alone would accept a zero 1e-31 source gradient
                earlier = ref90[NAMES.index(name)]
                nonzero = (expected != 0) & ((earlier - expected).abs() <= 1e-12 * expected.abs())
                torch.testing.assert_close(actual.double()[nonzero], expected[nonzero], atol=0, rtol=1e-9)
        assert info['fallback_subblocks'] > 0
        if info['extended_subblocks']:
            assert info['backward'] == 'complete_manual_chunk_source_vjp'
            assert info['shared_A_vjp'] == len(info['partition'])
            assert info['backward_direct_FS'] == info['direct_FS']
    return ROWS[-2:]


@pytest.mark.parametrize('case,dtype', [
    ('fp32_margin:0.00025', torch.float32), ('fp32_margin:0.0005', torch.float32),
    ('fp32_margin:0.001', torch.float32), ('feedback_cancellation', torch.float64),
    ('key_scale:-101', torch.float64), ('key_scale:101', torch.float64),
    ('key_scale:-500', torch.float64), ('key_scale:500', torch.float64),
    ('query_scale', torch.float64), ('input_margin', torch.float64), ('nearest_margin', torch.float64),
    ('gauge:33', torch.float64), ('gauge:-33', torch.float64),
    ('endpoint0', torch.float64), ('endpoint1', torch.float64),
    ('near_beta0', torch.float64), ('near_beta1', torch.float64),
    ('query_full_write:16', torch.float64), ('query_full_write:201', torch.float64),
])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
@pytest.mark.parametrize('feedback', [False, True])
def test_chunk_numerics_scalar(case, dtype, mode, feedback):
    xs, options = numerical_fixture(case, dtype)
    rows = compare_mp(xs, 1, mode, case, chunk_size=1, feedback_precomposition=feedback, **options)
    if case.startswith('fp32_margin'):
        assert all(row['telemetry']['partition'][0]['precision'] == 'torch.float64' for row in rows)
        assert all('source_sensitivity' in row['telemetry']['partition'][0]['reasons'] for row in rows)
    else:
        assert all(row['telemetry']['extended_subblocks'] == 1 for row in rows)


@pytest.mark.parametrize('case', ['feedback_cancellation', 'key_scale:-101'])
@pytest.mark.parametrize('mode', ['output', 'state', 'joint'])
def test_chunk_numerics_grouped_multichunk(case, mode):
    src, _ = numerical_fixture(case)
    T, K, g = 3, 4, 32
    q = torch.full((1, T, 1, K), .5, dtype=torch.float64)
    k = q * src[1].item()
    v = torch.full((1, T, 1, g), src[2].item(), dtype=q.dtype)
    v[..., 1::2].neg_()
    beta = torch.full((1, T, 1), src[3].item(), dtype=q.dtype)
    ell = torch.zeros_like(q)
    state = torch.full((1, 1, K, g), .5*src[5].item(), dtype=q.dtype)
    state[..., 1::2].neg_()
    xs = leaves((q, k, v, beta, ell, state, src[6]))
    rows = compare_mp(xs, g, mode, 'grouped_'+case, chunk_size=2, carry_mode='y_carry')
    for row in rows:
        assert [p['stop']-p['start'] for p in row['telemetry']['partition']] == [2, 1]
        assert row['telemetry']['extended_subblocks'] == 2
        assert row['telemetry']['carry_edges'] == 0


@pytest.mark.parametrize('mixed_curvature_dtype', [False, True])
@pytest.mark.parametrize('requested', [(0,), (0, 2), (0, 3, 4, 5), (1,), (0, 1, 2, 3, 4, 5), (6,), (0, 6)])
def test_chunk_numerics_requested_gradients(requested, mixed_curvature_dtype):
    values = [1., 1e-14, 1., .5, 0., 0., 0.]
    shapes = [(1, 1, 1, 1)]*3 + [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), ()]
    dtypes = [torch.float64 if mixed_curvature_dtype and i != 6 else torch.float32 for i in range(7)]
    xs = tuple(torch.tensor(v, dtype=dt).reshape(s).requires_grad_(i in requested)
               for i, (v, s, dt) in enumerate(zip(values, shapes, dtypes)))
    out = run(manual, xs, 1, scale=1., chunk_size=1)
    if 6 in requested:
        with pytest.raises(MODULES['reference_chunk'].ReferenceRangeError, match='source VJP is not representable'):
            torch.autograd.grad(out[0].sum(), [xs[i] for i in requested])
    else:
        cot = (torch.ones_like(xs[2]), torch.zeros_like(xs[5]))
        reference, raw = mp_canonical(xs, cot, 1, 90)
        grads = torch.autograd.grad(out[0].sum(), [xs[i] for i in requested])
        for i, grad in zip(requested, grads):
            torch.testing.assert_close(grad.double(), reference[2+i], **TOL32)
        ROWS.append(dict(label='requested_sources', requested=list(requested), truth90=dict(zip(NAMES, raw)),
                         metrics={NAMES[2+i]: metrics(grad, reference[2+i]) for i, grad in zip(requested, grads)}))


@pytest.mark.parametrize('query', [1e38, 3e38])
def test_chunk_numerics_output_representability(query):
    xs, _ = numerical_fixture('ordinary', torch.float32)
    with torch.no_grad():
        xs[0].fill_(query)
        xs[2].fill_(2.)
        xs[3].fill_(1.)
        xs[5].zero_()
        xs[6].fill_(.1)
    if query == 3e38:
        with pytest.raises(MODULES['reference_chunk'].ReferenceRangeError, match='public output'):
            run(manual, xs, 1, scale=1., chunk_size=1)
    else:
        result = run(manual, xs, 1, scale=1., chunk_size=1)
        assert torch.isfinite(result[0]).all()
        torch.testing.assert_close(result[0], 2*xs[0], atol=0, rtol=1e-7)


def test_chunk_numerics_extended_no_ad(monkeypatch):
    import mpmath

    xs, _ = numerical_fixture('feedback_cancellation')
    result = run(manual, xs, 1, scale=1., chunk_size=1)

    def forbidden(*args, **kwargs):
        raise AssertionError('Differentiation called inside the exceptional manual VJP')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(torch.autograd.functional, 'vjp', forbidden)
    monkeypatch.setattr(mpmath.ctx_mp.MPContext, 'diff', forbidden)
    (result[0].sum() + result[1].sum()).backward()
    assert all(x.grad is not None for x in xs)


def test_chunk_numerics_extended_carry_restart():
    xs = list(make(T=13, K=2, V=2, g=1))
    with torch.no_grad():
        xs[1][:, 3:7] *= 2.**-101
        xs[2][:, 3:7] *= 2.**-101
    rows = compare_mp(xs, 1, 'joint', 'extended_carry_restart', chunk_size=3, carry_mode='y_carry')
    info = rows[-1]['telemetry']
    assert info['extended_subblocks'] == 2 and info['carry_edges'] == 1
    assert info['carry_Fnext_owners'] == [(0, 0, 3, 4)]
    original = run(manual, xs, 1, scale=1., chunk_size=3, carry_mode='y_carry')
    yy = leaves(xs)
    replay = checkpoint(lambda *z: run(manual, z, 1, scale=1., chunk_size=3, carry_mode='y_carry',
                                       execution_plan=info['execution_plan']), *yy, use_reentrant=False)
    cot = tuple(torch.randn_like(t) for t in original)
    for a, b in zip((*original, *vjp(original, xs, cot)), (*replay, *vjp(replay, yy, cot))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('fn', [ordinary, manual])
@pytest.mark.parametrize('case', ['normal', 'fp32_margin:0.00025', 'feedback_cancellation'])
def test_chunk_numerics_ambient_autocast(fn, case):
    if case == 'normal':
        xs = make(T=5, dtype=torch.float32)
        g = 4
    else:
        xs, _ = numerical_fixture(case, torch.float32 if case.startswith('fp32') else torch.float64)
        g = 1
    with torch.autocast('cpu', enabled=False):
        reference = run(fn, xs, g, chunk_size=3, carry_mode='y_carry')
        cot = tuple(torch.randn_like(x) for x in reference)
        rg = vjp(reference, xs, cot)
    yy = leaves(xs)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = run(fn, yy, g, chunk_size=3, carry_mode='y_carry')
    # ordinary torch AD is normally invoked outside the forward AMP context
    with torch.autocast('cpu', enabled=False):
        gg = vjp(actual, yy, cot)
    for a, b in zip((*actual, *gg), (*reference, *rg)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if fn is manual:
        zz = leaves(xs)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = run(fn, zz, g, chunk_size=3, carry_mode='y_carry')
            grad_inside_amp = vjp(out, zz, cot)
        for a, b in zip(grad_inside_amp, rg):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('fn', [ordinary, manual])
def test_chunk_numerics_extended_packed_empty(fn):
    src, _ = numerical_fixture('feedback_cancellation')
    xs = leaves(tuple(x.expand(1, 3, *x.shape[2:]).clone() for x in src[:5]) +
                (src[5].expand(4, -1, -1, -1).clone(), src[6]))
    bounds = [0, 0, 2, 3, 3]
    cot = (torch.randn_like(xs[2]), torch.randn_like(xs[5]))
    expected = [torch.zeros_like(xs[2]), torch.zeros_like(xs[5]), *(torch.zeros_like(x) for x in xs)]
    for index, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        local = tuple(x[:, start:stop] for x in xs[:5]) + (xs[5][index:index+1], xs[6])
        cc = cot[0][:, start:stop], cot[1][index:index+1]
        ref, _ = mp_canonical(local, cc, 1, 160)
        expected[0][:, start:stop] = ref[0]
        expected[1][index:index+1] = ref[1]
        for j in range(5):
            expected[2+j][:, start:stop] = ref[2+j]
        expected[7][index:index+1] = ref[7]
        expected[8] += ref[8]
    info = {}
    result = run(fn, xs, 1, scale=1., chunk_size=1, carry_mode='y_carry',
                 cu_seqlens=torch.tensor(bounds), telemetry=info)
    actual = (*result, *vjp(result, xs, cot))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, **TOL64)
    assert info['extended_subblocks'] == 3 and info['direct_FS'] == 3
    ROWS.append(dict(label='extended_packed_empty', implementation=fn.__name__, telemetry=info,
                     metrics={n: metrics(a, b) for n, a, b in zip(NAMES, actual, expected)}))


def test_chunk_numerics_invalid_margin():
    xs, _ = numerical_fixture('nearest_margin')
    with torch.no_grad():
        xs[2].fill_(math.nextafter(math.sqrt(1/.6), math.inf))
    with pytest.raises(ValueError, match='curvature/group_size'):
        run(manual, xs, 1, scale=1., chunk_size=1)
