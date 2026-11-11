# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU-only formula, host-control and kernel-semantics checks; not a CUDA/Triton compiler test.

Run directly with Python to avoid the repository's accelerator-dependent pytest conftest.
GPU entry points are tested separately in test_phba_ablation.py.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / 'fla/ops/phba'
MODES = ('full', 'current_state_only', 'routed_states_only')


def load_reference():
    spec = importlib.util.spec_from_file_location('_phba_cpu_reference', OPS / 'naive.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_symbols(path, names, namespace=None):
    """Execute actual source definitions with explicit CPU dependencies, without importing FLA."""
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in body} == set(names)
    for node in body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == 'dataclass']
    module = types.ModuleType('_phba_cpu_' + path.stem + '_' + '_'.join(names))
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, dataclass=dataclass, math=math)
    module.__dict__.update(namespace or {})
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    exec(compile(selected, str(path), 'exec'), module.__dict__)
    return module


REF = load_reference()
HOST = load_symbols(OPS / 'parallel.py', ['BlockMajorConfig', '_readout_routes'])


def fixture(lengths=(97, 1, 65), c=32, h=2, kh=3):
    host = REF.HostLayout.from_lengths(lengths, c)
    t = sum(lengths)
    current = host.token_block()
    valid = torch.ones(host.Nc, dtype=torch.bool)
    valid[list(host.seq_block_ptr[:-1])] = False
    layout = types.SimpleNamespace(
        Nc=host.Nc, T=t, chunk_size=c, block_ptr=torch.tensor(host.block_ptr, dtype=torch.int32),
        block_ptr_host=host.block_ptr, prefix_valid=valid, current_block=current.to(torch.int32),
    )
    tok = torch.full((h, t, kh), -1, dtype=torch.int32)
    for seq in range(host.Nseq):
        for block in host.blocks_of_sequence(seq):
            candidates = list(range(host.seq_block_ptr[seq], block))[::-1][:kh]
            for head in range(h):
                ids = candidates[head % len(candidates):] + candidates[:head % len(candidates)] if candidates else []
                tok[head, host.block_ptr[block]:host.block_ptr[block + 1], :len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return host, layout, tok, REF.rstate_from_rtok(tok, layout)


def inputs(t, h=2, d=4, m=3, dtype=torch.float64):
    gen = torch.Generator().manual_seed(124)
    return [torch.randn(t, h, d, generator=gen, dtype=dtype) * .2 for _ in range(5)] + [
        torch.randn(t, h, m, generator=gen, dtype=dtype) * .3,
    ]


def evaluate(values, host, tok, state, mode, staged=False):
    tr, sr = HOST._readout_routes(tok, state, mode)
    q, k, v, qr, kr, gate = values
    kwargs = dict(q_raw=q, q_rope=qr, k_rope=kr, k_raw=k, v=v, a_storage=gate,
                  tau=8., layout=host, rtok=tr, rstate=sr)
    return REF.reference_a_chunkwise(**kwargs) if staged else REF.reference_b(**kwargs, dtype=q.dtype)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)

    __radd__ = __add__


class TorchTL:
    """Small CPU evaluator for the operations used by the two real forward kernel bodies."""
    int64, int32, float32 = torch.int64, torch.int32, torch.float32
    ids = (0, 0)

    @classmethod
    def program_id(cls, axis):
        return torch.tensor(cls.ids[axis])

    @staticmethod
    def cast(value, dtype):
        return torch.as_tensor(value).to(dtype)

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end)

    @staticmethod
    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def load(pointer, mask=True, other=0):
        index, mask = torch.broadcast_tensors(torch.as_tensor(pointer.offset).long(), torch.as_tensor(mask).bool())
        valid = index[mask]
        assert ((valid >= 0) & (valid < pointer.data.numel())).all(), 'unmasked out-of-bounds load'
        out = torch.full(index.shape, other, dtype=pointer.data.dtype)
        out[mask] = pointer.data[valid]
        return out

    @staticmethod
    def store(pointer, value, mask=True):
        index, value, mask = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset).long(), torch.as_tensor(value), torch.as_tensor(mask).bool(),
        )
        valid = index[mask]
        assert ((valid >= 0) & (valid < pointer.data.numel())).all(), 'unmasked out-of-bounds store'
        pointer.data[valid] = value[mask].to(pointer.data.dtype)

    @staticmethod
    def dot(a, b, out_dtype):
        return a.to(out_dtype) @ b.to(out_dtype)

    @staticmethod
    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def max(x, axis, keep_dims=False):
        return x.max(axis, keepdim=keep_dims).values

    @staticmethod
    def sum(x, axis, keep_dims=False):
        return x.sum(axis, keepdim=keep_dims)

    exp, log, where, trans = staticmethod(torch.exp), staticmethod(torch.log), staticmethod(torch.where), staticmethod(torch.t)


KERNELS = load_symbols(OPS / 'parallel.py', ['_update', '_history_partial', '_owner_merge'], {'tl': TorchTL})


def emulated_forward(values, host, layout, tok, state, mode, index32=True):
    q, k, v, qr, kr, gate = values
    t, h, d = q.shape
    m, kh = gate.shape[-1], tok.shape[-1]
    _, _, pk, pv = evaluate(values, host, tok, state, 'full')
    oloc, lloc = torch.empty_like(q), torch.empty(h, t)
    for head in range(h):
        mask = REF.route_masks(tok, state, host, head, m)[0]
        score = (qr[:, head] @ kr[:, head].T * d ** -.5).masked_fill(~mask, -torch.inf)
        oloc[:, head] = score.softmax(-1) @ v[:, head]
        lloc[head] = torch.logsumexp(score, -1)
    rows, offsets, experts, work, task_offsets = [], [0], [], [], [0]
    edge_pos = torch.full((t, h, kh), -1, dtype=torch.int32)
    for block in range(host.Nc):
        for head in range(h):
            start = len(rows)
            for row in range(t):
                for slot in range(kh):
                    if tok[head, row, slot] == block:
                        edge_pos[row, head, slot] = len(rows)
                        rows.append(row * h + head)
            offsets.append(len(rows))
            for pos in range(start, len(rows), 16):
                experts.append(block * h + head)
                work.append(pos)
            task_offsets.append(len(work))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32)
    rho, offset, task_offset, work_expert, work_edge = map(i32, (rows, offsets, task_offsets, experts, work))
    pm, pz, pu = torch.empty(len(rows)), torch.empty(len(rows)), torch.empty(len(rows), d)
    for task in range(len(work)):
        TorchTL.ids = (task, 0)
        kwargs = dict(qr=qr, qw=q, kr=kr, v=v, pk=pk, pv=pv, rho=rho, offset=offset,
                      task_offset=task_offset, work_expert=work_expert, work_edge=work_edge, bp=layout.block_ptr,
                      prefix_valid=layout.prefix_valid, pm=pm, pz=pz, pu=pu)
        KERNELS._history_partial(
            **{name: Pointer(x) for name, x in kwargs.items()},
            T=t, EXPERTS=host.Nc*h, SCALE=d**-.5, SQT=qr.stride(0), SQH=qr.stride(1),
            SWT=q.stride(0), SWH=q.stride(1), SKT=kr.stride(0), SKH=kr.stride(1),
            SVT=v.stride(0), SVH=v.stride(1), SPB=pk.stride(0), SPM=pk.stride(1), SPH=pk.stride(2),
            H=h, D=d, M=m, C=host.chunk_size, BQ=16, BC=16, BD=2**(d-1).bit_length(), INDEX32=index32,
            USE_HISTORY_TOKENS=mode != 'routed_states_only', USE_HISTORY_STATES=mode != 'current_state_only',
        )
    owners = [(b, row, min(row+16, host.block_ptr[b+1])) for b in range(host.Nc)
              for row in range(host.block_ptr[b], host.block_ptr[b+1], 16)]
    owner_block, owner_start, owner_end = map(i32, zip(*owners))
    out, lse = torch.empty_like(q), torch.empty(h, t)
    for work_id in range(len(owners)):
        for head in range(h):
            TorchTL.ids = (work_id, head)
            kwargs = dict(qw=q, pk=pk, pv=pv, oloc=oloc, lloc=lloc, edge_pos=edge_pos, pm=pm, pz=pz, pu=pu,
                          owner_block=owner_block, owner_start=owner_start, owner_end=owner_end,
                          prefix_valid=layout.prefix_valid, out=out, lse=lse)
            KERNELS._owner_merge(
                **{name: Pointer(x) for name, x in kwargs.items()},
                T=t, Q0=0, K=kh, SCALE=d**-.5, SWT=q.stride(0), SWH=q.stride(1),
                SPB=pk.stride(0), SPM=pk.stride(1), SPH=pk.stride(2),
                H=h, D=d, M=m, BR=16, BC=16, BD=2**(d-1).bit_length(), HISTORY=kh>0,
            )
    return out, lse


class AblationCPU(unittest.TestCase):
    def test_route_shapes_and_immutability(self):
        for kh in (0, 1, 3, 15):
            _, _, tok, state = fixture(kh=kh)
            for mode in MODES:
                with self.subTest(kh=kh, mode=mode):
                    old_t, old_s = tok.clone(), state.clone()
                    tr, sr = HOST._readout_routes(tok, state, mode)
                    self.assertEqual(tr.shape, tok.shape)
                    self.assertEqual(sr.shape, state.shape)
                    self.assertTrue(torch.equal(tok, old_t) and torch.equal(state, old_s))
                    self.assertTrue(torch.equal(sr[..., 0], state[..., 0]))
                    self.assertTrue(torch.equal(tr, torch.full_like(tok, -1) if mode == 'routed_states_only' else tok))
                    expected = state.clone()
                    if mode == 'current_state_only':
                        expected[..., 1:] = -1
                    self.assertTrue(torch.equal(sr, expected))

    def test_config_validation(self):
        self.assertEqual(HOST.BlockMajorConfig().phba_ablation, 'full')
        for mode in MODES:
            self.assertEqual(HOST.BlockMajorConfig(phba_ablation=mode).phba_ablation, mode)
        for bad in ('tokens_only', '', None, 1, True, [], torch.tensor(1)):
            with self.subTest(bad=str(bad)), self.assertRaises(ValueError):
                HOST.BlockMajorConfig(phba_ablation=bad)

    def test_fp64_staged_forward_and_all_six_gradients(self):
        for lengths, c, kh in [((11,), 3, 2), ((7, 1, 8), 3, 3), ((9,), 3, 0)]:
            host, _, tok, state = fixture(lengths, c, 2, kh)
            for mode in MODES:
                with self.subTest(lengths=lengths, mode=mode):
                    x = [v.requires_grad_() for v in inputs(sum(lengths))]
                    a = evaluate(x, host, tok, state, mode, staged=True)
                    b = evaluate(x, host, tok, state, mode)
                    for actual, expected in zip(a, b):
                        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                    weight = torch.randn_like(a[0])
                    ga = torch.autograd.grad(a[0], x, weight, retain_graph=True)
                    gb = torch.autograd.grad(b[0], x, weight)
                    for actual, expected in zip(ga, gb):
                        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)

    def test_gradcheck(self):
        host, _, tok, state = fixture((7,), 2, 1, 2)
        for mode in MODES:
            with self.subTest(mode=mode):
                x = tuple(v.requires_grad_() for v in inputs(7, h=1, d=2, m=2))
                self.assertTrue(torch.autograd.gradcheck(
                    lambda *values: evaluate(values, host, tok, state, mode, staged=True)[0], x, fast_mode=True,
                ))

    def test_topk_one_equivalence(self):
        host, _, tok, state = fixture((13, 7), 4, 2, 0)
        x = inputs(20)
        expected = evaluate(x, host, tok, state, 'full')[0]
        for mode in MODES:
            torch.testing.assert_close(evaluate(x, host, tok, state, mode)[0], expected, rtol=0, atol=0)

    def test_causality_and_packed_reset(self):
        host, _, tok, state = fixture((11, 9), 3, 2, 2)
        for mode in MODES:
            with self.subTest(mode=mode):
                x = inputs(20)
                expected = evaluate(x, host, tok, state, mode)[0]
                changed = [v.clone() for v in x]
                for v in changed:
                    v[7:11] += 2
                torch.testing.assert_close(evaluate(changed, host, tok, state, mode)[0][:7], expected[:7])
                for v in changed:
                    v[:11] -= 5
                torch.testing.assert_close(evaluate(changed, host, tok, state, mode)[0][11:], expected[11:])

    def test_routed_states_keeps_recurrent_history_gradient(self):
        host, _, tok, state = fixture((11,), 3, 1, 2)
        x = [v.requires_grad_() for v in inputs(11, h=1)]
        out = evaluate(x, host, tok, state, 'routed_states_only', staged=True)[0]
        grads = torch.autograd.grad(out[-1].sum(), x)
        self.assertGreater(grads[1][:9].abs().sum().item(), 0)
        self.assertGreater(grads[2][:9].abs().sum().item(), 0)
        self.assertEqual(grads[4][:9].abs().sum().item(), 0)

    def test_production_forward_kernel_bodies_on_cpu(self):
        cases = [((97,), 32, 3, 16, 3), ((65, 1, 33), 32, 2, 24, 17), ((33,), 32, 0, 16, 1), ((1,), 32, 7, 16, 3)]
        for lengths, c, kh, d, m in cases:
            host, layout, tok, state = fixture(lengths, c, 2, kh)
            values = inputs(sum(lengths), d=d, m=m, dtype=torch.float32)
            for mode in MODES:
                for index32 in (True, False):
                    with self.subTest(lengths=lengths, mode=mode, index32=index32):
                        actual = emulated_forward(values, host, layout, tok, state, mode, index32=index32)
                        expected = evaluate(values, host, tok, state, mode)
                        for a, b in zip(actual, expected[:2]):
                            self.assertTrue(torch.isfinite(a).all())
                            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_metadata_selects_actual_state_routes(self):
        def branch(route, nc, *args, **kwargs):
            return types.SimpleNamespace(route=route.clone(), capacity=route.numel())
        def derived(tok, layout, h, cap):
            return types.SimpleNamespace(derived=True)
        module = load_symbols(OPS / 'metadata.py', ['DirectMetadata', 'build_direct_metadata'], {
            '_branch': branch, 'derive_state': derived, 'kpad_tok': lambda x: x, 'kpad_state': lambda x: x,
            'fail': lambda code, message: ValueError(message),
        })
        _, layout, tok, state = fixture()
        for mode in MODES:
            with self.subTest(mode=mode):
                tr, sr = HOST._readout_routes(tok, state, mode)
                meta = module.build_direct_metadata(tr, sr, layout.Nc, derived_state_layout=layout,
                                                   derive_state_from_token=mode == 'full')
                if mode == 'full':
                    self.assertTrue(meta.state.derived)
                else:
                    self.assertTrue(torch.equal(meta.state.route, sr))
                    self.assertTrue(torch.equal(meta.tok.route, tr))

    def test_autograd_host_routing_contract(self):
        identity = lambda f: f
        captured = {}
        def branch(route, nc, *args, **kwargs):
            count, offset, rows = REF.metadata_reference(route, nc)
            rho = torch.full((route.numel(),), -1, dtype=torch.int32)
            flat = [r for seg in rows for r in seg]
            rho[:len(flat)] = torch.tensor(flat, dtype=torch.int32)
            return types.SimpleNamespace(count=count, offset=offset, rho=rho, source_slots=None)
        def metadata(tr, sr, nc, **kwargs):
            captured['metadata'] = (tr.clone(), sr.clone(), kwargs)
            return types.SimpleNamespace(tok=branch(tr, nc) if tr.shape[-1] else None, state=branch(sr, nc))
        def forward(qr, qw, kr, v, pk, pv, route, layout, **kwargs):
            captured['forward'] = (route.clone(), kwargs)
            return qr.clone(), torch.zeros(qr.shape[1], qr.shape[0])
        module = load_symbols(OPS / 'parallel.py', ['BlockMajorAttentionFn'], {
            'input_guard': identity, 'autocast_custom_fwd': identity, 'autocast_custom_bwd': identity,
            '_readout_routes': HOST._readout_routes, 'block_major_forward': forward,
            'workspace_plan': lambda *args: {'workspace_bytes': 0}, '_local_fa_batches': lambda *args: (),
        })
        meta_module = types.ModuleType('fla.ops.phba.metadata')
        meta_module.build_direct_metadata = metadata
        plan_module = types.ModuleType('fla.ops.phba.plans')
        plan_module.owner_branch_tape_plan = lambda *args: tuple(args)
        host, layout, tok, state = fixture((97,), 32, 2, 3)
        q = torch.randn(97, 2, 16)
        prefix = torch.randn(host.Nc, 3, 2, 16)
        with patch.dict(sys.modules, {'fla.ops.phba.metadata': meta_module, 'fla.ops.phba.plans': plan_module}):
            for mode in MODES:
                for need in (False, True):
                    with self.subTest(mode=mode, backward=need):
                        captured.clear()
                        ctx = types.SimpleNamespace(mark_non_differentiable=lambda *a: None)
                        ctx.save_for_backward = lambda *a: captured.update(saved=a)
                        module.BlockMajorAttentionFn.forward(ctx, q, q, q, q, prefix, prefix, tok, state, layout,
                                                            HOST.BlockMajorConfig(phba_ablation=mode), need)
                        self.assertTrue(torch.equal(captured['forward'][0], tok))
                        if need:
                            tr, sr = HOST._readout_routes(tok, state, mode)
                            self.assertTrue(torch.equal(captured['saved'][-2], tr))
                            self.assertTrue(torch.equal(captured['saved'][-1], sr))
                            self.assertEqual(captured['metadata'][2]['derive_state_from_token'], mode == 'full')
                            if mode != 'full':
                                self.assertIsNone(captured['forward'][1]['canonical_token'])
                        else:
                            self.assertNotIn('metadata', captured)

    def test_api_defaults_and_configuration_wiring(self):
        checks = [('fla/models/phba/configuration_phba.py', '__init__'), ('fla/layers/phba.py', '__init__'),
                  ('fla/ops/phba/chunk.py', 'chunk_phba')]
        for path, name in checks:
            tree = ast.parse((ROOT/path).read_text())
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
            fn = next(n for n in funcs if any(a.arg == 'phba_ablation' for a in n.args.args))
            defaults = dict(zip([a.arg for a in fn.args.args[-len(fn.args.defaults):]], fn.args.defaults))
            self.assertEqual(ast.literal_eval(defaults['phba_ablation']), 'full')
        for path, target in [('fla/models/phba/modeling_phba.py', 'PHBA'), ('fla/layers/phba.py', 'chunk_phba')]:
            tree = ast.parse((ROOT/path).read_text())
            calls = (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
            call = next(n for n in calls if n.func.id == target)
            self.assertIn('phba_ablation', [k.arg for k in call.keywords])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
