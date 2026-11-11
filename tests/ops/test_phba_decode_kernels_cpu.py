# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Execute kernel source bodies with CPU tensor semantics, NOT a Triton compiler test."""

import ast
import importlib.util
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("decode_ref", ROOT / "fla/ops/phba/decode.py")
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset
        self.dtype = types.SimpleNamespace(element_ty=tensor.dtype)

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    def __sub__(self, offset):
        return Pointer(self.tensor, self.offset - offset)


class TL:
    ids = (0, 0, 0)
    int64, int32, float32 = torch.int64, torch.int32, torch.float32
    constexpr = object()
    static_range = staticmethod(range)
    arange = staticmethod(torch.arange)
    exp = staticmethod(torch.exp)
    trans = staticmethod(lambda x: x.transpose(-2, -1))
    cast = staticmethod(lambda x, dtype: torch.as_tensor(x).to(dtype))
    full = staticmethod(lambda shape, value, dtype: torch.full(shape, value, dtype=dtype))
    sum = staticmethod(lambda x, axis: x.sum(dim=axis))
    max = staticmethod(lambda x, axis: x.max(dim=axis).values)
    min = staticmethod(lambda x, axis: x.min(dim=axis).values)
    cumsum = staticmethod(lambda x, axis: x.cumsum(dim=axis))
    dot = staticmethod(lambda x, y: x.float() @ y.float())
    maximum = staticmethod(lambda x, y: torch.maximum(torch.as_tensor(x), torch.as_tensor(y)))
    where = staticmethod(lambda mask, x, y: torch.where(torch.as_tensor(mask), torch.as_tensor(x), torch.as_tensor(y)))

    @classmethod
    def program_id(cls, axis):
        return torch.tensor(cls.ids[axis], dtype=torch.int64)

    @staticmethod
    def load(ptr, mask=True, other=0):
        index, mask = torch.broadcast_tensors(torch.as_tensor(ptr.offset).long(), torch.as_tensor(mask).bool())
        result = torch.full(index.shape, other, dtype=ptr.tensor.dtype)
        selected = index[mask]
        if selected.numel() and (selected.min() < 0 or selected.max() >= ptr.tensor.numel()):
            raise AssertionError("unmasked load outside allocated storage")
        result[mask] = ptr.tensor.reshape(-1)[selected]
        return result

    @staticmethod
    def store(ptr, value, mask=True):
        index, value, mask = torch.broadcast_tensors(torch.as_tensor(ptr.offset).long(), torch.as_tensor(value),
                                                    torch.as_tensor(mask).bool())
        selected = index[mask]
        if selected.numel() and (selected.min() < 0 or selected.max() >= ptr.tensor.numel()):
            raise AssertionError("unmasked store outside allocated storage")
        ptr.tensor.reshape(-1)[selected] = value[mask].to(ptr.tensor.dtype)


def load_kernels():
    tree = ast.parse((ROOT / "fla/ops/phba/decode_triton.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_')]
    for node in nodes:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    env = {"tl": TL}
    exec(compile(module, "phba_decode_kernel_cpu_semantics", "exec"), env)
    return types.SimpleNamespace(**env)


KERNELS = load_kernels()


def emulate_read(cache, q, qr, position, routes):
    b, h, d, k = cache.batch, cache.heads, cache.width, cache.topk
    partial = torch.full((b, h, k, d+2), float('nan'))
    out = torch.full((b, h, d), float('nan'))
    for bh in range(b*h):
        for route in range(k):
            TL.ids = (bh, route, 0)
            KERNELS._split_partial(
                *[Pointer(x.contiguous()) for x in (q, qr, cache.keys, cache.values, cache.pk, cache.pv, routes)],
                Pointer(partial), position, CAP=cache.capacity, NB=cache.nblocks, H=h, D=d, M=cache.slots,
                C=cache.chunk_size, K=k, BC=16, BD=2**(d-1).bit_length(),
                HISTORY_TOKENS=cache.mode != 'routed_states_only', HISTORY_STATES=cache.mode != 'current_state_only',
            )
        TL.ids = (bh, 0, 0)
        KERNELS._split_merge(Pointer(partial), Pointer(out), K=k, D=d,
                             BK=2**(k-1).bit_length(), BD=2**(d-1).bit_length())
    return out


class KernelCPU(unittest.TestCase):
    def test_boundary_kernel_both_states_and_masks(self):
        torch.manual_seed(42)
        cache = REF.PHBALayerCache(2, 99, 2, 72, 17, 32, 4, dtype=torch.float32, device='cpu')
        for tensor in (cache.keys, cache.values, cache.raw_tail, cache.state_k, cache.state_v):
            tensor.normal_(std=.2)
        cache.log_tail.uniform_(-.2, -.01)
        cache.write_tail.copy_(-cache.log_tail.expm1())
        expected = REF.block_update((cache.state_k.clone(), cache.state_v.clone()), cache.raw_tail,
                                    cache.values[:, 32:64], cache.log_tail, cache.write_tail)
        for bh in range(4):
            for tile in range(2):
                TL.ids = (bh, 0, tile)
                KERNELS._boundary_update(
                    *[Pointer(x) for x in (cache.raw_tail, cache.values, cache.log_tail, cache.write_tail,
                                          cache.state_k, cache.state_v, cache.pk, cache.pv)],
                    1, CAP=99, NB=cache.nblocks, H=2, D=72, M=17, C=32, BM=32, BD=64,
                )
        for actual, want in zip((cache.state_k, cache.state_v), expected):
            torch.testing.assert_close(actual, want, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(cache.pk[:, 2], cache.state_k.permute(0, 2, 1, 3), rtol=0, atol=0)

    def test_centroid_score_topk_kernels(self):
        torch.manual_seed(42)
        cache = REF.PHBALayerCache(2, 150, 2, 24, 3, 32, 4, dtype=torch.float32, device='cpu', stage='p1_route')
        cache.keys.normal_(std=.1)
        cache.centroids.fill_(float('nan'))
        query = torch.randn(2, 2, 24)*.1
        for bh in range(4):
            for block in range(4):
                TL.ids = (bh, block, 0)
                KERNELS._centroid_cache(Pointer(cache.keys), Pointer(cache.centroids), 0,
                                       CAP=150, NB=cache.nblocks, H=2, D=24, C=32, BD=32)
            TL.ids = (bh, 0, 0)
            KERNELS._routing_scores(Pointer(query), Pointer(cache.centroids), Pointer(cache.route_scores), 4,
                                   NB=cache.nblocks, H=2, D=24, BN=32, BD=32)
            KERNELS._routing_topk(Pointer(cache.route_scores), Pointer(cache.route_buffer), 4,
                                 NB=cache.nblocks, KH=3, BN=2**(cache.nblocks-1).bit_length())
        expected_centers = torch.stack([REF.block_centroid(cache.keys[:, b*32:(b+1)*32]) for b in range(4)], 1)
        torch.testing.assert_close(cache.centroids[:, :4], expected_centers)
        score = torch.einsum('bhd,bnhd->bhn', query, expected_centers)
        expected = torch.argsort(score, dim=-1, descending=True, stable=True)[..., :3].int()
        self.assertTrue(torch.equal(cache.route_buffer, expected))
        cache.route_scores.zero_()
        for bh in range(4):
            TL.ids = (bh, 0, 0)
            KERNELS._routing_topk(Pointer(cache.route_scores), Pointer(cache.route_buffer), 1,
                                 NB=cache.nblocks, KH=3, BN=2**(cache.nblocks-1).bit_length())
        self.assertTrue(torch.equal(cache.route_buffer, torch.tensor([0, -1, -1]).expand(2, 2, 3)))

    def test_split_kernels_empty_states_tails_and_modes(self):
        cases = [(1, 16, 3, 8, 'full'), (34, 24, 17, 4, 'routed_states_only'),
                 (67, 72, 33, 3, 'current_state_only'), (99, 32, 17, 4, 'full')]
        for length, d, m, k, mode in cases:
            torch.manual_seed(42)
            cache = REF.PHBALayerCache(2, length+5, 2, d, m, 32, k, dtype=torch.float32,
                                       device='cpu', stage='p1_route', mode=mode)
            for tensor in (cache.keys, cache.values, cache.pk, cache.pv):
                tensor.fill_(float('nan'))
            cache.pk[:, 0].zero_()
            cache.pv[:, 0].zero_()
            x = [torch.randn(2, length, 2, d)*.1 for _ in range(3)]
            gate = torch.randn(2, length, 2, m)*.3
            cache.prefill(x[0], x[1], x[2], gate)
            q, qr = [torch.randn(2, 2, d)*.1 for _ in range(2)]
            routes = cache.route(qr, length-1)
            expected = cache.read(q, qr, length-1, routes)
            actual = emulate_read(cache, q, qr, length-1, routes)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
