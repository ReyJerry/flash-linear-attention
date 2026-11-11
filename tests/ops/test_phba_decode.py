# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GPU acceptance for cached PHBA. CPU references do not replace these tests."""

import pytest
import torch

from fla.ops.phba import chunk_phba
from fla.ops.phba.decode import PHBALayerCache, block_centroid
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None,
                                reason='requires NVIDIA CUDA and the existing PHBA FlashAttention dependency')

CASES = [
    ('p0', 'full', 1, 32, 17, 32, 3, 31, 35, torch.float16),
    ('p1_state', 'full', 1, 128, 128, 128, 8, 127, 3, torch.bfloat16),
    ('p1_route', 'current_state_only', 2, 72, 17, 32, 4, 33, 35, torch.bfloat16),
    ('p1_split', 'full', 2, 72, 33, 64, 4, 63, 66, torch.bfloat16),
    ('p1_split', 'routed_states_only', 1, 16, 1, 32, 4, 1, 65, torch.float16),
    ('p1_split', 'current_state_only', 1, 32, 17, 32, 4, 65, 4, torch.float16),
    ('p1_split', 'full', 1, 32, 17, 256, 4, 255, 3, torch.bfloat16),
    ('p1_route', 'full', 1, 32, 17, 512, 2, 511, 3, torch.bfloat16),
    ('p1_split', 'full', 1, 32, 17, 32, 1, 31, 35, torch.float16),
]


@pytest.mark.parametrize('stage,mode,b,d,m,c,k,prompt,steps,dtype', CASES)
@torch.inference_mode()
def test_cached_decode_matches_full(stage, mode, b, d, m, c, k, prompt, steps, dtype):
    torch.manual_seed(42)
    h, length = 2, prompt + steps
    xs = [torch.randn(b, length, h, d, device=device, dtype=dtype)*.1 for _ in range(5)]
    xs.append(torch.randn(b, length, h, m, device=device, dtype=dtype)*.3)
    expected, _ = chunk_phba(*xs, chunk_size=c, topk=k, phba_ablation=mode)
    cache = PHBALayerCache(b, length, h, d, m, c, k, dtype=dtype, device=device, mode=mode, stage=stage)
    cache.prefill(xs[1][:, :prompt], xs[4][:, :prompt], xs[2][:, :prompt], xs[5][:, :prompt])
    out = torch.cat([cache.step(*(v[:, t:t+1] for v in xs)) for t in range(prompt, length)], dim=1)
    assert torch.isfinite(out).all()
    assert_close('cached outputs', expected[:, prompt:].float(), out.float(), .03 if dtype == torch.bfloat16 else .006)
    assert cache.length == length
    assert cache.raw_tail.shape[1] == c
    assert not out.requires_grad
    for block in range(length//c):
        torch.testing.assert_close(cache.centroids[:, block], block_centroid(cache.keys[:, block*c:(block+1)*c]),
                                   rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_routing_ties_empty_history_and_poisoned_capacity():
    cache = PHBALayerCache(2, 129, 2, 32, 17, 32, 8, device=device, stage='p1_split')
    for tensor in (cache.keys, cache.values, cache.pk, cache.pv, cache.centroids):
        tensor.fill_(float('nan'))
    cache.pk[:, 0].zero_()
    cache.pv[:, 0].zero_()
    zeros = torch.zeros(2, 97, 2, 32, device=device, dtype=torch.bfloat16)
    gate = torch.zeros(2, 97, 2, 17, device=device, dtype=torch.bfloat16)
    cache.prefill(zeros, zeros, zeros, gate)
    route = cache.route(zeros[:, 0], 97).clone()
    expected = torch.tensor([0, 1, 2, -1, -1, -1, -1], device=device, dtype=torch.int32).expand(2, 2, 7)
    assert torch.equal(route, expected)
    output = cache.step(zeros[:, :1], zeros[:, :1], zeros[:, :1], zeros[:, :1], zeros[:, :1], gate[:, :1])
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    assert (cache.route(zeros[:, 0], 0) == -1).all()
