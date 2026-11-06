# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.phba import SequenceLengthPlan, chunk_phba
from fla.ops.phba.reference import HostLayout, reference_a_chunkwise
from fla.ops.phba.reference_float64 import highest_precision
from fla.ops.phba.routing import build_routes
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(
    not IS_NVIDIA or find_spec_cached('flash_attn') is None,
    reason='PHBA requires NVIDIA CUDA and flash-attn2.8.3',
)


@pytest.mark.parametrize(
    ('lengths', 'H', 'D', 'M', 'chunk_size', 'topk', 'dtype'),
    [
        pytest.param((67,), 2, 32, 16, 32, 2, torch.float16, id='short-tail-fp16'),
        pytest.param((129,), 2, 64, 32, 64, 4, torch.bfloat16, id='chunk-boundary-bf16'),
        pytest.param((65, 1, 33), 2, 72, 80, 32, 2, torch.float16, id='varlen-one-token-nonpower-width'),
        pytest.param((257, 129, 1), 2, 128, 128, 128, 8, torch.bfloat16, id='varlen-eight-routes'),
        pytest.param((33,), 1, 16, 16, 32, 1, torch.float16, id='no-token-history'),
        pytest.param((65, 65), 1, 32, 17, 64, 3, torch.bfloat16, id='nonpower-slots'),
        pytest.param((513,), 2, 128, 128, 256, 4, torch.bfloat16, id='chunk256'),
        pytest.param((1025,), 2, 128, 64, 512, 2, torch.float16, id='chunk512'),
    ],
)
def test_chunk(lengths, H, D, M, chunk_size, topk, dtype):
    torch.manual_seed(42)
    T = sum(lengths)
    inputs = [torch.randn(1, T, H, D, dtype=dtype, device=device) * .1 for _ in range(5)]
    inputs.append(torch.randn(1, T, H, M, dtype=dtype, device=device) * .3)
    actual = [x.clone().requires_grad_() for x in inputs]
    expected = [x.clone().requires_grad_() for x in inputs]
    plan = SequenceLengthPlan.from_host(lengths, device)
    layout, rtok, rstate = build_routes(
        q_rope=inputs[3].squeeze(0),
        k_rope=inputs[4].squeeze(0),
        length_plan=plan,
        chunk_size=chunk_size,
        topk=topk,
    )
    do = torch.randn_like(inputs[0])
    with highest_precision():
        tri, _ = chunk_phba(
            q=actual[0], k=actual[1], v=actual[2], q_rope=actual[3], k_rope=actual[4], gate=actual[5],
            chunk_size=chunk_size, topk=topk, length_plan=plan,
        )
        refs = [x.squeeze(0) for x in expected]
        ref = reference_a_chunkwise(
            q_raw=refs[0], q_rope=refs[3], k_rope=refs[4], k_raw=refs[1], v=refs[2], a_storage=refs[5],
            tau=8., layout=HostLayout(chunk_size, layout.block_ptr_host, layout.seq_block_ptr_host),
            rtok=rtok, rstate=rstate, local_impl='flash_attn', fa_deterministic=True,
        )[0].unsqueeze(0)
        tri_grads = torch.autograd.grad(tri, actual, do)
        ref_grads = torch.autograd.grad(ref, expected, do)
    output_limit = .03 if dtype == torch.bfloat16 else .006
    gradient_limit = .06 if dtype == torch.bfloat16 else .015
    assert_close('output', ref.float(), tri.float(), output_limit)
    for name, reference, result in zip(('dq', 'dk', 'dv', 'dq_rope', 'dk_rope', 'dgate'), ref_grads, tri_grads):
        assert torch.isfinite(result).all()
        assert_close(name, reference.float(), result.float(), gradient_limit)
        relative_max = (reference.float() - result.float()).abs().max() / reference.float().abs().max().clamp_min(1e-5)
        assert relative_max < gradient_limit


@pytest.mark.parametrize(('chunk_size', 'topk'), [(16, 2), (64, 17)], ids=['unsupported-chunk', 'too-many-routes'])
def test_rejected_geometry(chunk_size, topk):
    x = torch.zeros(1, 33, 1, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='chunk_size'):
        chunk_phba(q=x, k=x, v=x, q_rope=x, k_rope=x, gate=x, chunk_size=chunk_size, topk=topk)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16], ids=['fp16', 'bf16'])
def test_prefix_final_state(dtype):
    from fla.ops.phba.reference import gate_from_storage, local_cumsum, prefix_forward

    torch.manual_seed(42)
    lengths = (65, 17)
    plan = SequenceLengthPlan.from_host(lengths, device)
    tensors = [torch.randn(1, sum(lengths), 1, 32, device=device, dtype=dtype) * .1 for _ in range(6)]
    _, final = chunk_phba(
        q=tensors[0], k=tensors[1], v=tensors[2], q_rope=tensors[3], k_rope=tensors[4], gate=tensors[5],
        chunk_size=32, topk=2, length_plan=plan, output_final_state=True,
    )
    host = HostLayout.from_lengths(lengths, 32)
    g, s = gate_from_storage(tensors[5].squeeze(0), 8., dtype)
    with highest_precision():
        cumulative = local_cumsum(g, host)
        expected = [prefix_forward(s, x.squeeze(0), cumulative, host)[1] for x in tensors[1:3]]
    for name, reference, actual in zip(('final_k', 'final_v'), expected, final):
        assert actual.dtype == torch.float32 and not actual.requires_grad
        assert torch.isfinite(actual).all()
        assert_close(name, reference, actual, .006)


def test_naive_formula_gradients():
    from fla.ops.phba.naive import naive_phba

    torch.manual_seed(42)
    values = [torch.randn(1, 35, 1, 8, dtype=torch.float64, device=device) * .1 for _ in range(6)]
    leaves = [x.requires_grad_() for x in values]
    token_routes = torch.full((1, 35, 1), -1, device=device, dtype=torch.int32)
    token_routes[:, 32:, :] = 0
    state_routes = torch.full((1, 35, 2), -1, device=device, dtype=torch.int32)
    state_routes[:, 32:, 0] = 1

    def formula(*inputs):
        return naive_phba(
            q=inputs[0], k=inputs[1], v=inputs[2], q_rope=inputs[3], k_rope=inputs[4], gate=inputs[5],
            lengths=(35,), token_routes=token_routes, state_routes=state_routes, chunk_size=32,
        )

    assert torch.autograd.gradcheck(formula, tuple(leaves), fast_mode=True)
