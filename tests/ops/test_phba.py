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
        pytest.param((67,), 2, 32, 16, 32, 2, torch.float16, id='short-tail-fp16', marks=pytest.mark.smoke),
        pytest.param((129,), 2, 64, 32, 64, 4, torch.bfloat16, id='chunk-boundary-bf16', marks=pytest.mark.smoke),
        pytest.param(
            (65, 1, 33), 2, 72, 80, 32, 2, torch.float16,
            id='varlen-one-token-nonpower-width',
            marks=pytest.mark.smoke,
        ),
        pytest.param((257, 129, 1), 2, 128, 128, 128, 8, torch.bfloat16, id='varlen-eight-routes', marks=pytest.mark.smoke),
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


@pytest.mark.parametrize('topk', [1, 4], ids=['no-history', 'history-and-tail'])
def test_routing_and_metadata(topk):
    from fla.ops.phba.metadata import build_direct_metadata, metadata_reference, sort_rho_segments
    from fla.ops.phba.routing.centroid import centroid_reference, compute_block_centroids
    from fla.ops.phba.routing.layout import verify_against_host
    from fla.ops.phba.routing.topk import check_route_invariants, route_reference

    plan = SequenceLengthPlan.from_host((97, 65), device)
    # integer block scores avoid near-tie ambiguity between reference and kernel reductions
    q = torch.ones(162, 2, 32, device=device, dtype=torch.bfloat16)
    k = (torch.arange(162, device=device) // 32 + 1).to(q.dtype)[:, None, None].expand_as(q).contiguous()
    layout, token, state = build_routes(q_rope=q, k_rope=k, length_plan=plan, chunk_size=32, topk=topk)
    verify_against_host(layout)
    check_route_invariants(token, state, layout)
    centroids = compute_block_centroids(k, layout)
    assert torch.equal(centroids, centroid_reference(k, layout.block_ptr_host))
    expected_token, expected_state, _ = route_reference(q, centroids, layout, topk - 1)
    assert torch.equal(token, expected_token)
    assert torch.equal(state, expected_state)
    metadata = build_direct_metadata(
        token, state, layout.Nc, backward_mode='query_owner_deterministic',
        retain_token_slots=True, derived_state_layout=layout,
    )
    for route, actual in ((token, metadata.tok), (state, metadata.state)):
        if route.shape[-1] == 0:
            assert actual is None
            continue
        count, offset, rows = metadata_reference(route, layout.Nc)
        assert torch.equal(actual.count.cpu(), count)
        assert torch.equal(actual.offset.cpu(), offset)
        assert sort_rho_segments(actual) == rows
        assert torch.all(actual.rho[int(offset[-1]):] == -1)


@pytest.mark.parametrize('width', [8, 17, 136], ids=['below-minimum', 'unaligned', 'above-maximum'])
def test_head_dimension_rejected_before_launch(width, monkeypatch):
    from fla.ops.phba import chunk

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid geometry reached FlashAttention initialization')

    monkeypatch.setattr(chunk, 'ensure_ready', unexpected)
    x = torch.zeros(1, 4, 1, width, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='head dimensions'):
        chunk_phba(q=x, k=x, v=x, q_rope=x, k_rope=x, gate=x)


def test_length_plan_batch_boundaries():
    x = torch.zeros(2, 4, 1, 16, device=device, dtype=torch.bfloat16)
    values = torch.ones_like(x)
    values[0].fill_(10)
    wrong = SequenceLengthPlan.from_host([8], device)
    with pytest.raises(ValueError, match='dense batch boundary'):
        chunk_phba(q=x, k=x, v=values, q_rope=x, k_rope=x, gate=x, chunk_size=32, topk=1, length_plan=wrong)
    actual, _ = chunk_phba(q=x, k=x, v=values, q_rope=x, k_rope=x, gate=x, chunk_size=32, topk=1)
    assert torch.equal(actual, values)


def test_length_plan_cumulative_boundaries(monkeypatch):
    from fla.ops.phba.plans import _resolve_length_plan

    plan = SequenceLengthPlan.from_host((4, 4), device)
    wrong = torch.tensor([0, 8], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match='different boundaries'):
        _resolve_length_plan(1, 8, plan.device, wrong, plan)
    assert _resolve_length_plan(1, 8, plan.device, plan.cu_seqlens.clone(), plan) is plan

    def unexpected(*args, **kwargs):
        pytest.fail('Reusing the owned length tensor must not compare device values')

    monkeypatch.setattr(torch, 'equal', unexpected)
    assert _resolve_length_plan(1, 8, plan.device, plan.cu_seqlens, plan) is plan


def test_low_memory_fallback_gradients(monkeypatch):
    from fla.ops.phba import plans

    torch.manual_seed(42)
    inputs = [torch.randn(1, 129, 2, 32, device=device, dtype=torch.bfloat16) * .1 for _ in range(6)]
    do = torch.randn_like(inputs[0])

    def evaluate():
        leaves = [x.clone().requires_grad_() for x in inputs]
        output, _ = chunk_phba(*leaves, chunk_size=32, topk=4)
        return output.detach(), torch.autograd.grad(output, leaves, do)

    expected, expected_grads = evaluate()
    monkeypatch.setattr(plans, 'OWNER_TAPE_CAP_BYTES', 0)
    monkeypatch.setattr(plans, 'PREFIX_REVERSE_WORKSPACE_CAP', 0)
    assert not plans.prefix_backward_plan(5, 2, 32, 32).parallel
    assert not plans.owner_branch_tape_plan(129, 2, 5, 32, 32, 3, 4, 32, 2).enabled
    actual, grads = evaluate()
    repeated, repeated_grads = evaluate()
    assert torch.equal(actual, repeated)
    assert_close('fallback_output', expected.float(), actual.float(), .03)
    for index, (reference, result, repeat) in enumerate(zip(expected_grads, grads, repeated_grads)):
        assert torch.isfinite(result).all()
        assert torch.equal(result.view(torch.uint8), repeat.view(torch.uint8))
        assert_close(f'fallback_gradient_{index}', reference.float(), result.float(), .06)


def test_length_plan_source_identity_and_mutation(monkeypatch):
    from fla.ops.phba.plans import _resolve_length_plan

    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    plan = SequenceLengthPlan.from_device(cu)
    with monkeypatch.context() as context:
        def unexpected(*args, **kwargs):
            pytest.fail('Unchanged source lengths must not require a device comparison')

        context.setattr(torch, 'equal', unexpected)
        assert _resolve_length_plan(1, 8, plan.device, cu, plan) is plan
    cu[1] = 3
    with pytest.raises(ValueError, match='different boundaries'):
        _resolve_length_plan(1, 8, plan.device, cu, plan)
