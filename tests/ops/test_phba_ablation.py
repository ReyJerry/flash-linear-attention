# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CUDA acceptance tests for PHBA readout ablations, including both backward schedules."""

import pytest
import torch

from fla.ops.phba import SequenceLengthPlan, chunk_phba, plans
from fla.ops.phba.metadata import build_direct_metadata
from fla.ops.phba.naive import HostLayout, highest_precision, metadata_reference, reference_a_chunkwise
from fla.ops.phba.parallel import BlockMajorConfig, _readout_routes, block_major_attention
from fla.ops.phba.routing import build_routes
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(
    not IS_NVIDIA or find_spec_cached('flash_attn') is None,
    reason='PHBA ablation kernels require NVIDIA CUDA and flash-attn2.8.3',
)
MODES = ('full', 'current_state_only', 'routed_states_only')


def expected_routes(tok, state, mode):
    # keep the oracle independent of the production route helper
    tok, state = tok.clone(), state.clone()
    if mode == 'current_state_only':
        state[:, :, 1:].fill_(-1)
    if mode == 'routed_states_only':
        tok.fill_(-1)
    return tok, state


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fallback', [False, True], ids=['tape', 'fallback'])
@pytest.mark.parametrize(
    ('lengths', 'width', 'slots', 'chunk_size', 'topk'),
    [((97,), 32, 17, 32, 3), ((65, 1, 33), 72, 80, 32, 4), ((257, 129), 128, 128, 128, 8),
     ((33,), 16, 1, 32, 1), ((1, 17), 32, 17, 32, 4), ((1025,), 32, 17, 512, 2)],
)
def test_ablation_chunk(mode, dtype, fallback, lengths, width, slots, chunk_size, topk, monkeypatch):
    if fallback:
        monkeypatch.setattr(plans, 'OWNER_TAPE_CAP_BYTES', 0)
    original_plan = plans.owner_branch_tape_plan
    seen = []

    def checked_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        seen.append(plan.enabled)
        assert plan.enabled == (not fallback)
        return plan

    monkeypatch.setattr(plans, 'owner_branch_tape_plan', checked_plan)
    torch.manual_seed(42)
    t, h = sum(lengths), 2
    plan = SequenceLengthPlan.from_host(lengths, device)
    values = [torch.randn(1, t, h, width, dtype=dtype, device=device) * .1 for _ in range(5)]
    values.append(torch.randn(1, t, h, slots, dtype=dtype, device=device) * .3)
    actual = [x.clone().requires_grad_() for x in values]
    reference = [x.clone().requires_grad_() for x in values]
    layout, tok, state = build_routes(
        q_rope=values[3][0],
        k_rope=values[4][0],
        length_plan=plan,
        chunk_size=chunk_size,
        topk=topk,
    )
    tr, sr = expected_routes(tok, state, mode)
    do = torch.randn_like(values[0])
    with highest_precision():
        result, _ = chunk_phba(
            q=actual[0],
            k=actual[1],
            v=actual[2],
            q_rope=actual[3],
            k_rope=actual[4],
            gate=actual[5],
            chunk_size=chunk_size,
            topk=topk,
            length_plan=plan,
            phba_ablation=mode,
        )
        x = [v[0] for v in reference]
        expected = reference_a_chunkwise(
            q_raw=x[0],
            q_rope=x[3],
            k_rope=x[4],
            k_raw=x[1],
            v=x[2],
            a_storage=x[5],
            tau=8.,
            layout=HostLayout(chunk_size, layout.block_ptr_host, layout.seq_block_ptr_host),
            rtok=tr,
            rstate=sr,
            local_impl='flash_attn',
            fa_deterministic=True,
        )[0].unsqueeze(0)
        assert result.grad_fn is not None
        actual_grads = torch.autograd.grad(result, actual, do)
        expected_grads = torch.autograd.grad(expected, reference, do)
    assert seen
    output_limit = .03 if dtype == torch.bfloat16 else .006
    gradient_limit = .06 if dtype == torch.bfloat16 else .015
    assert_close('output', expected.float(), result.float(), output_limit)
    for name, ref, grad in zip(('dq', 'dk', 'dv', 'dq_rope', 'dk_rope', 'dgate'), expected_grads, actual_grads):
        assert torch.isfinite(grad).all()
        assert_close(name, ref.float(), grad.float(), gradient_limit)
        relative_max = (ref.float() - grad.float()).abs().max() / ref.float().abs().max().clamp_min(1e-5)
        assert relative_max < gradient_limit
    # no-grad forward must select the same memory, without relying on backward metadata
    with torch.no_grad(), highest_precision():
        inference, _ = chunk_phba(
            q=values[0], k=values[1], v=values[2], q_rope=values[3], k_rope=values[4], gate=values[5],
            chunk_size=chunk_size, topk=topk, length_plan=plan, phba_ablation=mode,
        )
    assert_close('no-grad output', result.float(), inference.float(), output_limit)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('topk', [1, 4, 16])
def test_ablation_metadata(mode, topk):
    torch.manual_seed(123)
    plan = SequenceLengthPlan.from_host((97, 65), device)
    q = torch.randn(162, 2, 32, dtype=torch.bfloat16, device=device)
    layout, tok, state = build_routes(q_rope=q, k_rope=q, length_plan=plan, chunk_size=32, topk=topk)
    old_tok, old_state = tok.clone(), state.clone()
    tr, sr = _readout_routes(tok, state, mode)
    expected = expected_routes(tok, state, mode)
    assert torch.equal(tr, expected[0]) and torch.equal(sr, expected[1])
    assert torch.equal(tok, old_tok) and torch.equal(state, old_state)
    meta = build_direct_metadata(
        tr, sr, layout.Nc, derived_state_layout=layout, derive_state_from_token=mode == 'full',
    )
    for route, branch in ((tr, meta.tok), (sr, meta.state)):
        if not route.shape[-1]:
            assert branch is None
            continue
        count, offset, rows = metadata_reference(route, layout.Nc)
        assert torch.equal(branch.count.cpu(), count)
        assert torch.equal(branch.offset.cpu(), offset)
        rho = branch.rho.cpu()
        assert [rho[int(offset[i]):int(offset[i+1])].tolist() for i in range(count.numel())] == rows
        assert (rho[int(offset[-1]):] == -1).all()


@pytest.mark.parametrize('fallback', [False, True])
def test_routed_states_has_no_direct_history_token_gradient(fallback, monkeypatch):
    if fallback:
        monkeypatch.setattr(plans, 'OWNER_TAPE_CAP_BYTES', 0)
    torch.manual_seed(50)
    plan = SequenceLengthPlan.from_host((97,), device)
    shape = (97, 1, 32)
    qr, qw, kr, value = [torch.randn(shape, device=device, dtype=torch.float16).requires_grad_() for _ in range(4)]
    layout, tok, state = build_routes(q_rope=qr, k_rope=kr, length_plan=plan, chunk_size=32, topk=3)
    pk, pv = [torch.randn(layout.Nc, 17, 1, 32, device=device, dtype=torch.float16).requires_grad_() for _ in range(2)]
    result, _ = block_major_attention(
        qr, qw, kr, value, pk, pv, tok, state, layout,
        config=BlockMajorConfig(phba_ablation='routed_states_only'),
    )
    grads = torch.autograd.grad(result[-1].float().sum(), (kr, value, pk, pv))
    assert torch.count_nonzero(grads[0][:96]) == 0
    assert torch.count_nonzero(grads[1][:96]) == 0
    assert grads[2].float().abs().sum() > 0 and grads[3].float().abs().sum() > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_topk_one_ablation_equivalence(dtype):
    torch.manual_seed(42)
    plan = SequenceLengthPlan.from_host((65, 33), device)
    values = [torch.randn(1, 98, 1, 32, device=device, dtype=dtype) * .1 for _ in range(6)]
    outputs, finals, gradients = [], [], []
    for mode in MODES:
        x = [v.clone().requires_grad_() for v in values]
        out, final = chunk_phba(
            q=x[0], k=x[1], v=x[2], q_rope=x[3], k_rope=x[4], gate=x[5],
            topk=1, chunk_size=32, length_plan=plan, phba_ablation=mode, output_final_state=True,
        )
        outputs.append(out)
        finals.append(final)
        gradients.append(torch.autograd.grad(out.float().sum(), x))
    for i in (1, 2):
        torch.testing.assert_close(outputs[0], outputs[i], rtol=0, atol=0)
        for a, b in zip(finals[0], finals[i]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(gradients[0], gradients[i]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
