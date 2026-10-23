# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA is required to compile and execute the Triton kernels.', allow_module_level=True)

from fla.ops.hyperbolic_delta_rule import (
    chunk_hyperbolic_delta_rule,
    fused_recurrent_hyperbolic_delta_rule,
    naive_recurrent_hyperbolic_delta_rule,
)

OPS = [fused_recurrent_hyperbolic_delta_rule, chunk_hyperbolic_delta_rule]


def make_case(T, K, V, dtype, vector=False, N=2, packed=False, c=1.0):
    torch.manual_seed(42)
    B, H = (1 if packed else N), 2
    q = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(dtype)
    v = (0.2 * torch.randn(B, T, H, V, device='cuda').tanh() / max(c, 1) ** 0.5).to(dtype)
    beta = torch.rand((B, T, H, V) if vector else (B, T, H), device='cuda', dtype=torch.float32)
    state = torch.nn.functional.normalize(torch.randn(N, H, K, V, device='cuda'), dim=-2)
    state = state * (0.1 / max(c, 1) ** 0.5)
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, state))


def compare_with_reference(op, args, *, cu=None, c=1.0, size=16, final_only=False):
    q, k, v, beta, state = args
    extra = {'chunk_size': size} if op is chunk_hyperbolic_delta_rule else {'checkpoint_size': size}
    actual, final = op(
        q, k, v, beta,
        curvature=c,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        check_inputs=True,
        **extra,
    )
    oracle_args = tuple(x.detach().double().requires_grad_() for x in args)
    expected, expected_final = naive_recurrent_hyperbolic_delta_rule(
        *oracle_args[:4],
        curvature=c,
        initial_state=oracle_args[4],
        output_final_state=True,
        cu_seqlens=cu,
    )
    tolerances = {
        torch.float32: (2e-4, 2e-3),
        torch.float16: (8e-4, 8e-3),
        torch.bfloat16: (4e-3, 4e-2),
    }
    atol, rtol = tolerances[q.dtype]
    torch.testing.assert_close(actual.float(), expected.to(q.dtype).float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final.float(), atol=2e-4, rtol=2e-3)
    do, dh = torch.randn_like(actual), torch.randn_like(final)
    if final_only:
        ga = torch.autograd.grad(final, args, dh)
        gr = torch.autograd.grad(expected_final, oracle_args, dh.double(), allow_unused=True)
        gr = tuple(torch.zeros_like(x) if g is None else g for x, g in zip(oracle_args, gr))
    else:
        ga = torch.autograd.grad((actual, final), args, (do, dh))
        gr = torch.autograd.grad((expected, expected_final), oracle_args, (do.double(), dh.double()))
    for original, a, r in zip(args, ga, gr):
        torch.testing.assert_close(a.float(), r.to(original.dtype).float(), atol=atol * 3, rtol=rtol * 3)


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('T', [0, 1, 17, 65])
@pytest.mark.parametrize('K,V', [(7, 5), (64, 48)])
def test_forward_backward(op, dtype, T, K, V):
    compare_with_reference(op, make_case(T, K, V, dtype))


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('c', [0.0, 0.5, 2.0])
@pytest.mark.parametrize('size', [16, 32])
def test_vector_gate_and_chunk_size(op, c, size):
    compare_with_reference(op, make_case(39, 17, 19, torch.float32, vector=True, c=c), c=c, size=size)


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('gate', [0.0, 1.0])
def test_singular_projection_and_no_write(op, gate):
    args = list(make_case(35, 16, 17, torch.float32, vector=True))
    args[3] = torch.full_like(args[3], gate, requires_grad=True)
    compare_with_reference(op, args)


@pytest.mark.parametrize('op', OPS)
def test_packed_varlen_including_empty_and_final_only_loss(op):
    cu = torch.tensor([0, 0, 1, 18, 49, 49, 52], device='cuda', dtype=torch.int64)
    args = make_case(52, 17, 19, torch.float32, vector=True, N=6, packed=True)
    compare_with_reference(op, args, cu=cu)
    compare_with_reference(op, args, cu=cu, final_only=True)


@pytest.mark.parametrize('op', OPS)
def test_noncontiguous_and_output_without_final_state(op):
    args = make_case(19, 17, 19, torch.float32, vector=True)
    args = tuple(torch.stack((x.detach(), torch.zeros_like(x)), -1)[..., 0].requires_grad_() for x in args)
    compare_with_reference(op, args)
    out, final = op(*args[:4], initial_state=args[4], output_final_state=False)
    assert final is None
    gradients = torch.autograd.grad(out.sum(), args)
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize('prefix_op', OPS)
@torch.no_grad()
def test_prefill_decode_cache(prefix_op):
    q, k, v, beta, state = make_case(69, 32, 33, torch.float32)
    ref, ref_state = naive_recurrent_hyperbolic_delta_rule(
        q, k, v, beta, initial_state=state, output_final_state=True,
    )
    prefix, state = prefix_op(
        q[:, :67], k[:, :67], v[:, :67], beta[:, :67], initial_state=state, output_final_state=True,
    )
    outs = [prefix]
    for t in range(67, 69):
        out, state = fused_recurrent_hyperbolic_delta_rule(
            q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], beta[:, t:t+1], initial_state=state, output_final_state=True,
        )
        outs.append(out)
    torch.testing.assert_close(torch.cat(outs, 1), ref, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(state, ref_state, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize('op', OPS)
def test_invalid_geometric_input_is_rejected(op):
    q, k, v, beta, _ = make_case(3, 7, 5, torch.float32)
    with pytest.raises(ValueError, match='Require'):
        op(q, k, torch.full_like(v, 2), beta, check_inputs=True)


def test_learnable_curvature_triton_gradient():
    args = list(make_case(9, 16, 8, torch.float32, vector=True, N=1))
    raw_c = torch.tensor(0.2, device='cuda', dtype=torch.float32, requires_grad=True)
    c = torch.nn.functional.softplus(raw_c)
    out, final = fused_recurrent_hyperbolic_delta_rule(
        *args[:4], curvature=c, initial_state=args[4], output_final_state=True, checkpoint_size=16,
    )
    actual = torch.autograd.grad(out.square().sum() + final.square().sum(), (*args, raw_c))
    oracle_args = tuple(x.detach().double().requires_grad_() for x in args)
    raw_ref = raw_c.detach().double().requires_grad_()
    c_ref = torch.nn.functional.softplus(raw_ref)
    ref_out, ref_final = naive_recurrent_hyperbolic_delta_rule(
        *oracle_args[:4], curvature=c_ref, initial_state=oracle_args[4], output_final_state=True,
    )
    expected = torch.autograd.grad(ref_out.square().sum() + ref_final.square().sum(), (*oracle_args, raw_ref))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.float(), e.to(a.dtype).float(), atol=5e-3, rtol=5e-2)


def test_protected_triton_forward_backward_and_shared_bank_gradients():
    from fla.ops.hyperbolic_delta_rule.protected import (
        naive_protected_chunkwise_hyperbolic_delta_rule,
        protected_recurrent_hyperbolic_delta_rule,
    )

    torch.manual_seed(73)
    B, T, H, K, V = 2, 19, 1, 16, 3
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1).detach().requires_grad_()
    v = (0.08 * torch.randn(B, T, H, V, device='cuda')).requires_grad_()
    beta = torch.rand(B, T, H, V, device='cuda', dtype=torch.float32, requires_grad=True)
    h0 = torch.zeros(B, H, K, V, device='cuda', dtype=torch.float32, requires_grad=True)
    raw_c = torch.tensor(0.2, device='cuda', dtype=torch.float32, requires_grad=True)
    curvature = torch.nn.functional.softplus(raw_c)
    actual = protected_recurrent_hyperbolic_delta_rule(
        q, k, v, beta, curvature=curvature, initial_state=h0,
        protection_rank=2, checkpoint_size=16, mode='triton', checkpoint=True,
    )
    oracle_inputs = tuple(x.detach().double().requires_grad_() for x in (q, k, v, beta, h0))
    raw_ref = raw_c.detach().double().requires_grad_()
    c_ref = torch.nn.functional.softplus(raw_ref)
    expected = naive_protected_chunkwise_hyperbolic_delta_rule(
        *oracle_inputs[:4], curvature=c_ref, initial_state=oracle_inputs[4],
        protection_rank=2, chunk_size=16, checkpoint=False,
    )
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=1e-2, rtol=1e-1)
    torch.testing.assert_close(actual[1].float(), expected[1].float(), atol=1e-2, rtol=1e-1)
    do, dh = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    ga = torch.autograd.grad((actual[0], actual[1]), (q, k, v, beta, h0, raw_c), (do, dh))
    gr = torch.autograd.grad(
        (expected[0], expected[1]), (*oracle_inputs, raw_ref), (do.double(), dh.double()),
    )
    for a, e in zip(ga, gr):
        torch.testing.assert_close(a.float(), e.float(), atol=2e-2, rtol=2e-1)


def test_protected_triton_checkpoint_peak_memory_and_parity():
    from fla.ops.hyperbolic_delta_rule.protected import protected_recurrent_hyperbolic_delta_rule

    torch.manual_seed(117)
    B, T, H, K, V = 2, 65, 1, 16, 4
    base = (
        torch.randn(B, T, H, K, device='cuda', dtype=torch.float32),
        torch.nn.functional.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1),
        0.06 * torch.randn(B, T, H, V, device='cuda'),
        torch.rand(B, T, H, V, device='cuda'),
        torch.zeros(B, H, K, V, device='cuda'),
    )
    def run(checkpoint):
        args = tuple(x.detach().clone().requires_grad_() for x in base)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = protected_recurrent_hyperbolic_delta_rule(
            *args[:4], curvature=0.5, initial_state=args[4], protection_rank=2,
            checkpoint_size=16, mode='triton', checkpoint=checkpoint,
        )
        loss = out[0].square().mean() + out[1].square().mean() + out[2].square().mean() * 1e-3 + out[3].square().mean() * 1e-3
        grads = torch.autograd.grad(loss, args)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return tuple(x.detach() for x in out[:2]), tuple(g.detach() for g in grads), peak
    result_plain, grads_plain, peak_plain = run(False)
    result_ckpt, grads_ckpt, peak_ckpt = run(True)
    for a, b in zip(result_plain, result_ckpt):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)
    for a, b in zip(grads_plain, grads_ckpt):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-2)
    # Memory reduction is hardware / allocator dependent. Record both peaks;
    # avoid a brittle universal threshold in CI.
    print(f'protected peak allocated: no_checkpoint={peak_plain}, checkpoint={peak_ckpt}')


def test_triton_smooth_bank_update_matches_exact_reference_vjp():
    from fla.ops.hyperbolic_delta_rule.protected import _select_bank_smooth
    from fla.ops.hyperbolic_delta_rule.protected_triton import _TritonSmoothBankUpdate

    torch.manual_seed(401)
    B, T, H, K, V, R = 1, 5, 2, 16, 4, 2
    inputs = (
        torch.randn(B, H, R, K, device='cuda', dtype=torch.float32, requires_grad=True),
        torch.randn(B, H, R, V, device='cuda', dtype=torch.float32, requires_grad=True),
        torch.randn(B, H, R, device='cuda', dtype=torch.float32, requires_grad=True),
        torch.randn(B, T, H, K, device='cuda', dtype=torch.float32, requires_grad=True),
        torch.randn(B, T, H, V, device='cuda', dtype=torch.float32, requires_grad=True),
        torch.randn(B, T, H, device='cuda', dtype=torch.float32, requires_grad=True),
    )
    actual = _TritonSmoothBankUpdate.apply(*inputs, 0.7)
    expected = _select_bank_smooth(*inputs, temperature=0.7)
    for a, e in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(a, e, atol=2e-4, rtol=2e-4)
    weights = tuple(torch.randn_like(x) for x in actual[:3])
    ga = torch.autograd.grad(actual[:3], inputs, weights)
    ge = torch.autograd.grad(expected[:3], inputs, weights)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=3e-3, rtol=3e-3)


def test_chunk_optional_readout_matches_single_query_recurrence():
    torch.manual_seed(433)
    B, T, H, K, V = 2, 16, 2, 16, 3
    q = torch.randn(B, T, H, K, device='cuda', requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).detach().requires_grad_()
    v = (0.04 * torch.randn(B, T, H, V, device='cuda')).requires_grad_()
    beta = torch.rand(B, T, H, V, device='cuda', requires_grad=True)
    h0 = torch.zeros(B, H, K, V, device='cuda', requires_grad=True)
    from fla.ops.hyperbolic_delta_rule.chunk import chunk_hyperbolic_delta_rule
    y, state, readout = chunk_hyperbolic_delta_rule(
        q, k, v, beta, curvature=0.5, initial_state=h0,
        output_final_state=True, chunk_size=16, readout_q=k,
    )
    y_ref, state_ref = chunk_hyperbolic_delta_rule(
        q, k, v, beta, curvature=0.5, initial_state=h0,
        output_final_state=True, chunk_size=16,
    )
    readout_ref, _ = chunk_hyperbolic_delta_rule(
        k, k, v, beta, curvature=0.5, initial_state=h0,
        output_final_state=True, chunk_size=16, scale=1.,
    )
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(readout, readout_ref, atol=1e-4, rtol=1e-3)
    loss = y.square().mean() + state.square().mean() + readout.square().mean()
    gr = torch.autograd.grad(loss, (q, k, v, beta, h0))
    ref_loss = y_ref.square().mean() + state_ref.square().mean() + readout_ref.square().mean()
    ref_gr = torch.autograd.grad(ref_loss, (q, k, v, beta, h0))
    for a, b in zip(gr, ref_gr):
        torch.testing.assert_close(a, b, atol=3e-3, rtol=3e-2)


def test_single_launch_fused_protected_chunk_vjp():
    """Fused forward must agree with separately compiled Triton VJP replay."""
    from fla.ops.hyperbolic_delta_rule.protected_triton import (
        _FusedProtectedChunkFunction, _TritonSmoothBankUpdate, _run_chunk,
    )

    torch.manual_seed(547)
    B, T, H, K, V, R = 2, 16, 2, 16, 3, 2
    q = torch.randn(B, T, H, K, device='cuda', requires_grad=True)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device='cuda'), dim=-1,
    ).detach().requires_grad_()
    v = (torch.randn(B, T, H, V, device='cuda') * 0.03).requires_grad_()
    beta = torch.sigmoid(torch.randn(B, T, H, V, device='cuda')).detach().requires_grad_()
    h0 = torch.zeros(B, H, K, V, device='cuda', requires_grad=True)
    bk = torch.zeros(B, H, R, K, device='cuda')
    bv = torch.zeros(B, H, R, V, device='cuda')
    bs = torch.full((B, H, R), -1e6, device='cuda')
    scores = torch.randn(B, T, H, device='cuda', requires_grad=True)
    curvature = torch.tensor(0.2, device='cuda', requires_grad=True)
    actual = _FusedProtectedChunkFunction.apply(
        q, k, v, beta, h0, bk, bv, bs, scores, curvature, K**-0.5, 0.7,
    )
    ref_out, ref_state, realized = _run_chunk(
        q, k, v, beta, h0, bk, bv, bs, curvature, K**-0.5,
    )
    ref_bk, ref_bv, ref_bs, _ = _TritonSmoothBankUpdate.apply(
        bk, bv, bs, k, realized, scores, 0.7,
    )
    reference = (ref_out, ref_state, ref_bk, ref_bv, ref_bs)
    for a, e in zip(actual, reference):
        torch.testing.assert_close(a.float(), e.float(), atol=3e-3, rtol=3e-2)
    upstream = tuple(torch.randn_like(x) for x in actual)
    targets = (q, k, v, beta, h0, scores, curvature)
    actual_grad = torch.autograd.grad(actual, targets, upstream, allow_unused=False)
    ref_grad = torch.autograd.grad(reference, targets, upstream, allow_unused=False)
    for a, e in zip(actual_grad, ref_grad):
        torch.testing.assert_close(a.float(), e.float(), atol=3e-3, rtol=3e-2)


def test_value_parallel_protected_chunk_parity():
    """Multi-batch/multi-value regression including gradient into shared Key bank."""
    from fla.ops.hyperbolic_delta_rule.protected_triton import (
        _FusedProtectedChunkFunction, _run_value_parallel_chunk,
    )
    torch.manual_seed(641)
    B, T, H, K, V, R = 2, 16, 2, 16, 16, 2
    q = torch.randn(B, T, H, K, device='cuda', requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1).detach().requires_grad_()
    v = (torch.randn(B, T, H, V, device='cuda') * 0.03).requires_grad_()
    beta = torch.rand(B, T, H, V, device='cuda', requires_grad=True)
    state = torch.zeros(B, H, K, V, device='cuda', requires_grad=True)
    bk = torch.nn.functional.normalize(torch.randn(B, H, R, K, device='cuda'), dim=-1).detach().requires_grad_()
    bv = (0.03 * torch.randn(B, H, R, V, device='cuda')).requires_grad_()
    bs = torch.zeros(B, H, R, device='cuda', requires_grad=True)
    score = torch.randn(B, T, H, device='cuda', requires_grad=True)
    c = torch.tensor(0.15, device='cuda', requires_grad=True)
    args = (q, k, v, beta, state, bk, bv, bs, score, c)
    slow = _FusedProtectedChunkFunction.apply(*args, K**-0.5, 0.7)
    fast = _run_value_parallel_chunk(*args, K**-0.5, 0.7)
    for a, b in zip(fast, slow):
        torch.testing.assert_close(a, b, atol=5e-3, rtol=5e-2)
    grad_outputs = tuple(torch.randn_like(x) for x in slow)
    gt = torch.autograd.grad(slow, args, grad_outputs)
    gp = torch.autograd.grad(fast, args, grad_outputs)
    for a, b in zip(gp, gt):
        torch.testing.assert_close(a, b, atol=8e-3, rtol=8e-2)


def test_value_parallel_multi_chunk_checkpoint_parity():
    from fla.ops.hyperbolic_delta_rule.protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule
    torch.manual_seed(642)
    B, T, H, K, V = 2, 33, 1, 16, 16
    base = (
        torch.randn(B, T, H, K, device='cuda'),
        torch.nn.functional.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1),
        0.02 * torch.randn(B, T, H, V, device='cuda'),
        torch.rand(B, T, H, V, device='cuda'),
        torch.zeros(B, H, K, V, device='cuda'),
    )
    def run(checkpoint):
        args = tuple(x.detach().requires_grad_(True) for x in base)
        out = triton_protected_chunkwise_hyperbolic_delta_rule(
            *args[:4], initial_state=args[4], curvature=0.2, protection_rank=2,
            checkpoint_size=16, kernel_strategy='value_parallel',
            checkpoint=checkpoint,
        )
        loss = sum(z.float().square().mean() for z in out[:5])
        return out, torch.autograd.grad(loss, args)
    a, ga = run(False)
    b, gb = run(True)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.float(), y.float(), atol=2e-4, rtol=2e-3)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('K,V', [(16, 16), (64, 48)])
@pytest.mark.parametrize('T', [1, 16, 35])
def test_tensor_core_hyperbolic_chunk_forward_backward(dtype, K, V, T):
    args = make_case(T, K, V, dtype, vector=True, N=2, c=0.5)
    q, k, v, beta, h0 = args
    a, sa = chunk_hyperbolic_delta_rule(
        q, k, v, beta, curvature=0.5, initial_state=h0,
        output_final_state=True, chunk_size=16, tensor_core=True,
    )
    q2, k2, v2, b2, s2 = (x.detach().clone().requires_grad_() for x in args)
    b, sb = naive_recurrent_hyperbolic_delta_rule(
        q2, k2, v2, b2, curvature=0.5, initial_state=s2,
        output_final_state=True,
    )
    atol, rtol = {torch.float32:(8e-4, 8e-3),torch.float16:(1e-2,3e-2),
                  torch.bfloat16:(3e-2,8e-2)}[dtype]
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(sa, sb, atol=atol, rtol=rtol)
    do, dh = torch.randn_like(a), torch.randn_like(sa)
    ga = torch.autograd.grad((a,sa), args, (do,dh))
    gr = torch.autograd.grad((b,sb), (q2,k2,v2,b2,s2), (do,dh))
    for x,y in zip(ga,gr):
        torch.testing.assert_close(x.float(),y.float(),atol=atol*3,rtol=rtol*3)


def test_tensor_core_hyperbolic_chunk_curvature_grad():
    args = make_case(33, 64, 32, torch.float32, vector=True, N=2, c=0.5)
    c = torch.tensor(0.5,device='cuda',requires_grad=True)
    out, h = chunk_hyperbolic_delta_rule(
        *args[:4], curvature=c, initial_state=args[4],
        output_final_state=True, tensor_core=True,
    )
    grads = torch.autograd.grad(out.square().mean()+h.square().mean(),(*args,c))
    assert all(torch.isfinite(g).all() for g in grads)
    qr = tuple(x.detach().clone().requires_grad_() for x in args)
    cr = c.detach().clone().requires_grad_()
    ref, href = naive_recurrent_hyperbolic_delta_rule(
        *qr[:4], curvature=cr, initial_state=qr[4], output_final_state=True,
    )
    gre = torch.autograd.grad(ref.square().mean()+href.square().mean(),(*qr,cr))
    for g,gg in zip(grads,gre):
        torch.testing.assert_close(g,gg,atol=5e-3,rtol=5e-2)
