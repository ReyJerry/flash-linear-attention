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


def test_protected_triton_forward_backward():
    from fla.ops.hyperbolic_delta_rule.protected import _protected_reference, protected_recurrent_hyperbolic_delta_rule

    torch.manual_seed(73)
    B, T, H, K, V = 1, 7, 1, 16, 3
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1).detach().requires_grad_()
    v = (0.12 * torch.randn(B, T, H, V, device='cuda')).requires_grad_()
    beta = torch.rand(B, T, H, V, device='cuda', dtype=torch.float32, requires_grad=True)
    h0 = torch.zeros(B, H, K, V, device='cuda', dtype=torch.float32, requires_grad=True)
    curvature = torch.tensor(0.5, device='cuda', dtype=torch.float32, requires_grad=True)
    actual = protected_recurrent_hyperbolic_delta_rule(
        q, k, v, beta, curvature=curvature, initial_state=h0, protection_rank=2,
    )
    oracle_inputs = tuple(x.detach().double().requires_grad_() for x in (q, k, v, beta, h0))
    c_ref = curvature.detach().double().requires_grad_()
    expected = _protected_reference(
        *oracle_inputs[:4], curvature=c_ref, scale=K ** -0.5,
        initial_state=oracle_inputs[4], protected_keys=None, protected_values=None,
        protected_count=None, protection_rank=2,
    )
    torch.testing.assert_close(actual[0].float(), expected[0].to(q.dtype).float(), atol=5e-3, rtol=5e-2)
    torch.testing.assert_close(actual[1].float(), expected[1].to(q.dtype).float(), atol=5e-3, rtol=5e-2)
    do, dh = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    ga = torch.autograd.grad((actual[0], actual[1]), (q, k, v, beta, h0, curvature), (do, dh))
    gr = torch.autograd.grad((expected[0], expected[1]), (*oracle_inputs, c_ref), (do.double(), dh.double()))
    for a, e in zip(ga, gr):
        torch.testing.assert_close(a.float(), e.to(a.dtype).float(), atol=1e-2, rtol=1e-1)
