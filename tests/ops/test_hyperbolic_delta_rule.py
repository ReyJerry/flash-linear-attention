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
