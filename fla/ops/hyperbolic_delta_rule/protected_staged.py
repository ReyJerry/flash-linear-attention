# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in protected trajectory scheduling with bounded backward workspaces."""

import torch
import triton
from torch.autograd.function import once_differentiable

from fla.ops.hyperbolic_delta_rule.protected_kernels import shared_key_gram_fwd_kernel
from fla.ops.hyperbolic_delta_rule.protected_staged_kernels import (
    protected_bank_partials_bwd_kernel,
    protected_shared_trajectory_bwd_kernel,
    protected_shared_trajectory_fwd_kernel,
    protected_value_partials_bwd_kernel,
    protected_value_trajectory_bwd_kernel,
    protected_value_trajectory_fwd_kernel,
)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _reverse_window_chunks(batch, heads, key_dim, value_dim, rank, chunk_size, num_chunks):
    """Bound expanded gradient storage to 256 MiB, or one unavoidable chunk."""
    per_chunk = 4 * batch * heads * value_dim * (chunk_size * (2 * key_dim + rank) + rank * key_dim)
    return max(1, min(num_chunks, 16, (256 * 1024 ** 2) // per_chunk))


class _ProtectedStaged(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, state, bank_k, bank_v, bank_scores, scores, c, scale, temperature, chunk_size, window_chunks):
        batch, length, heads, key_dim = q.shape
        value_dim, rank = v.shape[-1], bank_k.shape[-2]
        chunks = triton.cdiv(length, chunk_size)
        bk, br = max(16, triton.next_power_of_2(key_dim)), max(16, triton.next_power_of_2(rank))
        value_br = triton.next_power_of_2(rank)
        fp32 = dict(device=q.device, dtype=torch.float32)
        keys = torch.empty((batch, chunks, heads, rank, key_dim), **fp32)
        score_track = torch.empty((batch, chunks, heads, rank), **fp32)
        gates = torch.empty((batch, length, heads, rank), **fp32)
        ainv = torch.empty((batch, chunks, heads, rank, rank), **fp32)
        state_cp = torch.empty((batch, chunks, heads, key_dim, value_dim), **fp32)
        value_cp = torch.empty((batch, chunks, heads, rank, value_dim), **fp32)
        output = torch.empty_like(v)
        final = torch.empty_like(state)
        final_k, final_v, final_scores = torch.empty_like(bank_k), torch.empty_like(bank_v), torch.empty_like(bank_scores)
        protected_shared_trajectory_fwd_kernel[(batch * heads,)](
            k, scores, bank_k, bank_scores, keys, score_track, gates, final_k, final_scores,
            T=length, H=heads, K=key_dim, R=rank, NC=chunks, C=chunk_size,
            TEMP=temperature, BK=bk, BR=br, num_warps=4, enable_fp_fusion=False,
        )
        shared_key_gram_fwd_kernel[(batch * chunks * heads,)](
            keys, score_track, ainv, H=heads, K=key_dim, R=rank, EPS=1e-5,
            BK=bk, BR=br, num_warps=4, enable_fp_fusion=False,
        )
        protected_value_trajectory_fwd_kernel[(batch * heads * value_dim,)](
            q, k, v, beta, state, bank_v, c, keys, score_track, ainv, gates,
            output, final, final_v, state_cp, value_cp,
            T=length, H=heads, K=key_dim, V=value_dim, R=rank, NC=chunks, C=chunk_size,
            SCALE=scale, EPS=1e-5, BK=bk, BR=value_br, num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(q, k, v, beta, bank_k, bank_scores, scores, c,
                              keys, score_track, ainv, gates, state_cp, value_cp)
        ctx.scale, ctx.temperature, ctx.chunk_size, ctx.window_chunks = scale, temperature, chunk_size, window_chunks
        ctx.set_materialize_grads(False)
        return output, final, final_k, final_v, final_scores

    @staticmethod
    @once_differentiable
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dfinal, dfinal_k, dfinal_v, dfinal_scores):
        q, k, v, beta, bank_k, bank_scores, scores, c, keys, score_track, ainv, gates, state_cp, value_cp = ctx.saved_tensors
        batch, length, heads, key_dim = q.shape
        value_dim, rank = v.shape[-1], bank_k.shape[-2]
        chunk_size, window = ctx.chunk_size, ctx.window_chunks
        chunks = triton.cdiv(length, chunk_size)
        bk, br, bv = max(16, triton.next_power_of_2(key_dim)), max(
            16, triton.next_power_of_2(rank)), triton.next_power_of_2(value_dim)
        value_br = triton.next_power_of_2(rank)
        programs = batch * heads * value_dim
        fp32 = dict(device=q.device, dtype=torch.float32)
        do = torch.zeros_like(v) if do is None else do.contiguous()
        ds = torch.zeros((batch, heads, key_dim, value_dim), **fp32) if dfinal is None else dfinal.float().contiguous().clone()
        dy = torch.zeros((batch, heads, rank, value_dim), **
                         fp32) if dfinal_v is None else dfinal_v.float().contiguous().clone()
        dfinal_k = torch.zeros_like(bank_k) if dfinal_k is None else dfinal_k.contiguous()
        dfinal_scores = torch.zeros_like(bank_scores) if dfinal_scores is None else dfinal_scores.contiguous()
        dc = torch.zeros((programs,), **fp32)
        dq, dk, dv, dbeta = (torch.empty_like(x, dtype=torch.float32) for x in (q, k, v, beta))
        dkeys, dgates = torch.empty_like(keys), torch.empty_like(gates)
        dscores, dbank_k, dbank_scores = torch.empty_like(scores), torch.empty_like(bank_k), torch.empty_like(bank_scores)
        scratch_state = torch.empty((programs, chunk_size, key_dim), **fp32)
        scratch_pk = torch.empty_like(scratch_state)
        scratch_v = torch.empty((programs, chunk_size), **fp32)
        scratch_alpha = torch.empty((programs, chunk_size, rank), **fp32)
        scratch_y = torch.empty_like(scratch_alpha)
        scratch_valid = torch.empty((programs, chunk_size), device=q.device, dtype=torch.uint8)
        dq_part = torch.empty((window * chunk_size, programs, key_dim), **fp32)
        dk_part = torch.empty_like(dq_part)
        dg_part = torch.empty((window * chunk_size, programs, rank), **fp32)
        dpk_part = torch.empty((window, programs, rank, key_dim), **fp32)
        for stop in range(chunks, 0, -window):
            start = max(0, stop - window)
            protected_value_trajectory_bwd_kernel[(programs,)](
                q, k, v, beta, c, keys, score_track, ainv, gates, state_cp, value_cp, do, ds, dy, dc,
                scratch_state, scratch_pk, scratch_v, scratch_alpha, scratch_valid, scratch_y,
                dq_part, dk_part, dv, dbeta, dpk_part, dg_part,
                T=length, START=start, STOP=stop, P=programs, H=heads, K=key_dim, V=value_dim, R=rank,
                NC=chunks, C=chunk_size, SCALE=ctx.scale, EPS=1e-5, BK=bk, BR=value_br,
                num_warps=4, enable_fp_fusion=False,
            )
            tokens = min(stop * chunk_size, length) - start * chunk_size
            protected_value_partials_bwd_kernel[(tokens, batch * heads, triton.cdiv(key_dim, 32))](
                dq_part, dk_part, dg_part, dq, dk, dgates,
                T=length, START=start * chunk_size, P=programs, H=heads, K=key_dim, V=value_dim, R=rank,
                BV=bv, BR=br, BK=32, num_warps=4, enable_fp_fusion=False,
            )
            protected_bank_partials_bwd_kernel[(stop - start, batch * heads, triton.cdiv(rank * key_dim, 32))](
                dpk_part, dkeys, START=start, P=programs, H=heads, K=key_dim, V=value_dim, R=rank, NC=chunks,
                BV=bv, BLOCK=32, num_warps=4, enable_fp_fusion=False,
            )
        scratch_keys = torch.empty((batch * heads, chunk_size, rank, key_dim), **fp32)
        scratch_scores = torch.empty((batch * heads, chunk_size, rank), **fp32)
        protected_shared_trajectory_bwd_kernel[(batch * heads,)](
            k, scores, keys, score_track, gates, dkeys, dgates, dfinal_k, dfinal_scores,
            scratch_keys, scratch_scores, dk, dscores, dbank_k, dbank_scores,
            T=length, H=heads, K=key_dim, R=rank, NC=chunks, C=chunk_size,
            TEMP=ctx.temperature, BK=bk, BR=br, num_warps=4, enable_fp_fusion=False,
        )
        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype), ds,
            dbank_k, dy, dbank_scores, dscores, dc.sum().reshape_as(c), None, None, None, None,
        )


def _run_staged_protected(q, k, v, beta, state, bank_k, bank_v, bank_scores, scores, c, scale, temperature, chunk_size):
    """Run normalized dense inputs; selection and compatibility checks belong to the public wrapper."""
    batch, length, heads, key_dim = q.shape
    if length == 0:
        return q.new_zeros((batch, 0, heads, v.shape[-1])), state, bank_k, bank_v, bank_scores
    window = _reverse_window_chunks(batch, heads, key_dim, v.shape[-1], bank_k.shape[-2], chunk_size,
                                    triton.cdiv(length, chunk_size))
    return _ProtectedStaged.apply(q, k, v, beta, state, bank_k, bank_v, bank_scores, scores, c,
                                  scale, float(temperature), chunk_size, window)
