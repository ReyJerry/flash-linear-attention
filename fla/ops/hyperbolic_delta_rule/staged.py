# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in staged unprotected WY autograd, preserving the strict TC precision."""

import torch
import triton
from torch.autograd.function import once_differentiable

from fla.ops.hyperbolic_delta_rule.staged_kernels import (
    hyperbolic_staged_output_bwd_kernel,
    hyperbolic_staged_output_fwd_kernel,
    hyperbolic_staged_parameters_bwd_kernel,
    hyperbolic_staged_prepare_fwd_kernel,
    hyperbolic_staged_state_bwd_kernel,
    hyperbolic_staged_state_fwd_kernel,
    hyperbolic_staged_values_bwd_kernel,
)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class StagedHyperbolicDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, h0, cu, curvature, scale, chunk_size):
        batch, length, heads, key_dim = q.shape
        value_dim, sequences = v.shape[-1], h0.shape[0]
        slots = triton.cdiv(batch * length, chunk_size) + sequences
        gram = torch.empty((slots, heads, chunk_size, chunk_size), dtype=torch.float32, device=q.device)
        query_key = torch.empty_like(gram)
        weight = torch.empty_like(v, dtype=torch.float32)
        checkpoints = torch.empty((slots, heads, key_dim, value_dim), dtype=torch.float32, device=q.device)
        output, final = torch.empty_like(v, dtype=q.dtype), torch.empty_like(h0)
        block_k = max(16, triton.next_power_of_2(key_dim))
        common = dict(H=heads, K=key_dim, BT=chunk_size, BK=block_k, num_warps=4, enable_fp_fusion=False)
        hyperbolic_staged_prepare_fwd_kernel[(slots * heads,)](
            q, k, cu, gram, query_key, N=sequences, **common,
        )
        hyperbolic_staged_state_fwd_kernel[(sequences * heads, triton.cdiv(value_dim, 16))](
            k, v, beta, h0, cu, curvature, gram, weight, checkpoints, final, V=value_dim, BV=16, **common,
        )
        hyperbolic_staged_output_fwd_kernel[(slots * heads, triton.cdiv(value_dim, 16))](
            q, v, cu, curvature, query_key, weight, checkpoints, output,
            V=value_dim, N=sequences, SCALE=scale, BV=16, **common,
        )
        ctx.save_for_backward(q, k, v, beta, cu, curvature, gram, query_key, weight, checkpoints, final)
        ctx.scale, ctx.chunk_size = scale, chunk_size
        ctx.set_materialize_grads(False)
        return output, final

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do, dht):
        q, k, v, beta, cu, curvature, gram, query_key, weight, checkpoints, final = ctx.saved_tensors
        heads, key_dim, value_dim = q.shape[-2], q.shape[-1], v.shape[-1]
        sequences, slots = final.shape[0], checkpoints.shape[0]
        do = torch.zeros_like(v, dtype=q.dtype) if do is None else do
        dht = torch.zeros_like(final) if dht is None else dht
        d_numerator = torch.empty_like(v, dtype=torch.float32)
        z = torch.empty_like(d_numerator)
        u = torch.empty_like(d_numerator)
        suffix = torch.empty_like(d_numerator)
        state_adjoint = torch.empty_like(checkpoints)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dv, dbeta = torch.empty_like(v), torch.empty_like(beta)
        dh0 = torch.empty_like(final)
        need_dc = ctx.needs_input_grad[6]
        dc = torch.empty((slots, heads, triton.cdiv(value_dim, 16)) if need_dc else (0,),
                         dtype=torch.float32, device=q.device)
        common = dict(
            H=heads, K=key_dim, V=value_dim, BT=ctx.chunk_size, BK=max(16, triton.next_power_of_2(key_dim)),
            BV=16, num_warps=4, enable_fp_fusion=False,
        )
        hyperbolic_staged_output_bwd_kernel[(slots * heads, triton.cdiv(value_dim, 16))](
            q, v, cu, curvature, query_key, weight, checkpoints, do, d_numerator, z, suffix,
            N=sequences, SCALE=ctx.scale, **common,
        )
        hyperbolic_staged_state_bwd_kernel[(sequences * heads, triton.cdiv(value_dim, 16))](
            q, k, v, beta, cu, curvature, gram, weight, checkpoints, final, dht,
            d_numerator, z, u, suffix, state_adjoint, dh0, **common,
        )
        hyperbolic_staged_values_bwd_kernel[(slots * heads, triton.cdiv(value_dim, 16))](
            k, v, beta, cu, curvature, gram, weight, checkpoints, z, u, suffix, dv, dbeta, dc,
            N=sequences, NEED_DC=need_dc, **common,
        )
        hyperbolic_staged_parameters_bwd_kernel[(slots * heads, triton.cdiv(key_dim, 32))](
            q, k, cu, weight, checkpoints, d_numerator, z, u, state_adjoint, dq, dk,
            N=sequences, **{name: value for name, value in common.items() if name != 'BK'},
        )
        return dq, dk, dv, dbeta, dh0, None, dc.sum().reshape_as(curvature) if need_dc else None, None, None
