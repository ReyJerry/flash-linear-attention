# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .naive import check_inputs as validate_inputs


class HyperbolicDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, h0, cu, curvature, scale, chunk_size, mode, save_checkpoints):
        import triton

        from .kernels import hyperbolic_delta_rule_chunk_fwd_kernel, hyperbolic_delta_rule_fwd_kernel

        B, T, H, K = q.shape
        V, N = v.shape[-1], h0.shape[0]
        total = B * T
        o = torch.empty_like(v, dtype=q.dtype)
        ht = torch.empty_like(h0)
        # floor(bos/BT)+sequence_index assigns disjoint checkpoint ranges, even
        # for zero-length and packed sequences, without reading cu_seqlens on CPU.
        slots = triton.cdiv(total, chunk_size) + N
        cp = torch.empty((slots, H, K, V) if save_checkpoints else (0,), device=q.device, dtype=torch.float32)
        common = dict(H=H, K=K, V=V, SCALE=scale, BT=chunk_size, SAVE=save_checkpoints)
        if mode == 'chunk':
            hyperbolic_delta_rule_chunk_fwd_kernel[(N * H, V)](
                q, k, v, beta, h0, cu, o, ht, cp, curvature,
                BK=max(16, triton.next_power_of_2(K)), **common, num_warps=4, enable_fp_fusion=False,
            )
        else:
            hyperbolic_delta_rule_fwd_kernel[(N * H, triton.cdiv(V, 16))](
                q, k, v, beta, h0, cu, o, ht, cp, curvature,
                BK=triton.next_power_of_2(K), BV=16, **common, num_warps=4, enable_fp_fusion=False,
            )
        ctx.save_for_backward(q, k, v, beta, h0, cu, cp, curvature)
        ctx.scale, ctx.chunk_size = scale, chunk_size
        ctx.set_materialize_grads(False)
        return o, ht

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dht):
        import triton

        from .kernels import hyperbolic_delta_rule_bwd_kernel

        q, k, v, beta, h0, cu, cp, curvature = ctx.saved_tensors
        B, T, H, K = q.shape
        V, N = v.shape[-1], h0.shape[0]
        do = torch.zeros_like(v, dtype=q.dtype) if do is None else do.contiguous()
        dht = torch.zeros_like(h0) if dht is None else dht.contiguous()
        nv = triton.cdiv(V, 16)
        dq = torch.empty((nv, B, T, H, K), device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv = torch.empty_like(v, dtype=torch.float32)
        db = torch.empty_like(beta, dtype=torch.float32)
        dh0 = torch.empty_like(h0)
        dc = torch.empty((nv, B, T, H), device=q.device, dtype=torch.float32)
        scratch = torch.empty((N, H, ctx.chunk_size, K, V), device=q.device, dtype=torch.float32)
        hyperbolic_delta_rule_bwd_kernel[(N * H, nv)](
            q, k, v, beta, cu, cp, do, dht, scratch, dq, dk, dv, db, dh0, dc, curvature,
            H=H, K=K, V=V, TOTAL=B * T, SCALE=ctx.scale,
            BT=ctx.chunk_size, BK=triton.next_power_of_2(K), BV=16,
            num_warps=4, enable_fp_fusion=False,
        )
        return (
            dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), dh0,
            None, dc.sum().reshape_as(curvature).to(curvature.dtype), None, None, None, None,
        )


def _apply_hyperbolic_delta_rule(
    q, k, v, beta, curvature, scale, initial_state, output_final_state,
    cu_seqlens, chunk_size, check_inputs, mode,
):
    if q.device.type != 'cuda':
        raise ValueError('Triton operators require CUDA; use the naive_* references on CPU.')
    if chunk_size not in (16, 32):
        raise ValueError('Triton checkpoint/chunk size must be 16 or 32.')
    N, scale = validate_inputs(q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_inputs)
    if q.shape[-1] > 256:
        raise ValueError('This implementation supports key head dimensions up to 256.')
    tensors = (q, k, v, beta) + (() if initial_state is None else (initial_state,))
    if any(x.dtype not in (torch.float16, torch.bfloat16, torch.float32) for x in tensors):
        raise ValueError('Triton inputs must use float16, bfloat16 or float32.')
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    B, T, H, K = q.shape
    if initial_state is None:
        h0 = torch.zeros((N, H, K, v.shape[-1]), device=q.device, dtype=torch.float32)
    else:
        h0 = initial_state.float().contiguous()
    cu = (torch.arange(B + 1, device=q.device, dtype=torch.int64) * T
          if cu_seqlens is None else cu_seqlens.contiguous())
    save = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, beta, h0))
    if isinstance(curvature, torch.Tensor):
        curvature_tensor = curvature.to(device=q.device, dtype=torch.float32).reshape(()).contiguous()
    else:
        curvature_tensor = torch.tensor(float(curvature), device=q.device, dtype=torch.float32)
    save = save or (torch.is_grad_enabled() and curvature_tensor.requires_grad)
    o, ht = HyperbolicDeltaRuleFunction.apply(
        q, k, v, beta, h0, cu, curvature_tensor, scale, chunk_size, mode, save,
    )
    return o, ht if output_final_state else None


def fused_recurrent_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    curvature: float | torch.Tensor = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_size: int = 16,
    check_inputs: bool = False,
):
    r"""Fused recurrent forward and checkpointed Triton backward.

    q/k: [B,T,H,K]; v: [B,T,H,V]; beta: [B,T,H] or [B,T,H,V].
    initial/final state: [N,H,K,V], canonical S=M/z; final state is FP32.
    Packed input uses B=1 and nondecreasing cu_seqlens [N+1] on the input device,
    with endpoints 0 and T. Empty sequences preserve their incoming state.

    Caller contract: finite inputs, beta in [0,1], nonzero keys,
    ||k||^2-c*v^2 > 0 and c*||S[:,v]||^2 < 1. Enable check_inputs to validate
    values (synchronizes the GPU). No clamping or implicit key normalization is
    performed. Curvature may be a scalar tensor; the Triton backward computes its
    first-order gradient. Double backward is not supported.
    """
    return _apply_hyperbolic_delta_rule(
        q, k, v, beta, curvature, scale, initial_state, output_final_state,
        cu_seqlens, checkpoint_size, check_inputs, 'fused_recurrent',
    )
