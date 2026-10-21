# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton operator wrapper for explicit Lorentz-protected KV updates."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .naive import check_inputs as validate_inputs


class ProtectedHyperbolicDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, h0, pk0, pv0, pc0, curvature, scale, save_tape):
        import triton

        from .protected_kernels import protected_hyperbolic_fwd_kernel

        B, T, H, K = q.shape
        V, R = v.shape[-1], pk0.shape[3]
        o, ht = torch.empty_like(v, dtype=q.dtype), torch.empty_like(h0)
        pk1, pv1, pc1 = torch.empty_like(pk0), torch.empty_like(pv0), torch.empty_like(pc0)
        tape_s = torch.empty((B, T, H, V, K) if save_tape else (0,), device=q.device, dtype=torch.float32)
        tape_pk = torch.empty((B, T, H, V, R, K) if save_tape else (0,), device=q.device, dtype=torch.float32)
        tape_pv = torch.empty((B, T, H, V, R) if save_tape else (0,), device=q.device, dtype=torch.float32)
        tape_pc = torch.empty((B, T, H, V) if save_tape else (0,), device=q.device, dtype=torch.int32)
        common = dict(H=H, K=K, V=V, R=R, T=T, SCALE=scale, SAVE=save_tape,
                      BK=max(16, triton.next_power_of_2(K)), BR=max(16, triton.next_power_of_2(R)))
        protected_hyperbolic_fwd_kernel[(B * H * V,)](
            q, k, v, beta, h0, pk0, pv0, pc0, o, ht, pk1, pv1, pc1,
            tape_s, tape_pk, tape_pv, tape_pc, curvature, **common,
            num_warps=8, enable_fp_fusion=False,
        )
        ctx.save_for_backward(q, k, v, beta, h0, tape_s, tape_pk, tape_pv, tape_pc, curvature)
        ctx.scale, ctx.rank, ctx.save_tape = scale, R, save_tape
        ctx.mark_non_differentiable(pk1, pv1, pc1)
        ctx.set_materialize_grads(False)
        return o, ht, pk1, pv1, pc1

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dht, dpk, dpv, dpc):
        import triton

        from .protected_kernels import protected_hyperbolic_bwd_kernel

        q, k, v, beta, h0, tape_s, tape_pk, tape_pv, tape_pc, curvature = ctx.saved_tensors
        B, T, H, K = q.shape
        V, R = v.shape[-1], ctx.rank
        if not ctx.save_tape:
            raise RuntimeError("Protected backward was invoked without saved state tapes.")
        do = torch.zeros_like(v) if do is None else do.contiguous()
        dht = torch.zeros_like(h0) if dht is None else dht.contiguous()
        dq = torch.empty((B, T, H, V, K), device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv, db = torch.empty_like(v, dtype=torch.float32), torch.empty_like(beta, dtype=torch.float32)
        dh0, dc = torch.empty_like(h0), torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        protected_hyperbolic_bwd_kernel[(B * H * V,)](
            q, k, v, beta, tape_s, tape_pk, tape_pv, tape_pc, curvature,
            do, dht, dq, dk, dv, db, dh0, dc,
            H=H, K=K, V=V, R=R, T=T, SCALE=ctx.scale,
            BK=max(16, triton.next_power_of_2(K)), BR=max(16, triton.next_power_of_2(R)), SAVE=save_tape,
            num_warps=8, enable_fp_fusion=False,
        )
        return (
            dq.sum(dim=3).to(q.dtype), dk.sum(dim=3).to(k.dtype), dv.to(v.dtype), db.to(beta.dtype),
            dh0, None, None, None, dc.sum().reshape_as(curvature).to(curvature.dtype), None, None,
        )


def protected_recurrent_hyperbolic_delta_rule_triton(
    q, k, v, beta, curvature, scale, initial_state, protected_keys, protected_values,
    protected_count, protection_rank, checkpoint_size, check_inputs,
):
    if q.device.type != "cuda":
        raise ValueError("Triton protected updates require CUDA.")
    if not isinstance(protection_rank, int) or not 1 <= protection_rank <= 16:
        raise ValueError("Triton protection_rank must be in [1, 16].")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if K > 128 or V > 128:
        raise ValueError("Protected Triton kernels support K,V <= 128.")
    cval = float(curvature.detach().item()) if isinstance(curvature, torch.Tensor) else float(curvature)
    if not torch.isfinite(torch.tensor(cval)) or cval <= 0:
        raise ValueError("Protected updates require finite curvature > 0.")
    validate_inputs(q, k, v, beta, initial_state, None, curvature, scale, check_inputs)
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    h0 = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32) if initial_state is None else initial_state.float().contiguous()
    if protected_keys is None:
        pk0 = torch.zeros((B, H, V, protection_rank, K), device=q.device, dtype=torch.float32)
        pv0 = torch.zeros((B, H, V, protection_rank), device=q.device, dtype=torch.float32)
        pc0 = torch.zeros((B, H, V), device=q.device, dtype=torch.int32)
    else:
        if protected_values is None:
            raise ValueError("protected_values is required with protected_keys.")
        pk0, pv0 = protected_keys.float().contiguous(), protected_values.float().contiguous()
        if protected_count is None:
            pc0 = (pk0.square().sum(-1) > 0).sum(-1).to(torch.int32)
        else:
            pc0 = protected_count.to(device=q.device, dtype=torch.int32).contiguous()
        if pk0.shape != (B, H, V, protection_rank, K) or pv0.shape != (B, H, V, protection_rank) or pc0.shape != (B, H, V):
            raise ValueError("Protected bank shape mismatch.")
        if bool(((pc0 < 0) | (pc0 > protection_rank)).any().item()):
            raise ValueError("protected_count must be in [0, protection_rank].")
    c = curvature.to(device=q.device, dtype=torch.float32).reshape(()).contiguous() if isinstance(curvature, torch.Tensor) else torch.tensor(cval, device=q.device, dtype=torch.float32)
    save_tape = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, beta, h0, c))
    scale = K ** -0.5 if scale is None else float(scale)
    return ProtectedHyperbolicDeltaRuleFunction.apply(q, k, v, beta, h0, pk0, pv0, pc0, c, scale, save_tape)


__all__ = ["protected_recurrent_hyperbolic_delta_rule_triton"]
