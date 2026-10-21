# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Triton protected projection + existing WY/Chunk recurrence."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .chunk import chunk_hyperbolic_delta_rule
from .naive import check_inputs as validate_inputs
from .protected import _EMPTY_SCORE, _select_bank


class _ProtectedProjection(torch.autograd.Function):
    """Triton projection with analytic VJP and shared-Key gradient reduction."""

    @staticmethod
    def forward(ctx, k, v, bank_k, bank_v, bank_scores, curvature, eps):
        import triton
        from .protected_kernels import shared_key_gram_fwd_kernel, protected_kv_project_fwd_kernel

        B, T, H, K = k.shape
        V, R = v.shape[-1], bank_k.shape[-2]
        bt = min(32, triton.next_power_of_2(max(T, 1)))
        bk, br = max(16, triton.next_power_of_2(K)), max(16, triton.next_power_of_2(R))
        ainv = torch.empty((B, H, R, R), device=k.device, dtype=torch.float32)
        shared_key_gram_fwd_kernel[(B * H,)](
            bank_k, bank_scores, ainv, H=H, K=K, R=R, EPS=eps,
            BK=bk, BR=br, num_warps=4, enable_fp_fusion=False,
        )
        kb = torch.empty((B, T, H, V, K), device=k.device, dtype=torch.float32)
        vb = torch.empty((B, T, H, V), device=v.device, dtype=torch.float32)
        alpha = torch.empty((B, T, H, V, R), device=k.device, dtype=torch.float32)
        ginv = torch.empty((B, H, V, R, R), device=k.device, dtype=torch.float32)
        valid = torch.empty((B, T, H, V), device=k.device, dtype=torch.uint8)
        c = curvature.to(device=k.device, dtype=torch.float32).reshape(()).contiguous()
        protected_kv_project_fwd_kernel[(B * H * V,)](
            k, v, bank_k, bank_v, bank_scores, ainv, c, kb, vb, alpha, ginv, valid,
            T=T, H=H, K=K, V=V, R=R, EPS=eps, BT=bt, BK=bk, BR=br,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(k, v, bank_k, bank_v, bank_scores, c, alpha, ginv, valid, kb, vb)
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        return kb, vb

    @staticmethod
    @once_differentiable
    def backward(ctx, dkb, dvb):
        import triton
        from .protected_kernels import protected_kv_project_bwd_kernel

        k, v, bank_k, bank_v, bank_scores, c, alpha, ginv, valid, kb, vb = ctx.saved_tensors
        B, T, H, K = k.shape
        V, R = v.shape[-1], bank_k.shape[-2]
        dkb = torch.zeros_like(kb) if dkb is None else dkb.contiguous().float()
        dvb = torch.zeros_like(vb) if dvb is None else dvb.contiguous().float()
        dk_v = torch.empty((B, T, H, V, K), device=k.device, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        dbank_k_v = torch.empty((B, H, V, R, K), device=k.device, dtype=torch.float32)
        dbank_v = torch.empty_like(bank_v, dtype=torch.float32)
        dc = torch.empty((B, H, V), device=k.device, dtype=torch.float32)
        protected_kv_project_bwd_kernel[(B * H * V,)](
            k, v, bank_k, bank_v, bank_scores, c, alpha, ginv, valid, dkb, dvb,
            dk_v, dv, dbank_k_v, dbank_v, dc,
            T=T, H=H, K=K, V=V, R=R, BT=min(32, triton.next_power_of_2(max(T, 1))),
            BK=max(16, triton.next_power_of_2(K)), BR=max(16, triton.next_power_of_2(R)),
            num_warps=4, enable_fp_fusion=False,
        )
        # One shared Key bank receives the sum of all independent Value-channel VJPs.
        return dk_v.sum(dim=3).to(k.dtype), dv.to(v.dtype), dbank_k_v.sum(dim=2).to(bank_k.dtype), dbank_v.to(bank_v.dtype), None, dc.sum().reshape_as(c), None


def _run_chunk(q, k, v, beta, state, bank_k, bank_v, bank_scores, curvature, scale):
    kb, vb = _ProtectedProjection.apply(k, v, bank_k, bank_v, bank_scores, curvature, 1.0e-5)
    outs, states = [], []
    # The projected Key can differ by Value channel. Reuse the repository's
    # Triton WY/Chunk recurrence per channel; gradients are concatenated back.
    for vi in range(v.shape[-1]):
        oi, si = chunk_hyperbolic_delta_rule(
            q, kb[:, :, :, vi, :], vb[:, :, :, vi:vi + 1], beta[:, :, :, vi:vi + 1],
            curvature=curvature, scale=scale, initial_state=state[:, :, :, vi:vi + 1],
            output_final_state=True, chunk_size=16, check_inputs=False,
        )
        outs.append(oi)
        states.append(si)
    return torch.cat(outs, -1), torch.cat(states, -1)


def triton_protected_chunkwise_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    protected_keys=None, protected_values=None, protected_scores=None,
    protected_count=None, selection_scores=None, protection_rank=4,
    checkpoint_size=16, check_inputs=False, checkpoint=True,
    temperature=0.25, novelty_weight=0.5,
):
    if q.device.type != 'cuda':
        raise ValueError("Triton protected mode requires CUDA.")
    if checkpoint_size not in (16, 32):
        raise ValueError("checkpoint_size must be 16 or 32.")
    if not isinstance(protection_rank, int) or not 1 <= protection_rank <= 16:
        raise ValueError("protection_rank must be in [1, 16].")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if K > 128 or V > 128:
        raise ValueError("Protected Triton currently supports K,V <= 128.")
    validate_inputs(q, k, v, beta, initial_state, None, curvature, scale, check_inputs)
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    c = curvature.to(device=q.device, dtype=torch.float32).reshape(()).contiguous() if isinstance(curvature, torch.Tensor) else torch.tensor(float(curvature), device=q.device, dtype=torch.float32)
    state = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32) if initial_state is None else initial_state.float().contiguous()
    if protected_keys is None:
        bank_k = torch.zeros((B, H, protection_rank, K), device=q.device, dtype=torch.float32)
        bank_v = torch.zeros((B, H, protection_rank, V), device=q.device, dtype=torch.float32)
        bank_scores = torch.full((B, H, protection_rank), _EMPTY_SCORE, device=q.device, dtype=torch.float32)
    else:
        if protected_values is None:
            raise ValueError("protected_values is required with protected_keys.")
        bank_k, bank_v = protected_keys.float().contiguous(), protected_values.float().contiguous()
        if protected_scores is not None:
            bank_scores = protected_scores.float().contiguous()
        else:
            if protected_count is None:
                protected_count = (bank_k.square().sum(-1) > 0).sum(-1)
            if protected_count.ndim == 3:
                protected_count = protected_count.amax(-1)
            slots = torch.arange(protection_rank, device=q.device).view(1, 1, -1)
            bank_scores = torch.where(slots < protected_count[..., None], 0.0, _EMPTY_SCORE).float()
    if bank_k.shape != (B, H, protection_rank, K) or bank_v.shape != (B, H, protection_rank, V):
        raise ValueError("Protected bank shape mismatch.")
    scores = k.norm(dim=-1) if selection_scores is None else selection_scores.to(device=q.device, dtype=torch.float32)
    if scores.shape != (B, T, H):
        raise ValueError("selection_scores must have shape [B,T,H].")
    scale = K ** -0.5 if scale is None else float(scale)
    outputs = []
    for start in range(0, T, checkpoint_size):
        end = min(start + checkpoint_size, T)
        qc, kc, vc, bc, sc = q[:, start:end], k[:, start:end], v[:, start:end], beta[:, start:end], scores[:, start:end]

        def run(qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c):
            out, next_state = _run_chunk(qc, kc, vc, bc, state, bank_k, bank_v, bank_scores, c, scale)
            # Hard top-k forward, straight-through soft gradients across chunk boundaries.
            from .protected import _select_bank
            next_k, next_v, next_scores, _ = _select_bank(
                bank_k, bank_v, bank_scores, kc, vc, sc,
                temperature=temperature, novelty_weight=novelty_weight,
            )
            return out, next_state, next_k, next_v, next_scores

        args = (qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c)
        needs_grad = torch.is_grad_enabled() and any(x.requires_grad for x in args)
        result = activation_checkpoint(run, *args, use_reentrant=False) if checkpoint and needs_grad else run(*args)
        out, state, bank_k, bank_v, bank_scores = result
        outputs.append(out)
    output = torch.cat(outputs, dim=1) if outputs else q.new_zeros((B, 0, H, V))
    count = (bank_scores.detach() > (_EMPTY_SCORE / 2)).sum(-1).to(torch.int32)
    return output.to(q.dtype), state, bank_k, bank_v, bank_scores, count


__all__ = ["triton_protected_chunkwise_hyperbolic_delta_rule"]
