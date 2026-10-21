# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable chunkwise hyperbolic protection reference and dispatch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_EPS = 1.0e-5
_EMPTY_SCORE = -1.0e6


def _project_chunk_torch(k, v, bank_k, bank_v, bank_scores, curvature, eps=_EPS):
    """Project a whole chunk using one shared key Gram inverse and Sherman-Morrison.

    k: [B,T,H,K], v: [B,T,H,V], bank_k: [B,H,R,K], bank_v: [B,H,R,V].
    The key-only Gram inverse is computed once per chunk, then reused for every
    value channel and token while the protection bank remains unchanged.
    """
    B, T, H, K = k.shape
    V, R = v.shape[-1], bank_k.shape[-2]
    active = bank_scores > (_EMPTY_SCORE / 2)
    mask = active.to(k.dtype)
    pk = bank_k * mask.unsqueeze(-1)
    pv = bank_v * mask.unsqueeze(-1)
    eye = torch.eye(R, device=k.device, dtype=k.dtype).view(1, 1, R, R)
    inactive_eye = torch.diag_embed((~active).to(k.dtype))
    gram_k = pk @ pk.transpose(-1, -2) + eps * eye + inactive_eye
    inv_k = torch.linalg.inv(gram_k)
    u = torch.einsum('bhij,bhjv->bhiv', inv_k, pv)
    denom = 1.0 - curvature * (pv * u).sum(dim=2)
    feasible = denom > eps
    safe_denom = denom.clamp_min(eps)
    u_v = u.permute(0, 1, 3, 2)
    ginv = inv_k.unsqueeze(2) + (
        curvature / safe_denom
    )[..., None, None] * u_v.unsqueeze(-1) * u_v.unsqueeze(-2)
    ginv = torch.where(feasible[..., None, None], ginv, inv_k.unsqueeze(2))
    rhs_k = torch.einsum('bhrk,bthk->bthr', pk, k)
    rhs = rhs_k.unsqueeze(3) - curvature * pv.permute(0, 1, 3, 2).unsqueeze(1) * v.unsqueeze(-1)
    alpha = torch.einsum('bhvij,bthvj->bthvi', ginv, rhs)
    kbar = k.unsqueeze(3) - torch.einsum('bhrk,bthvr->bthvk', pk, alpha)
    vbar = v - torch.einsum('bhrv,bthvr->bthv', pv, alpha)
    norm = kbar.square().sum(dim=-1) - curvature * vbar.square()
    valid = feasible.permute(0, 1, 2).unsqueeze(1) & (norm > eps)
    kbar = torch.where(valid.unsqueeze(-1), kbar, k.unsqueeze(3).expand_as(kbar))
    vbar = torch.where(valid, vbar, v)
    return kbar, vbar


def _recurrent_chunk_torch(q, k, v, beta, state, curvature, scale):
    """Sequential PyTorch oracle for one chunk; differentiable through every update."""
    outputs, realized = [], []
    for t in range(q.shape[1]):
        kt, qt, vt, bt = k[:, t], q[:, t], v[:, t], beta[:, t]
        # [B,H,V,K] and [B,H,K,V]
        residual = vt - torch.einsum('bhvk,bhkv->bhv', kt, state.permute(0, 1, 3, 2))
        a = kt.square().sum(dim=-1) - curvature * vt.square()
        valid = a > _EPS
        safe_a = torch.where(valid, a, torch.ones_like(a))
        u = torch.where(valid, bt * residual / safe_a, torch.zeros_like(residual))
        den = torch.where(valid, 1.0 + curvature * vt * u, torch.ones_like(u))
        candidate = (state + kt.permute(0, 1, 3, 2) * u.unsqueeze(-2)) / den.unsqueeze(-2)
        state = torch.where(valid.unsqueeze(-2), candidate, state)
        outputs.append(scale * torch.einsum('bhk,bhkv->bhv', qt, state))
        realized.append(torch.einsum('bhk,bhkv->bhv', k[:, t], state))
    if outputs:
        return torch.stack(outputs, dim=1), state, torch.stack(realized, dim=1)
    return v, state, v


def _select_bank(bank_k, bank_v, bank_scores, candidate_k, candidate_v, candidate_scores,
                 temperature=0.25, novelty_weight=0.5):
    """Hard top-k forward with a straight-through soft selection gradient.

    The selected forward bank contains exact candidate associations. The soft
    surrogate supplies gradients to candidate scores and association contents.
    """
    R = bank_k.shape[-2]
    active = bank_scores > (_EMPTY_SCORE / 2)
    if bank_k.shape[-2]:
        similarity = torch.einsum('bthk,bhrk->bthr', candidate_k, bank_k).abs()
        similarity = similarity.masked_fill(~active[:, None], 0.0)
        novelty = 1.0 - similarity.amax(dim=-1).clamp(0.0, 1.0)
    else:
        novelty = candidate_scores.new_ones(candidate_scores.shape)
    new_scores = torch.tanh(candidate_scores) + novelty_weight * novelty
    all_k = torch.cat((bank_k, candidate_k.permute(0, 2, 1, 3)), dim=2)
    all_v = torch.cat((bank_v, candidate_v.permute(0, 2, 1, 3)), dim=2)
    all_scores = torch.cat((bank_scores, new_scores.permute(0, 2, 1)), dim=-1)
    indices = all_scores.topk(R, dim=-1).indices
    hard = F.one_hot(indices, num_classes=all_scores.shape[-1]).to(all_scores.dtype)
    logits = all_scores.unsqueeze(-2) / max(float(temperature), 1.0e-3) + 12.0 * hard
    soft = torch.softmax(logits, dim=-1)
    weights = hard + soft - soft.detach()
    next_k = torch.einsum('bhrn,bhnk->bhrk', weights, all_k)
    next_v = torch.einsum('bhrn,bhnv->bhrv', weights, all_v)
    next_scores = (weights * all_scores.unsqueeze(-2)).sum(dim=-1)
    count = (next_scores > (_EMPTY_SCORE / 2)).sum(dim=-1).to(torch.int32)
    return next_k, next_v, next_scores, count


def naive_protected_chunkwise_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    protected_keys=None, protected_values=None, protected_scores=None,
    protected_count=None, selection_scores=None, protection_rank=4,
    chunk_size=16, checkpoint=True, temperature=0.25, novelty_weight=0.5,
):
    """Pure PyTorch reference with chunkwise Gram reuse and differentiable bank replacement."""
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError('Expected q/k [B,T,H,K] and v [B,T,H,V].')
    if chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32.')
    if not isinstance(protection_rank, int) or not 1 <= protection_rank <= 16:
        raise ValueError('protection_rank must be in [1, 16].')
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    c = torch.as_tensor(curvature, device=q.device, dtype=dtype).reshape(())
    state = q.new_zeros(B, H, K, V, dtype=dtype) if initial_state is None else initial_state.to(dtype)
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    bf = beta.to(dtype)
    if bf.ndim == 3:
        bf = bf.unsqueeze(-1).expand_as(vf)
    if selection_scores is None:
        selection_scores = kf.norm(dim=-1)
    if protected_keys is None:
        bank_k = kf.new_zeros(B, H, protection_rank, K)
        bank_v = vf.new_zeros(B, H, protection_rank, V)
        bank_scores = kf.new_full((B, H, protection_rank), _EMPTY_SCORE)
    else:
        bank_k, bank_v = protected_keys.to(dtype), protected_values.to(dtype)
        if protected_scores is not None:
            bank_scores = protected_scores.to(dtype)
        else:
            count = protected_count
            if count is None:
                count = (bank_k.square().sum(-1) > 0).sum(-1)
            slots = torch.arange(protection_rank, device=q.device).view(1, 1, -1)
            bank_scores = torch.where(slots < count[..., None], kf.new_zeros(()), kf.new_full((), _EMPTY_SCORE))
    if bank_k.shape != (B, H, protection_rank, K) or bank_v.shape != (B, H, protection_rank, V):
        raise ValueError('Protected bank shape mismatch.')
    scale = K ** -0.5 if scale is None else float(scale)
    outputs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        qc, kc, vc, bc = qf[:, start:end], kf[:, start:end], vf[:, start:end], bf[:, start:end]
        sc = selection_scores[:, start:end].to(dtype)

        def run_chunk(qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c):
            kbar, vbar = _project_chunk_torch(kc, vc, bank_k, bank_v, bank_scores, c)
            out, next_state, realized = _recurrent_chunk_torch(qc, kbar, vbar, bc, state, c, scale)
            next_bank_k, next_bank_v, next_scores, next_count = _select_bank(
                bank_k, bank_v, bank_scores, kc, realized, sc,
                temperature=temperature, novelty_weight=novelty_weight,
            )
            return out, next_state, next_bank_k, next_bank_v, next_scores, next_count

        args = (qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c)
        if checkpoint and torch.is_grad_enabled() and any(x.requires_grad for x in args):
            result = activation_checkpoint(run_chunk, *args, use_reentrant=False)
        else:
            result = run_chunk(*args)
        out, state, bank_k, bank_v, bank_scores, bank_count = result
        outputs.append(out)
    output = torch.cat(outputs, dim=1).to(q.dtype) if outputs else q.new_zeros(B, 0, H, V)
    return output, state, bank_k, bank_v, bank_scores, bank_count


def protected_recurrent_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    output_final_state=False, protected_keys=None, protected_values=None,
    protected_count=None, protection_rank=4, checkpoint_size=16, check_inputs=False,
    protected_scores=None, selection_scores=None, mode='auto', checkpoint=True,
    temperature=0.25, novelty_weight=0.5,
):
    """Dispatch the differentiable chunkwise protected rule.

    mode='naive' selects the PyTorch oracle; mode='triton' selects the Triton
    projection plus the existing WY/triangular-solve chunk operator.
    """
    if mode not in ('auto', 'naive', 'triton'):
        raise ValueError("mode must be 'auto', 'naive', or 'triton'.")
    use_triton = q.device.type == 'cuda' and mode != 'naive'
    if use_triton:
        from .protected_triton import triton_protected_chunkwise_hyperbolic_delta_rule
        result = triton_protected_chunkwise_hyperbolic_delta_rule(
            q, k, v, beta, curvature, scale, initial_state, protected_keys,
            protected_values, protected_scores, protected_count, selection_scores,
            protection_rank, checkpoint_size, check_inputs, checkpoint, temperature, novelty_weight,
        )
    else:
        result = naive_protected_chunkwise_hyperbolic_delta_rule(
            q, k, v, beta, curvature, scale, initial_state, protected_keys,
            protected_values, protected_scores, protected_count, selection_scores,
            protection_rank, checkpoint_size, checkpoint, temperature, novelty_weight,
        )
    return result


__all__ = [
    'naive_protected_chunkwise_hyperbolic_delta_rule',
    'protected_recurrent_hyperbolic_delta_rule',
]
