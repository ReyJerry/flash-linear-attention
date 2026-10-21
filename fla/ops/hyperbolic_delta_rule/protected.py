# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Chunkwise hyperbolic KV protection with a differentiable bounded bank."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_EPS = 1.0e-5
_EMPTY_SCORE = -1.0e6


def _project_chunk_torch(k, v, bank_k, bank_v, bank_scores, curvature, eps=_EPS):
    """Project one chunk, reusing the key-only Gram inverse across value channels.

    Shapes: k [B,T,H,K], v [B,T,H,V], bank_k [B,H,R,K],
    bank_v [B,H,R,V], scores [B,H,R]. The Lorentz Gram for each value
    channel is obtained from (K K^T + eps I) by Sherman-Morrison.
    """
    B, T, H, K = k.shape
    V, R = v.shape[-1], bank_k.shape[-2]
    active = bank_scores > (_EMPTY_SCORE / 2)
    mask = active.to(k.dtype)
    pk = bank_k * mask.unsqueeze(-1)
    pv = bank_v * mask.unsqueeze(-1)
    eye = torch.eye(R, device=k.device, dtype=k.dtype).expand(B, H, R, R)
    inactive_eye = torch.diag_embed((~active).to(k.dtype))
    gram_k = pk @ pk.transpose(-1, -2) + eps * eye + inactive_eye
    inv_k = torch.linalg.inv(gram_k)
    # U = A^-1 Y, one shared A^-1 for every value channel.
    u = torch.einsum('bhij,bhjv->bhiv', inv_k, pv)
    denom = 1.0 - curvature * (pv * u).sum(dim=2)
    feasible = denom > eps
    safe_denom = torch.where(feasible, denom, torch.ones_like(denom))
    u_v = u.permute(0, 1, 3, 2)
    ginv = inv_k.unsqueeze(2) + (
        curvature / safe_denom
    )[..., None, None] * u_v.unsqueeze(-1) * u_v.unsqueeze(-2)
    ginv = torch.where(feasible[..., None, None], ginv, inv_k.unsqueeze(2))
    rhs_k = torch.einsum('bhrk,bthk->bthr', pk, k)
    y_vr = pv.permute(0, 1, 3, 2)
    rhs = rhs_k.unsqueeze(3) - curvature * y_vr[:, None] * v.unsqueeze(-1)
    alpha = torch.einsum('bhvij,bthvj->bthvi', ginv, rhs)
    kbar = k.unsqueeze(3) - torch.einsum('bhrk,bthvr->bthvk', pk, alpha)
    vbar = v - torch.einsum('bhrv,bthvr->bthv', pv, alpha)
    norm = kbar.square().sum(dim=-1) - curvature * vbar.square()
    valid = feasible.unsqueeze(1) & (norm > eps)
    kbar = torch.where(valid.unsqueeze(-1), kbar, k.unsqueeze(3).expand_as(kbar))
    vbar = torch.where(valid, vbar, v)
    return kbar, vbar


def _recurrent_chunk_torch(q, kbar, vbar, beta, state, curvature, scale, readout_k):
    """Reference recurrence for per-value projected keys."""
    outputs, realized = [], []
    for t in range(q.shape[1]):
        qt = q[:, t]
        kt = kbar[:, t]  # [B,H,V,K]
        vt, bt = vbar[:, t], beta[:, t]  # [B,H,V]
        state_v = state.permute(0, 1, 3, 2)  # [B,H,V,K]
        residual = vt - (kt * state_v).sum(dim=-1)
        a = kt.square().sum(dim=-1) - curvature * vt.square()
        valid = a > _EPS
        safe_a = torch.where(valid, a, torch.ones_like(a))
        u = torch.where(valid, bt * residual / safe_a, torch.zeros_like(residual))
        den = torch.where(valid, 1.0 + curvature * vt * u, torch.ones_like(u))
        candidate = (state_v + kt * u.unsqueeze(-1)) / den.unsqueeze(-1)
        state = torch.where(valid.unsqueeze(-1), candidate, state_v).permute(0, 1, 3, 2)
        outputs.append(scale * torch.einsum('bhk,bhkv->bhv', qt, state))
        realized.append(torch.einsum('bhk,bhkv->bhv', readout_k[:, t], state))
    if outputs:
        return torch.stack(outputs, dim=1), state, torch.stack(realized, dim=1)
    return vbar, state, vbar


def _select_bank(
    bank_k, bank_v, bank_scores, candidate_k, candidate_v, candidate_scores,
    temperature=0.25, novelty_weight=0.5,
):
    """Hard top-k bank replacement with a straight-through soft assignment.

    Key slots are shared across all value channels. The hard forward pass keeps
    actual selected associations; the soft surrogate sends gradients to old and
    new keys, values, and selection scores.
    """
    R = bank_k.shape[-2]
    active = bank_scores > (_EMPTY_SCORE / 2)
    cand_norm = F.normalize(candidate_k, dim=-1)
    bank_norm = F.normalize(bank_k, dim=-1)
    similarity = torch.einsum('bthk,bhrk->bthr', cand_norm, bank_norm).abs().clamp(0, 1)
    similarity = similarity.masked_fill(~active[:, None], 0.0)
    novelty = 1.0 - similarity.amax(dim=-1)
    new_scores = torch.tanh(candidate_scores) + novelty_weight * novelty
    all_k = torch.cat((bank_k, candidate_k.permute(0, 2, 1, 3)), dim=2)
    all_v = torch.cat((bank_v.permute(0, 1, 3, 2), candidate_v.permute(0, 2, 3, 1)), dim=-1)
    all_scores = torch.cat((bank_scores, new_scores.permute(0, 2, 1)), dim=-1)
    indices = all_scores.topk(R, dim=-1).indices
    hard = F.one_hot(indices, num_classes=all_scores.shape[-1]).to(all_scores.dtype)
    logits = all_scores.unsqueeze(-2) / max(float(temperature), 1.0e-3) + 12.0 * hard
    soft = torch.softmax(logits, dim=-1)
    weights = hard + soft - soft.detach()
    next_k = torch.einsum('bhrn,bhnk->bhrk', weights, all_k)
    next_v = torch.einsum('bhrn,bhvn->bhrv', weights, all_v)
    next_scores = (weights * all_scores.unsqueeze(-2)).sum(dim=-1)
    count = (next_scores.detach() > (_EMPTY_SCORE / 2)).sum(dim=-1).to(torch.int32)
    return next_k, next_v, next_scores, count


def _initial_bank(q, k, v, protection_rank, protected_keys, protected_values, protected_scores, protected_count):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if protected_keys is None:
        bank_k = torch.zeros((B, H, protection_rank, K), device=q.device, dtype=dtype)
        bank_v = torch.zeros((B, H, protection_rank, V), device=q.device, dtype=dtype)
        bank_scores = torch.full((B, H, protection_rank), _EMPTY_SCORE, device=q.device, dtype=dtype)
    else:
        if protected_values is None:
            raise ValueError('protected_values is required when protected_keys is supplied.')
        bank_k, bank_v = protected_keys.to(dtype), protected_values.to(dtype)
        if bank_k.shape != (B, H, protection_rank, K) or bank_v.shape != (B, H, protection_rank, V):
            raise ValueError('Protected bank shape mismatch.')
        if protected_scores is not None:
            bank_scores = protected_scores.to(device=q.device, dtype=dtype)
        else:
            if protected_count is None:
                protected_count = (bank_k.square().sum(-1) > 0).sum(-1)
            if protected_count.ndim == 3:
                protected_count = protected_count.amax(dim=-1)
            slots = torch.arange(protection_rank, device=q.device).view(1, 1, -1)
            bank_scores = torch.where(
                slots < protected_count.to(device=q.device)[..., None],
                torch.zeros((), device=q.device, dtype=dtype),
                torch.full((), _EMPTY_SCORE, device=q.device, dtype=dtype),
            )
    if bank_scores.shape != (B, H, protection_rank):
        raise ValueError('protected_scores must have shape [B,H,R].')
    return bank_k, bank_v, bank_scores


def naive_protected_chunkwise_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    protected_keys=None, protected_values=None, protected_scores=None,
    protected_count=None, selection_scores=None, protection_rank=4,
    chunk_size=16, checkpoint=True, temperature=0.25, novelty_weight=0.5,
):
    """Pure-PyTorch reference with chunk checkpoints and differentiable bank replacement."""
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError('Expected q/k [B,T,H,K] and v [B,T,H,V].')
    if not isinstance(protection_rank, int) or not 1 <= protection_rank <= 16:
        raise ValueError('protection_rank must be in [1, 16].')
    if chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32.')
    if beta.shape not in (q.shape[:3], v.shape):
        raise ValueError('beta must have shape [B,T,H] or [B,T,H,V].')
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    bf = beta.to(dtype)
    if bf.ndim == 3:
        bf = bf.unsqueeze(-1).expand_as(vf)
    c = torch.as_tensor(curvature, device=q.device, dtype=dtype).reshape(())
    if not torch.isfinite(c).all() or c.detach().item() <= 0:
        raise ValueError('Protected mode requires finite curvature > 0.')
    state = torch.zeros((B, H, K, V), device=q.device, dtype=dtype) if initial_state is None else initial_state.to(dtype)
    bank_k, bank_v, bank_scores = _initial_bank(
        qf, kf, vf, protection_rank, protected_keys, protected_values, protected_scores, protected_count,
    )
    scores = kf.norm(dim=-1) if selection_scores is None else selection_scores.to(dtype)
    if scores.shape != (B, T, H):
        raise ValueError('selection_scores must have shape [B,T,H].')
    out_scale = K ** -0.5 if scale is None else float(scale)
    outputs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        qc, kc, vc, bc, sc = qf[:, start:end], kf[:, start:end], vf[:, start:end], bf[:, start:end], scores[:, start:end]

        def run_chunk(qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c):
            kbar, vbar = _project_chunk_torch(kc, vc, bank_k, bank_v, bank_scores, c)
            out, next_state, realized = _recurrent_chunk_torch(
                qc, kbar, vbar, bc, state, c, out_scale, readout_k=kc,
            )
            next_bank_k, next_bank_v, next_scores, _ = _select_bank(
                bank_k, bank_v, bank_scores, kc, realized, sc,
                temperature=temperature, novelty_weight=novelty_weight,
            )
            return out, next_state, next_bank_k, next_bank_v, next_scores

        args = (qc, kc, vc, bc, sc, state, bank_k, bank_v, bank_scores, c)
        use_checkpoint = checkpoint and torch.is_grad_enabled() and any(
            x.requires_grad for x in args
        )
        result = activation_checkpoint(run_chunk, *args, use_reentrant=False) if use_checkpoint else run_chunk(*args)
        out, state, bank_k, bank_v, bank_scores = result
        outputs.append(out)
    output = torch.cat(outputs, dim=1).to(q.dtype) if outputs else q.new_zeros(B, 0, H, V)
    bank_count = (bank_scores.detach() > (_EMPTY_SCORE / 2)).sum(dim=-1).to(torch.int32)
    return output, state, bank_k, bank_v, bank_scores, bank_count


def protected_recurrent_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    output_final_state=False, protected_keys=None, protected_values=None,
    protected_count=None, protection_rank=4, checkpoint_size=16, check_inputs=False,
    protected_scores=None, selection_scores=None, mode='auto', checkpoint=True,
    temperature=0.25, novelty_weight=0.5,
):
    """Dispatch to the PyTorch reference or the Triton chunkwise projection + WY path."""
    if mode not in ('auto', 'naive', 'triton'):
        raise ValueError("mode must be 'auto', 'naive', or 'triton'.")
    if mode == 'triton' and q.device.type != 'cuda':
        raise ValueError("mode='triton' requires CUDA.")
    if q.device.type == 'cuda' and mode != 'naive':
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
