# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch oracle and Triton dispatch for explicit protected KV associations."""

from __future__ import annotations

import torch


def _project(k, v, keys, values, curvature):
    if keys.shape[0] == 0:
        return k, v
    gram = keys @ keys.T - curvature * values[:, None] * values[None, :]
    rhs = keys @ k - curvature * values * v
    alpha = torch.linalg.solve(gram, rhs)
    return k - keys.T @ alpha, v - values @ alpha


def _protected_reference(q, k, v, beta, curvature, scale, initial_state,
                         protected_keys, protected_values, protected_count, protection_rank):
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected q/k [B,T,H,K] and v [B,T,H,V].")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if not isinstance(protection_rank, int) or not 1 <= protection_rank <= 16:
        raise ValueError("protection_rank must be in [1, 16].")
    c = torch.as_tensor(curvature, device=q.device, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
    if c.numel() != 1 or not torch.isfinite(c).all() or c.detach().item() <= 0:
        raise ValueError("Protected updates require finite curvature > 0.")
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    if beta.shape != v.shape:
        raise ValueError("beta must have shape [B,T,H] or [B,T,H,V].")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, beta, c = q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype), c.to(dtype)
    state = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.to(dtype)
    if protected_keys is None:
        pk = q.new_zeros(B, H, V, protection_rank, K)
        pv = q.new_zeros(B, H, V, protection_rank)
        pc = torch.zeros(B, H, V, dtype=torch.int32, device=q.device)
    else:
        if protected_values is None:
            raise ValueError("protected_values is required with protected_keys.")
        pk, pv = protected_keys.to(dtype), protected_values.to(dtype)
        if protected_count is None:
            pc = (pk.square().sum(-1) > 0).sum(-1).to(torch.int32)
        else:
            pc = protected_count.to(device=q.device, dtype=torch.int32)
    if pk.shape != (B, H, V, protection_rank, K) or pv.shape != (B, H, V, protection_rank) or pc.shape != (B, H, V):
        raise ValueError("Protected bank shape mismatch.")
    if bool(((pc < 0) | (pc > protection_rank)).any().item()):
        raise ValueError("protected_count must be in [0, protection_rank].")
    outputs, states, keys_out, values_out, counts_out = [], [], [], [], []
    out_scale = K ** -0.5 if scale is None else scale
    for b in range(B):
        batch_out, batch_state, batch_pk, batch_pv, batch_pc = [], [], [], [], []
        for h in range(H):
            head_out, head_state, head_pk, head_pv, head_pc = [], [], [], [], []
            for vi in range(V):
                s = state[b, h, :, vi]
                count = int(pc[b, h, vi].detach().item())
                keys = [x.detach() for x in pk[b, h, vi, :count]]
                vals = [x.detach() for x in pv[b, h, vi, :count]]
                y = []
                for t in range(T):
                    kt, vt, bt = k[b, t, h], v[b, t, h, vi], beta[b, t, h, vi]
                    key_bank = torch.stack(keys) if keys else kt.new_zeros((0, K))
                    val_bank = torch.stack(vals) if vals else vt.new_zeros((0,))
                    kb, vb = _project(kt, vt, key_bank, val_bank, c)
                    a = kb.square().sum() - c * vb.square()
                    valid = a > 1e-8
                    u = torch.where(valid, bt * (vb - kb @ s) / torch.where(valid, a, 1.0), torch.zeros_like(vb))
                    den = torch.where(valid, 1 + c * vb * u, torch.ones_like(u))
                    s = torch.where(valid, (s + kb * u) / den, s)
                    y.append(out_scale * (q[b, t, h] @ s))
                    realized = kt @ s
                    if count < protection_rank:
                        nk, nv = _project(kt, realized, key_bank, val_bank, c)
                        if bool((nk.square().sum() - c * nv.square() > 1e-5).detach().item()):
                            keys.append(kt.detach())
                            vals.append(realized.detach())
                            count += 1
                head_out.append(torch.stack(y) if y else v.new_zeros(0))
                head_state.append(s)
                head_pk.append(torch.stack(keys + [q.new_zeros(K)] * (protection_rank - len(keys))))
                head_pv.append(torch.stack(vals + [q.new_zeros(())] * (protection_rank - len(vals))))
                head_pc.append(count)
            batch_out.append(torch.stack(head_out, dim=-1))
            batch_state.append(torch.stack(head_state, dim=-1))
            batch_pk.append(torch.stack(head_pk))
            batch_pv.append(torch.stack(head_pv))
            batch_pc.append(torch.tensor(head_pc, device=q.device, dtype=torch.int32))
        outputs.append(torch.stack(batch_out, dim=1))
        states.append(torch.stack(batch_state))
        keys_out.append(torch.stack(batch_pk))
        values_out.append(torch.stack(batch_pv))
        counts_out.append(torch.stack(batch_pc))
    out = torch.stack(outputs) if outputs else v
    ht = torch.stack(states)
    return out.to(q.dtype), ht, torch.stack(keys_out), torch.stack(values_out), torch.stack(counts_out)


def protected_recurrent_hyperbolic_delta_rule(
    q, k, v, beta, curvature=1.0, scale=None, initial_state=None,
    output_final_state=False, protected_keys=None, protected_values=None,
    protected_count=None, protection_rank=4, checkpoint_size=16, check_inputs=False,
):
    """Update KV memory while preserving a bounded bank of realized associations.

    The bank is stop-gradient state. Dense, unpadded batches are supported.
    CUDA uses a Triton forward/backward; CPU uses the PyTorch oracle.
    """
    if q.device.type == "cuda":
        from .protected_triton import protected_recurrent_hyperbolic_delta_rule_triton
        result = protected_recurrent_hyperbolic_delta_rule_triton(
            q, k, v, beta, curvature, scale, initial_state, protected_keys,
            protected_values, protected_count, protection_rank, checkpoint_size, check_inputs,
        )
    else:
        result = _protected_reference(
            q, k, v, beta, curvature, scale, initial_state,
            protected_keys, protected_values, protected_count, protection_rank,
        )
    return result


__all__ = ["protected_recurrent_hyperbolic_delta_rule"]
