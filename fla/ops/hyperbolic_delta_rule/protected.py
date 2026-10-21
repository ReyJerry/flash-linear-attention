# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch reference for explicit protection of recent KV associations."""

from __future__ import annotations

import torch


def protected_recurrent_hyperbolic_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    curvature: float | torch.Tensor = 1.0, scale: float | None = None,
    initial_state: torch.Tensor | None = None, protected_keys: torch.Tensor | None = None,
    protected_values: torch.Tensor | None = None, protection_rank: int = 32,
):
    """Preserve recent realized KV associations with Lorentz-nullspace projection.

    q/k: [B,T,H,K], v: [B,T,H,V], beta: [B,T,H] or [B,T,H,V].
    State: [B,H,K,V]. Protected keys/values: [B,H,R,K] and [B,H,R,V].
    This slow reference path supports dense unpadded batches only.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected q/k [B,T,H,K] and v [B,T,H,V].")
    if not isinstance(protection_rank, int) or protection_rank < 1:
        raise ValueError("protection_rank must be positive.")
    c = torch.as_tensor(curvature, device=q.device, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
    if c.numel() != 1 or not torch.isfinite(c).all() or c.detach().item() <= 0:
        raise ValueError("Explicit protection requires finite curvature > 0.")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    if beta.shape != v.shape:
        raise ValueError("beta must have shape [B,T,H] or [B,T,H,V].")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, beta, c = q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype), c.to(dtype)
    state = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.to(dtype)
    if protected_keys is None:
        protected_keys, protected_values = q.new_zeros(B, H, 0, K), q.new_zeros(B, H, 0, V)
    if protected_keys.shape[:2] != (B, H) or protected_keys.shape[-1] != K:
        raise ValueError("protected_keys must have shape [B,H,R,K].")
    if protected_values.shape != (*protected_keys.shape[:3], V):
        raise ValueError("protected_values must have shape [B,H,R,V].")
    outputs, finals, saved_keys, saved_values = [], [], [], []
    out_scale = K ** -0.5 if scale is None else scale
    for b in range(B):
        batch_states, batch_keys, batch_values, head_outputs = [], [], [], []
        for h in range(H):
            s = state[b, h]
            old_k, old_v = list(protected_keys[b, h].unbind(0)), list(protected_values[b, h].unbind(0))
            this_head_outputs = []
            for t in range(T):
                kt, vt, bt = k[b, t, h], v[b, t, h], beta[b, t, h]
                a = kt.square().sum() - c * vt.square()
                if not torch.isfinite(a).all() or (a.detach() <= 0).any():
                    raise ValueError("Require ||k||^2 - c*v^2 > 0.")
                cols = []
                for j in range(V):
                    xi = torch.cat((s.new_ones(1), s[:, j]))
                    n = torch.cat((c * vt[j:j + 1], kt))
                    jn = torch.cat((-n[:1] / c, n[1:]))
                    if old_k:
                        normals = torch.stack([torch.cat((c * ov[j:j + 1], ok)) for ok, ov in zip(old_k, old_v)], dim=1)
                        jnormals = torch.cat((-normals[:1] / c, normals[1:]), dim=0)
                        nbar = n - normals @ (torch.linalg.pinv(normals.T @ jnormals, rtol=1e-7) @ (normals.T @ jn))
                    else:
                        nbar = n
                    jnbar = torch.cat((-nbar[:1] / c, nbar[1:]))
                    denom = nbar @ jnbar
                    if denom.detach().item() <= 1e-10:
                        cols.append(s[:, j])
                        continue
                    residual = nbar @ torch.cat((-xi[:1] / c, xi[1:]))
                    updated = xi - bt[j] * residual / denom * nbar
                    lorentz_norm = updated[1:].square().sum() - updated[0].square() / c
                    updated = updated / torch.sqrt((-c * lorentz_norm).clamp_min(1e-12))
                    if updated[0].detach().item() <= 0:
                        raise FloatingPointError("Projection left the positive Klein chart.")
                    cols.append(updated[1:] / updated[0])
                s = torch.stack(cols, dim=-1)
                this_head_outputs.append(out_scale * torch.einsum("k,kv->v", q[b, t, h], s))
                old_k.append(kt)
                old_v.append(torch.einsum("k,kv->v", kt, s))
                old_k, old_v = old_k[-protection_rank:], old_v[-protection_rank:]
            batch_states.append(s)
            head_outputs.append(torch.stack(this_head_outputs) if this_head_outputs else v.new_zeros(0, V))
            batch_keys.append(torch.stack(old_k) if old_k else q.new_zeros(0, K))
            batch_values.append(torch.stack(old_v) if old_v else q.new_zeros(0, V))
        finals.append(torch.stack(batch_states))
        outputs.append(torch.stack(head_outputs, dim=1))
        saved_keys.append(batch_keys)
        saved_values.append(batch_values)
    out = torch.stack(outputs) if outputs else v
    ht = torch.stack(finals)
    max_rank = max((x.shape[0] for row in saved_keys for x in row), default=0)
    pk, pv = q.new_zeros(B, H, max_rank, K), q.new_zeros(B, H, max_rank, V)
    for b in range(B):
        for h in range(H):
            r = saved_keys[b][h].shape[0]
            if r:
                pk[b, h, :r], pv[b, h, :r] = saved_keys[b][h], saved_values[b][h]
    return out.to(q.dtype), ht, pk, pv
