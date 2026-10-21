# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Independent PyTorch references. This module deliberately has no Triton imports.

State layout is [N, H, K, V], storing the canonical gauge S = M / z (z = 1).
This is the transpose of W in the derivation. Each value column is a Klein point.
Readout is inclusive: the current token is written before q_t reads the state.
"""

from __future__ import annotations

import math

import torch


def check_inputs(q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_values=True):
    if not isinstance(curvature, (int, float)) or not math.isfinite(curvature) or curvature < 0:
        raise ValueError('curvature must be a finite, nonnegative Python scalar; learned curvature is not supported.')
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or q.shape[:3] != v.shape[:3]:
        raise ValueError('Expected q/k [B,T,H,K] and v [B,T,H,V] with identical B,T,H.')
    B, T, H, K = q.shape
    V = v.shape[-1]
    if min(B, H, K, V) <= 0:
        raise ValueError('Batch, head count and head dimensions must be positive.')
    if beta.shape not in (q.shape[:3], v.shape):
        raise ValueError('beta must have shape [B,T,H] or [B,T,H,V].')
    tensors = (q, k, v, beta) + (() if initial_state is None else (initial_state,))
    if any(not x.is_floating_point() or x.device != q.device for x in tensors):
        raise ValueError('Inputs must be floating tensors on the same device.')
    N = B
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError('Packed input requires B=1 and cu_seqlens of shape [N+1].')
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError('cu_seqlens must have integer dtype.')
        if cu_seqlens.device != q.device:
            raise ValueError('cu_seqlens must be on the input device.')
        N = cu_seqlens.numel() - 1
        if check_values:
            cu = cu_seqlens.detach().cpu().tolist()
            if cu[0] != 0 or cu[-1] != T or any(a > b for a, b in zip(cu, cu[1:])):
                raise ValueError('cu_seqlens must start at 0, end at T, and be nondecreasing.')
    if initial_state is not None and initial_state.shape != (N, H, K, V):
        raise ValueError(f'initial_state must have shape {(N, H, K, V)}.')
    if scale is not None and (not isinstance(scale, (int, float)) or not math.isfinite(scale)):
        raise ValueError('scale must be a finite Python scalar or None.')
    if check_values:
        if any(not torch.isfinite(x).all().item() for x in tensors):
            raise ValueError('Inputs must be finite.')
        if not ((beta >= 0) & (beta <= 1)).all().item():
            raise ValueError('beta must lie in [0, 1].')
        # No denominator clamp: invalid geometry must not silently become a different rule.
        norm2 = k.double().square().sum(-1, keepdim=True)
        if not (norm2 - curvature * v.double().square() > 0).all().item():
            raise ValueError('Require ||k||^2 - curvature * v^2 > 0 for every token and value channel.')
        if initial_state is not None and curvature > 0:
            if not (curvature * initial_state.double().square().sum(-2) < 1).all().item():
                raise ValueError('Each initial-state column must lie strictly inside the Klein ball.')
    return N, (K ** -0.5 if scale is None else float(scale))


def prepare_key_value(k, raw_v, curvature: float = 1.0, value_margin: float = 0.5):
    """Normalize keys and bound values strictly inside the spacelike constraint.

    This is layer preprocessing, NOT part of the low-level update. A zero key is
    mapped to the first basis vector. Float32 preprocessing avoids half-precision
    rounding at the geometric boundary. Float64 is retained for reference tests.
    """
    if not math.isfinite(curvature) or curvature < 0 or not 0 < value_margin < 1:
        raise ValueError('Require curvature >= 0 and 0 < value_margin < 1.')
    dtype = torch.float64 if k.dtype == torch.float64 or raw_v.dtype == torch.float64 else torch.float32
    k = k.to(dtype)
    norm = k.norm(dim=-1, keepdim=True)
    basis = torch.zeros_like(k)
    basis[..., 0] = 1
    k = torch.where(norm > 1e-6, k / norm.clamp_min(1e-6), basis)
    # max(c, 1) makes the flat-curvature limit continuous and keeps values bounded.
    radius = value_margin / math.sqrt(max(curvature, 1.0))
    v = raw_v.to(dtype).tanh() * radius * k.norm(dim=-1, keepdim=True)
    return k, v


def _prepare(q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_values):
    N, scale = check_inputs(q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_values)
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in (q, k, v, beta)
                                + (() if initial_state is None else (initial_state,))) else torch.float32
    if beta.ndim == 3:
        beta = beta.unsqueeze(-1).expand_as(v)
    qf, kf, vf, bf = (x.to(dtype).reshape(B * T, H, x.shape[-1]) for x in (q, k, v, beta))
    h0 = (q.new_zeros(N, H, K, V, dtype=dtype) if initial_state is None else initial_state.to(dtype))
    bounds = list(range(0, B * T + 1, T)) if T else [0] * (B + 1)
    if cu_seqlens is not None:
        bounds = cu_seqlens.detach().cpu().tolist()
    return qf, kf, vf, bf, h0, bounds, scale


def naive_recurrent_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    curvature: float = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    check_values: bool = True,
):
    """Autograd reference using a projective gauge reset after every token."""
    qf, kf, vf, bf, h0, bounds, scale = _prepare(
        q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_values,
    )
    outputs, finals = [], []
    for n, (bos, eos) in enumerate(zip(bounds, bounds[1:])):
        s = h0[n]
        for t in range(bos, eos):
            kt, vt = kf[t], vf[t]
            a = kt.square().sum(-1, keepdim=True) - curvature * vt.square()
            r = vt - torch.einsum('hk,hkv->hv', kt, s)
            u = bf[t] * r / a
            d = 1 + curvature * vt * u
            s = (s + kt.unsqueeze(-1) * u.unsqueeze(-2)) / d.unsqueeze(-2)
            outputs.append(scale * torch.einsum('hk,hkv->hv', qf[t], s))
        finals.append(s)
    # Preserve zero gradients for empty input instead of returning a disconnected tensor.
    anchor = (qf.sum() + kf.sum() + vf.sum() + bf.sum() + h0.sum()) * 0
    out = torch.stack(outputs) if outputs else vf + anchor
    out = out.reshape(*v.shape).to(q.dtype)
    ht = torch.stack(finals) + anchor
    return out, ht if output_final_state else None


def naive_chunk_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    curvature: float = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    check_values: bool = True,
):
    """Autograd reference for C = KK^T - c vv^T and a per-value triangular solve.

    Normalize at chunk boundaries; return exactly the same canonical state as the
    recurrent reference in exact arithmetic. Chunk length controls conditioning as
    well as speed. This implementation is an oracle, not an optimized GPU kernel.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer.')
    qf, kf, vf, bf, h0, bounds, scale = _prepare(
        q, k, v, beta, initial_state, cu_seqlens, curvature, scale, check_values,
    )
    outputs, finals = [], []
    for n, (bos, eos) in enumerate(zip(bounds, bounds[1:])):
        s = h0[n]
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            qc, kc = (x[start:end].transpose(0, 1) for x in (qf, kf))
            vc, bc = (x[start:end].permute(1, 2, 0) for x in (vf, bf))
            a = kc.square().sum(-1).unsqueeze(1) - curvature * vc.square()
            alpha = bc / a
            gram = (kc @ kc.transpose(-1, -2)).unsqueeze(1)
            gram = gram - curvature * vc.unsqueeze(-1) * vc.unsqueeze(-2)
            eye = torch.eye(end - start, device=q.device, dtype=qf.dtype)
            triangular = eye + (alpha.unsqueeze(-1) * gram).tril(-1)
            rhs = alpha * (vc - (kc @ s).transpose(-1, -2))
            b = torch.linalg.solve_triangular(triangular, rhs.unsqueeze(-1), upper=False, unitriangular=True)[..., 0]
            den = 1 + curvature * (vc * b).cumsum(-1)
            qk = (qc @ kc.transpose(-1, -2)).tril()
            num = (qc @ s).transpose(-1, -2) + torch.einsum('hij,hvj->hvi', qk, b)
            outputs.append((scale * num / den).permute(2, 0, 1))
            s = (s + torch.einsum('hck,hvc->hkv', kc, b)) / den[..., -1].unsqueeze(-2)
        finals.append(s)
    anchor = (qf.sum() + kf.sum() + vf.sum() + bf.sum() + h0.sum()) * 0
    out = torch.cat(outputs) if outputs else vf + anchor
    out = out.reshape(*v.shape).to(q.dtype)
    ht = torch.stack(finals) + anchor
    return out, ht if output_final_state else None
