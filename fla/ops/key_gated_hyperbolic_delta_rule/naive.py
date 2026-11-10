# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Independent dense-solve reference for the canonical grouped KHDN recurrence.

Mathematical contract: https://github.com/ReyJerry/flash-linear-attention/pull/1.
The recurrence uses a dense g-by-g solve per token, head and Value group;
it imports no production kernels, Schur helpers or FLA device utilities.
"""

import math

import torch


def _validate(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, check_values):
    if q.ndim != 4 or k.shape != q.shape or log_decay.shape != q.shape:
        raise ValueError('q, k and log_decay must have identical shape [B,T,H,K].')
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('v must have shape [B,T,H,V] with the same B,T,H as q.')
    B, T, H, K = q.shape
    V = v.shape[-1]
    if min(B, H, K, V) < 1:
        raise ValueError('Batch, heads and Key/Value dimensions must be positive.')
    if beta.shape != q.shape[:3]:
        raise ValueError('beta must have shape [B,T,H], shared across all Value groups.')
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1 or V % group_size:
        raise ValueError('group_size must be a positive integer dividing V.')
    tensors = (q, k, v, beta, log_decay) + (() if initial_state is None else (initial_state,))
    if any(not x.is_floating_point() or x.device != q.device for x in tensors):
        raise ValueError('Inputs must be floating tensors on the same device.')
    if isinstance(curvature, torch.Tensor):
        if curvature.ndim != 0 or not curvature.is_floating_point() or curvature.device != q.device:
            raise ValueError('curvature must be a floating scalar tensor on the input device.')
        c_value = curvature.detach().item()
    else:
        if not isinstance(curvature, (int, float)):
            raise ValueError('curvature must be a Python scalar or a floating scalar tensor.')
        c_value = float(curvature)
    if not math.isfinite(c_value) or c_value < 0:
        raise ValueError('curvature must be finite and nonnegative.')
    if scale is not None and (not isinstance(scale, (int, float)) or not math.isfinite(scale)):
        raise ValueError('scale must be a finite Python scalar or None.')
    if cu_seqlens is None:
        bounds = [i * T for i in range(B + 1)]
    else:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError('Packed input requires B=1 and cu_seqlens of shape [N+1].')
        if cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.device != q.device:
            raise ValueError('cu_seqlens must be int32 or int64 on the input device.')
        bounds = cu_seqlens.detach().cpu().tolist()
        if bounds[0] != 0 or bounds[-1] != T or any(a > b for a, b in zip(bounds, bounds[1:])):
            raise ValueError('cu_seqlens must start at 0, end at T and be nondecreasing.')
    N = len(bounds) - 1
    if initial_state is not None and initial_state.shape != (N, H, K, V):
        raise ValueError(f'initial_state must have shape {(N, H, K, V)}.')
    if check_values:
        if any(not torch.isfinite(x).all().item() for x in tensors):
            raise ValueError('All inputs must be finite.')
        if not ((beta >= 0) & (beta <= 1)).all().item():
            raise ValueError('beta must lie in [0,1].')
        if not (log_decay <= 0).all().item():
            raise ValueError('log_decay must be nonpositive natural logarithms.')
        n = k.double().square().sum(-1)
        if not (n > 0).all().item():
            raise ValueError('Every valid token must have a nonzero Key.')
        values = v.double().reshape(B, T, H, V // group_size, group_size)
        if not (n.unsqueeze(-1) - (c_value / group_size) * values.square().sum(-1) > 0).all().item():
            raise ValueError('Require ||k||^2 - (curvature/group_size) * ||v_group||^2 > 0.')
        if initial_state is not None and c_value > 0:
            state = initial_state.double().reshape(N, H, K, V // group_size, group_size)
            if not ((c_value / group_size) * state.square().sum((-3, -1)) < 1).all().item():
                raise ValueError('Each initial-state group must lie strictly inside its Frobenius ball.')
    return bounds, K ** -0.5 if scale is None else float(scale)


def naive_recurrent_key_gated_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    curvature: float | torch.Tensor = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    group_size: int = 32,
    check_inputs: bool = True,
):
    """Compute inclusive readout and optional canonical state with ordinary autograd.

    q/k/log_decay use [B,T,H,K], v uses [B,T,H,V], beta uses [B,T,H].
    States use [N,H,K,V], where N=B or len(cu_seqlens)-1 for packed B=1.
    Value groups have width group_size dividing V; decay and beta are head shared.
    Noncontiguous tensors and duplicate packed boundaries (empty sequences) are supported.
    Inputs are neither normalized nor clamped; valid zero Keys are errors, including at beta=0.

    Require finite c>=0 constant within a call, beta in [0,1], finite log_decay<=0,
    n=||k||^2>0 and n-(c/g)||v_group||^2>0; for c>0, also (c/g)||S0_group||_F^2<1.
    check_inputs=False skips value/domain checks; callers retain these obligations.
    Shape, scalar and packed-boundary checks remain enabled.

    Arithmetic and final state retain FP64 if any source is FP64, otherwise FP32;
    output follows q.dtype. Default scale is K**-0.5. All seven source gradients,
    including curvature at zero and state across calls, remain connected.
    This is a CPU correctness oracle; ordinary AD here makes no promise about production double backward.

    Returns:
        output, final_state:
            Output of shape [B,T,H,V] and canonical state, or `None` when output_final_state=False.
    """
    bounds, scale = _validate(
        q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, check_inputs,
    )
    B, T, H, K = q.shape
    V = v.shape[-1]
    G = V // group_size
    sources = (q, k, v, beta, log_decay) + (() if initial_state is None else (initial_state,))
    if isinstance(curvature, torch.Tensor):
        sources += (curvature,)
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in sources) else torch.float32
    # explicit source promotion must also govern autocast-eligible contractions
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf, kf, vf, bf, lf = (x.to(dtype).reshape(B * T, *x.shape[2:]) for x in (q, k, v, beta, log_decay))
        c = torch.as_tensor(curvature, dtype=dtype, device=q.device)
        kappa = c / group_size
        h0 = q.new_zeros(len(bounds) - 1, H, K, V, dtype=dtype) if initial_state is None else initial_state.to(dtype)
        eye = torch.eye(group_size, dtype=dtype, device=q.device)
        outputs, finals = [], []
        for n, (bos, eos) in enumerate(zip(bounds, bounds[1:])):
            state = h0[n].reshape(H, K, G, group_size).permute(0, 2, 1, 3)
            for t in range(bos, eos):
                kt = kf[t]
                vt = vf[t].reshape(H, G, group_size)
                decayed = lf[t].exp()[:, None, :, None] * state
                residual = vt - torch.einsum('hk,hgkv->hgv', kt, decayed)
                norm2 = kt.square().sum(-1)
                matrix = norm2[:, None, None, None] * eye - kappa * vt.unsqueeze(-1) * vt.unsqueeze(-2)
                update = bf[t, :, None, None] * torch.linalg.solve(matrix, residual.unsqueeze(-1)).squeeze(-1)
                denominator = 1 + kappa * (vt * update).sum(-1)
                state = (decayed + kt[:, None, :, None] * update.unsqueeze(-2)) / denominator[:, :, None, None]
                outputs.append(scale * torch.einsum('hk,hgkv->hgv', qf[t], state).reshape(H, V))
            finals.append(state.permute(0, 2, 1, 3).reshape(H, K, V))
        # empty reductions connect zero VJPs without summing potentially large source values
        anchor = sum(x.reshape(-1)[:0].sum() for x in (qf, kf, vf, bf, lf, h0, c))
        output = torch.stack(outputs) if outputs else vf
        output = (output + anchor).reshape(B, T, H, V).to(q.dtype)
        final_state = torch.stack(finals) + anchor if output_final_state else None
        return output, final_state
