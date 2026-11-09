# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Untimed Sherman–Morrison reference for the per-token quality solve.

The production chunk graph solves the equivalent block system. This reference
keeps the canonical token recurrence visible and uses a dense solve whenever
the Sherman–Morrison denominator is too close to zero to certify safely.
"""

import torch

from fla.ops.key_gated_hyperbolic_delta_rule.naive import _validate


def _promoted_dtype(*tensors):
    return torch.float64 if any(x.dtype == torch.float64 for x in tensors if isinstance(x, torch.Tensor)) else torch.float32


def sherman_morrison_token_vjp(q, k, v, beta, ell, state, curvature, do, ds, scale=1.0):
    """Untimed manual seven-source VJP for one well-conditioned mathematical group."""
    kap = curvature / v.numel()
    decay = ell.exp()
    x = decay[:, None] * state
    residual = v - x.T @ k
    n, d = k.dot(k), k.dot(k) - kap * v.dot(v)
    y = residual / n + kap * v * v.dot(residual) / (n * d)
    u = beta * y
    z = 1 + kap * beta * v.dot(residual) / d
    updated = (x + torch.outer(k, u)) / z
    adjoint = ds + scale * torch.outer(q, do)
    dq = scale * (updated @ do)
    gx = adjoint / z
    gz = -(adjoint * updated).sum() / z
    gu = k @ gx + kap * gz * v
    dv = kap * gz * u
    dkap = gz * v.dot(u)
    dbeta = gu.dot(y)
    gr = beta * (gu / n + kap * v * v.dot(gu) / (n * d))
    dn = -gr.dot(y)
    dkap = dkap + gr.dot(v) * y.dot(v)
    dv = dv + gr + kap * (y.dot(v) * gr + gr.dot(v) * y)
    dk = gx @ u - x @ gr + 2 * dn * k
    gx = gx - torch.outer(k, gr)
    dell = (gx * x).sum(-1)
    dstate = decay[:, None] * gx
    return dq, dk, dv, dbeta, dell, dstate, dkap / v.numel()


def sherman_morrison_recurrent_key_gated_hyperbolic_delta_rule(
    q,
    k,
    v,
    beta,
    log_decay,
    curvature=1.0,
    initial_state=None,
    output_final_state=False,
    scale=None,
    cu_seqlens=None,
    group_size=32,
    check_inputs=True,
    telemetry=None,
):
    """Evaluate canonical KHDN with a rank-one quality solve.

    For each Value group, `H=nI-kappa vv^T`, `r=v-X^T k`, `s=v^T r`, and
    `d=n-kappa v^T v`. The fast formula is used only when `d` is finite and
    farther than the fixed FP32-relative safety guard from zero; otherwise the
    same dense solve as the independent oracle is used.
    """
    bounds, output_scale = _validate(
        q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, check_inputs,
    )
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = _promoted_dtype(q, k, v, beta, log_decay, initial_state, curvature)
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf, kf, vf, bf, lf = (x.to(dtype).reshape(B * T, *x.shape[2:]) for x in (q, k, v, beta, log_decay))
        c = torch.as_tensor(curvature, dtype=dtype, device=q.device)
        kappa = c / group_size
        state0 = q.new_zeros((len(bounds) - 1, H, K, V), dtype=dtype) if initial_state is None else initial_state.to(dtype)
        d_relative_tolerance = 2.0 ** -10
        fast_count = fallback_count = 0
        eye = torch.eye(group_size, dtype=dtype, device=q.device)
        outputs, finals = [], []
        for sequence, (start, end) in enumerate(zip(bounds, bounds[1:])):
            state = state0[sequence]
            for token in range(start, end):
                decayed = lf[token].exp()[:, :, None] * state
                heads = []
                for head in range(H):
                    kh, vh = kf[token, head], vf[token, head]
                    parts = []
                    for group in range(V // group_size):
                        sl = slice(group * group_size, (group + 1) * group_size)
                        x, vg = decayed[head, :, sl], vh[sl]
                        residual = vg - x.transpose(0, 1) @ kh
                        n, vv = kh.dot(kh), vg.dot(vg)
                        d = n - kappa * vv
                        inner = vg.dot(residual)
                        safe = bool(torch.isfinite(d).item() and d.detach().item() > d_relative_tolerance * n.detach().item())
                        if safe:
                            update = bf[token, head] * (residual / n + kappa * vg * inner / (n * d))
                            z = 1 + kappa * bf[token, head] * inner / d
                            fast_count += 1
                        else:
                            update = bf[token, head] * torch.linalg.solve(n * eye - kappa * torch.outer(vg, vg), residual)
                            z = 1 + kappa * vg.dot(update)
                            fallback_count += 1
                        parts.append((x + torch.outer(kh, update)) / z)
                    heads.append(torch.cat(parts, dim=1))
                state = torch.stack(heads)
                outputs.append(output_scale * (qf[token, :, :, None] * state).sum(1))
            finals.append(state)
        anchor = sum(x.reshape(-1)[:0].sum() for x in (qf, kf, vf, bf, lf, state0, c))
        output = (torch.stack(outputs) if outputs else vf) + anchor
        final = torch.stack(finals) + anchor if output_final_state else None
        if telemetry is not None:
            telemetry.update(sm_fast_solves=fast_count, sm_dense_fallbacks=fallback_count,
                             sm_d_relative_tolerance=d_relative_tolerance, sm_internal_dtype=str(dtype))
        return output.reshape(B, T, H, V).to(q.dtype), final
