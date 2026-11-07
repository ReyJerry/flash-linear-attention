# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Differentiable recurrent PHBA reference with frozen routes.'

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from fla.ops.phba.reference import HostLayout, route_masks


@contextlib.contextmanager
def highest_precision():
    """Reference B 与 fp32 routing oracle 显式关闭 TF32。"""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


def prefix_states_recurrent(k_raw, v, g_log, layout: HostLayout):
    """返回 PrefixK, PrefixV [Nc, M, H, D]（与 g_log 同 dtype，逐 token 递推，可导）。"""
    T, H, M = g_log.shape
    D = k_raw.shape[-1]
    A = torch.exp(g_log)
    I = -torch.expm1(g_log)  # noqa: E741
    pk, pv = [None] * layout.Nc, [None] * layout.Nc
    for seq in range(layout.Nseq):
        Sk = g_log.new_zeros((H, M, D))
        Sv = g_log.new_zeros((H, M, D))
        for b in layout.blocks_of_sequence(seq):
            pk[b], pv[b] = Sk.permute(1, 0, 2), Sv.permute(1, 0, 2)
            for t in range(layout.block_ptr[b], layout.block_ptr[b + 1]):
                a_t, i_t = A[t][..., None], I[t][..., None]                     # [H, M, 1]
                Sk = a_t * Sk + i_t * k_raw[t][:, None, :]
                Sv = a_t * Sv + i_t * v[t][:, None, :]
    return torch.stack(pk, 0), torch.stack(pv, 0)


def reference_b(q_raw, q_rope, k_rope, k_raw, v, a_storage, *, tau: float, layout: HostLayout,
                rtok: torch.Tensor, rstate: torch.Tensor, dtype: torch.dtype = torch.float32,
                scale: float | None = None):
    """返回 (O, L, PrefixK, PrefixV)，全部为 dtype；输入保持与调用方的 autograd 连接。"""
    with highest_precision():
        q_raw, q_rope, k_rope, k_raw, v, a = (x.to(dtype) for x in (q_raw, q_rope, k_rope, k_raw, v, a_storage))
        T, H, D = q_rope.shape
        scale = 1.0 / math.sqrt(D) if scale is None else scale
        g_log = F.logsigmoid(a) / tau
        pk, pv = prefix_states_recurrent(k_raw, v, g_log, layout)
        M = pk.shape[1]
        outs, lses = [], []
        for h in range(H):
            local, token, state = route_masks(rtok, rstate, layout, h, M)
            s_tt = (q_rope[:, h] @ k_rope[:, h].T) * scale
            s_tt = s_tt.masked_fill(~(local | token), float("-inf"))
            pk_h = pk[:, :, h].reshape(-1, D)
            pv_h = pv[:, :, h].reshape(-1, D)
            s_st = ((q_raw[:, h] @ pk_h.T) * scale).masked_fill(~state, float("-inf"))
            s = torch.cat([s_tt, s_st], dim=-1)
            lse = torch.logsumexp(s, dim=-1)
            p = torch.exp(s - lse[:, None])
            outs.append(p @ torch.cat([v[:, h], pv_h], dim=0))
            lses.append(lse)
        return torch.stack(outs, 1), torch.stack(lses, 0), pk, pv


__all__ = ["highest_precision", "prefix_states_recurrent", "reference_b"]
