# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'FP32 key centroids for causal same-sequence routing.'

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from fla.ops.phba.errors import fail

if TYPE_CHECKING:
    from fla.ops.phba.routing.layout import RoutingLayout

CENTROID_NUM_WARPS = 4
CENTROID_NUM_STAGES = 2


@triton.jit
def _centroid_kernel(
    k_ptr, block_ptr, out_ptr,
    stride_kt, stride_kh,
    H: tl.constexpr, D: tl.constexpr, BLOCK_TOK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)
    start = tl.load(block_ptr + i_b)
    end = tl.load(block_ptr + i_b + 1)
    n = end - start
    inv = 1.0 / tl.maximum(n, 1).to(tl.float32)
    o_t = tl.arange(0, BLOCK_TOK)
    o_d = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        m_d = (d0 + o_d) < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t0 in range(0, n, BLOCK_TOK):
            m_t = (t0 + o_t) < n
            rows = (start + t0 + o_t).to(tl.int64)
            x = tl.load(k_ptr + rows[:, None] * stride_kt + i_h * stride_kh + (d0 + o_d)[None, :],
                        mask=m_t[:, None] & m_d[None, :], other=0.0)
            acc += tl.sum(x.to(tl.float32), axis=0)
        dst = out_ptr + (tl.cast(i_b, tl.int64) * H + i_h) * D + d0 + o_d
        tl.store(dst, acc * inv, mask=m_d)


def compute_block_centroids(k_rope: torch.Tensor, layout: RoutingLayout, *, block_tok: int = 64,
                            block_d: int = 64) -> torch.Tensor:
    """Krope [T, H, Dh] → Centroid [Nc, H, Dh] fp32（在 no_grad 下调用；输入按 detach 处理）。"""
    if k_rope.dim() != 3 or k_rope.shape[0] != layout.T:
        raise fail("invalid_shape", f"Krope 必须是 [T, H, Dh]，T = {layout.T}，收到 {tuple(k_rope.shape)}")
    if k_rope.stride(-1) != 1:
        raise fail("non_contiguous_input", "Krope 的最后一维必须连续")
    T, H, D = k_rope.shape
    out = torch.empty((layout.Nc, H, D), dtype=torch.float32, device=k_rope.device)
    if layout.Nc == 0:
        return out
    _centroid_kernel[(layout.Nc, H)](
        k_rope.detach(), layout.block_ptr, out, k_rope.stride(0), k_rope.stride(1),
        H=H, D=D, BLOCK_TOK=block_tok, BLOCK_D=block_d,
        num_warps=CENTROID_NUM_WARPS, num_stages=CENTROID_NUM_STAGES,
    )
    return out


def centroid_reference(k_rope: torch.Tensor, block_ptr_host) -> torch.Tensor:
    """fp32 求和后除以有效 token 数（与 kernel 的累加顺序不同，按 fp32 容差比较）。"""
    T, H, D = k_rope.shape
    nc = len(block_ptr_host) - 1
    out = torch.zeros((nc, H, D), dtype=torch.float32, device=k_rope.device)
    for b in range(nc):
        a, e = block_ptr_host[b], block_ptr_host[b + 1]
        if e > a:
            out[b] = k_rope[a:e].float().sum(0) / (e - a)
    return out


__all__ = ["centroid_reference", "compute_block_centroids"]
