# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Algorithm 1 / 3 / 4 的 routing（triton化方案.md 4.2 节、4.9 节、4.11 节；阶段 C）。

    layout.py    BlockPtr / SeqBlockPtr / BlockSeq / CurrentBlock / PrefixValid / QueryWorkList（GPU 构造，不同步）
    centroid.py  Algorithm 3：有效 token 的 block centroid（fp32）
    topk.py      Algorithm 4：same-sequence + strict-history 的 multi-tile Top-K，以及 Rstate + PrefixValid

``build_routes`` 是 Algorithm 1 的 routing 部分（4.2 节 pipeline：block layout → centroid → tiled Top-K →
Rtok → Rstate = [current; Rtok] → PrefixValid）。route 是不可导的离散 metadata，全部在 no_grad 下计算。
Algorithm 5（direct metadata）在 metadata/direct_metadata.py，只在 needs_backward 且 direct 候选时调用。
"""

from __future__ import annotations

import torch

from fla.ops.phba.routing.centroid import compute_block_centroids
from fla.ops.phba.routing.layout import RoutingLayout, build_block_layout
from fla.ops.phba.routing.topk import route_topk


def build_routes(q_rope: torch.Tensor, k_rope: torch.Tensor, length_plan, *, chunk_size: int, topk: int, br: int = 8,
                 bc: int = 32, block_d: int = 64, block_tok: int = 64):
    """返回 (layout, Rtok [H, T, topk - 1], Rstate [H, T, topk])。"""
    layout: RoutingLayout = build_block_layout(length_plan, chunk_size, br)
    with torch.no_grad():
        cen = compute_block_centroids(k_rope, layout, block_tok=block_tok, block_d=block_d)
        rtok, rstate = route_topk(q_rope, cen, layout, int(topk) - 1, bc=bc, block_d=block_d)
    return layout, rtok, rstate


__all__ = ["RoutingLayout", "build_block_layout", "build_routes", "compute_block_centroids", "route_topk"]
