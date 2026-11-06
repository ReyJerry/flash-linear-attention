# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Discrete causal block routing from rotary queries and key centroids.'

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
