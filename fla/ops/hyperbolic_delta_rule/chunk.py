# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch

from .fused_recurrent import _apply_hyperbolic_delta_rule


def chunk_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    curvature: float | torch.Tensor = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    check_inputs: bool = False,
    readout_q: torch.Tensor | None = None,
):
    """Triton chunk forward with the same input/state contract as fused_recurrent.

    Computes a distinct (KK^T-c*vv^T) triangular solve for each value channel.
    Checkpointed backward differentiates the equivalent rational recurrence.
    This is a correctness-first chunk implementation, not a tuned tensor-core
    kernel. In particular, key/query Gram computations are repeated per value.
    """
    return _apply_hyperbolic_delta_rule(
        q, k, v, beta, curvature, scale, initial_state, output_final_state,
        cu_seqlens, chunk_size, check_inputs, 'chunk', readout_q=readout_q,
    )
