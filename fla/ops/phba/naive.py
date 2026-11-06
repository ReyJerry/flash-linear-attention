# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.phba.reference import HostLayout, reference_a_chunkwise
from fla.ops.phba.reference_float64 import prefix_states_recurrent, reference_b


def naive_phba(q, k, v, q_rope, k_rope, gate, lengths, token_routes, state_routes, chunk_size=128, tau=8.):
    """Differentiable FP64 PHBA formula for frozen routes and small reference cases."""
    shape = q.shape
    heads, width = shape[-2:]
    q, k, v, q_rope, k_rope = (x.reshape(-1, heads, width) for x in (q, k, v, q_rope, k_rope))
    out, _, _, _ = reference_b(
        q_raw=q,
        q_rope=q_rope,
        k_rope=k_rope,
        k_raw=k,
        v=v,
        a_storage=gate.reshape(-1, heads, gate.shape[-1]),
        tau=tau,
        layout=HostLayout.from_lengths(lengths, chunk_size),
        rtok=token_routes,
        rstate=state_routes,
        dtype=torch.float64,
    )
    return out.reshape(shape)


__all__ = ['HostLayout', 'naive_phba', 'prefix_states_recurrent', 'reference_a_chunkwise']
