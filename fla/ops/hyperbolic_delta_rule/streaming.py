# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.hyperbolic_delta_rule.protected import protected_recurrent_hyperbolic_delta_rule


def protected_streaming(q, k, v, beta, cache=None, checkpoint_size=16, **kwargs):
    """Resume a globally aligned protected chunk, replaying at most one partial chunk."""
    cache = {} if cache is None else cache
    pending = cache.get('pending')
    carry = cache.get('block_start', cache)
    skip = 0
    if pending is not None:
        skip = pending['q'].shape[1]
        q, k, v, beta = (torch.cat((pending[name], value), dim=1)
                         for name, value in zip(('q', 'k', 'v', 'beta'), (q, k, v, beta)))
    outputs = []
    result_cache = cache
    for start in range(0, q.shape[1], checkpoint_size):
        end = min(start + checkpoint_size, q.shape[1])
        block = {name: value[:, start:end].contiguous()
                 for name, value in zip(('q', 'k', 'v', 'beta'), (q, k, v, beta))}
        out, state, keys, values, scores, count = protected_recurrent_hyperbolic_delta_rule(
            **block, initial_state=carry.get('state'), protected_keys=carry.get('protected_keys'),
            protected_values=carry.get('protected_values'), protected_scores=carry.get('protected_scores'),
            protected_count=carry.get('protected_count'), output_final_state=True,
            checkpoint_size=checkpoint_size, **kwargs,
        )
        outputs.append(out)
        result_cache = dict(state=state, protected_keys=keys, protected_values=values,
                            protected_scores=scores, protected_count=count)
        if end - start < checkpoint_size:
            result_cache.update(block_start=carry, pending=block)
        else:
            carry = result_cache
    if not outputs:
        return v[:, :0], result_cache
    return torch.cat(outputs, dim=1)[:, skip:], result_cache
