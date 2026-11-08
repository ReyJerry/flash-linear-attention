# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Research checkpoint context: retain external Linear mm outputs, never custom-op tape."""
from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


def linear_policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mm.default and len(args) == 2 and all(isinstance(x, torch.Tensor) and x.ndim == 2 for x in args):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def apply(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
        use_reentrant=False, context_fn=partial(create_selective_checkpoint_contexts, linear_policy)))
    return dict(policy='save_external_aten_mm_outputs', allow_cache_entry_mutation=False, custom_operator_tape_cached=False)
