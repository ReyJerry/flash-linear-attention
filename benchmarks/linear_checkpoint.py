# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Retain external matrix products while recomputing custom-operator private tape."""
from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


def policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mm.default and len(args) == 2 and all(
            isinstance(value, torch.Tensor) and value.ndim == 2 for value in args):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


context_fn = partial(create_selective_checkpoint_contexts, policy)


def attest_policy(model, requested):
    if requested not in ('recompute', 'save_mm'):
        raise AssertionError('Unknown Linear checkpoint policy')
    layers = [module for module in model.modules() if getattr(module, 'gradient_checkpointing', False)]
    if requested == 'save_mm' and not layers:
        raise AssertionError('mm-save requires active checkpoint layers')
    for layer in layers:
        checkpoint = getattr(layer, '_gradient_checkpointing_func', None)
        actual = getattr(checkpoint, 'keywords', {}).get('context_fn')
        matches = (isinstance(actual, partial) and actual.func is create_selective_checkpoint_contexts
                   and actual.args == (policy,) and not actual.keywords)
        if matches is not (requested == 'save_mm') or requested == 'recompute' and actual is not None:
            raise AssertionError('Actual Linear checkpoint context differs from the requested policy')
    return {'policy': requested, 'checkpoint_layers': len(layers), 'custom_operator_tape_cached': False}
