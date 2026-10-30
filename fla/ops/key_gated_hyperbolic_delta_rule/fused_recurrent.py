# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Canonical decode/training recurrence using the C=1 stable chunk specialization.

Each device program scans one sequence/head/group. Preparation and explicit VJP
specialize to scalar feedback; no generic autograd runs inside backward.
"""

from fla.ops.key_gated_hyperbolic_delta_rule.chunk import chunk_key_gated_hyperbolic_delta_rule


class _RecurrentTelemetry(dict):
    def __init__(self, target):
        self.target = target

    def __setitem__(self, key, value):
        self.update({key: value})

    def pop(self, key, *args):
        return self.target.pop(key, *args)

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        if values.get('backend') == 'triton_chunk':
            values['backend'] = 'triton_recurrent'
        if values.get('backward') == 'manual_triton_chunk_source_vjp':
            values['backward'] = 'manual_triton_recurrent_source_vjp'
        self.target.update(values)


def fused_recurrent_key_gated_hyperbolic_delta_rule(
    q, k, v, beta, log_decay, curvature=1.0, initial_state=None, output_final_state=False,
    scale=None, cu_seqlens=None, group_size=32, check_inputs=True, checkpoint_size=32, telemetry=None,
):
    """Run first-order recurrent/decode with canonical state across calls.

    checkpoint_size is accepted for Layer compatibility. This strict initial
    implementation saves every token entry state; replay thinning is pending.
    """
    if isinstance(checkpoint_size, bool) or not isinstance(checkpoint_size, int) or checkpoint_size < 1:
        raise ValueError('checkpoint_size must be a positive integer.')
    info = None if telemetry is None else _RecurrentTelemetry(telemetry)
    if telemetry is not None:
        telemetry.update(recurrent_specialization='positive_mass_C1', checkpoint_size=checkpoint_size,
                         checkpoint_policy='save_every_token')
    return chunk_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, log_decay, curvature=curvature, initial_state=initial_state,
        output_final_state=output_final_state, scale=scale, cu_seqlens=cu_seqlens,
        group_size=group_size, check_inputs=check_inputs, chunk_size=1, telemetry=info,
    )
