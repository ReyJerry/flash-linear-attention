# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Expected computational launch schedule for the C128 no-carry implementation."""

FORWARD = (
    'common_decay_kernel', 'common_input_kernel', 'common_products_kernel', 'interval_cache_kernel',
    'diagonal_eligibility', 'interval_fwd_cross', 'interval_diagonal_tc_fwd', 'interval_fwd_diagonal',
    'interval_tail_kernel', 'common_lower_kernel', 'diagonal_inverse_kernel',
    *('inverse_offdiagonal_kernel',)*3, *('common_finish_tiles',)*3, 'common_tail_finish',
    'group_gram_kernel', 'group_gauge_kernel', 'group_system_tiles', 'diagonal_inverse_kernel',
    *('inverse_offdiagonal_kernel',)*3, 'group_feedback_tiles', 'state_fwd_kernel', 'output_fwd_kernel',
)
BACKWARD = (
    'output_bwd_kernel', 'state_bwd_kernel',
    *('shared_coefficients_bwd',)*4, 'shared_feedback_gl_bwd', 'shared_feedback_system_bwd',
    'parameter_bwd_kernel', 'gauge_collect', 'gauge_product_bwd', 'gauge_finish_bwd', 'common_tail_bwd',
    'common_ga_bwd', *('common_products_bwd',)*3, 'common_inverse_left_bwd', 'common_lower_bwd',
    'common_base_bwd', 'interval_bwd_cross', 'interval_diagonal_tc_bwd', 'interval_bwd_diagonal',
    'interval_gate_bwd', 'common_input_bwd',
)
ARITHMETIC = 'bf16_single_c128_preparation_and_workspace_bf16'


def attest_schedule(launches, backward):
    expected = FORWARD + (BACKWARD if backward else ())
    if [row['kernel'] for row in launches] != list(expected):
        raise AssertionError('C128 missing/duplicated/reordered computational stage')
    for row in launches:
        expected_phase = 'forward' if row['kernel'] in FORWARD else 'backward'
        if row['phase'] != expected_phase or row['arguments'].get('C') != 128:
            raise AssertionError('C128 stage phase or execution chunk differs')
    for name, argument, values in (
        ('diagonal_inverse_kernel', 'B', [16, 16]),
        ('inverse_offdiagonal_kernel', 'N', [16, 32, 64]*2),
        ('inverse_offdiagonal_kernel', 'SINGLE', [True]*6),
        ('common_finish_tiles', 'MODE', [0, 1, 2]),
        ('shared_coefficients_bwd', 'MODE', [0, 1, 2, 3] if backward else []),
        ('common_products_bwd', 'MODE', [0, 1, 2] if backward else []),
    ):
        if [row['arguments'].get(argument) for row in launches if row['kernel'] == name] != values:
            raise AssertionError(f'C128 specialization/order differs: {name}/{argument}')
    rho = [row['arguments']['RHO_MATERIALIZED'] for row in launches if 'RHO_MATERIALIZED' in row['arguments']]
    if not rho or any(type(value) is not bool or value != rho[0] for value in rho):
        raise AssertionError('C128 rho representation differs across stages')
