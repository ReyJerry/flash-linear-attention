# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Launch and arithmetic contract for the explicitly selected BF16 prototype."""

import re

ARITHMETIC = 'selective_bf16_adjoint_queryless'


def stages(chunk_size, backward=True, source_vjp='expanded'):
    if chunk_size not in (16, 32, 64, 128):
        raise AssertionError('Selective BF16 requires C16/C32/C64/C128')
    reverse = backward_stages(chunk_size, source_vjp)
    levels = (chunk_size // 16).bit_length() - 1
    forward = (
        'common_decay_kernel', 'common_input_kernel', 'interval_cache_kernel', 'diagonal_eligibility',
        *(('interval_fwd_cross',) if chunk_size >= 64 else ()),
        'interval_diagonal_tc_fwd', 'interval_fwd_diagonal', 'interval_tail_kernel', 'common_lower_kernel',
        'diagonal_inverse_kernel', *('inverse_offdiagonal_kernel',)*levels,
        *('common_finish_tiles',)*3, 'common_tail_finish', 'group_gram_kernel', 'group_gauge_kernel',
        'group_system_tiles', 'diagonal_inverse_kernel', *('inverse_offdiagonal_kernel',)*levels,
        'group_feedback_tiles', 'state_fwd_kernel', 'output_fwd_kernel',
    )
    return forward + (reverse if backward else ())


def backward_stages(chunk_size, source_vjp='expanded'):
    if source_vjp not in ('expanded', 'coupled', 'coupled_state', 'coupled_fused'):
        raise AssertionError('Unknown selective source VJP')
    if source_vjp != 'expanded':
        if chunk_size != 64:
            raise AssertionError('Coupled source VJP requires C64')
        tail = ('coupled_interval_sources_bwd', 'coupled_interval_gate_bwd', 'coupled_gamma_sources_bwd') if (
            source_vjp == 'coupled_fused') else (
                'common_base_bwd', 'interval_bwd_cross', 'interval_diagonal_tc_bwd', 'interval_bwd_diagonal', 'interval_gate_bwd')
        return (
            'output_bwd_kernel', 'state_bwd_kernel', 'coupled_residual_bwd', *('shared_coefficients_bwd',)*2,
            *('coupled_sources_bwd',)*4, 'parameter_bwd_kernel', 'mass_source_reduce', 'common_tail_bwd',
            'common_lower_bwd', *tail, 'common_input_bwd',
        )
    return (
        'output_bwd_kernel', 'state_bwd_kernel', *('shared_coefficients_bwd',)*4,
        'shared_feedback_gl_bwd', 'shared_feedback_system_bwd', 'parameter_bwd_kernel',
        'mass_source_reduce', 'common_tail_bwd', 'common_ga_bwd', *('common_products_bwd',)*3,
        'common_inverse_left_bwd', 'common_lower_bwd', 'common_base_bwd',
        'interval_bwd_cross' if chunk_size >= 64 else 'interval_bwd_small',
        'interval_diagonal_tc_bwd', 'interval_bwd_diagonal', 'interval_gate_bwd', 'common_input_bwd',
    )


def attest_schedule(launches, backward, chunk_size, source_vjp='expanded'):
    if [row['kernel'] for row in launches] != list(stages(chunk_size, backward, source_vjp)):
        raise AssertionError('Selective BF16 missing/duplicated/reordered computational stage')
    reverse = backward_stages(chunk_size, source_vjp)
    for row in launches:
        phase = 'backward' if row['kernel'] in reverse else 'forward'
        if row['phase'] != phase or row['arguments'].get('C') != chunk_size:
            raise AssertionError('Selective BF16 phase or execution chunk differs')
    curvature = [row['arguments']['NEED_DC'] for row in launches
                 if row['kernel'] in ('group_system_tiles', 'output_fwd_kernel', 'parameter_bwd_kernel')]
    if len(curvature) != (3 if backward else 2) or any(type(value) is not bool or value != curvature[0] for value in curvature):
        raise AssertionError('Selective BF16 curvature-gradient specialization differs across stages')
    widths = [width for width in (16, 32, 64) if width < chunk_size] * 2
    for name, argument, expected in (
        ('diagonal_inverse_kernel', 'B', [16, 16]),
        ('inverse_offdiagonal_kernel', 'N', widths),
        ('inverse_offdiagonal_kernel', 'SINGLE', [chunk_size !=
         128 and not (chunk_size == 64 and curvature[0])] * len(widths)),
        ('inverse_offdiagonal_kernel', 'NEED_DC', [curvature[0]] * len(widths)),
        ('common_finish_tiles', 'MODE', [0, 1, 2]),
        ('shared_coefficients_bwd', 'MODE', ([0, 3] if source_vjp != 'expanded' else [0, 1, 2, 3]) if backward else []),
        ('common_products_bwd', 'MODE', [0, 1, 2] if backward and source_vjp == 'expanded' else []),
        ('coupled_sources_bwd', 'MODE', [0, 1, 2, 3] if backward and source_vjp != 'expanded' else []),
    ):
        actual = [row['arguments'].get(argument) for row in launches if row['kernel'] == name]
        if actual != expected:
            raise AssertionError(f'Selective BF16 specialization differs: {name}/{argument}')
    if source_vjp != 'expanded' and curvature[0]:
        raise AssertionError('Coupled source VJP does not support curvature gradients')
    if backward:
        for name, key, expected in (
            ('state_bwd_kernel', 'COUPLED_STATE', source_vjp in ('coupled_state', 'coupled_fused')),
            ('parameter_bwd_kernel', 'COUPLED', source_vjp != 'expanded'),
            ('common_tail_bwd', 'COUPLED', source_vjp != 'expanded'),
            ('common_tail_bwd', 'SOURCE_OWNER', source_vjp == 'coupled_fused'),
            ('common_lower_bwd', 'COUPLED', source_vjp != 'expanded'),
        ):
            rows = [r for r in launches if r['kernel'] == name]
            if len(rows) != 1 or rows[0]['arguments'].get(key, False) is not expected:
                raise AssertionError(f'Selective source specialization differs: {name}/{key}')
        if source_vjp != 'expanded':
            row = next(r for r in launches if r['kernel'] == 'coupled_residual_bwd')
            if row['arguments'].get('PI_READY', False) is not (source_vjp != 'coupled'):
                raise AssertionError('Selective coupled Pi ownership differs')
    rho = [row['arguments']['RHO_MATERIALIZED'] for row in launches if 'RHO_MATERIALIZED' in row['arguments']]
    if not rho or any(type(value) is not bool or value != rho[0] for value in rho):
        raise AssertionError('Selective BF16 rho representation differs across stages')


def expected_dot_count(name, flags):
    chunk = flags.get('C')
    coupled_names = ('coupled_residual_bwd', 'coupled_sources_bwd', 'coupled_interval_sources_bwd',
                     'coupled_interval_gate_bwd', 'coupled_gamma_sources_bwd')
    if chunk not in (16, 32, 64, 128) or name not in (*stages(chunk), *coupled_names):
        raise AssertionError('Unknown selective BF16 kernel/geometry')
    if name in coupled_names:
        if chunk != 64:
            raise AssertionError('Coupled source kernel requires C64')
        return {'coupled_residual_bwd': 3 if flags.get('PI_READY') is True else 6,
                'coupled_sources_bwd': 3, 'coupled_interval_sources_bwd': 24,
                'coupled_interval_gate_bwd': 0, 'coupled_gamma_sources_bwd': 0}[name]
    if name in ('parameter_bwd_kernel', 'common_lower_bwd') and flags.get('COUPLED') is True:
        if chunk != 64:
            raise AssertionError('Coupled source kernel requires C64')
        return 0
    if name == 'inverse_offdiagonal_kernel':
        if type(flags.get('NEED_DC')) is not bool:
            raise AssertionError('Selective inverse requires a curvature-gradient specialization')
        single = chunk != 128 and not (chunk == 64 and flags['NEED_DC'])
        if flags.get('SINGLE') is not single:
            raise AssertionError('Selective inverse compensation differs')
        return 2 if single else 6
    if name == 'shared_coefficients_bwd':
        if flags.get('MODE') not in (0, 1, 2, 3):
            raise AssertionError('Unknown selective coefficient VJP')
        return 3 if flags['MODE'] == 3 else 2
    if name == 'output_fwd_kernel':
        if type(flags.get('NEED_DC')) is not bool:
            raise AssertionError('Selective readout requires an explicit curvature-gradient specialization')
        return 6 if flags['NEED_DC'] else 2
    # The pointer contractions split reductions of width128 into two width64 dots.
    if name == 'state_fwd_kernel':
        return 6 if chunk == 128 else 4
    if name == 'state_bwd_kernel':
        if flags.get('COUPLED_STATE') is True:
            if chunk != 64:
                raise AssertionError('Coupled state requires C64')
            return 15
        return 18 if chunk == 128 else 12
    return {
        'interval_fwd_cross': 2, 'interval_diagonal_tc_fwd': 2, 'common_finish_tiles': 1,
        'group_gram_kernel': 1, 'output_bwd_kernel': 6,
        'shared_feedback_gl_bwd': 2, 'parameter_bwd_kernel': 2, 'common_ga_bwd': 6,
        'common_products_bwd': 2, 'common_inverse_left_bwd': 2, 'common_lower_bwd': 2,
        'interval_bwd_cross': 8, 'interval_diagonal_tc_bwd': 8,
    }.get(name, 0)


def attest_compiled(launch, inventory):
    name, flags = launch['kernel'], launch['arguments']
    if name in ('state_fwd_kernel', 'state_bwd_kernel'):
        expected = dict(K=128, G=32, KD=128, VD=128, GS=32, CARRY=False)
        if any(flags.get(key) != value for key, value in expected.items()):
            raise AssertionError('Selective BF16 state geometry/carry differs')
    raw = inventory['raw_assembly']
    if re.search(r'\bf64\b|xf64\b', raw['ttir']) or re.search(r'\.f64\b', raw['ptx']):
        raise AssertionError('Selective BF16 forbids FP64 types/instructions')
    if not set(inventory['multiply_add_ir_dtypes']) <= {'f32'}:
        raise AssertionError('Selective BF16 requires FP32 non-dot arithmetic')
    dots, mma = inventory['dot_operations'], inventory['ptx_tensor_core_opcodes']
    if len(dots) != expected_dot_count(name, flags) or any(
        dot['operand_dtypes'] != ['bf16', 'bf16'] or dot['result_dtype'] != 'f32'
        or dot['max_num_imprecise_acc'] not in (None, 0) for dot in dots
    ):
        raise AssertionError('Selective BF16 dot count/operand/accumulation differs')
    if bool(dots) != bool(mma) or any(not re.search(r'\.f32\.bf16\.bf16(?:\.f32)?$', opcode) for opcode in mma):
        raise AssertionError('Selective BF16 requires matching BF16/FP32 Tensor Core instructions')
    if any(not opcode.endswith('.f32') for opcode in inventory['ptx_floating_opcodes']):
        raise AssertionError('Selective BF16 PTX arithmetic must remain FP32')
