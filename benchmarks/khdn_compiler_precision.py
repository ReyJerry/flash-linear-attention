# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Observe compiled KHDN precision without authorizing a numerical policy."""

import hashlib
import json
import re
from collections import Counter
from types import SimpleNamespace

from benchmarks.hyperbolic_benchmark_utils import (
    _compiler_without_comments,
    _mlir_without_location,
    _parse_mlir_contraction_types,
)


def _dot_types(ir):
    value = r'%[-\w.$]+(?:#\d+)?'
    encoding = r'#[\w.]+(?:<\{[^{}<>\n]*\}>)?'
    tensor = rf'tensor<(?:\d+x)+(?:f\d+|bf16)(?:,\s*{encoding})?>'
    pattern = re.compile(
        rf'{value}\s*=\s*tt\.dot\s+{value}\s*,\s*{value}\s*,\s*{value}'
        rf',\s*inputPrecision\s*=\s*(ieee|tf32|tf32x3)'
        rf'(?:,\s*maxNumImpreciseAcc\s*=\s*(\d+))?\s*:\s*'
        rf'({tensor})\s*\*\s*({tensor})\s*->\s*({tensor})',
    )
    rows = []
    for line in _compiler_without_comments(ir).splitlines():
        if not re.search(r'\btt\.dot\b', re.sub(r'"(?:\\.|[^"\\])*"', '""', line)):
            continue
        match = pattern.fullmatch(_mlir_without_location(line))
        if match is None:
            raise AssertionError('KHDN unrecognized dot operation/type syntax in compiled IR')
        descriptors = [re.match(r'tensor<((?:\d+x)+)(f\d+|bf16)', match[i]) for i in (3, 4, 5)]
        shapes = [tuple(int(x) for x in descriptor[1].rstrip('x').split('x')) for descriptor in descriptors]
        if any(len(shape) != 2 for shape in shapes) or (
                shapes[0][1] != shapes[1][0] or shapes[2] != (shapes[0][0], shapes[1][1])):
            raise AssertionError('KHDN unsupported or inconsistent compiled dot dimensions')
        dtypes = [descriptor[2] for descriptor in descriptors]
        if any(dtype not in ('f16', 'bf16', 'f32', 'f64') for dtype in dtypes):
            raise AssertionError('KHDN unrecognized compiled dot dtype')
        rows.append({'input_precision': match[1], 'operand_dtypes': dtypes[:2], 'result_dtype': dtypes[2],
                     'operand_shapes': shapes[:2], 'result_shape': shapes[2],
                     'max_num_imprecise_acc': None if match[2] is None else int(match[2])})
    return rows


def compiled_precision_inventory(compiled, *, allow_data_movement=False):
    """Return parsed instruction evidence; correctness and route qualification are separate gates."""
    assembly = getattr(compiled, 'asm', {})
    ir, ptx = assembly.get('ttir', ''), assembly.get('ptx', '')
    if not ir or not ptx:
        raise AssertionError('KHDN precision inventory requires compiled TTIR and PTX')
    scalar_types = _parse_mlir_contraction_types(ir)
    dots = _dot_types(ir)
    if not scalar_types and not dots and not allow_data_movement:
        raise AssertionError('KHDN precision inventory found no supported multiply/add/dot IR operations')
    if any(dtype not in ('f16', 'bf16', 'f32', 'f64') for dtype in scalar_types):
        raise AssertionError('KHDN unrecognized compiled multiply/add dtype')
    instructions = re.sub(r'"(?:\\.|[^"\\])*"', '""', _compiler_without_comments(ptx))
    observed = re.findall(r'(?:^|[;\n{}])\s*(?:@!?%[\w.$]+\s+)?([a-z][\w]*(?:\.[\w]+)+)(?=\s)', instructions)
    opcodes = [opcode for opcode in observed if opcode.split('.')[0] in
               ('add', 'sub', 'mul', 'fma', 'mad', 'div', 'sqrt', 'rcp')]
    if any(not re.fullmatch(r'(?:f\d+(?:x\d+)?|bf16(?:x\d+)?|[sbu]\d+)', opcode.split('.')[-1])
           for opcode in opcodes):
        raise AssertionError('KHDN unrecognized arithmetic opcode type in compiled PTX')
    floating = [opcode for opcode in opcodes if re.search(r'\.(?:f\d+(?:x\d+)?|bf16(?:x\d+)?)\b', opcode)]
    mma = [opcode for opcode in observed if re.fullmatch(
        r'(?:mma\.sync|wmma\.mma|wgmma\.mma_async|tcgen05\.mma)(?:\.[\w]+)+', opcode)]
    if not floating and not mma:
        if not (allow_data_movement and re.search(r'\btt.load\b', _compiler_without_comments(ir))
                and re.search(r'\btt.store\b', _compiler_without_comments(ir))
                and any(op.startswith('ld.global.') for op in observed)
                and any(op.startswith('st.global.') for op in observed)):
            raise AssertionError('KHDN precision inventory found no supported PTX arithmetic instructions')
    casts = [line.strip() for line in _compiler_without_comments(ir).splitlines()
             if re.search(r'\barith\.(?:truncf|extf)\b', line)]
    return {'raw_assembly': {'ttir': ir, 'ptx': ptx}, 'ir_float_casts': casts,
            'assembly_sha256': {name: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
                                for name, value in assembly.items() if isinstance(value, (str, bytes))},
            'ttir_sha256': hashlib.sha256(ir.encode()).hexdigest(),
            'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
            'multiply_add_ir_dtypes': dict(Counter(scalar_types)), 'dot_operations': dots,
            'ptx_floating_opcodes': dict(Counter(floating)), 'ptx_tensor_core_opcodes': dict(Counter(mma)),
            'interpretation': 'observed TTIR multiply/add and dot type clauses plus PTX arithmetic opcode text; '
                              'dot result type is not a proof of every internal accumulation step, '
                              'and source sites are not dynamic execution counts',
            'numerical_policy_qualified': False}


def attest_compiled_policy(launch, precision_policy):
    """Bind observed arithmetic to the requested family policy, without qualifying its numerics."""
    if precision_policy not in ('strict', 'mixed_readout', 'bf16_fs6', 'bf16', 'bf16_selective'):
        raise AssertionError('unknown KHDN precision policy')
    state_family = launch['kernel'] in ('state_fwd_kernel', 'state_bwd_kernel')
    mixed = precision_policy == 'mixed_readout' and state_family
    flags = launch['arguments']
    has_mixed_selector = state_family and not (precision_policy in (
        'bf16', 'bf16_selective') and launch['kernel'] == 'state_bwd_kernel')
    if has_mixed_selector and flags.get('MIXED') is not mixed:
        raise AssertionError('KHDN compiled MIXED flag differs from requested policy')
    if not has_mixed_selector and flags.get('MIXED', False) is not False:
        raise AssertionError('KHDN non-state family unexpectedly selects MIXED')
    fs6_forward = precision_policy == 'bf16_fs6' and launch['kernel'] == 'state_fwd_kernel'
    if flags.get('BF16_FS6', False) is not fs6_forward or (fs6_forward and 'BF16_FS6' not in flags):
        raise AssertionError('KHDN compiled BF16_FS6 flag differs from requested family policy')
    if precision_policy == 'bf16_fs6' and state_family:
        expected = dict(C=16, K=128, G=32, KD=128, VD=128, GS=32, CARRY=False, MIXED=False)
        if any(flags.get(key) != value for key, value in expected.items()):
            raise AssertionError('KHDN compiled bf16_fs6 geometry/flags differ')
    inventory = launch.get('compiled_precision_inventory', {})
    for key in ('ttir_sha256', 'ptx_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', inventory.get(key, '')):
            raise AssertionError('KHDN missing/invalid compiled precision hash')
    if not re.fullmatch('[0-9a-f]{64}', launch.get('kernel_source_sha256', '')):
        raise AssertionError('KHDN missing/invalid kernel source hash')
    raw = inventory.get('raw_assembly', {})
    parsed = compiled_precision_inventory(SimpleNamespace(asm=raw), allow_data_movement=(
        precision_policy in ('bf16', 'bf16_selective') and launch['kernel'] == 'gauge_collect' and flags.get('C') == 128))
    for key in ('ttir_sha256', 'ptx_sha256', 'multiply_add_ir_dtypes', 'dot_operations',
                'ptx_floating_opcodes', 'ptx_tensor_core_opcodes', 'ir_float_casts'):
        if json.dumps(inventory.get(key), sort_keys=True) != json.dumps(parsed[key], sort_keys=True):
            raise AssertionError(f'KHDN compiled precision observation/hash mismatch: {key}')
    if any(inventory.get('assembly_sha256', {}).get(key) != parsed['assembly_sha256'][key] for key in ('ttir', 'ptx')):
        raise AssertionError('KHDN assembly hash disagrees with raw compiled evidence')
    if precision_policy == 'bf16_selective':
        from benchmarks.khdn_selective_trace import attest_compiled
        attest_compiled(launch, inventory)
        return
    if precision_policy == 'bf16':
        _attest_bf16_arithmetic(launch, inventory)
        return
    scalar = inventory['multiply_add_ir_dtypes']
    if not scalar.get('f64') or set(scalar) != {'f64'}:
        raise AssertionError('KHDN scalar multiply/add arithmetic must remain FP64')
    dots, mma = inventory['dot_operations'], inventory['ptx_tensor_core_opcodes']
    floating = inventory['ptx_floating_opcodes']
    if not any(opcode.endswith('.f64') for opcode in floating):
        raise AssertionError('KHDN missing compiled FP64 arithmetic')
    if fs6_forward:
        _attest_fs6_arithmetic(inventory)
    elif mixed:
        if not dots or not mma or any(
            dot['input_precision'] != 'tf32x3' or dot['operand_dtypes'] != ['f32', 'f32']
            or dot['result_dtype'] != 'f32' or dot['max_num_imprecise_acc'] not in (None, 0)
            for dot in dots
        ):
            raise AssertionError('KHDN mixed state family requires actual FP32 TF32x3 dots and Tensor Core instructions')
        if any(not re.search(r'\.f32\.tf32\.tf32(?:\.f32)?$', opcode) for opcode in mma):
            raise AssertionError('KHDN mixed PTX Tensor Core operand/result types differ')
        if any(not opcode.endswith(('.f32', '.f64')) for opcode in floating):
            raise AssertionError('KHDN mixed PTX arithmetic dtype differs')
    elif dots or mma or any(not opcode.endswith('.f64') for opcode in floating):
        raise AssertionError('KHDN strict family requires FP64 arithmetic without Tensor Cores/dots')


def _attest_c128_bf16_matrix_arithmetic(name, inventory):
    """Admit source-bound matrix elementwise operations, never BF16 scalar reductions or controls."""
    shapes = {
        'common_tail_finish': (128, 32),
        'group_system_tiles': (128, 32),
        'common_base_bwd': (16, 128),
        'interval_bwd_diagonal': (32, 32),
    }
    observed = 0
    for source_line in _compiler_without_comments(inventory['raw_assembly']['ttir']).splitlines():
        probe = re.sub(r'"(?:\\.|[^"\\])*"', '""', source_line)
        if not re.search(r'\b(?:arith\.(?:mulf|addf|subf|divf|negf|maxnumf|minnumf)|math\.\w+)\b', probe):
            continue
        line = _mlir_without_location(source_line)
        operation = re.search(r'\b(arith\.(?:mulf|addf|subf|divf|negf|maxnumf|minnumf)|math\.\w+)\b', line)
        if not re.search(r'(?:\bbf16\b|xbf16\b)', line.rsplit(':', 1)[-1]):
            continue
        matrix = re.search(r':\s*tensor<(\d+)x(\d+)xbf16>\s*$', line)
        if (operation[1] not in ('arith.mulf', 'arith.addf', 'math.fma') or matrix is None
                or name not in shapes or tuple(map(int, matrix.groups())) != shapes[name]):
            raise AssertionError('KHDN C128 BF16 arithmetic is restricted to declared matrix elementwise sites')
        observed += 1
    if observed != inventory['multiply_add_ir_dtypes'].get('bf16', 0):
        raise AssertionError('KHDN C128 BF16 matrix arithmetic inventory differs')
    return observed > 0


def _attest_bf16_arithmetic(launch, inventory):
    """Check C16 product counts or the source-bound C128 BF16/FP32 specialization."""
    counts = dict(common_base_fwd_kernel=0, common_interval_fwd_kernel=0, common_system_fwd_kernel=9,
                  group_fwd_kernel=3, group_af_fwd_kernel=0, state_fwd_kernel=9, output_fwd_kernel=6,
                  output_bwd_kernel=8, state_bwd_kernel=12, parameter_bwd_kernel=12,
                  gauge_bwd_kernel=0, common_matrix_bwd_kernel=24, common_interval_bwd_kernel=0)
    name, flags = launch['kernel'], launch['arguments']
    c128 = flags.get('C') == 128
    if c128:
        from benchmarks.khdn_c128_trace import BACKWARD, FORWARD
        if name not in (*FORWARD, *BACKWARD):
            raise AssertionError('Unknown C128 computational kernel')
    elif name not in counts or flags.get('C') != 16:
        raise AssertionError('KHDN bf16 compiler gate requires C16 or C128 no-carry')
    if name in ('state_fwd_kernel', 'state_bwd_kernel'):
        expected = dict(K=128, G=32, KD=128, VD=128, GS=32, CARRY=False)
        if flags.get('MIXED', False) is not False or flags.get('BF16_FS6', False) is not False:
            raise AssertionError('KHDN bf16 cannot select a mixed or historical FS6 callee')
        if any(flags.get(key) != value for key, value in expected.items()):
            raise AssertionError('KHDN bf16 compiled geometry or carry policy differs')
    raw = inventory['raw_assembly']
    if re.search(r'\bf64\b|xf64\b', raw['ttir']) or re.search(r'\.f64\b', raw['ptx']):
        raise AssertionError('KHDN bf16 compiled FP64 type or instruction is forbidden')
    scalar_types = set(inventory['multiply_add_ir_dtypes'])
    matrix_bf16 = _attest_c128_bf16_matrix_arithmetic(name, inventory) if c128 else False
    if (not scalar_types <= ({'f32', 'bf16'} if matrix_bf16 else {'f32'})
            or (not c128 and scalar_types != {'f32'})
            or (c128 and name == 'gauge_collect' and scalar_types)):
        raise AssertionError('KHDN bf16 non-dot arithmetic precision differs')
    dots, mma = inventory['dot_operations'], inventory['ptx_tensor_core_opcodes']
    if (not c128 and len(dots) != counts[name]) or any(
        dot['operand_dtypes'] != ['bf16', 'bf16'] or dot['result_dtype'] != 'f32'
        or dot['max_num_imprecise_acc'] not in (None, 0) for dot in dots
    ):
        raise AssertionError('KHDN bf16 per-GEMM compiled dot count or operand/accumulator type differs')
    if bool(dots) != bool(mma) or any(not re.search(r'\.f32\.bf16\.bf16(?:\.f32)?$', opcode) for opcode in mma):
        raise AssertionError('KHDN bf16 requires matching BF16/FP32 Tensor Core instructions')
    if any(not (opcode.endswith('.f32') or matrix_bf16 and re.fullmatch(
            r'(?:add|mul|fma)(?:\.[\w]+)*\.bf16(?:x2)?', opcode)) for opcode in inventory['ptx_floating_opcodes']):
        raise AssertionError('KHDN bf16 PTX scalar arithmetic must be FP32')


def _attest_fs6_arithmetic(inventory):
    dots, mma = inventory['dot_operations'], inventory['ptx_tensor_core_opcodes']
    if len(dots) != 21 or any(
            dot['operand_dtypes'] != ['bf16', 'bf16'] or dot['result_dtype'] != 'f32'
            or [list(shape) for shape in dot['operand_shapes']] != [[16, 128], [128, 32]]
            or list(dot['result_shape']) != [16, 32] or dot['max_num_imprecise_acc'] not in (None, 0)
            for dot in dots):
        raise AssertionError('KHDN bf16_fs6 requires exactly 21 BF16[16,128] x BF16[128,32] -> FP32 dots')
    if not mma or any(opcode != 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32' for opcode in mma):
        raise AssertionError('KHDN bf16_fs6 requires native BF16/FP32 MMA PTX')
    if any(not opcode.endswith('.f64') for opcode in inventory['ptx_floating_opcodes']):
        raise AssertionError('KHDN bf16_fs6 reconstruction/readout arithmetic must remain FP64')
    lines = [_mlir_without_location(line) for line in _compiler_without_comments(
        inventory['raw_assembly']['ttir']).splitlines()
        if re.match(r'\s*%[-\w.$]+(?:#\d+)?\s*=\s*(?:arith\.(?:constant|extf)|tt\.dot)\b', line)]
    value = r'%[-\w.$]+(?:#\d+)?'
    zero_pattern = (rf'({value})\s*=\s*arith\.constant\s+dense<0(?:\.0+(?:e[+-]?0+)?)?>'
                    r'\s*:\s*tensor<16x32xf32>')
    zero = {match[1] for line in lines if (match := re.fullmatch(zero_pattern, line))}
    dot_pattern = rf'({value})\s*=\s*tt\.dot\s+{value}\s*,\s*{value}\s*,\s*({value}),.*'
    terms = [match.groups() for line in lines if (match := re.fullmatch(dot_pattern, line))]
    if len(terms) != 21 or any(acc not in zero for _, acc in terms):
        raise AssertionError('KHDN bf16_fs6 each digit dot requires a zero FP32 initial accumulator')
    for term, _ in terms:
        pattern = (rf'{value}\s*=\s*arith\.extf\s+{re.escape(term)}\s*:\s*tensor<16x32xf32>'
                   r'\s+to\s+tensor<16x32xf64>')
        if not any(re.fullmatch(pattern, line) for line in lines):
            raise AssertionError('KHDN bf16_fs6 each digit result must widen to FP64 before reconstruction')
