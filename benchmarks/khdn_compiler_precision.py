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


def compiled_precision_inventory(compiled):
    """Return parsed instruction evidence; correctness and route qualification are separate gates."""
    assembly = getattr(compiled, 'asm', {})
    ir, ptx = assembly.get('ttir', ''), assembly.get('ptx', '')
    if not ir or not ptx:
        raise AssertionError('KHDN precision inventory requires compiled TTIR and PTX')
    scalar_types = _parse_mlir_contraction_types(ir)
    dots = _dot_types(ir)
    if not scalar_types and not dots:
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
    if precision_policy not in ('strict', 'mixed_readout'):
        raise AssertionError('unknown KHDN precision policy')
    state_family = launch['kernel'] in ('state_fwd_kernel', 'state_bwd_kernel')
    mixed = precision_policy == 'mixed_readout' and state_family
    flags = launch['arguments']
    if state_family and flags.get('MIXED') is not mixed:
        raise AssertionError('KHDN compiled MIXED flag differs from requested policy')
    if not state_family and flags.get('MIXED', False) is not False:
        raise AssertionError('KHDN non-state family unexpectedly selects MIXED')
    inventory = launch.get('compiled_precision_inventory', {})
    for key in ('ttir_sha256', 'ptx_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', inventory.get(key, '')):
            raise AssertionError('KHDN missing/invalid compiled precision hash')
    if not re.fullmatch('[0-9a-f]{64}', launch.get('kernel_source_sha256', '')):
        raise AssertionError('KHDN missing/invalid kernel source hash')
    raw = inventory.get('raw_assembly', {})
    parsed = compiled_precision_inventory(SimpleNamespace(asm=raw))
    for key in ('ttir_sha256', 'ptx_sha256', 'multiply_add_ir_dtypes', 'dot_operations',
                'ptx_floating_opcodes', 'ptx_tensor_core_opcodes', 'ir_float_casts'):
        if json.dumps(inventory.get(key), sort_keys=True) != json.dumps(parsed[key], sort_keys=True):
            raise AssertionError(f'KHDN compiled precision observation/hash mismatch: {key}')
    if any(inventory.get('assembly_sha256', {}).get(key) != parsed['assembly_sha256'][key] for key in ('ttir', 'ptx')):
        raise AssertionError('KHDN assembly hash disagrees with raw compiled evidence')
    scalar = inventory['multiply_add_ir_dtypes']
    if not scalar.get('f64') or set(scalar) != {'f64'}:
        raise AssertionError('KHDN scalar multiply/add arithmetic must remain FP64')
    dots, mma = inventory['dot_operations'], inventory['ptx_tensor_core_opcodes']
    floating = inventory['ptx_floating_opcodes']
    if not any(opcode.endswith('.f64') for opcode in floating):
        raise AssertionError('KHDN missing compiled FP64 arithmetic')
    if mixed:
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
