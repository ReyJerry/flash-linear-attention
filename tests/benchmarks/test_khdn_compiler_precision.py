# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace

import pytest

from benchmarks.hyperbolic_benchmark_utils import _strict_compiler_evidence
from benchmarks.khdn_compiler_precision import compiled_precision_inventory

DOT = ('%o = tt.dot %a, %b, %c, inputPrecision = tf32x3 : '
       'tensor<32x128xf32> * tensor<128x32xf32> -> tensor<32x32xf32> loc(#loc338)')
MMA = 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%r0}, {%r1}, {%r2}, {%r3};'


def compiled(ir=DOT, ptx=MMA):
    return SimpleNamespace(asm={'ttir': ir, 'ptx': ptx})


def test_inventory_keeps_dot_operands_mode_and_result_distinct():
    result = compiled_precision_inventory(compiled())
    assert result['dot_operations'] == [{'input_precision': 'tf32x3', 'operand_dtypes': ['f32', 'f32'],
                                        'result_dtype': 'f32', 'operand_shapes': [(32, 128), (128, 32)],
                                         'result_shape': (32, 32), 'max_num_imprecise_acc': None}]
    assert result['ptx_tensor_core_opcodes'] == {MMA.split()[0]: 1}
    assert result['numerical_policy_qualified'] is False
    with pytest.raises(AssertionError):
        _strict_compiler_evidence(compiled())


def test_mixed_scalar_and_bf16_dot_do_not_inherit_autocast_or_storage_precision():
    ir = ('%s = arith.mulf %a, %b : f64\n%f = arith.addf %a, %b : tensor<16x16xf32>\n'
          '%o = tt.dot %a, %b, %c, inputPrecision = ieee : '
          'tensor<16x32xbf16> * tensor<32x16xbf16> -> tensor<16x16xf32>')
    ptx = 'fma.rn.f64 a,b,c,d; mul.f32 a,b,c;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 r,a,b;'
    result = compiled_precision_inventory(compiled(ir, ptx))
    assert result['multiply_add_ir_dtypes'] == {'f64': 1, 'f32': 1}
    assert result['dot_operations'][0]['operand_dtypes'] == ['bf16', 'bf16']
    assert result['dot_operations'][0]['result_dtype'] == 'f32'
    assert result['ptx_floating_opcodes'] == {'fma.rn.f64': 1, 'mul.f32': 1}
    with pytest.raises(AssertionError):
        _strict_compiler_evidence(compiled(ir, ptx))


def test_comments_locations_symbol_names_and_source_strings_are_not_instructions():
    ir = DOT + '\n// ' + DOT + '\n%v = arith.mulf %a, %b : f32 loc("tt.dot f64")'
    ptx = ('// ' + MMA + '\n.file 1 "' + MMA + '"\n'
           '.visible .entry bogus.mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32 () {}\n'
           '/* mul.f64 x,y,z; */\n@%p1 mul.f32 x,y,z;\n' + MMA)
    result = compiled_precision_inventory(compiled(ir, ptx))
    assert len(result['dot_operations']) == 1
    assert result['ptx_floating_opcodes'] == {'mul.f32': 1}
    assert sum(result['ptx_tensor_core_opcodes'].values()) == 1


@pytest.mark.parametrize('ir', [
    DOT.replace('tf32x3', 'unknown'),
    DOT.replace('tensor<32x128xf32>', 'tensor<32x128xf8>'),
    DOT.replace('tensor<32x32xf32>', 'tensor<32x64xf32>'),
    DOT.replace('tensor<128x32xf32>', 'tensor<64x32xf32>'),
    DOT.replace('tensor<32x128xf32>', 'tensor<1x32x128xf32>'),
    DOT.replace(' : ', ' :\n '),
    DOT.replace(' -> ', ' unexpected<f64> -> '),
    DOT.replace('%a, %b, %c', '%a, %b'),
    DOT.replace('tt.dot', '"tt.dot"'),
])
def test_unknown_or_inconsistent_dot_syntax_is_not_precision_evidence(ir):
    with pytest.raises(AssertionError):
        compiled_precision_inventory(compiled(ir))


@pytest.mark.parametrize('ir,ptx', [('', MMA), (DOT, ''), (DOT, '.file 1 "' + MMA + '"'),
                                    (DOT, 'mul.unknown x,y,z;')])
def test_missing_or_unknown_compiler_instructions_fail_closed(ir, ptx):
    with pytest.raises(AssertionError):
        compiled_precision_inventory(compiled(ir, ptx))


def test_strict_simt_record_remains_strict_without_dot_or_mma():
    kernel = compiled('%x = arith.mulf %a, %b : tensor<16x16xf64>', 'fma.rn.f64 a,b,c,d;')
    result = compiled_precision_inventory(kernel)
    assert result['dot_operations'] == [] and result['ptx_tensor_core_opcodes'] == {}
    assert _strict_compiler_evidence(kernel)['multiply_add_operand_accumulator_dtypes'] == ['float64']


def policy_launch(mixed=True):
    ir = '%x = arith.mulf %a, %b : tensor<16x16xf64>'
    ptx = 'fma.rn.f64 a,b,c,d;'
    if mixed:
        ir += '\n' + DOT + '\n%wide = arith.extf %o : tensor<32x32xf32> to tensor<32x32xf64>'
        ptx += '\n' + MMA
    inventory = compiled_precision_inventory(compiled(ir, ptx))
    return {'kernel': 'state_fwd_kernel', 'arguments': {'MIXED': mixed},
            'kernel_source_sha256': 'a' * 64, 'compiled_precision_inventory': inventory}


@pytest.mark.parametrize('policy', ['strict', 'mixed_readout'])
def test_compiled_policy_requires_actual_family_flags_dtypes_modes_and_hashes(policy):
    from benchmarks.khdn_compiler_precision import attest_compiled_policy
    row = policy_launch(policy == 'mixed_readout')
    attest_compiled_policy(row, policy)
    assert row['compiled_precision_inventory']['numerical_policy_qualified'] is False
    if policy == 'mixed_readout':
        assert row['compiled_precision_inventory']['ir_float_casts']
    with pytest.raises(AssertionError):
        attest_compiled_policy(row, 'strict' if policy == 'mixed_readout' else 'mixed_readout')


@pytest.mark.parametrize('mutation', [
    lambda row: row['arguments'].update(MIXED=False),
    lambda row: row['arguments'].pop('MIXED'),
    lambda row: row.update(kernel='common_bwd_kernel'),
    lambda row: row.pop('kernel_source_sha256'),
    lambda row: row['compiled_precision_inventory'].update(ttir_sha256='b' * 64),
    lambda row: row['compiled_precision_inventory'].update(ptx_sha256='b' * 64),
    lambda row: row['compiled_precision_inventory']['dot_operations'][0].update(input_precision='tf32'),
    lambda row: row['compiled_precision_inventory']['dot_operations'][0].update(operand_dtypes=['bf16', 'bf16']),
    lambda row: row['compiled_precision_inventory']['dot_operations'][0].update(result_dtype='f64'),
    lambda row: row['compiled_precision_inventory'].update(ptx_tensor_core_opcodes={}),
    lambda row: row['compiled_precision_inventory'].update(ir_float_casts=[]),
    lambda row: row['compiled_precision_inventory'].update(raw_assembly={}),
])
def test_compiled_policy_rejects_substitution_or_missing_observations(mutation):
    from benchmarks.khdn_compiler_precision import attest_compiled_policy
    row = policy_launch()
    mutation(row)
    with pytest.raises(AssertionError):
        attest_compiled_policy(row, 'mixed_readout')


@pytest.mark.parametrize('ir,ptx', [
    (DOT.replace('tf32x3', 'tf32'), MMA),
    (DOT.replace('tf32x3', 'ieee'), MMA),
    (DOT.replace('xf32>', 'xbf16>'), MMA),
    (DOT, MMA.replace('.tf32.tf32.', '.bf16.bf16.')),
    (DOT, 'fma.rn.f64 a,b,c,d;'),
    (DOT + '\n%x = arith.mulf %a, %b : f32', MMA),
    ('%x = arith.mulf %a, %b : f64', MMA),
])
def test_reparsed_but_wrong_compiler_arithmetic_still_rejects(ir, ptx):
    from benchmarks.khdn_compiler_precision import attest_compiled_policy
    row = policy_launch()
    row['compiled_precision_inventory'] = compiled_precision_inventory(compiled(
        '%x = arith.mulf %a, %b : f64\n' + ir, 'fma.rn.f64 a,b,c,d;\n' + ptx))
    with pytest.raises(AssertionError):
        attest_compiled_policy(row, 'mixed_readout')


def test_compiled_policy_roundtrips_json_and_rejects_assembly_hash_drift():
    import json

    from benchmarks.khdn_compiler_precision import attest_compiled_policy
    row = json.loads(json.dumps(policy_launch()))
    attest_compiled_policy(row, 'mixed_readout')
    row['compiled_precision_inventory']['assembly_sha256']['ptx'] = 'b' * 64
    with pytest.raises(AssertionError, match='assembly hash'):
        attest_compiled_policy(row, 'mixed_readout')
