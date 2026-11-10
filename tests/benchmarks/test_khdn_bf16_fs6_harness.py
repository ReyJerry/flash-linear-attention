# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU synthetic policy contracts and real source parsing; GPU NOT RUN."""

import __future__

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_khdn_training_integration import evidence
from test_khdn_training_integration import fake_core as fake_core

from benchmarks import benchmark_hyperbolic_training as benchmark
from benchmarks import hyperbolic_benchmark_utils as utils
from benchmarks import hyperbolic_training_distributed as training
from benchmarks.khdn_compiler_precision import compiled_precision_inventory

FLAGS = ['--methods', 'khdn_no_carry', '--khdn-precision-policy', 'bf16_fs6', '--chunk-size', '16',
         '--loss-backend', 'plain']


def fs6_raw():
    ir = ['%z = arith.constant dense<0.000000e+00> : tensor<16x32xf32>', '%x = arith.mulf %a, %b : f64']
    for i in range(21):
        ir.extend((f'%t{i} = tt.dot %a, %b, %z, inputPrecision = tf32 : '
                   'tensor<16x128xbf16> * tensor<128x32xbf16> -> tensor<16x32xf32>',
                   f'%d{i} = arith.extf %t{i} : tensor<16x32xf32> to tensor<16x32xf64>'))
    return {'ttir': '\n'.join(ir),
            'ptx': 'fma.rn.f64 a,b,c,d;\nmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 r,a,b,c;'}


@pytest.fixture
def source_modules():
    """Keep original decorators/line numbers; the CPU jit wrapper only stores the original function."""
    root = Path(__file__).resolve().parents[2] / 'fla/ops/key_gated_hyperbolic_delta_rule'
    chunk = ModuleType('cpu_source_contract.chunk')
    chunk.__file__ = str(root / 'chunk.py')
    kernels = ModuleType('cpu_source_contract.chunk_kernels')
    kernels.__file__ = str(root / 'chunk_kernels.py')
    functions = [node for node in ast.parse(Path(kernels.__file__).read_text()).body if isinstance(node, ast.FunctionDef)]
    scope = {'triton': SimpleNamespace(jit=lambda fn: SimpleNamespace(fn=fn))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), kernels.__file__, 'exec',
                 flags=__future__.annotations.compiler_flag), scope)
    for node in functions:
        setattr(kernels, node.name, scope[node.name])
    return chunk, kernels


@pytest.fixture
def source_binding(source_modules):
    return utils._khdn_fs6_source_binding(*source_modules)


def fs6_evidence(binding):
    calls = evidence('no_carry')
    row = calls[0]
    row['requested']['precision_policy'] = 'bf16_fs6'
    row['executed_options']['bf16_fs6'] = row['backward_options']['bf16_fs6'] = True
    row.update(output_all_finite=True, gradients_all_finite=True)
    for i, source in enumerate(row['executed_sources']):
        source['shape'] = [1, 33, 1, 128] if i < 3 else []
    row['telemetry'].update(precision_requested='bf16_fs6', precision_eligible=True,
                            precision_route='experimental_bf16_fs6', tensor_cores=True,
                            arithmetic='bf16_fs6_projection', backward_precision_route='experimental_bf16_fs6')
    for launch in row['launches']:
        name = launch['kernel']
        launch.update(phase='backward' if name.endswith('bwd_kernel') else 'forward',
                      kernel_source_sha256=binding['jit_functions'][name],
                      source_binding_sha256=utils._binding_hash(binding), selected_fs_callee=None)
        if name in ('state_fwd_kernel', 'state_bwd_kernel'):
            launch['arguments'].update(K=128, G=32, KD=128, VD=128)
        if name == 'state_fwd_kernel':
            launch['arguments']['BF16_FS6'] = True
            launch['selected_fs_callee'] = 'bf16_fs6'
            launch['compiled_precision_inventory'] = compiled_precision_inventory(SimpleNamespace(asm=fs6_raw()))
            launch['compiler'] = launch['compiled_precision_inventory'].copy()
            launch['compiler'].pop('raw_assembly')
            launch['y_diagnostics']['saved_state_y_fp64'] = True
    return calls


def attest(calls, binding, **options):
    return utils.attest_khdn(calls, 16, 32, 'no_carry', precision_policy='bf16_fs6', source_binding=binding, **options)


def test_fs6_synthetic_trace_separates_phases_and_roundtrips_raw(source_binding):
    result = attest(fs6_evidence(source_binding), source_binding)
    assert result['forward_arithmetic'] == 'bf16_fs6_projection' and result['backward_arithmetic'] == 'device_float64'
    assert result['route'] == 'experimental_bf16_fs6'
    assert result['numerical_policy_qualified'] is False and result['performance_accepted'] is False
    compact = utils._compact_khdn_assembly(result)
    assert len(compact['compiled_assemblies']) == 2
    reparsed = utils.reparse_khdn_trace(json.loads(json.dumps(compact)), expected_source_binding=source_binding)
    assert reparsed['backward_calls'] == 1
    next(iter(compact['compiled_assemblies'].values()))['ptx'] += '\nadd.f32 a,b,c;'
    with pytest.raises(AssertionError, match='mismatch'):
        utils.reparse_khdn_trace(compact, expected_source_binding=source_binding)


@pytest.mark.parametrize('mutation', [
    lambda row: row['executed_options'].pop('bf16_fs6'),
    lambda row: row['executed_options'].update(mixed=True),
    lambda row: row['telemetry'].update(backward_precision='bf16_fs6_projection'),
    lambda row: row['telemetry'].update(precision_route='experimental_mixed_readout'),
    lambda row: row['telemetry'].update(strict_replay_details={'replay': True}),
    lambda row: row['telemetry'].update(precision_eligible=False, qualification_reason='unqualified_geometry'),
    lambda row: row['telemetry'].update(exception_reason='host_substitution'),
    lambda row: row['executed_sources'][0].update(device='cpu'),
    lambda row: row['executed_sources'][0].update(shape=[1, 33, 1, 32]),
    lambda row: row['executed_sources'][3].update(dtype='torch.bfloat16'),
    lambda row: row.update(output_all_finite=False),
    lambda row: row.update(gradients_all_finite=False),
    lambda row: row['launches'][3]['y_diagnostics'].update(saved_state_y_fp64=False),
    lambda row: row['launches'][3].update(selected_fs_callee='mm_state_product'),
    lambda row: row['launches'][3].update(kernel_source_sha256='a' * 64),
    lambda row: row['launches'][3].update(source_binding_sha256='a' * 64),
    lambda row: row['launches'][3].update(phase='backward'),
    lambda row: row['launches'][3]['arguments'].update(BF16_FS6=False),
    lambda row: row['launches'][4]['arguments'].update(BF16_FS6=True),
    lambda row: row['launches'][4]['arguments'].update(MIXED=True),
    lambda row: row['launches'][4]['arguments'].update(KD=32),
    lambda row: row['launches'].reverse(),
    lambda row: row['launches'].pop(1),
])
def test_fs6_rejects_policy_geometry_phase_replay_source(mutation, source_binding):
    calls = fs6_evidence(source_binding)
    mutation(calls[0])
    with pytest.raises(AssertionError):
        attest(calls, source_binding)


@pytest.mark.parametrize('mutation', [
    lambda raw: raw.update(ttir=raw['ttir'].replace('%t20 = tt.dot', '%t20 = tt.bad')),
    lambda raw: raw.update(ttir=raw['ttir'].replace('xbf16>', 'xf32>')),
    lambda raw: raw.update(ttir=raw['ttir'].replace('dense<0.000000e+00>', 'dense<1.000000e+00>')),
    lambda raw: raw.update(ttir=raw['ttir'].replace('%t20 :', '%t19 :')),
    lambda raw: raw.update(ttir=raw['ttir'].replace('tensor<16x32xf64>', 'tensor<16x32xf32>')),
    lambda raw: raw.update(ptx=raw['ptx'].replace('.bf16.bf16.', '.tf32.tf32.')),
    lambda raw: raw.update(ptx=raw['ptx'] + '\nadd.f32 a,b,c;'),
])
def test_fs6_reparses_instruction_types_zero_accumulator_and_widening(mutation, source_binding):
    calls = fs6_evidence(source_binding)
    raw = fs6_raw()
    mutation(raw)
    calls[0]['launches'][3]['compiled_precision_inventory'] = compiled_precision_inventory(SimpleNamespace(asm=raw))
    with pytest.raises(AssertionError):
        attest(calls, source_binding)


def test_fs6_low_precision_reverse_and_strict_substitution_reject(source_binding):
    calls = fs6_evidence(source_binding)
    calls[0]['launches'][4]['compiled_precision_inventory'] = calls[0]['launches'][3]['compiled_precision_inventory']
    with pytest.raises(AssertionError):
        attest(calls, source_binding)
    with pytest.raises(AssertionError):
        utils.attest_khdn(fs6_evidence(source_binding), 16, 32, 'no_carry')


def test_fs6_checkpoint_early_stop_requires_per_layer_complete_launches(source_binding):
    calls = fs6_evidence(source_binding)
    calls[0]['layer'] = 'layers.0.attn'
    replay = copy.deepcopy(calls[0])
    replay.update(backward_calls=0, forward_interrupted='_StopRecomputationError',
                  telemetry={'precision_requested': 'bf16_fs6', 'precision_eligible': True})
    replay['launches'] = replay['launches'][:4]
    calls.append(replay)
    assert attest(calls, source_binding, expected_layers=['layers.0.attn'], expected_checkpoint=True)['checkpoint_recompute_observed']
    for invalid in (calls[:1], calls + [copy.deepcopy(replay)]):
        with pytest.raises(AssertionError, match='checkpoint graph'):
            attest(invalid, source_binding, expected_layers=['layers.0.attn'], expected_checkpoint=True)
    calls[1]['launches'].pop()
    with pytest.raises(AssertionError):
        attest(calls, source_binding, expected_layers=['layers.0.attn'], expected_checkpoint=True)


def test_fs6_real_source_binding_covers_helper_and_saved_primal(source_modules, source_binding):
    assert source_binding['selected_forward_callee'] == 'bf16_fs6'
    _, kernels = source_modules
    assert source_binding['files']['chunk_kernels'] == hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest()
    assert 'bf16_fs6' in source_binding['jit_functions']
    kernels.bf16_fs6 = kernels.mm_state_product
    assert utils._khdn_fs6_source_binding(*source_modules) != source_binding
    kernels.state_bwd_kernel = kernels.state_fwd_kernel
    with pytest.raises(AssertionError, match='saved-primal'):
        utils._khdn_fs6_source_binding(*source_modules)


def test_fs6_cli_tiny_oracle_seven_source_and_serialization_binding():
    args = benchmark.parse_arguments(FLAGS)
    configs = [benchmark.model_configuration(args, name, small=True) for name in ('khdn_no_carry', 'khdn_naive')]
    for config in configs:
        assert (config['head_dim'], config['hidden_size'], config['group_size'], config['chunk_size']) == (128, 256, 32, 16)
        assert config['learnable_curvature'] and config['num_hidden_layers'] == config['num_heads'] == 2
        assert config['use_short_conv'] and not config['fuse_cross_entropy']
    assert configs[0]['precision_policy'] == 'bf16_fs6' and configs[1]['precision_policy'] == 'strict'
    assert json.loads(json.dumps(configs[0]))['precision_policy'] == 'bf16_fs6'
    assert training.khdn_operator_call(lambda *a, **kw: kw, (None,) * 7, args, 'khdn_no_carry')['precision_policy'] == 'bf16_fs6'
    assert all(configs[0][key] == configs[1][key] for key in configs[0].keys() - {'attn_mode', 'precision_policy'})


@pytest.mark.parametrize('extra', [
    ['--khdn-strict'], ['--chunk-size', '32'], ['--head-dim', '32', '--hidden-size', '512'], ['--group-size', '16'],
    ['--methods', 'khdn_y_carry'], ['--methods', 'khdn_naive', '--check-only'], ['--dtype', 'fp32'],
    ['--no-short-conv'], ['--no-layer-checkpoint'], ['--loss-backend', 'native'],
])
def test_fs6_cli_refuses_outside_contract(extra):
    with pytest.raises(SystemExit):
        benchmark.parse_arguments(FLAGS + extra)


def test_fs6_guard_uses_new_helper_and_preserves_old_helper(request):
    core = request.getfixturevalue('fake_core')
    original = core.module._qualification_reason
    new = core.module._bf16_fs6_qualification_reason
    core.module._qualification_reason = lambda *a: 'unqualified_runtime'
    with utils.KHDNExceptionGuard('bf16_fs6') as guard:
        with pytest.raises(RuntimeError, match='unqualified_runtime'):
            core.module._bf16_fs6_qualification_reason(None, None)
        assert guard.events[0]['entry'] == '_bf16_fs6_qualification_reason'
        assert core.module._qualification_reason(None, None) == 'unqualified_runtime'
        with pytest.raises(RuntimeError, match='_strict_device_replay'):
            core.module._strict_device_replay()
    assert core.module._bf16_fs6_qualification_reason is new
    core.module._qualification_reason = original


def test_fs6_projection_diagnostic_preserves_strict_gate():
    values = {'P': [None] * 5 + [torch.ones(1, 2, 2, dtype=torch.float64)], 'SavedS': torch.ones(1, 2, 2, dtype=torch.float64),
              'SavedY': torch.full((1, 2, 2), 2.001, dtype=torch.float64), 'NG': 1, 'MIXED': False, 'BF16_FS6': True}
    result = utils._khdn_y_diagnostics(values)
    assert result['saved_state_y_fp64'] and result['residual_only_not_numerical_gate']
    values['BF16_FS6'] = False
    with pytest.raises(AssertionError):
        utils._khdn_y_diagnostics(values)


@pytest.mark.parametrize('policy', ['strict', 'mixed_readout', 'bf16_fs6'])
def test_real_hf_configuration_roundtrip_without_loading_gpu_modules(monkeypatch, tmp_path, policy):
    root = Path(__file__).resolve().parents[2]
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(__import__('sys').modules, name, module)
        spec.loader.exec_module(module)
        return module
    load('fla.models.configuration_utils', 'fla/models/configuration_utils.py')
    module = load('fs6_real_hf_config_cpu',
                  'fla/models/key_gated_hyperbolic_deltanet/configuration_key_gated_hyperbolic_deltanet.py')
    cls = module.KeyGatedHyperbolicDeltaNetConfig
    config = cls(**{**benchmark.model_configuration(benchmark.parse_arguments(FLAGS), 'khdn_no_carry', small=True),
                    'precision_policy': policy})
    config.save_pretrained(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['precision_policy'] == policy
    assert cls.from_pretrained(tmp_path).precision_policy == policy
    legacy = config.to_dict()
    legacy.pop('precision_policy')
    assert cls.from_dict(legacy).precision_policy == 'strict'


def test_fs6_trace_attest_keeps_full_source_binding_and_raw_roundtrip(monkeypatch, source_modules, source_binding):
    chunk, kernels = source_modules
    original = utils.importlib.import_module
    def module(name, *args, **kwargs):
        if name == utils.KHDN_PREFIX + '.chunk':
            return chunk
        if name == utils.KHDN_PREFIX + '.chunk_kernels':
            return kernels
        return original(name, *args, **kwargs)
    monkeypatch.setattr(utils.importlib, 'import_module', module)
    trace = utils.KHDNTrace(16, 32, 'no_carry', precision_policy='bf16_fs6')
    trace.source_binding, trace.kernel_module, trace.calls = source_binding, kernels, fs6_evidence(source_binding)
    result = trace.attest()
    assert result['source_binding'] == source_binding
    assert utils.reparse_khdn_trace(result, expected_source_binding=source_binding)['backward_calls'] == 1
