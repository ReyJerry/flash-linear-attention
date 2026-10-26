# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU synthetic evidence and fake entry tests only; no CUDA/kernel/numerical/performance validation."""

import copy
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_hyperbolic_training as benchmark
from benchmarks import hyperbolic_benchmark_utils as utils


def evidence(carry='y_carry', edges=2):
    kernels = [*utils.KHDN_STAGES, *(['carry_fwd_kernel'] if carry == 'y_carry' else [])]
    launches = []
    for name in kernels:
        launch = {'kernel': name, 'arguments': {'C': 16, 'GS': 32, 'CARRY': carry == 'y_carry'},
                  'compiler': {'ptx_sha256': 'synthetic-cpu-fixture', 'fp64_arithmetic_instructions': 1,
                               'tensor_core_instructions': 0, 'contraction_ir_sha256': 'synthetic-cpu-fixture',
                               'multiply_add_operand_accumulator_dtypes': ['float64']}}
        if name == 'carry_fwd_kernel':
            launch.update(observed_edges=edges, future_carry_tail_zero=True)
        if name == 'group_fwd_kernel':
            launch['prepared_diagnostics'] = dict(all_prepared_buffers_finite=True, prepared_denominator_min=.9,
                                                  positive_mass_min=1., positive_normalizer_min=.5)
        if name == 'state_fwd_kernel':
            launch['y_diagnostics'] = {'all_saved_state_y_finite': True}
        launches.append(launch)
    options = dict(c=16, g=32, carry=carry, bounds=[0, 33], final=False, scale=1.)
    return [{'requested': dict(chunk_size=16, group_size=32, carry_mode=carry),
             'executed_options': options, 'backward_options': options.copy(),
             'executed_sources': [{'device': 'cuda:0', 'dtype': 'torch.float32'} for _ in range(7)],
             'telemetry': dict(backend='triton_chunk', arithmetic='device_float64', tensor_cores=False,
                               fallback_subblocks=0, chunk_lengths=[16, 16, 1],
                               carry_edges=edges if carry == 'y_carry' else 0,
                               direct_FS=3 - edges if carry == 'y_carry' else 3,
                               backward='manual_triton_chunk_source_vjp', backward_precision='device_float64'),
             'consumed': dict(all_finite=True, curvature=.6, log_decay_min=-.2, log_decay_max=-.01,
                              feedback_delta_max=.02, key_norm2_min=1., key_norm2_max=1.,
                              initial_state_radius2_max=.01, beta_min=.1, beta_max=.9),
             'sequence_bounds': [0, 33], 'expected_chunk_lengths': [16, 16, 1], 'expected_edges': edges,
             'heads': 1, 'backward_calls': 1, 'launches': launches}]


@pytest.mark.parametrize('carry,edges', [('no_carry', 2), ('y_carry', 2), ('y_carry', 0)])
def test_synthetic_attestation_accepts_both_modes_and_zero_expected_edges(carry, edges):
    result = utils.attest_khdn(evidence(carry, edges), 16, 32, carry)
    assert result['route'] == 'strict_device_fp64_simt' and result['tensor_core_claim'] is False


@pytest.mark.parametrize('mutation', [
    lambda row: row.update(backward_calls=0),
    lambda row: row.update(executed_options=None),
    lambda row: row['executed_options'].update(c=32),
    lambda row: row['executed_options'].update(g=16),
    lambda row: row['executed_options'].update(carry='no_carry'),
    lambda row: row['executed_options'].update(bounds=[0, 32]),
    lambda row: row['requested'].update(carry_mode='no_carry'),
    lambda row: row['executed_sources'][0].update(device='cpu'),
    lambda row: row['telemetry'].pop('fallback_subblocks'),
    lambda row: row['telemetry'].pop('backend'),
    lambda row: row['telemetry'].pop('backward'),
    lambda row: row['telemetry'].pop('arithmetic'),
    lambda row: row['telemetry'].update(backend='triton_recurrent'),
    lambda row: row['telemetry'].update(backward='host_reviewed_manual_chunk_source_vjp'),
    lambda row: row['telemetry'].update(exception_reason='feedback_margin'),
    lambda row: row['telemetry'].update(fallback_subblocks=1),
    lambda row: row['telemetry'].update(backward_fallback_subblocks=1),
    lambda row: row['telemetry'].update(arithmetic='device_float32'),
    lambda row: row['telemetry'].update(tensor_cores=True),
    lambda row: row['telemetry'].update(chunk_lengths=[16, 16, 16]),
    lambda row: row['telemetry'].update(carry_edges=1),
    lambda row: row['telemetry'].update(direct_FS=0),
    lambda row: row['launches'].pop(2),
    lambda row: row['launches'].pop(4),
    lambda row: row['launches'][3]['arguments'].pop('CARRY'),
    lambda row: row['launches'][4]['arguments'].update(CARRY=False),
    lambda row: row['launches'][0]['arguments'].update(C=64),
    lambda row: row['launches'][0]['compiler'].update(tensor_core_instructions=1),
    lambda row: row['launches'][0]['compiler'].pop('ptx_sha256'),
    lambda row: row['launches'][0]['compiler'].update(fp64_arithmetic_instructions=0),
    lambda row: row['launches'][-1].update(future_carry_tail_zero=False),
    lambda row: row['launches'][-1].update(observed_edges=0),
    lambda row: row['launches'][1]['prepared_diagnostics'].update(prepared_denominator_min=0),
    lambda row: row['launches'][1]['prepared_diagnostics'].update(positive_normalizer_min=0),
    lambda row: row['launches'][3].pop('y_diagnostics'),
    lambda row: row['consumed'].update(curvature=0),
    lambda row: row['consumed'].update(all_finite=False),
    lambda row: row['consumed'].update(log_decay_max=0),
    lambda row: row['consumed'].update(feedback_delta_max=.3),
])
def test_synthetic_attestation_rejects_missing_or_substituted_evidence(mutation):
    calls = evidence()
    mutation(calls[0])
    with pytest.raises(AssertionError):
        utils.attest_khdn(calls, 16, 32, 'y_carry')


def test_no_carry_rejects_carry_launch_and_all_model_layers_need_backward():
    calls = evidence('no_carry')
    calls[0]['launches'].append(evidence()[0]['launches'][-1])
    with pytest.raises(AssertionError, match='no_carry'):
        utils.attest_khdn(calls, 16, 32, 'no_carry')
    with pytest.raises(AssertionError, match='model depth'):
        utils.attest_khdn(evidence(), 16, 32, 'y_carry', 24)


def test_checkpoint_interruption_keeps_observed_launches_without_fabricating_telemetry():
    calls = evidence()
    replay = copy.deepcopy(calls[0])
    replay.update(backward_calls=0, forward_interrupted='_StopRecomputationError', telemetry={})
    replay['launches'] = [row for row in replay['launches'] if not row['kernel'].endswith('bwd_kernel')]
    calls.append(replay)
    result = utils.attest_khdn(calls, 16, 32, 'y_carry')
    assert result['checkpoint_recompute_observed'] and calls[1]['telemetry'] == {}
    calls[1]['launches'].pop(2)
    with pytest.raises(AssertionError, match='kernel launch'):
        utils.attest_khdn(calls, 16, 32, 'y_carry')


@pytest.mark.parametrize('ptx', ['', 'add.f32 %f0, %f1, %f2;', 'fma.rn.f64 x,y,z,w; mma.sync.aligned.m16n8k16;'])
def test_compiled_evidence_rejects_missing_fp64_or_tensor_core_instructions(ptx):
    with pytest.raises(AssertionError):
        utils._strict_compiler_evidence(SimpleNamespace(asm={'ptx': ptx}))


def test_compiled_evidence_retains_hash_and_counts_real_instruction_text_only():
    ptx = '// mma.sync comment\nfma.rn.f64 %fd0,%fd1,%fd2,%fd3;'
    record = utils._strict_compiler_evidence(SimpleNamespace(
        asm={'ptx': ptx, 'ttir': 'arith.mulf %0, %1 : tensor<16x16xf64>'}))
    assert len(record['ptx_sha256']) == 64 and record['fp64_arithmetic_instructions'] == 1
    assert record['tensor_core_instructions'] == 0


@pytest.fixture
def fake_core(monkeypatch):
    module = ModuleType(utils.KHDN_PREFIX + '.chunk')
    kernels = ModuleType(utils.KHDN_PREFIX + '.chunk_kernels')
    package = ModuleType(utils.KHDN_PREFIX)
    alias = ModuleType('fla.layers.khdn_cpu_alias_fixture')

    class FakeKernel:
        def __init__(self, name):
            namespace = {}
            exec(f'def {name}(Q, C, GS, CARRY): pass', namespace)
            self.fn = namespace[name]

        def __getitem__(self, grid):
            return lambda *args, **kwargs: SimpleNamespace(asm={'ptx': 'fma.rn.f64 a,b,c,d;',
                                                                'ttir': 'arith.mulf %0, %1 : tensor<16x16xf64>'})

    for name in (*utils.KHDN_STAGES, 'carry_fwd_kernel'):
        setattr(kernels, name, FakeKernel(name))
        setattr(module, name, getattr(kernels, name))

    class FakeChunk:
        @staticmethod
        def forward(ctx, q, k, v, beta, ell, state, curvature, options, telemetry):
            ctx.options, ctx.telemetry = options, telemetry
            for name in utils.KHDN_STAGES[:4]:
                getattr(module, name)[(1,)](q, options['c'], options['g'], False)
            telemetry.update(backend='cpu_fake_fixture', chunk_lengths=[q.shape[1]])
            return q, state

        @staticmethod
        def backward(ctx, *args):
            for name in utils.KHDN_STAGES[4:]:
                getattr(module, name)[(1,)](torch.ones(1), ctx.options['c'], ctx.options['g'], False)
            ctx.telemetry['backward'] = 'cpu_fake_backward_fixture'
            return (None,) * 9

    module._Chunk = FakeChunk

    def entry(q, k, v, beta, log_decay, curvature=1., initial_state=None, output_final_state=False, scale=1.,
              cu_seqlens=None, group_size=32, check_inputs=False, chunk_size=16, carry_mode='no_carry', telemetry=None):
        ctx = SimpleNamespace()
        package.last_ctx = ctx
        options = dict(c=chunk_size, g=group_size, carry=carry_mode, bounds=[0, q.shape[1]])
        return module._Chunk.forward(ctx, q, k, v, beta, log_decay, initial_state, curvature, options, telemetry)

    def host_forward(*args, **kwargs):
        return 'cpu-fake-host'

    def host_backward(*args, **kwargs):
        return 'cpu-fake-host-backward'

    def recurrent_entry(*args, **kwargs):
        return 'cpu-fake-recurrent'

    def oracle_entry(*args, **kwargs):
        return 'cpu-fake-oracle'

    module.chunk_key_gated_hyperbolic_delta_rule = entry
    package.chunk_key_gated_hyperbolic_delta_rule = entry
    alias.imported_entry = entry
    module._host_call, module._host_backward = host_forward, host_backward
    alias.imported_host = host_forward
    recurrent = ModuleType(utils.KHDN_PREFIX + '.fused_recurrent')
    recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule = recurrent_entry
    naive = ModuleType(utils.KHDN_PREFIX + '.naive')
    naive.naive_recurrent_key_gated_hyperbolic_delta_rule = oracle_entry
    for item in (module, kernels, package, alias, recurrent, naive):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    monkeypatch.setattr(utils, '_khdn_contraction_sites', lambda *args: [])
    monkeypatch.setattr(utils, '_khdn_buffer_diagnostics', lambda *args: {'cpu_fixture': True})
    monkeypatch.setattr(utils, '_khdn_y_diagnostics', lambda *args: {'cpu_fixture': True})
    return SimpleNamespace(module=module, package=package, kernels=kernels, alias=alias, recurrent=recurrent, naive=naive)


def test_entry_aliases_backward_worker_and_replay_retain_one_live_dict_and_restore(fake_core):
    core = fake_core
    original = core.alias.imported_entry
    inputs = dict(q=torch.ones(1, 3, 1, 32), k=torch.ones(1, 3, 1, 32), v=torch.ones(1, 3, 1, 32) * .01,
                  beta=torch.ones(1, 3, 1) * .5, log_decay=torch.ones(1, 3, 1, 32) * -.1,
                  initial_state=torch.zeros(1, 1, 32, 32), curvature=torch.tensor(.6))
    supplied = {}
    errors = []

    def worker(ctx):
        try:
            core.module._Chunk.backward(ctx, None, None)
        except BaseException as error:
            errors.append(error)

    with utils.KHDNTrace(16, 32, 'no_carry') as trace:
        core.alias.imported_entry(**inputs, telemetry=supplied)
        first_ctx = core.package.last_ctx
        assert trace.calls[0]['telemetry'] is supplied and 'backward' not in supplied
        core.package.chunk_key_gated_hyperbolic_delta_rule(**inputs)
        thread = threading.Thread(target=worker, args=(first_ctx,))
        thread.start()
        thread.join()
        assert not errors
        assert supplied['backward'] == 'cpu_fake_backward_fixture'
        assert trace.calls[0]['backward_thread'] != trace.calls[0]['forward_thread']
        assert trace.calls[0]['backward_calls'] == 1 and len(trace.calls[0]['launches']) == 6
        assert len(trace.calls) == 2 and trace.calls[1]['backward_calls'] == 0
    assert core.alias.imported_entry is original and core.package.chunk_key_gated_hyperbolic_delta_rule is original
    assert not isinstance(core.module.state_fwd_kernel, utils._ObservedKHDNKernel)
    json.dumps(trace.calls)


@pytest.mark.parametrize('target', ['host_forward', 'host_backward', 'recurrent', 'oracle'])
def test_exception_sentinel_blocks_only_unexpected_entries_and_restores_aliases(fake_core, target):
    core = fake_core
    original = core.alias.imported_host
    functions = {'host_forward': lambda: core.alias.imported_host(), 'host_backward': lambda: core.module._host_backward(),
                 'recurrent': lambda: core.recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule(),
                 'oracle': lambda: core.naive.naive_recurrent_key_gated_hyperbolic_delta_rule()}
    normal = core.package.chunk_key_gated_hyperbolic_delta_rule
    with utils.KHDNExceptionGuard() as guard:
        assert core.package.chunk_key_gated_hyperbolic_delta_rule is normal
        with pytest.raises(RuntimeError, match='measurement rejected'):
            functions[target]()
        assert len(guard.evidence()['events']) == 1 and guard.evidence()['normal_path_wrappers'] == 0
    assert core.alias.imported_host is original and core.alias.imported_host() == 'cpu-fake-host'


def test_strict_opt_in_c64_and_bounded_operator_sweep_configuration():
    args = benchmark.parse_arguments(['--methods', 'khdn_no_carry', 'khdn_y_carry', '--khdn-strict', '--chunk-size', '64'])
    no, yes = (benchmark.model_configuration(args, name) for name in args.methods)
    assert no['carry_mode'] == 'no_carry' and yes['carry_mode'] == 'y_carry'
    yes['carry_mode'] = no['carry_mode']
    assert no == yes
    args = benchmark.parse_arguments(['--methods', 'khdn_no_carry', '--khdn-strict', '--operator-sweep',
                                     '--dtype', 'fp32', '--modes', 'bwd'])
    assert args.operator_lengths == [17, 33, 65]
    for options in (['--chunk-size', '64'], ['--operator-sweep'],
                    ['--methods', 'khdn_no_carry', '--khdn-strict', '--inner-checkpoint']):
        with pytest.raises(SystemExit):
            benchmark.parse_arguments(options)


def test_per_layer_backward_rejects_global_count_collision_and_accepts_forward_only_recompute():
    calls = evidence() + evidence()
    calls[0]['layer'], calls[1]['layer'] = 'layers.0.attn', 'layers.0.attn'
    with pytest.raises(AssertionError, match='per trained layer'):
        utils.attest_khdn(calls, 16, 32, 'y_carry', 2, ['layers.0.attn', 'layers.1.attn'])
    calls[1]['layer'] = 'layers.1.attn'
    replay = copy.deepcopy(calls[1])
    replay['backward_calls'] = 0
    replay['checkpoint_frames'] = ['recompute_fn']
    replay['launches'] = [row for row in replay['launches'] if not row['kernel'].endswith('bwd_kernel')]
    calls.append(replay)
    result = utils.attest_khdn(calls, 16, 32, 'y_carry', 2, ['layers.0.attn', 'layers.1.attn'])
    assert result['backward_layer_coverage'] == {'layers.0.attn': 1, 'layers.1.attn': 1}
    assert result['checkpoint_recompute_observed']
    replay['checkpoint_frames'] = []
    with pytest.raises(AssertionError, match='forward-only'):
        utils.attest_khdn(calls, 16, 32, 'y_carry', 2, ['layers.0.attn', 'layers.1.attn'])


@pytest.mark.parametrize('ir', ['', 'arith.mulf %0, %1 : tensor<16x16xf32>',
                                'arith.mulf %0, %1 : f64\narith.addf %2, %3 : f32'])
def test_compiler_precision_rejects_missing_or_mixed_internal_ir(ir):
    with pytest.raises(AssertionError, match='compiled IR'):
        utils._strict_compiler_evidence(SimpleNamespace(asm={'ptx': 'fma.rn.f64 a,b,c,d;', 'ttir': ir}))


def test_dense_checkpoint_harness_against_direct_loaded_independent_oracle_all_seven_gradients():
    import importlib.util
    from pathlib import Path

    from benchmarks import hyperbolic_training_distributed as training

    path = Path(__file__).resolve().parents[2] / 'fla/ops/key_gated_hyperbolic_delta_rule/naive.py'
    spec = importlib.util.spec_from_file_location('khdn_cpu_dense_oracle_checkpoint_fixture', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    generator = torch.Generator().manual_seed(73)
    q = torch.randn(1, 5, 1, 4, generator=generator, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(q.shape, generator=generator, dtype=torch.float64), dim=-1)
    v = torch.randn(1, 5, 1, 4, generator=generator, dtype=torch.float64) * .01
    beta = torch.rand(1, 5, 1, generator=generator, dtype=torch.float64) * .8 + .1
    ell = -torch.rand(q.shape, generator=generator, dtype=torch.float64) * .1 - .01
    state = torch.randn(1, 1, 4, 4, generator=generator, dtype=torch.float64) * .001
    args = SimpleNamespace(group_size=4, oracle_chunk_size=2)
    result = training.verify_checkpointed_oracle(oracle.naive_recurrent_key_gated_hyperbolic_delta_rule,
                                                 (q, k, v, beta, ell, state, torch.tensor(.6, dtype=torch.float64)),
                                                 args, 'khdn_no_carry')
    assert result['sequence_length'] == 5 and len(result['errors']) == 9
    assert max(result['errors'].values()) < 1e-12


def test_long_operator_shapes_are_explicit_and_pilot_defaults_stay_bounded():
    args = benchmark.parse_arguments(['--methods', 'khdn_no_carry', '--khdn-strict', '--operator-sweep', '--dtype', 'fp32',
                                     '--modes', 'fwd_bwd', '--operator-lengths', '2048', '8192', '16384'])
    assert args.operator_lengths == [2048, 8192, 16384] and args.oracle_chunk_size == 16
    with pytest.raises(SystemExit):
        benchmark.parse_arguments(['--methods', 'khdn_no_carry', '--khdn-strict', '--operator-sweep', '--dtype', 'fp32',
                                   '--modes', 'fwd_bwd', '--operator-lengths', '16385'])


def test_model_binding_labels_actual_attention_entries_and_removes_hooks(fake_core):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f_proj = torch.nn.Identity()
            self.A_log = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs):
            return fake_core.package.chunk_key_gated_hyperbolic_delta_rule(**inputs)[0]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Attention(), Attention()])

    model = Model()
    q = torch.ones(1, 3, 1, 32)
    inputs = dict(q=q, k=q, v=q * .01, beta=q[..., 0] * .5, log_decay=q * -.1,
                  curvature=torch.tensor(.6), initial_state=torch.zeros(1, 1, 32, 32))
    trace = utils.KHDNTrace(16, 32, 'no_carry', expected_backward_calls=2).bind_model(model)
    with trace:
        model.layers[0](inputs)
        model.layers[1](inputs)
    assert [row['layer'] for row in trace.calls] == ['layers.0', 'layers.1']
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.layers)


def test_trace_failure_retains_unpromoted_raw_observations(fake_core):
    with pytest.raises(RuntimeError) as raised, utils.KHDNTrace(16, 32, 'no_carry') as trace:
        trace.calls.append({'cpu_fake_partial': True})
        raise RuntimeError('cpu fixture failure')
    assert raised.value.khdn_evidence == [{'cpu_fake_partial': True}]


@pytest.mark.parametrize('key,value', [('key_norm2_max', 2.**81), ('initial_state_radius2_max', .6),
                                       ('beta_min', -.1), ('beta_max', 1.1)])
def test_synthetic_consumed_domain_rejects_invalid_bounds(key, value):
    calls = evidence()
    calls[0]['consumed'][key] = value
    with pytest.raises(AssertionError, match='consumed'):
        utils.attest_khdn(calls, 16, 32, 'y_carry')


@pytest.mark.parametrize('suffix', [' loc("f64")', ' // f64', ' /* f64 */', ' loc(fused["f64", "other"])'])
def test_ir_type_parser_ignores_location_and_comment_dtype_text(suffix):
    compiled = SimpleNamespace(asm={'ptx': 'fma.rn.f64 a,b,c,d;',
                                    'ttir': '%0 = arith.mulf %a, %b : tensor<16x16xf32>' + suffix})
    with pytest.raises(AssertionError, match='compiled IR'):
        utils._strict_compiler_evidence(compiled)


@pytest.mark.parametrize('operation', [
    '%0 = arith.mulf %a, %b : f64',
    '%0 = arith.addf %a, %b : tensor<16x16xf64, #blocked1> loc("f32")',
    '%0 = math.fma %a, %b, %c : vector<4xf64> // f32',
    '%0 = arith.mulf %a, %b : tensor<16xf64, #ttg.slice<{dim = 0, parent = #blocked2}>>',
])
def test_ir_type_parser_accepts_supported_actual_scalar_tensor_vector_types(operation):
    result = utils._strict_compiler_evidence(SimpleNamespace(asm={'ptx': 'fma.rn.f64 a,b,c,d;', 'ttir': operation}))
    assert result['multiply_add_ir_operations'] == 1
    assert result['multiply_add_operand_accumulator_dtypes'] == ['float64']


@pytest.mark.parametrize('operation', [
    '%0 = arith.mulf %a, %b :\n tensor<16x16xf64>',
    '%0 = arith.mulf %a, %b : !unknown<f64>',
    '%0 = arith.mulf %a, %b : tensor<16x16xf32> extra<f64>',
    '%0 = "arith.mulf"(%a, %b) : (f64, f64) -> f64',
    '%0 = math.fma %a, %b : f64',
    '%0 = arith.mulf %a, %b fastmath<contract> : f64',
    '%0 = arith.mulf %a, %b : f/*comment*/64',
    '%0 = arith.mulf %a, %b : f64 loc("a") %1 = arith.mulf %a, %b : f32 loc("f64")',
])
def test_ir_type_parser_fails_closed_on_unhandled_syntax(operation):
    with pytest.raises(AssertionError, match='compiled IR'):
        utils._strict_compiler_evidence(SimpleNamespace(asm={'ptx': 'fma.rn.f64 a,b,c,d;', 'ttir': operation}))


@pytest.mark.parametrize('mixed_opcode', ['mul.f32', 'mul.f32.f64', 'mul.unknown'])
def test_ptx_independently_rejects_mixed_arithmetic_even_with_valid_fp64_ir(mixed_opcode):
    ir = '%0 = arith.mulf %a, %b : f64'
    with pytest.raises(AssertionError, match='PTX'):
        utils._strict_compiler_evidence(SimpleNamespace(asm={'ptx': mixed_opcode + ' a,b,c; fma.rn.f64 a,b,c,d;', 'ttir': ir}))
    result = utils._strict_compiler_evidence(SimpleNamespace(asm={
        'ptx': '.file 1 "mul.f32 source"\n/* mul.f32 a,b,c; */\nfma.rn.f64 a,b,c,d; // mul.f32', 'ttir': ir}))
    assert result['ptx_inspected_floating_opcodes'] == {'fma.rn.f64': 1}


def _batch_binding_case(layout):
    from benchmarks import hyperbolic_training_distributed as training

    inputs = {'input_ids': torch.ones(2, 5, dtype=torch.int64)}
    if layout in ('packed', 'packed_with_mask'):
        inputs = {'input_ids': torch.ones(1, 10, dtype=torch.int64), 'cu_seqlens': torch.tensor([0, 0, 3, 7, 10])}
        if layout == 'packed_with_mask':
            inputs['attention_mask'] = torch.zeros(1, 10, dtype=torch.int64)
    elif layout == 'padding':
        inputs = {'input_ids': torch.ones(3, 5, dtype=torch.int64),
                  'attention_mask': torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]])}
    return inputs, training.khdn_batch_layout(inputs)


def _batch_bound_calls(domain):
    row = evidence('no_carry', edges=0)[0]
    bounds, shape = domain['sequence_bounds'], domain['operator_batch_time']
    lengths = [min(16, end - pos) for start, end in zip(bounds, bounds[1:]) for pos in range(start, end, 16)]
    row.update(layer='layer0', sequence_bounds=bounds, expected_chunk_lengths=lengths)
    row['executed_options']['bounds'] = bounds
    row['backward_options'] = copy.deepcopy(row['executed_options'])
    row['telemetry'].update(chunk_lengths=lengths, direct_FS=len(lengths))
    row['sources'] = {key: {'shape': [*shape, 1] if key == 'beta' else [*shape, 1, 32]}
                      for key in ('q', 'k', 'v', 'beta', 'log_decay')}
    for key, source in zip(('q', 'k', 'v', 'beta', 'log_decay'), row['executed_sources']):
        source['shape'] = row['sources'][key]['shape']
    return [row]


@pytest.mark.parametrize('layout', ['dense', 'packed', 'padding', 'packed_with_mask'])
@pytest.mark.parametrize('recompute', [False, True])
def test_route_binds_original_and_recompute_to_actual_batch(monkeypatch, layout, recompute):
    from benchmarks import hyperbolic_training_distributed as training

    inputs, domain = _batch_binding_case(layout)
    calls = _batch_bound_calls(domain)
    if recompute:
        replay = copy.deepcopy(calls[0])
        replay.update(backward_calls=0, checkpoint_frames=['recompute_fn'])
        replay['launches'] = [row for row in replay['launches'] if not row['kernel'].endswith('bwd_kernel')]
        calls.append(replay)

    class Trace:
        def __init__(self, *args):
            self.calls, self.layer_names = calls, ['layer0']

        def bind_model(self, model):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def attest(self):
            return utils.attest_khdn(self.calls, 16, 32, 'no_carry', 1, self.layer_names)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(num_hidden_layers=1, num_heads=1, head_dim=32, to_dict=lambda: {})

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.p.sum())

    monkeypatch.setattr(training, 'KHDNTrace', Trace)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    args = benchmark.parse_arguments(['--methods', 'khdn_no_carry', '--khdn-strict', '--dtype', 'fp32'])
    args.no_layer_checkpoint = not recompute
    batch = training.Batch(inputs, 1, inputs['input_ids'].numel(), domain['consumed_tokens'], 0, 'cpu-fixture')
    result = training.trace_route(Model(), batch, args, 'khdn_no_carry', benchmark)
    assert result['actual_token_domain'] == domain
    # A self-consistent tiny original or replay trace cannot borrow the full batch's label.
    tiny = {'operator_batch_time': [1, 1], 'sequence_bounds': [0, 1]}
    replacement = _batch_bound_calls(tiny)[0]
    if recompute:
        replacement.update(backward_calls=0, checkpoint_frames=['recompute_fn'])
        replacement['launches'] = [row for row in replacement['launches'] if not row['kernel'].endswith('bwd_kernel')]
    calls[-1] = replacement
    batch.inputs = {'input_ids': torch.ones(8, 2048, dtype=torch.int64)} if not recompute else inputs
    with pytest.raises(AssertionError, match='shape|token|batch|sequence'):
        training.trace_route(Model(), batch, args, 'khdn_no_carry', benchmark)


def test_token_domain_rejects_boundary_substitution_with_same_flattened_count():
    from benchmarks import hyperbolic_training_distributed as training

    inputs, domain = _batch_binding_case('packed')
    calls = _batch_bound_calls(domain)
    calls[0]['sequence_bounds'] = [0, 0, 4, 7, 10]
    with pytest.raises(AssertionError, match='sequence boundaries'):
        training.attest_khdn_batch(calls, domain)
    inputs['cu_seqlens'] = torch.tensor([0, 11])
    with pytest.raises(AssertionError, match='actual batch'):
        training.khdn_batch_layout(inputs)
