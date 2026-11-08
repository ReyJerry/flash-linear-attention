# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""BF16 trace evidence from actual public model forward/backward graphs."""

import contextlib
import copy
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_hyperbolic_training as benchmark
from benchmarks import hyperbolic_benchmark_utils as utils
from benchmarks import hyperbolic_training_distributed as training

FLAGS = ['--methods', 'khdn_no_carry', '--khdn-precision-policy', 'bf16', '--loss-backend', 'plain',
         '--chunk-size', '16', '--check-only', '--seed', '101', '--data-seed', '1009']


def test_bf16_cli_and_small_model_keep_physical_geometry():
    args = benchmark.parse_arguments(FLAGS)
    assert args.optimizer_backend == 'fp32'
    config = benchmark.model_configuration(args, 'khdn_no_carry', small=True)
    assert (config['hidden_size'], config['head_dim'], config['group_size']) == (256, 128, 32)
    assert config['learnable_curvature'] and config['precision_policy'] == 'bf16'
    oracle = benchmark.model_configuration(args, 'khdn_naive', small=True)
    assert oracle['precision_policy'] == 'strict' and oracle['attn_mode'] == 'naive'
    assert oracle['head_dim'] == 128


@pytest.mark.parametrize('extra', [
    ['--optimizer-backend', 'native'], ['--dtype', 'fp32'], ['--group-size', '16'], ['--chunk-size', '32'], ['--no-short-conv'],
    ['--head-dim', '64', '--hidden-size', '1024'], ['--methods', 'khdn_y_carry'], ['--methods', 'khdn_naive'],
])
def test_bf16_cli_rejects_unqualified_geometry(extra):
    with pytest.raises(SystemExit):
        benchmark.parse_arguments(FLAGS+extra)


def test_bf16_cli_does_not_authorize_full_timing_from_small_gate():
    with pytest.raises(SystemExit):
        benchmark.parse_arguments([value for value in FLAGS if value != '--check-only'])


def parity_results():
    expected = dict(initial_weights_sha256='a'*64, loss=torch.tensor(4.),
                    gradients={'weight': torch.full((100,), .2)},
                    validation=dict(sample_id=10000, batch_sha256='b'*64, valid_labels=16, mean_ce=4.))
    actual = copy.deepcopy(expected)
    actual['gradients']['weight'][0] += .01
    actual['validation']['mean_ce'] += .0003
    return actual, expected


def test_bf16_model_gate_preserves_elementwise_failures_as_diagnostics():
    actual, expected = parity_results()
    result = benchmark.compare_bf16_model_results(actual, expected)
    assert result['global_gradient_cosine'] >= .99
    assert result['parameter_gradient_errors']['weight']['legacy_elementwise_failures'] == 1
    with pytest.raises(AssertionError):
        benchmark.compare_model_results(actual, expected)


@pytest.mark.parametrize('mutation', [
    lambda row: row['gradients']['weight'].mul_(2),
    lambda row: row['gradients']['weight'].neg_(),
    lambda row: row['gradients']['weight'].zero_(),
    lambda row: row['gradients']['weight'].fill_(float('nan')),
    lambda row: row['gradients'].update(weight=None),
    lambda row: row.update(initial_weights_sha256='c'*64),
    lambda row: row.update(loss=torch.tensor(float('nan'))),
    lambda row: row['validation'].update(mean_ce=float('nan')),
    lambda row: row['validation'].update(sample_id=0),
    lambda row: row['validation'].update(valid_labels=0),
    lambda row: row['validation'].update(batch_sha256='c'*64),
])
def test_bf16_model_gate_rejects_invalid_gradient_or_validation(mutation):
    actual, expected = parity_results()
    mutation(actual)
    with pytest.raises(AssertionError):
        benchmark.compare_bf16_model_results(actual, expected)


def test_bf16_validation_ce_uses_actual_valid_labels_and_restores_training():
    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            assert not self.training and not kwargs['use_cache'] and 'labels' not in kwargs
            return SimpleNamespace(logits=torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)/10)

    inputs = dict(input_ids=torch.zeros(2, 4, dtype=torch.long),
                  labels=torch.tensor([[0, 1, -100, 2], [0, -100, 1, -100]]))
    batch = training.Batch(inputs, 3, 8, 8, 10000, 'a'*64)
    model = Model().train()
    result = training.khdn_validation_ce(model, batch, None,
                                         SimpleNamespace(autocast=lambda args: contextlib.nullcontext()), candidate=True)
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)/10
    truth = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), inputs['labels'][:, 1:].reshape(-1))
    assert model.training and abs(result['mean_ce']-truth.item()) < 1e-6
    assert result['sample_id'] == 10000 and result['valid_labels'] == 3


def test_bf16_tensor_guard_rejects_inputs_and_results():
    outside = torch.ones(1, dtype=torch.float64)
    with utils._NoFP64Training():
        assert torch.ones(1, dtype=torch.float32).dtype == torch.float32
        with pytest.raises(RuntimeError, match='FP64 tensor input'):
            outside.float()
        with pytest.raises(RuntimeError, match='FP64 tensor result'):
            torch.ones(1, dtype=torch.float64)
    assert torch.ones(1, dtype=torch.float64).dtype == torch.float64


@pytest.mark.parametrize('chunk_size', [16, 128])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_bf16_native_model_trace_and_serialized_rejection(tmp_path, checkpoint, chunk_size, monkeypatch):
    pytest.importorskip('triton')
    from benchmarks.fp32_adamw import FP32AdamW
    from fla.models import KeyGatedHyperbolicDeltaNetConfig, KeyGatedHyperbolicDeltaNetForCausalLM
    if chunk_size == 128:
        from fla.ops.key_gated_hyperbolic_delta_rule import chunk128 as bf16
    else:
        from fla.ops.key_gated_hyperbolic_delta_rule import bf16
    from fla.utils import IS_NVIDIA, device
    if not IS_NVIDIA:
        pytest.skip('Actual BF16 compiler/graph trace requires NVIDIA CUDA.')
    try:
        bf16._require_runtime(torch.zeros((), device=device))
    except ValueError as error:
        pytest.skip(str(error))
    torch.manual_seed(42)
    config = KeyGatedHyperbolicDeltaNetConfig(
        hidden_size=256, head_dim=128, num_heads=2, num_hidden_layers=2, intermediate_size=512,
        vocab_size=97, group_size=32, chunk_size=chunk_size, curvature=.6, learnable_curvature=True,
        precision_policy='bf16', use_short_conv=True, fuse_norm=False, fuse_swiglu=False,
        fuse_cross_entropy=False, use_cache=False,
    )
    model = KeyGatedHyperbolicDeltaNetForCausalLM(config).to(device).train()
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(1, 97, (2, chunk_size + 1), device=device)
    original_forward = vars(bf16._ChunkBF16)['forward']
    trace = utils.KHDNTrace(chunk_size, 32, 'no_carry', 2, 'bf16').bind_model(model)
    trace.expected_checkpoint = checkpoint
    with utils.KHDNExceptionGuard('bf16') as guard, trace:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(ids, labels=ids, use_cache=False).loss
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert vars(bf16._ChunkBF16)['forward'] is original_forward
    assert guard.evidence()['fp64_tensor_guard'] and not guard.evidence()['events']
    result = trace.attest()
    assert result['route'] == 'bf16' and result['backward_calls'] == 2
    assert result['checkpoint_recompute_observed'] is checkpoint
    assert not result['numerical_policy_qualified'] and not result['performance_accepted']
    (tmp_path/'trace.json.gz').write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
    binding = copy.deepcopy(result['source_binding'])
    assert utils.reparse_khdn_trace(json.loads(json.dumps(result)), expected_source_binding=binding)['backward_calls'] == 2
    mutations = [
        lambda row: row['requested'].update(precision_policy='strict'),
        lambda row: row['executed_options'].update(arithmetic='strict'),
        lambda row: row['telemetry'].update(host_fallback=True),
        lambda row: row['telemetry'].update(backward_precision_route='strict'),
        lambda row: row['launches'].pop(),
        lambda row: row['launches'].pop(6),
        lambda row: row['launches'].reverse(),
        lambda row: next(x for x in row['launches'] if x['kernel'] == 'state_fwd_kernel')[
            'arguments']['SavedU'].update(dtype='torch.float64'),
        lambda row: row['launches'][0]['arguments'].update(C=32),
        lambda row: next(x for x in row['launches'] if x['kernel'] == 'state_fwd_kernel')[
            'arguments']['SavedS'].update(dtype='torch.float64'),
        lambda row: next(x for x in row['launches'] if x['kernel'] == 'state_fwd_kernel')[
            'arguments']['Starts'].update(shape=[999]),
        lambda row: row['launches'][0].update(source_binding_sha256='0'*64),
        lambda row: row['launches'][0]['compiled_precision_inventory'].update(ptx_sha256='0'*64),
        lambda row: row.update(layer='unobserved_layer'),
    ]
    for mutate in mutations:
        altered = copy.deepcopy(result)
        row = next(value for value in altered['operator_calls'] if value['backward_calls'])
        mutate(row)
        with pytest.raises(AssertionError):
            utils.reparse_khdn_trace(altered, expected_source_binding=binding)
    batch = training.Batch(dict(input_ids=ids, labels=ids), 2*chunk_size, ids.numel(), ids.numel(), 0,
                           training.batch_digest(dict(input_ids=ids, labels=ids)))
    args = SimpleNamespace(khdn_precision_policy='bf16', no_layer_checkpoint=not checkpoint, dtype='bf16')
    optimizer = FP32AdamW(model.parameters(), lr=1e-4)
    records = []
    optimizer.zero_grad(set_to_none=True)
    with training.khdn_update_guard(args, 'khdn_no_carry', 'native_update', records,
                                    model=model, batches=[batch], source_binding=binding):
        training.weighted_loss(model, batch, args, batch.valid_labels, 1).backward()
    assert records[-1]['positive_observation']['attested_before_optimizer_step']
    optimizer.step()
    for corruption in ('policy', 'batch_identity', 'missing_layer'):
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer_before = training.state_digest(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        supplied = batch
        with monkeypatch.context() as patch:
            if corruption == 'policy':
                patch.setattr(model.model.layers[0].attn, 'precision_policy', 'strict')
            elif corruption == 'batch_identity':
                supplied = copy.copy(batch)
                supplied.inputs = {key: value.clone() for key, value in batch.inputs.items()}
            else:
                patch.setattr(model.model.layers[0].attn, 'forward',
                              lambda hidden_states, *args, **kwargs: (hidden_states, None, kwargs.get('past_key_values')))
            with pytest.raises(AssertionError):
                with training.khdn_update_guard(args, 'khdn_no_carry', corruption, records,
                                                model=model, batches=[batch], source_binding=binding):
                    training.weighted_loss(model, supplied, args, 32, 1).backward()
                optimizer.step()
        assert optimizer_before == training.state_digest(optimizer.state_dict())
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    (tmp_path/'update-guards.json.gz').write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))


def test_fp32_optimizer_factory_retains_parameter_group_policy(monkeypatch):
    import sys
    from types import ModuleType

    captured = {}

    def construct(groups, **kwargs):
        captured.update(groups=groups, **kwargs)
        return captured

    module = ModuleType('benchmarks.fp32_adamw')
    module.FP32AdamW = construct
    monkeypatch.setitem(sys.modules, 'benchmarks.fp32_adamw', module)
    model = torch.nn.Linear(3, 2)
    model.bias._no_weight_decay = True
    args = SimpleNamespace(lr=1e-4, weight_decay=.1, optimizer_backend='fp32')
    result = training.make_optimizer(model, args)
    assert result is captured and result['lr'] == args.lr and result['betas'] == (.9, .95)
    assert [group['weight_decay'] for group in result['groups']] == [.1, 0.]
    assert result['groups'][0]['params'][0] is model.weight
    assert result['groups'][1]['params'][0] is model.bias


def test_c128_cli_and_explicit_reference_configuration():
    args = benchmark.parse_arguments(FLAGS + ['--chunk-size', '128'])
    actual = benchmark.model_configuration(args, 'khdn_no_carry', small=True)
    oracle = benchmark.model_configuration(args, 'khdn_naive', small=True)
    assert actual['chunk_size'] == 128 and actual['precision_policy'] == 'bf16'
    assert oracle['chunk_size'] == 64 and oracle['attn_mode'] == 'naive'
    assert args.optimizer_backend == 'fp32'
    full = benchmark.parse_arguments([value for value in FLAGS if value != '--check-only'] + ['--chunk-size', '128'])
    assert not full.check_only and full.chunk_size == 128 and full.optimizer_backend == 'fp32'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'phase', 'inverse_level', 'inverse_precision', 'mode', 'rho'])
def test_c128_schedule_rejects_incomplete_or_wrong_specializations(mutation):
    from collections import defaultdict

    from benchmarks.khdn_c128_trace import BACKWARD, FORWARD, attest_schedule

    seen = defaultdict(int)
    launches = []
    for name in (*FORWARD, *BACKWARD):
        index = seen[name]
        seen[name] += 1
        args = dict(C=128)
        if name == 'diagonal_inverse_kernel':
            args['B'] = 16
        if name == 'inverse_offdiagonal_kernel':
            args.update(N=[16, 32, 64][index % 3], SINGLE=True)
        if name in ('common_finish_tiles', 'shared_coefficients_bwd', 'common_products_bwd'):
            args['MODE'] = index
        if name in ('state_fwd_kernel', 'state_bwd_kernel'):
            args['RHO_MATERIALIZED'] = False
        launches.append(dict(kernel=name, phase='forward' if name in FORWARD else 'backward', arguments=args))
    attest_schedule(launches, True)
    if mutation == 'missing':
        launches.pop()
    elif mutation == 'duplicate':
        launches.append(launches[-1])
    elif mutation == 'phase':
        launches[-1]['phase'] = 'forward'
    elif mutation == 'inverse_level':
        next(x for x in launches if x['kernel'] == 'inverse_offdiagonal_kernel')['arguments']['N'] = 64
    elif mutation == 'inverse_precision':
        next(x for x in launches if x['kernel'] == 'inverse_offdiagonal_kernel')['arguments']['SINGLE'] = False
    elif mutation == 'mode':
        next(x for x in launches if x['kernel'] == 'shared_coefficients_bwd')['arguments']['MODE'] = 3
    elif mutation == 'rho':
        next(x for x in launches if x['kernel'] == 'state_bwd_kernel')['arguments']['RHO_MATERIALIZED'] = True
    with pytest.raises(AssertionError):
        attest_schedule(launches, True)


@pytest.mark.parametrize('relative_gap', [.001228, .01])
def test_bf16_ppl_observation_preserves_excess_without_claiming_quality(relative_gap):
    import math
    actual, expected = parity_results()
    actual['validation']['mean_ce'] = expected['validation']['mean_ce'] + math.log1p(relative_gap)
    result = benchmark.compare_bf16_model_results(actual, expected)
    assert result['validation_ppl_relative_delta'] == pytest.approx(relative_gap)
    assert result['validation_ppl_observation_exceeded']
    assert result['validation_ppl_observation_limit'] == .001
    assert not result['training_quality_accepted']


@pytest.mark.parametrize('mutation', ['diagnostic', 'key_matrix', 'decay', 'inverse', 'rho', 'state', 'residual'])
def test_c128_saved_buffer_contract_rejects_precision_mismatches(mutation):
    from benchmarks.hyperbolic_benchmark_utils import _attest_bf16_sources

    fp32, bf16 = 'torch.float32', 'torch.bfloat16'
    flags = {
        'P': [{'dtype': bf16 if i < 18 and i not in (2, 3, 4) else fp32} for i in range(23)],
        'Gp': [{'dtype': bf16 if i in (0, 1, 3, 4) else fp32} for i in range(8)],
        'Carry': [{'dtype': fp32}, {'dtype': fp32}],
        **{name: {'dtype': bf16 if name in ('SavedS', 'SavedY', 'SavedU') else fp32}
           for name in ('SavedS', 'SavedY', 'SavedU', 'SavedR', 'SavedT', 'SavedW', 'SavedZP')},
        **{name: {'shape': [count], 'dtype': 'torch.int64'}
           for name, count in (('Starts', 1), ('Ends', 1), ('SeqChunks', 2))},
    }
    diagnostic = {'saved_state_y_bf16': True}
    output = {'SavedRO': {'dtype': bf16}}
    row = {
        'executed_sources': [{'shape': [1, 128, 1, 128], 'dtype': fp32} for _ in range(7)],
        'requested': {'chunk_size': 128}, 'backward_calls': 1,
        'output_all_finite': True, 'gradients_all_finite': True,
        'expected_chunk_lengths': [128], 'sequence_bounds': [0, 128],
        'launches': [
            {'kernel': 'state_fwd_kernel', 'arguments': flags, 'y_diagnostics': diagnostic},
            {'kernel': 'output_fwd_kernel', 'arguments': output},
        ],
    }
    _attest_bf16_sources(row, False)
    if mutation == 'diagnostic':
        diagnostic['saved_state_y_bf16'] = False
    elif mutation == 'key_matrix':
        flags['P'][0]['dtype'] = fp32
    elif mutation == 'decay':
        flags['P'][2]['dtype'] = bf16
    elif mutation == 'inverse':
        flags['Gp'][0]['dtype'] = fp32
    elif mutation == 'rho':
        flags['Gp'][2]['dtype'] = bf16
    elif mutation == 'state':
        flags['SavedS']['dtype'] = fp32
    else:
        output['SavedRO']['dtype'] = fp32
    with pytest.raises(AssertionError):
        _attest_bf16_sources(row, False)


def test_checkpoint_contract_distinguishes_fp32_optimizer_from_native(monkeypatch):
    monkeypatch.delenv('FLA_SOURCE_MANIFEST', raising=False)
    args = benchmark.parse_arguments(['--methods', 'kda', '--optimizer-backend', 'fp32', '--check-only'])
    current = training.checkpoint_contract(args, 'kda', 1)
    args.optimizer_backend = 'native'
    legacy = training.checkpoint_contract(args, 'kda', 1)
    assert current.pop('optimizer_backend') == 'triton_fp32_adamw_v1'
    assert 'optimizer_backend' not in legacy and current == legacy
