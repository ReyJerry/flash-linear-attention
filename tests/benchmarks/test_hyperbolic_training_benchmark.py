# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU coverage of training benchmark configuration and fail-closed evidence gates.

These tests do not execute or validate CUDA kernels or GPU timing accuracy.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
import torch

from benchmarks import benchmark_hyperbolic_training as benchmark


def test_training_benchmark_default_dry_run():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, 'benchmarks/benchmark_hyperbolic_training.py', '--dry-run'],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['status'] == 'DRY RUN; GPU NOT RUN'
    contract = payload['training_contract']
    assert contract['parameter_dtype'] == 'fp32' and contract['autocast_dtype'] == 'bf16'
    assert contract['outer_checkpoint'] and not contract['use_reentrant']
    assert not contract['protected_inner_checkpoint']
    config = payload['model_configurations']['hdn_tc']
    assert (config['num_hidden_layers'], config['hidden_size'], config['num_heads'], config['head_dim']) == (24, 2048, 16, 128)
    assert (config['intermediate_size'], config['vocab_size'], config['expand_v']) == (5184, 32000, 1.0)
    assert payload['metadata']['arguments']['warmup'] == 10
    assert payload['metadata']['arguments']['steps'] == 30
    assert len(payload['methods']) == 6


@pytest.mark.parametrize('arguments', [
    ['--head-dim', '129'], ['--hidden-size', '1024'], ['--seq-len', '1'], ['--vocab-size', '3'],
    ['--curvature', 'nan'], ['--lr', 'inf'], ['--weight-decay', '-1'], ['--warmup', '0'],
    ['--steps', '1'], ['--stability-steps', '-1'], ['--rank', '17'], ['--methods', 'hdn_tc', 'hdn_tc'],
])
def test_training_benchmark_rejects_invalid_configuration(arguments):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_arguments(arguments)
    assert error.value.code == 2


def test_training_model_configuration_preserves_comparison_contract():
    args = benchmark.parse_arguments(['--dtype', 'fp32', '--inner-checkpoint', '--learnable-curvature'])
    for name in benchmark.METHOD_NAMES:
        config = benchmark.model_configuration(args, name)
        assert config['expand_v'] == 1 and config['head_dim'] == 128
        assert config['use_cache'] is False and config['fuse_cross_entropy'] is True
        if name != 'gdn':
            assert config['tensor_core'] and config['learnable_curvature']
            assert config['protected_checkpoint']
            assert config['protection_rank'] == (4 if name.startswith('protected_') else 0)
    for name, flag in (('hdn_staged', 'FLA_HYPERBOLIC_STAGED'),
                       ('protected_shared_inverse', 'FLA_HYPERBOLIC_SHARED_INVERSE'),
                       ('protected_staged', 'FLA_HYPERBOLIC_PROTECTED_STAGED')):
        spec = benchmark.method_spec(name)
        assert spec['environment'][flag] == '1'
        assert spec['environment']['FLA_HYPERBOLIC_TC_FAST'] == '0'
        assert all(value == '0' for key, value in spec['environment'].items() if key != flag)


def test_training_benchmark_no_gpu_is_explicit(monkeypatch, tmp_path):
    path = tmp_path / 'training.json'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert benchmark.main(['--check-only', '--output', str(path)]) == 2
    payload = json.loads(path.read_text())
    assert payload['status'] == 'GPU NOT RUN: CUDA device unavailable'
    assert 'correctness' not in payload and 'measurements' not in payload


def test_weight_digest_and_parameter_gradient_comparison_fail_closed():
    first, second = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    second.load_state_dict(first.state_dict())
    digest = benchmark.weight_digest(first)
    assert benchmark.weight_digest(second) == digest
    with torch.no_grad():
        second.weight[0, 0] += 1
    assert benchmark.weight_digest(second) != digest
    reference = {'initial_weights_sha256': digest, 'loss': torch.tensor(0.5),
                 'gradients': {'weight': torch.tensor([0.2, -0.3])}}
    actual = {**reference, 'gradients': {'weight': reference['gradients']['weight'].clone()}}
    assert benchmark.compare_model_results(actual, reference)['parameter_gradient_errors']['weight']['max_abs'] == 0
    for bad in (
        {**actual, 'initial_weights_sha256': 'different'},
        {**actual, 'loss': torch.tensor(2.0)},
        {**actual, 'gradients': {}},
        {**actual, 'gradients': {'weight': None}},
        {**actual, 'gradients': {'weight': torch.tensor([float('nan'), 0.0])}},
        {**actual, 'gradients': {'weight': torch.tensor([2.0, 3.0])}},
    ):
        with pytest.raises(AssertionError):
            benchmark.compare_model_results(bad, reference)


def test_training_gradient_health_checks_all_parameters():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(AssertionError, match='Missing'):
        benchmark.gradient_health(model)
    model(torch.ones(1, 3)).square().sum().backward()
    expected = torch.cat([parameter.grad.flatten() for parameter in model.parameters()]).norm().item()
    assert benchmark.gradient_health(model) == pytest.approx(expected)
    model.weight.grad[0, 0] = float('nan')
    with pytest.raises(AssertionError, match='Non-finite'):
        benchmark.gradient_health(model)


def test_training_final_weights_checked_after_last_update():
    model = torch.nn.Linear(3, 2)
    assert benchmark.parameter_health(model) == {'all_finite': True, 'parameter_tensors_checked': 2}
    with torch.no_grad():
        model.bias[0] = float('nan')
    with pytest.raises(AssertionError, match='final optimizer update'):
        benchmark.parameter_health(model)


@pytest.mark.parametrize('name', ['hdn_tc', 'protected_value', 'protected_shared_inverse'])
def test_training_trace_requires_matching_forward_and_backward_flags(name):
    if name == 'hdn_tc':
        kernels, flag, enabled = ['hyperbolic_delta_tc_chunk_fwd', 'hyperbolic_delta_tc_chunk_bwd'], 'FAST', False
    else:
        kernels = ['protected_value_tile_fwd_kernel', 'protected_value_tile_bwd_kernel', 'soft_bank_update_bwd_kernel']
        flag, enabled = 'SHARED_INVERSE', name == 'protected_shared_inverse'
        if enabled:
            kernels.append('shared_key_gram_fwd_kernel')
    records = [{'kernel': kernel, 'flags': {flag: enabled}, 'launches': 1} for kernel in kernels]
    benchmark.attest_training(name, records)
    records[1]['flags'][flag] = not enabled
    with pytest.raises(AssertionError, match='policy|precision'):
        benchmark.attest_training(name, records)


def test_training_staged_trace_requires_all_backward_stages():
    for name, prefix, stages in (
        ('hdn_staged', 'hyperbolic_staged_', (
            'prepare_fwd', 'state_fwd', 'output_fwd', 'output_bwd', 'state_bwd', 'values_bwd', 'parameters_bwd')),
        ('protected_staged', 'protected_', (
            'shared_trajectory_fwd', 'value_trajectory_fwd', 'value_trajectory_bwd',
            'shared_trajectory_bwd', 'value_partials_bwd', 'bank_partials_bwd')),
    ):
        records = [{'kernel': prefix + stage + '_kernel', 'flags': {}, 'launches': 1} for stage in stages]
        if name == 'protected_staged':
            records.insert(0, {'kernel': 'shared_key_gram_fwd_kernel', 'flags': {}, 'launches': 1})
        benchmark.attest_training(name, records)
        with pytest.raises(AssertionError, match='required training kernel'):
            benchmark.attest_training(name, records[:-1])


def test_gdn_trace_records_execution_and_restores_profiler():
    namespace = {'__name__': 'fla.ops.common.backends.tilelang.chunk_bwd'}
    exec('def chunk_bwd_dqkwg_tilelang():\n    return 1\n', namespace)
    previous = sys.getprofile()
    with benchmark.GDNBackendTrace() as trace:
        assert namespace['chunk_bwd_dqkwg_tilelang']() == 1
    assert sys.getprofile() is previous
    records = trace.records()
    records += [{'function': 'fla.ops.gated_delta_rule.chunk.' + suffix, 'calls': 1}
                for suffix in ('chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd')]
    assert benchmark.attest_training('gdn', records)['tilelang_backward_observed'] is True
    with pytest.raises(AssertionError):
        benchmark.attest_training('gdn', records[:-1])


def test_gdn_entry_trace_observes_autograd_style_worker_thread(monkeypatch):
    name = 'fla.ops.gated_delta_rule.training_benchmark_test'
    module = ModuleType(name)
    exec('def chunk_gated_delta_rule_bwd():\n    return 1\n', vars(module))
    original = module.chunk_gated_delta_rule_bwd
    monkeypatch.setitem(sys.modules, name, module)
    with benchmark.GDNBackendTrace() as trace:
        thread = threading.Thread(target=module.chunk_gated_delta_rule_bwd)
        thread.start()
        thread.join()
    assert module.chunk_gated_delta_rule_bwd is original
    assert trace.records() == [{'function': name + '.chunk_gated_delta_rule_bwd', 'calls': 1}]


def test_training_speedups_use_only_matching_baselines():
    rows = {}
    for name, latency in (('hdn_tc', 8), ('hdn_staged', 4), ('protected_staged', 20), ('gdn', 2)):
        rows[name] = {'summary_ms': {key: {'median': latency} for key in (
            'fwd_cuda_ms', 'bwd_cuda_ms', 'optimizer_cuda_ms', 'total_cuda_ms', 'total_wall_ms')}}
    benchmark.add_comparisons(rows)
    assert rows['hdn_staged']['speedup_against_baseline']['total_wall_ms'] == 2
    assert rows['hdn_staged']['training_throughput_fraction_of_gdn'] == 0.5
    assert 'speedup_against_baseline' not in rows['protected_staged']
    assert 'speedup_against_baseline' not in rows['gdn']


def test_real_model_constructors_and_checkpoint_hooks_on_cpu(monkeypatch):
    """Real FLA/Transformers constructors only: no model forward/backward and no CUDA execution."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *args, **kwargs: self)
    args = benchmark.parse_arguments(['--learnable-curvature'])
    hashes = {}
    for name in benchmark.METHOD_NAMES:
        spec = benchmark.method_spec(name)
        with benchmark.environment(spec['environment']):
            model, info = benchmark.build_model(args, spec, small=True)
        assert all(parameter.device.type == 'cpu' and parameter.dtype == torch.float32 for parameter in model.parameters())
        assert info['parameter_count'] == sum(parameter.numel() for parameter in model.parameters())
        if spec['group'] in hashes:
            assert info['initial_weights_sha256'] == hashes[spec['group']]
        hashes[spec['group']] = info['initial_weights_sha256']
        blocks = [module for module in model.modules() if isinstance(module, GradientCheckpointingLayer)]
        assert len(blocks) == 2
        assert all(block.gradient_checkpointing for block in blocks)
        assert all(block._gradient_checkpointing_func.keywords['use_reentrant'] is False for block in blocks)
        if name != 'gdn':
            assert all(block.attn.tensor_core and not block.attn.protected_checkpoint for block in blocks)
        del model
