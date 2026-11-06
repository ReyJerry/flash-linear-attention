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
    config = benchmark.model_configuration(args, 'khdn_no_carry', small=True)
    assert (config['hidden_size'], config['head_dim'], config['group_size']) == (256, 128, 32)
    assert config['learnable_curvature'] and config['precision_policy'] == 'bf16'
    oracle = benchmark.model_configuration(args, 'khdn_naive', small=True)
    assert oracle['precision_policy'] == 'strict' and oracle['attn_mode'] == 'naive'
    assert oracle['head_dim'] == 128


@pytest.mark.parametrize('extra', [
    ['--dtype', 'fp32'], ['--group-size', '16'], ['--chunk-size', '32'], ['--no-short-conv'],
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
    lambda row: row['validation'].update(mean_ce=4.01),
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


@pytest.mark.parametrize('checkpoint', [False, True])
def test_bf16_native_model_trace_and_serialized_rejection(tmp_path, checkpoint, monkeypatch):
    pytest.importorskip('triton')
    from fla.models import KeyGatedHyperbolicDeltaNetConfig, KeyGatedHyperbolicDeltaNetForCausalLM
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
        vocab_size=97, group_size=32, chunk_size=16, curvature=.6, learnable_curvature=True,
        precision_policy='bf16', use_short_conv=True, fuse_norm=False, fuse_swiglu=False,
        fuse_cross_entropy=False, use_cache=False,
    )
    model = KeyGatedHyperbolicDeltaNetForCausalLM(config).to(device).train()
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(1, 97, (2, 17), device=device)
    original_forward = vars(bf16._ChunkBF16)['forward']
    trace = utils.KHDNTrace(16, 32, 'no_carry', 2, 'bf16').bind_model(model)
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
        lambda row: row['launches'][0]['arguments'].update(C=32),
        lambda row: row['launches'][3]['arguments']['SavedS'].update(dtype='torch.float64'),
        lambda row: row['launches'][3]['arguments']['Starts'].update(shape=[999]),
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
    batch = training.Batch(dict(input_ids=ids, labels=ids), 32, 34, 34, 0,
                           training.batch_digest(dict(input_ids=ids, labels=ids)))
    args = SimpleNamespace(khdn_precision_policy='bf16', no_layer_checkpoint=not checkpoint, dtype='bf16')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    records = []
    optimizer.zero_grad(set_to_none=True)
    with training.khdn_update_guard(args, 'khdn_no_carry', 'native_update', records,
                                    model=model, batches=[batch], source_binding=binding):
        training.weighted_loss(model, batch, args, 32, 1).backward()
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
