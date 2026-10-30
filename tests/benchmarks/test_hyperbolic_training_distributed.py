# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU contract tests; Gloo/tiny LM checks do not establish CUDA/NCCL or KHDN DDP acceptance."""

import ast
import copy
import hashlib
import io
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tarfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks import benchmark_hyperbolic_training as benchmark
from benchmarks import hyperbolic_benchmark_utils as benchmark_utils
from benchmarks import hyperbolic_training_distributed as training


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(97, 8)
        self.projection = torch.nn.Linear(8, 97)
        self.projection.bias._no_weight_decay = True

    def forward(self, input_ids, labels, num_items_in_batch, **kwargs):
        logits = self.projection(self.embedding(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 97), labels[:, 1:].reshape(-1),
                               ignore_index=-100, reduction='sum') / num_items_in_batch
        return SimpleNamespace(loss=loss)


def options():
    return benchmark.parse_arguments(['--methods', 'gdn', '--dtype', 'fp32'])


@pytest.mark.parametrize('layout', ['dense', 'padding', 'packed'])
def test_masks_match_exact_shifted_loss_and_are_reproducible(layout):
    args = options()
    args.data_layout = layout
    first, second = training.make_batch(args, 7, small=True), training.make_batch(args, 7, small=True)
    assert first.sha256 == second.sha256
    assert first.sha256 != training.make_batch(args, 8, small=True).sha256
    labels, inputs = first.inputs['labels'], first.inputs['input_ids']
    assert first.valid_labels == int(labels[:, 1:].ne(-100).sum())
    logits = torch.zeros(*inputs.shape, 97)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 97), labels[:, 1:].reshape(-1), reduction='sum')
    assert loss.item() == pytest.approx(first.valid_labels * np.log(97), rel=1e-6)
    if layout == 'packed':
        assert (labels[0, first.inputs['cu_seqlens'][1:-1].long()] == -100).all()
    if layout == 'padding':
        assert first.nonpadding_tokens == int(first.inputs['attention_mask'].sum())


def test_globally_empty_step_is_rejected_before_backward():
    runtime = object.__new__(training.Runtime)
    runtime.device, runtime.world_size = torch.device('cpu'), 1
    batch = training.make_batch(options(), 0, small=True)
    batch.valid_labels = 0
    with pytest.raises(ValueError, match='no valid shifted labels'):
        runtime.global_count([batch])


def test_scaling_same_global_strong_samples_and_distinct_weak_shards():
    args = options()
    strong_ids = []
    for world in (1, 8, 16):
        args.scaling = 'strong'
        plan = training.scaling_plan(args, world)
        assert plan['accumulation'] == 16 // world
        assert plan['global_input_slots_per_step'] == 262144
        ids = sorted(index for rank in range(world)
                     for index in training.sample_indices(5, rank, world, plan['accumulation']))
        assert len(ids) == len(set(ids)) == 16
        strong_ids.append(ids)
        args.scaling = 'weak'
        weak = training.scaling_plan(args, world)
        assert weak['global_input_slots_per_step'] == 16384 * world
        assert weak['accumulation'] == 1
    assert strong_ids[0] == strong_ids[1] == strong_ids[2]


@pytest.mark.parametrize('flags', [
    ['--final-validation'], ['--accumulation', '0'], ['--accumulation', '-1'], ['--task-timeout', '0'], ['--ddp-timeout', '-1'],
    ['--clip-grad', 'nan'], ['--trials', '0'], ['--modes', 'step', 'step'],
    ['--scaling', 'strong', '--batch', '1'], ['--scaling', 'weak', '--accumulation', '1'],
    ['--methods', 'khdn_naive'], ['--methods', 'khdn_no_carry'], ['--methods', 'khdn_y_carry'],
    ['--methods', 'gdn2'], ['--local-rank', '-1'], ['--methods', 'protected_value',
                                                    '--data-layout', 'packed'], ['--resume', '/tmp/state', '--methods', 'gdn', 'kda'],
])
def test_argument_gates(flags):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_arguments(flags)
    assert error.value.code == 2


def test_audit_and_tiny_g32_configuration():
    audit = training.audit_methods()
    assert audit['kda']['available'] and audit['comba']['available']
    assert audit['gdn2']['operator_available'] and audit['gdn2']['layer_available']
    assert not audit['gdn2']['available']
    args = benchmark.parse_arguments(['--methods', 'khdn_naive', '--check-only'])
    config = benchmark.model_configuration(args, 'khdn_naive', small=True)
    assert config['head_dim'] == config['group_size'] == 32
    assert config['curvature'] > 0 and config['attn_mode'] == 'naive'


def test_resume_optimizer_groups_rng_data_and_next_update(tmp_path):
    args = options()
    torch.manual_seed(11)
    random.seed(12)
    np.random.seed(13)
    model = TinyLM()
    optimizer = training.make_optimizer(model, args)
    batch = training.make_batch(args, 4, small=True)
    training.backward_step(model, [batch], args, batch.valid_labels, 1)
    training.clip_and_step(model, optimizer, args)
    contract, path = {'test': 'recovery'}, tmp_path / 'state.pt'
    training.save_checkpoint(path, model, optimizer, step=5, optimizer_step=3, contract=contract, rank=0)
    expected_rng = (random.random(), np.random.random(), torch.rand(2))
    optimizer.zero_grad(set_to_none=True)
    training.backward_step(model, [batch], args, batch.valid_labels, 1)
    training.clip_and_step(model, optimizer, args)
    expected, expected_optimizer = training.parameters(model), copy.deepcopy(optimizer.state_dict())
    # restoring a completed generation also restores/enforces the current no-decay policy
    optimizer.param_groups[1]['weight_decay'] = 0.7
    assert training.load_checkpoint(path, model, optimizer, contract=contract, rank=0,
                                    weight_decay=args.weight_decay) == (5, 3)
    assert (random.random(), np.random.random()) == expected_rng[:2]
    assert torch.equal(torch.rand(2), expected_rng[2])
    assert optimizer.param_groups[1]['weight_decay'] == 0
    optimizer.zero_grad(set_to_none=True)
    training.backward_step(model, [batch], args, batch.valid_labels, 1)
    training.clip_and_step(model, optimizer, args)
    training.error_metrics(training.parameters(model), expected, atol=0, rtol=0)
    for index, state in optimizer.state_dict()['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, expected_optimizer['state'][index][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='mismatch'):
        training.load_checkpoint(path, model, optimizer, contract={'wrong': 1}, rank=0, weight_decay=0.1)


def test_aggregate_reports_max_rank_and_three_token_counts():
    common = {'global_valid_labels': 3, 'input_slots': 8, 'nonpadding_tokens': 6, 'weighted_local_loss': 2.0}
    rows = training.aggregate_samples([[dict(common, wall_ms=2, cuda_ms=1)], [dict(common, wall_ms=4, cuda_ms=3)]])
    assert rows[0]['max_rank_wall_ms'] == 4
    assert rows[0]['global_input_slots'] == 16 and rows[0]['global_nonpadding_tokens'] == 12
    assert rows[0]['valid_labels_per_second'] == 750


def distributed_worker(rank, rendezvous, output):
    """Launched by torchrun in a bounded subprocess, never by an extra agent."""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', world_size=2, rank=rank,
                            timeout=timedelta(seconds=30))
    try:
        args = SimpleNamespace(dtype='fp32', data_seed=43, data_layout='packed', lr=1e-4, weight_decay=0.1, clip_grad=1.0)
        torch.manual_seed(17)
        model = TinyLM()
        serial = training.copy_model(model)
        ddp = DistributedDataParallel(model)
        batches = [training.make_batch(args, index, small=True) for index in range(4)]
        # final rank has no labels in either microstep, including the synchronizing last backward
        for index in (1, 3):
            batches[index].inputs['labels'].fill_(-100)
            batches[index].valid_labels = 0
        count = sum(batch.valid_labels for batch in batches)
        local = [batches[index] for index in training.sample_indices(0, rank, 2, 2)]
        optimizer, serial_optimizer = training.make_optimizer(ddp, args), training.make_optimizer(serial, args)
        training.backward_step(ddp, local, args, count, 2)
        for batch in batches:
            training.weighted_loss(serial, batch, args, count, 1).backward()
        training.error_metrics(training.gradients(ddp), training.gradients(serial), atol=1e-7, rtol=1e-6)
        training.clip_and_step(ddp, optimizer, args)
        training.clip_and_step(serial, serial_optimizer, args)
        training.error_metrics(training.parameters(ddp), training.parameters(serial), atol=1e-7, rtol=1e-6)
        Path(f'{output}.{rank}.json').write_text(json.dumps({'rank': rank, 'gloo_equivalence': True, 'cuda_accepted': False}))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gloo_unequal_labels_zero_rank_and_update(tmp_path):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc-per-node=2',
               '--master-addr=127.0.0.1', f'--master-port={port}',
               str(Path(__file__).resolve()), '--worker', str(tmp_path / 'rendezvous'), str(tmp_path / 'result')]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                               env={**os.environ, 'OMP_NUM_THREADS': '1', 'FLA_CI_ENV': '0'})
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f'owned Gloo test process group timed out: {stdout} {stderr}')
    result = SimpleNamespace(returncode=process.returncode, stdout=stdout, stderr=stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(json.loads((tmp_path / f'result.{rank}.json').read_text())['gloo_equivalence'] for rank in range(2))


def test_real_shared_loss_helper_sum_denominator_and_all_ignored():
    source = training.ROOT / 'fla/models/modeling_utils.py'
    module = ast.parse(source.read_text())
    owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and
                 any(isinstance(method, ast.FunctionDef) and method.name == '_compute_logits_and_loss' for method in node.body))
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '_compute_logits_and_loss')
    extracted = ast.Module(body=[ast.ClassDef(name='LossHost', bases=[], keywords=[], body=[method], decorator_list=[])],
                           type_ignores=[])
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.fix_missing_locations(extracted), str(source), 'exec'), namespace)
    host = namespace['LossHost']()
    host.config = SimpleNamespace(fuse_linear_cross_entropy=False, fuse_cross_entropy=False, use_l2warp=False)
    host.criterion, host.lm_head = None, torch.nn.Linear(3, 97)
    args = options()
    args.data_layout = 'packed'
    batch = training.make_batch(args, 2, small=True)
    states = torch.randn(*batch.inputs['input_ids'].shape, 3, requires_grad=True)
    denominator = batch.valid_labels / 8
    loss, logits = host._compute_logits_and_loss(states, labels=batch.inputs['labels'], num_items_in_batch=denominator)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 97), batch.inputs['labels'][:, 1:].reshape(-1),
                               reduction='sum') / denominator
    torch.testing.assert_close(loss, expected)
    batch.inputs['labels'].fill_(-100)
    zero, _ = host._compute_logits_and_loss(states, labels=batch.inputs['labels'], num_items_in_batch=denominator)
    assert float(zero.detach()) == 0 and torch.isfinite(zero)
    zero.backward()
    assert states.grad is not None and torch.equal(states.grad, torch.zeros_like(states.grad))


def test_native_causal_lms_explicitly_route_loss_denominator():
    for path in ('key_gated_hyperbolic_deltanet', 'hyperbolic_deltanet', 'gated_deltanet', 'kda', 'comba'):
        source = training.ROOT / f'fla/models/{path}/modeling_{path}.py'
        tree = ast.parse(source.read_text())
        if path in ('gated_deltanet', 'comba'):
            class_name = 'GatedDeltaNetForCausalLM' if path == 'gated_deltanet' else 'CombaForCausalLM'
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            assert not any(isinstance(node, ast.FunctionDef) and node.name == 'forward' for node in owner.body)
            assert any(isinstance(base, ast.Name) and base.id == 'FLAForCausalLM' for base in owner.bases)
            tree = ast.parse((training.ROOT / 'fla/models/modeling_utils.py').read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and any(key.arg == 'num_items_in_batch' and isinstance(key.value, ast.Name)
                           and key.value.id == 'num_items_in_batch' for key in node.keywords)]
        assert matches, f'{path} drops num_items_in_batch'


def test_recipe_parameter_inventory_full_dimensions_and_increment():
    args = options()
    expected = {'hdn_tc': 1400280064, 'gdn': 1401067264, 'kda': 1324881280,
                'comba': 1401068032, 'khdn_no_carry': 1412912512}
    for name, count in expected.items():
        assert training.recipe_parameter_counts(args, name)['total'] == count
    assert expected['khdn_no_carry'] - expected['hdn_tc'] == 12632448
    args.learnable_curvature = True
    assert training.recipe_parameter_counts(args, 'khdn_no_carry')['total'] == expected['khdn_no_carry'] + 24


@pytest.mark.parametrize('flags', [
    ['--modes', 'fwd', 'step'], ['--modes', 'step', 'fwd', '--scaling', 'strong'],
    ['--modes', 'fwd', '--resume', '/tmp/checkpoint'], ['--modes', 'bwd', '--checkpoint-dir', '/tmp/checkpoint'],
    ['--memory-only', '--trials', '3'], ['--memory-only', '--profile', '/tmp/profile'],
])
def test_reject_unsafe_trajectory_and_memory_combinations(flags):
    with pytest.raises(SystemExit):
        benchmark.parse_arguments(['--methods', 'gdn', *flags])


def test_checkpoint_contract_full_shape_checkpoint_loss_and_backend_policy(monkeypatch):
    first = benchmark.parse_arguments(['--methods', 'comba', '--batch', '4', '--seq-len', '8'])
    other = benchmark.parse_arguments(['--methods', 'comba', '--batch', '2', '--seq-len', '16'])
    assert training.checkpoint_contract(first, 'comba', 1) != training.checkpoint_contract(other, 'comba', 1)
    other = copy.copy(first)
    other.no_layer_checkpoint = True
    assert training.checkpoint_contract(first, 'comba', 1) != training.checkpoint_contract(other, 'comba', 1)
    other = copy.copy(first)
    other.loss_backend = 'plain'
    assert training.checkpoint_contract(first, 'comba', 1) != training.checkpoint_contract(other, 'comba', 1)
    before = training.checkpoint_contract(first, 'comba', 1)
    monkeypatch.setenv('FLA_DISABLE_BACKEND_DISPATCH', '1')
    assert before != training.checkpoint_contract(first, 'comba', 1)


def test_plain_ce_configuration_and_native_category():
    args = benchmark.parse_arguments(['--methods', 'gdn', 'kda', 'comba', 'hdn_tc', '--loss-backend', 'plain'])
    for name in args.methods:
        config = benchmark.model_configuration(args, name)
        assert config['fuse_cross_entropy'] is False and config['fuse_linear_cross_entropy'] is False
        assert training.effective_loss_backend(args, name) == 'plain_cross_entropy'
    args.loss_backend = 'native'
    assert training.effective_loss_backend(args, 'kda').startswith('fused_linear_cross_entropy')
    assert training.effective_loss_backend(args, 'gdn') == 'fused_cross_entropy'


def test_checkpoint_rejects_tampered_file_and_actual_config_before_mutation(tmp_path):
    args = options()
    model = TinyLM()
    optimizer = training.make_optimizer(model, args)
    path = tmp_path / 'state.pt'
    training.save_checkpoint(path, model, optimizer, step=3, contract={}, rank=0)
    before = training.state_digest(model.state_dict())
    data = torch.load(path, weights_only=False)
    data['next_data_step'] = 4
    torch.save(data, path)
    with pytest.raises(ValueError, match='manifest'):
        training.load_checkpoint(path, model, optimizer, contract={}, rank=0, weight_decay=args.weight_decay)
    assert training.state_digest(model.state_dict()) == before
    model.config = {'actual': 1}
    training.save_checkpoint(path, model, optimizer, step=3, contract={}, rank=0)
    model.config = {'actual': 2}
    with pytest.raises(ValueError, match='actual model config mismatch'):
        training.load_checkpoint(path, model, optimizer, contract={}, rank=0, weight_decay=args.weight_decay)


def checkpoint_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', world_size=2, rank=rank,
                            timeout=timedelta(seconds=20))
    try:
        args = SimpleNamespace(dtype='fp32', data_seed=43, data_layout='packed', lr=1e-4, weight_decay=0.1, clip_grad=1.0)
        contract = {'world_size': 2, 'recipe': 'actual Gloo checkpoint negative tests'}
        torch.manual_seed(19)
        model = DistributedDataParallel(TinyLM())
        optimizer = training.make_optimizer(model, args)
        root = Path(directory) / f'rank{rank}'
        first, second = root / 'first/state.pt', root / 'second/state.pt'
        batch = training.make_batch(args, 3, small=True)
        training.backward_step(model, [batch], args, batch.valid_labels * 2, 2)
        training.clip_and_step(model, optimizer, args)
        training.save_checkpoint(first, model, optimizer, step=3, contract=contract, rank=rank)
        optimizer.zero_grad(set_to_none=True)
        training.backward_step(model, [batch], args, batch.valid_labels * 2, 2)
        training.clip_and_step(model, optimizer, args)
        torch.manual_seed(100 + rank)
        training.save_checkpoint(second, model, optimizer, step=4, contract=contract, rank=rank)
        second_bytes = second.read_bytes()
        rows = []

        def reject(label, path):
            before = training.state_digest({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()})
            rng_before = torch.get_rng_state().clone()
            try:
                training.load_checkpoint(path, model, optimizer, contract=contract, rank=rank,
                                         weight_decay=args.weight_decay)
            except ValueError as error:
                rows.append({'case': label, 'collectively_rejected': True, 'error': str(error)})
            else:
                raise AssertionError(f'{label} accepted on rank {rank}')
            assert before == training.state_digest({'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()})
            assert torch.equal(rng_before, torch.get_rng_state())

        reject('individually_valid_mixed_generations', first if rank == 0 else second)
        partial = root / 'partial/state.pt'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(second_bytes)
        reject('missing_completion_manifest', second if rank == 0 else partial)
        if rank == 1:
            second.write_bytes(first.read_bytes())
        reject('stale_rank_file_under_current_manifest', second)
        second.write_bytes(second_bytes)
        # same positions and parameters, but divergent Adam moments, cannot publish a generation
        if rank == 1:
            next(iter(optimizer.state.values()))['exp_avg'].add_(1)
        bad = root / 'divergent/state.pt'
        try:
            training.save_checkpoint(bad, model, optimizer, step=4, contract=contract, rank=rank)
        except ValueError as error:
            rows.append({'case': 'different_optimizer_moments_at_save', 'collectively_rejected': True, 'error': str(error)})
        else:
            raise AssertionError('divergent optimizer states published')
        assert not training.checkpoint_manifest_path(bad).exists()
        assert training.load_checkpoint(second, model, optimizer, contract=contract, rank=rank,
                                        weight_decay=args.weight_decay) == (4, 4)
        saved = torch.load(second, weights_only=False)
        assert torch.equal(saved['rng']['torch'], torch.get_rng_state())
        assert training.state_digest(training.shared_checkpoint_state(saved)) == training.state_digest({
            **training.shared_checkpoint_state(saved), 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()})
        rows.append({'case': 'coherent_generation', 'accepted': True, 'rank_specific_rng_restored': True})
        Path(directory, f'checkpoint-rank{rank}.json').write_text(json.dumps(rows, indent=2))
    finally:
        dist.destroy_process_group()


def test_real_gloo_collectively_rejects_partial_stale_and_mixed_generations(tmp_path):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc-per-node=2',
               '--master-addr=127.0.0.1', f'--master-port={port}', str(Path(__file__).resolve()),
               '--checkpoint-worker', str(tmp_path / 'rendezvous'), str(tmp_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                               env={**os.environ, 'OMP_NUM_THREADS': '1', 'FLA_CI_ENV': '0'})
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f'owned checkpoint Gloo test timed out: {stdout} {stderr}')
    assert process.returncode == 0, stdout + stderr
    for rank in range(2):
        rows = json.loads((tmp_path / f'checkpoint-rank{rank}.json').read_text())
        assert len(rows) == 5
        assert all(row.get('collectively_rejected', row.get('accepted')) for row in rows)


@pytest.fixture
def archived_source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    contents = {'benchmarks/hyperbolic_benchmark_utils.py': Path(benchmark_utils.__file__).read_bytes(),
                'fla/example.py': b'VALUE = 7\n', 'run.sh': b'#!/bin/sh\nexit 0\n'}
    modes = {name: 0o755 if name.endswith('.sh') else 0o644 for name in contents}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for name, content in contents.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(content), modes[name]
            tar.addfile(entry, io.BytesIO(content))
        link = tarfile.TarInfo('source_alias')
        link.type, link.linkname = tarfile.SYMTYPE, 'fla/example.py'
        tar.addfile(link)
    archive.seek(0)
    with tarfile.open(fileobj=archive) as tar:
        tar.extractall(root, filter='data')
    files = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    files['source_alias'], modes['source_alias'] = 'symlink:fla/example.py', 0o777
    identity = {'base_commit': 'a' * 40, 'files': files, 'modes': modes}
    manifest = {**identity, 'snapshot_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                'selected_working_files': {'benchmarks/hyperbolic_benchmark_utils.py': files['benchmarks/hyperbolic_benchmark_utils.py']}}
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(manifest))
    return root, path, manifest


def test_metadata_from_real_extracted_archive_without_git(archived_source):
    root, path, manifest = archived_source
    assert not (root / '.git').exists()
    for name, mode in manifest['modes'].items():
        if not (root / name).is_symlink():
            (root / name).chmod(mode & ~0o222)
    command = [sys.executable, '-c', 'import json; from types import SimpleNamespace; '
               'from benchmarks.hyperbolic_benchmark_utils import metadata; print(json.dumps(metadata(SimpleNamespace())))']
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=20,
                               env={**os.environ, 'PYTHONPATH': str(root), 'FLA_SOURCE_MANIFEST': str(path),
                                    'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result['revision'] is None and result['working_tree_dirty'] is None
    assert result['source_kind'] == 'immutable_source_manifest'
    assert result['declared_base_commit'] == 'a' * 40
    assert result['source_tree_sha256'] == manifest['snapshot_sha256'] and result['declared_file_count'] == 4


@pytest.mark.parametrize('tamper', ['contents', 'missing', 'mode', 'extra', 'manifest_identity', 'base_commit'])
def test_source_manifest_rejects_tampering(archived_source, tamper):
    root, path, manifest = archived_source
    if tamper == 'contents':
        (root / 'fla/example.py').write_text('VALUE = 8\n')
    elif tamper == 'missing':
        (root / 'fla/example.py').unlink()
    elif tamper == 'mode':
        (root / 'fla/example.py').chmod(0o755)
    elif tamper == 'extra':
        (root / 'fla/undeclared.py').write_text('VALUE = 9\n')
    else:
        manifest['snapshot_sha256' if tamper == 'manifest_identity' else 'base_commit'] = 'b' * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='mismatch|missing'):
        benchmark_utils.validated_source_snapshot(root, path)


def test_git_metadata_and_unmanifested_archive_are_distinct(monkeypatch, archived_source):
    monkeypatch.delenv('FLA_SOURCE_MANIFEST', raising=False)
    metadata = benchmark_utils.source_metadata(training.ROOT)
    assert metadata['source_kind'] == 'git_checkout' and len(metadata['revision']) == 40
    with pytest.raises(ValueError, match='FLA_SOURCE_MANIFEST'):
        benchmark_utils.source_metadata(archived_source[0])


@pytest.mark.parametrize('failure_stage', ['measurement', 'checkpoint_save', 'checkpoint_reload', 'profile_export'])
def test_active_trial_survives_sample_checkpoint_and_profile_failures(monkeypatch, tmp_path, failure_stage):
    class CPURuntime:
        rank = local_rank = 0
        world_size = 1
        device = torch.device('cpu')

        def __init__(self, args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            pass

        def wrap(self, model):
            return model

        def gather(self, value):
            return [value]

    args = benchmark.parse_arguments(['--methods', 'comba', '--dtype', 'fp32', '--warmup', '1', '--steps', '2',
                                      '--stability-steps', '0', '--output', str(tmp_path / 'failure.json')])
    original_batch = training.make_batch
    calls = []

    def build(args, spec, small=False):
        model = TinyLM()
        return model, {'parameter_count': sum(p.numel() for p in model.parameters()),
                       'initial_weights_sha256': benchmark.weight_digest(model), 'configuration': {}}

    def sample(model, optimizer, batches, args, runtime, mode):
        calls.append([batch.sample_id for batch in batches])
        if failure_stage == 'measurement' and len(calls) == 3:
            raise RuntimeError('INJECTED measurement failure')
        labels = sum(batch.valid_labels for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        loss = training.backward_step(model, batches, args, labels, 1)
        training.clip_and_step(model, optimizer, args)
        return {'cpu_control_flow_only': True, 'wall_ms': 1., 'cuda_ms': 1., 'total_wall_ms': 1., 'total_cuda_ms': 1.,
                'fwd_cuda_ms': .2, 'bwd_cuda_ms': .6, 'optimizer_cuda_ms': .2, 'global_valid_labels': labels,
                'weighted_local_loss': float(loss), 'input_slots': sum(b.input_slots for b in batches),
                'nonpadding_tokens': sum(b.nonpadding_tokens for b in batches),
                'training_memory': {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0},
                'samples': [{'id': b.sample_id, 'sha256': b.sha256, 'valid_labels': b.valid_labels} for b in batches]}

    def fail(*args, **kwargs):
        raise RuntimeError('INJECTED ' + failure_stage)

    monkeypatch.setattr(benchmark, 'build_model', build)
    monkeypatch.setattr(training, 'Runtime', CPURuntime)
    monkeypatch.setattr(training, 'metadata', lambda args: {'source_tree_sha256': 'CPU control flow only'})
    monkeypatch.setattr(training, 'small_gate', lambda *args: {'cpu_control_flow_only': True})
    monkeypatch.setattr(training, 'trace_route', lambda *args: {'cpu_control_flow_only': True})
    monkeypatch.setattr(training, 'recipe_parameter_counts',
                        lambda *args: {'total': sum(p.numel() for p in TinyLM().parameters())})
    monkeypatch.setattr(training, 'make_batch', lambda args, index: original_batch(args, index, small=True))
    monkeypatch.setattr(training, 'timed_sample', sample)
    monkeypatch.setattr(training, 'memory_snapshot', lambda: {'allocated_bytes': 0, 'reserved_bytes': 0})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    if failure_stage in ('checkpoint_save', 'checkpoint_reload'):
        monkeypatch.setattr(training, 'save_checkpoint' if failure_stage == 'checkpoint_save' else 'load_checkpoint', fail)
    if failure_stage == 'profile_export':
        args.profile = tmp_path / 'profile'

        class Profiler:
            def __enter__(self):
                return self

            def __exit__(self, *exception):
                pass

            export_chrome_trace = fail
        monkeypatch.setattr(torch.profiler, 'profile', lambda **kwargs: Profiler())
    payload = {'training_contract': {}}
    with pytest.raises(RuntimeError, match='INJECTED'):
        training.run_benchmark(args, payload, benchmark)
    saved = json.loads(training.rank_path(args.output, 0).read_text())
    active = saved['measurements']['comba']['trials'][0]
    expected_samples = 1 if failure_stage == 'measurement' else 2
    assert len(active['modes']['step']['warmup']) == 1
    assert len(active['modes']['step']['raw_samples']) == expected_samples
    assert active['status'].startswith('FAILED') and active['position']['stage'] == failure_stage
    assert active['position']['global_optimizer_step'] == (5 if failure_stage == 'profile_export' else expected_samples + 1)
    journal = json.loads(Path(active['journal']).read_text())
    assert journal['modes']['step']['raw_samples'] == active['modes']['step']['raw_samples']
    assert journal['position'] == active['position']


if __name__ == '__main__' and '--worker' in sys.argv:
    index = sys.argv.index('--worker')
    distributed_worker(int(os.environ['RANK']), sys.argv[index + 1], sys.argv[index + 2])

if __name__ == '__main__' and '--checkpoint-worker' in sys.argv:
    index = sys.argv.index('--checkpoint-worker')
    checkpoint_worker(int(os.environ['RANK']), sys.argv[index + 1], sys.argv[index + 2])
