# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Distributed training, deterministic work partition, and recovery for the HDN benchmark."""

import ast
import contextlib
import copy
import hashlib
import importlib
import json
import math
import os
import random
import re
import signal
import socket
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from benchmarks.hyperbolic_benchmark_utils import (
    KHDNExceptionGuard,
    KHDNTrace,
    environment,
    metadata,
    summarize,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_CLASSES = {
    'gdn': ('gated_deltanet', 'GatedDeltaNet'),
    'kda': ('kda', 'KDA'),
    'comba': ('comba', 'Comba'),
    'khdn': ('key_gated_hyperbolic_deltanet', 'KeyGatedHyperbolicDeltaNet'),
    'hdn': ('hyperbolic_deltanet', 'HyperbolicDeltaNet'),
}


def audit_methods():
    """Audit source availability without importing accelerator dependencies or promising execution."""
    from benchmarks.benchmark_hyperbolic_training import EXTRA_METHOD_NAMES, METHOD_NAMES
    source = ROOT / 'fla/ops/key_gated_hyperbolic_delta_rule/__init__.py'
    tree = ast.parse(source.read_text())
    exports = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    optimized = 'chunk_key_gated_hyperbolic_delta_rule' in exports
    result = {}
    for name in METHOD_NAMES + EXTRA_METHOD_NAMES:
        available, reason = True, 'Native source present; actual CUDA forward/backward route must be attested at runtime.'
        if name.startswith('khdn_') and name != 'khdn_naive':
            available = optimized
            reason = ('Strict FP64 SIMT source present; explicit opt-in and actual operator/autograd/compiled-launch gates required.'
                      if optimized else 'optimized chunk_key_gated_hyperbolic_delta_rule is absent; no naive fallback is permitted')
        if name == 'gdn2':
            available = False
            reason = 'GDN2 layer and ops exist, but no native GDN2 CausalLM/config exists; full-model baseline unavailable'
        result[name] = {'available': available, 'reason': reason}
        if name in ('khdn_no_carry', 'khdn_y_carry'):
            result[name]['optimized_symbol_exported'] = optimized
    result['gdn2'].update(operator_available=(ROOT / 'fla/ops/gdn2/chunk.py').is_file(),
                          layer_available=(ROOT / 'fla/layers/gdn2.py').is_file())
    return result


def model_class(name):
    key = 'khdn' if name.startswith('khdn_') else name if name in ('gdn', 'kda', 'comba') else 'hdn'
    module_name, prefix = MODEL_CLASSES[key]
    module = importlib.import_module(f'fla.models.{module_name}')
    return getattr(module, prefix + 'Config'), getattr(module, prefix + 'ForCausalLM')


def recipe_parameter_counts(args, name):
    """Exact parameter inventory for the matched dense recipe; runtime construction cross-checks it."""
    d, h, k, f, layers = args.hidden_size, args.heads, args.head_dim, args.intermediate_size, args.layers
    per_layer = {'qkvo': 4 * d * d, 'output_gate': d * d, 'beta': d * h,
                 'short_conv': 0 if args.no_short_conv else 3 * d * 4,
                 'output_norm': k, 'block_norms': 2 * d, 'ffn': 3 * d * f}
    if name.startswith('khdn_'):
        per_layer.update(channel_gate_projection=2 * d * k, A_log=h, dt_bias=d,
                         curvature=int(args.learnable_curvature))
    elif name == 'kda':
        per_layer.update(output_gate=2 * d * k + d, channel_gate_projection=2 * d * k, A_log=h, dt_bias=d)
    elif name in ('gdn', 'comba'):
        per_layer.update(decay_projection=d * h, A_log=h, dt_bias=h)
        if name == 'comba':
            per_layer.update(inner_decay=h, correction=h)
    else:
        per_layer['curvature'] = int(args.learnable_curvature)
    outside = {'embedding_and_untied_lm_head': 2 * args.vocab_size * d, 'final_norm': d}
    return {'total': layers * sum(per_layer.values()) + sum(outside.values()), 'per_layer': per_layer,
            'outside_layers': outside, 'layers': layers,
            'method': 'source-audited exact parameter inventory; runtime model count must match'}


def parameter_counts(args):
    from benchmarks.benchmark_hyperbolic_training import model_configuration
    result = {}
    for name in args.methods:
        counted = recipe_parameter_counts(args, name)
        try:
            config_class, causal_class = model_class(name)
        except ModuleNotFoundError as error:
            if error.name != 'triton':
                raise
            counted['constructor_verification'] = 'NOT RUN: local Triton dependency absent'
        else:
            with torch.device('meta'):
                model = causal_class(config_class(**model_configuration(args, name)))
            actual = sum(p.numel() for p in model.parameters())
            if counted['total'] != actual:
                raise AssertionError(f'{name}: source inventory {counted["total"]} differs from actual {actual}')
            counted['constructor_verification'] = 'actual full model constructor on meta matched'
        result[name] = counted
    return result


def scaling_plan(args, world_size):
    if world_size not in (1, 8, 16):
        raise ValueError('benchmark world size must be 1, 8, or 16')
    if args.scaling == 'single' and world_size != 1:
        raise ValueError('torchrun with multiple ranks requires explicit --scaling strong or weak')
    if args.scaling != 'single' and args.batch * args.seq_len != 16384:
        raise ValueError('strong/weak scaling requires batch*seq-len=16384 input slots per microstep')
    if args.scaling != 'single' and args.accumulation is not None:
        raise ValueError('strong/weak accumulation is fixed; omit --accumulation')
    if args.accumulation is not None and args.accumulation < 1:
        raise ValueError('accumulation must be positive')
    accumulation = 16 // world_size if args.scaling == 'strong' else (args.accumulation or 1)
    if args.scaling == 'weak':
        accumulation = 1
    if accumulation < 1:
        raise ValueError('accumulation must be positive')
    return {'mode': args.scaling, 'world_size': world_size, 'accumulation': accumulation,
            'microstep_input_slots_per_rank': args.batch * args.seq_len,
            'global_input_slots_per_step': args.batch * args.seq_len * accumulation * world_size,
            'global_microbatches': accumulation * world_size}


def sample_indices(step, rank, world_size, accumulation):
    return [step * world_size * accumulation + micro * world_size + rank for micro in range(accumulation)]


@dataclass
class Batch:
    inputs: dict
    valid_labels: int
    input_slots: int
    nonpadding_tokens: int
    sample_id: int
    sha256: str

    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.inputs.items()}, self.valid_labels,
                     self.input_slots, self.nonpadding_tokens, self.sample_id, self.sha256)


def batch_digest(inputs):
    digest = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        digest.update(key.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def make_batch(args, sample_id, *, small=False):
    batch, length, vocab = (2, 7, 97) if small else (args.batch, args.seq_len, args.vocab_size)
    generator = torch.Generator().manual_seed((args.data_seed + sample_id * 1000003) % (2**63 - 1))
    tokens = torch.randint(3, vocab, (batch, length), generator=generator)
    labels = tokens.clone()
    inputs = {'input_ids': tokens, 'labels': labels}
    nonpadding = tokens.numel()
    if args.data_layout == 'padding':
        lengths = torch.randint(1, length + 1, (batch,), generator=generator)
        mask = torch.arange(length)[None, :] < lengths[:, None]
        labels.masked_fill_(~mask, -100)
        tokens.masked_fill_(~mask, 0)
        inputs['attention_mask'] = mask.long()
        nonpadding = int(mask.sum())
    if args.data_layout == 'packed':
        tokens, labels = tokens.reshape(1, -1), labels.reshape(1, -1)
        boundaries = [0]
        for row in range(batch):
            split = int(torch.randint(1, length, (), generator=generator))
            boundaries.extend((row * length + split, (row + 1) * length))
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
        labels[0, cu_seqlens[1:-1].long()] = -100
        inputs = {'input_ids': tokens, 'labels': labels, 'cu_seqlens': cu_seqlens}
    valid = int(labels[:, 1:].ne(-100).sum())
    return Batch(inputs, valid, tokens.numel(), nonpadding, sample_id, batch_digest(inputs))


class Runtime:
    """Own only the process group created here; launcher owns peer termination."""

    def __init__(self, args):
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get('LOCAL_RANK', '0'))
        self.device = torch.device('cuda', self.local_rank)
        self.args, self.owns_group = args, False

    def __enter__(self):
        if not 0 <= self.rank < self.world_size or not 0 <= self.local_rank < torch.cuda.device_count():
            raise ValueError('invalid global rank or local CUDA rank')
        torch.cuda.set_device(self.device)
        self.previous_alarm = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._timeout)
        signal.alarm(self.args.task_timeout)
        try:
            if self.world_size > 1 or ('RANK' in os.environ and self.args.scaling != 'single'):
                if dist.is_initialized():
                    raise RuntimeError('refuse to reuse an externally owned process group')
                dist.init_process_group(backend='nccl', timeout=timedelta(seconds=self.args.ddp_timeout),
                                        rank=self.rank, world_size=self.world_size, device_id=self.device)
                self.owns_group = True
        except BaseException:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous_alarm)
            raise
        return self

    def _timeout(self, signum, frame):
        raise TimeoutError(f'task timeout for pid={os.getpid()} rank={self.rank}')

    def __exit__(self, *exception):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous_alarm)
        if self.owns_group:
            dist.destroy_process_group()

    def wrap(self, model):
        if not self.owns_group:
            return model
        return DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank,
                                       broadcast_buffers=False, find_unused_parameters=False)

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        output = [None] * self.world_size
        dist.all_gather_object(output, value)
        return output

    def align(self):
        if self.world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(self.device)

    def global_count(self, batches):
        count = torch.tensor(sum(batch.valid_labels for batch in batches), device=self.device, dtype=torch.int64)
        if self.world_size > 1:
            dist.all_reduce(count)
        total = int(count.item())
        if total == 0:
            raise ValueError('global optimizer step has no valid shifted labels')
        return total


def copy_model(model):
    copied = copy.deepcopy(model)
    original = dict(model.named_parameters())
    for name, parameter in copied.named_parameters():
        if getattr(original[name], '_no_weight_decay', False):
            parameter._no_weight_decay = True
    return copied


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def optimizer_groups(model, weight_decay):
    groups = {False: [], True: []}
    names = {False: [], True: []}
    for name, parameter in unwrap(model).named_parameters():
        if parameter.requires_grad:
            exempt = bool(getattr(parameter, '_no_weight_decay', False))
            groups[exempt].append(parameter)
            names[exempt].append(name)
    return [{'params': groups[exempt], 'weight_decay': 0.0 if exempt else weight_decay,
             'param_names': names[exempt]} for exempt in (False, True) if groups[exempt]]


def make_optimizer(model, args):
    return torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95),
                             fused=next(model.parameters()).is_cuda)


def enforce_optimizer_policy(model, optimizer, weight_decay):
    expected = optimizer_groups(model, weight_decay)
    if len(expected) != len(optimizer.param_groups):
        raise ValueError('checkpoint optimizer parameter grouping differs')
    for group, policy in zip(optimizer.param_groups, expected):
        if len(group['params']) != len(policy['params']) or any(a is not b for a, b in zip(group['params'], policy['params'])):
            raise ValueError('checkpoint optimizer parameter order differs')
        if group.get('param_names') != policy['param_names']:
            raise ValueError('checkpoint optimizer parameter names differ')
        group['weight_decay'] = policy['weight_decay']


def weighted_loss(model, batch, args, global_labels, world_size):
    device = batch.inputs['input_ids'].device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and args.dtype == 'bf16'):
        # summed reduction divided by N/W compensates DDP rank averaging without dividing by accumulation
        return model(**batch.inputs, use_cache=False, num_items_in_batch=global_labels / world_size).loss


def backward_step(model, batches, args, global_labels, world_size, phase_events=None):
    losses = []
    for micro, batch in enumerate(batches):
        sync = model.no_sync() if isinstance(model, DistributedDataParallel) and micro < len(batches) - 1 else contextlib.nullcontext()
        with sync:
            if phase_events is not None:
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
            loss = weighted_loss(model, batch, args, global_labels, world_size)
            if phase_events is not None:
                events[1].record()
            loss.backward()
            if phase_events is not None:
                events[2].record()
                phase_events.append(events)
        losses.append(loss.detach())
    return torch.stack(losses).sum()


def clip_and_step(model, optimizer, args):
    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
    optimizer.step()


def optimizer_health(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise AssertionError('Non-finite optimizer state')


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def effective_loss_backend(args, name):
    if args.loss_backend == 'plain':
        return 'plain_cross_entropy'
    return 'fused_linear_cross_entropy (native KDA forward override)' if name == 'kda' else 'fused_cross_entropy'


def checkpoint_contract(args, name, world_size):
    from benchmarks.benchmark_hyperbolic_training import method_spec, model_configuration
    policy = {key: value for key, value in os.environ.items()
              if (key.startswith(('FLA_', 'TRITON_', 'NCCL_', 'TORCH_NCCL_')) and key not in ('FLA_SOURCE_MANIFEST', 'TRITON_CACHE_DIR', 'TRITON_DUMP_DIR'))
              or key in ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'CUDA_DEVICE_MAX_CONNECTIONS')}
    policy.update(method_spec(name)['environment'])
    snapshot = None
    if os.environ.get('FLA_SOURCE_MANIFEST'):
        from benchmarks.hyperbolic_benchmark_utils import validated_source_snapshot
        snapshot = validated_source_snapshot(ROOT, os.environ['FLA_SOURCE_MANIFEST'])['snapshot_sha256']
    return {'schema': 2, 'method': name, 'config': model_configuration(args, name), 'world_size': world_size,
            'scaling': scaling_plan(args, world_size), 'data_seed': args.data_seed, 'layout': args.data_layout,
            'batch': args.batch, 'seq_len': args.seq_len, 'cursor_unit': 'absolute_global_microbatch_offset',
            'modes': args.modes, 'outer_checkpoint': not args.no_layer_checkpoint, 'use_reentrant': False,
            'parameter_dtype': 'fp32', 'dtype': args.dtype, 'loss_backend': effective_loss_backend(args, name),
            'backend_environment': policy, 'source_snapshot_sha256': snapshot, 'matmul_precision': torch.get_float32_matmul_precision(),
            'allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'torch_version': str(torch.__version__),
            'lr': args.lr, 'weight_decay': args.weight_decay, 'clip_grad': args.clip_grad,
            'memory_only': args.memory_only}


def state_digest(value):
    """Hash shared state by tensor contents, independently of torch.save storage identities."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(f'{type(item).__name__}:{item!r};'.encode())
    visit(value)
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def actual_model_config(model):
    config = getattr(unwrap(model), 'config', None)
    return config.to_dict() if hasattr(config, 'to_dict') else config


def validate_requested_config(config, contract):
    if config is not None and 'config' in contract:
        if any(config.get(key) != value for key, value in contract['config'].items()):
            raise ValueError('actual model config differs from requested execution config')


def checkpoint_manifest_path(path):
    path = Path(path)
    return path.with_name(re.sub(r'\.rank[0-9]+$', '', path.stem) + '.complete.json')


def checkpoint_collective(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def checkpoint_agreement(record):
    records = checkpoint_collective(record)
    failures = [row.get('error') for row in records if row.get('error')]
    if failures:
        raise ValueError('collective checkpoint rejection: ' + '; '.join(failures))
    return records


def shared_checkpoint_state(state):
    return {key: state[key] for key in ('model', 'optimizer', 'scheduler', 'contract', 'config',
                                        'next_data_step', 'global_optimizer_step')}


def save_checkpoint(path, model, optimizer, *, step, contract, rank, scheduler=None, optimizer_step=None):
    """Publish completion only after every rank agrees on shared state and durable rank files."""
    from benchmarks.benchmark_hyperbolic_training import parameter_health
    generation = checkpoint_collective(uuid.uuid4().hex)[0]
    path = Path(path)
    record = {'rank': rank}
    try:
        if dist.is_initialized() and rank != dist.get_rank():
            raise ValueError('rank argument differs from process group')
        parameter_health(unwrap(model))
        optimizer_health(optimizer)
        state = {'model': unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict() if scheduler is not None else None,
                 'rng': capture_rng(), 'next_data_step': step,
                 'global_optimizer_step': step if optimizer_step is None else optimizer_step,
                 'contract': contract, 'rank': rank, 'config': actual_model_config(model), 'generation': generation}
        validate_requested_config(state['config'], contract)
        shared_digest = state_digest(shared_checkpoint_state(state))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.tmp.{generation}.{rank}')
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        record.update(file=path.name, sha256=file_digest(path), shared_state_sha256=shared_digest,
                      next_data_step=step, global_optimizer_step=state['global_optimizer_step'], generation=generation)
    except Exception as error:
        record['error'] = f'rank {rank}: {type(error).__name__}: {error}'
    records = checkpoint_agreement(record)
    if contract.get('world_size', len(records)) != len(records):
        raise ValueError('collective checkpoint rejection: contract world size differs from process group')
    identity = ('shared_state_sha256', 'next_data_step', 'global_optimizer_step', 'generation')
    if sorted(row['rank'] for row in records) != list(range(len(records))):
        raise ValueError('collective checkpoint rejection: incomplete rank set')
    if any(any(row[key] != records[0][key] for key in identity) for row in records):
        raise ValueError('collective checkpoint rejection: shared state or positions disagree at save')
    manifest = {'schema': 2, 'generation': generation, 'world_size': len(records),
                'shared_state_sha256': records[0]['shared_state_sha256'], 'ranks': records}
    publication = {'rank': rank}
    try:
        # identical manifests are replicated on node-local storage as well as shared filesystems
        write_json(checkpoint_manifest_path(path), manifest)
    except Exception as error:
        publication['error'] = f'rank {rank}: manifest publication: {error}'
    checkpoint_agreement(publication)
    return manifest


def load_checkpoint(path, model, optimizer, *, contract, rank, weight_decay, scheduler=None):
    """Validate every rank's completed generation before mutating any model, optimizer, or RNG."""
    path = Path(path)
    record = {'rank': rank}
    state = None
    try:
        if dist.is_initialized() and rank != dist.get_rank():
            raise ValueError('rank argument differs from process group')
        manifest = json.loads(checkpoint_manifest_path(path).read_text())
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if manifest['schema'] != 2 or manifest['world_size'] != world_size:
            raise ValueError('manifest schema/world mismatch')
        if sorted(row['rank'] for row in manifest['ranks']) != list(range(world_size)):
            raise ValueError('manifest rank set is incomplete')
        entry = next(row for row in manifest['ranks'] if row['rank'] == rank)
        if entry['file'] != path.name or entry['sha256'] != file_digest(path):
            raise ValueError('rank file is stale, partial or does not match the completion manifest')
        # locally produced trusted checkpoints contain Python/NumPy RNG state
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['contract'] != contract or state['rank'] != rank:
            raise ValueError('checkpoint recipe, partition, or rank mismatch')
        if state['config'] != actual_model_config(model):
            raise ValueError('actual model config mismatch')
        if (state['scheduler'] is None) != (scheduler is None):
            raise ValueError('checkpoint scheduler mismatch')
        if state['generation'] != manifest['generation'] or state['generation'] != entry['generation']:
            raise ValueError('checkpoint generation mismatch')
        validate_requested_config(state['config'], contract)
        shared_digest = state_digest(shared_checkpoint_state(state))
        if shared_digest != manifest['shared_state_sha256'] or shared_digest != entry['shared_state_sha256']:
            raise ValueError('checkpoint shared model/optimizer/position identity mismatch')
        if any(state[key] != entry[key] for key in ('next_data_step', 'global_optimizer_step')):
            raise ValueError('checkpoint position mismatch')
        record.update(generation=state['generation'], manifest_sha256=state_digest(manifest),
                      shared_state_sha256=shared_digest, next_data_step=state['next_data_step'],
                      global_optimizer_step=state['global_optimizer_step'])
    except Exception as error:
        record['error'] = f'rank {rank}: {type(error).__name__}: {error}'
    records = checkpoint_agreement(record)
    identity = ('generation', 'manifest_sha256', 'shared_state_sha256', 'next_data_step', 'global_optimizer_step')
    if sorted(row['rank'] for row in records) != list(range(len(records))):
        raise ValueError('collective checkpoint rejection: process ranks disagree')
    if any(any(row[key] != records[0][key] for key in identity) for row in records):
        raise ValueError('collective checkpoint rejection: mixed generation, shared state or positions')
    applied = {'rank': rank}
    try:
        unwrap(model).load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        enforce_optimizer_policy(model, optimizer, weight_decay)
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        optimizer_health(optimizer)
        from benchmarks.benchmark_hyperbolic_training import parameter_health
        parameter_health(unwrap(model))
    except Exception as error:
        applied['error'] = f'rank {rank}: state restore: {error}'
    checkpoint_agreement(applied)
    return state['next_data_step'], state['global_optimizer_step']


def rank_path(path, rank):
    path = Path(path)
    return path.with_name(f'{path.stem}.rank{rank}{path.suffix}')


def timed_sample(model, optimizer, batches, args, runtime, mode):
    """Fresh backward graph per sample; data/counts/zero_grad/reporting stay outside the window."""
    from benchmarks.benchmark_hyperbolic_training import gradient_health, parameter_health
    optimizer.zero_grad(set_to_none=True)
    total = runtime.global_count(batches)
    torch.cuda.reset_peak_memory_stats()
    prepared = None
    if mode == 'bwd':
        if len(batches) != 1:
            raise ValueError('backward-only requires one microbatch to avoid retaining multiple graphs')
        prepared = weighted_loss(model, batches[0], args, total, runtime.world_size)
    runtime.align()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    phase_events = []
    optimizer_start = torch.cuda.Event(enable_timing=True)
    events[0].record()
    started = time.perf_counter()
    if mode == 'bwd':
        loss = prepared
        loss.backward()
    elif mode == 'fwd':
        if len(batches) != 1:
            raise ValueError('forward-only requires one microbatch per sample')
        loss = weighted_loss(unwrap(model), batches[0], args, total, runtime.world_size)
    else:
        loss = backward_step(model, batches, args, total, runtime.world_size, phase_events)
        optimizer_start.record()
        if mode == 'step':
            clip_and_step(model, optimizer, args)
    events[1].record()
    torch.cuda.synchronize(runtime.device)
    wall_ms = (time.perf_counter() - started) * 1000
    cuda_ms = events[0].elapsed_time(events[1])
    training_peaks = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    value = float(loss.detach())
    if not math.isfinite(value):
        raise AssertionError('Non-finite training loss')
    health = None
    if not args.memory_only:
        health = gradient_health(model) if mode != 'fwd' else None
        parameter_health(unwrap(model))
        optimizer_health(optimizer)
    phase = {'fwd_cuda_ms': sum(e[0].elapsed_time(e[1]) for e in phase_events),
             'bwd_cuda_ms': sum(e[1].elapsed_time(e[2]) for e in phase_events),
             'optimizer_cuda_ms': optimizer_start.elapsed_time(events[1]) if mode == 'step' else 0.0}
    return {**phase, 'training_memory': training_peaks, 'total_wall_ms': wall_ms, 'total_cuda_ms': cuda_ms, 'loss': value,
            'wall_ms': wall_ms, 'cuda_ms': cuda_ms, 'weighted_local_loss': value, 'gradient_l2': health,
            'global_valid_labels': total, 'local_valid_labels': sum(b.valid_labels for b in batches),
            'input_slots': sum(b.input_slots for b in batches), 'nonpadding_tokens': sum(b.nonpadding_tokens for b in batches),
            'samples': [{'id': b.sample_id, 'sha256': b.sha256, 'valid_labels': b.valid_labels} for b in batches]}


def memory_snapshot():
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()}


def memory_peaks(baseline):
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    return {'baseline': baseline, 'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
            'extra_peak_allocated_bytes': max(0, allocated - baseline['allocated_bytes']),
            'extra_peak_reserved_bytes': max(0, reserved - baseline['reserved_bytes'])}


class NativeTrainingTrace:
    """Wrap actual native forward/backward entries, including autograd worker threads."""

    def __init__(self, name):
        self.name, self.saved, self.calls = name, [], {}

    def __enter__(self):
        import functools
        import inspect
        import sys
        prefix = f'fla.ops.{self.name}.'
        required = {f'chunk_{self.name}_fwd', f'chunk_{self.name}_bwd'}
        wrappers = {}
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith(prefix) or module is None:
                continue
            for key, value in list(vars(module).items()):
                if key not in required or not inspect.isfunction(value):
                    continue
                if id(value) not in wrappers:
                    qualified = f'{value.__module__}.{value.__name__}'

                    def wrap(function, qualified):
                        @functools.wraps(function)
                        def observed(*args, **kwargs):
                            self.calls[qualified] = self.calls.get(qualified, 0) + 1
                            return function(*args, **kwargs)
                        return observed
                    wrappers[id(value)] = wrap(value, qualified)
                self.saved.append((module, key, value))
                setattr(module, key, wrappers[id(value)])
        return self

    def __exit__(self, *exception):
        for module, key, value in reversed(self.saved):
            setattr(module, key, value)

    def attest(self):
        for suffix in ('fwd', 'bwd'):
            if not any(key.endswith(f'chunk_{self.name}_{suffix}') for key in self.calls):
                raise AssertionError(f'{self.name}: native training {suffix} was not observed')
        return {'executed_backend_calls': self.calls,
                'interpretation': 'Native entry calls; inner Triton/TileLang kernel attribution requires separate profiling.'}


@contextlib.contextmanager
def observe_key_gates(model, name):
    observations, hooks = {}, []
    if name.startswith('khdn_'):
        for module_name, attention in model.named_modules():
            if not hasattr(attention, 'f_proj') or not hasattr(attention, 'A_log'):
                continue

            def capture(module, inputs, output, attention=attention, key=module_name):
                raw = (output.detach().float() + attention.dt_bias.detach()
                       ).reshape(-1, attention.num_heads, attention.head_dim)
                decay = -attention.A_log.detach().exp()[None, :, None] * torch.nn.functional.softplus(raw)
                minimum, maximum = float(decay.min()), float(decay.max())
                curvature = (float(torch.nn.functional.softplus(attention.curvature_raw.detach()))
                             if attention.learnable_curvature else attention.curvature)
                if not torch.isfinite(decay).all() or maximum >= 0 or curvature <= 0:
                    raise AssertionError('KHDN gate must be finite, nonidentity and have positive curvature')
                observations[key] = {'log_decay_min': minimum, 'log_decay_max': maximum,
                                     'group_size': attention.group_size, 'curvature': curvature,
                                     'gate_projection_trainable': all(p.requires_grad for p in attention.f_proj.parameters())}
            hooks.append(attention.f_proj.register_forward_hook(capture))
    try:
        yield observations
    finally:
        for hook in hooks:
            hook.remove()


def khdn_batch_layout(inputs):
    """Derive the exact consumed token domain using the layer's explicit-packed-before-mask convention."""
    tokens = inputs['input_ids']
    if tokens.ndim != 2:
        raise AssertionError('KHDN input batch shape must be [B,T]')
    batch, length = tokens.shape
    cu, mask = inputs.get('cu_seqlens'), inputs.get('attention_mask')
    if cu is not None:
        if batch != 1 or cu.ndim != 1 or cu.numel() < 2 or cu.dtype not in (torch.int32, torch.int64):
            raise AssertionError('KHDN packed batch has invalid sequence boundaries')
        bounds = cu.detach().cpu().tolist()
        if bounds[0] != 0 or bounds[-1] != length or any(a > b for a, b in zip(bounds, bounds[1:])):
            raise AssertionError('KHDN packed sequence boundaries do not cover the actual batch')
        shape, layout = [batch, length], 'explicit_packed'
    elif mask is not None:
        if mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] < length:
            raise AssertionError('KHDN padding mask shape differs from actual batch')
        valid = mask[:, -length:]
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise AssertionError('KHDN padding mask must identify binary valid token positions')
        counts = valid.to(torch.int64).sum(-1).detach().cpu()
        bounds = [0, *counts.cumsum(0).tolist()]
        shape, layout = [1, bounds[-1]], 'unpadded'
    else:
        shape, bounds, layout = [batch, length], [i * length for i in range(batch + 1)], 'dense'
    return {'layout': layout, 'input_batch_shape': [batch, length], 'operator_batch_time': shape,
            'consumed_tokens': shape[0] * shape[1], 'sequence_bounds': bounds}


def attest_khdn_batch(calls, expected):
    for row in calls:
        if row.get('sequence_bounds') != expected['sequence_bounds']:
            raise AssertionError('KHDN observed sequence boundaries differ from actual batch token domain')
        for key in ('q', 'k', 'v', 'beta', 'log_decay'):
            shape = row.get('sources', {}).get(key, {}).get('shape', [])
            if shape[:2] != expected['operator_batch_time']:
                raise AssertionError(f'KHDN observed {key} B/T shape differs from actual batch token domain')
        executed = row.get('executed_sources', [])
        if len(executed) != 7 or any(value.get('shape', [])[:2] != expected['operator_batch_time'] for value in executed[:5]):
            raise AssertionError('KHDN executed source B/T shape differs from actual batch token domain')
    return expected


def trace_route(model, batch, args, name, benchmark):
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    strict = name in ('khdn_no_carry', 'khdn_y_carry')
    context = (KHDNTrace(args.chunk_size, args.group_size, 'y_carry' if name == 'khdn_y_carry' else 'no_carry',
                         model.config.num_hidden_layers, precision_policy=args.khdn_precision_policy or 'strict') if strict else
               NativeTrainingTrace(name) if name in ('kda', 'comba') else
               benchmark.GDNBackendTrace() if name == 'gdn' else
               contextlib.nullcontext() if name == 'khdn_naive' else benchmark.KernelTrace())
    if strict:
        expected_layout = khdn_batch_layout(batch.inputs)
        context.bind_model(model)
    with context as trace, observe_key_gates(model, name if not strict else '') as gates:
        loss = weighted_loss(model, batch, args, max(1, batch.valid_labels), 1)
        loss.backward()
        torch.cuda.synchronize()
    if strict:
        route = trace.attest()
        try:
            route['actual_token_domain'] = attest_khdn_batch(trace.calls, expected_layout)
        except AssertionError as error:
            error.khdn_evidence = trace.calls
            raise
        expected = (model.config.num_heads, model.config.head_dim)
        if any(tuple(row['sources']['q']['shape'][-2:]) != expected
               or row['sources']['v']['shape'][-1] != model.config.head_dim for row in trace.calls):
            raise AssertionError('KHDN observed head dimensions differ from the actual configured model')
        replay_layers = {row['layer'] for row in trace.calls
                         if 'recompute_fn' in row.get('checkpoint_frames', [])
                         or row.get('forward_interrupted') == '_StopRecomputationError'}
        if (not args.no_layer_checkpoint and replay_layers != set(trace.layer_names)
                or args.no_layer_checkpoint and replay_layers):
            raise AssertionError('KHDN observed outer checkpoint replay differs from requested per-layer policy')
        route['actual_model_config'] = model.config.to_dict()
        route['actual_batch_shape'] = list(batch.inputs['input_ids'].shape)
    elif name == 'khdn_naive':
        route = {'route': 'explicit naive recurrent; correctness only; GPU performance NOT ACCEPTED'}
    elif name in ('kda', 'comba'):
        route = trace.attest()
    else:
        route = benchmark.attest_training(name, trace.records())
    if not torch.isfinite(loss.detach()):
        raise AssertionError('Non-finite route-gate loss')
    route.update(loss=float(loss.detach()), gradient_l2=benchmark.gradient_health(model),
                 effective_loss_backend=effective_loss_backend(args, name))
    if name.startswith('khdn_') and not strict:
        if not gates:
            raise AssertionError('KHDN gate projection was not observed')
        route['key_gate_observations'] = gates
    route['untimed_trace_wall_ms'] = (time.perf_counter() - started) * 1000
    return route


def error_metrics(actual, reference, *, atol, rtol):
    if actual.keys() != reference.keys():
        raise AssertionError('parameter names differ in equivalence gate')
    result = {}
    for name in actual:
        first, second = actual[name], reference[name]
        if first is None or second is None or not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise AssertionError(f'missing or non-finite parameter/gradient: {name}')
        torch.testing.assert_close(first, second, atol=atol, rtol=rtol, msg=name)
        result[name] = {'max_abs': float((first - second).abs().max()),
                        'relative_rms': float((first - second).square().mean().sqrt()
                                              / second.square().mean().sqrt().clamp_min(1e-30))}
    return result


def gradients(model):
    return {name: p.grad.detach().clone() if p.grad is not None else None
            for name, p in unwrap(model).named_parameters() if p.requires_grad}


def parameters(model):
    return {name: p.detach().clone() for name, p in unwrap(model).named_parameters()}


@contextlib.contextmanager
def khdn_update_guard(args, name, stage, records):
    """Retain route rejection evidence for untimed numerical updates and recovery."""
    if name not in ('khdn_no_carry', 'khdn_y_carry'):
        yield
        return
    with KHDNExceptionGuard(precision_policy=args.khdn_precision_policy or 'strict') as guard:
        record = {'stage': stage, 'guard': guard.evidence()}
        records.append(record)
        try:
            yield
        except BaseException as error:
            record['failure'] = repr(error)
            error.khdn_evidence = {'update_route_guards': records,
                                   'operator_calls': getattr(error, 'khdn_evidence', None)}
            raise


def small_gate(args, name, runtime, benchmark):
    """Compare real serial accumulation and DDP gradients/updates before timing any full model."""
    spec = benchmark.method_spec(name)
    model, info = benchmark.build_model(args, spec, small=True)
    serial = copy_model(model)
    # exercise unequal labels and a zero-label final microstep while retaining the collective schedule
    gate_args = copy.copy(args)
    gate_args.data_layout = 'dense' if name.startswith('protected_') else 'packed'
    global_batches = [make_batch(gate_args, index, small=True) for index in range(16)]
    for index, batch in enumerate(global_batches):
        batch.inputs['labels'][:, -(index % 3 + 1):] = -100
        batch.valid_labels = int(batch.inputs['labels'][:, 1:].ne(-100).sum())
    global_batches[-1].inputs['labels'].fill_(-100)
    global_batches[-1].valid_labels = 0
    for batch in global_batches:
        batch.sha256 = batch_digest(batch.inputs)
    total = sum(batch.valid_labels for batch in global_batches)
    route_batch = global_batches[0].to(runtime.device)
    route = trace_route(model, route_batch, args, name, benchmark)
    oracle_parity = khdn_model_oracle_gate(args, name, benchmark) if name in ('khdn_no_carry', 'khdn_y_carry') else None
    model.zero_grad(set_to_none=True)
    ddp = runtime.wrap(model)
    local_batches = [global_batches[index].to(runtime.device)
                     for index in sample_indices(0, runtime.rank, runtime.world_size, 16 // runtime.world_size)]
    ddp_optimizer, serial_optimizer = make_optimizer(ddp, args), make_optimizer(serial, args)
    initial = parameters(model)
    update_guards = []
    with khdn_update_guard(args, name, 'accumulated_update', update_guards):
        backward_step(ddp, local_batches, args, total, runtime.world_size)
        for batch in global_batches:
            weighted_loss(serial, batch.to(runtime.device), args, total, 1).backward()
    errors = error_metrics(gradients(ddp), gradients(serial),
                           atol=benchmark.GRADIENT_TOLERANCE[0], rtol=benchmark.GRADIENT_TOLERANCE[1])
    clip_and_step(ddp, ddp_optimizer, args)
    clip_and_step(serial, serial_optimizer, args)
    final_ddp, final_serial = parameters(ddp), parameters(serial)
    updates = error_metrics({key: value - initial[key] for key, value in final_ddp.items()},
                            {key: value - initial[key] for key, value in final_serial.items()}, atol=2e-6, rtol=2e-2)
    benchmark.parameter_health(model)
    result = {'model': info, 'route': route, 'world_size': runtime.world_size, 'global_valid_labels': total,
              'all_parameter_gradient_errors': errors, 'all_parameter_update_errors': updates,
              'update_route_guards': update_guards,
              'zero_label_final_global_microbatch': 15,
              'global_samples': [{'id': b.sample_id, 'sha256': b.sha256, 'valid_labels': b.valid_labels}
                                 for b in global_batches],
              'scope': 'Actual model serial accumulation versus DDP update' if isinstance(ddp, DistributedDataParallel) else
                       'Actual single-device accumulation; DDP equivalence NOT RUN',
              'ddp_accepted': False}
    if name.startswith('khdn_'):
        result['strict_vs_explicit_oracle' if args.khdn_precision_policy != 'mixed_readout'
               else 'mixed_vs_explicit_strict_oracle'] = oracle_parity
        gates = {}
        for module_name, module in model.named_modules():
            if hasattr(module, 'f_proj') and hasattr(module, 'A_log'):
                gates[module_name] = {'group_size': module.group_size, 'curvature': module.curvature,
                                      'gate_trainable': all(p.requires_grad for p in module.f_proj.parameters()),
                                      'A_log_min': float(module.A_log.min()), 'A_log_max': float(module.A_log.max()),
                                      'dt_bias_min': float(module.dt_bias.min()), 'dt_bias_max': float(module.dt_bias.max()),
                                      'decay_function': module.decay_function}
        result['key_gate_parameters'] = gates
    # recovery is checked on an actual small model; each rank saves its own RNG and data position
    import tempfile
    recovery_batches = [make_batch(args, 16 + runtime.rank, small=True).to(runtime.device)]
    recovery_total = runtime.global_count(recovery_batches)
    with tempfile.TemporaryDirectory(prefix=f'khdn-recovery-rank{runtime.rank}-') as temporary:
        checkpoint = Path(temporary) / 'state.pt'
        contract = {'small_model': info['configuration'], 'world_size': runtime.world_size, 'method': name}
        save_checkpoint(checkpoint, ddp, ddp_optimizer, step=1, contract=contract, rank=runtime.rank)
        ddp_optimizer.zero_grad(set_to_none=True)
        with khdn_update_guard(args, name, 'uninterrupted_recovery', update_guards):
            backward_step(ddp, recovery_batches, args, recovery_total, runtime.world_size)
        clip_and_step(ddp, ddp_optimizer, args)
        uninterrupted, uninterrupted_rng = parameters(ddp), capture_rng()
        next_data, step = load_checkpoint(checkpoint, ddp, ddp_optimizer, contract=contract,
                                          rank=runtime.rank, weight_decay=args.weight_decay)
        if next_data != 1 or step != 1:
            raise AssertionError('checkpoint did not restore data and optimizer positions')
        ddp_optimizer.zero_grad(set_to_none=True)
        with khdn_update_guard(args, name, 'resumed_recovery', update_guards):
            backward_step(ddp, recovery_batches, args, recovery_total, runtime.world_size)
        clip_and_step(ddp, ddp_optimizer, args)
        recovery = error_metrics(parameters(ddp), uninterrupted, atol=2e-5, rtol=2e-4)
        resumed_rng = capture_rng()
        if not torch.equal(resumed_rng['torch'], uninterrupted_rng['torch']):
            raise AssertionError('resume changed CPU RNG trajectory')
        if resumed_rng['python'] != uninterrupted_rng['python'] or not np.array_equal(resumed_rng['numpy'][1],
                                                                                      uninterrupted_rng['numpy'][1]):
            raise AssertionError('resume changed Python/NumPy RNG trajectory')
        if resumed_rng['cuda'] is not None and not torch.equal(resumed_rng['cuda'], uninterrupted_rng['cuda']):
            raise AssertionError('resume changed CUDA RNG trajectory')
        result['resume_parameter_errors'] = recovery
        result['resume_scope'] = 'small real model next-update replay; full-model finite save still required'
    del ddp_optimizer, serial_optimizer, ddp, serial, model
    return result


def aggregate_samples(per_rank):
    lengths = {len(rows) for rows in per_rank}
    if len(lengths) != 1:
        raise ValueError('rank sample counts differ')
    rows = []
    for index in range(next(iter(lengths))):
        ranks = [samples[index] for samples in per_rank]
        labels = {row['global_valid_labels'] for row in ranks}
        if len(labels) != 1:
            raise ValueError('global label counts disagree')
        milliseconds = max(row['wall_ms'] for row in ranks)
        slots = sum(row['input_slots'] for row in ranks)
        nonpadding = sum(row['nonpadding_tokens'] for row in ranks)
        valid = next(iter(labels))
        rows.append({**{key: max(row.get(key, 0.0) for row in ranks)
                        for key in ('fwd_cuda_ms', 'bwd_cuda_ms', 'optimizer_cuda_ms', 'total_cuda_ms', 'total_wall_ms')},
                     'max_rank_wall_ms': milliseconds, 'max_rank_cuda_ms': max(row['cuda_ms'] for row in ranks),
                     'global_input_slots': slots, 'global_nonpadding_tokens': nonpadding, 'global_valid_labels': valid,
                     'global_mean_loss': sum(row['weighted_local_loss'] for row in ranks) / len(ranks),
                     'input_slots_per_second': slots * 1000 / milliseconds,
                     'nonpadding_tokens_per_second': nonpadding * 1000 / milliseconds,
                     'valid_labels_per_second': valid * 1000 / milliseconds})
    return rows


def train_trial(args, name, trial, runtime, benchmark, result=None):
    """Persist rank-local progress after each successful sample, outside every timing window."""
    if len(args.modes) != 1:
        raise ValueError('one mode per invocation is required for an independent trajectory')
    result = {} if result is None else result
    mode = args.modes[0]
    journal = rank_path(args.output.with_name(f'{args.output.stem}.{name}.trial{trial}.json'), runtime.rank)
    next_data, optimizer_steps = 0, 0
    result.update(method=name, trial=trial, modes={}, status='IN_PROGRESS', journal=str(journal),
                  configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  comparison_category='controlled_plain_ce' if args.loss_backend == 'plain' else 'native_fastest_qualified',
                  effective_loss_backend=effective_loss_backend(args, name),
                  timing_qualification='memory-only observations' if args.memory_only else 'pending GPU/review acceptance')

    def persist(stage, sample_index=None):
        result['position'] = {'method': name, 'trial': trial, 'mode': mode, 'stage': stage,
                              'sample_index': sample_index, 'next_global_microbatch': next_data,
                              'global_optimizer_step': optimizer_steps}
        write_json(journal, result)

    def batches_at(cursor, count):
        return [make_batch(args, cursor + micro * runtime.world_size + runtime.rank).to(runtime.device)
                for micro in range(count)]

    def sample(count, execution_mode):
        nonlocal next_data, optimizer_steps
        batches = batches_at(next_data, count)
        if name in ('khdn_no_carry', 'khdn_y_carry'):
            with KHDNExceptionGuard(precision_policy=args.khdn_precision_policy or 'strict') as guard:
                result['timing_exception_guard'] = guard.evidence()
                row = timed_sample(model, optimizer, batches, args, runtime, execution_mode)
        else:
            row = timed_sample(model, optimizer, batches, args, runtime, execution_mode)
        row['start_global_microbatch'] = next_data
        next_data += count * runtime.world_size
        optimizer_steps += int(execution_mode == 'step')
        return row

    persist('model_construction')
    try:
        spec = benchmark.method_spec(name)
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        model, info = benchmark.build_model(args, spec)
        optimizer = make_optimizer(model, args)
        contract = checkpoint_contract(args, name, runtime.world_size)
        result.update(model=info, contract=contract, resumed=bool(args.resume))
        plan = scaling_plan(args, runtime.world_size)
        if args.resume:
            persist('checkpoint_resume')
            next_data, optimizer_steps = load_checkpoint(args.resume / f'{name}.rank{runtime.rank}.pt', model, optimizer,
                                                         contract=contract, rank=runtime.rank, weight_decay=args.weight_decay)
        # checkpoint agreement happens before wrapping and before any model route execution
        model = runtime.wrap(model)
        if info['parameter_count'] != recipe_parameter_counts(args, name)['total']:
            raise AssertionError(f'{name}: actual parameter count disagrees with source inventory')
        result['optimizer_groups'] = [{key: value for key, value in group.items() if key != 'params'}
                                      for group in optimizer.param_groups]
        persist('route_gate')
        route_batch = batches_at(next_data, 1)[0]
        result['route_evidence'] = trace_route(unwrap(model), route_batch, args, name, benchmark)
        optimizer.zero_grad(set_to_none=True)
        del route_batch
        if mode != 'step':
            persist('phase_optimizer_initialization')
            result['phase_optimizer_initialization'] = sample(plan['accumulation'], 'step')
            persist('phase_optimizer_initialized')
        count = plan['accumulation'] if mode == 'step' else 1
        measurements = {'warmup': [], 'raw_samples': [],
                        'scope': 'one microbatch per rank; independent phase-only trajectory' if mode != 'step' else
                                 'all accumulation microsteps: forward+loss+backward+communication+clip+optimizer'}
        result['modes'][mode] = measurements
        for index in range(args.warmup):
            persist('warmup', index)
            measurements['warmup'].append(sample(count, mode))
            persist('warmup_completed', index)
        optimizer.zero_grad(set_to_none=True)
        if args.memory_only:
            # one cache release before the dedicated series; never between steady-state samples
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        baseline = memory_snapshot()
        measurements['memory_baseline'] = baseline
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.steps):
            persist('measurement', index)
            measurements['raw_samples'].append(sample(count, mode))
            persist('measurement_completed', index)
        samples = measurements['raw_samples']
        allocated = max(row['training_memory']['peak_allocated_bytes'] for row in samples)
        reserved = max(row['training_memory']['peak_reserved_bytes'] for row in samples)
        memory = {'baseline': baseline, 'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
                  'extra_peak_allocated_bytes': max(0, allocated - baseline['allocated_bytes']),
                  'extra_peak_reserved_bytes': max(0, reserved - baseline['reserved_bytes']),
                  'allocated_scope': 'sample peak captured before post-step diagnostics',
                  'reserved_scope': 'process caching allocator; may retain earlier diagnostics/methods/trials',
                  'fresh_process_recipe_requested': args.memory_only,
                  'interleaved_health_diagnostics': not args.memory_only}
        measurements['local_memory'] = memory
        persist('measurement_summary')
        gathered = runtime.gather(samples)
        aggregated = aggregate_samples(gathered)
        manifests = [entry for row in samples for entry in row['samples']]
        measurements.update(aggregate_samples=aggregated,
                            summary_ms=summarize([row['max_rank_wall_ms'] for row in aggregated]),
                            input_slots_per_second=summarize([row['input_slots_per_second'] for row in aggregated]),
                            memory_per_rank=runtime.gather(memory),
                            global_sample_manifest=sorted((entry for rank in runtime.gather(manifests) for entry in rank),
                                                          key=lambda entry: entry['id']))
        if args.memory_only:
            persist('deferred_health_gate')
            benchmark.gradient_health(model)
            benchmark.parameter_health(unwrap(model))
            optimizer_health(optimizer)
        if mode == 'step':
            result.update(samples=samples, warmup_samples=measurements['warmup'],
                          tokens_per_second=measurements['input_slots_per_second'], memory=measurements['memory_per_rank'],
                          summary_ms={key: summarize([row[key] for row in aggregated]) for key in
                                      ('fwd_cuda_ms', 'bwd_cuda_ms', 'optimizer_cuda_ms', 'total_cuda_ms', 'total_wall_ms')})
        result['stability_samples'] = []
        if mode == 'step':
            for index in range(args.stability_steps):
                persist('stability', index)
                result['stability_samples'].append(sample(plan['accumulation'], 'step'))
                persist('stability_completed', index)
            persist('checkpoint_save')
            checkpoint_root = args.checkpoint_dir or args.output.parent / (args.output.stem + '.checkpoints')
            checkpoint = checkpoint_root / f'trial{trial}' / f'{name}.rank{runtime.rank}.pt'
            manifest = save_checkpoint(checkpoint, model, optimizer, step=next_data, contract=contract, rank=runtime.rank,
                                       optimizer_step=optimizer_steps)
            result.update(checkpoint=str(checkpoint), checkpoint_generation=manifest['generation'],
                          checkpoint_manifest=str(checkpoint_manifest_path(checkpoint)))
            persist('checkpoint_reload')
            restored = load_checkpoint(checkpoint, model, optimizer, contract=contract, rank=runtime.rank,
                                       weight_decay=args.weight_decay)
            if restored != (next_data, optimizer_steps):
                raise AssertionError('full-model recovery lost data or optimizer position')
            persist('recovery_update')
            result['full_model_recovery'] = sample(plan['accumulation'], 'step')
            result['full_model_recovery']['scope'] = 'finite next update after collectively validated generation'
            persist('recovery_completed')
        else:
            result['recovery_scope'] = 'phase-only experiment; no stability training, checkpoint, or resume'
        if args.profile:
            persist('profile')
            path = args.profile / f'{name}.trial{trial}.rank{runtime.rank}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                result['profile_sample'] = sample(count, mode)
            persist('profile_export')
            profiler.export_chrome_trace(str(path))
            result['separate_profile'] = str(path)
            persist('profile_completed')
        result['final_parameter_health'] = benchmark.parameter_health(unwrap(model))
        result['global_optimizer_step'], result['next_data_step'] = optimizer_steps, next_data
        result['status'] = 'COMPLETED; ACCEPTANCE PENDING'
        persist('completed')
        return result
    except BaseException as error:
        if hasattr(error, 'khdn_evidence'):
            result['khdn_failure_evidence'] = error.khdn_evidence
        result['status'] = 'FAILED; PARTIAL EVIDENCE ONLY'
        result['position'].update(next_global_microbatch=next_data, global_optimizer_step=optimizer_steps,
                                  counter_scope='completed sample calls; the failing call may have partially updated state')
        result['failure'] = traceback.format_exc()
        write_json(journal, result)
        raise


def run_benchmark(args, payload, benchmark):
    runtime = Runtime(args)
    output = rank_path(args.output, runtime.rank)
    payload['run_identity'] = {'host': socket.gethostname(), 'pid': os.getpid(), 'parent_pid': os.getppid(),
                               'rank': runtime.rank, 'local_rank': runtime.local_rank, 'world_size': runtime.world_size,
                               'run_id': os.environ.get('TORCHELASTIC_RUN_ID'), 'pod_uid': os.environ.get('POD_UID'),
                               'image_digest': os.environ.get('IMAGE_DIGEST'), 'job_id': os.environ.get('JOB_ID')}
    payload['training_contract'].update(
        optimizer='AdamW(fused on CUDA, betas=(0.9,0.95)); _no_weight_decay groups enforced after resume',
        loss='sum CE / (global valid shifted labels / world size); no extra accumulation division',
        timing_scope='aligned synchronized wall; DDP communication, clipping and optimizer included; '
                     'data/counts/zero_grad/health/reporting excluded; phases measured independently',
        checkpoint='no recurrent cache across optimizer steps; per-rank RNG/model/config/optimizer/data/step',
    )
    payload['ddp_acceptance'] = 'NOT ACCEPTED: independent Astra review and real 1/8/16 GPU validation remain required'
    try:
        with runtime:
            torch.backends.cuda.matmul.allow_tf32 = False
            payload['metadata'] = metadata(args)
            hardware = runtime.gather(payload['metadata'])
            required = ('revision', 'source_kind', 'source_tree_sha256', 'source_manifest_sha256',
                        'torch', 'cuda_runtime', 'triton', 'nccl')
            if any(any(rank.get(key) != hardware[0].get(key) for key in required) for rank in hardware):
                raise RuntimeError('ranks differ in source or required software identity')
            payload['rank_metadata'] = hardware
            payload['correctness'], payload['measurements'] = {}, {}
            legacy = copy.copy(args)
            legacy.methods = [name for name in args.methods if name in benchmark.METHOD_NAMES]
            if legacy.methods:
                payload['legacy_small_model_parity'] = runtime.gather(benchmark.small_model_gate(legacy))
            for name in args.methods:
                with environment(benchmark.method_spec(name)['environment']):
                    gate = small_gate(args, name, runtime, benchmark)
                    payload['correctness'][name] = runtime.gather(gate)
            payload['status'] = 'GPU SMALL MODEL UPDATE/RECOVERY GATES PASSED; DDP ACCEPTANCE PENDING'
            write_json(output, payload)
            if not args.check_only:
                initial_hashes = {}
                for trial in range(args.trials):
                    # reverse paired order on alternating independent trials
                    order = args.methods if trial % 2 == 0 else list(reversed(args.methods))
                    for name in order:
                        method_result = payload['measurements'].setdefault(name, {
                            'trials': [], 'legacy_summary_scope': 'trial0 only; not cross-trial confirmation',
                            'legacy_comparison_scope': 'trial0 only; main owns cross-trial confirmation'})
                        active = {}
                        method_result['trials'].append(active)
                        payload['active_trial'] = {'method': name, 'trial': trial, 'rank': runtime.rank}
                        write_json(output, payload)
                        with environment(benchmark.method_spec(name)['environment']):
                            result = train_trial(args, name, trial, runtime, benchmark, result=active)
                        group = benchmark.method_spec(name)['group']
                        digest = result['model']['initial_weights_sha256']
                        if group in initial_hashes and initial_hashes[group] != digest:
                            raise AssertionError(f'{name}: comparison group initial weights differ')
                        initial_hashes[group] = digest
                        if trial == 0:
                            method_result.update(result)
                        if 'step' in args.modes and not args.memory_only:
                            benchmark.add_comparisons(payload['measurements'])
                        payload.pop('active_trial', None)
                        write_json(output, payload)
            final_identity = metadata(args)
            if any(final_identity.get(key) != payload['metadata'].get(key)
                   for key in ('source_tree_sha256', 'source_kind', 'source_manifest_sha256')):
                raise RuntimeError('source files changed during this run; timing evidence rejected')
            payload['status'] = 'GPU HARNESS RUN COMPLETED; DDP ACCEPTANCE PENDING INDEPENDENT REVIEW'
            write_json(output, payload)
            identities = runtime.gather(payload['run_identity'])
            if runtime.rank == 0:
                payload['rank_identities'] = identities
                payload['rank_results'] = [str(rank_path(args.output, rank)) for rank in range(runtime.world_size)]
                write_json(args.output, payload)
    except BaseException as error:
        if hasattr(error, 'khdn_evidence'):
            payload['khdn_failure_evidence'] = error.khdn_evidence
        payload['status'] = 'GPU HARNESS FAILED; configuration retained; no automatic retry or shape reduction'
        payload['failure'] = traceback.format_exc()
        write_json(output, payload)
        raise
    return 0


def khdn_model_oracle_call(entry, q, k, v, beta, log_decay, **options):
    """Evaluate independent FP64 truth while preserving public dtypes and all source edges."""
    sources = dict(q=q, k=k, v=v, beta=beta, log_decay=log_decay,
                   initial_state=options.get('initial_state'), curvature=options.get('curvature'))
    state_dtype = (torch.float64 if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64
                                        for value in sources.values()) else torch.float32)
    promoted = dict(options)
    promoted.update({key: value.double() for key, value in sources.items() if isinstance(value, torch.Tensor)})
    output, state = entry(**promoted)
    if output.dtype != torch.float64 or state is not None and state.dtype != torch.float64:
        raise AssertionError('Model Oracle must compute its output and state in FP64')
    return output.to(q.dtype), None if state is None else state.to(state_dtype)


@contextlib.contextmanager
def khdn_model_oracle_precision():
    """Scope FP64 reference arithmetic across untimed forward and checkpoint backward."""
    import inspect

    package = importlib.import_module('fla.ops.key_gated_hyperbolic_delta_rule')
    original = package.naive_recurrent_key_gated_hyperbolic_delta_rule
    source = Path(inspect.getfile(original))
    evidence = {'reference_source': str(source), 'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'internal_dtype': 'torch.float64', 'public_boundary': 'original output/state dtypes', 'calls': []}

    def oracle(q, k, v, beta, log_decay, **options):
        output, state = khdn_model_oracle_call(original, q, k, v, beta, log_decay, **options)
        sources = dict(q=q, k=k, v=v, beta=beta, log_decay=log_decay,
                       initial_state=options.get('initial_state'), curvature=options.get('curvature'))
        evidence['calls'].append({'shape': list(q.shape),
                                  'source_dtypes': {key: str(value.dtype) if isinstance(value, torch.Tensor) else None
                                                    for key, value in sources.items()},
                                  'output_dtype': str(output.dtype), 'state_dtype': str(state.dtype) if state is not None else None})
        return output, state

    package.naive_recurrent_key_gated_hyperbolic_delta_rule = oracle
    try:
        yield evidence
    finally:
        package.naive_recurrent_key_gated_hyperbolic_delta_rule = original


def khdn_model_oracle_gate(args, name, benchmark):
    """Use identical g32 KHDN models and the explicit dense Oracle, including learned-curvature gradients."""
    gate_args = copy.copy(args)
    gate_args.group_size, gate_args.learnable_curvature = 32, True
    gate_args.batch, gate_args.seq_len, gate_args.vocab_size = 2, 2 * args.chunk_size + 1, 97
    gate_args.head_dim = benchmark.model_configuration(gate_args, name, small=True)['head_dim']
    result = {}
    for layout in ('dense', 'packed'):
        gate_args.data_layout = layout
        batch = make_batch(gate_args, 0).to('cuda')
        outputs = []
        for method in (name, 'khdn_naive'):
            model, info = benchmark.build_model(gate_args, benchmark.method_spec(method), small=True)
            oracle_context = khdn_model_oracle_precision() if method == 'khdn_naive' else contextlib.nullcontext()
            with oracle_context as oracle_evidence:
                route = trace_route(model, batch, gate_args, method, benchmark)
            if method == 'khdn_naive':
                if not oracle_evidence['calls']:
                    raise AssertionError('Model Oracle did not execute the explicit FP64 reference')
                result[layout]['oracle_precision'] = oracle_evidence
            outputs.append({'loss': torch.tensor(route['loss']), 'initial_weights_sha256': info['initial_weights_sha256'],
                            'gradients': {key: value.grad.detach().float().cpu().clone()
                                          for key, value in model.named_parameters() if value.requires_grad}})
            if method == name:
                result[layout] = {'actual_route': route, 'model': info}
            del model
        parity = benchmark.compare_model_results(*outputs)
        required = ('f_proj.0.weight', 'f_proj.1.weight', 'A_log', 'dt_bias', 'curvature_raw')
        for suffix in required:
            if not any(key.endswith(suffix) for key in outputs[0]['gradients']):
                raise AssertionError(f'KHDN Oracle gate did not exercise parameter gradient {suffix}')
        result[layout]['parity'] = parity
    source_gate = khdn_operator_gate(gate_args, name, khdn_operator_sources(gate_args, 2 * args.chunk_size + 1))
    return {'cases': result, 'source_vjp_gate': source_gate, 'scope': 'same FP32 parameters/public autocast, g32, learned positive curvature, '
            'identical weights and outer checkpoint; explicit naive dense-solve Oracle only in this untimed small gate',
            'dense_fp64_truth_scope': 'independent operator and model Oracle; model reference internally promotes prepared '
            'sources differentiably to FP64 and restores public output/state dtypes; outer autocast and parameters unchanged'}


def khdn_operator_sources(args, length):
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    shape = (1, length, 1, args.head_dim)
    q = torch.randn(shape, generator=generator, device='cuda')
    k = torch.nn.functional.normalize(torch.randn(shape, generator=generator, device='cuda'), dim=-1)
    v = .02 * torch.randn(shape, generator=generator, device='cuda')
    beta = .1 + .8 * torch.rand(shape[:3], generator=generator, device='cuda')
    ell = -.01 - .1 * torch.rand(shape, generator=generator, device='cuda')
    state = .001 * torch.randn((1, 1, args.head_dim, args.head_dim), generator=generator, device='cuda')
    return (q, k, v, beta, ell, state, q.new_tensor(args.curvature))


def khdn_operator_call(entry, sources, args, name, *, oracle=False):
    options = {} if oracle else {'chunk_size': args.chunk_size,
                                 'carry_mode': 'y_carry' if name == 'khdn_y_carry' else 'no_carry',
                                 'precision_policy': args.khdn_precision_policy or 'strict'}
    return entry(*sources[:5], curvature=sources[6], initial_state=sources[5], output_final_state=True,
                 group_size=args.group_size, check_inputs=True, **options)


def khdn_operator_gate(args, name, sources):
    """Compare unrounded dense FP64 truth to the public FP32 route with the frozen operator budgets."""
    import fla.ops.key_gated_hyperbolic_delta_rule as ops
    actual_sources = tuple(value.detach().clone().requires_grad_() for value in sources)
    truth_sources = tuple(value.detach().double().requires_grad_() for value in sources)
    oracle_checkpoint = verify_checkpointed_oracle(ops.naive_recurrent_key_gated_hyperbolic_delta_rule, sources, args, name)
    truth = checkpointed_dense_oracle(ops.naive_recurrent_key_gated_hyperbolic_delta_rule, truth_sources, args, name)
    with KHDNTrace(args.chunk_size, args.group_size, 'y_carry' if name == 'khdn_y_carry' else 'no_carry',
                   precision_policy=args.khdn_precision_policy or 'strict') as trace:
        actual = khdn_operator_call(ops.chunk_key_gated_hyperbolic_delta_rule, actual_sources, args, name)
        cotangents = tuple(torch.cos(torch.arange(value.numel(), device=value.device)).reshape(value.shape).to(value)
                           for value in actual)
        actual_grads = torch.autograd.grad(actual, actual_sources, cotangents)
    try:
        truth_grads = torch.autograd.grad(truth, truth_sources, tuple(value.double() for value in cotangents))
        errors = {}
        for category, values, references, names, budget in (
            ('forward', actual, truth, ('output', 'state'), (2e-5, 2e-4, 1e-4)),
            ('backward', actual_grads, truth_grads, ('q', 'k', 'v', 'beta', 'log_decay', 'S0', 'curvature'), (5e-5, 5e-4, 2e-4)),
        ):
            for key, value, reference in zip(names, values, references):
                if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
                    raise AssertionError(f'KHDN nonfinite {category}/{key}')
                difference = value.double() - reference
                atol, rtol, ratio = budget
                torch.testing.assert_close(value.double(), reference, atol=atol, rtol=rtol, msg=key)
                rms = difference.square().mean().sqrt()
                scale = reference.square().mean().sqrt().clamp_min(atol)
                if rms > ratio * scale:
                    raise AssertionError(f'KHDN {category}/{key} exceeds frozen relative-RMS budget')
                errors[f'{category}/{key}'] = {'max_abs': float(difference.abs().max()), 'relative_rms': float(rms / scale),
                                               'atol': atol, 'rtol': rtol, 'relative_rms_budget': ratio}
    except BaseException as error:
        error.khdn_evidence = trace.calls
        raise
    return {'route': trace.attest(), 'errors_against_dense_fp64_truth': errors,
            'independent_oracle_checkpoint_verification': oracle_checkpoint,
            'scope': 'same rounded FP32 sources promoted to dense FP64 truth; exceptional high-precision routes excluded'}


def run_operator_sweep(args, payload):
    """Bounded real entry-point sweep using fresh graphs and the shared synchronized sampling code."""
    import fla.ops.key_gated_hyperbolic_delta_rule as ops
    from benchmarks.hyperbolic_benchmark_utils import Method, _sample
    payload['scope'] = 'operator only, B=H=1, K=V=head-dim; separate from full-model and DDP acceptance'
    payload['operator_cases'] = []
    initial_identity = payload['metadata']
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for length in args.operator_lengths:
            sources = khdn_operator_sources(args, length)
            for name in args.methods:
                case = {'method': name, 'shape': list(sources[0].shape), 'chunk_size': args.chunk_size,
                        'group_size': args.group_size, 'mode': args.modes[0], 'status': 'GATE_PENDING'}
                payload['operator_cases'].append(case)
                write_json(args.output, payload)
                case['correctness'] = khdn_operator_gate(args, name, sources)
                if args.check_only:
                    case['status'] = 'ACTUAL OPERATOR GATE PASSED; TIMING NOT RUN'
                    write_json(args.output, payload)
                    continue
                method = Method(name, lambda leaves, name=name: khdn_operator_call(
                    ops.chunk_key_gated_hyperbolic_delta_rule, leaves, args, name), {}, 'khdn', 'state_fwd_kernel')
                case['warmup'], case['raw_samples'] = [], []
                with KHDNExceptionGuard(precision_policy=args.khdn_precision_policy or 'strict') as guard:
                    case['exception_guard'] = guard.evidence()
                    case['first_sample_including_possible_compilation'] = _sample(method, sources, args.modes[0])
                    for _ in range(args.warmup):
                        case['warmup'].append(_sample(method, sources, args.modes[0]))
                    for _ in range(args.steps):
                        case['raw_samples'].append(_sample(method, sources, args.modes[0]))
                        write_json(args.output, payload)
                    if args.profile:
                        args.profile.mkdir(parents=True, exist_ok=True)
                        path = args.profile / f'{name}.T{length}.C{args.chunk_size}.{args.modes[0]}.json'
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
                            _sample(method, sources, args.modes[0])
                        profiler.export_chrome_trace(str(path))
                        case['profile_trace'] = str(path)
                case['wall_ms'] = summarize([sample['wall_ms'] for sample in case['raw_samples']])
                case['cuda_ms'] = summarize([sample['cuda_ms'] for sample in case['raw_samples']])
                case['status'] = 'ACTUAL OPERATOR RUN COMPLETED; NOT MODEL/DDP ACCEPTANCE'
                write_json(args.output, payload)
        final_identity = metadata(args)
        if any(final_identity.get(key) != initial_identity.get(key)
               for key in ('source_tree_sha256', 'source_kind', 'source_manifest_sha256')):
            raise RuntimeError('source changed during operator sweep; timing rejected')
        payload['status'] = ('GPU OPERATOR CORRECTNESS GATES PASSED; TIMING NOT RUN' if args.check_only else
                             'GPU OPERATOR SWEEP COMPLETED; NO FULL-MODEL/DDP ACCEPTANCE')
    except BaseException as error:
        if hasattr(error, 'khdn_evidence'):
            payload['khdn_failure_evidence'] = error.khdn_evidence
        payload['status'] = 'GPU OPERATOR SWEEP FAILED; PARTIAL EVIDENCE ONLY'
        payload['failure'] = traceback.format_exc()
        write_json(args.output, payload)
        raise
    write_json(args.output, payload)
    return 0


def checkpointed_dense_oracle(entry, sources, args, name):
    """Keep canonical state and all seven source-gradient chains through independent dense temporal chunks."""
    from torch.utils.checkpoint import checkpoint
    q, k, v, beta, ell, state, curvature = sources
    outputs = []
    for start in range(0, q.shape[1], args.oracle_chunk_size):
        end = min(q.shape[1], start + args.oracle_chunk_size)
        block = (q[:, start:end], k[:, start:end], v[:, start:end], beta[:, start:end], ell[:, start:end], state, curvature)

        def run(*inputs):
            return khdn_operator_call(entry, inputs, args, name, oracle=True)
        output, state = checkpoint(run, *block, use_reentrant=False, preserve_rng_state=False)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def verify_checkpointed_oracle(entry, sources, args, name):
    """Prove the memory-bounded harness against the uncheckpointed dense Oracle on a bounded same-source witness."""
    length = min(sources[0].shape[1], 2 * args.oracle_chunk_size + 1)
    small = tuple(value[:, :length] if index < 5 else value for index, value in enumerate(sources))
    direct_sources = tuple(value.detach().double().requires_grad_() for value in small)
    checkpoint_sources = tuple(value.detach().double().requires_grad_() for value in small)
    direct = khdn_operator_call(entry, direct_sources, args, name, oracle=True)
    bounded = checkpointed_dense_oracle(entry, checkpoint_sources, args, name)
    cotangents = tuple(torch.cos(torch.arange(value.numel(), device=value.device)).reshape(value.shape).to(value)
                       for value in direct)
    direct_gradients = torch.autograd.grad(direct, direct_sources, cotangents)
    bounded_gradients = torch.autograd.grad(bounded, checkpoint_sources, cotangents)
    errors = {}
    for key, actual, reference in zip(('output', 'state', 'q', 'k', 'v', 'beta', 'log_decay', 'S0', 'curvature'),
                                      (*bounded, *bounded_gradients), (*direct, *direct_gradients)):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9, msg=key)
        errors[key] = float((actual - reference).detach().abs().max())
    return {'sequence_length': length, 'temporal_chunk_size': args.oracle_chunk_size, 'errors': errors,
            'scope': 'independent dense Oracle checkpointed versus uncheckpointed; same FP64 sources, all seven gradients',
            'atol_rtol': [1e-10, 1e-9]}
