# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness-gated, FP32-parameter HDN/GDN full-model training benchmark.

CUDA execution is required. --help and --dry-run also work on CPU hosts.
Example:
    python benchmarks/benchmark_hyperbolic_training.py --batch 8 --seq-len 2048 --methods hdn_tc hdn_staged gdn

GDN is a performance comparison, not a mathematically equivalent reference.
"""

import argparse
import functools
import gc
import hashlib
import inspect
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.hyperbolic_benchmark_utils import (  # noqa: E402
    KernelTrace,
    environment,
    metadata,
    summarize,
    write_json,
)

METHOD_NAMES = ('hdn_tc', 'hdn_staged', 'protected_value', 'protected_shared_inverse', 'protected_staged', 'gdn')
FLAGS = {
    'FLA_CI_ENV': '0', 'FLA_HYPERBOLIC_TC_FAST': '0', 'FLA_HYPERBOLIC_STAGED': '0',
    'FLA_HYPERBOLIC_SHARED_INVERSE': '0', 'FLA_HYPERBOLIC_PROTECTED_STAGED': '0',
}
LOSS_TOLERANCE = (1e-4, 1e-3)  # existing model packed-loss parity budget
GRADIENT_TOLERANCE = (2e-3, 2e-2)  # existing layer all-parameter gradient parity budget


def method_spec(name):
    flags = FLAGS.copy()
    if name == 'hdn_staged':
        flags['FLA_HYPERBOLIC_STAGED'] = '1'
    if name == 'protected_shared_inverse':
        flags['FLA_HYPERBOLIC_SHARED_INVERSE'] = '1'
    if name == 'protected_staged':
        flags['FLA_HYPERBOLIC_PROTECTED_STAGED'] = '1'
    group = 'gdn' if name == 'gdn' else ('protected' if name.startswith('protected_') else 'hdn')
    baseline = {'hdn': 'hdn_tc', 'protected': 'protected_value', 'gdn': None}[group]
    return {'name': name, 'group': group, 'environment': flags, 'baseline': baseline if baseline != name else None}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', choices=METHOD_NAMES, nargs='+', default=list(METHOD_NAMES))
    for name, default in (('batch', 8), ('seq-len', 2048), ('layers', 24), ('hidden-size', 2048),
                          ('heads', 16), ('head-dim', 128), ('intermediate-size', 5184), ('vocab-size', 32000)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16', help='autocast dtype; parameters stay FP32')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--learnable-curvature', action='store_true')
    parser.add_argument('--inner-checkpoint', action='store_true', help='enable protected inner checkpoint (default off)')
    parser.add_argument('--no-layer-checkpoint', action='store_true', help='disable outer nonreentrant layer checkpoint')
    parser.add_argument('--no-short-conv', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=43)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--stability-steps', type=int, default=10, help='additional continuous untimed optimizer steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--check-only', action='store_true', help='run only small-model loss/gradient and route gates')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--profile', type=Path, help='directory for one post-measurement Chrome trace per method')
    parser.add_argument('--output', type=Path, default=Path('profile/hyperbolic-opt/training-benchmark.json'))
    args = parser.parse_args(argv)
    dimensions = (args.batch, args.layers, args.hidden_size, args.heads, args.intermediate_size)
    if min(dimensions) <= 0 or args.seq_len < 2 or args.vocab_size < 4:
        parser.error('model dimensions must be positive; seq-len>=2 and vocab-size>=4')
    if not 16 <= args.head_dim <= 128 or args.head_dim % 16:
        parser.error('this benchmark requires head-dim in {16,32,...,128} to exercise the requested TC/protected routes')
    if args.hidden_size != args.heads * args.head_dim:
        parser.error('hidden-size must equal heads*head-dim for this matched-architecture comparison')
    if not 1 <= args.rank <= 16 or not math.isfinite(args.curvature) or not 0 < args.curvature <= 2:
        parser.error('rank must be in [1,16] and curvature in (0,2]')
    if args.warmup < 1 or args.steps < 2 or args.stability_steps < 0:
        parser.error('warmup>=1, steps>=2 and stability-steps>=0 are required')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('lr must be finite and positive; weight-decay must be finite and nonnegative')
    if len(args.methods) != len(set(args.methods)):
        parser.error('methods must not contain duplicates')
    return args


def model_configuration(args, name, *, small=False):
    common = dict(
        attn_mode='chunk', hidden_size=32 if small else args.hidden_size,
        num_hidden_layers=2 if small else args.layers, num_heads=2 if small else args.heads,
        head_dim=16 if small else args.head_dim, expand_v=1.0,
        intermediate_size=64 if small else args.intermediate_size, vocab_size=97 if small else args.vocab_size,
        max_position_embeddings=max(args.seq_len, 2 * args.chunk_size + 1),
        use_cache=False, use_short_conv=not args.no_short_conv,
        fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=True, fuse_linear_cross_entropy=False,
    )
    if name != 'gdn':
        common.update(tensor_core=True, curvature=args.curvature, learnable_curvature=args.learnable_curvature,
                      protection_rank=args.rank if name.startswith('protected_') else 0,
                      protected_checkpoint=args.inner_checkpoint, chunk_size=args.chunk_size)
    return common


def weight_digest(model):
    """Hash initial CPU weights without retaining a second full model/state copy."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != 'cpu':
            raise ValueError('Hash initial weights before moving the model to CUDA.')
        digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        digest.update(memoryview(tensor.detach().contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def build_model(args, spec, *, small=False):
    torch.manual_seed(args.seed)
    if spec['name'] == 'gdn':
        from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**model_configuration(args, spec['name'], small=small)))
    else:
        from fla.models.hyperbolic_deltanet import HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM
        model = HyperbolicDeltaNetForCausalLM(HyperbolicDeltaNetConfig(**model_configuration(args, spec['name'], small=small)))
    model.float().train()
    digest = weight_digest(model)
    if not args.no_layer_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameter_count = sum(x.numel() for x in model.parameters())
    if any(x.dtype != torch.float32 for x in model.parameters()):
        raise AssertionError('Training benchmark requires FP32 parameters.')
    return model.cuda(), {'initial_weights_sha256': digest, 'parameter_count': parameter_count,
                          'parameter_dtype': 'torch.float32', 'configuration': model.config.to_dict()}


def autocast(args):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.dtype == 'bf16')


def loss_for(model, tokens, args):
    with autocast(args):
        return model(input_ids=tokens, labels=tokens, use_cache=False).loss


class GDNBackendTrace:
    """Observe training entries across autograd threads and additional backend calls."""

    def __init__(self):
        self.calls, self.saved, self.wrapped_names = Counter(), [], set()

    def _wrap(self, function):
        name = f'{function.__module__}.{function.__qualname__}'
        self.wrapped_names.add(name)

        @functools.wraps(function)
        def observed(*args, **kwargs):
            self.calls[name] += 1
            return function(*args, **kwargs)
        return observed

    def _profile(self, frame, event, arg):
        if event == 'call':
            module = frame.f_globals.get('__name__', '')
            name = frame.f_code.co_name
            if module.startswith(('fla.ops.gated_delta_rule', 'fla.ops.common')) and 'verifier' not in name:
                qualified = f'{module}.{getattr(frame.f_code, "co_qualname", name)}'
                if qualified not in self.wrapped_names:
                    self.calls[qualified] += 1

    def __enter__(self):
        # sys.setprofile is thread-local; direct entry wrappers also observe autograd worker threads.
        entries = ('chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd',
                   'chunk_bwd_dqkwg', 'chunk_bwd_dqkwg_tilelang')
        wrapped = {}
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith(('fla.ops.gated_delta_rule', 'fla.ops.common')) or module is None:
                continue
            owners = [module] + [value for value in vars(module).values()
                                 if inspect.isclass(value) and value.__module__ == module_name]
            for owner in owners:
                for name in entries:
                    value = vars(owner).get(name)
                    if inspect.isfunction(value):
                        if id(value) not in wrapped:
                            wrapped[id(value)] = self._wrap(value)
                        self.saved.append((owner, name, value))
                        setattr(owner, name, wrapped[id(value)])
        self.previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self.previous)
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)

    def records(self):
        return [{'function': name, 'calls': count} for name, count in sorted(self.calls.items())]


def attest_training(name, records):
    if name == 'gdn':
        observed = [x['function'] for x in records]
        for suffix in ('chunk_gated_delta_rule_fwd', 'chunk_gated_delta_rule_bwd'):
            if not any(x.endswith(suffix) for x in observed):
                raise AssertionError(f'GDN training backend function {suffix} was not executed')
        return {'executed_backend_calls': records,
                'tilelang_backward_observed': any(x.endswith('chunk_bwd_dqkwg_tilelang') or
                                                  ('.backends.tilelang.' in x and x.endswith('.chunk_bwd_dqkwg'))
                                                  for x in observed),
                'interpretation': 'Executed backend functions, including mixed routes. Absence of an observed TileLang '
                                  'entry is not a claim of pure Triton; use the optional CUDA profiler for kernel names.'}
    if name == 'hdn_staged':
        required = [f'hyperbolic_staged_{stage}_kernel' for stage in (
            'prepare_fwd', 'state_fwd', 'output_fwd', 'output_bwd', 'state_bwd', 'values_bwd', 'parameters_bwd')]
    elif name == 'protected_staged':
        required = [f'protected_{stage}_kernel' for stage in (
            'shared_trajectory_fwd', 'value_trajectory_fwd', 'value_trajectory_bwd',
            'shared_trajectory_bwd', 'value_partials_bwd', 'bank_partials_bwd')]
        required.append('shared_key_gram_fwd_kernel')
    elif name.startswith('protected_'):
        required = ['protected_value_tile_fwd_kernel', 'protected_value_tile_bwd_kernel', 'soft_bank_update_bwd_kernel']
        if name == 'protected_shared_inverse':
            required.append('shared_key_gram_fwd_kernel')
    else:
        required = ['hyperbolic_delta_tc_chunk_fwd', 'hyperbolic_delta_tc_chunk_bwd']
    for suffix in required:
        matching = [x for x in records if x['kernel'].endswith(suffix)]
        if not matching:
            raise AssertionError(f'{name}: required training kernel {suffix} was not launched')
        if suffix in ('protected_value_tile_fwd_kernel', 'protected_value_tile_bwd_kernel'):
            if any(x['flags'].get('SHARED_INVERSE') != (name == 'protected_shared_inverse') for x in matching):
                raise AssertionError(f'{name}: actual forward/backward SHARED_INVERSE policy differs')
        if suffix in ('hyperbolic_delta_tc_chunk_fwd', 'hyperbolic_delta_tc_chunk_bwd'):
            if any(x['flags'].get('FAST') is not False for x in matching):
                raise AssertionError(f'{name}: actual forward/backward precision differs from strict TC')
    return {'kernel_launches': records}


def gradient_health(model):
    gradients = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError(f'Missing training gradient: {name}')
            gradients.append(torch.linalg.vector_norm(parameter.grad.detach().float()))
    norm = torch.linalg.vector_norm(torch.stack(gradients)).item()
    if not math.isfinite(norm):
        raise AssertionError('Non-finite parameter gradients')
    return norm


def parameter_health(model):
    """Untimed final check also catches a non-finite last optimizer update."""
    finite = [torch.isfinite(parameter.detach()).all() for parameter in model.parameters()]
    if not torch.stack(finite).all().item():
        raise AssertionError('Non-finite parameter after the final optimizer update')
    return {'all_finite': True, 'parameter_tensors_checked': len(finite)}


def traced_backward(model, tokens, args, spec):
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with GDNBackendTrace() if spec['name'] == 'gdn' else KernelTrace() as trace:
        loss = loss_for(model, tokens, args)
        loss.backward()
        torch.cuda.synchronize()
    value = loss.item()
    if not math.isfinite(value):
        raise AssertionError('Non-finite loss')
    evidence = attest_training(spec['name'], trace.records())
    evidence.update(loss=value, gradient_l2=gradient_health(model),
                    untimed_trace_wall_ms=(time.perf_counter() - started) * 1000)
    return evidence


def compare_model_results(actual, expected):
    if actual['initial_weights_sha256'] != expected['initial_weights_sha256']:
        raise AssertionError('Baseline and candidate did not start from identical weights')
    if actual['gradients'].keys() != expected['gradients'].keys():
        raise AssertionError('Parameter-gradient names differ')
    torch.testing.assert_close(actual['loss'], expected['loss'], atol=LOSS_TOLERANCE[0], rtol=LOSS_TOLERANCE[1])
    errors = {}
    for name, gradient in actual['gradients'].items():
        reference = expected['gradients'][name]
        if gradient is None or reference is None or not torch.isfinite(gradient).all() or not torch.isfinite(reference).all():
            raise AssertionError(f'Missing or non-finite gradient: {name}')
        torch.testing.assert_close(gradient, reference, atol=GRADIENT_TOLERANCE[0], rtol=GRADIENT_TOLERANCE[1], msg=name)
        errors[name] = {'max_abs': (gradient - reference).abs().max().item(),
                        'relative_rms': ((gradient - reference).square().mean().sqrt()
                                         / reference.square().mean().sqrt().clamp_min(1e-30)).item()}
    return {'loss_abs': abs(actual['loss'].item() - expected['loss'].item()), 'parameter_gradient_errors': errors,
            'loss_atol_rtol': LOSS_TOLERANCE, 'gradient_atol_rtol': GRADIENT_TOLERANCE}


def small_model_gate(args):
    requested = set(args.methods)
    if requested.intersection(('hdn_tc', 'hdn_staged')):
        requested.add('hdn_tc')
    if any(x.startswith('protected_') for x in requested):
        requested.add('protected_value')
    tokens = torch.randint(3, 97, (2, 2 * args.chunk_size + 1), generator=torch.Generator().manual_seed(args.data_seed)).cuda()
    references, evidence = {}, {}
    for name in METHOD_NAMES:
        if name not in requested:
            continue
        spec = method_spec(name)
        with environment(spec['environment']):
            model, model_info = build_model(args, spec, small=True)
            result = traced_backward(model, tokens, args, spec)
            actual = {'loss': torch.tensor(result['loss']), 'initial_weights_sha256': model_info['initial_weights_sha256'],
                      'gradients': {key: value.grad.detach().cpu().clone() for key, value in model.named_parameters()
                                    if value.requires_grad}}
            if spec['baseline']:
                result['errors_against_baseline'] = compare_model_results(actual, references[spec['group']])
            else:
                references[spec['group']] = actual
            result.update(model=model_info, gdn_is_performance_only=name == 'gdn')
            evidence[name] = result
            del actual, model
        gc.collect()
        torch.cuda.empty_cache()
    return {'status': 'GPU SMALL MODEL LOSS/ALL-PARAMETER GRADIENT GATE PASSED', 'methods': evidence,
            'tokens_sha256': hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
            'scope': 'Small dense training models; operator tests separately cover packed/cache/initial-state contracts.'}


def optimizer_step(model, optimizer, tokens, args):
    """Only forward/loss, backward and optimizer are timed; health checks are outside."""
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    started = time.perf_counter()
    events[0].record()
    loss = loss_for(model, tokens, args)
    events[1].record()
    loss.backward()
    events[2].record()
    optimizer.step()
    events[3].record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    value = loss.item()
    if not math.isfinite(value):
        raise AssertionError('Non-finite training loss')
    return {
        'fwd_cuda_ms': events[0].elapsed_time(events[1]), 'bwd_cuda_ms': events[1].elapsed_time(events[2]),
        'optimizer_cuda_ms': events[2].elapsed_time(events[3]), 'total_cuda_ms': events[0].elapsed_time(events[3]),
        'total_wall_ms': wall_ms, 'loss': value, 'gradient_l2': gradient_health(model),
        'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
    }


def train_method(args, spec, data, expected_hash):
    with environment(spec['environment']):
        model, info = build_model(args, spec)
        if expected_hash is not None and info['initial_weights_sha256'] != expected_hash:
            raise AssertionError(f'{spec["name"]}: initial weights differ from its HDN comparison group')
        route = traced_backward(model, data[0], args, spec)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True)
        warmup = [optimizer_step(model, optimizer, data[index], args) for index in range(args.warmup)]
        torch.cuda.reset_peak_memory_stats()
        start_allocated, start_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        samples = [optimizer_step(model, optimizer, data[args.warmup + index], args) for index in range(args.steps)]
        memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'start_allocated_bytes': start_allocated, 'start_reserved_bytes': start_reserved,
                  'scope': 'After optimizer-state warmup; includes FP32 parameters, gradients, optimizer state and pregenerated data.'}
        stability = [optimizer_step(model, optimizer, data[args.warmup + args.steps + index], args)
                     for index in range(args.stability_steps)]
        profile_path = None
        if args.profile:
            args.profile.mkdir(parents=True, exist_ok=True)
            profile_path = args.profile / f'{spec["name"]}.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                optimizer_step(model, optimizer, data[-1], args)
            prof.export_chrome_trace(str(profile_path))
        result = {
            'model': info, 'route_evidence': route, 'warmup_samples': warmup, 'samples': samples,
            'stability_samples': stability, 'memory': memory,
            'final_parameter_health': parameter_health(model),
            'numerical_validation_scope': 'Full-size loss, gradient and final-weight finiteness plus actual training route. '
                                          'Baseline/candidate loss and all-parameter gradient parity is gated only on small models; '
                                          'continuous full-size loss trajectories are recorded, not asserted equivalent.',
            'summary_ms': {key: summarize([row[key] for row in samples])
                           for key in ('fwd_cuda_ms', 'bwd_cuda_ms', 'optimizer_cuda_ms', 'total_cuda_ms', 'total_wall_ms')},
            'tokens_per_second': summarize([args.batch * args.seq_len * 1000 / row['total_wall_ms'] for row in samples]),
            'profile_trace': str(profile_path) if profile_path else None,
            'continuous_optimizer_steps': args.warmup + args.steps + args.stability_steps + bool(args.profile),
        }
        del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def add_comparisons(results):
    for name, result in results.items():
        baseline = method_spec(name)['baseline']
        if baseline in results:
            result['speedup_against_baseline'] = {
                'baseline': baseline,
                **{key: results[baseline]['summary_ms'][key]['median'] / result['summary_ms'][key]['median']
                   for key in ('fwd_cuda_ms', 'bwd_cuda_ms', 'optimizer_cuda_ms', 'total_cuda_ms', 'total_wall_ms')},
            }
        if 'gdn' in results and name != 'gdn':
            result['training_throughput_fraction_of_gdn'] = (
                results['gdn']['summary_ms']['total_wall_ms']['median'] / result['summary_ms']['total_wall_ms']['median']
            )


def main(argv=None):
    args = parse_arguments(argv)
    specs = [method_spec(name) for name in args.methods]
    payload = {
        'status': 'CONFIGURED', 'metadata': metadata(args), 'methods': specs,
        'training_contract': {
            'parameter_dtype': 'fp32', 'autocast_dtype': args.dtype, 'expand_v': 1.0,
            'optimizer': 'AdamW(fused=True, betas=(0.9,0.95))',
            'outer_checkpoint': not args.no_layer_checkpoint, 'use_reentrant': False,
            'protected_inner_checkpoint': args.inner_checkpoint,
            'timing_scope': 'Forward includes loss; backward includes outer checkpoint replay; optimizer included in total. '
                            'CUDA events per phase, synchronized wall total. zero_grad/data creation/health/trace are outside timing.',
            'comparison': 'Identical initial HDN weights within groups; GDN performance-only, never a numerical oracle.',
            'numerical_gate_scope': 'Small-model loss and all-parameter gradient parity; full-size runs check numerical '
                                    'finiteness and actual routes, without asserting full-size trajectory equivalence.',
        },
    }
    if args.dry_run:
        payload['status'] = 'DRY RUN; GPU NOT RUN'
        payload['model_configurations'] = {name: model_configuration(args, name) for name in args.methods}
        print(json.dumps(payload, indent=2))
        return 0
    if not torch.cuda.is_available():
        payload['status'] = 'GPU NOT RUN: CUDA device unavailable'
        write_json(args.output, payload)
        print(payload['status'])
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    payload['metadata'] = metadata(args)
    try:
        payload['correctness'] = small_model_gate(args)
        payload['status'] = 'GPU SMALL MODEL GATE PASSED'
        write_json(args.output, payload)
        if not args.check_only:
            total = args.warmup + args.steps + args.stability_steps
            data = torch.randint(3, args.vocab_size, (total, args.batch, args.seq_len),
                                 generator=torch.Generator().manual_seed(args.data_seed)).cuda()
            payload['data_sha256'] = hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
            payload['measurements'] = {}
            hashes = {}
            for spec in specs:
                result = train_method(args, spec, data, hashes.get(spec['group']))
                hashes[spec['group']] = result['model']['initial_weights_sha256']
                payload['measurements'][spec['name']] = result
                add_comparisons(payload['measurements'])
                payload['status'] = 'GPU TRAINING BENCHMARK PARTIALLY COMPLETED'
                write_json(args.output, payload)
                print(f'{spec["name"]}: {result["tokens_per_second"]["median"]:.1f} tokens/s (median wall time)')
            payload['status'] = 'GPU SMALL MODEL GATE AND FULL TRAINING BENCHMARK COMPLETED'
        write_json(args.output, payload)
    except Exception:
        payload['status'] = 'GPU VALIDATION OR TRAINING BENCHMARK FAILED'
        payload['failure'] = traceback.format_exc()
        write_json(args.output, payload)
        raise
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
