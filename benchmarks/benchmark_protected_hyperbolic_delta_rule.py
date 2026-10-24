# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness-gated protected HDN benchmarks with raw CUDA and wall timings.

Example:
    python benchmarks/benchmark_protected_hyperbolic_delta_rule.py --seq-len 1024 --modes fwd bwd fwd_bwd
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.hyperbolic_benchmark_utils import (  # noqa: E402
    Method,
    attest,
    compare,
    environment,
    evaluate,
    measure,
    metadata,
    validate_arguments,
    write_json,
)

FLAGS = {
    'FLA_CI_ENV': '0', 'FLA_HYPERBOLIC_STAGED': '0', 'FLA_HYPERBOLIC_TC_FAST': '0',
    'FLA_HYPERBOLIC_SHARED_INVERSE': '0', 'FLA_HYPERBOLIC_PROTECTED_STAGED': '0',
}


def make_inputs(args, *, reference_case=False):
    torch.manual_seed(args.seed)
    batch, length, heads, key_dim, value_dim = args.batch, args.seq_len, args.heads, args.key_dim, args.value_dim
    if reference_case:
        batch, length, heads = min(batch, 2), min(length, 2 * args.chunk_size + 1), min(heads, 2)
        key_dim, value_dim = min(key_dim, 32), min(value_dim, 16)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    q = torch.nn.functional.normalize(torch.randn(batch, length, heads, key_dim, device='cuda'), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1)
    v = 0.15 * torch.randn(batch, length, heads, value_dim, device='cuda').tanh()
    if not args.layer_inputs:
        k, v = k.to(dtype), v.to(dtype)
    beta = torch.rand_like(v, dtype=torch.float32)
    state = 0.05 * torch.nn.functional.normalize(torch.randn(batch, heads, key_dim, value_dim, device='cuda'), dim=-2)
    c = torch.tensor(args.curvature, device='cuda')
    bank_k = torch.zeros(batch, heads, args.rank, key_dim, device='cuda')
    bank_v = torch.zeros(batch, heads, args.rank, value_dim, device='cuda')
    bank_scores = torch.full((batch, heads, args.rank), -1e6, device='cuda')
    bank = 'partial' if reference_case else args.initial_bank
    occupied = max(1, args.rank // 2) if bank == 'partial' else (args.rank if bank == 'full' else 0)
    if occupied:
        bank_k[:, :, :occupied] = torch.nn.functional.normalize(torch.randn_like(bank_k[:, :, :occupied]), dim=-1)
        bank_v[:, :, :occupied] = torch.einsum('bhrk,bhkv->bhrv', bank_k[:, :, :occupied], state)
        bank_scores[:, :, :occupied] = 0
    selection = torch.randn(batch, length, heads, device='cuda') * 0.1
    return {
        'protected': (q, k, v, beta, state, bank_k, bank_v, bank_scores, selection, c),
        'unprotected': (q, k, v, beta, state, c),
    }


def _protected(args, inputs, strategy='value_parallel', reference=False):
    from fla.ops.hyperbolic_delta_rule.protected import protected_recurrent_hyperbolic_delta_rule
    return protected_recurrent_hyperbolic_delta_rule(
        *inputs[:4], curvature=inputs[9], initial_state=inputs[4], output_final_state=True,
        protected_keys=inputs[5], protected_values=inputs[6], protected_scores=inputs[7],
        selection_scores=inputs[8], protection_rank=args.rank, checkpoint_size=args.chunk_size,
        checkpoint=not args.no_checkpoint, temperature=args.temperature,
        mode='naive' if reference else 'triton', kernel_strategy=strategy,
    )


def _unprotected(args, inputs, reference=False):
    from fla.ops.hyperbolic_delta_rule.chunk import chunk_hyperbolic_delta_rule
    from fla.ops.hyperbolic_delta_rule.naive import naive_recurrent_hyperbolic_delta_rule
    if reference:
        return naive_recurrent_hyperbolic_delta_rule(
            *inputs[:4], curvature=inputs[5], initial_state=inputs[4], output_final_state=True,
        )
    return chunk_hyperbolic_delta_rule(
        *inputs[:4], curvature=inputs[5], initial_state=inputs[4], output_final_state=True, chunk_size=args.chunk_size,
    )


def _decomposed(args, inputs):
    from fla.ops.hyperbolic_delta_rule.protected_triton import _run_chunk, _TritonSmoothBankUpdate
    q, k, v, beta, state, bank_k, bank_v, bank_scores, selection, c = inputs
    outputs = []
    for start in range(0, q.shape[1], args.chunk_size):
        qc, kc, vc, bc, sc = (x[:, start:start + args.chunk_size].contiguous() for x in (q, k, v, beta, selection))
        out, state, realized = _run_chunk(qc, kc, vc, bc, state, bank_k, bank_v, bank_scores, c, q.shape[-1] ** -0.5)
        bank_k, bank_v, bank_scores, count = _TritonSmoothBankUpdate.apply(
            bank_k, bank_v, bank_scores, kc, realized, sc, args.temperature,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1), state, bank_k, bank_v, bank_scores, count


def methods(args):
    baseline = 'protected_value_parallel'
    result = [
        Method('unprotected_chunk', lambda x: _unprotected(args, x), FLAGS.copy(), 'unprotected',
               'hyperbolic_delta_rule_chunk_fwd_kernel'),
        Method(baseline, lambda x: _protected(args, x), FLAGS.copy(), 'protected',
               'protected_value_tile_fwd_kernel', {'SHARED_INVERSE': False}),
        Method('protected_shared_inverse', lambda x: _protected(args, x),
               {**FLAGS, 'FLA_HYPERBOLIC_SHARED_INVERSE': '1'}, 'protected',
               'protected_value_tile_fwd_kernel', {'SHARED_INVERSE': True}, baseline),
    ]
    if args.compare_strategies:
        result.extend([
            Method('protected_auto', lambda x: _protected(args, x, 'auto'), FLAGS.copy(), 'protected',
                   'protected_value_tile_fwd_kernel' if args.value_dim >= 8 else 'protected_fused_chunk_fwd_kernel', baseline=baseline),
            Method('protected_fused', lambda x: _protected(args, x, 'fused'), FLAGS.copy(), 'protected',
                   'protected_fused_chunk_fwd_kernel', baseline=baseline),
        ])
    if args.include_staged:
        result.append(Method('protected_decomposed', lambda x: _decomposed(args, x), FLAGS.copy(), 'protected',
                             'protected_kv_project_fwd_kernel', baseline=baseline))
    return result


def correctness(args, candidates, inputs):
    evidence = {'small_reference': {}, 'requested_shape': {}}
    small = make_inputs(args, reference_case=True)
    oracles = {}
    for group, run in (('protected', lambda x: _protected(args, x, reference=True)),
                       ('unprotected', lambda x: _unprotected(args, x, reference=True))):
        oracle_inputs = tuple(x.double() for x in small[group])
        oracles[group], _ = evaluate(run, oracle_inputs, args.seed + 1)
    for method in candidates:
        with environment(method.environment):
            actual, trace = evaluate(method.run, small[method.group], args.seed + 1, trace=True)
        attest(method, trace)
        if method.name == 'protected_shared_inverse' and not any(
            x['kernel'].endswith('shared_key_gram_fwd_kernel') for x in trace
        ):
            raise AssertionError('Shared inverse preparation did not execute.')
        # these are the existing protected parity and TC dtype budgets, not new relaxed thresholds
        if method.group == 'protected':
            out_tol, grad_tol = (5e-3, 5e-2), (8e-3, 8e-2)
        else:
            out_tol = {'fp32': (2e-4, 2e-3), 'fp16': (8e-4, 8e-3), 'bf16': (4e-3, 4e-2)}[args.dtype]
            grad_tol = (out_tol[0] * 3, out_tol[1] * 3)
        errors = compare(actual, oracles[method.group], out_tol, grad_tol, cast_reference=True)
        evidence['small_reference'][method.name] = {'errors': errors, 'kernel_launches': trace}
    full = {}
    for method in candidates:
        with environment(method.environment):
            actual, trace = evaluate(method.run, inputs[method.group], args.seed + 1, trace=True)
        attest(method, trace)
        row = {'kernel_launches': trace}
        if method.baseline:
            if method.name == 'protected_shared_inverse':
                out_tol, grad_tol = (2e-4, 2e-3), (3e-3, 3e-2)
            else:
                out_tol, grad_tol = (5e-3, 5e-2), (8e-3, 8e-2)
            row['errors_against_baseline'] = compare(actual, full[method.baseline], out_tol, grad_tol)
        full[method.name] = actual
        evidence['requested_shape'][method.name] = row
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (('batch', 1), ('seq-len', 4096), ('heads', 8), ('key-dim', 64), ('value-dim', 64), ('rank', 4)):
        parser.add_argument('--' + flag, type=int, default=default)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument('--dtype', choices=('fp32', 'bf16', 'fp16'), default='fp16')
    parser.add_argument('--layer-inputs', action='store_true', help='retain prepared K/V in FP32 while Q uses --dtype')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--curvature', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.25)
    parser.add_argument('--initial-bank', choices=('empty', 'partial', 'full'), default='empty')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--backward', action='store_true', help='compatibility alias for --modes fwd_bwd')
    parser.add_argument('--modes', choices=('fwd', 'bwd', 'fwd_bwd'), nargs='+')
    parser.add_argument('--include-staged', action='store_true', help='include the legacy decomposed projection/WY/bank route')
    parser.add_argument('--compare-strategies', action='store_true', help='also include auto and head-fused protected routes')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print validated configuration without importing FLA or using CUDA')
    parser.add_argument('--output', type=Path, default=Path('profile/hyperbolic-opt/protected-benchmark.json'))
    args = parser.parse_args()
    validate_arguments(parser, args)
    if max(args.key_dim, args.value_dim) > 128 or not 1 <= args.rank <= 16:
        parser.error('protected kernels require K,V<=128 and rank in [1,16]')
    if not 0 < args.curvature <= 2 or args.temperature < 1e-3:
        parser.error('the protected reference gate requires curvature in (0,2] and temperature >=1e-3')
    candidates = methods(args)
    payload = {'status': 'CONFIGURED', 'metadata': metadata(args), 'methods': [
        {'name': x.name, 'environment': x.environment, 'expected_kernel': x.expected_kernel,
         'expected_flags': x.expected_flags, 'baseline': x.baseline} for x in candidates
    ]}
    if args.dry_run:
        payload['status'] = 'DRY RUN; GPU NOT RUN'
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
        inputs = make_inputs(args)
        payload['input_dtypes'] = {group: [str(x.dtype) for x in data] for group, data in inputs.items()}
        payload['correctness'] = correctness(args, candidates, inputs)
        payload['status'] = 'GPU CORRECTNESS PASSED'
        write_json(args.output, payload)
        if not args.check_only:
            payload['measurements'] = measure(candidates, inputs, args.modes, args.warmup, args.repeats, args.batch * args.seq_len)
            payload['status'] = 'GPU CORRECTNESS AND BENCHMARK COMPLETED'
        write_json(args.output, payload)
    except Exception:
        payload['status'] = 'GPU VALIDATION OR BENCHMARK FAILED'
        payload['failure'] = traceback.format_exc()
        write_json(args.output, payload)
        raise
    print(f'Results: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
