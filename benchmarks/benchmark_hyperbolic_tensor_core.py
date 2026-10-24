# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness-gated old, recurrent, Tensor Core and staged HDN benchmarks.

Example:
    python benchmarks/benchmark_hyperbolic_tensor_core.py --seq-len 4096 --layer-inputs --modes fwd bwd fwd_bwd
"""

import argparse
import json
import math
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
        batch, heads = min(batch, 2), min(heads, 2)
        length, value_dim = min(length, 2 * args.chunk_size + 1), min(value_dim, 32)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    q = torch.nn.functional.normalize(torch.randn(batch, length, heads, key_dim, device='cuda'), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1)
    v = 0.15 * torch.randn(batch, length, heads, value_dim, device='cuda').tanh()
    if not args.layer_inputs:
        k, v = k.to(dtype), v.to(dtype)
    beta = torch.rand_like(v, dtype=torch.float32)
    state = 0.05 * torch.nn.functional.normalize(torch.randn(batch, heads, key_dim, value_dim, device='cuda'), dim=-2)
    c = torch.tensor(args.curvature, device='cuda')
    return q, k, v, beta, state, c


def run(args, inputs, mode):
    from fla.ops.hyperbolic_delta_rule.chunk import chunk_hyperbolic_delta_rule
    from fla.ops.hyperbolic_delta_rule.fused_recurrent import fused_recurrent_hyperbolic_delta_rule
    from fla.ops.hyperbolic_delta_rule.naive import naive_recurrent_hyperbolic_delta_rule
    common = dict(curvature=inputs[5], initial_state=inputs[4], output_final_state=True, scale=args.scale)
    if mode == 'reference':
        return naive_recurrent_hyperbolic_delta_rule(*inputs[:4], **common)
    if mode == 'recurrent':
        return fused_recurrent_hyperbolic_delta_rule(*inputs[:4], checkpoint_size=args.chunk_size, **common)
    return chunk_hyperbolic_delta_rule(
        *inputs[:4], chunk_size=args.chunk_size, tensor_core=mode == 'tc', **common,
    )


def methods(args):
    tc_supported = args.key_dim <= 128
    baseline = 'tensor_core_wy' if tc_supported else 'tensor_core_fallback'
    result = [
        Method('old_chunk', lambda x: run(args, x, 'chunk'), FLAGS.copy(), 'unprotected',
               'hyperbolic_delta_rule_chunk_fwd_kernel'),
        Method(baseline, lambda x: run(args, x, 'tc'), FLAGS.copy(), 'unprotected',
               'hyperbolic_delta_tc_chunk_fwd' if tc_supported else 'hyperbolic_delta_rule_chunk_fwd_kernel',
               {'FAST': False} if tc_supported else None, 'old_chunk'),
        Method('staged_wy' if tc_supported else 'staged_fallback', lambda x: run(args, x, 'tc'),
               {**FLAGS, 'FLA_HYPERBOLIC_STAGED': '1'}, 'unprotected',
               'hyperbolic_staged_prepare_fwd_kernel' if tc_supported else 'hyperbolic_delta_rule_chunk_fwd_kernel',
               baseline=baseline),
        Method('fused_recurrent', lambda x: run(args, x, 'recurrent'), FLAGS.copy(), 'unprotected',
               'hyperbolic_delta_rule_fwd_kernel', baseline=baseline),
    ]
    if args.legacy_fast:
        result.append(Method('tensor_core_legacy_fast', lambda x: run(args, x, 'tc'),
                             {**FLAGS, 'FLA_HYPERBOLIC_TC_FAST': '1'}, 'unprotected',
                             'hyperbolic_delta_tc_chunk_fwd', {'FAST': True}, baseline))
    return result


def _attest_training(method, trace):
    attest(method, trace)
    if method.name == 'staged_wy':
        required = ('hyperbolic_staged_state_fwd_kernel', 'hyperbolic_staged_output_fwd_kernel',
                    'hyperbolic_staged_output_bwd_kernel', 'hyperbolic_staged_state_bwd_kernel',
                    'hyperbolic_staged_values_bwd_kernel', 'hyperbolic_staged_parameters_bwd_kernel')
    elif method.expected_kernel == 'hyperbolic_delta_tc_chunk_fwd':
        required = ('hyperbolic_delta_tc_chunk_bwd',)
    else:
        required = ('hyperbolic_delta_rule_bwd_kernel',)
    for name in required:
        if not any(record['kernel'].endswith(name) for record in trace):
            raise AssertionError(f'{method.name}: required training kernel {name} was not launched')
    if method.expected_kernel == 'hyperbolic_delta_tc_chunk_fwd':
        expected_fast = method.expected_flags['FAST']
        if any(record['flags'].get('FAST') != expected_fast for record in trace
               if record['kernel'].endswith('hyperbolic_delta_tc_chunk_bwd')):
            raise AssertionError(f'{method.name}: backward did not preserve the requested FAST policy')
    return trace


def _compare_reference(args, method, actual, oracle):
    legacy = method.name in ('old_chunk', 'fused_recurrent') or args.key_dim > 128
    if legacy:
        out_tol = {'fp32': (2e-4, 2e-3), 'fp16': (8e-4, 8e-3), 'bf16': (4e-3, 4e-2)}[args.dtype]
    else:
        out_tol = {'fp32': (8e-4, 8e-3), 'fp16': (1e-2, 3e-2), 'bf16': (3e-2, 8e-2)}[args.dtype]
    errors = compare(actual, oracle, out_tol, (out_tol[0] * 3, out_tol[1] * 3), cast_reference=True)
    if legacy:
        # The existing recurrent/chunk tests keep FP32 state tolerance independent of output storage dtype.
        torch.testing.assert_close(actual['outputs'][1], oracle['outputs'][1], atol=2e-4, rtol=2e-3)
        errors['outputs_1'].update(atol=2e-4, rtol=2e-3)
    return errors


def correctness(args, candidates, inputs):
    evidence = {'small_reference': {}, 'requested_shape': {}}
    small = make_inputs(args, reference_case=True)
    oracle, _ = evaluate(lambda x: run(args, x, 'reference'), tuple(x.double() for x in small), args.seed + 1)
    for method in candidates:
        with environment(method.environment):
            actual, trace = evaluate(method.run, small, args.seed + 1, trace=True)
        _attest_training(method, trace)
        errors = _compare_reference(args, method, actual, oracle)
        evidence['small_reference'][method.name] = {'errors': errors, 'kernel_launches': trace}
    full = {}
    for method in candidates:
        with environment(method.environment):
            actual, trace = evaluate(method.run, inputs, args.seed + 1, trace=True)
        _attest_training(method, trace)
        row = {'kernel_launches': trace}
        if method.baseline:
            out_tol = {'fp32': (8e-4, 8e-3), 'fp16': (1e-2, 3e-2), 'bf16': (3e-2, 8e-2)}[args.dtype]
            row['errors_against_baseline'] = compare(actual, full[method.baseline], out_tol, (out_tol[0] * 3, out_tol[1] * 3))
        evidence['requested_shape'][method.name] = row
        full[method.name] = actual
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (('batch', 1), ('seq-len', 4096), ('heads', 8), ('key-dim', 64), ('value-dim', 64)):
        parser.add_argument('--' + flag, type=int, default=default)
    parser.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='bf16')
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument(
        '--layer-inputs', action='store_true', help='keep prepared K/V in FP32; Q storage is selected by --dtype',
    )
    parser.add_argument('--curvature', type=float, default=0.5)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backward', action='store_true', help='compatibility alias for --modes fwd_bwd')
    parser.add_argument('--modes', choices=('fwd', 'bwd', 'fwd_bwd'), nargs='+')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument(
        '--legacy-fast', action='store_true', help='also compare the existing all-GEMM TC_FAST precision trade-off',
    )
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('profile/hyperbolic-opt/tensor-core-benchmark.json'))
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.key_dim > 256:
        parser.error('the existing operator supports K<=256; K>128 uses its chunk fallback')
    if args.legacy_fast and (args.dtype == 'fp32' or args.key_dim > 128):
        parser.error('--legacy-fast requires FP16/BF16 Query storage and K<=128, otherwise FAST would not execute')
    if not 0 <= args.curvature <= 2 or (args.scale is not None and not math.isfinite(args.scale)):
        parser.error('the benchmark requires curvature in [0,2] and a finite optional scale')
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
        payload['input_dtypes'] = [str(x.dtype) for x in inputs]
        payload['correctness'] = correctness(args, candidates, inputs)
        payload['status'] = 'GPU CORRECTNESS PASSED'
        write_json(args.output, payload)
        if not args.check_only:
            payload['measurements'] = measure(
                candidates, {'unprotected': inputs}, args.modes, args.warmup, args.repeats, args.batch * args.seq_len,
            )
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
