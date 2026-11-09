#!/usr/bin/env python3
"""GPU correctness and timing gate for Exact Group RHS Fusion.

Run on a qualified H20/SM90 environment, for example:

  python benchmarks/benchmark_khdn_group_rhs_fusion.py --warmup 10 --repeat 30 \
      --output group_rhs_fusion_h20.json

Without a qualified GPU the script prints ``GPU NOT RUN`` and exits 0.  It
never treats a CPU/reference result as GPU validation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from fla.ops.key_gated_hyperbolic_delta_rule import chunk_key_gated_hyperbolic_delta_rule


def make_inputs(batch: int, length: int, heads: int, value_dim: int, device: torch.device):
    torch.manual_seed(20261109)
    q = torch.randn(batch, length, heads, 128, device=device, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn_like(q).float(), dim=-1
    ).to(torch.bfloat16)
    v = (0.06 * torch.tanh(torch.randn(batch, length, heads, value_dim,
                                        device=device, dtype=torch.float32))).to(torch.bfloat16)
    beta = (0.15 + 0.7 * torch.rand(batch, length, heads,
                                    device=device, dtype=torch.float32)).to(torch.bfloat16)
    ell = (-0.03 - 0.2 * torch.rand(batch, length, heads, 128,
                                    device=device, dtype=torch.float32)).to(torch.bfloat16)
    state = torch.zeros(batch, heads, 128, value_dim, device=device, dtype=torch.float32)
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, ell, state))


def run_once(inputs, fused: bool, *, chunk_size: int):
    q, k, v, beta, ell, state = inputs
    telemetry = {}
    out, final = chunk_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, ell, curvature=0.5, initial_state=state,
        output_final_state=True, group_size=32, chunk_size=chunk_size,
        precision_policy='bf16_selective', source_vjp='expanded',
        group_rhs_fusion=fused, telemetry=telemetry,
    )
    loss = out.float().square().mean() + final.float().square().mean()
    grads = torch.autograd.grad(loss, (q, k, v, beta, ell, state), allow_unused=False)
    return out.detach(), final.detach(), tuple(x.detach() for x in grads), telemetry


def max_error(a, b):
    return max(float((x.float() - y.float()).abs().max().item()) for x, y in zip(a, b))


def timed(inputs, fused: bool, warmup: int, repeat: int, chunk_size: int):
    for _ in range(warmup):
        run_once(inputs, fused, chunk_size=chunk_size)
    torch.cuda.synchronize()
    samples = []
    last = None
    for _ in range(repeat):
        start = time.perf_counter()
        last = run_once(inputs, fused, chunk_size=chunk_size)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return last, {
        'median_ms': statistics.median(samples),
        'p95_ms': sorted(samples)[max(0, min(len(samples) - 1, int(0.95 * len(samples)) - 1))],
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--value-dim', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--atol', type=float, default=5e-2)
    parser.add_argument('--rtol', type=float, default=5e-2)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        result = {'status': 'GPU NOT RUN', 'reason': 'CUDA is unavailable'}
        print(json.dumps(result, indent=2))
        return 0
    device = torch.device('cuda')
    if torch.cuda.get_device_capability(device) != (9, 0):
        result = {'status': 'GPU NOT RUN', 'reason': f'unqualified capability {torch.cuda.get_device_capability(device)}'}
        print(json.dumps(result, indent=2))
        return 0

    inputs = make_inputs(args.batch, args.length, args.heads, args.value_dim, device)
    legacy, legacy_timing = timed(inputs, False, args.warmup, args.repeat, args.chunk_size)
    fused, fused_timing = timed(inputs, True, args.warmup, args.repeat, args.chunk_size)
    legacy_out, legacy_final, legacy_grads, legacy_meta = legacy
    fused_out, fused_final, fused_grads, fused_meta = fused
    output_error = max_error((legacy_out, legacy_final), (fused_out, fused_final))
    gradient_error = max_error(legacy_grads, fused_grads)
    if output_error > args.atol + args.rtol * float(legacy_out.float().abs().max().item()):
        raise AssertionError(f'Group RHS fused output mismatch: {output_error}')
    if gradient_error > args.atol + args.rtol * max(float(x.float().abs().max().item()) for x in legacy_grads):
        raise AssertionError(f'Group RHS fused gradient mismatch: {gradient_error}')
    if legacy_meta.get('group_rhs_fusion') is not False or fused_meta.get('group_rhs_fusion') is not True:
        raise AssertionError(f'wrong dispatch metadata: legacy={legacy_meta} fused={fused_meta}')
    if legacy_meta.get('group_feedback_tiles_launches') != 1 or fused_meta.get('group_feedback_tiles_launches', 0) != 0:
        raise AssertionError(f'wrong feedback launch metadata: legacy={legacy_meta} fused={fused_meta}')

    result = {
        'status': 'GPU PASS',
        'device': torch.cuda.get_device_name(device),
        'shape': [args.batch, args.length, args.heads, 128, args.value_dim],
        'chunk_size': args.chunk_size,
        'output_max_abs_error': output_error,
        'gradient_max_abs_error': gradient_error,
        'legacy': legacy_timing,
        'fused': fused_timing,
        'speedup_legacy_over_fused': legacy_timing['median_ms'] / fused_timing['median_ms'],
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
