#!/usr/bin/env python3
"""GPU correctness and timing gate for the block-level group action solve.

This compares the exact full-inverse route with the exact RHS/transpose action
route on the same inputs. It never attempts the rejected per-token SM path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from fla.ops.key_gated_hyperbolic_delta_rule import chunk_key_gated_hyperbolic_delta_rule


def make_inputs(batch, length, heads, value_dim, device):
    torch.manual_seed(20261110)
    q = torch.randn(batch, length, heads, 128, device=device, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
    v = (0.06 * torch.tanh(torch.randn(batch, length, heads, value_dim,
                                        device=device, dtype=torch.float32))).to(torch.bfloat16)
    beta = (0.15 + 0.7 * torch.rand(batch, length, heads,
                                    device=device, dtype=torch.float32)).to(torch.bfloat16)
    ell = (-0.03 - 0.2 * torch.rand(batch, length, heads, 128,
                                    device=device, dtype=torch.float32)).to(torch.bfloat16)
    state = torch.zeros(batch, heads, 128, value_dim, device=device, dtype=torch.float32)
    return tuple(x.detach().requires_grad_() for x in (q, k, v, beta, ell, state))


def run_once(inputs, action, chunk_size, options):
    q, k, v, beta, ell, state = inputs
    telemetry = {}
    out, final = chunk_key_gated_hyperbolic_delta_rule(
        q, k, v, beta, ell, curvature=0.5, initial_state=state,
        output_final_state=True, group_size=32, chunk_size=chunk_size,
        precision_policy='bf16_selective', source_vjp=options.source_vjp,
        value_tile=options.value_tile, state_value_tile=options.state_value_tile,
        state_residual_mode=options.state_residual_mode,
        group_rhs_fusion=True, group_action_solve=action, telemetry=telemetry,
    )
    loss = out.float().square().mean() + final.float().square().mean()
    grads = torch.autograd.grad(loss, (q, k, v, beta, ell, state), allow_unused=False)
    return out.detach(), final.detach(), tuple(x.detach() for x in grads), telemetry


def max_error(a, b):
    return max(float((x.float() - y.float()).abs().max().item()) for x, y in zip(a, b))


def timed(inputs, action, warmup, repeat, chunk_size, options):
    for _ in range(warmup):
        run_once(inputs, action, chunk_size, options)
    torch.cuda.synchronize()
    samples, last = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        last = run_once(inputs, action, chunk_size, options)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return last, {
        'median_ms': statistics.median(samples),
        'p95_ms': ordered[max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))],
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
    parser.add_argument('--chunk-size', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--value-tile', type=int, choices=(32, 64, 128), default=32)
    parser.add_argument('--state-value-tile', type=int, choices=(32, 64, 128), default=32)
    parser.add_argument('--state-residual-mode', choices=('stable', 'direct'), default='stable')
    parser.add_argument('--source-vjp', choices=('expanded', 'coupled_state', 'coupled_fused'), default='expanded')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--atol', type=float, default=7e-2)
    parser.add_argument('--rtol', type=float, default=7e-2)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({'status': 'GPU NOT RUN', 'reason': 'CUDA is unavailable'}, indent=2))
        return 0
    device = torch.device('cuda')
    if torch.cuda.get_device_capability(device) != (9, 0):
        print(json.dumps({'status': 'GPU NOT RUN', 'reason': f'unqualified capability {torch.cuda.get_device_capability(device)}'}, indent=2))
        return 0

    inputs = make_inputs(args.batch, args.length, args.heads, args.value_dim, device)
    inverse, inverse_timing = timed(inputs, False, args.warmup, args.repeat, args.chunk_size, args)
    action, action_timing = timed(inputs, True, args.warmup, args.repeat, args.chunk_size, args)
    inverse_out, inverse_final, inverse_grads, inverse_meta = inverse
    action_out, action_final, action_grads, action_meta = action
    output_error = max_error((inverse_out, inverse_final), (action_out, action_final))
    gradient_error = max_error(inverse_grads, action_grads)
    scale = max(float(inverse_out.float().abs().max().item()), 1.0)
    grad_scale = max(max(float(x.float().abs().max().item()) for x in inverse_grads), 1.0)
    if output_error > args.atol + args.rtol * scale:
        raise AssertionError(f'action solve output mismatch: {output_error}')
    if gradient_error > args.atol + args.rtol * grad_scale:
        raise AssertionError(f'action solve gradient mismatch: {gradient_error}')
    if inverse_meta.get('triangular_route') != 'full_inverse' or action_meta.get('triangular_route') != 'action_solve':
        raise AssertionError(f'wrong triangular dispatch: inverse={inverse_meta} action={action_meta}')

    result = {
        'status': 'GPU PASS',
        'device': torch.cuda.get_device_name(device),
        'shape': [args.batch, args.length, args.heads, 128, args.value_dim],
        'chunk_size': args.chunk_size,
        'output_max_abs_error': output_error,
        'gradient_max_abs_error': gradient_error,
        'full_inverse': inverse_timing,
        'action_solve': action_timing,
        'speedup_full_inverse_over_action': inverse_timing['median_ms'] / action_timing['median_ms'],
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
