# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Microbenchmark protected vs. unprotected HyperbolicDeltaNet CUDA paths.

Example:
  python benchmarks/benchmark_protected_hyperbolic_delta_rule.py --seq-len 4096 --heads 8 --key-dim 64 --value-dim 64 --rank 4
"""

import argparse
import statistics
import time

import torch

from fla.ops.hyperbolic_delta_rule.chunk import chunk_hyperbolic_delta_rule
from fla.ops.hyperbolic_delta_rule.protected import protected_recurrent_hyperbolic_delta_rule


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    peak = 0
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        peak = max(peak, torch.cuda.max_memory_allocated())
    return statistics.median(samples), peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=4096)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--key-dim', type=int, default=64)
    parser.add_argument('--value-dim', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark.')
    device = 'cuda'
    B, T, H, K, V = args.batch, args.seq_len, args.heads, args.key_dim, args.value_dim
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float16)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = 0.05 * torch.randn(B, T, H, V, device=device, dtype=torch.float16)
    beta = torch.rand(B, T, H, V, device=device, dtype=torch.float32)
    state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    def baseline():
        args_in = [x.detach().requires_grad_(args.backward) for x in (q, k, v, beta, state)]
        out, final = chunk_hyperbolic_delta_rule(
            *args_in[:4], curvature=0.5, initial_state=args_in[4],
            output_final_state=True, chunk_size=args.chunk_size,
        )
        if args.backward:
            (out.float().square().mean() + final.square().mean()).backward()
        return out, final

    def protected():
        args_in = [x.detach().requires_grad_(args.backward) for x in (q, k, v, beta, state)]
        out = protected_recurrent_hyperbolic_delta_rule(
            *args_in[:4], curvature=0.5, initial_state=args_in[4],
            protection_rank=args.rank, checkpoint_size=args.chunk_size,
            mode='triton', checkpoint=True,
        )
        if args.backward:
            (out[0].float().square().mean() + out[1].square().mean()).backward()
        return out[0], out[1]

    for name, fn in (('unprotected_chunk', baseline), ('protected_chunk', protected)):
        ms, peak = measure(fn, args.warmup, args.repeats)
        tokens_s = B * T * 1000.0 / ms
        print(f'{name}: {ms:.3f} ms, {tokens_s:.0f} tokens/s, peak={peak / 2**30:.3f} GiB')


if __name__ == '__main__':
    main()
