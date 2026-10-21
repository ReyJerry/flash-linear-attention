# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run only after CUDA correctness tests; no performance results are prefilled."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', default='128,512,2048')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--value-dim', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; no timings were measured.')
    if min(args.heads, args.head_dim, args.value_dim, args.repeats) <= 0:
        parser.error('Dimensions and repeats must be positive.')
    from fla.ops.hyperbolic_delta_rule import chunk_hyperbolic_delta_rule, fused_recurrent_hyperbolic_delta_rule
    torch.manual_seed(42)
    print(f'device={torch.cuda.get_device_name()} torch={torch.__version__} backward={args.backward}')
    print('mode,tokens,milliseconds,ktokens_per_second,peak_allocated_mib')
    for length in (int(x) for x in args.lengths.split(',')):
        if length <= 0:
            parser.error('All sequence lengths must be positive.')
        q = torch.randn(1, length, args.heads, args.head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1)
        v = 0.25 * torch.randn(1, length, args.heads, args.value_dim, device='cuda').tanh()
        beta = torch.rand(1, length, args.heads, device='cuda')
        inputs = tuple(x.detach().requires_grad_(args.backward) for x in (q, k, v, beta))
        for op in (fused_recurrent_hyperbolic_delta_rule, chunk_hyperbolic_delta_rule):
            def run():
                for x in inputs:
                    x.grad = None
                y, _ = op(*inputs)
                if args.backward:
                    y.float().square().mean().backward()
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                run()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / args.repeats
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f'{op.__name__},{length},{ms:.4f},{length / ms:.3f},{peak:.2f}')


if __name__ == '__main__':
    main()
