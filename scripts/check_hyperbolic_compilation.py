# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Offline Triton compilation; this does not launch or validate a CUDA kernel."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('shared-inverse',), default='shared-inverse')
    parser.add_argument('--arch', type=int, choices=(80, 90, 100), default=90)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--key-dim', type=int, default=128)
    parser.add_argument('--value-dim', type=int, default=128)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.key_dim <= 128 or not 1 <= args.value_dim <= 128 or not 1 <= args.rank <= 16:
        parser.error('requires 1 <= K,V <= 128 and 1 <= rank <= 16')
    cache = Path(__file__).resolve().parents[1] / 'profile/hyperbolic-opt/triton-cache'
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache))

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as compile_kernel

    from fla.ops.hyperbolic_delta_rule.protected_kernels import shared_key_gram_fwd_kernel
    from fla.ops.hyperbolic_delta_rule.protected_value_tile_kernels import (
        protected_value_tile_bwd_kernel,
        protected_value_tile_fwd_kernel,
    )

    kernels = (shared_key_gram_fwd_kernel, protected_value_tile_fwd_kernel, protected_value_tile_bwd_kernel)
    constants = dict(
        T=args.chunk_size, H=2, K=args.key_dim, V=args.value_dim, R=args.rank,
        SCALE=args.key_dim ** -0.5, EPS=1e-5,
        BR=max(16, triton.next_power_of_2(args.rank)), BK=max(16, triton.next_power_of_2(args.key_dim)),
    )
    records = []
    for kernel in kernels:
        strategies = (False, True) if 'SHARED_INVERSE' in kernel.arg_names else (None,)
        for shared in strategies:
            values = dict(constants, SHARED_INVERSE=shared)
            constexprs = {p.name: values[p.name] for p in kernel.params if p.is_constexpr}
            signature = {}
            for param in kernel.params:
                if param.is_constexpr:
                    continue
                if param.name == 'ScratchValid':
                    signature[param.name] = '*u8'
                elif param.name in ('Q', 'Keys', 'Values', 'O', 'DO'):
                    signature[param.name] = f'*{args.dtype}'
                else:
                    signature[param.name] = '*fp32'
            compiled = compile_kernel(
                ASTSource(kernel, signature=signature, constexprs=constexprs),
                target=GPUTarget('cuda', args.arch, 32),
                options=dict(num_warps=4, enable_fp_fusion=False),
            )
            row = dict(
                kernel=kernel.__name__, shared_inverse=shared, dtype=args.dtype,
                arch=args.arch, status='COMPILED_ONLY_GPU_NOT_RUN',
                shared_memory_bytes=compiled.metadata.shared,
                cubin_bytes=len(compiled.asm.get('cubin', b'')),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    report = dict(status='OFFLINE_COMPILE_ONLY', triton=triton.__version__, cases=records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
