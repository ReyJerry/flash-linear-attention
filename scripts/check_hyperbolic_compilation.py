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
import platform
import re
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('shared-inverse', 'protected-staged', 'staged', 'tensor-core'),
                        default='shared-inverse')
    parser.add_argument('--arch', type=int, choices=(80, 90, 100), default=90)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    parser.add_argument('--key-dim', type=int, default=128)
    parser.add_argument('--value-dim', type=int, default=128)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32), default=16)
    parser.add_argument('--layer-inputs', action='store_true', help='FP32 Key/Value with Query in --dtype')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=16384)
    parser.add_argument('--cu-dtype', choices=('int32', 'int64'), default='int64', help='dense public calls use int64')
    parser.add_argument('--constant-curvature', action='store_true', help='compile staged NEED_DC=False')
    parser.add_argument('--resources', action='store_true',
                        help='record offline ptxas registers/spills; does not measure occupancy')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.key_dim <= 128 or not 1 <= args.value_dim <= 128 or not 1 <= args.rank <= 16:
        parser.error('requires 1 <= K,V <= 128 and 1 <= rank <= 16')
    if min(args.batch, args.heads, args.seq_len) <= 0:
        parser.error('batch, heads and seq-len must be positive')
    cache = Path(__file__).resolve().parents[1] / 'profile/hyperbolic-opt/triton-cache'
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache))

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as compile_kernel

    if args.component == 'shared-inverse':
        from fla.ops.hyperbolic_delta_rule.protected_kernels import shared_key_gram_fwd_kernel
        from fla.ops.hyperbolic_delta_rule.protected_value_tile_kernels import (
            protected_value_tile_bwd_kernel,
            protected_value_tile_fwd_kernel,
        )
        kernels = (shared_key_gram_fwd_kernel, protected_value_tile_fwd_kernel, protected_value_tile_bwd_kernel)
    elif args.component == 'protected-staged':
        from fla.ops.hyperbolic_delta_rule import protected_staged_kernels
        from fla.ops.hyperbolic_delta_rule.protected_kernels import shared_key_gram_fwd_kernel
        kernels = (shared_key_gram_fwd_kernel,) + tuple(getattr(protected_staged_kernels, f'protected_{name}_kernel') for name in (
            'shared_trajectory_fwd', 'value_trajectory_fwd', 'value_trajectory_bwd',
            'value_partials_bwd', 'bank_partials_bwd', 'shared_trajectory_bwd',
        ))
    elif args.component == 'staged':
        from fla.ops.hyperbolic_delta_rule import staged_kernels
        kernels = tuple(getattr(staged_kernels, f'hyperbolic_staged_{name}_kernel') for name in (
            'prepare_fwd', 'state_fwd', 'output_fwd', 'output_bwd', 'state_bwd', 'values_bwd', 'parameters_bwd',
        ))
    else:
        from fla.ops.hyperbolic_delta_rule.tensor_core_kernels import (
            hyperbolic_delta_tc_chunk_bwd,
            hyperbolic_delta_tc_chunk_fwd,
        )
        kernels = (hyperbolic_delta_tc_chunk_fwd, hyperbolic_delta_tc_chunk_bwd)
    constants = dict(
        T=args.chunk_size, H=args.heads, K=args.key_dim, V=args.value_dim, R=args.rank,
        SCALE=args.key_dim ** -0.5, EPS=1e-5,
        BR=max(16, triton.next_power_of_2(args.rank)), BK=max(16, triton.next_power_of_2(args.key_dim)),
        NC=triton.cdiv(args.seq_len, args.chunk_size), C=args.chunk_size, TEMP=0.25,
        P=args.batch * args.heads * args.value_dim, BV=triton.next_power_of_2(args.value_dim), BLOCK=32,
        N=args.batch, BT=args.chunk_size, NEED_DC=not args.constant_curvature,
        TOTAL=args.batch * args.seq_len, SAVE=True, FAST=False, HALF=args.dtype == 'fp16', BF16=args.dtype == 'bf16',
    )
    if args.component in ('staged', 'tensor-core'):
        constants['BV'] = 16
    records = []
    for kernel in kernels:
        strategies = (False, True) if 'SHARED_INVERSE' in kernel.arg_names else (None,)
        for shared in strategies:
            values = dict(constants, SHARED_INVERSE=shared)
            if kernel.__name__ in ('protected_value_trajectory_fwd_kernel', 'protected_value_trajectory_bwd_kernel'):
                values['BR'] = triton.next_power_of_2(args.rank)
            if kernel.__name__ == 'protected_value_partials_bwd_kernel':
                values['BK'] = 32
            constexprs = {p.name: values[p.name] for p in kernel.params if p.is_constexpr}
            signature = {}
            for param in kernel.params:
                if param.is_constexpr:
                    continue
                if param.name == 'Cu':
                    signature[param.name] = '*i32' if args.cu_dtype == 'int32' else '*i64'
                elif (
                    args.component in ('staged', 'tensor-core') and param.name in ('Key', 'Val')
                    or args.component == 'staged' and param.name in ('DK', 'DV')
                ):
                    signature[param.name] = '*fp32' if args.layer_inputs else f'*{args.dtype}'
                elif args.component in ('staged', 'tensor-core') and param.name in ('DQ', 'Out', 'DOut'):
                    signature[param.name] = f'*{args.dtype}'
                elif param.name in ('T', 'START', 'STOP'):
                    signature[param.name] = 'i32'
                elif param.name == 'ScratchValid':
                    signature[param.name] = '*u8'
                elif param.name in ('Keys', 'Values', 'O', 'DO'):
                    signature[param.name] = '*fp32' if args.layer_inputs else f'*{args.dtype}'
                elif param.name == 'Q':
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
                layer_inputs=args.layer_inputs, chunk_size=args.chunk_size,
                key_dim=args.key_dim, value_dim=args.value_dim, rank=args.rank,
                batch=args.batch, heads=args.heads, seq_len=args.seq_len,
                cu_dtype=args.cu_dtype if args.component in ('staged', 'tensor-core') else None,
                need_dc=not args.constant_curvature if args.component == 'staged' else None,
                arch=args.arch, status='COMPILED_ONLY_GPU_NOT_RUN',
                shared_memory_bytes=compiled.metadata.shared,
                cubin_bytes=len(compiled.asm.get('cubin', b'')),
                signature=signature, constexprs=constexprs,
            )
            if args.resources:
                # Match Triton's assembler flags for enable_fp_fusion=False.
                # This reports static resources, not measured occupancy/latency.
                ptxas = Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas'
                target = re.search(r'\.target\s+(sm_\d+a?)', compiled.asm['ptx']).group(1)
                with tempfile.TemporaryDirectory(prefix='hdn-ptxas-') as directory:
                    ptx, cubin = Path(directory) / 'kernel.ptx', Path(directory) / 'kernel.cubin'
                    ptx.write_text(compiled.asm['ptx'])
                    assembled = subprocess.run(
                        [str(ptxas), '-lineinfo', '--fmad=false', '-v', f'--gpu-name={target}', str(ptx), '-o', str(cubin)],
                        text=True, capture_output=True, check=True,
                    )
                    row['ptxas_resources'] = assembled.stderr
            records.append(row)
            print(json.dumps(row), flush=True)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, text=True, capture_output=True, check=True).stdout
    report = dict(
        status='OFFLINE_COMPILE_ONLY', component=args.component, triton=triton.__version__,
        python=platform.python_version(), revision=revision, working_tree_dirty=bool(dirty),
        compiler_options=dict(num_warps=4, enable_fp_fusion=False), cases=records,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
