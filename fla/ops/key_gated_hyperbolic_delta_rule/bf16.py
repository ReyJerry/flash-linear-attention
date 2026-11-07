# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""First-order BF16 contractions with FP32 control/state and no high-precision replay."""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import torch
import triton

from fla.ops.key_gated_hyperbolic_delta_rule import bf16_kernels as kernels
from fla.ops.key_gated_hyperbolic_delta_rule.naive import _validate
from fla.utils import IS_NVIDIA, autocast_custom_bwd, autocast_custom_fwd, input_guard


@lru_cache(maxsize=1)
def _compiler_digest():
    from triton._C import libtriton
    with Path(libtriton.__file__).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_runtime(q):
    if not q.is_cuda or not IS_NVIDIA:
        raise ValueError('bf16 KHDN requires NVIDIA CUDA; high-precision fallback is disabled.')
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError('bf16 KHDN requires the qualified SM90 device; no strict fallback is permitted.')
    if torch.__version__ != '2.9.0+cu128' or triton.__version__ != '3.5.0':
        raise ValueError('bf16 KHDN requires the qualified PyTorch/Triton runtime.')
    if _compiler_digest() != '1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32':
        raise ValueError('bf16 KHDN requires the qualified compiler binary; no strict fallback is permitted.')


def buffers(count, shapes, device):
    return tuple(torch.empty((count, *s), dtype=torch.float32, device=device) for s in shapes)


def shapes(c, k):
    common = [(c, k)] * 9 + [(c, c)] * 7 + [(k, c)] * 2 + [(c,)] * 5
    group = [(c, c)] * 5 + [(c,)] * 3
    shared = [(c, c), (c, c), (c, k), (c, c), (c,), (c, k), (c, c), (k, c),
              (c,), (c,), (c,), (k,), (c,), (k,), (c, k)]
    return common, group, shared


def launch(kernel, grid, args, telemetry, **kwargs):
    compiled = kernel[grid](*args, **kwargs)
    ir, ptx = compiled.asm['ttir'], compiled.asm['ptx']
    if re.search(r'\bf64\b|xf64\b', ir) or re.search(r'\.f64\b', ptx):
        raise RuntimeError('FP64 compiled arithmetic detected: ' + kernel.__name__)
    if telemetry is not None:
        telemetry['native_kernel_calls'] = telemetry.get('native_kernel_calls', 0) + 1
    return compiled


class _ChunkBF16(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, ell, state, curve, opts, telemetry):
        sources = (q, k, v, beta, ell, state, curve)
        _, _, h, kd = q.shape
        vd, gs, c = v.shape[-1], opts['g'], opts['c']
        ng, kt, gt = vd // gs, triton.next_power_of_2(kd), triton.next_power_of_2(gs)
        starts, ends, seq_chunks = [], [], [0]
        for bos, eos in zip(opts['bounds'], opts['bounds'][1:]):
            for pos in range(bos, eos, c):
                starts.append(pos)
                ends.append(min(pos+c, eos))
            seq_chunks.append(len(starts))
        prev, nxt = [-1]*len(starts), [-1]*len(starts)
        for first, last in zip(seq_chunks, seq_chunks[1:]):
            for j in range(first, last):
                prev[j] = j-1 if j > first else -1
                nxt[j] = j+1 if j+1 < last else -1
        meta = tuple(torch.tensor(x, dtype=torch.int64, device=q.device) for x in (starts, ends, seq_chunks, prev, nxt))
        nc = len(starts)
        cs, gr, _ = shapes(c, kt)
        p, gp = buffers(nc*h, cs, q.device), buffers(nc*h*ng, gr, q.device)
        carry = buffers(nc*h if opts['carry'] else 0, [(c, kt), (c, c)], q.device)
        saved_s, saved_y = buffers(nc*h*ng, [(kt, gt), (c, gt)], q.device)
        saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp = buffers(
            nc*h*ng, [(c, gt), (c,), (c,), (c, gt), (c,), (c,)], q.device)
        output = torch.empty_like(v, dtype=q.dtype)
        final = torch.empty_like(state, dtype=torch.float32)
        if nc:
            launch(kernels.common_base_fwd_kernel, (nc*h,), (q, k, beta, ell, *
                   meta[:2], p, h, kd, c, kt, opts['scale']), telemetry, num_warps=4)
            launch(kernels.common_interval_fwd_kernel, (nc*h*c,),
                   (ell, *meta[:2], p, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.common_system_fwd_kernel, (nc*h,),
                   (*meta[:2], p, h, c, kt), telemetry, num_warps=4)
            if opts['carry']:
                launch(kernels.carry_fwd_kernel, (nc*h,), (meta[1], meta[0],
                       meta[4], p, carry, h, c, kt), telemetry, num_warps=4)
            launch(kernels.group_fwd_kernel, (nc*h*ng,), (v, curve, *
                   meta[:2], p, gp, h, vd, gs, ng, c, gt), telemetry, num_warps=4)
            launch(kernels.group_af_fwd_kernel, (nc*h*ng, c), (p, gp, ng, c), telemetry, num_warps=4)
        launch(kernels.state_fwd_kernel, ((len(seq_chunks)-1)*h*ng,),
               (v, state, output, final, *meta[:3], p, gp, saved_s, saved_y, saved_u, saved_r, saved_t, saved_w, saved_zp, carry,
                h, kd, vd, gs, ng, c, kt, gt, True, opts['carry'], False), telemetry, BF16_FS6=False, num_warps=4)
        if nc:
            launch(kernels.output_fwd_kernel, (nc*h*ng,),
                   (v, output, *meta[:2], p, gp, saved_s, saved_u, saved_r, saved_t, saved_ro,
                    h, vd, gs, ng, c, kt, gt), telemetry, num_warps=4)
        if not torch.isfinite(output).all() or not torch.isfinite(final).all():
            raise RuntimeError('nonfinite candidate forward; FP64 fallback forbidden')
        ctx.save_for_backward(*sources, *meta, *p, *gp, saved_s, saved_y, *carry,
                              saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp)
        ctx.opts, ctx.telemetry, ctx.nc = opts, telemetry, nc
        ctx.set_materialize_grads(False)
        if telemetry is not None:
            telemetry.update(backend='triton_chunk', precision_route='bf16', precision_eligible=True,
                             arithmetic='bf16_output_source_single_c16', working_dtype='float32', host_fallback=False,
                             readout_precision='fp32', state_control_precision='fp32',
                             contraction_precision='bf16_operands_fp32_accumulation', fallback_subblocks=0,
                             saved_buffer_bytes=sum(x.numel()*x.element_size() for x in (*p, *gp, saved_s, saved_y, *carry, saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp)))
        return output, final

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, ds):
        if torch.is_grad_enabled():
            raise RuntimeError('first-order gradients only')
        saved, opts = ctx.saved_tensors, ctx.opts
        sources, meta, p, gp = saved[:7], saved[7:12], saved[12:35], saved[35:43]
        saved_s, saved_y = saved[43:45]
        carry = saved[45:47]
        saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp = saved[47:53]
        local = (saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp)
        q, k, v, beta, ell, state, curve = sources
        _, _, h, kd = q.shape
        vd, gs, c = v.shape[-1], opts['g'], opts['c']
        ng, kt, gt = vd // gs, triton.next_power_of_2(kd), triton.next_power_of_2(gs)
        do = torch.zeros_like(v) if do is None else do.contiguous()
        ds = torch.zeros_like(state) if ds is None else ds.contiguous()
        assert do.dtype in (torch.float32, torch.bfloat16) and ds.dtype in (torch.float32, torch.bfloat16)
        # retain FP32 partials until aliased source contributions have been combined
        grads = [torch.empty_like(x, dtype=torch.float32) for x in sources]
        dc = torch.empty((ctx.nc*h*ng,), device=q.device, dtype=torch.float32)
        _, _, ss = shapes(c, kt)
        shared = buffers(ctx.nc*h*ng, ss[:-1], q.device) + buffers(ctx.nc*h*ng if opts['carry'] else 0, ss[-1:], q.device)
        gauge = buffers(ctx.nc*h, [(c,), (c,)], q.device)
        output_adj = buffers(ctx.nc*h*ng, [(kt, gt), (c, gt), (c,), (c,)], q.device)
        edge_y = buffers(ctx.nc*h*ng if opts['carry'] else 0, [(c, gt)], q.device)[0]
        if ctx.nc:
            launch(kernels.output_bwd_kernel, (ctx.nc*h*ng,),
                   (v, do, grads[2], *meta[:2], p, gp, saved_s, saved_u, saved_r, saved_t, saved_ro, shared, output_adj,
                    h, kd, vd, gs, ng, c, kt, gt), ctx.telemetry, num_warps=4)
        launch(kernels.state_bwd_kernel, (state.shape[0]*h*ng,),
               (v, curve, ds, grads[2], grads[5], *meta[:3], p, gp, saved_s, local, shared, output_adj, carry, edge_y,
                h, kd, vd, gs, ng, c, kt, gt, opts['carry']), ctx.telemetry, num_warps=4)
        if ctx.nc:
            launch(kernels.parameter_bwd_kernel, (ctx.nc*h*ng,),
                   (v, curve, do, grads[2], dc, *meta[:2], meta[3], meta[4], p, gp, saved_s, saved_y, local,
                    shared, output_adj, edge_y, h, kd, vd, gs, ng, c, kt, gt, opts['carry']), ctx.telemetry, num_warps=4)
        if ctx.nc:
            launch(kernels.gauge_bwd_kernel, (ctx.nc*h,), (p, shared, gauge, ng, c), ctx.telemetry, num_warps=4)
            launch(kernels.common_matrix_bwd_kernel, (ctx.nc*h,),
                   (grads[0], grads[1], grads[3], grads[4], *meta[:2], meta[3], p, shared,
                    h, kd, ng, c, kt, opts['carry']), ctx.telemetry, num_warps=4)
            launch(kernels.common_interval_bwd_kernel, (ctx.nc*h,),
                   (ell, grads[0], grads[1], grads[3], grads[4], *meta[:2], p, shared, gauge,
                    h, kd, ng, c, kt, opts['scale']), ctx.telemetry, num_warps=4)
        grads[6] = dc.sum()
        if any(not torch.isfinite(x).all() for x in grads):
            raise RuntimeError('nonfinite candidate backward; FP64 fallback forbidden')
        result = [x.to(source.dtype) if need else None for x, source, need in zip(grads, sources, ctx.needs_input_grad[:7])]
        for group in opts['aliases']:
            owners = [i for i in group if ctx.needs_input_grad[i]]
            if owners:
                total = grads[group[0]]
                for i in group[1:]:
                    total = total + grads[i]
                for i in group:
                    result[i] = None
                result[owners[0]] = total.to(sources[owners[0]].dtype)
        if ctx.telemetry is not None:
            ctx.telemetry.update(backward='manual_triton_chunk_source_vjp', backward_precision='device_float32',
                                 backward_precision_route='bf16', backward_fallback_subblocks=0)
        return (*result, None, None)


def chunk_bf16(q, k, v, beta, log_decay, curvature=1., initial_state=None, output_final_state=False,
               scale=None, cu_seqlens=None, group_size=32, chunk_size=16, carry_mode='no_carry',
               arithmetic='bf16', telemetry=None):
    _require_runtime(q)
    inputs = (q, k, v, beta, log_decay) + (() if initial_state is None else (initial_state,))
    if isinstance(curvature, torch.Tensor):
        inputs += (curvature,)
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in inputs):
        raise ValueError('candidate only accepts BF16/FP32; no FP64 inputs or arithmetic')
    if not q.is_cuda or arithmetic != 'bf16' or carry_mode not in ('no_carry', 'y_carry'):
        raise ValueError('CUDA BF16/FP32 candidate and explicit carry mode required')
    if isinstance(chunk_size, bool) or chunk_size not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError('invalid execution chunk size')
    bounds, scale = _validate(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, False)
    c = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(curvature, dtype=torch.float32, device=q.device)
    state = initial_state if initial_state is not None else torch.zeros(
        (len(bounds)-1, q.shape[2], q.shape[3], v.shape[-1]), dtype=torch.float32, device=q.device)
    sources = (q, k, v, beta, log_decay, state, c)
    if any(not torch.isfinite(x).all() for x in sources) or not ((beta >= 0) & (beta <= 1)).all() or not (log_decay <= 0).all():
        raise ValueError('invalid nonfinite/gate source')
    n = k.float().square().sum(-1)
    vv = v.float().reshape(*v.shape[:-1], v.shape[-1]//group_size, group_size).square().sum(-1)
    radius = state.float().reshape(*state.shape[:-1], -1, group_size).square().sum((-3, -1))
    if not torch.isfinite(n).all() or not (n > 0).all() or not (n[..., None]-c.float()/group_size*vv > 0).all() or not (c.float()/group_size*radius < 1).all():
        raise ValueError('invalid/unrepresentable FP32 physical domain')
    aliases = {}
    for i, x in enumerate(sources):
        aliases.setdefault(id(x), []).append(i)
    opts = dict(bounds=bounds, scale=scale, g=group_size, c=chunk_size, carry=carry_mode ==
                'y_carry', arithmetic=arithmetic, aliases=[a for a in aliases.values() if len(a) > 1])
    if telemetry is not None:
        telemetry.update(source_alias_groups=opts['aliases'], precision_requested='bf16')
    output, final = _ChunkBF16.apply(*(x.contiguous() for x in sources), opts, telemetry)
    return output, final if output_final_state else None
