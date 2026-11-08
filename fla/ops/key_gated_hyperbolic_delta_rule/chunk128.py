# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""First-order BF16 contractions with FP32 control/state and no high-precision replay."""
import hashlib
import re
import weakref
from functools import lru_cache
from pathlib import Path

import torch
import triton

from fla.ops.key_gated_hyperbolic_delta_rule import chunk128_kernels as kernels
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


def buffers(count, shapes, device, dtype=torch.float32):
    return tuple(torch.empty((count, *s), dtype=dtype, device=device) for s in shapes)


def shapes(c, k):
    common = [(c, k)] * 9 + [(c, c)] * 7 + [(k, c)] * 2 + [(c,)] * 5
    group = [(c, c)] * 5 + [(c,)] * 3
    shared = [(c, c), (c, c), (c, k), (c, c), (c,), (c, k), (c, c), (k, c),
              (c,), (c,), (c,), (k,), (c,), (k,), (c, k)]
    return common, group, shared


_checked_kernels = weakref.WeakSet()


def launch(kernel, grid, args, telemetry, **kwargs):
    pipelined = kernel.__name__ in {
        'common_finish_tiles', 'common_ga_bwd', 'common_products_bwd',
        'common_inverse_left_bwd', 'shared_coefficients_bwd', 'gauge_product_bwd'}
    kwargs.setdefault("num_stages", 2 if pipelined else 1)
    compiled = kernel[grid](*args, **kwargs)
    if compiled not in _checked_kernels:
        ir, ptx = compiled.asm['ttir'], compiled.asm['ptx']
        if re.search(r'\bf64\b|xf64\b', ir) or re.search(r'\.f64\b', ptx):
            raise RuntimeError('FP64 compiled arithmetic detected: ' + kernel.__name__)
        _checked_kernels.add(compiled)
    if telemetry is not None:
        telemetry['native_kernel_calls'] = telemetry.get('native_kernel_calls', 0) + 1
    return compiled


def blocked_inverse_into(a, result, count, c, telemetry):
    launch(kernels.diagonal_inverse_kernel, (count, c//16), (a, result, c, 16), telemetry, num_warps=1)
    for width in (16, 32, 64):
        launch(kernels.inverse_offdiagonal_kernel, (count, c//(2*width)),
               (a, result, c, width, True), telemetry, num_warps=8 if width == 64 else 4)


def _all_finite(tensors):
    if any(not x.is_contiguous() for x in tensors):
        return torch.stack([torch.isfinite(x).all() for x in tensors]).all()
    sizes = tuple(x.numel() for x in tensors)
    total = sum(sizes)
    flags = torch.empty((max(1, triton.cdiv(total, 1024)),), dtype=torch.int32, device=tensors[0].device)
    launch(kernels.finite_check_kernel, (flags.numel(),), (tuple(tensors), sizes, flags, 1024), None, num_warps=4)
    return ~flags.any()


_source_validation_fallbacks = 0


def _reference_source_code(sources, group_size):
    q, k, v, beta, log_decay, state, c = sources
    source_ok = _all_finite(sources) & ((beta >= 0) & (beta <= 1)).all() & (log_decay <= 0).all()
    n = k.float().square().sum(-1)
    vv = v.float().reshape(*v.shape[:-1], v.shape[-1]//group_size, group_size).square().sum(-1)
    radius = state.float().reshape(*state.shape[:-1], -1, group_size).square().sum((-3, -1))
    domain_ok = torch.stack((torch.isfinite(n).all(), (n > 0).all(),
                             (n[..., None]-c.float()/group_size*vv > 0).all(),
                             (c.float()/group_size*radius < 1).all())).all()
    # preserve source-error precedence while reading one combined CUDA flag
    invalid = int((~source_ok).to(torch.int32) + 2 * (~domain_ok).to(torch.int32))
    return invalid


def _source_code_route(sources, group_size):
    global _source_validation_fallbacks
    q, k, v, beta, log_decay, state, c = sources
    if (k.shape[-1] != 128 or group_size != 32 or v.shape[-1] not in (32, 64, 128)
            or any(not x.is_contiguous() for x in sources)):
        _source_validation_fallbacks += 1
        return _reference_source_code(sources, group_size), False
    rows = q.numel() // 128
    input_blocks = triton.cdiv(rows, 16)
    state_blocks = state.shape[0]*state.shape[1]*(v.shape[-1]//group_size)
    flags = torch.empty((input_blocks+state_blocks,), dtype=torch.int32, device=q.device)
    if input_blocks:
        launch(kernels.input_domain_flags, (input_blocks,),
               (q, k, v, beta, log_decay, c, flags, rows, 128, v.shape[-1], group_size), None, num_warps=4)
    launch(kernels.state_domain_flags, (state_blocks,),
           (state, c, flags, input_blocks, 128, v.shape[-1], group_size), None, num_warps=4)
    partial = torch.empty((triton.cdiv(flags.numel(), 1024),), dtype=torch.int32, device=q.device)
    code = torch.empty((), dtype=torch.int32, device=q.device)
    launch(kernels.reduce_domain_flags, (partial.numel(),), (flags, partial, flags.numel(), 1024), None, num_warps=4)
    launch(kernels.reduce_domain_flags, (1,),
           (partial, code, partial.numel(), triton.next_power_of_2(partial.numel())), None, num_warps=4)
    invalid = int(code)
    if invalid & 4:
        _source_validation_fallbacks += 1
        return _reference_source_code(sources, group_size), False
    return invalid & 3, not bool(invalid & 8)


def _source_code(sources, group_size):
    return _source_code_route(sources, group_size)[0]


@lru_cache(maxsize=16)
def _make_chunk_metadata(device, bounds, chunk_size):
    starts, ends, seq_chunks = [], [], [0]
    for bos, eos in zip(bounds, bounds[1:]):
        for pos in range(bos, eos, chunk_size):
            starts.append(pos)
            ends.append(min(pos+chunk_size, eos))
        seq_chunks.append(len(starts))
    prev, nxt = [-1]*len(starts), [-1]*len(starts)
    for first, last in zip(seq_chunks, seq_chunks[1:]):
        for index in range(first, last):
            prev[index] = index-1 if index > first else -1
            nxt[index] = index+1 if index+1 < last else -1
    # Cached tensors may be reused by autograd after inference and on other streams.
    with torch.cuda.device(device), torch.inference_mode(False):
        meta = tuple(torch.tensor(value, dtype=torch.int64, device=device)
                     for value in (starts, ends, seq_chunks, prev, nxt))
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
    return meta, len(starts), ready


def chunk_metadata(device, bounds, chunk_size):
    meta, count, ready = _make_chunk_metadata(device, bounds, chunk_size)
    stream = torch.cuda.current_stream(device)
    stream.wait_event(ready)
    for tensor in meta:
        tensor.record_stream(stream)
    return meta, count


class _ChunkBF16(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, ell, state, curve, opts, telemetry):
        sources = (q, k, v, beta, ell, state, curve)
        need_curve_grad = ctx.needs_input_grad[6]
        _, _, h, kd = q.shape
        vd, gs, c = v.shape[-1], opts['g'], opts['c']
        ng, kt, gt = vd // gs, triton.next_power_of_2(kd), triton.next_power_of_2(gs)
        meta, nc = chunk_metadata(q.device, tuple(opts['bounds']), c)
        nseq = len(opts['bounds'])-1
        cs, gr, _ = shapes(c, kt)
        p = tuple(torch.empty((nc*h, *shape), dtype=torch.bfloat16 if i < 18 and i not in (2, 3, 4) else torch.float32,
                              device=q.device) for i, shape in enumerate(cs))
        diagonal_flags = torch.empty((nc*h, c//32), dtype=torch.bool, device=q.device)
        gp = tuple(torch.empty((nc*h*ng if (i != 4 or need_curve_grad) and (i != 2 or opts['rho_materialized']) else 0, *shape),
                               dtype=torch.bfloat16 if i in (0, 1, 3, 4) else torch.float32, device=q.device) for i, shape in enumerate(gr))
        carry = buffers(nc*h if opts['carry'] else 0, [(c, kt), (c, c)], q.device)
        saved_s, saved_y = buffers(nc*h*ng, [(kt, gt), (c, gt)], q.device, dtype=torch.bfloat16)
        saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp = tuple(
            torch.empty((nc*h*ng, *shape), dtype=torch.bfloat16 if index in (0, 3) else torch.float32, device=q.device)
            for index, shape in enumerate([(c, gt), (c,), (c,), (c, gt), (c,), (c,)]))
        output = torch.empty_like(v, dtype=q.dtype)
        final = torch.empty_like(state, dtype=torch.float32)
        if nc:
            launch(kernels.common_decay_kernel, (nc*h, kt//32),
                   (ell, *meta[:2], p, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.common_input_kernel, (nc*h, c//16),
                   (q, k, beta, *meta[:2], p, h, kd, c, kt, opts['scale']), telemetry, num_warps=4)
            launch(kernels.common_products_kernel, (nc*h, (c//16)**2), (p, c), telemetry, num_warps=4)
            launch(kernels.interval_cache_kernel, (nc*h, c//32, kt//32),
                   (ell, *meta[:2], p, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.diagonal_eligibility, (nc*h, c//32),
                   (ell, *meta[:2], p, diagonal_flags, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.interval_fwd_cross, (nc*h, 4),
                   (ell, *meta[:2], p, h, kd, c, kt), telemetry, num_warps=8)
            launch(kernels.interval_diagonal_tc_fwd, (nc*h, c//32),
                   (ell, *meta[:2], p, diagonal_flags, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.interval_fwd_diagonal, (nc*h, c//32),
                   (ell, *meta[:2], p, diagonal_flags, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.interval_tail_kernel, (nc*h, c//32, kt//32),
                   (ell, *meta[:2], p, h, kd, c, kt), telemetry, num_warps=4)
            launch(kernels.common_lower_kernel, (nc*h,), (p, c), telemetry, num_warps=8)
            blocked_inverse_into(p[10], p[11], nc*h, c, telemetry)
            for finish_mode in range(3):
                launch(kernels.common_finish_tiles, (nc*h, (kt if finish_mode == 2 else c)//64, (kt if finish_mode == 0 else c)//64),
                       (p, *meta[:2], h, c, kt, finish_mode), telemetry, num_warps=4)
            launch(kernels.common_tail_finish, (nc*h, c//32), (p, *meta[:2], h, c, kt), telemetry, num_warps=4)
            if opts['carry']:
                launch(kernels.carry_fwd_kernel, (nc*h,), (meta[1], meta[0],
                       meta[4], p, carry, h, c, kt), telemetry, num_warps=8)
            launch(kernels.group_gram_kernel, (nc*h*ng,),
                   (v, *meta[:2], gp, h, vd, gs, ng, c, gt), telemetry, num_warps=8)
            launch(kernels.group_gauge_kernel, (nc*h*ng,), (curve, p, gp, gs, ng, c), telemetry, num_warps=8)
            launch(kernels.group_system_tiles, (nc*h*ng, c//32),
                   (curve, p, gp, gs, ng, c, need_curve_grad), telemetry, num_warps=8, RHO_MATERIALIZED=opts['rho_materialized'])
            blocked_inverse_into(gp[1], gp[0], nc*h*ng, c, telemetry)
            launch(kernels.group_feedback_tiles, (nc*h*ng, c//32),
                   (curve, p, gp, gs, ng, c), telemetry, num_warps=4)
        # The group system is only needed to build the inverse; feedback uses its column scales directly.
        gp = (gp[0], gp[1].new_empty((0,)), *gp[2:])
        launch(kernels.state_fwd_kernel, (nseq*h*ng,),
               (v, curve, state, output, final, *meta[:3], p, gp, saved_s, saved_y, saved_u, saved_r, saved_t, saved_w, saved_zp, carry,
                h, kd, vd, gs, ng, c, kt, gt, True, opts['carry'], False), telemetry, BF16_FS6=False, num_warps=4, RHO_MATERIALIZED=opts['rho_materialized'])
        if nc:
            launch(kernels.output_fwd_kernel, (nc*h*ng,),
                   (v, output, *meta[:2], p, gp, saved_s, saved_u, saved_r, saved_t, saved_ro,
                    h, vd, gs, ng, c, kt, gt), telemetry, num_warps=8, RHO_MATERIALIZED=opts['rho_materialized'])
        if not _all_finite((output, final)):
            raise RuntimeError('nonfinite candidate forward; FP64 fallback forbidden')
        ctx.save_for_backward(*sources, *meta, *p, *gp, saved_s, saved_y, *carry,
                              saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp, diagonal_flags)
        ctx.opts, ctx.telemetry, ctx.nc = opts, telemetry, nc
        ctx.set_materialize_grads(False)
        if telemetry is not None:
            telemetry.update(rho_materialized=opts['rho_materialized'], backend='triton_chunk', precision_route='bf16', precision_eligible=True,
                             arithmetic='bf16_single_c128_preparation_and_workspace_bf16', working_dtype='float32', host_fallback=False,
                             readout_precision='fp32', state_control_precision='fp32',
                             contraction_precision='bf16_operands_fp32_accumulation', fallback_subblocks=0,
                             saved_buffer_bytes=sum(x.numel()*x.element_size() for x in (*p, *gp, saved_s, saved_y, *carry, saved_u, saved_r, saved_t, saved_ro, saved_w, saved_zp, diagonal_flags)))
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
        diagonal_flags = saved[53]
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
        need_curve_grad = ctx.needs_input_grad[6]
        dc = torch.empty((ctx.nc*h*ng if need_curve_grad else 0,), device=q.device, dtype=torch.float32)
        _, _, ss = shapes(c, kt)
        shared = tuple(torch.empty((ctx.nc*h*(1 if i in (0, 1, 2, 3, 5, 6, 7) else ng), *shape),
                                   device=q.device, dtype=torch.bfloat16 if i in (0, 1, 2, 3, 5, 6, 7) else torch.float32)
                       for i, shape in enumerate(ss[:-1]))
        shared += buffers(ctx.nc*h if opts['carry'] else 0, ss[-1:], q.device)
        gauge = buffers(ctx.nc*h, [(c,), (c,)], q.device)
        output_adj = buffers(ctx.nc*h*ng, [(kt, gt), (c, gt), (c,), (c,)], q.device)
        edge_y = buffers(ctx.nc*h*ng if opts['carry'] else 0, [(c, gt)], q.device)[0]
        if ctx.nc:
            launch(kernels.output_bwd_kernel, (ctx.nc*h*ng,),
                   (v, do, grads[2], *meta[:2], p, gp, saved_s, saved_u, saved_r, saved_t, saved_ro, shared, output_adj,
                    h, kd, vd, gs, ng, c, kt, gt), ctx.telemetry, num_warps=8, RHO_MATERIALIZED=opts['rho_materialized'])
        launch(kernels.state_bwd_kernel, (state.shape[0]*h*ng,),
               (v, curve, ds, grads[2], grads[5], *meta[:3], p, gp, saved_s, local, shared, output_adj, carry, edge_y,
                h, kd, vd, gs, ng, c, kt, gt, opts['carry']), ctx.telemetry, num_warps=4, RHO_MATERIALIZED=opts['rho_materialized'])
        if ctx.nc:
            for coefficient_mode in range(4):
                launch(kernels.shared_coefficients_bwd,
                       (ctx.nc*h, (kt if coefficient_mode == 3 else c)//64, (c if coefficient_mode in (1, 3) else kt)//64),
                       (do, grads[2], *meta[:2], gp, saved_s, saved_u, saved_r, saved_t, output_adj, shared,
                        h, vd, gs, ng, c, kt, gt, coefficient_mode), ctx.telemetry, num_warps=4, RHO_MATERIALIZED=opts['rho_materialized'])
            launch(kernels.shared_feedback_gl_bwd, (ctx.nc*h, c//64, c//64),
                   (gp, saved_y, output_adj, shared, ng, c, gt), ctx.telemetry, num_warps=8)
            launch(kernels.shared_feedback_system_bwd, (ctx.nc*h, c//64, c//64),
                   (curve, gp, saved_y, saved_w, output_adj, shared, ng, gs, c, gt), ctx.telemetry, num_warps=4, RHO_MATERIALIZED=opts['rho_materialized'])
            launch(kernels.parameter_bwd_kernel, (ctx.nc*h*ng,),
                   (v, curve, do, grads[2], dc, *meta[:2], meta[3], meta[4], p, gp, saved_s, saved_y, local,
                    shared, output_adj, edge_y, h, kd, vd, gs, ng, c, kt, gt, opts['carry']),
                   ctx.telemetry, NEED_DC=need_curve_grad, num_warps=8, RHO_MATERIALIZED=opts['rho_materialized'])
        if ctx.nc:
            assert c == kt == 128
            matrix_work = buffers(ctx.nc*h, [(kt, c), (c, c), (c, c), (c, kt)], q.device, dtype=torch.bfloat16)
            launch(kernels.gauge_collect, (ctx.nc*h, c//32, c//64), (shared, matrix_work, ng, c), ctx.telemetry, num_warps=4)
            launch(kernels.gauge_product_bwd, (ctx.nc*h, c//64, c//64), (p, matrix_work, c), ctx.telemetry, num_warps=4)
            launch(kernels.gauge_finish_bwd, (ctx.nc*h, c//16), (p, shared,
                   matrix_work, gauge, ng, c), ctx.telemetry, num_warps=4)
            launch(kernels.common_tail_bwd, (ctx.nc*h,),
                   (*meta[:2], p, shared, matrix_work, grads[1], grads[3], h, kd, ng, c, kt),
                   ctx.telemetry, num_warps=8)
            launch(kernels.common_ga_bwd, (ctx.nc*h, c//64, c//64),
                   (*meta[:2], p, shared, matrix_work, h, ng, c, kt), ctx.telemetry, num_warps=4)
            for matrix_mode in range(3):
                launch(kernels.common_products_bwd, (ctx.nc*h, c//64, (c if matrix_mode == 1 else kt)//64),
                       (*meta[:2], p, shared, matrix_work, h, ng, c, kt, matrix_mode), ctx.telemetry, num_warps=4)
            launch(kernels.common_inverse_left_bwd, (ctx.nc*h, c//64, c//64),
                   (p, matrix_work, c), ctx.telemetry, num_warps=4)
            launch(kernels.common_lower_bwd, (ctx.nc*h, c//16),
                   (p, shared, matrix_work, grads[3], *meta[:2], h, ng, c), ctx.telemetry, num_warps=4)
            launch(kernels.common_base_bwd, (ctx.nc*h, c//16),
                   (p, shared, matrix_work, grads[0], grads[1], grads[3], grads[4], *meta[:2],
                    h, kd, ng, c, kt), ctx.telemetry, num_warps=4)
            launch(kernels.interval_bwd_cross, (ctx.nc*h, c//64, kt//64),
                   (ell, grads[0], grads[1], *meta[:2], p, shared, matrix_work[:2], h, kd, ng, c, kt), ctx.telemetry, num_warps=8)
            launch(kernels.interval_diagonal_tc_bwd, (ctx.nc*h, c//32, kt//32),
                   (ell, grads[0], grads[1], *meta[:2], p, shared, matrix_work[:2], diagonal_flags, h, kd, ng, c, kt),
                   ctx.telemetry, num_warps=4)
            launch(kernels.interval_bwd_diagonal, (ctx.nc*h, c//32, kt//32),
                   (ell, grads[0], grads[1], *meta[:2], p, shared, matrix_work[:2], diagonal_flags, h, kd, ng, c, kt), ctx.telemetry, num_warps=4)
            launch(kernels.interval_gate_bwd, (ctx.nc*h, kt//32),
                   (grads[4], *meta[:2], p, matrix_work[:2], h, kd, c, kt), ctx.telemetry, num_warps=4)
            launch(kernels.common_input_bwd, (ctx.nc*h, c//32),
                   (grads[0], grads[1], grads[3], *meta[:2], p, shared, gauge, h, kd, ng, c, kt, opts['scale']),
                   ctx.telemetry, num_warps=4)
        grads[6] = dc.sum() if need_curve_grad else torch.zeros_like(curve, dtype=torch.float32)
        if not _all_finite(grads):
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
               scale=None, cu_seqlens=None, group_size=32, chunk_size=128, carry_mode='no_carry',
               arithmetic='bf16', telemetry=None):
    if q.ndim != 4 or v.ndim != 4 or q.shape[-1] != 128 or v.shape[-1] not in (32, 64, 128) or group_size != 32:
        raise ValueError('C128 requires K=128, V in {32, 64, 128}, and group_size=32.')
    _require_runtime(q)
    if carry_mode != 'no_carry':
        raise ValueError('C128 currently supports carry_mode=no_carry only')
    inputs = (q, k, v, beta, log_decay) + (() if initial_state is None else (initial_state,))
    if isinstance(curvature, torch.Tensor):
        inputs += (curvature,)
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in inputs):
        raise ValueError('candidate only accepts BF16/FP32; no FP64 inputs or arithmetic')
    if not q.is_cuda or arithmetic != 'bf16' or carry_mode not in ('no_carry', 'y_carry'):
        raise ValueError('CUDA BF16/FP32 candidate and explicit carry mode required')
    if isinstance(chunk_size, bool) or chunk_size != 128:
        raise ValueError('invalid execution chunk size')
    bounds, scale = _validate(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, False)
    c = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(curvature, dtype=torch.float32, device=q.device)
    state = initial_state if initial_state is not None else torch.zeros(
        (len(bounds)-1, q.shape[2], q.shape[3], v.shape[-1]), dtype=torch.float32, device=q.device)
    sources = (q, k, v, beta, log_decay, state, c)
    invalid, rho_bounded = _source_code_route(sources, group_size)
    if invalid & 1:
        raise ValueError('invalid nonfinite/gate source')
    if invalid & 2:
        raise ValueError('invalid/unrepresentable FP32 physical domain')
    aliases = {}
    for i, x in enumerate(sources):
        aliases.setdefault(id(x), []).append(i)
    opts = dict(bounds=bounds, scale=scale, rho_materialized=not rho_bounded, g=group_size, c=chunk_size, carry=carry_mode ==
                'y_carry', arithmetic=arithmetic, aliases=[a for a in aliases.values() if len(a) > 1])
    if telemetry is not None:
        telemetry.update(source_alias_groups=opts['aliases'], precision_requested='bf16')
    output, final = _ChunkBF16.apply(*(x.contiguous() for x in sources), opts, telemetry)
    return output, final if output_final_state else None
