# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in mixed KHDN candidate with accurate readout; strict e39 remains the default."""

import math
import time

import torch
import triton

from fla.ops.key_gated_hyperbolic_delta_rule.chunk_kernels import (
    carry_fwd_kernel,
    common_bwd_kernel,
    common_fwd_kernel,
    group_af_fwd_kernel,
    group_fwd_kernel,
    state_bwd_kernel,
    state_fwd_kernel,
)
from fla.ops.key_gated_hyperbolic_delta_rule.naive import _validate
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _reset_telemetry(telemetry, requested):
    if telemetry is None:
        return
    for key in (
        'backend', 'backward', 'arithmetic', 'tensor_cores', 'backward_precision', 'fallback_subblocks',
        'carry_edges', 'common_preparations', 'direct_FS', 'chunk_lengths', 'saved_buffer_bytes',
        'host_transfer_and_forward_seconds', 'host_details', 'backward_host_details', 'backward_host_seconds',
        'backward_fallback_subblocks', 'shared_A_vjp', 'precision_requested', 'precision_route', 'precision_eligible',
        'qualification_reason', 'qualification', 'readout_precision', 'state_control_precision',
        'contraction_precision', 'backward_qualification_reason', 'strict_replay_details',
        'exception_reason', 'backward_exception_reason', 'backward_precision_route',
        'backward_adjoint_profile', 'backward_adjoint_limit',
        'backward_output_cotangent_max', 'backward_state_cotangent_max',
    ):
        telemetry.pop(key, None)
    telemetry.update(precision_requested=requested)


def _buffers(count, shapes, device):
    return tuple(torch.empty((count, *shape), dtype=torch.float64, device=device) for shape in shapes)


def _shapes(c, k):
    common = [(c, k)] * 9 + [(c, c)] * 7 + [(k, c)] * 2 + [(c,)] * 5
    group = [(c, c)] * 5 + [(c,)] * 3
    shared = [(c, c), (c, c), (c, k), (c, c), (c,), (c, k), (c, c), (k, c),
              (c,), (c,), (c,), (c, k), (c,), (c, k), (c, k)]
    return common, group, shared


def _host_arguments(sources, options, check_inputs):
    from fla.ops.key_gated_hyperbolic_delta_rule.reference_chunk import _arguments
    q, k, v, beta, ell, state, c = sources
    cu = torch.tensor(options['bounds'], dtype=torch.int64) if q.shape[0] == 1 else None
    normalized, opts = _arguments(q, k, v, beta, ell, c, state, options['scale'], cu, options['g'],
                                  check_inputs, options['c'], options['carry'], 'zero_detached', None, True, 0)
    # the device contract requires FP64 preparation even when the CPU public dtype is FP32
    opts.update(output_final_state=options['final'], working_dtype=torch.float64)
    return normalized, opts


def _host_call(sources, options, telemetry):
    from fla.ops.key_gated_hyperbolic_delta_rule.reference_backward import _ManualChunk
    start = time.perf_counter()
    host = tuple(x.to('cpu') for x in sources)
    normalized, opts = _host_arguments(host, options, True)
    details = {}
    output, final = _ManualChunk.apply(*normalized, opts, details, None)
    output = output.to(sources[0].device)
    final = None if final is None else final.to(sources[0].device)
    torch.cuda.synchronize(sources[0].device)
    if telemetry is not None:
        for key in ('qualification', 'readout_precision', 'state_control_precision', 'contraction_precision'):
            telemetry.pop(key, None)
        telemetry.update(backend='host_reviewed_chunk_exception', backward='host_reviewed_manual_chunk_source_vjp',
                         precision_route='host_exception', arithmetic='host_reviewed_chunk', tensor_cores=False,
                         fallback_subblocks=sum(len(p) for p in details['execution_plan']), carry_edges=details['carry_edges'],
                         host_transfer_and_forward_seconds=time.perf_counter() - start, host_details=details)
    return output, final


def _exception_reason(sources, g, scale):
    q, k, v, beta, ell, state, c = sources
    # inspect original sources of every dtype before device preparation
    if any(x.numel() and (x.detach().abs().max().item() > 2.**100
                          or ((x != 0) & (x.detach().abs() < 2.**-100)).any().item())
           for x in (q, k, v, state, c)):
        return 'source_exponent_range'
    if not k.numel():
        return None
    kd = k.detach().double()
    n = kd.square().sum(-1)
    if ((n < 2.**-80) | (n > 2.**80)).any().item():
        return 'raw_key_norm_range'
    delta = c.detach().double() / g * v.detach().double().reshape(*v.shape[:-1], -1, g).square().sum(-1) / n[..., None]
    if (delta > .25).any().item():
        return 'feedback_margin'
    radius2 = c.detach().double() / g * state.detach().double().reshape(*state.shape[:-1], -1, g).square().sum((-3, -1))
    if (radius2 > .5).any().item():
        return 'entry_ball_margin'
    from fla.ops.key_gated_hyperbolic_delta_rule.reference_arithmetic import source_roundoff_bits, source_roundoff_reason
    if source_roundoff_reason(source_roundoff_bits(q, k, v, state, scale), torch.float64):
        return 'state_readout_source_roundoff'
    return None


def _host_backward(sources, options, do, ds, needs, telemetry):
    from fla.ops.key_gated_hyperbolic_delta_rule.reference_backward import _backward
    from fla.ops.key_gated_hyperbolic_delta_rule.reference_chunk import _execute
    start = time.perf_counter()
    host = tuple(x.detach().cpu() for x in sources)
    normalized, opts = _host_arguments(host, options, False)
    details = {}
    _, _, tapes = _execute(normalized, opts, details)
    result = _backward(tapes, normalized, opts, do.cpu(), ds.cpu(), details, needs)
    result = tuple(None if x is None else x.to(sources[0].device) for x in result)
    torch.cuda.synchronize(sources[0].device)
    if telemetry is not None:
        telemetry.update(backward='host_reviewed_manual_chunk_source_vjp', backward_host_details=details,
                         backward_precision_route='host_exception', backward_precision=details.get('backward_precision'),
                         backward_host_seconds=time.perf_counter() - start,
                         backward_fallback_subblocks=sum(len(p) for p in details['execution_plan']))
    return result


def _qualification_reason(sources, options):
    """Select the empirical profile; matching version strings do not attest compiler binaries."""
    q, k, v, beta, ell, state, curve = sources
    if (q.dtype not in (torch.float32, torch.bfloat16)
            or k.dtype != q.dtype or v.dtype != q.dtype
            or any(x.dtype != torch.float32 for x in (beta, ell, state, curve))):
        return 'unqualified_source_dtype'
    if (options['c'] not in (16, 32, 64) or q.shape[3] not in (16, 128)
            or options['g'] not in (16, 32) or v.shape[-1] not in (16, 128)
            or q.shape[0] > 8 or q.shape[2] > 16
            or not q.numel() or max(b-a for a, b in zip(options['bounds'], options['bounds'][1:])) > 2048):
        return 'unqualified_geometry'
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return 'unqualified_device'
    if torch.__version__ != '2.9.0+cu128' or triton.__version__ != '3.5.0':
        return 'unqualified_runtime'
    if any(not torch.isfinite(x).all().item() for x in sources):
        return 'unqualified_nonfinite_source'
    if any(x.numel() and x.detach().abs().max().item() > 8 for x in (q, k, v, state)):
        return 'unqualified_source_magnitude'
    scale = float(options['scale'])
    if not math.isfinite(scale):
        return 'unqualified_scale'
    if abs(scale) * q.detach().abs().max().item() > 8:
        return 'unqualified_effective_query_magnitude'
    n = k.detach().double().square().sum(-1)
    if ((n < .25) | (n > 4.)).any().item():
        return 'unqualified_key_norm'
    if ell.numel() and ell.detach().min().item() < -16.:
        return 'unqualified_decay'
    if curve.detach().item() > 1.:
        return 'unqualified_curvature'
    return None


def _mixed_adjoint_qualification(sources, options, do, ds):
    """Keep v1 and admit the separately measured small-model adjoint envelope."""
    maxima = [x.detach().abs().max().item() if x.numel() else 0. for x in (do, ds)]
    finite = all(math.isfinite(x) for x in maxima)
    q, _, v, _, _, state, _ = sources
    model_geometry = (
        q.shape[0] <= 2 and q.shape[2] <= 2 and q.shape[3] == v.shape[3] == 128
        and state.shape[0] <= 4 and options['g'] == 32 and options['c'] == 16
        and options['carry'] == 'no_carry'
        and max(b-a for a, b in zip(options['bounds'], options['bounds'][1:])) <= 35
    )
    legacy = finite and max(maxima) <= 8.
    limit = 128. if model_geometry and not legacy else 8.
    accepted = finite and max(maxima) <= limit
    reason = None if accepted else (
        'unqualified_cotangent_magnitude' if model_geometry or not finite
        else 'unqualified_model_adjoint_geometry')
    return reason, dict(
        backward_adjoint_profile=('empirical_profile_v1' if legacy else 'model_adjoint_v1') if accepted else 'unqualified',
        backward_adjoint_limit=limit, backward_output_cotangent_max=maxima[0],
        backward_state_cotangent_max=maxima[1],
    )


def _strict_device_replay(sources, options, do, ds, needs, telemetry):
    details = {}
    strict_options = dict(options, mixed=False)
    with torch.enable_grad():
        replay = tuple(x.detach().requires_grad_(need) for x, need in zip(sources, needs))
        output, final = _Chunk.apply(*replay, strict_options, details)
        values, cots = (output,), (do,)
        if final is not None:
            values, cots = (*values, final), (*cots, ds)
        partials = iter(torch.autograd.grad(values, [x for x in replay if x.requires_grad], cots))
        result = tuple(next(partials) if need else None for need in needs)
    if telemetry is not None:
        telemetry.update(backward='strict_device_replay', backward_precision=details.get('backward_precision'),
                         backward_precision_route='strict_device_replay',
                         backward_qualification_reason='unqualified_cotangent_magnitude',
                         strict_replay_details=details)
    return result


class _Chunk(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, ell, state, curvature, options, telemetry):
        sources = (q, k, v, beta, ell, state, curvature)
        b, t, h, kd = q.shape
        vd, gs, c = v.shape[-1], options['g'], options['c']
        ng, kt, gt = vd // gs, triton.next_power_of_2(kd), triton.next_power_of_2(gs)
        starts, ends, seq_chunks = [], [], [0]
        for bos, eos in zip(options['bounds'], options['bounds'][1:]):
            for pos in range(bos, eos, c):
                starts.append(pos)
                ends.append(min(pos + c, eos))
            seq_chunks.append(len(starts))
        prev, nxt = [-1] * len(starts), [-1] * len(starts)
        for first, last in zip(seq_chunks, seq_chunks[1:]):
            for j in range(first, last):
                prev[j] = j - 1 if j > first else -1
                nxt[j] = j + 1 if j + 1 < last else -1
        metadata = tuple(torch.tensor(x, device=q.device, dtype=torch.int64) for x in (starts, ends, seq_chunks, prev, nxt))
        nc = len(starts)
        common_shapes, group_shapes, _ = _shapes(c, kt)
        p = _buffers(nc * h, common_shapes, q.device)
        groups = _buffers(nc * h * ng, group_shapes, q.device)
        carry = _buffers(nc * h, [(c, kt), (c, c)], q.device)
        saved_s, saved_y = _buffers(nc * h * ng, [(kt, gt), (c, gt)], q.device)
        output = torch.empty(v.shape, dtype=q.dtype, device=q.device)
        public_dtype = torch.float64 if any(x.dtype == torch.float64 for x in sources) else torch.float32
        final = torch.empty(state.shape, dtype=public_dtype, device=q.device) if options['final'] else None
        if nc:
            common_fwd_kernel[(nc*h,)](q, k, beta, ell, *metadata[:2], p, h, kd, c, kt, options['scale'], num_warps=4)
            if options['carry'] == 'y_carry':
                carry_fwd_kernel[(nc*h,)](metadata[1], metadata[0], metadata[4], p, carry, h, c, kt, num_warps=4)
            group_fwd_kernel[(nc*h*ng,)](v, curvature, *metadata[:2], p, groups, h, vd, gs, ng, c, gt, num_warps=4)
            group_af_fwd_kernel[(nc*h*ng, c)](p, groups, ng, c, num_warps=4)
        state_fwd_kernel[((len(seq_chunks)-1)*h*ng,)](
            v, state, output, final if final is not None else state, *metadata[:3], p, groups, saved_s, saved_y, carry,
            h, kd, vd, gs, ng, c, kt, gt, options['final'], options['carry'] == 'y_carry',
            options.get('mixed', False), num_warps=4,
        )
        ctx.save_for_backward(*sources, *metadata, *p, *groups, saved_s, saved_y, *carry)
        ctx.options, ctx.telemetry, ctx.nc = options, telemetry, nc
        ctx.edges = sum(x >= 0 for x in nxt) * h if options['carry'] == 'y_carry' else 0
        ctx.set_materialize_grads(False)
        if telemetry is not None:
            telemetry.update(backend='triton_chunk' if nc else 'empty_identity', fallback_subblocks=0, carry_edges=ctx.edges,
                             precision_route='experimental_mixed_readout' if options.get('mixed', False) else 'strict',
                             arithmetic=('strict_readout_tf32x3_state_products'
                                         if options.get('mixed', False) else 'device_float64'),
                             tensor_cores=bool(options.get('mixed', False) and nc and min(c, kt, gt) >= 16),
                             common_preparations=nc*h,
                             direct_FS=nc*h-ctx.edges, chunk_lengths=[z-a for a, z in zip(starts, ends)],
                             saved_buffer_bytes=sum(x.numel()*x.element_size()
                                                    for x in (*p, *groups, saved_s, saved_y, *carry)))
            if options.get('mixed', False):
                telemetry.update(qualification='empirical_profile_v1', readout_precision='fp64',
                                 state_control_precision='fp64',
                                 contraction_precision='17_selected_tf32x3_fp32_accumulate_sites')
        return output, final

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, output_grad, final_grad):
        if torch.is_grad_enabled():
            raise RuntimeError('KHDN supports first-order gradients only; double backward is not supported.')
        saved = ctx.saved_tensors
        sources, metadata, p, groups = saved[:7], saved[7:12], saved[12:35], saved[35:43]
        saved_s, saved_y = saved[43:45]
        carry = saved[45:47]
        q, k, v, beta, ell, state, curvature = sources
        options = ctx.options
        b, t, h, kd = q.shape
        vd, gs, c = v.shape[-1], options['g'], options['c']
        ng, kt, gt = vd // gs, triton.next_power_of_2(kd), triton.next_power_of_2(gs)
        do = torch.zeros_like(v) if output_grad is None else output_grad.contiguous()
        ds = torch.zeros_like(state) if final_grad is None else final_grad.contiguous()
        extreme_cotangent = any(x.numel() and (((x != 0) & (x.abs() < 2.**-100)).any().item()
                                               or x.abs().max().item() > 2.**100) for x in (do, ds))
        from fla.ops.key_gated_hyperbolic_delta_rule.reference_arithmetic import cotangent_precision_bits
        accuracy_replay = cotangent_precision_bits(sources, options['scale'], do, ds, torch.float64) > 53
        if extreme_cotangent or accuracy_replay:
            if ctx.telemetry is not None:
                ctx.telemetry['backward_exception_reason'] = 'cotangent_accuracy' if accuracy_replay else 'cotangent_range'
            result = _host_backward(sources, options, do, ds, ctx.needs_input_grad[:7], ctx.telemetry)
            return (*result, None, None)
        if options.get('mixed', False):
            adjoint_reason, adjoint_details = _mixed_adjoint_qualification(sources, options, do, ds)
            if ctx.telemetry is not None:
                ctx.telemetry.update(adjoint_details)
            if adjoint_reason is not None:
                result = _strict_device_replay(sources, options, do, ds, ctx.needs_input_grad[:7], ctx.telemetry)
                if ctx.telemetry is not None:
                    ctx.telemetry['backward_qualification_reason'] = adjoint_reason
                return (*result, None, None)
        grads = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in sources]
        # scalar curvature owners retain FP64 until the final requested-source cast
        dc = torch.empty((ctx.nc*h*ng,), device=q.device, dtype=torch.float64)
        _, _, shared_shapes = _shapes(c, kt)
        shared = _buffers(ctx.nc*h*ng, shared_shapes, q.device)
        state_bwd_kernel[(state.shape[0]*h*ng,)](
            v, curvature, do, ds, grads[2], grads[5], dc, *metadata[:3], p, groups, saved_s, saved_y, shared, carry,
            h, kd, vd, gs, ng, c, kt, gt, options['carry'] == 'y_carry', options.get('mixed', False), num_warps=4,
        )
        if ctx.nc:
            common_bwd_kernel[(ctx.nc*h,)](
                ell, grads[0], grads[1], grads[3], grads[4], *metadata[:2], metadata[3], p, shared,
                h, kd, ng, c, kt, options['scale'], num_warps=4,
            )
        grads[6] = dc.sum().to(curvature.dtype)
        if any(need and not torch.isfinite(x).all().item() for x, need in zip(grads, ctx.needs_input_grad[:7])):
            result = _host_backward(sources, options, do, ds, ctx.needs_input_grad[:7], ctx.telemetry)
            return (*result, None, None)
        if ctx.telemetry is not None:
            ctx.telemetry.update(backward='manual_triton_chunk_source_vjp' if ctx.nc else 'empty_identity',
                                 backward_precision_route='experimental_mixed_readout' if options.get(
                                     'mixed', False) else 'strict',
                                 backward_precision=('strict_readout_tf32x3_state_products'
                                                     if options.get('mixed', False) else 'device_float64'),
                                 shared_A_vjp=ctx.nc*h)
        return (*[x if need else None for x, need in zip(grads, ctx.needs_input_grad[:7])], None, None)


def chunk_key_gated_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    curvature: float | torch.Tensor = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    group_size: int = 32,
    check_inputs: bool = True,
    chunk_size: int = 32,
    carry_mode: str = 'no_carry',
    telemetry: dict | None = None,
    precision_policy: str = 'strict',
):
    """Evaluate e39 strict or the explicitly requested, empirically bounded mixed candidate.

    Output follows q.dtype; canonical state is FP32 unless a source is FP64.
    Exceptional ranges use the separately reported reviewed host implementation.
    Telemetry describes the latest call; use a separate dictionary for each live autograd graph.
    """
    _reset_telemetry(telemetry, precision_policy)
    bounds, scale = _validate(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, False)
    if not q.is_cuda:
        raise ValueError('The production KHDN entry requires CUDA tensors.')
    if isinstance(chunk_size, bool) or chunk_size not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError('chunk_size must be a power of two from 1 through 64.')
    if carry_mode not in ('no_carry', 'y_carry'):
        raise ValueError('carry_mode must be no_carry or y_carry.')
    dtype = torch.float64 if any(x is not None and isinstance(x, torch.Tensor) and x.dtype == torch.float64
                                 for x in (q, k, v, beta, log_decay, curvature, initial_state)) else torch.float32
    c = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(curvature, device=q.device, dtype=dtype)
    state = (torch.zeros((len(bounds)-1, q.shape[2], q.shape[3], v.shape[-1]), device=q.device, dtype=dtype)
             if initial_state is None else initial_state)
    sources = (q, k, v, beta, log_decay, state, c)
    options = dict(bounds=bounds, scale=scale, g=group_size, c=chunk_size, final=output_final_state, carry=carry_mode)
    if precision_policy not in ('strict', 'mixed_readout'):
        raise ValueError('precision_policy must be strict or mixed_readout.')
    qualification = 'strict_requested' if precision_policy == 'strict' else _qualification_reason(sources, options)
    options['mixed'] = qualification is None
    if telemetry is not None:
        telemetry.update(precision_eligible=qualification is None)
        if qualification:
            telemetry.update(qualification_reason=qualification)
    reason = _exception_reason(sources, group_size, scale)
    if reason:
        if telemetry is not None:
            telemetry['exception_reason'] = reason
        return _host_call(sources, options, telemetry)
    if check_inputs:
        from fla.ops.key_gated_hyperbolic_delta_rule.reference_chunk import _domain
        _domain(q, k, v, beta, log_decay, c.detach(), state, group_size)
    with torch.cuda.device(q.device):
        output, final = _Chunk.apply(*sources, options, telemetry)
        if not torch.isfinite(output).all().item() or (final is not None and not torch.isfinite(final).all().item()):
            if telemetry is not None:
                telemetry['exception_reason'] = 'device_result_range'
            return _host_call(sources, options, telemetry)
        return output, final
