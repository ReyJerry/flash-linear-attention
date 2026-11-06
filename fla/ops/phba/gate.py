# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fuse gate algebra while retaining native transcendental evaluations and FP32 rounding."""
import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.jit(do_not_specialize=['N'])
def _gate_neg_store_kernel(u, out, N, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    value = tl.load(u + i, i < N, other=0)
    negated = (value.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
    tl.store(out + i, negated, i < N)


@triton.jit(do_not_specialize=['N'])
def _gate_algebra_backward_kernel(u, ds, dg, out, N, HAS_S: tl.constexpr, HAS_G: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    if HAS_S:
        storage_grad = tl.load(ds + i, i < N, other=0).to(tl.float32)
        saved = tl.load(u + i, i < N, other=0)
        factor = saved + 1.0
        negated = (storage_grad.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
        value = negated * factor
    if HAS_G:
        direct = tl.load(dg + i, i < N, other=0)
        if HAS_S:
            value = direct + value
        else:
            value = direct
    value = value * 0.125
    tl.store(out + i, value, i < N)


class GateAlgebraFn(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, log_sigmoid, storage_dtype):
        g = log_sigmoid / 8
        u = torch.expm1(g)
        s = torch.empty_like(u, dtype=storage_dtype)
        _gate_neg_store_kernel[(triton.cdiv(u.numel(), 256),)](
            u, s, u.numel(), B=256, num_warps=4, num_stages=1, enable_fp_fusion=False)
        ctx.save_for_backward(g, u)
        ctx.set_materialize_grads(False)
        return g, s

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dg, ds):
        g, u = ctx.saved_tensors
        if ds is None and dg is None:
            return None, None
        if torch.is_grad_enabled():
            value = (-ds.float()) * (torch.expm1(g) + 1) if ds is not None else None
            if dg is not None:
                value = dg + value if value is not None else dg
            return value / 8, None
        ds = ds.contiguous() if ds is not None else None
        dg = dg.contiguous() if dg is not None else None
        out = torch.empty_like(u)
        _gate_algebra_backward_kernel[(triton.cdiv(u.numel(), 256),)](
            u, ds, dg, out, u.numel(), HAS_S=ds is not None, HAS_G=dg is not None,
            B=256, num_warps=4, num_stages=1, enable_fp_fusion=False)
        return out, None


def gate_algebra(log_sigmoid, storage_dtype, tau):
    if (tau == 8 and log_sigmoid.is_cuda and log_sigmoid.dtype == torch.float32
            and log_sigmoid.is_contiguous() and storage_dtype in (torch.bfloat16, torch.float16)):
        return GateAlgebraFn.apply(log_sigmoid, storage_dtype)
    g = log_sigmoid / tau
    return g, (-torch.expm1(g)).to(storage_dtype)
