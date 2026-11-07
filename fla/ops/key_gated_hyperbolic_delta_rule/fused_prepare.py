# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Isolated first-order preprocessing; BF16 queries, FP32 geometry and gate controls."""
import re

import torch

from fla.ops.key_gated_hyperbolic_delta_rule import fused_prepare_kernels as kernels
from fla.utils import autocast_custom_bwd, autocast_custom_fwd


def launch(kernel, grid, values, telemetry=None, **kwargs):
    compiled = kernel[grid](*values, **kwargs)
    assert not re.search(r'\bf64\b|xf64\b', compiled.asm['ttir'])
    assert not re.search(r'\.f64\b', compiled.asm['ptx'])
    return compiled


class Preprocessing(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, q, k, v, br, gr, a, dt, curve, margin):
        assert q.shape == k.shape == v.shape and q.shape[-1] == 128
        assert all(x.is_cuda and x.is_contiguous() and x.dtype in (torch.float32, torch.bfloat16)
                   for x in (q, k, v, br, gr, a, dt, curve))
        batch, length, heads, dim = q.shape
        assert br.shape == (batch, length, heads) and gr.numel() == q.numel()
        assert a.numel() == heads and dt.numel() == heads*dim and curve.ndim == 0
        rows = batch*length*heads
        assert rows > 0 and 0 < margin < 1
        qo = torch.empty_like(q, dtype=torch.bfloat16)
        ko, vo, ell = (torch.empty_like(q, dtype=torch.float32) for _ in range(3))
        beta = torch.empty_like(br, dtype=torch.float32)
        norms = torch.empty((rows, 3), device=q.device, dtype=torch.float32)
        launch(kernels.preprocessing_fwd, (rows,),
               (q, k, v, br, gr, a, dt, curve, qo, ko, vo, beta, ell, norms, heads, dim, margin), num_warps=4)
        ctx.save_for_backward(q, k, v, br, gr, a, dt, curve, norms, beta, ell)
        ctx.margin = margin
        ctx.set_materialize_grads(False)
        return qo, ko, vo, beta, ell

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, gq, gk, gv, gb, ge):
        if torch.is_grad_enabled():
            raise RuntimeError('First-order preprocessing VJP only')
        q, k, v, br, gr, a, dt, curve, norms, beta, ell = ctx.saved_tensors
        batch, length, heads, dim = q.shape
        rows = batch*length*heads
        grads = []
        for value, like in zip((gq, gk, gv, gb, ge), (q, k, v, beta, ell)):
            grads.append(torch.zeros_like(like, dtype=torch.float32) if value is None else value.contiguous())
        dq, dk, dv, db = (torch.empty_like(x) for x in (q, k, v, br))
        dg = torch.empty_like(gr, dtype=torch.float32)
        da = torch.empty((batch, length, heads), device=q.device, dtype=torch.float32)
        need_c = ctx.needs_input_grad[7]
        dc = torch.empty((rows if need_c else 0,), device=q.device, dtype=torch.float32)
        launch(kernels.preprocessing_bwd, (rows,),
               (q, k, v, gr, a, dt, curve, norms, beta, ell, *grads, dq, dk, dv, db, dg, da, dc,
                heads, dim, ctx.margin, need_c), num_warps=4)
        da = da.sum((0, 1)).to(a.dtype)
        ddt = dg.reshape(batch, length, heads*dim).sum((0, 1)).reshape_as(dt).to(dt.dtype)
        dc = dc.sum().to(curve.dtype) if need_c else None
        return dq, dk, dv, db, dg.to(gr.dtype), da, ddt, dc, None


def fused_preprocessing(q, k, v, br, gr, a, dt, curve, margin):
    if not isinstance(curve, torch.Tensor):
        curve = torch.tensor(curve, device=q.device, dtype=torch.float32)
    return Preprocessing.apply(*(x.contiguous() for x in (q, k, v, br, gr, a, dt, curve)), margin)
