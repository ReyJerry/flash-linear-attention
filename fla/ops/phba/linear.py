# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

from fla.modules.activations.ops import swiglu_fwd, swiglu_fwdbwd
from fla.ops.phba.backends.cublaslt import _gemm_inputs_aligned, exact_backend
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def exact_weight_gradient(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return torch.mm(a, b)
    backend = exact_backend(a.device)
    return torch.mm(a, b) if backend is None else backend.weight_gradient(a=a, b=b)


class _Linear(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        inputs = x.reshape(-1, x.shape[-1])
        grad = grad.reshape(-1, weight.shape[0])
        dx = torch.mm(grad, weight).reshape(x.shape) if ctx.needs_input_grad[0] else None
        dw = exact_weight_gradient(a=grad.t(), b=inputs) if ctx.needs_input_grad[1] else None
        return dx, dw


class _SwiGLULinear(torch.autograd.Function):
    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x=x, y=y, output_contiguous=True)
        out = F.linear(z, weight, bias)
        ctx.save_for_backward(x, y, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    @autocast_custom_bwd
    def backward(ctx, grad):
        x, y, weight = ctx.saved_tensors
        grad = grad.reshape(-1, grad.shape[-1])
        dz = F.linear(grad, weight.t()).view_as(x)
        dx, dy, z = swiglu_fwdbwd(x=x, y=y, g=dz, use_weight=True, output_contiguous=True)
        matrix = z.reshape(-1, z.shape[-1])
        eligible = (not torch.compiler.is_compiling()
                    and type(grad) is torch.Tensor and type(matrix) is torch.Tensor
                    and grad.is_cuda and grad.dtype == matrix.dtype == torch.bfloat16
                    and tuple(grad.shape) == (8192, 2048) and tuple(matrix.shape) == (8192, 4272)
                    and grad.is_contiguous() and matrix.is_contiguous() and not torch.is_grad_enabled()
                    and _gemm_inputs_aligned(grad, matrix))
        backend = exact_backend(grad.device) if eligible else None
        if backend is not None:
            dw = backend.weight_gradient(a=grad.t(), b=matrix)
        else:
            dw = torch.einsum('bo,bi->oi', grad, matrix)
        return dx, dy, dw, grad.sum(0) if ctx.has_bias else None
