# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.activations import swiglu
from fla.modules.activations.ops import swiglu_fwd, swiglu_fwdbwd
from fla.modules.phba.linear import PHBALinear, exact_backend
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


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
                    and grad.is_contiguous() and matrix.is_contiguous() and not torch.is_grad_enabled())
        backend = exact_backend(grad.device) if eligible else None
        if backend is not None:
            dw = backend.weight_gradient(a=grad.t(), b=matrix)
        else:
            dw = torch.einsum('bo,bi->oi', grad, matrix)
        return dx, dy, dw, grad.sum(0) if ctx.has_bias else None


class PHBAMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True,
    ):
        super().__init__()
        if hidden_act != 'swish':
            raise ValueError('PHBA supports the swish-gated MLP')
        hidden_ratio = 4 if hidden_ratio is None else hidden_ratio
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        self.fuse_swiglu = fuse_swiglu
        self.gate_proj = PHBALinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = PHBALinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = PHBALinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate, value = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return _SwiGLULinear.apply(gate, value, self.down_proj.weight, self.down_proj.bias)
        return self.down_proj(swiglu(gate, value))
