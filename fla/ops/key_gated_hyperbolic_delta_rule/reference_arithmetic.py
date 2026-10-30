# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Small arbitrary-exponent arithmetic adapter for the exceptional CPU chunk graph.

Only array arithmetic and explicit triangular substitution are provided. There is no
AD engine, recurrent model, or alternative VJP here. Each array owns an isolated
mpmath context, so independent tests and callers cannot change its precision.
"""

import math

import mpmath
import numpy as np
import torch


def source_roundoff_bits(q, k, v, state, scale):
    """Screen absolute cancellation in state, readout and their source VJP before preparation."""
    def max_bits(x):
        largest = x.detach().double().abs().max().item() if x.numel() else 0.
        return math.log2(largest) if largest else -math.inf

    query = max_bits(q) + (math.log2(abs(scale)) if scale else -math.inf)
    state_bits = max(0., max_bits(state))
    target_bits = 0.
    if k.numel():
        kd = k.detach().double().abs()
        largest = kd.amax(-1, keepdim=True)
        safe = torch.where(largest > 0, largest, torch.ones_like(largest))
        log_norm = safe.squeeze(-1).log2() + .5 * (kd / safe).square().sum(-1).log2()
        log_value = v.detach().double().abs().amax(-1).log2()
        target_bits = max(0., (log_value - log_norm).max().item())
    # curvature adjoints contain state-squared times target and state times target-squared terms
    cancellation = max(0., query) + 2 * max(state_bits, target_bits) + target_bits
    return cancellation


def source_roundoff_reason(bits, dtype):
    budget = 2e-5 if dtype == torch.float32 else 1e-10
    if bits + math.log2(torch.finfo(dtype).eps) > math.log2(budget / 8):
        return ('state_readout_source_roundoff',)
    return ()


def cotangent_precision_bits(sources, scale, output_grad, final_grad, dtype):
    """Significand demand from original sources and the actual unscaled adjoint amplitude."""
    largest = max((x.detach().double().abs().max().item() for x in (output_grad, final_grad) if x.numel()), default=0.)
    if not math.isfinite(largest):
        raise ValueError('Nonfinite incoming cotangent.')
    if not largest:
        return -math.inf
    q, k, v, _, _, state, _ = sources
    budget = 2e-5 if dtype == torch.float32 else 1e-10
    return 1 + source_roundoff_bits(q, k, v, state, scale) + math.log2(largest) - math.log2(budget / 8)


def precision_context(sources):
    # reserve guard bits for products and source cancellation over the input exponent span
    exponents = []
    for source in sources:
        values = source.detach().double().abs()
        nonzero = values[values > 0]
        if nonzero.numel():
            exponents.extend((math.frexp(nonzero.min().item())[1], math.frexp(nonzero.max().item())[1]))
    span = max(exponents, default=0) - min(exponents, default=0)
    ctx = mpmath.mp.clone()
    ctx.prec = max(300, 128 + 4 * span)
    return ctx


class MPArray(np.ndarray):
    """Object array retaining its arithmetic context, including scalar and empty views."""

    def __new__(cls, values, ctx):
        obj = np.asarray(values, dtype=object).view(cls)
        obj.ctx = ctx
        return obj

    def __array_finalize__(self, source):
        self.ctx = getattr(source, 'ctx', None)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return value if isinstance(value, MPArray) else MPArray(value, self.ctx)

    @property
    def device(self):
        return 'cpu'

    def new_zeros(self, shape):
        return MPArray(np.full(shape, self.ctx.zero, dtype=object), self.ctx)

    new_empty = new_zeros

    def new_ones(self, shape):
        return MPArray(np.full(shape, self.ctx.one, dtype=object), self.ctx)

    def square(self):
        return self * self

    def exp(self):
        return MPArray(np.frompyfunc(self.ctx.exp, 1, 1)(self), self.ctx)

    def reciprocal(self):
        return 1 / self

    def detach(self):
        return self

    def sum(self, axis=None, **kwargs):
        return MPArray(np.asarray(self).sum(axis=axis, **kwargs), self.ctx)

    def prod(self, axis=None, **kwargs):
        return MPArray(np.asarray(self).prod(axis=axis, **kwargs), self.ctx)


class MPArithmetic:
    def __init__(self, ctx):
        self.ctx = ctx
        self.linalg = self

    def array(self, value):
        return MPArray(value, self.ctx)

    def eye(self, count, **kwargs):
        out = self.array(np.full((count, count), self.ctx.zero, dtype=object))
        np.fill_diagonal(out, self.ctx.one)
        return out

    def zeros_like(self, x):
        return x.new_zeros(x.shape)

    def stack(self, values, dim=0):
        return self.array(np.stack(values, axis=dim))

    def cat(self, values, dim=0):
        return self.array(np.concatenate(values, axis=dim))

    def outer(self, x, y):
        return self.array(np.outer(x, y))

    def diag(self, x):
        return self.array(np.diag(x))

    def tril(self, x, diagonal=0):
        return self.array(np.tril(x, k=diagonal))

    def isfinite(self, x):
        return np.asarray(np.frompyfunc(self.ctx.isfinite, 1, 1)(x), dtype=bool)

    def solve_triangular(self, matrix, rhs, upper=False):
        out = rhs.new_zeros(rhs.shape)
        count = len(matrix)
        for i in (reversed(range(count)) if upper else range(count)):
            sl = slice(i + 1, count) if upper else slice(0, i)
            out[i] = (rhs[i] - matrix[i, sl] @ out[sl]) / matrix[i, i]
        return out


def arithmetic(x):
    return MPArithmetic(x.ctx) if isinstance(x, MPArray) else torch


def extended(x, ctx):
    if isinstance(x, MPArray) and x.ctx is ctx:
        return x
    raw = x.detach().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return MPArray(np.frompyfunc(lambda v: ctx.mpf(v), 1, 1)(raw), ctx)


def tensor(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)


def like(x, reference):
    return extended(x, reference.ctx) if isinstance(reference, MPArray) else tensor(x, reference.dtype)


def exact_domain(k, v, c, state, group_size):
    """Resolve ambiguous floating norm comparisons using the original binary sources."""
    ctx = precision_context((k, v, c, state))
    kk, vv, cc, ss = (extended(x, ctx) for x in (k, v, c, state))
    n = kk.square().sum(-1)
    values = vv.reshape(*v.shape[:-1], v.shape[-1] // group_size, group_size).square().sum(-1)
    states = ss.reshape(*state.shape[:2], state.shape[2], -1, group_size).square().sum((-3, -1))
    return bool((n[..., None] > cc / group_size * values).all()), bool((cc / group_size * states < 1).all())
