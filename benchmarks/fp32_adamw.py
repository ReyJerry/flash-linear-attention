# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FP32 AdamW numerical path, including device-side bias corrections."""
import re
import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def advance_steps(Meta, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
    ptr = tl.load(Meta+i*6+4, i < N, 0).to(tl.pointer_type(tl.float32))
    step = tl.load(ptr, i < N, 0.)
    tl.store(ptr, step+1., i < N)


@triton.jit
def adamw_fp32_kernel(Meta, Config, Jobs, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    owner = tl.load(Jobs+task*2)
    start = tl.load(Jobs+task*2+1)
    pp = tl.load(Meta+owner*6).to(tl.pointer_type(tl.float32))
    gp = tl.load(Meta+owner*6+1).to(tl.pointer_type(tl.float32))
    mp = tl.load(Meta+owner*6+2).to(tl.pointer_type(tl.float32))
    vp = tl.load(Meta+owner*6+3).to(tl.pointer_type(tl.float32))
    sp = tl.load(Meta+owner*6+4).to(tl.pointer_type(tl.float32))
    size = tl.load(Meta+owner*6+5)
    lr = tl.load(Config+owner*5)
    beta1, beta2 = tl.load(Config+owner*5+1), tl.load(Config+owner*5+2)
    eps, wd = tl.load(Config+owner*5+3), tl.load(Config+owner*5+4)
    step = tl.load(sp)
    correction1 = 1.-tl.exp2(step*tl.log2(beta1))
    correction2_sqrt = tl.sqrt(1.-tl.exp2(step*tl.log2(beta2)))
    step_size = lr/correction1
    ar = tl.arange(0, BLOCK).to(tl.int64)
    stop = tl.minimum(start+CHUNK, size)
    for offset in range(start, stop, BLOCK):
        i = offset+ar
        valid = i < stop
        p, g = tl.load(pp+i, valid, 0.), tl.load(gp+i, valid, 0.)
        m, v = tl.load(mp+i, valid, 0.), tl.load(vp+i, valid, 0.)
        p -= lr*wd*p
        m = beta1*m+(1.-beta1)*g
        v = beta2*v+(1.-beta2)*g*g
        denominator = tl.sqrt(v)/correction2_sqrt+eps
        p -= step_size*m/denominator
        tl.store(pp+i, p, valid)
        tl.store(mp+i, m, valid)
        tl.store(vp+i, v, valid)


_checked = weakref.WeakSet()
images = {}


def launch(kernel, grid, values):
    image = kernel.warmup(*values, grid=grid, num_warps=4, num_stages=1)
    if image not in _checked:
        assert not re.search(r'\bf64\b|xf64\b', image.asm['ttir']), kernel.__name__
        assert not re.search(r'\.f64\b', image.asm['ptx']), kernel.__name__
        images[kernel.__name__] = dict(ttir=image.asm['ttir'], ptx=image.asm['ptx'], registers=getattr(
            image, 'n_regs', None), spills=getattr(image, 'n_spills', None))
        _checked.add(image)
    launched = kernel[grid](*values, num_warps=4, num_stages=1)
    images[kernel.__name__].update(registers=launched.n_regs, spills=launched.n_spills)


class FP32AdamW(torch.optim.AdamW):
    """Training variant for contiguous CUDA FP32 parameters and unscaled gradients."""

    def __init__(self, params, **kwargs):
        kwargs['fused'] = True
        super().__init__(params, **kwargs)
        self._layout_key = None
        self._jobs = None
        self._last_metadata = self._last_config = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if getattr(self, 'grad_scale', None) is not None or getattr(self, 'found_inf', None) is not None:
            raise ValueError('Explicit unscaled BF16/FP32 training required')
        records = []
        config = []
        key = []
        mutated = []
        device = None
        for group in self.param_groups:
            if group.get('amsgrad') or group.get('maximize') or group.get('differentiable') or group.get('capturable'):
                raise ValueError('Unsupported optimizer variant')
            if isinstance(group['lr'], torch.Tensor):
                raise ValueError('Python scalar learning rate required')
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32 or not p.is_cuda or not p.is_contiguous() or p.grad.dtype != torch.float32 or not p.grad.is_contiguous():
                    raise ValueError('Contiguous CUDA FP32 parameters and gradients required')
                if device is None:
                    device = p.device
                assert p.device == device
                state = self.state[p]
                if not state:
                    state['step'] = torch.zeros((), dtype=torch.float32, device=device)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                assert all(v.dtype == torch.float32 and v.device == device for v in state.values())
                records.append([p.data_ptr(), p.grad.data_ptr(), state['exp_avg'].data_ptr(),
                               state['exp_avg_sq'].data_ptr(), state['step'].data_ptr(), p.numel()])
                config.append([group['lr'], *group['betas'], group['eps'], group['weight_decay']])
                key.append((id(p), p.numel()))
                mutated.extend((p, state['step'], state['exp_avg'], state['exp_avg_sq']))
        if not records:
            return loss
        key = tuple(key)
        if key != self._layout_key:
            jobs = [(owner, start) for owner, (_, size) in enumerate(key) for start in range(0, size, 65536)]
            self._jobs = torch.tensor(jobs, dtype=torch.int64, device=device).reshape(-1, 2)
            self._layout_key = key
        self._last_metadata = torch.tensor(records, dtype=torch.int64, device=device)
        self._last_config = torch.tensor(config, dtype=torch.float32, device=device)
        launch(advance_steps, (triton.cdiv(len(records), 128),), (self._last_metadata, len(records), 128))
        if self._jobs.numel():
            launch(adamw_fp32_kernel, (self._jobs.shape[0],),
                   (self._last_metadata, self._last_config, self._jobs, 65536, 1024))
        torch.autograd.graph.increment_version(mutated)
        return loss


def make_optimizer(model, args, groups):
    return FP32AdamW(groups(model, args.weight_decay), lr=args.lr, betas=(.9, .95))
