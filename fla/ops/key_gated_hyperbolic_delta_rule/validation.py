# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Explicit finite-validation ownership for one experimental training step."""

import torch


class FiniteValidationScope:
    """Collect device checks and enforce them before a bound optimizer can update.

    Source/domain admission remains immediate. The caller passes this scope
    explicitly to the chunk operator, including checkpoint recomputation.
    Normal scope exit checks pending flags even without an optimizer step.
    """

    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.flags = []
        self.device = None
        self.open = False
        self.entered = False
        self.committed = False
        self.checks = 0
        self.flushes = 0
        self._hook = None

    @property
    def accepting(self):
        return self.open and not self.committed

    def __enter__(self):
        if self.entered:
            raise RuntimeError('FiniteValidationScope cannot be reused')
        self.entered = True
        self.open = True
        if self.optimizer is not None:
            self._hook = self.optimizer.register_step_pre_hook(self._before_step)
        return self

    def _before_step(self, optimizer, args, kwargs):
        self.commit()

    def add(self, finite):
        if not self.accepting:
            raise RuntimeError('Finite validation scope is not accepting checks')
        if not isinstance(finite, torch.Tensor) or finite.dtype != torch.bool or finite.numel() != 1:
            raise TypeError('Finite validation requires a scalar boolean tensor')
        if self.device is None:
            self.device = finite.device
        if finite.device != self.device:
            raise ValueError('A finite validation scope supports one device')
        self.flags.append(finite.reshape(()))
        self.checks += 1

    def flush(self):
        if not self.accepting:
            raise RuntimeError('Finite validation scope is not accepting checks')
        if self.flags:
            if not torch.stack(self.flags).all():
                raise RuntimeError('nonfinite KHDN training outputs or gradients; FP64 fallback forbidden')
            self.flags.clear()
            self.flushes += 1

    def commit(self):
        if not self.committed:
            self.flush()
            self.committed = True

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None and not self.committed:
                self.commit()
        finally:
            self.flags.clear()
            self.open = False
            if self._hook is not None:
                self._hook.remove()
                self._hook = None
        return False
