# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'PHBA input validation, structured errors and isolated autograd probes.'

from __future__ import annotations

import contextlib
from typing import Any

import torch

CLASS_A = "A"


CLASS_C = "C"


REASON_CLASSES: dict[str, str] = {
    "device_mismatch": CLASS_A,
    "dtype_mismatch": CLASS_A,
    "non_contiguous_input": CLASS_A,
    "invalid_shape": CLASS_A,
    "invalid_cu_seqlens": CLASS_A,
    "int32_overflow": CLASS_A,
    "routing_unsupported": CLASS_A,
    "fa_layout_probe_failed": CLASS_A,
    "fa_device_unsupported": CLASS_A,
    "invalid_config": CLASS_A,
    "prefix_kernel_unsupported": CLASS_C,
}


class PHBAError(RuntimeError):
    """PHBA 的统一异常。``fallback_reason`` 必须是 ``REASON_CLASSES`` 中登记的名字。"""

    def __init__(self, reason: str, message: str = "", **details: Any) -> None:
        if reason not in REASON_CLASSES:
            raise ValueError(f"未登记的 PHBA fallback_reason: {reason!r}")
        self.fallback_reason = reason
        self.fallback_class = REASON_CLASSES[reason]
        self.details = details
        text = f"[{reason}] {message}" if message else f"[{reason}]"
        super().__init__(text)


def fail(reason: str, message: str = "", **details: Any) -> PHBAError:
    """构造异常（调用方 ``raise fail(...)``），便于在表达式里使用。"""
    return PHBAError(reason, message, **details)


ALIGNMENT_BYTES = 16


def prepare_grad_output(grad_out: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, int]:
    """backward 入口的 dO。返回 (dO, grad_output_copy_bytes)。

    dtype 与 O 相同（autograd 已转换，这里只断言）；最后一维不连续或 data_ptr % 16 != 0 时强制新分配 aligned contiguous
    copy。不能只调用 .contiguous()：未对齐但 stride 合法的 view 会被原样返回。loss = O.sum() 时 dO 是 stride 全 0 的广播 tensor。
    """
    if grad_out.dtype != out.dtype:
        raise fail("dtype_mismatch", f"dO 的 dtype {grad_out.dtype} 与 O 的 {out.dtype} 不一致")
    return prepare_grad_output_meta(grad_out, out.dtype, tuple(out.shape))


def prepare_grad_output_meta(grad_out: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
    """与 prepare_grad_output 相同，只用 O 的 dtype / shape（autograd.Function 不必为此保存 O）。"""
    if grad_out.dtype != dtype:
        raise fail("dtype_mismatch", f"dO 的 dtype {grad_out.dtype} 与 O 的 {dtype} 不一致")
    if tuple(grad_out.shape) != tuple(shape):
        raise fail("invalid_shape", f"dO shape {tuple(grad_out.shape)} 与 O shape {tuple(shape)} 不一致")
    if grad_out.stride(-1) == 1 and grad_out.data_ptr() % ALIGNMENT_BYTES == 0:
        return grad_out, 0
    fresh = torch.empty(grad_out.shape, dtype=grad_out.dtype, device=grad_out.device)
    fresh.copy_(grad_out)
    return fresh, fresh.numel() * fresh.element_size()


@contextlib.contextmanager
def isolated_autograd():
    """预检 / probe 自己做 forward + backward：开启 grad、退出 inference_mode，并压入恒等 saved-tensor hooks，
    使它们即使在 non-reentrant activation checkpoint 的 forward 内运行，也不会被 checkpoint 的 hook 记录
    （checkpoint 按保存的 tensor 个数与 metadata 校验重算结果）。"""
    with torch.inference_mode(False), torch.enable_grad(), \
            torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        yield
