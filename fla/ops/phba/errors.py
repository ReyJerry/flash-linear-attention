# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Structured PHBA input and kernel validation errors.'

from __future__ import annotations

from typing import Any

CLASS_A = "A"
CLASS_C = "C"

REASON_CLASSES: dict[str, str] = {
    "device_mismatch": CLASS_A,
    "dtype_mismatch": CLASS_A,
    "non_contiguous_input": CLASS_A,
    "invalid_shape": CLASS_A,
    "invalid_cu_seqlens": CLASS_A,
    "int32_overflow": CLASS_A,
    "n_query_split_unsupported": CLASS_A,
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

    def to_record(self) -> dict[str, Any]:
        return {"fallback_reason": self.fallback_reason, "fallback_class": self.fallback_class,
                "message": str(self), "details": dict(self.details)}


def fail(reason: str, message: str = "", **details: Any) -> PHBAError:
    """构造异常（调用方 ``raise fail(...)``），便于在表达式里使用。"""
    return PHBAError(reason, message, **details)
