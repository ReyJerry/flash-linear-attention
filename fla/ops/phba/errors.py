# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""PHBA 失败分类（triton化方案.md 5.2 节）。

每个失败带固定的 ``fallback_reason`` 与 ``fallback_class``：

    A  直接报错，不 fallback（与 sparse / prefix 实现方式无关，或 packed 同样不支持）
    B  direct 专属问题：只有 sparse_backend_requested = auto 且 packed_allowed 时才退到 packed
    C  prefix 专属问题：生产路径报 prefix_kernel_unsupported；allow_prefix_reference_cpu 时退到 reference_cpu_mode

原因名以方案为准；方案没有给出名字的几项（标注“本模块命名”）在这里固定下来，后续实现与 benchmark 只使用这里的名字。
本模块只依赖标准库。
"""

from __future__ import annotations

from typing import Any

CLASS_A = "A"
CLASS_B = "B"
CLASS_C = "C"

REASON_CLASSES: dict[str, str] = {
    # A 类：输入与环境（2.1 节、4.2 节、5.2 节）
    "non_cuda_input": CLASS_A,              # 本模块命名：CPU 或非 CUDA 输入（生产路径）
    "device_mismatch": CLASS_A,             # 本模块命名：输入与 SequenceLengthPlan 不在同一设备
    "dtype_mismatch": CLASS_A,
    "non_contiguous_input": CLASS_A,
    "gate_layout_mismatch": CLASS_A,
    "invalid_gate_input": CLASS_A,          # tau 非法、gate 输入域非法
    "gate_numeric_overflow": CLASS_A,       # correctness / debug 模式的 device finite assert
    "alignment_unsupported": CLASS_A,
    "invalid_shape": CLASS_A,
    "invalid_cu_seqlens": CLASS_A,
    "int32_overflow": CLASS_A,
    "gqa_unsupported": CLASS_A,             # 本模块命名：H != Hkv
    "head_dim_mismatch": CLASS_A,           # 本模块命名：Dq != Dv
    "dropout_unsupported": CLASS_A,         # 本模块命名：dropout_p != 0
    "n_query_split_unsupported": CLASS_A,
    "checkpoint_mode_unsupported": CLASS_A,
    "decode_unsupported": CLASS_A,
    "routing_unsupported": CLASS_A,
    "worklist_exceeds_budget": CLASS_A,
    "rho_exceeds_budget": CLASS_A,
    "common_exceeds_budget": CLASS_A,
    "packed_exceeds_budget": CLASS_A,
    "packed_not_admissible": CLASS_A,
    "runtime_oom": CLASS_A,
    "runtime_version_mismatch": CLASS_A,
    "fa_layout_probe_failed": CLASS_A,
    "fa_device_unsupported": CLASS_A,
    "fa_head_dim_unsupported": CLASS_A,
    "invalid_config": CLASS_A,              # 本模块命名：层配置或 kernel 配置本身不合法（非 2 的幂、小于 MIN_DOT 等）
    # B 类：direct 专属（5.2 节）
    "direct_dh_unsupported": CLASS_B,
    "J_exceeds_Jmax": CLASS_B,
    "direct_exceeds_budget": CLASS_B,
    "direct_grid_unsupported": CLASS_B,     # 本模块命名：direct sparse backward 的 grid 超出 2.1 节上限
    "direct_unsupported": CLASS_B,          # 显式 direct 时步骤 1 / 2 的 B 类失败
    "direct_runtime_failure": CLASS_B,      # 显式 direct 或 checkpoint 下步骤 3 的失败
    # C 类：prefix（5.2 节）
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


def reason_class(reason: str | None) -> str:
    if reason is None:
        return "none"
    return REASON_CLASSES[reason]
