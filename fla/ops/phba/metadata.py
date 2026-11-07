# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Fixed-capacity block-major route metadata for deterministic training.'

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.errors import fail
from fla.ops.phba.plans import INT32_MAX, kpad_state, kpad_tok

META_BT = 128
META_NUM_WARPS = 4


@triton.jit(do_not_specialize=["T", "K", "stride_rh", "stride_rt"])
def _count_kernel(route_ptr, count_ptr, T, K, stride_rh, stride_rt, H: tl.constexpr, BT: tl.constexpr,
                  KPAD: tl.constexpr):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    slot = tl.arange(0, KPAD)
    m = (t < T)[:, None] & (slot < K)[None, :]
    b = tl.load(route_ptr + h * stride_rh + t.to(tl.int64)[:, None] * stride_rt + slot[None, :], mask=m, other=-1)
    valid = b >= 0
    e = tl.where(valid, b, 0) * H + h
    tl.atomic_add(count_ptr + e, 1, mask=valid)


@dataclass(frozen=True)
class BranchMetadata:
    count: torch.Tensor          # [Nc * H] int32
    offset: torch.Tensor         # [Nc * H + 1] int32；offset[-1] = Nedge（device）
    rho: torch.Tensor            # [capacity] int32；只有前 Nedge 个有效
    capacity: int
    # retain original route slots for forward metadata reuse
    source_slots: torch.Tensor | None = None

    @property
    def nedge(self) -> torch.Tensor:
        return self.offset[-1:]


@dataclass(frozen=True)
class DirectMetadata:
    tok: BranchMetadata | None   # Kh = 0 时为 None
    state: BranchMetadata
    kh: int

    def bytes(self) -> int:
        n = 0
        for br in (self.tok, self.state):
            if br is not None:
                n += (br.count.numel() + br.offset.numel() + br.rho.numel()) * 4
                if br.source_slots is not None:
                    n += br.source_slots.numel() * br.source_slots.element_size()
        return n


def _branch(
    route: torch.Tensor,
    nc: int,
    kpad: int,
    *,
    scan_impl: str,
    retain_slots: bool = False,
    sorted_counts: bool = False,
) -> BranchMetadata:
    H, T, K = route.shape
    if route.dtype != torch.int32 or route.stride(-1) != 1:
        raise fail("invalid_shape", f"route 必须是最后一维连续的 int32，收到 {route.dtype} stride {route.stride()}")
    cap = T * H * K
    if cap > INT32_MAX or nc * H + 1 > INT32_MAX:
        raise fail("int32_overflow", f"Rho 上界 T*H*K = {cap} 或 Nc*H 超出 int32")
    if sorted_counts:
        if scan_impl != 'torch_cumsum':
            raise fail('invalid_config', f'unsupported metadata scan: {scan_impl}')
        from fla.ops.phba.canonical import canonical_rho
        rho, slots, count, offset = canonical_rho(route, nc, return_slots=retain_slots, return_counts=True)
        return BranchMetadata(count=count, offset=offset, rho=rho, capacity=cap, source_slots=slots)
    dev = route.device
    n_e = nc * H
    count = torch.zeros(n_e, dtype=torch.int32, device=dev)
    # Offset[0] = 0；不用标量写入（会同步）
    offset = torch.zeros(n_e + 1, dtype=torch.int32, device=dev)
    grid = (triton.cdiv(T, META_BT), H)
    args = (T, K, route.stride(0), route.stride(1))
    _count_kernel[grid](route, count, *args, H=H, BT=META_BT, KPAD=kpad, num_warps=META_NUM_WARPS)
    if scan_impl != "torch_cumsum":
        raise fail("invalid_config", f"metadata_scan_impl = {scan_impl!r} is unsupported; use torch_cumsum")
    torch.cumsum(count, dim=0, dtype=torch.int32, out=offset[1:])
    from fla.ops.phba.canonical import canonical_rho

    source_slots = None
    result = canonical_rho(route, nc, return_slots=retain_slots)
    if retain_slots:
        rho, source_slots = result
    else:
        rho = result
    return BranchMetadata(count=count, offset=offset, rho=rho, capacity=cap, source_slots=source_slots)


def build_direct_metadata(rtok: torch.Tensor, rstate: torch.Tensor, nc: int, *,
                          metadata_scan_impl: str = "torch_cumsum",
                          retain_token_slots: bool = False, derived_state_layout=None) -> DirectMetadata:
    """Rtok [H, T, Kh]、Rstate [H, T, Ks = Kh + 1] → token / state 两套 block-major metadata。"""
    H, T, kh = rtok.shape
    if rstate.shape != (H, T, kh + 1):
        raise fail("invalid_shape", f"Rstate 必须是 [H, T, Kh + 1] = {(H, T, kh + 1)}，收到 {tuple(rstate.shape)}")
    tok = _branch(rtok, nc, kpad_tok(kh), scan_impl=metadata_scan_impl,
                  retain_slots=retain_token_slots, sorted_counts=derived_state_layout is not None) if kh > 0 else None
    if derived_state_layout is not None and tok is not None:
        from fla.ops.phba.derived_state import derive_state
        state = derive_state(tok, derived_state_layout, H, T * H * (kh + 1))
    else:
        state = _branch(rstate, nc, kpad_state(kh + 1), scan_impl=metadata_scan_impl)
    return DirectMetadata(tok=tok, state=state, kh=kh)


def metadata_reference(route: torch.Tensor, nc: int) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """逐 slot 读取 route 的参照：返回 (Count, Offset, 每个 e 的 row 列表（升序）)。"""
    H, T, K = route.shape
    count = torch.zeros(nc * H, dtype=torch.int64)
    rows: list[list[int]] = [[] for _ in range(nc * H)]
    r = route.cpu()
    for h in range(H):
        for t in range(T):
            for k in range(K):
                b = int(r[h, t, k])
                if b >= 0:
                    count[b * H + h] += 1
                    rows[b * H + h].append(t * H + h)
    offset = torch.zeros(nc * H + 1, dtype=torch.int64)
    offset[1:] = torch.cumsum(count, 0)
    return count.to(torch.int32), offset.to(torch.int32), [sorted(x) for x in rows]


def sort_rho_segments(meta: BranchMetadata) -> list[list[int]]:
    """把每个 (block, head) 段内的 row 排序后返回（需要同步；correctness 与确定性调试模式用）。"""
    off = meta.offset.tolist()
    rho = meta.rho[: off[-1]].tolist()
    return [sorted(rho[off[e]:off[e + 1]]) for e in range(len(off) - 1)]


__all__ = ["BranchMetadata", "DirectMetadata", "build_direct_metadata", "metadata_reference", "sort_rho_segments"]
