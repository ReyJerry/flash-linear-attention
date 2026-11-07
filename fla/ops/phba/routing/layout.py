# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Block and query ownership layouts for dense and packed sequences.'

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fla.ops.phba.errors import fail
from fla.ops.phba.plans import SequenceLengthPlan, _block_layout, _require_int


@dataclass(frozen=True)
class RoutingLayout:
    chunk_size: int
    br: int
    Nseq: int
    Nc: int
    Nwork: int
    T: int
    block_ptr_host: tuple[int, ...] = field(repr=False)
    seq_block_ptr_host: tuple[int, ...] = field(repr=False)
    block_ptr: torch.Tensor = field(repr=False, compare=False)
    seq_block_ptr: torch.Tensor = field(repr=False, compare=False)
    block_seq: torch.Tensor = field(repr=False, compare=False)
    current_block: torch.Tensor = field(repr=False, compare=False)
    prefix_valid: torch.Tensor = field(repr=False, compare=False)
    work_block: torch.Tensor = field(repr=False, compare=False)
    work_start: torch.Tensor = field(repr=False, compare=False)
    work_end: torch.Tensor = field(repr=False, compare=False)
    work_seq: torch.Tensor = field(repr=False, compare=False)

    @property
    def device(self) -> torch.device:
        return self.block_ptr.device

    def tensors(self) -> dict[str, torch.Tensor]:
        return {n: getattr(self, n) for n in ("block_ptr", "seq_block_ptr", "block_seq", "current_block", "prefix_valid",
                                              "work_block", "work_start", "work_end", "work_seq")}


def _i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.int32).contiguous()


@torch.inference_mode(False)
def build_block_layout(length_plan: SequenceLengthPlan, chunk_size: int, br: int) -> RoutingLayout:
    """由 SequenceLengthPlan 构造（必要时）并返回缓存的 RoutingLayout。"""
    c = _require_int("chunk_size", chunk_size)
    br = _require_int("Br", br)
    cache = length_plan._routing_layouts  # noqa: SLF001
    key = (c, br)
    layout = cache.get(key)
    if layout is None:
        layout = _construct(length_plan, c, br)
        cache[key] = layout
    return layout


def _construct(lp: SequenceLengthPlan, c: int, br: int) -> RoutingLayout:
    block_ptr_host, seq_block_ptr_host, nwork = _block_layout(lp.lengths, c, br)
    nc, nseq, T = len(block_ptr_host) - 1, lp.Nseq, lp.T
    dev = lp.device
    # BlockPtr / SeqBlockPtr 与 prefix states 共用 SequenceLengthPlan.block_layout(C) 的 device 副本（同一份 source of truth）
    blayout = lp.block_layout(c)
    block_ptr = blayout.block_ptr.to(torch.int64)
    seq_block_ptr = blayout.seq_block_ptr.to(torch.int64)
    nblk = seq_block_ptr[1:] - seq_block_ptr[:-1]
    block_seq = torch.repeat_interleave(torch.arange(nseq, device=dev), nblk, output_size=nc)       # [Nc]
    block_len = block_ptr[1:] - block_ptr[:-1]                            # [Nc]，最后一个 block 可能不满
    current_block = torch.repeat_interleave(torch.arange(nc, device=dev), block_len, output_size=T)  # [T]
    prefix_valid = torch.arange(nc, device=dev) > seq_block_ptr[block_seq]                         # [Nc] bool
    # 每个 block 的 query tile 数
    nq = torch.div(block_len + (br - 1), br, rounding_mode="floor")
    work_block = torch.repeat_interleave(torch.arange(nc, device=dev), nq, output_size=nwork)       # [Nwork]
    work_ptr = torch.zeros(nc + 1, dtype=torch.int64, device=dev)
    work_ptr[1:] = torch.cumsum(nq, 0)
    tile = torch.arange(nwork, device=dev) - work_ptr[work_block]
    work_start = block_ptr[work_block] + tile * br
    work_end = torch.minimum(work_start + br, block_ptr[work_block + 1])
    work_seq = block_seq[work_block]
    return RoutingLayout(
        chunk_size=c, br=br, Nseq=nseq, Nc=nc, Nwork=nwork, T=T,
        block_ptr_host=block_ptr_host, seq_block_ptr_host=seq_block_ptr_host,
        block_ptr=_i32(block_ptr), seq_block_ptr=_i32(seq_block_ptr), block_seq=_i32(block_seq),
        current_block=_i32(current_block), prefix_valid=prefix_valid.contiguous(),
        work_block=_i32(work_block), work_start=_i32(work_start), work_end=_i32(work_end), work_seq=_i32(work_seq),
    )


def verify_against_host(layout: RoutingLayout) -> None:
    """调试模式：GPU 构造的 layout 与 host 值逐元素一致，SeqBlockPtr 与 BlockSeq 互相一致。需要同步。"""
    t = {n: x.tolist() for n, x in layout.tensors().items()}
    if tuple(t["block_ptr"]) != layout.block_ptr_host:
        raise fail("invalid_cu_seqlens", "GPU BlockPtr 与 BlockPtr_host 不一致")
    if tuple(t["seq_block_ptr"]) != layout.seq_block_ptr_host:
        raise fail("invalid_cu_seqlens", "GPU SeqBlockPtr 与 host 不一致")
    sbp = t["seq_block_ptr"]
    for b, s in enumerate(t["block_seq"]):
        if not sbp[s] <= b < sbp[s + 1]:
            raise fail("invalid_cu_seqlens", f"BlockSeq[{b}] = {s} 与 SeqBlockPtr 不一致")
    if len(t["work_block"]) != layout.Nwork or len(t["current_block"]) != layout.T:
        raise fail("invalid_cu_seqlens", "QueryWorkList / CurrentBlock 长度与 host 计算的 Nwork / T 不一致")


__all__ = ["RoutingLayout", "build_block_layout", "verify_against_host"]
