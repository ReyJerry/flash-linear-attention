# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Algorithm 1 的 block layout（triton化方案.md 2.2 节、2.6 节、4.2 节、4.3 节、4.11 节、4.17 节）。

    BlockPtr      [Nc + 1] int32   block b 的 token 区间 [BlockPtr[b], BlockPtr[b+1])；BlockPtr[Nc] = T
    SeqBlockPtr   [Nseq + 1] int32 sequence s 的 block 区间 [SeqBlockPtr[s], SeqBlockPtr[s+1])
    BlockSeq      [Nc] int32       block b 所属的 sequence
    CurrentBlock  [T] int32        token t 所在的 block
    PrefixValid   [Nc] bool        b > SeqBlockPtr[BlockSeq[b]]（每个 sequence 的首 block 为 0）
    QueryWork*    [Nwork] int32    Block / Start / End / Sequence：每项是一个 block 内的 Br 个 query 行，不跨 block、不跨 sequence

全部在 GPU 上由 SequenceLengthPlan 的 device cu_seqlens 构造（第一阶段用 PyTorch tensor 操作，4.2 节允许）：
tensor 尺寸（Nc、Nwork、T）全部来自 host 计算的值，不用 device 上的 0 维 tensor 作尺寸，不调用 nonzero / .item()，
repeat_interleave 显式给 output_size，因此不引入 device→host 同步。结果缓存在 SequenceLengthPlan 内（按 (C, Br)）。

调试模式（``verify_against_host``）逐元素比较 GPU 结果与 host 计算的 BlockPtr / SeqBlockPtr / Nwork（需要同步）。
"""

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
    _replay_checkpoints: dict = field(default_factory=dict, repr=False, compare=False)

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
    # BlockPtr / SeqBlockPtr 与 prefix/chunkwise 共用 SequenceLengthPlan.block_layout(C) 的 device 副本（同一份 source of truth）
    blayout = lp.block_layout(c)
    block_ptr = blayout.block_ptr.to(torch.int64)
    seq_block_ptr = blayout.seq_block_ptr.to(torch.int64)
    nblk = seq_block_ptr[1:] - seq_block_ptr[:-1]
    block_seq = torch.repeat_interleave(torch.arange(nseq, device=dev), nblk, output_size=nc)       # [Nc]
    block_len = block_ptr[1:] - block_ptr[:-1]                            # [Nc]，最后一个 block 可能不满
    current_block = torch.repeat_interleave(torch.arange(nc, device=dev), block_len, output_size=T)  # [T]
    prefix_valid = torch.arange(nc, device=dev) > seq_block_ptr[block_seq]                         # [Nc] bool
    nq = torch.div(block_len + (br - 1), br, rounding_mode="floor")       # 每个 block 的 query tile 数
    work_block = torch.repeat_interleave(torch.arange(nc, device=dev), nq, output_size=nwork)       # [Nwork]
    work_ptr = torch.zeros(nc + 1, dtype=torch.int64, device=dev)
    work_ptr[1:] = torch.cumsum(nq, 0)
    tile = torch.arange(nwork, device=dev) - work_ptr[work_block]
    work_start = block_ptr[work_block] + tile * br
    work_end = torch.minimum(work_start + br, block_ptr[work_block + 1])
    work_seq = block_seq[work_block]
    return RoutingLayout(
        chunk_size=c, br=br, Nseq=nseq, Nc=nc, Nwork=nwork, T=T,
        _replay_checkpoints=lp._replay_checkpoints,
        block_ptr_host=block_ptr_host, seq_block_ptr_host=seq_block_ptr_host,
        block_ptr=_i32(block_ptr), seq_block_ptr=_i32(seq_block_ptr), block_seq=_i32(block_seq),
        current_block=_i32(current_block), prefix_valid=prefix_valid.contiguous(),
        work_block=_i32(work_block), work_start=_i32(work_start), work_end=_i32(work_end), work_seq=_i32(work_seq),
    )


def verify_against_host(layout: RoutingLayout) -> None:
    """调试模式（4.2 节）：GPU 构造的 layout 与 host 值逐元素一致，SeqBlockPtr 与 BlockSeq 互相一致。需要同步。"""
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
