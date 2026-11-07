# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Strict-history top-k selection and prefix-state route masks.'

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from fla.ops.phba.errors import fail
from fla.ops.phba.plans import MAX_KH, MIN_DOT, kpad_tok

if TYPE_CHECKING:
    from fla.ops.phba.routing.layout import RoutingLayout

TOPK_NUM_WARPS = 4
TOPK_NUM_STAGES = 2


@triton.jit(do_not_specialize=["Kh", "stride_rt_h", "stride_rt_t", "stride_rs_h", "stride_rs_t"])
def _topk_kernel(
    q_ptr, cen_ptr, work_block_ptr, work_start_ptr, work_end_ptr, work_seq_ptr, seq_block_ptr, prefix_valid_ptr,
    rtok_ptr, rstate_ptr,
    stride_qt, stride_qh,
    Kh, stride_rt_h, stride_rt_t, stride_rs_h, stride_rs_t,
    H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BR_DOT: tl.constexpr, BC: tl.constexpr,
    BLOCK_D: tl.constexpr, KPAD: tl.constexpr,
):
    w = tl.program_id(0)
    h = tl.program_id(1)
    cb = tl.load(work_block_ptr + w)
    start = tl.load(work_start_ptr + w)
    end = tl.load(work_end_ptr + w)
    seq = tl.load(work_seq_ptr + w)
    b0 = tl.load(seq_block_ptr + seq)

    lane = tl.arange(0, BR_DOT)
    rows = start + lane
    qvalid = (lane < BR) & (rows < end)
    rows64 = rows.to(tl.int64)
    o_d = tl.arange(0, BLOCK_D)
    kcol = tl.arange(0, KPAD)
    bcol = tl.arange(0, BC)

    best_s = tl.full([BR_DOT, KPAD], float("-inf"), tl.float32)
    best_i = tl.full([BR_DOT, KPAD], -1, tl.int32)
    for blk0 in range(b0, cb, BC):
        bids = blk0 + bcol
        bvalid = bids < cb
        score = tl.zeros([BR_DOT, BC], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            m_d = (d0 + o_d) < D
            q = tl.load(q_ptr + rows64[:, None] * stride_qt + h * stride_qh + (d0 + o_d)[None, :],
                        mask=qvalid[:, None] & m_d[None, :], other=0.0)
            c = tl.load(cen_ptr + (bids.to(tl.int64)[:, None] * H + h) * D + (d0 + o_d)[None, :],
                        mask=bvalid[:, None] & m_d[None, :], other=0.0)
            score += tl.dot(q, tl.trans(c.to(q.dtype)), out_dtype=tl.float32)
        score = tl.where(qvalid[:, None] & bvalid[None, :], score, float("-inf"))

        # 固定 KPAD 轮的选择合并：已有区（rank 顺序）与新 tile（block id 递增）
        old_s = best_s
        old_i = best_i
        new_s = tl.full([BR_DOT, KPAD], float("-inf"), tl.float32)
        new_i = tl.full([BR_DOT, KPAD], -1, tl.int32)
        for r in range(KPAD):
            m_old = tl.max(old_s, axis=1)
            a_old = tl.argmax(old_s, axis=1, tie_break_left=True)
            m_new = tl.max(score, axis=1)
            a_new = tl.argmax(score, axis=1, tie_break_left=True)
            # ties retain the earlier candidate with the smaller block id
            take_new = m_new > m_old
            sel_s = tl.where(take_new, m_new, m_old)
            id_old = tl.sum(tl.where(kcol[None, :] == a_old[:, None], old_i, 0), axis=1)
            id_new = blk0 + a_new
            sel_i = tl.where(take_new, id_new, id_old)
            # invalid scores retain the route sentinel
            sel_i = tl.where(sel_s > float("-inf"), sel_i, -1)
            new_s = tl.where(kcol[None, :] == r, sel_s[:, None], new_s)
            new_i = tl.where(kcol[None, :] == r, sel_i[:, None], new_i)
            old_s = tl.where((m_new <= m_old)[:, None] & (kcol[None, :] == a_old[:, None]), float("-inf"), old_s)
            score = tl.where(take_new[:, None] & (bcol[None, :] == a_new[:, None]), float("-inf"), score)
        best_s = new_s
        best_i = new_i

    slot_ok = kcol < Kh
    m_store = qvalid[:, None] & slot_ok[None, :]
    tl.store(rtok_ptr + h * stride_rt_h + rows64[:, None] * stride_rt_t + kcol[None, :], best_i, mask=m_store)

    pv_cur = tl.load(prefix_valid_ptr + cb).to(tl.int32)
    cur = tl.where(pv_cur != 0, cb, -1)
    tl.store(rstate_ptr + h * stride_rs_h + rows64 * stride_rs_t, tl.zeros([BR_DOT], dtype=tl.int32) + cur,
             mask=qvalid)
    has = best_i >= 0
    pv = tl.load(prefix_valid_ptr + tl.where(has, best_i, 0), mask=has, other=0).to(tl.int32)
    hist = tl.where(has & (pv != 0), best_i, -1)
    tl.store(rstate_ptr + h * stride_rs_h + rows64[:, None] * stride_rs_t + 1 + kcol[None, :], hist, mask=m_store)


def route_topk(q_rope: torch.Tensor, centroids: torch.Tensor, layout: RoutingLayout, kh: int, *, bc: int = 32,
               block_d: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (Rtok [H, T, Kh], Rstate [H, T, Kh + 1])，int32，-1 无效。不可导（输入按 detach 处理）。"""
    if not 0 <= kh <= MAX_KH:
        raise fail("routing_unsupported", f"Kh = {kh} 不在 [0, {MAX_KH}]（KPAD_tok <= 16）")
    T, H, D = q_rope.shape
    if T != layout.T or centroids.shape != (layout.Nc, H, D):
        raise fail("invalid_shape", f"Qrope {tuple(q_rope.shape)} / Centroid {tuple(centroids.shape)} 与 layout 不一致")
    if q_rope.stride(-1) != 1 or not centroids.is_contiguous():
        raise fail("non_contiguous_input", "Qrope 最后一维与 Centroid 必须连续")
    if centroids.dtype != torch.float32:
        raise fail("dtype_mismatch", f"Centroid 必须是 fp32，收到 {centroids.dtype}")
    if bc < MIN_DOT or block_d < MIN_DOT:
        raise fail("invalid_config", f"Bc / BLOCK_D 必须 >= {MIN_DOT}")
    dev = q_rope.device
    rtok = torch.empty((H, T, kh), dtype=torch.int32, device=dev)
    if kh == 0:
        cur = torch.where(layout.prefix_valid[layout.current_block.long()], layout.current_block,
                          torch.full_like(layout.current_block, -1))
        rstate = cur.view(1, T, 1).expand(H, T, 1).contiguous()
        return rtok, rstate
    rstate = torch.empty((H, T, kh + 1), dtype=torch.int32, device=dev)
    br = layout.br
    _topk_kernel[(layout.Nwork, H)](
        q_rope.detach(), centroids, layout.work_block, layout.work_start, layout.work_end, layout.work_seq,
        layout.seq_block_ptr, layout.prefix_valid.view(torch.uint8), rtok, rstate,
        q_rope.stride(0), q_rope.stride(1),
        kh, rtok.stride(0), rtok.stride(1), rstate.stride(0), rstate.stride(1),
        H=H, D=D, BR=br, BR_DOT=max(br, MIN_DOT), BC=bc, BLOCK_D=block_d, KPAD=kpad_tok(kh),
        num_warps=TOPK_NUM_WARPS, num_stages=TOPK_NUM_STAGES,
    )
    return rtok, rstate


# references
def _history_mask(layout: RoutingLayout, device) -> torch.Tensor:
    """[T, Nc] bool：BlockSeq[b] == seq(t) 且 b < CurrentBlock[t]。"""
    cb = layout.current_block.to(device).long()
    bs = layout.block_seq.to(device).long()
    b = torch.arange(layout.Nc, device=device)
    return (bs[None, :] == bs[cb][:, None]) & (b[None, :] < cb[:, None])


def routing_scores(q_rope: torch.Tensor, centroids: torch.Tensor, layout: RoutingLayout, *, production: bool):
    """[H, T, Nc] 分数，无效候选为 -inf。

    production=True ：Centroid 先转为 Qrope 的 dtype（与 kernel 相同的 cast），乘积与求和在 fp64 中计算后舍入为 fp32；
                      bf16 / fp16 的乘积在 fp32 / fp64 中都是精确的，所以只剩“求和顺序”一项与 kernel 不同
    production=False：fp32 oracle，Qrope.float() @ Centroid.float().T（Centroid 不 cast），TF32 关闭
    """
    dev = q_rope.device
    mask = _history_mask(layout, dev)
    if production:
        q = q_rope.detach().to(torch.float64)
        c = centroids.to(q_rope.dtype).to(torch.float64)
        s = torch.einsum("thd,bhd->htb", q, c).to(torch.float32)
    else:
        from fla.ops.phba.naive import highest_precision

        with highest_precision():
            s = torch.einsum("thd,bhd->htb", q_rope.detach().float(), centroids.float())
    return s.masked_fill(~mask[None], float("-inf"))


def topk_from_scores(scores: torch.Tensor, kh: int) -> torch.Tensor:
    """稳定 Top-K：分数降序，同分时 block id 较小者优先；-inf 记为 -1。scores [H, T, Nc] → [H, T, Kh] int32。"""
    H, T, nc = scores.shape
    if kh == 0 or nc == 0:
        return torch.full((H, T, kh), -1, dtype=torch.int32, device=scores.device)
    vals, idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    k = min(kh, nc)
    ids = torch.where(vals[..., :k] > float("-inf"), idx[..., :k], torch.full_like(idx[..., :k], -1))
    out = torch.full((H, T, kh), -1, dtype=torch.int32, device=scores.device)
    out[..., :k] = ids.to(torch.int32)
    return out


def rstate_from_rtok(rtok: torch.Tensor, layout: RoutingLayout) -> torch.Tensor:
    H, T, kh = rtok.shape
    pv = layout.prefix_valid.to(rtok.device)
    cb = layout.current_block.to(rtok.device).long()
    cur = torch.where(pv[cb], cb, torch.full_like(cb, -1)).to(torch.int32).view(1, T, 1).expand(H, T, 1)
    hist = torch.where((rtok >= 0) & pv[rtok.long().clamp(min=0)], rtok, torch.full_like(rtok, -1))
    return torch.cat([cur, hist], dim=-1).contiguous()


def route_reference(q_rope, centroids, layout: RoutingLayout, kh: int):
    """production-matched routing reference：返回 (Rtok, Rstate, scores)。"""
    scores = routing_scores(q_rope, centroids, layout, production=True)
    rtok = topk_from_scores(scores, kh)
    return rtok, rstate_from_rtok(rtok, layout), scores


def compare_routes(rtok: torch.Tensor, ref: torch.Tensor, scores: torch.Tensor, *, tie_tol: float) -> dict:
    """kernel route 与 reference 逐行比较（集合相同即匹配；路由的写回顺序也按 rank，额外报告逐位一致）。

    不一致的行只允许出现在 reference 的“近并列”处：对称差中每个候选的 reference 分数与该行第 Kh 名的分数之差都 <= tie_tol
    （kernel 的 tl.dot 与 reference 的求和顺序不同，只会在这种情况下改变选择）。
    """
    H, T, kh = rtok.shape
    exact = bool(torch.equal(rtok, ref))
    a = torch.sort(rtok, dim=-1).values
    b = torch.sort(ref, dim=-1).values
    differ = (a != b).any(-1)
    n_differ = int(differ.sum())
    bad = 0
    if n_differ:
        hh, tt = differ.nonzero(as_tuple=True)
        for h, t in zip(hh.tolist(), tt.tolist()):
            sa = set(rtok[h, t].tolist()) - {-1}
            sb = set(ref[h, t].tolist()) - {-1}
            ref_valid = [x for x in ref[h, t].tolist() if x >= 0]
            boundary = float(scores[h, t, ref_valid[-1]]) if ref_valid else float("-inf")
            for x in sa ^ sb:
                if not abs(float(scores[h, t, x]) - boundary) <= tie_tol:
                    bad += 1
                    break
    return {"exact": exact, "rows": H * T, "rows_differ": n_differ, "rows_differ_not_near_tie": bad}


def routing_fidelity(rtok: torch.Tensor, oracle_scores: torch.Tensor) -> dict:
    """fp32 oracle 的 fidelity 报告（只报告）：route 完全一致率、Top-K 重合率、Top-1 一致率、第 Kh / Kh+1 名分差分布。"""
    H, T, kh = rtok.shape
    if kh == 0:
        return {}
    oracle = topk_from_scores(oracle_scores, kh)
    valid_rows = (oracle >= 0).any(-1)
    n = int(valid_rows.sum())
    if n == 0:
        return {"rows": 0}
    same_set = (torch.sort(rtok, -1).values == torch.sort(oracle, -1).values).all(-1) & valid_rows
    inter = ((rtok[..., :, None] == oracle[..., None, :]) & (rtok[..., :, None] >= 0)).any(-1).sum(-1).float()
    denom = (oracle >= 0).sum(-1).clamp(min=1).float()
    top1 = (rtok[..., 0] == oracle[..., 0]) & valid_rows
    vals = torch.sort(oracle_scores, dim=-1, descending=True).values
    gaps = None
    if vals.shape[-1] > kh:
        g = vals[..., kh - 1] - vals[..., kh]
        g = g[torch.isfinite(g)]
        if g.numel():
            q = torch.quantile(g.double().cpu(), torch.tensor([0.0, 0.01, 0.5], dtype=torch.float64))
            gaps = {"min": float(q[0]), "p01": float(q[1]), "median": float(q[2])}
    return {
        "rows": n,
        "route_exact_rate": float(same_set.sum()) / n,
        "topk_overlap": float((inter / denom)[valid_rows].mean()),
        "top1_agreement": float(top1.sum()) / n,
        "oracle_gap_k_k1": gaps,
    }


def check_route_invariants(rtok: torch.Tensor, rstate: torch.Tensor, layout: RoutingLayout) -> None:
    """调试模式：distinct、[0, Nc)、同一 sequence、< CurrentBlock；Rstate slot 0 / PrefixValid。需要同步。"""
    H, T, kh = rtok.shape
    dev = rtok.device
    cb = layout.current_block.to(dev).long()
    bs = layout.block_seq.to(dev).long()
    r = rtok.long()
    valid = r >= 0
    if bool((r[valid] >= layout.Nc).any()) or bool((r < -1).any()):
        raise fail("invalid_shape", "Rtok 有越界的 block id")
    safe = r.clamp(min=0)
    if bool((valid & (bs[safe] != bs[cb][None, :, None])).any()):
        raise fail("invalid_shape", "Rtok 含其他 sequence 的 block")
    if bool((valid & (safe >= cb[None, :, None])).any()):
        raise fail("invalid_shape", "Rtok 含非严格历史的 block")
    if kh > 1:
        s = torch.sort(torch.where(valid, r, torch.arange(-kh, 0, device=dev).expand_as(r)), -1).values
        if bool((s[..., 1:] == s[..., :-1]).any()):
            raise fail("invalid_shape", "Rtok 同一行有重复 block id")
    expect = rstate_from_rtok(rtok, layout)
    if not torch.equal(rstate, expect):
        raise fail("invalid_shape", "Rstate 与 [current; Rtok] + PrefixValid 不一致")


__all__ = [
    "check_route_invariants", "compare_routes", "route_reference", "route_topk", "routing_fidelity", "routing_scores",
    "rstate_from_rtok", "topk_from_scores",
]
