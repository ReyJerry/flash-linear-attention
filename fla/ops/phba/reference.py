# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Storage-aware PyTorch reference for PHBA prefix states and joint attention.'

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HostLayout:
    """Reference 只需要 host 上的 block 划分（与 plans.BlockLayout 的 host 部分相同）。"""

    chunk_size: int
    block_ptr: tuple[int, ...]
    seq_block_ptr: tuple[int, ...]

    @property
    def Nc(self) -> int:  # noqa: N802
        return len(self.block_ptr) - 1

    @property
    def Nseq(self) -> int:  # noqa: N802
        return len(self.seq_block_ptr) - 1

    @classmethod
    def from_lengths(cls, lengths, chunk_size: int) -> HostLayout:
        block_ptr, seq_block_ptr, t = [0], [0], 0
        for length in lengths:
            n = -(-length // chunk_size)
            for j in range(n):
                t += min(chunk_size, length - j * chunk_size)
                block_ptr.append(t)
            seq_block_ptr.append(seq_block_ptr[-1] + n)
        return cls(chunk_size, tuple(block_ptr), tuple(seq_block_ptr))

    @classmethod
    def from_block_layout(cls, layout) -> HostLayout:
        return cls(layout.chunk_size, tuple(layout.block_ptr_host), tuple(layout.seq_block_ptr_host))

    def blocks_of_sequence(self, s: int) -> range:
        return range(self.seq_block_ptr[s], self.seq_block_ptr[s + 1])

    def token_block(self, device=None) -> torch.Tensor:
        lens = [self.block_ptr[i + 1] - self.block_ptr[i] for i in range(self.Nc)]
        return torch.repeat_interleave(torch.arange(self.Nc, device=device), torch.tensor(lens, device=device))


def compute_dtype(storage: torch.dtype) -> torch.dtype:
    return torch.float64 if storage == torch.float64 else torch.float32


def _st(x: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    """cast 到 storage 再回到 compute dtype（模拟“storage 操作数进入 dot / 写回 HBM”）。"""
    return x.to(storage).to(x.dtype)


# Prefix-State（chunkwise）
def local_cumsum(x: torch.Tensor, layout: HostLayout, *, reverse: bool = False, dtype: torch.dtype | None = None):
    cd = dtype or compute_dtype(x.dtype)
    out = torch.empty(x.shape, dtype=cd, device=x.device)
    for b in range(layout.Nc):
        r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
        seg = x[r0:r1].to(cd)
        out[r0:r1] = seg.flip(0).cumsum(0).flip(0) if reverse else seg.cumsum(0)
    return out


def prefix_forward(s, z, gk, layout: HostLayout):
    """s: [T, H, M] storage；z: [T, H, D] storage；gk: [T, H, M] compute。

    返回 (Prefix [Nc, M, H, D] storage, final state [Nseq, H, M, D] compute dtype)。
    """
    storage, cd = z.dtype, gk.dtype
    T, H, M = s.shape
    D = z.shape[-1]
    out = torch.empty((layout.Nc, M, H, D), dtype=storage, device=z.device)
    finals = []
    for seq in range(layout.Nseq):
        S = torch.zeros((H, M, D), dtype=cd, device=z.device)
        for b in layout.blocks_of_sequence(seq):
            out[b] = S.permute(1, 0, 2).to(storage)
            r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
            g = gk[r0:r1]
            g_last = g[-1]
            S = S * torch.exp(g_last)[..., None]
            W = _st(s[r0:r1].to(cd) * torch.exp(g_last[None] - g), storage)
            S = S + torch.einsum("lhm,lhd->hmd", W, z[r0:r1].to(cd))
        finals.append(S)
    return out, torch.stack(finals, 0)


def prefix_backward_branch(s, z, gk, prefix, dprefix, layout: HostLayout):
    """_prefix_bwd_kernel 的公式（一个 branch）。返回 compute dtype 的 (ds [T,H,M], dz [T,H,D], dg [T,H,M])。"""
    storage, cd = z.dtype, gk.dtype
    T, H, M = s.shape
    D = z.shape[-1]
    ds = torch.zeros((T, H, M), dtype=cd, device=z.device)
    dz = torch.zeros((T, H, D), dtype=cd, device=z.device)
    dg = torch.zeros((T, H, M), dtype=cd, device=z.device)
    for seq in range(layout.Nseq):
        grad_next = torch.zeros((H, M, D), dtype=cd, device=z.device)
        for b in reversed(layout.blocks_of_sequence(seq)):
            r0, r1 = layout.block_ptr[b], layout.block_ptr[b + 1]
            g = gk[r0:r1]
            g_last = g[-1]
            eg = torch.exp(g_last)
            k_f = s[r0:r1].to(cd)
            exp_diff = torch.exp(g_last[None] - g)
            gn_st = _st(grad_next, storage)
            v_val = z[r0:r1].to(cd)
            dot_lk = torch.einsum("lhd,hmd->lhm", v_val, gn_st)
            grad_k = exp_diff * dot_lk
            k_scaled = _st(k_f * exp_diff, storage)
            grad_v = torch.einsum("lhm,hmd->lhd", k_scaled, gn_st)
            grad_g = -(k_f * grad_k)
            sum_k_gradk = (k_f * grad_k).sum(0)
            h_val = prefix[b].to(cd).permute(1, 0, 2)
            dot_state = (h_val * grad_next).sum(-1)
            grad_g[-1] = grad_g[-1] + (eg * dot_state + sum_k_gradk)
            ds[r0:r1], dz[r0:r1], dg[r0:r1] = grad_k, grad_v, grad_g
            grad_next = dprefix[b].to(cd).permute(1, 0, 2) + eg[..., None] * grad_next
    return ds, dz, dg


class RefChunkwisePrefixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_val, k_raw, v, g_log, layout: HostLayout):
        cd = compute_dtype(k_raw.dtype)
        gk = local_cumsum(g_log, layout, dtype=cd)
        pk, _ = prefix_forward(s_val, k_raw, gk, layout)
        pv, _ = prefix_forward(s_val, v, gk, layout)
        ctx.layout = layout
        ctx.save_for_backward(s_val, k_raw, v, g_log, pk, pv)
        return pk, pv

    @staticmethod
    def backward(ctx, dpk, dpv):
        s_val, k_raw, v, g_log, pk, pv = ctx.saved_tensors
        layout = ctx.layout
        cd = compute_dtype(k_raw.dtype)
        gk = local_cumsum(g_log, layout, dtype=cd)
        dpk = torch.zeros_like(pk) if dpk is None else dpk
        dpv = torch.zeros_like(pv) if dpv is None else dpv
        ds_k, dk, dg_k = prefix_backward_branch(s_val, k_raw, gk, pk, dpk, layout)
        ds_v, dv, dg_v = prefix_backward_branch(s_val, v, gk, pv, dpv, layout)
        d_g_log = local_cumsum(dg_k + dg_v, layout, reverse=True, dtype=cd)
        return (ds_k + ds_v).to(s_val.dtype), dk.to(k_raw.dtype), dv.to(v.dtype), d_g_log.to(g_log.dtype), None


# joint attention（冻结 route）
def route_masks(rtok: torch.Tensor, rstate: torch.Tensor, layout: HostLayout, h: int, M: int):
    """返回 (local [T,T], token [T,T], state [T, Nc*M]) 三个 bool mask（head h）。"""
    T = rtok.shape[1]
    dev = rtok.device
    tb = layout.token_block(dev)
    idx = torch.arange(T, device=dev)
    local = (tb[:, None] == tb[None, :]) & (idx[None, :] <= idx[:, None])

    def block_mask(routes: torch.Tensor) -> torch.Tensor:
        m = torch.zeros((T, layout.Nc + 1), dtype=torch.bool, device=dev)       # 最后一列收集 -1
        ids = routes[h].long().clamp(min=-1)
        ids = torch.where(ids < 0, torch.full_like(ids, layout.Nc), ids)
        m.scatter_(1, ids, True)
        return m[:, : layout.Nc]

    token = block_mask(rtok)[:, tb] if rtok.shape[-1] > 0 else torch.zeros((T, T), dtype=torch.bool, device=dev)
    state = block_mask(rstate).repeat_interleave(M, dim=1)
    return local, token, state


def _masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask, float("-inf"))


def _rowmax(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=-1).values if x.shape[-1] else torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)


def _joint_forward_head(q, k, vv, qr, pk, pv, masks, scale, storage, local_seed=None):
    local, token, state = masks
    s_qk = (q @ k.T) * scale
    s_loc = _masked(s_qk, local)
    if local_seed is None:
        m_loc = _rowmax(s_loc)
        p_loc = torch.exp(s_loc - m_loc[:, None])
        l_loc = p_loc.sum(-1)
        o_loc = _st((_st(p_loc, storage) @ vv) / l_loc[:, None], storage)
        lse_loc = m_loc + torch.log(l_loc)
    else:
        o_loc, lse_loc = local_seed
    s_tok = _masked(s_qk, token)
    s_st = _masked((qr @ pk.T) * scale, state)
    m = torch.maximum(lse_loc, torch.maximum(_rowmax(s_tok), _rowmax(s_st)))
    p_tok = torch.exp(s_tok - m[:, None])
    p_st = torch.exp(s_st - m[:, None])
    z = torch.exp(lse_loc - m) + p_tok.sum(-1) + p_st.sum(-1)
    u = o_loc * torch.exp(lse_loc - m)[:, None] + _st(p_tok, storage) @ vv + _st(p_st, storage) @ pv
    return (u / z[:, None]).to(storage), m + torch.log(z), (s_loc, s_tok, s_st)


class RefJointAttentionFn(torch.autograd.Function):
    """输入 q_rope, k_rope, v, q_raw [T,H,D] 与 prefix_k, prefix_v [Nc,M,H,D]（storage）；输出 O [T,H,D] storage 与 L [H,T]。"""

    @staticmethod
    def forward(ctx, q_rope, k_rope, v, q_raw, prefix_k, prefix_v, rtok, rstate, layout: HostLayout, scale: float,
                local_impl='torch', fa_segment_batch_max=65535, fa_rows_per_call_max=1 << 20, fa_deterministic=False):
        if local_impl not in ('torch', 'flash_attn'):
            raise ValueError(f'unknown Reference A local_impl: {local_impl}')
        storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
        T, H, D = q_rope.shape
        M = prefix_k.shape[1]
        ctx.local_impl, ctx.fa_deterministic = local_impl, fa_deterministic
        oloc = lloc = None
        extra = []
        if local_impl == 'flash_attn':
            from types import SimpleNamespace

            from fla.ops.phba.flash_attention import ensure_ready, local_attention
            from fla.ops.phba.plans import _local_fa_batches

            if q_rope.device.type != 'cuda' or storage not in (torch.float16, torch.bfloat16):
                raise ValueError('FlashAttention Reference A requires CUDA fp16/bf16 inputs')
            if any(x.device != q_rope.device or x.dtype != storage or x.shape != q_rope.shape
                   or not x.is_contiguous() or x.data_ptr() % 16 for x in (q_rope, k_rope, v)):
                raise ValueError('FlashAttention Reference A requires matching aligned contiguous Q/K/V')
            if fa_segment_batch_max < 1 or fa_rows_per_call_max < 1:
                raise ValueError('FlashAttention batch limits must be positive')
            ensure_ready(q_rope.device, storage, D)
            block_ptr = torch.tensor(layout.block_ptr, device=q_rope.device, dtype=torch.int32)
            fa_layout = SimpleNamespace(chunk_size=layout.chunk_size, block_ptr_host=layout.block_ptr,
                                        block_ptr=block_ptr)
            ctx.batches = _local_fa_batches(layout.block_ptr, layout.chunk_size,
                                            fa_segment_batch_max, fa_rows_per_call_max)
            oloc, lloc = local_attention(q_rope, k_rope, v, fa_layout, scale=scale,
                                         fa_segment_batch_max=fa_segment_batch_max,
                                         fa_rows_per_call_max=fa_rows_per_call_max)
            extra.append(block_ptr)
        out = torch.empty((T, H, D), dtype=storage, device=q_rope.device)
        lse = torch.empty((H, T), dtype=cd, device=q_rope.device)
        for h in range(H):
            masks = route_masks(rtok, rstate, layout, h, M)
            seed = None if oloc is None else (oloc[:, h].to(cd), lloc[h].to(cd))
            o, l, _ = _joint_forward_head(
                q_rope[:, h].to(cd), k_rope[:, h].to(cd), v[:, h].to(cd), q_raw[:, h].to(cd),
                prefix_k[:, :, h].reshape(-1, D).to(cd), prefix_v[:, :, h].reshape(-1, D).to(cd),
                masks, scale, storage, local_seed=seed)
            out[:, h], lse[h] = o, l
        ctx.layout, ctx.scale = layout, scale
        ctx.save_for_backward(q_rope, k_rope, v, q_raw, prefix_k, prefix_v, rtok, rstate, out, lse, *extra)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, d_out, _d_lse):
        q_rope, k_rope, v, q_raw, prefix_k, prefix_v, rtok, rstate, out, lse, *extra = ctx.saved_tensors
        layout, scale = ctx.layout, ctx.scale
        storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
        T, H, D = q_rope.shape
        nc, M = prefix_k.shape[0], prefix_k.shape[1]
        grads = {n: torch.zeros((T, H, D), dtype=cd, device=q_rope.device) for n in ("dq", "dk", "dv", "dqr")}
        dpk = torch.zeros((nc, M, H, D), dtype=cd, device=q_rope.device)
        dpv = torch.zeros_like(dpk)
        d_out = d_out.to(storage).contiguous()
        local_grads = None
        if ctx.local_impl == 'flash_attn':
            from fla.ops.phba.flash_attention import local_backward

            local_grads = local_backward(d_out, q_rope, k_rope, v, out, lse, extra[0], ctx.batches,
                                         deterministic=ctx.fa_deterministic, softmax_scale=scale)
        for h in range(H):
            masks = route_masks(rtok, rstate, layout, h, M)
            q, k, vv, qr = (x[:, h].to(cd) for x in (q_rope, k_rope, v, q_raw))
            pk = prefix_k[:, :, h].reshape(-1, D).to(cd)
            pv = prefix_v[:, :, h].reshape(-1, D).to(cd)
            s_qk = (q @ k.T) * scale
            s_loc, s_tok = _masked(s_qk, masks[0]), _masked(s_qk, masks[1])
            s_st = _masked((qr @ pk.T) * scale, masks[2])
            L = lse[h]
            do = d_out[:, h].to(cd)
            delta = (do * out[:, h].to(cd)).sum(-1)
            do_st = _st(do, storage)

            def branch(s, q_in, k_in, v_in):
                p = torch.exp(s - L[:, None])
                dp = do_st @ v_in.T
                ds = p * (dp - delta[:, None])
                dq = (_st(ds, storage) @ k_in) * scale
                dk = (_st(ds, storage).T @ q_in) * scale
                dv = _st(p, storage).T @ do_st
                return dq, dk, dv

            if local_grads is None:
                dq_l, dk_l, dv_l = (_st(x, storage) for x in branch(s_loc, q, k, vv))
            else:
                dq_l, dk_l, dv_l = (x[:, h].to(cd) for x in local_grads)
            dq_t, dk_t, dv_t = branch(s_tok, q, k, vv)
            dqr, dpk_h, dpv_h = branch(s_st, qr, pk, pv)
            grads["dq"][:, h] = dq_l + dq_t
            grads["dk"][:, h] = dk_l + dk_t
            grads["dv"][:, h] = dv_l + dv_t
            grads["dqr"][:, h] = dqr
            dpk[:, :, h] = dpk_h.reshape(nc, M, D)
            dpv[:, :, h] = dpv_h.reshape(nc, M, D)
        result = (grads["dq"].to(storage), grads["dk"].to(storage), grads["dv"].to(storage), grads["dqr"].to(storage),
                  dpk.to(storage), dpv.to(storage))
        return result + (None,) * (len(ctx.needs_input_grad) - len(result))


def rstate_from_rtok_host(rtok: torch.Tensor, layout: HostLayout) -> torch.Tensor:
    """Rstate = [current_block; Rtok]，PrefixValid = 0（每个 sequence 的首 block）的 entry 为 -1。"""
    H, T, kh = rtok.shape
    dev = rtok.device
    tb = layout.token_block(dev)
    first = torch.zeros(layout.Nc, dtype=torch.bool, device=dev)
    first[list(layout.seq_block_ptr[:-1])] = True
    pv = ~first
    cur = torch.where(pv[tb], tb, torch.full_like(tb, -1)).to(torch.int32).view(1, T, 1).expand(H, T, 1)
    hist = torch.where((rtok >= 0) & pv[rtok.long().clamp(min=0)], rtok, torch.full_like(rtok, -1))
    return torch.cat([cur, hist], dim=-1)


def alg6_reference(q_rope, q_raw, k_rope, v, prefix_k, prefix_v, oloc, lloc, rtok, layout: HostLayout, scale: float):
    """joint-attention merge 的参照：以给定的 (Oloc storage, Lloc fp32 [H, T]) 为种子，在 current prefix、历史 token、
    历史 aligned prefix 上做一次 joint softmax（与 kernel 的逐 tile online-softmax 代数等价）。
    P 在进入 dot 前转为 storage dtype；返回 (O storage [T,H,D], L compute [H,T])。"""
    storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
    T, H, D = q_rope.shape
    M = prefix_k.shape[1]
    rstate = rstate_from_rtok_host(rtok, layout)
    out = torch.empty((T, H, D), dtype=storage, device=q_rope.device)
    lse = torch.empty((H, T), dtype=cd, device=q_rope.device)
    for h in range(H):
        _, token, state = route_masks(rtok, rstate, layout, h, M)
        q, k, vv, qr = (x[:, h].to(cd) for x in (q_rope, k_rope, v, q_raw))
        pk = prefix_k[:, :, h].reshape(-1, D).to(cd)
        pv = prefix_v[:, :, h].reshape(-1, D).to(cd)
        s_tok = _masked((q @ k.T) * scale, token)
        s_st = _masked((qr @ pk.T) * scale, state)
        l0 = lloc[h].to(cd)
        m = torch.maximum(l0, torch.maximum(_rowmax(s_tok), _rowmax(s_st)))
        p_tok = torch.exp(s_tok - m[:, None])
        p_st = torch.exp(s_st - m[:, None])
        a0 = torch.exp(l0 - m)
        z = a0 + p_tok.sum(-1) + p_st.sum(-1)
        u = oloc[:, h].to(cd) * a0[:, None] + _st(p_tok, storage) @ vv + _st(p_st, storage) @ pv
        out[:, h] = (u / z[:, None]).to(storage)
        lse[h] = m + torch.log(z)
    return out, lse


def local_reference(q_rope, k_rope, v, layout: HostLayout, scale: float):
    """local 分支（当前 block 内因果）的公式参照：返回 (Oloc storage [T,H,D], Lloc compute [H,T])。"""
    storage, cd = q_rope.dtype, compute_dtype(q_rope.dtype)
    T, H, D = q_rope.shape
    tb = layout.token_block(q_rope.device)
    idx = torch.arange(T, device=q_rope.device)
    local = (tb[:, None] == tb[None, :]) & (idx[None, :] <= idx[:, None])
    oloc = torch.empty((T, H, D), dtype=storage, device=q_rope.device)
    lloc = torch.empty((H, T), dtype=cd, device=q_rope.device)
    for h in range(H):
        s = _masked((q_rope[:, h].to(cd) @ k_rope[:, h].to(cd).T) * scale, local)
        m = _rowmax(s)
        p = torch.exp(s - m[:, None])
        l_ = p.sum(-1)
        oloc[:, h] = ((_st(p, storage) @ v[:, h].to(cd)) / l_[:, None]).to(storage)
        lloc[h] = m + torch.log(l_)
    return oloc, lloc


# operator 级入口
def gate_from_storage(a_storage: torch.Tensor, tau: float, storage: torch.dtype):
    """与 fla/layers/phba.py 相同：g_log = logsigmoid(fp32(a)) / tau；s = storage(-expm1(g_log))（autograd）。"""
    cd = compute_dtype(storage)
    g_log = F.logsigmoid(a_storage.to(cd)) / tau
    return g_log, (-torch.expm1(g_log)).to(storage)


def reference_a_chunkwise(q_raw, q_rope, k_rope, k_raw, v, a_storage, *, tau: float, layout: HostLayout,
                          rtok: torch.Tensor, rstate: torch.Tensor, scale: float | None = None,
                          local_impl='torch', fa_segment_batch_max=65535, fa_rows_per_call_max=1 << 20,
                          fa_deterministic=False):
    """Reference A(chunkwise_cuda) 的 operator：RoPE 在外（q_rope / k_rope 为输入），返回 (O, L, PrefixK, PrefixV)。

    autograd 只负责把各手写 backward 的 Function 与 gate 的逐元素运算连起来；dV 由 attention 与 Prefix 两个 Function
    的返回值相加（storage dtype，与生产实现中 autograd 的行为一致）。
    """
    storage = v.dtype
    scale = 1.0 / math.sqrt(q_rope.shape[-1]) if scale is None else scale
    g_log, s_val = gate_from_storage(a_storage, tau, storage)
    pk, pv = RefChunkwisePrefixFn.apply(s_val, k_raw, v, g_log, layout)
    out, lse = RefJointAttentionFn.apply(q_rope, k_rope, v, q_raw, pk, pv, rtok, rstate, layout, scale,
                                         local_impl, fa_segment_batch_max, fa_rows_per_call_max, fa_deterministic)
    return out, lse, pk, pv


__all__ = [
    "HostLayout",
    "RefChunkwisePrefixFn",
    "RefJointAttentionFn",
    "alg6_reference",
    "compute_dtype",
    "gate_from_storage",
    "local_cumsum",
    "local_reference",
    "prefix_backward_branch",
    "prefix_forward",
    "reference_a_chunkwise",
    "route_masks",
    "rstate_from_rtok_host",
]
