# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'FlashAttention local branches using the joint PHBA outputs and log normalizers.'

from __future__ import annotations

import inspect
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fla.ops.phba.contracts import isolated_autograd
from fla.ops.phba.errors import fail
from fla.ops.phba.plans import FA_SEGMENT_BATCH_LIMIT, _local_fa_batches, fa_head_dim_round

if TYPE_CHECKING:
    from fla.ops.phba.routing.layout import RoutingLayout

FA_MIN_COMPUTE_CAPABILITY = (8, 0)

# 锁定的 flash-attn 2.8.3.post1 低层 op 的参数名集合（flash_attn/flash_attn_interface.py）。
# forward 没有 num_splits，也没有 deterministic。
FORWARD_PARAMS = frozenset({
    "q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k",
    "dropout_p", "softmax_scale", "causal", "window_size_left", "window_size_right", "softcap",
    "alibi_slopes", "return_softmax", "block_table", "leftpad_k", "seqused_k", "zero_tensors",
})
BACKWARD_PARAMS = frozenset({
    "dout", "q", "k", "v", "out", "softmax_lse", "dq", "dk", "dv", "cu_seqlens_q", "cu_seqlens_k",
    "max_seqlen_q", "max_seqlen_k", "dropout_p", "softmax_scale", "causal", "window_size_left",
    "window_size_right", "softcap", "alibi_slopes", "deterministic", "rng_state", "zero_tensors",
})


@dataclass(frozen=True)
class FAOps:
    forward: Callable[..., Any]
    backward: Callable[..., Any]
    forward_signature_source: Callable[..., Any]
    backward_signature_source: Callable[..., Any]


def _load_ops() -> FAOps:
    try:
        from flash_attn import flash_attn_interface as fai
    except Exception as exc:  # noqa: BLE001
        raise fail("fa_layout_probe_failed", f"无法导入 flash_attn.flash_attn_interface: {type(exc).__name__}: {exc}")
    # 2.8.3 在 torch >= 2.4 时把低层函数注册为 torch.ops.flash_attn.*；签名以 Python 定义为准（含参数名）
    fwd_src = getattr(fai, "_flash_attn_varlen_forward", None)
    bwd_src = getattr(fai, "_flash_attn_varlen_backward", None)
    if fwd_src is None or bwd_src is None:
        raise fail("fa_layout_probe_failed",
                   "flash_attn_interface 中没有 _flash_attn_varlen_forward / _flash_attn_varlen_backward")
    fwd = getattr(fai, "_wrapped_flash_attn_varlen_forward", fwd_src)
    bwd = getattr(fai, "_wrapped_flash_attn_varlen_backward", bwd_src)
    return FAOps(forward=fwd, backward=bwd, forward_signature_source=fwd_src, backward_signature_source=bwd_src)


def _param_names(fn: Callable[..., Any]) -> frozenset:
    """参数名集合。flash-attn 2.8.3 在 torch >= 2.4 时用 torch.library.custom_op 包装低层函数
    （CustomOpDef，签名是 *args / **kwargs），所以依次尝试：注册的 op schema、原始 Python 函数、普通签名。"""
    schema = getattr(fn, "_schema", None)
    if schema is not None and hasattr(schema, "arguments"):
        return frozenset(a.name for a in schema.arguments)
    default = getattr(fn, "default", None)                       # torch.ops.<ns>.<op> 的 OpOverloadPacket
    schema = getattr(default, "_schema", None)
    if schema is not None and hasattr(schema, "arguments"):
        return frozenset(a.name for a in schema.arguments)
    target = getattr(fn, "_init_fn", None) or inspect.unwrap(fn)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise fail("fa_layout_probe_failed", f"无法读取 {fn!r} 的签名: {exc}")
    kinds = {p.kind for p in sig.parameters.values()}
    if inspect.Parameter.VAR_POSITIONAL in kinds or inspect.Parameter.VAR_KEYWORD in kinds:
        raise fail("fa_layout_probe_failed", f"{fn!r} 的签名含 *args / **kwargs，无法逐参数核对")
    return frozenset(sig.parameters)


def check_schema(forward_fn: Callable[..., Any] | None = None, backward_fn: Callable[..., Any] | None = None) -> None:
    """参数名集合必须与锁定版本完全一致；不一致时报 fa_layout_probe_failed，并列出多出与缺少的参数。"""
    if forward_fn is None or backward_fn is None:
        ops = _load_ops()
        forward_fn = forward_fn or ops.forward_signature_source
        backward_fn = backward_fn or ops.backward_signature_source
    for name, fn, expected in (("forward", forward_fn, FORWARD_PARAMS), ("backward", backward_fn, BACKWARD_PARAMS)):
        actual = _param_names(fn)
        if actual != expected:
            raise fail("fa_layout_probe_failed",
                       f"FA varlen {name} 的参数名集合与锁定版本不一致：多出 {sorted(actual - expected)}，"
                       f"缺少 {sorted(expected - actual)}", op=name)


def check_device(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        raise fail("fa_device_unsupported", f"FlashAttention 需要 CUDA 设备，收到 {device}")
    cap = torch.cuda.get_device_capability(device)
    if cap < FA_MIN_COMPUTE_CAPABILITY:
        raise fail("fa_device_unsupported", f"FA2 要求 sm80 及以上，当前设备 compute capability = {cap}")
    return cap


def _ops() -> FAOps:
    global _OPS
    if _OPS is None:
        _OPS = _load_ops()
    return _OPS


_OPS: FAOps | None = None


def varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (out [total_q, H, Dh] storage dtype, lse [H, total_q] fp32)。所有参数显式传入。"""
    out, lse, _s_dmask, _rng = _ops().forward(
        q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q), max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0, softmax_scale=float(softmax_scale), causal=bool(causal),
        window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None,
        return_softmax=False, block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False,
    )
    return out, lse


def varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> torch.Tensor:
    """out / softmax_lse 必须是 PHBA 的 joint O / L（不是分支自己的 O / LSE）。softmax_lse 必须是连续的 [H, total_q]。"""
    if not softmax_lse.is_contiguous():
        raise fail("fa_layout_probe_failed", "softmax_lse 必须是连续的 [H, total_q]（切片要先 .contiguous()）")
    return _ops().backward(
        dout=dout, q=q, k=k, v=v, out=out, softmax_lse=softmax_lse, dq=dq, dk=dk, dv=dv,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q), max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0, softmax_scale=float(softmax_scale), causal=bool(causal),
        window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None,
        deterministic=bool(deterministic), rng_state=None, zero_tensors=False,
    )


@dataclass(frozen=True)
class ProbeResult:
    device: str
    compute_capability: tuple[int, int]
    dtype: str
    head_dim: int
    lse_shape: tuple[int, ...]
    lse_stride: tuple[int, ...]
    softmax_d_shape: tuple[int, ...]
    dh_round: int
    max_abs_err_out: float
    max_abs_err_dq: float


def _reference_attention(q, k, v, cu, scale, causal):
    """fp32 reference，返回 out 与 lse[H, T]（逐 sequence，小 shape）。"""
    T, H, D = q.shape
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(H, T, dtype=torch.float32, device=q.device)
    cu = cu.tolist()
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        qs, ks, vs = (x[a:b].float().transpose(0, 1) for x in (q, k, v))   # [H, L, D]
        scores = qs @ ks.transpose(1, 2) * scale
        if causal:
            L = b - a
            mask = torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        lse[:, a:b] = torch.logsumexp(scores, dim=-1)
        out[a:b] = (torch.softmax(scores, dim=-1) @ vs).transpose(0, 1)
    return out, lse


def runtime_smoke_probe(device: torch.device, dtype: torch.dtype, head_dim: int, *, num_heads: int = 2) -> ProbeResult:
    """小 shape 的 forward + backward。检查：
    - forward：out 与输入同 dtype / shape；lse 为 fp32、shape == [H, total_q]、连续；数值与 fp32 reference 接近；
    - backward：接受调用方给出的 joint 形状 out / lse（这里用 forward 自己的 out / lse，布局与 joint 相同），
      写入调用方分配的 dq / dk / dv；softmax_d 的 shape 为 [H, total_q + 128 * batch]（workspace 公式的依据）；
      dq 与 autograd 对 fp32 reference 的结果接近。
    """
    cap = check_device(device)
    scale = 1.0 / math.sqrt(head_dim)
    lengths = (5, 17)                                   # 两条 sequence，一条不满 16，覆盖 varlen 与尾部
    cu = torch.tensor((0, 5, 22), dtype=torch.int32, device=device)
    T, H = sum(lengths), num_heads
    gen = torch.Generator(device=device).manual_seed(1234)
    q, k, v, do = (torch.randn(T, H, head_dim, device=device, dtype=dtype, generator=gen) for _ in range(4))

    def bad(msg: str) -> Exception:
        return fail("fa_layout_probe_failed", f"{msg}（device={device}, dtype={dtype}, Dh={head_dim}）")

    try:
        out, lse = varlen_forward(q, k, v, cu, cu, max(lengths), max(lengths), softmax_scale=scale, causal=True)
    except Exception as exc:  # noqa: BLE001
        raise bad(f"forward 调用失败: {type(exc).__name__}: {exc}")
    if out.shape != q.shape or out.dtype != dtype:
        raise bad(f"forward out 的 shape / dtype = {tuple(out.shape)} / {out.dtype}")
    if lse.dtype != torch.float32 or tuple(lse.shape) != (H, T) or not lse.is_contiguous():
        raise bad(f"softmax_lse 应为连续的 fp32 [H, total_q] = [{H}, {T}]，实际 {lse.dtype} {tuple(lse.shape)} "
                  f"stride {tuple(lse.stride())}")

    qr, kr, vr = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    ref_out, ref_lse = _reference_attention(qr, kr, vr, cu, scale, causal=True)
    tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    err_out = float((out.float() - ref_out.detach()).abs().max())
    err_lse = float((lse - ref_lse.detach()).abs().max())
    if err_out > tol or err_lse > 1e-2:
        raise bad(f"forward 数值与 fp32 reference 不符：max|Δout| = {err_out:.3e}, max|Δlse| = {err_lse:.3e}")

    # joint 形状的 out / lse：由调用方新分配，避免与 forward 的返回 tensor 共享存储
    joint_out = out.clone()
    joint_lse = lse.clone()
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    try:
        softmax_d = varlen_backward(do, q, k, v, joint_out, joint_lse, dq, dk, dv, cu, cu, max(lengths), max(lengths),
                                    softmax_scale=scale, causal=True, deterministic=True)
    except Exception as exc:  # noqa: BLE001
        raise bad(f"backward 不接受 joint 形状的 out / softmax_lse: {type(exc).__name__}: {exc}")
    batch = len(lengths)
    if not isinstance(softmax_d, torch.Tensor) or tuple(softmax_d.shape) != (H, T + 128 * batch):
        got = tuple(softmax_d.shape) if isinstance(softmax_d, torch.Tensor) else type(softmax_d).__name__
        raise bad(f"softmax_d 应为 [H, total_q + 128 * batch] = [{H}, {T + 128 * batch}]，实际 {got}；"
                  "FlashAttention backward workspace layout differs from the supported ABI")
    ref_out.backward(do.float())
    err_dq = float((dq.float() - qr.grad).abs().max())
    scale_ref = float(qr.grad.abs().max()) + 1e-6
    if err_dq / scale_ref > 5e-2:
        raise bad(f"backward 的 dq 与 fp32 reference 不符：max|Δdq| = {err_dq:.3e}（参考最大值 {scale_ref:.3e}）")
    return ProbeResult(
        device=str(device), compute_capability=tuple(cap), dtype=str(dtype).replace("torch.", ""), head_dim=head_dim,
        lse_shape=tuple(lse.shape), lse_stride=tuple(lse.stride()), softmax_d_shape=tuple(softmax_d.shape),
        dh_round=fa_head_dim_round(head_dim), max_abs_err_out=err_out, max_abs_err_dq=err_dq,
    )


_READY: dict[tuple[str, str, int], ProbeResult] = {}
_READY_LOCK = threading.Lock()
_SCHEMA_OK = False


def ensure_ready(device: torch.device, dtype: torch.dtype, head_dim: int) -> ProbeResult:
    """初始化检查（每个进程、每个 (device, dtype, Dh) 只做一次）：设备架构、参数表、runtime smoke probe。
    只缓存成功结果；失败每次都重新抛出（A 类，不 fallback）。"""
    global _SCHEMA_OK
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (str(device), str(dtype), int(head_dim))
    cached = _READY.get(key)
    if cached is not None:
        return cached
    with _READY_LOCK:
        cached = _READY.get(key)
        if cached is not None:
            return cached
        check_device(device)
        if not _SCHEMA_OK:
            check_schema()
            _SCHEMA_OK = True
        # probe 内部对 fp32 reference 做一次 autograd backward；isolated_autograd 使它在 no_grad 下也能运行，
        # 并且不会被外层 activation checkpoint 的 saved-tensor hook 记录
        with isolated_autograd(), torch.cuda.device(device):
            result = runtime_smoke_probe(device, dtype, head_dim)
        _READY[key] = result
        return result


def local_backward(dout, q, k, v, joint_out, joint_lse, block_ptr, batches, *, deterministic=False, softmax_scale=None):
    """Use the same block batches as forward, caller-provided joint O/L, storage gradients.

    C++ mha_varlen_bwd requires dq/dk/dv dtype == q.dtype and stride(-1)==1.
    L[:, t0:t1] must be copied: FA indexes L as contiguous [H, batch_total_q].
    """
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    grads = tuple(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))
    for batch in batches:
        t0, t1 = batch.t0, batch.t1
        cu = block_ptr[batch.b0:batch.b1 + 1] - t0
        varlen_backward(dout[t0:t1], q[t0:t1], k[t0:t1], v[t0:t1], joint_out[t0:t1],
                        joint_lse[:, t0:t1].contiguous(), *(g[t0:t1] for g in grads), cu, cu,
                        batch.max_seqlen, batch.max_seqlen, softmax_scale=scale,
                        causal=True, deterministic=deterministic)
    return grads


__all__ = [
    "BACKWARD_PARAMS", "FORWARD_PARAMS", "ProbeResult",
    "check_device", "check_schema", "ensure_ready", "runtime_smoke_probe",
    "varlen_backward", "varlen_forward",
]


def local_attention(q_rope, k_rope, v, layout: RoutingLayout, *, scale: float,
                    fa_segment_batch_max: int = FA_SEGMENT_BATCH_LIMIT, fa_rows_per_call_max: int = 1 << 20):
    """local 分支：当前 block 内的因果 attention，FA 低层 varlen forward；返回 (Oloc storage [T,H,Dh], Lloc fp32 [H,T])。"""
    from fla.ops.phba.flash_attention import varlen_forward

    c = layout.chunk_size
    batches = _local_fa_batches(layout.block_ptr_host, c, fa_segment_batch_max, fa_rows_per_call_max)
    if len(batches) == 1:
        return varlen_forward(q_rope, k_rope, v, layout.block_ptr, layout.block_ptr, c, c, softmax_scale=scale,
                              causal=True)
    T, H, D = q_rope.shape
    oloc = torch.empty_like(q_rope)
    lloc = torch.empty((H, T), dtype=torch.float32, device=q_rope.device)
    for b in batches:
        cu_b = layout.block_ptr[b.b0:b.b1 + 1] - b.t0                    # device 运算，rebase 到 0
        o_b, l_b = varlen_forward(q_rope[b.t0:b.t1], k_rope[b.t0:b.t1], v[b.t0:b.t1], cu_b, cu_b,
                                  b.max_seqlen, b.max_seqlen, softmax_scale=scale, causal=True)
        oloc[b.t0:b.t1] = o_b
        lloc[:, b.t0:b.t1] = l_b
    return oloc, lloc
