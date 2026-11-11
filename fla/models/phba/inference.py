# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Explicit cached PHBA inference; training and the existing generation ABI are unchanged."""

from __future__ import annotations

import torch

from fla.ops.phba.decode import STAGES, PHBALayerCache


class PHBADecoder:
    """Equal-length, unpadded prompts and one-token incremental decoding.

    Construct from an eval-mode PHBAForCausalLM. Prefill/decode return only
    [B,1,vocab] logits. This does not enable model.generate/use_cache=True.
    No backward, packed input, beam reorder, offload, or CUDA graph support.
    """

    def __init__(self, model, max_length: int, *, stage: str = "p0"):
        if type(max_length) is not int or max_length < 1 or stage not in STAGES:
            raise ValueError("invalid decode stage or maximum length")
        if getattr(model.config, "model_type", None) != "phba":
            raise ValueError("PHBADecoder requires a PHBA model")
        if model.training:
            raise ValueError("call model.eval() before constructing a decoder")
        self.model, self.max_length, self.stage = model, max_length, stage
        self.caches = []
        self.failed, self.batch = False, None

    def clear(self):
        """Release request caches without changing model parameters or config."""
        self.caches.clear()
        self.failed, self.batch = False, None

    def _check_ids(self, ids, *, decoding):
        if self.model.training:
            raise ValueError("cached inference requires model.eval()")
        if self.failed:
            raise RuntimeError("previous inference failed; clear and prefill again")
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or min(ids.shape) < 1:
            raise ValueError("input_ids must be nonempty dense [B,T] integers")
        if ids.device != self.model.model.embeddings.weight.device:
            raise ValueError("input_ids must be on the model device")
        if decoding:
            if not self.caches or ids.shape != (self.batch, 1):
                raise ValueError("decode requires prefill and one token per row")
            lengths = {cache.length for cache in self.caches}
            if len(lengths) != 1 or next(iter(lengths)) >= self.max_length:
                raise ValueError("inconsistent cache lengths or exhausted capacity")
        elif self.caches or ids.shape[1] > self.max_length:
            raise ValueError("prefill needs a fresh decoder and must fit capacity")

    def _project(self, attn, hidden, offset):
        b, t, _ = hidden.shape
        h, d = attn.num_heads, attn.head_dim
        q = attn.q_norm(attn.q_proj(hidden).reshape(b, t, h, d))
        k = attn.k_norm(attn.k_proj(hidden).reshape(b, t, h, d))
        v = attn.v_proj(hidden).reshape(b, t, h, d)
        qr, kr = attn.rotary(q=q, k=k, seqlen_offset=offset,
                             max_seqlen=max(self.max_length, attn.max_position_embeddings or 0))
        gate = attn.gsa_gate_proj(hidden).reshape(b, t, h, attn.gsa_num_slots)
        return tuple(x.contiguous() for x in (q, k, v, qr, kr, gate))

    @staticmethod
    def _output_projection(attn, hidden, attention):
        b, t, h, d = attention.shape
        if attn.use_output_gate:
            gate = attn.g_proj(hidden).reshape(b, t, h, d)
            attention = attn.o_norm(attention, gate)
        else:
            attention = attn.o_norm(attention)
        return attn.o_proj(attention.reshape(b, t, h*d))

    def _finish_block(self, block, residual, attention):
        # preserve fused residual/norm arithmetic instead of introducing a rounded residual addition
        if self.model.config.fuse_norm:
            hidden, residual = block.mlp_norm(attention, residual, True)
        else:
            residual = residual + attention
            hidden = block.mlp_norm(residual)
        return residual + block.mlp(hidden)

    @staticmethod
    def _prefill_attention(cache, projected):
        from fla.ops.phba.flash_attention import ensure_ready
        from fla.ops.phba.parallel import BlockMajorConfig, block_major_attention
        from fla.ops.phba.plans import SequenceLengthPlan
        from fla.ops.phba.routing import build_routes

        q, k, v, qr, kr, gate = projected
        b, t, h, d = q.shape
        ensure_ready(q.device, q.dtype, d)
        cache.prefill(k, kr, v, gate)
        plan = SequenceLengthPlan.from_host((t,) * b, q.device)
        layout, rtok, rstate = build_routes(
            q_rope=qr.reshape(-1, h, d), k_rope=kr.reshape(-1, h, d), length_plan=plan,
            chunk_size=cache.chunk_size, topk=cache.topk,
        )
        n = (t + cache.chunk_size - 1) // cache.chunk_size
        pk = cache.pk[:, :n].reshape(b*n, cache.slots, h, d).contiguous()
        pv = cache.pv[:, :n].reshape(b*n, cache.slots, h, d).contiguous()
        out, _ = block_major_attention(
            qr=qr.reshape(-1, h, d), qw=q.reshape(-1, h, d), kr=kr.reshape(-1, h, d), v=v.reshape(-1, h, d),
            pk=pk, pv=pv, rtok=rtok, rstate=rstate, layout=layout,
            config=BlockMajorConfig(phba_ablation=cache.mode),
        )
        return out.reshape(b, t, h, d)

    @torch.inference_mode()
    def prefill(self, input_ids):
        """Build all caches and emit only the final prompt position's vocabulary logits."""
        self._check_ids(input_ids, decoding=False)
        self.batch = input_ids.shape[0]
        try:
            hidden = self.model.model.embeddings(input_ids)
            for block in self.model.model.layers:
                residual = hidden
                normalized = block.attn_norm(hidden)
                attn = block.attn
                projected = self._project(attn, normalized, 0)
                cache = PHBALayerCache(
                    self.batch, self.max_length, attn.num_heads, attn.head_dim, attn.gsa_num_slots,
                    attn.moba_chunk_size, attn.moba_topk, dtype=projected[0].dtype, device=projected[0].device,
                    tau=attn.gate_logit_normalizer, mode=attn.phba_ablation, stage=self.stage,
                )
                self.caches.append(cache)
                output = self._prefill_attention(cache, projected)
                attention = self._output_projection(attn, normalized, output)
                hidden = self._finish_block(block, residual, attention)
                del projected, normalized, output, attention, residual
            return self.model.lm_head(self.model.model.norm(hidden[:, -1:]))
        except Exception:
            self.failed = True
            raise

    @torch.inference_mode()
    def decode(self, input_ids):
        """Consume one token per row without recomputing historical projections."""
        self._check_ids(input_ids, decoding=True)
        try:
            hidden = self.model.model.embeddings(input_ids)
            for block, cache in zip(self.model.model.layers, self.caches):
                residual = hidden
                normalized = block.attn_norm(hidden)
                projected = self._project(block.attn, normalized, cache.length)
                output = cache.step(*projected)
                attention = self._output_projection(block.attn, normalized, output)
                hidden = self._finish_block(block, residual, attention)
            return self.model.lm_head(self.model.model.norm(hidden))
        except Exception:
            self.failed = True
            raise

    def memory_bytes(self):
        totals = {}
        for cache in self.caches:
            for name, size in cache.memory_bytes().items():
                totals[name] = totals.get(name, 0) + size
        return totals
