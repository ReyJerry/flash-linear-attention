# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Explicit inference-only PHBA caches; no autograd or Transformers Cache ABI."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

STAGES = ("p0", "p1_state", "p1_route", "p1_split")
MODES = ("full", "current_state_only", "routed_states_only")


def block_update(state, raw, values, log_gate, write):
    """Chunk update with storage-rounded write operands and FP32/FP64 carry.

    State uses [B,H,M,D]; block inputs use [B,C,H,D/M]. Disable autocast
    when calling: accumulation precision is part of this reference's contract.
    """
    dtype = state[0].dtype
    cumulative = log_gate.to(dtype).cumsum(1)
    last = cumulative[:, -1]
    weights = (write.to(dtype) * (last[:, None] - cumulative).exp()).to(raw.dtype).to(dtype)
    decay = last.exp()[..., None]
    return tuple(s * decay + torch.einsum("bthm,bthd->bhmd", weights, z.to(dtype))
                 for s, z in zip(state, (raw, values)))


def block_centroid(keys):
    """FP32 mean, with the production centroid's 64-token summation grouping."""
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    total = torch.zeros_like(keys[:, 0], dtype=dtype)
    for start in range(0, keys.shape[1], 64):
        total += keys[:, start:start + 64].to(dtype).sum(1)
    return total / keys.shape[1]



def completed_centroids(keys, chunk_size):
    """Vectorized reference pooling for complete blocks, retaining 64-token grouping."""
    b, t, h, d = keys.shape
    n = t // chunk_size
    packed = keys[:, :n*chunk_size].reshape(b, n, chunk_size, h, d)
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    total = torch.zeros(b, n, h, d, device=keys.device, dtype=dtype)
    for start in range(0, chunk_size, 64):
        total += packed[:, :, start:start+64].to(dtype).sum(2)
    return total / chunk_size


class PHBALayerCache:
    """Dense equal-length cached prefill and single-token readout.

    Token KV: [B,capacity,H,D]. Prefix snapshots: [B,Nblock,M,H,D]. Only
    one block of raw keys/gates is retained. No packing, beam reorder, offload
    or concurrent requests sharing a cache. Caches are not model parameters.
    """

    def __init__(self, batch, capacity, heads, width, slots, chunk_size=128, topk=8,
                 *, dtype=torch.bfloat16, device="cuda", tau=8.0, mode="full", stage="p0"):
        dimensions = (batch, capacity, heads, width, slots, chunk_size, topk)
        if any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("cache dimensions must be positive integers")
        if topk > 16 or slots > 128 or width > 128:
            raise ValueError("PHBA requires topk<=16 and slots/head width<=128")
        if stage not in STAGES or mode not in MODES:
            raise ValueError("invalid PHBA decode stage or ablation mode")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("decode supports CPU references and NVIDIA CUDA only")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("unsupported cache storage dtype")
        if self.device.type == "cuda" and (dtype not in (torch.float16, torch.bfloat16)
                                          or width < 16 or width % 8
                                          or chunk_size not in (32, 64, 128, 256, 512)):
            raise ValueError("CUDA PHBA needs fp16/bf16, aligned head widths and supported chunks")
        self.batch, self.capacity, self.heads, self.width = batch, capacity, heads, width
        self.slots, self.chunk_size, self.topk = slots, chunk_size, topk
        self.dtype, self.tau, self.mode, self.stage = dtype, tau, mode, stage
        self.acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.nblocks = (capacity + chunk_size - 1) // chunk_size + 1
        self.length, self.ready = 0, False
        kw = dict(device=self.device, dtype=dtype)
        self.keys = torch.empty(batch, capacity, heads, width, **kw)
        self.values = torch.empty_like(self.keys)
        self.pk = torch.empty(batch, self.nblocks, slots, heads, width, **kw)
        self.pv = torch.empty_like(self.pk)
        self.pk[:, 0].zero_()
        self.pv[:, 0].zero_()
        self.raw_tail = torch.empty(batch, chunk_size, heads, width, **kw)
        self.log_tail = torch.empty(batch, chunk_size, heads, slots, device=self.device, dtype=self.acc_dtype)
        self.write_tail = torch.empty(batch, chunk_size, heads, slots, **kw)
        self.state_k = torch.zeros(batch, heads, slots, width, device=self.device, dtype=self.acc_dtype)
        self.state_v = torch.zeros_like(self.state_k)
        self.live_k = torch.zeros_like(self.state_k) if stage == "p0" else None
        self.live_v = torch.zeros_like(self.state_v) if stage == "p0" else None
        self.centroids = torch.empty(batch, self.nblocks, heads, width, device=self.device, dtype=self.acc_dtype)
        self.cached_routing = stage in ("p1_route", "p1_split")
        self.route_buffer = torch.empty(batch, heads, topk-1, device=self.device, dtype=torch.int32)
        self.route_scores = torch.empty(batch, heads, self.nblocks, device=self.device, dtype=self.acc_dtype)
        self.partials = (torch.empty(batch, heads, topk, width+2, device=self.device, dtype=self.acc_dtype)
                         if stage == "p1_split" else None)

    def _validate(self, *xs, gate, length=None):
        t = xs[0].shape[1] if xs[0].ndim == 4 else -1
        shape = (self.batch, t, self.heads, self.width)
        if t < 1 or (length is not None and t != length) or any(tuple(x.shape) != shape for x in xs):
            raise ValueError("projected tensors must have matching [B,T,H,D] shapes")
        if tuple(gate.shape) != (self.batch, t, self.heads, self.slots):
            raise ValueError("gate must have shape [B,T,H,M]")
        if any(x.device != self.keys.device or x.dtype != self.dtype for x in (*xs, gate)):
            raise ValueError("inputs must match cache device and storage dtype")
        return t

    def _gates(self, gate):
        log_gate = F.logsigmoid(gate.to(self.acc_dtype)) / self.tau
        return log_gate, (-torch.expm1(log_gate)).to(self.dtype)

    def _save_boundary(self, block):
        self.pk[:, block].copy_(self.state_k.permute(0, 2, 1, 3))
        self.pv[:, block].copy_(self.state_v.permute(0, 2, 1, 3))

    def _refresh_live(self, block, length):
        """Eager reference: refresh the current state with boundary-compatible rounding."""
        start = block * self.chunk_size
        live = block_update((self.state_k, self.state_v), self.raw_tail[:, :length],
                            self.values[:, start:start+length], self.log_tail[:, :length],
                            self.write_tail[:, :length])
        self.live_k.copy_(live[0])
        self.live_v.copy_(live[1])

    def _finish_block(self, block):
        c = self.chunk_size
        if self.stage == "p0":
            self.state_k.copy_(self.live_k)
            self.state_v.copy_(self.live_v)
            self._save_boundary(block + 1)
        elif self.device.type == "cuda":
            from fla.ops.phba.decode_triton import finish_block

            finish_block(self, block)
        else:
            state = block_update((self.state_k, self.state_v), self.raw_tail,
                                 self.values[:, block*c:(block+1)*c], self.log_tail, self.write_tail)
            self.state_k.copy_(state[0])
            self.state_v.copy_(state[1])
            self._save_boundary(block + 1)
        if self.cached_routing and self.device.type == "cuda":
            from fla.ops.phba.decode_triton import cache_centroids

            cache_centroids(self, block, 1)
        else:
            self.centroids[:, block].copy_(block_centroid(self.keys[:, block*c:(block+1)*c]))

    @torch.inference_mode()
    def prefill(self, raw_keys, rope_keys, values, gate):
        """Initialize once, retaining an FP32 anchor before the incomplete tail.

        A partial prompt tail is buffered, not included in the anchor and then
        counted a second time when decoding completes that block.
        """
        if self.ready:
            raise ValueError("prefill requires a fresh cache")
        t = self._validate(raw_keys, rope_keys, values, gate=gate)
        if t > self.capacity:
            raise ValueError("prompt exceeds cache capacity")
        with torch.autocast(device_type=self.device.type, enabled=False):
            c, h, m, d = self.chunk_size, self.heads, self.slots, self.width
            complete, tail = divmod(t, c)
            end = complete * c
            self.keys[:, :t].copy_(rope_keys)
            self.values[:, :t].copy_(values)
            g, s = self._gates(gate)
            if complete and self.device.type == "cuda":
                from fla.ops.phba.plans import SequenceLengthPlan
                from fla.ops.phba.prefix import chunkwise_prefix_states

                plan = SequenceLengthPlan.from_host((end,) * self.batch, self.device)
                pk, pv, hk, hv = chunkwise_prefix_states(
                    s_val=s[:, :end].reshape(-1, h, m).contiguous(),
                    k_raw=raw_keys[:, :end].reshape(-1, h, d).contiguous(),
                    v=values[:, :end].reshape(-1, h, d).contiguous(),
                    g_log=g[:, :end].reshape(-1, h, m).contiguous(),
                    layout=plan.block_layout(c), output_final_state=True,
                )
                self.pk[:, :complete].copy_(pk.reshape(self.batch, complete, m, h, d))
                self.pv[:, :complete].copy_(pv.reshape(self.batch, complete, m, h, d))
                self.state_k.copy_(hk)
                self.state_v.copy_(hv)
                self._save_boundary(complete)
            else:
                for block in range(complete):
                    sl = slice(block*c, (block+1)*c)
                    state = block_update((self.state_k, self.state_v), raw_keys[:, sl], values[:, sl], g[:, sl], s[:, sl])
                    self.state_k.copy_(state[0])
                    self.state_v.copy_(state[1])
                    self._save_boundary(block + 1)
            if self.cached_routing and self.device.type == "cuda":
                from fla.ops.phba.decode_triton import cache_centroids

                cache_centroids(self, 0, complete)
            elif complete:
                self.centroids[:, :complete].copy_(completed_centroids(self.keys[:, :end], c))
            if tail:
                self.raw_tail[:, :tail].copy_(raw_keys[:, end:])
                self.log_tail[:, :tail].copy_(g[:, end:])
                self.write_tail[:, :tail].copy_(s[:, end:])
            if self.stage == "p0":
                self.live_k.copy_(self.state_k)
                self.live_v.copy_(self.state_v)
                if tail:
                    self._refresh_live(complete, tail)
            self.length, self.ready = t, True

    def route(self, q_rope, position):
        """Single-query top-K with cached or recomputed centroids; lower IDs break ties."""
        count, kh = position // self.chunk_size, self.topk - 1
        if self.cached_routing and self.device.type == "cuda":
            from fla.ops.phba.decode_triton import route_one

            return route_one(self, q_rope, count)
        result = torch.full((self.batch, self.heads, kh), -1, dtype=torch.int32, device=self.device)
        if not kh or not count:
            return result
        centers = (self.centroids[:, :count] if self.cached_routing else
                   completed_centroids(self.keys[:, :count*self.chunk_size], self.chunk_size))
        centers = centers.to(self.dtype).to(self.acc_dtype)
        scores = torch.einsum("bhd,bnhd->bhn", q_rope.to(self.acc_dtype), centers)
        selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :min(kh, count)]
        result[..., :selected.shape[-1]].copy_(selected)
        return result

    def read(self, q_raw, q_rope, position, routes):
        """Torch selected-memory joint readout; no all-prefix attention matrix."""
        if self.stage == "p1_split":
            if self.device.type == "cuda":
                from fla.ops.phba.decode_triton import split_read

                return split_read(self, q_raw, q_rope, position, routes)
            return split_read_reference(self, q_raw, q_rope, position, routes)
        b, h, d, c, m = self.batch, self.heads, self.width, self.chunk_size, self.slots
        current = position // c
        ids = torch.cat((torch.full((b, h, 1), current, device=self.device, dtype=torch.int32), routes), dim=-1)
        row = torch.arange(b, device=self.device)[:, None, None, None]
        head = torch.arange(h, device=self.device)[None, :, None, None]
        lane = torch.arange(c, device=self.device)[None, None, None, :]
        token_ids = ids[..., None].long().clamp_min(0) * c + lane
        token_valid = (ids[..., None] >= 0) & (token_ids <= position)
        if self.mode == "routed_states_only":
            token_valid[:, :, 1:] = False
        safe_token = token_ids.clamp_max(self.capacity - 1)
        keys = self.keys[row, safe_token, head].to(self.acc_dtype)
        vals = self.values[row, safe_token, head].to(self.acc_dtype)
        token_scores = (keys * q_rope.to(self.acc_dtype)[:, :, None, None]).sum(-1) * d**-0.5
        token_scores = token_scores.masked_fill(~token_valid, -torch.inf).flatten(2)
        slots = torch.arange(m, device=self.device)[None, None, None, :]
        skeys = self.pk[row, ids[..., None].long().clamp_min(0), slots, head].to(self.acc_dtype)
        svals = self.pv[row, ids[..., None].long().clamp_min(0), slots, head].to(self.acc_dtype)
        state_valid = (ids[..., None] > 0).expand(b, h, self.topk, m).clone()
        if self.mode == "current_state_only":
            state_valid[:, :, 1:] = False
        state_scores = (skeys * q_raw.to(self.acc_dtype)[:, :, None, None]).sum(-1) * d**-0.5
        state_scores = state_scores.masked_fill(~state_valid, -torch.inf).flatten(2)
        scores = torch.cat((token_scores, state_scores), dim=-1)
        all_values = torch.cat((vals.flatten(2, 3), svals.flatten(2, 3)), dim=2)
        valid = torch.cat((token_valid.flatten(2), state_valid.flatten(2)), dim=-1)
        # unused capacity can contain NaNs; mask values before multiplying by zero probabilities
        all_values = all_values.masked_fill(~valid[..., None], 0)
        prob = scores.softmax(-1).to(self.dtype).to(self.acc_dtype)
        return (prob[..., None] * all_values).sum(2).to(self.dtype)

    @torch.inference_mode()
    def step(self, q_raw, raw_keys, values, q_rope, rope_keys, gate):
        """Consume one new token per row; no backward and no full-prefix recomputation."""
        if not self.ready:
            raise ValueError("call prefill before decode")
        self._validate(q_raw, raw_keys, values, q_rope, rope_keys, gate=gate, length=1)
        if self.length >= self.capacity:
            raise ValueError("decode exceeds preallocated cache capacity")
        with torch.autocast(device_type=self.device.type, enabled=False):
            pos = self.length
            block, offset = divmod(pos, self.chunk_size)
            self.keys[:, pos].copy_(rope_keys[:, 0])
            self.values[:, pos].copy_(values[:, 0])
            self.raw_tail[:, offset].copy_(raw_keys[:, 0])
            log_gate, write = self._gates(gate[:, 0])
            self.log_tail[:, offset].copy_(log_gate)
            self.write_tail[:, offset].copy_(write)
            if self.stage == "p0":
                self._refresh_live(block, offset + 1)
            routes = self.route(q_rope[:, 0], pos)
            out = self.read(q_raw[:, 0], q_rope[:, 0], pos, routes)
            self.length += 1
            if offset + 1 == self.chunk_size:
                self._finish_block(block)
            return out[:, None]

    def memory_bytes(self):
        """Persistent allocated storage including slack, counting shared storage only once."""
        groups = {"token_kv": (self.keys, self.values), "prefix_snapshots": (self.pk, self.pv),
                  "fp32_state": (self.state_k, self.state_v, self.live_k, self.live_v),
                  "tail": (self.raw_tail, self.log_tail, self.write_tail), "centroids": (self.centroids,),
                  "routing_workspace": (self.route_buffer, self.route_scores), "split_workspace": (self.partials,)}
        seen, sizes = set(), {}
        for name, tensors in groups.items():
            sizes[name] = 0
            for tensor in tensors:
                if tensor is None:
                    continue
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    sizes[name] += storage.nbytes()
                    seen.add(key)
        return sizes



def split_read_reference(cache, q_raw, q_rope, position, routes):
    """CPU reference for per-route statistics and exact joint-softmax merging.

    Low-precision probability rounding occurs per partial; this is mathematically
    equivalent, not bitwise identical to the dense normalized-probability reference.
    """
    b, h, d = cache.batch, cache.heads, cache.width
    output = torch.empty(b, h, d, dtype=cache.dtype, device=cache.device)
    current, c = position // cache.chunk_size, cache.chunk_size
    for batch in range(b):
        for head in range(h):
            parts = []
            for index, block in enumerate([current] + routes[batch, head].tolist()):
                if block < 0:
                    continue
                scores, values = [], []
                if index == 0 or cache.mode != "routed_states_only":
                    end = min((block+1)*c, position+1)
                    keys = cache.keys[batch, block*c:end, head].to(cache.acc_dtype)
                    vals = cache.values[batch, block*c:end, head].to(cache.acc_dtype)
                    scores.append((keys * q_rope[batch, head].to(cache.acc_dtype)).sum(-1) * d**-.5)
                    values.append(vals)
                if block > 0 and (index == 0 or cache.mode != "current_state_only"):
                    keys = cache.pk[batch, block, :, head].to(cache.acc_dtype)
                    vals = cache.pv[batch, block, :, head].to(cache.acc_dtype)
                    scores.append((keys * q_raw[batch, head].to(cache.acc_dtype)).sum(-1) * d**-.5)
                    values.append(vals)
                if not scores:
                    continue
                score, vals = torch.cat(scores), torch.cat(values)
                maximum = score.max()
                prob = (score - maximum).exp()
                numerator = (prob.to(cache.dtype).to(cache.acc_dtype)[:, None] * vals).sum(0)
                parts.append((maximum, prob.sum(), numerator))
            maximum = torch.stack([p[0] for p in parts]).max()
            weights = [(p[0]-maximum).exp() for p in parts]
            output[batch, head] = (sum(w*p[2] for w, p in zip(weights, parts)) /
                                   sum(w*p[1] for w, p in zip(weights, parts)))
    return output
