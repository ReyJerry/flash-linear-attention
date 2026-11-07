# PHBA operator

PHBA combines causal local token attention, selected historical token blocks, and gated prefix-state slots in one joint softmax. Routing is discrete and is treated as constant in backward.

For gate logits `a_t` and positive normalizer `tau`, define `g_t = log_sigmoid(a_t) / tau` and `s_t = -expm1(g_t)`. Prefix states follow the recurrence

```text
S_{t+1} = diag(exp(g_t)) S_t + s_t [k_t, v_t]^T.
```

A block's prefix is the state before its first token. Token routes select up to `topk - 1` strictly earlier blocks from the same sequence. State routes contain the current block and token-route blocks, excluding the first prefix of each sequence. Causal local tokens, historical tokens and valid prefix slots share the final normalization.

## Storage and gradients

The CUDA implementation stores activations, insertion weights and prefix matrices in FP16/BF16 and uses FP32 accumulation. Prefix processing preserves its chunk-local cumulative-sum and dot-product order. The block-major backward assigns producers and owners deterministically, so route scatter order does not change floating-point accumulation order.

Both reference families live in `naive.py`. `naive.naive_phba` is the differentiable FP64 recurrent formula with frozen routes. `naive.reference_a_chunkwise` follows the staged storage conversions and supplies the storage-aware GPU comparison. BF16/FP16 comparisons use the existing PHBA numerical envelopes; migration against the frozen implementation also checks exact tensor bytes. References do not approximate the sparse workload with dense FLOPs for performance claims.

The operator returns `(output, final_state)`. When requested, final state is a pair of FP32 key/value states with no backward contract. Training supports first-order gradients for all six floating inputs. The internal attention backward is once-differentiable.

## Layouts and bounds

Inputs use `[B, T, H, D]` and gate logits use `[B, T, H, M]`. Packed inputs have `B == 1` and cumulative int32 lengths. A `SequenceLengthPlan` owns immutable geometry tensors and can be shared between layers; it never caches query, key, value or gradient data.

Pointer arithmetic uses int64. Canonical sorting uses int32 keys only when the expert/query key range is safe, with int64 fallback otherwise. Invalid routes and metadata tails are preserved. Forward processing uses bounded query windows; deterministic backward uses the geometry plan saved by forward, with bounded grouped workspaces and a fixed-order fallback.

## Implementation layout

The public entry is `chunk.py`, with sequence geometry and workspace limits in `plans.py`. `parallel.py` owns joint attention forward and autograd dispatch; `prefix.py` owns the recurrent prefix computation. Internal responsibilities are grouped as follows:

| Module           | Responsibility                                                               |
| ---------------- | ---------------------------------------------------------------------------- |
| `routing.py`     | Block layout, centroids and discrete top-k routes                             |
| `metadata.py`    | Canonical token metadata and derived prefix-state metadata                    |
| `chunk_bwd.py`   | Attention backward dispatch, branch kernels and deterministic query ownership |
| `tape.py`        | Grouped deterministic producer/owner workspaces and kernels                   |
| `utils.py`       | Input contracts, structured errors and isolated preflight autograd            |
| `naive.py`       | FP64 and storage-aware references, including routing and metadata oracles     |

`flash_attention.py` adapts the local-attention interface. `linear.py` and `backends/` provide optional projection acceleration, with native fallbacks and packaged source resources. Profiling and algorithm-screening tools live under `benchmarks/`, outside the production operator. Internal module paths are implementation details; use the exports from `fla.ops.phba` for the public operator and sequence plan.
