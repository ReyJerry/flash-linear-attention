# PHBA cached decoding: P0/P1 implementation and acceptance

This adds an explicit inference-only API without changing existing PHBA training, ablation kernels, checkpoint parameter names, or the Hugging Face `use_cache=False` interface. It does not implement `model.generate()`, Transformers Cache, beam search, ragged/packed decoding, multi-device sharding, CUDA graphs, or quantization. Dense equal-length batches are supported. The model adapter is intended for CUDA FP16/BF16; operator references run on CPU.

## Stages

| Stage | State maintenance | Routing | Readout |
| --- | --- | --- | --- |
| `p0` | Eager current-block refresh | Re-pool historical key blocks; one-query sort | Torch selected-memory joint softmax |
| `p1_state` | Update only at block completion; Triton boundary kernel | Same as P0 | Same as P0 |
| `p1_route` | Same as above | Cached centroids; single-query Triton scores/top-K | Same as P0 |
| `p1_split` | Same as above | Same as above | Direct per-route Triton streaming + joint-softmax merge |

P0 is a correctness/reference stage, NOT a previously optimized production decode baseline. Its eager state refresh reconstructs the current partial block from the FP32 boundary anchor so that snapshots have the same storage-rounding definition as chunkwise training. It does not use an artificial discarded per-token state. The partial state is used to publish the next snapshot at the boundary. Report stage improvements as within-implementation comparisons, not as a speedup over an optimized MoBA or conventional recurrent kernel.

All stages retain the same block partition, gate formula, exact top-K budget and prefix definition. `topk` includes the current block. No independent state router is introduced. All three `phba_ablation` readout modes are supported, but the paper's main inference comparison must use `full`.

## Cache semantics and numerical contract

Cache K/V contains RoPE keys and values, not all historical raw keys. Snapshots use activation storage precision; the boundary anchor uses FP32. Only the current incomplete block's raw K and gates are retained; V is reused from the KV cache. Routing centroids use FP32, then are rounded to the query storage dtype for routing, as in the existing router.

For a block, let g be log retention, G_i its cumulative sum, and s the storage-rounded insertion weight. The boundary update is

`S_next = exp(G_last) * S_before + [s_i * exp(G_last-G_i)]_storage^T Z`.

The update is mathematically equivalent to token recurrence. Crucially, its storage-rounded coefficients differ from an arbitrarily reassociated low-precision token recurrence. P0 recomputes a partial block from its anchor; P1 only evaluates the complete block when the next prefix is needed. An incomplete prefill tail is NOT already incorporated into the anchor; otherwise it would be counted twice.

The final token of a block reads the old current prefix and is included in local causal attention. Only after readout does the implementation publish the next boundary snapshot. At a block boundary both FP32 carry and the storage snapshot are updated. Centroids are finalized once per completed block.

Split readout produces `(maximum, exp_sum, unnormalized_output)` per selected block and merges using log-sum-exp rescaling. Tokens use the rotary query; states use the raw query. Empty state blocks and invalid routes contribute zero probability mass. This is one joint softmax, not an average of separately normalized branches.

Mathematical equivalence does NOT imply bitwise identity across Torch and Triton reductions. Near-tied routing scores need particular attention: accumulation order can change a discrete top-K decision. GPU tests and real-checkpoint short-reference checks are mandatory. CPU semantic tests execute kernel source bodies with PyTorch-backed primitives, not a Triton compiler or CUDA device.

## Explicit model API

```python
import torch
from fla.models.phba import PHBAForCausalLM
from fla.models.phba.inference import PHBADecoder

model = PHBAForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).cuda().eval()
decoder = PHBADecoder(model, max_length=prompt_ids.shape[1] + 256, stage="p1_split")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = decoder.prefill(prompt_ids)  # [B,1,vocab], not [B,T,vocab]
    next_ids = logits.argmax(-1)
    logits = decoder.decode(next_ids)    # consume one new token per row
    next_ids = logits.argmax(-1)
decoder.clear()
```

Set the ablation mode before constructing the model. Existing `PHBA.forward(use_cache=True)` still rejects that ABI; use the explicit adapter instead. This intentionally leaves ongoing training untouched. The adapter preserves fused residual/norm handling and supports the existing output gate and q/k normalization. Cache buffers are external to `state_dict`. Failed model requests require `clear()` before reuse.

The optimized path uses no full-prefix recomputation during decode and no training-style edge-count/scatter metadata. It does still use Python dispatch, small per-step allocations, and host sequence positions; it is not CUDA-graph-ready. Prefill reuses existing chunkwise state and sparse-attention kernels. Temporary copies can occur while placing prefix snapshots into preallocated capacity, especially for B>1; this is not a minimum-prefill-peak-memory claim.

## Acceptance

Run standalone CPU tests from the repository root:

```bash
python tests/ops/test_phba_decode_cpu.py
python tests/ops/test_phba_decode_kernels_cpu.py
```

Then run actual CUDA acceptance and existing regressions in the repository's supported environment, including its FlashAttention dependency:

```bash
FLA_CI_ENV=0 python benchmarks/phba/run_decode_acceptance.py --output-dir /tmp/phba-decode-acceptance
```

This produces `acceptance.json`, JUnit XML and logs. A missing GPU, skipped new GPU cases, failed numerical checks or failing regression prevents acceptance. Do not relax tolerances or suppress failing cases. Benchmark commands require a passing manifest whose source digest still matches.

## Stage and component comparisons

```bash
python benchmarks/phba/benchmark_decode.py \
  --checkpoint "$PHBA_CHECKPOINT" \
  --validation-report /tmp/phba-decode-acceptance/acceptance.json \
  --output-dir "$RESULTS_DIR/phba-stages" \
  --stages p0 p1_state p1_route p1_split --lengths 32768 65536 --steps 256 --repeats 5

python benchmarks/phba/benchmark_decode_components.py \
  --validation-report /tmp/phba-decode-acceptance/acceptance.json \
  --length 65536 --output "$RESULTS_DIR/phba-components-64k.json"
```

Resolve the checkpoint and results environment variables to real local paths. The full-model benchmark loads a trained local checkpoint and defaults to shared teacher-forced continuation tokens across stages. `--tokens` supplies a token tensor; otherwise synthetic token IDs are explicitly used. `--greedy` enables free generation. `--steps 256` means 256 incremental forward calls AFTER prefill, not 256 total generated tokens including the token obtained from prefill. All timing includes the incremental model call and argmax. Warm-up is excluded; no per-token host synchronization is inserted. CUDA event intervals include any stream idle gaps, not just summed kernel durations. Wall latency is also reported.

JSONL contains per-token intervals, boundary indices, ordinary/boundary latency statistics, prefill/decode peak allocated and reserved memory, and cache storage breakdown. CSV/Markdown summarize repeats. Boundary updates may not appear in p95/p99, so report explicit boundary/max latency. Storage accounting deduplicates views and includes allocated capacity slack. It is not a measurement of temporary workspace or allocator reserved memory; those are reported separately by CUDA peaks.

Component diagnostics compare eager/deferred state construction, re-pooled/cached routing, and Torch/direct split readout for one synthetic layer. They are not end-to-end 1.3B results. The runner does not implement the separate Transformer/MoBA/NHA/KDA adapters; integrate those in gsa2 without treating cache-free generation as cached decode.

## Validation status of this patch

Implemented and executed locally: staged CPU tests (8 P0, 9 after deferred state, 10 after centroid caching, 12 after split readout), then 3 kernel-source CPU semantic tests, plus Python syntax compilation. Final CPU suite: 15 test methods, all passing. No CUDA device, Triton package or Transformers package was available locally. Actual GPU compilation, trained-model equivalence, GPU performance, 32K/64K peak memory and the model adapter therefore remain unverified. No GPU speedup or memory result is claimed.
