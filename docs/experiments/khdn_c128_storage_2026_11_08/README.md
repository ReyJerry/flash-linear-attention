# C128 BF16 storage and scheduling research update

This draft reduces preparation/storage costs while retaining the grouped KHDN recurrence, g32 mathematical groups and a real 128-token execution chunk. [逐项进展与审批说明](PROGRESS.zh-CN.md).

**This is not a production-qualified release.** The published code includes a final FP32-reduction correction and a narrower compiled-precision policy update that have host tests but have not yet been rerun on GPU. The timing, initial-gradient and trajectory measurements below precede that correction. Original numerical failures remain failures.

## Implementation changes

- Main contractions and hierarchical off-diagonal inverse products use single BF16 products with FP32 accumulation. The 16-row diagonal inverse remains FP32; its single-warp layout and tiled upper-zero stores avoid cross-warp reductions and a large address tile.
- Most common/group preparation matrices and saved S/Y/U/RO snapshots use BF16. Canonical state carried between chunks and the final state remain FP32. Decay scales, prefix/suffix factors, mass vectors and normalization controls retain FP32 storage.
- State contractions load reduction tiles of 64. Reverse-scan loads are delayed until use; a V reload avoids retaining a full tile across a GEMM. No extra GEMM or global state-adjoint buffer is introduced.
- Shared feedback matrix and scalar-system gradients run separately. The other four shared coefficient gradients retain their faster grouped reduction; the larger concatenated alternative was measured and rejected.
- Preprocessing handles eight tokens per head, caches successful compiled-image precision checks and retains FP32 parameter-gradient reductions. Immutable chunk metadata is cached with stream events and lifetime recording, including inference-to-autograd reuse.
- The public trace contract records the new stages, buffer dtypes and single-product inverses. A final audit found BF16 accumulation in the erasure projection of the measured candidate. `erase` and `erase_vjp` now explicitly promote their reduction input to FP32. The compiler checker admits only declared rank-two BF16 elementwise sites; BF16 scalar/vector reductions, unsupported shapes, FP16 and FP64 remain rejected. **This final correction still needs GPU verification and retiming.**

Public symbols, model state-dict format and default precision/chunk settings are unchanged. The C128 experimental operator remains restricted to its documented SM90 runtime, K128, V32/64/128, g32 and no-carry. The Layer uses K=V128. No FP64 training fallback or six-product compensation is introduced.

## Same-card full-model short screen

H20, PyTorch 2.9.0+cu128, Triton 3.5.0; 24 layers, hidden2048, H16, K=V128, FFN5184, vocabulary32000, B8×T2048, g32, ShortConv, outer checkpointing, plain CE, BF16 autocast, FP32 parameters and the same explicit FP32 AdamW in all arms. Compiler SHA256: `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`.

Two warmup and three steady samples per mode; five timed optimizer updates plus one separate profile step per arm. Step includes clipping, optimizer and public checks, excluding transfer and zero-grad. F+B is directly measured. These are research screens, not the final three-trial 10-warmup/30-steady protocol.

| Arm | Direct F+B ms | Complete step ms | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Combined BF16 storage, previous diagonal layout | 2127.871 | 2151.434 | 26.14 |
| Selected single-warp diagonal layout | 2106.447 | 2141.073 | 26.14 |
| Matched-output KDA | 1710.857 | 1735.064 | 25.58 |

KHDN and matched KDA each contain 1,412,912,512 parameters. The selected measured step is 1.23400× KDA, about 81.04% of its throughput; the paired step reduction is 0.482%. The separately profiled diagonal stage falls from 42.669 to 21.700 ms. Neither kernel time nor separate experimental gains are added together to predict whole-step speed.

The preceding storage experiments measured 1,137,704,960 bytes less peak allocation for preparation storage and 267,649,024 bytes for the separate saved-state workspace experiment. The selected combined run is about 0.56 GiB above matched KDA. These measurements have different paired baselines and are not additive.

## Numerical and integration evidence

- The selected layout's eight fixed NaN-poisoned cases retain one beta-gradient RMS failure: 0.000288358 versus 0.000188444, in case4. All corresponding elementwise checks and finite checks pass. The failure is not waived. Cases cover tails/multiple chunks, packed/empty segments, nonzero state, beta endpoints, c=0, strong gates, distinct groups and both input dtypes.
- Before the final diagonal-layout change, the combined implementation captured all 459 initial parameter gradients against a frozen strict reference with identical initial weights, RNG and batch: cosine0.9901701637 and norm ratio0.9997752024 pass the adopted model-level gate, narrowly for cosine. The older coordinate budget still has18,632 failures. Initial same-batch FP32-CE-derived PPL difference is−0.092845%. This capture makes zero optimizer updates.
- That same preceding source completed16 consecutive updates and a fresh-process8→16 continuation. Shared model/optimizer state content hashes, RNG, training records and validation results match exactly. The frozen BF16 C16 control was reused after public-source, initialization, data and optimizer checks. On two heldout **synthetic** batches, paired PPL differences at0/4/8/12/16 updates are +0.04590%, −0.06856%, −0.24405%, −0.09188%, +0.12741%. The0.1% line is an observation, not automatic acceptance or rejection. This is not real-corpus or long-term quality evidence.
- Public GPU integration before the final reduction/policy correction: **12 original-distribution C128 numeric tests fail; four metadata lifecycle tests pass; no test errors or skips.** The CLI stops at its compiled-arithmetic check. These failures are recorded in the release data and are not replaced by the fixed-case or model-level results.
- Current host checks: **404 passed, four GPU tests skipped**. This includes matrix-BF16 admission and negative tests for scalar/vector BF16 reductions, wrong shapes and forbidden arithmetic. Host checks do not qualify CUDA execution.

The selected-layout timings, preceding-source model/trajectory evidence, and post-measurement correction are separate version boundaries. No new-source full-model quality or distributed acceptance is claimed.

## Hardware diagnosis and remaining work

The earlier streamed-state source has paired selected-kernel NCU evidence:1,008 KHDN and504 KDA launches, with all459 clipped gradients exactly matching each arm's unprofiled comparison. Its KHDN state kernels show substantial local-memory requests while the corresponding KDA kernels show none; occupancy alone does not explain the gap. These counters belong to that earlier source, not the final publication. Cache-flushed replay times are not ordinary training timings.

Open work includes the public GPU recheck after the reduction correction, raw-operator numerical qualification, cache/packed model regressions, four-shape final timing, full1/8/16-GPU gradient/update/recovery/scaling, and longer/model-quality validation. Optional C128 Y-carry remains unqualified. The PR and engineering goal remain incomplete.

## Evidence verification

`observations.json.gz` contains raw short timings, operator metrics, initial-gradient statistics, trajectory/recovery records and the failed public-integration summary. `measured-sources.tar.gz` is a source evidence overlay, not a portable runtime or full checkout. Unrelated protected files, transport scripts, credentials, large tensor/checkpoint files and raw GPU traces are excluded. `source-binding.json` identifies the explicit post-measurement `erase`/`erase_vjp` correction and import/unused-assignment normalization.

Run at this publication revision:

```bash
python docs/experiments/khdn_c128_storage_2026_11_08/summarize.py
```

The verifier recomputes hashes, source differences, timing medians, gradient aggregates, paired PPL and recovery-record equality. It does not independently reread the large gradient/checkpoint tensors or manufacture GPU qualification. The earlier `khdn_c128_2026_11_08` package remains historical evidence pinned to d9103472; run its current-source AST verifier from that revision.
