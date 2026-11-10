# BF16/FP32 research: cost, error and tiny distributed updates

The isolated direct-BF16 candidate completes the tiny same-precision 1/8/16-GPU update and checkpoint checks below, but still fails the original full-model cross-precision gradient budget.
It is not exposed by the public operator or Config, and is not qualified for production training.
The published `strict`, `mixed_readout` and `bf16_fs6` paths still contain FP64; their historical passes do not satisfy the newer BF16-first, FP32-where-needed, no-FP64 training target.
The historical results retain their original budgets and failure labels; the public default is unchanged.

The candidates use BF16 matrix operands, FP32 accumulation and FP32 control/state arithmetic.
The compensated variant splits each operand into three BF16 components and sums six products; the residual variant additionally retains readout-product and state-rounding residuals.
The direct variant is the priority for further evaluation because compensation has a measured cost, while its effect on sustained training quality remains unresolved.
Explicit, untimed FP64 references are separate from candidate execution.
Source/compiler fingerprints, job IDs and source-record hashes are in [evidence.json](evidence.json).

## Same-card kernel-chain cost

One H20 replayed the actual full-model Layer0 sources and incoming output gradient at B8/T2048/H16/K=V128, g32/C16/no_carry.
Each variant had 10 warmups and 30 steady samples, with interleaved execution order.
The timed region was a fixed-buffer chain of six forward/manual-backward kernels with fresh forward state each iteration.
It excludes allocation, validation, PyTorch autograd, the other model layers, loss and optimizer.

| Candidate                  | Median ms | p95 ms  | Saved forward buffers, bytes | Original payload checks |
| -------------------------- | --------: | ------: | ---------------------------: | ----------------------- |
| Direct BF16                | 67.593    | 67.727  | 3,306,160,128                 | Fail                    |
| Six-product compensation   | 93.835    | 94.719  | 3,306,160,128                 | Pass on this payload    |
| Additional state residuals | 136.271   | 136.540 | 4,514,119,680                 | Pass on this payload    |

Direct BF16 has 1.388× the compensated variant's kernel-chain throughput and 2.016× the residual variant's.
These are research cost measurements of finite implementations, including an implementation that fails the original accuracy check.
They establish neither qualified training throughput nor an end-to-end comparison with KDA.
The extra state residuals cost 45.23% more time and 36.54% more saved forward storage than the compensated variant on this payload.
Saved-buffer bytes are not peak model memory.

Run `python3 summarize.py` in this directory with Python 3.10+ to recompute the medians, nearest-rank p95 and ratios from all 120 original observations in [cost-samples.json.gz](cost-samples.json.gz).
The script verifies their hash and sample partition; it does not rerun GPU training or reproduce the external raw-tensor audits.

## Full 24-layer zero-update comparison

The fixed initial batch used 24 layers, hidden2048, 16 heads, K=V128, FFN5184, vocabulary32000, B8/T2048, g32/C16/no_carry, fixed c=1, trainable Key-channel gates, ShortConv, checkpointing and plain cross-entropy.
All 459 trainable tensors, totaling 1,412,912,512 coordinates, were compared with retained strict gradients.
Every candidate completed one forward/backward; the reproduced strict control ran forward only.
No optimizer was constructed and all final weight hashes matched the initial weights.

| Candidate                  | FP32 evaluation CE difference | Gradient relative L2 | Gradient cosine | Elements outside original budget |
| -------------------------- | ----------------------------: | -------------------: | --------------: | -------------------------------: |
| Direct BF16                | +0.000056267                  | 0.111466             | 0.993787        | 13,001                           |
| Six-product compensation   | +0.000455856                  | 0.058640             | 0.998281        | 5,110                            |
| Additional state residuals | +0.000536919                  | 0.058522             | 0.998288        | 4,976                            |

The strict FP32 evaluation CE was 10.7865219116.
Small loss differences alone do not establish acceptable optimization error or sustained training quality.
The raw candidate gradients remain in the external experiment archive; this export contains the audited aggregate records.
All three original gradient failures remain failures.
A separate [prospective BF16 contract](acceptance-contract.txt) was approved by the user on November 5, 2026.
It requires new seeds and validation runs; these historical failures are not retroactively reclassified.
User approval does not imply maintainer approval or completed qualification.

## Tiny same-precision distributed checks

The actual model has two layers, hidden256, two heads, K=V128, FFN512, vocabulary97, g32/C16, learned positive curvature, trainable Key-channel gates, FP32 parameters, BF16 autocast, ShortConv and layer checkpointing.
The same 16 B2/T7 microbatches contain 224 input slots and 126 valid labels after packed-boundary masking, unequal-label masking and a completely zero-label last microbatch.
W1 uses a genuine NCCL/DDP group; W8 uses one node and W16 two nodes, with unique GPU/rank bindings.
These are tiny fixtures, not the full-model 262144-token strong-scaling or 16384-token-per-rank weak-scaling workloads.

Each microbatch loss is divided by the true global valid-label count.
Default DDP averages the parameter gradients; the driver multiplies the synchronized FP32 gradients by world size once, after the final backward and before clipping.
Earlier local microbatches use `no_sync`.
The serial comparison uses the same per-microbatch denominator and the identical candidate, model, data and shared-kernel tuning choices.

| GPU count | Maximum gradient difference from W1 | Maximum update difference from W1 |
| --------: | ----------------------------------: | --------------------------------: |
|         1 |                                   0 |                                 0 |
|         8 |                      5.96046448e-08 |                     3.7252903e-09 |
|        16 |                       7.4505806e-08 |                     3.7252903e-09 |

All 43 tensors satisfy the original gradient budget `(atol=0.002, rtol=0.02)` and update budget `(atol=2e-6, rtol=0.02)`.
The separately computed serial reference arrays are byte-identical across the three world sizes.
Within each group, all ranks have byte-identical gradients, model parameters and Adam state.
Each rank performs four physical Adam calls: the first distributed update, its serial control, an uninterrupted next update, and a replay of that next update after loading the saved state.
Every replay matches the uninterrupted model, Adam, RNG and full parameter gradients byte for byte.
The recovery microbatches differ across world sizes; only the identical first global update is compared across world sizes.

All actual forward/backward calls are counted per layer and stage; the six KHDN kernel families have recorded TTIR/PTX without FP64 instructions.
A runtime tensor guard also rejects FP64 tensors during candidate forward, backward, clipping and Adam operations.
The first W8 launch failed the full-training CLI's scaling check before model updates; it was fully reaped.
The corrected tiny-fixture entry preserves the W1 model configuration, data, optimizer, kernel and numerical budgets.
Both attempts are retained in the evidence records.

## Finite training and remaining work

A separate single-seed synthetic two-layer study ran 64 actual updates for each of direct, compensated, residual and explicit strict-reference arms, all finite.
At step 64, recomputing CE from the saved logits gave differences from strict of +0.000397365, +0.000106927 and +0.000422917 respectively.
This is not a real-data quality result or a multiple-seed acceptance test.
The direct arm also restored checkpoint32 in a fresh process and reached checkpoint64 with byte-identical model, Adam and RNG state; the original full per-step gradients were not retained for that separate 64-step study.

Validation uses explicit FP32 logits with autocast disabled: on the pinned environment, a returned FP32 loss tensor under BF16 autocast had coarser arithmetic than explicit FP32 evaluation.
Historical reported training losses remain separate and are not silently replaced.

Remaining gates include qualifying the approved prospective BF16 contract on new seeds, integrating a reviewed public BF16/FP32 policy, complete supported-domain/source-gradient coverage and targeted sanitizer checks, full-model finite training and resume, all four full sequence lengths, genuine full-model 1/8/16-GPU strong/weak scaling, and repeated same-environment training throughput.
The tiny results do not close those gates, grant external review approval, or change the published default.
The GPU probe archives and large tensor/checkpoint records are external research artifacts; the checked-in summary is not a standalone GPU reproduction package.
