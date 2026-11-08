# Public coupled execution and native checkpoint policy

The public chunk operator accepts `precision_policy='bf16_selective', source_vjp='coupled_fused'` for fixed-curvature C64/K128/g32/no-carry. `source_vjp='coupled'` retains the original state adjoint; `coupled_state` also replaces that adjoint, and `coupled_fused` uses the measured32-token source owners. `expanded` remains the default. Other chunk sizes and learned curvature keep their existing expanded paths; an explicitly incompatible coupled selection raises before execution.

`KeyGatedHyperbolicDeltaNetConfig(source_vjp=...)` carries the choice through Model and Layer. The default configuration omits the new field, and its serialized names/values match the prior class. Parameter names and state-dict tensors are unchanged. Short evaluation/decode retains the existing recurrent BF16 route; coupled selection governs supported chunk execution.

The native training CLI adds `--khdn-source-vjp` and `--linear-checkpoint-policy {recompute,save_mm}`. `save_mm` retains external matrix-product outputs with PyTorch nonreentrant selective checkpointing; the custom operator private tape is recomputed. The builder and pre-update checks validate the actual context factory, including copied models. Default checkpoint contracts keep their old fields; save_mm adds a policy tag, so recovery cannot silently switch strategies. The policy is available for KHDN no-carry and KDA; it is not a KHDN-only optimization.

## Verification

Pinned H20/Torch2.9.0+cu128/Triton3.5.0 with compiler SHA `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`:

- 216 affected host checks pass. Two local constructor checks require unavailable local Triton;19 real model cases run on H20 instead.
- 33 public GPU operator/constant cases pass without skips, including all three coupled selections, FP32/BF16, dense/packed/tails, cancellation, strong gates and the retained public curvature path.
- 19 actual model trace cases pass across legacy/selective C16/32/64/128, coupled selections, checkpointing and mm-save. They perform real optimizer updates and reject corrupted source choices, precision, ownership, metadata, batch/model identity and checkpoint contexts before updates.
- The raw compiled assembly and stage plans of all19 traces, plus19 update-guard records, were independently reread/reparsed on CPU. Kernel and helper hashes were also checked against actual frozen source files. Expanded arithmetic counts are retained; new coupled contractions get their own exact contracts.
- The native check-only CLI passes41 parameter-gradient/update comparisons with serial accumulation and same-process fresh-model recovery. The16-microbatch fixture includes unequal label counts and a zero-label last microbatch. It also compares the fixed-curvature model and six requested operator sources with independent untimed dense truth. Expanded selection separately retains learned-curvature/seven-source coverage.
- A current public-source full24 B8×T2048 initial-gradient screen captures all459 parameter tensors against frozen strict truth. The coupled C64 candidate has gradient cosine `0.9924001`, norm ratio `0.9996324`, relative L2 `0.1232652`, and15,321 coordinates beyond the older elementwise BF16 diagnostic. It performs zero optimizer updates; this is an initial-gradient gate, not continuous training quality.

These are bounded integration checks, not final24-layer quality, formal throughput, cross-process recovery or distributed acceptance. The earlier1900ms research measurement predates this public integration and does not qualify its final source identity. Current inference cache/packed numerical repair remains a separate experiment.

The private inference repair passes the original eight protected cache/packed/restore checks after three BF16 products with FP32 intermediates and final BF16 logits. A same-H20 full24 B1×T2048 latency screen shows the cost: baseline prefill/decode `282.30/1845.09 ms`, fused repair `443.72/2120.45 ms`. It remains a diagnostic and is excluded from the public implementation. Training arithmetic and model defaults are unchanged.

## Reproduction

With the pinned runtime:

```bash
FLA_CI_ENV=0 pytest tests/ops/test_key_gated_hyperbolic_bf16.py -k 'selective_coupled_source_vjp or selective_public_policy_source_vjp or fixed_curvature_cache_lifetime'
FLA_CI_ENV=0 pytest tests/benchmarks/test_khdn_bf16_harness.py -k native_model_trace_and_serialized_rejection
python -m benchmarks.benchmark_hyperbolic_training \
  --methods khdn_no_carry --khdn-precision-policy bf16_selective \
  --khdn-source-vjp coupled_fused --chunk-size 64 --group-size 32 \
  --linear-checkpoint-policy save_mm --optimizer-backend fp32 \
  --loss-backend plain --check-only --seed 101 --data-seed 1009
```

Actual training kernels use BF16 products and FP32 control/accumulation; FP64 is restricted to the explicitly selected untimed oracle. No tolerance changes or six-product compensation are part of this integration. Original failed cases and older measurements remain in their frozen records.
