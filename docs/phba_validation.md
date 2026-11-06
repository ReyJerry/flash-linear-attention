# PHBA migration validation

This validation compares the mature deterministic PHBA implementation with its integration into official FLA main at `cfb0a3c416541e5ab4befe0867a91044347298e3`. It checks numerical preservation and training-compute throughput; it is not a model-quality or convergence experiment.

## Source and environment

The frozen baseline source manifest has SHA256 `2e60e05ed5eb071f66937b3f45248ab3cd73fbf3dfffe514a1823515723b92da`. Its native AdamW312 helper has SHA256 `66884aa5bc6d3ba60c87dd8fa3b67d221a769cf5b2abc42117b94169ab6453e5`. Both roles retain the mature CTA ordering, exact GEMM selection, metadata reuse and derivation, terminal gradient stores, gate algebra, int32 sorting, and AdamW312 schedule.

The migrated PHBA Python/C++/JSON file manifest has SHA256 `477fb25f99902700c98a18b7934634ddccb1e6fe309db11ccc83dbd999f00842`. It is calculated by hashing each file, then hashing `json.dumps(path_to_sha256, sort_keys=True).encode()` with repository-relative paths. All 34 PHBA implementation files match the tested snapshot. Documentation and benchmark-driver changes are tracked separately.

Hardware is NVIDIA H20. Both roles use PyTorch 2.9.0+cu128, Triton 3.5.0 and FlashAttention 2.8.3.post1. The baseline uses Transformers 4.57.6; the official-main integration uses Transformers 5.6.2, as required by the new FLA base. This is therefore an integration comparison including the framework-version transition. Native normalization/activation/rotary launch configurations are matched explicitly.

## Numerical and API gates

The four PHBA test files pass 27 tests with no skips on H20. Coverage includes FP16/BF16 dense and variable-length outputs and all six operator-input gradients, tails and non-power-of-two slots, final states, FP64 formula gradcheck, unsupported inputs, registration/save-load, checkpoint gradient identity, right padding, cache-free generation, optimizer fallback and all three screened GEMM geometries. The one warning is the existing PyTorch TF32 API deprecation; it is not a relaxed numerical-tolerance warning.

A tiny paired model matches initial weights before loading, captured activations, logits, loss, and all 25 raw/clipped parameter gradients exactly. The 1,264,423,936-parameter model matches all 267 initial parameter tensors, input, loss, gradient norm, and raw/clipped gradients. Reference loss is `10.784871101379395`; gradient norm is `2.495190382003784`. The exact H20 GEMM backend is asserted active.

The full 267-tensor optimizer gate compares raw/clipped gradients, parameters, `exp_avg`, `exp_avg_sq` and step counters with native AdamW across seven steps. Learning rates are `[0, 1e-4, 3e-4, 1e-3, 0, 2e-4, 1e-4]`, with nonzero initial moments and varied initial steps. Every comparison is byte-identical, and every original parameter's step increments once per update.

All dense and packed throughput pairs additionally match first-step raw/clipped gradients, loss/norm across measured steps, and all optimizer-state hashes after 70 steps. These throughput runs use lr=0; the separate nonzero-learning-rate gate verifies actual parameter updates.

## Paired complete-step measurements

Each row uses three seeds (`3407`, `3408`, `3409`) with baseline and candidate on the same physical H20, alternating execution order. The recipe is 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000, 128 prefix slots, B1 x 8192, FP32 parameters, BF16 autocast, no activation checkpointing, clipping at 1, AdamW betas 0.9/0.95 and the same weight-decay exclusions. Each run has 20 warmup and 50 measured steps. Packed lengths are `[2048, 3072, 3072]`; labels at sequence starts are masked before shifted language-model loss.

Synchronized wall time includes zero-grad, forward/loss, backward, clipping, optimizer, allocations and per-step dispatch. Input creation and untimed audit hooks are excluded. Both roles execute 192 exact GEMM calls and the same native AdamW packet schedule on the audited step. Peak allocated memory includes warmup after the first two initialization steps.

| Workload | Baseline ms/step | Migrated ms/step | Paired throughput gain | Peak allocated GiB, baseline / migrated |
| --- | ---: | ---: | ---: | ---: |
| Dense C64/K16 | 1036.64 | 1033.24 | +0.330% | 41.295302 / 41.294661 |
| Dense C128/K8 | 846.07 | 842.69 | +0.402% | 39.422864 / 39.423200 |
| Dense C256/K4 | 768.15 | 766.52 | +0.213% | 38.495252 / 38.494672 |
| Dense C512/K2 | 744.39 | 742.04 | +0.316% | 38.012922 / 38.012891 |
| Packed C128/K8 | 778.84 | 776.80 | +0.264% | 39.422927 / 39.423263 |

Latency columns are arithmetic means of three runs; gain is the geometric mean of their paired throughput ratios. Dense geometric-mean gain across 12 pairs is **+0.315%**. One dense pair, C256/K4 seed 3409, is **-0.038%**. Packed gain is **+0.264%**, with seed 3408 at **-0.198%**. These measurements support preserving mature performance; they do not establish a statistically significant universal speedup. Peak allocated differences are below 1 MiB. No dense-FLOPs proxy or hardware-floor claim is made.

An initial packed harness attempt stopped before timed steps because expected autotune entries were compared in insertion order while observed entries were sorted. Canonicalizing both lists by key fixed the record comparison; kernel configurations and numerical gates were unchanged. The reported packed measurements are from the subsequent completed study.

## Reproduction and scope

Run the commands in [the PHBA guide](phba.md). For the packed benchmark, add `--lengths 2048,3072,3072`. Reproducing the paired audit additionally requires the frozen baseline and the recorded native launch choices; ordinary benchmark invocations can autotune differently. Raw per-step records and numerical hashes remain in the experiment evidence archive, outside the source tree.

All 827 Python files present in the official base remain byte-identical. PHBA self-registers only after importing `fla.models.phba`; existing model/layer exports and benchmark registries are untouched. Header checks, the repository's pre-commit hooks and whitespace checks pass. The wheel contains C++ source and algorithm descriptors, with no prebuilt shared library. Installing it into an isolated target and running the same tests outside the source checkout passes 28 tests, including an assertion that imports resolve to the installed wheel. Its 654 packaged Python/C++/JSON source files match the locally built wheel byte-for-byte.

Persistent decode caching, context/tensor parallelism, full-graph compilation, other accelerators and training convergence are outside this validation. Numerical identity claims apply to the tested runtime, shapes and matched launch configurations.
