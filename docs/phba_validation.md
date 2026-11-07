# PHBA validation

The numerical and throughput records compare PHBA revision `3f39ae4aa64411221ea935b267630f212cd2edc1` with `96e17c10b970249cafcbb9a15ff42b9907a52ec4`. The subsequent smoke-CI and line-wrapping update leaves the runtime computation ASTs unchanged. Both roles use NVIDIA H20, PyTorch 2.9.0+cu128, Triton 3.5.0, Transformers 5.6.2 and FlashAttention 2.8.3.post1. This record covers numerical preservation, public API boundaries, release packaging and training-compute throughput.

## Correctness and API coverage

The final source gate passes 64 tests with no skips: 47 PHBA operator/model/projection tests and 17 benchmark tests. The pinned H20 gate sets `PHBA_REQUIRE_EXACT_GEMM=1`; each of the three screened projection geometries must execute its cuBLASLt call. Separate tests verify native gradients when the CUDA version, library fingerprint or build prerequisites do not match.

Coverage includes FP16/BF16 dense and packed outputs and all six input gradients, final states, independent routing/metadata references, FP64 formula gradcheck, save/load, central model exports, checkpoint gradients, padding and cache-free generation. New boundary cases reject cross-batch or conflicting sequence plans and invalid head dimensions before attention initialization. Affine-free normalization is checked for both norm implementations and through model gradients. Forced low-memory prefix/tape fallbacks match the reference path and repeat deterministically. Original cumulative-length tensors retain rotary cache identity only while safe to reuse; in-place length changes are tested against fresh tensors.

The shared configuration, Transformers API, benchmark and split-wheel contract gate passes 574 tests. Three additional split-package tests pass for core-first installation, import from the checkout with only core installed, and separate namespace installation paths. Existing numerical tolerances remain unchanged. The sole numerical-suite warning is PyTorch's TF32 API deprecation.

The default 1,264,423,936-parameter model matches all 267 initial parameter tensors, inputs, loss, gradient norm and raw/clipped gradient bytes. The installed split wheels pass the same full-model gate, with all three exact GEMM sessions active. Loss is `10.784871101379395`; gradient norm is `2.495190382003784`. Every complete-step pair also matches all optimizer-state hashes after 70 steps.

The 35 Triton kernel computation ASTs and selected GEMM descriptors remain unchanged. Shared Python edits are limited to central model exports, the benchmark registry, the split-package builder and its contract test; 823 other pre-existing non-PHBA Python files are byte-identical. The PHBA runtime plus central-export file manifest for that numerical/throughput revision has SHA256 `e2a8dedffefffbab62a468ac3f9403600f90cbdfa1157c874f091c4c9bbaddfe`, computed from `json.dumps(path_to_sha256, sort_keys=True).encode()` using repository-relative paths. Header, pre-commit and whitespace checks pass.

## Official split-package release

Validation uses `scripts/build_packages.py`, then builds and installs both `fla-core` and `flash-linear-attention` wheels outside the checkout. The two C++ helpers and algorithm JSON are present, byte-identical to source, in the core wheel only. Automatic model registration works after importing FLA. The installed-package suite passes 48 tests, including import binding; the complete model's 267 gradients remain byte-identical. No prebuilt shared libraries or optimizer implementation are added.

The standard operator registry runs both dense and packed `chunk_phba` forward/backward cases and preserves distinct result keys across JSON serialization. CI configuration installs and checks the FlashAttention interface when PHBA tests are selected on NVIDIA runners. Upstream-hosted workflows have not been executed as part of this local H20 validation.

## Smoke selection and execution

The H100 smoke workflow marks four existing PHBA output/backward cases covering FP16/BF16, dense/packed inputs, tails, one-token sequences and non-power-of-two widths. It installs `flash-attn>=2.8.3.post1,<2.9` through the selected Python interpreter and checks the attention schema and CUDA device before testing. The PHBA branch uses `FLA_CI_ENV=0` and requires a nonempty JUnit report without skipped, failed or errored cases.

The exact dependency and test shell bodies pass on H20 with the software versions recorded above: **4 passed, 18 deselected, 0 skipped**, with one TF32 API deprecation warning. Each selected case executes `chunk_phba` and compares its output and all six floating input gradients against the reference at unchanged tolerances. Separate report checks confirm that empty, skipped, failed, errored and mixed passed/skipped runs are rejected. Dependency discovery includes `tests/ops/test_phba.py`. This verifies the PHBA smoke commands on H20; the H100/PyTorch 2.12 runner itself has not been executed.

## Paired complete-step throughput

Each case uses three same-device paired seeds, 3407/3408/3409, with alternating role order. The recipe is 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000, 128 prefix slots, B1 x 8192, FP32 parameters, BF16 autocast, no activation checkpointing, clipping at 1, fused native PyTorch AdamW, betas 0.9/0.95, and weight decay 0.1 excluding one-dimensional parameters. Each run has 20 warmup and 50 measured steps with lr=0. Native norm/activation/rotary configurations are matched.

Packed lengths are 2048/3072/3072; sequence-boundary labels are masked before shifted language-model loss. Synchronized wall time includes zero-grad, forward/loss, backward, clipping, optimizer, allocations and dispatch. Input generation and audit hooks are outside measured steps.

| Workload       | Baseline ms/step | Current ms/step | Paired throughput gain | Peak allocated GiB, baseline / current |
| -------------- | ---------------: | --------------: | ---------------------: | -------------------------------------: |
| Dense C64/K16  |          1040.17 |         1040.78 |                -0.059% |                  41.294661 / 41.294661 |
| Dense C128/K8  |           849.72 |          849.88 |                -0.019% |                  39.423200 / 39.423200 |
| Dense C256/K4  |           773.08 |          773.88 |                -0.104% |                  38.494672 / 38.494672 |
| Dense C512/K2  |           749.66 |          749.22 |                +0.058% |                  38.012891 / 38.012891 |
| Packed C128/K8 |           783.20 |          783.28 |                -0.010% |                  39.423263 / 39.423263 |

Latency columns are arithmetic means; gains are geometric means of paired throughput ratios. Dense geometric-mean throughput change is -0.031%, and packed change is -0.010%. Individual pairs include both positive and negative changes; C256/K4 averages -0.104%. These results indicate effectively preserved throughput, not a universal speedup or an exact-zero-latency-change guarantee. No dense-FLOPs hardware-floor estimate is used.

## Reproduction and scope

See [the PHBA guide](phba.md) for source installation, model construction and benchmark commands. Exact paired reproduction additionally requires the recorded native launch choices and reference hashes; raw experiment artifacts remain outside the source tree.

Default `elementwise_affine=True` checkpoint parameter names and training numerics are preserved. Setting it to `False` now intentionally removes normalization scale parameters. Invalid sequence plans are rejected instead of silently changing sample boundaries. Persistent decode caching, context/tensor parallelism, full-graph compilation, other accelerators and convergence are outside this validation. The lr=0 runs measure training computation. Optimizer selection remains the training framework's responsibility.
