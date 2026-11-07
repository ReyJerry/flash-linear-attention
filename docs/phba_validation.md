# PHBA validation

This record validates the alignment-guard implementation on NVIDIA H20, PyTorch 2.9.0+cu128, Triton 3.5.0, Transformers 5.6.2 and FlashAttention 2.8.3.post1. The source/runtime manifest has SHA256 `9b9f7d2671c210e2803480bc80ccbe869a1512022295c7f25944e0c22b1b5a9c`, computed from `json.dumps(path_to_sha256, sort_keys=True).encode()` using repository-relative paths. The manifest covers the tracked `fla` runtime and the complete-step benchmark script.

## Correctness and API coverage

The strict source gate passes **79 tests, no skips**: 62 PHBA operator/model/projection cases and 17 benchmark cases. `FLA_CI_ENV=0 PHBA_REQUIRE_EXACT_GEMM=1` makes numerical failures and an unavailable pinned H20 backend fail the gate. The sole warning is PyTorch's TF32 API deprecation.

Projection coverage checks all three screened GEMM geometries against native outputs and gradient bytes. It includes FP32 autocast inputs, BF16 views aligned to 16 bytes, unaligned activations and incoming gradients, and separate unaligned A/B GEMM operands. Tests assert actual cuBLASLt execution for eligible calls and zero calls for the alignment fallback. Fused SwiGLU materializes a fresh aligned activation; unaligned pre-activation inputs therefore remain eligible when the actual GEMM operands are aligned. Its unaligned-gradient path retains the original einsum operation.

Operator/model coverage includes FP16/BF16 dense and packed outputs and all six input gradients, final states, independent routing/metadata references, FP64 formula gradcheck, save/load, central exports, checkpoint gradients, padding, cache-free generation, affine-free norms, invalid head dimensions and conflicting sequence plans. Forced low-workspace prefix/tape fallbacks are checked against their references and repeat deterministically. Existing tolerances are unchanged. All 35 Triton kernel computation ASTs and the screened GEMM descriptors remain unchanged by this update.

Every current complete-step pair matches all 267 initial parameter tensors, inputs, loss, gradient norm, raw/clipped gradient bytes and optimizer-state hashes after 70 steps. The installed split wheels also pass the default full-model gate with all three exact GEMM sessions active: loss `10.784871101379395`, gradient norm `2.495190382003784`, and no raw or clipped gradient mismatches.

## Official split-package installation

The formal `scripts/build_packages.py` path builds both wheels and source distributions. All 653 packaged Python/C++/JSON files match the current source bytes. The two C++ helpers and algorithm JSON are present in `fla-core` only; no prebuilt shared library or optimizer is added. Outside the checkout, the installed PHBA suite passes **63 tests, no skips**, including import binding, registration and alignment/fallback checks.

Four host-side split-package contract/install-order tests pass. Their full-runtime-dependency companion is intentionally deselected on the local host; the separate installed GPU suite and full-model gate above execute on H20. The unchanged shared configuration, Transformers API, benchmark and split-wheel gate previously passed 574 tests at `3f39ae4aa64411221ea935b267630f212cd2edc1`; that count is prior evidence, not a new run.

## Smoke selection

The H100 smoke workflow selects four existing PHBA forward/backward cases, installs and preflights FlashAttention, uses strict numerical checks for PHBA, and rejects empty or skipped JUnit reports. Its exact shell commands passed **4 cases, 18 deselected, 0 skipped** on H20 at `47490bcdb3fbca795cff60ac6f9a96277698a020`. Those same four marked cases also pass in the current 79-case source gate. The H100/PyTorch 2.12 runner itself has not been executed. Empty, skipped, failed, errored and mixed passed/skipped reports are rejected by the validated workflow guard.

## Paired complete-step throughput

Baseline: `47490bcdb3fbca795cff60ac6f9a96277698a020`, before the alignment guard. Each workload uses three same-device paired seeds, 3407/3408/3409, with alternating role order and matched native normalization launch choices. The recipe is 1,264,423,936 parameters, 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000, 128 prefix slots, B1 x 8192, FP32 parameters/BF16 autocast, no activation checkpointing, native fused AdamW, lr=0, betas 0.9/0.95, clipping at 1, and weight decay 0.1 excluding one-dimensional parameters. Each run has 20 warmup and 50 measured complete steps. Packed lengths are 2048/3072/3072, with sequence-boundary labels masked before shifted language-model loss. Zero-grad, forward/loss, backward, clipping, optimizer, allocations, alignment checks and dispatch are timed together. Input generation and validation hooks are outside measured steps.

| Workload       | Baseline ms/step | Current ms/step | Paired throughput change |
| -------------- | ---------------: | --------------: | -----------------------: |
| Dense C64/K16  | 1040.04          | 1040.20         | -0.016%                  |
| Dense C128/K8  | 850.14           | 850.64          | -0.059%                  |
| Dense C256/K4  | 774.45           | 774.02          | +0.055%                  |
| Dense C512/K2  | 749.79           | 749.71          | +0.011%                  |
| Packed C128/K8 | 783.24           | 784.21          | -0.123%                  |

Latency columns are arithmetic means; throughput changes are geometric means of paired throughput ratios. Dense geometric-mean throughput change is -0.0021%; packed change is -0.1231%. All positive and negative pairs are retained. These measurements show effectively preserved throughput, not an exact-zero latency change or a universal speedup. Peak allocated memory is unchanged in each matched workload. No dense-FLOPs hardware-floor estimate is used. The lr=0 runs measure training computation, not convergence.

The earlier API/release migration comparison against `96e17c10b970249cafcbb9a15ff42b9907a52ec4` is preserved in [the preceding validation record](https://github.com/ReyJerry/flash-linear-attention/blob/47490bcdb3fbca795cff60ac6f9a96277698a020/docs/phba_validation.md). Its dense/packed changes were -0.031%/-0.010%; these are a different baseline comparison.

## Reproduction and scope

See [the PHBA guide](phba.md) for installation, supported APIs, execution paths and benchmark commands. Exact paired reproduction additionally requires the recorded native normalization launch choices and reference hashes; raw experiment artifacts remain outside the source tree. Shared Python edits in the full PR remain limited to central exports, benchmark registration, the split-package builder and its contract test. Other model implementations are unchanged.

Default PHBA parameter names and training mathematics are preserved. Invalid sequence plans are rejected; `elementwise_affine=False` omits normalization scale parameters. Persistent decode caching, context/tensor parallelism, full-graph compilation, other accelerators and convergence remain outside validation. Optimizer selection remains the training framework's responsibility.
