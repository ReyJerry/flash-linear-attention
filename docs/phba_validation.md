# PHBA validation

This record validates the 20-file PHBA operator layout on NVIDIA H20, PyTorch 2.9.0+cu128, Triton 3.5.0, Transformers 5.6.2 and FlashAttention 2.8.3.post1. The source/runtime manifest has SHA256 `a8762cefc40d9388d197ea19de439bd4fbed270ff36eb35b409dcaba2926168c`, computed from `json.dumps(path_to_sha256, sort_keys=True).encode()` using repository-relative paths. The manifest covers the tracked `fla` files and complete-step benchmark script. The baseline is `73a9e98a0a1e5155cdd5bde838ff70ab6bc42bbc`.

## Correctness and API coverage

The strict source gate passes **105 tests, no skips**: 22 operator, 25 projection, 41 model and 17 benchmark cases. `FLA_CI_ENV=0 PHBA_REQUIRE_EXACT_GEMM=1` makes numerical failures and an unavailable pinned H20 backend fail the gate. Seven additional PHBA configuration/Transformers cases pass with no skips (550 others deselected), as do 18 CPU-compatible cases with CUDA hidden (23 accelerator cases deselected). Warnings concern the TF32 API, advertised fused-linear-CE precision, CPU platform detection and standalone smoke-marker registration; no numerical failure is downgraded to a warning.

Projection coverage checks all three screened GEMM geometries against native outputs and gradient bytes. It includes FP32 autocast inputs, BF16 views aligned to 16 bytes, unaligned activations and incoming gradients, and separate unaligned A/B GEMM operands. Tests assert actual cuBLASLt execution for eligible calls and zero calls for the alignment fallback. Fused SwiGLU materializes a fresh aligned activation; unaligned pre-activation inputs therefore remain eligible when the actual GEMM operands are aligned. Its unaligned-gradient path retains the original einsum operation.

The fresh before/after audit covers nine combinations of FP16/BF16 dense/packed/no-history inputs and training, inference or forced low-workspace execution. Outputs, final states and all supported gradients are byte-identical. Instrumentation confirms execution of integer metadata kernels, sparse dK/dV, query-owner dQ and sequential prefix backward. Six reference-mode comparisons cover the FP64 formula and wrapper, plus FP16/BF16 storage-aware references with PyTorch or FlashAttention local branches; returned tensors and all six input gradients match byte-for-byte.

All 33 production Triton JIT ASTs are identical. The 171 retained top-level function/class bodies are AST-equivalent after excluding import relocation and the explicitly removed unused error-record method. Constants retain their expressions. Public operator entry files, all model/layer/module files and the H20 backend resources are byte-identical to the baseline. Runtime inspection also verifies public signatures, defaults, exports and the sequence-plan class module. The operator test changes only imports; its numerical assertions and tolerances are unchanged.

Every complete-step pair matches all 267 initial parameter tensors, inputs, first-step loss/norm and first-step raw/clipped gradient bytes. Final parameter and optimizer-state hashes also match after 70 steps. The freshly installed split wheels also pass the 1.264B model gate with all three exact GEMM sessions active: loss `10.784871101379395`, gradient norm `2.495190382003784`, and zero raw or clipped gradient mismatches.

## Official split-package installation

The formal `scripts/build_packages.py` path builds both wheels and source distributions. All **643** packaged Python/C++/JSON files match the source bytes: 456 in `fla-core` and 187 in `flash-linear-attention`. The 11 superseded private module files are absent, as are the earlier obsolete reference modules; both reference families and their active oracles live in `naive.py`. The two C++ helpers and algorithm JSON are present in `fla-core` only; no prebuilt shared library or optimizer is added.

Fresh installation outside the checkout passes **89 tests, no skips**, including import binding, registration and alignment/fallback checks. Four host-side split-package contract/install-order tests also pass. Their full-runtime-dependency companion is intentionally deselected on the local host; the separate installed GPU suite and full-model gate execute on H20. The unchanged shared integration previously passed 574 tests at `3f39ae4aa64411221ea935b267630f212cd2edc1`; that count is prior evidence, not a new run.

## Smoke selection

The H100 smoke workflow selects four existing PHBA forward/backward cases, installs and preflights FlashAttention, uses strict numerical checks for PHBA, and rejects empty or skipped JUnit reports. Its exact shell commands passed **4 cases, 18 deselected, 0 skipped** on H20 at `47490bcdb3fbca795cff60ac6f9a96277698a020`. Those same four marked cases also pass in the current 105-case source gate. The H100/PyTorch 2.12 runner itself has not been executed. Empty, skipped, failed, errored and mixed passed/skipped reports are rejected by the validated workflow guard.

## Paired complete-step throughput

Baseline: `73a9e98a0a1e5155cdd5bde838ff70ab6bc42bbc`, before the four core module consolidations. The candidate is identified by the runtime manifest above. Each workload uses three same-device paired seeds, 3407/3408/3409, with alternating role order and matched native normalization launch choices. The recipe is 1,264,423,936 parameters, 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000, 128 prefix slots, B1 x 8192, FP32 parameters/BF16 autocast, no activation checkpointing, native fused AdamW, lr=0, betas 0.9/0.95, clipping at 1, and weight decay 0.1 excluding one-dimensional parameters. Each run has 20 warmup and 50 measured complete steps. Packed lengths are 2048/3072/3072, with sequence-boundary labels masked before shifted language-model loss. Zero-grad, forward/loss, backward, clipping, optimizer, allocations, alignment checks and dispatch are timed together. Input generation and validation hooks are outside measured steps.

| Workload       | Baseline ms/step | Candidate ms/step | Paired throughput change |
| -------------- | ---------------: | ----------------: | -----------------------: |
| Dense C64/K16  | 1039.42          | 1040.51           | -0.104%                  |
| Dense C128/K8  | 849.74           | 849.49            | +0.029%                  |
| Dense C256/K4  | 775.46           | 775.41            | +0.007%                  |
| Dense C512/K2  | 750.44           | 750.08            | +0.048%                  |
| Packed C128/K8 | 782.26           | 782.84            | -0.074%                  |

Latency columns are arithmetic means; throughput changes are geometric means of paired throughput ratios. Dense geometric-mean throughput change is -0.0051%; packed change is -0.0738%. All positive and negative pairs are retained. Throughput is effectively preserved in these measurements; this is not an exact-zero latency change or a universal speedup. Peak allocated memory is unchanged in every matched workload. No dense-FLOPs hardware-floor estimate or new NCU bottleneck claim is made. The lr=0 runs measure training computation, not convergence.

The preceding path-cleanup measurements are preserved in [the previous validation record](https://github.com/ReyJerry/flash-linear-attention/blob/73a9e98a0a1e5155cdd5bde838ff70ab6bc42bbc/docs/phba_validation.md). They use a different baseline and are not the numbers above.

## Normalized L2Warp sensitivity

The four focused cases below pass again in the current source and installed suites. Mutation sensitivity was verified at `73a9e98a`; the model test file and loss implementation are byte-identical in this consolidation, so those fault-injection results are retained evidence.

Four focused cases cover native CE with integer/tensor item counts, fused CE and fused-linear CE. An identity head and one valid CE target on a zero hidden row isolate the L2 contribution: ignored-token hidden gradients and head-weight gradients have no CE component. The test compares them with an independently differentiated FP64 max-logit penalty, both without item-count normalization and after normalization. It additionally checks the reciprocal item-count scaling directly and requires a nonzero signal.

These FP32 loss-head checks use `rtol=1.3e-6, atol=0`; no absolute-error shortcut can hide the normalized term. Existing total-gradient and full-model AMP checks remain unchanged. The focused test is a separate loss-head contract check, not a claim that all mixed-precision fusion variants are bitwise equal.

Two temporary process-local mutations validate sensitivity. With L2Warp disabled only when `num_items_in_batch` is supplied, all four new cases fail while the ten older loss/fusion cases still pass. With forward values preserved but normalized gradients incorrectly rescaled to the unnormalized value, all four new cases also fail (two older normalization cases fail as well). These failures are expected mutation results, separate from the green source/CPU/installed suites; no production mutation is committed.

## Model test scope

PHBA has 41 dedicated model cases, alongside 22 operator, 25 projection and two PHBA registry cases. Fifteen existing shared benchmark cases complete the 105-case source gate. Seven shared configuration/Transformers PHBA cases and an 18-case CPU subset are also executed separately. The GPU equivalence, fusion and interface tests execute real two-layer PHBA models.

| Area                                | Current PHBA coverage                                                                                                                                                                                                 | Boundary                                                                               |
| ---                                 | ---                                                                                                                                                                                                                   | ---                                                                                    |
| Dense/packed equivalence            | Batched dense BF16, ragged BF16 and left-padded FP16 with a one-token sequence; all hidden states, logits, loss and every parameter gradient compared                                                                 | Labels explicitly mask padding and sequence starts before shifted LM loss              |
| Padding and generation              | Right/left padding in BF16 and left padding in FP16 against independent sequences; cache-free greedy generation, including a left-padded batch crossing a chunk boundary                                              | Persistent cache decoding remains unsupported                                          |
| Loss contracts                      | Mean/sum, label smoothing, ignore indices, custom criteria, integer/tensor item counts, unchanged input labels and an independent L2Warp gradient oracle                                                              | Lightweight CPU contract cases use a zero-layer model                                  |
| Loss/fusion combinations            | Native CE, fused CE and fused-linear CE; all four norm/SwiGLU fusion settings in five representative cases; L2Warp, packed inputs and tied weights included                                                           | Representative combinations, not a full Cartesian product                              |
| Forward/decoder interfaces          | Token IDs/embeddings, positional/keyword calls, training/eval, tuple/dict returns, hidden states, no-cache outputs, logits slicing/index selection and legacy selection alias; head/loss and all gradients checked    | Supported cache-free contract only                                                     |
| Initialization                      | Native/fused and affine-free norms, QKV biases, tied weights, pad embeddings, rotary frequencies, meta-to-CPU materialization, NaN-poisoned reinitialization, residual rescale/zero strategies and parameter identity | Ordinary tensors; DTensor initialization remains unvalidated                           |
| Registration and checkpointing      | Save/load, exports, shared embedding accessors/ties/resize/config round-trip, and non-reentrant checkpoint gradient equality                                                                                          | Reentrant checkpointing is not claimed                                                 |
| Unsupported or unvalidated features | Runtime/config cache rejection is checked; existing dimension/plan rejection tests remain                                                                                                                             | AttnRes/persistent caches unsupported; CP/TP and other accelerators outside validation |

Native loss-interface comparisons preserve PyTorch's autocast input precision. Fused-loss comparisons use an explicit FP32 cross-entropy reference, matching their accumulation contract. Loss comparisons use the existing 1e-3 bound and full-model AMP gradient comparisons use 0.006; no existing tolerance was relaxed. The original L2Warp CPU oracle remains, supplemented by the isolated FP64-oracle scaling checks above. Meta initialization runs in a subprocess so the shared operator-test NaN-allocation wrappers cannot pollute Torch's cached device-factory set.

The existing PHBA model and projection test files are unchanged in this consolidation. The operator test changes only its imports; all assertions and tolerances retain identical ASTs. Other models' tests and production modules are untouched. PHBA remains outside the hand-written shared forward/decoder/initialization lists; its dedicated tests now cover the supported equivalent contracts directly. The scripts remain `tests/models/test_modeling_phba.py`, `tests/ops/test_phba.py`, `tests/layers/test_phba_linear.py` and `tests/benchmarks/test_phba_registry.py`.

## Reproduction and scope

See [the PHBA guide](phba.md) for installation, supported APIs, execution paths and benchmark commands. Exact paired reproduction additionally requires the recorded native normalization launch choices and reference hashes; raw experiment artifacts remain outside the source tree. Shared Python edits in the full PR remain limited to central exports, benchmark registration, the split-package builder and its contract test. Other model implementations are unchanged.

Default PHBA parameter names and training mathematics are preserved. Invalid sequence plans are rejected; `elementwise_affine=False` omits normalization scale parameters. Persistent decode caching, context/tensor parallelism, full-graph compilation, other accelerators and convergence remain outside validation. Optimizer selection remains the training framework's responsibility.
