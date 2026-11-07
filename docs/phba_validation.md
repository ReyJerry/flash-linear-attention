# PHBA validation

The deterministic CUDA implementation is checked for numerical preservation, supported model APIs, packaging and complete-step throughput. The comparison baseline is `6fcadf9242c303dac5dcf73310597d625443c472` on the same official FLA base. Both roles use NVIDIA H20, PyTorch 2.9.0+cu128, Triton 3.5.0, Transformers 5.6.2 and FlashAttention 2.8.3.post1.

## Numerical coverage

The PHBA operator/model/projection suite passes 26 tests with no skips. Coverage includes FP16/BF16 dense and packed outputs and all six input gradients, tail blocks, non-power-of-two dimensions/slots, final states, FP64 formula gradcheck, unsupported inputs, registration/save-load, checkpoint gradients, padding, cache-free generation and screened/native projection paths. Independent routing and canonical metadata comparisons cover both no-history and history-with-tail cases. References and existing tolerances are retained.

The 1,264,423,936-parameter model matches all 267 initial parameter tensors, input, loss, gradient norm, and raw/clipped gradient bytes against the validated reference. Loss is `10.784871101379395` and gradient norm is `2.495190382003784`. All three screened H20 GEMM geometries execute. Every complete-step pair also matches loss/norm across measured steps and all optimizer-state hashes after 70 steps.

The wheel passes 27 tests outside the source checkout, including installed-import binding. Its 653 packaged Python/C++/JSON files match the tested source. It includes the cuBLASLt helper source and validated algorithm descriptors, but no optimizer, training extension, prebuilt shared library or candidate-enumeration helper. A separate tiny-model profiling run verifies that stage ranges are recorded by the external profiling command.

All 827 pre-existing non-PHBA Python files remain byte-identical. The 35 Triton kernel computation ASTs and all selected GEMM descriptors are unchanged. Runtime helper removal and relocation are additionally covered by the byte gates above. The manifest of 33 PHBA implementation files has SHA256 `aed1966c9176347b2e969a40920e2cdcbd7e5f6ddc437f3ff604a705b8cce02e`, calculated as SHA256 of `json.dumps(path_to_sha256, sort_keys=True).encode()` using repository-relative Python/C++/JSON paths. Header, pre-commit and whitespace checks pass. The only numerical-suite warning is PyTorch's TF32 API deprecation.

## Complete-step measurements

Each case uses three same-device paired seeds, 3407/3408/3409, with alternating baseline/candidate order. The recipe is 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000, 128 prefix slots, B1 x 8192, FP32 parameters, BF16 autocast, no activation checkpointing, clipping at 1, fused native PyTorch AdamW, betas 0.9/0.95, and weight decay 0.1 excluding one-dimensional parameters. Native normalization/activation/rotary launch configurations are matched. Each run has 20 warmup and 50 measured steps with lr=0.

Packed lengths are 2048/3072/3072, totaling 8192 tokens; sequence-boundary labels are masked before shifted language-model loss. Synchronized wall time includes zero-grad, forward/loss, backward, clipping, optimizer, allocations and dispatch. Input creation and numerical audit hooks are outside measured steps. Peak allocated memory includes warmup after the first two initialization steps.

| Workload       | Baseline ms/step | Current ms/step | Paired throughput gain | Peak allocated GiB, baseline / current |
| -------------- | ---------------: | --------------: | ---------------------: | -------------------------------------: |
| Dense C64/K16  |          1040.40 |         1040.48 |                -0.008% |                  41.294661 / 41.294661 |
| Dense C128/K8  |           849.98 |          850.71 |                -0.086% |                  39.423200 / 39.423200 |
| Dense C256/K4  |           772.97 |          773.08 |                -0.015% |                  38.494672 / 38.494672 |
| Dense C512/K2  |           749.75 |          749.06 |                +0.091% |                  38.012891 / 38.012891 |
| Packed C128/K8 |           782.90 |          782.84 |                +0.008% |                  39.423263 / 39.423263 |

Latency columns are arithmetic means of the runs; gains are geometric means of paired throughput ratios. Dense rows use three pairs; the packed row pools six pairs across both role orders for each seed. Across the 12 dense pairs the geometric-mean gain is -0.004%, with a minimum individual pair of -0.391%. The first three packed pairs had a gain of -0.047%; a confirmation with reversed role order measured +0.063%. Both rounds are pooled. The packed geometric-mean gain is +0.008%, with a minimum individual pair of -0.058%. Small differences must be interpreted as measurements with run-to-run variation, not a universal speedup guarantee. No dense-FLOPs proxy or hardware-floor inference is used.

An additional four-case, three-seed comparison keeps the same external AdamW312 training extension on both sides. Its geometric-mean gain is -0.009%, with a minimum pair of -0.084%. That optimizer is not part of FLA or this change. Its separately archived seven-step, nonzero-learning-rate gate matches all 267 parameters, raw/clipped gradients, moments and step counters against native AdamW; each original step increments once. This auxiliary comparison isolates model cleanup from training-side optimizer selection.

## Reproduction and limits

Use the commands in [the PHBA guide](phba.md); add `--lengths 2048,3072,3072` for packed input. Ordinary benchmark invocations may autotune differently; reproducing the exact paired audit also requires the frozen launch choices and reference hashes from the experiment records. Raw per-step records and profiler traces stay outside the source tree.

Persistent decode caching, context/tensor parallelism, full-graph compilation, other accelerators and training convergence are outside this validation. The lr=0 runs measure training computation; the separate nonzero-learning-rate gate verifies optimizer updates. Import `fla.models.phba` explicitly for Transformers auto registration. Existing model/layer exports and benchmark registries remain unchanged.
