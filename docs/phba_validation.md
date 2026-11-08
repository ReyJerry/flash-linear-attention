# PHBA validation

PHBA is tested on NVIDIA H20 with PyTorch 2.9.0+cu128, Triton 3.5.0, Transformers 5.6.2 and FlashAttention 2.8.3.post1. The test and installation results below correspond to revision `9a2a5b54a7799c7c6a7647b42bf9e6c04e797af1`.

## Tests

| Suite                    | Result     | Coverage                                                                |
| ------------------------ | ---------- | ----------------------------------------------------------------------- |
| Source                   | 111 passed | 28 operator, 25 projection, 41 model and 17 benchmark cases             |
| Installed split wheels   | 95 passed  | Operator, projection, model, import binding and registration            |
| Shared model integration | 7 passed   | PHBA configuration and Transformers integration                         |
| CPU subset               | 18 passed  | Model configuration, loss contracts and initialization                  |
| H20 smoke                | 8 passed   | Four PHBA forward/backward cases and four FlashAttention contract cases |
| Host package contracts   | 4 passed   | Package ownership, installation order and namespace imports             |

The source, installed and H20 smoke suites have zero skips. Source tests use `FLA_CI_ENV=0 PHBA_REQUIRE_EXACT_GEMM=1`, which enforces numerical assertions and execution of the pinned H20 backend. The smoke selection has 20 deselected cases. The host suite selects the four standalone packaging tests; runtime imports are covered by the installed H20 suite. H100/PyTorch 2.12 CI is pending.

```bash
FLA_CI_ENV=0 PHBA_REQUIRE_EXACT_GEMM=1 python -m pytest -q \
  tests/ops/test_phba.py tests/models/test_modeling_phba.py \
  tests/layers/test_phba_linear.py tests/benchmarks
```

## Numerical checks

Paired checks against `9e0a46499798e080969ecc2f6fe39d5d1aa1f131` cover nine standard operator cases and nine public-path cases. Outputs, final states and supported gradients match byte-for-byte. Coverage includes FP16/BF16, dense and packed inputs, inference, no-history metadata, int64 sorting, unaligned gradients and both history-index widths. Fourteen geometry/budget cases check the saved plans and launch settings.

The public-path cases exercise the 256MiB workspace policy, including the 32K tape fallback, the 4097 one-token prefix fallback and FlashAttention segmentation across 65535+1 batches. The metadata tests cover counts with and without source slots, rejection of slots without counts, and a head stride exceeding signed int32. The large-stride case executes on H20 and requires an 8GiB route allocation plus workspace.

Both references live in `fla/ops/phba/naive.py`: `naive_phba` implements the differentiable FP64 formula, and `reference_a_chunkwise` follows the FP16/BF16 storage conversions. Six paired reference modes compare outputs and all six input gradients. The FP64 formula also passes autograd gradcheck.

Projection tests execute all three screened H20 GEMM geometries and compare outputs and gradients against native bytes. They cover FP32 autocast inputs, aligned BF16 views, unaligned activations and gradients, and separate unaligned GEMM operands. Dispatch assertions check both accelerated calls and native fallbacks. SwiGLU materializes an aligned activation; its unaligned-gradient case uses the native einsum operation.

FlashAttention contract tests check output, contiguous FP32 log-normalizers, packed backward workspace and Q/K/V gradients. Output bounds are 5e-3 for FP16 and 2e-2 for BF16; log-normalizers use 1e-2 and relative maximum gradient error uses 5e-2. Production preflight checks dependencies, device support and the parameter schema.

Across all 15 complete-step pairs, initial parameters, inputs, first-step raw/clipped gradients and every recorded loss/norm match. Final parameters and optimizer states match after 70 steps. The installed 1.264B model audit matches all **267 raw and clipped parameter gradients**, with three exact GEMM sessions active: loss **10.784871101379395**, gradient norm **2.495190382003784**. Producer launches use ordered scheduling and zero spills; register counts are 168 for the baseline and 164 or 168 for PHBA.

## Model coverage

The 41 dedicated model cases exercise real two-layer PHBA models for attention, fusion and interface comparisons. Lightweight loss-contract cases use zero-layer models.

| Area                            | Checks                                                                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense, packed and padded inputs | Hidden states, logits, loss and every parameter gradient; FP16/BF16; one-token sequences; left/right padding                                                    |
| Loss and fusion                 | Mean/sum reduction, label smoothing, ignore indices, custom criteria, item-count scaling, native/fused CE, fused-linear CE, norm/SwiGLU fusion and tied weights |
| Forward and generation          | IDs/embeddings, positional/keyword calls, tuple/dict outputs, hidden states, logits selection and cache-free greedy generation                                  |
| Initialization                  | Affine-free and fused norms, QKV biases, tied weights, pad embeddings, rotary frequencies, meta-to-CPU materialization and residual initialization              |
| Integration                     | Save/load, exports, configuration round-trip, embedding accessors/resize and non-reentrant checkpoint gradients                                                 |

Loss-interface comparisons use a 1e-3 bound and full-model AMP gradient comparisons use 0.006. Native CE follows PyTorch autocast input precision; fused-loss cases use an explicit FP32 CE reference. Meta initialization runs in a subprocess to isolate Torch's cached device-factory set from the operator tests' NaN-allocation wrappers.

Four normalized-L2Warp cases compare the additional gradient with an independent FP64 max-logit penalty at `rtol=1.3e-6, atol=0`. An identity head and a zero hidden row isolate the L2 contribution, including reciprocal `num_items_in_batch` scaling. At revision `73a9e98a`, disabling the normalized branch or removing its gradient scaling causes all four cases to fail. These tests and the loss implementation are unchanged.

## Split-package installation

The official `scripts/build_packages.py` path builds both wheels and source distributions. The release checks verify **640 Python/C++/JSON files** against their build source: 453 in `fla-core` and 187 in `flash-linear-attention`.

`fla-core` contains `lt_exact.cpp`, `native_autocast.cpp` and `exact_gemm_algorithms.json`; all three resources are also present in its source distribution. Layers and models belong to `flash-linear-attention`. Installation tests run outside the checkout and verify import ownership, registration, H20 execution and native fallbacks.

```bash
uv run --no-project --python 3.12 --with pytest --with packaging \
  --with setuptools --with wheel python -m pytest --noconftest -q \
  tests/test_split_package_release.py -k "not full_split_import"
```

`--noconftest` isolates host packaging tests from accelerator fixtures.

## Complete-step throughput

Baseline: `9e0a46499798e080969ecc2f6fe39d5d1aa1f131`. Three same-card paired seeds (3407/3408/3409) use alternating execution order and matched native normalization launch choices. Each run has 20 warmup and 50 measured steps.

The model has 1,264,423,936 parameters, 24 layers, hidden width 2048, 16 heads, FFN width 4272, vocabulary 32000 and 128 prefix slots. Inputs are B1 x 8192, with FP32 parameters and BF16 autocast. The compute benchmark uses no activation checkpointing, native fused AdamW, lr=0, betas 0.9/0.95, clip=1 and weight decay 0.1 excluding 1D parameters. Packed lengths are 2048/3072/3072; sequence-boundary labels are masked before shifted LM loss.

Timing includes zero-grad, forward/loss, backward, clipping, optimizer, allocations and dispatch. Inputs and validation hooks are prepared outside timed steps. Values below are arithmetic means of the three runs. Step time and peak allocated/reserved memory refer to PHBA.

| Workload       | Baseline tokens/s | PHBA tokens/s | Step ms | Allocated GiB | Reserved GiB |
| -------------- | ----------------- | ------------- | ------- | ------------- | ------------ |
| Dense C64/K16  | 7,872.60          | 7,864.23      | 1041.68 | 41.294        | 43.855       |
| Dense C128/K8  | 9,640.89          | 9,629.95      | 850.68  | 39.422        | 40.754       |
| Dense C256/K4  | 10,582.27         | 10,559.89     | 775.77  | 38.494        | 39.602       |
| Dense C512/K2  | 10,931.07         | 10,927.47     | 749.67  | 38.013        | 39.266       |
| Packed C128/K8 | 10,460.22         | 10,443.82     | 784.39  | 39.422        | 40.754       |

```bash
python benchmarks/benchmark_phba.py --chunk-size 128 --topk 8 \
  --warmup-steps 20 --steps 50 --optimizer torch.optim:AdamW --output result.json
```

Add `--lengths 2048,3072,3072` for packed inputs. Exact paired reproduction uses the recorded normalization launch choices and reference hashes. The standard operator entry is `python -m benchmarks.ops.run --op chunk_phba --no-base`.

## Validation snapshots

The SHA256 manifests cover tracked non-Markdown `fla` files and the complete-step benchmark script. Digests use `json.dumps(path_to_sha256, sort_keys=True).encode()` with repository-relative paths.

| Snapshot                                         | SHA256                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| Complete-step measurements                       | `c48f2fec053cc03d983b2389dfa0d7f964b4c0a39021e8aab03bee1255cb8395` |
| Source and installed suites, revision `9a2a5b54` | `ad8aa642ccda2097172c506d56d356e079a5cec280f01f26fd33f8a97eff0b91` |
| Formatting revision `83f9b9c9`                   | `7ad406baca29530111d99ef52797c0cba34a3b6c5ae68945c063834e3b03ef8f` |

Runtime Python ASTs are identical across these snapshots. See [the PHBA guide](phba.md) for API constraints, installation and execution paths.
