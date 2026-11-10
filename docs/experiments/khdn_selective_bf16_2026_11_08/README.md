# Selective BF16 KHDN: numerical repair and chunk selection

The experimental C64/C128 candidate now passes the original twelve operator
fixtures at both chunk lengths, the separate historical eight-case suite, and
the recorded legal near-boundary input. C64 wins both short and long full-model
screens. It is the preferred candidate for further qualification.

This release archives experimental code and evidence. It does not replace the
public implementation or promote a model default. Full model cache/packed
regressions, finite training, formal timing and current-source distributed
qualification remain open. Earlier failures remain in their historical archives.

## Candidate and precision

The source overlay applies to commit `cd0da267569732c3aae0462b7bd9e90a975f5516`.
The same selected host/kernel bytes are used in operator qualification, both
full-model endpoint screens and the C64 initial-gradient screen.

The mass-system rewrite uses the exact inverse of the scalar lower triangular
system: diagonal `1/n_i`, strictly lower entries `beta_i/n_i`. It incorporates
that inverse into the group feedback system and replaces dense mass-system
source adjoints with scalar scans. It preserves the canonical recurrence,
inclusive readout, Key-channel decay, g32 and all seven source derivatives.
An independent CPU dense-solve/autograd derivation is included in the source
archive. Its FP64 arithmetic is an explicitly selected, untimed oracle.

Selected precision is more conservative than the earlier unqualified fast
candidate:

- Internal preparation, state snapshots and gradient work buffers use FP32.
- Forward contractions use single BF16 products, except C128 inverse
  off-diagonal preparation, which uses three BF16 products.
- State/output backward and the state-tail shared coefficient VJP use three
  BF16 products. Other selected preparation/local adjoints use two products,
  retaining the high and low parts of the gradient operand.
- Accumulation/control remain FP32. There is no FP64 candidate training,
  hidden high-precision replay, or six-product compensation.
- The redundant scaled-query copy is removed. A prior isolated comparison was
  within the budgets but was **not bitwise identical**; the combined candidate
  was qualified again on the original fixtures.

The archived direct-state-basis and optional curvature-buffer experiments are
not part of this selected source. A compiler-pipeline-sensitive repeat-backward
defect in a prior preparation-storage variant was isolated; the selected
candidate uses the verified one-stage launch policy. This is not a general
claim about the compiler.

## Numerical evidence

All listed budgets are unchanged. Operator output/state budgets are
`atol=0.002, rtol=0.02, RMS=0.01*max(reference_RMS, atol)`; source-gradient
budgets are `0.005, 0.03, 0.015`, respectively.

| Coverage | Result | Boundary |
| --- | --- | --- |
| Four chunk lengths 16/32/64/128, dense/packed, NaN-poisoned allocations | 8 cases pass; memcheck reports zero errors | Bounded operator cases |
| Original seed42/T129 output/state/joint fixtures, FP32/BF16, dense/packed | 24 executions pass at C64/C128; repeat backward exact | Original sources and cotangents reused |
| Explicit trainable c=0 | 8 cases pass | Curvature derivative retained |
| Exact legal near-boundary raw input | 2 cases pass | One counterexample repaired, not a full-domain proof |
| Historical eight-case strong-gate/nonunit-Key/Value-width suite | 16 executions pass | Original generator and budgets retained |
| Lengths 63/64/65/127/128/129, dense/packed, FP32/BF16 | 48 executions pass | C64/C128, nonzero initial state, beta endpoints |
| Source aliases and nonleaf input graphs | 8 executions pass, repeat backward and cross-stream replay exact | Same supported candidate, no kernel change |

The historical beta-gradient RMS case is separate from previously approved
scalar-curvature exceptions. Its new C64/C128 RMS values are
`3.49278294e-5` and `2.57656607e-5`, below the unchanged `1.88443750e-4` limit.
These results do not retroactively qualify the older failing source.

## Full-model endpoint screens

H20/SM90, Torch 2.9.0+cu128, Triton 3.5.0; compiler SHA256
`1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`.
The model has 24 layers, hidden2048, H16, K=V128, FFN5184, vocabulary32000,
g32, ShortConv, checkpointing, BF16 autocast and plain CE. KHDN and the explicit
output-projection-aligned KDA each have 1,412,912,512 parameters. Parameters and
AdamW state are FP32.

Each arm uses two warmup and three steady samples per directly measured F+B
and complete step, followed by a separate profiled update. Complete steps
include clipping, optimizer and public checks. Transfers and zero-grad follow
the archived harness protocol outside timing. Initial KHDN weights and input
batches match across arms; endpoint arm order is reversed. Actual launch grids
and KDA TileLang backward calls are asserted before measurement.

| Shape | Arm | Direct F+B ms | Step ms | Tokens/s | Allocated GiB | Reserved GiB | KDA throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B8×T2048 | C64 selected | 2157.355 | 2193.867 | 7468.09 | 27.215 | 31.719 | 79.044% |
| B8×T2048 | C64 inverse-three control | 2179.532 | 2195.121 | 7463.83 | 27.216 | 31.725 | 78.998% |
| B8×T2048 | C128 selected | 2327.050 | 2350.274 | 6971.10 | 28.019 | 32.844 | 73.783% |
| B8×T2048 | Matched KDA | 1710.615 | 1734.112 | 9448.07 | 25.584 | 30.158 | 100% |
| B1×T16384 | C64 selected | 2277.467 | 2299.021 | 7126.51 | 27.196 | 31.732 | 77.026% |
| B1×T16384 | C64 inverse-three control | 2287.463 | 2311.316 | 7088.60 | 27.196 | 31.732 | 76.617% |
| B1×T16384 | C128 selected | 2428.869 | 2458.086 | 6665.35 | 27.999 | 32.721 | 72.042% |
| B1×T16384 | Matched KDA | 1747.129 | 1770.852 | 9252.04 | 25.584 | 30.158 | 100% |

C64 selected has approximately 7.1%/6.9% higher throughput than C128.
The inverse-three C64 control separately passes the original twelve fixtures
and also wins against C128. The selected pair differs in inverse precision;
that joint comparison must not be attributed solely to chunk length.

These are research screens, not three independent trials of at least
10 warmup/30 steady samples. They supersede neither the numerical identity nor
the qualification status of earlier measurements. In particular, the older
faster BF16-buffer candidate was numerically unqualified. Its speed cannot be
reported as the performance of this repaired candidate.

### Current profile

The separate B8×T2048 profiles contain 48 state-forward and 24 state-backward
calls, including checkpoint replay. State forward/backward totals are
73.117/71.140 ms at C64 and 63.215/57.075 ms at C128: the larger chunk really
does shorten the state work. However, C128 has higher inverse preparation,
group system, public preparation VJP and output costs. For example:

| Kernel family | C64 ms | C128 ms |
| --- | ---: | ---: |
| Inverse off-diagonal preparation | 12.835 | 45.788 |
| Group system preparation | 14.816 | 35.933 |
| Common finish products | 19.247 | 37.482 |
| Common A-gradient | 12.759 | 29.451 |
| Cross-interval backward | 28.072 | 41.863 |

These are actual profiled kernel durations, not a sum of independent timing
medians or causal percentages of wall time. The selected C64 profile records
121 scalar reads versus one for matched KDA. CPU synchronization duration
includes waiting for GPU work and cannot be counted as additive avoidable
overhead. New final-source NCU attribution remains incomplete.

## Full-model initial gradient and inference limits

The selected C64 initial-gradient screen captures all 459 parameter gradients
at the same initial weights, input batch and RNG as the saved independent
strict reference. No optimizer update is performed.

| Metric | Value |
| --- | ---: |
| Gradient cosine | 0.9923959902 |
| Gradient norm ratio | 0.9996165341 |
| Gradient relative L2 | 0.1232977492 |
| Same-batch initial PPL difference | +0.01134937% |
| Old elementwise budget violations | 15,325 |

The approved model initial-gradient gate passes. Relative L2 is about 12.33%,
not 0.04%; the old elementwise failures remain. The PPL number is an initial
same-batch comparison, not unseen-validation or continuous-training quality.
MAIN verified receipt/result hashes and reaggregated all per-tensor records;
the large gradient tensors remain in persistent storage and have not yet
been independently reread for this release.

Original C16 model regression diagnostics remain unresolved: baseline,
selected core and disabling BF16 reduced-precision reduction each have
3 passed/5 failed. An inference-only all-FP32 Linear control passes all eight
tests, but it is not deployed or timed. Keeping BF16 Linear operands with
FP32 outputs gives 4/6 inference cases passing with the accurate core and
2/6 with the selected core. These controls support further investigation of
rounding amplification; they do not justify silently replacing model
inference with FP32 or recurrent execution. Protected test-source copies and
their full source archives are excluded from this release.

## Files and reproduction

- `evidence.json.gz`: source-bound operator metrics, timing observations,
  profile events, model configurations, gradient records and terminal receipts.
- `operator-sources.tar.gz` and `model-sources.tar.gz`: immutable selected
  candidate, measured scripts, required source overlays and the independent
  mass-system algebra check.
- `source-binding.json`: baseline, per-file hashes and experiment identities.
- `summarize.py`: CPU artifact/hash, numerical-record and timing verification.

Run `python docs/experiments/khdn_selective_bf16_2026_11_08/summarize.py` from
the repository. GPU drivers retain their explicit runtime/cache/reference
paths. Source overlays apply to the named baseline; this package is not a
portable one-command GPU runner. Original source/cotangent fixtures and large
gradient tensors remain in the identified persistent experiments, with hashes
in the evidence. The self-contained boundary and algebra checks are also
included. Do not treat CPU archive verification as another GPU execution.

Remaining work includes public optimized C64 integration and native CLI trace,
model cache/packed repair, selected-source finite training and unseen quality,
four-shape formal timing, dynamic profiling, and 1/8/16-GPU gradients, updates,
recovery and strong/weak scaling. Y-carry is not qualified by this no-carry
candidate. The PR and engineering goal remain incomplete.
