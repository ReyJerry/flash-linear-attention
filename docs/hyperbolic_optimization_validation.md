# HyperbolicDeltaNet optimization audit and validation

## Frozen baseline

The audited target is `ReyJerry/flash-linear-attention`, branch `add-hyperbolic-delta-net`, commit `405835cc0a44ab4b82d7984f6f5b33e98210ee9b`. The isolated checkout started clean; a separate detached worktree preserves that exact baseline. No pre-existing work was reset or overwritten, and no changes target `main`.

The audit read `AGENTS.md`, `CONTRIBUTING.md`, the repository skill index, design/correctness coverage, dispatch, optimization-loop and its measurement traps, NVIDIA performance, PR readiness, and both HyperbolicDeltaNet model/streaming documents. Related operator, layer/model, tests and all three hyperbolic benchmarks were inspected. GitHub searches for `hyperbolic` returned no matching issues/PRs in the fork or upstream during this audit; recent target-branch commits were also checked. Creating the precision RFC revealed that Issues are disabled on the fork (HTTP 410), so an empty search is not evidence of an approved precision design.

## Existing implementation and dependencies

| Area | Actual route and existing work | Remaining candidate |
| --- | --- | --- |
| Unprotected | Layer/model `tensor_core=True` -> `chunk_hyperbolic_delta_rule` -> `_apply_hyperbolic_delta_rule` -> `HyperbolicDeltaRuleFunction` -> `tensor_core_kernels.py`; analytic WY VJP, FP32 canonical states, packed sequences and C=16/32 already exist. | Shared chunk preparation and parallel readout/parameter gradients; retain forward/reverse state dependencies. |
| Precision | `_mm` uses TF32x3 unless legacy FAST and a matching low-precision Query dtype select FP16/BF16. The layer can retain FP32 Query under autocast. | Per-GEMM lowering needs an RFC and additional numerical eligibility rules. |
| Protected projection | `_ProtectedProjection` already shares A inverse; fused head-wide kernels share it inside each CTA. Value-parallel forward and backward still rebuild it per Value. | Share the unchanged inverse in Value-parallel scheduling. |
| Protected state/bank | `auto` uses Value-parallel at V>=8; each Python chunk invokes recurrence and then continuous bank update. Score/key evolution is shared; realized original-Key readouts drive Value-bank updates. | Precompute shared key/score gates and bank boundaries; scan Value/state trajectories independently with full coupled VJP. |
| Memory/checkpoint | Unused TC backward scratch has already been removed. Protected inner checkpoint is configurable; streaming preserves partial prefixes and globally aligned chunks. | New schedules need explicit cache lifetime and bounded gradient workspace models; do not repeat the already-landed buffer/checkpoint cleanups. |

Readout-Q and K>128 keep the existing non-TC chunk fallback. Short decode keeps its recurrent dispatch. Protected packed input is not silently added to an implementation that only accepts dense input. New scheduling options default off until CUDA validation and timing establish their supported domain.

## Development environment and baseline results

The local host has 9 reported CPUs, Python 3.12.14, PyTorch 2.8.0+cpu, Triton 3.4.0, Transformers 5.20.2 and pytest 9.1.1 in an isolated virtual environment. `torch.cuda.is_available()` is false, CUDA device count is zero, and `nvidia-smi` is unavailable. Connected remote devices were offline. No GPU correctness, profiler run, latency or training speedup is claimed.

The fixed baseline commands actually executed were:

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_reference.py tests/ops/test_hyperbolic_delta_rule_host.py
# 173 passed; 0 failed, 0 skipped.

FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false tests/layers/test_hyperbolic_streaming.py tests/layers/test_hyperbolic_deltanet.py tests/models/test_hyperbolic_deltanet.py
# 2 passed, 8 skipped (CUDA unavailable); 1 CPU-platform warning.
```

The host tests check launch contracts with mocked kernels. The reference tests execute differentiable CPU formulas. Neither establishes GPU correctness. Triton interpreter tests, when explicitly selected, execute actual Triton source on CPU; offline `sm_90` compilation produces device binaries without launching them. Those two additional checks also do not establish hardware correctness or speed.

## Independent gates and promotion

Scheduling candidates require mathematical forward/VJP equivalence, actual CPU regressions, explicit old/new dispatch, unchanged reference tolerances, CUDA test definitions, a reproducible benchmark and diff/static review before branch submission. They remain experimental when CUDA execution is unavailable. Full production promotion additionally needs real CUDA forward/state/all-gradient tests and same-hardware forward, backward and complete-step measurements; CPU pass counts and theoretical FLOPs cannot substitute for them.

Detailed equations, memory/performance models, supported domains and validation commands are maintained with each candidate:

- [Shared Bank-Key inverse](hyperbolic_shared_inverse.md).
- [Unprotected staged forward/backward](hyperbolic_staged.md).
- [Protected shared Key/Score and Value trajectories](hyperbolic_protected_staged.md).
- [Per-GEMM precision RFC draft and CPU error study](hyperbolic_precision.md): production policy blocked by the required RFC issue being unavailable, not an implemented optimization.

Raw local XML/JSON outputs and compiler caches are under ignored `profile/hyperbolic-opt/`. Model/kernel performance tables must use `GPU NOT RUN` until actual measurements exist. Existing numerical thresholds and tests are retained. Pre-existing Ruff findings in untouched portions of legacy protected files are distinguished from new-file or changed-line findings rather than hidden by repository-wide formatting.

## Submitted evidence: B2 and A2

B2 was independently published as [`7e2a09e52590aede18ab08735129ef29aa8ac834`](https://github.com/ReyJerry/flash-linear-attention/commit/7e2a09e52590aede18ab08735129ef29aa8ac834). The submission gate ran 210 CPU algebra/reference/host tests (zero failures/skips), 3 separately selected actual-source Triton CPU interpreter cases (zero failures/skips), and 10 benchmark host tests (zero failures/skips). Fifteen offline CUDA compilation cases covered three storage dtypes and the shared preparation plus old/shared forward/backward variants. Thirteen CUDA test cases were defined but not executed. The benchmark's no-CUDA invocation returned exit code 2 and wrote `GPU NOT RUN`, rather than emitting timings.

A2's independent evidence-only submission contains a full GEMM inventory, separate-edge analytic VJP, 103 passing CPU mathematical tests (zero failures/skips; final run 1.52 s), a deterministic feasible-domain counterexample, and an error-study script. The 16-row operand-rounding study rejected three BF16 candidates under the unchanged budgets. Its ordinary completion exit code means the study finished; `--require-all-pass` returns nonzero for rejected candidates. No production selective-precision kernel, GPU selective-precision test, or performance result is claimed. The existing global FAST route remains unchanged. This is a recorded blocker and numerical finding, not completion of A2.

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false tests/ops/test_hyperbolic_delta_rule_precision.py
# 103 passed, 0 failed, 0 skipped.
python scripts/analyze_hyperbolic_precision.py --json profile/hyperbolic-opt/precision-cpu.json
```

## B1 submission gate

The protected trajectory implementation retains the previous default and is requested only by `FLA_HYPERBOLIC_PROTECTED_STAGED=1` on an eligible Value-parallel route. The independent mathematical gate preceded kernel implementation. The final combined CPU gate ran **233 passed, 0 failed, 0 skipped, 63 deselected** in 12.87 s: 167 original reference tests, 6 original host tests, 37 B2 mathematical tests and 23 B1 mathematical/host tests. There was one expected CPU-platform warning. Separately, **11 actual-source Triton CPU interpreter tests passed**, including final-Bank-only loss paths and checkpoint replay; these are not CUDA tests.

The protected/common benchmark host suite passed **18 tests**, zero failures/skips, in 12.42 s. It checks actual-path attestation, error gates, deterministic cotangents, backward-only timing, CLI validation, no-CUDA reporting and checkpoint-policy parity for the legacy decomposed comparison. Backward latency excludes forward-graph preparation; its separately measured memory peak includes the graph and its preparation, as explicitly recorded in JSON. Per-operator timings do not include an optimizer or full model.

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false tests/ops/test_hyperbolic_delta_rule_reference.py tests/ops/test_hyperbolic_delta_rule_host.py tests/ops/test_hyperbolic_delta_rule_shared_inverse.py tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k 'not cuda and not interpreter'
TRITON_INTERPRET=1 FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k interpreter
FLA_CI_ENV=0 python -m pytest --noconftest -q tests/benchmarks/test_hyperbolic_benchmarks.py
```

The reusable offline compiler was executed for all seven entry kernels (six new kernels plus the reused Gram preparation) in three representative configurations: BF16 Query with FP32 Key/Value at K=V128/R4/C16; FP32 at K=V128/R16/C32; FP16 storage at K7/V3/R3/C16. All 21 compilation cases passed. Additional independent compiler cases and same-flags assembler resource comparisons are recorded in the B1 document. The public CUDA matrix is defined and collected, but **GPU NOT RUN** remains the hardware status.

```bash
PYTHONPATH=. FLA_CI_ENV=0 python scripts/check_hyperbolic_compilation.py --component protected-staged --dtype bf16 --layer-inputs --resources --output profile/hyperbolic-opt/b1-compile.json
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k cuda
python benchmarks/benchmark_protected_hyperbolic_delta_rule.py --batch 1 --seq-len 16384 --heads 16 --key-dim 128 --value-dim 128 --rank 4 --dtype bf16 --layer-inputs --compare-trajectories --no-checkpoint --modes fwd bwd fwd_bwd --warmup 20 --repeats 50 --output profile/hyperbolic-opt/protected-16k.json
```

The benchmark was actually invoked on this CPU host and exited 2 with `GPU NOT RUN: CUDA device unavailable`; it wrote no correctness or performance result. GPU commands above are instructions for the next hardware gate, not evidence that it ran. Profile the same candidate and baseline on the same hardware before interpreting launch reduction or register counts as acceleration.

B1 was published independently as [`d4320de6bbca107b91b585ed90fcc6c74c41f56d`](https://github.com/ReyJerry/flash-linear-attention/commit/d4320de6bbca107b91b585ed90fcc6c74c41f56d). A2's evidence-only commit is [`5838aac1c2280586c6b5addb751ed3b28948ce63`](https://github.com/ReyJerry/flash-linear-attention/commit/5838aac1c2280586c6b5addb751ed3b28948ce63).

## A1 submission gate

The selected implementation has three forward launches and four backward launches. It shares chunk Gram/readout preparation, retains only the necessary forward/reverse state dependencies, and performs local parameter VJPs across chunks. A full-Key local-gradient prototype was rejected after offline compilation exposed large spills. The final local VJP separates Value gradients from Q/Key gradients, reuses the dead solve-result buffer for `da`, and writes disjoint Key tiles without expanded whole-sequence `dq/dk` partials. The squared Key norm keeps the original full-Key reduction to preserve the forward/backward feasibility decision; only `KS` is contracted in smaller tiles. The unchanged numerical thresholds caught and guard the near-boundary case.

Actual final gates:

- **250 CPU mathematical/reference/host tests passed**, zero failures/skips: 173 existing reference/host cases plus 77 A1 algebra tests. These do not establish CUDA correctness.
- **29 actual-source Triton CPU interpreter/host-routing cases passed**, zero failures/skips, in 37.89 s, after the final norm fix. These include two near-boundary old/new kernel comparisons and explicitly labeled logical-device mocks for public dispatch. CPU tensors remained on CPU throughout.
- **11 TC benchmark host tests passed**, including strict/FAST/fallback attestation, all seven staged entry kernels, no-CUDA status and the unchanged fixed FP32 final-state tolerance for old chunk/recurrent comparisons.
- **59 CUDA cases collected, GPU NOT RUN**, including mixed storage, packed/empty sequences, 16K length, supported gradients, cache segments and fallbacks.
- **28 final staged offline compilation cases passed**: seven entry kernels in FP32 C16/C32 with int64 offsets and fixed curvature; BF16 Query with FP32 Key/Value and int32 offsets with curvature gradient; and FP16 asymmetric K17/V19, C32 with int32 offsets and curvature gradient. Four additional unchanged-TC compilation/resource cases establish the same-environment baseline.

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false tests/ops/test_hyperbolic_delta_rule_reference.py tests/ops/test_hyperbolic_delta_rule_host.py tests/ops/test_hyperbolic_delta_rule_staged.py -k 'not cuda and not interpreter'
TRITON_INTERPRET=1 FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_staged.py -k interpreter
FLA_CI_ENV=0 python -m pytest --noconftest -q tests/benchmarks/test_hyperbolic_tensor_core_benchmark.py
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule_staged.py -k cuda
python benchmarks/benchmark_hyperbolic_tensor_core.py --batch 1 --seq-len 16384 --heads 16 --key-dim 128 --value-dim 128 --dtype fp32 --layer-inputs --modes fwd bwd fwd_bwd --warmup 20 --repeats 50 --output profile/hyperbolic-opt/unprotected-16k.json
```

The benchmark's CPU-host invocation was actually executed and returned exit 2 with `GPU NOT RUN: CUDA device unavailable`, writing `profile/hyperbolic-opt/a1-gpu-status-final.json` without timings. The CUDA command and timing command still require hardware execution. The final offline audit explicitly records high register counts and remaining state-scan spills; the absence of huge local-VJP spills is not a GPU acceleration result. The staged path remains opt-in and the original TC/recurrent/readout/large-Key fallbacks remain available.

A1 was independently published as [`69d716145989323cead6d292a0a90a126c7085e5`](https://github.com/ReyJerry/flash-linear-attention/commit/69d716145989323cead6d292a0a90a126c7085e5).

## Integrated CPU and benchmark gate

After all three scheduling implementations and the complete-training runner were frozen, the combined CPU invocation passed **468 tests, 0 failed, 0 skipped, 151 deselected**, in 33.38 s, with one expected CPU-platform warning. The XML report is `profile/hyperbolic-opt/final-cpu-and-benchmarks.xml`. The count is a single combined run, not a sum of repeatedly executed gates:

| Test file | Passed | What actually executes |
| --- | ---: | --- |
| `tests/ops/test_hyperbolic_delta_rule_reference.py` | 167 | Existing differentiable CPU reference |
| `tests/ops/test_hyperbolic_delta_rule_host.py` | 6 | Existing host dispatch/launch contracts |
| `tests/ops/test_hyperbolic_delta_rule_shared_inverse.py` | 37 | B2 independent math/VJP/finite differences |
| `tests/ops/test_hyperbolic_delta_rule_protected_staged.py` | 23 | B1 independent math/VJP and host contracts |
| `tests/ops/test_hyperbolic_delta_rule_staged.py` | 77 | A1 independent math/VJP/gradcheck |
| `tests/ops/test_hyperbolic_delta_rule_precision.py` | 103 | A2 math, cast contracts and rejection cases |
| `tests/benchmarks/test_hyperbolic_benchmarks.py` | 18 | Shared/protected benchmark host behavior |
| `tests/benchmarks/test_hyperbolic_tensor_core_benchmark.py` | 11 | Unprotected benchmark host behavior |
| `tests/benchmarks/test_hyperbolic_training_benchmark.py` | 26 | Full-model benchmark host behavior and real CPU model construction |

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false \
  --junitxml=profile/hyperbolic-opt/final-cpu-and-benchmarks.xml \
  tests/ops/test_hyperbolic_delta_rule_reference.py \
  tests/ops/test_hyperbolic_delta_rule_host.py \
  tests/ops/test_hyperbolic_delta_rule_shared_inverse.py \
  tests/ops/test_hyperbolic_delta_rule_protected_staged.py \
  tests/ops/test_hyperbolic_delta_rule_staged.py \
  tests/ops/test_hyperbolic_delta_rule_precision.py \
  tests/benchmarks/test_hyperbolic_benchmarks.py \
  tests/benchmarks/test_hyperbolic_tensor_core_benchmark.py \
  tests/benchmarks/test_hyperbolic_training_benchmark.py \
  -k 'not cuda and not interpreter'
```

The existing layer/streaming/model suite was also rerun after integration: **2 passed, 8 skipped, 0 failed**, in 8.13 s. The eight skips require CUDA; they are not passing layer/model forward/backward checks. This matches the frozen CPU baseline. The report is `profile/hyperbolic-opt/final-layer-model.xml`.

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false \
  --junitxml=profile/hyperbolic-opt/final-layer-model.xml \
  tests/layers/test_hyperbolic_streaming.py \
  tests/layers/test_hyperbolic_deltanet.py \
  tests/models/test_hyperbolic_deltanet.py
```

The three new scheduling test files collected **108 CUDA cases** (13 B2, 36 B1, 59 A1), without executing them. No GPU tests are counted among the 468 CPU passes. Actual CUDA execution remains a release gate.

A final single-process integration run of the actual-source Triton CPU interpreter suites passed **43 tests, 0 failed, 0 skipped, 245 deselected**, in 130.48 s: 3 B2, 11 B1 and 29 A1 cases. Its report is `profile/hyperbolic-opt/final-interpreter.xml`. This checks the integrated kernels and host routing on CPU, not CUDA execution. The 460 warnings were retained: 458 NumPy scalar-conversion deprecations from Triton's interpreter, one CPU-platform warning, and one `exp` overflow warning in the low-temperature legacy-sigmoid case. That case passed old/new output and all-gradient parity and explicit finite-gradient assertions; no warning or tolerance was suppressed to obtain a pass.

```bash
TRITON_INTERPRET=1 FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q -o log_cli=false \
  --junitxml=profile/hyperbolic-opt/final-interpreter.xml \
  tests/ops/test_hyperbolic_delta_rule_shared_inverse.py \
  tests/ops/test_hyperbolic_delta_rule_protected_staged.py \
  tests/ops/test_hyperbolic_delta_rule_staged.py -k interpreter
```

## Complete-training comparison

`benchmarks/benchmark_hyperbolic_training.py` adds the requested 24-layer comparison, separately from the operator microbenchmarks. The [complete-training guide](hyperbolic_training_benchmark.md) documents all commands, method selection, checkpoint differences and JSON fields. Default parameters are FP32 with BF16 autocast, AdamW, outer nonreentrant layer checkpoint enabled, and protected inner checkpoint disabled. Methods are run sequentially with matching initial-weight hashes within each HDN group. GDN is a performance comparison, not an output or gradient oracle.

The runner first requires a small-model loss and all-parameter-gradient parity gate, then records the full model's actual path, continuous loss/gradient/final-weight finite checks, forward/backward/optimizer times, total synchronized CUDA/wall times, raw distributions and peak allocated/reserved memory. It does not label full-size finite checks as full-size trajectory equivalence. The default is 10 warmup, 30 measured and 10 additional continuous stability steps. Compile, tracing, health checks, data preparation and `zero_grad` remain outside the reported timing scope. That scope is explicit in the JSON.

The CPU host tests execute real small model constructors, verify FP32 parameters, same-group weight hashes and actual nonreentrant checkpoint hooks, while replacing only `.cuda()` with an identity for that construction check. They do not run model forward/backward on CPU or CUDA. The runner's no-CUDA `--check-only` invocation was actually executed and returned exit 2, writing `profile/hyperbolic-opt/training-final-no-gpu.json` with `GPU NOT RUN` and no correctness/timing result.

| Optimization | Implementation status | Hardware status | Publication |
| --- | --- | --- | --- |
| A1 staged forward/backward | Implemented, opt-in; math/CPU/interpreter and offline compilation checked | GPU NOT RUN | `69d71614` |
| A2 per-GEMM precision | Production change blocked by required RFC; draft and CPU evidence only | No selective-precision GPU implementation or measurement | `5838aac1` is evidence only |
| B1 Key/Score and Value trajectories | Implemented, opt-in; full coupled VJP retained | GPU NOT RUN | `d4320de6` |
| B2 shared Bank-Key inverse | Implemented only where real Value-CTA duplication remained | GPU NOT RUN | `7e2a09e5` |

All new scheduling flags default off. Actual forward/backward/complete-training speedup and peak-memory changes are unmeasured. No candidate is promoted to formal GPU training based on this CPU-only validation. The remaining external gates are a usable/approved precision RFC and actual supported NVIDIA hardware for correctness, resource profiling, same-environment timing and longer training validation.

## Final static review

The final lint pass also used the repository-pinned **Ruff 0.14.10**, installed in a separate scratch tool directory; the development Ruff 0.17.2 was not treated as a substitute for that pin. All added Python files and changed Python files other than the two legacy protected files passed the pinned check. The complete legacy files retained eight pre-existing findings: one `__all__` ordering finding in `protected_kernels.py`, and one unused import plus six import-format findings in `protected_triton.py`. The same pinned tool reproduced those findings on the frozen baseline, which had nine across those two files; an edited import block already removed the ninth. No new finding was ignored and no broad legacy formatting was included.

Python compilation passed for all changed/new Python files. New-file autopep8 2.3.2 checks and `git diff --check` were clean. Existing autopep8 changes suggested in untouched portions of `fused_recurrent.py` were reproduced on the baseline and left outside this scheduling change. These results do not claim that the entire repository's lint or GPU CI passed.
