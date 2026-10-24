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
