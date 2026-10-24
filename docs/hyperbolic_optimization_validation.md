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

Raw local XML/JSON outputs and compiler caches are under ignored `profile/hyperbolic-opt/`. Model/kernel performance tables must use `GPU NOT RUN` until actual measurements exist. Existing numerical thresholds and tests are retained. Pre-existing Ruff findings in untouched portions of legacy protected files are distinguished from new-file or changed-line findings rather than hidden by repository-wide formatting.
