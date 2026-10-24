# Shared protection-bank inverse

This experimental scheduling option applies to the existing Value-parallel protected path. Set `FLA_HYPERBOLIC_SHARED_INVERSE=1` to compute the key-only inverse once per chunk and reuse it in forward and backward. The unset/default value retains the existing implementation. CUDA correctness and same-hardware timing are required before promoting this option.

## Baseline and actual duplication

The baseline is `405835cc0a44ab4b82d7984f6f5b33e98210ee9b`. `_ProtectedProjection` already calls `shared_key_gram_fwd_kernel`, and the head-wide fused forward/backward share an inverse within their CTA. `_ValueParallelChunk`, selected by `auto` when V >= 8, instead computes the same inverse independently in every Value CTA in both forward and backward. This option addresses only that remaining duplication.

For active bank keys P and one channel's bank values y, define the padded, regularized key Gram A exactly as in the existing kernel: active entries are PPᵀ + epsilon I, and inactive rows/columns form an identity block. The existing `_invert_small` routine computes A⁻¹ in FP32 with the same IEEE key product and pivot handling. No inverse algorithm or precision is replaced.

For each Value channel:

\[
u=A^{-1}y,\quad d=1-cy^Tu,\quad G^{-1}=A^{-1}+\frac{c}{d}uu^T.
\]

The existing feasibility branch is unchanged: when d <= epsilon, projection is rejected. Given a token key k and value v, alpha = G⁻¹(Pk-cyv), k' = k-Pᵀalpha and v' = v-yᵀalpha. The projected norm check and its identity fallback remain unchanged. A and its inverse depend only on the fixed chunk-start bank keys and occupancy mask, so sharing them cannot change any exact-arithmetic token, state, or bank update. A new inverse must be computed after the bank changes; no cache is reused across bank versions.

## Complete projection VJP

Let gk and gv be cotangents of k' and v'. On an accepted projection, set

\[
g_\alpha=-P g_k-y g_v,\qquad z=G^{-T}g_\alpha.
\]

Then

\[
\begin{aligned}
\bar k &= g_k+P^Tz, & \bar v &= g_v-cy^Tz,\\
\bar P &= -\alpha g_k^T+z(k-P^T\alpha)^T-\alpha(P^Tz)^T,\\
\bar y &= -\alpha g_v-czv+cz(y^T\alpha)+c\alpha(y^Tz),\\
\bar c &= (y^Tz)(y^T\alpha-v).
\end{aligned}
\]

Rejected projections use alpha=z=0, giving the identity VJP. Mask inactive bank rows and sum key-bank contributions across tokens and Value channels. These are the existing analytic kernel equations, including the inverse dependence on P, y and c. The shared inverse is an internal numerical cache inside the custom autograd Function: returning a separate gradient for it would double-count that dependence. Continuous bank updates and their score gradients continue through the existing differentiable bank-update Function. Discrete occupancy predicates retain their existing piecewise semantics.

The recurrent state and inclusive Query/original-Key readouts are unchanged. Consequently, the existing state/Query/beta/curvature adjoints compose with this projection VJP without modification. Sharing remains valid for beta 0/1, nonzero canonical initial state and c=0 at the projection level; the public protected reference's positive-curvature restriction is not expanded. Empty calls never launch a chunk and preserve the incoming state/bank. Protected streaming still controls the globally aligned chunk-start bank.

## Performance and memory model

For N chunks, the old Value-parallel route performs approximately 2*N*B*H*V key Grams and inversions over a forward/backward pair. The option performs N*B*H, saving the per-Value duplication and backward reconstruction. Per-channel Sherman-Morrison and projection work remain. The Gram costs O(R²K), the unchanged inverse O(BR³), and BR is at least 16 in the current implementation. There is one extra small launch per chunk in forward, and no extra backward launch.

The cache is 4*B*H*R² bytes per chunk, not per Value. Across 16,384 tokens, H=16, C=16 and R=4 this is 1 MiB per layer; at R=16 it is 16 MiB. Reads are repeated per Value CTA, but may benefit from cache residency. Checkpointing can replay preparation; memory lifetime depends on the outer layer checkpoint policy.

Possible regressions include the extra launch at small V, too few preparation CTAs at small B*H, or latency already dominated by token projection/replay rather than inverse formation. Sharing alone does not remove the Python chunk loop, gradient partials, or bank-update launches. No speedup is asserted without measured forward, backward and complete-step results. The original route remains directly selectable for same-input comparisons.

## Validation

Before kernel edits, the independent float64 math gate passed 37 tests: shared versus repeated inverse construction; comparison with the existing PyTorch projection reference; all five projection gradient families against an independently derived VJP; empty/partial/full and infeasible banks; ranks 1/3/16; c=0/0.3/1; finite differences; and nearly dependent bank keys. The initial gradcheck fixture exposed a dimension-slicing error, which was corrected before rerunning the unchanged tolerances.

```bash
FLA_CI_ENV=0 python -m pytest --noconftest -q -o log_cli=false tests/ops/test_hyperbolic_delta_rule_shared_inverse.py -k 'not cuda and not interpreter'
```

Triton interpreter checks execute actual kernels on CPU but do not establish CUDA lowering, GPU numerical behavior or performance. Offline sm_90 compilation, when recorded, only establishes compilation. GPU tests and benchmark results must be reported separately as RUN or GPU NOT RUN.

The completed post-implementation checks on the CPU development host were:

- Existing reference/host regressions plus this mathematical gate: **210 passed, 0 failed, 0 skipped, 16 deselected** (the deselected cases require explicit interpreter or CUDA execution).
- Actual Triton CPU interpreter: **3 passed, 0 failed, 0 skipped, 50 deselected**; ranks 1/3/16, empty/partial/full banks, beta 0/1/random, forward/state/realized outputs and all eight chunk input-gradient families. The saved inverse policy also survives an environment-variable change before backward.
- Benchmark host/CLI checks: **10 passed, 0 failed, 0 skipped**. These check derivative timing separation, numerical-gate rejection, launch-flag attestation, environment restoration, statistical summaries and invalid arguments.
- Offline sm_90 compilation: **15 cases compiled** across FP32/FP16/BF16 storage, preparation and old/shared forward/backward, K=V=128 and R=4. No kernel was launched on CUDA.
- New/changed kernel and new-file Ruff checks, Python syntax checks and diff whitespace checks passed. Legacy protected-wrapper/shared-kernel files have pre-existing import/unused-export lint findings reproduced on the baseline; no repository-wide formatting was applied.
- GPU correctness and timing: **GPU NOT RUN**. The benchmark was invoked without CUDA and exited 2 with a machine-readable `GPU NOT RUN` status, as designed.

Reproduce actual-source CPU and offline checks:

```bash
TRITON_INTERPRET=1 FLA_CI_ENV=0 python -m pytest --noconftest -q -o log_cli=false tests/ops/test_hyperbolic_delta_rule_shared_inverse.py -k interpreter
PYTHONPATH=. python scripts/check_hyperbolic_compilation.py --component shared-inverse --dtype fp32 --output profile/hyperbolic-opt/b2-compile-fp32.json
PYTHONPATH=. python scripts/check_hyperbolic_compilation.py --component shared-inverse --dtype fp16 --output profile/hyperbolic-opt/b2-compile-fp16.json
PYTHONPATH=. python scripts/check_hyperbolic_compilation.py --component shared-inverse --dtype bf16 --output profile/hyperbolic-opt/b2-compile-bf16.json
```

Run CUDA validation before performance conclusions:

```bash
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule_shared_inverse.py -k cuda
FLA_HYPERBOLIC_SHARED_INVERSE=1 FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule.py -k 'value_parallel or protected_triton or smooth_bank'
FLA_HYPERBOLIC_SHARED_INVERSE=1 FLA_CI_ENV=0 python -m pytest -q tests/layers/test_hyperbolic_streaming.py tests/layers/test_hyperbolic_deltanet.py tests/models/test_hyperbolic_deltanet.py
python benchmarks/benchmark_protected_hyperbolic_delta_rule.py --batch 1 --seq-len 2048 --heads 16 --key-dim 128 --value-dim 128 --rank 4 --dtype bf16 --layer-inputs --no-checkpoint --modes fwd bwd fwd_bwd --warmup 20 --repeats 50 --output profile/hyperbolic-opt/b2-gpu.json
```

The benchmark compares explicit old/shared settings on identical inputs, verifies the actual preparation and Value-kernel launch flags, checks an independent small float64 reference and the requested-shape output/all-gradient parity, then rotates method order while measuring synchronized CUDA and wall-time samples. JSON includes the commit, environment, dtype, errors, raw samples, median/p95/range, speedup against the matching baseline, and separate peak allocated/reserved memory trials. Compilation/reference checks and cache-clearing memory trials are outside steady-state timing. The benchmark does not claim a model-level training speedup.
