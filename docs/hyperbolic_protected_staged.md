# Protected HyperbolicDeltaNet trajectory scheduling

This opt-in implementation schedules the existing smooth-bank equations in separate shared-key/score and per-Value trajectories. It preserves the frozen bank at each logical chunk boundary, canonical state, inclusive readout, fixed occupancy predicates, regularized Gram inverse, and first-order gradients. It does not introduce hard selection, detached bank updates, inverse state reconstruction, or lower precision. The previous protected implementations remain the default and the comparison paths. The output retains the legacy Value-dtype storage followed by the public Query-dtype cast. Low-precision Key gradients preserve the separate input-dtype rounding of the original projection and bank-update autograd edges.

Baseline: `405835cc0a44ab4b82d7984f6f5b33e98210ee9b`. GPU correctness and speed are **GPU NOT RUN** on the CPU-only development host; passing CPU mathematics or Triton interpretation does not establish CUDA correctness or performance.

## Dependency decomposition

For a bank with key rows `P`, Value column `y_v`, and shared scores `s`, let the candidate key and selection score at token `t` be `k_t` and `sigma_t`. Once the bank is full, the update gate is

```math
p_r=\operatorname{softmax}(-s/\tau)_r,\quad
u_r=\operatorname{sigmoid}((\sigma_t-s_r)/\tau),\quad g_r=p_ru_r.
```

An unfilled bank uses the existing next-empty-slot one-hot gate instead. The occupancy comparison is discrete and has no derivative, exactly as in the existing implementation. The continuous updates are

```math
P'_r=(1-g_r)P_r+g_rk_t,\quad
s'_r=(1-g_r)s_r+g_r\sigma_t,\quad
y'_{rv}=(1-g_r)y_{rv}+g_r\rho_{tv},
```

where the realized association is `rho_tv = k_t^T S_tv`, with the original key and the post-update canonical state. Neither the gate nor the key/score transition depends on a Value, state, or realized readout. Consequently all chunk-start key banks, score banks, and token gates can be computed before the Value trajectories. This claim applies to the current smooth selector; a future selector depending on realized Values would invalidate the decomposition.

At chunk start, mask inactive bank rows and construct the unchanged regularized key-only matrix `A = P P^T + eps I` on active rows, with identity on inactive rows. Its inverse is computed using the existing `shared_key_gram_fwd_kernel` and `_invert_small`, then shared by all Value channels. For each Value column,

```math
w_v=A^{-1}y_v,\quad d_v=1-cy_v^Tw_v,\quad
G_v^{-1}=A^{-1}+\frac{c}{d_v}w_vw_v^T.
```

The same feasibility test and fallback are preserved. Token constraints use

```math
\alpha_{tv}=G_v^{-1}(Pk_t-cy_vv_{tv}),\quad
\widetilde k_{tv}=k_t-P^T\alpha_{tv},\quad
\widetilde v_{tv}=v_{tv}-y_v^T\alpha_{tv}.
```

The bank used by this projector remains frozen throughout the chunk. A separate running Value bank receives the realized readouts through the already computed gates, and becomes the projector bank only at the next chunk. The state update and readout remain

```math
a=\widetilde k^T\widetilde k-c\widetilde v^2,\quad
r=\widetilde v-\widetilde k^TS,\quad
u=\beta r/a,\quad d=1+c\widetilde v u,
```

```math
S'=(S+\widetilde k u)/d,\qquad o=\mathrm{scale}\,q^TS'.
```

The existing `a > eps` predicate and zero update on rejection are retained. No division by beta occurs, so beta zero and one remain supported. No inverse token transition is used; beta one can make that transition singular. The low-level equations also have a finite flat-curvature limit; this does not expand the positive-curvature contract of the public protected PyTorch reference.

## Reverse dependency and exact VJP

The reverse Value trajectory carries both the state adjoint and the Value-bank adjoint across chunk boundaries. For each reverse token, the bank transition first contributes

```math
\bar\rho_{tv}=\sum_r g_r\bar y'_{rv},\quad
\bar g^{(v)}_r=\bar y'_{rv}(\rho_{tv}-y_{rv}),\quad
\bar y_{rv}=(1-g_r)\bar y'_{rv}.
```

This supplies `bar S' += k * bar rho`, `bar k += S' * bar rho` before the state-update VJP. With the accumulated post-update state adjoint `D`,

```math
\bar n=D/d,\quad \bar d=-D^TS'/d,\quad
\bar u=\bar n^T\widetilde k+c\widetilde v\bar d,\quad
\bar r=\bar u\beta/a,\quad \bar a=-\bar u u/a,
```

```math
\bar{\widetilde k}=\bar n u-S\bar r+2\widetilde k\bar a,\quad
\bar{\widetilde v}=cu\bar d+\bar r-2c\widetilde v\bar a,\quad
\bar S=\bar n-\widetilde k\bar r,\quad
\bar\beta=\bar u r/a.
```

Curvature receives `v_tilde * u * bar d - v_tilde^2 * bar a`, plus the projection contribution below. On a rejected update, the derivatives of `u` through `r`, `a`, and beta are zero. The inclusive output also adds `bar q = scale * S' * bar o` and `bar S' += scale * q * bar o`.

For the projection, let `a_proj = alpha`, `gk = bar k_tilde`, `gv = bar v_tilde`, and `z = G^{-T}(-P gk - y gv)`. On a rejected projection use alpha and z equal to zero. Then

```math
\bar k=gk+P^Tz,\quad \bar v=gv-cy^Tz,
```

```math
\bar P=-\alpha gk^T+z(k-P^T\alpha)^T-\alpha(P^Tz)^T,
```

```math
\bar y=-\alpha gv-czv+cz(y^T\alpha)+c\alpha(y^Tz),\quad
\bar c=(y^Tz)(y^T\alpha-v).
```

Inactive rows are masked. This is the full inverse-dependent VJP, not merely the right-hand-side derivative; the cached inverse must not also contribute a second autograd derivative. Projection contributions to the chunk-start Value bank are added after reversing that chunk's running-bank transitions.

After deterministic reduction over Value channels, the shared reverse scan receives the sum of projector-bank key adjoints and gate adjoints. Let `D_P`, `D_s` denote adjoints after a token, and `D_g` the Value-derived gate adjoint. Form

```math
e_r=D_{g,r}+D_{P,r}^T(k_t-P_r)+D_{s,r}(\sigma_t-s_r),\quad
h_r=e_rp_ru_r(1-u_r)/\tau,
```

```math
j_r=p_r\bigl(e_ru_r-\sum_a p_ae_au_a\bigr)/\tau.
```

For a full bank,

```math
\bar k_t=\sum_r g_rD_{P,r},\quad
\bar\sigma_t=\sum_r g_rD_{s,r}+\sum_r h_r,
```

```math
\bar s_r=(1-g_r)D_{s,r}-j_r-h_r,\qquad
\bar P_r=(1-g_r)D_{P,r}.
```

For empty-slot insertion, the gate is fixed and `h = j = 0`. Add each chunk-start projector-bank key adjoint before reversing the previous chunk. This preserves gradients to initial banks, selection scores, original keys, Value-specific paths, state, beta, and curvature. When selection scores are omitted, the wrapper's differentiable key norm supplies the extra key derivative.

## Kernel and memory model

Forward has three launches: one shared key/score scan per batch/head; one shared inverse over all batch/chunk/head positions; and one persistent Value scan per batch/head/Value. The former `2 * ceil(T/C)` per-chunk launches and repeated contiguous chunk copies are eliminated. The shared key/score scan remains sequential, and the Value state scan remains sequential; neither dependency is claimed to disappear. The shared inverse is prepared once per chunk rather than independently in each Value CTA. Value kernels only consume the cached inverse and do not perform a matrix multiply or inversion, so they pad the rank to its next power of two instead of the former minimum of 16. Shared Gram/inverse and shared-score kernels retain their original minimum padding of 16, inversion algorithm, and softmax semantics.

Backward uses a bounded sequence of reverse windows. A window has one Value reverse kernel and two deterministic reduction kernels, followed by one shared key/score reverse kernel for the whole sequence. It does not use atomic accumulation. Token state/projection/Value-bank replay scratch is one chunk per Value CTA and is reused. Window-expanded q/k partials are bounded to approximately 256 MiB in total, or one minimum chunk if that is larger; at most 16 chunks share one window. It avoids whole-sequence `[B,T,H,V,K]` gradient buffers, which would be 16 GiB each for total tokens 16384, H16, K128, V128. The legacy per-chunk backward already uses chunk-local partials; this figure describes an avoided whole-sequence expansion in a naive persistent rewrite, not a memory saving measured against the baseline.

For total tokens 16384, H16, K128, V128, R4, C16, the FP32 retained data are approximately 1 GiB state checkpoints, 32 MiB Value-bank checkpoints, 32 MiB key-bank trajectory, 0.25 MiB score trajectory, 1 MiB inverse trajectory, and 4 MiB token gates. These are per-layer live tensors, not a prediction of whole-model peak memory. Layer checkpointing, autograd lifetimes, replay scratch, expanded gradient windows, and allocator behavior affect the actual peak.

The arithmetic for projection and recurrent state updates remains essentially unchanged. Expected gains come from launch reduction, shared inverse reuse, removal of chunk-copy/cat overhead, and independent Value execution through multiple chunks. A one-chunk call has three forward launches instead of the original Value-parallel path's two; short prefill and decode may regress and must be measured independently. Risks include a long shared-score scan at low batch/head count, persistent-kernel register pressure, checkpoint memory, and reduction/partial-buffer traffic in backward. A longer window lowers launch count but increases memory; it is not assumed to be faster. There is no measured speedup or production recommendation until CUDA correctness and same-hardware forward/backward/end-to-end comparisons pass.

## Compatibility and verification

The opt-in dispatcher uses `FLA_HYPERBOLIC_PROTECTED_STAGED=1` with the existing normalized dense protected inputs. The flag applies to the existing `value_parallel` strategy and to `auto` only when V is at least 8. An explicit `fused` request and the small-Value `auto` route keep the fused implementation. Default routing, public return values, state-dict/config fields, and the previous comparison strategies remain unchanged. Logical chunk size remains 16 or 32; changing it still changes protected-bank scheduling. Window size only changes backward scheduling. With `checkpoint=True`, the opt-in wrapper checkpoints the whole staged call using non-reentrant PyTorch checkpointing, rather than separately checkpointing each logical chunk. With `checkpoint=False`, it directly retains the staged checkpoints. The whole-call choice can recompute more work during backward and must be benchmarked separately, especially under an outer layer checkpoint. Packed protected inputs remain unsupported by the existing public interface.

The existing `protected_streaming` wrapper continues to define global boundaries, partial-prefix replay, `pending`, `block_start`, and empty calls. The staged entry processes the supplied block using exactly its starting bank; it cannot advance the projection bank at a different token boundary. Empty calls return the incoming state and banks. CUDA tests compare full, chunked, token-wise and irregular streaming through the existing public wrapper.

The independent mathematical gate ran before kernel implementation: **19 CPU tests passed, zero failures and zero skips**. It compares the separated and coupled FP64 references and all ten input gradient families, independently checks both shared and Value analytic VJPs, and runs finite-difference gradcheck. It includes beta zero/one, c zero/positive, nonzero initial state, partial/full/empty occupancy, tails, and chunk sizes 16/32.

```bash
FLA_CI_ENV=0 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k 'not cuda and not interpreter'
```

Actual Triton interpreter and offline compiler coverage, when run, is CPU verification only. CUDA tests and the protected benchmark must be executed on the target GPU before selecting this route for training; the existing oracle and tolerances remain unchanged.


## Recorded CPU and compiler checks

After implementation, the frozen reference suite and the new CPU mathematics/boundary tests passed together: **190 passed, zero failures, zero skips** (167 existing reference tests plus 23 new CPU tests). The selected command deselected accelerator/interpreter cases explicitly; it did not count them as passes.

The actual new Triton kernels also passed **11 CPU interpreter cases**, comparing output/state/bank values and all ten gradient families against the previous Triton Value-parallel and smooth-bank kernels. Coverage includes rank 1 and non-power-of-two rank 3, multi-batch/head/Value, chunk sizes 16/32, tails, reverse-window boundaries, beta zero/one, c zero/positive, whole-call non-reentrant checkpointing, mixed Query/Key/Value storage, and a temperature below the reference's clamp threshold compared against the unchanged raw-temperature Triton baseline. Interpreter warnings were platform availability, NumPy scalar-conversion deprecation, and a saturating sigmoid's exponential overflow; assertions remained strict and all checked gradients were finite.

Twenty-four final offline compilation cases cover the six new kernels, rank 1/4/16, FP32/FP16/BF16 storage, C16/C32, and production K/V128 on the sm90 target. These are compilation checks, not GPU execution. GPU test collection finds **36 CUDA cases** covering direct reference comparisons, the 16K-length regime, streaming, explicit fallbacks, output/gradient storage, operator checkpointing, and rank 16, bank/score-only losses, the flat-curvature boundary, and a small model with outer layer checkpointing and inner protected checkpointing disabled. They are **GPU NOT RUN** here.

With FP32 Query/Key/Value storage at H16, K=V128, R4, C16, the production-matching offline assembler flags (`-lineinfo --gpu-name=sm_90a --fmad=false`, four warps) report:

| Value kernel | Previous shared-inverse path | Staged path | Spill stores / loads |
| --- | ---: | ---: | ---: |
| Forward registers per thread | 79 | 56 | 0 / 0 bytes in both |
| Forward shared memory per block | 4,128 bytes | 64 bytes | — |
| Backward registers per thread | 126 | 167 | 0 / 0 bytes in both |
| Backward shared memory per block | 4,608 bytes | 64 bytes | — |

The backward register increase is a remaining performance risk even though no spill was emitted for this case. These figures do not measure occupancy or latency. The final CPU interpreter command is:

```bash
TRITON_INTERPRET=1 FLA_CI_ENV=0 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k interpreter
```

Run actual CUDA checks in a separate process without `TRITON_INTERPRET`:

```bash
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule_protected_staged.py -k cuda
```

The correctness-gated protected benchmark is extended alongside this implementation; use its raw JSON, synchronized forward/backward timings, allocated/reserved memory, and dispatch evidence to decide whether the opt-in route should be retained for a workload. No latency or throughput result has been measured in this environment.
