# Staged unprotected HyperbolicDeltaNet

This is an opt-in scheduling prototype for the existing unprotected Tensor Core rule, based on `405835cc0a44ab4b82d7984f6f5b33e98210ee9b`. It preserves FP32 state, FP32 reductions, TF32x3 GEMMs, the per-Value triangular substitution, and the public canonical-state convention. GPU correctness and speed must be measured before selecting it for training by default. It does not change the protected bank or its logical chunk boundaries.

Set `FLA_HYPERBOLIC_STAGED=1` together with `chunk_hyperbolic_delta_rule(..., tensor_core=True)` or the existing model `tensor_core=True` configuration to request this route. The unset/default environment retains the previous implementation. Short evaluation sequences that select recurrent mode, optional extra readout, K greater than 128, and active legacy low-precision TC mode retain their existing paths.

## Dependency graph and forward equivalence

For one chunk and head, let `Q,K` have shape `[C,K]`, `V,Beta,W` have shape `[C,V]`, and the entry state `S` have shape `[K,V]`. Define, independently for each Value column,

```math
a_i=\|k_i\|^2-cv_i^2,\quad \alpha_i=\beta_i/a_i,\quad G=KK^\top,
\quad L_v=I+\operatorname{tril}(\operatorname{diag}(\alpha_v)(G-cv_vv_v^\top),-1).
```

The homogeneous recurrence with entry gauge `z=1` gives

```math
W_v=L_v^{-1}[\alpha_v\odot(v_v-KS_v)],\quad
z=1+c\operatorname{cumsum}(V\odot W),
\quad S_+=(S+K^\top W)/z_{\mathrm{last}}.
```

To see the triangular equation, substitute `M_{i-1}=S+sum_{j<i} k_j W_j` and `z_{i-1}=1+c sum_{j<i} v_j W_j` into `W_i=alpha_i(z_{i-1}v_i-k_i^T M_{i-1})`. Moving earlier weights to the left gives exactly `L_v W_v=alpha_v(v_v-KS_v)`. No inverse state transition is used; beta=1 may make a token transition singular and remains supported.

The inclusive readout is

```math
P=\operatorname{tril}(QK^\top),\qquad O=\mathrm{scale}(QS+PW)/z.
```

The strict lower mask in `L` and inclusive mask in `P` differ intentionally. Resetting the homogeneous gauge at each chunk boundary produces the same canonical `S=M/z`; it introduces no extra forgetting or clipping.

The implementation stages are:

1. All chunks compute shared `G` and `P` once per head. These depend only on the current input chunk.
2. One CTA per sequence, head and 16 Value columns scans chunks. It solves for the actual state-dependent `W`, writes `W` and the entry state, and updates the state. It performs no query readout.
3. All chunks compute output using their saved entry state and weights.

`W` cannot be precomputed independently of the entry state. The per-Value systems remain distinct. This change moves independent work out of the scan without replacing the solver or constructing per-Value inverse matrices.

## Independent first-order VJP

Let `dO` be the output cotangent, `dS+` the incoming state cotangent from the next chunk, and overbars denote cotangents. Define

```math
D=\mathrm{scale}\,\overline O/z,\quad
R_i=\sum_{t\ge i}-\overline O_t O_t/z_t,\quad
X=\overline S_+/z_{\mathrm{last}},
\quad J_i=R_i-\sum_K(\overline S_+\odot S_+)/z_{\mathrm{last}}.
```

The readout preparation stage computes `D`, `P^T D` and `R` independently for all chunks. The reverse state scan performs

```math
\overline W=P^\top D+KX+cV\odot J,\qquad
L_v^\top Z_v=\overline W_v,\qquad U=\alpha\odot Z,
\qquad \overline S=Q^\top D+X-K^\top U.
```

The transposed triangular solve stays inside the reverse state scan: its right-hand side depends on the future state cotangent. A forward solve is not repeated in backward because the actual forward `W` is retained. The scan carries the forward next-state value backwards, starting from `HT`, so the next-state numerator GEMM is not recomputed.

After the state cotangents are available, local parameter derivatives run across all chunks. Let `prefix` and `suffix` below exclude the current token, and define

```math
F=\operatorname{tril}(DW^\top),\quad E=-\operatorname{tril}(UW^\top,-1),\quad
p=\operatorname{prefix}(V\odot W),\quad t=\operatorname{suffix}(U\odot V),
```

```math
\overline\alpha=Z\odot[V-KS-\operatorname{tril}(G,-1)W+cV\odot p],\quad
\overline a=-\overline\alpha\odot\mathrm{Beta}/a^2.
```

The independent VJP is

```math
\overline Q=DS^\top+FK,
\qquad \overline K=F^\top Q+WX^\top-US^\top+(E+E^\top)K+2K\sum_V\overline a,
```

```math
\overline V=cW\odot J+c(U\odot p+W\odot t)+U-2cV\odot\overline a,
\quad\overline{\mathrm{Beta}}=\overline\alpha/a,
\quad\overline c=\sum(V\odot W\odot J+U\odot V\odot p-\overline a\odot V^2).
```

The formula for `d_alpha` is valid at beta=0; dividing `W` or `U` by alpha would lose that derivative. Padding rows have zero alpha and their gradients are masked. `L` has a fixed unit diagonal, so its cotangent is strictly lower triangular. The final local kernel accumulates Value-tile contributions to `dQ`, `dK`, `F`, and `E` in FP32 and writes final `dQ/dK` once. It uses no floating-point atomics.

### Resource-driven local VJP split

Offline sm90 compilation exposed unacceptable spilling in the first local VJP prototype: with K=V=128, four warps, strict TF32x3 and fusion disabled, C=16 used a 7240-byte per-thread stack and C=32 used 8808 bytes. These are compiler resource results, not measured GPU latencies. Disabling software pipelining did not resolve the problem, so that whole-K local kernel is rejected.

The replacement retains the preceding equations and separates their data dependencies. A Value-local stage consumes `Z` to calculate `d_alpha`, `da`, `dV`, `dBeta` and the curvature contribution. Once the reverse state scan has completed, `Z` has no remaining consumer except `d_alpha`; that stage can therefore overwrite its own disjoint token/Value slice with `da`. No saved forward tensor is mutated. A following Q/Key stage reads `da` and accumulates the final `dQ/dK` in disjoint 32-column Key tiles. This creates neither floating-point atomics nor expanded global `dQ/dK` partials. Empty chunk slots still initialize their curvature partials to zero.

The split adds one backward launch and a read of the reused `da` buffer. `F` and `E` are recomputed for each of `ceil(K/32)` output Key tiles; unlike full-K output contractions, these are small `[C,16]@[16,C]` products. The shared input Gram `G` and Query-Key product `P` remain computed once per chunk/head. This trade exchanges duplicated small cotangent products for substantially smaller live state/gradient tiles; acceptance requires an offline resource recheck followed by the existing exact-kernel CPU tests and later CUDA profiling.

The Value-local `KS` product is contracted in 32-column Key tiles: `KS=sum_j K_j S_j`. This is the same real-valued expression, with FP32 accumulation and strict TF32x3 products retained. It changes the FP32 reduction grouping and is checked against the unchanged existing tolerances; it does not introduce lower-precision operands. The squared Key norm deliberately retains the original full-Key `tl.sum(key*key)` reduction used in both state scans. Grouping that reduction by Key tiles could change `a>0` very near the feasible-domain boundary and make `U` and `da` inconsistent. The norm is reduced before the tiled `KS` loop so that its full Key tile need not stay live across the contractions.

Within each Q/Key tile, the common cotangents `F`, `E` and `sum_V da` are contracted and applied to `Q/K` before a second Value pass adds `DS^T`, `WX^T` and `-US^T`. This follows directly from linearity of the displayed VJP. It prevents the `F/E` accumulators and the state operands from being simultaneously live, without a temporary gradient write or an early low-precision cast. The price is rereading `D/W/U` in the second pass; their small per-chunk tiles may be cached, but this is not assumed in the performance claim.

The final sm90 offline resource check uses four warps, H=16, N=1, K=V=128, FP32 inputs, int64 packed-offset storage (also used by public dense calls), `enable_fp_fusion=False` and fixed curvature:

| Kernel | C | Registers/thread | Spill stores / loads | Shared memory |
| --- | ---: | ---: | ---: | ---: |
| Value-local VJP | 16 | 128 | 0 / 0 bytes | 8192 bytes |
| Value-local VJP | 32 | 183 | 0 / 0 bytes | 12288 bytes |
| Q/Key VJP | 16 | 123 | 0 / 0 bytes | 3104 bytes |
| Q/Key VJP | 32 | 255 | 16 / 16 bytes | 8192 bytes |

The large spill problem is removed; C=32 still has high register pressure and a small spill. An eight-warp experiment retained 255 registers and did not establish a runtime benefit, so it is not selected. The compiler figures do not prove latency or occupancy on the target GPU, and C=32 requires particular attention in the performance matrix.

The complete seven-kernel resource audit also retained the other scheduling risks; it did not inspect only the improved local VJP:

| Stage | C16 registers | C16 spill stores / loads | C32 registers | C32 spill stores / loads |
| --- | ---: | ---: | ---: | ---: |
| Shared preparation | 255 | 0 / 0 B | 255 | 1108 / 1192 B |
| Forward state scan | 255 | 244 / 228 B | 255 | 388 / 372 B |
| Parallel output | 239 | 0 / 0 B | 255 | 0 / 0 B |
| Parallel output VJP | 255 | 28 / 28 B | 255 | 60 / 60 B |
| Reverse state scan | 255 | 688 / 612 B | 255 | 1404 / 1360 B |
| Value-local VJP | 128 | 0 / 0 B | 183 | 0 / 0 B |
| Q/Key VJP | 123 | 0 / 0 B | 255 | 16 / 16 B |

For the unchanged strict TC baseline with the same H/K/V, int64 offsets and compiler flags, the single forward kernel reports 2292/2688 B spill stores/loads at C16 and 4088/5276 B at C32; the single backward kernel reports 4928/5728 B and 12968/17868 B respectively. These are static assembler figures per compiled kernel, not dynamically counted traffic. Different grids and loop counts mean their sums cannot be treated as measured bandwidth savings or speedup. Persistent state scans still have high register occupancy and residual spills.

Reproduce the resource comparison without a CUDA device:

```bash
PYTHONPATH=. FLA_CI_ENV=0 python scripts/check_hyperbolic_compilation.py --component tensor-core --resources --chunk-size 16 --output profile/hyperbolic-opt/strict-tc-compile.json
PYTHONPATH=. FLA_CI_ENV=0 python scripts/check_hyperbolic_compilation.py --component staged --resources --constant-curvature --chunk-size 16 --output profile/hyperbolic-opt/staged-compile.json
```

Repeat with `--chunk-size 32`, and use `--cu-dtype int32` for packed-offset storage. The used `tl.range` scheduling arguments are present in the repository's minimum Triton 3.3 API; no dependency minimum or default backend was changed.

## Numerical and boundary contract

The legal domain and public validation remain those of the existing operator: beta in `[0,1]`, positive `a`, finite inputs, and initial state columns strictly inside the Klein ball for positive curvature. Curvature zero, nonzero initial state, scalar/per-Value beta, partial chunks, and empty packed sequences preserve their existing meaning. No tolerance or feasible-domain threshold is relaxed.

All checkpoint and token pointer arithmetic is widened to int64 before products. The checkpoint slot for sequence `n`, chunk `j` is `floor(cu[n]/C)+n+j`, matching the existing allocation bound `ceil(total/C)+N`. Parallel kernels recover the owning sequence by a device-side binary search over strictly increasing slot bases. This avoids copying packed lengths to the CPU or creating a data-dependent launch grid. Holes are not read, and curvature partials for holes are initialized to zero.

Different summation order can change FP32 rounding. Equivalence is mathematical, not a claim of bitwise identity. The original route remains the default and the comparison route. Legacy broad `TC_FAST=1` precision is not silently reinterpreted as the strict staged policy; that existing route remains available. Independent mixed-precision work requires its own numerical design review.

## Performance model before implementation

With `nv=ceil(V/16)`, the original forward and backward each form `G` and `P` inside every Value CTA. Shared preparation forms each only once per chunk/head and retains them through backward. This removes `4*nv-2` GEMMs of shape `[C,K]@[K,C]` per chunk/head across forward plus backward. Other GEMMs are rescheduled rather than assumed free.

The forward state scan contains `KS`, triangular substitution and `K^T W`; query products move to parallel stages. The reverse scan contains `KX`, the transposed solve, `Q^T D`, and `K^T U`. It no longer computes output numerators, a forward triangular replay, or local `dQ/dK` contractions. Parameter-gradient contractions operate on all chunks and aggregate shared Gram cotangents before multiplication by `K`.

The cost is three forward and four backward kernel launches instead of one each, extra global reads/writes, and retention of state checkpoints for parallel inference output as well as training. The Q/Key VJP uses two Value passes and repeats the small `F/E` products for each 32-column Key tile, as detailed above. For small sequences or small V this may lose. GPU profiling must measure occupancy, spills, and end-to-end training before promotion.

For total tokens 16384, H=16, K=V=128, C=16, ignoring padding:

| Buffer | FP32 size | Lifetime |
| --- | ---: | --- |
| Shared G and P | 32 MiB combined | Forward through backward |
| Actual W | 128 MiB | Forward through backward |
| Entry state checkpoints | 1 GiB | Forward through backward |
| D, Z, U and normalization suffix | 512 MiB combined | Backward |
| Normalized future state cotangents X | 1 GiB | Backward |
| Original expanded dQ and dK | 2 GiB combined | Removed from staged backward |

`dW_readout` is reused for `Z` and then `da`, and the readout normalization suffix is updated in place. The saved boundary cotangents replace part of the removed gradient expansion; net temporary-buffer reduction is approximately 352 MiB, not 2 GiB. Curvature gradients, when requested, add one scalar per chunk/head/Value tile. These are operator buffer estimates, not whole-model peak-memory measurements. No speedup percentage is predicted without hardware evidence.

## Validation

Before kernel implementation, 75 independent CPU float64 tests passed: staged forward versus canonical recurrence, staged autograd versus recurrence autograd, the three-stage manual VJP versus both, all six gradient families, beta=0/1/random, c=0/0.4/2.5, T=0/1/17/35, C=16/32, scalar beta, packed empty sequences, and finite-difference gradcheck. Output/state tolerance is `atol=1e-11, rtol=1e-10`; VJP tolerance is `atol=1e-10, rtol=1e-9`.

```bash
FLA_CI_ENV=0 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_staged.py -k 'not cuda and not interpreter'
TRITON_INTERPRET=1 FLA_CI_ENV=0 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_staged.py -k interpreter
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_hyperbolic_delta_rule_staged.py -k cuda
```

The expanded algebra gate has 77 passing tests, including tiny positive key norms and near-feasible values. Fifteen additional tests execute all seven actual kernels through Triton's CPU interpreter: dense and packed/empty sequences, K/V up to 128, both chunk sizes, nonzero initial state, beta=0/1, final-only/output-only/curvature-only backward, noncontiguous input, scalar-gate expansion, and split-cache parity. Twelve further CPU interpreter tests exercise the real public host dispatch using an explicitly mocked device label during validation and real CPU tensors after normalization. They cover mixed Q/K/V storage dtypes, packed int32 lengths, fixed and learnable curvature, scalar beta and an environment change between forward and backward. They are host-routing/interpreter tests, not CUDA-device validation. All use the existing TC tolerances; CPU-backend and NumPy interpreter deprecation warnings are not numerical-tolerance warnings.

Two additional interpreter cases guard the full-Key norm reduction at a deliberately ill-conditioned feasible point. The fixture has FP64 `a` approximately `3.245e-9 > 0`, while the existing CPU FP32 reduction gives approximately `3.576e-7`; a grouped 32-column reduction gives nonpositive `a`. With beta=0, output remains well behaved but the beta derivative exposes an inconsistent denominator immediately. These cases compare the actual old TC kernels and staged kernels with unchanged tolerances. They establish preservation of the existing FP32 path, not FP64 gradient accuracy at that condition number. There are 29 interpreter/host-routing cases in total.

The 59-case CUDA matrix includes FP32/FP16/BF16, mixed Q/K/V dtypes, supported gradient families, 1025/4097/16384-token comparison with the existing TC implementation, packed tails and empty sequences, cached irregular segments, the near-boundary denominator fixture, and readout/large-key/legacy-fast fallbacks. These CUDA tests have not run in the CPU-only development environment. Their branch assertions fail if a supposed staged comparison silently uses another implementation.

CPU algebra, CPU Triton interpretation, and offline compilation are separate checks. None certifies CUDA execution, numerical error on Tensor Cores, races, or performance. The CUDA test matrix must run on the candidate commit before production use.
