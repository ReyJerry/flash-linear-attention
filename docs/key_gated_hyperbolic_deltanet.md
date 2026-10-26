# Key-Gated Hyperbolic DeltaNet: model, numerical contract and validation

Key-Gated Hyperbolic DeltaNet (KHDN) has a separate model identity with per-Key-channel forgetting and grouped joint Lorentz projection. Existing HyperbolicDeltaNet, protected HDN, GDN and KDA APIs, checkpoints, kernels and numerical policies remain unchanged. The bounded correctness evidence below does not establish full-training readiness or a speed claim.

## Current implementation status

`fla.ops.key_gated_hyperbolic_delta_rule.naive_recurrent_key_gated_hyperbolic_delta_rule` now provides the independent PyTorch dense-solve recurrence. It supports head-shared beta/decay, multiple heads and groups, noncontiguous inputs, packed empty sequences, canonical initial/final states and all seven first-order source gradients. It retains FP64 arithmetic when any source is FP64, otherwise FP32; the public output follows the Query dtype. It does not implement the production chunk schedule or establish GPU correctness.

The formal CPU suite passes **731 tests**: 242 independent dense-Oracle cases and 489 chunk/manual cases. Coverage includes seven-input gradients, grouped geometry, endpoint derivatives, main-size multiple groups, exact strong-decay state/derivative, streaming, packed/empty inputs, numerical range and ambient-autocast isolation. The unchanged HDN reference/host suite separately passes **173 tests**. A focused validation correction preserves legal finite initial states at zero curvature even when squaring a large state would overflow, retaining the curvature derivative.

The separate Layer and Hugging Face Config/Model/ForCausalLM identities implement the specified Key-channel gate, canonical cache metadata, packed-label masking and gate initialization/loading policy. The explicit `attn_mode='naive'` route uses the independent dense Oracle. The `chunk` and `fused_recurrent` entries execute Triton forward and handwritten backward, with separately identified range-sensitive host execution. Gate parameters retain FP32 values through conversion and assignment; callers must honor their `_no_weight_decay` tags when creating optimizer groups. Inference-produced cache tensors can reenter gradient-enabled execution as ordinary tensors, while already differentiable cache history remains connected.

The original explicit-naive Layer/Model gate passed **67 tests**, including 11 CUDA cases, with no failures/skips; it excluded seven then-unimplemented device cases. The strict device implementation subsequently passed **80 tests in one H20 run**: 73 operator cases and those seven Layer/Model cases, using actual shared components. Both carry modes and C16/32/64 pass the frozen output/state/seven-source budgets with FP32/BF16/FP16 public inputs. Internal arithmetic is FP64 SIMT, with no Tensor Core speed claim. The unchanged three-case memcheck and racecheck gates both report zero errors after a compiler-layout repair. Independent review covers the source graph and the repair's semantics. See [device implementation and limits](../fla/ops/key_gated_hyperbolic_delta_rule/README.md).

Independent review accepts the complete CPU source graph and its bounded numerical gate, including the repaired near-boundary/range cases, optional public-state handling and preservation of extreme cotangent components. The native CPU path supplies independent dense Oracle, ordinary chunk AD and complete handwritten VJP comparisons. Exceptional arbitrary-precision calls use the manual graph and independent high-precision canonical truth; they do not supply a third AD witness. See [the CPU mathematics and support contract](key_gated_hyperbolic_math.md). Broader long-context and mixed-precision validation, parallel output/backward scheduling, scalable checkpointing, profile-driven optimization, full training and 1/8/16-card KHDN acceptance remain pending. The existing native GDN tiny-model DDP checks establish harness mechanics separately, not KHDN training readiness.

Run the CPU reference independently of accelerator imports:

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 python -m pytest --noconftest -q tests/ops/test_key_gated_hyperbolic_delta_rule_reference.py
```

Development baseline: `46c54ef0d57609d66a5c158fa4a8cce124ca1483` on `add-hyperbolic-delta-net`. The implementation will retain later branch work. The specification is frozen by SHA-256 `faacdb1ded8cadf9716333adb0827dbbc444ff0af13ad6683affc3ca8e4d03a8`; the execution requirements are frozen by SHA-256 `86d39cdd0dc2000d81ac4478cb89fde56c428be8d32941972952de95456d1fb6`.

## Model contract

For each head and Value group of width `g`, canonical state is `S: [K,g]`, `kappa=c/g`, `n=k^T k` and `X=diag(exp(log_decay)) S`. The update is

```text
G = n I_g - kappa v v^T
u = beta solve(G, v - X^T k)
d = 1 + kappa v^T u
S_next = (X + k u^T) / d
o = scale q^T S_next
```

The domain is finite `c>=0`, `0<=beta<=1`, finite elementwise `log_decay<=0`, `n>0`, `n-kappa ||v||^2>0` and, for positive curvature, `kappa ||S||_F^2<1`. Beta and Key-channel decay are shared across the head's Value groups; curvature is constant within one operator call. The original Key norm is used. Group width changes the model; execution chunk length does not.

Primary configuration is `K=V=128`, `g=32`, positive curvature, trainable nonidentity Key-channel forgetting and `C=32`, with `C=16/64` execution comparisons and a separate `g=16` ablation. Beta zero still applies forgetting; beta one writes the complete current grouped association. A learnable curvature equal to zero retains its generally nonzero derivative.

The gate is `log_decay=-exp(A_log)*softplus(W2 W1 x + dt_bias)`, with bias-free linear projections and FP32 `A_log/dt_bias`. This does not use KDA's bounded-sigmoid gate or beta-times-two option. Existing output gating remains distinct. Inputs/outputs use `[B,T,H,K/V]`; beta uses `[B,T,H]`; authoritative recurrent state uses `[N,H,K,V]`. Packed empty sequences preserve their initial state and terminal-state gradient.

## Exact chunk and numerical proposal

Input-only preparation is shared across chunks, with one head-shared Key system and one feedback system per Value group:

```text
gamma = beta/n
L = I + tril(diag(gamma) G_key_relative, -1)
A = inverse(L)
F = A diag(gamma) K_entry_decay
Y = F S
R = A * (V V^T)
T_beta = I + diag(beta) J_strict_lower
Q_s = solve(T_beta, diag(n))
eta = solve(T_beta, beta)
Z = Q_s - kappa R
h = rowdot(V,Y)
w = solve(Z, eta-kappa h)
U = diag(w)V - L Y
```

The input-dependent coefficients of the last solve are precomposed before the state scan. `Z` has a positive, generally nonunit diagonal; existing unit-diagonal/low-precision helpers cannot be reused without the corresponding scaling and source-gradient audit. Endpoint formulas do not divide by beta, `1-beta`, or curvature.

Positive masses obey `z_i=(1-beta_i)z_previous+n_i w_i`. Outputs and final state use a directly computed strict-past residual and current target, `f(R,T,r,t)=R/(r+t)+(t/(r+t))*T`. Residuals are never reconstructed by subtracting the target from an already rounded normalized output. An optional input-dependent time gauge rescales the feedback matrix, RHS, residual, scan and cross-chunk interface consistently. Its derivative may be held fixed only after proving arbitrary-gauge invariance and full source-gradient parity.

Direct local-interval decay accumulation or an audited same-model strict fallback must preserve the valid strong-decay case `ell=[-1e8,-1]`, `q=k=1`, `v=[.5,.1]`, `beta=[1,0]`, `S0=0`, `c=.6`: the final state and second-decay derivative are `.5*exp(-1)`. Naive FP32 prefix subtraction is inadequate. Fallbacks will be reported separately from optimized-path measurements.

The new implementation starts with a frozen strict numerical contract. Any later per-GEMM precision change requires its own explicit operand/accumulator/storage audit, unchanged numerical gate and review; this proposal does not approve blanket BF16 lowering of feedback, normalization or state arithmetic.

## Optional exact Y-carry

No-carry remains the baseline. The candidate computes the next block projection through two structured branches using `E=F_next D0`, `H=F_next P_past` and the corresponding target projection; `D0` is diagonal-plus-rank-one, not a general dense Key transform. It shares the same feedback preparation, precision, gauge and checkpoint policy with no-carry. It neither removes all feedback nor guarantees less arithmetic.

The complete handwritten first-order VJP covers `q,k,v,beta,log_decay,S0,c`, all gate/preprocessing parameters and final-state cotangents. Carry-derived coefficients retain all source gradients. Only the initial/reanchored block differentiates `Y=FS` directly; subsequent blocks return the carry adjoint through the preceding block. Current outputs must have zero future-input gradients. Public caches contain canonical state, never future-dependent Y or absolute cumulative gates. Unsupported double backward will be explicitly rejected.

## Independent validation and promotion

The mathematical audit and a separate independent review precede production kernels. Three distinct CPU checks are required: FP64 recurrent dense `g x g` solve, stable chunk ordinary autograd, and full handwritten VJP, plus finite differences. Coverage includes beta/curvature endpoints, physical padding, packed empty segments, nonzero initial state, multiple heads/groups, near-boundary and strongly decaying inputs, streaming and causality.

The supplied finite research smoke was reproduced without modifying its thresholds: 33 checks passed and gradcheck succeeded. The frozen existing HDN reference/host gate passed 173 tests. Neither result validates new production kernels. At proposal time all new KHDN GPU tests, full training, performance, DDP and resume validation are NOT RUN.

Hardware validation will first run one idle H20 and then a finite manifest on the authorized two-node 16-H20 allocation. Performance compares matched no-carry/Y-carry and separately remeasured GDN, KDA, old HDN TC/staged and available supplementary baselines. A full 24-layer hidden-2048 model uses the four 16,384-token microbatch configurations, with accurate parameter counts including the new gate projections. Measurements retain raw samples, at least 10 warmups and 30 samples, three final independent repeats, numerical/path evidence, memory peaks and separate profiler runs.

True DDP at world sizes 1/8/16 is a separate gate: strong scaling fixes 262,144 input tokens per optimizer step with accumulation 16/2/1; weak scaling fixes 16,384 tokens per rank with accumulation one. Unequal valid-label counts require global weighting and a representative single-device versus DDP gradient/update comparison. Finite continuous training and save/resume must pass before formal-training suitability is claimed.

If exact Y-carry does not produce a repeatable complete forward/backward or training win, no-carry remains the default and the regression is reported. No model substitution, gate removal, curvature-zero setting, group-width change, tolerance relaxation or skipped failure will be used to claim completion or acceleration.
