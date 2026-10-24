# HyperbolicDeltaNet selective precision RFC draft

This document proposes an experimental per-GEMM precision policy for the unprotected Tensor Core path at baseline `405835cc0a44ab4b82d7984f6f5b33e98210ee9b`. **The policy is not implemented or approved.** Existing defaults and `FLA_HYPERBOLIC_TC_FAST` behavior remain unchanged.

The repository requires an RFC issue before a precision change. Creating the required issue in `ReyJerry/flash-linear-attention` returned GitHub HTTP 410, “Issues has been disabled in this repository.” This document and the CPU mathematical tests preserve the proposal for review; they do not substitute for that RFC or authorize changing validated numerical behavior. Enabling repository issues or designating an authorized RFC issue location is required before implementation.

## Existing dispatch and precision

`HyperbolicDeltaNet.forward` normalizes Query in FP32 and casts it to the incoming hidden-state dtype. `prepare_key_value` keeps normalized Key and bounded Value in FP32. BF16 autocast therefore does not imply BF16 Query storage: FP32 residual/embedding activations can leave Query in FP32.

`HyperbolicDeltaRuleFunction.forward` enables the existing global `FAST` flag only when `mode == 'chunk_tc'`, Query storage is FP16/BF16, and `FLA_HYPERBOLIC_TC_FAST=1`. This flag is stored on the autograd context and reused in backward. The kernel's `_mm` converts both inputs to FP16/BF16 for every GEMM under that flag; otherwise it uses FP32 operands with `input_precision='tf32x3'`. All these GEMMs have FP32 accumulators.

The proposal would separate operand precision from Query/output storage. An explicitly selected strict policy would disable the legacy global fast flag. An omitted policy would preserve the legacy behavior. Proposed names such as `qk_fp16` are design labels in this document, not implemented environment values.

## Complete GEMM inventory

Let `C` be chunk length, `D` Key dimension and `U` the Value tile width (currently 16). `S` has shape `D × U`, `W` has shape `C × U`, and `P = tril(QKᵀ)`. All operands below are FP32 after load in accurate mode; all accumulations and stored intermediate states are FP32. Input tensors may have different storage dtypes.

| Stage | Product | Shape | Use and numerical sensitivity | Proposed first experiment |
| --- | --- | --- | --- | --- |
| Forward and replay | `KKᵀ` | `C × D` by `D × C` | Strict lower triangle enters every per-Value triangular system and changes state evolution | Separate later experiment; strict initially |
| Forward and replay | `KS` | `C × D` by `D × U` | Residual of the state update | Strict |
| Forward and replay | `QKᵀ` | `C × D` by `D × C` | Inclusive causal readout coefficients; no forward-state dependency | FP16 first; BF16 separately |
| Forward and replay | `QS` | `C × D` by `D × U` | State readout numerator | Strict |
| Forward and replay | `PW` | `C × C` by `C × U` | May combine large cancelling homogeneous terms | Strict |
| Forward and replay | `KᵀW` | `D × C` by `C × U` | Canonical state transition numerator | Strict |
| Backward | `PᵀDout` | `C × C` by `C × U` | Readout contribution to `dW` | Strict |
| Backward | `Dout Sᵀ` | `C × U` by `U × D` | Direct Query gradient | Strict |
| Backward | `Dout Wᵀ` | `C × U` by `U × C` | Readout coefficient gradient | Strict |
| Backward | `tril(Dout Wᵀ) K` | `C × C` by `C × D` | Query gradient through QK; must use its rounded Key operand if QK is lowered | FP32 adjoint at the matching operand |
| Backward | `QᵀDout` | `D × C` by `C × U` | Initial-state adjoint from numerator | Strict, original Query |
| Backward | `W Doutᵀ` | `C × U` by `U × C` | Transpose of the readout coefficient adjoint, currently recomputed | Strict; possible independent transpose reuse |
| Backward | `triu(W Doutᵀ) Q` | `C × C` by `C × D` | Key gradient through QK | FP32 adjoint at the matching rounded Query |
| Backward | `W (dS/z)ᵀ` | `C × U` by `U × D` | Key gradient from state update | Strict |
| Backward | `K (dS/z)` | `C × D` by `D × U` | Future-state contribution to `dW` | Strict, original Key |
| Backward | `-(αZ) Wᵀ` | `C × U` by `U × C` | Strict-lower Key-Gram adjoint summed over Value channels | Strict |
| Backward | `(dG+dGᵀ) K` | `C × C` by `C × D` | Key gradient through Gram | FP32 adjoint at the matching rounded Key if Gram is lowered |
| Backward | `tril(G,-1) W` | `C × C` by `C × U` | Alpha gradient, hence Beta, Key norm, Value and curvature gradients | Strict, recomputed forward Gram |
| Backward | `Kᵀ(αZ)` | `D × C` by `C × U` | Reverse state propagation | Strict |
| Backward | `(αZ)Sᵀ` | `C × U` by `U × D` | Key gradient through the residual | Strict |

The forward has six GEMMs per chunk/Value tile. Backward has six replay GEMMs and fourteen adjoint GEMMs. These counts exclude the scalar/vector triangular substitutions, normalization, prefix sums, load/store work and Python gradient reductions. The projected keys, bank inverses and protected operators are outside this proposal.

## Forward equations and precision boundaries

For one chunk and Value channel, define

```math
a_{iv}=\|k_i\|^2-cv_{iv}^2,\qquad \alpha_{iv}=\beta_{iv}/a_{iv},
```

```math
L_v=I+\operatorname{tril}\!\left(\operatorname{diag}(\alpha_v)
 (KK^\top-cv_vv_v^\top),-1\right),\quad
L_vW_v=\operatorname{diag}(\alpha_v)(v_v-KS_v).
```

```math
z=1+c\operatorname{cumsum}(v\odot W),\quad
O=\mathrm{scale}\,(QS+\operatorname{tril}(QK^\top)W)/z,\quad
S'=\left(S+K^\top W\right)/z_{\mathrm{last}}.
```

The initial experiment would round only the operands used by `QKᵀ`: `P = tril(Qr Krᵀ)`. Original `Q` remains in `QS`; original `K` remains in the residual, norm, Gram and state update. The forward state is consequently independent of this particular precision choice. This exact dependency statement does not imply unchanged readout gradients or long-training behavior.

A separate Gram experiment would use `Gr = Kg Kgᵀ` in the strict lower triangle of `L`. It must leave `a` computed from original FP32 Key; replacing the norm with a lowered Gram diagonal silently changes another numerical operation. Gram perturbations affect `W`, `z`, every later state and every supported gradient.

Neither experiment lowers `a`, `α`, canonical-state storage, normalization denominators, triangular accumulators or curvature products. FP16's narrower exponent range and BF16's coarser mantissa are separate risks. FP16 is a plausible first choice for the normalized original Q/K supplied by the layer; this says nothing about projected keys, states, adjoints, or arbitrary low-level inputs.

## Complete VJP and the cast-contract question

Write `D = scale dO/z`, `E = dS'/z_last`, `Pbar = tril(DWᵀ)`, and

```math
\rho=\operatorname{reverse\_cumsum}(-dO\odot O/z)
 -\sum_D(dS'\odot S')/z_{\mathrm{last}}.
```

The subtraction broadcasts the endpoint normalization contribution to every token. Then

```math
dW=P^\top D+KE+cv\odot\rho,\quad L_v^\top Z_v=dW_v,
\quad U=\alpha\odot Z,\quad Gbar=-\operatorname{tril}(UW^\top,-1).
```

The Query and Key adjoints require separate edges:

```math
dQ=DS^\top+Pbar\,K_r,
```

```math
dK=WE^\top-US^\top+2K\sum_v da
 +Pbar^\top Q_r+(Gbar+Gbar^\top)K_g.
```

For QK-only rounding `Kg = K`; for strict arithmetic all three rounded-edge operands equal their original operands. Reusing original `K` for the rounded-Gram derivative or original `Q` for the rounded-QK derivative does not differentiate the proposed bilinear operands.

For completeness, let `p = exclusive_cumsum(vW)` and `t = exclusive_reverse_cumsum(Uv)`. The remaining gradients are

```math
d\alpha=Z\odot\left(v-KS-\operatorname{tril}(G,-1)W+cv\odot p\right),
\quad da=-d\alpha\odot\beta/a^2,
```

```math
dv=cW\odot\rho+c(U\odot p+W\odot t)+U-2cv\odot da,
\quad d\beta=d\alpha/a,
```

```math
dS=Q^\top D+E-K^\top U,\qquad
dc=\sum(vW\rho+Uv p-da\,v^2).
```

These formulas avoid division by `α` and remain defined for `beta=0`; no transition inverse is needed at `beta=1`. Applying the boundary state adjoint in reverse chunk order yields the full sequence VJP.

The proposed numerical contract still needs an explicit design decision. An FP32 analytic adjoint evaluated at rounded matrix operands is different from differentiating an actual PyTorch `float32 -> float16 -> float32` cast graph: PyTorch also rounds cotangents when they cross the half-typed intermediate. The CPU suite checks these as distinct operations. It does not use a hidden stop-gradient construction or assert that quantization is differentiable. Its finite-difference test treats the already rounded operand values as independent floating inputs. No straight-through quantization policy has been approved or installed in the operator.

## Safety evidence and rejected shortcuts

Normalized Key is insufficient to certify lowered Gram. A deterministic valid-domain example uses `T=32`, `K=16`, `V=1`, identical normalized keys, `beta=1`, `c=1`, `v=0.9999`, and zero initial state. The strict FP64 equations retain state norm approximately `0.9999`. Replacing only the Gram operands with FP16 or BF16 rounded values produces state norm approximately `1.00010001`, outside the Klein ball; the FP16 experiment also produces a negative homogeneous denominator. `test_reduced_gram_can_leave_the_feasible_domain` preserves this counterexample.

Therefore a norm/range-only dispatch guard is rejected. Any future reduced-Gram proposal must address the feasible-domain margin, solve conditioning, normalization sensitivity and long-sequence error propagation. The layer's default value margin bounds `a`, but low-level callers have a wider contract. A safe fallback must preserve that wider contract. The need for condition/range checks also belongs in the performance model, including repeated checks across Value tiles.

An input guard must execute on the GPU or rely on an already enforced caller contract; adding a host `.item()` synchronization to every chunk would be unsuitable. Its branch and selected policy must match in forward and backward. A stored autograd policy prevents environment changes between forward and backward from selecting different arithmetic. A selective policy must not accidentally inherit global `TC_FAST=1` for state or adjoint GEMMs.

## CPU measurements and their limits

The independent test suite executes 103 tests on CPU with PyTorch `2.8.0+cpu`. It checks all nine separate mathematical input edges, rounded FP16/BF16 operand values, `T=1/16/32`, asymmetric K/V, `c=0/0.5/2`, `beta=0/1/random`, nonzero initial state, first-order gradients, a finite-difference gradcheck, QK-only state independence, cast-cotangent behavior and the feasible-domain counterexample. These are mathematical tests, not tests of a new production dispatch or Triton implementation.

The reproducible study uses FP64 algebra and FP64 analytic adjoints with only QK/Gram operand rounding. It compares against strict FP64 WY arithmetic, using the existing TC FP32 forward/state budgets (`atol=8e-4`, `rtol=8e-3`), their existing threefold gradient budgets, and the existing curvature-gradient budget (`5e-3`, `5e-2`). It does not emulate TF32x3, FP32 accumulation, GPU rounding or a cast graph's cotangent storage. The scale is 1, as in the layer, and the initial state is nonzero.

| T / K / V / C | QK FP16 | QK+Gram FP16 | QK BF16 | QK+Gram BF16 |
| --- | --- | --- | --- | --- |
| 17 / 16 / 8 / 16 | Passes CPU rounding budget | Passes CPU rounding budget | Passes CPU rounding budget | Passes CPU rounding budget |
| 257 / 64 / 16 / 32 | Passes CPU rounding budget | Passes CPU rounding budget | **Rejected: dV** | **Rejected: dV** |
| 4096 / 128 / 16 / 16 | Passes CPU rounding budget | Passes CPU rounding budget | Passes CPU rounding budget | Passes CPU rounding budget |
| 16384 / 128 / 16 / 16 | Passes CPU rounding budget | Passes CPU rounding budget | Passes CPU rounding budget | **Rejected: dV** |

For `T=257`, the two rejected BF16 candidates have maximum absolute dV errors approximately `0.00566` and `0.00505`. At `T=16384`, the rejected BF16 Gram candidate has maximum absolute dV error approximately `0.00434`. Each rejection is determined elementwise against its absolute-plus-relative budget; a small global relative RMS error does not override a failed elementwise check. No threshold was relaxed. FP16 passing these sampled CPU studies is insufficient for GPU approval, and the near-boundary counterexample still rejects unconditional Gram lowering.

Reproduce the evidence from the repository root:

```bash
FLA_CI_ENV=0 python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_precision.py
python scripts/analyze_hyperbolic_precision.py --json profile/hyperbolic-opt/precision-cpu.json
python scripts/analyze_hyperbolic_precision.py --require-all-pass --json profile/hyperbolic-opt/precision-cpu-gate.json
```

The last command deliberately exits nonzero because some candidate policies fail; the first study command records all candidates without presenting a completed study as universal numerical acceptance. Raw JSON includes per-output errors, error/budget ratios, seeds, versions and revision metadata.

## Performance model and required GPU work

Changing input precision does not remove any mathematical FLOPs or launches. With `C=16`, `D=128`, `U=16`, the current forward GEMMs perform about 335,872 scalar FLOPs per chunk/Value tile and backward GEMMs about 966,656. The two QK evaluations account for about 10.1% of these combined GEMM FLOPs; lowering both QK and Gram would affect about 20.1%. These are arithmetic counts, not runtime fractions. Tensor Core instruction throughput, conversions, layouts, the triangular loop, memory traffic and register pressure can change the measured fraction substantially.

The proposed QK experiment replaces two `C × D` by `D × C` accurate input products per forward/backward pair with native low-precision input products. FP32 adjoint products remain accurate. No extra full-sequence operand buffers are inherently required; tile-local casts can still add instructions and live registers. Sharing Gram/QK across Value tiles in a staged architecture removes repeated work first, so the benefit of precision changes cannot be multiplied by the benefit of staging. Guard reductions and recomputing matching rounded operands must be included in the timed path.

After an RFC is available and the contract is accepted, implement strict versus selective policies as explicit, independently selectable paths. Validate all current dtype, packed, empty, tail, initial/final-state and curvature-gradient contracts, then add layer-shaped FP32 Q/K with BF16 autocast, long sequences, small/large input scales, beta endpoints and near-boundary conditions. Preserve the current numerical thresholds. Query storage alone must not attest to the executed GEMM policy.

GPU correctness must precede performance claims. Measure forward, backward and their combined time, full model training with checkpointing, allocated/reserved memory, raw repeated timings, and actual dispatch. Compare strict/global-fast/selective policies separately; BF16 and FP16 experiments are numerical trade-offs, not pure scheduling speedups. The current existing global-fast control can already be exercised with `FLA_HYPERBOLIC_TC_FAST=0/1`; it is not an implementation of this RFC.

**GPU NOT RUN. No selective-precision kernel, speedup or production readiness is claimed.**
