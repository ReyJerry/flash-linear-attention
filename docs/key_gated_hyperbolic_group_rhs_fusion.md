# Exact Group RHS Fusion

## Scope

The selective BF16 KHDN path constructs one lower triangular group system per
chunk and value group. Let `Z` denote the lower triangular group system; the current production
path materializes its inverse action as `Gp[0]`, let `d`, `nu`, and `beta` be
the per-token diagonal factors, and let `h` be the old-state/value feedback
vector. The current implementation forms

\[
a_f = d\odot\nu\odot\beta, \qquad
r = d\odot\nu\odot\left(h+\beta\odot P h\right),
\]

where `P` is the strict exclusive-prefix operator. It then evaluates

\[
w = Z^{-1}a_f - \kappa Z^{-1}r.
\]

This is exactly equal, in real arithmetic and in the same FP32 kernel
arithmetic up to the reassociation of two additions, to

\[
\boxed{
w = Z^{-1}\left[d\odot\nu\odot
\left(\beta-\kappa\left(h+\beta\odot P h\right)\right)\right].
}
\]

The fused route therefore performs one action of the same triangular inverse
instead of two, removes the materialized feedback vector and removes the
`group_feedback_tiles` launch. It does not change `Z`, the mass scan, the
state recurrence, the output readout, or any VJP formula. The legacy route is
kept behind `group_rhs_fusion=False` for differential debugging.

## Why the transformation is safe

The only identity used is linearity of the fixed matrix action:

\[
Z^{-1}a_f-\kappa Z^{-1}r
=Z^{-1}(a_f-\kappa r).
\]

`Z`, `d`, `nu`, `beta`, `h`, and `P h` are already computed before the state
kernel consumes `w`. Consequently the transformation does not move a state
dependent operation across a recurrence boundary and does not assume that the
per-value systems are shared. Differentiating the boxed expression gives the
same VJP by the chain rule; the CPU proof test compares all five independent
inputs (`d`, `nu`, `beta`, `h`, and `kappa`) against the two-action expression.

## Expected work reduction and limits

For every group/chunk, the algebraic work decreases from two triangular
matrix-vector actions to one, plus one less global write/read of a length-`C`
vector and one less kernel launch. This is a guaranteed reduction in the
candidate's operation count. A GPU speedup is still an empirical property:
register pressure, occupancy, and the cost of the preceding inverse build can
hide the reduction. The GPU script must compare both routes on the same
qualified H20 runtime before the optimization is called a measured speedup.

## Validation status

* CPU exact-value and VJP proof: `tests/ops/test_key_gated_hyperbolic_group_rhs_fusion.py`.
* GPU differential and timing script: `benchmarks/benchmark_khdn_group_rhs_fusion.py`.
* The H20 run is intentionally not claimed in environments without the
  qualified CUDA/Triton runtime.
