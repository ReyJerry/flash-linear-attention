# Block-level Group Action-only Solve

## Route

The selective KHDN group preparation writes a normalized unit-diagonal lower
triangular system `Zhat` into `Gp[1]`. The old route constructs the complete
inverse `Gp[0]` with a diagonal inverse kernel and several off-diagonal inverse
kernels, then applies that inverse in forward and backward.

The action-only route keeps `Zhat` and evaluates only the actions that are
needed:

\[
\operatorname{solve}_{\downarrow}(\widehat Z,r)_i
=\frac{r_i-\sum_{j<i}\widehat Z_{ij}x_j}{\widehat Z_{ii}},
\qquad
\operatorname{solve}_{\uparrow}(\widehat Z^\top,r)_i
=\frac{r_i-\sum_{j>i}\widehat Z_{ji}x_j}{\widehat Z_{ii}}.
\]

The current construction sets `Zhat_ii=1`; retaining the explicit diagonal
operation makes the helper correct if that invariant is changed later. Both
solves run as compile-time loops inside the existing `(sequence, head,
group)` state program. They do not launch one kernel per token and they never
materialize a dense inverse.

For the default fused RHS, the forward action is

\[
r=d\odot\nu\odot
\left(\beta-\kappa(h+\beta\odot Ph)\right),
\qquad
w=\operatorname{solve}_{\downarrow}(\widehat Z,r).
\]

The state backward replaces the old inverse transpose action with

\[
f=d\odot\operatorname{solve}_{\uparrow}(\widehat Z^\top,\bar w),
\]

after which the existing mass-system VJP and parameter VJP are unchanged.
This is the implicit VJP of the same linear solve; it does not differentiate
through a Python or Triton iteration and it does not approximate the gradient.

## Work model and safety boundary

The action route removes the `O(C^3)` inverse construction and the `O(C^2)`
dense-inverse storage for every group/chunk. It replaces each required inverse
matvec with an exact `O(C^2)` substitution. Therefore the algebraic work and
peak group-buffer footprint are strictly lower before considering compiler
behavior. Register pressure can still make a naïve substitution slower on a
specific GPU, so `group_action_solve=False` remains the default until the
qualified H20 script reports a complete-step improvement.

The route is currently admitted only for `C=16/32/64`; `C=128` stays on the
full-inverse route. A failure must be handled by selecting the full-inverse
route before the state/output kernels launch; no per-group or mid-scan fallback
is permitted. The old route remains available for differential debugging.

## Validation status

* CPU solve and transpose-solve proof: `tests/ops/test_key_gated_hyperbolic_group_action_solve.py`.
* GPU correctness and timing: `benchmarks/benchmark_khdn_group_action_solve.py`.
* No H20 run is claimed from an environment without the qualified runtime.
