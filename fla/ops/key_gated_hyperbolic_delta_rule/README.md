# KHDN device operator

The strict implementation described below is the historical correctness route. BF16 training is selected explicitly with `precision_policy="bf16"`; it rejects unsupported inputs and never falls back to FP64.

`chunk_key_gated_hyperbolic_delta_rule` implements the reviewed strict-past, positive-mass chunk graph in Triton. `no_carry` is the default. `y_carry` prepares shared next-block E/H coefficients, advances both S and Y, and folds E/H/target source derivatives into the next block's F owner. The public cache contains canonical S only. Calls reinitialize Y from S.

This is an initial correctness implementation. Its device arithmetic is FP64, including when public inputs/outputs are FP32, BF16 or FP16. It is not an FP32 internal kernel or a Tensor Core performance result. The initial numerical evidence is recorded in `evidence/production-first-report.md`; the subsequent compiler-layout repair and its validation are recorded in `evidence/race-repair-report.md`.

## Interface and routes

Inputs are q/k/log_decay `[B,T,H,K]`, v `[B,T,H,V]`, beta `[B,T,H]`, scalar curvature, and optional canonical state `[N,H,K,V]`. Packed calls require B=1 and int32/int64 cumulative lengths, including repeated boundaries. Value group size divides V; padding the execution tile does not change curvature/group_size. The production Layer uses g1/16/32 and chunk sizes 16/32/64. Smaller power-of-two chunks through C1 are available for the recurrence specialization.

Output follows q.dtype. State is FP64 if any source is FP64 and FP32 otherwise. Source gradients follow each original source dtype. Noncontiguous sources are made contiguous by the wrapper without detaching gradients. Empty sequences preserve S and its gradient, with zero other source gradients. An unrequested public final state is neither constructed nor cast. Only first-order derivatives are supported; `create_graph=True` raises explicitly.

`fused_recurrent_key_gated_hyperbolic_delta_rule` specializes this graph to C1: feedback is scalar and one program scans each sequence/head/group. It supports training and decode with the same state contract. `checkpoint_size` is accepted for Layer compatibility; the current implementation saves every token entry state, regardless of that parameter. Checkpoint replay thinning is pending.

Range-sensitive inputs are screened before preparing coefficients. Raw Key norms outside `[2^-80,2^80]`, feedback ratio above .25, entry ball squared radius above .5, or nonzero FP64 source amplitudes outside `[2^-100,2^100]` use the **reviewed host chunk route**. Original sources are transferred before precision selection. Extreme cotangents (nonzero magnitudes below `2^-100` or above `2^100`) cause host backward replay using original sources and the complete reviewed source VJP. Nonfinite requested device results also replay on the host; representability errors are reported only for requested public values/sources. There is no clipping.

These are conservative implementation routing boundaries, not mathematical domain restrictions or a proof of uniform device accuracy throughout the accepted region. Host exceptions may use arbitrary precision and preserve finite valid results; they incur synchronization, transfer and CPU costs. They never count as pure GPU validation or throughput. Telemetry distinguishes `triton_chunk`, `triton_recurrent`, `empty_identity`, and `host_reviewed_chunk_exception`, including backward-only host replay. The wrapper also synchronizes for validation and range/finite checks. CUDA graph capture and asynchronous dispatch are not claimed.

## Stages and gradient owners

| Stage               | Parallel owner         | Work and source ownership                                                                                                               |
| ------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Common preparation  | chunk/head             | Local interval gates; unit-diagonal L inverse; F, strict-past output/state coefficients; beta products                                  |
| Carry preparation   | chunk/head             | E/H from current state coefficients and next F; no cross-document edge                                                                  |
| Group preparation   | chunk/head/Value group | Value Gram, zero-state gauge, non-unit Zhat inverse, af/Jf                                                                              |
| Forward scan        | sequence/head/group    | Canonical S, actual Y when enabled, positive masses, direct residual output/state                                                       |
| Reverse scan        | sequence/head/group    | Recompute local graph from saved S/Y; normalize cotangents; explicit group feedback VJP; dV/dS0/group dc and shared-owner contributions |
| Common source VJP   | chunk/head             | Deterministic group sum, next-F owner, one shared inverse VJP, endpoint-safe beta products, both interval endpoints, dq/dk/dbeta/dell   |
| Curvature reduction | scalar                 | FP64 device reduction of every sequence/head/chunk/group owner, then source-dtype cast                                                  |

The output and group VJPs are currently fused into the reverse state program; extracting their state-independent work into separate parallel kernels remains an optimization. No recurrent backward or generic inverse autograd substitutes for the C>1 chunk graph. No dense K-by-K state transform is materialized.

The inverse implementation uses explicit forward substitution with its actual diagonal. The group backward applies the saved inverse transpose to the w cotangent and uses the implicit solve source VJP. It does not differentiate an inverse algorithm. The common A VJP executes once after group reduction.

Triton 3.5 can lower replicated FP64 reduction layouts to overlapping shared-memory writes. The af matvec therefore runs in a separate kernel with one scalar output owner per chunk/head/group/row, reading the existing FP64 inverse, nu and eta buffers. C1 uses a scalar beta×aq broadcast and scalar extraction when changing the Key-reduction axis into a row. These boundaries preserve the source equations and add no preparation buffers. They add a kernel launch and C1 scalar-selection work; performance is not established by correctness checks.

## Precision and liveness

| Operation                                    | Load/operand precision       | Accumulator/intermediate                              | Store                                          |
| -------------------------------------------- | ---------------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| Source loads                                 | Original source dtype        | Explicit promotion to FP64                            | Original sources retained                      |
| Gate intervals and exponentials              | FP64                         | FP64 cumsum, CUDA libdevice exp                       | Common buffers FP64                            |
| Key/Value Grams and every matrix product     | FP64                         | FP64 SIMT outer-product accumulation; no Tensor Cores | FP64                                           |
| L and Zhat solves                            | FP64 including diagonal      | FP64 forward substitution/division                    | FP64                                           |
| Gauge, masses, state and residuals           | FP64                         | FP64                                                  | FP64 saved entries; public cast only at output |
| Reverse normalization and all source algebra | FP64 cotangents/coefficients | FP64                                                  | Owners FP64; final source-dtype cast           |
| Group/curvature sums                         | FP64                         | Deterministic group order; device FP64 scalar sum     | Source dtype                                   |

For padded execution widths C/K/G, common storage is `11CK + 7C² + 5C` FP64 elements per chunk/head. Group preparation is `5C² + 3C` per chunk/head/group. Saved entry S/Y require `KG + CG` per chunk/head/group. Carry storage is `CK + C²` per chunk/head (also allocated on no_carry in this initial implementation). The reverse owner buffers require `4C² + 6CK + 5C` per chunk/head/group, plus one curvature owner. Metadata, original sources, public results and source gradients are additional.

These buffers live through backward. Local output/mass/state residuals are recomputed from the saved S/Y and full-precision prepared coefficients; no rounded public output is used to reconstruct a residual. C1 recurrent saves all entry states. Relative decay uses a C-by-K register tile per left endpoint; there is no global C²K interval or interval-gradient scratch. Matrix multiply uses SIMT FP64 and can spill registers; compiler/profile data must guide any subsequent tiling or precision work. Live-buffer accounting is not a measured peak-memory or speed claim.


## Experimental C128 BF16 execution

The public chunk entry accepts `precision_policy="bf16", chunk_size=128, carry_mode="no_carry"` for K=128, mathematical group size 32, and V in {32, 64, 128}. The Layer/config entry supports K=V128. This path requires the pinned SM90/PyTorch/Triton runtime checked by `chunk128.py`; other chunk sizes retain their existing routes. The default precision and checkpoint schema are unchanged.

C128 is the state execution chunk. Internal 16/32/64 tiles prepare its full block system; they do not advance canonical state through eight separate C16 updates. Output forward/backward and parameter VJPs run separately from the state scans. Hierarchical triangular inversion retains three BF16 products, while the main contractions use one BF16 product with FP32 accumulation. Canonical state, scalar controls and normalization remain FP32.

The training Layer fuses query/key normalization, the Value transform, beta and Key-channel gates. Bounded gauges reconstruct the relative rho matrix from saved scales; inputs outside that bound retain the materialized FP32 representation. Source validation and finite checks remain active. Trainable curvature retains its source gradient; fixed curvature omits only work exclusively needed for that gradient.

`benchmarks.fp32_adamw.FP32AdamW` provides device FP32 bias corrections, moments and updates for contiguous CUDA FP32 parameters with unscaled gradients. The training benchmark selects this optimizer for its BF16 KHDN policy; matched controls can select `--optimizer-backend fp32` explicitly. Optimizer writes advance tensor version counters, including after checkpoint restoration.

C128 is still under qualification. The native small-model update/restore check passes, but current operator tests retain gradient elementwise failures on some dense/packed inputs. The C128 compiler trace and pre-update graph checks support checkpointed and non-checkpointed execution. Public CLI validation, continuous full-model quality, long training and distributed acceptance remain incomplete. Initial heldout PPL differences beyond 0.1% are recorded as observations; they do not by themselves establish or reject training quality. Research timing does not establish those qualifications.
