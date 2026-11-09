# C64 state-graph research and finite-validation ownership

These are opt-in research routes for canonical grouped KHDN, with mathematical group width 32 and K=V128. They retain inclusive readout, Key-channel decay and independent group feedback. No default, numerical tolerance or model checkpoint format changes. The full engineering goal remains incomplete and paused; this publication records existing results only.

## State and output grouping

The bounded C64 gauge route can batch shared left contractions across physical Value tiles of 64 or 128, while keeping the mass system, feedback, normalization and scalar coefficients independent for each mathematical group. FP32 queries and materialized-rho inputs explicitly use the original group-width-32 route; telemetry records the fallback. A fallback does not qualify the wide implementation.

The same-card 24-layer B8×T2048 screen uses `coupled_fused`, `save_mm`, BF16 operands with FP32 controls/state/AdamW, two warmups and five steady samples per F/B/F+B/step mode. The 36-case GPU gate and four tiny real updates passed. Complete-step medians were:

| Physical output / state tile | Step (ms) |
| ---------------------------- | --------: |
| 32 / 32                      | 1899.478  |
| 64 / 32                      | 1933.027  |
| 64 / 64                      | 2016.290  |
| 128 / 32                     | 1917.951  |
| 128 / 128                    | 2128.485  |

The wide candidates are archived research options, with no default promotion or training-speed claim. Increasing the physical tile did not yield a whole-step win in this implementation.

## Direct state residual

The explicit `state_residual_mode='direct'` research route uses the identity `A L = I` to form the residual from the target transform and `N Y`. The unit diagonal of A stays exact in FP32; the strict-past contraction uses a single BF16 product. A coefficient-weighted cancellation bound selects the stable residual expression for sensitive inputs. It retains the feedback and mass-system adjoints and all direct endpoint, target, erase and normalization sources.

Forty untimed CPU cases compare canonical recurrence, the stable block graph and the direct graph, including all seven source gradients. The GPU research gate covers fixed curvature, six differentiable sources, tails, initial state, repeated Keys, β endpoints, fixed points, strong decay and raw margins. Trainable curvature continues to use the original implementation; CPU seven-source agreement does not qualify a new trainable-curvature GPU route.

The prepared-layout combination passed 22 focused GPU cases and a real tiny update. Its paired complete step changed from 1885.993 to 1910.456 ms. A later direct/V64 combination passed the same gate and memcheck, but measured 2015.796 ms against a 1890.113 ms stable baseline. These are regressions, so `state_residual_mode='stable'` remains the default. The producer/consumer fusion, removal of Delta/Pi global exchange, tape-lifetime redesign and a new state backend have not been promoted on this evidence.

## Explicit finite-validation scope

`FiniteValidationScope` collects the existing CUDA finite-check flags and reads their combined status before an associated optimizer update. Source/domain admission remains immediate, and the GPU finite scans still run. It changes the ownership and timing of error reporting, so it is explicit per call; `strict` remains the default.

```python
from fla.ops.key_gated_hyperbolic_delta_rule.validation import FiniteValidationScope

optimizer.zero_grad(set_to_none=True)
with FiniteValidationScope(optimizer) as checks:
    loss = model(input_ids=tokens, labels=tokens, use_cache=False, validation_scope=checks).loss
    loss.backward()
    optimizer.step()
```

The model must use gradient-enabled `bf16_selective` chunk training. The scope must stay open through checkpoint recomputation and backward. The optimizer pre-hook commits pending flags before its update; a failure leaves parameters and optimizer state unchanged. Normal scope exit also checks pending flags without an update. `flush()` validates pending flags while leaving the scope open; scopes cannot be reused. Hybrid attention blocks and inference calls reject this new option. The scope is a caller-owned transient object, never a model/config/checkpoint field. It validates KHDN operator outputs and source gradients, not arbitrary external model operations.

The native model gate passes 11 GPU cases: strict/deferred loss, all parameter gradients and actual update parity with checkpoint off, recompute and `save_mm`, plus nonfinite-update rejection and unsupported-usage checks. The finite-validation implementation separately passes four CPU lifecycle tests. The native full-model screen uses the same source, policy and hardware within each shape, with 2 warmups and 5 steady samples. It includes real FP32 AdamW updates and commits validation inside the timed window.

| Shape     | Strict KHDN step (ms) | Deferred KHDN step (ms) | Matched KDA step (ms) | Deferred throughput / KDA |
| --------- | --------------------: | ----------------------: | --------------------: | ------------------------: |
| B8×T2048  | 1891.730              | 1844.700                | 1420.427              | 77.000%                   |
| B1×T16384 | 2017.265              | 1967.730                | 1455.807              | 73.984%                   |

Deferred validation improves throughput by 2.55% and 2.52%, respectively. All 459 initial parameter-gradient hashes and every sample's loss and batch identity agree between strict and deferred modes for both shapes. Peak allocated is 47,016,119,296 versus 47,016,131,584 bytes; reserved memory is identical within each pair. These are short research screens, not the formal timing protocol or training-quality acceptance.

The same-source short-shape CPU/CUDA profiles retain 72 finite-scan kernel calls in each KHDN mode. Actual device-to-host copies fall from 121 to 50, and stream synchronizations from 123 to 52. Summed GPU kernel duration stays approximately 1792.6/1792.9 ms, versus matched KDA's 1414.3 ms. This supports a synchronization benefit while leaving the additional attention computation unresolved; profiler sums are not training-step latency. Each profiled arm performs three actual updates and matches all 459 clipped-gradient hashes to its unprofiled reference.

The [evidence bundle](evidence.json.gz) preserves the 168 raw timing observations, identities, compiler-resource traces and independent verification summaries. Full compressed Chrome traces remain in the experiment archive, identified by SHA256 in that bundle.

## Evidence limits

The measured source snapshot for native scope integration is `074e6c33bca0ea66d45532b1f8c3abf7dbdcebbbc65e209d2c528068027a30c9`, based on commit `256826f3b015fe600d5f710cd93e3e7e97c6db2f` plus inventoried working files. The pack is verified before and after extraction, including regularized symlink bytes, file modes, manifest and runner entries. One initial native timing runner failed because it omitted `SCREEN_VARIANT`; its 11-case native GPU gate passed, but it supplied no full-model timings.

The historical 1900.263 ms / 75.22% comparison is not final-source performance qualification. Formal four-shape repeats, continuous training and unseen quality, cache/packed model regressions, recovery and 1/8/16-GPU acceptance remain open. BF16-first training, no FP64 training/fallback, no six-product compensation and the original failed fixtures remain required.

## Pending direct-state guard preparation

The current opt-in direct-state prototype additionally prepares an eight-column-block L1 bound from the shared strict inverse. At scan time it multiplies each block's bound by the largest residual-operand row norm in that block. The bound is conservative, uses the existing cancellation threshold, and can select the stable fallback more often. Its purpose is to remove the full C×C weighted-bound live range from the state loop, at a cost of 8 MiB of shared preparation data for the main operator shape.

The prepared-guard candidate snapshot is `3e4ee578119182a2469a01b2ee0d0726ce532d4ac9701f9833b875e59d57d6ac`. Its CPU bound checks, 22-case GPU gate and one real tiny update pass. On the same full-model research screen, stable is 1841.618 ms and prepared direct is 1854.747 ms, a 0.71% time regression; direct fallback selected 85,516 of 786,432 group-chunk entries. It is archived and stable remains the default. The earlier passing direct-state GPU tests and native-scope timing do not qualify this new combination.


## Cost audit

Actual staged KHDN graph cost audit, commit fba02a4c, C64/K128/g32/V128.
This is nominal arithmetic/source analysis, NOT a latency prediction or lower-bound proof.

Forward per mathematical group:
Stable: Y=F*S; U=w*V-N*(L*Y); tail=P17*Us, P17 already contains the A precomposition.
Direct: Y=F*S; Delta=w*V+N*strictLower(A)*(N^-1*w*V)-N*Y; tail=rawP*Delta_s.
Both request three primary dense contractions: (C,K)x(K,g), (C,C)x(C,g), (K,C)x(C,g).
Both output stages request two dense contractions: qer*S and To*U versus qer*S and rawCr*Delta.
The direct graph deletes common To=rawCr*A and P17=erasedRawP*A preparation, shared once across all groups. It does not delete an additional explicit A*U from the existing online forward, because that multiplication was already precomposed.
The direct cancellation branch can request extra stable work, and preparing/using bounds and raw erase terms costs time/registers.

State/source backward per group, ignoring other unchanged work:
Stable coupled: Ahat^T*(b*V) to form Pi, B^T*(N*Pi) for the state source, then Ahat*U to reconstruct Delta in the producer.
Direct: Ahat^T*gDelta for the w adjoint, F^T*N*(gDelta+kappa*b*V) for the state source, then Ahat^T*(gDelta+kappa*b*V) to supply Pi to source consumers.
The residual-only identity gY=-N*gDelta removes the residual transpose chain algebraically. The w(Y)/mass feedback and final raw-source Pi still need their contractions. In the current staged implementation the above sets each contain three contractions, not three versus one. Shapes differ and scheduling changes, so performance can still change; measured backward improvement is legitimate, but it is not evidence that the entire backward graph lost all those products.

Shared Value batching:
Four (M,R)x(R,g32) contractions and one (M,R)x(R,V128) contraction both have 2*M*R*V FLOPs. Batching changes launch/left-operand reuse and matrix utilization, while retaining group-specific systems and feedback. A fourfold reduction in call count is not a fourfold arithmetic saving. Larger state/adjoint tiles can lose occupancy or spill, as the actual wide candidates demonstrate.

Main shape total tokens16384, heads16, C64:4096 head/chunks and4groups each.
Primary online forward dots per group = 2*C*K*g +2*C*C*g+2*K*C*g =1,310,720 FLOPs.
Primary output dots per group =2*C*K*g+2*C*C*g=786,432 FLOPs.
Two deleted shared precompositions per head/chunk =2*C^3+2*K*C^2=1,572,864 FLOPs, about6.442GFLOPs/operator forward at this shape (single BF16-product nominal arithmetic).
These savings were shared across groups already; they should not be multiplied again by4. FP32 operations, scans, inverse, tensor conversion, recomputation, memory exchanges, guards and host dependencies remain outside these dot counts.

Consequence: The identities and group batching are valid candidates, but the initial large-throughput expectation double-counted work already removed by staging/precomposition. A real next structural change needs a new explicit whole-graph cost budget. No fixed low-rank log-depth scan, architecture impossibility, default promotion or final acceptance follows from this audit.


## Current structural status after the rank-one screen

The [Sherman–Morrison experiment](../khdn_sherman_morrison_2026_11_09/README.md) now has completed two-endpoint full-step evidence. Its canonical rank-one audit, CPU seven-source checks, eight H20 cases and a tiny learned-curvature update pass, but full-step time regresses by 43.27% / 110.07%. Its GPU code is archived outside the live training path. The block route remains selected; stage 2/3 did not proceed.

| Work item                      | Verified progress                                                                                     | Remaining limitation                                                                                          |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| State/output/source separation | Parallel output and separated source VJPs are implemented; direct-residual algebra has focused checks | State scans still run along each sequence; direct/group candidates have no clear step gain                    |
| Shared preparation             | Common Key/decay matrices are prepared and reused; the whole-graph cost audit is recorded above       | Wider group consumers and interval/system preparation have no demonstrated structural step benefit            |
| Register/layout pressure       | Several split kernels and layouts have measured resource records                                      | Wide candidates lose full-step performance; dedicated H20 backend remains incomplete                          |
| Tape and partial exchange      | Earlier no_carry and tail-partial reductions are retained                                             | Delta/Pi global exchange and state tape remain; SM replay adds no useful allocated-memory saving              |
| Chunk length                   | Research paths support 16/32/64/128; current representative comparisons use C64                       | A larger Chunk alone has no qualified performance advantage                                                   |
| Precision and complete costs   | BF16-first/FP32 controls and an opt-in deferred-validation scope have measured costs                  | Source_vjp defaults to expanded, and formal quality/throughput/recovery/distributed qualification remain open |

The latest same-source short screens retain block KHDN at 1849.929 / 1969.768 ms against matched KDA at 1420.894 / 1456.728 ms, corresponding to 76.808% / 73.954% of KDA throughput. These observations use save_mm in all arms and the provenance caveats in the linked report. They do not qualify this documentation/reference publication as a newly timed production revision or complete the engineering goal.
