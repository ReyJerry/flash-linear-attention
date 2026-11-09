# C64 state-graph research and finite-validation ownership

These are opt-in research routes for canonical grouped KHDN, with mathematical group width 32 and K=V128. They retain inclusive readout, Key-channel decay and independent group feedback. No default, numerical tolerance or model checkpoint format changes. The full engineering goal remains incomplete.

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

| Shape | Strict KHDN step (ms) | Deferred KHDN step (ms) | Matched KDA step (ms) | Deferred throughput / KDA |
| ----- | -------------------: | ---------------------: | -------------------: | ------------------------: |
| B8×T2048 | 1891.730 | 1844.700 | 1420.427 | 77.000% |
| B1×T16384 | 2017.265 | 1967.730 | 1455.807 | 73.984% |

Deferred validation improves throughput by 2.55% and 2.52%, respectively. All 459 initial parameter-gradient hashes and every sample's loss and batch identity agree between strict and deferred modes for both shapes. Peak allocated is 47,016,119,296 versus 47,016,131,584 bytes; reserved memory is identical within each pair. These are short research screens, not the formal timing protocol or training-quality acceptance.

The same-source short-shape CPU/CUDA profiles retain 72 finite-scan kernel calls in each KHDN mode. Actual device-to-host copies fall from 121 to 50, and stream synchronizations from 123 to 52. Summed GPU kernel duration stays approximately 1792.6/1792.9 ms, versus matched KDA's 1414.3 ms. This supports a synchronization benefit while leaving the additional attention computation unresolved; profiler sums are not training-step latency. Each profiled arm performs three actual updates and matches all 459 clipped-gradient hashes to its unprofiled reference.

The [evidence bundle](evidence.json.gz) preserves the 168 raw timing observations, identities, compiler-resource traces and independent verification summaries. Full compressed Chrome traces remain in the experiment archive, identified by SHA256 in that bundle.

## Evidence limits

The measured source snapshot for native scope integration is `074e6c33bca0ea66d45532b1f8c3abf7dbdcebbbc65e209d2c528068027a30c9`, based on commit `256826f3b015fe600d5f710cd93e3e7e97c6db2f` plus inventoried working files. The pack is verified before and after extraction, including regularized symlink bytes, file modes, manifest and runner entries. One initial native timing runner failed because it omitted `SCREEN_VARIANT`; its 11-case native GPU gate passed, but it supplied no full-model timings.

The historical 1900.263 ms / 75.22% comparison is not final-source performance qualification. Formal four-shape repeats, continuous training and unseen quality, cache/packed model regressions, recovery and 1/8/16-GPU acceptance remain open. BF16-first training, no FP64 training/fallback, no six-product compensation and the original failed fixtures remain required.

## Pending direct-state guard preparation

The current opt-in direct-state prototype additionally prepares an eight-column-block L1 bound from the shared strict inverse. At scan time it multiplies each block's bound by the largest residual-operand row norm in that block. The bound is conservative, uses the existing cancellation threshold, and can select the stable fallback more often. Its purpose is to remove the full C×C weighted-bound live range from the state loop, at a cost of 8 MiB of shared preparation data for the main operator shape.

The pending candidate snapshot is `3e4ee578119182a2469a01b2ee0d0726ce532d4ac9701f9833b875e59d57d6ac`. Its CPU bound checks pass; the GPU gate and complete-step comparison are in progress. The earlier passing direct-state GPU tests and the measured native-scope timing do not qualify this new combination. It remains opt-in; no direct-state default promotion or speed claim is made.
