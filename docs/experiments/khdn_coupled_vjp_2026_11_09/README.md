# Experimental coupled KHDN source VJP

The internal `bf16_selective.chunk_bf16` entry offers `source_vjp='coupled'`, `'coupled_state'` and `'coupled_fused'` for fixed-curvature C64/K128/g32/no-carry calls. `expanded` remains the default, including calls through the public operator, Layer and training CLI. Other chunk sizes and trainable curvature retain their existing implementation. These are research paths, without a production or throughput promotion.

The canonical recurrence, inclusive readout, physical-domain checks, input dtypes and numerical budgets are unchanged. New contractions use at most three BF16 products with FP32 accumulation. Candidate execution rejects FP64; the separate CPU proof explicitly uses untimed FP64.

## Coupled source equations

Let `N=diag(nu)`, `rho_ij=nu_i/nu_j` and `Ahat=A*rho`. The implementation reads rho through the existing qualified materialized/bounded representation. It never introduces division by underflowed absolute scales. Reuse the complete U adjoint `gu` and the scalar mass-system adjoint `b`:

```
Delta = Ahat @ U
Pi    = gu + kappa * Ahat.T @ (b[:, None] * V)
dL    = -tril(rho * (Pi @ Delta.T), -1)
dB    = -(nu[:, None] * Pi) @ S.T
dV    = w[:, None] * Pi + kappa * b[:, None] * Delta
```

Mass beta/n sources retain the endpoint-safe scalar scan. Direct Qer, target, normalization and endpoint erase sources retain their existing owners. Strict-past readout and raw tail sources are generated directly from `Delta`; the raw tail adjoint passes through the existing symmetric endpoint erase before contraction. The endpoint Key/gamma derivative keeps its forward coefficient dependency. The old inverse/precomposition and Value-Gram pullbacks are bypassed, so they are not counted twice.

`coupled` retains the existing state gradient. `coupled_state` replaces its residual contribution by `-B.T @ (nu[:,None]*Pi)` and reuses Pi for local sources. `coupled_fused` additionally combines C64 cross/diagonal preparation sources with B/Qer sources in 32-token owners, then performs the gate scan separately. Unsafe diagonal regions retain local interval evaluation. A temporary Pi/Delta tensor never overwrites autograd-saved state or U; repeated backward allocates fresh work.

The shared fixed-curvature helper caches up to sixteen immutable FP32 device constants across preprocessing and the selective core. Tensor curvature is passed through, including its autograd edge. CUDA events preserve cross-stream readiness; inference creates a normal Tensor, and version/gradient changes invalidate a cached constant. No parameter, model buffer or checkpoint key is added.

## Evidence and limits

Independent CPU checks compare a dense canonical token recurrence, expanded block autograd and the manual seven-source VJP: 19 cases pass, maximum absolute difference `2.22e-15`. A separate source-owner layout proof covers seven lengths/strong-gate cases, maximum difference `1.20e-14`. CPU results do not establish GPU rounding or concurrency correctness.

The original Stage A passed eight focused GPU cases and a real small-model optimizer update. Eight additional public C64/C128 cases passed all seven source budgets, including the final conditional curvature precision fix. Stage B and the fused owner separately passed the same eight focused cases and real updates. Cases include FP32/BF16, dense/packed with empty segments, nonzero state, tail chunks, beta endpoints, repeated Keys, a fixed point, strong gates and a raw near-boundary input; repeated backward is byte-exact. These subsets are not the final regression matrix.

The scalar cache passed its CUDA lifecycle/stream/eviction checks and four byte-exact preprocessing/operator forward-backward comparisons across FP32/BF16. It removes 96 HtoD copies in the measured 24-layer step: 98→2; stream synchronizations fall from 219→123. Scalar reads remain121. Synchronization durations include GPU waiting and must not be interpreted as entirely avoidable overhead.

The focused gate retains output budgets `(atol=.002, rtol=.02, RMS=.01)` and gradient budgets `(.005,.03,.015)`. Fixed-curvature GPU paths request six source gradients; independent mathematical validation also covers curvature, while trainable curvature continues through the original selected implementation. Full final-source regression, quality, inference cache/packed repair, four-shape formal timing and1/8/16-GPU qualification remain open.

## Complete-step screens

All rows below are separate same-H20 comparisons with24layers,1,412,912,512parameters, hidden2048, H16/K=V128/g32, FFN5184, vocabulary32000, ShortConv, plain CE, BF16 autocast, FP32 parameters/AdamW and nonreentrant layer checkpointing. Each arm uses2warmup/5steady samples with real updates. They are research screens, not final10warmup/30steady three-trial measurements. Do not add improvements across rows.

| Graph change | Shape | Expanded step ms | Candidate step ms | Matched KDA step ms | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Local coupled sources | B8×T2048 | 2185.579 | 2174.171 | 1725.301 | Near-neutral |
| Local coupled sources | B1×T16384 | 2299.597 | 2299.547 | 1761.049 | Neutral |
| State adjoint and local sources | B8×T2048 | 2176.313 | 2183.771 | 1735.693 | Near-neutral |
| State adjoint and local sources | B1×T16384 | 2299.221 | 2292.548 | 1770.604 | Near-neutral |
| Previous64-token fused owner | B8×T2048 | 2175.866 | 2219.000 | 1733.607 | Regressed |
| Previous64-token owner plus scalar cache | B1×T16384 | 2298.452 | 2339.761 | 1770.042 | Regressed |

In the short comparison the same64-token fused owner with the scalar cache takes2209.980ms, versus2219.000ms without it. That isolates the cache on this graph, and does not prove the final default-path improvement. The64-token source owner takes127.576ms accumulated over24backward calls and reports255registers/162spills; its32-token replacement is a separate candidate. Compiler spill counts do not measure actual local-memory traffic.

The32-token owner reports6spills and65.255ms accumulated over24source calls, followed by6.764ms in the gate scan. On its own paired H20 screen, short complete step is2158.484→2144.315ms (matched KDA1723.716ms); long step is2278.470→2271.750ms (KDA1759.510ms). Short steady ranges are2156.487–2162.573ms and2143.249–2146.767ms respectively. This is a small measured improvement, without formal repeat qualification. The64-token profile came from a different H20, so its duration is diagnostic rather than a same-card64→32speedup claim.

An independent complete-model screen retains external `aten.mm` outputs during nonreentrant checkpointing; custom operator private tape is still recomputed. Each architecture's459initial parameter-gradient hashes match its ordinary-checkpoint control exactly. Tiny two-step losses, gradients, parameters and Adam states also match.

| Short full24 policy | Step ms | Allocated GiB | Reserved GiB |
| --- | ---: | ---: | ---: |
| Expanded KHDN, fresh scalar | 2173.009 | 27.215 | 31.719 |
| Expanded KHDN, scalar cache | 2165.131 | 27.216 | 31.725 |
| Expanded KHDN, scalar cache and mm-save | 1911.216 | 43.787 | 47.645 |
| Matched KDA, ordinary checkpoint | 1734.185 | 25.584 | 30.158 |
| Matched KDA, mm-save | 1428.567 | 43.781 | 47.250 |

Both architectures use the same64GiBallocated ceiling. KHDN improves11.73% in step time against its cached-scalar control. Its throughput is90.74% of ordinary-checkpoint KDA, but only74.75% of KDA with the same mm-save policy. This does not close the fair-policy architecture gap. A subsequent same-card short comparison measures the actual combination: expanded KHDN with mm-save1913.888ms, coupled32KHDN with mm-save1900.263ms, matched KDA with mm-save1429.324ms. The combined candidate reaches75.22% of equally checkpointed KDA throughput. It is an actual measurement, not the sum of separate gains; final combined-source model qualification is pending.

A first fused-owner attempt failed compilation because `tl.cat` cannot guarantee ordering in the pinned Triton; the working version uses join/permute/reshape. A later submission failed the archive-hash admission check before output-directory creation or GPU execution because upload was incomplete. Both failures are retained in the experiment record, and are not numerical passes.


## Reproduction

Run `python docs/experiments/khdn_coupled_vjp_2026_11_09/verify_math.py` and `verify_layout.py` with CPU PyTorch for the independent mathematical checks. On the pinned H20 runtime, `FLA_CI_ENV=0 pytest tests/ops/test_key_gated_hyperbolic_bf16.py -k "selective_coupled_source_vjp or selective_public_policy_source_vjp or fixed_curvature_cache_lifetime"` exercises the internal research branches and the retained public curvature path. `python docs/experiments/khdn_coupled_vjp_2026_11_09/summarize.py` verifies the462raw timing observations and execution-source hashes in `evidence.json.gz`. `mm_policy.py` contains the research checkpoint policy. The listed subsets and complete-step screens do not qualify final continuous training or distributed execution.
