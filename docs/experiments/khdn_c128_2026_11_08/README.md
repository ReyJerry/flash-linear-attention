# C128 BF16 KHDN research release

This release adds a real 128-token state-execution chunk to the public BF16 KHDN operator, Layer/configuration and training CLI. Mathematical groups remain g32 and canonical state remains FP32. The selected M64 cross-interval preparation changes physical GPU tiles, not the state chunk or mathematical groups.

**The engineering goal and production qualification are incomplete.** The PR stays draft. Original numerical failures remain visible below; neither a successful CLI run nor the timing table means all supported inputs or long training have passed.

## Current behavior

- Common Key preparation is shared across Value groups. Group feedback is prepared before the state scan. Output forward/backward and parameter VJPs run separately from state scans.
- Positive mass uses affine scans. State backward reuses adjoint contractions instead of rebuilding one residual GEMM.
- C128 executes 16 state iterations at T2048. Internal 16/32/64 tiles do not perform eight separate C16 canonical updates.
- Main contractions use one BF16 product with FP32 accumulation. Hierarchical inverse construction retains three BF16 products per contraction. No six-product compensation or FP64 fallback is used.
- Common matrix gradient owners are reduced before storage, selected workspaces use BF16, and bounded gauges reconstruct rho without a stored full matrix. Materialized rho remains available outside that bound. Source/domain/finite validation remains active.
- Training preprocessing fuses Q/K normalization, the Value transform, beta and Key gates. The FP32 AdamW backend includes device FP32 bias corrections and advances tensor version counters after updates.

The new C128 operator supports K128, V32/64/128, g32 and no-carry. Layer/configuration support K=V128. Existing small-chunk/recurrent selections, signatures, model state-dict format and default `strict` policy remain. FP32 optimizer checkpoints carry an explicit backend tag; cross-backend recovery is rejected. C128 Y-carry and a BF16 default transition are not qualified.

The qualified environment is H20/SM90, PyTorch 2.9.0+cu128 and Triton 3.5.0, with compiler SHA256 `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`. Other runtimes are rejected by this experimental path. Explicit untimed correctness references may use FP64; the candidate does not replay through them.

## Same-card full-model short screen

24 layers, hidden2048, 16 heads, K=V128, FFN5184, vocabulary32000, B8×T2048, g32, positive curvature, ShortConv, outer layer checkpointing, plain CE, BF16 autocast and FP32 parameters. All arms use the same FP32 AdamW implementation. KHDN and matched KDA each contain 1,412,912,512 parameters. Matched KDA replaces its 24 output-gate projections to match the KHDN projection/parameter recipe; its attention operator remains KDA, with actual TileLang backward observed.

Each arm has two warmup and three steady samples per mode. F+B is measured directly. Step includes forward/loss/backward/clipping/optimizer and public checks, excluding transfer and zero-grad. Each arm performs five timed optimizer steps and one additional profile step. These are research screens, **not** the final three-trial, 10-warmup/30-steady protocol.

| Arm | Direct F+B ms | Complete step ms |
| --- | ---: | ---: |
| Integrated C128 baseline | 2320.848 | 2343.867 |
| Selected M64 interval layout | 2277.149 | 2299.478 |
| Matched KDA | 1710.886 | 1734.396 |

The selected layout reduces step time by 1.894% in this pair. Its step is 1.32581× matched KDA, corresponding to about 75.43% of KDA throughput. This does not establish parity or an architectural lower bound. Raw arm ID `pair` denotes the M64 interval candidate.

| Kernel, summed over one profiled step | Baseline ms | M64 ms | Registers/thread before → after | Reported spills before → after |
| --- | ---: | ---: | ---: | ---: |
| Cross-interval forward, 48 calls | 34.816 | 30.340 | 255 → 174 | 106 → 0 |
| Cross-interval backward, 24 calls | 76.796 | 35.966 | 255 → 242 | 0 → 0 |
| Common lower-system preparation, 48 calls | 3.583 | 4.334 | 124 → 128 | 0 → 0 |

The new cross-interval kernels emit WGMMA instructions. Padding increases nominal dot arithmetic, while larger matrix tiles and fewer owners improve measured execution. The profile is a separate iteration and is not an exact causal decomposition of the timing medians. Final-source NCU counters and final four-shape timing remain open.

Previously archived Torch fused-AdamW runs remain historical evidence. The installed Torch 2.9 CUDA header declares double bias-correction arithmetic ([source audit](optimizer-source-audit.json)), so those runs do not prove the new whole-step no-FP64 requirement. The current paired table uses the explicit FP32 optimizer for both architectures.

## Numerical and integration evidence

- Eight fixed, NaN-poisoned operator cases pass outputs, final states and all seven source gradients against an explicit untimed CPU FP64 canonical oracle. They cover tails, multiple chunks, packed/empty segments, nonzero initial state, c=0, beta endpoints, strong gates, non-unit Keys, distinct groups and both rho representations. Budgets are unchanged: output/state `(atol .002, rtol .02, relative RMS .01)`; gradients `(.005, .03, .015)`.
- A separate 12-case diagnostic on the repository's original source distribution retains **133 elementwise failures over 5,336,112 comparisons and four scalar-dc RMS failures**. The latter are about 1.52%–1.91% against 1.5%. All six non-curvature gradient classes pass their RMS budgets. The C16 control passes these same cases. These records are failures, not a blanket waiver or an all-input pass.
- The actual integrated full24 model captures all 459 parameter gradients (1,412,912,512 elements) against the frozen strict reference with identical initial weights, RNG and batch. Gradient cosine is **0.9922968213**, norm ratio **1.0016784296**, satisfying the adopted initial-model contract. The initial same-batch FP32-CE-derived PPL difference is **−0.014018%**. The old coordinate budget still reports **15,447 failures**. This run makes zero optimizer updates and is not a heldout/continuous-quality result.
- The public C128 CLI completes its W1 small-model gates, including dense/packed model/oracle comparisons, a source VJP check, 16 accumulated samples with an all-ignored final sample, actual FP32 optimizer updates and next-update recovery. All 43 parameter gradient/update/recovery comparisons are exactly equal in the same-precision comparisons. This is not W8/W16 or full24 recovery acceptance.
- Compiled-image/source binding, stage-order checks, checkpoint graph ownership and positive pre-update guards remain enabled. Earlier C128 checkpoint-on/off trace tests are archived separately; the final M64 public CLI uses the checkpointed training recipe.
- Publication host regressions: 283 passed, four GPU cases deselected, plus three targeted checkpoint-contract/recovery checks passed; GPU evidence is reported separately. Scoped pre-commit passed. Formatting preserves executable ASTs. A legacy strict-trace metadata compatibility fix has no BF16 execution effect.

The later user quality policy treats 0.1% PPL as an observation line. The harness records excess and keeps `training_quality_accepted=False`; a single acceptable initial point does not approve long training. Mathematical and same-precision distributed/update/recovery budgets are unchanged.

## Six-issue status and remaining work

| Issue | Current implementation | Remaining |
| --- | --- | --- |
| State-chain work | Parallel output/local VJPs, affine mass, removed residual recomputation, true C128 | State scan still has limited occupancy; optional carry comparison incomplete |
| Preparation | Shared Key systems, blocked inversion, cached relative intervals, M64 cross-interval forward/backward | More efficient group/system preparation and final hardware attribution |
| Registers | M64 interval forward drops reported spills to zero; backward drops to242 registers | Other state/parameter kernels remain resource limited |
| Buffers/partials | Compact no-carry/tail buffers, shared gradient owners, BF16 workspaces, implicit rho | Remaining source copies and cross-stage data exchange |
| Chunk size | C128 public operator, Layer/config and CLI | Final four-shape acceptance on this exact version |
| Full cost/precision | Single BF16 main products, selected three-product inverses, FP32 optimizer, matched step measurement | Final timing, continuous quality, cache/packed regressions, full1/8/16 updates/recovery/scaling |

Two larger alternatives were tested and not selected: paired BV64 state kernels increased step time by0.99%/5.06% at8/16warps; block32 factored group feedback reduced summed GPU kernel time in one profile but increased measured step time by1.255%. Neither replaces the selected code. Historical C16 distributed and long-shape evidence does not qualify this C128/M64 version. No real-corpus or long-term training claim is made.

## Reproduction and evidence scope

On the pinned runtime, the public entry is:

```bash
python benchmarks/benchmark_hyperbolic_training.py \
  --methods khdn_no_carry --khdn-precision-policy bf16 --chunk-size 128 \
  --optimizer-backend fp32 --loss-backend plain --check-only
```

Full runs may omit `--check-only`; the numerical/update/recovery gates still execute before measurement. Final timing additionally requires `--final-validation --trials 3 --warmup 10 --steps 30`. The archived matched-KDA driver supplies the projection-aligned control.

`observations.json.gz` contains raw short timings, operator metrics, all per-parameter initial-gradient statistics, compact CLI evidence and execution receipts. `measured-sources.tar.gz` contains the owned measured source overlay and cost-driver/operator-overlay sources. Protected unrelated parallel/exploratory files are omitted; this is not a full snapshot reconstruction. Production/model/optimizer executable ASTs match publication. Harness differences after measurement are formatting, the legacy strict-trace compatibility fix, and the optimizer backend tag in checkpoint metadata.

Large raw CUDA traces, compiler listings, gradient tensors and checkpoints remain outside Git. Aggregate recomputation from published parameter records is not an independent reread of the full5.6GB captured-gradient set. The archived driver retains its original runtime/cache dependencies; it is provenance, not a portable runtime installer.

```bash
python docs/experiments/khdn_c128_2026_11_08/summarize.py
```

The verifier checks published hashes, owned-source AST identity, raw timing medians, gradient aggregates and retained failures. It does not manufacture GPU qualification from metadata.
