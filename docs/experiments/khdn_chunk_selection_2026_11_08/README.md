# KHDN chunk selection and numerical diagnostics

Research evidence based on `0c667850d8e5aed44637099d2c671753e472f29d`. C64 is the preferred optimization candidate after both endpoint screens; it is **not a qualified new default**. The optimized C16/32/64/128 implementation remains in the reproducible experimental source bundle. The product change accompanying this report removes two overwritten stores from the existing public C128 kernel and adds two public regression cases. It does not switch the public smaller-chunk backend or change model defaults, mathematics, precision, or tolerances.

## Same-generation chunk comparison

One H20, Torch 2.9.0+cu128, Triton 3.5.0; 24 layers, hidden 2048, H16, K=V128, g32, FFN5184, vocabulary32000, positive curvature and trainable Key gates. Both KHDN sizes use the same optimized source, BF16 contractions/storage, FP32 state/control, fused preprocessing, weights, data, checkpointing, plain CE and FP32 AdamW. Matched KDA includes the full output-gate projection; each model has 1,412,912,512 parameters. Actual execution grids and 24 KDA TileLang backward calls are recorded.

Each arm has two warmups and three steady samples for directly timed F+B and complete step, followed by a separate profiled update. Each arm performs six actual optimizer updates. These are screening runs, not the final three-trial, ten-warmup/thirty-steady protocol. Short arm order is C64/C128/KDA; long order is KDA/C128/C64. Transfers and zero-grad are outside the timed region; clipping, optimizer and public input/finite checks are included.

| Shape       | Arm         | F+B ms   | Step ms  | Tokens/s | Peak allocated bytes |
| ----------- | ----------- | -------: | -------: | -------: | -------------------: |
| B8×T2048    | C64         | 2013.892 | 2036.431 | 8045.45  | 27616996352          |
| B8×T2048    | C128        | 2094.469 | 2117.449 | 7737.61  | 28071025664          |
| B8×T2048    | Matched KDA | 1710.372 | 1733.505 | 9451.37  | 27470129152          |
| B1×T16384   | C64         | 2097.362 | 2119.218 | 7731.16  | 27591044096          |
| B1×T16384   | C128        | 2180.253 | 2199.662 | 7448.42  | 28049005568          |
| B1×T16384   | Matched KDA | 1746.538 | 1770.193 | 9255.49  | 27470129152          |

C64 throughput is 3.98% and 3.80% above C128, and 85.12% and 83.53% of matched KDA, respectively. Allocated memory is lower by 433.0 and 436.7 MiB. Reserved memory is included in the raw samples. The short-screen `interval_execution_tile=64` binding is a stale descriptive field for C64: its actual cross-interval forward tile is32, as shown by source and launch geometry. The long screen corrects that label. A null source-fallback counter is not proof of zero fallbacks.

The short profile shows the tradeoff: state forward/backward totals are 48.864/51.614 ms at C64 versus 45.016/43.913 ms at C128. Cross-interval forward/backward totals fall from 27.285/35.143 to16.076/23.428 ms, and shared feedback-system backward from20.510 to10.190 ms. These are summed kernel times from separate profile steps, not causal percentages of ordinary step latency or new NCU counters.

## Original numerical failures remain open

The original twelve seed42/T129 fixtures cover FP32/BF16 sources, dense/packed input and output/state/joint cotangents. Identical source and cotangent tensors are used for both sizes and every ablation. CPU FP64 dense canonical solves are explicit untimed truth; candidates and compiled kernels contain no FP64. All108 first-round executions are finite and repeat backward exactly. Published C128 and private same-generation C128 have identical metric records.

| Variant                         | C64 bad elements / RMS failures | C128 bad elements / RMS failures |
| ------------------------------- | -----------------------------: | ------------------------------: |
| Baseline                        | 1178 / 7                       | 1533 / 5                        |
| Erasure operands FP32 before mul| 1177 / 7                       | 1536 / 5                        |
| Only SavedRO FP32               | 1157 / 7                       | 1523 / 5                        |
| Only SavedS FP32                | 1176 / 7                       | 1532 / 5                        |

None passes all twelve cases. Error counts include all output/state/seven-source tensors, not just the first failing assertion. Each metric retains its original absolute/relative/RMS budget, maximum-error coordinate, sign disagreements and finite status. No proposed storage restoration above is promoted or subjected to expensive full-model qualification because it did not resolve the failures.

A second diagnostic reuses those archived fixtures at C64 and changes one preparation-storage domain at a time, while retaining single BF16 dots. FP32 Key transforms produce1080 bad elements/8 RMS failures; FP32 Key-system matrices produce30375/26; FP32 group-system matrices produce418/6. The Key-system variant also fails exact repeat backward in one joint/FP32/dense case and needs implementation investigation; its regression cannot be attributed solely to precision. The other variants repeat exactly. Group-system storage is a useful localization lead, not a qualified fix. The separate historical fixed-case beta RMS failure and prior approved scalar-curvature exceptions retain their original scope.

## Legal raw-input boundary

The specified B=T=H1, K128, V=g32 case with q=k=e1, curvature1, beta.5, zero gate/state and every FP32 Value component `.9998779296875` has positive FP32 physical margin. Public C128 and experimental C64/C128 accept the source through the conservative source-guard fallback, then fail the finite-forward check. Actual GPU observations identify Gram diagonal32, gauge Inf and system diagonal0.

A diagnostic retaining the Gram in FP32 and replacing only its diagonal with the original-input FP32 squared norm gives Gram diagonal31.9921875, gauge2048.5 and system diagonal0.000244140625. Output, final state and all seven gradients pass the existing budgets at both C64 and C128 for this case. Off-diagonal dots remain BF16. This establishes a local repair for this counterexample, not general-domain qualification; no clamp, epsilon, detach, curvature change or support-domain restriction is introduced. The default Layer's value margin excludes this particular input. Additional storage cost and broader-domain behavior remain to be evaluated before integration.

## Overwritten input stores

`common_input_kernel` wrote P3/P4, `common_products_kernel` did not read either, and `interval_cache_kernel` subsequently overwrote every element with prefix/suffix factors. Only the two initial stores are deleted. The allocations, decay values used elsewhere, and actual prefix/suffix producers remain. Logical eliminated writes are256 MiB/operator-forward at the main shape; this is not a measured DRAM-traffic or capacity reduction.

Eight NaN-poisoned dense/packed/empty/tail cases across C16/32/64/128 pass the original output/state/seven-source budgets and are bitwise equal to the same-source baseline. Twenty-four lifecycle checks verify the buffers remain poisoned before the true producer and become finite afterwards. The public C128 integration separately passes the two new regression cases, with81 other tests deselected, not passed.

| C64 full24 B8×T2048 | F+B ms   | Complete step ms |
| ------------------ | -------: | ---------------: |
| Before             | 2011.596 | 2035.285         |
| Dead stores removed| 2008.747 | 2037.416         |
| Matched KDA        | 1711.355 | 1734.121         |

Complete step is0.10% slower in this short screen: **no net training speedup is claimed**. Common-input summed GPU time in a separate profile is12.443→8.978 ms. Retaining the removal eliminates proven redundant writes without changing tested results; it does not close the performance gap. The full-model timing uses the experimental C64 backend; the published C128 deletion has public numerical integration coverage, not a new C128 full-model timing.

## Reproduction and remaining work

Run `python docs/experiments/khdn_chunk_selection_2026_11_08/summarize.py` for artifact hashes, raw-sample medians, failure counts and source bindings. `source-binding.json` maps each immutable run to its supervisor receipt and result hash. Reconstruct an experiment from the base commit plus its `repo-overlay` in `measured-sources.tar.gz`, then use its frozen driver/candidate modules and recorded runtime. The bundle preserves experiment source and protocols; remote node/runtime paths require adaptation outside the recorded environment. Raw large fixtures and compiled images remain in the experiment archive; published records include their identities and derived metrics, not an independent re-execution of those tensors.

CPU checks executed for this change: source/dataflow and artifact verification, Python parsing and scoped pre-commit. The public kernel's modified function has the same executable AST as the frozen tested candidate. The dependency finder returned no paths for this dynamically imported kernel; operator coverage was selected explicitly. No CPU mathematical change or full reference-suite rerun is claimed.

Still required: original operator numerical qualification, the preparation repeatability investigation, a cost-effective general boundary repair, minimum useful precision selection, further triangular/layout/partial optimizations, public optimized smaller-chunk integration, native model trace/cache/packed qualification, B4×T4096 and B2×T8192 plus formal four-shape timing, current-source1/8/16-GPU gradient/update/recovery/scaling, and longer quality validation. C128 Y-carry remains unqualified. Earlier-source distributed, recovery and PPL evidence does not automatically qualify these candidates.
