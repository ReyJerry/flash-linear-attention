# KHDN experimental mixed precision

`precision_policy='strict'` remains the default. The explicit `mixed_readout` option retains accurate readout and state/control arithmetic while using seventeen TF32x3 matrix-product sites with FP32 operands and accumulation. It is available through the chunk operator and the KHDN Config → Model → Layer chain. It is an experimental execution policy, not a different model, a general accuracy certificate, or a speed claim.

BF16 autocast controls the surrounding model operations. It does not override explicitly typed Triton operations. The measured Layer preparation supplies FP32 physical source tensors even under BF16 autocast. TF32x3 is distinct from plain BF16 multiplication; generated TTIR and PTX must be inspected to report the actual mode.

## Arithmetic and selection

Both readout products, `Qer @ S` and `(To * rho) @ U`, and their four direct source/transpose VJP products retain FP64 operands and accumulation. State/control, prepared coefficients, triangular solves, normalization and common-source reductions also retain FP64. The remaining selected state/carry contractions use FP32 operands, explicit TF32x3 products and FP32 accumulation before extension into FP64 buffers. Products with padded dimensions below 16 use the strict primitive. Public outputs, final state and source-gradient dtypes are unchanged.

The retained readout boundary matters: the legal moderate-input expression `(1+2^-13)*(1-2^-13)-1` equals `-2^-26`, but a three-product TF32x3 readout loses the residual product and returns zero. The unchanged near-zero RMS budget rejects that error. Restoring only the readout while rounding cross-chunk state to FP32 also fails a multi-chunk neighbor. These finite counterexamples justify the current boundary; they do not prove that every internal operation must always use FP64.

The experimental selector currently admits matching FP32/BF16 q/k/v, FP32 controls/state/curvature, C16/32/64, K16/128, V16/128, g16/32, batch ≤8, heads ≤16 and individual sequence lengths ≤2048. It checks SM90 and the tested Torch2.9.0+cu128/Triton3.5.0 version strings, source magnitudes ≤8, effective query magnitude `abs(scale)*maxabs(q) ≤8`, squared Key norms in [.25,4], decay ≥−16 and curvature ≤1, with the small-model curvature extension below. Original mathematical validation, feedback/state margin and source/cotangent exception handling remain active.

These bounds describe a finite experiment envelope, not proof that every tensor in its Cartesian product meets the numerical budgets. Version strings do not identify compiler binaries. Unsupported selection falls back to strict execution and reports why. A mixed forward with an incoming cotangent above the experimental magnitude bound can replay a strict primal/VJP from the original sources; this is reported separately and does not replace the already returned forward output. Existing exceptional arbitrary-precision paths remain distinct.

The original backward envelope admits unscaled output/final-state cotangents up to magnitude 8. A separately tested small-model envelope admits magnitudes up to 128 only for K=V128, g32, C16, no_carry, batch/heads ≤2, at most four sequences, and longest sequence ≤35. It was motivated by actual BF16-model boundary adjoints of 56.429 and 115.348; their mixed partials matched retained strict partials bitwise and passed independent FP64 truth. Existing source/range/accuracy guards run first. This empirical extension is not training acceptance; other geometries or larger adjoints retain strict replay. Backward telemetry records the profile, limit and actual two incoming maxima.

A separate experimental `model_curvature_v1` profile admits finite curvature in (1,2] only for the same small-model geometry: K=V128, g32, C16, no_carry, batch/heads ≤2, at most four sequences and longest sequence ≤16 (one chunk). It retains the original ≤8/≤128 adjoint rules and every source/cotangent accuracy guard. Curvature >2, multi-chunk sequences or other geometry selects strict; C32/C64, y_carry and the original [8,2048,16,128] Layer0 shape receive no curvature extension. Public negative/nonfinite-curvature errors and the c=0 source derivative are unchanged. Forward telemetry identifies the new profile separately from the original `empirical_profile_v1`.

The endpoint2 bounds a local empirical extension motivated by learned curvature 1.000059843 after the first Adam update. Curvature1 is not a singularity: the retained native margins enforce `(c/g)||v_group||²/||k||² ≤1/4` and `(c/g)||S0_group||² ≤1/2`, independently of whether c crosses1. The first margin bounds the per-token dense-solve condition number by4/3; it does not certify accumulated cancellation or every chunk transform. All seventeen selected TF32x3 products and FP64 controls/readouts remain unchanged. The earlier profile admitted c1/T35 joint adjoints128 that fail the dBeta elementwise budget; the combined-adjoint screening below routes those cases to strict-device replay. Direct all-seven-source comparisons with unchanged FP64 truth and budgets are still required; neither model training nor the entire Cartesian envelope is certified by the selector.

For extended small-model backward calls spanning more than one chunk, native mixed execution additionally requires `N * (maxabs(dOutput) + maxabs(dFinal)) <= 128`, where `N` is the maximum chunk count of any individual sequence. Absent or unused final-state gradients count as zero; packed total tokens and sequence count are not substituted for `N`. The legacy max-magnitude≤8 rule remains unchanged, and single-chunk calls retain independent maxima≤128, including joint128. Qualifying multi-chunk extended calls report `model_joint_adjoint_v1`; excluded calls retain the reviewed strict-device replay and report `unqualified_joint_chunk_adjoint`. Telemetry records both effective maxima, maximum chunk count, their sum and the resulting product.

This combined bound is an empirical screen motivated by additive VJP contributions and sequential chunk propagation. It is not a uniform bound on source-dependent Jacobians, cancellation or floating-point error. Replay uses a strict primal/VJP from original sources and does not change the mixed forward already returned; both forward and all requested source partials must still meet the unchanged dense-reference budgets. Replay is reported separately from native mixed training. Curvature bounds, source/accuracy guards, mathematical equations, public dtypes and all17 selected product precisions are unchanged by this backward screen.

When the same original Tensor object occupies multiple source arguments, their source partials are accumulated before the public source-dtype conversion and delivered once to that input edge. The public entry records object identity before contiguity copies; distinct detached leaves sharing storage remain independent. Native execution uses FP64 destination buffers for the aliased partials while retaining the existing product arithmetic. Strict replay reconstructs one leaf per original input edge. Exceptional host backward sums in its existing FP64 or arbitrary-precision context before conversion and representability checks. This preserves cancellation that would be lost by separately rounding argument partials before autograd accumulates an aliased leaf. It does not change the independently implemented dense Oracle or the requested-source and first-order contracts.

Telemetry reports `source_alias_groups` by argument index (q,k,v,beta,log_decay,S0,c) and `source_gradient_aggregation`. Host-forward details are in `host_details`; exceptional backward details are in `backward_host_details`; replay retains nested strict details. This identity rule applies to repeated leaf and non-leaf Tensor objects, including noncontiguous inputs. It does not merge distinct views or independently detached leaves by storage pointer, nor certify arbitrary cancellation elsewhere in an upstream model graph.

Configuration values are serialized. Explicit mixed selection requires chunk mode; short evaluation/decode calls that use the recurrent entry remain strict and preserve the canonical cache. Existing configurations without the field load with strict policy. No cache tensor is silently downcast by this option.

Telemetry separates `precision_requested`, `precision_eligible`, actual `precision_route`, and `backward_precision_route`. Host fallback overrides the forward route, and strict replay retains nested details. A reused dictionary describes its latest call and clears owned stale fields; use a fresh dictionary for each live autograd graph. Host-forward manual backward details are in `host_details`, not a fabricated native backward record.

## Compiler binding

The original Triton3.5 compiler can emit duplicated shared-memory writers for replicated FP64 layouts. Different bytes were observed in the original witness. The runtime used for the integrated gates includes a narrowly reviewed predicate patch to `lowerLdStShared`, selecting one lane/warp representative per shared address. The patch does not change the numerical equations or suppress sanitizer diagnostics.

| Component | Tested identity |
| --- | --- |
| Triton source | `c3c476f357f1e9768ea4e45aa5c17528449ab9ef` |
| LLVM revision | `7d5de3033187c8a3bb4d2e322f5462cdaf49808f` |
| Original `libtriton.so` SHA256 | `4478f4ab08b25844c68c7f41eec6c865aeb5f0918d2d1da37173cc2fbd4f5bd9` |
| Tested patched `libtriton.so` SHA256 | `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32` |
| Preserved overlay archive SHA256 | `46d99552aa1ef43282d96a1a7e6dab3cebeb70fc3230c8ab4230cec01b9b160c` |

The source patch is [triton-3.5-shared-store-predicate.patch](../scripts/patches/triton-3.5-shared-store-predicate.patch). The tested build used Linux x86_64/Python3.11, GCC11.4, CMake3.31.6, pybind11 2.13.1 and Ninja1.13. The archive contains the overlay package, per-file hashes, exact configure arguments and build/source bindings. A rebuild must record its own library hash and requalify generated kernels; an identical version string does not inherit the tested hash or evidence.

Extract the verified archive into a fresh node-local directory, check its package hashes and library hash, and prepend its `overlay` directory to `PYTHONPATH` before the intended immutable FLA source. Keep the existing Torch environment and global installation unchanged. Compiler/runtime caches must be local and separate by run, Pod and GPU. Assert the actual imported compiler path/hash in every execution process. Do not place compiler caches or builds on a shared filesystem.

The patch is qualified for the inspected shared-store callback and tested strict/mixed paths. It is not a blanket validation of every Triton backend, asynchronous/multicast layout or unrelated operator. The separate unmerged parallel-backward candidate still has numerical failures under racecheck in small-width y_carry cases and is not accepted.

## Current evidence and remaining acceptance

The integrated source and patched library passed 253 unique operator tests across two H20 nodes, with zero failures/errors/skips and an audited disjoint test partition. This includes strict and mixed routes, all seven source derivatives, prior numerical regressions and the new route-accounting checks. A focused 39-case run overlaps those tests; counts must not be added.

Five affected mixed cases passed INFO-level racecheck with zero hazards/errors/warnings; the same five passed memcheck with zero errors. Sanitizer diagnostics execute the unchanged dense Oracle on CPU through differentiable copies because its GPU cuBLAS solve reports independent INFO hazards. The full 253-case numerical suite retains the original GPU Oracle, unchanged comparison function and unchanged tolerances. No sanitizer filter or suppression was used.

The integrated public mixed wrapper also passed the actual Layer0 `[8,2048,16,128]` capture against retained strict truth: output and all six requested partials pass; S0 and final-state coverage comes from separate all-seven-source tests. Its maximum output absolute error is `3.8708094507455826e-9`. This is one layer's arithmetic/path validation, not a model optimizer trajectory or throughput measurement.

The model policy wiring passed nine CPU lifecycle/dispatch checks using the independent reference and a CPU activation stand-in. Actual BF16 model loss, parameter gradients, optimizer updates, training save/resume, four full-model shapes, final timing repeats and 1/8/16-rank scaling still require their own exact-source/runtime evidence. The experimental T2048 and curvature bounds cannot be silently expanded to claim those missing results.

## Bounded operator measurement

One same-H20 comparison used the actual Layer0 inputs `[8,2048,16,128]`, g32/C32/no_carry and the Layer's `check_inputs=False` call setting, after the numerical and safety gates. It alternated strict/mixed order for 10 warmup pairs and 30 measured pairs and created a fresh backward graph for every sample. Both routes passed the captured output/six-requested-partial checks and had no host fallback or strict replay.

| Median latency | Strict | Mixed readout |
| --- | ---: | ---: |
| Forward | 1183.035 ms | 478.909 ms |
| Backward | 1620.978 ms | 697.464 ms |
| Measured F+B | 2804.031 ms | 1176.404 ms |
| F+B p95 | 2807.279 ms | 1179.082 ms |

The measured F+B ratio is 2.38356× for this one operator shape. Both allocate a peak of 15,640,839,680 bytes and reserve 15,672,016,896 bytes; storage remains FP64. A prior run with public default input validation enabled is retained separately and is not pooled with these samples. This is neither a full-model/GDN comparison nor DDP throughput, and it does not satisfy the final three-independent-repeat requirement. It predates the small-model adjoint-envelope extension; the unchanged kernel bytes and ≤8 captured adjoints identify the measured execution scope.
