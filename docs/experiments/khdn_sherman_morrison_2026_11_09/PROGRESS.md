# KHDN progress — November 9, 2026

The retained training path is the existing stable block implementation. The
Sherman–Morrison (SM) candidate passed focused correctness checks but lost both
representative complete-step screens, so it is not enabled in the live training
package. The engineering objective remains incomplete and paused.

## Published code and evidence boundary

This update publishes the untimed per-token SM reference, its ten CPU tests,
three frozen GPU candidate source/test files as `.py.txt`, and this progress report. Existing training kernels remain at the production
source represented by `fba02a4c0c1fea25af2281b5c9a03ab1b089592f`.

The complete 39-file local archive is preserved in local commit
`78c650f5a72c225d5b42f1509528f2b954637dba`. Its raw evidence files, portable
verifier and expanded reports are not yet uploaded. The five unique candidate
source/test contents are available through the live untimed reference/CPU test
and the three archived GPU files.
Do not treat that local commit as an accessible GitHub revision. The figures below
are the result of local verification, not a claim that readers can already
reproduce the full archive from this partial publication.

## Complete-step research screens

Each endpoint uses one H20, 24 layers, 1,412,912,512 parameters, H16/K=V128,
g32/C64, fixed positive curvature, no_carry, ShortConv, plain cross-entropy,
BF16 autocast, FP32 master parameters and AdamW. All arms use save_mm and deferred
finite validation. Each forward, backward, directly measured F+B and actual
optimizer-update step has two warmups and five steady observations.

| Shape | Path | F ms | B ms | Direct F+B ms | Full step ms | Tokens/s | Peak allocated GiB | Peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B8×T2048 | Retained block | 521.122 | 1301.892 | 1821.011 | 1849.929 | 8856.56 | 43.7872 | 47.6445 |
| B8×T2048 | SM candidate | 573.021 | 2051.347 | 2621.684 | 2650.325 | 6181.88 | 43.7948 | 46.9238 |
| B8×T2048 | Matched KDA | 392.496 | 1005.097 | 1396.574 | 1420.894 | 11530.77 | 43.7811 | 47.2500 |
| B1×T16384 | Retained block | 551.188 | 1406.752 | 1955.560 | 1969.768 | 8317.73 | 43.7872 | 47.6270 |
| B1×T16384 | SM candidate | 969.526 | 3143.322 | 4112.337 | 4137.872 | 3959.52 | 43.7948 | 46.9258 |
| B1×T16384 | Matched KDA | 407.173 | 1036.573 | 1438.980 | 1456.728 | 11247.12 | 43.7811 | 47.2500 |

The retained block reaches 76.808% / 73.954% of matched KDA throughput. SM
increases step time by 43.266% / 110.069% and allocated memory by 8,126,464 bytes.
Both SM endpoints execute the intended GPU route with zero fallback; matched KDA
executes its TileLang backward. Valid block samples reuse completed same-card,
same-source measurements; SM and KDA run consecutively. These are not a fresh
single-process three-arm paired run or formal four-shape qualification.

The measured frozen snapshot is
`ae35de862dd6837be805e9836de59ca644e659cb614256bf39078ad0a9c89cc1`.
All 883 shared tracked frozen-source files match FBA byte for byte; five candidate
source/test files separately match the frozen manifest. This publication itself
has not been timed. Historical 1900.263 ms / 75.22% is not final-source performance
qualification.

## Why the rank-one candidate did not win

The canonical per-token system H=nI−κvvᵀ supports the exact SM formula. The
cross-token C64 feedback system is generally not rank one. The tested candidate
therefore performs token-serial state updates; C64 describes its checkpoint
interval, not 64 parallel updates. Backward replays each segment into global
scratch and reduces group partials. This does not establish carry-only fusion or
a dedicated H20 shared-memory wide backend. Compiler-reported zero spills alone
cannot establish dynamic occupancy or memory-traffic efficiency.

The GPU candidate uses FP32 controls/state with BF16/FP32 inputs and no FP64
training or fallback. Ill-conditioned calls retain the original block fallback.
The live reference is explicitly untimed; FP64 is selected only as correctness
truth by its tests and is not connected to training dispatch.

## Verification and remaining work

The local archive verifier checked 168 raw observations, 42 actual optimizer
updates and five exact frozen candidate source files. Ten CPU reference/VJP tests
pass; 78 CPU quantity comparisons have maximum absolute error
6.661338147750939e-16 under the existing tolerances. Archived H20 checks include
eight operator cases, one actual two-layer learned-curvature optimizer update and
eleven compiler images with no FP64 training arithmetic. Per-quantity GPU max/RMS
errors were not recorded; full-model gradient cosine and unseen model quality are
not qualified by these checks. Scoped pre-commit and diff checks pass. No new GPU
experiment was launched for this publication.

The prepared direct-residual guard also has no full-step win: stable 1841.618 ms
versus direct 1854.747 ms (+0.71%), with 85,516 guarded fallbacks among 786,432
checks. Earlier group V64/V128 candidates likewise have no clear full-step gain.
Native validation-scope research screens yield about 2.5% throughput improvement;
that is a validation-policy benefit, not a state-graph breakthrough.

Producer/consumer fusion, Delta/Pi global-traffic removal and a dedicated H20
state backend have not met their required full-step benefit condition. Continuous
training/unseen quality, known cache/packed regressions, formal four-shape timing,
recovery and 1/8/16-GPU acceptance remain open. The draft PR stays open and
unmerged; protected parallel work is excluded.
