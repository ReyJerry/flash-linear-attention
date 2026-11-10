# Public selective BF16 integration checkpoint

The public operator, Layer/config and training CLI now accept the explicit
experimental policy `bf16_selective`, with execution chunks16/32/64/128,
K=V128/g32/no-carry in the model harness. The raw operator additionally supports
V32/64. Existing policy defaults and checkpoint parameter names are retained.

The policy uses mass-system preconditioning, FP32 internal buffers and selected
BF16 contractions with FP32 accumulation. No FP64 candidate training or
six-product compensation is enabled. Fixed-curvature C64 retains single-product
forward contractions. When curvature requires gradients, output readout uses
three-product BF16 compensation; C64 inverse off-diagonal preparation also uses
three products. C128 inverse preparation remains compensated.

The legacy C128 eligibility check now widens its query matrix before absolute
value, preserving the declared FP32 control arithmetic. An exhaustive CPU check
of all65536 BF16 encodings confirms identical absolute/admission behavior,
including NaN rejection and infinities.

## Verification boundaries

- Current local host checks:275 passed,12 GPU-dependent skips; one constructor
  test requiring unavailable local Triton was excluded. Scoped pre-commit and
  diff checks are required before publication.
- Earlier public integration snapshot:64 regression cases passed, comprising
  24 original source/cotangent cases,8 c=0 cases,8 fixed-curvature comparisons,
  8 four-chunk cases and16 historical strong-gate cases. The fixed-curvature
  comparisons matched the frozen candidate's output/state/six gradients exactly.
- That snapshot also passed12 actual H20 model execution/checkpoint/optimizer
  trace tests across legacy and selective policies. Compiler and pre-step
  corruption checks are included; this is not full-model quality qualification.
- Expanded sine-cotangent coverage exposed a scalar-curvature failure. Readout
  compensation passed7/8 expanded cases; the remaining C64/BF16 RMS error was
  0.00321145 against0.00252146. An independent readout-plus-inverse compensation
  control passed this case:dc0.16919465 versus reference0.16809728. The failure
  remains recorded for the preceding source.
- **The final conditional C64 inverse change is included for review but has not
  completed combined GPU regression or end-to-end timing.** Earlier test results
  above must not be presented as acceptance of every byte in this checkpoint.

The frozen complete-step screens in [README.md](README.md) are historical
candidate measurements:2193.867ms at B8/T2048 and2299.021ms at B1/T16384 for C64,
versus matched KDA1734.112/1770.852ms. They are not measurements of this final
integration checkpoint. Cache/packed model regressions, finite training and
unseen quality, four-shape formal timing and current-source1/8/16-GPU acceptance
remain incomplete. The proposed fixed-curvature scalar cache is not implemented.

This checkpoint is an experimental draft, not a production promotion.
