# PHBA

PHBA combines causal block routing, local attention, and gated prefix states. The integration keeps the validated deterministic block-major training path, canonical metadata, gate algebra, and screened H20 projection gradients together under PHBA-owned modules.

## Public entry points

The model follows the repository's configuration, decoder-layer, and causal-language-model structure. Import `fla.models.phba` to register `PHBAConfig`, `PHBAModel`, and `PHBAForCausalLM` with Transformers auto classes. Public operators live in `fla.ops.phba` and the attention layer in `fla.layers.phba`.

Existing non-PHBA Python files, including shared model/layer export lists, remain unchanged. PHBA-specific adapters preserve checkpoint parameter names and keep optional acceleration isolated from other models.

## Validation contract

The migration is compared with the frozen, previously validated PHBA implementation. Validation covers outputs, every supported gradient, deterministic metadata, parameter/state-dict compatibility, save/load and registration, supported dense and variable-length inputs, and clear rejection of unsupported inputs. Optional optimizer packetization also requires nonzero-learning-rate multi-step parameter, gradient, momentum, and step-counter equality.

Performance is measured on the same H20 devices with identical input shapes, weights, seeds, precision, native normalization choices, and optimizer settings. Full-step measurements include allocation, descriptor/view construction, sorting, conversion, and dispatch. No dense FLOPs proxy is used to infer a hardware limit.

Implementation and validation on this development branch are in progress. The draft pull request records the current validation status.
