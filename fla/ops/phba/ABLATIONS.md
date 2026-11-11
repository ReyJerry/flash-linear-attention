# PHBA readout ablations

The `phba_ablation` config, layer and `chunk_phba` argument selects one of three modes. Existing configurations default to `full`; parameter names and tensor shapes are unchanged.

| Mode | Current causal tokens | Routed historical tokens | Current prefix | Routed historical prefixes |
| --- | --- | --- | --- | --- |
| `full` | yes | yes | yes | yes |
| `current_state_only` | yes | yes | yes | no |
| `routed_states_only` | yes | no | yes | yes |

`routed_states_only` removes only historical exact-token readout, not local causal attention. All modes retain the same gate, prefix recurrence, block partition and top-K algorithm. All prefix snapshots are still constructed. A current prefix is the state **before** the current block, not a state updated through the query token.

```python
from fla.models.phba import PHBAConfig, PHBAForCausalLM

config = PHBAConfig(
    # Keep the rest of the matched training recipe unchanged.
    phba_ablation="current_state_only",  # or "routed_states_only"
    use_cache=False,
)
model = PHBAForCausalLM(config)
```

Choose the mode before constructing the model. Changing `model.config` alone after construction does not update existing layers. Configuration save/load retains the mode, and old checkpoints load with `full` by default. These modes support the same cache-free training/prefill interface as the original PHBA implementation; they do not add KV-cache decoding.

## Implementation contract

The historical forward kernel uses compile-time token/state switches. Local attention and current-state merging are unchanged. Original routes schedule forward history work even when historical token readout is disabled. Backward receives shape-preserving readout routes, with disabled edges replaced by `-1`.

Full mode keeps the original token-derived state metadata and forward metadata reuse. Ablations build state metadata from the actual state readout routes and do not reuse backward token metadata for forward scheduling. Existing attention backward kernels and state recurrence kernels are unchanged. Both tape and fallback backward paths require GPU acceptance testing.

This minimal implementation retains fixed-capacity route workspaces and may launch empty backward token tasks in `routed_states_only`. It is intended for matched quality ablations, not a claim of optimal throughput or minimum state-cache memory. Routes have the same selection rule, but independently trained variants can learn different routing scores.

## CPU checks

Run from the repository root with CPU PyTorch installed:

```bash
python tests/ops/test_phba_ablation_cpu.py
```

The standalone runner avoids importing FLA's accelerator-only package initialization. It loads the unchanged `naive.py` reference directly and selects actual host definitions using Python AST. It tests shape-preserving route masking, input immutability, config defaults/wiring, host forward/backward route dispatch, causal and packed boundaries, FP64 staged/manual backward versus recurrent/autograd references, finite-difference gradcheck for all three modes, and the `topk=1` identity.

It also executes the actual `_history_partial` and `_owner_merge` source bodies with a small PyTorch-backed CPU evaluator, checking masks, empty prefixes, tails and both index widths against the reference. This checks kernel semantics, **not Triton compilation, GPU instruction behavior or GPU performance**. Metadata kernel launches are replaced by explicit CPU oracles in host-control tests. Full Hugging Face model integration is covered by the GPU suite below, not by the CPU AST checks.

## GPU acceptance gate

In an environment satisfying the existing PHBA dependencies (including FlashAttention 2.8.3):

```bash
FLA_CI_ENV=0 pytest -q tests/ops/test_phba_ablation.py tests/models/test_modeling_phba_ablation.py
FLA_CI_ENV=0 pytest -q tests/ops/test_phba.py tests/models/test_modeling_phba.py tests/layers/test_phba_linear.py
```

The new suite compares BF16/FP16 output and all six input gradients with the storage-aware reference, forces tape and fallback backward separately, checks actual effective-route metadata, no-grad forward, empty-prefix cases, and `topk=1` equivalence. The model suite checks a training/optimizer step with and without non-reentrant checkpointing, configuration persistence, strict state-dict loading and old-config defaults.

GPU tests must pass before long training runs. CPU success alone is not GPU acceptance. For the paper, train each variant under the same recipe as the full model; changing an already-trained checkpoint only at evaluation is a diagnostic, not a matched training ablation.
