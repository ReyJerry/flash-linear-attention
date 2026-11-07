# PHBA

PHBA combines causal block routing, local token attention, and gated prefix states. The implementation provides the deterministic block-major training path, canonical metadata, fixed-order gradient accumulation, and bounded temporary workspaces.

## Installation and model registration

Install FLA with its CUDA and PHBA dependencies. FlashAttention2.8.3 is required for the joint local-attention backward interface.

```bash
pip install -e '.[cuda,phba]'
```

PHBA is registered by the shared `fla.models` exports. Import the configuration directly or use Transformers auto classes:

```python
import torch
from transformers import AutoModelForCausalLM
from fla.models import PHBAConfig

config = PHBAConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    num_heads=16,
    num_kv_heads=16,
    intermediate_size=4272,
    max_position_embeddings=8192,
    moba_chunk_size=128,
    moba_topk=8,
    gsa_num_slots=128,
    qk_norm=False,
    use_output_gate=False,
    use_cache=False,
)
model = AutoModelForCausalLM.from_config(config).cuda().train()
tokens = torch.randint(config.vocab_size, (1, 8192), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(input_ids=tokens, labels=tokens).loss
loss.backward()
```

The public model classes are `PHBAConfig`, `PHBAModel`, and `PHBAForCausalLM` in `fla.models.phba`. The layer is `fla.layers.phba.PHBA`; the operator is `fla.ops.phba.chunk_phba`. `fla.models` exports all three classes; importing `fla` also registers PHBA with Transformers. Checkpoints retain the PHBA projection, normalization, MLP, embedding and language-head parameter names.

## Supported execution

The operator accepts CUDA FP16 or BF16 activation tensors and accumulates prefix and attention intermediates in FP32. Models can retain FP32 parameters under autocast. Query, key and value head counts must match; head dimensions are multiples of 8 from 16 through 128; prefix slot counts are 1–128. Supported chunk sizes are 32, 64, 128, 256, 512 and topk is 1–16, including the current block.

Dense inputs, packed variable-length inputs, nonempty padded sequences, first-order gradients, and non-reentrant gradient checkpointing are supported. `elementwise_affine=False` removes learnable scale parameters from all PHBA normalization layers. Packed operator inputs use `[1, total_tokens, heads, width]` and int32 cumulative sequence lengths. `SequenceLengthPlan.from_host` can supply known sequence lengths without a device-to-host length read. Dense `[B,T,...]` inputs with `B>1` require exactly `B` sequences of length `T`; packed inputs require `B=1`. When both a plan and `cu_seqlens` are supplied, their boundaries must agree. Passing `plan.cu_seqlens` itself, or a fresh, unmodified source used by `from_device`, avoids a device comparison. Other separate tensors require a value comparison and can synchronize. Model calls may pass `phba_length_plan` alone for packed sequences, and rotary positions use its boundaries. Padding masks and explicit packed `cu_seqlens` are mutually exclusive.

The model supports cache-free training, prefill and generation with `use_cache=False`. `output_attentions=True`, persistent KV/prefix caching, an additional sliding window, and unequal query/key/value head counts are not implemented by this deterministic path. Final prefix states returned by the operator are FP32 and nondifferentiable. Context parallelism, tensor parallelism and full-graph compilation are outside the current validation scope.

## Optional H20 acceleration

PHBA projections retain the native linear path unless the exact-screened H20 geometry and runtime match. The accelerated path is restricted to PyTorch 2.9.0, CUDA 12.8, the validated cuBLASLt library, BF16 autocast and its three weight-gradient geometries. The C++ helpers build from packaged source into the Torch extension cache; no prebuilt machine-specific binary is included. CUDA development headers and a C++ compiler are needed for this optional path. Other supported cases retain native PyTorch operations.

Training uses the optimizer selected by the training framework. PHBA model construction does not modify optimizer selection or global Torch dispatchers.

## Validation and benchmarking

```bash
FLA_CI_ENV=0 pytest tests/ops/test_phba.py tests/models/test_modeling_phba.py \
  tests/layers/test_phba_linear.py

python benchmarks/benchmark_phba.py --chunk-size 128 --topk 8 \
  --warmup-steps 20 --steps 50 --optimizer torch.optim:AdamW \
  --output phba-throughput.json
```

The standard operator registry includes `chunk_phba` with dense and packed shapes: run `python -m benchmarks.ops.run --op chunk_phba --no-base`. The dedicated training benchmark measures complete model steps. Its default recipe is 1.264B parameters, B1×8192, FP32 parameters, BF16 autocast, no activation checkpointing and lr=0 training computation. It measures zero-grad, forward/loss, backward, clipping and optimizer updates together. Input generation and optional numerical audit hooks are outside the measured steps. The lr=0 recipe measures compute throughput and is not a convergence experiment.

The references and numerical storage contract are described in [the operator README](../fla/ops/phba/README.md). Migration validation additionally compares the frozen mature implementation using the same initial weights, inputs and native normalization choices. See [migration validation](phba_validation.md) for the completed numerical gates, paired dense/packed throughput, memory measurements and limitations.

Profiling is separate from the production operator: run `python benchmarks/profile_phba.py` with the same benchmark arguments to export a Chrome trace beside the JSON result. Its instrumented timings must not be used as throughput measurements. Offline cuBLASLt candidate-inspection code is under `benchmarks/phba_tools`; production loads only the validated descriptors.
