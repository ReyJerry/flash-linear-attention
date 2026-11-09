# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

pytest.importorskip('triton')

from benchmarks.fp32_adamw import FP32AdamW  # noqa: E402
from benchmarks.hyperbolic_benchmark_utils import _NoFP64Training  # noqa: E402
from benchmarks.linear_checkpoint import context_fn  # noqa: E402
from fla.models.key_gated_hyperbolic_deltanet import (  # noqa: E402
    KeyGatedHyperbolicDeltaNetConfig,
    KeyGatedHyperbolicDeltaNetForCausalLM,
)
from fla.ops.key_gated_hyperbolic_delta_rule import bf16_selective  # noqa: E402
from fla.ops.key_gated_hyperbolic_delta_rule.validation import FiniteValidationScope  # noqa: E402
from fla.utils import IS_NVIDIA  # noqa: E402

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='Selective KHDN requires NVIDIA SM90.')


def make_model(checkpoint, source_vjp='coupled_fused'):
    torch.manual_seed(42)
    config = KeyGatedHyperbolicDeltaNetConfig(
        hidden_size=256, num_hidden_layers=2, num_heads=2, head_dim=128, intermediate_size=512,
        vocab_size=128, group_size=32, chunk_size=64, curvature=1., learnable_curvature=False,
        precision_policy='bf16_selective', source_vjp=source_vjp, fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False, use_cache=False,
    )
    model = KeyGatedHyperbolicDeltaNetForCausalLM(config).cuda().train()
    if checkpoint != 'none':
        options = dict(use_reentrant=False)
        if checkpoint == 'save_mm':
            options['context_fn'] = context_fn
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)
    return model


@pytest.mark.parametrize('checkpoint', ['none', 'recompute', 'save_mm'])
@pytest.mark.parametrize('source_vjp', ['expanded', 'coupled_fused'])
def test_native_scope_model_gradient_update_and_checkpoint_parity(checkpoint, source_vjp, monkeypatch):
    model = make_model(checkpoint, source_vjp)
    inputs = torch.randint(0, 128, (2, 65), device='cuda')
    optimizer = FP32AdamW(model.parameters(), lr=1e-4)
    snapshots = {name: p.detach().clone() for name, p in model.named_parameters()}
    with _NoFP64Training(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        expected.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    optimizer.step()
    updated = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(snapshots[name])
    observed = []
    entry = bf16_selective.chunk_bf16

    def observe(*args, **kwargs):
        observed.append(kwargs.get('validation_scope'))
        return entry(*args, **kwargs)

    monkeypatch.setattr(bf16_selective, 'chunk_bf16', observe)
    with _NoFP64Training(), FiniteValidationScope(optimizer) as checks:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = model(input_ids=inputs, labels=inputs, use_cache=False, validation_scope=checks).loss
        actual.backward()
        assert checks.checks == (4 if checkpoint == 'none' else 6)
        assert len(observed) == (2 if checkpoint == 'none' else 4) and all(x is checks for x in observed)
        assert torch.equal(actual, expected)
        assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
        optimizer.step()
        assert checks.committed and checks.flushes == 1
    assert all(torch.equal(p, updated[name]) for name, p in model.named_parameters())
    assert len(optimizer.state) == len(snapshots)
    assert all(int(state['step']) == 1 for state in optimizer.state.values())


def test_native_scope_nonfinite_checkpoint_blocks_optimizer_state_mutation():
    model = make_model('save_mm')
    inputs = torch.randint(0, 128, (2, 65), device='cuda')
    optimizer = FP32AdamW(model.parameters(), lr=1e-4)
    snapshots = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match='nonfinite KHDN training'), _NoFP64Training(), FiniteValidationScope(optimizer) as checks:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=inputs, labels=inputs, use_cache=False, validation_scope=checks).loss
        (loss * float('nan')).backward()
        optimizer.step()
    assert not optimizer.state and all(torch.equal(p, saved) for p, saved in zip(model.parameters(), snapshots))


@pytest.mark.parametrize('invalid', ['closed', 'eval', 'no_grad', 'precision'])
def test_native_scope_rejects_unsupported_usage(invalid):
    model = make_model('none')
    inputs = torch.randint(0, 128, (1, 1), device='cuda')
    if invalid == 'closed':
        with pytest.raises(ValueError, match='open FiniteValidationScope'):
            model(input_ids=inputs, use_cache=False, validation_scope=FiniteValidationScope())
        return
    with FiniteValidationScope() as checks:
        if invalid == 'eval':
            model.eval()
        if invalid == 'precision':
            for block in model.model.layers:
                block.attn.precision_policy = 'bf16'
        with torch.set_grad_enabled(invalid != 'no_grad'), pytest.raises(ValueError, match='gradient-enabled'):
            model(input_ids=inputs, use_cache=False, validation_scope=checks)
