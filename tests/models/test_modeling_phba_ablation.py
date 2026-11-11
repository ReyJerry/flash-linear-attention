# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models.phba import PHBAConfig, PHBAForCausalLM
from fla.utils import IS_NVIDIA, device, find_spec_cached

pytestmark = pytest.mark.skipif(
    not IS_NVIDIA or find_spec_cached('flash_attn') is None,
    reason='PHBA model acceptance requires NVIDIA CUDA and flash-attn2.8.3',
)


@pytest.mark.parametrize('mode', ['full', 'current_state_only', 'routed_states_only'])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_ablation_model_training_and_save(mode, checkpointing, tmp_path):
    config = PHBAConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=128,
        moba_chunk_size=32,
        moba_topk=3,
        gsa_num_slots=17,
        use_output_gate=False,
        use_cache=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        fuse_norm=False,
        fuse_swiglu=False,
        phba_ablation=mode,
    )
    model = PHBAForCausalLM(config).to(device).train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for layer in model.model.layers:
        assert layer.attn.phba_ablation == mode
    tokens = torch.randint(config.vocab_size, (2, 97), device=device)
    labels = tokens.clone()
    labels[:, 0] = -100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(input_ids=tokens, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for layer in model.model.layers:
        assert layer.attn.gsa_gate_proj.weight.grad is not None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    config.save_pretrained(tmp_path)
    restored = PHBAConfig.from_pretrained(tmp_path)
    assert restored.phba_ablation == mode
    copy = PHBAForCausalLM(restored).to(device)
    copy.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    copy.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        a = model(input_ids=tokens).logits
        b = copy(input_ids=tokens).logits
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    old_config = config.to_dict()
    old_config.pop('phba_ablation')
    assert PHBAConfig(**old_config).phba_ablation == 'full'
    full = PHBAForCausalLM(PHBAConfig(**old_config))
    full.load_state_dict(model.state_dict(), strict=True)
