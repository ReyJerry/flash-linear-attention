# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models.phba import PHBAConfig, PHBAForCausalLM
from fla.utils import IS_NVIDIA, device, find_spec_cached


def make_config(**kwargs):
    options = dict(
        hidden_size=128, num_hidden_layers=2, num_heads=2, intermediate_size=256, vocab_size=128,
        max_position_embeddings=128, moba_chunk_size=32, moba_topk=2, gsa_num_slots=16,
        use_cache=False, use_output_gate=False, fuse_cross_entropy=False,
    )
    options.update(kwargs)
    return PHBAConfig(**options)


def test_registration_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    config = make_config()
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, PHBAForCausalLM)
    model.save_pretrained(tmp_path)
    loaded_config = AutoConfig.from_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(loaded_config, PHBAConfig)
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
@pytest.mark.parametrize(
    ('qk_norm', 'use_output_gate', 'dtype'),
    [(False, False, torch.bfloat16), (True, True, torch.float16)],
    ids=['baseline-bf16', 'normalized-output-gate-fp16'],
)
def test_modeling(qk_norm, use_output_gate, dtype):
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config(qk_norm=qk_norm, use_output_gate=use_output_gate)).to(device).train()
    tokens = torch.randint(0, 128, (2, 67), device=device)
    with torch.autocast(device_type='cuda', dtype=dtype):
        result = model(input_ids=tokens, labels=tokens)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert result.logits.shape == (2, 67, 128)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cache_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='use_cache=False'):
        make_config(use_cache=True)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_checkpoint_gradients():
    torch.manual_seed(42)
    first = PHBAForCausalLM(make_config()).to(device).train()
    second = PHBAForCausalLM(make_config()).to(device).train()
    second.load_state_dict(first.state_dict())
    second.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tokens = torch.randint(0, 128, (1, 67), device=device)
    for model in (first, second):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=tokens, labels=tokens).loss.backward()
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(left.grad, right.grad), name
