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
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached


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


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_right_padding():
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    tokens = torch.randint(1, 128, (2, 67), device=device)
    mask = torch.ones_like(tokens)
    mask[1, 49:] = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        padded = model(input_ids=tokens, attention_mask=mask).logits
        for index, length in enumerate((67, 49)):
            expected = model(input_ids=tokens[index:index + 1, :length]).logits
            assert_close('padded_logits', expected.float(), padded[index:index + 1, :length].float(), .006)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_cache_free_generation():
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    prompt = torch.tensor([[1, 4, 7]], device=device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        expected = prompt
        for _ in range(2):
            token = model(input_ids=expected).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), dim=-1)
        actual = model.generate(
            input_ids=prompt, max_new_tokens=2, do_sample=False, use_cache=False,
            pad_token_id=0, eos_token_id=None,
        )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('fuse_norm', [False, True], ids=['native-norm', 'fused-norm'])
def test_affine_configuration(fuse_norm):
    model = PHBAForCausalLM(make_config(elementwise_affine=False, fuse_norm=fuse_norm, qk_norm=True, use_output_gate=True))
    norms = [model.model.norm]
    for layer in model.model.layers:
        norms.extend((layer.attn_norm, layer.mlp_norm, layer.attn.q_norm, layer.attn.k_norm, layer.attn.o_norm))
    assert all(module.weight is None for module in norms)


@pytest.mark.parametrize('width', [8, 17, 136], ids=['below-minimum', 'unaligned', 'above-maximum'])
def test_configuration_head_dimensions(width):
    with pytest.raises(ValueError, match='head dimensions'):
        make_config(hidden_size=width * 2)


def test_central_model_exports():
    from fla.models import PHBAConfig as ExportedConfig
    from fla.models import PHBAForCausalLM as ExportedCausalLM
    from fla.models import PHBAModel

    assert ExportedConfig is PHBAConfig
    assert ExportedCausalLM is PHBAForCausalLM
    assert AutoConfig.for_model('phba').model_type == PHBAModel.config_class.model_type


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_model_plan_boundaries_and_affine_gradients():
    from fla.ops.phba import SequenceLengthPlan

    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config(elementwise_affine=False, qk_norm=True, use_output_gate=True)).to(device).train()
    tokens = torch.randint(0, 128, (2, 33), device=device)
    with pytest.raises(ValueError, match='dense batch boundary'):
        model(input_ids=tokens, phba_length_plan=SequenceLengthPlan.from_host((66,), device))
    packed = tokens.reshape(1, -1)
    plan = SequenceLengthPlan.from_host((33, 33), device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(input_ids=packed, labels=packed, phba_length_plan=plan)
        expected = model(input_ids=packed, labels=packed, cu_seqlens=plan.cu_seqlens)
    assert torch.equal(actual.logits, expected.logits)
    actual.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='different boundaries'):
        model(input_ids=packed, cu_seqlens=torch.tensor([0, 66], device=device, dtype=torch.int32), phba_length_plan=plan)


@pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None, reason='PHBA CUDA integration')
def test_updated_cumulative_lengths_do_not_reuse_rotary_indices():
    torch.manual_seed(42)
    model = PHBAForCausalLM(make_config()).to(device).eval()
    tokens = torch.randint(0, 128, (1, 66), device=device)
    cu = torch.tensor([0, 33, 66], dtype=torch.int32, device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=tokens, cu_seqlens=cu)
        cu[1] = 32
        actual = model(input_ids=tokens, cu_seqlens=cu).logits
        expected = model(input_ids=tokens, cu_seqlens=cu.clone()).logits
    assert torch.equal(actual, expected)
