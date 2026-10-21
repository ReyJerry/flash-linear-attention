# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA and the full FLA runtime are required for model integration.', allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM

from fla.models import HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM


def tiny_config(**kwargs):
    defaults = dict(
        hidden_size=32,
        head_dim=16,
        num_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=97,
        use_short_conv=False,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        pad_token_id=0,
    )
    defaults.update(kwargs)
    return HyperbolicDeltaNetConfig(**defaults)


def test_training_step_and_registration(tmp_path):
    torch.manual_seed(11)
    config = tiny_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, HyperbolicDeltaNetConfig)
    model = AutoModelForCausalLM.from_config(loaded).cuda().train()
    assert isinstance(model, HyperbolicDeltaNetForCausalLM)
    ids = torch.randint(1, config.vocab_size, (2, 19), device='cuda')
    result = model(ids, labels=ids, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path).cuda().eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, restored(ids).logits)


@torch.no_grad()
def test_lm_cache_and_generate():
    torch.manual_seed(11)
    model = HyperbolicDeltaNetForCausalLM(tiny_config()).cuda().eval()
    ids = torch.randint(1, model.config.vocab_size, (1, 71), device='cuda')
    expected = model(ids, use_cache=False).logits
    prefix = model(ids[:, :67], use_cache=True)
    cache, outs = prefix.past_key_values, [prefix.logits]
    for t in range(67, 71):
        step = model(ids[:, t:t+1], past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        outs.append(step.logits)
    torch.testing.assert_close(torch.cat(outs, 1), expected, atol=1e-3, rtol=1e-2)
    generated = model.generate(ids[:, :7], do_sample=False, max_new_tokens=2, eos_token_id=None)
    assert generated.shape == (1, 9)


@torch.no_grad()
def test_packed_language_model_loss_has_no_cross_sequence_target():
    torch.manual_seed(11)
    model = HyperbolicDeltaNetForCausalLM(tiny_config()).cuda().eval()
    a = torch.randint(1, 97, (1, 13), device='cuda')
    b = torch.randint(1, 97, (1, 17), device='cuda')
    ids = torch.cat((a, b), 1)
    labels = ids.clone()
    cu = torch.tensor([0, 13, 30], device='cuda', dtype=torch.int64)
    packed = model(ids, labels=labels, cu_seqlens=cu, use_cache=False)
    ra, rb = model(a, labels=a, use_cache=False), model(b, labels=b, use_cache=False)
    torch.testing.assert_close(packed.logits, torch.cat((ra.logits, rb.logits), 1), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(packed.loss, (12 * ra.loss + 16 * rb.loss) / 28, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(labels, ids)
