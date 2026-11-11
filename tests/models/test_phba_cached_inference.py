# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models.phba import PHBAConfig, PHBAForCausalLM
from fla.models.phba.inference import PHBADecoder
from fla.utils import IS_NVIDIA, assert_close, device, find_spec_cached

pytestmark = pytest.mark.skipif(not IS_NVIDIA or find_spec_cached('flash_attn') is None,
                                reason='cached model acceptance requires NVIDIA CUDA and FlashAttention')


@pytest.mark.parametrize('stage,mode,batch,fused,output_gate,qk_norm', [
    ('p0', 'full', 1, False, False, False),
    ('p1_state', 'full', 1, False, True, False),
    ('p1_route', 'current_state_only', 2, True, False, True),
    ('p1_split', 'full', 2, True, True, True),
    ('p1_split', 'routed_states_only', 1, False, False, False),
])
@torch.inference_mode()
def test_model_logits_cache_and_training_path_unchanged(stage, mode, batch, fused, output_gate, qk_norm):
    torch.manual_seed(42)
    config = PHBAConfig(hidden_size=64, num_hidden_layers=2, num_heads=2, num_kv_heads=2,
                        intermediate_size=128, vocab_size=128, max_position_embeddings=128,
                        moba_chunk_size=32, moba_topk=3, gsa_num_slots=17, use_output_gate=output_gate,
                        qk_norm=qk_norm, use_cache=False, fuse_norm=fused, fuse_swiglu=fused,
                        fuse_cross_entropy=False, fuse_linear_cross_entropy=False, phba_ablation=mode)
    model = PHBAForCausalLM(config).to(device).eval()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    tokens = torch.randint(config.vocab_size, (batch, 69), device=device)
    decoder = PHBADecoder(model, max_length=69, stage=stage)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(tokens, use_cache=False).logits
        initial = decoder.prefill(tokens[:, :31])
        assert initial.shape == (batch, 1, config.vocab_size)
        assert_close('prefill logits', expected[:, 30:31].float(), initial.float(), .03)
        actual = torch.cat([decoder.decode(tokens[:, t:t+1]) for t in range(31, 69)], dim=1)
        assert_close('cached model logits', expected[:, 31:].float(), actual.float(), .03)
        with pytest.raises(ValueError, match='capacity'):
            decoder.decode(tokens[:, :1])
        decoder.clear()
        again = decoder.prefill(tokens[:, :31])
        torch.testing.assert_close(again, initial, rtol=0, atol=0)
    assert not model.config.use_cache
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    decoder.clear()
