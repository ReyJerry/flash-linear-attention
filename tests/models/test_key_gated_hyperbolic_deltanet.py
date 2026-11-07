# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import copy
import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('triton', reason='Actual FLA import requires the Triton runtime, including on CPU.')
pytest.importorskip('transformers')

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models import (
    HyperbolicDeltaNetConfig,
    KeyGatedHyperbolicDeltaNetBlock,
    KeyGatedHyperbolicDeltaNetConfig,
    KeyGatedHyperbolicDeltaNetForCausalLM,
    KeyGatedHyperbolicDeltaNetModel,
    KeyGatedHyperbolicDeltaNetPreTrainedModel,
)
from fla.models.cache_utils import Cache


def tiny_config(**kwargs):
    options = dict(
        hidden_size=64, head_dim=32, num_heads=2, num_hidden_layers=2, intermediate_size=96,
        vocab_size=97, use_short_conv=False, fuse_norm=False, fuse_swiglu=False,
        fuse_cross_entropy=False, pad_token_id=0, attn_mode='naive', curvature=0.6, group_size=32,
    )
    options.update(kwargs)
    return KeyGatedHyperbolicDeltaNetConfig(**options)


def make_optimizer(model, lr=1e-4, weight_decay=0.01):
    excluded = [param for param in model.parameters() if getattr(param, '_no_weight_decay', False)]
    regular = [param for param in model.parameters() if not getattr(param, '_no_weight_decay', False)]
    return torch.optim.AdamW([
        dict(params=regular, weight_decay=weight_decay), dict(params=excluded, weight_decay=0.0),
    ], lr=lr)


def assert_parameter_gradient(name, actual, expected):
    assert actual is not None and expected is not None, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    assert actual.count_nonzero() > 0 and expected.count_nonzero() > 0, name
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2, msg=name)
    if '.f_proj.' in name or name.endswith(('A_log', 'dt_bias', 'curvature_raw')):
        relative_error = (actual.double() - expected.double()).norm() / expected.double().norm()
        assert relative_error <= 2e-2, f'{name}: relative gradient error {relative_error.item()}'


def test_identity_registration_config_roundtrip_and_imports(tmp_path):
    config = tiny_config(decay_rank=24, chunk_size=64, carry_mode='y_carry')
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, KeyGatedHyperbolicDeltaNetConfig)
    fields = ('model_type', 'group_size', 'decay_mode', 'decay_rank', 'decay_function', 'curvature', 'readout')
    for name in fields:
        assert getattr(loaded, name) == getattr(config, name)
    assert loaded.model_type == 'key_gated_hyperbolic_deltanet'
    assert loaded.decay_function == 'negative_exp_softplus' and loaded.readout == 'inclusive'
    assert isinstance(AutoModel.from_config(loaded), KeyGatedHyperbolicDeltaNetModel)
    model = AutoModelForCausalLM.from_config(loaded)
    assert isinstance(model, KeyGatedHyperbolicDeltaNetForCausalLM)
    assert isinstance(model.model.layers[0], KeyGatedHyperbolicDeltaNetBlock)
    assert model.config_class is KeyGatedHyperbolicDeltaNetConfig
    assert set(model._no_split_modules) == {'KeyGatedHyperbolicDeltaNetBlock'}
    assert issubclass(KeyGatedHyperbolicDeltaNetModel, KeyGatedHyperbolicDeltaNetPreTrainedModel)
    assert AutoConfig.for_model('hyperbolic_deltanet').__class__ is HyperbolicDeltaNetConfig
    defaults = KeyGatedHyperbolicDeltaNetConfig()
    assert (defaults.head_dim, defaults.group_size, defaults.chunk_size, defaults.decay_rank) == (128, 32, 32, 128)
    assert defaults.attn_mode == 'chunk' and defaults.carry_mode == 'no_carry'


@pytest.mark.parametrize('kwargs', [
    dict(protection_rank=0), dict(protected_checkpoint=False), dict(tensor_core=False),
    dict(safe_gate=False), dict(lower_bound=None), dict(num_v_heads=4), dict(group_size=64),
    dict(head_dim=16), dict(group_size=32.0), dict(decay_rank=0), dict(decay_mode='scalar'), dict(readout='exclusive'),
    dict(decay_function='sigmoid'), dict(chunk_size=8), dict(model_type='hyperbolic_deltanet'),
])
def test_config_rejects_incompatible_semantics(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)


def test_reject_old_config_and_from_pretrained_identity(tmp_path):
    with pytest.raises(TypeError, match='legacy HDN'):
        KeyGatedHyperbolicDeltaNetForCausalLM(HyperbolicDeltaNetConfig())
    config = tiny_config().to_dict()
    config['model_type'] = 'hyperbolic_deltanet'
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='model_type'):
        KeyGatedHyperbolicDeltaNetConfig.from_pretrained(tmp_path)


def assert_initialized_gates(model):
    for block in model.model.layers:
        layer = block.attn
        assert layer.group_size == 32 and layer.head_v_dim >= 32
        assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
        assert layer.A_log.shape == (2,) and layer.dt_bias.shape == (64,)
        assert layer.A_log._no_weight_decay and layer.dt_bias._no_weight_decay
        assert layer.A_log._is_hf_initialized and layer.dt_bias._is_hf_initialized
        assert torch.isfinite(layer.A_log).all() and torch.isfinite(layer.dt_bias).all()
        assert ((layer.A_log.exp() >= 1) & (layer.A_log.exp() <= 16)).all()
        dt = F.softplus(layer.dt_bias)
        assert ((dt >= 0.001 - 1e-8) & (dt <= 0.1 + 1e-8)).all()
        assert all(module.bias is None for module in layer.f_proj)


def test_hf_gate_init_post_init_dtype_and_actual_optimizer_groups():
    torch.manual_seed(61)
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True))
    assert_initialized_gates(model)
    saved = {name: param.detach().clone() for name, param in model.named_parameters()}
    model.post_init()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, saved[name], msg=name)
    layer = model.model.layers[0].attn
    model._init_weights(layer)
    for name in ('A_log', 'dt_bias', 'curvature_raw'):
        torch.testing.assert_close(getattr(layer, name), saved['model.layers.0.attn.' + name])
    # HF module-name matching and ordinary whole-model conversions both retain gate precision.
    model.to(dtype=torch.bfloat16)
    for idx, block in enumerate(model.model.layers):
        assert block.attn.A_log.dtype == block.attn.dt_bias.dtype == torch.float32
        assert block.attn.curvature_raw.dtype == torch.float32
        assert block.attn.f_proj[0].weight.dtype == torch.bfloat16
        for name in ('A_log', 'dt_bias', 'curvature_raw'):
            torch.testing.assert_close(getattr(block.attn, name), saved[f'model.layers.{idx}.attn.{name}'], atol=0, rtol=0)
    optimizer = make_optimizer(model, lr=1e-4, weight_decay=0.1)
    by_id = {id(param): group for group in optimizer.param_groups for param in group['params']}
    for block in model.model.layers:
        for param in (block.attn.A_log, block.attn.dt_bias):
            assert by_id[id(param)]['weight_decay'] == 0 and param.dtype == torch.float32
        assert by_id[id(block.attn.f_proj[0].weight)]['weight_decay'] == 0.1


def test_meta_materialization_and_stable_curvature_initialization():
    config = tiny_config(learnable_curvature=True, curvature=1000.0)
    with torch.device('meta'):
        model = KeyGatedHyperbolicDeltaNetForCausalLM(config)
    assert all(param.is_meta for param in model.parameters())
    model.to_empty(device='cpu')
    model.apply(model._init_weights)
    assert_initialized_gates(model)
    for block in model.model.layers:
        torch.testing.assert_close(F.softplus(block.attn.curvature_raw), torch.tensor(1000.0))
    assert all(torch.isfinite(param).all() for param in model.parameters())


@pytest.mark.parametrize('checkpoint', [False, True])
def test_actual_loss_and_every_parameter_gradient(checkpoint):
    torch.manual_seed(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True)).to(device).train()
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(1, 97, (2, 7), device=device)
    result = model(ids, labels=ids, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
    make_optimizer(model, lr=1e-4).step()


def test_save_load_from_pretrained_preserves_gates_and_logits(tmp_path):
    torch.manual_seed(31)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True)).to(device).train()
    ids = torch.randint(1, 97, (1, 7), device=device)
    optimizer = make_optimizer(model, lr=1e-3)
    model(ids, labels=ids, use_cache=False).loss.backward()
    optimizer.step()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.eval()
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path)
    restored, info = KeyGatedHyperbolicDeltaNetForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    restored.to(device).eval()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), msg=name)
    for block in restored.model.layers:
        assert block.attn.A_log._no_weight_decay and block.attn.dt_bias._no_weight_decay
        assert block.attn.A_log.dtype == block.attn.dt_bias.dtype == torch.float32
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)
    restored.post_init()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), msg=name)
    bf16 = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert bf16.model.layers[0].attn.f_proj[0].weight.dtype == torch.bfloat16
    assert bf16.model.layers[0].attn.A_log.dtype == bf16.model.layers[0].attn.dt_bias.dtype == torch.float32
    for idx, block in enumerate(bf16.model.layers):
        for name in ('A_log', 'dt_bias', 'curvature_raw'):
            expected_gate = model.get_parameter(f'model.layers.{idx}.attn.{name}').detach().cpu()
            torch.testing.assert_close(getattr(block.attn, name), expected_gate, atol=0, rtol=0)
    resumed_optimizer = make_optimizer(restored, lr=1e-3)
    resumed_optimizer.load_state_dict(optimizer_state)
    for candidate, opt in ((model, optimizer), (restored, resumed_optimizer)):
        for group in opt.param_groups:
            for param in group['params']:
                if getattr(param, '_no_weight_decay', False):
                    assert group['weight_decay'] == 0
        opt.zero_grad(set_to_none=True)
        candidate(ids, labels=ids, use_cache=False).loss.backward()
        opt.step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), msg=name)


@torch.no_grad()
def test_model_cache_generate_beam_and_restore():
    torch.manual_seed(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config()).to(device).eval()
    ids = torch.randint(1, 97, (2, 9), device=device)
    expected = model(ids, use_cache=False).logits
    prefix = model(ids[:, :5], use_cache=True)
    cache = Cache.from_legacy_cache(prefix.past_key_values.to_legacy_cache(), seen_tokens=5)
    outputs = [prefix.logits]
    for t in range(5, 9):
        outputs.append(model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=3e-5, rtol=3e-4)
    cache.reorder_cache(torch.tensor([1, 0]))
    assert cache[0]['recurrent_state']['state'].shape == (2, 2, 32, 32)
    for beams in (1, 2):
        generated = model.generate(ids[:, :3], do_sample=False, num_beams=beams, max_new_tokens=2, eos_token_id=None)
        assert generated.shape == (2, 5)


@torch.no_grad()
def test_packed_label_mask_and_empty_sequence():
    torch.manual_seed(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config()).to(device).eval()
    a, b = torch.randint(1, 97, (1, 5), device=device), torch.randint(1, 97, (1, 7), device=device)
    ids = torch.cat((a, b), 1)
    labels = ids.clone()
    cu = torch.tensor([0, 5, 5, 12], dtype=torch.int64, device=device)
    packed = model(ids, labels=labels, cu_seqlens=cu, use_cache=True)
    left = model(a, labels=a, use_cache=False)
    right = model(b, labels=b, use_cache=False)
    torch.testing.assert_close(packed.logits, torch.cat((left.logits, right.logits), 1), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(packed.loss, (4 * left.loss + 6 * right.loss) / 10, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(labels, ids)
    assert torch.count_nonzero(packed.past_key_values[0]['recurrent_state']['state'][1]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual optimized model comparison requires CUDA kernels.')
def test_real_optimized_model_loss_and_all_gradients():
    import fla.ops.key_gated_hyperbolic_delta_rule as ops

    if not hasattr(ops, 'chunk_key_gated_hyperbolic_delta_rule'):
        pytest.skip('Production chunk entry is not implemented yet.')
    torch.manual_seed(51)
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(attn_mode='chunk', learnable_curvature=True)).cuda().train()
    oracle = copy.deepcopy(model)
    for block in oracle.model.layers:
        block.attn.mode = 'naive'
    ids = torch.randint(1, 97, (1, 35), device='cuda')
    actual = model(ids, labels=ids, use_cache=False)
    expected = oracle(ids, labels=ids, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(actual.loss, expected.loss, atol=1e-4, rtol=1e-3)
    actual.loss.backward()
    expected.loss.backward()
    for name, param in model.named_parameters():
        assert_parameter_gradient(name, param.grad, oracle.get_parameter(name).grad)


def test_optimizer_step_excludes_gate_decay():
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True))
    optimizer = make_optimizer(model, lr=0.01, weight_decay=0.1)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for name, param in model.named_parameters():
        if name.endswith(('A_log', 'dt_bias')):
            torch.testing.assert_close(param, before[name], atol=0, rtol=0)
        elif '.f_proj.' in name:
            assert not torch.equal(param, before[name]), name


@pytest.mark.parametrize('meta,assign', [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize('inference', [False, True])
def test_load_state_dict_preserves_fp32_controls_and_hf_markers(meta, assign, inference):
    source = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True))
    state = {name: value.detach().clone().bfloat16() for name, value in source.state_dict().items()}
    with torch.inference_mode(inference):
        with torch.device('meta' if meta else 'cpu'):
            model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True))
        loaded = model.load_state_dict(state, assign=assign)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        for idx, block in enumerate(model.model.layers):
            for name in ('A_log', 'dt_bias', 'curvature_raw'):
                param = getattr(block.attn, name)
                assert param.dtype == torch.float32 and not param.is_inference()
                assert isinstance(param._version, int) and param._is_hf_initialized
                torch.testing.assert_close(param, state[f'model.layers.{idx}.attn.{name}'].float(), atol=0, rtol=0)
                if name != 'curvature_raw':
                    assert param._no_weight_decay
            saved = {name: getattr(block.attn, name).clone() for name in ('A_log', 'dt_bias', 'curvature_raw')}
            model._init_weights(block.attn)
            for name, expected in saved.items():
                torch.testing.assert_close(getattr(block.attn, name), expected, atol=0, rtol=0)


def test_model_inference_construction_loading_and_cached_generation(tmp_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.inference_mode():
        model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True)).to(device).eval()
        model.save_pretrained(tmp_path)
        loaded = KeyGatedHyperbolicDeltaNetForCausalLM.from_pretrained(tmp_path).to(device).eval()
        ids = torch.randint(1, 97, (1, 4), device=device)
        for candidate in (model, loaded):
            prefix = candidate(ids[:, :3], use_cache=True)
            step = candidate(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
            assert torch.isfinite(step.logits).all()
            generated = candidate.generate(ids, do_sample=False, max_new_tokens=2, eos_token_id=None)
            assert generated.shape == (1, 6)
            param = candidate.model.layers[0].attn.curvature_raw
            assert not param.is_inference()
            param.add_(0.1)
            with pytest.raises(ValueError, match='stale'):
                candidate(ids[:, :1], past_key_values=prefix.past_key_values, use_cache=True)


def test_inference_meta_materialization_keeps_controls_versioned():
    with torch.inference_mode():
        with torch.device('meta'):
            model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True))
        model.to_empty(device='cpu')
        model.apply(model._init_weights)
        for block in model.model.layers:
            for name in ('A_log', 'dt_bias', 'curvature_raw'):
                param = getattr(block.attn, name)
                assert not param.is_inference() and param.dtype == torch.float32
                assert isinstance(param._version, int)
            torch.testing.assert_close(F.softplus(block.attn.curvature_raw), torch.tensor(0.6))
        layer = model.model.layers[0].attn
        x, cache = torch.randn(1, 3, 64), Cache()
        assert torch.isfinite(layer(x, past_key_values=cache, use_cache=True)[0]).all()
        layer.curvature_raw.add_(0.1)
        with pytest.raises(ValueError, match='stale'):
            layer(x[:, :1], past_key_values=cache, use_cache=True)


def test_optimized_gradient_acceptance_rejects_suppressed_paths():
    torch.manual_seed(51)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KeyGatedHyperbolicDeltaNetForCausalLM(tiny_config(learnable_curvature=True)).to(device).train()
    ids = torch.randint(1, 97, (1, 35), device=device)
    model(ids, labels=ids, use_cache=False).loss.backward()
    checked = []
    for name, param in model.named_parameters():
        if '.f_proj.' in name or name.endswith('curvature_raw'):
            reference = param.grad
            assert_parameter_gradient(name, reference, reference)
            for factor in (0.0, 0.01, 0.5):
                with pytest.raises(AssertionError):
                    assert_parameter_gradient(name, reference * factor, reference)
            checked.append(name)
    assert len(checked) == 6
