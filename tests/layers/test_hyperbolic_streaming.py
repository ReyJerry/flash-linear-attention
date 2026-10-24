# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from fla.layers.hyperbolic_deltanet import HyperbolicDeltaNet
from fla.models.cache_utils import Cache
from fla.models.hyperbolic_deltanet import HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM

@pytest.mark.parametrize('device,conv', [('cpu',False),('cuda',False),('cuda',True)])
@pytest.mark.parametrize('chunk', [16,32])
def test_protected_segmentation(device,chunk,conv):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('requires CUDA')
    torch.manual_seed(127)
    model=HyperbolicDeltaNet(hidden_size=32,head_dim=16,num_heads=2,protection_rank=2,
        chunk_size=chunk,use_short_conv=conv,fuse_norm=False,mode='naive' if device=='cpu' else 'chunk',layer_idx=0).to(device=device,dtype=torch.float64 if device=='cpu' else torch.float32).eval()
    x=torch.randn(2,37,32,device=device,dtype=next(model.parameters()).dtype)*.2
    with torch.no_grad():
        expected=model(x,use_cache=False)[0]
        full=Cache(); full_out,_,full=model(x,past_key_values=full,use_cache=True)
        torch.testing.assert_close(full_out,expected,rtol=3e-4,atol=3e-5)
        for widths in ([1]*37,[3,14,2,18]):
            cache=Cache();outputs=[];start=0
            for width in widths:
                out,_,cache=model(x[:,start:start+width],past_key_values=cache,use_cache=True)
                outputs.append(out);start+=width
            torch.testing.assert_close(torch.cat(outputs,1),expected,rtol=3e-4,atol=3e-5)
            for key in ('state','protected_keys','protected_values','protected_scores','protected_count'):
                torch.testing.assert_close(cache[0]['recurrent_state'][key],full[0]['recurrent_state'][key],rtol=3e-4,atol=3e-5)
            assert cache[0]['recurrent_state']['pending']['q'].shape[1]==37%chunk

def test_tensor_core_model_config_and_training(monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('requires CUDA')
    import fla.layers.hyperbolic_deltanet as module
    actual=module.chunk_hyperbolic_delta_rule; called=[]
    def spy(*args,**kwargs):
        called.append(kwargs.get('tensor_core'));return actual(*args,**kwargs)
    monkeypatch.setattr(module,'chunk_hyperbolic_delta_rule',spy)
    c=HyperbolicDeltaNetConfig(hidden_size=32,head_dim=16,num_heads=2,num_hidden_layers=1,vocab_size=64,
        intermediate_size=64,tensor_core=True,use_short_conv=False,fuse_norm=False,fuse_swiglu=False,fuse_cross_entropy=False)
    c=HyperbolicDeltaNetConfig.from_dict(c.to_dict());assert c.tensor_core
    m=HyperbolicDeltaNetForCausalLM(c).cuda().train()
    ids=torch.randint(0,64,(2,35),device='cuda'); loss=m(ids,labels=ids,use_cache=False).loss
    loss.backward();assert called and all(called)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
