# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone CPU references; never imports FLA's accelerator initialization."""

import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("phba_decode_cpu", ROOT / "fla/ops/phba/decode.py")
DECODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DECODE)


def fixture(length=25, batch=2, heads=2, width=8, slots=3, dtype=torch.float64):
    torch.manual_seed(42)
    x = [torch.randn(batch, length, heads, width, dtype=dtype) * .2 for _ in range(5)]
    x.append(torch.randn(batch, length, heads, slots, dtype=dtype) * .5)
    return x


def oracle(x, c=4, topk=3, mode="full", tau=8.):
    """Independent explicit causal softmax with chunk-prefix memory."""
    q, k, v, qr, kr, gate = x
    b, t, h, d = q.shape
    m = gate.shape[-1]
    cd = torch.float64 if q.dtype == torch.float64 else torch.float32
    log = F.logsigmoid(gate.to(cd)) / tau
    insert = (-log.expm1()).to(q.dtype)
    sk, sv = [torch.zeros(b, h, m, d, dtype=cd) for _ in range(2)]
    keys, values = [], []
    for begin in range(0, t, c):
        keys.append(sk.to(q.dtype).clone())
        values.append(sv.to(q.dtype).clone())
        end = min(t, begin+c)
        weights = []
        for i in range(begin, end):
            weights.append((insert[:, i].to(cd) * log[:, i+1:end].sum(1).exp()).to(q.dtype).to(cd))
        decay = log[:, begin:end].sum(1).exp()[..., None]
        sk, sv = sk * decay, sv * decay
        for i, wi in enumerate(weights, begin):
            sk += wi[..., None] * k[:, i, :, None].to(cd)
            sv += wi[..., None] * v[:, i, :, None].to(cd)
    outputs = torch.empty_like(q)
    for batch in range(b):
        for pos in range(t):
            current = pos // c
            for head in range(h):
                centers = [kr[batch, j*c:(j+1)*c, head].to(cd).mean(0).to(q.dtype).to(cd)
                           for j in range(current)]
                scores = [(qr[batch, pos, head].to(cd) * z).sum().item() for z in centers]
                routes = sorted(range(current), key=lambda j: (-scores[j], j))[:topk-1]
                token_idx = list(range(current*c, pos+1))
                if mode != "routed_states_only":
                    token_idx += [i for block in routes for i in range(block*c, (block+1)*c)]
                ks, vs = kr[batch, token_idx, head].to(cd), v[batch, token_idx, head].to(cd)
                score = (ks * qr[batch, pos, head].to(cd)).sum(-1) * d**-.5
                state_ids = [current] + (routes if mode != "current_state_only" else [])
                state_ids = [i for i in state_ids if i > 0]
                if state_ids:
                    pk = torch.cat([keys[i][batch, head] for i in state_ids]).to(cd)
                    pv = torch.cat([values[i][batch, head] for i in state_ids]).to(cd)
                    score = torch.cat((score, (pk * q[batch, pos, head].to(cd)).sum(-1) * d**-.5))
                    vs = torch.cat((vs, pv))
                p = score.softmax(-1).to(q.dtype).to(cd)
                outputs[batch, pos, head] = (p[:, None] * vs).sum(0)
    return outputs


class DecodeCPU(unittest.TestCase):
    def test_cache_matches_explicit_oracle(self):
        cases = [(1, "full", torch.float64), (4, "current_state_only", torch.float64),
                 (7, "routed_states_only", torch.float64), (9, "full", torch.float32)]
        for prompt, mode, dtype in cases:
            with self.subTest(prompt=prompt, mode=mode):
                x = fixture(dtype=dtype)
                cache = DECODE.PHBALayerCache(2, 25, 2, 8, 3, 4, 3, dtype=dtype, device="cpu", mode=mode)
                cache.prefill(x[1][:, :prompt], x[4][:, :prompt], x[2][:, :prompt], x[5][:, :prompt])
                expected = oracle(x, mode=mode)
                for t in range(prompt, 25):
                    actual = cache.step(*(z[:, t:t+1] for z in x))
                    torch.testing.assert_close(actual, expected[:, t:t+1], rtol=2e-5, atol=1e-7)
                    self.assertFalse(actual.requires_grad)

    def test_block_formula_matches_token_recurrence(self):
        x = fixture(length=13)
        g = F.logsigmoid(x[-1]) / 8
        s = -g.expm1()
        initial = tuple(torch.randn(2, 2, 3, 8, dtype=torch.float64) for _ in range(2))
        actual = DECODE.block_update(initial, x[1], x[2], g, s)
        expected = [z.clone() for z in initial]
        for i in range(13):
            for j, raw in enumerate((x[1], x[2])):
                expected[j] = g[:, i].exp()[..., None]*expected[j] + s[:, i, ..., None]*raw[:, i, :, None]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=2e-13, atol=2e-14)

    def test_tail_is_not_counted_twice(self):
        x = fixture(length=17)
        def run(prompt):
            cache = DECODE.PHBALayerCache(2, 17, 2, 8, 3, 4, 3, dtype=torch.float64, device="cpu")
            cache.prefill(x[1][:, :prompt], x[4][:, :prompt], x[2][:, :prompt], x[5][:, :prompt])
            for i in range(prompt, 17):
                cache.step(*(z[:, i:i+1] for z in x))
            return cache
        a, b = run(3), run(12)
        for name in ("state_k", "state_v"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        torch.testing.assert_close(a.pk[:, :5], b.pk[:, :5], rtol=0, atol=0)

    def test_delayed_update_preserves_outputs_and_anchors(self):
        cases = [(3, torch.float64, "full"), (8, torch.bfloat16, "current_state_only"),
                 (1, torch.float16, "routed_states_only")]
        for prompt, dtype, mode in cases:
            x = fixture(length=21, dtype=dtype)
            caches = [DECODE.PHBALayerCache(2, 21, 2, 8, 3, 4, 3, dtype=dtype,
                      device="cpu", stage=stage, mode=mode) for stage in ("p0", "p1_state")]
            for cache in caches:
                cache.prefill(x[1][:, :prompt], x[4][:, :prompt], x[2][:, :prompt], x[5][:, :prompt])
            for pos in range(prompt, 21):
                before = caches[1].state_k.clone()
                out = [cache.step(*(z[:, pos:pos+1] for z in x)) for cache in caches]
                torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)
                torch.testing.assert_close(caches[0].state_k, caches[1].state_k, rtol=0, atol=0)
                if (pos+1) % 4:
                    torch.testing.assert_close(before, caches[1].state_k, rtol=0, atol=0)
            self.assertIsNone(caches[1].live_k)
            self.assertLess(caches[1].memory_bytes()["fp32_state"], caches[0].memory_bytes()["fp32_state"])

    def test_topk_ties_and_causality(self):
        x = fixture(length=14)
        x[4].zero_()
        cache = DECODE.PHBALayerCache(2, 14, 2, 8, 3, 4, 3, dtype=torch.float64, device="cpu")
        cache.prefill(x[1][:, :13], x[4][:, :13], x[2][:, :13], x[5][:, :13])
        routes = cache.route(x[3][:, 13], 13)
        self.assertTrue(torch.equal(routes, torch.tensor([0, 1], dtype=torch.int32).expand(2, 2, 2)))

    def test_cached_centroids_and_routes_match_recomputed(self):
        cases = [(3, torch.float64, 3), (8, torch.bfloat16, 4), (7, torch.float16, 16)]
        for prompt, dtype, topk in cases:
            x = fixture(length=25, dtype=dtype)
            caches = [DECODE.PHBALayerCache(2, 25, 2, 8, 3, 4, topk, dtype=dtype,
                      device="cpu", stage=stage) for stage in ("p1_state", "p1_route")]
            for cache in caches:
                cache.prefill(x[1][:, :prompt], x[4][:, :prompt], x[2][:, :prompt], x[5][:, :prompt])
            for pos in range(prompt, 25):
                routes = [cache.route(x[3][:, pos], pos) for cache in caches]
                self.assertTrue(torch.equal(*routes))
                out = [cache.step(*(z[:, pos:pos+1] for z in x)) for cache in caches]
                torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)
                n = (pos+1)//4
                torch.testing.assert_close(caches[0].centroids[:, :n], caches[1].centroids[:, :n], rtol=0, atol=0)

    def test_split_softmax_is_same_in_real_arithmetic(self):
        cases = [(1, "full", 4), (7, "routed_states_only", 3), (9, "current_state_only", 16)]
        for prompt, mode, topk in cases:
            x = fixture(length=21)
            caches = [DECODE.PHBALayerCache(2, 21, 2, 8, 3, 4, topk, dtype=torch.float64,
                      device="cpu", stage=stage, mode=mode) for stage in ("p1_route", "p1_split")]
            for cache in caches:
                cache.prefill(x[1][:, :prompt], x[4][:, :prompt], x[2][:, :prompt], x[5][:, :prompt])
            for pos in range(prompt, 21):
                out = [cache.step(*(z[:, pos:pos+1] for z in x)) for cache in caches]
                torch.testing.assert_close(out[0], out[1], rtol=2e-12, atol=2e-14)

    def test_all_stages_causal_and_batch_independent(self):
        x = fixture(length=17)
        for stage in DECODE.STAGES:
            def run(values):
                cache = DECODE.PHBALayerCache(2, 17, 2, 8, 3, 4, 3, dtype=torch.float64, device="cpu", stage=stage)
                cache.prefill(values[1][:, :5], values[4][:, :5], values[2][:, :5], values[5][:, :5])
                return torch.cat([cache.step(*(z[:, i:i+1] for z in values)) for i in range(5, 17)], dim=1)
            expected = run(x)
            altered = [v.clone() for v in x]
            for tensor in altered:
                tensor[:, 12:] += 5
                tensor[0, :12] -= 3
            actual = run(altered)
            torch.testing.assert_close(actual[1, :7], expected[1, :7], rtol=0, atol=0)

    def test_capacity_and_shape_errors_before_mutation(self):
        x = fixture(length=5)
        cache = DECODE.PHBALayerCache(2, 5, 2, 8, 3, 4, 3, dtype=torch.float64, device="cpu")
        cache.prefill(x[1], x[4], x[2], x[5])
        original = cache.keys.clone()
        with self.assertRaises(ValueError):
            cache.step(*(z[:, :1] for z in x))
        self.assertEqual(cache.length, 5)
        self.assertTrue(torch.equal(cache.keys, original))
        with self.assertRaises(ValueError):
            cache.prefill(x[1], x[4], x[2], x[5])

    def test_topk_one_modes_identical(self):
        x = fixture(length=13)
        outputs = []
        for mode in DECODE.MODES:
            cache = DECODE.PHBALayerCache(2, 13, 2, 8, 3, 4, 1, dtype=torch.float64, device="cpu", mode=mode)
            cache.prefill(x[1][:, :3], x[4][:, :3], x[2][:, :3], x[5][:, :3])
            outputs.append(torch.cat([cache.step(*(z[:, i:i+1] for z in x)) for i in range(3, 13)], dim=1))
        for out in outputs[1:]:
            torch.testing.assert_close(out, outputs[0], rtol=0, atol=0)

    def test_no_persistent_prompt_raw_keys_or_logits(self):
        cache = DECODE.PHBALayerCache(1, 1024, 2, 8, 3, 4, 3, dtype=torch.bfloat16, device="cpu")
        sizes = cache.memory_bytes()
        self.assertEqual(sizes["token_kv"], 2*1024*2*8*2)
        self.assertEqual(cache.raw_tail.shape[1], 4)
        self.assertNotIn("logits", vars(cache))
        self.assertGreater(sum(sizes.values()), 0)

    def test_poisoned_unused_capacity(self):
        x = fixture(length=7)
        cache = DECODE.PHBALayerCache(2, 20, 2, 8, 3, 4, 4, dtype=torch.float64, device="cpu")
        for name in ("keys", "values", "pk", "pv", "raw_tail", "log_tail", "write_tail", "centroids"):
            getattr(cache, name).fill_(float("nan"))
        cache.pk[:, 0].zero_()
        cache.pv[:, 0].zero_()
        cache.prefill(x[1][:, :1], x[4][:, :1], x[2][:, :1], x[5][:, :1])
        for i in range(1, 7):
            out = cache.step(*(z[:, i:i+1] for z in x))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
