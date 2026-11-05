"""CPU fixtures for the isolated depth routes; no SGLang/GPU imports required."""
import copy
import math
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from experiments.block_depth.native_routes import (
    DepthGate, DepthRoutes, ShallowReadout, selected_round_indices, profile_completed_request, bitwise_equal,
)


class MutatingNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6) * self.weight
        residual.add_(hidden)
        hidden.copy_(self.forward(residual))
        return hidden, residual


class Layer(nn.Module):
    def __init__(self, ordinal, calls):
        super().__init__()
        self.ordinal, self.calls = ordinal, calls

    def forward(self, positions, hidden, batch, residual):
        self.calls.append(self.ordinal)
        residual = hidden.clone() if residual is None else residual + hidden
        return 0.1 * residual + self.ordinal, residual


def fixture():
    width = 8
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=width),
                            norm=MutatingNorm(width),
                            layers=nn.ModuleList([Layer(i + 1, calls) for i in range(5)]))
    batch = SimpleNamespace(input_embeds=torch.arange(16 * width, dtype=torch.float32).reshape(16, width) / 100,
                            positions=torch.arange(16), seq_lens=torch.tensor([31]),
                            input_ids=torch.arange(16), batch_size=1)
    originals = []
    def original_forward(batch, *args, **kwargs):
        originals.append(True)
        hidden, residual = batch.input_embeds.clone(), None
        for layer in model.layers:
            hidden, residual = layer(batch.positions, hidden, batch, residual)
        hidden, _ = model.norm(hidden, residual)
        return SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden), can_run_graph=True)
    metadata_calls = []
    runner = SimpleNamespace(forward=original_forward,
        attn_backend=SimpleNamespace(init_forward_metadata=lambda value: metadata_calls.append(value)))
    head_calls = []
    head = torch.arange(17 * width, dtype=torch.float32).reshape(17, width)
    def sample(*, hidden_states, lm_head):
        head_calls.append(tuple(hidden_states.shape))
        return (hidden_states @ lm_head.T).argmax(-1)
    worker = SimpleNamespace(draft_model=model, draft_model_runner=runner, block_size=16,
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=head))),
        _greedy_sample_from_vocab_parallel_head=sample)
    routes = DepthRoutes(worker, cpu_fixture=True)
    return routes, batch, calls, originals, head_calls


class NativeRouteTests(unittest.TestCase):
    def test_initialization_preserves_rng_and_backbone(self):
        state = torch.get_rng_state().clone()
        routes, batch, calls, originals, heads = fixture()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(routes.model.norm.weight, routes.readout.norm.weight))
        self.assertNotEqual(routes.model.norm.weight.data_ptr(), routes.readout.norm.weight.data_ptr())
        self.assertEqual(tuple(routes.readout.A.shape), (32, 8))
        self.assertEqual(tuple(routes.readout.B.shape), (8, 32))
        self.assertEqual(torch.count_nonzero(routes.readout.B).item(), 0)

    def test_original_five_is_physically_unwrapped(self):
        routes, batch, calls, originals, heads = fixture()
        self.assertIs(routes.runner.forward, routes.original_forward)
        routes.set_mode("FORCED3")
        routes.set_mode("FIVE")
        self.assertIs(routes.runner.forward, routes.original_forward)
        routes.runner.forward(batch)
        self.assertEqual(calls, [1, 2, 3, 4, 5])
        self.assertEqual(routes.segments.launches, dict(static_three=0, prefix=0, shallow=0, deep=0))
        self.assertEqual(routes.routes, [])

    def test_forced_five_preserves_residual_input_and_native_hidden(self):
        routes, batch, calls, originals, heads = fixture()
        expected = routes.original_forward(batch).logits_output.hidden_states.clone()
        before = batch.input_embeds.clone()
        calls.clear()
        routes.set_mode("FORCED5")
        actual = routes.runner.forward(batch).logits_output.hidden_states
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(batch.input_embeds, before, rtol=0, atol=0)
        self.assertEqual(calls, [1, 2, 3, 4, 5])

    def test_three_really_skips_both_layers_and_projects_once(self):
        routes, batch, calls, originals, heads = fixture()
        tokens, depth = routes.proposal(batch, "FORCED3")
        self.assertEqual(depth, 3)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(heads, [(15, 8)])
        self.assertEqual(tokens.numel(), 15)
        self.assertEqual(originals, [])

    def test_alternation_is_whole_block_and_full_route_still_matches(self):
        routes, batch, calls, originals, heads = fixture()
        routes.set_mode("ALTERNATE")
        for _ in range(6):
            routes.runner.forward(batch)
        self.assertEqual(routes.routes, [3, 5, 3, 5, 3, 5])
        self.assertEqual(calls.count(4), 3)
        self.assertEqual(calls.count(5), 3)
        self.assertEqual(calls.count(3), 6)

    def test_sidecar_norm_does_not_mutate_native_residual_pair(self):
        readout = ShallowReadout(MutatingNorm(8), 8)
        hidden, residual = torch.ones(16, 8), torch.full((16, 8), 2.0)
        before = hidden.clone(), residual.clone()
        normalized, features = readout.features(hidden, residual)
        self.assertTrue(torch.equal(hidden, before[0]))
        self.assertTrue(torch.equal(residual, before[1]))
        torch.testing.assert_close(readout.finish(normalized, features), normalized, rtol=0, atol=0)

    def test_gate_pools_only_proposals_and_uses_97_features(self):
        gate = DepthGate("cpu")
        features = torch.arange(16, dtype=torch.float32).reshape(16, 1).repeat(1, 32)
        gate.weight[0] = 1
        gate.weight[32] = 2
        gate.weight[64] = 3
        gate.weight[96] = 4
        expected = 2 + 2 * 5.5 + 3 * 11.5 + 4 * math.log(32)
        self.assertAlmostEqual(gate(features, torch.tensor(31)).item(), expected, places=4)
        features[0] = 9999
        self.assertAlmostEqual(gate(features, torch.tensor(31)).item(), expected, places=4)

    def test_forced_modes_still_compute_gate_and_cpu_decision(self):
        routes, batch, calls, originals, heads = fixture()
        routes.gate.continue_cost_ms.fill_(100)
        _, forced = routes.proposal(batch, "FORCED5")
        _, adaptive = routes.proposal(batch, "ADAPTIVE")
        self.assertEqual((forced, adaptive), (5, 3))
        self.assertEqual(routes.segments.launches["prefix"], 2)

    def test_audit_shadows_do_not_advance_live_route_or_history(self):
        routes, batch, calls, originals, heads = fixture()
        routes.set_mode("ALTERNATE", audit=True)
        initial = batch.input_ids.clone()
        routes.runner.forward(batch)
        self.assertEqual(routes.routes, [3])
        self.assertEqual(routes.round_index, 1)
        self.assertEqual(len(routes.audit_records), 1)
        self.assertTrue(routes.audit_records[0]["all_15_equal"])
        self.assertEqual(routes.head_invocations, 2)  # live head remains the worker's responsibility
        self.assertTrue(torch.equal(initial, batch.input_ids))
        routes.close()
        self.assertIs(routes.runner.forward, routes.original_forward)
        self.assertIs(routes.worker._greedy_sample_from_vocab_parallel_head, routes.original_head)

    def test_replay_sampler_keeps_endpoints_and_distinct_indices(self):
        self.assertEqual(selected_round_indices(0), [])
        self.assertEqual(selected_round_indices(1), [0])
        self.assertEqual(selected_round_indices(4), [0, 1, 2, 3])
        indices = selected_round_indices(139)
        self.assertEqual(len(set(indices)), 8)
        self.assertEqual((indices[0], indices[-1]), (0, 138))


class ReplayRestoreTests(unittest.TestCase):
    def run_profile_fixture(self, directory, fail_at=None):
        path = Path(directory) / "profile.json"
        cache = [(torch.arange(256 * 2).reshape(256, 2).float() + 1000 * layer,
                  torch.arange(256 * 2).reshape(256, 2).float() + 2000 * layer)
                 for layer in range(5)]
        # Unused allocator slots may hold NaNs; restoration is a bitwise contract.
        cache[0][0][50] = float("nan")
        original_cache = [(key.clone(), value.clone()) for key, value in cache]
        row = torch.arange(1, 129, dtype=torch.int32).reshape(1, 128)
        original_row = row.clone()
        pool = SimpleNamespace(req_to_token=row, free_slots=[0])
        class Allocator:
            def __init__(self):
                self.free_pages = torch.arange(16, 256)
                self.release_pages = torch.empty(0, dtype=torch.int64)
            def backup_state(self):
                return self.free_pages, self.release_pages
            def restore_state(self, state):
                self.free_pages, self.release_pages = state
            def alloc(self, count):
                result = self.free_pages[:count]
                self.free_pages = self.free_pages[count:]
                return result
        allocator = Allocator()
        original_allocator = tuple(value.clone() for value in allocator.backup_state())
        buffers = SimpleNamespace(input_ids=torch.arange(16), positions=torch.arange(16),
                                  out_cache_loc=torch.arange(16), input_embeds=torch.ones(16, 8),
                                  req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([28]),
                                  seq_lens_cpu=torch.tensor([28]), num_token_non_padded=torch.tensor([16]))
        metadata = SimpleNamespace(page_table=torch.arange(128), max_seq_len_k=28)
        graph = SimpleNamespace(buffers=buffers, raw_bs=1, raw_num_token=16, bs=1, _replay_ntpb=16)
        runner = SimpleNamespace(req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda index: cache[index]), graph_runner=graph,
            attn_backend=SimpleNamespace(target_verify_metadata={1: metadata}, forward_metadata=metadata))
        worker = SimpleNamespace(page_size=1, use_compact_draft_cache=False,
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(
                get_input_embeddings=lambda: lambda ids: torch.ones(len(ids), 8)))),
            _ensure_draft_block_buffers=lambda size: None, _mask_token_id=99,
            _draft_block_spec_info=SimpleNamespace(draft_token_num=16))
        layers = [SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(layer_id=index)))
                  for index in range(5)]
        call_count = [0]
        def proposal(batch, mode):
            # A timed execution must have a durable pending marker before work.
            pending = json.loads(path.read_text())["active_operation"]
            self.assertIn(pending["kind"], {"warmup", "timed_route"})
            call_count[0] += 1
            for key, value in cache:
                key[batch.out_cache_loc] = -100
                value[batch.out_cache_loc] = -200
            metadata.page_table.add_(1)
            buffers.input_ids.add_(1)
            if call_count[0] == fail_at:
                raise RuntimeError("fixture GPU interruption")
            return torch.arange(15), 5 if mode in {"FIVE", "ADAPTIVE5"} else 3
        controller = SimpleNamespace(worker=worker, runner=runner, model=SimpleNamespace(layers=layers),
            segments=SimpleNamespace(prepare=lambda batch: None), proposal=proposal,
            receipt=lambda: {"fixture": True})
        blocks = [dict(output_start=start, draft_tokens=list(range(15)),
                       positions=list(range(20 + start, 36 + start))) for start in (0, 8)]
        modules = {
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                CaptureHiddenMode=SimpleNamespace(NULL=0), ForwardBatch=SimpleNamespace,
                ForwardMode=SimpleNamespace(TARGET_VERIFY=1)),
            "sglang.srt.speculative.spec_info": SimpleNamespace(SpeculativeAlgorithm=SimpleNamespace(DFLASH=1)),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch.object(torch.cuda, "synchronize"):
            if fail_at is None:
                result = profile_completed_request(controller, list(range(20)), list(range(50)),
                    blocks, "fixture", 0, path)
                self.assertEqual(result["status"], "passed")
            else:
                with self.assertRaisesRegex(RuntimeError, "fixture GPU interruption"):
                    profile_completed_request(controller, list(range(20)), list(range(50)),
                        blocks, "fixture", 0, path)
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["restoration_status"], "passed")
        self.assertTrue(all(receipt["restoration_checks"].values()))
        self.assertTrue(torch.equal(row, original_row))
        for actual, expected in zip(allocator.backup_state(), original_allocator):
            self.assertTrue(torch.equal(actual, expected))
        for (key, value), (old_key, old_value) in zip(cache, original_cache):
            self.assertTrue(bitwise_equal(key, old_key))
            self.assertTrue(bitwise_equal(value, old_value))
        return receipt

    def test_completed_replay_restores_overlapping_freed_cache_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = self.run_profile_fixture(directory)
        self.assertEqual(receipt["accounting"]["returned_timed_measurements"], 32)
        self.assertEqual(receipt["accounting"]["unknown_timed_measurements"], 0)
        self.assertEqual(receipt["accounting"]["unattempted_timed_measurements"], 0)

    def test_failed_replay_preserves_partial_records_and_unknown_work(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = self.run_profile_fixture(directory, fail_at=6)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["accounting"]["returned_timed_measurements"], 1)
        self.assertEqual(receipt["accounting"]["unknown_timed_measurements"], 1)
        self.assertEqual(receipt["accounting"]["unattempted_timed_measurements"], 30)
        self.assertEqual(receipt["active_operation"]["kind"], "timed_route")


if __name__ == "__main__":
    unittest.main()
