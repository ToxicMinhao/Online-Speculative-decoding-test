"""CPU contracts for the opening-inclusive native census.

The stub follows the pinned runtime's important ordering: target prefill ingests
the prompt into draft KV, then the scheduler commits its first output token.
It does not simulate GPU numerics or replace the required native parity gate.
"""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import torch

from experiments.lost_repair.native_worker import CensusBackend, CensusSuccessors
from experiments.verification_transition.budget import WorkBudget


class Mode:
    def __init__(self, extend=False):
        self.extend = extend

    def is_extend(self):
        return self.extend


class Request:
    def __init__(self, cap):
        self.origin_input_ids = [10, 11, 12]
        self.output_ids = []
        self.sampling_params = SimpleNamespace(max_new_tokens=cap)
        self.grammar = None

    def finished(self):
        return (len(self.output_ids) >= self.sampling_params.max_new_tokens
                or bool(self.output_ids and self.output_ids[-1] == 99))


class Cache:
    def __init__(self, layers):
        self.rows = [(torch.arange(64, dtype=torch.float32).reshape(32, 2) + i,
                      torch.arange(64, dtype=torch.float32).reshape(32, 2) - i)
                     for i in range(layers)]

    def get_kv_buffer(self, layer):
        return self.rows[layer]


class Allocator:
    def __init__(self):
        self.state = (torch.arange(16, 32),)

    def backup_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state


def runtime_fixture(cap=4, canonical=(20, 21, 22, 23)):
    pool = SimpleNamespace(
        req_to_token=torch.arange(32, dtype=torch.int32).reshape(1, 32),
        free_slots=[1, 2],
    )
    allocator = Allocator()
    layers = [SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(layer_id=i)))
              for i in range(5)]

    def runner(count):
        return SimpleNamespace(model=SimpleNamespace(layers=layers[:count]),
                               req_to_token_pool=pool,
                               token_to_kv_pool=Cache(count),
                               token_to_kv_pool_allocator=allocator)

    target, draft = runner(2), runner(5)
    worker = SimpleNamespace(
        page_size=1, use_compact_draft_cache=False, block_size=16,
        draft_model=draft.model, model_runner=target, draft_model_runner=draft,
        device="cpu", coefficient=2., ingestion_calls=[], trace=[],
    )
    batch = SimpleNamespace(
        reqs=[Request(cap)], req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        spec_info=None, sampling_info=SimpleNamespace(is_all_greedy=True),
        forward_mode=Mode(True), is_extend_in_batch=False, req_to_token_pool=pool,
    )

    def ingest(batch, info):
        end = int(batch.seq_lens[0])
        start = end - len(info.target_hidden)
        locations = pool.req_to_token[0, start:end].long()
        worker.ingestion_calls.append((start, end, batch.forward_mode.is_extend(),
                                       int(info.draft_seq_lens[0])))
        for key, value in draft.token_to_kv_pool.rows:
            key[locations] = info.target_hidden * worker.coefficient
            value[locations] = info.target_hidden * worker.coefficient + 1
        # Pinned dflash_worker consumes these fields during prefill as well.
        info.draft_seq_lens = batch.seq_lens.clone()
        info.ctx_lens.zero_()
        info.target_hidden = info.target_hidden[:0]

    @torch.inference_mode()
    def forward(batch, **kwargs):
        if batch.forward_mode.is_extend():
            info = SimpleNamespace(
                target_hidden=torch.arange(6, dtype=torch.float32).reshape(3, 2),
                ctx_lens=torch.tensor([3], dtype=torch.int32),
                draft_seq_lens=torch.tensor([0], dtype=torch.int32),
                verified_id=torch.tensor([canonical[0]]),
            )
            worker._append_target_hidden_to_draft_kv(batch, info)
            batch.spec_info = info
            return canonical[0]

        req = batch.reqs[0]
        output_start = len(req.output_ids)
        seed = int(draft.token_to_kv_pool.rows[0][0][2, 0])
        proposed = [seed + i for i in range(15)]
        req.output_ids.append(canonical[output_start])
        batch.seq_lens += 1
        batch.seq_lens_cpu += 1
        end = int(batch.seq_lens[0])
        info = batch.spec_info
        info.target_hidden = torch.tensor([[2 * (end - 1), 2 * (end - 1) + 1]],
                                          dtype=torch.float32)
        info.ctx_lens = torch.tensor([1], dtype=torch.int32)
        info.verified_id = torch.tensor([canonical[output_start]])
        worker._append_target_hidden_to_draft_kv(batch, info)
        worker.trace.append(dict(output_start=output_start - 1, draft_tokens=proposed,
                                 committed_tokens=[canonical[output_start]],
                                 finished=req.finished()))
        return worker.trace[-1]

    worker._append_target_hidden_to_draft_kv = ingest
    worker.forward_batch_generation = forward
    return SimpleNamespace(draft_worker=worker), batch


class OpeningCensusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.budget = WorkBudget(Path(self.tmp.name) / "work.jsonl")

    def prepare(self, cap=4, canonical=(20, 21, 22, 23)):
        scheduler, batch = runtime_fixture(cap, canonical)
        request = dict(request_id="r", prompt_ids=[10, 11, 12], max_new_tokens=cap)
        session = CensusSuccessors(scheduler, request, "KPLUS6", range(1, cap + 1),
                                   self.budget)
        original_forward = scheduler.draft_worker.forward_batch_generation
        session.install()
        self.addCleanup(session.close)
        # The scheduler owns this first output commit, after native prefill.
        seed = scheduler.draft_worker.forward_batch_generation(batch)
        batch.reqs[0].output_ids.append(seed)
        batch.forward_mode = Mode()
        return session, scheduler.draft_worker, batch, original_forward

    def test_opening_owns_prompt_ingestion_not_an_empty_draft_cache(self):
        session, worker, batch, _ = self.prepare()
        self.assertEqual(int(batch.spec_info.draft_seq_lens[0]), 3)
        self.assertEqual(len(batch.spec_info.target_hidden), 0)
        self.assertEqual(int(batch.spec_info.ctx_lens[0]), 0)
        worker.forward_batch_generation(batch)
        opening = session.successors[0]
        self.assertEqual(opening.output_ids, [20])
        self.assertEqual(opening.locations.dtype, torch.int64)
        self.assertEqual(opening.mapping.dtype, torch.int32)
        self.assertEqual(len(opening.ingestions), 1)
        self.assertEqual(opening.ingestions[0].draft_prefix_length, 0)
        self.assertEqual(opening.ingestions[0].end, 3)
        self.assertEqual(len(opening.draft_reference_cache), 5)
        self.assertEqual(opening.metadata()["remaining_output_budget"], 3)

    def test_opening_neutral_replay_restores_prompt_cache_and_proposals(self):
        session, worker, batch, _ = self.prepare()
        original = worker.forward_batch_generation(batch)
        state = session.successors[0]
        for runner in (worker.model_runner, worker.draft_model_runner):
            for key, value in runner.token_to_kv_pool.rows:
                key.fill_(999)
                value.fill_(999)
        worker.ingestion_calls.clear()
        restored = session.rebuild(state, "KPLUS6")
        self.assertEqual(worker.ingestion_calls, [(0, 3, True, 0)])
        self.assertTrue(session.reconstruction["target_prefix_cache_exact"])
        self.assertTrue(session.reconstruction["neutral_draft_prefix_cache_exact"])
        self.assertEqual(worker.forward_batch_generation(restored), original)
        self.assertEqual(state.output_ids, [20])

    def test_opening_cross_checkpoint_replays_with_destination_weights(self):
        session, worker, batch, _ = self.prepare()
        worker.forward_batch_generation(batch)
        state = session.successors[0]
        worker.coefficient = 3.
        restored = session.rebuild(state, "KPLUS4")
        self.assertTrue(session.reconstruction["target_prefix_cache_exact"])
        self.assertIsNone(session.reconstruction["neutral_draft_prefix_cache_exact"])
        self.assertFalse(session.reconstruction["source_draft_cache_transplanted"])
        expected = torch.arange(6, dtype=torch.float32).reshape(3, 2) * 3
        self.assertTrue(torch.equal(worker.draft_model_runner.token_to_kv_pool.rows[0][0][:3],
                                    expected))
        self.assertEqual(restored.reqs[0].sampling_params.max_new_tokens, 4)

    def test_later_neutral_replay_keeps_original_ingestion_shapes(self):
        session, worker, batch, _ = self.prepare()
        worker.forward_batch_generation(batch)
        state = session.successors[1]
        original = worker.forward_batch_generation(batch)
        worker.ingestion_calls.clear()
        restored = session.rebuild(state, "KPLUS6")
        self.assertEqual(worker.ingestion_calls, [(0, 3, True, 0), (3, 4, False, 3)])
        self.assertEqual(worker.forward_batch_generation(restored), original)

    def test_cap_census_includes_every_start_and_not_the_terminal_endpoint(self):
        session, worker, batch, original_forward = self.prepare()
        while not batch.reqs[0].finished():
            worker.forward_batch_generation(batch)
        self.assertEqual(session.round_index, 3)
        self.assertEqual(list(session.successors), [0, 1, 2])
        self.assertEqual([v.output_ids for v in session.successors.values()],
                         [[20], [20, 21], [20, 21, 22]])
        self.assertEqual(len(worker.trace), 3)
        self.assertEqual(worker.trace[-1]["committed_tokens"], [23])
        self.assertTrue(worker.trace[-1]["finished"])
        session.close()
        self.assertIs(worker.forward_batch_generation, original_forward)

    def test_eos_round_remains_in_trace_without_a_spurious_next_start(self):
        session, worker, batch, _ = self.prepare(cap=8, canonical=(20, 21, 99))
        while not batch.reqs[0].finished():
            worker.forward_batch_generation(batch)
        self.assertEqual(session.round_index, 2)
        self.assertEqual(list(session.successors), [0, 1])
        self.assertEqual(worker.trace[-1]["committed_tokens"], [99])
        self.assertTrue(worker.trace[-1]["finished"])
        self.assertEqual(session.successors[1].original_max_new_tokens, 8)

    def test_prefill_termination_has_zero_verification_starts(self):
        for cap, canonical in [(1, (20,)), (8, (99,))]:
            with self.subTest(cap=cap, canonical=canonical):
                session, worker, batch, _ = self.prepare(cap, canonical)
                self.assertTrue(batch.reqs[0].finished())
                self.assertEqual(session.successors, {})
                self.assertEqual(session.round_index, 0)
                self.assertEqual(worker.trace, [])
                session.close()

    def test_opening_rejects_absent_prefill_history_or_zero_draft_metadata(self):
        session, _, batch, _ = self.prepare()
        original = session.ingestions
        session.ingestions = []
        with self.assertRaisesRegex(ValueError, "ingested prefill"):
            session.capture_opening(batch)
        session.ingestions = original
        batch.spec_info.draft_seq_lens = torch.zeros_like(batch.spec_info.draft_seq_lens)
        with self.assertRaisesRegex(ValueError, "ingested prefill"):
            session.capture_opening(batch)


class CensusAdapterTests(unittest.TestCase):
    def test_completion_uses_the_inherited_transition_rpc_namespace(self):
        record = dict(response_ids=[20, 21], finish_reason="length",
                      metadata=dict(verification_calls=1))
        adapter = object.__new__(CensusBackend)
        adapter.backend = SimpleNamespace(
            publish=Mock(), generate=Mock(return_value=record), stop_tokens=[99],
        )
        operations = []

        def dispatch(name, **arguments):
            # verification_transition.native_worker.main sends only this
            # namespace to Scheduler.collective_rpc; other names are backend
            # operations such as generate/publish/check_weights.
            if not name.startswith("transition_"):
                raise KeyError(name)
            operations.append((name, arguments))
            if name == "transition_begin":
                return dict(status="armed")
            return dict(native_rounds=1, successors={"0": {"handle": "start0"}})

        adapter._rpc = dispatch
        request = dict(request_id="r", request_index=1, prompt_ids=[10],
                       max_new_tokens=4, stop_token_ids=[99])
        with tempfile.TemporaryDirectory() as directory:
            actual, states = adapter.capture(request, "KPLUS6", Path(directory))
        self.assertIs(actual, record)
        self.assertEqual(states, {"0": {"handle": "start0"}})
        self.assertEqual([name for name, _ in operations],
                         ["transition_begin", "transition_finish_capture"])
        self.assertEqual(operations[-1][1]["canonical_response_ids"], [20, 21])
        self.assertEqual(operations[-1][1]["canonical_finish_reason"], "length")
        adapter.backend.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
