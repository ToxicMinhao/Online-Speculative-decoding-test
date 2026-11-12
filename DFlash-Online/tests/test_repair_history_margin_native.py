"""CPU tests for actual-score observation and immutable cache compaction."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.repair_history_margin import native_worker as native
from experiments.verification_transition.native import tensor_bits_equal
from experiments.verification_transition.budget import WorkBudget
from test_lost_repair_native import runtime_fixture, Mode


class NativeMarginTests(unittest.TestCase):
    def setUp(self):
        native._QA_ROWS.clear()
        self.addCleanup(native._QA_ROWS.clear)

    def scores(self):
        values = torch.zeros((15, 7), dtype=torch.bfloat16)
        values[:, 3] = 3
        values[:, 1] = 2
        return values

    def test_native_reduction_excludes_canonical_and_respects_acceptance(self):
        scores = self.scores()
        rows, _ = native.native_margin_rows(scores, [3] * 15, [3] * 15, 2,
                                            source_id='r', start_offset=0)
        self.assertEqual([r['slot'] for r in rows], [1, 2])
        self.assertEqual([r['margin'] for r in rows], [1., 1.])
        self.assertTrue(all(r['competitor_id'] == 1 for r in rows))
        self.assertTrue(all(r['native_dtype'] == 'torch.bfloat16' for r in rows))
        self.assertEqual(native.audit_native_margin_samples(list(native._QA_ROWS.values()))['status'], 'passed')

    def test_zero_tie_keeps_native_lowest_token_choice(self):
        values = self.scores()
        values[0, 1] = 3
        rows, _ = native.native_margin_rows(values, [1] + [3] * 14,
                                            [1] + [3] * 14, 1,
                                            source_id='r', start_offset=2)
        self.assertEqual(rows[0]['margin'], 0.)
        self.assertEqual(rows[0]['competitor_id'], 3)
        self.assertEqual(native.audit_native_margin_samples(list(native._QA_ROWS.values()))['status'], 'passed')

    def test_accepted_wrong_token_and_negative_margin_fail(self):
        with self.assertRaisesRegex(ValueError, 'not canonical'):
            native.native_margin_rows(self.scores(), [3] * 15, [2] * 15, 1,
                                      source_id='r', start_offset=0)
        with self.assertRaisesRegex(ValueError, 'negative'):
            native.native_margin_rows(self.scores(), [2] * 15, [2] * 15, 1,
                                      source_id='r', start_offset=0)

    def test_nonfinite_is_not_hidden(self):
        values = self.scores()
        values[0, 5] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            native.native_margin_rows(values, [3] * 15, [3] * 15, 1,
                                      source_id='r', start_offset=0)

    def test_hash_sample_is_global_bottom32_independent_of_visit_order(self):
        scores = self.scores()
        expected = sorted(native.qa_key('r', start, 1) for start in range(50))[:32]
        for start in reversed(range(50)):
            native.native_margin_rows(scores, [3] * 15, [3] * 15, 1,
                                      source_id='r', start_offset=start)
        self.assertEqual(sorted(native._QA_ROWS), expected)
        self.assertEqual(native.audit_native_margin_samples(list(native._QA_ROWS.values()))['sample_count'], 32)

    def test_original_matmul_is_called_and_same_tensor_returned(self):
        left, right = torch.randn(3, 4), torch.randn(4, 8)
        original, observed = torch.matmul, []
        expected = original(left, right)
        with native.observe_original_matmul(observed.append):
            result = torch.matmul(left, right)
        self.assertIs(torch.matmul, original)
        self.assertIs(result, observed[0])
        self.assertTrue(torch.equal(result, expected))

    def test_matmul_observer_restores_after_failure(self):
        original = torch.matmul
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with native.observe_original_matmul(lambda value: None):
                raise RuntimeError('fixture')
        self.assertIs(torch.matmul, original)

    def test_independent_audit_rejects_scalar_tampering(self):
        native.native_margin_rows(self.scores(), [3] * 15, [3] * 15, 1,
                                  source_id='r', start_offset=0)
        rows = list(native._QA_ROWS.values())
        rows[0]['margin'] = 2.
        self.assertEqual(native.audit_native_margin_samples(rows)['status'], 'failed')


class HeadWrapperTests(unittest.TestCase):
    def test_observer_uses_actual_native_scores_and_restores_head(self):
        native._QA_ROWS.clear()
        self.addCleanup(native._QA_ROWS.clear)
        with tempfile.TemporaryDirectory() as folder:
            scheduler, batch = runtime_fixture()
            worker = scheduler.draft_worker
            worker.tp_size = 1
            head = SimpleNamespace(weight=torch.eye(100, dtype=torch.bfloat16),
                shard_indices=SimpleNamespace(num_added_elements=0,
                    org_vocab_start_index=0, num_org_elements=100))
            calls = []
            def original_head(*, hidden_states, lm_head, chunk_size=256):
                logits = torch.matmul(hidden_states, lm_head.weight.T)
                calls.append(logits.clone())
                return logits.argmax(-1)
            worker._greedy_sample_from_vocab_parallel_head = original_head
            original_forward = worker.forward_batch_generation
            recorder = SimpleNamespace(trace=[])
            def forward(batch, **kwargs):
                if batch.forward_mode.is_extend():
                    return original_forward(batch, **kwargs)
                start = len(batch.reqs[0].output_ids) - 1
                values = torch.zeros((15, 100), dtype=torch.bfloat16)
                values[:, 21 + start] = 3
                worker._greedy_sample_from_vocab_parallel_head(
                    hidden_states=values, lm_head=head)
                result = original_forward(batch, **kwargs)
                recorder.trace.append(dict(output_start=start, accepted_draft_tokens=1))
                return result
            worker.forward_batch_generation = forward
            request = dict(request_id='r', source_id='s', prompt_ids=[10, 11, 12],
                           canonical_response_ids=[20, 21, 22, 23], max_new_tokens=4)
            session = native.MarginSuccessors(scheduler, request, 'K5', range(1, 5),
                WorkBudget(Path(folder) / 'work.jsonl'))
            with patch.object(native.sglang_hooks, '_ACTIVE', recorder):
                session.install()
                seed = worker.forward_batch_generation(batch)
                batch.reqs[0].output_ids.append(seed)
                batch.forward_mode = Mode()
                while not batch.reqs[0].finished():
                    worker.forward_batch_generation(batch)
                session.close()
            self.assertIs(worker._greedy_sample_from_vocab_parallel_head, original_head)
            self.assertIs(worker.forward_batch_generation, forward)
            self.assertEqual(session.observed_head_calls, 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual([r['margin'] for r in session.margin_rows], [3., 3., 3.])
            self.assertEqual([r['chosen_id'] for r in session.margin_rows], [21, 22, 23])
            self.assertEqual(native.audit_native_margin_samples(
                list(native._QA_ROWS.values()))['status'], 'passed')


class CacheCompactionTests(unittest.TestCase):
    def states(self):
        full = torch.arange(20).reshape(10, 2)
        return [SimpleNamespace(locations=torch.arange(n),
                target_cache=[(full[:n].clone(), (full[:n] + 1).clone())],
                draft_reference_cache=[((full[:n] + 2).clone(), (full[:n] + 3).clone())])
                for n in (3, 5, 10)]

    def test_compaction_preserves_exact_values_and_shares_cpu_storage(self):
        states = self.states()
        before = [[tuple(x.clone() for x in layer) for layer in s.target_cache] for s in states]
        result = native.compact_prefix_caches(SimpleNamespace(successors=dict(enumerate(states))))
        self.assertEqual(result['states'], 3)
        self.assertEqual(result['prefixes_checked'], 12)
        self.assertLess(result['after_bytes'], result['before_bytes'])
        for state, expected in zip(states, before):
            self.assertTrue(tensor_bits_equal(state.target_cache[0][0], expected[0][0]))
            self.assertEqual(state.target_cache[0][0].untyped_storage().data_ptr(),
                             states[-1].target_cache[0][0].untyped_storage().data_ptr())

    def test_compaction_refuses_changing_historical_rows(self):
        states = self.states()
        states[0].target_cache[0][0][0, 0] = 99
        with self.assertRaisesRegex(ValueError, 'Earlier native cache rows changed'):
            native.compact_prefix_caches(SimpleNamespace(successors=dict(enumerate(states))))

    def test_compaction_preserves_native_rebuild_in_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler, batch = runtime_fixture()
            worker = scheduler.draft_worker
            session = native.CensusSuccessors(scheduler,
                dict(request_id='r', prompt_ids=[10, 11, 12], max_new_tokens=4),
                'K5', range(1, 5), WorkBudget(Path(folder) / 'work.jsonl'))
            session.install()
            seed = worker.forward_batch_generation(batch)
            batch.reqs[0].output_ids.append(seed)
            batch.forward_mode = Mode()
            while not batch.reqs[0].finished():
                worker.forward_batch_generation(batch)
            session.close()
            native.compact_prefix_caches(session)
            for state in session.successors.values():
                session.rebuild(state, 'K5')
                self.assertTrue(session.reconstruction['target_prefix_cache_exact'])
                self.assertTrue(session.reconstruction['neutral_draft_prefix_cache_exact'])


class MetadataOffloadTests(unittest.TestCase):
    def test_field_roundtrip_preserves_values_devices_and_runtime_services(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler, batch = runtime_fixture()
            session = native.CensusSuccessors(scheduler,
                dict(request_id='r', prompt_ids=[10, 11, 12], max_new_tokens=4),
                'K5', range(1, 5), WorkBudget(Path(folder) / 'work.jsonl'))
            session.install()
            worker = scheduler.draft_worker
            seed = worker.forward_batch_generation(batch)
            batch.reqs[0].output_ids.append(seed)
            batch.forward_mode = Mode()
            worker.forward_batch_generation(batch)
            session.close()
            state = session.successors[0]
            original_mapping = state.mapping.clone()
            original_lens = state.batch.seq_lens.clone()
            original_pool = state.batch.req_to_token_pool
            receipt = native.offload_successor_metadata(state)
            self.assertGreater(receipt['tensors'], 0)
            self.assertEqual(receipt['device_bytes_released'], 0)
            restored = native.materialize_successor_metadata(state)
            self.assertIsNot(restored, state)
            self.assertTrue(torch.equal(restored.mapping, original_mapping))
            self.assertEqual(restored.mapping.dtype, torch.int32)
            self.assertEqual(restored.locations.dtype, torch.int64)
            self.assertTrue(torch.equal(restored.batch.seq_lens, original_lens))
            self.assertEqual(restored.batch.seq_lens_cpu.device.type, 'cpu')
            self.assertIs(restored.batch.req_to_token_pool, original_pool)
            session.rebuild(restored, 'K5')
            self.assertTrue(session.reconstruction['target_prefix_cache_exact'])
            self.assertTrue(session.reconstruction['neutral_draft_prefix_cache_exact'])
            self.assertEqual(state.batch.reqs[0].output_ids, [20])
            with self.assertRaisesRegex(ValueError, 'already offloaded'):
                native.offload_successor_metadata(state)

    def test_mapping_handles_nested_containers_without_touching_services(self):
        service = SimpleNamespace(buffer=torch.ones(2))
        original = {'a': [torch.arange(3), (torch.zeros(2),)], 'service': service}
        paths = []
        mapped = native._map_tensors(original, ('x',),
            lambda tensor, path: paths.append(path) or tensor.clone())
        self.assertEqual(paths, [('x', 'a', 0), ('x', 'a', 1, 0)])
        self.assertIs(mapped['service'], service)
        self.assertIsNot(mapped['a'][0], original['a'][0])


if __name__ == '__main__':
    unittest.main()
