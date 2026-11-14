"""CPU mocked-native contracts; these do not establish A800 numerical parity."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.threatened_repair_replay import native_probe as native
from experiments.threatened_repair_replay import probe_client as client
from experiments.verification_transition.budget import WorkBudget
from experiments.verification_transition.native import tensor_bits_equal
from test_lost_repair_native import runtime_fixture, Mode


def record_fixture():
    prompt, response = [10, 11, 12], [20, 21, 22, 23, 24]
    blocks = []
    for start in range(4):
        blocks.append(dict(output_start=start, anchor_token=response[start],
            draft_tokens=[999] * 15, target_argmax=[response[start+1]] + [888] * 15,
            positions=list(range(len(prompt)+start, len(prompt)+start+16)), verify_size=16,
            accepted_draft_tokens=0, native_accepted_draft_tokens=0,
            raw_accepted_draft_tokens_before_stop=0, committed_progress=1,
            committed_tokens=[response[start+1]], stop_clipped=False, finished=start == 3))
    return dict(record_id='r', checkpoint_id='K6', prompt_ids=prompt, response_ids=response,
        max_new_tokens=5, stop_token_ids=[99], finish_reason='length',
        metadata=dict(trace=blocks, verification_calls=4, block_size=16,
            runtime='pinned_SGLang_FA3', capture=dict(prefill_chunks=1),
            cache_policy='idle_full_flush_before_every_request_and_publication'))


def keys_fixture():
    return [dict(request_id='r', canonical_start=i, round_ordinal=i) for i in range(4)]


def observed(record, ordinal):
    block = deepcopy(record['metadata']['trace'][ordinal])
    end = block['output_start'] + block['committed_progress'] + 1
    return dict(status='completed', blocks=[block], response_ids=record['response_ids'][:end],
        output_matches_canonical_prefix=True, native_finished=block['finished'],
        original_cap=record['max_new_tokens'], cap_unchanged=True,
        reconstruction=dict(status='passed'))


class SourceAndGateTests(unittest.TestCase):
    def test_saved_parent_and_four_sealed_keys_are_exact(self):
        source = record_fixture()
        self.assertEqual(native.validate_parent(source, keys_fixture())[1], source['response_ids'])
        for mutate in (lambda r: r['metadata'].update(verification_calls=3),
                       lambda r: r['metadata']['capture'].update(prefill_chunks=2),
                       lambda r: r['metadata']['trace'][0]['draft_tokens'].pop()):
            changed = deepcopy(source)
            mutate(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                native.validate_parent(changed, keys_fixture())
        wrong = keys_fixture()
        wrong[0]['canonical_start'] = 19
        with self.assertRaises(ValueError):
            native.validate_parent(source, wrong)

    def test_gate_checks_all_fifteen_including_unscored_tail_before_target(self):
        calls = []
        class Verify:
            def prepare_for_verify(self, batch):
                calls.append('target_preparation')
        original = Verify.prepare_for_verify
        block = record_fixture()['metadata']['trace'][0]
        info = Verify()
        info.draft_token = torch.tensor([block['anchor_token']] + block['draft_tokens'])
        info.positions = torch.tensor(block['positions'])
        info.draft_token_num = 16
        with native.require_saved_proposals(block, Verify) as receipt:
            info.prepare_for_verify(None)
        self.assertEqual(receipt['calls'], 1)
        self.assertIs(Verify.prepare_for_verify, original)
        info.draft_token[-1] += 1
        with self.assertRaisesRegex(ValueError, 'proposals/positions'):
            with native.require_saved_proposals(block, Verify):
                info.prepare_for_verify(None)
        self.assertEqual(calls, ['target_preparation'])
        self.assertIs(Verify.prepare_for_verify, original)

    def test_missing_gate_call_and_posterior_mismatch_fail(self):
        class Verify:
            def prepare_for_verify(self, batch):
                pass
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            with native.require_saved_proposals(record_fixture()['metadata']['trace'][0], Verify):
                pass
        source = record_fixture()
        row = deepcopy(source['metadata']['trace'][0])
        row['target_argmax'][-1] += 1
        with self.assertRaisesRegex(ValueError, 'posterior/commit'):
            native.validate_replayed_block(source, 0, row, source['response_ids'][:2], False)


class CacheTests(unittest.TestCase):
    def test_only_selected_boundaries_and_destination_cache_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler, batch = runtime_fixture(cap=5, canonical=(20, 21, 22, 23, 24))
            worker = scheduler.draft_worker
            session = native.ProbeSuccessors(scheduler,
                dict(request_id='r', prompt_ids=[10, 11, 12], max_new_tokens=5),
                'K6', [0, 1, 2, 3], WorkBudget(Path(folder) / 'work.jsonl'))
            session.install()
            try:
                seed = worker.forward_batch_generation(batch)
                batch.reqs[0].output_ids.append(seed)
                batch.forward_mode = Mode()
                while not batch.reqs[0].finished():
                    worker.forward_batch_generation(batch)
            finally:
                session.close()
            self.assertEqual(set(session.successors), {0, 1, 2, 3})
            before = native.cache_storage_bytes([session])
            native.compact_prefix_caches(session)
            self.assertLess(native.cache_storage_bytes([session]), before)
            for state in session.successors.values():
                native.offload_successor_metadata(state)
            state = session.successors[2]
            session.rebuild(state, 'K6')
            self.assertTrue(session.reconstruction['neutral_draft_prefix_cache_exact'])
            worker.coefficient = 3.
            session.rebuild(state, 'T')
            self.assertTrue(session.reconstruction['target_prefix_cache_exact'])
            self.assertFalse(session.reconstruction['source_draft_cache_transplanted'])
            self.assertFalse(tensor_bits_equal(worker.draft_model_runner.token_to_kv_pool.rows[0][0][state.locations],
                                               state.draft_reference_cache[0][0]))

    def test_opening_not_selected_is_not_captured(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler, batch = runtime_fixture(cap=6, canonical=(20, 21, 22, 23, 24, 25))
            session = native.ProbeSuccessors(scheduler,
                dict(request_id='r', prompt_ids=[10, 11, 12], max_new_tokens=6),
                'K6', [1, 2, 3, 4], WorkBudget(Path(folder) / 'work.jsonl'))
            session.install()
            try:
                seed = scheduler.draft_worker.forward_batch_generation(batch)
                batch.reqs[0].output_ids.append(seed)
                batch.forward_mode = Mode()
                while not batch.reqs[0].finished():
                    scheduler.draft_worker.forward_batch_generation(batch)
            finally:
                session.close()
            self.assertEqual(set(session.successors), {1, 2, 3, 4})


class ScoringTests(unittest.TestCase):
    def test_first_error_and_no_native_serving_claim(self):
        source = record_fixture()
        row = client.score_observation(source, keys_fixture()[0], observed(source, 0),
                                       window=8, state='T', checkpoint='T')
        self.assertEqual(row['accepted_prefix'], 0)
        self.assertEqual(row['first_error'], 1)
        self.assertTrue(row['diagnostic_only'])
        self.assertNotIn('verification_calls', row)

    def test_matched_eos_is_valid_and_cap_match_is_censored(self):
        source = record_fixture()
        source['response_ids'] = [20, 99]
        source['max_new_tokens'] = 2048
        source['finish_reason'] = 'eos'
        block = source['metadata']['trace'][0]
        block.update(draft_tokens=[99]+[999]*14, target_argmax=[99]+[888]*15,
            accepted_draft_tokens=1, committed_tokens=[99], raw_accepted_draft_tokens_before_stop=1,
            stop_clipped=True, finished=True)
        got = client.score_observation(source, keys_fixture()[0], observed(source, 0),
                                       window=8, state='T', checkpoint='T')
        self.assertEqual((got['accepted_prefix'], got['valid_proposals'], got['first_error']), (1, 1, None))
        self.assertTrue(got['matched_eos'])
        self.assertFalse(got['censored_at_cap'])
        source['response_ids'] = [20, 21]
        source['max_new_tokens'] = 2
        source['finish_reason'] = 'length'
        block.update(draft_tokens=[21]+[999]*14, target_argmax=[21]+[888]*15, committed_tokens=[21])
        got = client.score_observation(source, keys_fixture()[0], observed(source, 0),
                                       window=8, state='T', checkpoint='T')
        self.assertTrue(got['censored_at_cap'])
        self.assertFalse(got['matched_eos'])

    def test_wrong_input_or_terminal_cannot_be_scored(self):
        source = record_fixture()
        for mutate in (lambda r: r.update(response_ids=[999]),
                       lambda r: r.update(native_finished=True),
                       lambda r: r['blocks'][0]['positions'].__setitem__(0, 999)):
            raw = observed(source, 0)
            mutate(raw)
            with self.assertRaises(ValueError):
                client.score_observation(source, keys_fixture()[0], raw,
                                         window=8, state='T', checkpoint='T')

    def test_export_grid_requires_all_states_and_explicit_valid_reuse(self):
        rows = [dict(window=w, state=s, checkpoint=f'{w}/{s}')
                for w in client.WINDOWS for s in client.STATES]
        self.assertEqual(len(client.validate_exports(rows)), 64)
        with self.assertRaises(ValueError):
            client.validate_exports(rows[:-1])
        rows[2].update(reused_from='N', checkpoint=rows[1]['checkpoint'])
        client.validate_exports(rows)
        rows[2]['checkpoint'] = 'different'
        with self.assertRaises(ValueError):
            client.validate_exports(rows)


class ClientExecutionTests(unittest.TestCase):
    def test_initial_checkpoint_is_explicitly_published_once(self):
        backend = object.__new__(client.ProbeBackend)
        backend.backend = SimpleNamespace(checkpoint_id='K6', publish=Mock())
        backend.published_checkpoint = None
        backend.select_checkpoint('K6')
        backend.select_checkpoint('K6')
        backend.select_checkpoint('T')
        self.assertEqual(backend.backend.publish.call_args_list[0].args, ('K6',))
        self.assertEqual(backend.backend.publish.call_count, 2)

    def test_reconstruction_receives_the_sealed_host_reserve(self):
        backend = object.__new__(client.ProbeBackend)
        backend.backend = SimpleNamespace(stop_tokens=[99])
        backend.select_checkpoint = Mock()
        backend._rpc = Mock(return_value={})
        backend.host_reserve_bytes = 32 * 1024**3
        backend.reconstruct(record_fixture(), keys_fixture(), 'K6')
        self.assertEqual(backend._rpc.call_args.kwargs['host_reserve_bytes'], 32 * 1024**3)

    def test_reverse_alias_is_allowed_but_cycle_is_not(self):
        exports = {(8, state): dict(window=8, state=state, checkpoint=state) for state in client.STATES}
        exports[(8, 'U')].update(checkpoint='T', reused_from='T')
        self.assertEqual(client.ordered_states(exports, 8), ['P', 'N', 'T', 'U'])
        exports[(8, 'T')]['reused_from'] = 'U'
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            client.ordered_states(exports, 8)

    def test_partial_state_failure_preserves_completed_and_invalid_rows(self):
        source, keys = record_fixture(), keys_fixture()
        exports = {(8, state): dict(window=8, state=state, checkpoint=state) for state in client.STATES}
        first, bad = observed(source, 0), observed(source, 1)
        bad['response_ids'] = [999]
        backend = SimpleNamespace(one_block=Mock(side_effect=[first, bad]))
        snapshots = {client.probe_key(key): {} for key in keys}
        receipt = dict(physical_probe_blocks=0, probe_attempts=0, invalid_probe_rows=0,
                       conceptual_probe_rows=0, reused_probe_rows=0)
        outcomes = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaises(ValueError):
                client.evaluate_exports(backend, exports, keys, {'r': source}, snapshots,
                    path, SimpleNamespace(check_time=lambda: None), receipt, outcomes)
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(outcomes[0]['valid'])
            self.assertFalse(outcomes[1]['valid'])
            self.assertEqual(receipt['probe_attempts'], 2)
            self.assertEqual(receipt['physical_probe_blocks'], 2)
            self.assertEqual(receipt['invalid_probe_rows'], 1)
            self.assertEqual(len((path / 'PROBE_OUTCOMES.jsonl').read_text().splitlines()), 2)
            self.assertTrue((path / 'window_008/P/r/0001.json').exists())

    def test_explicit_no_threat_aliases_preserve_zero_contrasts_without_work(self):
        source, keys = record_fixture(), keys_fixture()
        exports = {(8, state): dict(window=8, state=state, checkpoint=state) for state in client.STATES}
        for arm in ('U', 'T'):
            exports[(8, arm)].update(checkpoint='N', reused_from='N')
        responses = [observed(source, i) for i in range(4)] * 2
        backend = SimpleNamespace(one_block=Mock(side_effect=responses))
        snapshots = {client.probe_key(key): {} for key in keys}
        receipt = dict(physical_probe_blocks=0, probe_attempts=0, invalid_probe_rows=0,
                       conceptual_probe_rows=0, reused_probe_rows=0)
        outcomes = []
        with tempfile.TemporaryDirectory() as folder:
            client.evaluate_exports(backend, exports, keys, {'r': source}, snapshots,
                Path(folder), SimpleNamespace(check_time=lambda: None), receipt, outcomes)
        self.assertEqual(len(outcomes), 16)
        self.assertEqual(backend.one_block.call_count, 8)
        self.assertEqual(receipt['reused_probe_rows'], 8)
        self.assertTrue(all(row['reused_from'] == 'N' for row in outcomes[8:]))


class ReconstructionEntryTests(unittest.TestCase):
    def execute(self, folder, *, corrupt=False, available=100 * 1024**3):
        source = record_fixture()
        scheduler, batch = runtime_fixture(cap=5, canonical=tuple(source['response_ids']))
        batch.reqs[0].check_finished = lambda: None
        scheduler._online_publication = {'checkpoint_path': 'K6'}
        ordinary = scheduler.draft_worker.forward_batch_generation
        class Verify:
            def prepare_for_verify(self, batch):
                return None
        def forward(batch, **kwargs):
            recorder = native.sglang_hooks._ACTIVE
            if batch.forward_mode.is_extend():
                seed = ordinary(batch, **kwargs)
                recorder.prefill(0, source['prompt_ids'], seed, block_size=16)
                return SimpleNamespace(next_token_ids=torch.tensor([seed]))
            ordinal = len(batch.reqs[0].output_ids) - 1
            block = source['metadata']['trace'][ordinal]
            verify = Verify()
            verify.draft_token = torch.tensor([block['anchor_token']] + block['draft_tokens'])
            verify.positions = torch.tensor(block['positions'])
            verify.draft_token_num = 16
            verify.prepare_for_verify(batch)
            before = list(batch.reqs[0].output_ids)
            result = ordinary(batch, **kwargs)
            recorder.verification(before=before, after=list(batch.reqs[0].output_ids),
                candidates=verify.draft_token.tolist(), positions=block['positions'],
                posterior=list(block['target_argmax']), commit_rows=1, native_accepted=0,
                finished=batch.reqs[0].finished())
            if corrupt:
                recorder.trace[-1]['target_argmax'][-1] += 1
            return result
        scheduler.draft_worker.forward_batch_generation = forward
        budget = WorkBudget(Path(folder) / 'work.jsonl')
        modules = {
            'sglang.srt.model_executor.forward_batch_info': SimpleNamespace(ForwardMode=SimpleNamespace(DECODE=Mode())),
            'sglang.srt.speculative.dflash_info': SimpleNamespace(DFlashVerifyInput=Verify),
        }
        with patch.dict('sys.modules', modules), patch.object(native.base, '_BUDGET', budget), \
             patch.object(native, 'make_batch', return_value=batch), \
             patch.object(native, 'check_cache_prefix', side_effect=lambda b, n, old=None: torch.arange(n)), \
             patch.object(native.sglang_hooks, 'strict_idle'), patch.object(native.sglang_hooks, 'flush_idle', return_value=0.), \
             patch.object(native, 'host_memory_available', return_value=available), \
             patch.object(torch.cuda, 'synchronize'):
            return native.transition_reconstruct_saved(scheduler, source, keys_fixture(), 'K6',
                                                        str(Path(folder) / 'reconstruct.json'))

    def setUp(self):
        self.original_sessions = list(native.base._SESSIONS)
        native.base._SESSIONS.clear()
        self.addCleanup(lambda: native.base._SESSIONS.__setitem__(slice(None), self.original_sessions))

    def test_full_saved_path_admits_only_parity_checked_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.execute(folder)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['rounds_replayed'], 4)
            self.assertEqual(set(result['snapshots']), {'0', '1', '2', '3'})
            self.assertEqual(len(native.base._SESSIONS), 1)
            self.assertIsNone(native.base._SESSION)
            self.assertIsNone(native.sglang_hooks._ACTIVE)
            self.assertEqual(result['new_generations'], 0)
            self.assertEqual(result['host_reserve_bytes'], 32 * 1024**3)

    def test_native_snapshot_admission_keeps_32_gib_host_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(MemoryError, 'sealed host-memory reserve'):
                self.execute(folder, available=31 * 1024**3)
            self.assertEqual(len(native.base._SESSIONS), 0)
            self.assertEqual(client.read(Path(folder) / 'reconstruct.json')['status'], 'failed')

    def test_failure_is_saved_without_admitting_partial_snapshots(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'posterior/commit'):
                self.execute(folder, corrupt=True)
            result = json.loads((Path(folder) / 'reconstruct.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(result['completed_trace']), 1)
            self.assertEqual(len(native.base._SESSIONS), 0)
            self.assertIsNone(native.base._SESSION)
            self.assertIsNone(native.sglang_hooks._ACTIVE)


class QualificationEntryTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        target, draft = root / 'target', root / 'K6'
        config = dict(num_hidden_layers=2, num_key_value_heads=1, num_attention_heads=1,
                      head_dim=4, hidden_size=4, dflash_config=dict(target_layer_ids=[0]))
        client.write_json(target / 'config.json', config)
        client.write_json(draft / 'config.json', config)
        source = record_fixture()
        source['checkpoint_id'] = str(draft)
        client.write_json(root / 'parent.json', source)
        keys = keys_fixture() + [dict(request_id=f'other_{i}', canonical_start=j, round_ordinal=j)
                                for i in range(63) for j in range(4)]
        client.write_json(root / 'keys.json', keys)
        plan = dict(target_path=str(target), checkpoints={'K6': {'export': str(draft)}},
                    probe_keys=str(root / 'keys.json'), probe_requests=[{'request_id': 'r'}],
                    parent_records={'r': str(root / 'parent.json')}, worker_python='unused')
        budget = WorkBudget(root / 'work.jsonl', limits={'draft_block': 20})
        backend = SimpleNamespace(loading_s=1., close=Mock())
        def charge(kind):
            ticket = budget.attempt(kind)
            budget.complete(ticket)
        def reconstruct(record, subset, checkpoint):
            self.assertEqual(record, source)
            self.assertEqual(subset, keys_fixture())
            self.assertEqual(checkpoint, str(draft))
            charge('model_load')  # Explicit K6 publication, separate from engine.
            charge('target_call')  # One saved prefill.
            for _ in range(4):
                charge('draft_block')
                charge('target_call')
            return dict(status='passed', rounds_replayed=4,
                        snapshots={str(i): {'ordinal': i} for i in range(4)})
        def one_block(snapshot, checkpoint):
            charge('context_forward')
            charge('draft_block')
            charge('target_call')
            return observed(source, snapshot['ordinal'])
        backend.reconstruct = Mock(side_effect=reconstruct)
        backend.one_block = Mock(side_effect=one_block)
        return plan, budget, backend

    def patches(self, stack, budget, backend, available=100 * 1024**3):
        stack.enter_context(patch('experiments.verification_transition.budget.connect_budget', return_value=budget))
        factory = stack.enter_context(patch.object(client, 'ProbeBackend', return_value=backend))
        stack.enter_context(patch.object(client, 'host_memory_available', return_value=available))
        return factory

    def test_disposable_qualification_uses_one_parent_four_starts_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            plan, budget, backend = self.fixture(folder)
            factory = self.patches(stack, budget, backend)
            result = client.qualify(plan, Path(folder) / 'qualification', {})
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['neutral_blocks_valid'], 4)
            self.assertEqual(result['unattempted_neutral_blocks'], 0)
            self.assertEqual(result['new_generations'], 0)
            self.assertEqual(result['adam_steps'], 0)
            self.assertTrue(result['worker_closed'])
            backend.close.assert_called_once()
            backend.reconstruct.assert_called_once()
            self.assertEqual(backend.one_block.call_count, 4)
            self.assertEqual(result['work_delta']['completed'],
                             {'model_load': 2, 'target_call': 9, 'draft_block': 8, 'context_forward': 4})
            self.assertEqual(factory.call_args.kwargs['host_reserve_bytes'], 32 * 1024**3)
            self.assertEqual(client.read(Path(folder) / 'qualification/RESULT.json'), result)

    def test_failed_neutral_keeps_raw_record_closes_worker_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            plan, budget, backend = self.fixture(folder)
            ordinary = backend.one_block.side_effect
            def bad(snapshot, checkpoint):
                raw = ordinary(snapshot, checkpoint)
                raw['blocks'][0]['draft_tokens'][-1] += 1
                return raw
            backend.one_block.side_effect = bad
            self.patches(stack, budget, backend)
            output = Path(folder) / 'qualification'
            with self.assertRaisesRegex(ValueError, 'all fifteen'):
                client.qualify(plan, output, {})
            result = client.read(output / 'RESULT.json')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['neutral_blocks_attempted'], 1)
            self.assertEqual(result['neutral_blocks_completed'], 1)
            self.assertEqual(result['neutral_blocks_valid'], 0)
            self.assertEqual(result['unattempted_neutral_blocks'], 3)
            self.assertTrue((output / '0000_NEUTRAL.json').exists())
            self.assertTrue(result['worker_closed'])
            backend.close.assert_called_once()
            backend.one_block.assert_called_once()

    def test_host_admission_stops_before_model_work_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            plan, budget, backend = self.fixture(folder)
            factory = self.patches(stack, budget, backend, available=31 * 1024**3)
            output = Path(folder) / 'qualification'
            with self.assertRaises(MemoryError):
                client.qualify(plan, output, {})
            result = client.read(output / 'RESULT.json')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['unattempted_neutral_blocks'], 4)
            self.assertEqual(result['work_delta']['attempted'], {})
            factory.assert_not_called()

    def test_resource_estimate_uses_sealed_reserve_and_default_32_gib(self):
        with tempfile.TemporaryDirectory() as folder:
            plan, _, _ = self.fixture(folder)
            estimate = client.resource_estimate(plan, keys_fixture())
            self.assertEqual((estimate['requests'], estimate['starts']), (1, 4))
            self.assertEqual(estimate['host_reserve_bytes'], 32 * 1024**3)
            plan['host_reserve_bytes'] = 40 * 1024**3
            self.assertEqual(client.resource_estimate(plan, keys_fixture())['host_reserve_bytes'], 40 * 1024**3)


if __name__ == '__main__':
    unittest.main()
