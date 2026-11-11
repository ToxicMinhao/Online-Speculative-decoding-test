"""CPU integration tests for all128 originals, the two-packet gate and path rebind."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.ok_coding.storage import identity
from experiments.survival_head.feedback import compare_prepared_packets
from experiments.survival_head.prepare_feedback import (authenticate_inputs, prepare, rebound_schedule,
                                                       retire_redundant_capture, run)
from online_feedback.storage import write_json
from test_feedback import prepared_packet, source_record


def fixture(root):
    old = root / 'original'
    rows, records, requests, features = [], [], [], []
    for index in range(128):
        record = source_record()
        identifier = f'feedback_{index:03d}'
        record['record_id'] = identifier
        records.append(record)
        canonical, feature = old / f'{index:03d}.json', old / f'{index:03d}.pt'
        write_json(canonical, record)
        packet = prepared_packet(record)
        if index < 2:
            torch.save(packet, feature)
            feature_identity = identity(feature)
        else:
            feature_identity = dict(bytes=12345, sha256='f' * 64)  # honestly retired, not required on disk
        rows.append(dict(record_id=identifier,
            canonical=dict(path=str(canonical), **identity(canonical)),
            feature=dict(path=str(feature), **feature_identity)))
        features.append(dict(record_id=identifier, length=8, prompt_length=2,
            response_tokens=6, adjacent_response_positions=5, path=str(feature)))
        requests.append(dict(request_id=identifier, prompt_ids=[2, 3], dataset='gsm8k',
            event_id='event_005', max_new_tokens=2048, collect_feedback=True))
    pool = old / 'POOL.json'
    write_json(pool, dict(dataset='gsm8k', event=5, producer='KPLUS4', records=rows))
    schedule = old / 'RESULT.json'
    write_json(schedule, dict(status='completed', dataset='gsm8k', method='KPLUS',
        event_id='event_005', start_step=512, end_step=640, optimizer_steps=128,
        passes=2, response_presentations=256, features=features,
        records=[dict(record_id=r['record_id'], **{'pass': p}, sampled_anchors=[[1, 2, 3]])
                 for p in (0, 1) for r in records]))
    manifest = old / 'TRAINING.json'
    write_json(manifest, dict(dataset='gsm8k', start_ordinal=512, stop_ordinal=768, train_requests=requests))
    return dict(root=str(root / 'pilot'), prior_pool=str(pool), prior_pool_identity=identity(pool),
        original_schedule=str(schedule), original_schedule_identity=identity(schedule),
        retained_packet_paths=[row['feature']['path'] for row in rows[:2]],
        training_manifest=str(manifest), origin_export='/original/KPLUS4',
        target='/target', worker_python='/worker/python', serving_gpu='0')


def fake_reconstruct(backend, record, path, *, retained_packet=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    packet = prepared_packet(record)
    torch.save(packet, path)
    capture = path.with_suffix('.replay.capture.pt')
    terminal = path.with_suffix('.terminal.pt')
    torch.save(dict(input_ids=packet['input_ids'][0, :-1].tolist(),
        hidden_states=packet['hidden_states'][0, :-1],
        target_last_hidden_states=packet['target_last_hidden_states'][0, :-1]), capture)
    torch.save(dict(input_ids=packet['input_ids'][0].tolist(), positions=[7],
        final_feature_logits_used_for_labels=False, hidden_states=packet['hidden_states'][0, -1:],
        target_last_hidden_states=packet['target_last_hidden_states'][0, -1:]), terminal)
    comparison = None if retained_packet is None else compare_prepared_packets(retained_packet, path)
    result = dict(status='passed', record_id=record['record_id'], packet_comparison=comparison,
        terminal_attempts=1, terminal_forwards=1,
        preparation=dict(terminal_feature=dict(tensor_path=str(terminal))),
        replay=dict(status='passed', full_posterior_match=True, all_commits_exact=True,
            prefill_attempts=1, prefill_forwards=1, verification_attempts=2, verification_forwards=2,
            record=dict(metadata=dict(capture=dict(feature_path=str(capture))))))
    write_json(path.with_suffix('.reconstruction.json'), result)
    return result


class FeedbackPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = fixture(self.root)

    def test_complete_pool_uses_two_reused_bytes_then_only126_reconstructed_finals(self):
        backend = Mock(loading_s=1.5)
        factory = Mock(return_value=backend)
        with patch('experiments.survival_head.prepare_feedback.reconstruct_feedback', side_effect=fake_reconstruct) as replay:
            result = prepare(self.plan, backend_factory=factory)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(replay.call_count, 128)
        self.assertEqual(result['observed_work']['target_forwards'], 512)
        self.assertEqual(result['observed_work']['verification_forwards'], 256)
        self.assertEqual(result['observed_work']['terminal_forwards'], 128)
        self.assertEqual(result['observed_work']['new_generations'], 0)
        self.assertEqual(result['reused_original_packet_bytes'], 2)
        self.assertEqual(result['reconstructed_missing_packets'], 126)
        self.assertEqual(factory.call_args.kwargs['worker_module'], 'online_feedback.survival_feedback_worker')
        for index in range(2):
            self.assertEqual(identity(Path(self.plan['root']) / 'feedback/features' / f'{index:03d}.pt'),
                             identity(self.plan['retained_packet_paths'][index]))
            self.assertIsNotNone(replay.call_args_list[index].kwargs['retained_packet'])
        self.assertTrue(all(call.kwargs['retained_packet'] is None for call in replay.call_args_list[2:]))
        backend.generate.assert_not_called()
        backend.close.assert_called_once()
        from experiments.survival_head.data import read, verify_shared_packets
        original = read(self.plan['original_schedule'])
        rebound = read(result['presentation_schedule'])
        self.assertEqual(original['records'], rebound['records'])
        self.assertNotEqual(original['features'][0]['path'], rebound['features'][0]['path'])
        rebound['features'] = original['features']
        self.assertEqual(original, rebound)
        self.assertEqual(verify_shared_packets(read(result['pool']))['status'], 'passed')
        self.assertFalse(list((Path(self.plan['root']) / 'feedback').rglob('*.replay.capture.pt')))
        self.assertFalse(list((Path(self.plan['root']) / 'feedback').rglob('*.terminal.pt')))

    def test_qualification_mismatch_blocks_all126_missing_reconstructions(self):
        backend = Mock(loading_s=1.5)
        calls = []
        def failing(owner, record, path, **kwargs):
            result = fake_reconstruct(owner, record, path, **kwargs)
            calls.append(record['record_id'])
            result['packet_comparison']['full_packet_bitwise_equal'] = False
            return result
        with patch('experiments.survival_head.prepare_feedback.reconstruct_feedback', side_effect=failing), \
             self.assertRaisesRegex(ValueError, 'not passed exactly'):
            prepare(self.plan, backend_factory=Mock(return_value=backend))
        self.assertEqual(calls, ['feedback_000', 'feedback_001'])
        self.assertFalse((Path(self.plan['root']) / 'feedback/features/002.pt').exists())
        self.assertFalse((Path(self.plan['root']) / 'feedback/POOL.json').exists())
        backend.close.assert_called_once()

    def test_all_original_records_are_authenticated_before_backend_start(self):
        (self.root / 'original/127.json').write_text('{}')
        factory = Mock()
        with self.assertRaisesRegex(ValueError, 'canonical JSON identity'):
            prepare(self.plan, backend_factory=factory)
        factory.assert_not_called()
        self.assertFalse(Path(self.plan['root']).exists())

    def test_original_schedule_identity_and_event_boundary_are_checked(self):
        from experiments.survival_head.data import read
        schedule = read(self.plan['original_schedule'])
        schedule['start_step'] = 640
        write_json(self.plan['original_schedule'], schedule)
        with self.assertRaisesRegex(ValueError, 'Sealed original input'):
            authenticate_inputs(self.plan)
        self.plan['original_schedule_identity'] = identity(self.plan['original_schedule'])
        with self.assertRaisesRegex(ValueError, 'event boundary'):
            authenticate_inputs(self.plan)

    def test_rebind_rejects_metadata_change(self):
        authenticated = authenticate_inputs(self.plan)
        schedule = deepcopy(authenticated['schedule'])
        schedule['features'][0]['length'] += 1
        with self.assertRaisesRegex(ValueError, 'changed feature metadata'):
            rebound_schedule(schedule, authenticated['records'],
                [self.root / f'pilot_{i:03d}.pt' for i in range(128)])

    def test_retired126_source_features_are_never_required_or_opened(self):
        authenticated = authenticate_inputs(self.plan)
        self.assertEqual(authenticated['expected_work']['prefill_forwards'], 128)
        self.assertEqual(authenticated['expected_work']['terminal_forwards'], 128)
        self.assertEqual(authenticated['qualification_record_ids'], ['feedback_000', 'feedback_001'])
        self.assertFalse((self.root / 'original/002.pt').exists())

    def test_retirement_records_proven_redundancy_and_preserves_prepared_copy(self):
        path = self.root / 'pilot/feedback/features/000.pt'
        result = fake_reconstruct(None, source_record(), path)
        original = identity(path)
        receipt = retire_redundant_capture(path, result, root=self.root / 'pilot/feedback')
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(len(receipt['removed_paths']), 2)
        self.assertEqual(identity(path), original)
        self.assertIn('free_bytes_before', receipt)
        self.assertIn('free_bytes_after', receipt)

    def test_retirement_rejects_nonidentical_rows_without_deleting_either_copy(self):
        path = self.root / 'pilot/feedback/features/000.pt'
        result = fake_reconstruct(None, source_record(), path)
        capture = path.with_suffix('.replay.capture.pt')
        raw = torch.load(capture, weights_only=False)
        raw['hidden_states'][0, 0] += 1
        torch.save(raw, capture)
        with self.assertRaisesRegex(ValueError, 'rows differ'):
            retire_redundant_capture(path, result, root=self.root / 'pilot/feedback')
        self.assertTrue(capture.is_file())
        self.assertTrue(path.with_suffix('.terminal.pt').is_file())

    def test_retirement_never_follows_original_capture_path(self):
        path = self.root / 'pilot/feedback/features/000.pt'
        result = fake_reconstruct(None, source_record(), path)
        result['replay']['record']['metadata']['capture']['feature_path'] = str(self.root / 'original/000.pt')
        with self.assertRaisesRegex(ValueError, 'newly created'):
            retire_redundant_capture(path, result, root=self.root / 'pilot/feedback')
        self.assertTrue((self.root / 'original/000.pt').is_file())

    def test_direct_entry_rechecks_priority_before_loading_any_backend(self):
        with patch('experiments.survival_head.preflight.verify', side_effect=ValueError('PR36 owns window')) as admission, \
             patch('experiments.survival_head.prepare_feedback.prepare') as helper, \
             self.assertRaisesRegex(ValueError, 'PR36 owns window'):
            run(self.plan['root'])
        admission.assert_called_once_with(self.plan['root'], scope='qualification')
        helper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
