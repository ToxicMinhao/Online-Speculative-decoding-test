"""CPU source and tensor contracts for retired teacher reconstruction."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.survival_head.feedback import (
    compare_prepared_packets, expected_reconstruction_work, reconstruct_feedback,
    require_reconstruction_qualification, source_contract_sha256, validate_saved_record,
)
from online_feedback.sglang_hooks import RequestCapture
from online_feedback.survival_feedback_worker import OPERATIONS, survival_teacher
from types import SimpleNamespace


def source_record():
    with tempfile.TemporaryDirectory() as directory:
        capture = RequestCapture([2, 3], Path(directory) / 'capture.json', False)
        capture.prefill(0, [2, 3], 10, block_size=16)
        capture.verification(before=[10], after=[10, 11, 12, 20],
            candidates=[10, 11, 12] + [88] * 13, positions=list(range(2, 18)),
            posterior=[11, 12, 20] + [66] * 13, commit_rows=3,
            native_accepted=2, finished=False)
        capture.verification(before=[10, 11, 12, 20], after=[10, 11, 12, 20, 21, 99],
            candidates=[20, 21, 99] + [77] * 13, positions=list(range(5, 21)),
            posterior=[21, 99, 50] + [67] * 13, commit_rows=2,
            native_accepted=1, finished=True)
        document = capture.finish([10, 11, 12, 20, 21, 99])
    document['collect_features'] = True
    document['feature_path'] = '/retired/original/native_capture.pt'
    return dict(record_id='gsm8k_feedback_001', checkpoint_id='/original/KPLUS4', purpose='serving',
        prompt_ids=[2, 3], response_ids=[10, 11, 12, 20, 21, 99], stop_token_ids=[99],
        max_new_tokens=2048, finish_reason='eos', metadata=dict(
            runtime='pinned_SGLang_FA3', block_size=16, verification_calls=2,
            cache_policy='idle_full_flush_before_every_request_and_publication',
            trace=deepcopy(document['blocks']), capture=document))


def prepared_packet(record=None):
    record = record or source_record()
    tokens = record['prompt_ids'] + record['response_ids']
    hidden = torch.arange(len(tokens) * 4).reshape(1, len(tokens), 4).to(torch.bfloat16)
    return dict(input_ids=torch.tensor([tokens]),
        loss_mask=torch.tensor([[0, 0] + [1] * len(record['response_ids'])], dtype=torch.float32),
        hidden_states=hidden, target_last_hidden_states=hidden[:, :, :2].clone(),
        record_id=record['record_id'], prompt_length=2,
        label_binding=dict(capture_backend='same_frozen_SGLang_native_capture',
            native_capture_validated=True, label_binding_scope='native_prefill_and_committed_verification',
            final_feature_logits_used_for_labels=False, input_ids=tokens,
            target_argmax_ids=record['response_ids'], target_layer_ids=[1, 2],
            prompt_length=2, record_id=record['record_id'], label_mismatches=[]))


class FeedbackSourceTests(unittest.TestCase):
    def test_valid_record_requires_no_retired_tensor_to_validate(self):
        record = source_record()
        _, response, blocks = validate_saved_record(record)
        self.assertEqual(response, record['response_ids'])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(expected_reconstruction_work(record), dict(
            new_generations=0, draft_forwards=0, prefill_forwards=1,
            verification_forwards=2, terminal_forwards=1, target_forwards=4,
            prefill_input_rows=2, verification_input_rows=32, terminal_input_rows=8,
            retained_native_rows=7, retained_terminal_rows=1))

    def test_rejects_unretained_prefill_chunk_boundaries(self):
        record = source_record()
        record['metadata']['capture']['prefill_chunks'] = 2
        with self.assertRaisesRegex(ValueError, 'chunk geometry'):
            validate_saved_record(record)

    def test_rejects_changed_canonical_commit_or_native_geometry(self):
        for key, value in [('positions', list(range(3, 19))), ('verify_size', 15),
                           ('committed_progress', 4), ('finished', True),
                           ('native_accepted_draft_tokens', 3), ('stop_clipped', True),
                           ('draft_tokens', [99] * 15), ('target_argmax', [7] * 16)]:
            with self.subTest(key=key):
                record = source_record()
                for trace in (record['metadata']['trace'], record['metadata']['capture']['blocks']):
                    trace[0][key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_saved_record(record)

    def test_source_digest_ignores_retired_path_and_timing_but_binds_discarded_posterior(self):
        record = source_record()
        digest = source_contract_sha256(record)
        record['metadata']['capture']['feature_path'] = '/pilot/new/path.pt'
        record['metadata']['capture']['time_to_first_token_s'] = 999
        self.assertEqual(digest, source_contract_sha256(record))
        for trace in (record['metadata']['trace'], record['metadata']['capture']['blocks']):
            trace[0]['target_argmax'][-1] += 1
        self.assertNotEqual(digest, source_contract_sha256(record))

    def test_no_generation_worker_operation(self):
        self.assertNotIn('generate', OPERATIONS)
        self.assertEqual(set(OPERATIONS), {'reconstruct', 'teacher', 'check_weights'})

    def test_qualification_needs_exactly_two_distinct_matching_passes(self):
        def passed(identifier):
            return dict(status='passed', record_id=identifier,
                packet_comparison=dict(status='passed', full_packet_bitwise_equal=True),
                replay=dict(status='passed', full_posterior_match=True, all_commits_exact=True))
        rows = [passed('a'), passed('b')]
        self.assertEqual(require_reconstruction_qualification(rows, ['a', 'b'])['status'], 'passed')
        for malformed, ids in [(rows[:1], ['a']), (rows, ['a', 'a']), (rows, ['b', 'a'])]:
            with self.assertRaises(ValueError):
                require_reconstruction_qualification(malformed, ids)
        rows[1]['packet_comparison']['full_packet_bitwise_equal'] = False
        with self.assertRaisesRegex(ValueError, 'not passed exactly'):
            require_reconstruction_qualification(rows, ['a', 'b'])


class FeedbackPacketTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_bitwise_packet_equality_checks_native_and_terminal_rows_separately(self):
        reference = prepared_packet()
        expected, actual = self.root / 'old.pt', self.root / 'new.pt'
        torch.save(reference, expected)
        torch.save(deepcopy(reference), actual)
        result = compare_prepared_packets(expected, actual)
        self.assertTrue(result['full_packet_bitwise_equal'])
        reference['target_last_hidden_states'][0, -1, 0] += 1
        torch.save(reference, actual)
        result = compare_prepared_packets(expected, actual)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['fields']['target_last_hidden_states']['native_rows_exact'])
        self.assertFalse(result['fields']['target_last_hidden_states']['terminal_row_exact'])

    def test_native_row_mismatch_and_nonfinite_are_not_tolerance_accepted(self):
        expected, actual = self.root / 'old.pt', self.root / 'new.pt'
        reference = prepared_packet()
        torch.save(reference, expected)
        reference['hidden_states'][0, 2, 1] += 1
        torch.save(reference, actual)
        result = compare_prepared_packets(expected, actual)
        self.assertFalse(result['fields']['hidden_states']['native_rows_exact'])
        self.assertTrue(result['fields']['hidden_states']['terminal_row_exact'])
        reference['hidden_states'][0, 2, 1] = float('nan')
        torch.save(reference, actual)
        self.assertFalse(compare_prepared_packets(expected, actual)['fields']['hidden_states']['finite'])

    def test_orchestrator_replays_then_uses_original_terminal_and_preserves_source(self):
        record = source_record()
        before = deepcopy(record)
        restored = deepcopy(record)
        replay = dict(status='passed', source_contract_sha256=source_contract_sha256(record),
            full_posterior_match=True, all_commits_exact=True, record=restored,
            prefill_forwards=1, verification_forwards=2)
        backend = Mock()
        backend._call.return_value = replay
        backend.prepare_feedback.return_value = dict(status='prepared', terminal_feature=dict(target_forward_count=1))
        result = reconstruct_feedback(backend, record, self.root / 'features.pt')
        self.assertEqual(result['target_forwards'], 4)
        self.assertEqual(result['terminal_forwards'], 1)
        self.assertFalse(result['native_tensor_parity_qualified'])
        self.assertEqual(record, before)
        self.assertEqual(backend._call.call_args.args, ('reconstruct',))
        backend.generate.assert_not_called()
        self.assertEqual(backend.prepare_feedback.call_args.args[0], restored)

    def test_terminal_failure_retains_attempt_and_does_not_repeat_target_replay(self):
        record = source_record()
        backend = Mock()
        backend._call.return_value = dict(status='passed',
            source_contract_sha256=source_contract_sha256(record), record=record,
            full_posterior_match=True, all_commits_exact=True)
        backend.prepare_feedback.side_effect = RuntimeError('terminal failure')
        with self.assertRaisesRegex(RuntimeError, 'terminal failure'):
            reconstruct_feedback(backend, record, self.root / 'features.pt')
        receipt = json.loads((self.root / 'features.reconstruction.json').read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['terminal_attempts'], 1)
        self.assertEqual(receipt['terminal_forwards'], 0)
        self.assertEqual(backend._call.call_count, 1)

    def test_terminal_target_completion_survives_later_packet_failure(self):
        runner = SimpleNamespace(forward=Mock(return_value='native result'))
        original = runner.forward
        scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
        def teacher(owner, **kwargs):
            self.assertEqual(owner.tp_worker.model_runner.forward('input'), 'native result')
            raise RuntimeError('tensor persistence failed after forward')
        path = self.root / 'features.terminal.json'
        with patch('online_feedback.survival_feedback_worker.sglang_hooks.online_teacher', side_effect=teacher), \
             self.assertRaisesRegex(RuntimeError, 'persistence failed'):
            survival_teacher(scheduler, [1, 2, 3], str(path))
        work = json.loads(path.with_suffix('.work.json').read_text())
        self.assertEqual(work['status'], 'failed')
        self.assertEqual(work['target_forward_attempts'], 1)
        self.assertEqual(work['target_forward_completions'], 1)
        self.assertIs(runner.forward, original)

    def test_terminal_counted_wrapper_rejects_a_second_forward(self):
        runner = SimpleNamespace(forward=Mock(return_value='native result'))
        scheduler = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
        def teacher(owner, **kwargs):
            owner.tp_worker.model_runner.forward('first')
            owner.tp_worker.model_runner.forward('unallowed second')
        path = self.root / 'features.terminal.json'
        with patch('online_feedback.survival_feedback_worker.sglang_hooks.online_teacher', side_effect=teacher), \
             self.assertRaisesRegex(RuntimeError, 'one-forward allowance'):
            survival_teacher(scheduler, [1, 2, 3], str(path))
        work = json.loads(path.with_suffix('.work.json').read_text())
        self.assertEqual(work['target_forward_attempts'], 1)
        self.assertEqual(work['target_forward_completions'], 1)


if __name__ == '__main__':
    unittest.main()
