"""CPU-only scientific controls for the persistent OSD campaign driver."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.osd_persistent import run as R


class EvaluationBackend:
    """Small deterministic records; never imports or constructs a model."""

    def __init__(self, mutate=None):
        self.checkpoint_id = '/released'
        self.calls = []
        self.switches = []
        self.mutate = mutate

    def publish(self, checkpoint):
        self.checkpoint_id = str(checkpoint)
        self.switches.append(self.checkpoint_id)
        return {'checkpoint_id': self.checkpoint_id, 'activation_s': 1000.0}

    def generate(self, prompt_ids, **kwargs):
        method = 'F' if self.checkpoint_id == '/released' else 'O'
        index = len(self.calls)
        self.calls.append((method, list(prompt_ids), kwargs))
        response = [prompt_ids[0] + 10, 99]
        record = {
            'checkpoint_id': self.checkpoint_id,
            'prompt_ids': list(prompt_ids), 'response_ids': response,
            'finish_reason': 'eos', 'max_new_tokens': kwargs['max_new_tokens'],
            'metadata': {
                'verification_calls': 4 if method == 'F' else 3,
                'accepted_draft_tokens': 1, 'committed_progress': 1,
                'request_latency_s': 2.0 if method == 'F' else 1.0,
                'native_generate_s': 0.5,
                'feedback_capture_in_timed_request': False,
            },
        }
        if self.mutate:
            self.mutate(record, index, method)
        # The real backend writes the raw records; preserve that boundary here.
        R.write_json(kwargs['save_to'], record)
        return record


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'evaluation'
        self.requests = [
            {'request_id': 'heldout_1', 'request_index': 1,
             'prompt_ids': [1, 2], 'max_new_tokens': 8},
            {'request_id': 'heldout_2', 'request_index': 2,
             'prompt_ids': [3, 4], 'max_new_tokens': 8},
        ]
        self.log = patch.object(R, 'log')
        self.log.start()
        self.addCleanup(self.log.stop)

    def evaluate(self, backend, requests=None):
        return R.evaluate(backend, requests or self.requests,
                          '/released', '/trained', self.root)

    def test_three_fresh_repeats_rotate_order_and_exclude_activation_cost(self):
        backend = EvaluationBackend()
        summary = self.evaluate(backend)
        self.assertEqual(''.join(call[0] for call in backend.calls), 'FOOFFOOFFOOF')
        self.assertEqual(len(backend.calls), 12)
        self.assertTrue(all(call[2]['purpose'] == 'evaluation' for call in backend.calls))
        self.assertEqual(summary['pair_count'], 6)
        self.assertEqual(summary['valid_pairs'], 6)
        self.assertEqual(summary['invalid_pairs'], 0)
        self.assertEqual(summary['F']['request_latency_s'], 12.0)
        self.assertEqual(summary['O']['request_latency_s'], 6.0)
        self.assertEqual(summary['call_reduction_pct'], 25.0)
        self.assertEqual(summary['request_time_reduction_pct'], 50.0)
        self.assertEqual(len(list(self.root.glob('*_comparison.json'))), 6)
        with (self.root / 'PER_REQUEST.csv').open(newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)

    def test_disagreement_excludes_both_methods_and_preserves_raw_evidence(self):
        def mutate(record, index, method):
            if index == 1:
                record['response_ids'] = [777, 99]

        summary = self.evaluate(EvaluationBackend(mutate), self.requests[:1])
        self.assertEqual((summary['valid_pairs'], summary['invalid_pairs']), (2, 1))
        self.assertEqual(summary['F']['verification_calls'], 8)
        self.assertEqual(summary['O']['verification_calls'], 6)
        comparison = json.loads((self.root / '0000_r0_comparison.json').read_text())
        self.assertFalse(comparison['valid'])
        self.assertIn('canonical frozen output mismatch', comparison['errors'][0])
        raw = json.loads((self.root / '0000_r0_O.json').read_text())
        self.assertEqual(raw['response_ids'], [777, 99])

    def test_same_changed_outputs_in_a_later_repeat_remain_invalid(self):
        def mutate(record, index, method):
            if index in (2, 3):
                record['response_ids'] = [777, 99]

        summary = self.evaluate(EvaluationBackend(mutate), self.requests[:1])
        self.assertEqual((summary['valid_pairs'], summary['invalid_pairs']), (2, 1))
        comparison = json.loads((self.root / '0000_r1_comparison.json').read_text())
        self.assertEqual(len(comparison['errors']), 2)
        self.assertEqual(comparison['canonical_response_ids'], [11, 99])

    def test_input_or_capture_mismatch_excludes_a_complete_pair(self):
        def mutate(record, index, method):
            if index == 0:
                record['prompt_ids'] = [321]
            if index == 2:
                record['metadata']['feedback_capture_in_timed_request'] = True

        summary = self.evaluate(EvaluationBackend(mutate), self.requests[:1])
        self.assertEqual((summary['valid_pairs'], summary['invalid_pairs']), (1, 2))
        self.assertEqual(summary['distinct_requests'], 1)

    def test_all_invalid_pairs_produce_no_gain_claim(self):
        def mutate(record, index, method):
            if method == 'O':
                record['response_ids'] = [777, 99]

        summary = self.evaluate(EvaluationBackend(mutate), self.requests[:1])
        self.assertEqual((summary['valid_pairs'], summary['invalid_pairs']), (0, 3))
        self.assertEqual(summary['distinct_requests'], 0)
        self.assertIsNone(summary['call_reduction_pct'])
        self.assertIsNone(summary['request_time_reduction_pct'])
        self.assertIsNone(summary['F']['output_tokens_per_s'])

    def test_ratios_use_pooled_counts_not_mean_of_request_ratios(self):
        rows = []
        for request, frozen_calls, online_calls, tokens in [('a', 2, 1, 2), ('b', 98, 98, 100)]:
            for method, calls in [('F', frozen_calls), ('O', online_calls)]:
                rows.append({
                    'request_id': request, 'method': method, 'valid_pair': True,
                    'output_tokens': tokens, 'verification_calls': calls,
                    'accepted_draft_tokens': 0, 'committed_progress': tokens - 1,
                    'request_latency_s': float(calls),
                })
        summary = R.summarize_pairs(rows)
        self.assertAlmostEqual(summary['call_reduction_pct'], 1.0)
        self.assertAlmostEqual(summary['F']['calls_per_1000_output_tokens'], 100000 / 102)
        self.assertAlmostEqual(summary['O']['output_tokens_per_s'], 102 / 99)


class EventGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.receipt = {'training': {
            'event_id': 'event_002', 'optimizer_steps': 128,
            'response_presentations': 256, 'end_step': 256,
            'checkpoint_path': '/trained',
        }}
        self.trainer = Mock()
        self.trainer.status.return_value = {
            'global_step': 256, 'presentations': 512,
            'completed_events': ['event_001', 'event_002'],
            'accumulation_remainder': 0,
        }
        self.backend = Mock()
        self.backend.checkpoint_id = '/trained'
        self.backend.check_weights.return_value = {'all_equal': True}

    def check(self, receipt=None):
        R.check_event(receipt or self.receipt, self.trainer, self.backend, self.root, 2, 128)

    def test_valid_full_event_publishes_retirement_gate_without_efficacy_test(self):
        self.check()
        marker = json.loads((self.root / 'PUBLICATION_CHECK.json').read_text())
        self.assertEqual(marker['status'], 'passed')
        self.assertEqual(marker['event_id'], 'event_002')
        self.assertTrue(marker['all_equal'])

    def test_wrong_steps_presentations_or_cumulative_step_block_retirement(self):
        for key in ('optimizer_steps', 'response_presentations', 'end_step'):
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.receipt)
                receipt['training'][key] -= 1
                with self.assertRaises(AssertionError):
                    self.check(receipt)
                self.assertFalse((self.root / 'PUBLICATION_CHECK.json').exists())
        self.backend.check_weights.assert_not_called()

    def test_native_weight_mismatch_blocks_retirement(self):
        self.backend.check_weights.return_value = {'all_equal': False}
        with self.assertRaisesRegex(AssertionError, 'native draft'):
            self.check()
        self.assertFalse((self.root / 'PUBLICATION_CHECK.json').exists())


class FirstUpdatedRequestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {'prompt_ids': [1, 2], 'max_new_tokens': 8}

    def check(self, backend):
        backend.checkpoint_id = '/trained'
        return R.check_first_updated_request(
            backend, self.request, '/released', '/trained', self.root)

    def test_updated_frozen_reload_is_capture_disabled_and_restores_current_export(self):
        backend = EvaluationBackend()
        result = self.check(backend)
        self.assertEqual([call[0] for call in backend.calls], ['O', 'F', 'O'])
        self.assertEqual(backend.switches, ['/released', '/trained'])
        self.assertEqual(backend.checkpoint_id, '/trained')
        self.assertTrue(all(call[2]['purpose'] == 'evaluation' for call in backend.calls))
        self.assertTrue(result['same_input'])
        self.assertTrue(result['same_output'])
        self.assertTrue(result['reloaded_calls_equal'])
        self.assertFalse(result['feedback_collected'])
        self.assertFalse(result['performance_win_required'])

    def test_harmful_call_count_still_passes_correctness_gate(self):
        def mutate(record, index, method):
            if method == 'O':
                record['metadata']['verification_calls'] = 5
                record['metadata']['request_latency_s'] = 10.0

        result = self.check(EvaluationBackend(mutate))
        self.assertTrue(result['same_output'])
        self.assertTrue(result['reloaded_calls_equal'])

    def test_changed_output_stops_with_all_three_raw_records_retained(self):
        def mutate(record, index, method):
            if method == 'O':
                record['response_ids'] = [777, 99]

        with self.assertRaisesRegex(AssertionError, 'evidence preserved'):
            self.check(EvaluationBackend(mutate))
        result = json.loads((self.root / 'NATIVE_REPLAY_CHECK.json').read_text())
        self.assertFalse(result['same_output'])
        for label in ('updated', 'frozen', 'reloaded'):
            self.assertTrue((self.root / f'qualification_{label}.json').is_file())

    def test_reload_count_mismatch_stops_even_with_identical_outputs(self):
        def mutate(record, index, method):
            if index == 2:
                record['metadata']['verification_calls'] += 1

        with self.assertRaisesRegex(AssertionError, 'evidence preserved'):
            self.check(EvaluationBackend(mutate))
        result = json.loads((self.root / 'NATIVE_REPLAY_CHECK.json').read_text())
        self.assertTrue(result['same_output'])
        self.assertFalse(result['reloaded_calls_equal'])

    def test_input_mismatch_stops_even_with_identical_outputs(self):
        def mutate(record, index, method):
            if index == 2:
                record['prompt_ids'] = [123]

        with self.assertRaisesRegex(AssertionError, 'evidence preserved'):
            self.check(EvaluationBackend(mutate))
        result = json.loads((self.root / 'NATIVE_REPLAY_CHECK.json').read_text())
        self.assertFalse(result['same_input'])


if __name__ == '__main__':
    unittest.main()
