"""Scientific boundary tests for the fixed replay runner; no model execution."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.osd_components import run
from experiments.osd_components.protocol import ARMS, method_order, record_errors, weight_check_errors


def request(index):
    return {'request_id': f'q{index}', 'panel_ordinal': index, 'original_index': 7 + index * 13,
            'prompt_ids': [1, 2], 'response_ids': [10, 11, 12, 13, 99],
            'finish_reason': 'eos', 'stop_token_ids': [99], 'max_new_tokens': 2048,
            'reference_calls': {'F': 2, 'O': 2}}


def publication(checkpoint):
    helpers = {'helper_matches_current_layers': True,
               'helper_norm_matches_current_layers': True, 'fused_enabled': True}
    return {'checkpoint_id': checkpoint, 'activation_s': 10000.0,
            'checkpoint_structure_valid': True,
            'native_status': {'status': 'passed', 'poisoned': None, 'capture_pending': False,
                              'cache_tokens': 0, 'attention_backend': 'fa3', 'block_size': 16,
                              'publication': {'status': 'passed', 'checkpoint_path': checkpoint,
                                              'cache_after_tokens': 0, **helpers}}}


class Backend:
    def __init__(self, requests, mutation=None):
        self.requests = requests
        self.checkpoint_id = '/F'
        self.mutation = mutation
        self.executions, self.switches, self.weight_checks = [], [], []

    def publish(self, checkpoint):
        self.checkpoint_id = checkpoint
        self.switches.append(checkpoint)
        return publication(checkpoint)

    def check_weights(self, checkpoint):
        self.weight_checks.append(checkpoint)
        return {'status': 'passed', 'all_equal': True, 'tensors': 58,
                'checkpoint_path': checkpoint,
                'source_tensor_count': 58, 'native_tensor_count': 43,
                'helper_matches_current_layers': True,
                'helper_norm_matches_current_layers': True, 'fused_enabled': True}

    def generate(self, prompt_ids, **kwargs):
        self.executions.append((self.checkpoint_id, prompt_ids, kwargs))
        expected = self.requests[kwargs['request_index']]
        blocks = [dict(output_start=0, anchor_token=10, committed_progress=2,
                       committed_tokens=[11, 12], accepted_draft_tokens=1),
                  dict(output_start=2, anchor_token=12, committed_progress=2,
                       committed_tokens=[13, 99], accepted_draft_tokens=1)]
        record = {key: copy.deepcopy(expected[key]) for key in (
            'prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens')}
        record.update(checkpoint_id=self.checkpoint_id, purpose=kwargs['purpose'],
                      metadata={'verification_calls': 2, 'accepted_draft_tokens': 2,
                                'committed_progress': 4, 'request_latency_s': 1.0,
                                'native_generate_s': 0.9, 'trace': blocks,
                                'feedback_capture_in_timed_request': False,
                                'block_size': 16, 'context_limit': 4096,
                                'cache_policy': 'idle_full_flush_before_every_request_and_publication',
                                'runtime': 'pinned_SGLang_FA3',
                                'capture': {'collect_features': False, 'feature_bytes': 0,
                                            'native_capture_validated': True, 'cache_maintenance_s': 0.1}})
        if self.mutation:
            self.mutation(record, kwargs)
        write_json(kwargs['save_to'], record)
        return record


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'evaluation'
        self.requests = [request(i) for i in range(3)]
        self.arms = {method: '/' + method for method in ARMS}

    def evaluate(self, backend):
        return run.evaluate(backend, self.requests, self.arms, self.root,
                            qualification_requests=1)

    def test_exact_budget_order_capture_off_and_separate_activation(self):
        backend = Backend(self.requests)
        result = self.evaluate(backend)
        self.assertEqual(result['returned_generations'], 36)
        self.assertEqual(result['unattempted_generations'], 0)
        self.assertEqual(result['qualification'], 'passed')
        self.assertEqual(len(backend.weight_checks), 4)
        self.assertEqual(backend.switches[0], '/F')
        self.assertEqual(len(backend.switches), 36)
        for index, req in enumerate(self.requests):
            expected = [method for repeat in range(3) for method in method_order(index, repeat)]
            actual = [row[0][1:] for row in backend.executions[index * 12:(index + 1) * 12]]
            self.assertEqual(actual, expected)
        self.assertTrue(all(row[2]['purpose'] == 'evaluation' for row in backend.executions))
        self.assertTrue(all('response_ids' not in row[2] for row in backend.executions))
        summary = json.loads((self.root / 'report/SUMMARY.json').read_text())
        self.assertEqual(summary['cohort']['valid_requests'], 3)
        rows = [json.loads(line) for line in (self.root / 'MEASUREMENTS.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['request_latency_s'] == 1 for row in rows))

    def test_initial_output_mismatch_stops_before_remaining_requests(self):
        def mutate(record, kwargs):
            if record['checkpoint_id'] == '/C-only':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'First-group'):
            self.evaluate(Backend(self.requests, mutate))
        status = json.loads((self.root / 'STATUS.json').read_text())
        self.assertEqual(status['returned_generations'], 12)
        self.assertEqual(status['unattempted_generations'], 24)

    def test_later_output_mismatch_is_retained_and_excludes_whole_request(self):
        def mutate(record, kwargs):
            if kwargs['request_index'] == 1 and record['checkpoint_id'] == '/C-only':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        result = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(result['returned_generations'], 36)
        summary = json.loads((self.root / 'report/SUMMARY.json').read_text())
        self.assertEqual(summary['cohort']['valid_requests'], 2)
        self.assertEqual(summary['cohort']['status'], 'PARTIAL')

    def test_initial_f_publication_failure_prevents_any_generation(self):
        backend = Backend(self.requests)
        receipt = publication('/F')
        receipt['native_status']['publication']['helper_matches_current_layers'] = False
        with patch.object(backend, 'publish', return_value=receipt) as publish:
            with self.assertRaisesRegex(RuntimeError, 'Native publication failed'):
                self.evaluate(backend)
        publish.assert_called_once_with('/F')
        self.assertEqual(backend.executions, [])
        self.assertEqual(backend.weight_checks, [])
        saved = json.loads((self.root / 'ACTIVATIONS.jsonl').read_text())
        self.assertIn('helpers', saved['errors'])

    def test_weight_equality_requires_native_checkpoint_identity(self):
        backend = Backend(self.requests)
        receipt = backend.check_weights('/F')
        self.assertEqual(weight_check_errors(receipt, '/F'), [])
        for change in ('missing', 'different'):
            altered = copy.deepcopy(receipt)
            if change == 'missing':
                altered.pop('checkpoint_path')
            else:
                altered['checkpoint_path'] = '/another-checkpoint'
            self.assertIn('checkpoint_identity', weight_check_errors(altered, '/F'))

    def test_later_structural_errors_stop_after_persisting_invalid_measurement(self):
        changes = {
            'cache_policy': lambda m: m.update(cache_policy='warm'),
            'runtime_geometry': lambda m: m.update(context_limit=2048),
            'capture_contract': lambda m: m['capture'].update(collect_features=True),
            'feedback_capture_or_purpose': lambda m: m.update(feedback_capture_in_timed_request=True),
            'measurement_accounting': lambda m: m.update(committed_progress=3),
        }
        for expected_error, change in changes.items():
            with self.subTest(expected_error=expected_error):
                def mutate(record, kwargs):
                    if kwargs['request_index'] == 1:
                        change(record['metadata'])

                root = self.root.parent / expected_error
                backend = Backend(self.requests, mutate)
                with self.assertRaisesRegex(RuntimeError, 'Structural measurement validation failed'):
                    run.evaluate(backend, self.requests, self.arms, root,
                                 qualification_requests=1)
                status = json.loads((root / 'STATUS.json').read_text())
                self.assertEqual(status['qualification'], 'passed')
                self.assertEqual(status['returned_generations'], 13)
                self.assertEqual(status['unknown_generation_attempts'], 0)
                self.assertEqual(status['unattempted_generations'], 23)
                rows = [json.loads(line) for line in (root / 'MEASUREMENTS.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 13)
                self.assertFalse(rows[-1]['valid_measurement'])
                self.assertIn(expected_error, rows[-1]['errors'])
                self.assertTrue(Path(rows[-1]['record_path']).is_file())
                summary = json.loads((root / 'report/SUMMARY.json').read_text())
                self.assertEqual(summary['cohort']['valid_requests'], 1)

    def test_failed_native_attempt_is_unknown_not_retried(self):
        backend = Backend(self.requests)
        with patch.object(backend, 'generate', side_effect=RuntimeError('worker stopped')):
            with self.assertRaisesRegex(RuntimeError, 'worker stopped'):
                self.evaluate(backend)
        status = json.loads((self.root / 'STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['returned_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['unattempted_generations'], 35)

    def test_measurement_bounds_reject_accepted_tokens_larger_than_output(self):
        backend = Backend(self.requests)
        record = backend.generate([1, 2], request_index=0, purpose='evaluation',
                                  save_to=self.root.parent / 'one.json')
        record['metadata']['accepted_draft_tokens'] = 100
        self.assertIn('accepted_token_bounds', record_errors(record, self.requests[0], 'F', '/F'))

    def test_each_arm_has_equal_position_exposure(self):
        positions = {arm: [0] * 4 for arm in ARMS}
        for ordinal in range(64):
            for repeat in range(3):
                for position, arm in enumerate(method_order(ordinal, repeat)):
                    positions[arm][position] += 1
        self.assertTrue(all(counts == [48] * 4 for counts in positions.values()))


if __name__ == '__main__':
    unittest.main()
