"""Boundary checks for one ordinary event and its capture-disabled replay."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.osd_one_event import run
from experiments.osd_one_event.protocol import (
    ARMS, TOTAL_GENERATIONS, method_order, record_errors, validate_manifest,
    validate_training_accounting,
)
from experiments.osd_one_event.report import summarize


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



def trainer_status():
    return {'global_step': 128, 'presentations': 256, 'completed_events': ['event_001'],
            'accumulation_remainder': 0, 'objective': 'osd', 'reduction': 'response_mean'}


def training_receipt():
    return {'training': {'start_step': 0, 'end_step': 128, 'optimizer_steps': 128,
                         'response_presentations': 256}}


class OneEventTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.requests = [request(index) for index in range(3)]
        self.checkpoints = {'F': '/F', 'O': '/O'}

    def evaluate(self, backend):
        return run.evaluate(backend, self.requests, self.checkpoints,
                            self.root / 'evaluation', qualification_requests=1)

    def test_budget_order_capture_and_inference_cost(self):
        backend = Backend(self.requests)
        status = self.evaluate(backend)
        self.assertEqual(TOTAL_GENERATIONS, 512)
        self.assertEqual(status['returned_generations'], 18)
        self.assertEqual(status['unattempted_generations'], 0)
        self.assertEqual(status['qualification'], 'passed')
        self.assertEqual(backend.weight_checks, ['/F', '/O'])
        for index in range(3):
            expected = [method for repeat in range(3) for method in method_order(index, repeat)]
            actual = [row[0][1:] for row in backend.executions[index * 6:(index + 1) * 6]]
            self.assertEqual(actual, expected)
        self.assertTrue(all(row[2]['purpose'] == 'evaluation' for row in backend.executions))
        self.assertTrue(all('response_ids' not in row[2] for row in backend.executions))
        summary = json.loads((self.root / 'evaluation/report/SUMMARY.json').read_text())
        self.assertEqual(summary['valid_requests'], 3)
        self.assertEqual(summary['F']['verification_calls'], 6)
        self.assertEqual(summary['F']['summed_mean_request_latency_s'], 3)
        self.assertEqual(summary['F']['all_repeats_request_latency_s'], 9)
        self.assertEqual(summary['call_request_directions']['tied'], 3)
        self.assertGreater(status['activation_s'], 10000)

    def test_position_exposure_is_balanced_over_the_full_panel(self):
        positions = {method: [0, 0] for method in ARMS}
        for ordinal in range(64):
            for repeat in range(3):
                for position, method in enumerate(method_order(ordinal, repeat)):
                    positions[method][position] += 1
        self.assertEqual(positions, {'F': [96, 96], 'O': [96, 96]})

    def test_historical_final_osd_count_is_not_a_new_osd_requirement(self):
        for row in self.requests:
            row['reference_calls']['O'] = 999
        result = self.evaluate(Backend(self.requests))
        self.assertEqual(result['qualification'], 'passed')

    def test_historical_frozen_count_remains_required(self):
        self.requests[0]['reference_calls']['F'] = 999
        with self.assertRaisesRegex(RuntimeError, 'historical_count_mismatch'):
            self.evaluate(Backend(self.requests))
        rows = (self.root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()
        self.assertFalse(json.loads(rows[0])['valid_measurement'])

    def test_initial_mismatch_stops_after_first_complete_group(self):
        def mutate(record, kwargs):
            if record['checkpoint_id'] == '/O':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'First-group'):
            self.evaluate(Backend(self.requests, mutate))
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['returned_generations'], 6)
        self.assertEqual(status['unattempted_generations'], 12)

    def test_later_mismatch_is_visible_and_excludes_whole_request(self):
        def mutate(record, kwargs):
            if kwargs['request_index'] == 1 and record['checkpoint_id'] == '/O':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        result = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(result['returned_generations'], 18)
        summary = json.loads((self.root / 'evaluation/report/SUMMARY.json').read_text())
        self.assertEqual(summary['valid_requests'], 2)
        self.assertEqual(summary['invalid_or_incomplete_requests'], 1)

    def test_measurement_bounds_stop_and_remain_invalid_in_report(self):
        def mutate(record, kwargs):
            record['metadata']['accepted_draft_tokens'] = 100
        with self.assertRaisesRegex(RuntimeError, 'accepted_token_bounds'):
            self.evaluate(Backend(self.requests, mutate))
        summary = json.loads((self.root / 'evaluation/report/SUMMARY.json').read_text())
        self.assertEqual(summary['valid_requests'], 0)
        row = json.loads((self.root / 'evaluation/MEASUREMENTS.jsonl').read_text())
        self.assertFalse(row['valid_measurement'])
        self.assertIn('accepted_token_bounds', row['errors'])

    def test_failed_generation_is_unknown_and_not_retried(self):
        backend = Backend(self.requests)
        with patch.object(backend, 'generate', side_effect=RuntimeError('native stopped')):
            with self.assertRaisesRegex(RuntimeError, 'native stopped'):
                self.evaluate(backend)
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['returned_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['unattempted_generations'], 17)

    def test_no_historical_osd_gate_does_not_disable_its_canonical_gate(self):
        backend = Backend(self.requests)
        backend.checkpoint_id = '/O'
        record = backend.generate([1, 2], request_index=0, purpose='evaluation', save_to=self.root / 'record.json')
        record['response_ids'][1] = 777
        record['metadata']['trace'][0]['committed_tokens'][0] = 777
        self.assertIn('reference_response_ids', record_errors(record, self.requests[0], 'O', '/O'))

    def test_harmful_updated_calls_continue_but_repeat_disagreement_stops(self):
        def make_harmful(record, kwargs):
            if record['checkpoint_id'] == '/O':
                record['metadata'].update(
                    verification_calls=3, accepted_draft_tokens=1,
                    trace=[dict(output_start=0, anchor_token=10, committed_progress=1,
                                committed_tokens=[11], accepted_draft_tokens=0),
                           dict(output_start=1, anchor_token=11, committed_progress=1,
                                committed_tokens=[12], accepted_draft_tokens=0),
                           dict(output_start=2, anchor_token=12, committed_progress=2,
                                committed_tokens=[13, 99], accepted_draft_tokens=1)])
        result = self.evaluate(Backend(self.requests, make_harmful))
        self.assertEqual(result['returned_generations'], 18)
        summary = json.loads((self.root / 'evaluation/report/SUMMARY.json').read_text())
        self.assertEqual(summary['call_reduction_pct'], -50)
        self.assertEqual(summary['call_request_directions']['regressed'], 3)

        def make_inconsistent(record, kwargs):
            if '_r1_' in kwargs['request_id']:
                make_harmful(record, kwargs)
        with self.assertRaisesRegex(RuntimeError, 'do not reproduce'):
            run.evaluate(Backend(self.requests, make_inconsistent), self.requests, self.checkpoints,
                         self.root / 'inconsistent', qualification_requests=1)
        summary = json.loads((self.root / 'inconsistent/report/SUMMARY.json').read_text())
        self.assertEqual(summary['valid_requests'], 0)
        # JSONL is durable before the end-of-request repeat check. Rebuilding a
        # report must rediscover this failure even if those earlier flags pass.
        rows = [json.loads(line) for line in
                (self.root / 'inconsistent/MEASUREMENTS.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['valid_measurement'] for row in rows))
        rebuilt, per_request = summarize(rows, self.requests)
        self.assertEqual(rebuilt['valid_requests'], 0)
        self.assertFalse(per_request[0]['valid'])
        self.assertIn('repeat_count_mismatch', per_request[0]['errors'])

    def test_training_accounting_rejects_extra_event_or_presentation(self):
        validate_training_accounting(training_receipt(), trainer_status())
        status = trainer_status()
        status['completed_events'].append('event_002')
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            validate_training_accounting(training_receipt(), status)
        receipt = training_receipt()
        receipt['training']['response_presentations'] = 258
        with self.assertRaisesRegex(RuntimeError, 'response_presentations'):
            validate_training_accounting(receipt, trainer_status())

    def test_only_one_update_follows_all_feedback(self):
        calls = []
        root = self.root
        class Trainer:
            def status(self):
                return trainer_status()
        class Controller:
            trainer = Trainer()
            def __init__(self):
                self.root = root / 'loop'
            def serve(self, prompt_ids, **kwargs):
                calls.append(('serve', kwargs['request_id']))
                write_json(self.root / 'serving' / (kwargs['request_id'] + '.costs.json'), {'serve_call_s': 1.1})
                return {'checkpoint_id': '/F', 'prompt_ids': prompt_ids, 'metadata': {
                    'request_latency_s': 1.0, 'capture': {'capture_host_s': .1,
                    'feature_persistence_s': .2, 'cache_maintenance_s': .01}}}
            def update(self, event_id):
                calls.append(('update', event_id))
                return training_receipt()
        requests = [{'request_id': f'q{i}', 'request_index': i, 'prompt_ids': [i], 'max_new_tokens': 2048}
                    for i in range(128)]
        progress = {'feedback_attempted': 0, 'feedback_returned': 0,
                    'feedback_controller_s': 0.0, 'feedback_request_latency_s': 0.0,
                    'feedback_nested_costs': {'capture_host_s': 0., 'feature_persistence_s': 0.,
                                              'cache_maintenance_s': 0.}}
        with patch.object(run, 'admit_update') as admission:
            run.collect_and_update(Controller(), requests, root, '/F', progress)
        self.assertEqual(calls[-1], ('update', 'event_001'))
        self.assertEqual(len(calls), 129)
        self.assertTrue(all(kind == 'serve' for kind, _ in calls[:-1]))
        self.assertEqual(progress['updates_completed'], 1)
        self.assertEqual(progress['optimizer_steps'], 128)
        admission.assert_called_once()
        self.assertEqual(len(admission.call_args.args[2]), 128)

    def test_manifest_preserves_budget_and_disjoint_panel(self):
        manifest = {
            'train_requests': [{'request_id': f'a{i}', 'prompt_ids': [i], 'max_new_tokens': 2048} for i in range(128)],
            'eval_requests': [{**request(i), 'prompt_ids': [1000 + i]} for i in range(64)],
            'policy': {'train_context_limit': 2272, 'serving_context_limit': 4096}}
        validate_manifest(manifest)
        changed = copy.deepcopy(manifest)
        changed['eval_requests'][0]['prompt_ids'] = [0]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed['train_requests'].pop()
        with self.assertRaisesRegex(ValueError, '128 feedback'):
            validate_manifest(changed)


if __name__ == '__main__':
    unittest.main()
