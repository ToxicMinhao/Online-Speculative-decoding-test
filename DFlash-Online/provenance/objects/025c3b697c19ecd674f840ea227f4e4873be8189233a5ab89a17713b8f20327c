"""CPU checks for fresh four-arm replays, strict validity and complete accounting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.prefix_ranking.evaluate import evaluate
from experiments.prefix_ranking.protocol import (
    ARMS, GENERATIONS, method_order, load_ordinary_counts, record_errors,
)
from experiments.prefix_ranking.report import summarize
from test_osd_one_event import Backend, request


def harmful(record):
    record['metadata'].update(
        verification_calls=3, accepted_draft_tokens=1,
        trace=[dict(output_start=0, anchor_token=10, committed_progress=1,
                    committed_tokens=[11], accepted_draft_tokens=0),
               dict(output_start=1, anchor_token=11, committed_progress=1,
                    committed_tokens=[12], accepted_draft_tokens=0),
               dict(output_start=2, anchor_token=12, committed_progress=2,
                    committed_tokens=[13, 99], accepted_draft_tokens=1)])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.requests = [request(index) for index in range(3)]
        self.checkpoints = {method: '/' + method for method in ARMS}
        self.counts = {row['request_id']: {'F': 2, 'O': 2} for row in self.requests}

    def evaluate(self, backend, name='evaluation'):
        return evaluate(backend, self.requests, self.checkpoints, self.root / name,
                        ordinary_counts=self.counts, qualification_requests=1)

    def rows(self, name='evaluation'):
        path = self.root / name / 'MEASUREMENTS.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]

    def summary(self, name='evaluation'):
        return json.loads((self.root / name / 'report/SUMMARY.json').read_text())

    def test_full_budget_rotation_capture_and_request_time(self):
        backend = Backend(self.requests)
        status = self.evaluate(backend)
        self.assertEqual(GENERATIONS, 768)
        self.assertEqual(status['returned_generations'], 36)
        self.assertEqual(status['unattempted_generations'], 0)
        self.assertEqual(status['qualification'], 'passed')
        self.assertEqual(status['optimizer_steps'], 0)
        self.assertEqual(status['feedback_collections'], 0)
        self.assertEqual(backend.weight_checks, ['/F', '/O', '/U', '/P'])
        for ordinal in range(3):
            actual = [call[0][1:] for call in backend.executions[ordinal * 12:(ordinal + 1) * 12]]
            expected = [method for repeat in range(3) for method in method_order(ordinal, repeat)]
            self.assertEqual(actual, expected)
        for checkpoint, prompt, options in backend.executions:
            self.assertEqual(options['purpose'], 'evaluation')
            self.assertEqual(options['max_new_tokens'], 2048)
            self.assertNotIn('response_ids', options)
            self.assertEqual(prompt, self.requests[options['request_index']]['prompt_ids'])
        summary = self.summary()
        self.assertEqual(summary['valid_requests'], 3)
        self.assertEqual(summary['F']['verification_calls'], 6)
        self.assertEqual(summary['F']['summed_mean_request_latency_s'], 3)
        self.assertEqual(summary['F']['all_repeats_request_latency_s'], 9)
        self.assertEqual(summary['comparisons']['P_vs_O']['call_request_directions']['tied'], 3)
        self.assertGreater(status['activation_s'], 10000)
        self.assertTrue((self.root / 'evaluation/report/ALL_REPEATS.csv').is_file())
        self.assertTrue((self.root / 'evaluation/report/REPEAT_TOTALS.csv').is_file())

    def test_order_balanced_in_each_position_over_64_requests(self):
        positions = {method: [0] * 4 for method in ARMS}
        for ordinal in range(64):
            for repeat in range(3):
                for position, method in enumerate(method_order(ordinal, repeat)):
                    positions[method][position] += 1
        self.assertEqual(positions, {method: [48] * 4 for method in ARMS})

    def test_qualification_96_generations_is_inside_768(self):
        requests = [request(index) for index in range(64)]
        counts = {row['request_id']: {'F': 2, 'O': 2} for row in requests}
        result = evaluate(Backend(requests), requests, self.checkpoints, self.root / 'full', ordinary_counts=counts)
        receipt = json.loads((self.root / 'full/QUALIFICATION.json').read_text())
        self.assertEqual(receipt['generations'], 96)
        self.assertEqual(result['returned_generations'], 768)

    def test_old_final_osd_count_ignored_but_retained_one_event_required(self):
        for row in self.requests:
            row['reference_calls']['O'] = 999
        self.assertEqual(self.evaluate(Backend(self.requests))['status'], 'completed')
        self.counts['q0']['O'] = 999
        with self.assertRaisesRegex(RuntimeError, 'retained_one_event_count_mismatch'):
            self.evaluate(Backend(self.requests), 'wrong')
        self.assertEqual(self.summary('wrong')['valid_requests'], 0)
        self.assertEqual(self.rows('wrong')[-1]['errors'], ['retained_one_event_count_mismatch'])

    def test_fresh_frozen_count_must_reproduce_retained_baseline(self):
        self.counts['q0']['F'] = 999
        with self.assertRaisesRegex(RuntimeError, 'retained_one_event_count_mismatch'):
            self.evaluate(Backend(self.requests))
        self.assertEqual(len(self.rows()), 1)

    def test_initial_output_mismatch_stops_immediately_and_retains_raw(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/P':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['returned_generations'], 4)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 32)
        row = self.rows()[-1]
        self.assertFalse(row['valid_measurement'])
        self.assertEqual(json.loads(Path(row['record_path']).read_text())['response_ids'][1], 777)

    def test_late_output_mismatch_stops_after_passed_initial_gate(self):
        def mutate(record, options):
            if options['request_index'] == 1 and record['checkpoint_id'] == '/O':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['qualification'], 'passed')
        self.assertEqual(status['returned_generations'], 13)
        self.assertEqual(status['unattempted_generations'], 23)
        self.assertEqual(self.summary()['valid_requests'], 1)
        self.assertEqual(self.summary()['invalid_or_incomplete_requests'], 2)

    def test_capture_enabled_is_a_correctness_failure(self):
        def mutate(record, options):
            record['metadata']['feedback_capture_in_timed_request'] = True
            record['metadata']['capture']['collect_features'] = True
        with self.assertRaisesRegex(RuntimeError, 'capture'):
            self.evaluate(Backend(self.requests, mutate))
        self.assertFalse(self.rows()[0]['valid_measurement'])

    def test_repeat_disagreement_stops_and_report_does_not_trust_valid_flag(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/P' and '_r1_' in options['request_id']:
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.evaluate(Backend(self.requests, mutate))
        rows = self.rows()
        for row in rows:
            row['valid_measurement'], row['errors'] = True, []
        summary, per_request, _, _ = summarize(rows, self.requests, checkpoints=self.checkpoints, ordinary_counts=self.counts)
        self.assertEqual(summary['valid_requests'], 0)
        self.assertIn('repeat_count_mismatch', per_request[0]['errors'])

    def test_harmful_and_null_results_continue_unchanged(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/P':
                harmful(record)
        result = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(result['returned_generations'], 36)
        comparison = self.summary()['comparisons']['P_vs_O']
        self.assertEqual(comparison['call_reduction_pct'], -50)
        self.assertEqual(comparison['call_request_directions']['regressed'], 3)
        self.assertEqual(comparison['call_gross_extra'], 3)

    def test_report_revalidates_scalar_bounds_and_native_record(self):
        self.evaluate(Backend(self.requests))
        rows = self.rows()
        rows[0]['accepted_draft_tokens'] = 100
        self.assertTrue(rows[0]['valid_measurement'])
        summary, per_request, _, _ = summarize(rows, self.requests, checkpoints=self.checkpoints, ordinary_counts=self.counts)
        self.assertEqual(summary['valid_requests'], 2)
        self.assertIn('accepted_token_bounds', per_request[0]['errors'])
        self.assertIn('row_native_disagreement_accepted_draft_tokens', per_request[0]['errors'])
        rows = self.rows()
        native_path = Path(rows[0]['record_path'])
        record = json.loads(native_path.read_text())
        record['metadata']['capture']['feature_bytes'] = 8
        write_json(native_path, record)
        summary, per_request, _, _ = summarize(rows, self.requests, checkpoints=self.checkpoints, ordinary_counts=self.counts)
        self.assertEqual(summary['valid_requests'], 2)
        self.assertIn('capture_contract', per_request[0]['errors'])

    def test_unknown_generation_and_unattempted_work_remain_distinct(self):
        backend = Backend(self.requests)
        with patch.object(backend, 'generate', side_effect=RuntimeError('native interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'native interrupted'):
                self.evaluate(backend)
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 1)
        self.assertEqual(status['returned_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['unattempted_generations'], 35)
        self.assertEqual(self.summary()['valid_requests'], 0)

    def test_malformed_returned_record_is_preserved_and_counted_as_returned(self):
        def mutate(record, options):
            del record['metadata']['capture']
        with self.assertRaisesRegex(RuntimeError, 'malformed_native_record'):
            self.evaluate(Backend(self.requests, mutate))
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['returned_generations'], 1)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 35)
        self.assertFalse(self.rows()[0]['valid_measurement'])
        self.assertEqual(self.summary()['valid_requests'], 0)

    def test_failed_publication_does_not_count_a_generation(self):
        backend = Backend(self.requests)
        original = backend.publish
        def failed(checkpoint):
            result = original(checkpoint)
            result['native_status']['publication']['helper_matches_current_layers'] = False
            return result
        with patch.object(backend, 'publish', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'publication'):
                self.evaluate(backend)
        status = json.loads((self.root / 'evaluation/STATUS.json').read_text())
        self.assertEqual(status['attempted_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 36)

    def test_retained_count_loader_uses_complete_valid_one_event_repeats(self):
        self.evaluate(Backend(self.requests))
        rows = [row for row in self.rows() if row['method'] in ('F', 'O')]
        path = self.root / 'ordinary.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertEqual(load_ordinary_counts(path, self.requests), self.counts)
        rows[0]['accepted_draft_tokens'] = 100
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'invalid evidence'):
            load_ordinary_counts(path, self.requests)

    def test_pooled_time_ratio_is_not_mean_request_percentage(self):
        def mutate(record, options):
            times = [1.0, 4.0, 10.0] if record['checkpoint_id'] != '/P' else [0.5, 3.0, 12.0]
            record['metadata']['request_latency_s'] = times[options['request_index']]
            record['metadata']['native_generate_s'] = times[options['request_index']] * 0.9
        self.evaluate(Backend(self.requests, mutate))
        summary = self.summary()
        comparison = summary['comparisons']['P_vs_O']
        self.assertAlmostEqual(comparison['request_time_reduction_pct'], 100 * (1 - 15.5 / 15))
        self.assertEqual(comparison['latency_request_directions'], {'improved': 2, 'regressed': 1, 'tied': 0})
        self.assertEqual(comparison['latency_gross_saved'], 1.5)
        self.assertEqual(comparison['latency_gross_extra'], 2.0)


if __name__ == '__main__':
    unittest.main()
