"""CPU-only retained-F tests with the reviewed fake native backend."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_osd_one_event import request
from test_prefix_ranking_evaluation import harmful
from experiments.fresh_confirmation.test_evaluate import Backend
from .evaluate import evaluate
from .protocol import (
    ARMS, GENERATIONS, method_order, measurement_errors, prepare_requests, validate_inputs,
)
from .report import summarize


class ContinuationEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = [request(index) for index in range(3)]
        self.checkpoints = {'O': '/O', 'K': '/K'}
        self.counts = {row['request_id']: 2 for row in self.requests}

    def evaluate(self, backend, name='evaluation'):
        return evaluate(backend, self.requests, self.checkpoints, self.root / name,
                        frozen_counts=self.counts, qualification_requests=1)

    def read(self, name, directory='evaluation'):
        return json.loads((self.root / directory / name).read_text())

    def rows(self, name='MEASUREMENTS.jsonl'):
        return [json.loads(line) for line in (self.root / 'evaluation' / name).read_text().splitlines()]

    def test_exact_384_budget_rotation_193_publications_and_included48_gate(self):
        requests = [request(index) for index in range(64)]
        counts = {row['request_id']: 2 for row in requests}
        backend = Backend(requests)
        status = evaluate(backend, requests, self.checkpoints, self.root / 'evaluation', frozen_counts=counts)
        self.assertEqual(GENERATIONS, 384)
        for key in ('planned_generations', 'attempted_generations', 'returned_generations',
                    'completed_valid_generations'):
            self.assertEqual(status[key], 384)
        for key in ('invalid_generations', 'pending_canonical_generations',
                    'unknown_generation_attempts', 'unattempted_generations',
                    'optimizer_steps', 'feedback_collections'):
            self.assertEqual(status[key], 0)
        self.assertEqual(self.read('QUALIFICATION.json')['generations'], 48)
        self.assertEqual(backend.weight_checks, ['/O', '/K'])
        expected = [arm for ordinal in range(64) for repeat in range(3)
                    for arm in method_order(ordinal, repeat)]
        self.assertEqual([call[0][1:] for call in backend.executions], expected)
        self.assertEqual(len(backend.switches), 193)
        self.assertEqual(len(self.rows('ACTIVATIONS.jsonl')), 193)
        self.assertEqual(status['activation_s'], 193 * 10000.)
        self.assertEqual(set(row['method'] for row in self.rows()), set(ARMS))
        positions = {arm: [0, 0] for arm in ARMS}
        for ordinal in range(64):
            for repeat in range(3):
                for position, arm in enumerate(method_order(ordinal, repeat)):
                    positions[arm][position] += 1
        self.assertEqual(positions, {'O': [96, 96], 'K': [96, 96]})
        for _, prompt, options in backend.executions:
            self.assertEqual(options['purpose'], 'evaluation')
            self.assertEqual(options['max_new_tokens'], 2048)
            self.assertNotIn('response_ids', options)
            self.assertEqual(prompt, requests[options['request_index']]['prompt_ids'])
        self.assertEqual(self.read('report/SUMMARY.json')['historical_F']['verification_calls'], 128)
        self.assertTrue(all(row['source_method'] == 'F' for row in self.read('CANONICAL.json').values()))

    def test_evolving_O_counts_allowed_and_F_counts_are_only_historical_context(self):
        for row in self.requests:
            row['reference_calls']['O'] = 999
        self.counts = dict(q0=10, q1=20, q2=30)
        def mutate(record, options):
            if record['checkpoint_id'] == '/O':
                harmful(record)
        status = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(status['status'], 'completed')
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(summary['O']['verification_calls'], 9)
        self.assertEqual(summary['K']['verification_calls'], 6)
        self.assertEqual(summary['historical_F']['verification_calls'], 60)
        self.assertEqual(summary['historical_F_comparisons']['O_vs_F']['call_reduction_pct'], 85)
        self.assertEqual(summary['historical_F_comparisons']['K_vs_F']['call_reduction_pct'], 90)
        self.assertNotIn('F', summary)
        self.assertFalse(any('latency' in key or 'time' in key for key in summary['historical_F']))
        self.assertEqual(set(summary['comparisons']), {'K_vs_O'})
        self.assertEqual(set(summary['historical_F_comparisons']), {'O_vs_F', 'K_vs_F'})

    def test_first_O_must_match_retained_F_and_cannot_replace_canonical(self):
        def mutate(record, options):
            record['response_ids'][1] = 777
            record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read('STATUS.json')
        self.assertEqual(status['returned_generations'], 1)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['unattempted_generations'], 17)
        self.assertEqual(self.read('0000_r0_O.json')['response_ids'][1], 777)
        self.assertEqual(self.read('CANONICAL.json')['q0']['response_ids'][1], 11)
        self.assertEqual(self.read('report/SUMMARY.json')['valid_requests'], 0)

    def test_K_first_F_mismatch_stops_immediately_with_honest_common_cohort(self):
        self.counts = dict(q0=2, q1=200, q2=300)
        def mutate(record, options):
            if options['request_id'] == '0001_r0_K':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read('STATUS.json')
        self.assertEqual(status['qualification'], 'passed')
        self.assertEqual(status['returned_generations'], 7)
        self.assertEqual(status['completed_valid_generations'], 6)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['pending_canonical_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 11)
        self.assertFalse((self.root / 'evaluation/0001_r0_O.json').exists())
        events = [row for row in self.rows('VALIDATION_EVENTS.jsonl') if row['label'] == '0001_r0_K']
        self.assertEqual([row['status'] for row in events], ['invalid'])
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(summary['valid_requests'], 1)
        self.assertEqual(summary['historical_F']['verification_calls'], 2)

    def test_exact_prompt_termination_identity_and_capture_are_gates(self):
        for field in ('prompt', 'finish', 'identity', 'capture'):
            with self.subTest(field=field):
                def mutate(record, options):
                    if field == 'prompt':
                        record['prompt_ids'][0] = 777
                    elif field == 'finish':
                        record['finish_reason'] = 'length'
                    elif field == 'identity':
                        record['session_id'] = 'incorrect'
                    else:
                        record['metadata']['capture']['collect_features'] = True
                with self.assertRaises(RuntimeError):
                    self.evaluate(Backend(self.requests, mutate), field)
                status = self.read('STATUS.json', field)
                self.assertEqual(status['returned_generations'], 1)
                self.assertEqual(status['invalid_generations'], 1)
                self.assertEqual(status['unknown_generation_attempts'], 0)

    def test_repeat_count_disagreement_invalidates_all_affected_arm_rows(self):
        def mutate(record, options):
            if options['request_id'] == '0000_r1_K':
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read('STATUS.json')
        self.assertEqual(status['returned_generations'], 3)
        self.assertEqual(status['invalid_generations'], 2)
        self.assertEqual(status['completed_valid_generations'], 1)
        self.assertTrue(all('repeat_count_mismatch' in row['errors']
                            for row in self.rows() if row['method'] == 'K'))

    def test_interrupted_O_keeps_preceding_K_valid_and_counts_unknown_attempt(self):
        def interrupt(record, options):
            if options['request_id'] == '0001_r0_O':
                raise RuntimeError('worker reply lost')
        with self.assertRaisesRegex(RuntimeError, 'worker reply lost'):
            self.evaluate(Backend(self.requests, interrupt))
        status = self.read('STATUS.json')
        self.assertEqual(status['attempted_generations'], 8)
        self.assertEqual(status['returned_generations'], 7)
        self.assertEqual(status['completed_valid_generations'], 7)
        self.assertEqual(status['pending_canonical_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['invalid_generations'], 0)
        self.assertEqual(status['unattempted_generations'], 10)
        self.assertEqual(self.rows()[-1]['validation_status'], 'valid')

    def test_nonfinite_and_nonscalar_returned_records_are_durable_invalid(self):
        for index, value in enumerate((float('nan'), {'bad': 'count'}, [1, 2])):
            with self.subTest(value=value):
                def mutate(record, options):
                    field = 'request_latency_s' if isinstance(value, float) else 'verification_calls'
                    record['metadata'][field] = value
                directory = 'bad_' + str(index)
                with self.assertRaises(RuntimeError):
                    self.evaluate(Backend(self.requests, mutate), directory)
                status = self.read('STATUS.json', directory)
                self.assertEqual(status['returned_generations'], 1)
                self.assertEqual(status['invalid_generations'], 1)
                self.assertEqual(status['unknown_generation_attempts'], 0)
                self.assertEqual(status['unattempted_generations'], 17)
                self.assertTrue((self.root / directory / '0000_r0_O.json').is_file())
                self.assertTrue((self.root / directory / 'report/SUMMARY.json').is_file())
                self.assertFalse((self.root / directory / 'REPORT_FAILURE.json').exists())

    def test_failed_publication_and_failed_weights_have_no_generation_or_overwrite(self):
        for field in ('publication', 'weights'):
            backend = Backend(self.requests)
            if field == 'publication':
                publish = backend.publish
                def failed(checkpoint):
                    result = publish(checkpoint)
                    result['native_status']['cache_tokens'] = 1
                    return result
                backend.publish = failed
            else:
                check = backend.check_weights
                def failed(checkpoint):
                    result = check(checkpoint)
                    result['all_equal'] = False
                    return result
                backend.check_weights = failed
            with self.assertRaises(RuntimeError):
                self.evaluate(backend, field)
            self.assertEqual(self.read('STATUS.json', field)['attempted_generations'], 0)
            directory = self.root / field
            before = {str(path): path.read_bytes() for path in directory.rglob('*') if path.is_file()}
            with self.assertRaises(FileExistsError):
                self.evaluate(Backend(self.requests), field)
            self.assertEqual(before, {str(path): path.read_bytes() for path in directory.rglob('*') if path.is_file()})

    def test_report_revalidates_scalars_and_raw_outputs_against_retained_F(self):
        self.evaluate(Backend(self.requests))
        rows = self.rows()
        for field, value in (('request_latency_s', True), ('native_generate_s', float('nan')),
                             ('verification_calls', []), ('cache_maintenance_s', -1)):
            self.assertTrue(measurement_errors({**rows[0], field: value}, self.requests[0]))
        altered = copy.deepcopy(rows)
        altered[0]['accepted_draft_tokens'] = 900
        summary, _, _, _ = summarize(altered, self.requests, checkpoints=self.checkpoints, frozen_counts=self.counts)
        self.assertEqual(summary['valid_requests'], 2)
        path = Path(rows[0]['record_path'])
        record = json.loads(path.read_text())
        record['response_ids'][1] = 777
        record['metadata']['trace'][0]['committed_tokens'][0] = 777
        path.write_text(json.dumps(record))
        summary, per_request, _, _ = summarize(rows, self.requests, checkpoints=self.checkpoints, frozen_counts=self.counts)
        self.assertEqual(summary['valid_requests'], 2)
        self.assertIn('reference_response_ids', per_request[0]['errors'])
        self.assertEqual(summary['historical_F']['verification_calls'], 4)

    def test_harmful_results_complete_and_counts_costs_progress_lengths_are_separate(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/K':
                harmful(record)
                record['metadata']['request_latency_s'] = 2.
        status = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(status['status'], 'completed')
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(summary['O']['verification_calls'], 6)
        self.assertEqual(summary['K']['verification_calls'], 9)
        self.assertEqual(summary['K']['summed_mean_request_latency_s'], 6)
        self.assertEqual(summary['K']['all_repeats_request_latency_s'], 18)
        self.assertEqual(summary['K']['accepted_draft_tokens'], 3)
        self.assertEqual(summary['K']['committed_progress'], 12)
        self.assertEqual(summary['K']['output_tokens'], 15)
        self.assertEqual(summary['K']['committed_progress_per_call'], 12 / 9)
        self.assertEqual(summary['comparisons']['K_vs_O']['call_reduction_pct'], -50)
        self.assertEqual(summary['comparisons']['K_vs_O']['request_time_reduction_pct'], -100)
        self.assertEqual(summary['comparisons']['K_vs_O']['call_request_directions'],
                         dict(improved=0, regressed=3, tied=0))
        self.assertEqual(summary['canonical_outputs']['minimum_length'], 5)
        self.assertEqual(summary['canonical_outputs']['maximum_length'], 5)
        self.assertEqual(summary['canonical_outputs']['eos_requests'], 3)
        self.assertEqual(summary['canonical_outputs']['output_cap_hits'], 0)
        self.assertGreater(status['activation_s'], 10000)

    def test_heterogeneous_reductions_are_pooled_not_averaged(self):
        self.counts = dict(q0=2, q1=20, q2=30)
        def mutate(record, options):
            ordinal = options['request_index']
            repeat = int(options['request_id'].split('_')[1][1:])
            method = record['checkpoint_id'][1:]
            means = {'O': (1., 9., 10.), 'K': (1., 3., 10.)}
            record['metadata']['request_latency_s'] = means[method][ordinal] * (.5, 1., 1.5)[repeat]
            record['metadata']['native_generate_s'] = record['metadata']['request_latency_s'] * .9
            if method == 'O' and ordinal == 1:
                harmful(record)
        self.evaluate(Backend(self.requests, mutate))
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(summary['O']['verification_calls'], 7)
        self.assertEqual(summary['K']['verification_calls'], 6)
        self.assertEqual(summary['O']['summed_mean_request_latency_s'], 20)
        self.assertEqual(summary['K']['summed_mean_request_latency_s'], 14)
        self.assertAlmostEqual(summary['comparisons']['K_vs_O']['call_reduction_pct'], 100 / 7)
        self.assertAlmostEqual(summary['comparisons']['K_vs_O']['request_time_reduction_pct'], 30)
        self.assertAlmostEqual(summary['historical_F_comparisons']['O_vs_F']['call_reduction_pct'], 100 * 45 / 52)
        self.assertAlmostEqual(summary['historical_F_comparisons']['K_vs_F']['call_reduction_pct'], 100 * 46 / 52)

    def test_inputs_require_retained_canonical_exact_count_keys_and_only_O_K(self):
        for field in ('response_ids', 'finish_reason'):
            malformed = copy.deepcopy(self.requests)
            malformed[0].pop(field)
            with self.assertRaises(ValueError):
                validate_inputs(malformed, self.checkpoints, self.counts, 1)
        for counts in ({'q0': 2}, {**self.counts, 'extra': 2}, {**self.counts, 'q0': True}):
            with self.assertRaises(ValueError):
                validate_inputs(self.requests, self.checkpoints, counts, 1)
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            validate_inputs(self.requests, {**self.checkpoints, 'F': '/F'}, self.counts, 1)
        without_ordinals = [{key: value for key, value in row.items() if key != 'panel_ordinal'}
                            for row in self.requests]
        self.assertEqual([row['panel_ordinal'] for row in prepare_requests(without_ordinals)], [0, 1, 2])

    def test_import_has_no_model_training_optimizer_or_CUDA_side_effect(self):
        script = "import sys; import experiments.ok_continuation.evaluate; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert 'online_feedback.training' not in sys.modules"
        process = subprocess.run([sys.executable, '-c', script],
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
