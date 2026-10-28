"""CPU fake-backend tests; no native worker, optimizer or CUDA construction."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_osd_one_event import Backend as ExistingBackend, request as existing_request
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate
from .protocol import ARMS, GENERATIONS, method_order, measurement_errors, validate_inputs
from .report import summarize


def request(index):
    return {key: value for key, value in existing_request(index).items()
            if key not in ('response_ids', 'finish_reason')}


class Backend(ExistingBackend):
    def __init__(self, requests, mutation=None):
        expected = [existing_request(row['panel_ordinal']) for row in requests]
        super().__init__(expected)
        self.after_return = mutation

    def generate(self, prompt_ids, **kwargs):
        record = super().generate(prompt_ids, **kwargs)
        record.update(record_id=kwargs['request_id'], session_id=kwargs['session_id'],
                      request_index=kwargs['request_index'])
        if self.after_return is not None:
            self.after_return(record, kwargs)
        return record


class KRCConfirmationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = [request(index) for index in range(3)]
        self.checkpoints = {'K1': '/K1', 'KRC': '/KRC'}

    def evaluate(self, backend, name='evaluation'):
        return evaluate(backend, self.requests, self.checkpoints, self.root / name, qualification_requests=1)

    def read(self, name):
        return json.loads((self.root / 'evaluation' / name).read_text())

    def rows(self, name='MEASUREMENTS.jsonl'):
        return [json.loads(line) for line in (self.root / 'evaluation' / name).read_text().splitlines()]

    def test_exact_two_arm_384_budget_rotation_and_included48_gate(self):
        requests = [request(index) for index in range(64)]
        backend = Backend(requests)
        status = evaluate(backend, requests, self.checkpoints, self.root / 'evaluation')
        self.assertEqual(GENERATIONS, 384)
        self.assertEqual(status['planned_generations'], 384)
        self.assertEqual(status['returned_generations'], 384)
        self.assertEqual(status['completed_valid_generations'], 384)
        self.assertEqual(status['invalid_generations'], 0)
        self.assertEqual(status['pending_canonical_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 0)
        self.assertEqual(self.read('QUALIFICATION.json')['generations'], 48)
        self.assertEqual(backend.weight_checks, ['/K1', '/KRC', '/KRC', '/K1'])
        self.assertEqual(status['weight_checks'], {
            'before': {'K1': True, 'KRC': True},
            'after': {'KRC': True, 'K1': True},
        })
        self.assertEqual(status['activation_count'], 193)
        expected = [arm for ordinal in range(64) for repeat in range(3) for arm in method_order(ordinal, repeat)]
        self.assertEqual([call[0][1:] for call in backend.executions], expected)
        for _, _, options in backend.executions:
            self.assertEqual(options['purpose'], 'evaluation')
            self.assertNotIn('response_ids', options)
        positions = {arm: [0, 0] for arm in ARMS}
        for ordinal in range(64):
            for repeat in range(3):
                for position, arm in enumerate(method_order(ordinal, repeat)):
                    positions[arm][position] += 1
        self.assertEqual(positions, {'K1': [96, 96], 'KRC': [96, 96]})
        for repeat in range(3):
            first = [method_order(ordinal, repeat)[0] for ordinal in range(64)]
            self.assertEqual(first.count('K1'), 32)
            self.assertEqual(first.count('KRC'), 32)
        self.assertEqual(len(self.read('CANONICAL.json')), 64)
        self.assertFalse(any('F' == row['method'] for row in self.rows()))
        summary = self.read('report/SUMMARY.json')
        self.assertTrue(summary['all_planned_requests_valid'])
        self.assertEqual(summary['leave_one_out']['omissions'], 64)
        for arm in ARMS:
            self.assertEqual(summary[arm]['terminations'], {'eos': 64, 'length': 0})
            self.assertEqual(summary[arm]['output_length_mean'], 5)

    def test_first_K1_is_canonical_and_historical_count_is_not_a_gate(self):
        for row in self.requests:
            row['reference_calls']['K1'] = 999
        status = self.evaluate(Backend(self.requests))
        self.assertEqual(status['status'], 'completed')
        events = self.rows('VALIDATION_EVENTS.jsonl')
        delayed = [row for row in events if row['label'] == '0001_r0_KRC']
        self.assertEqual([row['status'] for row in delayed], ['canonical_pending', 'valid'])
        canonical = self.read('CANONICAL.json')['q1']
        self.assertEqual(canonical['source_method'], 'K1')
        self.assertEqual(canonical['source_repeat'], 0)
        self.assertTrue(canonical['source_record_path'].endswith('0001_r0_K1.json'))
        self.assertEqual(len(self.rows()), 18)
        self.assertEqual(self.read('report/SUMMARY.json')['valid_requests'], 3)

    def test_KRC_first_mismatch_is_delayed_then_stops_with_both_records(self):
        def mutate(record, options):
            if options['request_id'] == '0001_r0_KRC':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read('STATUS.json')
        self.assertEqual(status['qualification'], 'passed')
        self.assertEqual(status['returned_generations'], 8)
        self.assertEqual(status['completed_valid_generations'], 7)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertEqual(status['pending_canonical_generations'], 0)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['unattempted_generations'], 10)
        self.assertEqual([row['method'] for row in self.rows()[-2:]], ['KRC', 'K1'])
        self.assertEqual(self.read('0001_r0_KRC.json')['response_ids'][1], 777)
        self.assertEqual(self.read('report/SUMMARY.json')['valid_requests'], 1)

    def test_later_K1_output_mismatch_retains_first_canonical(self):
        def mutate(record, options):
            if options['request_id'] == '0000_r1_K1':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(self.read('STATUS.json')['returned_generations'], 4)
        self.assertEqual(self.read('CANONICAL.json')['q0']['response_ids'][1], 11)

    def test_repeat_count_disagreement_invalidates_affected_arm_rows(self):
        def mutate(record, options):
            if options['request_id'] == '0000_r1_KRC':
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read('STATUS.json')
        self.assertEqual(status['returned_generations'], 3)
        self.assertEqual(status['invalid_generations'], 2)
        self.assertEqual(status['completed_valid_generations'], 1)
        self.assertTrue(all('repeat_count_mismatch' in row['errors'] for row in self.rows() if row['method'] == 'KRC'))

    def test_pending_KRC_and_interrupted_K1_are_not_completed_or_unattempted(self):
        def interrupt(record, options):
            if options['request_id'] == '0001_r0_K1':
                raise RuntimeError('worker reply lost')
        with self.assertRaisesRegex(RuntimeError, 'worker reply lost'):
            self.evaluate(Backend(self.requests, interrupt))
        status = self.read('STATUS.json')
        self.assertEqual(status['attempted_generations'], 8)
        self.assertEqual(status['returned_generations'], 7)
        self.assertEqual(status['completed_valid_generations'], 6)
        self.assertEqual(status['pending_canonical_generations'], 1)
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['invalid_generations'], 0)
        self.assertEqual(status['unattempted_generations'], 10)
        self.assertEqual(self.rows()[-1]['validation_status'], 'canonical_pending')

    def test_nonfinite_and_nonscalar_returned_records_keep_durable_failure_accounting(self):
        for value in (float('nan'), {'bad': 'count'}, [1, 2]):
            with self.subTest(value=value):
                def mutate(record, options):
                    if isinstance(value, float):
                        record['metadata']['request_latency_s'] = value
                    else:
                        record['metadata']['verification_calls'] = value
                destination = 'bad_' + str(len(list(self.root.iterdir())))
                with self.assertRaises(RuntimeError):
                    self.evaluate(Backend(self.requests, mutate), destination)
                directory = self.root / destination
                status = json.loads((directory / 'STATUS.json').read_text())
                self.assertEqual(status['returned_generations'], 1)
                self.assertEqual(status['invalid_generations'], 1)
                self.assertEqual(status['unknown_generation_attempts'], 0)
                self.assertEqual(status['unattempted_generations'], 17)
                self.assertTrue((directory / '0000_r0_K1.json').is_file())
                self.assertTrue((directory / 'report/SUMMARY.json').is_file())
                self.assertFalse((directory / 'REPORT_FAILURE.json').exists())

    def test_failed_publication_counts_no_generation_and_old_outputs_not_overwritten(self):
        backend = Backend(self.requests)
        publish = backend.publish
        def failed(checkpoint):
            result = publish(checkpoint)
            result['native_status']['cache_tokens'] = 1
            return result
        backend.publish = failed
        with self.assertRaisesRegex(RuntimeError, 'publication'):
            self.evaluate(backend)
        self.assertEqual(self.read('STATUS.json')['attempted_generations'], 0)
        before = {str(path): path.read_bytes() for path in (self.root / 'evaluation').rglob('*') if path.is_file()}
        with self.assertRaises(FileExistsError):
            self.evaluate(Backend(self.requests))
        self.assertEqual(before, {str(path): path.read_bytes() for path in (self.root / 'evaluation').rglob('*') if path.is_file()})

    def test_strict_scalar_validator_and_report_revalidate_raw_canonical(self):
        self.evaluate(Backend(self.requests))
        rows, canonicals = self.rows(), self.read('CANONICAL.json')
        for field, value in (('request_latency_s', True), ('native_generate_s', float('nan')),
                             ('verification_calls', []), ('cache_maintenance_s', -1)):
            row = {**rows[0], field: value}
            self.assertTrue(measurement_errors(row, self.requests[0], canonicals['q0']))
        altered = copy.deepcopy(rows)
        altered[0]['accepted_draft_tokens'] = 900
        summary, _, _, _ = summarize(altered, self.requests, checkpoints=self.checkpoints, canonicals=canonicals)
        self.assertEqual(summary['valid_requests'], 2)
        canonical = copy.deepcopy(canonicals)
        canonical['q0']['response_ids'][1] = 777
        summary, per_request, _, _ = summarize(rows, self.requests, checkpoints=self.checkpoints, canonicals=canonical)
        self.assertEqual(summary['valid_requests'], 2)
        self.assertIn('canonical_source_output', per_request[0]['errors'])

    def test_metrics_pool_repeat_zero_calls_and_three_mean_times_harmful_continues(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/KRC':
                harmful(record)
                record['metadata']['request_latency_s'] = 2.
        status = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(status['status'], 'completed')
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(set(summary['comparisons']), {'KRC_vs_K1'})
        self.assertEqual(summary['K1']['verification_calls'], 6)
        self.assertEqual(summary['KRC']['verification_calls'], 9)
        self.assertEqual(summary['KRC']['summed_mean_request_latency_s'], 6)
        self.assertEqual(summary['KRC']['all_repeats_request_latency_s'], 18)
        self.assertEqual(summary['comparisons']['KRC_vs_K1']['call_reduction_pct'], -50)
        self.assertEqual(summary['comparisons']['KRC_vs_K1']['request_time_reduction_pct'], -100)
        self.assertNotIn('F', summary)
        self.assertEqual((status['optimizer_steps'], status['feedback_collections']), (0, 0))

    def test_heterogeneous_request_ratios_are_pooled_not_averaged(self):
        def mutate(record, options):
            ordinal = options['request_index']
            repeat = int(options['request_id'].split('_')[1][1:])
            method = record['checkpoint_id'][1:]
            means = {'K1': (1., 9., 10.), 'KRC': (1., 3., 10.)}
            record['metadata']['request_latency_s'] = means[method][ordinal] * (.5, 1., 1.5)[repeat]
            record['metadata']['native_generate_s'] = record['metadata']['request_latency_s'] * .9
            if method == 'K1' and ordinal == 1:
                harmful(record)
        self.evaluate(Backend(self.requests, mutate))
        summary = self.read('report/SUMMARY.json')
        self.assertEqual(summary['K1']['verification_calls'], 7)
        self.assertEqual(summary['KRC']['verification_calls'], 6)
        self.assertEqual(summary['K1']['summed_mean_request_latency_s'], 20)
        self.assertEqual(summary['KRC']['summed_mean_request_latency_s'], 14)
        self.assertAlmostEqual(summary['comparisons']['KRC_vs_K1']['call_reduction_pct'], 100/7)
        self.assertAlmostEqual(summary['comparisons']['KRC_vs_K1']['request_time_reduction_pct'], 30)

    def test_preexisting_canonical_or_fake_arm_rejected_before_backend_work(self):
        with self.assertRaisesRegex(ValueError, 'first K1'):
            evaluate(Backend(self.requests), [existing_request(0)], self.checkpoints, self.root / 'invalid', qualification_requests=1)
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            validate_inputs(self.requests, {**self.checkpoints, 'F': '/F'}, 1)

    def test_import_has_no_model_training_optimizer_or_CUDA_side_effect(self):
        script = "import sys; import experiments.krc_confirmation.evaluate; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert 'online_feedback.training' not in sys.modules"
        process = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_final_weight_failure_preserves_generations_but_excludes_checkpoint(self):
        backend = Backend(self.requests)
        native_check = backend.check_weights

        def fail_final(checkpoint):
            result = native_check(checkpoint)
            if len(backend.weight_checks) == 3:
                result['all_equal'] = False
                result['status'] = 'failed'
            return result

        backend.check_weights = fail_final
        with self.assertRaisesRegex(RuntimeError, 'Native tensor equality'):
            self.evaluate(backend)
        status = self.read('STATUS.json')
        self.assertEqual(status['returned_generations'], 17)
        self.assertEqual(status['attempted_generations'], 17)
        self.assertEqual(status['unattempted_generations'], 1)
        self.assertEqual(status['unknown_generation_attempts'], 0)
        self.assertEqual(status['weight_checks']['after'], {'K1': False})
        self.assertEqual(status['invalid_generations'], 9)
        self.assertEqual(self.read('report/SUMMARY.json')['valid_requests'], 0)
        self.assertTrue(all('checkpoint_weight_integrity' in row['errors']
                            for row in self.rows() if row['method'] == 'K1'))
        self.assertFalse((self.root / 'evaluation/WEIGHTS_AFTER_KRC.json').exists())

    def test_initial_weight_failure_attempts_no_generation(self):
        backend = Backend(self.requests)
        native_check = backend.check_weights

        def fail(checkpoint):
            result = native_check(checkpoint)
            result['all_equal'] = False
            return result

        backend.check_weights = fail
        with self.assertRaisesRegex(RuntimeError, 'Native tensor equality'):
            self.evaluate(backend)
        status = self.read('STATUS.json')
        self.assertEqual(status['attempted_generations'], 0)
        self.assertEqual(status['unattempted_generations'], 18)
        self.assertEqual(status['weight_checks']['before'], {'K1': False})

    def test_leave_one_out_keeps_all_requests_in_primary_and_reports_gross_changes(self):
        def mutate(record, options):
            arm = record['checkpoint_id']
            ordinal = options['request_index']
            if (arm == '/K1' and ordinal == 0) or (arm == '/KRC' and ordinal == 1):
                harmful(record)

        self.evaluate(Backend(self.requests, mutate))
        summary = self.read('report/SUMMARY.json')
        comparison = summary['comparisons']['KRC_vs_K1']
        self.assertEqual(summary['valid_requests'], 3)
        self.assertEqual(comparison['call_reduction_pct'], 0)
        self.assertEqual(comparison['call_gross_saved'], 1)
        self.assertEqual(comparison['call_gross_extra'], 1)
        self.assertEqual(comparison['call_net_saved'], 0)
        self.assertEqual(comparison['call_request_directions'], {
            'improved': 1, 'regressed': 1, 'tied': 1,
        })
        sensitivity = summary['leave_one_out']
        self.assertEqual(sensitivity['omissions'], 3)
        self.assertEqual(sensitivity['minimum_call_reduction_pct'], -25)
        self.assertAlmostEqual(sensitivity['maximum_call_reduction_pct'], 20)
        self.assertEqual(sensitivity['nonpositive_call_reduction_omissions'], 2)
        self.assertTrue((self.root / 'evaluation/report/LEAVE_ONE_OUT.csv').is_file())

    def test_termination_and_runtime_mismatch_stop_without_retry(self):
        for field, value in (('finish_reason', 'length'), ('block_size', 10)):
            with self.subTest(field=field):
                def mutate(record, options):
                    if field == 'finish_reason':
                        record[field] = value
                    else:
                        record['metadata'][field] = value

                destination = 'mismatch_' + field
                with self.assertRaisesRegex(RuntimeError, 'correctness'):
                    self.evaluate(Backend(self.requests, mutate), destination)
                status = json.loads((self.root / destination / 'STATUS.json').read_text())
                self.assertEqual(status['returned_generations'], 1)
                self.assertEqual(status['invalid_generations'], 1)
                self.assertEqual(status['unattempted_generations'], 17)
                self.assertFalse(status['automatic_retry'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
