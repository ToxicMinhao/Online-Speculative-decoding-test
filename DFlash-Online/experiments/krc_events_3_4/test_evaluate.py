"""CPU fixtures for three-arm budgets, native validity and count-only history."""

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.reference_replay_event.test_evaluate import Backend, panel as old_panel
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate, summarize
from .evaluation_protocol import (
    ARMS, HISTORICAL_ARMS, method_order, prepare_requests, rotation_balance,
    validate_inputs, validate_plan,
)


def panel(name='development', count=64):
    value = old_panel(name, count)
    counts = value.pop('retained_krc_counts')
    value['expected_control_counts'] = ({'O4': dict(counts), 'K4': dict(counts)}
                                        if name == 'development' else None)
    history = HISTORICAL_ARMS if name == 'development' else ('K1', 'KRC1', 'K2', 'KRC2')
    value['historical_counts'] = {arm: {key: 4 if arm.endswith('1') else 3 for key in counts}
                                  for arm in history}
    return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = panel(count=3)
        self.checkpoints = {arm: '/' + arm for arm in ARMS}

    def evaluate(self, mutation=None, name='evaluation', gate=1, backend=None):
        selected = self.panel
        return evaluate(backend or Backend(selected['requests'], mutation), selected['requests'],
            self.checkpoints, self.root / name, expected_control_counts=selected['expected_control_counts'],
            historical_counts=selected['historical_counts'], qualification_requests=gate)

    def read(self, path):
        return json.loads((self.root / path).read_text())

    def rows(self):
        return [json.loads(line) for line in (self.root / 'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]

    def test_full_1152_budget_and_included72_gate(self):
        total = 0
        for name in ('development', 'additional'):
            self.panel = panel(name)
            backend = Backend(self.panel['requests'])
            result = self.evaluate(name=name, gate=8 if name == 'development' else 0, backend=backend)
            self.assertEqual(result['completed_valid_generations'], 576)
            self.assertEqual(result['unattempted_generations'], 0)
            self.assertEqual(result['optimizer_steps'], 0)
            self.assertEqual(result['feedback_collections'], 0)
            self.assertEqual(backend.weight_checks, ['/O4', '/K4', '/KRC4'])
            self.assertEqual([row[0][1:] for row in backend.executions],
                [arm for index in range(64) for repeat in range(3) for arm in method_order(index, repeat)])
            self.assertTrue(all(row[2]['purpose'] == 'evaluation' and row[2]['max_new_tokens'] == 2048
                                for row in backend.executions))
            total += result['returned_generations']
        self.assertEqual(total, 1152)
        self.assertEqual(self.read('development/QUALIFICATION.json')['generations'], 72)
        self.assertFalse((self.root / 'additional/QUALIFICATION.json').exists())
        self.assertEqual(self.read('additional/STATUS.json')['qualification'], 'not_required_prior_panel_gate')

    def test_rotation_all_positions_and_starts_exactly_balance_across_repeats(self):
        balance = rotation_balance(64)
        self.assertTrue(all(sorted(row.values()) == [21, 21, 22] for row in balance['starts_by_repeat']))
        self.assertEqual(balance['starts_all_repeats'], {arm: 64 for arm in ARMS})
        for index in range(64):
            for position in range(3):
                self.assertEqual({method_order(index, repeat)[position] for repeat in range(3)}, set(ARMS))

    def test_both_development_controls_gate_exact_per_request_counts(self):
        for arm, attempted in (('O4', 1), ('K4', 2)):
            def mutate(record, options):
                if record['checkpoint_id'] == '/' + arm:
                    harmful(record)
            with self.assertRaisesRegex(RuntimeError, 'retained_' + arm + '_verification_calls'):
                self.evaluate(mutate, name=arm)
            result = self.read(arm + '/STATUS.json')
            self.assertEqual((result['returned_generations'], result['invalid_generations'],
                              result['unattempted_generations']), (attempted, 1, 27 - attempted))
            self.assertEqual(self.read(arm + '/0000_r0_' + arm + '.json')['metadata']['verification_calls'], 3)

    def test_additional_controls_have_no_historical_gate_and_absent_O_stays_absent(self):
        self.panel = panel('additional', 3)
        def mutate(record, options):
            if record['checkpoint_id'] in ('/O4', '/K4'):
                harmful(record)
        self.assertEqual(self.evaluate(mutate, gate=0)['status'], 'completed')
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['O4']['verification_calls'], 9)
        self.assertEqual(summary['K4']['verification_calls'], 9)
        self.assertIsNone(summary['event_count_curve'][0]['O'])
        self.assertIsNone(summary['event_count_curve'][1]['O'])
        self.assertEqual(summary['event_count_curve'][2]['O'], 9)
        self.assertNotIn('O1', summary['historical_counts'])
        self.assertEqual(self.read('evaluation/PROTOCOL.json')['historical_count_gates'], [])
        self.assertTrue(all(row['source_method'] == 'K1' for row in self.read('evaluation/CANONICAL.json').values()))

    def test_harmful_KRC4_continues_and_O_is_primary(self):
        def mutate(record, options):
            if record['checkpoint_id'] == '/KRC4':
                harmful(record)
        self.assertEqual(self.evaluate(mutate)['status'], 'completed')
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['primary_comparison'], 'KRC4_vs_O4')
        self.assertEqual(set(summary['comparisons']), {'KRC4_vs_O4', 'K4_vs_O4', 'KRC4_vs_K4'})
        for comparison in ('KRC4_vs_O4', 'KRC4_vs_K4'):
            self.assertEqual(summary['comparisons'][comparison]['call_reduction_pct'], -50)
            self.assertEqual(summary['comparisons'][comparison]['call_gross_extra'], 3)
            self.assertEqual(summary['comparisons'][comparison]['call_request_directions'],
                             dict(improved=0, regressed=3, tied=0))
        self.assertEqual(summary['KRC4']['verification_calls'], 9)
        self.assertEqual(summary['count_learning_curve']['KRC4_vs_KRC2']['call_reduction_pct'], 0)
        self.assertEqual(summary['relative_advantage_change_percentage_points']['KRC_vs_K_event2_to4'], -50)
        self.assertFalse(any('latency' in key for value in summary['count_learning_curve'].values() for key in value))

    def test_canonical_tokens_termination_and_repeat_counts_stop(self):
        def wrong_tokens(record, options):
            record['response_ids'][1] = 777
            record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.evaluate(wrong_tokens)
        self.assertEqual(self.read('evaluation/CANONICAL.json')['development_0']['response_ids'][1], 11)
        def wrong_finish(record, options):
            record['finish_reason'] = 'length'
        with self.assertRaisesRegex(RuntimeError, 'finish_reason'):
            self.evaluate(wrong_finish, name='finish')
        def inconsistent(record, options):
            if options['request_id'] == '0000_r1_KRC4':
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.evaluate(inconsistent, name='repeat')
        self.assertEqual(self.read('repeat/STATUS.json')['invalid_generations'], 2)

    def test_unknown_reply_nonfinite_and_malformed_raw_are_preserved(self):
        def interrupted(record, options):
            if options['request_id'] == '0000_r0_K4':
                raise RuntimeError('lost reply')
        with self.assertRaisesRegex(RuntimeError, 'lost reply'):
            self.evaluate(interrupted)
        result = self.read('evaluation/STATUS.json')
        self.assertEqual((result['returned_generations'], result['unknown_generation_attempts'],
                          result['unattempted_generations']), (1, 1, 25))
        def nonfinite(record, options):
            record['metadata']['request_latency_s'] = float('nan')
        with self.assertRaises(RuntimeError):
            self.evaluate(nonfinite, name='nan')
        self.assertIn('NaN', (self.root / 'nan/0000_r0_O4.json').read_text())
        self.assertEqual(self.read('nan/STATUS.json')['invalid_generations'], 1)
        self.assertFalse((self.root / 'nan/REPORT_FAILURE.json').exists())
        def malformed(record, options):
            record.pop('metadata')
        with self.assertRaises(RuntimeError):
            self.evaluate(malformed, name='malformed')
        self.assertEqual(self.read('malformed/STATUS.json')['invalid_generations'], 1)
        self.assertFalse((self.root / 'malformed/REPORT_FAILURE.json').exists())

    def test_publication_and_weight_failures_precede_every_generation(self):
        class BadPublication(Backend):
            def publish(self, checkpoint):
                receipt = super().publish(checkpoint)
                receipt['native_status']['cache_tokens'] = 1
                return receipt
        with self.assertRaisesRegex(RuntimeError, 'cache_empty'):
            self.evaluate(backend=BadPublication(self.panel['requests']))
        result = self.read('evaluation/STATUS.json')
        self.assertEqual((result['attempted_generations'], result['unattempted_generations']), (0, 27))
        self.assertTrue((self.root / 'evaluation/ACTIVATIONS.jsonl').exists())
        class BadWeights(Backend):
            def check_weights(self, checkpoint):
                receipt = super().check_weights(checkpoint)
                receipt['all_equal'] = False
                return receipt
        with self.assertRaisesRegex(RuntimeError, 'native_equality'):
            self.evaluate(name='weights', backend=BadWeights(self.panel['requests']))
        self.assertEqual(self.read('weights/STATUS.json')['attempted_generations'], 0)
        self.assertTrue((self.root / 'weights/WEIGHTS_O4.json').exists())

    def test_qualified_cache_and_capture_contracts_are_reused(self):
        def cache(record, options):
            record['metadata']['cache_policy'] = 'reuse_previous_request'
        with self.assertRaisesRegex(RuntimeError, 'cache_policy'):
            self.evaluate(cache)
        def capture(record, options):
            record['metadata']['capture']['collect_features'] = True
        with self.assertRaisesRegex(RuntimeError, 'capture_contract'):
            self.evaluate(capture, name='capture')
        self.assertEqual(self.read('capture/STATUS.json')['invalid_generations'], 1)

    def test_nested_timers_pooled_ratio_repeat_totals_and_loading_excluded(self):
        def mutate(record, options):
            index = options['request_index']
            repeat = int(options['request_id'].split('_')[1][1:])
            latency = {'/O4': (1., 9., 10.), '/K4': (1., 5., 10.), '/KRC4': (1., 3., 10.)}[
                record['checkpoint_id']][index] * (.5, 1., 1.5)[repeat]
            record['metadata']['request_latency_s'] = latency
            record['metadata']['native_generate_s'] = .9 * latency
            record['metadata']['capture']['cache_maintenance_s'] = .1 * latency
        self.evaluate(mutate)
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['KRC4']['summed_mean_request_latency_s'], 14)
        self.assertAlmostEqual(summary['KRC4']['summed_mean_native_generate_s'], 12.6)
        self.assertAlmostEqual(summary['KRC4']['summed_mean_cache_maintenance_s'], 1.4)
        self.assertAlmostEqual(summary['comparisons']['KRC4_vs_O4']['request_time_reduction_pct'], 30)
        self.assertGreater(summary['execution']['activation_s'], 10000)
        self.assertIn('loading', summary['latency_scope'])
        totals = list(csv.DictReader((self.root / 'evaluation/report/REPEAT_TOTALS.csv').read_text().splitlines()))
        self.assertEqual(len(totals), 9)
        self.assertEqual({row['method'] for row in totals}, set(ARMS))
        self.assertEqual([float(row['request_latency_s']) for row in totals if row['method'] == 'KRC4'], [7, 14, 21])

    def test_report_rechecks_native_counts_and_common_cohort(self):
        self.evaluate()
        rows = self.rows()
        raw = self.read('evaluation/0000_r0_KRC4.json')
        harmful(raw)
        (self.root / 'evaluation/0000_r0_KRC4.json').write_text(json.dumps(raw))
        summary, requests, *_ = summarize(rows, self.panel['requests'], checkpoints=self.checkpoints,
            expected_control_counts=self.panel['expected_control_counts'], historical_counts=self.panel['historical_counts'])
        self.assertEqual(summary['valid_requests'], 2)
        self.assertTrue(requests[0]['errors'])
        self.assertEqual(summary['O4']['verification_calls'], 4)
        self.assertEqual(summary['historical_counts']['O1']['requests'], 2)
        self.assertEqual(summary['historical_counts']['O1']['verification_calls'], 8)

    def test_protocol_exact_panels_history_and_checkpoint_contract(self):
        plan = dict(retained_o4_checkpoint='/O4', retained_k4_checkpoint='/K4',
                    panels=[panel(), panel('additional')])
        self.assertEqual(validate_plan(plan)['planned_generations'], 1152)
        self.assertEqual(validate_plan(plan)['included_gate_generations'], 72)
        for index, counts in ((0, None), (1, plan['panels'][0]['expected_control_counts'])):
            changed = copy.deepcopy(plan)
            changed['panels'][index]['expected_control_counts'] = counts
            with self.assertRaisesRegex(ValueError, 'Only development'):
                validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed['panels'][1]['historical_counts']['O1'] = changed['panels'][1]['historical_counts']['K1']
        with self.assertRaisesRegex(ValueError, 'exact historical'):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed['panels'][1]['requests'][0]['canonical_source_method'] = 'F'
        with self.assertRaisesRegex(ValueError, 'canonical'):
            validate_plan(changed)
        requests = prepare_requests(self.panel['requests'], self.panel['expected_control_counts'])
        for checkpoints in ({'O4': '/same', 'K4': '/same', 'KRC4': '/different'},
                            {'K4': '/K4', 'KRC4': '/KRC4'}):
            with self.assertRaisesRegex(ValueError, 'three distinct'):
                validate_inputs(requests, checkpoints, self.panel['expected_control_counts'], 1)
        counts = copy.deepcopy(self.panel['expected_control_counts'])
        counts['O4']['development_0'] = True
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            validate_inputs(requests, self.checkpoints, counts, 1)

    def test_import_has_no_training_or_CUDA_construction(self):
        code = ("import sys; import experiments.krc_events_3_4.evaluate; "
                "assert 'torch' not in sys.modules; assert 'online_feedback.training' not in sys.modules")
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
