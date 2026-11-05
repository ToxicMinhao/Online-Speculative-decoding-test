"""Synthetic receipt checks; these numbers are fixtures, never campaign results."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('krc34_report_builder', HERE / 'build_report.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def panel(name):
    counts = {'O4': 2994, 'K4': 2942, 'KRC4': 2960} if name == 'development' else {'O4': 3150, 'K4': 3130, 'KRC4': 3100}
    result = dict(valid_requests=64, comparisons={}, event_count_curve=[
        dict(event=1, measurement='historical', O=3075 if name == 'development' else None,
             K=3027 if name == 'development' else 3150, KRC=3005 if name == 'development' else 3125),
        dict(event=2, measurement='historical', O=3059 if name == 'development' else None,
             K=2973 if name == 'development' else 3136, KRC=2967 if name == 'development' else 3116),
        dict(event=4, measurement='fresh', O=counts['O4'], K=counts['K4'], KRC=counts['KRC4'])])
    for arm, calls in counts.items():
        result[arm] = dict(verification_calls=calls, summed_mean_request_latency_s=calls / 50,
                          summed_mean_native_generate_s=calls / 60, summed_mean_cache_maintenance_s=.1)
    for candidate, reference in B.PAIRS:
        delta = counts[reference] - counts[candidate]
        result['comparisons'][candidate + '_vs_' + reference] = dict(
            call_request_directions=dict(improved=20, regressed=30, tied=14),
            call_gross_saved=100 + max(delta, 0), call_gross_extra=100 + max(-delta, 0))
    return result


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / 'raw/campaign_001'
        self.plan = dict(execution_source=dict(execution_git_commit='a' * 40,
            dirty_files=[], working_source='/qualified/source'),
            panels=[dict(name='development'), dict(name='additional')])
        B.write_json(self.root / 'raw/sealed/PLAN.json', self.plan)

    def complete(self):
        B.write_json(self.raw / 'STATUS.json', dict(status='completed',
            evaluation_totals=dict(completed_valid_generations=1152, invalid_generations=0,
                                   unknown_generation_attempts=0, unattempted_generations=0)))
        B.write_json(self.root / 'FINAL_AUDIT.json', dict(status='passed', issues=[]))
        for name in ('development', 'additional'):
            directory = self.raw / 'evaluation' / name
            B.write_json(directory / 'report/SUMMARY.json', panel(name))
            B.write_json(directory / 'STATUS.json', dict(status='completed', completed_valid_generations=576,
                invalid_generations=0, unknown_generation_attempts=0, unattempted_generations=0,
                activation_s=2, weight_validation_s=1))
            repeats = [dict(repeat=repeat, method=arm, requests=64, verification_calls=panel(name)[arm]['verification_calls'],
                **{timer: panel(name)[arm]['summed_mean_' + timer] * (.99, 1, 1.01)[repeat]
                   for timer in B.TIMERS})
                for repeat in range(3) for arm in B.ARMS]
            B.write_csv(directory / 'report/REPEAT_TOTALS.csv', repeats)
            (directory / 'report/PER_REQUEST.csv').write_text('request_id,valid\nfixture,true\n')
            (directory / 'report/ALL_REPEATS.csv').write_text('repeat,valid\n0,true\n')

    def test_pooled_percent_mixed_correction_result_and_absent_history(self):
        self.complete()
        summary = B.build(self.root, postprocess_commit='b' * 40)
        self.assertTrue(summary['completed_audited_endpoint'])
        rows = B.comparison_rows(summary['panels']['development'])
        self.assertAlmostEqual(rows[0]['call_reduction_pct'], 100 * 34 / 2994)
        self.assertAlmostEqual(rows[2]['call_reduction_pct'], -100 * 18 / 2942)
        text = (self.root / 'REPORT.md').read_text()
        self.assertIn('Against K, the advantage reverses', text)
        self.assertIn('advantage persists and widens', text)
        self.assertIn('Earlier O counts are absent', text)
        self.assertIn('within-arm change does not establish relative benefit', text)
        self.assertEqual(summary['source']['execution_git_commit'], 'a' * 40)
        self.assertEqual(summary['postprocessing']['git_commit'], 'b' * 40)
        self.assertIn('| 2 | historical | absent | 3136 | 3116 |', text)
        self.assertEqual((self.root / 'analysis/development/PER_REQUEST.csv').read_bytes(),
                         (self.raw / 'evaluation/development/report/PER_REQUEST.csv').read_bytes())
        self.assertEqual(len(summary['panels']['development']['repeats']), 9)

    def test_no_audit_or_incomplete_cohort_never_promotes_endpoint(self):
        self.complete()
        (self.root / 'FINAL_AUDIT.json').unlink()
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])
        self.assertIn('No completed endpoint conclusion', (self.root / 'REPORT.md').read_text())
        B.write_json(self.root / 'FINAL_AUDIT.json', dict(status='passed', issues=[]))
        partial = panel('additional')
        partial['valid_requests'] = 63
        B.write_json(self.raw / 'evaluation/additional/report/SUMMARY.json', partial)
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])

    def test_failure_without_panel_results_keeps_unknown_and_unattempted(self):
        status = dict(status='failed', evaluation_totals=dict(completed_valid_generations=72,
            invalid_generations=0, unknown_generation_attempts=1, unattempted_generations=1079),
            panels=dict(development=dict(completed_valid_generations=72, invalid_generations=0,
                unknown_generation_attempts=1, unattempted_generations=503),
                additional=dict(status='unattempted', completed_valid_generations=0,
                    invalid_generations=0, unknown_generation_attempts=0, unattempted_generations=576)))
        B.write_json(self.raw / 'STATUS.json', status)
        summary = B.build(self.root)
        self.assertEqual(summary['status'], status)
        self.assertFalse(summary['completed_audited_endpoint'])
        self.assertEqual(summary['panels']['additional']['execution']['unattempted_generations'], 576)
        self.assertNotIn('O4', summary['panels']['additional'])
        self.assertIn('72 / 0 / 1 / 503', (self.root / 'REPORT.md').read_text())
        self.assertIsNone(summary['training'][0]['Adam_steps'])

    def test_training_cost_nesting_and_exact_retained_paths(self):
        costs = dict(events={'event_003': dict(event_wall_s=100, peak_allocated_bytes=30e9,
            peak_reserved_bytes=32e9, event=dict(training_s=80, checkpoint_save_s=8, export_s=10, state_retention_s=.1),
            correction=dict(counts=dict(Adam_completed=128, presentations_completed=256, K_backward_completed=256),
                work=dict(original_response_forwards=256, pre_reference_response_forwards=256,
                          post_k_reference_response_forwards=256, violation_backward_completed=220),
                costs=dict(pre_reference_s=5, post_k_s=20, correction_s=5, integrity_s=3,
                           pre_reference_integrity_s=1, first_window_gate_s=1)))})
        receipt = dict(status='completed', start_step=256, end_step=384,
            full_state_path='/run/KRC/events/event_003/training_state.pt',
            checkpoint_path='/run/KRC/events/event_003/export')
        B.write_json(self.raw / 'KRC/events/event_003/RESULT.json', receipt)
        rows = B.training_rows(self.raw, costs, dict(training=dict(events={'event_003': dict(
            realized_nonzero_corrections=125, correction_branches=dict(applied=125, zero_violation=3),
            exposure=dict(kept_anchors=65000, available_occurrences=950000,
                          selection=dict(selected_position_occurrences=550000)),
            reference_presentations=256, reference_anchors=65000,
            reference_available_occurrences=950000, reference_protected_occurrences=440000)})))
        self.assertEqual(rows[0]['intrinsic_reference_correction_s'], 30)
        self.assertEqual(rows[0]['ordinary_K_and_other_overhead_residual_s'], 45)
        self.assertEqual(rows[0]['event_wall_s'], 100)
        self.assertEqual(rows[0]['full_state_path'], receipt['full_state_path'])
        self.assertEqual(rows[0]['BF16_nonzero_corrections'], 125)
        self.assertEqual(rows[0]['ordinary_anchors'], 65000)
        self.assertEqual(rows[0]['ordinary_available_occurrences'], 950000)
        self.assertEqual(rows[0]['ordinary_selected_occurrences'], 550000)
        self.assertEqual(rows[0]['reference_presentations'], 256)
        self.assertEqual(rows[0]['reference_available_occurrences'], 950000)
        self.assertEqual(rows[0]['reference_protected_occurrences'], 440000)
        self.assertIsNone(rows[1]['event_wall_s'])

    def test_failed_window_before_correction_keeps_partial_work(self):
        correction = dict(status='failed', counts=dict(Adam_completed=1, presentations_completed=2),
                          work=dict(original_response_forwards=2), steps=[dict(step=1, correction=None)])
        B.write_json(self.raw / 'events/event_003/CORRECTION_EVIDENCE.json', correction)
        B.write_json(self.raw / 'STATUS.json', dict(status='failed'))
        summary = B.build(self.root)
        self.assertFalse(summary['completed_audited_endpoint'])
        event = summary['training'][0]
        self.assertEqual(event['status'], 'failed')
        self.assertEqual(event['Adam_steps'], 1)
        self.assertEqual(event['ordinary_presentations'], 2)
        self.assertEqual(json.loads(event['correction_branches']), {'unavailable': 1})
        self.assertIsNone(event['event_wall_s'])
        self.assertIsNone(event['full_state_path'])
        audited = {'training': {'events': {'event_003': {'correction_branches': {'not_applied': 1}}}}}
        self.assertEqual(json.loads(B.training_rows(self.raw, {}, audited)[0]['correction_branches']),
                         {'not_applied': 1})

    def test_missing_or_contradictory_execution_and_repeat_artifacts_block_endpoint(self):
        self.complete()
        directory = self.raw / 'evaluation/additional'
        path = directory / 'report/REPEAT_TOTALS.csv'
        original = path.read_text()
        path.unlink()
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])
        self.assertIn('Missing panel report files: REPEAT_TOTALS.csv', (self.root / 'REPORT.md').read_text())
        path.write_text(original)
        rows = list(csv.DictReader(original.splitlines()))
        rows[0]['request_latency_s'] = str(float(rows[0]['request_latency_s']) + 10)
        B.write_csv(path, rows)
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])
        path.write_text(original)
        status = B.read(directory / 'STATUS.json')
        status['unknown_generation_attempts'] = 1
        B.write_json(directory / 'STATUS.json', status)
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])
        status['unknown_generation_attempts'] = 0
        B.write_json(directory / 'STATUS.json', status)
        total = B.read(self.raw / 'STATUS.json')
        total['evaluation_totals']['completed_valid_generations'] = 1151
        B.write_json(self.raw / 'STATUS.json', total)
        self.assertFalse(B.build(self.root)['completed_audited_endpoint'])

    def test_persistence_tie_and_prior_regression_wording(self):
        value = panel('development')
        value['event_count_curve'][-1]['KRC'] = value['event_count_curve'][-1]['K']
        self.assertIn('advantage disappears into a tie', B.persistence(value))
        value['event_count_curve'][1]['KRC'] = value['event_count_curve'][1]['K'] + 1
        self.assertIn('the advantage does not emerge', B.persistence(value))


if __name__ == '__main__':
    unittest.main()
