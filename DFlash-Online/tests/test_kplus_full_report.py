"""Treatment diagnostics are counted once at the original-response boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_full_gsm8k import report as R
from experiments.kplus_full_gsm8k import training as T
from experiments.kplus_full_gsm8k.campaign import ARM_LIMITS


class ReportTests(unittest.TestCase):
    def test_response_metrics_once_not_optimizer_accumulations(self):
        sums = dict(kplus_W1=1., kplus_W2=2., kplus_response_count=1.,
            kplus_mixed_response=1., kplus_error_weight_moved=.3,
            kplus_error_weight_moved_share=.1, kplus_conservation_error=0.)
        records = [dict(**{'pass': i // 128}, prompt_tokens=2, response_tokens=3, active_anchors=4,
            metrics={'kplus_W1': 999.}, response_observation={'sum_metrics': sums}) for i in range(256)]
        status = dict(status='completed', events_completed=58, tail_completed=49,
            actual_work=dict(attempts=ARM_LIMITS, completed=ARM_LIMITS, unknown=dict.fromkeys(ARM_LIMITS, 0)),
            event_costs=[], worker_wall_s=1., backend_loading_s=1., trainer_loading_s=1., initial_validation_s=1.)
        def read(path):
            path = Path(path)
            if path.name == 'STATUS.json':
                return status
            if path.name == 'RESULT.json':
                return dict(records=records, objective_work={})
            if path.parent.name == 'boundaries':
                return dict(status='passed', event=int(path.stem.split('_')[1]))
            return {}
        with patch.object(R, 'read', side_effect=read), patch.object(T, 'validate_event_receipt'):
            result = R.training_summary(Path('/fixture'))
        totals = result['exposure']
        self.assertEqual(totals['presentations'], 14848)
        self.assertEqual(totals['kplus_W1'], 14848)
        self.assertEqual(totals['kplus_W2'], 29696)
        self.assertEqual(totals['mixed_response_frequency'], 1.)
        self.assertAlmostEqual(totals['pooled_error_weight_moved_share'], .1)
        self.assertAlmostEqual(totals['mean_response_moved_share'], .1)
        self.assertEqual(len(result['pass_rows']), 116)
        self.assertEqual(len(result['objective_work']), 58)


class ReportingRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = dict(status='failed', automatic_retry=False,
            error="ValueError('Complete final endpoint failed raw-evidence validation')",
            training_quiescence=dict(resource={'status': 'passed'}, timing={'status': 'passed'}),
            post_evaluation_boundary=dict(resource={'status': 'passed'}, timing={'status': 'passed'}))
        self.original_audit = dict(status='invalid_or_incomplete', rows=[],
            errors=["missing_or_malformed_evidence: FileNotFoundError(2, 'No such file or directory')"])
        self.training = dict(status='passed', pass_rows=[], worker_wall_s=1.,
            exposure=dict(mixed_response_frequency=.5, pooled_error_weight_moved_share=.1))
        self.audit = dict(status='passed', errors=[], rows=[{} for _ in range(6144)])
        self.write('sealed/PLAN.json', dict(references={'K': '/retained/K58'}, evaluation_requests=[],
            canonicals=str(self.root / 'sealed/CANONICALS.json'),
            baseline_counts=str(self.root / 'sealed/BASELINE_COUNTS.json'),
            execution_source={'commit': 'synthetic_fixture'}))
        self.write('sealed/CANONICALS.json', {})
        self.write('sealed/BASELINE_COUNTS.json', {})
        self.write('campaign_001/arms/KPLUS/STATUS.json', {'final_checkpoint': '/new/KPLUS58'})
        self.write('campaign_001/STATUS.json', self.campaign)
        self.write('campaign_001/FINAL_AUDIT.json', self.original_audit)
        self.write('POSTFLIGHT.json', {'status': 'passed'})
        for name, text in [('EXIT_CODE', '1\n'), ('CAMPAIGN_EXIT_CODE', '1\n'),
                           ('REPORT_EXIT_CODES', 'report=1\npostflight=0\n'),
                           ('final_report/REPORT.md', 'Original failed report must remain unchanged.\n')]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.original_bytes = {path.relative_to(self.root).as_posix(): path.read_bytes()
                               for path in self.root.rglob('*') if path.is_file()}

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    def build(self, output, *, recover=False, audit=None):
        with patch.object(R, 'training_summary', return_value=self.training), \
                patch.object(R, 'audit_stage', return_value=self.audit if audit is None else audit) as raw_audit, \
                patch.object(R, 'summarize', return_value={}) as statistics, \
                patch.object(R, 'reduce_final', return_value=[]):
            result = R.build_report(self.root, output, recover_completed_audit=recover)
        self.assertEqual(raw_audit.call_args.kwargs['evidence_root'], self.root)
        return result, statistics

    def test_explicit_recovery_accepts_complete_audit_and_preserves_original_failures(self):
        result, statistics = self.build(self.root / 'recovered_report', recover=True)
        self.assertEqual(result['status'], 'valid_complete')
        self.assertEqual(result['campaign']['status'], 'failed')
        self.assertEqual(result['reporting_recovery']['status'], 'recovered_by_saved_evidence_audit')
        self.assertEqual(result['reporting_recovery']['model_work'], 0)
        statistics.assert_called_once()
        for name, original in self.original_bytes.items():
            self.assertEqual((self.root / name).read_bytes(), original)
        for name, digest in result['reporting_recovery']['preserved_evidence'].items():
            self.assertEqual(digest, hashlib.sha256(self.original_bytes[name]).hexdigest())

    def test_failed_campaign_without_opt_in_never_computes_endpoint(self):
        result, statistics = self.build(self.root / 'default_attempt')
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertIn('campaign_not_completed', result['errors'])
        self.assertIsNone(result['endpoint'])
        self.assertIsNone(result['reporting_recovery'])
        statistics.assert_not_called()

    def test_incomplete_or_invalid_raw_audit_cannot_be_recovered(self):
        for name, changed in [('incomplete', dict(self.audit, rows=self.audit['rows'][:-1])),
                              ('invalid', dict(self.audit, status='invalid_or_incomplete', errors=['native_counter_mismatch']))]:
            with self.subTest(case=name):
                result, statistics = self.build(self.root / name, recover=True, audit=changed)
                self.assertEqual(result['status'], 'invalid_or_incomplete')
                self.assertIsNone(result['endpoint'])
                self.assertIsNone(result['reporting_recovery'])
                statistics.assert_not_called()
                self.assertFalse((self.root / name / 'MEASUREMENTS.csv').exists())

    def test_recovery_requires_exact_failure_and_complete_training(self):
        for defect in ('other_error', 'completed_campaign', 'cleanup_failure', 'automatic_retry',
                       'partial_training', 'timing_failure', 'resource_failure', 'wrong_original_audit',
                       'original_rows_present', 'wrong_exit', 'postflight_failure'):
            for name, original in self.original_bytes.items():
                (self.root / name).write_bytes(original)
            campaign, training = deepcopy(self.campaign), deepcopy(self.training)
            if defect == 'other_error':
                campaign['error'] = "ValueError('Native count mismatch')"
            elif defect == 'completed_campaign':
                campaign['status'] = 'completed'
            elif defect == 'cleanup_failure':
                campaign['cleanup_errors'] = ['worker did not close']
            elif defect == 'automatic_retry':
                campaign['automatic_retry'] = True
            elif defect == 'partial_training':
                training['status'] = 'failed'
            elif defect == 'timing_failure':
                campaign['post_evaluation_boundary']['timing']['status'] = 'failed'
            elif defect == 'resource_failure':
                campaign['training_quiescence']['resource']['status'] = 'failed'
            elif defect == 'wrong_original_audit':
                self.write('campaign_001/FINAL_AUDIT.json', dict(self.original_audit, errors=['native_counter_mismatch']))
            elif defect == 'original_rows_present':
                self.write('campaign_001/FINAL_AUDIT.json', dict(self.original_audit, rows=[{}]))
            elif defect == 'wrong_exit':
                (self.root / 'EXIT_CODE').write_text('0\n')
            else:
                self.write('POSTFLIGHT.json', {'status': 'failed'})
            with self.subTest(case=defect), self.assertRaises(ValueError):
                R.completed_audit_recovery(self.root, campaign, training, self.audit)

    def test_opt_in_does_not_silently_accept_a_different_original_status(self):
        self.write('campaign_001/STATUS.json', dict(self.campaign, status='completed'))
        result, statistics = self.build(self.root / 'wrong_original_status', recover=True)
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertIsNone(result['endpoint'])
        statistics.assert_not_called()

    def test_recovery_requires_new_separate_report_directory(self):
        for output in (None, self.root / 'final_report'):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'separate output'):
                R.build_report(self.root, output, recover_completed_audit=True)
        existing = self.root / 'existing_report'
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            R.build_report(self.root, existing, recover_completed_audit=True)
        self.assertEqual((self.root / 'final_report/REPORT.md').read_bytes(), self.original_bytes['final_report/REPORT.md'])


if __name__ == '__main__':
    unittest.main()
