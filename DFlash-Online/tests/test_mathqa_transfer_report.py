"""Saved failure statuses cannot be silently converted into endpoint gains."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mathqa_transfer import report as R, statistics as S
from test_mathqa_transfer_statistics import rows


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.values = rows()
        self.requests = [dict(panel_ordinal=i, request_id=f'mathqa:{i}') for i in range(1024)]
        self.plan = dict(evaluation_requests=self.requests,
                        checkpoints={method: '/export/' + method for method in S.METHODS},
                        execution_source=dict(commit='0'*40, dirty=False),
                        control_request=self.requests[0], control_canonical={}, control_counts={})
        self.campaign = dict(status='completed', loading_s=5., activation_s=3., campaign_wall_s=10.,
            actual_work=dict(planned_generations=9219, attempted_generations=9219,
                returned_generations=9219, completed_valid_generations=9219,
                invalid_generations=0, pending_canonical_generations=0,
                unknown_generation_attempts=0, unattempted_generations=0,
                optimizer_steps=0, training_presentations=0, feedback_collections=0,
                teacher_preparations=0, extra_warmup_generations=0))
        self.controls = dict(status='passed', errors=[], rows=[dict(method=m) for m in S.METHODS])
        self.original_control_auditor = R.audit_controls
        patcher = patch.object(R, 'audit_controls', return_value=self.controls)
        self.control_auditor = patcher.start()
        self.addCleanup(patcher.stop)
        self.audit = dict(status='passed', errors=[], rows=self.values)
        for stage in ('PREFLIGHT', 'POSTFLIGHT'):
            self.write(stage + '.json', dict(status='passed', model_work=0))
        self.write('sealed/PLAN.json', self.plan)
        self.write('campaign_001/STATUS.json', self.campaign)
        self.write('campaign_001/CONTROL_AUDIT.json', self.controls)
        self.write('campaign_001/FINAL_AUDIT.json', {'status': 'passed', 'errors': []})
        raw = self.root/'campaign_001/evaluation/final/MEASUREMENTS.jsonl'
        raw.parent.mkdir(parents=True)
        raw.write_text('\n'.join(json.dumps(row) for row in self.values)+'\n')

    def write(self, name, value):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value)+'\n')

    def test_complete_report_leads_with_latency_and_keeps_controls_separate(self):
        with patch.object(R, 'validate_plan'), patch.object(R, 'audit_endpoint', return_value=self.audit), \
                patch.object(S, 'RESAMPLES', 10):
            result = R.build_report(self.root)
        self.assertEqual(result['status'], 'valid_complete')
        self.assertEqual(result['endpoint']['execution_count'], 9216)
        self.assertEqual(result['raw_repeat_rows_preserved'], 9216)
        self.assertEqual(result['costs']['loading_s'], 5.)
        self.assertEqual(result['budget']['total_generations'], 9219)
        text = (self.root/'final_report/REPORT.md').read_text()
        self.assertLess(text.index('## Complete-request time'), text.index('## Drafting counts'))
        self.assertIn('confirmatory', text)
        self.assertTrue((self.root/'final_report/ALL_REPEATS.csv').exists())
        self.assertTrue((self.root/'final_report/PAIRED_REQUESTS.csv').exists())

    def test_failed_audit_controls_or_campaign_never_runs_statistics(self):
        for failure in ('audit', 'control', 'campaign', 'original_audit'):
            self.write('campaign_001/STATUS.json', self.campaign)
            self.write('campaign_001/CONTROL_AUDIT.json', self.controls)
            self.write('campaign_001/FINAL_AUDIT.json', {'status': 'passed', 'errors': []})
            audit = self.audit
            if failure == 'audit':
                audit = dict(self.audit, status='invalid', errors=['native_output_mismatch'])
            elif failure == 'control':
                self.write('campaign_001/CONTROL_AUDIT.json', dict(self.controls, rows=self.controls['rows'][:2]))
            elif failure == 'campaign':
                self.write('campaign_001/STATUS.json', dict(self.campaign, status='failed'))
            else:
                self.write('campaign_001/FINAL_AUDIT.json', {'status': 'failed', 'errors': ['old_failure']})
            with self.subTest(failure=failure), patch.object(R, 'validate_plan'), \
                    patch.object(R, 'audit_endpoint', return_value=audit), patch.object(R, 'summarize') as summarize:
                result = R.build_report(self.root, self.root/failure)
            summarize.assert_not_called()
            self.assertEqual(result['status'], 'invalid_or_incomplete')
            self.assertIsNone(result['endpoint'])
            self.assertEqual(result['raw_repeat_rows_preserved'], 9216)
            self.assertFalse((self.root/failure/'PAIRED_REQUESTS.csv').exists())
            self.assertTrue((self.root/failure/'RAW_REPEAT_ROWS.csv').exists())

    def test_report_requires_fresh_destination_to_preserve_prior_findings(self):
        destination = self.root/'existing'
        destination.mkdir()
        (destination/'REPORT.md').write_text('Original evidence')
        with self.assertRaises(FileExistsError):
            R.build_report(self.root, destination)
        self.assertEqual((destination/'REPORT.md').read_text(), 'Original evidence')

    def test_raw_controls_admission_and_accounting_block_gain_claims(self):
        failures = ('raw_control', 'PREFLIGHT', 'POSTFLIGHT', 'extra_generation',
                    'unattempted', 'optimizer_step', 'missing_work')
        for failure in failures:
            self.control_auditor.return_value = self.controls
            for stage in ('PREFLIGHT', 'POSTFLIGHT'):
                self.write(stage + '.json', dict(status='passed', model_work=0))
            campaign = dict(self.campaign, actual_work=dict(self.campaign['actual_work']))
            if failure == 'raw_control':
                self.control_auditor.return_value = dict(self.controls, status='invalid', errors=['native_capture_changed'])
            elif failure in ('PREFLIGHT', 'POSTFLIGHT'):
                self.write(failure + '.json', dict(status='failed', model_work=0))
            elif failure == 'extra_generation':
                campaign['actual_work']['attempted_generations'] += 1
            elif failure == 'unattempted':
                campaign['actual_work']['unattempted_generations'] = 1
            elif failure == 'optimizer_step':
                campaign['actual_work']['optimizer_steps'] = 1
            else:
                del campaign['actual_work']['returned_generations']
            self.write('campaign_001/STATUS.json', campaign)
            with self.subTest(failure=failure), patch.object(R, 'validate_plan'), \
                    patch.object(R, 'audit_endpoint', return_value=self.audit), patch.object(R, 'summarize') as summarize:
                result = R.build_report(self.root, self.root/failure)
            summarize.assert_not_called()
            self.assertEqual(result['status'], 'invalid_or_incomplete')

    def test_audit_sidecars_use_production_root(self):
        # The real evaluator may not yet be imported by other CPU-only tests.
        from experiments.mathqa_transfer import evaluation
        with patch.object(evaluation, 'audit_stage', return_value={}) as audit:
            R.audit_endpoint(self.root, self.plan)
        self.assertEqual(audit.call_args.kwargs['evidence_root'], self.root)
        self.assertEqual(audit.call_args.args[0], self.root/'campaign_001/evaluation/final')
        # Call the original implementation, rather than setUp's mock.
        with patch.object(evaluation, 'audit_stage', return_value={}) as audit:
            self.original_control_auditor(self.root, self.plan)
        self.assertEqual(audit.call_args.kwargs['evidence_root'], self.root)
        self.assertEqual(audit.call_args.args[0], self.root/'campaign_001/evaluation/controls')
        self.assertEqual(audit.call_args.kwargs['control'], dict(canonical={}, counts={}))


if __name__ == '__main__':
    unittest.main()
