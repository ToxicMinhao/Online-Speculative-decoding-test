"""Published evidence distinguishes complete results from stopped execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.adaflash_c1 import report as R
from experiments.adaflash_c1 import statistics as S
from online_feedback.storage import write_json
from test_adaflash_c1_statistics import rows


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        canonicals = self.root / 'sealed/CANONICALS.json'
        write_json(canonicals, {})
        write_json(self.root / 'sealed/PLAN.json', dict(
            execution_source=dict(commit='a' * 40, dirty=False), binding='sealed-config',
            configs=dict(A='/sealed/config_A.yaml'), original_config='/sealed/ORIGINAL_CONFIG.yaml',
            loss={'fixture': True}, references=dict(O='/retained/O58', KPLUS='/retained/KPLUS58'),
            evaluation_requests=[], canonicals=str(canonicals)))

    def test_partial_report_preserves_work_without_an_efficacy_claim(self):
        progress = dict(status='failed', events_completed=3, error='retained failure')
        write_json(self.root / 'campaign_001/arms/A/STATUS.json', progress)
        audit = dict(status='invalid_or_incomplete', errors=['unattempted endpoint'], rows=[], progress={})
        with patch.object(R, 'audit_stage', return_value=audit), \
                patch.object(R, 'training_summary', side_effect=ValueError('partial trajectory')):
            result = R.build_report(self.root)
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertIsNone(result['endpoint'])
        self.assertEqual(result['arm_progress'], progress)
        self.assertIn('unattempted endpoint', result['errors'])
        text = (self.root / 'results/REPORT.md').read_text()
        self.assertNotIn('Primary decision:', text)
        self.assertIn('a' * 40, text)

    def test_complete_report_records_checkpoints_cost_scope_and_primary_decision(self):
        write_json(self.root / 'campaign_001/STATUS.json', {'status': 'completed'})
        training = dict(worker_wall_s=12., pass_rows=[], event_costs=[],
                        numerical_qualification={'status': 'passed'})
        audit = dict(status='passed', errors=[], rows=rows(), progress={})
        with patch.object(R, 'audit_stage', return_value=audit), \
                patch.object(R, 'training_summary', return_value=training), \
                patch.object(S, 'RESAMPLES', 10):
            result = R.build_report(self.root)
        self.assertEqual(result['status'], 'valid_complete')
        self.assertEqual(result['endpoint']['interpretation']['endpoint_label'], 'favors_KPLUS')
        self.assertEqual(set(result['checkpoints']), {'O', 'KPLUS', 'A'})
        self.assertEqual(result['configuration']['binding'], 'sealed-config')
        self.assertEqual(len((self.root / 'results/PAIRED_REQUESTS.csv').read_text().splitlines()), 1025)
        self.assertEqual(len((self.root / 'results/ALL_REPEATS.csv').read_text().splitlines()), 9217)
        text = (self.root / 'results/REPORT.md').read_text()
        self.assertIn('negative entries remain unchanged', text.lower())
        self.assertIn('nested timers must not be added', text)
        self.assertIn('not independent GPU replication', text)
        self.assertEqual(json.loads((self.root / 'results/SUMMARY.json').read_text())['status'], 'valid_complete')


if __name__ == '__main__':
    unittest.main()
