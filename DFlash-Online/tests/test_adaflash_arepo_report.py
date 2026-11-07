"""Complete and stopped supplement reports retain validity and provenance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.adaflash_arepo import report as R
from experiments.adaflash_arepo import statistics as S
from online_feedback.storage import write_json
from test_adaflash_arepo_statistics import rows


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        canonicals = self.root / 'sealed/CANONICALS.json'
        write_json(canonicals, {})
        write_json(self.root / 'sealed/PLAN.json', dict(
            execution_source=dict(commit='a' * 40, dirty=False), binding='sealed-config',
            configs=dict(AREPO='/sealed/config_AREPO.yaml'), original_config='/sealed/ORIGINAL_CONFIG.yaml',
            loss={'fixture': True}, references=dict(O='/retained/O58', KPLUS='/retained/KPLUS58', A='/retained/A58'),
            evaluation_requests=[], canonicals=str(canonicals)))

    def save_measurements(self, values):
        path = self.root / 'campaign_001/evaluation/final/MEASUREMENTS.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in values))

    def test_failed_qualification_has_no_endpoint_claim_or_main_work(self):
        write_json(self.root / 'qualification/QUALIFICATION.json', dict(status='failed', optimizer_steps=0))
        audit = dict(status='invalid_or_incomplete', errors=['unattempted endpoint'], rows=[], progress={})
        with patch.object(R, 'audit_stage', return_value=audit):
            result = R.build_report(self.root)
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertIsNone(result['endpoint'])
        self.assertEqual(result['qualification']['optimizer_steps'], 0)
        self.assertIn('source_loss_qualification_not_passed', result['errors'])
        text = (self.root / 'results/REPORT.md').read_text()
        self.assertNotIn('Primary decision:', text)
        self.assertIn('a' * 40, text)

    def test_partial_report_preserves_raw_invalid_rows(self):
        write_json(self.root / 'qualification/QUALIFICATION.json', dict(status='passed', optimizer_steps=6))
        progress = dict(status='failed', events_completed=58, error='retained native failure')
        write_json(self.root / 'campaign_001/arms/AREPO/STATUS.json', progress)
        values = rows()[:2]
        values[1].update(valid_measurement=False, validation_status='invalid', errors=['output mismatch'])
        self.save_measurements(values)
        audit = dict(status='invalid_or_incomplete', errors=['output mismatch'], rows=[],
                     progress=dict(invalid_generations=1, unattempted_generations=12286))
        with patch.object(R, 'audit_stage', return_value=audit), \
                patch.object(R, 'training_summary', side_effect=ValueError('partial trajectory')):
            result = R.build_report(self.root)
        self.assertEqual(result['arm_progress'], progress)
        self.assertEqual(result['evaluation_audit']['progress']['unattempted_generations'], 12286)
        self.assertEqual(len((self.root / 'results/ALL_REPEATS.csv').read_text().splitlines()), 3)
        self.assertIn('output mismatch', (self.root / 'results/ALL_REPEATS.csv').read_text())
        self.assertFalse((self.root / 'results/PAIRED_REQUESTS.csv').exists())

    def test_complete_report_records_four_methods_and_both_acceptance_counters(self):
        write_json(self.root / 'qualification/QUALIFICATION.json', dict(status='passed', optimizer_steps=6))
        write_json(self.root / 'qualification/RESULT.json', dict(status='passed', branches=6))
        write_json(self.root / 'campaign_001/STATUS.json', {'status': 'completed'})
        training = dict(worker_wall_s=12., pass_rows=[], event_costs=[])
        values = rows()
        self.save_measurements(values)
        audit = dict(status='passed', errors=[], rows=values, progress={})
        with patch.object(R, 'audit_stage', return_value=audit), \
                patch.object(R, 'training_summary', return_value=training), \
                patch.object(S, 'RESAMPLES', 10):
            result = R.build_report(self.root)
        self.assertEqual(result['status'], 'valid_complete')
        self.assertEqual(result['endpoint']['interpretation']['endpoint_label'], 'favors_KPLUS')
        self.assertEqual(set(result['checkpoints']), {'O', 'KPLUS', 'A', 'AREPO'})
        self.assertEqual(result['qualification']['details']['branches'], 6)
        self.assertEqual(len((self.root / 'results/PAIRED_REQUESTS.csv').read_text().splitlines()), 1025)
        self.assertEqual(len((self.root / 'results/ALL_REPEATS.csv').read_text().splitlines()), 12289)
        text = (self.root / 'results/REPORT.md').read_text()
        self.assertIn('Runtime accepted counter', text)
        self.assertIn('Matched accepted draft tokens', text)
        self.assertIn('nested timers must not be added', text)
        self.assertIn('historical context', text)
        self.assertIn('not independent GPU replication', text)
        self.assertEqual(json.loads((self.root / 'results/SUMMARY.json').read_text())['status'], 'valid_complete')


if __name__ == '__main__':
    unittest.main()
