"""Partial reports preserve all trajectory cells and returned invalid evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.role_loss_pilot.report import build_report
from online_feedback.storage import write_json


class PartialReportTests(unittest.TestCase):
    def test_missing_campaign_reports_six_cells_and_partial_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            datasets={}
            for dataset in ('gsm8k','mathqa'):
                panel=root/f'{dataset}_panel.json';refs=root/f'{dataset}_refs.json'
                write_json(panel,dict(requests=[]));write_json(refs,{})
                datasets[dataset]=dict(panel_manifest=str(panel),references=str(refs))
            write_json(root/'sealed/PLAN.json',dict(datasets=datasets,execution_source={'commit':'a'*40},binding='b'*64))
            folder=root/'campaign_001/evaluation/gsm8k/event_002'
            folder.mkdir(parents=True)
            (folder/'MEASUREMENTS.jsonl').write_text(json.dumps(dict(request_id='invalid_request',validation_status='invalid'))+'\n')
            result=build_report(root)
            self.assertEqual(result['status'],'invalid_or_incomplete')
            self.assertEqual(len(result['training']),6)
            self.assertEqual(len(result['diagnostics']),6)
            self.assertEqual(len(result['audits']),4)
            self.assertEqual(result['recommendation']['outcome'],'invalid_or_incomplete')
            self.assertIn('invalid_request',(root/'results/ALL_REPEATS.csv').read_text())
            self.assertIn('No complete efficacy conclusion',(root/'results/REPORT.md').read_text())


if __name__=='__main__':unittest.main()
