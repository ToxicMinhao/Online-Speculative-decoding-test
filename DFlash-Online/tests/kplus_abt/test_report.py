import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_abt import report, statistics, contract
from test_campaign import rows, result


class Report(unittest.TestCase):
    def test_partial_report_preserves_all_missing_cases_and_suppresses_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';(root/'sealed').mkdir(parents=True)
            plan=dict(root=str(root),budgets=contract.budgets(),analysis=contract.ANALYSIS,
                binding='a'*64,execution_source={'commit':'b'*40},datasets={d:{} for d in contract.DATASETS})
            (root/'sealed/PLAN.json').write_text(json.dumps(plan))
            output=Path(temp)/'report'
            summary=report.build(root,output)
            self.assertEqual(summary['status'],'partial_or_incomplete')
            self.assertEqual(summary['valid_arm_events'],0)
            self.assertTrue(all(not row['inference_permitted'] for row in summary['endpoints'].values()))
            self.assertEqual(len(summary['expected_tasks']),22)
            qualification=json.loads((output/'QUALIFICATION.json').read_text())
            self.assertEqual(len(qualification['cases']),12)
            self.assertTrue(all(row['status']=='no_receipt' for row in qualification['cases']))
            text=(output/'REPORT.md').read_text()
            self.assertIn('No complete valid dataset',text)
            self.assertNotIn('0.0000 [0.0000',text)
            self.assertGreater(len(json.loads((output/'INPUT_MANIFEST.json').read_text())['missing']),16)

    def test_incomplete_fourth_arm_does_not_reach_endpoint_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence=report.Evidence(temp,{'root':temp})
            valid={(d,e,a):True for d in contract.DATASETS for e in contract.EVENTS for a in contract.ARMS}
            valid['gsm8k',6,'ADV']=False
            with patch.object(report,'audit_stage') as audit:
                receipt,paired,repeats=report.endpoint(evidence,'gsm8k',{},valid,{'status':'passed'})
            audit.assert_not_called()
            self.assertFalse(receipt['inference_permitted'])
            self.assertEqual((paired,repeats),([],[]))

    def test_failed_endpoint_keeps_partial_repeats_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'campaign_001/evaluation/gsm8k';folder.mkdir(parents=True)
            observed=dict(request_id='r0',method='ADV',repeat=0,validation_status='invalid',errors=['fixture'])
            (folder/'MEASUREMENTS.jsonl').write_text(json.dumps(observed)+'\n')
            evidence=report.Evidence(root,{'root':str(root)})
            receipt,paired,repeats=report.endpoint(evidence,'gsm8k',{}, {},None)
            self.assertFalse(receipt['inference_permitted'])
            self.assertEqual(paired,[])
            self.assertEqual(repeats,[observed])

    def test_input_changed_between_reads_remains_detectable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'receipt.json';path.write_text('{"status":"running"}')
            evidence=report.Evidence(root,{'root':str(root)})
            evidence.read(path)
            path.write_text('{"status":"completed"}')
            evidence.read(path)
            self.assertEqual(evidence.stability()['status'],'changed_during_analysis')
            self.assertTrue(evidence.errors)

    def test_contrasts_use_summed_calls_not_mean_request_ratios(self):
        sample=rows()
        for row in sample:
            if row['panel_ordinal']==0 and row['method']=='KPLUS':
                row.update(verification_calls=50,accepted_draft_tokens=14,native_accepted_draft_tokens=14)
        summary=statistics.summarize(sample,dataset='gsm8k',event=6)
        contrasts=report.contrast_rows({'gsm8k':{'status':'passed','summary':summary}})
        main=next(row for row in contrasts if row['contrast']=='ADV_vs_KPLUS')
        self.assertAlmostEqual(main['calls_ratio'],2048/2600)
        mean_request_ratio=(8/50+255*.8)/256
        self.assertNotAlmostEqual(main['calls_ratio'],mean_request_ratio)
        self.assertEqual(main['call_wins'],256)
        self.assertEqual(len(contrasts),3)

    def test_nested_radius_metrics_and_nonfinite_rejection(self):
        observed=report.radius_summary([[0.,2.],[1.,1.]])
        self.assertEqual(observed,dict(count=4,mean=1.,minimum=0.,maximum=2.,total=4.))
        with self.assertRaises(ValueError): report.radius_summary([[float('nan')]])

    def test_intrinsic_training_subtracts_observer_once_and_keeps_startup_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'campaign_001/gsm8k/event_005/KPLUS';base.mkdir(parents=True)
            value=result();value['costs']=dict(training_s=20.,checkpoint_save_s=3.,export_s=4.)
            for row in value['diagnostic_windows']: row.update(wall_s=1.,rows=[])
            value.update(boundary=dict(status='passed',export_exact=True),
                         state_identity={'sha256':'a'*64,'bytes':1},weights_identity={'sha256':'b'*64,'bytes':1})
            (base/'RESULT.json').write_text(json.dumps(value))
            (base/'STATUS.json').write_text(json.dumps(dict(status='completed',wall_s=40.,loading_s=5.,startup={'wall_s':6.})))
            evidence=report.Evidence(root,{'root':str(root)})
            _,_,costs,_,_,_,_=report.training(evidence)
            intrinsic=[r for r in costs if r['component']=='intrinsic_training_s']
            self.assertEqual(intrinsic[0]['value'],16.)
            self.assertEqual(next(r['value'] for r in costs if r['scope']=='discarded_startup'),6.)


if __name__=='__main__': unittest.main()
