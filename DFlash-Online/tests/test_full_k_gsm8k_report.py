"""Report integration with tiny native fixtures and externally audited training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.full_k_gsm8k import evaluation as E, statistics as S, report as R
from experiments.full_k_gsm8k.contract import NEW_BUDGET
from test_full_k_gsm8k_evaluation import FullBackend, requests


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = requests(8)
        self.checkpoints = dict(F='/F',O='/O58',**{arm:str(self.root/'campaign_001/arms'/arm/
            'trainer/events/event_058/export') for arm in E.ARMS})
        self.canonicals = {r['request_id']:dict(response_ids=[10,11,12,13,151645],finish_reason='eos',
            source_record_path='/original/'+r['request_id']+'.json',F_checkpoint='/F',F_counts=dict(
                verification_calls=2,accepted_draft_tokens=2,committed_progress=4,output_tokens=5)) for r in self.inputs}
        self.plan = dict(execution_source={'commit':'a'*40,'dirty':False},
                         checkpoints=self.checkpoints,evaluation_requests=self.inputs)
        write_json(self.root/'sealed/PLAN.json',self.plan)
        write_json(self.root/'sealed/CANONICALS.json',self.canonicals)
        self.training = dict(status='passed',errors=[],budget=dict(NEW_BUDGET),arms={})
        for arm in E.ARMS:
            self.training['arms'][arm] = dict(events_completed=58,optimizer_steps=7424,
                response_presentations=14848,online_generations=7473,terminal_preparation_forwards=7424,
                correctness_generations=4,intermediate_generations=384,actual_work={'fixture':True},
                costs={'training_s':1.,'blocking_update_s':2.,'controller_serving_s':3.,'loading_s':.1,
                       'online_wall_s':6.},exposure={'sampled_anchors':4},final_checkpoint=self.checkpoints[arm])
        self.addCleanup(patch.stopall)
        patch.multiple(E,PANEL_SIZE=8,REQUEST_BLOCK=2,INTERMEDIATE_SIZE=4).start()
        patch.multiple(R,PANEL_SIZE=8,INTERMEDIATE_SIZE=4).start()
        patch.object(R,'validate_plan',return_value={'status':'passed'}).start()
        patch.object(S,'RESAMPLES',100).start()

    def execute_fixture(self):
        for arm in E.ARMS:
            for event in E.INTERMEDIATE_EVENTS:
                root = self.root/'campaign_001/arms'/arm
                checkpoint = str(root/'trainer/events'/f'event_{event:03d}'/'export')
                E.evaluate_intermediate(FullBackend(self.inputs),self.inputs,checkpoint,
                    root/'evaluation'/f'event_{event:03d}',arm=arm,event=event,canonicals=self.canonicals)
        E.evaluate_final(FullBackend(self.inputs),self.inputs,self.checkpoints,
            self.root/'campaign_001/evaluation/final',canonicals=self.canonicals)

    def test_complete_report_reconstructs_all_rows_and_fixed_curve(self):
        self.execute_fixture()
        write_json(self.root/'campaign_001/FINAL_LOADING.json',{'native_loading_s':3.5})
        write_json(self.root/'campaign_001/STATUS.json',{'status':'completed','campaign_wall_s':100.})
        result = R.write_campaign_report(self.root,training_audit=self.training)
        self.assertEqual(result['status'],'valid')
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['analysis']['request_count'],8)
        self.assertEqual(len(result['learning_curve']),8)
        self.assertEqual(result['evaluation_accounting']['totals']['completed_valid_generations'],120)
        self.assertEqual([r['event'] for r in result['learning_curve']],[14,29,44,58]*2)
        report = (self.root/'final_report/REPORT.md').read_text()
        self.assertIn('inconclusive endpoint difference',report.lower())
        self.assertIn('different learning recipes',report)
        self.assertIn('same-LR',report)
        self.assertIn('not evidence of faster learning than O',report)
        self.assertIn('a'*40,report)
        self.assertIn('Verification Status: ANALYZED',report)
        self.assertIn('not an independent GPU reproduction',report)
        self.assertIn('not whole-system memory peaks',report)
        self.assertEqual(result['costs']['final_loading']['native_loading_s'],3.5)
        self.assertEqual(result['costs']['campaign']['campaign_wall_s'],100.)
        self.assertTrue((self.root/'final_report/RESOURCES.json').exists())

    def test_resource_summary_keeps_nested_compiler_names_and_memory_scope(self):
        source = self.training['arms']['K']
        source.update(peak_allocated_bytes=10*2**30,peak_reserved_bytes=12*2**30,
                      unavailable_costs=['serving-worker compilation'])
        source['resource_records'] = [
            {'event':1,'training_compiler_timers':{
                'outer':{'calls':1,'total_s':8.,'durations_s':[8.]},
                'inner':{'calls':2,'total_s':5.,'durations_s':[2.,3.]}}},
            {'event':2,'training_compiler_timers':{
                'outer':{'calls':1,'total_s':1.,'durations_s':[1.]}}}]
        resources = R.resource_summary(self.training)
        self.assertEqual(resources['K']['compiler_by_timer_name'],
                         {'outer':{'calls':2,'total_s':9.},'inner':{'calls':2,'total_s':5.}})
        self.assertNotIn('total_compilation_s',resources['K'])
        self.assertEqual(resources['K']['event_records'],source['resource_records'])
        self.assertIn('separate serving worker',resources['K']['memory_scope'])
        self.assertIsNone(resources['K_HALF']['peak_allocated_bytes'])

    def test_missing_cells_preserve_unattempted_and_issue_no_effect(self):
        result = R.write_campaign_report(self.root,training_audit=self.training)
        self.assertEqual(result['status'],'invalid_or_incomplete')
        self.assertIsNone(result['analysis'])
        self.assertEqual(result['evaluation_accounting']['totals']['unattempted_generations'],120)
        self.assertIn('no efficacy conclusion',(self.root/'final_report/REPORT.md').read_text())

    def test_incomplete_training_rejects_otherwise_complete_evaluation(self):
        self.execute_fixture()
        bad = deepcopy(self.training)
        bad['arms']['K_HALF']['optimizer_steps'] -= 1
        result = R.write_campaign_report(self.root,training_audit=bad)
        self.assertEqual(result['status'],'invalid_or_incomplete')
        self.assertIsNone(result['analysis'])
        self.assertTrue(any('K_HALF' in error for error in result['errors']))

    def test_report_does_not_trust_saved_validity_for_native_error(self):
        self.execute_fixture()
        path = self.root/'campaign_001/evaluation/final/0000_r0_F.json'
        value = json.loads(path.read_text())
        value['metadata']['trace'][0]['draft_tokens'][0] += 101
        write_json(path,value)
        result = R.write_campaign_report(self.root,training_audit=self.training)
        self.assertEqual(result['status'],'invalid_or_incomplete')
        self.assertIsNone(result['analysis'])


if __name__ == '__main__':
    unittest.main()
