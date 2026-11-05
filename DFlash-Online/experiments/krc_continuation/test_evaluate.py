"""CPU K2/KRC2 evaluation contracts, historical controls and panel separation."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.reference_replay_event.test_evaluate import Backend, panel as old_panel
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate
from .evaluation_protocol import method_order, validate_plan
from .report import summarize


def panel(name='development',count=64):
    value=old_panel(name,count)
    counts=value.pop('retained_krc_counts')
    value['expected_control_counts']=counts if name=='development' else None
    value['historical_counts']={'K1':{key:4 for key in counts},'KRC1':{key:3 for key in counts}}
    return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.panel=panel(count=3);self.checkpoints={'K2':'/K2','KRC2':'/KRC2'}

    def evaluate(self,mutation=None,name='evaluation',gate=1):
        p=self.panel
        return evaluate(Backend(p['requests'],mutation),p['requests'],self.checkpoints,self.root/name,
            expected_control_counts=p['expected_control_counts'],historical_counts=p['historical_counts'],
            qualification_requests=gate)

    def read(self,path):return json.loads((self.root/path).read_text())

    def test_dev_exact384_budget_balanced_order_and48included_gate(self):
        self.panel=panel();p=self.panel;backend=Backend(p['requests'])
        result=evaluate(backend,p['requests'],self.checkpoints,self.root/'evaluation',
            expected_control_counts=p['expected_control_counts'],historical_counts=p['historical_counts'])
        self.assertEqual(result['completed_valid_generations'],384)
        self.assertEqual(result['unattempted_generations'],0)
        self.assertEqual(self.read('evaluation/QUALIFICATION.json')['generations'],48)
        self.assertEqual([x[0][1:] for x in backend.executions],
                         [arm for i in range(64) for r in range(3) for arm in method_order(i,r)])
        self.assertEqual(backend.weight_checks,['/K2','/KRC2'])
        self.assertEqual(len(backend.switches),193)
        self.assertTrue(self.read('evaluation/PROTOCOL.json')['historical_K2_count_gate'])

    def test_additional_K2_does_not_gate_against_K1_counts(self):
        self.panel=panel('additional',3)
        def mutation(record,options):
            if record['checkpoint_id']=='/K2':harmful(record)
        result=self.evaluate(mutation,gate=0)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(self.read('evaluation/PROTOCOL.json')['historical_K2_count_gate'])
        self.assertEqual(result['qualification'],'not_required_prior_panel_gate')
        self.assertTrue(all(v['source_method']=='K1' for v in self.read('evaluation/CANONICAL.json').values()))
        summary=self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['K2']['verification_calls'],9)
        self.assertEqual(summary['historical_counts']['K1']['verification_calls'],12)
        self.assertEqual(summary['historical_counts']['KRC1']['verification_calls'],9)
        self.assertEqual(summary['count_learning_curve']['K2_vs_K1']['call_reduction_pct'],25)

    def test_dev_K2_count_mismatch_stops_and_keeps_raw(self):
        def mutation(record,options):
            if record['checkpoint_id']=='/K2':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'retained_K2_verification_calls'):self.evaluate(mutation)
        result=self.read('evaluation/STATUS.json')
        self.assertEqual((result['returned_generations'],result['invalid_generations'],result['unattempted_generations']),(1,1,17))
        self.assertEqual(self.read('evaluation/0000_r0_K2.json')['metadata']['verification_calls'],3)

    def test_harmful_KRC2_continues_and_history_is_count_only(self):
        def mutation(record,options):
            if record['checkpoint_id']=='/KRC2':harmful(record)
        self.assertEqual(self.evaluate(mutation)['status'],'completed')
        s=self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(s['comparisons']['KRC2_vs_K2']['call_reduction_pct'],-50)
        self.assertEqual(s['count_learning_curve']['KRC1_vs_K1']['call_reduction_pct'],25)
        self.assertEqual(s['count_learning_curve']['KRC2_vs_KRC1']['call_reduction_pct'],0)
        self.assertEqual(s['relative_advantage_change_percentage_points'],-75)
        self.assertFalse(any('latency' in key for rows in s['count_learning_curve'].values() for key in rows))

    def test_canonical_and_repeat_gates_preserve_invalid_and_unknown(self):
        def wrong(record,options):
            record['response_ids'][1]=777;record['metadata']['trace'][0]['committed_tokens'][0]=777
        with self.assertRaisesRegex(RuntimeError,'reference_response_ids'):self.evaluate(wrong)
        self.assertEqual(self.read('evaluation/CANONICAL.json')['development_0']['response_ids'][1],11)
        def repeat(record,options):
            if options['request_id']=='0000_r1_KRC2':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat_count_mismatch'):self.evaluate(repeat,name='repeat')
        self.assertEqual(self.read('repeat/STATUS.json')['invalid_generations'],2)
        def unknown(record,options):raise RuntimeError('lost reply')
        with self.assertRaisesRegex(RuntimeError,'lost reply'):self.evaluate(unknown,name='unknown')
        self.assertEqual(self.read('unknown/STATUS.json')['unknown_generation_attempts'],1)
        self.assertEqual(self.read('unknown/STATUS.json')['unattempted_generations'],17)

    def test_nested_timings_and_report_revalidation(self):
        def mutation(record,options):
            i=options['request_index'];r=int(options['request_id'].split('_')[1][1:])
            latency={'/K2':(1.,9.,10.),'/KRC2':(1.,3.,10.)}[record['checkpoint_id']][i]*(.5,1.,1.5)[r]
            record['metadata']['request_latency_s']=latency
            record['metadata']['native_generate_s']=.9*latency
            record['metadata']['capture']['cache_maintenance_s']=.1*latency
        self.evaluate(mutation)
        s=self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(s['KRC2']['summed_mean_request_latency_s'],14)
        self.assertAlmostEqual(s['KRC2']['summed_mean_native_generate_s'],12.6)
        self.assertAlmostEqual(s['KRC2']['summed_mean_cache_maintenance_s'],1.4)
        self.assertAlmostEqual(s['comparisons']['KRC2_vs_K2']['request_time_reduction_pct'],30)
        rows=list(csv.DictReader((self.root/'evaluation/report/REPEAT_TOTALS.csv').read_text().splitlines()))
        self.assertEqual(len(rows),6);self.assertIn('native_generate_s',rows[0])
        rows=[json.loads(x) for x in (self.root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        rows[0]['accepted_draft_tokens']=999
        s,requests,*_=summarize(rows,self.panel['requests'],checkpoints=self.checkpoints,
            expected_control_counts=self.panel['expected_control_counts'],historical_counts=self.panel['historical_counts'])
        self.assertEqual(s['valid_requests'],2)
        self.assertEqual(s['historical_counts']['K1']['requests'],2)
        self.assertTrue(requests[0]['errors'])

    def test_plan_requires_dev_gate_and_no_additional_gate(self):
        keys=('config','startup_ordinary_receipt','pinned_initial','krc1_state','krc1_receipt',
              'event2_ordinary_receipt','event2_k_receipt','reference_schedule','retained_k2_checkpoint',
              'target','draft','worker_python')
        plan={key:'path-'+key for key in keys};plan['panels']=[panel(),panel('additional')]
        self.assertEqual(validate_plan(plan)['planned_generations'],768)
        for index,value in ((0,None),(1,plan['panels'][1]['historical_counts']['K1'])):
            changed=copy.deepcopy(plan);changed['panels'][index]['expected_control_counts']=value
            with self.assertRaisesRegex(ValueError,'Only development'):validate_plan(changed)
        changed=copy.deepcopy(plan);changed['panels'][1]['requests'][0]['canonical_source_method']='F'
        with self.assertRaisesRegex(ValueError,'canonical'):validate_plan(changed)


if __name__=='__main__':unittest.main()
