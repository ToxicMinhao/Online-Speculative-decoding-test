"""CPU OFF/KRC2 evaluation contracts, historical controls and panel separation."""
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
    value['expected_control_counts']=counts
    value['historical_counts']={'K2':{key:4 for key in counts},'KRC1':{key:3 for key in counts}}
    return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.panel=panel(count=3);self.checkpoints={'KRC2_OFF':'/KRC2_OFF','KRC2':'/KRC2'}

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
        self.assertEqual(backend.weight_checks,['/KRC2_OFF','/KRC2'])
        self.assertEqual(len(backend.switches),193)
        self.assertTrue(self.read('evaluation/PROTOCOL.json')['retained_KRC2_count_gate'])

    def test_both_panels_require_retained_KRC2_count_reproduction(self):
        for name in ('development', 'additional'):
            self.panel=panel(name, 3)
            def mutation(record, options):
                if record['checkpoint_id']=='/KRC2': harmful(record)
            with self.assertRaisesRegex(RuntimeError, 'retained_KRC2_verification_calls'):
                self.evaluate(mutation, name=name, gate=0)
            result=self.read(name+'/STATUS.json')
            self.assertEqual((result['returned_generations'],result['invalid_generations'],result['unattempted_generations']), (2,1,16))
            self.assertEqual(self.read(name+'/0000_r0_KRC2.json')['metadata']['verification_calls'],3)

    def test_harmful_OFF_continues_with_correct_primary_orientation(self):
        def mutation(record,options):
            if record['checkpoint_id']=='/KRC2_OFF': harmful(record)
        self.assertEqual(self.evaluate(mutation)['status'],'completed')
        s=self.read('evaluation/report/SUMMARY.json')
        self.assertAlmostEqual(s['comparisons']['KRC2_vs_KRC2_OFF']['call_reduction_pct'],100/3)
        self.assertEqual(s['comparisons']['KRC2_vs_KRC2_OFF']['call_request_directions'],dict(improved=3,regressed=0,tied=0))
        self.assertEqual(s['count_learning_curve']['KRC2_OFF_vs_KRC1']['call_reduction_pct'],0)
        self.assertEqual(s['count_learning_curve']['KRC2_OFF_vs_K2']['call_reduction_pct'],25)
        self.assertFalse(any('latency' in key for rows in s['count_learning_curve'].values() for key in rows))

    def test_canonical_and_repeat_gates_preserve_invalid_and_unknown(self):
        def wrong(record,options):
            record['response_ids'][1]=777;record['metadata']['trace'][0]['committed_tokens'][0]=777
        with self.assertRaisesRegex(RuntimeError,'reference_response_ids'):self.evaluate(wrong)
        self.assertEqual(self.read('evaluation/CANONICAL.json')['development_0']['response_ids'][1],11)
        def repeat(record,options):
            if options['request_id']=='0000_r1_KRC2_OFF':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat_count_mismatch'):self.evaluate(repeat,name='repeat')
        self.assertEqual(self.read('repeat/STATUS.json')['invalid_generations'],2)
        def unknown(record,options):raise RuntimeError('lost reply')
        with self.assertRaisesRegex(RuntimeError,'lost reply'):self.evaluate(unknown,name='unknown')
        self.assertEqual(self.read('unknown/STATUS.json')['unknown_generation_attempts'],1)
        self.assertEqual(self.read('unknown/STATUS.json')['unattempted_generations'],17)

    def test_nested_timings_and_report_revalidation(self):
        def mutation(record,options):
            i=options['request_index'];r=int(options['request_id'].split('_')[1][1:])
            latency={'/KRC2_OFF':(1.,9.,10.),'/KRC2':(1.,3.,10.)}[record['checkpoint_id']][i]*(.5,1.,1.5)[r]
            record['metadata']['request_latency_s']=latency
            record['metadata']['native_generate_s']=.9*latency
            record['metadata']['capture']['cache_maintenance_s']=.1*latency
        self.evaluate(mutation)
        s=self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(s['KRC2']['summed_mean_request_latency_s'],14)
        self.assertAlmostEqual(s['KRC2']['summed_mean_native_generate_s'],12.6)
        self.assertAlmostEqual(s['KRC2']['summed_mean_cache_maintenance_s'],1.4)
        self.assertAlmostEqual(s['comparisons']['KRC2_vs_KRC2_OFF']['request_time_reduction_pct'],30)
        rows=list(csv.DictReader((self.root/'evaluation/report/REPEAT_TOTALS.csv').read_text().splitlines()))
        self.assertEqual(len(rows),6);self.assertIn('native_generate_s',rows[0])
        rows=[json.loads(x) for x in (self.root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        rows[0]['accepted_draft_tokens']=999
        s,requests,*_=summarize(rows,self.panel['requests'],checkpoints=self.checkpoints,
            expected_control_counts=self.panel['expected_control_counts'],historical_counts=self.panel['historical_counts'])
        self.assertEqual(s['valid_requests'],2)
        self.assertEqual(s['historical_counts']['KRC1']['requests'],2)
        self.assertTrue(requests[0]['errors'])

    def test_plan_requires_KRC2_counts_both_panels_and_exact_panel_canonicals(self):
        keys=('config','startup_ordinary_receipt','pinned_initial','krc1_state','krc1_receipt',
              'event2_ordinary_receipt','event2_k_receipt','presentation_schedule','krc2_receipt',
              'retained_krc2_checkpoint','target','draft','worker_python')
        plan={key:'path-'+key for key in keys};plan['panels']=[panel(),panel('additional')]
        self.assertEqual(validate_plan(plan)['planned_generations'],768)
        for index in (0,1):
            changed=copy.deepcopy(plan);changed['panels'][index]['expected_control_counts']=None
            with self.assertRaisesRegex(ValueError,'Both panels'):validate_plan(changed)
        changed=copy.deepcopy(plan);changed['panels'][1]['requests'][0]['canonical_source_method']='F'
        with self.assertRaisesRegex(ValueError,'canonical'):validate_plan(changed)


if __name__=='__main__':unittest.main()
