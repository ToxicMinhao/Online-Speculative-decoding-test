"""CPU-only tests for the fixed conditional pilot controller and native adapter."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.functional_step_geometry import evaluation as E, statistics as S, pilot as P
from experiments.functional_step_geometry.contract import MAIN_BUDGET, DIAGNOSTIC_BUDGET
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import FullBackend, requests


def presentations():
    return [dict(record_id=f'train_{index%128:05d}', **{'pass':index//128},
                 sampled_anchors=[dict(positions=[[index%128]], keep_mask=[[True]])])
            for index in range(256)]


def training_result(dataset, arm, event):
    value = dict(status='completed', method=arm, dataset=dataset, event_id=f'event_{event:03d}',
        start_step=(event-1)*128, end_step=event*128, optimizer_steps=128,
        response_presentations=256, passes=2, records=presentations())
    if arm == 'FSG':
        value['diagnostic_windows'] = [dict(dataset=dataset,event=event,window=window,
            status='completed', restoration_exact=True, response_forwards=40,backwards=4)
            for window in (1,32,65,128)]
    return value


def valid_rows():
    rows = []
    for index in range(256):
        for repeat in range(2):
            for method in E.METHODS:
                calls = 10+index%7-(method=='FSG')
                rows.append(dict(seed=17,dataset='gsm8k',event=6,phase='final',
                    panel_ordinal=index,request_id=f'eval{index}',repeat=repeat,method=method,
                    verification_calls=calls,accepted_draft_tokens=50,
                    native_accepted_draft_tokens=99-calls,committed_progress=99,output_tokens=100,
                    checkpoint_id='/'+method,finish_reason='eos',cap_hit=False,
                    validation_status='valid',valid_measurement=True,errors=[],
                    request_latency_s=calls+repeat,native_generate_s=calls-.1+repeat,cache_maintenance_s=.1))
    return rows


class SchedulingTests(unittest.TestCase):
    def test_fixed_work_and_causal_collection(self):
        tasks = P.tasks()
        self.assertEqual(len(tasks),18)
        self.assertEqual(sum(t['kind']=='train_arm' for t in tasks),12)
        self.assertEqual(sum(t['kind']=='collect_shared' for t in tasks),4)
        for dataset in E.DATASETS:
            selected = [task for task in tasks if task['dataset']==dataset]
            self.assertEqual(selected[0],dict(kind='collect_shared',dataset=dataset,event=5,
                producer='KPLUS',producer_event=4))
            self.assertEqual([t.get('arm') for t in selected[1:4]],list(E.METHODS))
            self.assertEqual(selected[4]['producer_event'],5)
            self.assertEqual(selected[-1]['kind'],'evaluate_dataset')

    def test_diagnostic_windows_wrap_inside_the_original_pass(self):
        rows=P.diagnostic_pairs(presentations(),dataset='gsm8k',event=5)
        self.assertEqual([row['window'] for row in rows],[1,32,65,128])
        self.assertEqual(rows[0]['fitting_ids'],['train_00000','train_00001'])
        self.assertEqual(rows[0]['off_fit_ids'],['train_00064','train_00065'])
        self.assertEqual(rows[-1]['fitting_ids'],['train_00126','train_00127'])
        self.assertEqual(rows[-1]['off_fit_ids'],['train_00062','train_00063'])
        self.assertEqual(rows[-1]['pass_index'],1)
        self.assertEqual(len({tuple(row['diagnostic_rng_key']) for row in rows}),4)

    def test_diagnostic_pairing_rejects_changed_pass_or_duplicate_ids(self):
        for mode in ('pass', 'duplicate', 'pool'):
            rows=presentations()
            if mode=='pass':rows[128]['pass']=0
            elif mode=='duplicate':rows[1]['record_id']=rows[0]['record_id']
            else:rows[128]['record_id']='foreign'
            with self.assertRaises(ValueError):
                P.diagnostic_pairs(rows,dataset='gsm8k',event=5)

    def test_exact_native_anchor_and_diagnostic_accounting(self):
        baseline=training_result('gsm8k','KPLUS',5)
        result=training_result('gsm8k','FSG',5)
        P.validate_update(result,dataset='gsm8k',arm='FSG',event=5,baseline=baseline)
        result['records'][7]['sampled_anchors'][0]['positions'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'anchors differ'):
            P.validate_update(result,dataset='gsm8k',arm='FSG',event=5,baseline=baseline)
        for change in ('missing','over_budget','restoration'):
            result=training_result('gsm8k','FSG',5)
            if change=='missing':result['diagnostic_windows'].pop()
            elif change=='over_budget':result['diagnostic_windows'][0]['response_forwards']=41
            else:result['diagnostic_windows'][0]['restoration_exact']=False
            with self.assertRaises(ValueError):
                P.validate_update(result,dataset='gsm8k',arm='FSG',event=5,baseline=baseline)

    def test_balanced_per_request_orders_are_presealed(self):
        total=0
        for dataset in E.DATASETS:
            slots=E.evaluation_schedule(requests(256),dataset)
            self.assertEqual(slots,E.evaluation_schedule(requests(256),dataset))
            total+=len(slots)
            balance=E.schedule_balance(slots)
            self.assertEqual(set(balance['pair_precedence'].values()),{256})
            for values in balance['positions'].values():self.assertLessEqual(max(values)-min(values),2)
            for index in range(256):
                orders=[[r['method'] for r in slots if r['repeat']==repeat and r['panel_ordinal']==index]
                        for repeat in (0,1)]
                self.assertEqual(orders[1],orders[0][::-1])
        self.assertEqual(total,3072)
        with self.assertRaises(ValueError):E.evaluation_schedule(requests(256),'gsm8k',4)


class AdmissionAndControllerTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.plan=dict(execution_source=dict(commit='a'*40),binding='b'*64,
            realization_revision='norm_feasible_bf16_r1',
            main_budget=MAIN_BUDGET,diagnostic_budget=DIAGNOSTIC_BUDGET,
            startup_budget=dict(response_forwards=24,backwards=24,optimizer_steps=0),
            pilot_admission=str(self.root/'gate.json'),
            diagnostic_factory='experiments.functional_step_geometry.pilot:diagnostic_pairs',
            main_storage=dict(required_initial_free_bytes=1,no_future_deletion_credit=True))
        self.receipt=dict(status='passed',pilot_permitted=True,execution_commit='a'*40,binding='b'*64,
            checks={name:True for name in P.REQUIRED_GATES},nontrivial_accepted_fsg_cases=1,
            scalar_matching_exercised_cases=1,realization_revision='norm_feasible_bf16_r1')
        write_json(self.root/'gate.json',self.receipt)
        write_json(self.root/'sealed/PLAN.json',self.plan)

    def test_admission_rejects_all_fallback_and_missing_checks(self):
        P.require_pilot_admission(self.plan,self.root)
        for field in P.REQUIRED_GATES:
            changed=deepcopy(self.receipt);changed['checks'][field]=False
            write_json(self.root/'gate.json',changed)
            with self.assertRaises(ValueError):P.require_pilot_admission(self.plan,self.root)
        changed=deepcopy(self.receipt);changed['nontrivial_accepted_fsg_cases']=0
        write_json(self.root/'gate.json',changed)
        with self.assertRaises(ValueError):P.require_pilot_admission(self.plan,self.root)

    def test_missing_diagnostics_and_low_storage_block_before_work(self):
        changed=deepcopy(self.plan);changed.pop('diagnostic_factory')
        with self.assertRaisesRegex(ValueError,'not yet bound'):P.require_pilot_admission(changed,self.root)
        changed=deepcopy(self.plan);changed['main_storage']['required_initial_free_bytes']=10**30
        with self.assertRaisesRegex(ValueError,'storage'):P.require_pilot_admission(changed,self.root)

    def test_complete_controller_keeps_shared_pool_and_own_state(self):
        observed=[];results={}
        def collect(plan,dataset,event,producer):
            observed.append(('collect',dataset,event))
            self.assertEqual(producer,results.get((dataset,event-1,'KPLUS')))
            return dict(dataset=dataset,event=event,producer=f'KPLUS{event-1}',records=[{}]*128)
        def train(plan,dataset,event,arm,pool,previous):
            observed.append(('train',dataset,event,arm))
            self.assertEqual(previous,results.get((dataset,event-1,arm)))
            value=training_result(dataset,arm,event);results[dataset,event,arm]=value;return value
        def evaluate(plan,dataset,final):
            observed.append(('evaluate',dataset))
            self.assertEqual(set(final),set(E.METHODS))
            return dict(status='passed',rows=[{}]*1536)
        with patch.object(P,'verify_shared_packets',return_value={'status':'passed'}) as verify:
            result=P.run(self.root,collect_shared=collect,train_arm=train,evaluate_dataset=evaluate)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(observed),18)
        self.assertEqual(verify.call_count,28)
        self.assertEqual(result['unattempted_tasks'],0)

    def test_failed_operation_is_not_retried(self):
        called=[]
        def fail(*args):called.append(1);raise RuntimeError('retained evidence failure')
        with self.assertRaises(RuntimeError):
            P.run(self.root,collect_shared=fail,train_arm=fail,evaluate_dataset=fail)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(called,[1]);self.assertEqual(result['unattempted_tasks'],17)
        self.assertEqual(result['attempted_task']['kind'],'collect_shared')

    def test_storage_uses_measured_sizes_and_no_retirement_credit(self):
        budget=P.storage_estimate(2097259104,6297245019)
        self.assertTrue(budget['no_future_deletion_credit'])
        self.assertEqual(budget['retained_new_boundaries'],12)
        self.assertEqual(budget['required_initial_free_bytes'],286367339007)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.inputs=requests(4);self.paths={method:'/'+method for method in E.METHODS}
        self.refs={row['request_id']:dict(response_ids=[10,11,12,13,151645],finish_reason='eos',
            verified=True,binding_digest='retained',prompt_ids=row['prompt_ids']) for row in self.inputs}

    def test_native_raw_audit_and_exact_outputs(self):
        folder=self.root/'valid';backend=FullBackend(self.inputs)
        result=E.evaluate_stage(backend,self.inputs,self.paths,folder,canonicals=self.refs,
            dataset='gsm8k',event=6,production=False)
        self.assertEqual(result['completed_valid_generations'],24)
        audit=E.audit_stage(folder,self.inputs,self.paths,canonicals=self.refs,dataset='gsm8k',event=6,production=False)
        self.assertEqual(audit['errors'],[])
        f=folder/'0000_r0_KPLUS.json';row=json.loads(f.read_text());row['response_ids'][0]=999;write_json(f,row)
        self.assertEqual(E.audit_stage(folder,self.inputs,self.paths,canonicals=self.refs,
            dataset='gsm8k',event=6,production=False)['status'],'invalid_or_incomplete')

    def test_bad_and_unknown_records_preserve_unattempted_work(self):
        for mode in ('bad','unknown'):
            folder=self.root/mode
            def mutate(record,options):record['response_ids'][0]=999
            backend=FullBackend(self.inputs,mutate if mode=='bad' else None)
            if mode=='unknown':backend.generate=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('worker died'))
            with self.assertRaises(RuntimeError):E.evaluate_stage(backend,self.inputs,self.paths,folder,
                canonicals=self.refs,dataset='gsm8k',event=6,production=False)
            status=json.loads((folder/'STATUS.json').read_text())
            self.assertEqual(status['unattempted_generations'],23)
            self.assertEqual(status['unknown_generation_attempts'],int(mode=='unknown'))
            self.assertEqual(status['invalid_generations'],int(mode=='bad'))

    def test_ratio_of_sums_bootstrap_uses_requests_not_repeats(self):
        values=valid_rows();summary=S.summarize(values,dataset='gsm8k',event=6)
        before=np.asarray([10+i%7 for i in range(256)])
        indices=np.random.Generator(np.random.PCG64(20261108)).integers(0,256,(10000,256))
        oracle=100*(1-(before-1)[indices].sum(1)/before[indices].sum(1))
        self.assertEqual(summary['comparisons']['FSG_vs_KPLUS']['call_reduction']['ci95_pct'],
                         np.quantile(oracle,[.025,.975],method='linear').tolist())
        self.assertEqual(summary['method_totals']['KPLUS']['verification_calls'],int(before.sum()))
        self.assertEqual(summary['method_totals']['KPLUS']['request_latency_s'],float(before.sum()+128))
        self.assertEqual(summary['primary'],['FSG_vs_KPLUS','FSG_vs_SCALAR'])
        self.assertEqual(set(summary['comparisons']),{'FSG_vs_KPLUS','FSG_vs_SCALAR','SCALAR_vs_KPLUS'})

    def test_invalid_repeats_are_not_summarized(self):
        for kind in ('missing','repeat','dataset','calls'):
            rows=valid_rows()
            if kind=='missing':rows.pop()
            elif kind=='repeat':rows[-1]['repeat']=0
            elif kind=='dataset':rows[-1]['dataset']='mathqa'
            else:rows[-1]['verification_calls']+=1
            with self.assertRaises(ValueError):S.reduce_stage(rows,dataset='gsm8k',event=6)


if __name__=='__main__':unittest.main()
