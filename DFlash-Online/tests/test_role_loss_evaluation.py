"""Canonical safety, exact schedule and paired-request statistical contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.role_loss_pilot import evaluation as E, statistics as S
from experiments.role_loss_pilot.report import independent_scalar_audit, pooled_observations
from test_full_k_gsm8k_evaluation import FullBackend, requests


def rows(event=4, dataset='gsm8k'):
    result=[]
    for i in range(256):
        for repeat in range(1 if event==2 else 2):
            for method in E.METHODS:
                calls=10+i%7-(method=='ROLE')
                result.append(dict(seed=17,dataset=dataset,event=event,
                    phase='final' if event==4 else 'intermediate',panel_ordinal=i,request_id=f'eval{i}',
                    repeat=repeat,method=method,verification_calls=calls,accepted_draft_tokens=50,
                    native_accepted_draft_tokens=99-calls,committed_progress=99,output_tokens=100,
                    checkpoint_id='/'+method,finish_reason='eos',cap_hit=False,
                    validation_status='valid',valid_measurement=True,errors=[],
                    request_latency_s=calls+repeat,native_generate_s=calls-.1+repeat,cache_maintenance_s=0.))
    return result


class ScheduleTests(unittest.TestCase):
    def test_all_timed_work_and_position_balance(self):
        total=0
        for dataset in E.DATASETS:
            for event in (2,4):
                schedule=E.evaluation_schedule(requests(256),dataset,event)
                total+=len(schedule)
                self.assertEqual(len(schedule),768 if event==2 else 1536)
                counts=E.schedule_balance(schedule)['positions']
                for arm in E.METHODS:
                    self.assertLessEqual(max(counts[arm])-min(counts[arm]),1)
                self.assertEqual(schedule[:32],[dict(schedule[0],panel_ordinal=i,request_id=f'eval{i}') for i in range(32)])
        self.assertEqual(total,4608)

    def test_stage_rejects_changed_schedule_and_missing_authentication(self):
        inputs=requests(256)
        refs={r['request_id']:dict(response_ids=[151645],finish_reason='eos',verified=True,
            prompt_ids=r['prompt_ids'],binding_digest='fixed') for r in inputs}
        paths={m:'/'+m for m in E.METHODS}
        schedule=E.evaluation_schedule(inputs,'gsm8k',4)
        E.validate_evaluation_inputs(inputs,paths,refs,dataset='gsm8k',event=4,schedule=schedule)
        with self.assertRaisesRegex(ValueError,'presealed'):
            E.validate_evaluation_inputs(inputs,paths,refs,dataset='gsm8k',event=4)
        schedule[0]['method']='ROLE'
        with self.assertRaisesRegex(ValueError,'order changed'):
            E.validate_evaluation_inputs(inputs,paths,refs,dataset='gsm8k',event=4,schedule=schedule)


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.inputs=requests(4)
        self.refs={r['request_id']:dict(response_ids=[10,11,12,13,151645],finish_reason='eos',
            source_record_path='/historical/'+r['request_id'],prompt_ids=r['prompt_ids'],verified=True,
            binding_digest='binding') for r in self.inputs}
        self.paths={m:'/'+m for m in E.METHODS}

    def run_stage(self,backend,folder,event=4):
        return E.evaluate_stage(backend,self.inputs,self.paths,self.root/folder,canonicals=self.refs,
            dataset='gsm8k',event=event,production=False)

    def test_exact_tokens_native_counters_repeats_and_raw_audit(self):
        backend=FullBackend(self.inputs)
        value=self.run_stage(backend,'valid')
        audit=E.audit_stage(self.root/'valid',self.inputs,self.paths,canonicals=self.refs,
                           dataset='gsm8k',event=4,production=False)
        self.assertEqual(value['completed_valid_generations'],24)
        self.assertEqual(audit['errors'],[])
        self.assertTrue(all(row['native_accepted_draft_tokens']==2 for row in audit['rows']))
        path=self.root/'valid/0000_r0_KPLUS.json'
        raw=json.loads(path.read_text());raw['response_ids'][0]=999;path.write_text(json.dumps(raw))
        audit=E.audit_stage(self.root/'valid',self.inputs,self.paths,canonicals=self.refs,
                           dataset='gsm8k',event=4,production=False)
        self.assertEqual(audit['status'],'invalid_or_incomplete')

    def test_invalid_first_completion_preserves_unattempted(self):
        def mutation(record,options): record['response_ids'][0]=999
        with self.assertRaises(RuntimeError):self.run_stage(FullBackend(self.inputs,mutation),'failed')
        status=json.loads((self.root/'failed/STATUS.json').read_text())
        self.assertEqual(status['invalid_generations'],1)
        self.assertEqual(status['unattempted_generations'],23)
        self.assertEqual(status['unknown_generation_attempts'],0)

    def test_unknown_attempt_is_not_retried_or_lost(self):
        backend=FullBackend(self.inputs)
        with patch.object(backend,'generate',side_effect=RuntimeError('native failure')):
            with self.assertRaises(RuntimeError):self.run_stage(backend,'failed')
        status=json.loads((self.root/'failed/STATUS.json').read_text())
        self.assertEqual(status['unknown_generation_attempts'],1)
        self.assertEqual(status['unattempted_generations'],23)
        self.assertEqual(status['returned_generations'],0)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_matches_direct_oracle_two_repeat_means(self):
        values=rows()
        summary=S.summarize(values,dataset='gsm8k',event=4)
        before=np.asarray([10+i%7 for i in range(256)])
        indices=np.random.Generator(np.random.PCG64(20261107)).integers(0,256,(10000,256))
        expected=100*(1-(before-1)[indices].sum(1)/before[indices].sum(1))
        effect=summary['comparisons']['ROLE_vs_KPLUS']
        self.assertEqual(effect['call_reduction']['ci95_pct'],np.quantile(expected,[.025,.975],method='linear').tolist())
        self.assertEqual(summary['method_totals']['KPLUS']['request_latency_s'],float(before.sum()+128))
        self.assertEqual(summary['method_totals']['KPLUS']['verification_calls'],int(before.sum()))
        self.assertEqual(independent_scalar_audit(values,summary)['status'],'passed')

    def test_incomplete_repeat_cross_dataset_and_count_disagreement_rejected(self):
        for mutation in ('missing','repeat','dataset','counts'):
            values=rows()
            if mutation=='missing':values.pop()
            elif mutation=='repeat':values[-1]['repeat']=0
            elif mutation=='dataset':values[-1]['dataset']='mathqa'
            else:values[-1]['native_accepted_draft_tokens']+=1
            with self.assertRaises(ValueError):S.reduce_stage(values,dataset='gsm8k',event=4)

    def test_event2_is_descriptive_and_cap_records_not_filtered(self):
        values=rows(2)
        for row in values:
            if row['panel_ordinal']==0:row.update(cap_hit=True,finish_reason='length',output_tokens=2048,committed_progress=2047,
                    native_accepted_draft_tokens=2047-row['verification_calls'])
        summary=S.summarize(values,dataset='gsm8k',event=2)
        self.assertIsNone(summary['primary'])
        self.assertEqual(summary['request_count'],256)
        self.assertEqual(len(S.reduce_stage(values,dataset='gsm8k',event=2)),256)

    def test_pooled_ratios_use_numerators_not_means(self):
        observations=[dict(loss=1.,sum_metrics={'n':2},ratio_metrics={'r':[1,2]}),
                      dict(loss=3.,sum_metrics={'n':3},ratio_metrics={'r':[9,10]})]
        pooled=pooled_observations(observations)
        self.assertEqual(pooled['ratio_metrics']['r'],dict(numerator=10.,denominator=12.,value=10/12))
        self.assertEqual(pooled['actual_mean_response_loss'],2.)


if __name__=='__main__':unittest.main()
