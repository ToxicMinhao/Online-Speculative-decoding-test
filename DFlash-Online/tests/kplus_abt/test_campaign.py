from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_abt import campaign, contract, evaluation, statistics, storage
from experiments.kplus_abt.data import seal_shared_packets, verify_shared_packets
from experiments.kplus_abt.native_pilot import ViewWorkCallback
from experiments.ok_coding.campaign import WorkLedger, read_work


def result(arm='KPLUS', event=5, dataset='gsm8k'):
    return dict(status='completed', method=arm, dataset=dataset,
        event_id=f'event_{event:03d}', start_step=(event-1)*128, end_step=event*128,
        optimizer_steps=128, response_presentations=256, passes=2,
        records=[dict(record_id=f'r{i}', **{'pass':p}, sampled_anchors=[dict(anchor=i)])
                 for p in range(2) for i in range(128)],
        abt_work=dict(response_forwards=256 if arm=='KPLUS' else 512,
                      response_backwards=256 if arm=='KPLUS' else 512, optimizer_steps=128),
        diagnostic_windows=[dict(window=w,status='completed',response_forwards=2,
            response_backwards=0,optimizer_steps=0,rng_restored_exact=True)
            for w in contract.DIAGNOSTIC_WINDOWS])


def rows():
    result = []
    for i in range(256):
        for method, calls, latency in [('KPLUS',10,2.),('CLEAN2',10,2.),('RANDOM',9,1.9),('ADV',8,1.8)]:
            for repeat in range(2):
                result.append(dict(phase='final',event=6,dataset='gsm8k',seed=17,
                    method=method,validation_status='valid',valid_measurement=True,errors=[],
                    panel_ordinal=i,request_id=f'r{i}',repeat=repeat,
                    verification_calls=calls,accepted_draft_tokens=64-calls,
                    native_accepted_draft_tokens=64-calls,committed_progress=64,output_tokens=65,
                    checkpoint_id='/'+method,finish_reason='eos',cap_hit=False,
                    request_latency_s=latency+repeat*.2,native_generate_s=latency-.1+repeat*.2,
                    cache_maintenance_s=.01))
    return result


class Contracts(unittest.TestCase):
    def test_work_counters_separate_reuse_and_startup(self):
        allwork=[contract.expected_work(a,6) for a in contract.ARMS for _ in contract.DATASETS]
        self.assertEqual(sum(x['optimizer_steps'] for x in allwork),2048)
        self.assertEqual(sum(x['ordinary_presentations'] for x in allwork),4096)
        self.assertEqual(sum(x['response_forwards'] for x in allwork),7168)
        self.assertEqual(contract.QUALIFICATION_BUDGET['response_forwards'],68)
        self.assertTrue(contract.QUALIFICATION_BUDGET['feedback_included_in_main_unique_total'])
        self.assertEqual(contract.STARTUP_BUDGET['optimizer_steps'],0)
        self.assertEqual(contract.DIAGNOSTIC_BUDGET['response_backwards'],0)

    def test_shared_feedback_sequence(self):
        tasks=campaign.tasks()
        self.assertEqual(len(tasks),22)
        self.assertEqual([(x['dataset'],x['event'],x['producer_event']) for x in tasks if x['kind']=='collect_shared'],
                         [('gsm8k',5,4),('gsm8k',6,5),('mathqa',5,4),('mathqa',6,5)])
        self.assertEqual([x['arm'] for x in tasks[1:5]],list(contract.ARMS))
        self.assertEqual(tasks[10]['kind'],'evaluate_dataset')

    def test_reject_wrong_views_schedule_diagnostics(self):
        original=result()
        candidate=result('ADV')
        campaign.validate_update(candidate,dataset='gsm8k',arm='ADV',event=5,baseline=original)
        for change in ('views','anchor','diagnostic','response'):
            bad=deepcopy(candidate)
            if change=='views': bad['abt_work']['response_forwards']=256
            elif change=='anchor': bad['records'][0]['sampled_anchors'][0]['anchor']=999
            elif change=='diagnostic': bad['diagnostic_windows'].pop()
            else: bad['records'][0]['record_id']='r1'
            with self.assertRaises(ValueError):
                campaign.validate_update(bad,dataset='gsm8k',arm='ADV',event=5,baseline=original)

    def test_failure_is_not_retried_or_counted_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'sealed').mkdir()
            (root/'sealed/PLAN.json').write_text(json.dumps(dict(execution_source={'commit':'abc'})))
            called=[]
            def collect(*args):
                called.append(args[1:3]);raise RuntimeError('fixture failure')
            with patch.object(campaign,'require_admission',return_value={'status':'passed'}):
                with self.assertRaises(RuntimeError):
                    campaign.run(root,collect_shared=collect,train_arm=None,evaluate_dataset=None)
            saved=json.loads((root/'campaign_001/STATUS.json').read_text())
            self.assertEqual(called,[('gsm8k',5)])
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['completed_tasks'],[])
            self.assertEqual(saved['unattempted_tasks'],21)

    def test_peak_counts_all_eight_final_states_and_staging(self):
        base=storage.capacity_plan(2097259104,6297245019)
        self.assertEqual(base['required_initial_free_bytes'],138374886251)
        staged=storage.capacity_plan(2097259104,6297245019,retrieval_staging=True)
        self.assertEqual(staged['required_initial_free_bytes'],167259424027)
        more=storage.capacity_plan(2097259104,6297245019,retained_qualification_bytes=1000)
        self.assertEqual(more['required_initial_free_bytes']-base['required_initial_free_bytes'],1050)
        self.assertEqual(base['retained_final_boundaries'],8)

    def test_real_view_callbacks_retain_unknown_work_and_enforce_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger=WorkLedger(Path(temp)/'work.jsonl',dict(response_forwards=1,response_backwards=1))
            callback=ViewWorkCallback(ledger)
            callback(kind='forward',view='clean',phase='started')
            work=read_work(ledger.path,ledger.limits)
            self.assertEqual(work['unknown']['response_forwards'],1)
            callback(kind='forward',view='clean',phase='completed')
            with self.assertRaises(RuntimeError):
                callback(kind='forward',view='second',phase='started')
            with self.assertRaises(ValueError):
                callback(kind='backward',view='second',phase='completed')
            self.assertEqual(read_work(ledger.path,ledger.limits)['completed']['response_forwards'],1)

    def test_shared_pool_tampering_is_detected_without_tensor_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);features=[];records=[]
            for i in range(128):
                feature=root/f'{i}.pt';feature.write_bytes(b'fixture-not-a-tensor')
                record=root/f'{i}.json';record.write_text(json.dumps(dict(record_id=f'r{i}')))
                features.append(feature);records.append(record)
            pool=seal_shared_packets(features,records,dataset='gsm8k',event=5,output=root/'POOL.json')
            self.assertEqual(pool['schema'],contract.SCHEMA)
            self.assertEqual(verify_shared_packets(pool)['packets'],128)
            features[-1].write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_shared_packets(pool)


class Evaluation(unittest.TestCase):
    def test_four_arm_order_is_balanced_and_repeat_reversed(self):
        requests=[dict(request_id=f'r{i}') for i in range(256)]
        schedule=evaluation.evaluation_schedule(requests,'gsm8k')
        self.assertEqual(len(schedule),2048)
        balance=evaluation.schedule_balance(schedule)
        self.assertEqual(balance['positions'],{m:[128]*4 for m in contract.ARMS})
        self.assertEqual(set(balance['pair_precedence'].values()),{256})
        for i in range(256):
            first=[s['method'] for s in schedule if s['repeat']==0 and s['panel_ordinal']==i]
            second=[s['method'] for s in schedule if s['repeat']==1 and s['panel_ordinal']==i]
            self.assertEqual(first,second[::-1])

    def test_missing_fourth_arm_is_not_three_arm_success(self):
        sample=rows()
        with self.assertRaises(ValueError):
            statistics.reduce_stage([r for r in sample if r['method']!='ADV'],dataset='gsm8k',event=6)
        with self.assertRaises(ValueError):
            statistics.reduce_stage(sample+sample[:1],dataset='gsm8k',event=6)

    def test_statistics_average_repeats_then_compare_three_controls(self):
        sample=rows()
        reduced=statistics.reduce_stage(sample,dataset='gsm8k',event=6)
        self.assertEqual(len(reduced),256)
        self.assertAlmostEqual(reduced[0]['KPLUS']['request_latency_s'],2.1)
        summary=statistics.summarize(sample,dataset='gsm8k',event=6)
        self.assertEqual(set(summary['comparisons']),{'ADV_vs_KPLUS','ADV_vs_CLEAN2','ADV_vs_RANDOM'})
        self.assertAlmostEqual(summary['comparisons']['ADV_vs_KPLUS']['call_reduction']['point_pct'],20.)
        self.assertEqual(summary['method_totals']['KPLUS']['verification_calls'],2560)
        self.assertEqual(summary['primary'],['ADV_vs_KPLUS'])


if __name__ == '__main__':
    unittest.main()
