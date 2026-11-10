from copy import deepcopy
import json
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.survival_head import report as R, contract as C


def rows():
    result=[]
    for i in range(128):
        for method,calls,latency,prefix in [('KPLUS',10,2.,1),('LOCAL',9,1.9,2),('SURVIVAL',8,1.8,3)]:
            for repeat in range(2):
                result.append(dict(phase='final',event=5,dataset='gsm8k',seed=17,
                    method=method,validation_status='valid',valid_measurement=True,errors=[],
                    panel_ordinal=i,request_id=f'r{i}',repeat=repeat,
                    verification_calls=calls,accepted_draft_tokens=64-calls,
                    native_accepted_draft_tokens=64-calls,committed_progress=64,output_tokens=65,
                    checkpoint_id='/'+method,finish_reason='eos',cap_hit=False,
                    request_latency_s=latency+repeat*.2,native_generate_s=latency-.1+repeat*.2,
                    cache_maintenance_s=.01,first_round_present=True,first_round_start_digest='a'*64,
                    first_round_accepted_prefix_tokens=prefix,first_round_raw_prefix_tokens=prefix,
                    first_round_output_start=0))
    return result


def training_result(arm):
    return dict(status='completed',method=arm,dataset='gsm8k',event_id='event_005',
        start_step=512,end_step=640,optimizer_steps=128,response_presentations=256,passes=2,
        records=[dict(record_id=f'r{i}',**{'pass':p},sampled_anchors=[{'anchor':i}])
                 for p in range(2) for i in range(128)],
        boundary=dict(status='passed',export_exact=True),survival_work=C.expected_work(arm),
        survival_head=None if arm=='KPLUS' else dict(steps=128,state_path='/head.pt',
            state_identity=dict(sha256='b'*64,bytes=10)))


class ReportTests(unittest.TestCase):
    def test_paired_128_means_three_comparisons_and_prefix_counts(self):
        reduced=R.reduce_stage(rows(),dataset='gsm8k',event=5)
        self.assertEqual(len(reduced),128)
        self.assertAlmostEqual(reduced[0]['KPLUS']['request_latency_s'],2.1)
        summary=R.summarize(rows(),dataset='gsm8k',event=5)
        self.assertEqual(set(summary['comparisons']),{'SURVIVAL_vs_KPLUS','SURVIVAL_vs_LOCAL','LOCAL_vs_KPLUS'})
        effect=summary['comparisons']['SURVIVAL_vs_KPLUS']
        self.assertAlmostEqual(effect['call_reduction']['point_pct'],20.)
        self.assertAlmostEqual(effect['request_latency_s_reduction']['point_pct'],100*(1-1.9/2.1))
        self.assertEqual(effect['first_native_round']['gross_prefix_tokens_gained'],256)
        self.assertEqual(effect['first_native_round']['wins'],128)
        self.assertEqual(R.recommend(summary)['outcome'],'positive_screen_eligible_for_fresh_confirmation')

    def test_pooled_calls_not_mean_of_request_ratios(self):
        sample=rows()
        for row in sample:
            if row['panel_ordinal']==0 and row['method']=='KPLUS':
                row.update(verification_calls=50,accepted_draft_tokens=14,native_accepted_draft_tokens=14)
        summary=R.summarize(sample,dataset='gsm8k',event=5)
        effect=summary['comparisons']['SURVIVAL_vs_KPLUS']['call_reduction']
        self.assertAlmostEqual(effect['point_pct'],100*(1-1024/1320))
        self.assertNotAlmostEqual(effect['point_pct'],100*(1-(8/50+127*.8)/128))

    def test_missing_arm_duplicate_repeat_and_mismatched_seed_start_rejected(self):
        for change in ('missing','duplicate','repeatcount','start','prefixrepeat'):
            sample=rows()
            if change=='missing': sample=[r for r in sample if r['method']!='LOCAL']
            if change=='duplicate': sample.append(deepcopy(sample[0]))
            if change=='repeatcount': sample[1]['verification_calls']+=1
            if change=='start':
                sample[0]['first_round_start_digest']='b'*64
                sample[1]['first_round_start_digest']='b'*64
            if change=='prefixrepeat': sample[1]['first_round_accepted_prefix_tokens']=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                R.reduce_stage(sample,dataset='gsm8k',event=5)

    def test_mixed_sweep_direction_cannot_advance(self):
        sample=rows()
        for row in sample:
            if row['method']=='SURVIVAL': row['request_latency_s']=1. if row['repeat']==0 else 2.3
        summary=R.summarize(sample,dataset='gsm8k',event=5)
        self.assertGreater(summary['comparisons']['SURVIVAL_vs_LOCAL']['request_latency_s_reduction']['point_pct'],0)
        decision=R.recommend(summary)
        self.assertFalse(decision['fresh_confirmation_eligible'])
        self.assertFalse(decision['automatic_followup_authorized'])

    def test_intervals_crossing_zero_provisional_and_head_specific_open(self):
        summary=R.summarize(rows(),dataset='gsm8k',event=5)
        summary['comparisons']['SURVIVAL_vs_LOCAL']['call_reduction']['ci95_pct']=[-1.,5.]
        decision=R.recommend(summary)
        self.assertEqual(decision['outcome'],'provisional_signal_eligible_for_fresh_confirmation')
        self.assertEqual(decision['prefix_specific_explanation'],'open_both_heads_improve_unresolved_head_to_head')

    def test_first_round_losses_and_ties_are_kept(self):
        sample=rows()
        for row in sample:
            if row['method']=='SURVIVAL' and row['panel_ordinal']<2:
                value=row['panel_ordinal']
                row.update(first_round_accepted_prefix_tokens=value,first_round_raw_prefix_tokens=value)
        summary=R.summarize(sample,dataset='gsm8k',event=5)
        first=summary['comparisons']['SURVIVAL_vs_KPLUS']['first_native_round']
        self.assertEqual((first['wins'],first['losses'],first['ties']),(126,1,1))
        self.assertEqual(first['gross_prefix_tokens_lost'],1)

    def test_training_anchor_and_head_boundary_validation(self):
        base=training_result('KPLUS'); candidate=training_result('SURVIVAL')
        self.assertEqual(R.validate_training_result(candidate,'SURVIVAL',base)['status'],'passed')
        candidate['records'][0]['sampled_anchors']=[{'anchor':999}]
        with self.assertRaises(ValueError): R.validate_training_result(candidate,'SURVIVAL',base)

    def test_passes_may_have_different_native_shuffle_but_match_across_arms(self):
        base=training_result('KPLUS');candidate=training_result('SURVIVAL')
        for value in (base,candidate): value['records'][128:]=value['records'][128:][::-1]
        self.assertEqual(R.validate_training_result(candidate,'SURVIVAL',base)['status'],'passed')

    def test_own_native_trainer_receipts_and_resource_fields_integrate(self):
        # Import a compact synthetic state fixture, without any model load.
        sys.path.insert(0,str(Path(__file__).parents[1]))
        try:
            fixtures=importlib.import_module('test_survival_head_training')
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for arm in C.ARMS:
                _,state,_=fixtures.continuation(root,arm)
                value=state['events'][4]
                value.update(boundary=dict(status='passed',export_exact=True),checkpoint_path='/'+arm,
                    costs=dict(training_s=10.),survival_costs=dict(head_bce_response_mean=.25,
                        head_bce_pooled=.25,backbone_clipped_windows=2,backbone_optimizer_windows=128,
                        peak_reserved_bytes=2**30))
                folder=root/'campaign_001/gsm8k/event_005'/arm;folder.mkdir(parents=True)
                (folder/'RESULT.json').write_text(json.dumps(value))
                (folder/'STATUS.json').write_text(json.dumps(dict(status='completed',wall_s=15.)))
            _,valid,costs,statuses=R.training(R.Evidence(root,{'root':str(root)}))
            self.assertTrue(all(valid.values()),statuses)
            self.assertEqual(sum(r['component']=='training_s' for r in costs),3)
            self.assertEqual(sum(r['component']=='head_bce_response_mean' for r in costs),3)

    def test_incomplete_report_preserves_three_missing_arms_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';(root/'sealed').mkdir(parents=True)
            plan=dict(root=str(root),budgets=C.budgets(),analysis=C.ANALYSIS,
                binding='a'*64,execution_source={'commit':'b'*40})
            (root/'sealed/PLAN.json').write_text(json.dumps(plan))
            summary=R.build(root,Path(temp)/'report')
            self.assertEqual(summary['status'],'partial_or_incomplete')
            self.assertEqual(summary['valid_arm_events'],0)
            self.assertFalse(summary['endpoint']['inference_permitted'])
            status=json.loads((Path(temp)/'report/TRAINING_STATUS.json').read_text())
            self.assertEqual(len(status),3)
            self.assertIn('No complete valid three-arm endpoint',(Path(temp)/'report/REPORT.md').read_text())

    def test_failed_endpoint_keeps_partial_repeats_and_input_changes_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'campaign_001/evaluation/gsm8k';folder.mkdir(parents=True)
            row=dict(method='LOCAL',validation_status='invalid',errors=['fixture'])
            path=folder/'MEASUREMENTS.jsonl';path.write_text(json.dumps(row)+'\n')
            evidence=R.Evidence(root,{'root':str(root)})
            status,paired,retained=R.endpoint(evidence,{}, {},None)
            self.assertEqual(retained,[row]);self.assertEqual(paired,[])
            path.write_text('{}\n')
            self.assertEqual(evidence.stability()['status'],'changed_during_analysis')


if __name__ == '__main__': unittest.main()
