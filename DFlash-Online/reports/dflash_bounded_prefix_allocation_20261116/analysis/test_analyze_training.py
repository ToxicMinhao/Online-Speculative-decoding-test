"""Synthetic retained-evidence fixtures; no model or Torch import."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_training as A


def write(root,name,value,lines=False):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))


def observation(ordinal,event):
    counts=[0]*17;counts[0]=counts[2]=1
    groups=[dict(prefix_length=2,slot=slot,occurrences=1,native_weight_sum=1.,candidate_weight_sum=weight,
        signed_weight_change=weight-1,absolute_weight_change=.25) for slot,weight in ((1,1.25),(2,.75))]
    summary=dict(blocks=2,valid_positions=4,prefix_positions=2,changed_blocks=1,changed_positions=2,
        prefix_weight_before=2.,prefix_weight_after=2.,weight_moved=.25,mass_error_abs_sum=0.,
        mass_violations=0,bounds_violations=0,support_violations=0,nonfinite_values=0,delta_numerator=.01,
        prefix_length_counts=counts,by_prefix_length_and_slot=groups)
    return dict(scope='one_original_response_forward',record_id=f'r{ordinal%128}',event_id=f'event_{event:03d}',
        presentation_ordinal=ordinal,pass_index=ordinal//128,sampled_anchors=[1,2],loss=.5,
        prefix_allocation=summary,sum_metrics=dict(kplus_response_count=1,kplus_W1=1.,kplus_W2=1.,
            kplus_coefficient1=1-.5/3,kplus_coefficient2=1+.5/3,kplus_first_error_weight_after=2.))


def fixture(root):
    write(root,'sealed/PLAN.json',dict(execution_commit='a'*40,binding='b'*64))
    for event in (7,8):
        base=f'tasks/C{event}'
        observations=[observation(i,event) for i in range(256)]
        records=[dict(response_observation=row,sampled_anchors=row['sampled_anchors'],record_id=row['record_id']) for row in observations]
        windows=[dict(window=i+1,global_step=(event-1)*128+i+1,learning_rate=.0003,
            record_ids=[row['record_id'] for row in observations[2*i:2*i+2]]) for i in range(128)]
        write(root,base+'/RESULT.json',dict(status='completed',execution_commit='a'*40,binding='b'*64,
            endpoint={'presentations':event*256},boundary=dict(export_exact=True,wall_s=1.),
            startup=dict(status='completed',optimizer_steps=0,wall_s=2.),loading_s=3.,event_wall_s=10.,
            costs=dict(training_s=8.,checkpoint_save_s=1.,export_s=1.,event_wall_s_before_commit=10.),
            publication_wait_s=0.,physical_work={'draft_forwards':512}))
        write(root,base+'/STATUS.json',dict(status='completed',wall_s=16.,sampled_process_tree_peak_rss_bytes=1000))
        write(root,base+'/MEMORY.json',dict(peak_gpu_allocated_bytes=1000,peak_gpu_reserved_bytes=2000))
        event_dir=base+f'/trainer/events/event_{event:03d}'
        write(root,event_dir+'/RESULT.json',dict(status='completed',records=records,start_step=(event-1)*128,
            end_step=event*128,optimizer_steps=128))
        write(root,event_dir+'/RESPONSE_OBSERVATIONS.jsonl',observations,lines=True)
        write(root,event_dir+'/WINDOW_OBSERVATIONS.jsonl',windows,lines=True)
    cases=[dict(case=case,status='completed',wall_s=1.,preclip_observer_s=.1,block_observer_s=.1,
           pair_wall_excluding_observers_s=.8,unchanged_native_inputs_and_first_error_terms=True,
           forward=[dict(sums={'kplus_W1':1.})],preclip={'fixture':'grad'},endpoint={'fixture':'state'},
           block_inputs=[{'fixture':'block'}],allocation={'fixture':'coefficients'}) for case in
           ('native','cap_zero','active','continued_next_pair','reloaded_next_pair')]
    write(root,'qualification/training/STATUS.json',dict(status='passed',cases=cases,exact_native_cap0=True,
        exact_own_resume=True,actual_intervention=True,origin_restored=True,optimizer_applications=5,
        response_presentations=10,loading_s=2.,startup_s=2.,wall_s=10.,peak_gpu_allocated_bytes=2000,
        peak_gpu_reserved_bytes=3000))
    write(root,'WORK.json',dict(attempted=A.EXPECTED_WORK,completed=A.EXPECTED_WORK,pending={}))
    write(root,'CLOCK.json',dict(active={},finished=[dict(occupied_card_s=32.),dict(occupied_card_s=10.)],completed_card_s=42.))


class TrainingReaderTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'evidence';fixture(self.root)

    def test_complete_reader_counts_original_rows_once_and_keeps_nested_costs(self):
        result=A.analyze(self.root,self.root.parent/'report')
        self.assertEqual(result['status'],'passed',result['errors'])
        self.assertEqual([r['responses'] for r in result['events']],[256,256])
        self.assertEqual(result['events'][0]['totals']['weight_moved'],64.)
        self.assertEqual(result['events'][0]['moved_prefix_mass_share'],.125)
        self.assertEqual(result['training_costs'][0]['training_s'],8.)
        self.assertEqual(result['training_costs'][0]['task_wall_s'],16.)
        self.assertEqual(len((self.root.parent/'report/RESPONSE_ALLOCATIONS.csv').read_text().splitlines()),513)
        self.assertEqual(len((self.root.parent/'report/PREFIX_LENGTH_SLOT.csv').read_text().splitlines()),5)

    def test_violation_and_wrong_denominator_are_not_silently_averaged_away(self):
        row=observation(0,7)
        row['prefix_allocation']['support_violations']=1
        row['prefix_allocation']['prefix_length_counts'][2]=2
        audit=A.Audit(self.root)
        A.summarize_observations([row],audit,7)
        self.assertTrue(any('support_violations' in x for x in audit.errors))
        self.assertTrue(any('denominator' in x for x in audit.errors))

    def test_embedded_duplicates_are_compared_not_used_as_additional_samples(self):
        path=self.root/'tasks/C7/trainer/events/event_007/RESULT.json'
        receipt=json.loads(path.read_text())
        receipt['records'][0]['response_observation']['prefix_allocation']['weight_moved']=10.
        path.write_text(json.dumps(receipt))
        result=A.analyze(self.root,self.root.parent/'report')
        self.assertEqual(result['status'],'failed')
        self.assertIn('C7: original-row identity',result['errors'])
        self.assertEqual(result['events'][0]['totals']['weight_moved'],64.)

    def test_pending_work_and_failed_qualification_prevent_pass(self):
        write(self.root,'WORK.json',dict(attempted=A.EXPECTED_WORK,completed=A.EXPECTED_WORK,pending={'1':{'task':'unknown'}}))
        path=self.root/'qualification/training/STATUS.json'
        saved=json.loads(path.read_text());saved['exact_own_resume']=False;path.write_text(json.dumps(saved))
        result=A.analyze(self.root,self.root.parent/'report')
        self.assertEqual(result['status'],'failed')
        self.assertIn('qualification: exact_own_resume',result['errors'])
        self.assertIn('exact completed work including startup',result['errors'])

    def test_raw_qualification_difference_overrides_pass_boolean(self):
        path=self.root/'qualification/training/STATUS.json'
        saved=json.loads(path.read_text());saved['cases'][1]['preclip']['fixture']='different';path.write_text(json.dumps(saved))
        result=A.analyze(self.root,self.root.parent/'report')
        self.assertEqual(result['status'],'failed')
        self.assertIn('qualification native/cap0 raw equality: preclip',result['errors'])

    def test_retained_FP32_cancellation_is_bounded_but_corruption_is_rejected(self):
        # C7 presentation51, prefix length4, slot3: actual retained summaries.
        group=dict(occurrences=37.,native_weight_sum=27.80467414855957,
            candidate_weight_sum=26.06849479675293,signed_weight_change=-1.736161470413208,
            absolute_weight_change=1.7587625980377197)
        audit=A.Audit(self.root)
        self.assertFalse(audit.near(group['candidate_weight_sum']-group['native_weight_sum'],
                                    group['signed_weight_change'],group['occurrences']))
        error,bound=A.signed_reduction_roundoff(group)
        self.assertEqual(error,1.7881393432617188e-05)
        self.assertLess(error,bound)
        group['signed_weight_change'] += .01
        error,bound=A.signed_reduction_roundoff(group)
        self.assertGreater(error,bound)


if __name__=='__main__':unittest.main()
