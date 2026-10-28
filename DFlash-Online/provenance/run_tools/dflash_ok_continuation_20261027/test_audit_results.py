"""Local-only tests of independent audit mechanics; no current-run data used."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_results as audit

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fixed completed historical artifacts are fixtures, never modified.
        cls.native=audit.load(RESEARCH/'dflash_first_error_fresh_confirmation_20261027/raw/campaign_001/evaluation/0000_r0_O.json')
        cls.ordinary=audit.load(RESEARCH/'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json')
        cls.selected=audit.load(RESEARCH/'dflash_first_error_execution_20261026/raw/campaign_001/K/events/event_001/RESULT.json')

    def native_check(self,record):
        request={k:self.native[k] for k in ('prompt_ids','response_ids','finish_reason','stop_token_ids')}
        issues=[]
        result=audit.native_record(record,request,self.native['checkpoint_id'],capture=False,
            session=self.native['session_id'],label=self.native['record_id'],
            index=self.native['request_index'],issues=issues)
        return result,issues

    def test_historical_trace_independent_reconstruction(self):
        result,issues=self.native_check(self.native)
        self.assertEqual([],issues)
        self.assertEqual(36,result['verification_calls'])
        self.assertEqual(len(self.native['response_ids'])-1,result['committed_progress'])

    def test_first_mismatch_and_committed_counters_cannot_be_faked(self):
        for field in ('raw_accepted_draft_tokens_before_stop','accepted_draft_tokens',
                      'native_accepted_draft_tokens','committed_progress'):
            with self.subTest(field=field):
                record=copy.deepcopy(self.native)
                record['metadata']['trace'][0][field]+=1
                record['metadata']['capture']['blocks']=copy.deepcopy(record['metadata']['trace'])
                self.assertTrue(self.native_check(record)[1])

    def test_exact_canonical_and_termination_are_required(self):
        for change in ('token','termination','capture','known_seed'):
            with self.subTest(change=change):
                record=copy.deepcopy(self.native)
                if change=='token':record['response_ids'][2]+=1
                elif change=='termination':record['finish_reason']='length'
                elif change=='capture':record['metadata']['capture']['collect_features']=True
                else:record['metadata']['trace'][0]['anchor_token']+=1
                self.assertTrue(self.native_check(record)[1])

    def test_native_serial_and_wrapper_duplication(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence=audit.Evidence(folder);record=copy.deepcopy(self.native)
            record['metadata']['native_trace_path']='/remote/request_000129.json'
            write(Path(folder)/'campaign_001/native/request_000129.json',record['metadata']['capture'])
            audit.bind_native(evidence,record,129)
            self.assertEqual([],evidence.issues)
            audit.bind_native(evidence,record,130)
            self.assertEqual(2,len(evidence.issues))

    def test_old_O_K_boundary_exposure_matches(self):
        issues=[]
        ordinary=audit.training_exposure(self.ordinary,1,'O',issues)
        selected=audit.training_exposure(self.selected,1,'K',issues,self.ordinary)
        self.assertEqual([],issues)
        self.assertEqual(949560,ordinary['available_occurrences'])
        self.assertEqual(ordinary['available_occurrences'],selected['available_occurrences'])
        self.assertEqual(534063,selected['selection']['selected_position_occurrences'])

    def test_double_counted_accumulation_totals_are_rejected(self):
        receipt=copy.deepcopy(self.selected)
        receipt['selection_totals']['selected_position_occurrences']=sum(
            r['metrics']['selected_position_occurrences'] for r in receipt['records'])
        issues=[];audit.training_exposure(receipt,1,'K',issues)
        self.assertTrue(any('boundary-only' in value for value in issues))

    def test_mismatched_anchor_and_reference_work_are_rejected(self):
        receipt=copy.deepcopy(self.selected)
        receipt['records'][0]['sampled_anchors'][0]['positions'][0][0]+=1
        receipt['objective_work']['reference_forward_s']=1
        issues=[];audit.training_exposure(receipt,1,'K',issues,self.ordinary)
        self.assertTrue(any('matched normal sampling' in value for value in issues))
        self.assertTrue(any('reference/auxiliary' in value for value in issues))

    def test_first_error_inclusion_and_epsilon_bounds(self):
        receipt=copy.deepcopy(self.selected)
        row=receipt['records'][0]['metrics']
        row['selected_position_occurrences']=0
        row['loss_denominator']+=10
        issues=[];audit.training_exposure(receipt,1,'K',issues)
        self.assertTrue(any('includes first error' in value for value in issues))
        self.assertTrue(any('one epsilon' in value for value in issues))

    def receipt_fixture(self,folder,event=2,arm='K'):
        parent=dict(state=f'/old/{arm}/state.pt',export=f'/old/{arm}/export')
        digest=dict(bytes=100,sha256='a'*64)
        saved=dict(status='passed',completed_event=event-1,objective=audit.OBJECTIVES[arm],
            global_step=(event-1)*128,presentations=(event-1)*256,horizon=7424,draft_tensors=58,
            optimizer_parameters=58,adam_step=(event-1)*128,cuda_rng_states=1,
            scientific_extra_steps=0,export_exact=True,state_path=parent['state'],
            export_path=parent['export'],state_identity=digest)
        live=dict(status='passed',exact_resume=True,mismatches=[],mismatch_count=0,
            global_step=(event-1)*128,presentations=(event-1)*256,no_rng_reset=True,scientific_extra_steps=0)
        closed=dict(closed=True,failed=False,global_step=event*128,presentations=event*256,
            completed_events=[f'event_{n:03d}' for n in range(1,event+1)],objective=audit.OBJECTIVES[arm],
            reduction='response_mean',accumulation_remainder=0,horizon=7424,full_state_path='/new/state.pt')
        checked=dict(status='passed',event_id=f'event_{event:03d}',objective=audit.OBJECTIVES[arm],
            start_step=(event-1)*128,end_step=event*128,optimizer_steps=128,response_presentations=256,
            passes=2,reduction='response_mean',schedule_matched=arm=='K',anchor_draws_checked=256,scientific_extra_steps=0)
        costs={k:1 for k in ('loading_and_resume_validation_s','event_wall_s','loading_peak_allocated_bytes',
            'loading_peak_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')}
        root=Path(folder)/f'campaign_001/resume_{event:03d}_{arm}'
        for name,data in [('SAVED',saved),('LIVE',live),('CLOSED',closed),('EVENT_CHECK',checked),('COSTS',costs)]:
            write(root/(name+'.json'),data)
        return parent,dict(files={parent['state']:digest}),root

    def test_each_resume_event_and_arm_requires_complete_contract(self):
        for n in (2,3,4):
            for arm in audit.ARMS:
                with self.subTest(event=n,arm=arm),tempfile.TemporaryDirectory() as folder:
                    parent,seal,_=self.receipt_fixture(folder,n,arm);evidence=audit.Evidence(folder)
                    result=audit.resume_receipts(evidence,n,arm,parent,seal)
                    self.assertEqual([],evidence.issues)
                    self.assertEqual('passed',result['status'])

    def test_wrong_parent_reset_rng_and_live_mismatch_rejected(self):
        for name,field,value in [('SAVED','state_path','/wrong/O/state.pt'),('SAVED','adam_step',0),
            ('LIVE','no_rng_reset',False),('LIVE','mismatches',['optimizer']),
            ('EVENT_CHECK','scientific_extra_steps',1),('CLOSED','global_step',128)]:
            with self.subTest(name=name,field=field),tempfile.TemporaryDirectory() as folder:
                parent,seal,root=self.receipt_fixture(folder)
                path=root/(name+'.json');data=audit.load(path);data[field]=value;write(path,data)
                evidence=audit.Evidence(folder);audit.resume_receipts(evidence,2,'K',parent,seal)
                self.assertTrue(evidence.issues)

    def test_missing_closed_never_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            parent,seal,root=self.receipt_fixture(folder);(root/'CLOSED.json').unlink()
            evidence=audit.Evidence(folder);result=audit.resume_receipts(evidence,2,'K',parent,seal)
            self.assertEqual('incomplete',result['status'])
            self.assertTrue(evidence.missing)

    def test_storage_includes_atomic_overlap_and_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence=audit.Evidence(folder);path=Path(folder)/'storage.json'
            tensors=3*(4096*6*2+12)+(3*3+2)*2097259104
            required=int(tensors*1.1)+12*2**30
            row=dict(rows=3,new_states=2,exports=2,overlap_states=1,tensor_bytes=tensors,
                tensor_margin=.1,compact_allowance_bytes=2*2**30,work_bytes=2*2**30,
                reserve_bytes=8*2**30,required_bytes=required,free_bytes=required,admitted=True)
            write(path,row);audit.storage_receipt(path,evidence,rows=3,first=True)
            self.assertEqual([],evidence.issues)
            row['free_bytes']-=1;write(path,row);audit.storage_receipt(path,evidence,rows=3,first=True)
            self.assertTrue(evidence.issues)

    def test_empty_run_is_incomplete_and_no_model_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            result=audit.audit(folder)
            self.assertEqual('incomplete',result['status'])
            self.assertFalse(result['completed_campaign'])
            self.assertEqual(0,result['new_model_forwards'])
            self.assertIn('Missing evidence',audit.markdown(result))
        self.assertNotIn('torch',sys.modules)

    def test_rotation_and_serial_window_boundaries(self):
        requests=[dict(request_id=f'r{i}',panel_ordinal=i) for i in range(64)]
        result=audit.slots(requests)
        self.assertEqual(384,len(result))
        self.assertEqual(['O','K','K','O','O','K','K','O'],[row[2] for row in result[:8]])
        self.assertEqual(128+384+128+128+384,1152)
        self.assertEqual(193,sum(i==0 or result[i-1][2]!=row[2] for i,row in enumerate(result)))

    def failed_evaluation_fixture(self,folder,*,eos_at_cap=False):
        root=Path(folder)/'campaign_001/evaluation_002'
        requests=[dict(request_id=f'r{i}',panel_ordinal=i,original_index=i,
            prompt_ids=[10],response_ids=[11,99],finish_reason='eos',stop_token_ids=[99]) for i in range(64)]
        if eos_at_cap:requests[0]['response_ids']=[11]*2047+[99]
        plan=dict(eval_requests=requests,frozen_counts={r['request_id']:1 for r in requests})
        checkpoints=dict(O='/O/export',K='/K/export')
        write(root/'STATUS.json',dict(status='failed',returned_generations=384))
        write(root/'FROZEN_COUNTS.json',plan['frozen_counts'])
        write(root/'PROTOCOL.json',dict(arms=['O','K'],comparisons=[['K','O']],repeats=3,
            checkpoints=checkpoints,rotation='(panel_ordinal + repeat) % 2',count_trajectory_repeat=0,
            historical_count_gate=False,capture=False,extra_warmup_generations=0))
        rows=[]
        for request,repeat,arm in audit.slots(requests):
            label=f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
            counts=dict(verification_calls=1,accepted_draft_tokens=0,
                committed_progress=len(request['response_ids'])-1,output_tokens=len(request['response_ids']))
            row=dict(**counts,label=label,request_id=request['request_id'],method=arm,repeat=repeat,
                panel_ordinal=request['panel_ordinal'],original_index=request['original_index'],
                validation_status='valid',valid_measurement=True,structural_errors=[],errors=[],
                finish_reason='eos',cap_hit=False,prompt_tokens=1,record_path='/remote/'+label+'.json',
                request_latency_s=1.,native_generate_s=.5,cache_maintenance_s=.1)
            rows.append(row)
            write(root/(label+'.json'),dict(metadata=dict(request_latency_s=1.,native_generate_s=.5,
                capture=dict(cache_maintenance_s=.1))))
            write(root/(label+'_VALIDATION.json'),dict(label=label,request_id=request['request_id'],
                method=arm,repeat=repeat,status='valid',valid=True,errors=[],
                canonical_source='retained_manifest_eval_requests',canonical_method='F'))
        (root/'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return plan,checkpoints

    def evaluated_fixture(self,folder,*,eos_at_cap=False):
        plan,checkpoints=self.failed_evaluation_fixture(folder,eos_at_cap=eos_at_cap)
        evidence=audit.Evidence(folder)
        def counts(record,request,*args,**kwargs):
            return dict(verification_calls=1,accepted_draft_tokens=0,
                committed_progress=len(request['response_ids'])-1,output_tokens=len(request['response_ids']))
        with patch.object(audit,'native_record',side_effect=counts),patch.object(audit,'bind_native'):
            result=audit.evaluation(evidence,2,plan,checkpoints,129)
        return result,evidence

    def test_failed_child_with_all384_valid_rows_cannot_pass_completed_root(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluated,evidence=self.evaluated_fixture(folder)
            self.assertEqual([],evidence.issues)
            self.assertEqual(384,evaluated['returned'])
            self.assertEqual('incomplete',evaluated['status'])
            events={str(n):dict(pool=dict(status='complete'),
                O=dict(resume=dict(status='passed'),training=dict(status='completed')),
                K=dict(resume=dict(status='passed'),training=dict(status='completed'))) for n in (2,3,4)}
            complete_eval=dict(status='complete',returned=384,execution=dict(status='completed'))
            events['2']['evaluation']=evaluated;events['4']['evaluation']=complete_eval
            execution=dict(status=dict(status='completed'),events=events)
            write(Path(folder)/'sealed/PLAN.json',{})
            write(Path(folder)/'sealed/SEAL.json',{})
            (Path(folder)/'EXIT_CODE').write_text('0\n')
            with patch.object(audit,'provenance',return_value={}),patch.object(audit,'execution',return_value=execution):
                result=audit.audit(folder)
                self.assertEqual('failed',result['status'])
                self.assertFalse(result['completed_campaign'])
                self.assertTrue(any('completed event2 evaluation' in issue for issue in result['issues']))
                events['2']['evaluation']=complete_eval
                result=audit.audit(folder)
                self.assertEqual('passed',result['status'])
                self.assertTrue(result['completed_campaign'])

    def test_eos_at_output_cap_is_not_length_termination(self):
        with tempfile.TemporaryDirectory() as folder:
            result,evidence=self.evaluated_fixture(folder,eos_at_cap=True)
            self.assertEqual(384,result['returned'])
            self.assertEqual([],evidence.issues)
            path=Path(folder)/'campaign_001/evaluation_002/MEASUREMENTS.jsonl'
            rows=audit.lines(path);rows[0]['cap_hit']=True
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            plan=dict(eval_requests=[dict(request_id=f'r{i}',panel_ordinal=i,original_index=i,
                prompt_ids=[10],response_ids=[11]*2047+[99] if i==0 else [11,99],
                finish_reason='eos',stop_token_ids=[99]) for i in range(64)],frozen_counts={f'r{i}':1 for i in range(64)})
            evidence=audit.Evidence(folder)
            def counts(record,request,*args,**kwargs):
                return dict(verification_calls=1,accepted_draft_tokens=0,
                    committed_progress=len(request['response_ids'])-1,output_tokens=len(request['response_ids']))
            with patch.object(audit,'native_record',side_effect=counts),patch.object(audit,'bind_native'):
                audit.evaluation(evidence,2,plan,dict(O='/O/export',K='/K/export'),129)
            self.assertTrue(any('cap_hit' in issue for issue in evidence.issues))


if __name__=='__main__':
    unittest.main()
