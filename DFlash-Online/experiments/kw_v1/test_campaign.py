"""CPU checks for budget gates, chronology, complete-state guards and storage."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.ok_continuation.test_resume import fixture, event
from experiments.ok_events_5_8 import resume as old
from . import prepare as P, resume as R, storage as S
from .campaign import require_qualification,reconcile_work,validate_completed_work

class SelectionTests(unittest.TestCase):
    def test_source_cursor_counts_rejected_source_rows(self):
        indices=list(range(512,1025));indices.remove(700)
        audit=dict(selected_dataset_indices=indices,skipped_training_candidates=[dict(source_row=700)],prior_training_count=512)
        rows=[dict(dataset_index=i) for i in indices]
        self.assertEqual(P.previous_source_cursor(audit,rows),1025)
        audit['skipped_training_candidates']=[]
        with self.assertRaisesRegex(ValueError,'every consumed'):P.previous_source_cursor(audit,rows)

    def test_final_deduplicates_before_isolated_shuffle(self):
        questions=['a','b','a','c','d','e','f','g']
        selected,skipped,eligible=P.select_panel(questions,excluded_test_ids={1},excluded_normalized={'c'},seed=20261030,count=3)
        self.assertEqual(eligible,[0,4,5,6,7])
        self.assertNotIn(2,selected)
        import random
        expected=list(eligible);random.Random(20261030).shuffle(expected)
        self.assertEqual(selected,expected[:3])

    def test_qualification_selection_uses_stored_length_and_id(self):
        rows=[dict(record_id=f'r{i:03d}',length=20 if i<5 else 10,path=f'/x/{i}.pt') for i in range(128)]
        self.assertEqual([r['record_id'] for r in P.qualification_rows(rows)],['r000','r001','r002','r003'])

class ResumeTests(unittest.TestCase):
    def test_new_boundary_admitted_old_boundary_guard_unchanged(self):
        for number in (8,9,10,11,12):
            state=fixture('osd_first_error',number)
            self.assertEqual(R.validate_snapshot(state,objective='osd_first_error',config_text=state['config_text'],
                completed_event=number,expected_cuda_rng_states=0)['global_step'],number*128)
            if number>8:
                with self.assertRaises(ValueError):old.validate_snapshot(state,objective='osd_first_error',
                    config_text=state['config_text'],completed_event=number,expected_cuda_rng_states=0)
        with self.assertRaises(ValueError):R.validate_event_receipt(event(13),event_number=13,objective='osd')

    def test_ordinary_arm_rejects_kw_and_natural_draw_mismatch(self):
        state=fixture('osd_first_error',8);state['algorithm_contract']['response_weight_controller']={}
        with self.assertRaisesRegex(ValueError,'Ordinary O/K'):
            R.validate_snapshot(state,objective='osd_first_error',config_text=state['config_text'],
                completed_event=8,expected_cuda_rng_states=0)
        ordinary,k=event(9),event(9,'osd_first_error')
        R.validate_event_receipt(k,event_number=9,objective='osd_first_error',reference_schedule=ordinary)
        k['records'][0]['sampled_anchors'][0]['positions'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'anchors differ'):
            R.validate_event_receipt(k,event_number=9,objective='osd_first_error',reference_schedule=ordinary)

class BudgetTests(unittest.TestCase):
    def test_full_campaign_storage_includes_all_features_and_pending_overlap(self):
        plan=dict(windows=[dict(event_index=n,requests=[dict(prompt_ids=[1]*100,max_new_tokens=2048)]*128)
            for n in (9,10,11,12)])
        budget=S.full_campaign_budget(plan)
        features=4*128*2148*S.FEATURE_ROW_BYTES*2
        self.assertEqual(budget['final_incremental_payload_bytes'],features+6*4*S.WEIGHTS)
        self.assertGreater(budget['peak_required_bytes'],S.required_bytes(budget['final_incremental_payload_bytes']))
        self.assertEqual(budget['assumed_transfer_bytes'],0)
        self.assertEqual(budget['reserve_bytes'],8*S.GIB)

    def test_failed_preview_attempt_is_charged_even_without_completed_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'KW/events/event_009';folder.mkdir(parents=True)
            rows=[dict(kind='credits',phase='started'),dict(kind='credits',phase='completed'),
                  dict(kind='previews',phase='started')]
            (folder/'KW_WORK.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            progress=dict(status='failed',phase='training_KW',active_feedback=None,feedback_attempted=128)
            reconcile_work(root,progress)
            self.assertEqual(progress['optimizer_previews'],1)
            self.assertEqual(progress['credit_opportunities'],1)
            self.assertEqual(progress['kw_work_unknown']['previews'],1)
            self.assertEqual(progress['kw_completed_window_previews'],0)

    def test_completion_requires_exact_completed_kw_work_and_zero_unknown(self):
        progress=dict(status='running',optimizer_steps=1536,presentations=3072,feedback_returned=512,
            evaluation_returned=4032,evaluation_valid=4032,optimizer_previews=1024,credit_opportunities=504,
            arm_counters={a:1536 for a in ('O','K','KW')},
            kw_work_attempts=dict(previews=1024,live_steps=512,credits=504),
            kw_work_completed_lower_bound=dict(previews=1024,live_steps=512,credits=504),
            kw_work_unknown=dict(previews=0,live_steps=0,credits=0))
        validate_completed_work(progress)
        for field in ('kw_work_attempts','kw_work_completed_lower_bound','kw_work_unknown'):
            for kind in ('previews','live_steps','credits'):
                bad=deepcopy(progress);bad['status']='failed'
                bad[field][kind]+=1 if field=='kw_work_unknown' else -1
                before=deepcopy(bad)
                with self.subTest(field=field,kind=kind),self.assertRaises(ValueError):
                    validate_completed_work(bad)
                self.assertEqual(bad,before)
            missing=deepcopy(progress);del missing[field]
            with self.subTest(missing=field),self.assertRaises(ValueError):
                validate_completed_work(missing)

    def test_qualification_gate_requires_counters_and_memory_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'IMPLEMENTATION_QUALIFICATION.json'
            sealed=Path(temp)/'sealed';sealed.mkdir()
            (sealed/'PLAN.json').write_text(json.dumps(dict(initial={'K':{'state_identity':{'sha256':'a'*64}}},
                execution_source={'commit':'b'*40})))
            (sealed/'QUALIFICATION_INPUTS.json').write_text('{}')
            record=dict(status='passed',work=dict(P.QUALIFICATION_BUDGET),
                full_model_memory_checked=True,native_neutral_exact=True,preview_isolation=True)
            record['bindings']=dict(source_k8_sha256='a'*64,
                qualification_inputs_sha256=hashlib.sha256(b'{}').hexdigest(),execution_commit='b'*40)
            path.write_text(json.dumps(record));require_qualification(temp)
            record['work']['response_forwards']=9;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'outside ceiling'):require_qualification(temp)

if __name__=='__main__':unittest.main()
