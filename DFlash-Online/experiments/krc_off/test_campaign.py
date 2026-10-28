"""CPU checks for exact restore, narrow OFF migration and included work gates."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.ok_continuation.test_resume import event
from . import campaign as C
from . import protocol as P
from .test_evaluate import panel, Backend


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def off_report():
    return dict(status='completed', start_global_step=128, end_global_step=256,
                first_A={'status':'passed'}, extra_qualification_optimizer_steps=0,
                work=dict(response_forwards=256, K_backwards=256, Adam_steps=128,
                          presentations=256, reference_forwards=0, reference_backwards=0,
                          protected_prefix_captures=0, direct_corrections=0, probe_forwards=0))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.o2=event(2, 'osd')
        self.k2=dict(event(2, 'osd_first_error'), checkpoint_path='K2')
        self.krc2=dict(event(2, 'osd_first_error'), checkpoint_path='KRC2')
        self.krc1=dict(event(1, 'osd_first_error'), checkpoint_path='KRC1')

    def test_sealed_schedule_preserves_capped_record_and_original_metadata(self):
        self.o2['features'][0].update(finish_reason='length', response_tokens=2048)
        before=deepcopy(self.o2)
        self.assertTrue(P.validate_schedule(self.o2, before)['passed'])
        self.assertEqual(self.o2, before)
        self.assertTrue(P.validate_retained(self.krc1, event(2,'osd'),self.k2,self.krc2)['passed'])
        with self.assertRaisesRegex(ValueError, 'Evaluation'):
            P.validate_schedule(self.o2, before, evaluation_ids=['r000'])
        changed=deepcopy(self.o2); changed['records'][0]['sampled_anchors'][0]['positions'][0][0]+=1
        with self.assertRaisesRegex(ValueError, 'schedule differs'): P.validate_schedule(changed, self.o2)

    def test_zero_reference_budget_and_complete_global_boundary(self):
        report=off_report()
        self.assertEqual(P.check_off_report(report)['core_response_forwards'],256)
        for mutate in (lambda x:x['work'].update(reference_forwards=1),
                       lambda x:x['work'].update(protected_prefix_captures=1),
                       lambda x:x['work'].update(Adam_steps=129),
                       lambda x:x.update(end_global_step=128),
                       lambda x:x['first_A'].update(status='failed')):
            changed=deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):P.check_off_report(changed)
        status=dict(global_step=256,presentations=512,completed_events=['event_001','event_002'],
                    accumulation_remainder=0,objective='osd_first_error',horizon=7424)
        self.assertEqual(P.check_event(self.k2,self.o2,status)['end_step'],256)
        status['global_step']=128
        with self.assertRaisesRegex(ValueError,'status'):P.check_event(self.k2,self.o2,status)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.o2=event(2,'osd')
        self.k2=dict(event(2,'osd_first_error'),checkpoint_path='retained-K2')
        self.krc2=dict(event(2,'osd_first_error'),checkpoint_path='retained-KRC2')
        self.krc1=dict(event(1,'osd_first_error'),checkpoint_path='retained-KRC1')
        self.plan=dict(config=str(self.root/'config.yaml'),startup_ordinary_receipt=str(self.root/'O1.json'),
            pinned_initial='released-startup-only.pt',krc1_state='complete-KRC1-state.pt',
            krc1_receipt=str(self.root/'KRC1.json'),event2_ordinary_receipt=str(self.root/'O2.json'),
            event2_k_receipt=str(self.root/'K2.json'),presentation_schedule=str(self.root/'schedule.json'),
            krc2_receipt=str(self.root/'KRC2.json'),retained_krc2_checkpoint='retained-KRC2',
            target='target',draft='draft',worker_python='worker',panels=[panel(),panel('additional')])
        save(self.root/'sealed/PLAN.json',self.plan)
        (self.root/'config.yaml').write_text('exact config')
        for name,value in [('O1.json',event()),('O2.json',self.o2),('K2.json',self.k2),
                           ('KRC1.json',self.krc1),('KRC2.json',self.krc2),('schedule.json',self.o2)]:
            save(self.root/name,value)

    def tearDown(self): self.assertFalse(torch.cuda.is_initialized())

    def runtime(self, *, resume_failure=False, first_A_failure=False, evaluation_failure=False):
        outer=self; events=[];current=[None];report=off_report()
        receipt=dict(self.k2,checkpoint_path='new-KRC2_OFF',costs={'training_s':1},objective_work={})
        state={'identity':'KRC1 complete state with own RNG'}
        class Trainer:
            def __init__(self,config,root,**kwargs):
                outer.assertEqual(kwargs,dict(resume_state='complete-KRC1-state.pt',
                                             objective='osd_first_error',reduction='response_mean'))
                events.append('restore_KRC1');current[0]=self
                self.global_step=128;self.presentations=256;self.closed=self.failed=False
            def enable_off_continuation(self, *, resume_check, first_A_records, output_dir):
                outer.assertIn('exact_live_resume',events)
                outer.assertTrue(resume_check['exact_resume'])
                outer.assertEqual(first_A_records,outer.krc2['records'][:2])
                events.append('disable_correction_after_resume')
            def train_event(self,event,paths,*,passes,presentation_schedule):
                outer.assertEqual((event,len(paths),passes),('event_002',128,2))
                outer.assertEqual(presentation_schedule,outer.o2)
                events.append('train_event2')
                if first_A_failure:
                    self.global_step=129;self.presentations=258
                    report.update(status='failed',end_global_step=129,first_A={'status':'failed'})
                    report['work'].update(response_forwards=2,K_backwards=2,Adam_steps=1,presentations=2)
                    report['failure_value']=float('nan')
                    raise RuntimeError('first A mismatch')
                self.global_step=256;self.presentations=512
                return receipt
            def off_report(self):return report
            def close(self):self.closed=True;events.append('trainer_closed')
            def status(self):
                return dict(global_step=self.global_step,presentations=self.presentations,
                    completed_events=['event_001','event_002'] if self.global_step==256 else ['event_001'],
                    accumulation_remainder=0,objective='osd_first_error',horizon=7424,closed=self.closed,failed=self.failed)
        class Native:
            loading_s=1
            def __init__(self,*args,**kwargs):
                outer.assertTrue(current[0].closed);events.append('native_loaded')
                outer.assertEqual((kwargs['block_size'],kwargs['context_limit']),(16,4096))
            def close(self):events.append('backend_closed')
        class WorkLog:
            def __init__(self,path):self.counts=Counter()
        def load(path,export,**kwargs):
            events.append('snapshot_and_export_exact')
            outer.assertEqual((path,export),('complete-KRC1-state.pt','retained-KRC1'))
            outer.assertEqual(kwargs,dict(objective='osd_first_error',config_text='exact config',completed_event=1))
            return state,dict(status='passed',wall_s=0,export_exact=True)
        def verify(trainer,expected,*,receipt_path):
            outer.assertIs(expected,state);events.append('exact_live_resume')
            if resume_failure:raise ValueError('Live resume differs')
            return dict(status='passed',exact_resume=True,global_step=128,presentations=256)
        def startup(config,ordinary,pin,log,root):
            events.append('startup_two_zero_Adam');outer.assertNotIn('restore_KRC1',events)
            log.counts.update(startup_response_evaluation_completed=2,startup_backward_completed=2)
            return dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
                        exact_initial_state_restored=True,separate_disposable_runtime=True)
        def pin(root):events.append('donated_buffer_false');return {'effective':False}
        def evaluate(backend,requests,checkpoints,root,*,expected_control_counts,historical_counts,qualification_requests):
            events.append('evaluate_'+root.name)
            outer.assertEqual(checkpoints,{'KRC2':'retained-KRC2','KRC2_OFF':'new-KRC2_OFF'})
            outer.assertEqual(qualification_requests,8 if root.name=='development' else 0)
            outer.assertIsNotNone(expected_control_counts)
            value=dict(C.unattempted_panel(),status='completed',returned_generations=384,attempted_generations=384,
                       completed_valid_generations=384,unattempted_generations=0)
            if evaluation_failure:
                value.update(status='failed',returned_generations=1,attempted_generations=1,
                             completed_valid_generations=0,invalid_generations=1,unattempted_generations=383)
                save(root/'STATUS.json',value);raise RuntimeError('native mismatch')
            save(root/'STATUS.json',value);return value
        modules={'online_feedback.krc_off':SimpleNamespace(KRCOffTrainer=Trainer),
                 'online_feedback.sglang_backend':SimpleNamespace(SGLangDFlashBackend=Native),
                 'experiments.prefix_guard.qualify':SimpleNamespace(WorkLog=WorkLog,pin_retained_graph_runtime=pin,
                     require_runtime_pin=lambda:None,run_startup=startup)}
        from . import evaluate as E
        from experiments.ok_continuation import resume as R
        stack=ExitStack();stack.enter_context(patch.dict(sys.modules,modules))
        stack.enter_context(patch.object(R,'load_snapshot',side_effect=load))
        stack.enter_context(patch.object(R,'verify_live_resume',side_effect=verify))
        stack.enter_context(patch.object(E,'evaluate',side_effect=evaluate))
        stack.enter_context(patch.object(torch,'load',return_value={}))
        for name in ('reset_peak_memory_stats','empty_cache'):stack.enter_context(patch.object(torch.cuda,name))
        for name in ('max_memory_allocated','max_memory_reserved'):stack.enter_context(patch.object(torch.cuda,name,return_value=0))
        return stack,events

    def test_exact_restore_precedes_contract_change_and_training_closes_before_native(self):
        stack,events=self.runtime()
        with stack:C.run(self.root)
        self.assertLess(events.index('exact_live_resume'),events.index('disable_correction_after_resume'))
        self.assertLess(events.index('trainer_closed'),events.index('native_loaded'))
        self.assertEqual(events.count('train_event2'),1)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual((result['global_optimizer_step'],result['new_optimizer_steps'],result['new_presentations']),(256,128,256))
        self.assertEqual(result['actual_global_Adam_step'],256)
        self.assertEqual(result['evaluation_totals']['returned_generations'],768)

    def test_resume_failure_stops_before_contract_change_training_or_serving(self):
        stack,events=self.runtime(resume_failure=True)
        with stack,self.assertRaisesRegex(ValueError,'Live resume'):C.run(self.root)
        for step in ('disable_correction_after_resume','train_event2','native_loaded'):self.assertNotIn(step,events)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['evaluation_totals']['unattempted_generations'],768)

    def test_first_A_failure_preserves_completed_update_and_stops(self):
        stack,events=self.runtime(first_A_failure=True)
        with stack,self.assertRaisesRegex(RuntimeError,'first A mismatch'):C.run(self.root)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['actual_global_Adam_step'],129)
        self.assertEqual(result['actual_K_presentations_completed'],2)
        cost=json.loads((self.root/'campaign_001/COSTS.json').read_text())
        self.assertEqual(cost['off']['failure_value'],'nan')
        self.assertNotIn('native_loaded',events)

    def test_native_failure_keeps_second_panel_unattempted(self):
        stack,events=self.runtime(evaluation_failure=True)
        with stack,self.assertRaisesRegex(RuntimeError,'native mismatch'):C.run(self.root)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['evaluation_totals']['unattempted_generations'],767)
        self.assertEqual(result['panels']['additional']['status'],'unattempted')

    def test_real_two_panel_adapter768_only_one_gate(self):
        from . import evaluate as E
        actual=E.evaluate;backend=Backend(self.plan['panels'][0]['requests'])
        def wrapped(backend,requests,*args,**kwargs):
            backend.requests=requests;return actual(backend,requests,*args,**kwargs)
        progress={'status':'running','panels':{x['name']:C.unattempted_panel() for x in self.plan['panels']}}
        with patch.object(E,'evaluate',side_effect=wrapped):
            C.evaluate_panels(backend,self.plan['panels'],{'KRC2_OFF':'/KRC2_OFF','KRC2':'/KRC2'},self.root,progress)
        self.assertEqual(len(backend.executions),768)
        self.assertEqual(progress['evaluation_totals']['completed_valid_generations'],768)
        self.assertTrue((self.root/'evaluation/development/QUALIFICATION.json').exists())
        self.assertFalse((self.root/'evaluation/additional/QUALIFICATION.json').exists())


if __name__=='__main__':unittest.main()
