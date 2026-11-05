"""CPU continuation orchestration: exact resume before manager, bounded work and failure."""
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


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def correction_report():
    return dict(status='completed',start_step=128,native_global_step=256,
        first_window=dict(status='passed',extra_optimizer_steps=0,extra_forwards=0),
        counts=dict(K_backward_attempted=256,K_backward_completed=256,
        K_backward_failed=0,Adam_attempted=128,Adam_completed=128,Adam_failed=0,
        presentations_completed=256,windows_completed=128),
        work=dict(original_response_forwards=256,pre_reference_response_forwards=256,
                  post_k_reference_response_forwards=256,post_c_reference_response_forwards=0,
                  post_k_learning_response_forwards=0,post_c_learning_response_forwards=0,
                  violation_backward_completed=200,violation_backward_failed=0,direct_corrections=100),
        steps=[dict(step=i,global_step=128+i) for i in range(1,129)],probes=[])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.o2=event(2,'osd')
        self.k2=event(2,'osd_first_error');self.k2['checkpoint_path']='K2'
        self.krc1=event(1,'osd_first_error');self.krc1['checkpoint_path']='KRC1'

    def test_local_view_only_changes_counter_construction_not_scientific_source(self):
        before=deepcopy(self.o2)
        schedule=P.build_continuation_schedule(self.o2,evaluation_ids=['eval'])
        self.assertEqual(self.o2,before)
        self.assertEqual(schedule['continuation']['start_global_step'],128)
        self.assertEqual(schedule['windows'][0]['step'],1)
        self.assertEqual(schedule['windows'][-1]['step'],128)
        self.assertEqual(schedule['windows'][63]['reference'],schedule['windows'][0]['learning'])
        self.assertEqual(schedule['windows'][127]['reference'],schedule['windows'][64]['learning'])
        self.assertEqual(schedule['windows'][0]['learning'][0]['sampled_anchors'],self.o2['records'][0]['sampled_anchors'])
        self.assertEqual(P.validate_continuation_schedule(schedule,self.o2)['global_steps'],[129,256])
        self.assertTrue(P.validate_retained(self.krc1,self.o2,self.k2)['passed'])

    def test_wrong_global_counter_pool_or_evaluation_overlap_rejected(self):
        for mutate in (lambda x:x['records'][1].update(optimizer_step_after=1),
                       lambda x:x['records'].pop(),lambda x:x.update(start_step=0)):
            changed=deepcopy(self.o2);mutate(changed)
            with self.assertRaises(ValueError):P.build_continuation_schedule(changed)
        with self.assertRaisesRegex(ValueError,'Evaluation'):P.build_continuation_schedule(self.o2,evaluation_ids=['r000'])
        changed=deepcopy(self.k2);changed['records'][0]['sampled_anchors'][0]['positions'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'anchors'):P.validate_retained(self.krc1,self.o2,changed)

    def test_capped_feature_and_complete128_pool_are_preserved(self):
        self.o2['features'][0].update(record_id='train_00187',finish_reason='length',response_tokens=2048)
        for row in self.o2['records']:
            if row['record_id']=='r000':row['record_id']='train_00187'
        schedule=P.build_continuation_schedule(self.o2)
        self.assertEqual(len(schedule['features']),128)
        self.assertEqual(schedule['features'][0],self.o2['features'][0])
        self.assertEqual(sum('train_00187' in w['learning_record_ids'] for w in schedule['windows']),2)
        self.assertEqual(sum('train_00187' in w['reference_record_ids'] for w in schedule['windows']),2)

    def test_core_budget768_no_sparse_diagnostics_and_event256_boundary(self):
        report=correction_report();self.assertEqual(P.check_correction_report(report)['core_response_forwards'],768)
        for mutate in (lambda x:x['work'].update(post_k_learning_response_forwards=2),
                       lambda x:x['probes'].append({'step':1}),
                       lambda x:x['counts'].update(Adam_completed=129),
                       lambda x:x.update(native_global_step=128),
                       lambda x:x['first_window'].update(extra_forwards=2)):
            changed=deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):P.check_correction_report(changed)
        status=dict(global_step=256,presentations=512,completed_events=['event_001','event_002'],
                    accumulation_remainder=0,objective='osd_first_error',horizon=7424)
        self.assertEqual(P.check_event(self.k2,self.o2,status)['end_step'],256)
        status['global_step']=128
        with self.assertRaisesRegex(ValueError,'status'):P.check_event(self.k2,self.o2,status)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.o2=event(2,'osd');self.k2=dict(event(2,'osd_first_error'),checkpoint_path='retained-K2')
        self.krc1=dict(event(1,'osd_first_error'),checkpoint_path='retained-KRC1')
        self.plan=dict(config=str(self.root/'config.yaml'),startup_ordinary_receipt=str(self.root/'O1.json'),
            pinned_initial='released-startup-only.pt',krc1_state='complete-KRC1-state.pt',
            krc1_receipt=str(self.root/'KRC1.json'),event2_ordinary_receipt=str(self.root/'O2.json'),
            event2_k_receipt=str(self.root/'K2.json'),reference_schedule=str(self.root/'schedule.json'),
            retained_k2_checkpoint='retained-K2',target='target',draft='draft',worker_python='worker',
            panels=[panel(),panel('additional')])
        save(self.root/'sealed/PLAN.json',self.plan);(self.root/'config.yaml').write_text('exact config')
        for name,value in [('O1.json',event()),('O2.json',self.o2),('K2.json',self.k2),('KRC1.json',self.krc1)]:
            save(self.root/name,value)
        self.schedule=P.build_continuation_schedule(self.o2);save(self.root/'schedule.json',self.schedule)

    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())

    def runtime(self,*,resume_failure=False,window_failure=False,evaluation_failure=False):
        outer=self;events=[];current=[None];report=correction_report()
        receipt=dict(self.k2,checkpoint_path='new-KRC2',costs={'training_s':1},objective_work={})
        state={'identity':'KRC1 complete state with own RNG'}
        class Trainer:
            def __init__(self,config,root,**kwargs):
                outer.assertEqual(kwargs,dict(resume_state='complete-KRC1-state.pt',objective='osd_first_error',
                                               reduction='response_mean'))
                events.append('restore_KRC1');current[0]=self
                self.global_step=128;self.presentations=256;self.closed=self.failed=False
            def train_event(self,event,paths,*,passes,presentation_schedule):
                outer.assertEqual((event,len(paths),passes),('event_002',128,2))
                outer.assertEqual(presentation_schedule,outer.o2)
                events.append('train_event2')
                if window_failure:
                    report['status']='failed';report['counts'].update(Adam_completed=1,presentations_completed=2)
                    report['failure_value']=float('nan')
                    raise RuntimeError('correction integrity')
                self.global_step=256;self.presentations=512
                return receipt
            def close(self):self.closed=True;events.append('trainer_closed')
            def status(self):
                return dict(global_step=self.global_step,presentations=self.presentations,
                    completed_events=['event_001','event_002'] if self.global_step==256 else ['event_001'],
                    accumulation_remainder=0,objective='osd_first_error',horizon=7424,closed=self.closed,failed=self.failed)
        class Manager:
            def __init__(self,trainer,*,schedule,feature_paths,output_dir):
                outer.assertIn('exact_live_resume',events);events.append('manager_no_probes')
                outer.assertEqual(schedule,outer.schedule)
                outer.assertEqual(feature_paths,{row['record_id']:Path(row['path']) for row in outer.o2['features']})
            def __enter__(self):return self
            def close(self):events.append('manager_closed')
            def report(self):return report
        class Backend:
            loading_s=1
            def __init__(self,*args,**kwargs):
                outer.assertTrue(current[0].closed);events.append('native_loaded')
                outer.assertEqual((kwargs['block_size'],kwargs['context_limit']),(16,4096))
            def close(self):events.append('backend_closed')
        class WorkLog:
            def __init__(self,path):self.counts=Counter()
        def load(path,export,**kwargs):
            events.append('snapshot_and_export_exact');outer.assertEqual((path,export),('complete-KRC1-state.pt','retained-KRC1'))
            outer.assertEqual(kwargs,dict(objective='osd_first_error',config_text='exact config',completed_event=1))
            return state,dict(status='passed',wall_s=0,export_exact=True)
        def verify(trainer,expected,*,receipt_path):
            outer.assertIs(expected,state);events.append('exact_live_resume')
            if resume_failure:raise ValueError('Live resume differs')
            return dict(exact_resume=True)
        def startup(config,ordinary,pin,log,root):
            events.append('startup_two_zero_Adam');outer.assertNotIn('restore_KRC1',events)
            log.counts.update(startup_response_evaluation_completed=2,startup_backward_completed=2)
            return dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
                        exact_initial_state_restored=True,separate_disposable_runtime=True)
        def pin(root):events.append('donated_buffer_false');return {'effective':False}
        def evaluate(backend,requests,checkpoints,root,*,expected_control_counts,historical_counts,qualification_requests):
            events.append('evaluate_'+root.name);outer.assertEqual(checkpoints,{'K2':'retained-K2','KRC2':'new-KRC2'})
            outer.assertEqual(qualification_requests,8 if root.name=='development' else 0)
            outer.assertEqual(expected_control_counts is None,root.name=='additional')
            value=dict(C.unattempted_panel(),status='completed',returned_generations=384,attempted_generations=384,
                       completed_valid_generations=384,unattempted_generations=0)
            if evaluation_failure:
                value.update(status='failed',returned_generations=1,attempted_generations=1,
                             completed_valid_generations=0,invalid_generations=1,unattempted_generations=383)
                save(root/'STATUS.json',value);raise RuntimeError('native mismatch')
            save(root/'STATUS.json',value);return value
        modules={'online_feedback.krc_continuation':SimpleNamespace(KRCContinuationTrainer=Trainer,KRCContinuationEvent=Manager),
                 'online_feedback.sglang_backend':SimpleNamespace(SGLangDFlashBackend=Backend),
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

    def test_fullstate_resume_precedes_manager_and_evaluation_follows_close(self):
        stack,events=self.runtime()
        with stack:C.run(self.root)
        self.assertEqual(events[:5],['donated_buffer_false','snapshot_and_export_exact','startup_two_zero_Adam','restore_KRC1','exact_live_resume'])
        self.assertLess(events.index('exact_live_resume'),events.index('manager_no_probes'))
        self.assertLess(events.index('trainer_closed'),events.index('native_loaded'))
        self.assertEqual(events.count('train_event2'),1)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual((result['global_optimizer_step'],result['new_optimizer_steps'],result['new_presentations']),(256,128,256))
        self.assertEqual(result['actual_global_Adam_step'],256)
        self.assertEqual(result['evaluation_totals']['returned_generations'],768)

    def test_resume_failure_stops_before_window_or_native(self):
        stack,events=self.runtime(resume_failure=True)
        with stack,self.assertRaisesRegex(ValueError,'Live resume'):C.run(self.root)
        self.assertNotIn('manager_no_probes',events);self.assertNotIn('train_event2',events);self.assertNotIn('native_loaded',events)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['evaluation_totals']['unattempted_generations'],768)

    def test_post_Adam_failure_preserves_actual_global129_and_nonfinite_evidence(self):
        stack,events=self.runtime(window_failure=True)
        with stack,self.assertRaisesRegex(RuntimeError,'correction integrity'):C.run(self.root)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['global_optimizer_step'],128)
        self.assertEqual(result['actual_global_Adam_step'],129)
        self.assertEqual(result['actual_K_presentations_completed'],2)
        cost=json.loads((self.root/'campaign_001/COSTS.json').read_text())
        self.assertEqual(cost['correction']['failure_value'],'nan')
        self.assertNotIn('native_loaded',events)

    def test_native_failure_preserves_unattempted_second_panel(self):
        stack,events=self.runtime(evaluation_failure=True)
        with stack,self.assertRaisesRegex(RuntimeError,'native mismatch'):C.run(self.root)
        result=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(result['evaluation_totals']['unattempted_generations'],767)
        self.assertEqual(result['panels']['additional']['status'],'unattempted')
        self.assertNotIn('evaluate_additional',events)

    def test_real_two_panel_adapter768_only_one_gate(self):
        from . import evaluate as E
        actual=E.evaluate;backend=Backend(self.plan['panels'][0]['requests'])
        def wrapped(backend,requests,*args,**kwargs):
            backend.requests=requests;return actual(backend,requests,*args,**kwargs)
        progress={'status':'running','panels':{x['name']:C.unattempted_panel() for x in self.plan['panels']}}
        with patch.object(E,'evaluate',side_effect=wrapped):
            C.evaluate_panels(backend,self.plan['panels'],{'K2':'/K2','KRC2':'/KRC2'},self.root,progress)
        self.assertEqual(len(backend.executions),768)
        self.assertEqual(progress['evaluation_totals']['completed_valid_generations'],768)
        self.assertTrue((self.root/'evaluation/development/QUALIFICATION.json').exists())
        self.assertFalse((self.root/'evaluation/additional/QUALIFICATION.json').exists())


if __name__=='__main__':unittest.main()
