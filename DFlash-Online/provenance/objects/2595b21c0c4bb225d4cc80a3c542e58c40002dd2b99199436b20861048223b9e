"""CPU orchestration checks for the single KC event and retained K comparison."""
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

from . import campaign as C
from experiments.prefix_guard.qualify import persist_comparison


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def correction_report():
    return dict(status='completed', counts=dict(K_backward_attempted=256,K_backward_completed=256,K_backward_failed=0,
                Adam_attempted=128,Adam_completed=128,Adam_failed=0,presentations_completed=256,windows_completed=128),
                work=dict(original_response_forwards=256,post_k_response_forwards=256,
                         post_c_response_forwards=6,violation_backward_completed=200,violation_backward_failed=0,
                         direct_corrections=100),
                steps=[dict(step=i) for i in range(1,129)],
                probes=[dict(step=i) for i in (1,64,128)])


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ordinary=dict(status='completed',objective='osd',passes=2,optimizer_steps=128,
            response_presentations=256,start_step=0,end_step=128,
            features=[dict(path=f'feature-{i}.pt') for i in range(128)],
            records=[dict(record_id=f'r{i}',**{'pass':i//128},sampled_anchors=[i],
                          optimizer_stepped=bool(i%2),optimizer_step_after=(i+1)//2) for i in range(256)])
        self.k=dict(self.ordinary,objective='osd_first_error',checkpoint_path='retained-K')
        self.reference=dict(status='completed',completed_optimizer_steps=1,completed_presentations=2,
            core=dict(K_update_norm=1.0,responses=[{'record_id':'r0'},{'record_id':'r1'}],
                      post_k={'responses':[{'old_prefix_tokens_lost':137},{'old_prefix_tokens_lost':16}]},
                      correction={'status':'applied','V':.003},
                      post_c={'responses':[{'old_prefix_tokens_lost':100},{'old_prefix_tokens_lost':13}]}))
        self.plan=dict(config='qualified-config',ordinary_receipt=str(self.root/'O.json'),
            k_receipt=str(self.root/'K.json'),manifest=str(self.root/'manifest.json'),
            reference_diagnostic=str(self.root/'reference.json'),pinned_initial='original-initial.pt',
            target='target',draft='draft',worker_python='worker')
        save(self.root/'sealed/PLAN.json',self.plan)
        save(self.root/'O.json',self.ordinary); save(self.root/'K.json',self.k)
        save(self.root/'reference.json',self.reference)
        save(self.root/'manifest.json',{'eval_requests':[{'ordinal':i} for i in range(64)]})

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_window_checks_corrected_records_and_keeps_failure(self):
        path=self.root/'first.json'
        C.first_window_gate(deepcopy(self.reference['core']),self.reference,path)
        changed=deepcopy(self.reference['core'])
        changed['post_c']['responses'][0]['old_prefix_tokens_lost']+=1
        with self.assertRaises(AssertionError):
            C.first_window_gate(changed,self.reference,path)
        self.assertFalse(json.loads(path.read_text())['exact_equal'])

    def test_budget_is_518_forwards_and_only_three_corrected_rechecks(self):
        report=correction_report(); C.check_correction_report(report)
        report['work']['post_c_response_forwards']=8
        with self.assertRaisesRegex(ValueError,'518'):
            C.check_correction_report(report)
        report=correction_report(); report['work']['direct_corrections']=129
        with self.assertRaisesRegex(ValueError,'budget'):
            C.check_correction_report(report)

    def test_schedule_cannot_change_and_only_one_event_is_permitted(self):
        receipt=dict(self.k)
        C.check_event(receipt,self.ordinary)
        receipt=deepcopy(receipt); receipt['records'][0]['sampled_anchors']=[999]
        with self.assertRaisesRegex(ValueError,'trajectory'):
            C.check_event(receipt,self.ordinary)
        receipt=dict(self.k,end_step=256)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            C.check_event(receipt,self.ordinary)

    def runtime(self,*,first_window_failure=False):
        events=[]; outer=self; manager_instance=None
        report=correction_report()
        if first_window_failure:
            report['failed_value']=float('nan')
        receipt=dict(self.k,checkpoint_path='new-KC',costs={'training_s':1},objective_work={})
        class Trainer:
            def __init__(self,config,root,**kwargs):
                events.append('fresh_KC');self.global_step=self.presentations=0
                self.closed=self.failed=False
                outer.assertEqual(kwargs,dict(objective='osd_first_error',reduction='response_mean',ranking_lambda=0))
            def train_event(self,event,features,*,passes,presentation_schedule):
                events.append('one_train_event')
                outer.assertEqual((event,len(features),passes),('event_001',128,2))
                outer.assertEqual(presentation_schedule,outer.ordinary)
                current=deepcopy(outer.reference['core'])
                if first_window_failure:
                    current['post_c']['responses'][0]['old_prefix_tokens_lost']+=1
                self.global_step,self.presentations=1,2
                if first_window_failure:
                    self.global_step,self.presentations=0,1
                    report['counts'].update(Adam_completed=1,presentations_completed=2)
                manager_instance.callback(current)
                events.append('first_window_passed')
                self.global_step,self.presentations=128,256
                return receipt
            def close(self): self.closed=True;events.append('trainer_closed')
            def status(self): return dict(closed=self.closed,failed=self.failed)
        class Manager:
            def __init__(self,trainer,*,probe_steps,output_dir,first_window_check):
                nonlocal manager_instance
                manager_instance=self;self.callback=first_window_check
                outer.assertEqual(probe_steps,(1,64,128))
                report['status']='running'
            def __enter__(self): events.append('manager_installed');return self
            def report(self): return report
            def close(self):
                events.append('manager_closed')
                report['status']='completed'
        class Backend:
            loading_s=1
            def __init__(self,*args,**kwargs):
                outer.assertIn('trainer_closed',events);events.append('native_loaded')
                outer.assertEqual((kwargs['context_limit'],kwargs['block_size']),(4096,16))
            def close(self): events.append('backend_closed')
        class WorkLog:
            def __init__(self,path):self.counts=Counter()
        def pin(root):events.append('pin_false');return dict(effective=False)
        def startup(config,ordinary,pin,log,root):
            events.append('startup_two_zero_steps')
            outer.assertNotIn('fresh_KC',events)
            log.counts.update(startup_response_evaluation_completed=2,startup_backward_completed=2)
            return dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
                        exact_initial_state_restored=True,separate_disposable_runtime=True)
        def initialize(*args):events.append('initial_exact');return dict(exact_pinned_initial_state=True)
        def evaluate(backend,requests,checkpoints,root):
            events.append('evaluate_384')
            outer.assertEqual(checkpoints,{'K':'retained-K','KC':'new-KC'})
            outer.assertEqual(len(requests),64)
            return dict(status='completed',returned_generations=384)
        modules={
            'online_feedback.training':SimpleNamespace(SpecForgeEventTrainer=Trainer),
            'online_feedback.prefix_correction_event':SimpleNamespace(PrefixCorrectionEvent=Manager),
            'online_feedback.sglang_backend':SimpleNamespace(SGLangDFlashBackend=Backend),
            'experiments.prefix_guard.qualify':SimpleNamespace(WorkLog=WorkLog,
                pin_retained_graph_runtime=pin,require_runtime_pin=lambda:None,
                run_startup=startup,prepare_main=initialize),
        }
        from . import evaluate as E
        stack=ExitStack()
        stack.enter_context(patch.dict(sys.modules,modules))
        stack.enter_context(patch.object(C,'validate_scope'))
        # Use the real exact first-window comparison with its imported helper.
        modules['experiments.prefix_guard.qualify'].persist_comparison=persist_comparison
        stack.enter_context(patch.object(E,'evaluate',side_effect=evaluate))
        stack.enter_context(patch.object(torch,'load',return_value={}))
        for name in ('reset_peak_memory_stats','empty_cache'):
            stack.enter_context(patch.object(torch.cuda,name))
        for name in ('max_memory_allocated','max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda,name,return_value=0))
        return stack,events

    def test_main_keeps_startup_training_and_serving_separate(self):
        stack,events=self.runtime()
        with stack:C.run(self.root)
        self.assertEqual(events[:4],['pin_false','startup_two_zero_steps','fresh_KC','initial_exact'])
        self.assertEqual(events.count('one_train_event'),1)
        self.assertLess(events.index('first_window_passed'),events.index('evaluate_384'))
        status=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual((status['main_optimizer_steps'],status['main_presentations']),(128,256))
        self.assertEqual(status['new_qualification_optimizer_steps'],0)
        costs=json.loads((self.root/'campaign_001/COSTS.json').read_text())
        self.assertEqual(costs['startup']['optimizer_steps'],0)
        self.assertEqual(costs['correction']['work']['post_c_response_forwards'],6)

    def test_first_window_failure_stops_before_native_evaluation(self):
        stack,events=self.runtime(first_window_failure=True)
        with stack,self.assertRaises(AssertionError):C.run(self.root)
        self.assertNotIn('native_loaded',events)
        self.assertNotIn('first_window_passed',events)
        status=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['main_optimizer_steps'],0)
        self.assertEqual(status['native_adam_steps_completed'],1)
        self.assertEqual(status['native_K_presentations_completed'],2)
        self.assertTrue((self.root/'campaign_001/FIRST_WINDOW_REPRODUCTION.json').exists())
        costs=json.loads((self.root/'campaign_001/COSTS.json').read_text())
        self.assertEqual(costs['correction']['failed_value'],'nan')


if __name__=='__main__': unittest.main()
