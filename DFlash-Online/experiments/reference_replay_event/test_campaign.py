"""Bounded CPU orchestration checks; no model, optimizer or CUDA construction."""
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
from . import protocol as P
from .test_evaluate import panel, Backend
from experiments.reference_correction_event.test_reference_schedule import original_receipt
from experiments.prefix_guard.qualify import persist_comparison


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.ordinary=original_receipt()
        self.k=deepcopy(self.ordinary); self.k['objective']='osd_first_error'
        for i,row in enumerate(self.k['records']):
            row.update(learning_rate=.0003 if i%2 else None, metrics={'loss':.125},
                       metrics_scope='accumulation_window' if i%2 else 'presentation')
        self.schedule=P.build_replay_schedule(self.ordinary)
        self.plan=dict(config='config',ordinary_receipt=str(self.root/'O.json'),
            k_receipt=str(self.root/'K.json'),pinned_initial='initial.pt',
            reference_schedule='original-reference-schedule.json',replay_schedule=str(self.root/'schedule.json'),
            target='target',draft='draft',worker_python='worker',retained_krc_checkpoint='retained-KRC',
            panels=[panel(),panel('additional')])
        save(self.root/'sealed/PLAN.json',self.plan)
        save(self.root/'O.json',self.ordinary); save(self.root/'K.json',self.k)
        save(self.root/'schedule.json',self.schedule)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_A_exact_metrics_anchors_and_boundaries_without_extra_fields(self):
        rows=deepcopy(self.k['records'][:2]); rows[0].update(role='A',training_s=999)
        receipt=C.first_update_gate(rows,self.k,self.root/'gate.json')
        self.assertTrue(receipt['exact_equal'])
        self.assertIn('No retained post-update tensor equivalence',receipt['scope'])
        rows[1]['metrics']['loss']+=.001
        with self.assertRaises(AssertionError): C.first_update_gate(rows,self.k,self.root/'bad.json')
        self.assertFalse(json.loads((self.root/'bad.json').read_text())['exact_equal'])

    def runtime(self, first_failure=False, evaluation_failure=False):
        outer=self; events=[]; current=[None]
        replay={'counts':{'outer_windows_completed':128,'Adam_completed':256,'response_forwards_completed':512}}
        receipt=dict(checkpoint_path='new-KR',reference_replay=replay,costs={'training_s':1},objective_work={})
        class Trainer:
            def __init__(self,config,root,**kwargs):
                events.append('fresh_KR');current[0]=self
                self.global_step=self.presentations=0;self.closed=self.failed=False
                outer.assertEqual(kwargs,dict(objective='osd_first_error',reduction='response_mean',ranking_lambda=0))
            def train_replay_event(self,event,features,schedule,*,first_update_check):
                events.append('one_replay_event')
                outer.assertEqual((event,len(features)),('event_001',128))
                outer.assertEqual(schedule,outer.schedule)
                self.global_step,self.presentations=1,2
                rows=deepcopy(outer.k['records'][:2])
                if first_failure: rows[1]['metrics']['loss']+=.001
                first_update_check(rows); events.append('first_A_passed')
                self.global_step,self.presentations=256,512
                return receipt
            def replay_report(self):
                return {'counts':{'Adam_completed':self.global_step,'response_forwards_completed':self.presentations}}
            def status(self):return dict(closed=self.closed,failed=self.failed)
            def close(self):self.closed=True;events.append('trainer_closed')
        class Backend:
            loading_s=1
            def __init__(self,*args,**kwargs):
                outer.assertTrue(current[0].closed); events.append('native_loaded')
                outer.assertEqual((kwargs['block_size'],kwargs['context_limit']),(16,4096))
            def close(self):events.append('backend_closed')
        class WorkLog:
            def __init__(self,path):self.counts=Counter()
        def startup(config,ordinary,pinned,log,root):
            events.append('startup_two_zero_Adam')
            outer.assertNotIn('fresh_KR',events)
            log.counts.update(startup_response_evaluation_completed=2,startup_backward_completed=2)
            return dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
                        exact_initial_state_restored=True,separate_disposable_runtime=True)
        def pin(root):events.append('donated_buffer_false');return {'effective':False}
        def initialize(*args):events.append('exact_initial');return {'exact_pinned_initial_state':True}
        def evaluate(backend,requests,checkpoints,root,*,retained_krc_counts,qualification_requests):
            events.append('evaluate_'+root.name)
            outer.assertEqual(checkpoints,dict(KR='new-KR',KRC='retained-KRC'))
            outer.assertEqual(len(requests),64)
            outer.assertEqual(qualification_requests,8 if root.name=='development' else 0)
            status=dict(C.unattempted_panel(),status='completed',attempted_generations=384,
                        returned_generations=384,completed_valid_generations=384,unattempted_generations=0)
            if evaluation_failure:
                status.update(status='failed',attempted_generations=1,returned_generations=1,
                              completed_valid_generations=0,invalid_generations=1,unattempted_generations=383)
                save(root/'STATUS.json',status)
                raise RuntimeError('native gate failed')
            save(root/'STATUS.json',status)
            return status
        modules={
            'online_feedback.reference_replay_event':SimpleNamespace(ReferenceReplayEventTrainer=Trainer),
            'online_feedback.sglang_backend':SimpleNamespace(SGLangDFlashBackend=Backend),
            'experiments.prefix_guard.qualify':SimpleNamespace(WorkLog=WorkLog,run_startup=startup,
                pin_retained_graph_runtime=pin,require_runtime_pin=lambda:None,
                prepare_main=initialize,persist_comparison=persist_comparison)}
        from . import evaluate as E
        stack=ExitStack();stack.enter_context(patch.dict(sys.modules,modules))
        stack.enter_context(patch.object(P,'check_training',return_value={'passed':True}))
        stack.enter_context(patch.object(E,'evaluate',side_effect=evaluate))
        stack.enter_context(patch.object(torch,'load',return_value={}))
        for name in ('reset_peak_memory_stats','empty_cache'):
            stack.enter_context(patch.object(torch.cuda,name))
        for name in ('max_memory_allocated','max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda,name,return_value=0))
        return stack,events

    def test_one_event_512presentations256steps_then_two_panels_after_close(self):
        stack,events=self.runtime()
        with stack:C.run(self.root)
        self.assertEqual(events[:4],['donated_buffer_false','startup_two_zero_Adam','fresh_KR','exact_initial'])
        self.assertEqual(events.count('one_replay_event'),1)
        self.assertLess(events.index('trainer_closed'),events.index('evaluate_development'))
        self.assertLess(events.index('evaluate_development'),events.index('evaluate_additional'))
        status=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual((status['main_presentations'],status['main_optimizer_steps']),(512,256))
        self.assertEqual(status['evaluation_totals']['returned_generations'],768)
        self.assertEqual(status['evaluation_totals']['unattempted_generations'],0)
        costs=json.loads((self.root/'campaign_001/COSTS.json').read_text())
        self.assertEqual(costs['startup']['optimizer_steps'],0)
        self.assertEqual(costs['new_qualification_optimizer_steps'],0)

    def test_first_A_failure_durably_stops_before_native(self):
        stack,events=self.runtime(first_failure=True)
        with stack,self.assertRaises(AssertionError):C.run(self.root)
        self.assertNotIn('native_loaded',events)
        status=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(status['actual_Adam_steps_completed'],1)
        self.assertEqual(status['evaluation_totals']['unattempted_generations'],768)
        self.assertTrue((self.root/'campaign_001/FIRST_A_REPRODUCTION.json').exists())

    def test_first_panel_failure_keeps_second_panel_wholly_unattempted(self):
        stack,events=self.runtime(evaluation_failure=True)
        with stack,self.assertRaisesRegex(RuntimeError,'native gate failed'):C.run(self.root)
        self.assertNotIn('evaluate_additional',events)
        status=json.loads((self.root/'campaign_001/STATUS.json').read_text())
        self.assertEqual(status['panels']['additional']['status'],'unattempted')
        self.assertEqual(status['evaluation_totals']['unattempted_generations'],767)
        self.assertEqual(status['evaluation_totals']['invalid_generations'],1)
        self.assertEqual(status['evaluation_totals']['returned_generations'],1)

    def test_real_two_panel_evaluator_executes768_with_one48_gate(self):
        backend=Backend(self.plan['panels'][0]['requests'])
        # Switch only CPU fixture expectations between panels, not scientific inputs.
        from . import evaluate as E
        native_evaluate=E.evaluate
        def wrapped(backend,requests,*args,**kwargs):
            backend.requests=requests
            return native_evaluate(backend,requests,*args,**kwargs)
        progress={'status':'running','panels':{p['name']:C.unattempted_panel() for p in self.plan['panels']}}
        with patch.object(E,'evaluate',side_effect=wrapped):
            C.evaluate_panels(backend,self.plan['panels'],dict(KR='/KR',KRC='/KRC'),self.root,progress)
        self.assertEqual(progress['evaluation_totals']['returned_generations'],768)
        self.assertEqual(len(backend.executions),768)
        self.assertEqual(json.loads((self.root/'evaluation/development/QUALIFICATION.json').read_text())['generations'],48)
        self.assertFalse((self.root/'evaluation/additional/QUALIFICATION.json').exists())
        self.assertEqual(progress['panels']['additional']['qualification'],'not_required_prior_panel_gate')


if __name__=='__main__': unittest.main()
