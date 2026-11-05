"""CPU orchestration fixtures; no native generation or optimizer work is run."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from . import campaign as C


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class Harness:
    def __init__(self, root):
        self.root, self.calls, self.trainers = Path(root), [], []
        self.fail_feedback = self.fail_training = None
        self.fail_evaluation = self.fail_preparation = self.tamper_pool = False
        self.fail_backend_close = self.fail_trainer_close = self.fail_storage = False
        self.fail_storage_init = False
        self.backend = None
        self.plan = dict(new_budget=dict(C.NEW_BUDGET), stop_step=1024, inherited_horizon=7424,
            no_automatic_retry=True, config=str(self.root/'config.yaml'), target='target', draft='released',
            worker_python='python', initial={}, windows=[], panels=[])
        (self.root/'config.yaml').write_text('qualified configuration\n')
        for arm in ('O', 'K'):
            state = self.root/'initial'/arm/'state.pt'
            export = state.parent/'export'
            export.mkdir(parents=True)
            write(state, dict(arm=arm, step=512, presentations=1024,
                events=[dict(number=n, features=[dict(record_id=f'old_{n}')]) for n in range(1,5)]))
            (export/'model.safetensors').write_bytes(arm.encode())
            self.plan['initial'][arm] = dict(state=str(state), export=str(export), receipt=str(state.parent/'RESULT.json'))
        for n in (5, 6, 7, 8):
            self.plan['windows'].append(dict(event_index=n, event_id=f'event_{n:03d}', requests=[
                dict(request_id=f'train_{i:05d}', request_index=i+1, prompt_ids=[i+1], max_new_tokens=2048)
                for i in range((n-1)*128, n*128)]))
        for panel_index, name in enumerate(('development', 'additional', 'fresh')):
            requests = [dict(request_id=f'{name}_{i}', panel_ordinal=i, original_index=9000+panel_index*64+i,
                prompt_ids=[9000+panel_index*64+i], max_new_tokens=2048, stop_token_ids=[99]) for i in range(64)]
            panel = dict(name=name, requests=requests, canonical_mode='fresh' if name == 'fresh' else 'retained')
            if name == 'fresh':
                panel['selection_seed'] = 20261029
            else:
                for request in requests:
                    request.update(response_ids=[11,99], finish_reason='eos',
                        canonical_source_method='F' if name == 'development' else 'K1')
                panel['historical_counts'] = {arm: {r['request_id']: 1 for r in requests} for arm in ('O4', 'K4')}
            self.plan['panels'].append(panel)
        self.seal()

    def seal(self):
        write(self.root/'sealed/PLAN.json', self.plan)

    def backend_class(self):
        h = self
        class Backend:
            def __init__(self, target, draft, **kwargs):
                assert (target,draft) == ('target','released')
                assert kwargs['serving_gpu'] == '0' and kwargs['context_limit'] == 4096 and kwargs['block_size'] == 16
                self.loading_s, self.checkpoint_id, self.feedback_count, self.closed = .1, None, 0, False
                h.backend = self
            def publish(self, checkpoint):
                self.checkpoint_id = checkpoint
                h.calls.append(('publish',checkpoint))
                return dict(errors=[])
            def check_weights(self, checkpoint):
                return dict(errors=[])
            def generate(self, prompt, **kwargs):
                assert kwargs['purpose'] == 'serving' and kwargs['max_new_tokens'] == 2048
                assert all(t.closed for t in h.trainers)
                self.feedback_count += 1
                h.calls.append(('generate',self.checkpoint_id,kwargs['request_id']))
                if self.feedback_count == h.fail_feedback:
                    raise RuntimeError('native interrupted')
                record = dict(record_id=kwargs['request_id'], prompt_ids=prompt, response_ids=[11,99],
                    checkpoint_id=self.checkpoint_id, metadata=dict(request_latency_s=.1,
                        capture=dict(capture_host_s=.01, feature_persistence_s=.01, cache_maintenance_s=.001)))
                write(kwargs['save_to'],record)
                return record
            def prepare_feedback(self, record, path):
                if h.fail_preparation:
                    return dict(status='ineligible_short_response')
                path.write_bytes(record['record_id'].encode())
                return dict(status='prepared', feature_path=str(path), teacher_preparation_s=.1, feature_persistence_s=.01)
            def close(self):
                self.closed = True
                if h.fail_backend_close:
                    raise RuntimeError('backend cleanup failed')
        return Backend

    def trainer_class(self):
        h = self
        class Trainer:
            def __init__(self, config, output, *, resume_state, arm, completed_event):
                assert config == h.plan['config']
                self.state, self.output, self.arm = read(resume_state), Path(output), arm
                assert self.state['arm'] == arm and self.state['step'] == completed_event*128
                self.n, self.closed = completed_event+1, False
                self.events_root, self.latest = self.output/'events', self.output/'training_state.pt'
                h.calls.append(('resume',arm,self.n,str(resume_state),self.state['step']))
                h.trainers.append(self)
            def status(self):
                return dict(global_step=self.state['step'], presentations=self.state['presentations'], closed=self.closed)
            def train_event(self, event_id, features, *, passes, presentation_schedule):
                assert event_id == f'event_{self.n:03d}' and passes == 2 and len(features) == 128
                assert (presentation_schedule is None) == (self.arm == 'O')
                h.calls.append(('train',self.arm,self.n,tuple(features),deepcopy(presentation_schedule)))
                if h.fail_training == (self.arm,self.n):
                    self.state['step'] += 3
                    self.state['presentations'] += 7
                    raise RuntimeError('training interrupted')
                start = self.state['step']
                self.state['step'] += 128
                self.state['presentations'] += 256
                self.state['events'].append(dict(number=self.n,features=[dict(record_id=Path(p).stem) for p in features]))
                export = self.events_root/event_id/'export'
                export.mkdir(parents=True)
                (export/'model.safetensors').write_bytes(f'{self.arm}{self.n}'.encode())
                pending = self.latest.with_suffix('.pending')
                write(pending,self.state)
                pending.replace(self.latest)
                result = dict(status='completed',event_id=event_id,objective=C.OBJECTIVES[self.arm],
                    start_step=start,end_step=self.state['step'],optimizer_steps=128,response_presentations=256,
                    full_state_path=str(self.latest),export_path=str(export),features=list(features),
                    records=[dict(record_id=Path(features[i%128]).stem,pass_index=i//128) for i in range(256)],
                    costs=dict(training_s=.2),objective_work=dict(current_forward=256))
                write(export.parent/'RESULT.json',result)
                if h.tamper_pool and self.arm == 'O':
                    Path(features[0]).write_bytes(b'changed')
                return result
            def close(self):
                self.closed = True
                h.calls.append(('trainer_close',self.arm,self.n))
                if h.fail_trainer_close:
                    raise RuntimeError('trainer cleanup failed')
        return Trainer

    def load_snapshot(self,state,export,*,objective,config_text,completed_event):
        value = read(state)
        assert value['arm'] == {'osd':'O','osd_first_error':'K'}[objective]
        assert value['step'] == completed_event*128 and value['presentations'] == completed_event*256
        assert [e['number'] for e in value['events']] == list(range(1,completed_event+1))
        assert Path(export,'model.safetensors').is_file() and config_text == 'qualified configuration\n'
        return value,dict(status='passed')

    def verify_live(self,trainer,state,receipt_path):
        assert trainer.state == state
        write(receipt_path,dict(status='passed'))

    def validate_receipt(self,result,*,event_number,objective,reference_schedule,trainer_status):
        assert result['start_step'] == (event_number-1)*128 and result['end_step'] == event_number*128
        assert result['objective'] == objective and trainer_status['global_step'] == event_number*128
        if objective == 'osd_first_error':
            assert result['features'] == reference_schedule['features'] and result['records'] == reference_schedule['records']
        else:
            assert reference_schedule is None
        return dict(status='passed')

    def evaluate(self,backend,requests,checkpoints,root,*,event_number,panel_name,canonical_mode,qualification_requests):
        assert all(t.closed for t in self.trainers)
        assert requests == next(p['requests'] for p in self.plan['panels'] if p['name'] == panel_name)
        assert set(checkpoints) == {'O','K'}
        assert qualification_requests == (8 if (event_number,panel_name) == (6,'development') else 0)
        h_event = [('evaluate',event_number,panel_name,deepcopy(checkpoints),qualification_requests)]
        self.calls.extend(h_event)
        count = 17 if self.fail_evaluation else 384
        status = dict(status='failed' if self.fail_evaluation else 'completed',returned_generations=count,
            attempted_generations=count+int(self.fail_evaluation),unknown_generation_attempts=int(self.fail_evaluation),
            completed_valid_generations=count,invalid_generations=0,pending_canonical_generations=0)
        write(root/'STATUS.json',status)
        if self.fail_evaluation:
            raise RuntimeError('evaluation interrupted')
        return status

    def storage_class(self):
        h = self
        class Storage:
            def __init__(self,plan,output):
                assert plan == h.plan
                if h.fail_storage_init:
                    raise RuntimeError('storage initialization failed')
            def admit_collection(self,window):
                h.calls.append(('admit_collection',window['event_index']))
                if h.fail_storage:
                    raise RuntimeError('storage admission failed')
            def admit_preparation(self,window,records):
                assert len(records) == 128
            def admit_training(self,arm,event,pool,parents):
                assert Path(parents[arm]['export'],'model.safetensors').is_file()
            def after_event(self,event,parents,pool,*,evaluated):
                assert all(t.closed for t in h.trainers)
                h.calls.append(('after_event',event,evaluated))
            def finalize(self,parents):
                h.calls.append(('finalize',))
        return Storage

    @contextmanager
    def patched(self):
        modules = {
            'online_feedback.sglang_backend':module('online_feedback.sglang_backend',SGLangDFlashBackend=self.backend_class()),
            'online_feedback.qualify_sglang':module('online_feedback.qualify_sglang',check_recipe=lambda config:None),
            'experiments.ok_events_5_8.evaluate':module('experiments.ok_events_5_8.evaluate',evaluate=self.evaluate),
            'experiments.ok_events_5_8.storage':module('experiments.ok_events_5_8.storage',StorageLifecycle=self.storage_class()),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,modules))
            for name,value in dict(OKContinuationTrainer=self.trainer_class(),load_snapshot=self.load_snapshot,
                    verify_live_resume=self.verify_live,verify_saved_boundary=self.verify_live,
                    validate_event_receipt=self.validate_receipt,publication_errors=lambda r,c:r['errors'],
                    weight_check_errors=lambda r,c:r['errors']).items():
                stack.enter_context(patch.object(C,name,value))
            for name,value in dict(is_initialized=lambda:False,reset_peak_memory_stats=lambda:None,
                    max_memory_allocated=lambda:0,max_memory_reserved=lambda:0).items():
                stack.enter_context(patch.object(torch.cuda,name,value))
            yield C


class CampaignTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.h = Harness(root.name)

    def status(self):
        return read(self.h.root/'campaign_001/STATUS.json')

    def test_full_budget_causal_pools_own_resumes_retention_and_closed_training(self):
        originals = {arm:Path(source['state']).read_bytes() for arm,source in self.h.plan['initial'].items()}
        with self.h.patched() as driver:
            driver.run(self.h.root)
        status = self.status()
        self.assertEqual((status['status'],status['optimizer_steps'],status['presentations']),('completed',1024,2048))
        self.assertEqual((status['feedback_returned'],status['evaluation_returned'],status['evaluation_valid']),(512,1920,1920))
        self.assertEqual(status['arm_counters'],dict(O=1024,K=1024))
        evaluations = [c for c in self.h.calls if c[0] == 'evaluate']
        self.assertEqual([(c[1],c[2]) for c in evaluations],[(6,'development'),(6,'additional'),(8,'development'),(8,'additional'),(8,'fresh')])
        for n in (5,6,7,8):
            calls = [c for c in self.h.calls if c[0] == 'train' and c[2] == n]
            self.assertEqual([c[1] for c in calls],['O','K'])
            self.assertEqual(calls[0][3],calls[1][3])
            feedback = [c for c in self.h.calls if c[0] == 'generate' and c[2] in {r['request_id'] for r in self.h.plan['windows'][n-5]['requests']}]
            expected = self.h.plan['initial']['O']['export'] if n == 5 else str(self.h.root/f'campaign_001/O/events/event_{n-1:03d}/export')
            self.assertEqual({c[1] for c in feedback},{expected})
            after = self.h.calls.index(('after_event',n,n in (6,8)))
            self.assertTrue(all(self.h.calls.index(c) < after for c in evaluations if c[1] == n))
        for arm in ('O','K'):
            self.assertEqual(Path(self.h.plan['initial'][arm]['state']).read_bytes(),originals[arm])
            for n in (6,8):
                retained = self.h.root/f'campaign_001/{arm}/events/event_{n:03d}/training_state.pt'
                self.assertEqual(read(retained)['step'],n*128)
        self.assertTrue(self.h.backend.closed and all(t.closed for t in self.h.trainers))

    def test_plan_scope_overlap_and_horizon_fail_before_native_loading(self):
        mutations = [lambda p:p.update(stop_step=1152),lambda p:p.update(inherited_horizon=1024),
            lambda p:p['new_budget'].update(feedback=513),lambda p:p['windows'].reverse(),
            lambda p:p['windows'][0]['requests'][0].update(prompt_ids=p['panels'][2]['requests'][0]['prompt_ids'])]
        original = deepcopy(self.h.plan)
        for mutate in mutations:
            self.h.plan = deepcopy(original)
            mutate(self.h.plan)
            self.h.seal()
            with self.h.patched(),self.assertRaises(ValueError):
                C.run(self.h.root)
            self.assertIsNone(self.h.backend)

    def test_interrupted_feedback_and_no_automatic_retry(self):
        self.h.fail_feedback = 17
        with self.h.patched(),self.assertRaisesRegex(RuntimeError,'native interrupted'):
            C.run(self.h.root)
        status = self.status()
        self.assertEqual((status['feedback_attempted'],status['feedback_returned'],status['feedback_unknown']),(17,16,1))
        self.assertEqual(status['optimizer_steps'],0)
        self.assertFalse(self.h.trainers)
        with self.h.patched(),self.assertRaises(FileExistsError):
            C.run(self.h.root)

    def test_partial_training_counts_and_cleanup_preserve_primary_failure(self):
        self.h.fail_training,self.h.fail_trainer_close = ('O',5),True
        with self.h.patched(),self.assertRaisesRegex(RuntimeError,'training interrupted'):
            C.run(self.h.root)
        status = self.status()
        self.assertEqual((status['optimizer_steps'],status['presentations']),(3,7))
        self.assertTrue(status['training_interrupted'])
        self.assertTrue((self.h.root/'campaign_001/resume_005_O/CLEANUP_ERROR.json').is_file())

    def test_evaluation_interruption_stops_before_event7_preserving_event6(self):
        self.h.fail_evaluation = True
        with self.h.patched(),self.assertRaisesRegex(RuntimeError,'evaluation interrupted'):
            C.run(self.h.root)
        status = self.status()
        self.assertEqual((status['feedback_returned'],status['optimizer_steps']),(256,512))
        self.assertEqual((status['evaluation_returned'],status['evaluation_attempted'],status['evaluation_unknown']),(17,18,1))
        self.assertEqual(status['events_completed'],[5,6])
        self.assertTrue(all(t.closed for t in self.h.trainers))

    def test_shared_pool_mutation_stops_before_K(self):
        self.h.tamper_pool = True
        with self.h.patched(),self.assertRaisesRegex(ValueError,'pool changed'):
            C.run(self.h.root)
        self.assertEqual([(t.arm,t.n) for t in self.h.trainers],[('O',5)])
        self.assertEqual(self.status()['optimizer_steps'],128)

    def test_ineligible_feedback_is_not_replaced(self):
        self.h.fail_preparation = True
        with self.h.patched(),self.assertRaisesRegex(ValueError,'no substitution'):
            C.run(self.h.root)
        self.assertEqual(self.status()['feedback_returned'],128)
        self.assertFalse(self.h.trainers)

    def test_storage_failure_precedes_first_generation(self):
        self.h.fail_storage = True
        with self.h.patched(),self.assertRaisesRegex(RuntimeError,'storage admission'):
            C.run(self.h.root)
        self.assertEqual(self.status()['feedback_attempted'],0)
        self.assertTrue(self.h.backend.closed)

    def test_storage_initialization_failure_has_durable_failed_status(self):
        self.h.fail_storage_init = True
        with self.h.patched(),self.assertRaisesRegex(RuntimeError,'storage initialization'):
            C.run(self.h.root)
        self.assertEqual(self.status()['status'],'failed')
        self.assertEqual(self.status()['feedback_attempted'],0)
        self.assertIsNone(self.h.backend)

    def test_shared_pool_rejects_duplicate_feature_or_response_identity(self):
        pool = dict(features=[str(n) for n in range(128)],record_ids=[str(n) for n in range(128)],
            response_generations=128,immutable=True)
        for field in ('features','record_ids'):
            bad = deepcopy(pool)
            bad[field][1] = bad[field][0]
            with self.assertRaisesRegex(ValueError,'128-response feedback pool'):
                C.verify_pool(bad)


if __name__ == '__main__':
    unittest.main()
