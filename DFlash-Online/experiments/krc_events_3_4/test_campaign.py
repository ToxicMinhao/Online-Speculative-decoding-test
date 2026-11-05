"""Independent CPU integration checks for bounded KRC3/4 orchestration."""
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

from experiments.ok_continuation.test_resume import event, fixture, runtime
from online_feedback.rng import capture_rng
from . import campaign as C
from . import protocol as P
from .test_evaluate import Backend, panel


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def correction_report(number):
    start = (number - 1) * 128
    return dict(status='completed', start_step=start, native_global_step=start + 128,
        first_window=dict(status='passed', extra_optimizer_steps=0, extra_forwards=0),
        counts=dict(K_backward_attempted=256, K_backward_completed=256,
            K_backward_failed=0, Adam_attempted=128, Adam_completed=128, Adam_failed=0,
            presentations_completed=256, windows_completed=128),
        work=dict(original_response_forwards=256, pre_reference_response_forwards=256,
            post_k_reference_response_forwards=256, post_c_reference_response_forwards=0,
            post_k_learning_response_forwards=0, post_c_learning_response_forwards=0,
            violation_backward_completed=200, violation_backward_failed=0, direct_corrections=100),
        steps=[dict(step=i) for i in range(1, 129)], probes=[])


class ProtocolTests(unittest.TestCase):
    def test_both_local_schedules_keep_global_source_anchors_and_cyclic_pairs(self):
        events = []
        for number in (3, 4):
            ordinary = event(number, 'osd')
            before = deepcopy(ordinary)
            schedule = P.build_continuation_schedule(ordinary, number, evaluation_ids=['eval'])
            self.assertEqual(ordinary, before)
            self.assertEqual(schedule['windows'][63]['reference'], schedule['windows'][0]['learning'])
            self.assertEqual(schedule['windows'][127]['reference'], schedule['windows'][64]['learning'])
            self.assertEqual(P.validate_schedule(schedule, ordinary, number)['global_steps'],
                             [(number - 1) * 128 + 1, number * 128])
            self.assertEqual(P.check_correction_report(correction_report(number), number)
                             ['core_response_forwards'], 768)
            events.append((number, ordinary, event(number, 'osd_first_error')))
        self.assertTrue(P.validate_retained(event(2, 'osd_first_error'), events)['passed'])
        with self.assertRaisesRegex(ValueError, 'ordered events'):
            P.validate_retained(event(2, 'osd_first_error'), events[::-1])

    def test_capped_event4_record_is_preserved_in_both_roles_and_passes(self):
        ordinary = event(4)
        ordinary['features'][15].update(record_id='train_00399', length=2231,
                                        response_tokens=2048, finish_reason='length')
        for row in ordinary['records']:
            if row['record_id'] == 'r015':
                row['record_id'] = 'train_00399'
        schedule = P.build_continuation_schedule(ordinary, 4)
        self.assertEqual(schedule['features'][15], ordinary['features'][15])
        for role in ('learning_record_ids', 'reference_record_ids'):
            self.assertEqual(sum('train_00399' in row[role] for row in schedule['windows']), 2)
        with self.assertRaisesRegex(ValueError, 'Evaluation'):
            P.build_continuation_schedule(ordinary, 4, evaluation_ids=['train_00399'])


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def prepared(self):
        state = fixture('osd_first_error', number=3)
        trainer = runtime(state)
        path = self.root / 'state.pt'
        torch.save(state, path)
        trainer._restore(path)
        trainer._krc_late_event = None
        trainer._anchor_observations = [{'positions': [[7]], 'keep_mask': [[True]]}]
        return trainer, state

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_completed_boundary_allows_last_anchor_without_clearing_rng(self):
        trainer, state = self.prepared()
        observations = deepcopy(trainer._anchor_observations)
        result = C.verify_saved_boundary(trainer, state, self.root / 'check.json')
        self.assertTrue(result['exact_boundary'])
        self.assertTrue(result['no_rng_restore'])
        self.assertEqual(result['extra_model_passes'], 0)
        self.assertEqual(trainer._anchor_observations, observations)
        self.assertTrue(torch.equal(state['rng']['torch'], capture_rng()['torch']))

    def test_boundary_rejects_changed_state_and_stale_runtime_buffers(self):
        mutations = {
            'horizon': lambda t: setattr(t, 'horizon', 512),
            'accumulation': lambda t: setattr(t, 'accumulation', 1),
            'reduction': lambda t: setattr(t, 'reduction', 'upstream_weight_mean'),
            'ratio_totals': lambda t: t.core._ratio_totals.update(loss=[1, 2]),
            'sum_totals': lambda t: t.core._sum_totals.update(loss=1),
            'gradient': lambda t: setattr(next(t.bundle.draft_model.parameters()), 'grad',
                                          torch.ones_like(next(t.bundle.draft_model.parameters()))),
            'reference': lambda t: setattr(t.bundle.model, 'reference', object()),
            'ranking': lambda t: setattr(t.bundle.model, 'ranking_enabled', True),
            'schedule': lambda t: setattr(t, '_presentation_schedule', object()),
            'controller': lambda t: setattr(t, '_krc_late_event', object()),
            'micro': lambda t: setattr(t.core, '_micro', t.presentations + 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                trainer, state = self.prepared()
                mutate(trainer)
                with self.assertRaisesRegex(ValueError, 'boundary changed'):
                    C.verify_saved_boundary(trainer, state, self.root / (name + '.json'))
                self.assertEqual(json.loads((self.root / (name + '.json')).read_text())['status'], 'failed')


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = dict(config=str(self.root / 'config.yaml'),
            startup_ordinary_receipt=str(self.root / 'O1.json'), pinned_initial='initial-startup-only.pt',
            krc2_state='KRC2-complete.pt', krc2_checkpoint='KRC2-export',
            krc2_receipt=str(self.root / 'KRC2.json'), retained_o4_checkpoint='retained-O4',
            retained_k4_checkpoint='retained-K4', target='target', draft='draft', worker_python='worker',
            execution_source={'git_commit': 'test-only'}, panels=[panel(), panel('additional')], events=[])
        self.ordinary = {}
        self.schedules = {}
        for number in (3, 4):
            self.ordinary[number] = event(number)
            self.schedules[number] = P.build_continuation_schedule(self.ordinary[number], number)
            entry = dict(event_number=number, ordinary_receipt=str(self.root / f'O{number}.json'),
                         k_receipt=str(self.root / f'K{number}.json'),
                         reference_schedule=str(self.root / f'schedule{number}.json'))
            self.plan['events'].append(entry)
            save(Path(entry['ordinary_receipt']), self.ordinary[number])
            save(Path(entry['k_receipt']), event(number, 'osd_first_error'))
            save(Path(entry['reference_schedule']), self.schedules[number])
        save(self.root / 'sealed/PLAN.json', self.plan)
        save(self.root / 'KRC2.json', event(2, 'osd_first_error'))
        save(self.root / 'O1.json', event())
        (self.root / 'config.yaml').write_text('exact configuration')

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def read(self, path):
        return json.loads((self.root / path).read_text())

    def mocked_runtime(self, *, resume_failure=False, window_failure=False, retention_failure=False):
        outer = self
        events = []
        current = [None]
        managers = []
        class Trainer:
            def __init__(self, config, root, **kwargs):
                outer.assertEqual(kwargs, {'resume_state': 'KRC2-complete.pt'})
                outer.assertIsNone(current[0])
                current[0] = self
                self.root = root
                self.global_step, self.presentations = 256, 512
                self.closed = self.failed = False
                events.append('restore2')
            def train_event(self, event_id, paths, *, passes, presentation_schedule):
                number = int(event_id[-3:])
                outer.assertEqual((len(paths), passes), (128, 2))
                outer.assertEqual(presentation_schedule, outer.ordinary[number])
                outer.assertIs(managers[-1].original_trainer, self)
                events.append('train' + str(number))
                if window_failure and (window_failure is True or window_failure == number):
                    managers[-1].value.update(status='failed', failure_value=float('nan'))
                    managers[-1].value['counts'].update(Adam_completed=1, presentations_completed=2)
                    raise RuntimeError('post Adam correction integrity failure')
                self.global_step, self.presentations = number * 128, number * 256
                event_root = self.root / 'events' / event_id
                export = event_root / 'export'
                export.mkdir(parents=True)
                (export / 'model.safetensors').write_bytes(f'export{number}'.encode())
                full_state = event_root / 'training_state.pt'
                full_state.write_bytes(f'complete_state{number}'.encode())
                if retention_failure and number == 4:
                    (self.root / 'events/event_003/training_state.pt').write_bytes(b'overwritten')
                return dict(event(number, 'osd_first_error'), checkpoint_path=str(export),
                            full_state_path=str(full_state),
                            costs={'training_s': 1, 'state_retention_s': 0}, objective_work={})
            def status(self):
                return dict(global_step=self.global_step, presentations=self.presentations,
                    completed_events=[f'event_{i:03d}' for i in range(1, self.global_step // 128 + 1)],
                    accumulation_remainder=0, horizon=7424, objective='osd_first_error',
                    closed=self.closed, failed=self.failed)
            def close(self):
                self.closed = True
                events.append('trainer_closed')
        class Manager:
            def __init__(self, trainer, *, schedule, feature_paths, output_dir):
                number = trainer.global_step // 128 + 1
                outer.assertEqual(feature_paths, {row['record_id']: Path(row['path'])
                                                  for row in outer.ordinary[number]['features']})
                outer.assertEqual(schedule, outer.schedules[number])
                self.original_trainer = trainer
                self.start_step = trainer.global_step
                self.output_dir = output_dir
                self.value = correction_report(number)
                self.closed = False
                self.number = number
                managers.append(self)
                events.append('manager' + str(number))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
            def close(self):
                if not self.closed:
                    events.append('manager_closed' + str(self.number))
                    self.closed = True
            def report(self):
                outer.assertTrue(self.closed)
                self.value['native_global_step'] = self.start_step + self.value['counts']['Adam_completed']
                return self.value
        class Native:
            loading_s = 0
            def __init__(self, *args, **kwargs):
                outer.assertTrue(current[0].closed)
                outer.assertEqual(events.count('restore2'), 1)
                events.append('native_loaded')
            def close(self):
                events.append('native_closed')
        class WorkLog:
            def __init__(self, path):
                self.counts = Counter()
        def load(path, export, *, completed_event, **kwargs):
            event_root = outer.root / 'campaign_001/KRC/events' / f'event_{completed_event:03d}'
            expected = ('KRC2-complete.pt', 'KRC2-export') if completed_event == 2 else (
                str(event_root / 'training_state.pt'), str(event_root / 'export'))
            outer.assertEqual((path, export), expected)
            events.append('snapshot' + str(completed_event))
            return {'event_number': completed_event}, {'wall_s': 0, 'status': 'passed'}
        def resume(trainer, state, **kwargs):
            outer.assertEqual(state, {'event_number': 2})
            events.append('live_resume')
            if resume_failure:
                raise ValueError('Live resume mismatch')
        def boundary(trainer, state, receipt_path):
            outer.assertIs(trainer, current[0])
            outer.assertTrue(managers[-1].closed)
            number = state['event_number']
            events.append('boundary' + str(number))
            save(receipt_path, {'status': 'passed'})
        def startup(*args):
            outer.assertIsNone(current[0])
            events.append('startup')
            return dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                        exact_initial_state_restored=True, separate_disposable_runtime=True)
        def evaluate(backend, requests, checkpoints, root, **kwargs):
            outer.assertEqual(checkpoints, dict(O4='retained-O4', K4='retained-K4',
                KRC4=str(outer.root / 'campaign_001/KRC/events/event_004/export')))
            outer.assertEqual(kwargs['qualification_requests'], 8 if root.name == 'development' else 0)
            events.append('evaluate_' + root.name)
            result = dict(C.unattempted_panel(), status='completed', attempted_generations=576,
                          returned_generations=576, completed_valid_generations=576, unattempted_generations=0)
            save(root / 'STATUS.json', result)
            return result
        modules = {
            'online_feedback.krc_late_continuation': SimpleNamespace(
                KRCLateContinuationTrainer=Trainer, KRCLateContinuationEvent=Manager),
            'online_feedback.sglang_backend': SimpleNamespace(SGLangDFlashBackend=Native),
            'experiments.prefix_guard.qualify': SimpleNamespace(WorkLog=WorkLog,
                pin_retained_graph_runtime=lambda _: {}, require_runtime_pin=lambda: None, run_startup=startup),
        }
        from experiments.ok_continuation import resume as R
        from . import evaluate as E
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(R, 'load_snapshot', side_effect=load))
        stack.enter_context(patch.object(R, 'verify_live_resume', side_effect=resume))
        stack.enter_context(patch.object(C, 'verify_saved_boundary', side_effect=boundary))
        stack.enter_context(patch.object(E, 'evaluate', side_effect=evaluate))
        stack.enter_context(patch.object(torch, 'load', return_value={}))
        for name in ('reset_peak_memory_stats', 'empty_cache'):
            stack.enter_context(patch.object(torch.cuda, name))
        for name in ('max_memory_allocated', 'max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda, name, return_value=0))
        return stack, events

    def test_single_continuous_trainer_and_completed_managers_before_serving(self):
        stack, events = self.mocked_runtime()
        with stack:
            C.run(self.root)
        self.assertEqual(events.count('restore2'), 1)
        self.assertLess(events.index('live_resume'), events.index('manager3'))
        self.assertLess(events.index('manager_closed3'), events.index('boundary3'))
        self.assertLess(events.index('boundary3'), events.index('manager4'))
        self.assertLess(events.index('boundary4'), events.index('trainer_closed'))
        self.assertLess(events.index('trainer_closed'), events.index('native_loaded'))
        status = self.read('campaign_001/STATUS.json')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual((status['global_optimizer_step'], status['new_optimizer_steps'],
                          status['new_presentations']), (512, 256, 512))
        self.assertEqual(status['events']['event_003']['full_state_path'],
                         str(self.root / 'campaign_001/KRC/events/event_003/training_state.pt'))
        self.assertEqual(self.read('campaign_001/STATE_RETENTION.json')['status'], 'passed')
        self.assertEqual(status['evaluation_totals']['completed_valid_generations'], 1152)

    def test_resume_failure_keeps_all_work_unattempted(self):
        stack, events = self.mocked_runtime(resume_failure=True)
        with stack, self.assertRaisesRegex(ValueError, 'Live resume mismatch'):
            C.run(self.root)
        self.assertNotIn('manager3', events)
        self.assertNotIn('native_loaded', events)
        self.assertEqual(self.read('campaign_001/STATUS.json')['evaluation_totals']
                         ['unattempted_generations'], 1152)

    def test_changed_event3_artifact_stops_before_native_evaluation(self):
        stack, events = self.mocked_runtime(retention_failure=True)
        with stack, self.assertRaisesRegex(ValueError, 'state or export changed'):
            C.run(self.root)
        self.assertNotIn('native_loaded', events)
        self.assertEqual(self.read('campaign_001/STATE_RETENTION.json')['status'], 'failed')
        self.assertEqual(self.read('campaign_001/STATUS.json')['evaluation_totals']
                         ['unattempted_generations'], 1152)

    def test_post_adam_failure_retains_actual_counters_and_nonfinite_evidence(self):
        stack, events = self.mocked_runtime(window_failure=True)
        with stack, self.assertRaisesRegex(RuntimeError, 'post Adam correction'):
            C.run(self.root)
        status = self.read('campaign_001/STATUS.json')
        self.assertEqual(status['global_optimizer_step'], 256)
        self.assertEqual(status['actual_global_Adam_step'], 257)
        self.assertEqual(status['actual_new_Adam_steps_completed'], 1)
        self.assertEqual(status['actual_K_presentations_completed'], 2)
        self.assertEqual(self.read('campaign_001/COSTS.json')['interrupted_event']['failure_value'], 'nan')
        self.assertNotIn('manager4', events)
        self.assertNotIn('native_loaded', events)
        self.assertEqual(status['evaluation_totals']['unattempted_generations'], 1152)

    def test_event4_failure_counts_completed_event3_and_partial_event4_once(self):
        stack, events = self.mocked_runtime(window_failure=4)
        with stack, self.assertRaisesRegex(RuntimeError, 'post Adam correction'):
            C.run(self.root)
        status = self.read('campaign_001/STATUS.json')
        self.assertEqual(status['global_optimizer_step'], 384)
        self.assertEqual(status['actual_global_Adam_step'], 385)
        self.assertEqual(status['actual_new_Adam_steps_completed'], 129)
        self.assertEqual(status['actual_K_presentations_completed'], 258)
        self.assertEqual(list(status['events']), ['event_003'])
        self.assertNotIn('native_loaded', events)

    def test_real_evaluation_adapter_includes72_gate_and_keeps_second_panel_unattempted_on_failure(self):
        from . import evaluate as E
        original = E.evaluate
        backend = Backend(self.plan['panels'][0]['requests'])
        calls = [0]
        def fail_after_gate(record, options):
            calls[0] += 1
            if calls[0] == 73:
                raise RuntimeError('lost native reply after included gate')
        backend.mutation = fail_after_gate
        def wrapped(backend, requests, *args, **kwargs):
            backend.requests = requests
            return original(backend, requests, *args, **kwargs)
        progress = dict(status='running', panels={row['name']: C.unattempted_panel()
                                                 for row in self.plan['panels']})
        with patch.object(E, 'evaluate', side_effect=wrapped), self.assertRaisesRegex(RuntimeError, 'lost native'):
            C.evaluate_panels(backend, self.plan['panels'], dict(O4='/O4', K4='/K4', KRC4='/KRC4'),
                              self.root, progress)
        self.assertEqual(self.read('evaluation/development/QUALIFICATION.json')['generations'], 72)
        totals = progress['evaluation_totals']
        self.assertEqual(totals['completed_valid_generations'], 72)
        self.assertEqual(totals['unknown_generation_attempts'], 1)
        self.assertEqual(totals['unattempted_generations'], 1079)
        self.assertEqual(progress['panels']['additional']['status'], 'unattempted')
        self.assertFalse((self.root / 'evaluation/additional').exists())


if __name__ == '__main__':
    unittest.main()
