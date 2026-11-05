"""CPU orchestration fixtures: independent restore/startup and exact endpoint gates."""
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
from experiments.krc_events_3_4.test_campaign import correction_report
from . import campaign as C, protocol as P, state as S


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = dict(config=str(self.root / 'config.yaml'),
            startup_ordinary_receipt=str(self.root / 'O1.json'), pinned_initial='startup-only.pt',
            budget=deepcopy(P.BUDGET), evaluation_ids=['evaluation'], no_automatic_retry=True,
            no_new_feedback=True, execution_source=dict(execution_git_commit='1' * 40, dirty_files=[]), episodes=[])
        (self.root / 'config.yaml').write_text('exact original config')
        save(self.root / 'O1.json', event())
        for number in (2, 4):
            item = dict(event_number=number, parent_event=number - 1, start_step=(number - 1) * 128,
                end_step=number * 128, parent_state=f'KRC{number-1}.pt', parent_checkpoint=f'KRC{number-1}-export',
                endpoint_state=f'KRC{number}.pt', endpoint_checkpoint=f'KRC{number}-export',
                parent_receipt=str(self.root / f'parent{number}.json'),
                endpoint_receipt=str(self.root / f'endpoint{number}.json'),
                ordinary_receipt=str(self.root / f'O{number}.json'),
                reference_schedule=str(self.root / f'schedule{number}.json'),
                window_plan=str(self.root / f'windows{number}.json'), pool='immutable-pool')
            self.plan['episodes'].append(item)
            save(Path(item['parent_receipt']), event(number - 1, 'osd_first_error'))
            save(Path(item['endpoint_receipt']), event(number, 'osd_first_error'))
            ordinary = event(number)
            save(Path(item['ordinary_receipt']), ordinary)
            save(Path(item['reference_schedule']), P.reference_schedule(ordinary, number, evaluation_ids=['evaluation']))
            save(Path(item['window_plan']), P.build_window_plan(ordinary, number, evaluation_ids=['evaluation']))
        save(self.root / 'sealed/PLAN.json', self.plan)
        save(self.root / 'sealed/SEAL.json', {'files': {}})

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def runtime(self, *, endpoint_failure=False, update_failure=False):
        outer = self
        sequence, managers, trainers = [], [], []
        class Trainer:
            def __init__(self, config, root, *, resume_state):
                self.number = int(resume_state[3]) + 1
                if trainers:
                    outer.assertTrue(trainers[-1].closed)
                self.global_step, self.presentations = (self.number - 1) * 128, (self.number - 1) * 256
                self.root, self.closed, self.failed = root, False, False
                trainers.append(self)
                sequence.append('restore' + str(self.number))
            def train_event(self, event_id, paths, *, passes, presentation_schedule):
                sequence.append('train' + str(self.number))
                outer.assertEqual((len(paths), passes), (128, 2))
                outer.assertEqual(presentation_schedule, event(self.number))
                if update_failure:
                    managers[-1].value['counts'].update(Adam_completed=1, presentations_completed=2)
                    raise RuntimeError('post-Adam isolation failed')
                self.global_step, self.presentations = self.number * 128, self.number * 256
                self.root.mkdir(parents=True)
                state = self.root / 'training_state.pt'
                state.write_bytes(b'complete replay state')
                export = self.root / 'export'
                export.mkdir()
                (export / 'model.safetensors').write_bytes(b'export')
                return dict(event(self.number, 'osd_first_error'), full_state_path=str(state),
                            checkpoint_path=str(export), costs={}, objective_work={})
            def status(self):
                return dict(global_step=self.global_step, presentations=self.presentations,
                    completed_events=[f'event_{i:03d}' for i in range(1, self.global_step // 128 + 1)],
                    accumulation_remainder=0, objective='osd_first_error', horizon=7424,
                    closed=self.closed, failed=self.failed)
            def close(self):
                self.closed = True
                sequence.append('closed' + str(self.number))
        class Manager:
            def __init__(self, trainer, *, schedule, feature_paths, output_dir, window_plan, mechanism_dir):
                self.number, self.output_dir = trainer.number, output_dir
                self.value = correction_report(self.number)
                self.value['mechanism'] = dict(work=dict(pre_capture_response_forwards=40,
                    candidate_response_forwards=240, d_gradient_response_forwards=20, d_gradient_backwards=20),
                    rows=[dict(step=step, candidate_order=list(P.CANDIDATES), restoration_passed=True,
                        k_only_witness_passed=True, candidates={c: {'status': 'valid'} for c in P.CANDIDATES})
                        for step in P.WINDOWS])
                outer.assertEqual([row['step'] for row in window_plan], list(P.WINDOWS))
                managers.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
            def close(self):
                pass
            def report(self):
                return self.value
        class WorkLog:
            def __init__(self, root):
                self.counts = Counter()
        def startup(*args):
            if trainers:
                outer.assertTrue(trainers[-1].closed)
            sequence.append('startup')
            return dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
                        exact_initial_state_restored=True, separate_disposable_runtime=True)
        def load(path, export, *, completed_event, **kwargs):
            sequence.append('load_' + str(path))
            return {'number': completed_event}, {'wall_s': 0}
        def comparison(actual, original, *, event_number, **kwargs):
            sequence.append('compare' + str(event_number))
            if endpoint_failure:
                raise ValueError('Decoded scientific endpoint differs')
            return dict(wall_s=0, status='passed', exact_decoded_scientific_state=True)
        def retire(*args):
            sequence.append('retired2')
            return {'wall_s': 0}
        from experiments.ok_continuation import resume as R
        modules = {'experiments.prefix_guard.qualify': SimpleNamespace(WorkLog=WorkLog,
            pin_retained_graph_runtime=lambda _: {}, require_runtime_pin=lambda: None, run_startup=startup)}
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(C, 'controller_types', return_value={2: (Trainer, Manager), 4: (Trainer, Manager)}))
        stack.enter_context(patch.object(R, 'load_snapshot', side_effect=load))
        stack.enter_context(patch.object(R, 'verify_live_resume', return_value={'exact_resume': True}))
        stack.enter_context(patch.object(S, 'verify_live_endpoint', return_value={'exact_boundary': True}))
        stack.enter_context(patch.object(S, 'compare_endpoints', side_effect=comparison))
        stack.enter_context(patch.object(S, 'retire_verified_replay', side_effect=retire))
        stack.enter_context(patch.object(torch, 'load', return_value={}))
        for name in ('reset_peak_memory_stats', 'empty_cache'):
            stack.enter_context(patch.object(torch.cuda, name))
        for name in ('max_memory_allocated', 'max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda, name, return_value=0))
        return stack, sequence

    def test_two_independent_episodes_four_startup_forwards_no_generations(self):
        stack, sequence = self.runtime()
        with stack:
            C.run(self.root)
        self.assertEqual(sequence.count('startup'), 2)
        self.assertLess(sequence.index('closed2'), sequence.index('restore4'))
        self.assertLess(sequence.index('compare2'), sequence.index('retired2'))
        self.assertLess(sequence.index('retired2'), sequence.index('restore4'))
        status = C.read(self.root / 'campaign_001/STATUS.json')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['actual_new_Adam_steps_completed'], 256)
        self.assertEqual(status['serving_generations'], 0)
        work = C.read(self.root / 'campaign_001/COSTS.json')['work_ledger']
        self.assertEqual(work['core_response_forwards'], 2136)
        self.assertEqual(work['startup_response_forwards'], 4)

    def test_endpoint_mismatch_stops_before_retirement_and_independent_event4(self):
        stack, sequence = self.runtime(endpoint_failure=True)
        with stack, self.assertRaisesRegex(ValueError, 'endpoint differs'):
            C.run(self.root)
        self.assertNotIn('retired2', sequence)
        self.assertNotIn('restore4', sequence)
        self.assertTrue((self.root / 'campaign_001/event_002/replay/training_state.pt').exists())
        self.assertEqual(C.read(self.root / 'campaign_001/STATUS.json')['status'], 'failed')

    def test_partial_failure_records_actual_native_steps_no_automatic_retry(self):
        stack, sequence = self.runtime(update_failure=True)
        with stack, self.assertRaisesRegex(RuntimeError, 'isolation failed'):
            C.run(self.root)
        status = C.read(self.root / 'campaign_001/STATUS.json')
        self.assertEqual(status['actual_new_Adam_steps_completed'], 1)
        self.assertEqual(status['actual_K_presentations_completed'], 2)
        self.assertFalse(status['automatic_retry'])
        self.assertNotIn('restore4', sequence)


if __name__ == '__main__':
    unittest.main()
