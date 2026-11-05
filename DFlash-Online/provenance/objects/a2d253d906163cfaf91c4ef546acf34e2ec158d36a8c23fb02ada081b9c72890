"""CPU continuation checks; production constants stay 128 + 128 steps.

The real-trainer fixture uses four steps per event to keep these tests small.
It retains native BF16 Adam, current K masks, normal sampling and full restart.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.ok_continuation.resume import verify_live_resume
from online_feedback import krc_continuation as C
from online_feedback import training as T
from online_feedback.reference_correction_event import ReferenceCorrectionEvent
from online_feedback.rng import capture_rng
from test_reference_correction_event import ReferenceWindowTrainer, small_schedule
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class ContinuationWindowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trainer = ReferenceWindowTrainer(self.root)
        self.trainer.global_step = 128
        self.trainer.presentations = self.trainer.core._micro = 256
        self.trainer.contract['reference_correction_event'] = C.krc1_reference_contract()
        for parameter in self.trainer.backend.optimizer.model_params:
            self.trainer.backend.optimizer.optimizer.state[parameter] = {
                'step': torch.tensor(128.), 'exp_avg': torch.zeros_like(parameter),
                'exp_avg_sq': torch.zeros_like(parameter),
            }
        patch = mock.patch.object(C, 'EVENT_WINDOWS', 2)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_two_windows_map_global_steps_refresh_prefixes_and_do_no_probes(self):
        optimizer = self.trainer.backend.optimizer.optimizer
        with C.KRCContinuationEvent(
                self.trainer, schedule=self.trainer.schedule,
                feature_paths=self.trainer.feature_paths, output_dir=self.root / 'logs') as manager:
            for index in range(4):
                self.trainer.present(index)
                if index % 2:
                    self.assertIsNone(manager.active)
                    self.assertFalse(hasattr(self.trainer.bundle.model, 'prefix_guard'))
                    self.assertFalse(self.trainer.bundle.model.lm_head._forward_pre_hooks)
        report = manager.report()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual((self.trainer.global_step, self.trainer.presentations), (130, 260))
        self.assertIs(self.trainer.backend.optimizer.optimizer, optimizer)
        self.assertEqual(report['first_window']['status'], 'passed')
        self.assertEqual(report['first_window']['optimizer_counters'], [129.])
        self.assertEqual(report['first_window']['extra_forwards'], 0)
        self.assertEqual(report['first_window']['extra_optimizer_steps'], 0)
        self.assertEqual(report['counts']['Adam_completed'], 2)
        self.assertEqual(report['counts']['K_backward_completed'], 4)
        self.assertEqual(report['probes'], [])
        for name in ('original_response_forwards', 'pre_reference_response_forwards',
                     'post_k_reference_response_forwards'):
            self.assertEqual(report['work'][name], 4)
        for name in ('post_c_reference_response_forwards', 'post_k_learning_response_forwards',
                     'post_c_learning_response_forwards'):
            self.assertEqual(report['work'][name], 0)
        self.assertEqual([row['correction']['status'] for row in report['steps']],
                         ['applied', 'empty_protection'])
        for local, global_step in ((1, 129), (2, 130)):
            saved = json.loads((self.root / 'logs' / f'window_{local:03d}.json').read_text())
            coordinate = saved['summary']['integrity']['step_coordinates']
            self.assertEqual(coordinate['event_local_step'], local)
            self.assertEqual(coordinate['global_step_after_native_Adam'], global_step)
            self.assertIsNone(saved['details']['learning'])
        for state in optimizer.state.values():
            self.assertEqual(float(state['step']), 130.)
        self.assertIsNone(manager.trainer)
        self.assertFalse(self.trainer._krc_continuation_active)
        manager.close()

    def test_contract_migration_changes_only_optional_probes(self):
        before = deepcopy(self.trainer.contract)
        rng = capture_rng()
        manager = C.KRCContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                        feature_paths=self.trainer.feature_paths)
        expected = deepcopy(before)
        expected['reference_correction_event']['probe_steps'] = []
        self.assertEqual(self.trainer.contract, expected)
        self.assertTrue(trees_equal(rng, capture_rng()))
        self.assertEqual(manager.report()['diagnostic_migration']['changed_fields'], ['probe_steps'])
        manager.close()
        bad = deepcopy(before['reference_correction_event'])
        bad['epsilon'] = 1e-6
        self.trainer.contract['reference_correction_event'] = bad
        with self.assertRaisesRegex(ValueError, 'exact restored KRC1 contract'):
            C.KRCContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                  feature_paths=self.trainer.feature_paths)

    def test_first_window_accounting_failure_stops_without_extra_work(self):
        manager = C.KRCContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                        feature_paths=self.trainer.feature_paths,
                                        output_dir=self.root / 'failed')
        original = manager._check_first_window
        def corrupt_accounting():
            manager.counts['K_backward_completed'] -= 1
            original()
        manager._check_first_window = corrupt_accounting
        with self.assertRaisesRegex(ValueError, 'first-window check failed'):
            with manager:
                self.trainer.present(0)
                self.trainer.present(1)
        gate = json.loads((self.root / 'failed' / 'FIRST_WINDOW.json').read_text())
        self.assertEqual(gate['status'], 'failed')
        self.assertEqual(manager.counts['Adam_completed'], 1)
        self.assertEqual(manager.work['original_response_forwards'], 2)
        self.assertEqual(manager.work['post_c_reference_response_forwards'], 0)
        self.assertEqual(manager.steps[-1]['status'], 'failed')
        self.assertFalse(hasattr(self.trainer.bundle.model, 'prefix_guard'))


class RealContinuationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        for patch in (mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,)),
                      mock.patch.object(C, 'RESUME_STEP', 4),
                      mock.patch.object(C, 'EVENT_WINDOWS', 4)):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def prepare(self):
        ordinary = self.trainer('ordinary', objective='osd_first_error')
        first = ordinary.train_event('event_001', self.features[:4])
        second = ordinary.train_event('event_002', self.features[4:])
        source = self.trainer('KRC1', objective='osd_first_error')
        first_schedule = small_schedule(first['records'], 2)
        paths = {row['record_id']: Path(row['path']) for row in first['features']}
        with ReferenceCorrectionEvent(source, schedule=first_schedule, feature_paths=paths,
                                      probe_steps=(1, 64, 128)) as manager:
            self.assertEqual(manager.contract, C.krc1_reference_contract())
            source.train_event('event_001', self.features[:4], presentation_schedule=first)
        state_path = self.root / 'KRC1_read_only.pt'
        state_path.write_bytes(source.latest.read_bytes())
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        schedule = small_schedule(second['records'], 2)
        paths = {row['record_id']: Path(row['path']) for row in second['features']}
        return source, state_path, state, second, schedule, paths

    def resume(self, state_path, name='KRC2'):
        trainer = C.KRCContinuationTrainer(self.config, self.root / name, resume_state=state_path)
        self.addCleanup(trainer.close)
        return trainer

    def test_exact_full_resume_matches_uninterrupted_continuation_and_normal_draws(self):
        source, path, state, ordinary, schedule, paths = self.prepare()
        # The existing serializer intentionally saves the event before the
        # completed external receipt; no status key may be assumed here.
        self.assertNotIn('status', state['events'][0])
        with ReferenceCorrectionEvent(source, schedule=schedule, feature_paths=paths,
                                      probe_steps=()) as uninterrupted:
            expected = source.train_event('event_002', self.features[4:], presentation_schedule=ordinary)
        expected_state = self.saved(source)
        resumed = self.resume(path)
        self.assertFalse(resumed.latest.exists())  # No copy of the old bulky state.
        gate = verify_live_resume(resumed, state)
        self.assertTrue(gate['exact_resume'])
        self.assertTrue(trees_equal(state['rng'], capture_rng()))
        with C.KRCContinuationEvent(resumed, schedule=schedule, feature_paths=paths,
                                    output_dir=self.root / 'continuation') as manager:
            actual = resumed.train_event('event_002', self.features[4:], presentation_schedule=ordinary)
        actual_state = self.saved(resumed)
        for name in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected_state[name], actual_state[name]), name)
        for name in ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'metrics'):
            self.assertEqual([row[name] for row in actual['records']],
                             [row[name] for row in expected['records']], name)
        self.assertEqual((actual['start_step'], actual['end_step']), (4, 8))
        self.assertEqual(len(actual_state['events']), 2)
        self.assertEqual(actual_state['algorithm_contract']['reference_correction_event']['probe_steps'], [])
        self.assertEqual(manager.report()['first_window']['status'], 'passed')
        self.assertEqual(manager.work['original_response_forwards'], 8)
        self.assertEqual(manager.work['pre_reference_response_forwards'], 8)
        self.assertEqual(manager.work['post_k_reference_response_forwards'], 8)
        self.assertEqual(manager.probes, [])
        self.assertTrue(trees_equal(state, torch.load(path, map_location='cpu', weights_only=False)))
        for value in resumed.backend.optimizer.optimizer.state.values():
            self.assertEqual(float(value['step']), 8.)

    def test_bad_saved_contract_and_missing_complete_state_are_rejected(self):
        _, path, state, _, _, _ = self.prepare()
        changed = deepcopy(state)
        changed['algorithm_contract']['reference_correction_event']['probe_steps'] = []
        changed_path = self.root / 'wrong_contract.pt'
        torch.save(changed, changed_path)
        with self.assertRaisesRegex(ValueError, 'Resume algorithm'):
            self.resume(changed_path, 'bad')
        with self.assertRaisesRegex(ValueError, 'complete recovery state'):
            self.resume(None, 'missing')
        restored = self.resume(path)
        with self.assertRaisesRegex(ValueError, 'Install the KRC'):
            restored.train_event('event_002', self.features[4:])

    def test_wrong_retained_anchor_draw_stops_instead_of_forcing_it(self):
        _, path, _, ordinary, schedule, paths = self.prepare()
        ordinary = deepcopy(ordinary)
        anchors = ordinary['records'][0]['sampled_anchors'][0]['positions'][0]
        anchors[0] += 1
        resumed = self.resume(path)
        with C.KRCContinuationEvent(resumed, schedule=schedule, feature_paths=paths) as manager:
            with self.assertRaisesRegex(ValueError, 'Sampled anchors differ'):
                resumed.train_event('event_002', self.features[4:], presentation_schedule=ordinary)
        self.assertEqual(manager.counts['Adam_completed'], 0)
        self.assertEqual(manager.counts['K_backward_completed'], 0)
        self.assertEqual(resumed.global_step, 4)


if __name__ == '__main__':
    unittest.main()
