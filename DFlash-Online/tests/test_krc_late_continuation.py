"""CPU qualification of bounded events3/4; no production scientific work.

Real-trainer tests retain BF16 Adam, masks, natural shuffle/anchor sampling and
complete states, using four updates per event and a tiny eager-attention model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from experiments.ok_continuation.resume import verify_live_resume
from online_feedback import krc_continuation as old
from online_feedback import krc_late_continuation as C
from online_feedback import training as T
from online_feedback.reference_correction_event import ReferenceCorrectionEvent
from online_feedback.rng import capture_rng
from test_reference_correction_event import ReferenceWindowTrainer, small_schedule
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class LateWindowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trainer = ReferenceWindowTrainer(self.root)
        self.trainer.global_step = 256
        self.trainer.presentations = self.trainer.core._micro = 512
        self.trainer.contract['reference_correction_event'] = C.krc2_reference_contract()
        for parameter in self.trainer.backend.optimizer.model_params:
            self.trainer.backend.optimizer.optimizer.state[parameter] = {
                'step': torch.tensor(256.), 'exp_avg': torch.zeros_like(parameter),
                'exp_avg_sq': torch.zeros_like(parameter),
            }
        patch = mock.patch.object(C, 'EVENT_WINDOWS', 2)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_active_and_empty_windows_and_next_event_keep_offsets_and_no_probes(self):
        optimizer = self.trainer.backend.optimizer.optimizer
        original_step = self.trainer.core.train_step
        reports = []
        for event_number in (3, 4):
            with C.KRCLateContinuationEvent(
                    self.trainer, schedule=self.trainer.schedule,
                    feature_paths=self.trainer.feature_paths,
                    output_dir=self.root / str(event_number)) as manager:
                for index in range(4):
                    self.trainer.present(index)
            reports.append(manager.report())
            self.assertEqual(self.trainer.core.train_step, original_step)
            self.assertIs(self.trainer.backend.optimizer.optimizer, optimizer)
            self.assertIsNone(self.trainer._krc_late_event)
            self.assertFalse(hasattr(self.trainer.bundle.model, 'prefix_guard'))
            self.assertFalse(self.trainer.bundle.model.lm_head._forward_pre_hooks)
            self.assertEqual(manager.report()['status'], 'completed')
            self.assertEqual(manager.report()['probes'], [])
            for name in ('original_response_forwards', 'pre_reference_response_forwards',
                         'post_k_reference_response_forwards'):
                self.assertEqual(manager.work[name], 4)
            for name in ('post_c_reference_response_forwards', 'post_k_learning_response_forwards',
                         'post_c_learning_response_forwards'):
                self.assertEqual(manager.work[name], 0)
            self.assertEqual(manager.counts['Adam_completed'], 2)
            manager.close()
        self.assertEqual([row['correction']['status'] for row in reports[0]['steps']],
                         ['applied', 'empty_protection'])
        self.assertEqual([report['first_window']['optimizer_counters'] for report in reports],
                         [[257.], [259.]])
        self.assertEqual([report['native_global_step'] for report in reports], [258, 260])
        for index, report in enumerate(reports):
            for local, row in enumerate(report['steps'], 1):
                self.assertEqual(row['integrity']['step_coordinates'], dict(
                    event_local_step=local, global_step_before=256 + 2 * index + local - 1,
                    global_step_after_native_Adam=256 + 2 * index + local))
            self.assertEqual(report['first_window']['extra_forwards'], 0)
            self.assertEqual(report['first_window']['extra_optimizer_steps'], 0)
            self.assertEqual(report['first_window']['status'], 'passed')

    def test_contract_has_no_migration_or_rng_draw_and_old_guard_remains(self):
        contract = deepcopy(self.trainer.contract)
        rng = capture_rng()
        manager = C.KRCLateContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                            feature_paths=self.trainer.feature_paths)
        self.assertEqual(contract, self.trainer.contract)
        self.assertTrue(trees_equal(rng, capture_rng()))
        self.assertEqual(manager.report()['diagnostic_migration']['changed_fields'], [])
        manager.close()
        with self.assertRaisesRegex(ValueError, 'exact restored KRC1 contract'):
            old.KRCContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                     feature_paths=self.trainer.feature_paths)
        self.trainer.contract['reference_correction_event'] = old.krc1_reference_contract()
        with self.assertRaisesRegex(ValueError, 'no-probe KRC2 contract'):
            C.KRCLateContinuationEvent(self.trainer, schedule=self.trainer.schedule,
                                       feature_paths=self.trainer.feature_paths)

    def test_first_actual_window_failure_adds_no_model_work(self):
        manager = C.KRCLateContinuationEvent(
            self.trainer, schedule=self.trainer.schedule, feature_paths=self.trainer.feature_paths,
            output_dir=self.root / 'failure')
        original = manager._check_first_window
        def fail_counter():
            manager.counts['K_backward_completed'] -= 1
            original()
        manager._check_first_window = fail_counter
        with self.assertRaisesRegex(ValueError, 'first-window check failed'):
            with manager:
                self.trainer.present(0)
                self.trainer.present(1)
        self.assertEqual(manager.counts['Adam_completed'], 1)
        self.assertEqual(manager.work['original_response_forwards'], 2)
        self.assertEqual(manager.work['post_c_reference_response_forwards'], 0)
        self.assertEqual(manager.steps[-1]['status'], 'failed')
        self.assertEqual(json.loads((self.root / 'failure' / 'FIRST_WINDOW.json').read_text())['status'],
                         'failed')


class RealLateContinuationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        for index in range(8, 16):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = raw['label_binding']['record_id'] = f'response_{index}'
            path = self.root / f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)
        for patch in (mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,)),
                      mock.patch.object(C, 'RESUME_STEP', 8),
                      mock.patch.object(C, 'EVENT_WINDOWS', 4),
                      mock.patch.object(C, 'FINAL_STEP', 16)):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def prepare(self):
        ordinary = self.trainer('ordinary', objective='osd')
        receipts = [ordinary.train_event(f'event_{number:03d}', self.pool(number))
                    for number in range(1, 5)]
        source = self.trainer('KRC_source', objective='osd_first_error')
        for number in (1, 2):
            receipt, schedule, paths = self.bindings(receipts, number)
            probes = (1, 64, 128) if number == 1 else ()
            with ReferenceCorrectionEvent(source, schedule=schedule, feature_paths=paths,
                                          probe_steps=probes):
                source.train_event(f'event_{number:03d}', self.pool(number),
                                   presentation_schedule=receipt)
        state_path = self.root / 'KRC2_read_only.pt'
        state_path.write_bytes(source.latest.read_bytes())
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        return source, state_path, state, receipts

    def pool(self, number):
        return self.features[(number - 1) * 4:number * 4]

    def bindings(self, receipts, number):
        receipt = receipts[number - 1]
        return (receipt, small_schedule(receipt['records'], pairs_per_pass=2),
                {row['record_id']: Path(row['path']) for row in receipt['features']})

    def resume(self, state_path, name='KRC34', config=None):
        trainer = C.KRCLateContinuationTrainer(config or self.config, self.root / name,
                                              resume_state=state_path)
        self.addCleanup(trainer.close)
        return trainer

    def test_exact_resume_two_events_and_immutable_event3_state(self):
        source, path, state, ordinary = self.prepare()
        self.assertNotIn('status', state['events'][1])
        expected_receipts = []
        expected_states = []
        for number in (3, 4):
            receipt, schedule, paths = self.bindings(ordinary, number)
            with ReferenceCorrectionEvent(source, schedule=schedule, feature_paths=paths,
                                          probe_steps=()):
                expected_receipts.append(source.train_event(f'event_{number:03d}', self.pool(number),
                                                            presentation_schedule=receipt))
            expected_states.append(self.saved(source))
        resumed = self.resume(path)
        self.assertFalse(resumed.latest.exists())
        self.assertTrue(verify_live_resume(resumed, state)['exact_resume'])
        self.assertTrue(trees_equal(state['rng'], capture_rng()))
        optimizer = resumed.backend.optimizer.optimizer
        retained_third = None
        third_bytes = None
        for index, number in enumerate((3, 4)):
            receipt, schedule, paths = self.bindings(ordinary, number)
            with C.KRCLateContinuationEvent(resumed, schedule=schedule, feature_paths=paths,
                                           output_dir=self.root / f'windows{number}') as manager:
                actual = resumed.train_event(f'event_{number:03d}', self.pool(number),
                                             presentation_schedule=receipt)
            actual_state = self.saved(resumed)
            for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations',
                        'algorithm_contract', 'config_text', 'horizon'):
                self.assertTrue(trees_equal(expected_states[index][key], actual_state[key]), key)
            for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'metrics'):
                self.assertEqual([row[key] for row in actual['records']],
                                 [row[key] for row in expected_receipts[index]['records']], key)
            self.assertEqual((actual['start_step'], actual['end_step']), ((number - 1) * 4, number * 4))
            self.assertEqual(len(actual_state['events']), number)
            self.assertEqual(actual_state['horizon'], 7424)
            self.assertIs(resumed.backend.optimizer.optimizer, optimizer)
            self.assertEqual(manager.report()['first_window']['optimizer_counters'],
                             [float((number - 1) * 4 + 1)])
            self.assertEqual(manager.work['original_response_forwards'], 8)
            self.assertEqual(manager.work['pre_reference_response_forwards'], 8)
            self.assertEqual(manager.work['post_k_reference_response_forwards'], 8)
            self.assertEqual(manager.probes, [])
            immutable = Path(actual['full_state_path'])
            self.assertTrue(immutable.samefile(resumed.latest))
            self.assertEqual(json.loads((immutable.parent / 'RESULT.json').read_text()), actual)
            self.assertGreaterEqual(actual['costs']['state_retention_s'], 0.)
            if number == 3:
                retained_third = immutable
                third_bytes = hashlib.sha256(immutable.read_bytes()).hexdigest()
            else:
                self.assertFalse(retained_third.samefile(resumed.latest))
                self.assertEqual(hashlib.sha256(retained_third.read_bytes()).hexdigest(), third_bytes)
                saved_third = torch.load(retained_third, map_location='cpu', weights_only=False)
                self.assertEqual((saved_third['global_step'], saved_third['presentations']), (12, 24))
                self.assertEqual((actual_state['global_step'], actual_state['presentations']), (16, 32))
        self.assertTrue(trees_equal(state, torch.load(path, map_location='cpu', weights_only=False)))
        self.assertEqual([float(value['step']) for value in optimizer.state.values()],
                         [16.] * len(optimizer.state))
        with self.assertRaisesRegex(ValueError, 'boundary or local window count'):
            C.KRCLateContinuationEvent(resumed, schedule=schedule, feature_paths=paths)

    def test_rejects_wrong_history_config_contract_and_incomplete_start(self):
        _, path, state, _ = self.prepare()
        variants = []
        changed = deepcopy(state)
        changed['events'][1]['end_step'] -= 1
        variants.append(('history', changed, 'own event history'))
        changed = deepcopy(state)
        changed['algorithm_contract']['reference_correction_event']['probe_steps'] = [1, 64, 128]
        variants.append(('probes', changed, 'Resume algorithm'))
        changed = deepcopy(state)
        changed['algorithm_contract'].pop('reference_correction_event')
        variants.append(('K2', changed, 'Resume algorithm'))
        changed = deepcopy(state)
        changed['presentations'] -= 1
        variants.append(('partial', changed, 'complete optimizer windows'))
        changed = deepcopy(state)
        changed['events'] = changed['events'][:1]
        variants.append(('missing_event', changed, 'own event history'))
        for name, payload, message in variants:
            with self.subTest(name=name):
                candidate = self.root / (name + '.pt')
                torch.save(payload, candidate)
                with self.assertRaisesRegex(ValueError, message):
                    self.resume(candidate, name=name)
        config = yaml.safe_load(self.config.read_text())
        config['training']['max_steps'] = 512
        config['online_feedback']['max_events'] = 4
        short = self.root / 'short.yaml'
        short.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, '7424-step horizon'):
            self.resume(path, name='short', config=short)
        with self.assertRaisesRegex(ValueError, 'complete KRC2 recovery state'):
            self.resume(None, name='missing')
        with self.assertRaisesRegex(ValueError, 'fresh output directory'):
            self.resume(path, name='KRC_source')

    def test_wrong_anchor_draw_stops_before_ordinary_work_without_forcing(self):
        _, path, _, ordinary = self.prepare()
        ordinary = deepcopy(ordinary)
        ordinary[2]['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        receipt, schedule, paths = self.bindings(ordinary, 3)
        resumed = self.resume(path)
        with C.KRCLateContinuationEvent(resumed, schedule=schedule, feature_paths=paths) as manager:
            with self.assertRaisesRegex(ValueError, 'Sampled anchors differ'):
                resumed.train_event('event_003', self.pool(3), presentation_schedule=receipt)
        self.assertEqual(manager.counts['Adam_completed'], 0)
        self.assertEqual(manager.counts['K_backward_completed'], 0)
        self.assertEqual(resumed.global_step, 8)
        self.assertTrue(resumed.failed)

    def test_event_and_reference_bindings_and_optional_work_are_guarded(self):
        _, path, _, ordinary = self.prepare()
        receipt, schedule, paths = self.bindings(ordinary, 3)
        resumed = self.resume(path)
        with self.assertRaisesRegex(ValueError, 'retained ordinary presentation schedule'):
            resumed.train_event('event_003', self.pool(3))
        with self.assertRaisesRegex(ValueError, 'Install the KRC late'):
            resumed.train_event('event_003', self.pool(3), presentation_schedule=receipt)
        with C.KRCLateContinuationEvent(resumed, schedule=schedule, feature_paths=paths) as manager:
            with self.assertRaisesRegex(ValueError, 'Optional learning'):
                resumed.train_event('event_003', self.pool(3), reference_check=lambda *_: None,
                                    presentation_schedule=receipt)
            with self.assertRaisesRegex(ValueError, 'different event boundary'):
                resumed.train_event('event_004', self.pool(4), presentation_schedule=receipt)
            wrong_counters = deepcopy(receipt)
            wrong_counters['records'][0]['optimizer_step_after'] = 0
            with self.assertRaisesRegex(ValueError, 'different cumulative counters'):
                resumed.train_event('event_003', self.pool(3), presentation_schedule=wrong_counters)
            manager.schedule['windows'][0]['reference_batch_index'] = 0
            with self.assertRaisesRegex(ValueError, 'same-pass next pair'):
                resumed.train_event('event_003', self.pool(3), presentation_schedule=receipt)
            self.assertEqual(manager.counts['Adam_completed'], 0)


if __name__ == '__main__':
    unittest.main()
