"""Focused CPU fixtures for strict O/K boundary extension and natural draws."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from experiments.ok_continuation.test_resume import event, fixture, runtime
from experiments.ok_continuation import resume as old
from online_feedback import training as T
from online_feedback.rng import capture_rng, restore_rng
from test_training_contract import trees_equal
import test_osd_recipe as legacy
from . import resume as R


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = capture_rng()
        self.addCleanup(lambda: restore_rng(rng))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_each_new_snapshot_keeps_complete_own_state_and_old_guards(self):
        for arm, objective in R.OBJECTIVES.items():
            for number in range(4, 9):
                state = fixture(objective, number)
                path = self.root / f'{arm}{number}.pt'
                torch.save(state, path)
                export = self.root / f'{arm}{number}'
                export.mkdir()
                save_file(state['draft_state_dict'], export / 'model.safetensors')
                loaded, receipt = R.load_snapshot(path, export, objective=objective,
                    config_text=state['config_text'], completed_event=number, expected_cuda_rng_states=0)
                self.assertEqual(receipt['global_step'], number * 128)
                self.assertTrue(receipt['export_exact'])
                trainer = runtime(state)
                trainer._restore(path)
                self.assertTrue(R.verify_live_resume(trainer, loaded)['exact_resume'])
                self.assertTrue(R.verify_saved_boundary(trainer, loaded, self.root / 'boundary.json')['exact_boundary'])
                if number > 4:
                    with self.assertRaisesRegex(ValueError, '1..4'):
                        old.validate_snapshot(state, objective=objective, config_text=state['config_text'],
                            completed_event=number, expected_cuda_rng_states=0)

    def test_new_receipts_require_exact_O_K_draws_and_boundaries(self):
        for number in range(5, 9):
            ordinary, k = event(number), event(number, 'osd_first_error')
            self.assertTrue(R.validate_event_receipt(k, event_number=number,
                objective='osd_first_error', reference_schedule=ordinary)['schedule_matched'])
            k['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
            with self.assertRaisesRegex(ValueError, 'anchors differ'):
                R.validate_event_receipt(k, event_number=number, objective='osd_first_error',
                    reference_schedule=ordinary)
        with self.assertRaisesRegex(ValueError, '1..8'):
            R.validate_event_receipt(event(9), event_number=9, objective='osd')

    def test_no_cross_arm_correction_partial_history_or_counter_substitution(self):
        original = fixture('osd', 4)
        first_id = next(iter(original['backend']['optimizer']['optimizer_state_dict']['state']))
        mutations = {
            'objective': lambda s: s['algorithm_contract']['online_target_objective'].update(name='osd_first_error'),
            'correction': lambda s: s['algorithm_contract'].update(reference_correction_event={}),
            'history': lambda s: s['events'].pop(),
            'counter': lambda s: s['backend']['optimizer']['optimizer_state_dict']['state'][first_id].update(step=torch.tensor(0.)),
            'accumulation': lambda s: s.update(presentations=1023),
            'horizon': lambda s: s.update(horizon=1024),
        }
        for label, mutate in mutations.items():
            bad = deepcopy(original)
            mutate(bad)
            with self.subTest(label=label), self.assertRaises(ValueError):
                R.validate_snapshot(bad, objective='osd', config_text=original['config_text'],
                    completed_event=4, expected_cuda_rng_states=0)


class RealContinuationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        for index in range(8, 32):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw['record_id'] = raw['label_binding']['record_id'] = f'response_{index}'
            path = self.root / f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)
        for context in (patch.object(T, 'ranking_eos_token_ids', return_value=(28,)),
                        patch.object(R, 'STEPS', 4), patch.object(R, 'PRESENTATIONS', 8)):
            context.start()
            self.addCleanup(context.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def pool(self, number):
        return self.features[(number - 1) * 4:number * 4]

    def prepare(self):
        originals, receipts, states, parents = {}, {}, {}, {}
        for arm, objective in R.OBJECTIVES.items():
            original = originals[arm] = self.trainer('original_' + arm, objective=objective)
            receipts[arm], states[arm] = {}, {}
            for number in range(1, 9):
                receipt = original.train_event(f'event_{number:03d}', self.pool(number),
                    presentation_schedule=receipts['O'][number] if arm == 'K' else None)
                receipts[arm][number] = receipt
                if number == 4:
                    parent = self.root / (arm + '4_read_only.pt')
                    parent.write_bytes(original.latest.read_bytes())
                    parents[arm] = parent
                if number >= 5:
                    states[arm][number] = self.saved(original)
        return receipts, states, parents

    def test_own_full_restore5_to8_matches_uninterrupted_native_path(self):
        receipts, expected, parents = self.prepare()
        for arm in R.OBJECTIVES:
            original_bytes = parents[arm].read_bytes()
            parent = parents[arm]
            for number in range(5, 9):
                trainer = R.OKContinuationTrainer(self.config, self.root / ('resumed_' + arm),
                    resume_state=parent, arm=arm, completed_event=number - 1)
                self.addCleanup(trainer.close)
                restored = torch.load(parent, weights_only=False)
                self.assertTrue(R.verify_live_resume(trainer, restored)['exact_resume'])
                actual = trainer.train_event(f'event_{number:03d}', self.pool(number),
                    presentation_schedule=receipts['O'][number] if arm == 'K' else None)
                saved = self.saved(trainer)
                for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations',
                            'algorithm_contract', 'config_text', 'horizon'):
                    self.assertTrue(trees_equal(saved[key], expected[arm][number][key]), (arm, number, key))
                for key in ('record_id', 'pass', 'sampled_anchors', 'learning_rate', 'metrics'):
                    self.assertEqual([row[key] for row in actual['records']],
                                     [row[key] for row in receipts[arm][number]['records']])
                with self.assertRaisesRegex(ValueError, 'next sealed event'):
                    trainer.train_event(f'event_{number+1:03d}', self.pool(number),
                        presentation_schedule=receipts['O'][number] if arm == 'K' else None)
                parent = trainer.latest
                trainer.close()
            self.assertEqual(parents[arm].read_bytes(), original_bytes)

    def test_wrapper_rejects_missing_schedule_probes_wrong_arm_and_natural_anchor_mismatch(self):
        receipts, _, parents = self.prepare()
        for arm in ('O', 'K'):
            trainer = R.OKContinuationTrainer(self.config, self.root / ('guard_' + arm),
                resume_state=parents[arm], arm=arm, completed_event=4)
            self.addCleanup(trainer.close)
            with self.assertRaisesRegex(ValueError, 'O samples normally'):
                trainer.train_event('event_005', self.pool(5),
                    presentation_schedule=None if arm == 'K' else receipts['O'][5])
            with self.assertRaisesRegex(ValueError, 'no reference'):
                trainer.train_event('event_005', self.pool(5), reference_check=lambda *args: None)
        with self.assertRaisesRegex(ValueError, 'Resume algorithm'):
            R.OKContinuationTrainer(self.config, self.root / 'wrong_arm',
                resume_state=parents['O'], arm='K', completed_event=4)
        with self.assertRaisesRegex(ValueError, 'boundary 4..7'):
            R.OKContinuationTrainer(self.config, self.root / 'ninth',
                resume_state=parents['O'], arm='O', completed_event=8)
        bad_schedule = deepcopy(receipts['O'][5])
        bad_schedule['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        bad = R.OKContinuationTrainer(self.config, self.root / 'bad_anchors',
            resume_state=parents['K'], arm='K', completed_event=4)
        self.addCleanup(bad.close)
        with self.assertRaisesRegex(ValueError, 'Sampled anchors differ'):
            bad.train_event('event_005', self.pool(5), presentation_schedule=bad_schedule)
        self.assertEqual(bad.global_step, 16)
        self.assertTrue(bad.failed)


if __name__ == '__main__':
    unittest.main()
