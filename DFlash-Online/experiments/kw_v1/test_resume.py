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


class RealContinuationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        for index in range(8, 48):
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
            for number in range(1, 13):
                receipt = original.train_event(f'event_{number:03d}', self.pool(number),
                    presentation_schedule=receipts['O'][number] if arm == 'K' else None)
                receipts[arm][number] = receipt
                if number == 8:
                    parent = self.root / (arm + '8_read_only.pt')
                    parent.write_bytes(original.latest.read_bytes())
                    parents[arm] = parent
                if number >= 9:
                    states[arm][number] = self.saved(original)
        return receipts, states, parents

    def test_own_full_restore9_to12_matches_uninterrupted_native_path(self):
        receipts, expected, parents = self.prepare()
        for arm in R.OBJECTIVES:
            original_bytes = parents[arm].read_bytes()
            parent = parents[arm]
            for number in range(9, 13):
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
                resume_state=parents[arm], arm=arm, completed_event=8)
            self.addCleanup(trainer.close)
            with self.assertRaisesRegex(ValueError, 'O samples normally'):
                trainer.train_event('event_009', self.pool(9),
                    presentation_schedule=None if arm == 'K' else receipts['O'][9])
            with self.assertRaisesRegex(ValueError, 'no reference'):
                trainer.train_event('event_009', self.pool(9), reference_check=lambda *args: None)
        with self.assertRaisesRegex(ValueError, 'Resume algorithm'):
            R.OKContinuationTrainer(self.config, self.root / 'wrong_arm',
                resume_state=parents['O'], arm='K', completed_event=8)
        with self.assertRaisesRegex(ValueError, 'boundary 8..11'):
            R.OKContinuationTrainer(self.config, self.root / 'thirteenth',
                resume_state=parents['O'], arm='O', completed_event=12)
        bad_schedule = deepcopy(receipts['O'][9])
        bad_schedule['records'][0]['sampled_anchors'][0]['positions'][0][0] += 1
        bad = R.OKContinuationTrainer(self.config, self.root / 'bad_anchors',
            resume_state=parents['K'], arm='K', completed_event=8)
        self.addCleanup(bad.close)
        with self.assertRaisesRegex(ValueError, 'Sampled anchors differ'):
            bad.train_event('event_009', self.pool(9), presentation_schedule=bad_schedule)
        self.assertEqual(bad.global_step, 32)
        self.assertTrue(bad.failed)


if __name__ == '__main__':
    unittest.main()
