"""CPU integration of consecutive KC windows, with the tested correction core."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.prefix_correction import PrefixCorrection
from online_feedback.prefix_correction_event import AuxiliaryIntegrity, PrefixCorrectionEvent
from online_feedback.rng import capture_rng
from test_prefix_correction import ToyBlockModel
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class WindowTrainer:
    """Real BF16 Adam, deterministic tiny forwards, and explicit accumulation."""

    def __init__(self):
        torch.manual_seed(17)
        self.bundle = SimpleNamespace(model=ToyBlockModel())
        self.backend = SimpleNamespace(
            optimizer=DirectBF16AdamW(self.bundle.model.draft_model, learning_rate=2.,
                                     weight_decay=0., max_grad_norm=1.),
            backward=lambda loss: loss.backward(),
        )
        self.objective = 'osd_first_error'
        self.accumulation = 2
        self.global_step = self.presentations = 0
        self.contract = {}
        self.core = SimpleNamespace(_micro=0, accumulation_remainder=0, train_step=self._step)

    def _step(self, batch, context=None):
        model = self.bundle.model
        inputs, features, labels = batch.inputs, batch.features, batch.labels
        anchors, keep, hidden = model._forward_draft_blocks(inputs, features, torch.ones_like(inputs))
        logits = model.lm_head(hidden).unsqueeze(1)
        validity = torch.tensor([[[0., 1., 1.]]])
        recorder = getattr(model, 'prefix_guard', None)
        if recorder is not None:
            recorder.begin_response(inputs, features, torch.ones_like(inputs), anchors, keep, labels, validity)
            recorder.chunk(logits, labels, validity, 0)
            recorder.finish_response(torch.zeros(()), torch.ones(()))
        # This known native step breaks the first window's prefix. A zero
        # second-window gradient still advances existing Adam moments/counters.
        loss = logits[..., 0].mean() * batch.loss_scale / 2
        torch.rand(1)  # Ordinary training RNG progression must survive replays.
        self.backend.backward(loss)
        self.core._micro += 1
        self.core.accumulation_remainder = self.core._micro % 2
        stepped = self.core.accumulation_remainder == 0
        if stepped:
            self.backend.optimizer.step()
        return SimpleNamespace(optimizer_stepped=stepped)

    def present(self, ordinal):
        labels = torch.full((1, 1, 3), 0 if ordinal < 2 else 1, dtype=torch.long)
        batch = SimpleNamespace(
            sample_ids=[f'response_{ordinal}'], inputs=labels[:, 0], labels=labels,
            features=torch.tensor([[[1., 0.], [1., 0.], [1., 0.]]], dtype=torch.bfloat16),
            loss_scale=1. if ordinal < 2 else 0.,
        )
        result = self.core.train_step(batch)
        self.presentations += 1
        self.global_step += int(result.optimizer_stepped)
        return result


class EventWindowTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_consecutive_active_and_empty_windows_refresh_masks_keep_adam_and_free_buffers(self):
        trainer = WindowTrainer()
        gated = []
        with tempfile.TemporaryDirectory() as directory:
            with PrefixCorrectionEvent(trainer, output_dir=directory, probe_steps=(1,),
                                       first_window_check=gated.append) as session:
                for index in range(4):
                    trainer.present(index)
                    if index % 2:
                        self.assertIsNone(session.active)
                        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))
                        self.assertFalse(trainer.bundle.model.lm_head._forward_pre_hooks)
                self.assertEqual([row['correction']['status'] for row in session.steps],
                                 ['applied', 'empty_protection'])
            report = session.report()
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['counts']['Adam_completed'], 2)
            self.assertEqual(report['counts']['K_backward_completed'], 4)
            self.assertEqual(report['counts']['windows_completed'], 2)
            self.assertEqual(report['work']['original_response_forwards'], 4)
            self.assertEqual(report['work']['post_k_response_forwards'], 4)
            self.assertEqual(report['work']['post_c_response_forwards'], 2)
            self.assertEqual(report['work']['direct_corrections'], 1)
            self.assertEqual(len(report['probes']), 1)
            self.assertEqual(len(gated), 1)
            self.assertGreater(report['steps'][0]['correction']['global_protected_weight'], 0)
            self.assertEqual(report['steps'][1]['correction']['global_protected_weight'], 0)
            self.assertEqual((trainer.global_step, trainer.presentations, trainer.core._micro), (2, 4, 4))
            for state in trainer.backend.optimizer.optimizer.state.values():
                self.assertEqual(float(state['step']), 2.)
                self.assertTrue(torch.count_nonzero(state['exp_avg']))
            self.assertEqual(len(list(Path(directory).glob('window_*.json'))), 2)
            self.assertEqual(trainer.contract['prefix_correction_event']['arm'], 'KC')
            self.assertIsNone(session.trainer)
            session.close()

    def test_manager_matches_manual_tested_core_for_weights_adam_rng_and_counts(self):
        manual = WindowTrainer()
        for start in (0, 2):
            recorder = PrefixCorrection(manual, record_ids=(f'response_{start}', f'response_{start+1}'))
            with recorder.capture_k():
                manual.present(start)
                manual.present(start + 1)
            recorder.measure_post_k()
            recorder.apply_once()
            if start == 0:
                recorder.measure_post_c()
            recorder.close()
        expected_weights = deepcopy(manual.bundle.model.state_dict())
        expected_adam = deepcopy(manual.backend.optimizer.state_dict())
        expected_rng = capture_rng()
        observed = WindowTrainer()
        with PrefixCorrectionEvent(observed, probe_steps=(1,)):
            for index in range(4):
                observed.present(index)
        self.assertTrue(trees_equal(expected_weights, observed.bundle.model.state_dict()))
        self.assertTrue(trees_equal(expected_adam, observed.backend.optimizer.state_dict()))
        self.assertTrue(trees_equal(expected_rng, capture_rng()))

    def test_reproduction_failure_stops_after_first_actual_adam_with_no_second_window(self):
        trainer = WindowTrainer()
        def mismatch(report):
            self.assertEqual(report['responses'][0]['record_id'], 'response_0')
            raise ValueError('Retained first window mismatch')
        session = PrefixCorrectionEvent(trainer, first_window_check=mismatch, probe_steps=(1,))
        with self.assertRaisesRegex(ValueError, 'Retained first window mismatch'):
            with session:
                for index in range(4):
                    trainer.present(index)
        self.assertEqual(session.counts['Adam_completed'], 1)
        self.assertEqual(session.counts['presentations_completed'], 2)
        self.assertEqual(session.counts['windows_completed'], 0)
        self.assertEqual(trainer.global_step, 0)  # Trainer increments only after wrapper returns.
        self.assertEqual(session.steps[0]['actual_Adam_steps_completed'], 1)
        self.assertEqual(session.steps[0]['status'], 'failed')
        self.assertEqual(session.report()['status'], 'failed')
        self.assertIsNone(session.active)
        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))

    def test_runtime_check_rejects_adam_mutation_after_correction(self):
        trainer = WindowTrainer()
        original = PrefixCorrection.apply_once
        def corrupt(recorder):
            result = original(recorder)
            state = next(iter(trainer.backend.optimizer.optimizer.state.values()))
            state['exp_avg'].add_(1)
            return result
        session = PrefixCorrectionEvent(trainer, probe_steps=())
        with mock.patch.object(PrefixCorrection, 'apply_once', corrupt):
            with self.assertRaisesRegex(ValueError, 'correction changed protected training state: optimizer'):
                with session:
                    trainer.present(0)
                    trainer.present(1)
        self.assertEqual(session.counts['Adam_completed'], 1)
        self.assertEqual(session.report()['work']['direct_corrections'], 1)

    def test_half_window_is_incomplete_without_extra_optimizer_or_replay(self):
        trainer = WindowTrainer()
        with PrefixCorrectionEvent(trainer, probe_steps=()) as session:
            trainer.present(0)
        self.assertEqual(session.status, 'incomplete')
        self.assertEqual(session.counts['Adam_completed'], 0)
        self.assertEqual(session.work['post_k_response_forwards'], 0)
        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))

    def test_nonfinite_failure_evidence_is_saved_without_masking_the_error(self):
        trainer = WindowTrainer()
        def fail(recorder):
            recorder.k_update_norm = float('nan')
            raise ValueError('Original nonfinite failure')
        with tempfile.TemporaryDirectory() as directory:
            session = PrefixCorrectionEvent(trainer, output_dir=directory, probe_steps=())
            with mock.patch.object(PrefixCorrection, 'measure_post_k', fail):
                with self.assertRaisesRegex(ValueError, 'Original nonfinite failure'):
                    with session:
                        trainer.present(0)
                        trainer.present(1)
            saved = json.loads((Path(directory) / 'window_001.json').read_text())
            self.assertEqual(saved['details']['K_update_norm'], 'nan')
            self.assertEqual(saved['summary']['status'], 'failed')
            self.assertEqual(session.counts['Adam_completed'], 1)

    def test_capture_allocation_failure_is_accounted_without_new_forward(self):
        trainer = WindowTrainer()
        session = PrefixCorrectionEvent(trainer, probe_steps=())
        with mock.patch('online_feedback.prefix_correction_event.PrefixCorrection',
                        side_effect=RuntimeError('allocation failed')):
            with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                with session:
                    trainer.present(0)
        self.assertEqual(session.steps[0]['status'], 'failed')
        self.assertEqual(session.counts['presentations_attempted'], 0)
        self.assertEqual(session.counts['Adam_completed'], 0)

    def test_integrity_detects_read_only_parameter_mutation_and_rng_changes(self):
        trainer = WindowTrainer()
        trainer.present(0)
        trainer.present(1)
        snapshot = AuxiliaryIntegrity(trainer)
        with torch.no_grad():
            next(trainer.bundle.model.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'parameter_versions'):
            snapshot.check('read_only', parameters_read_only=True)
        snapshot = AuxiliaryIntegrity(trainer)
        torch.rand(1)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            snapshot.check('read_only', parameters_read_only=True)


class NativeTrainerIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_K_two_step_event_exports_KC_contract_and_keeps_optimizer_progress(self):
        trainer = self.trainer('KC', objective='osd_first_error')
        with PrefixCorrectionEvent(trainer, probe_steps=(1, 2)) as session:
            receipt = trainer.train_event('e1', self.features[:4], passes=1)
        self.assertEqual((receipt['optimizer_steps'], receipt['response_presentations']), (2, 4))
        self.assertEqual(session.counts['Adam_completed'], 2)
        self.assertEqual(session.counts['K_backward_completed'], 4)
        self.assertEqual(session.work['original_response_forwards'], 4)
        self.assertEqual(session.work['post_k_response_forwards'], 4)
        self.assertEqual(session.work['post_c_response_forwards'], 4)
        self.assertEqual([row['step'] for row in session.steps], [1, 2])
        state = self.saved(trainer)
        self.assertEqual(state['algorithm_contract']['prefix_correction_event']['arm'], 'KC')
        self.assertEqual((state['global_step'], state['presentations']), (2, 4))
        self.assertEqual(trainer.core._micro, 4)
        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))
        for moments in trainer.backend.optimizer.optimizer.state.values():
            self.assertEqual(float(moments['step']), 2.)


if __name__ == '__main__':
    unittest.main()
