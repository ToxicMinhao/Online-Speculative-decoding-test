"""Real event-trainer wiring with tiny DFlash and the upstream BF16 optimizer.

Only model loading, distributed wrapping and export are replaced. Configuration,
strategy selection, loss, sampling, accumulation, optimizer and restart are real.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from specforge.algorithms.common.providers import StepRuntimeConfig
from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend

from online_feedback import training as T
from online_feedback.objective import OSDTrainStrategy
from online_feedback.qualify import check_restart
from test_objective_integration import tiny_author
from test_osd_objective import osd_batch
from test_training_contract import trees_equal


class OSDTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        template = Path(__file__).parents[1] / 'configs/qualification.yaml'
        self.config = self.root / 'config.yaml'
        self.config.write_text(template.read_text())

        def load_tiny_bundle(*args, **kwargs):
            model = tiny_author()
            return SimpleNamespace(
                model=model, draft_model=model.draft_model, target_head=None,
                target_hidden_size=16, target_vocab_size=29,
                strategy_kwargs=StepRuntimeConfig({}, {'tiny_model': True}, frozenset()),
            )

        original_prepare = FSDPTrainingBackend.prepare_model

        def prepare_without_distributed(backend, model, *, optimizer_target):
            return original_prepare(backend, model, wrap=False, optimizer_target=optimizer_target)

        def export_fixture(state, config, output, **kwargs):
            Path(output).mkdir()
            (Path(output) / 'model.safetensors').write_bytes(b'tiny export fixture')

        patches = [
            mock.patch.object(T.SpecForgeEventTrainer, '_initialize_distributed'),
            mock.patch('specforge.training.assembly.build_model_bundle', side_effect=load_tiny_bundle),
            mock.patch.object(FSDPTrainingBackend, 'prepare_model', new=prepare_without_distributed),
            mock.patch('specforge.export.to_hf.export_to_hf', side_effect=export_fixture),
            mock.patch('specforge.optimizer.print_on_rank0'),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.features = []
        for index, length in enumerate((12, 10)):
            raw = osd_batch(tiny_author())
            raw = {name: value[:, :length] for name, value in raw.items()}
            raw.update(record_id=f'response_{index}', prompt_length=4)
            raw['label_binding'] = {
                'capture_backend': 'same_frozen_HF_target_canonical_replay',
                'record_id': raw['record_id'], 'prompt_length': 4,
                'target_argmax_ids': raw['input_ids'][0, 4:].tolist(), 'label_mismatches': [],
            }
            path = self.root / f'feedback_{index}.pt'
            torch.save(raw, path)
            self.features.append(path)

    def trainer(self, name='run', **kwargs):
        trainer = T.SpecForgeEventTrainer(self.config, self.root / name, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def saved(self, trainer):
        return torch.load(trainer.latest, map_location='cpu', weights_only=False)

    def test_runtime_uses_osd_and_real_optimizer_for_both_response_reductions(self):
        for reduction in ('response_mean', 'upstream_weight_mean'):
            with self.subTest(reduction=reduction):
                trainer = self.trainer(reduction, objective='osd', reduction=reduction)
                self.assertIsInstance(trainer.author_strategy, OSDTrainStrategy)
                self.assertIsInstance(trainer.backend.optimizer, BF16Optimizer)
                self.assertEqual(trainer.bundle.model.objective, 'osd')
                original = {name: value.clone() for name, value in trainer.bundle.draft_model.state_dict().items()}
                event = trainer.train_event('e1', self.features)
                self.assertEqual((event['optimizer_steps'], event['response_presentations']), (2, 4))
                self.assertEqual([row['record_id'] for row in event['records']],
                                 ['response_0', 'response_1', 'response_0', 'response_1'])
                self.assertEqual([row['optimizer_stepped'] for row in event['records']],
                                 [False, True, False, True])
                self.assertFalse(trees_equal(original, trainer.bundle.draft_model.state_dict()))
                self.assertIsNone(trainer.bundle.model.reference)
                contract = self.saved(trainer)['algorithm_contract']['online_target_objective']
                self.assertEqual(contract['teacher'], 'frozen_target_predecessor_hidden')
                self.assertEqual((contract['temperature'], contract['hard_label_mix']), (1.0, 0.0))
                self.assertEqual(contract['denominator_epsilon'], 1e-6)
                self.assertNotIn('reference', contract)

    def test_missing_or_mismatched_teacher_is_rejected_before_any_step(self):
        trainer = self.trainer(objective='osd')
        original = {name: value.clone() for name, value in trainer.bundle.draft_model.state_dict().items()}
        for teacher in (None, torch.ones(1, 12, 15), torch.ones(1, 11, 16)):
            with self.subTest(shape=None if teacher is None else teacher.shape):
                raw = torch.load(self.features[0], weights_only=True)
                if teacher is None:
                    del raw['target_last_hidden_states']
                else:
                    raw['target_last_hidden_states'] = teacher
                path = self.root / 'invalid.pt'
                torch.save(raw, path)
                with self.assertRaisesRegex(ValueError, 'teacher features|Final target features'):
                    trainer.train_event('invalid', [path], passes=2)
                self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
                self.assertEqual(trainer.events, [])
                self.assertFalse((trainer.events_root / 'invalid').exists())
                self.assertTrue(trees_equal(original, trainer.bundle.draft_model.state_dict()))

    def test_resume_keeps_osd_optimizer_anchors_and_next_event_exact(self):
        uninterrupted = self.trainer('original', objective='osd')
        uninterrupted.train_event('e1', self.features)
        checkpoint = self.root / 'checkpoint.pt'
        checkpoint.write_bytes(uninterrupted.latest.read_bytes())
        expected_event = uninterrupted.train_event('e2', self.features)
        expected = self.saved(uninterrupted)
        resumed = self.trainer('resumed', objective='osd', resume_state=checkpoint)
        actual_event = resumed.train_event('e2', self.features)
        actual = self.saved(resumed)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected[key], actual[key]), key)
        self.assertEqual([r['sampled_anchors'] for r in expected_event['records']],
                         [r['sampled_anchors'] for r in actual_event['records']])

    def test_osd_checkpoint_cannot_resume_into_ce(self):
        osd = self.trainer('osd', objective='osd')
        osd.train_event('e1', self.features)
        ce = self.trainer('ce', objective='hard_ce')
        original = {name: value.clone() for name, value in ce.bundle.draft_model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'Resume algorithm or schedule differs'):
            ce._restore(osd.latest)
        self.assertTrue(trees_equal(original, ce.bundle.draft_model.state_dict()))
        self.assertEqual(ce.global_step, 0)

    def test_qualification_restart_preserves_osd_and_nondefault_reduction(self):
        trainer = self.trainer(objective='osd', reduction='upstream_weight_mean')
        trainer.train_event('e1', self.features)
        controller = SimpleNamespace(trainer=trainer)
        result = check_restart(controller, self.config, trainer.output_root)
        self.addCleanup(controller.trainer.close)
        self.assertTrue(all(result.values()))
        self.assertEqual(controller.trainer.objective, 'osd')
        self.assertEqual(controller.trainer.reduction, 'upstream_weight_mean')
        self.assertIsInstance(controller.trainer.author_strategy, OSDTrainStrategy)


if __name__ == '__main__':
    unittest.main()
