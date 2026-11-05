"""Additive K+ binding, unchanged old guards, and tiny native continuity."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import save_file
from experiments.full_k_gsm8k import training as old
from experiments.kplus_full_gsm8k import training as T
from online_feedback.rng import capture_rng
import test_full_k_training as native_fixtures
from test_full_k_training import own_state, BINDING


def plus_state(event=0):
    state, kwargs = own_state('K', event)
    del state['algorithm_contract'][old.CONTRACT_KEY]
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(BINDING)
    kwargs.pop('arm')
    kwargs['expected_contract'] = deepcopy(state['algorithm_contract'])
    for entry in state['events']:
        entry.update(method='KPLUS', redistribution=T.kplus_contract())
        for row in entry['records']:
            row['response_observation'] = dict(record_id=row['record_id'], pass_index=row['pass'],
                sum_metrics=dict(kplus_response_count=1, kplus_W1=1., kplus_W2=2.,
                                 kplus_first_error_weight_after=3.))
    return state, kwargs


class KPlusStateTests(unittest.TestCase):
    def test_exact_full_horizon_state_and_native_guard_projection(self):
        for event in (0, 1, 58):
            state, args = plus_state(event)
            before = capture_rng()
            result = T.validate_snapshot(state, **args)
            self.assertEqual(result['arm'], 'KPLUS')
            self.assertEqual(result['global_step'], event * 128)
            self.assertFalse(old.differences(before, capture_rng()))
            self.assertNotIn(old.CONTRACT_KEY, state['algorithm_contract'])

    def test_foreign_method_rejected_before_native_validator(self):
        state, args = plus_state()
        state['algorithm_contract'][T.CONTRACT_KEY]['redistribution']['redistribution_fraction'] = .1
        with patch.object(old, 'validate_snapshot') as native, self.assertRaisesRegex(ValueError, 'KPLUS'):
            T.validate_snapshot(state, **args)
        native.assert_not_called()

    def test_old_two_arm_guard_does_not_admit_kplus(self):
        with self.assertRaises(ValueError):
            old.experiment_contract('KPLUS', BINDING)
        state, args = plus_state()
        with self.assertRaises(ValueError):
            old.validate_snapshot(state, arm='K', **args)

    def test_changed_rng_adam_and_conservation_fail(self):
        state, args = plus_state(1)
        for mutate in (
            lambda s: s['rng'].pop('numpy'),
            lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=1.5e-4),
            lambda s: s['events'][0]['records'][0]['response_observation']['sum_metrics'].update(kplus_first_error_weight_after=4.),
        ):
            bad = deepcopy(state)
            mutate(bad)
            with self.assertRaises(ValueError):
                T.validate_snapshot(bad, **args)


class KPlusTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('KPLUS_TINY_CPU_CHILD') == '1':
            native_fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('KPLUS_TINY_CPU_CHILD') == '1':
            native_fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('KPLUS_TINY_CPU_CHILD') == '1':
            native_fixtures.FullKTinyIntegrationTests.setUp(self)

    def build_plus(self, name, **kwargs):
        config = self.root / (name + '.yaml')
        config.write_text(self.original_config_text)
        trainer = T.KPlusTrainer(config, self.root / name, binding=BINDING,
            original_config_text=self.original_config_text, expected_cuda_rng_states=0, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_native_initial_identity_complete_event_and_exact_restore(self):
        if os.environ.get('KPLUS_TINY_CPU_CHILD') != '1':
            # GPU visibility must be disabled before the child imports Torch;
            # patching a live parent's CUDA state would hide an unsafe fixture.
            repository = Path(__file__).resolve().parents[1]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', KPLUS_TINY_CPU_CHILD='1')
            environment['PYTHONPATH'] = os.pathsep.join((str(repository), str(repository / 'tests')))
            result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                'test_kplus_full_training.KPlusTinyTests.test_native_initial_identity_complete_event_and_exact_restore'],
                cwd=repository, env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        original = native_fixtures.FullKTinyIntegrationTests.build(self, 'original')
        released = self.root / 'released'
        released.mkdir()
        save_file(dict(original.bundle.draft_model.state_dict()), released / 'model.safetensors')
        initial = original.save_initial()['state_path']
        original.close()
        trainer = self.build_plus('plus')
        before = capture_rng()
        check = trainer.save_initial(initial)
        self.assertTrue(check['original_K_initialization']['exact_released_weights_empty_adam_rng'])
        self.assertFalse(old.differences(before, capture_rng()))
        first = trainer.train_event('event_001', self.features)
        trainer.verify_boundary(first['export_path'])
        state = torch.load(trainer.latest, weights_only=False)
        checkpoint = self.root / 'boundary.pt'
        checkpoint.write_bytes(trainer.latest.read_bytes())
        self.assertEqual(first['objective_work']['work']['response_forwards'], 256)
        self.assertEqual(first['objective_work']['work']['draft_forwards'], 256)
        self.assertEqual(sum(r['response_observation']['sum_metrics']['kplus_response_count'] for r in first['records']), 256)
        trainer.close()
        restored = self.build_plus('restored', resume_state=checkpoint, completed_event=1)
        self.assertTrue(old.verify_live(restored, state)['exact_resume'])
        self.assertEqual(restored.global_step, 128)
        self.assertIsNotNone(restored.bundle.model.first_error_redistribution)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
