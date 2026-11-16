"""CPU recovery and native-loop integration for the fixed allocation candidate."""
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.bounded_prefix_allocation import training as T
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng
import test_recent_repair_retention_training as retained_fixtures

BINDING = "d" * 64


def origin():
    with patch.object(torch.cuda, "is_initialized", return_value=False):
        return retained_fixtures.CompleteStateTests().origin()


def candidate():
    source, original = origin()
    state = T.fork_state(source, binding=BINDING, config_text=source['config_text'], run_id=source['run_id'])
    arguments = dict(binding=BINDING, config_text=source['config_text'], original_config_text=original,
        completed_event=6, expected_contract=deepcopy(state['algorithm_contract']),
        expected_recipe=deepcopy(state['backend']['optimizer']['recipe']), expected_cuda_rng_states=0)
    return source, state, arguments


class CandidateStateTests(unittest.TestCase):
    def test_fork_preserves_tensors_moments_rng_and_history(self):
        source, state, arguments = candidate()
        for key in ('draft_state_dict', 'backend', 'rng', 'events'):
            self.assertIs(source[key], state[key])
        before = capture_rng()
        receipt = T.validate_snapshot(state, **arguments)
        self.assertEqual(receipt['global_step'], 768)
        self.assertEqual(differences(before, capture_rng()), [])
        self.assertNotIn(T.CONTRACT_KEY, source['algorithm_contract'])
        self.assertEqual(state['algorithm_contract'][T.CONTRACT_KEY]['allocation']['relative_cap'], .25)
        self.assertEqual(state['backend']['optimizer']['recipe']['optimizer_options']['lr'], 3e-4)

    def test_serialized_contract_is_required_and_changed_learning_settings_rejected(self):
        _, state, arguments = candidate()
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        T.validate_snapshot(restored, **arguments)
        self.assertEqual(differences(state, restored), [])
        changes = [
            lambda x: x['algorithm_contract'][T.CONTRACT_KEY]['allocation'].update(relative_cap=0.),
            lambda x: x['algorithm_contract'].update(native_prefix_roles='E'),
            lambda x: x['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=.000288),
            lambda x: x.update(global_step=769),
        ]
        for change in changes:
            bad = deepcopy(state)
            change(bad)
            with self.assertRaises(ValueError):
                T.validate_snapshot(bad, **arguments)
        with self.assertRaises(ValueError):
            T.fork_state(restored, binding=BINDING, config_text=state['config_text'], run_id='again')

    def test_own_controller_reinstallation_rejects_foreign_or_withholding_hooks(self):
        from types import SimpleNamespace
        from online_feedback.prefix_allocation import PrefixAllocation
        trainer = object.__new__(T.CandidateTrainer)
        trainer.relative_cap = .25
        trainer.qualification_only = False
        trainer._qualification_cap = .25
        model = SimpleNamespace(objective='osd_first_error', loss_dtype=torch.bfloat16,
            first_error_redistribution=object(), prefix_allocation=PrefixAllocation())
        trainer.bundle = SimpleNamespace(model=model)
        trainer._install_allocation()
        trainer._check_allocation()
        model.prefix_allocation = SimpleNamespace(relative_cap=.25)
        with self.assertRaises(ValueError):
            trainer._install_allocation()
        model.prefix_allocation = PrefixAllocation()
        model.native_prefix_roles = SimpleNamespace(role='E')
        with self.assertRaises(ValueError):
            trainer._check_allocation()
        with self.assertRaises(ValueError):
            trainer._install_allocation()

    def test_native_configuration_pins_are_unchanged(self):
        _, state, arguments = candidate()
        for section, field, value in [('training','num_anchors',32), ('training','learning_rate',.000288),
                                      ('online_feedback','weight_decay',0), ('data','max_length',2048)]:
            raw = yaml.safe_load(state['config_text'])
            raw[section][field] = value
            with self.assertRaises(ValueError):
                T.native.validate_config_text(yaml.safe_dump(raw), 'K', arguments['original_config_text'])


class CandidateTinyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('ALLOCATION_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('ALLOCATION_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('ALLOCATION_TINY_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def subprocess_or_continue(self):
        if os.environ.get('ALLOCATION_TINY_CHILD') == '1':
            return False
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', ALLOCATION_TINY_CHILD='1', OMP_NUM_THREADS='1')
        env['PYTHONPATH'] = os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
            __name__ + '.' + type(self).__name__ + '.' + self._testMethodName], cwd=root,
            env=env, text=True, capture_output=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def make_parent(self):
        from test_full_k_training import FullKTinyIntegrationTests
        from experiments.lost_repair.training_probe import sample_schedule
        from safetensors.torch import save_file
        base = FullKTinyIntegrationTests.build(self, 'base')
        T.install_kplus(base.bundle.model)
        state = deepcopy(T.native.live_state(base))
        history, _ = origin()
        state.update(events=history['events'], global_step=768, presentations=1536)
        state['algorithm_contract'].pop(T.native.CONTRACT_KEY)
        state['algorithm_contract'].update({key:value for key,value in history['algorithm_contract'].items()
            if key in (T.historical.CONTRACT_KEY, T.geometry.CONTRACT_KEY)})
        optimizer = state['backend']['optimizer']['optimizer_state_dict']
        for index, parameter in zip(optimizer['param_groups'][0]['params'], base.backend.optimizer.model_params):
            optimizer['state'][index] = dict(step=torch.tensor(768.), exp_avg=torch.zeros_like(parameter),
                                            exp_avg_sq=torch.zeros_like(parameter))
        schedule = sample_schedule(base.bundle.model, self.features, state['rng'], max_length=base.max_length)
        schedule.pop('final_rng')
        path = self.root/'parent.pt'
        T.save_torch(path, state)
        export = self.root/'parent_export'
        export.mkdir()
        save_file(state['draft_state_dict'], export/'model.safetensors')
        base.close()
        return path, export, schedule

    def build_candidate(self, name, path, export, schedule=None, completed_event=6, qualification_only=False):
        trainer = T.CandidateTrainer(self.config, self.root/name, binding=BINDING,
            origin_state=path, origin_export=export, original_config_text=self.original_config_text,
            completed_event=completed_event, schedule=schedule,
            expected_cuda_rng_states=0, qualification_only=qualification_only)
        self.addCleanup(trainer.close)
        return trainer

    def test_persistent_C7_and_own_C8_resume_preserve_contract_and_dynamic_counters(self):
        if self.subprocess_or_continue():
            return
        path, export, schedule = self.make_parent()
        trainer = self.build_candidate('C7', path, export, schedule)
        event7 = trainer.train_event('event_007', self.features)
        trainer.verify_boundary(event7['export_path'])
        saved = self.root/'C7.pt'
        saved.write_bytes(trainer.latest.read_bytes())
        self.assertEqual((trainer.global_step, trainer.presentations), (896,1792))
        self.assertTrue(all('prefix_allocation' in row['response_observation'] for row in event7['records']))
        self.assertIsNone(getattr(trainer.bundle.model, 'native_prefix_roles', None))
        new_features = []
        for index, old in enumerate(self.features):
            raw = torch.load(old, weights_only=True)
            raw['record_id'] = f'own_C7_feedback_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            destination = self.root/f'own_{index}.pt'
            torch.save(raw, destination)
            new_features.append(destination)
        uninterrupted = trainer.train_event('event_008', new_features)
        final = torch.load(trainer.latest, weights_only=False)
        trainer.close()
        resumed = self.build_candidate('C8_reloaded', saved, event7['export_path'], completed_event=7)
        self.assertEqual(resumed.bundle.model.prefix_allocation.relative_cap, .25)
        repeated = resumed.train_event('event_008', new_features)
        resumed.verify_boundary(repeated['export_path'])
        reloaded = torch.load(resumed.latest, weights_only=False)
        for key in ('draft_state_dict','backend','rng','global_step','presentations'):
            self.assertEqual(differences(final[key], reloaded[key]), [], key)
        for key in ('record_id','pass','sampled_anchors','metrics'):
            self.assertEqual([row[key] for row in uninterrupted['records']], [row[key] for row in repeated['records']])
        self.assertEqual((resumed.global_step, resumed.presentations), (1024,2048))
        with self.assertRaises(ValueError):
            resumed.train_event('event_009', new_features)
        with self.assertRaises(ValueError):
            resumed.qualification_allocation(0.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_qualification_serializes_own_mid_event_and_repeats_next_update(self):
        if self.subprocess_or_continue():
            return
        path, export, schedule = self.make_parent()
        trainer = self.build_candidate('disposable', path, export, schedule, qualification_only=True)
        features = T.SpecForgeEventTrainer._prepare_event(trainer, 'event_007', self.features, 2)
        trainer._active_event = 'qualification'
        (trainer.events_root/trainer._active_event).mkdir()
        trainer._presentation_schedule = T.PresentationSchedule(T.retention.schedule_adapter(schedule,features),
            [item['record_id'] for item in features], 2, features=features)
        trainer.bundle.model.begin_event()
        order = torch.randperm(len(features)).tolist()
        trainer.run_pair([features[index] for index in order[:2]],0,0)
        snapshot = self.root/'temporary.pt'
        from experiments.functional_step_geometry.qualify import runtime_state
        runtime = runtime_state(trainer.bundle.model)
        trainer.save_qualification(snapshot)
        saved = torch.load(snapshot, weights_only=False)
        self.assertIn('draft_state_dict', saved)
        self.assertEqual(saved['qualification_state']['schedule']['position'],2)
        rows = trainer.run_pair([features[index] for index in order[2:4]],0,2)
        end = deepcopy(T.native.live_state(trainer))
        physical_work = deepcopy(trainer.work)
        trainer.restore_qualification(snapshot,runtime=runtime)
        self.assertEqual(trainer.work, physical_work)
        replay = trainer.run_pair([features[index] for index in order[2:4]],0,2)
        self.assertEqual(differences(end, T.native.live_state(trainer)), [])
        for key in ('metrics','sampled_anchors'):
            self.assertEqual([row[key] for row in rows],[row[key] for row in replay])
        self.assertEqual(trainer.global_step,770)
        self.assertEqual(trainer._presentation_schedule.position,4)
        self.assertTrue(all(value['step'].item()==769 for value in saved['backend']['optimizer']['optimizer_state_dict']['state'].values()))
        trainer.qualification_allocation(None)
        self.assertIsNone(trainer.bundle.model.prefix_allocation)
        trainer.qualification_allocation(0.)
        self.assertEqual(trainer.bundle.model.prefix_allocation.relative_cap,0.)
        with self.assertRaises(ValueError):
            trainer.train_event('event_007',self.features)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
