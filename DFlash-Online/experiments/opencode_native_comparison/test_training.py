"""CPU-only CODE contracts, compact witnesses and native tiny continuity."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import save_file

from experiments.opencode_native_comparison import configuration as C
from experiments.opencode_native_comparison import training as T
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch

# Existing fixtures build 58 tiny tensors and the real native loss/Adam runtime.
TESTS = Path(__file__).resolve().parents[2] / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import test_full_k_training as fixtures

BINDING = 'c' * 64


def state_fixture(arm='O_CODE58', event=0):
    state, old = fixtures.own_state('K', event)
    original = old['original_config_text']
    state['config_text'] = C.build_config(original, arm=arm, output_dir='/code/' + arm)
    state['algorithm_contract'].pop(T.native.CONTRACT_KEY)
    state['algorithm_contract']['online_target_objective']['name'] = T.OBJECTIVES[arm]
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(arm, BINDING)
    for entry in state['events']:
        entry.update(method=arm, dataset='OpenCodeInstruct', objective=T.OBJECTIVES[arm],
                     **{T.CONTRACT_KEY: T.experiment_contract(arm, BINDING)})
        for row in entry['records']:
            row['response_observation'] = dict(record_id=row['record_id'], pass_index=row['pass'],
                sampled_anchors=deepcopy(row['sampled_anchors']), loss=1., ratio_metrics={},
                sum_metrics={'adaflash_response_count': 1.} if arm == 'AREPO_CODE58' else {})
            if arm == 'C_CODE58':
                row['response_observation']['prefix_allocation'] = dict.fromkeys((
                    'changed_positions', 'changed_blocks', 'weight_moved', 'prefix_weight_before',
                    'prefix_weight_after', 'mass_error_abs_sum', 'delta_numerator', 'mass_violations',
                    'bounds_violations', 'support_violations', 'nonfinite_values'), 0.)
    kwargs = dict(arm=arm, binding=BINDING, config_text=state['config_text'],
        original_config_text=original, completed_event=event,
        expected_contract=deepcopy(state['algorithm_contract']),
        expected_recipe=deepcopy(state['backend']['optimizer']['recipe']), expected_cuda_rng_states=0)
    return state, kwargs


class CodeContractTests(unittest.TestCase):
    def setUp(self):
        rng = capture_rng()
        self.addCleanup(lambda: restore_rng(rng))

    def test_context_is_the_only_scientific_configuration_change(self):
        state, args = state_fixture()
        C.validate_config_text(state['config_text'], args['original_config_text'])
        for section, key, value in (('data', 'max_length', 2272), ('training', 'seed', 29),
                ('training', 'learning_rate', 1.5e-4), ('training', 'num_anchors', 256),
                ('online_feedback', 'shuffle_responses', False), ('training', 'objective_chunk_blocks', 4),
                ('model', 'draft_checkpoint_path', '/trained')):
            raw = yaml.safe_load(state['config_text'])
            raw[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_config_text(yaml.safe_dump(raw), args['original_config_text'])

    def test_all_arms_keep_complete_native_states_and_full_horizon(self):
        for arm in T.ARMS:
            for event in (0, 1, 58):
                with self.subTest(arm=arm, event=event):
                    state, args = state_fixture(arm, event)
                    before = deepcopy(state)
                    rng = capture_rng()
                    result = T.validate_snapshot(state, **args)
                    self.assertEqual(result['global_step'], event * 128)
                    self.assertEqual(result['context_limit'], 4096)
                    self.assertFalse(T.differences(before, state))
                    self.assertFalse(T.differences(rng, capture_rng()))

    def test_contract_weight_optimizer_and_history_failures_remain_visible(self):
        source, args = state_fixture('C_CODE58', 1)
        for mutate in (
            lambda s: s['algorithm_contract'][T.CONTRACT_KEY].update(arm='O_CODE58'),
            lambda s: s['algorithm_contract'][T.CONTRACT_KEY]['allocation'].update(relative_cap=.5),
            lambda s: s.update(presentations=255),
            lambda s: s['rng'].pop('numpy'),
            lambda s: s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=.001),
            lambda s: s['events'][0]['records'][0]['response_observation'].update(loss=float('nan')),
            lambda s: s['events'][0]['records'][0]['response_observation']['prefix_allocation'].update(mass_violations=1),
        ):
            bad = deepcopy(source)
            mutate(bad)
            with self.assertRaises(ValueError):
                T.validate_snapshot(bad, **args)

    def test_compact_witness_has_no_weight_tensors_and_roundtrips_exactly(self):
        state, args = state_fixture()
        T.validate_snapshot(state, **args)
        before = deepcopy(state)
        witness = T.initial_witness(state)
        self.assertNotIn('draft_state_dict', witness)
        self.assertTrue(all('sha256' in value for value in witness['weights'].values()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'witness.pt'
            save_torch(path, witness)
            decoded = torch.load(path, weights_only=False)
            self.assertFalse(T.differences(witness, decoded))
        self.assertFalse(T.differences(state, before))

    def test_initial_comparison_checks_weights_rng_Adam_and_config(self):
        first, _ = state_fixture()
        other = deepcopy(first)
        other['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract('C_CODE58', BINDING)
        other['algorithm_contract']['online_target_objective']['name'] = 'osd_first_error'
        a, b = T.initial_witness(first), T.initial_witness(other)
        self.assertTrue(T.compare_initial_witnesses(a, b)['exact_weights_empty_Adam_RNG'])
        for mutate in (
            lambda w: next(iter(w['weights'].values())).update(sha256='0' * 64),
            lambda w: w['rng']['torch'].__setitem__(0, w['rng']['torch'][0] ^ 1),
            lambda w: w['backend']['optimizer']['optimizer_state_dict']['state'].update({0: {}}),
            lambda w: w['algorithm_contract'][T.CONTRACT_KEY].update(context_limit=2272),
            lambda w: w.update(config_text=w['config_text'].replace('0.0003', '0.0002')),
        ):
            bad = deepcopy(b)
            mutate(bad)
            with self.assertRaises(ValueError):
                T.compare_initial_witnesses(a, bad)

    def test_historical_K0_comparison_allows_only_declared_CODE_changes(self):
        historical, old_args = fixtures.own_state('K', 0)
        current, args = state_fixture('C_CODE58', 0)
        result = T.compare_historical_initial(historical, current,
            original_config_text=old_args['original_config_text'], expected_cuda_rng_states=0)
        self.assertTrue(result['exact_weights_empty_Adam_RNG'])
        self.assertEqual(historical['config_text'], old_args['config_text'])
        bad = deepcopy(current)
        bad['rng']['torch'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'Historical released initialization'):
            T.compare_historical_initial(historical, bad,
                original_config_text=old_args['original_config_text'], expected_cuda_rng_states=0)

    def test_installation_uses_only_the_declared_existing_components(self):
        for arm in T.ARMS:
            model = SimpleNamespace(objective=T.OBJECTIVES[arm])
            with patch.object(T, 'install_kplus') as kplus, \
                    patch.object(T, 'install_prefix_allocation') as allocation, \
                    patch.object(T, 'install_adaflash_repo') as arepo:
                T.install_arm(model, arm)
                self.assertEqual(kplus.call_count, int(arm == 'C_CODE58'))
                self.assertEqual(allocation.call_count, int(arm == 'C_CODE58'))
                self.assertEqual(arepo.call_count, int(arm == 'AREPO_CODE58'))

    def test_production_cannot_force_anchors_or_switch_allocation(self):
        trainer = object.__new__(T.CodeTrainer)
        trainer.qualification_only, trainer.arm = False, 'C_CODE58'
        with self.assertRaises(ValueError):
            trainer.qualification_allocation(0.)
        with self.assertRaises(ValueError):
            trainer.train_event('event_001', [], presentation_schedule={})

    def test_recovery_metadata_and_serialized_file_have_fixed_envelopes(self):
        state, _ = state_fixture('C_CODE58',1)
        before=deepcopy(state)
        result=T.recovery_size_plan(state)
        self.assertEqual(result['complete_state_bound_bytes'],6_791_777_312)
        self.assertEqual(result['metadata_envelope_bytes'],500_000_000)
        self.assertGreater(result['tensor_payload_bytes'],0)
        self.assertFalse(T.differences(before,state))
        with patch.object(T,'RECOVERY_METADATA_BYTES',1):
            with self.assertRaisesRegex(RuntimeError,'500MB metadata'):
                T.recovery_size_plan(state)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt'
            save_torch(path,state)
            self.assertEqual(T.validate_recovery_file(path)['status'],'passed')
            with patch.object(T,'MAX_RECOVERY_BYTES',1):
                with self.assertRaisesRegex(RuntimeError,'Serialized recovery'):
                    T.validate_recovery_file(path)

    def test_readonly_resume_authenticates_own_complete_state_without_work(self):
        from experiments.ok_coding.storage import identity
        state,args=state_fixture('C_CODE58',1)
        initial,_=state_fixture('C_CODE58',0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint=root/'export'
            checkpoint.mkdir()
            save_file(state['draft_state_dict'],checkpoint/'model.safetensors')
            save_torch(root/'state.pt',state)
            save_torch(root/'initial.pt',T.initial_witness(initial))
            kwargs=dict(state_path=root/'state.pt',export_path=checkpoint,
                initial_witness_path=root/'initial.pt',arm=args['arm'],binding=BINDING,
                config_text=args['config_text'],original_config_text=args['original_config_text'],
                completed_event=1,state_identity=identity(root/'state.pt'),
                export_identity=identity(checkpoint/'model.safetensors'),expected_cuda_rng_states=0)
            rng=capture_rng()
            result=T.authenticate_resume(**kwargs)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['optimizer_applications'],0)
            self.assertEqual(result['trainer_initializations'],0)
            self.assertFalse(result['automatic_resume'])
            self.assertFalse(T.differences(rng,capture_rng()))
            with self.assertRaises(ValueError):
                T.authenticate_resume(**dict(kwargs,completed_event=2))
            (checkpoint/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                T.authenticate_resume(**kwargs)


class TinyCodeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('OPENCODE_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('OPENCODE_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('OPENCODE_TINY_ARM'):
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def build(self, name, arm, **kwargs):
        path = self.root / (name + '.yaml')
        # Resume keeps the scientific config text exactly, despite its new state destination.
        path.write_text(C.build_config(self.original_config_text, arm=arm, output_dir=self.root/'common'))
        trainer = T.CodeTrainer(path, self.root/name, arm=arm, binding=BINDING,
            original_config_text=self.original_config_text, expected_cuda_rng_states=0, **kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_real_own_feedback_event_and_exact_resume_in_isolated_processes(self):
        arm = os.environ.get('OPENCODE_TINY_ARM')
        if arm is None:
            root = Path(__file__).resolve().parents[2]
            for current in T.ARMS:
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENCODE_TINY_ARM=current)
                environment['PYTHONPATH'] = os.pathsep.join(str(root / p) for p in ('.', 'tests', 'training', 'inference'))
                run = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                    __name__ + '.TinyCodeTrainingTests.test_real_own_feedback_event_and_exact_resume_in_isolated_processes'],
                    cwd=root, env=environment, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, current + '\n' + run.stdout + run.stderr)
            return
        trainer = self.build('first', arm)
        released = self.root/'released'
        released.mkdir()
        save_file(dict(trainer.bundle.draft_model.state_dict()), released/'model.safetensors')
        before = capture_rng()
        initial = trainer.save_initial()
        self.assertFalse(T.differences(before, capture_rng()))
        self.assertFalse((trainer.output_root/'initial_state.pt').exists())
        self.assertTrue(Path(initial['witness_path']).is_file())
        result = trainer.train_event('event_001', self.features)
        self.assertEqual(result['method'], arm)
        self.assertEqual(result['optimizer_steps'], 128)
        self.assertEqual(result['response_presentations'], 256)
        trainer.verify_boundary(result['export_path'])
        boundary = self.root/'boundary.pt'
        boundary.write_bytes(trainer.latest.read_bytes())
        saved = torch.load(boundary, weights_only=False)
        next_features = []
        for index, path in enumerate(self.features):
            packet = torch.load(path, weights_only=True)
            packet['record_id'] = f'event2_{index}'
            packet['label_binding']['record_id'] = packet['record_id']
            destination = self.root/f'next_{index}.pt'
            torch.save(packet, destination)
            next_features.append(destination)
        expected = trainer.train_event('event_002', next_features)
        final = torch.load(trainer.latest, weights_only=False)
        trainer.close()
        resumed = self.build('resumed', arm, resume_state=boundary, completed_event=1)
        self.assertTrue(T.verify_live(resumed, saved)['exact_resume'])
        actual = resumed.train_event('event_002', next_features)
        resumed.verify_boundary(actual['export_path'])
        decoded = torch.load(resumed.latest, weights_only=False)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertFalse(T.differences(final[key], decoded[key]), key)
        for key in ('record_id', 'pass', 'sampled_anchors', 'metrics'):
            self.assertEqual([row[key] for row in expected['records']], [row[key] for row in actual['records']])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
