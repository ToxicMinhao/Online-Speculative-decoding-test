"""CPU-only checks for persistent withholding, continuation and exact LR forks."""
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.persistent_prefix_withholding import state as S
from experiments.persistent_prefix_withholding.training import PersistentPrefixTrainer, ROLE_BOUNDARIES, boundary_gradient_check
from experiments.ok_continuation.resume import differences
from online_feedback.native_prefix_roles import NativePrefixRoles, controlled_rows, role_masks
from online_feedback.first_error import first_error_mask
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng
import test_recent_repair_retention_training as retained_fixtures

BINDING = "c" * 64


def original():
    # This is an explicitly CPU synthetic state, regardless of another test's
    # visible-device setup. Never query or manufacture a GPU RNG fixture.
    with patch.object(torch.cuda, "is_initialized", return_value=False):
        return retained_fixtures.CompleteStateTests().origin()


def config_for(state, arm):
    config = yaml.safe_load(state['config_text'])
    config['training']['learning_rate'] = S.arm_lr(arm)
    return yaml.safe_dump(config)


def fork(arm="E"):
    origin, original_config = original()
    config = config_for(origin, arm)
    own, receipt = S.fork_state(origin, arm=arm, binding=BINDING, config_text=config, run_id=origin['run_id'])
    kwargs = dict(arm=arm, binding=BINDING, config_text=config, original_config_text=original_config,
        completed_event=6, expected_contract=deepcopy(own['algorithm_contract']),
        expected_recipe=deepcopy(own['backend']['optimizer']['recipe']), expected_cuda_rng_states=0)
    return origin, own, receipt, kwargs


class BoundaryGradientTests(unittest.TestCase):
    def test_zero_current_error_gradient_is_valid_but_prefix_leak_and_nonfinite_fail(self):
        row = dict(prefix_occurrences=10, error_occurrences=0,
                   gradient=dict(finite=True, excluded_nonzero=0, prefix_nonzero=0))
        receipt = dict(role="E", responses=[dict(chunks=[row])])
        self.assertEqual(boundary_gradient_check(receipt)["status"], "passed")
        for mutation in (dict(prefix_nonzero=1), dict(excluded_nonzero=1), dict(finite=False)):
            changed = deepcopy(receipt)
            changed["responses"][0]["chunks"][0]["gradient"].update(mutation)
            self.assertEqual(boundary_gradient_check(changed)["status"], "failed")
        changed = deepcopy(receipt)
        changed["responses"][0]["chunks"][0]["gradient"] = None
        self.assertEqual(boundary_gradient_check(changed)["status"], "failed")


class PersistentStateTests(unittest.TestCase):
    def test_original_origin_accepted_before_migration(self):
        state, original_config = original()
        before = deepcopy(state)
        S.validate_legacy_origin(state, original_config, 6, expected_cuda_rng_states=0)
        self.assertEqual(differences(state, before), [])

    def test_scalar_migration_changes_only_declared_metadata_not_tensors_or_history(self):
        origin, own, receipt, kwargs = fork('S')
        self.assertEqual(S.SCALAR_LR.hex(), '0x1.2dfd694ccab3ep-12')
        self.assertNotEqual(S.SCALAR_LR, 2.88e-4)
        for name in ('draft_state_dict', 'rng', 'events'):
            self.assertIs(origin[name], own[name])
        before = origin['backend']['optimizer']['optimizer_state_dict']
        after = own['backend']['optimizer']['optimizer_state_dict']
        self.assertIs(before['state'], after['state'])
        self.assertEqual(before['param_groups'][0]['lr'], 3e-4)
        self.assertEqual(after['param_groups'][0]['lr'], S.SCALAR_LR)
        self.assertEqual(own['backend']['optimizer']['recipe']['optimizer_options']['lr'], S.SCALAR_LR)
        self.assertEqual(receipt['learning_rate_repr'], repr(S.SCALAR_LR))
        rng = capture_rng()
        S.validate_snapshot(own, **kwargs)
        self.assertEqual(differences(rng, capture_rng()), [])
        self.assertEqual(origin['events'][0]['records'][1]['learning_rate'], 3e-4)

    def test_validation_rejects_foreign_arm_recipe_literal_rounding_and_historical_change(self):
        _, own, _, kwargs = fork('S')
        changes = [
            lambda x:x['algorithm_contract'][S.CONTRACT_KEY].update(arm='E'),
            lambda x:x['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=2.88e-4),
            lambda x:x['backend']['optimizer']['recipe']['optimizer_options'].update(lr=2.88e-4),
            lambda x:x['events'][0]['records'][1].update(learning_rate=S.SCALAR_LR),
            lambda x:x.update(global_step=769),
            lambda x:x['algorithm_contract'].update(unapproved_extension={}),
        ]
        for change in changes:
            bad = deepcopy(own)
            change(bad)
            with self.assertRaises(ValueError):
                S.validate_snapshot(bad, **kwargs)

    def test_serialized_role_contract_and_own_state_survive_round_trip(self):
        _, own, _, kwargs = fork('E')
        stream = io.BytesIO()
        torch.save(own, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        self.assertEqual(differences(own, restored), [])
        self.assertEqual(restored['algorithm_contract'][S.CONTRACT_KEY]['prefix_gradient'], 'stopped')
        S.validate_snapshot(restored, **kwargs)
        with self.assertRaises(ValueError):
            S.fork_state(restored, arm='E', binding=BINDING, config_text=kwargs['config_text'], run_id='again')

    def test_qualification_exception_does_not_relax_production_N_boundary(self):
        source, text = original()
        with self.assertRaises(ValueError):
            S.fork_state(source, arm='N', binding=BINDING, config_text=source['config_text'], run_id=source['run_id'])
        own, _ = S.fork_state(source, arm='N', binding=BINDING, config_text=source['config_text'],
                             run_id=source['run_id'], qualification_only=True)
        args = dict(arm='N', binding=BINDING, config_text=source['config_text'], original_config_text=text,
            completed_event=6, expected_contract=own['algorithm_contract'],
            expected_recipe=own['backend']['optimizer']['recipe'], expected_cuda_rng_states=0)
        with self.assertRaises(ValueError):
            S.validate_snapshot(own, **args)
        S.validate_snapshot(own, **args, qualification_only=True)

    def test_current_config_allows_only_declared_lr_and_output_identity(self):
        _, own, _, kwargs = fork('E')
        for section, field, value in [('training','num_anchors',32),('training','seed',23),
                                      ('online_feedback','weight_decay',0),('data','max_length',2048)]:
            changed = yaml.safe_load(own['config_text'])
            changed[section][field] = value
            with self.assertRaises(ValueError):
                S.validate_config_text(yaml.safe_dump(changed), 'E', kwargs['original_config_text'])


class PersistentRoleTests(unittest.TestCase):
    def test_role_reinstated_before_each_forward_and_no_hidden_N(self):
        trainer = object.__new__(PersistentPrefixTrainer)
        trainer._arm = 'E'
        trainer.qualification_only = False
        trainer.qualification_role = None
        trainer.presentation_ordinal = 0
        trainer.role_controller = NativePrefixRoles('E')
        trainer.bundle = SimpleNamespace(model=SimpleNamespace(native_prefix_roles=None))
        for ordinal in (0, 1, 2, 127, 128, 255):
            trainer.presentation_ordinal = ordinal
            trainer.role_controller.role = 'N'
            trainer.bundle.model.native_prefix_roles = None
            trainer._install_current_role()
            self.assertIs(trainer.bundle.model.native_prefix_roles, trainer.role_controller)
            self.assertEqual(trainer.role_controller.role, 'E')
            self.assertEqual(trainer.role_controller.capture_gradients, ordinal in ROLE_BOUNDARIES)
        with self.assertRaises(AttributeError):
            trainer.arm = 'N'

    def test_native_paths_refuse_unintended_controller(self):
        trainer = object.__new__(PersistentPrefixTrainer)
        trainer.role_controller = None
        trainer.bundle = SimpleNamespace(model=SimpleNamespace(native_prefix_roles=NativePrefixRoles('E')))
        with self.assertRaises(ValueError):
            trainer._install_current_role()

    def test_repair_changes_current_roles_and_original_denominator_is_preserved(self):
        labels = torch.tensor([[[0, 1, 2, 3]]])
        weights = torch.tensor([[[0., 1., .5, .25]]])
        proposals = torch.tensor([[[0, 1, 9, 3]]])
        selected = first_error_mask(proposals, labels, weights)
        prefix, errors = role_masks(proposals, labels, weights, selected)
        self.assertEqual(errors.nonzero()[:, -1].tolist(), [2])
        terms = torch.tensor([[[0., 2., 4., 8.]]], requires_grad=True)
        denominator = (weights * selected).sum() + 1e-6
        native = (terms * weights * selected).sum() / denominator
        error = (controlled_rows(terms, prefix, errors, 'E') * weights * selected).sum() / denominator
        self.assertTrue(torch.equal(native, error))
        error.backward()
        self.assertEqual(float(terms.grad[0,0,1]), 0.)
        self.assertGreater(float(terms.grad[0,0,2]), 0.)
        proposals[0,0,2] = 2
        proposals[0,0,3] = 9
        _, new_errors = role_masks(proposals, labels, weights, first_error_mask(proposals, labels, weights))
        self.assertEqual(new_errors.nonzero()[:, -1].tolist(), [3])

    def test_two_persistent_updates_and_resume_preserve_own_adam_continuity(self):
        def make():
            model = torch.nn.Linear(3, 1, bias=False).to(torch.bfloat16)
            model.weight.data.copy_(torch.tensor([[.125, -.125, .25]], dtype=torch.bfloat16))
            return model, DirectBF16AdamW(model)
        def update(model, optimizer, predictions):
            values = model(torch.eye(3, dtype=torch.bfloat16)).view(1,1,3).square()
            labels = torch.tensor([[[0, 1, 2]]])
            weights = torch.tensor([[[0., 1., .5]]])
            selected = first_error_mask(predictions, labels, weights)
            prefix, errors = role_masks(predictions, labels, weights, selected)
            loss = (controlled_rows(values, prefix, errors, 'E') * weights * selected).sum() / ((weights*selected).sum()+1e-6)
            loss.backward()
            optimizer.step()
        model, optimizer = make()
        update(model, optimizer, torch.tensor([[[0, 9, 2]]]))
        state = deepcopy(optimizer.state_dict())
        first = model.weight.detach().clone()
        update(model, optimizer, torch.tensor([[[0, 1, 9]]]))
        uninterrupted = model.weight.detach().clone()
        restored, second_optimizer = make()
        restored.weight.data.copy_(first)
        second_optimizer.load_state_dict(state)
        update(restored, second_optimizer, torch.tensor([[[0, 1, 9]]]))
        self.assertTrue(torch.equal(restored.weight, uninterrupted))
        self.assertEqual(differences(optimizer.state_dict(), second_optimizer.state_dict()), [])
        self.assertEqual(next(iter(second_optimizer.optimizer.state.values()))['step'].item(), 2.)

    def test_zero_error_gradient_preserves_zero_vs_none_and_still_advances_adam(self):
        model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
        optimizer = DirectBF16AdamW(model)
        for _ in range(2):
            values = model(torch.eye(2, dtype=torch.bfloat16)).square()
            inactive = torch.zeros_like(values, dtype=torch.bool)
            loss = controlled_rows(values, ~inactive, inactive, 'E').sum()
            loss.backward()
            self.assertIsNotNone(model.weight.grad)
            self.assertEqual(int(torch.count_nonzero(model.weight.grad)), 0)
            optimizer.step()
        self.assertEqual(next(iter(optimizer.optimizer.state.values()))['step'].item(), 2.)
        self.assertIsNone(model.weight.grad)


class PersistentNativeTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get("PERSISTENT_TINY_CHILD") == "1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        import os
        if os.environ.get("PERSISTENT_TINY_CHILD") == "1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        import os
        if os.environ.get("PERSISTENT_TINY_CHILD") == "1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def test_native_adapter_event7_resume_event8_and_scalar_migration(self):
        import os
        import subprocess
        import sys
        if os.environ.get("PERSISTENT_TINY_CHILD") != "1":
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PERSISTENT_TINY_CHILD="1")
            env['PYTHONPATH'] = os.pathsep.join(str(root/p) for p in ('.', 'tests', 'training', 'inference'))
            result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                __name__ + '.PersistentNativeTinyTests.' + self._testMethodName], cwd=root,
                env=env, text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        from test_full_k_training import FullKTinyIntegrationTests
        from experiments.lost_repair.training_probe import sample_schedule
        from online_feedback.kplus import install_kplus
        from online_feedback.storage import save_torch
        from safetensors.torch import save_file
        from experiments.full_k_gsm8k import training as native
        base = FullKTinyIntegrationTests.build(self, 'base')
        install_kplus(base.bundle.model)
        state = deepcopy(native.live_state(base))
        history, _ = original()
        state.update(events=history['events'], global_step=768, presentations=1536)
        state['algorithm_contract'].pop(native.CONTRACT_KEY)
        state['algorithm_contract'].update({k:v for k,v in history['algorithm_contract'].items()
            if k in (S.historical.CONTRACT_KEY, S.geometry.CONTRACT_KEY)})
        optimizer = state['backend']['optimizer']['optimizer_state_dict']
        for index, parameter in zip(optimizer['param_groups'][0]['params'], base.backend.optimizer.model_params):
            optimizer['state'][index] = dict(step=torch.tensor(768.), exp_avg=torch.zeros_like(parameter),
                                            exp_avg_sq=torch.zeros_like(parameter))
        schedule = sample_schedule(base.bundle.model, self.features, state['rng'], max_length=base.max_length)
        schedule.pop('final_rng')
        origin = self.root/'origin.pt'
        save_torch(origin, state)
        export = self.root/'parent_export'
        export.mkdir()
        save_file(state['draft_state_dict'], export/'model.safetensors')
        base.close()
        kwargs = dict(binding=BINDING, original_config_text=self.original_config_text,
                      expected_cuda_rng_states=0)
        trainer = PersistentPrefixTrainer(self.config, self.root/'E', arm='E', origin_state=origin,
            origin_export=export, completed_event=6, schedule=schedule, **kwargs)
        self.addCleanup(trainer.close)
        result = trainer.train_event('event_007', self.features)
        trainer.verify_boundary(result['export_path'], self.root/'E_BOUNDARY.json')
        saved = self.root/'E7.pt'
        saved.write_bytes(trainer.latest.read_bytes())
        self.assertEqual(trainer.global_step, 896)
        self.assertEqual(len(result['records']), 256)
        self.assertTrue(all(row['prefix_roles']['role'] == 'E' for row in result['records']))
        self.assertTrue(all(row['prefix_roles']['responses'][0]['chunks'] for row in result['records']))
        trainer.close()
        resumed = PersistentPrefixTrainer(self.config, self.root/'E_resumed', arm='E', origin_state=saved,
            origin_export=result['export_path'], completed_event=7, **kwargs)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.global_step, 896)
        self.assertEqual(resumed.role_controller.role, 'E')
        new_features = []
        for index, path in enumerate(self.features):
            raw = torch.load(path, weights_only=True)
            raw['record_id'] = f'own_event8_{index}'
            raw['label_binding']['record_id'] = raw['record_id']
            target = self.root/f'event8_{index}.pt'
            torch.save(raw, target)
            new_features.append(target)
        final = resumed.train_event('event_008', new_features)
        resumed.verify_boundary(final['export_path'])
        self.assertEqual(resumed.global_step, 1024)
        self.assertEqual(resumed.presentations, 2048)
        with self.assertRaises(ValueError):
            resumed.train_event('event_009', new_features)
        resumed.close()
        scalar_config = self.root/'S.yaml'
        raw = yaml.safe_load(self.config.read_text())
        raw['training']['learning_rate'] = S.SCALAR_LR
        scalar_config.write_text(yaml.safe_dump(raw))
        scalar = PersistentPrefixTrainer(scalar_config, self.root/'S', arm='S', origin_state=origin,
            origin_export=export, completed_event=6, schedule=schedule, **kwargs)
        self.addCleanup(scalar.close)
        self.assertEqual(scalar.backend.optimizer.get_learning_rate(), S.SCALAR_LR)
        self.assertEqual(scalar.backend.optimizer.recipe['optimizer_options']['lr'], S.SCALAR_LR)
        self.assertIsNone(scalar.role_controller)
        self.assertEqual(scalar.global_step, 768)
        scalar.close()
        from experiments.persistent_prefix_withholding import qualification as Q
        from unittest.mock import patch
        receipts = []
        for slot in ('A', 'B'):
            disposable = PersistentPrefixTrainer(self.config, self.root/('qualification_' + slot),
                arm='N', origin_state=origin, origin_export=export, completed_event=6,
                schedule=schedule, qualification_only=True, **kwargs)
            self.addCleanup(disposable.close)
            # The tiny fixture is an integration/state oracle; native required
            # role support remains separately gated on the actual fixed pair.
            with patch.object(Q, 'gradient_role_check', return_value=dict(status='passed', fixture_only=True)):
                receipts.append(Q.qualify(disposable, self.features, self.root/('checks_' + slot), device_slot=slot))
            disposable.close()
        self.assertEqual(Q.compare_cards(*receipts)['attempted_optimizer_applications'], 8)
        self.assertEqual([receipt['optimizer_applications_completed'] for receipt in receipts], [5, 3])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
