"""CPU fixtures exercise exact warm import and the separate metadata contract.

Only the original model-sized parent validator is stubbed. The real native
loader, RNG restoration, live-state comparison and optimizer isolation run.
"""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from online_feedback.rng import capture_rng, restore_rng
from experiments.ok_continuation.resume import differences
from experiments.repair_conditional_kl_pilot import training as native
from . import fork


class TinyBackend:
    def __init__(self, model):
        self.model = model
        self.optimizer = {}

    def load_state_dict(self, state):
        self.optimizer = state['optimizer']

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer)


class OriginalContractTests(unittest.TestCase):
    """All original validators run on the established 58-tiny-tensor fixture."""

    def setUp(self):
        from experiments.repair_conditional_kl_pilot import test_training as fixtures
        self.rng = capture_rng()
        with patch.object(fixtures, 'BINDING', fork.PARENT_BINDING):
            self.state, old = fixtures.state_fixture('PAIR', event=1, maximum_events=1)
        self.arguments = dict(parent_identity=fork.PARENT_IDENTITY,
            parent_config_text=old['config_text'], original_config_text=old['original_config_text'],
            expected_contract=old['expected_contract'], expected_recipe=old['expected_recipe'],
            expected_cuda_rng_states=0)

    def tearDown(self):
        restore_rng(self.rng)
        self.assertFalse(torch.cuda.is_initialized())

    def test_complete_original_history_and_58_warm_states_pass_unchanged(self):
        before = deepcopy(self.state)
        receipt = fork.validate_parent_snapshot(self.state, **self.arguments)
        self.assertEqual(receipt['native_validation']['adam_state_entries'], 58)
        self.assertEqual(receipt['native_validation']['global_step'], 128)
        self.assertFalse(differences(before, self.state))

    def test_original_pair_history_and_optimizer_checks_remain_strict(self):
        for mutate in (
            lambda state: state['events'][0]['records'][0]['response_observation']['ratio_metrics'].clear(),
            lambda state: state['events'][0][fork.OLD_CONTRACT_KEY].update(maximum_events=2),
            lambda state: state['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=.001),
        ):
            changed = deepcopy(self.state)
            mutate(changed)
            with self.assertRaises(ValueError):
                fork.validate_parent_snapshot(changed, **self.arguments)


class ForkTests(unittest.TestCase):
    def setUp(self):
        self.rng = capture_rng()
        self.config = yaml.safe_dump(dict(
            run_id='original_pair', output_dir='/original/PAIR',
            training=dict(learning_rate=3e-4, gradient_accumulation_steps=2),
            online_feedback=dict(max_events=1),
        ))
        self.old_contract = {fork.OLD_CONTRACT_KEY:
            native.experiment_contract('PAIR', fork.PARENT_BINDING, maximum_events=1)}
        model = torch.nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
        self.state = dict(
            draft_state_dict=deepcopy(model.state_dict()),
            backend=dict(optimizer=dict(
                recipe={'fixture': True}, optimizer_state_dict=dict(
                    state={0: dict(step=torch.tensor(128.),
                                   exp_avg=torch.tensor([[.2, -.3]], dtype=torch.bfloat16),
                                   exp_avg_sq=torch.tensor([[.1, .2]], dtype=torch.bfloat16))},
                    param_groups=[dict(params=[0], lr=3e-4, betas=(.9, .999), eps=1e-8)]))),
            rng=capture_rng(), global_step=128, presentations=256,
            events=[dict(event_id='event_001', method='PAIR', features=[{'record_id': 'old'}],
                         **{fork.OLD_CONTRACT_KEY: deepcopy(self.old_contract[fork.OLD_CONTRACT_KEY])})],
            strategy='dflash', world_size=1, reduction='response_mean',
            accumulation_steps=2, horizon=7424, config_text=self.config,
            algorithm_contract=deepcopy(self.old_contract), run_id='original_pair',
        )
        model.weight.data.zero_()
        self.trainer = SimpleNamespace(
            arm='PAIR', closed=False, failed=False,
            bundle=SimpleNamespace(draft_model=model,
                                   model=SimpleNamespace(repair_conditional_kl=True)),
            backend=TinyBackend(model),
            strategy=SimpleNamespace(checkpoint_state_filter=lambda value: value),
            core=SimpleNamespace(accumulation_remainder=0, _micro=0,
                                 _ratio_totals={}, _sum_totals={}),
            global_step=0, presentations=0, events=[],
            cfg=SimpleNamespace(run_id='new'), config_text='new metadata',
            contract={'new': 'metadata'}, reduction='response_mean', accumulation=2, horizon=7424,
        )
        self.arguments = dict(parent_identity=deepcopy(fork.PARENT_IDENTITY),
            parent_config_text=self.config, original_config_text='original source text',
            expected_contract=deepcopy(self.old_contract), expected_recipe={'fixture': True},
            expected_cuda_rng_states=0)
        self.validator = patch.object(native, 'validate_snapshot', return_value={'status': 'passed'})
        self.validate = self.validator.start()
        self.addCleanup(self.validator.stop)

    def tearDown(self):
        restore_rng(self.rng)
        self.assertFalse(torch.cuda.is_initialized())

    def contract(self, arm='CONTROL'):
        return fork.fork_contract(arm, source_binding='a' * 64)

    def test_validates_parent_using_original_contract_before_any_import(self):
        parent = deepcopy(self.state)
        receipt = fork.validate_parent_snapshot(self.state, **self.arguments)
        args = self.validate.call_args.kwargs
        self.assertEqual((args['arm'], args['binding'], args['maximum_events'], args['completed_event']),
                         ('PAIR', fork.PARENT_BINDING, 1, 1))
        self.assertEqual(args['config_text'], self.config)
        self.assertEqual(args['expected_contract'], self.old_contract)
        self.assertEqual(receipt['global_step'], 128)
        self.assertFalse(differences(parent, self.state))

    def test_wrong_parent_boundaries_and_rebound_history_fail_before_native_validation(self):
        for change in (dict(global_step=0), dict(presentations=0), dict(events=[]),
                       dict(events=[{'event_id': 'event_002'}]), dict(config_text='new')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                fork.validate_parent_snapshot(dict(self.state, **change), **self.arguments)
        foreign = deepcopy(self.state)
        foreign['algorithm_contract'][fork.OLD_CONTRACT_KEY]['arm'] = 'BASE'
        with self.assertRaises(ValueError):
            fork.validate_parent_snapshot(foreign, **self.arguments)
        self.validate.assert_not_called()

    def test_identity_and_original_native_failure_are_not_bypassed(self):
        bad = dict(self.arguments, parent_identity=dict(bytes=1, sha256='b' * 64))
        with self.assertRaises(ValueError):
            fork.validate_parent_snapshot(self.state, **bad)
        self.validate.return_value = {'status': 'failed'}
        with self.assertRaises(ValueError):
            fork.import_parent_state(self.trainer, self.state, contract=self.contract(), **self.arguments)
        self.assertEqual(self.trainer.global_step, 0)

    def test_exact_import_both_arms_preserves_parent_and_separates_metadata(self):
        parent = deepcopy(self.state)
        for arm in fork.ARMS:
            torch.rand(3)
            receipt = fork.import_parent_state(self.trainer, self.state,
                contract=self.contract(arm), **self.arguments)
            self.assertTrue(receipt['exact_weights_optimizer_rng_counters_history'])
            self.assertFalse(receipt['allocator_runtime_verified'])
            self.assertFalse(differences(fork.scientific_state(native.native.live_state(self.trainer)),
                                         fork.scientific_state(self.state)))
            self.assertEqual(self.trainer.contract, {'new': 'metadata'})
            self.assertEqual(self.trainer.config_text, 'new metadata')
            self.assertEqual(self.trainer.core._micro, 256)
            self.assertFalse(differences(parent, self.state))
        self.trainer.events[0]['features'].append({'record_id': 'new'})
        self.trainer.backend.optimizer['optimizer_state_dict']['state'][0]['step'].add_(1)
        self.trainer.backend.optimizer['optimizer_state_dict']['state'][0]['exp_avg'].zero_()
        self.trainer.bundle.draft_model.weight.data.add_(1)
        self.assertFalse(differences(parent, self.state))

    def test_runtime_requires_pair_and_clean_boundary(self):
        for attribute, value in (('arm', 'BASE'), ('closed', True), ('failed', True)):
            original = getattr(self.trainer, attribute)
            setattr(self.trainer, attribute, value)
            with self.assertRaises(ValueError):
                fork.import_parent_state(self.trainer, self.state, contract=self.contract(), **self.arguments)
            setattr(self.trainer, attribute, original)
        self.trainer.bundle.model.repair_conditional_kl = False
        with self.assertRaises(ValueError):
            fork.import_parent_state(self.trainer, self.state, contract=self.contract(), **self.arguments)
        self.trainer.bundle.model.repair_conditional_kl = True
        self.trainer.bundle.draft_model.weight.grad = torch.ones_like(self.trainer.bundle.draft_model.weight)
        with self.assertRaises(ValueError):
            fork.import_parent_state(self.trainer, self.state, contract=self.contract(), **self.arguments)
        self.validate.assert_not_called()

    def test_new_contract_exact_scope_and_pair_enabled(self):
        for arm in fork.ARMS:
            contract = self.contract(arm)
            self.assertEqual(fork.validate_fork_contract(contract), contract)
            self.assertTrue(contract['pair_enabled'])
            self.assertEqual(contract['native_runtime_arm'], 'PAIR')
            self.assertEqual(contract['final_step'], 256)
            for key, value in (('pair_enabled', False), ('reset_optimizer', True),
                               ('new_events', ['event_001']), ('prefix_relative_cap', .5)):
                with self.assertRaises(ValueError):
                    fork.validate_fork_contract(dict(contract, **{key: value}))
        for arm in ('BASE', 'PAIR', None):
            with self.assertRaises(ValueError):
                self.contract(arm)

    def test_metadata_binding_allows_only_run_and_output_identity(self):
        parent = fork.validate_parent_snapshot(self.state, **self.arguments)
        changed = yaml.safe_load(self.config)
        changed.update(run_id='new_frontier', output_dir='/new/frontier')
        receipt = fork.bind_fork_metadata(parent, contract=self.contract('FRONTIER_P'),
            parent_config_text=self.config, new_config_text=yaml.safe_dump(changed))
        self.assertEqual(receipt['status'], 'bound_not_runtime_qualified')
        self.assertFalse(receipt['allocator_runtime_verified'])
        changed['training']['learning_rate'] = 1e-4
        with self.assertRaises(ValueError):
            fork.bind_fork_metadata(parent, contract=self.contract(),
                parent_config_text=self.config, new_config_text=yaml.safe_dump(changed))
        with self.assertRaises(ValueError):
            fork.bind_fork_metadata(dict(parent, status='failed'), contract=self.contract(),
                parent_config_text=self.config, new_config_text=self.config)


if __name__ == '__main__':
    unittest.main()
