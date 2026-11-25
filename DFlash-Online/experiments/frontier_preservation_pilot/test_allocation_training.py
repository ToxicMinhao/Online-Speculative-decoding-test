"""New fork validators preserve the old parent history and native state gates."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.repair_conditional_kl_pilot import test_training as fixtures
from .allocation_training import (ForkTrainer, history_digest, production_contract,
                                  validate_boundary, validate_event)
from .fork import FORK_CONTRACT_KEY, OLD_CONTRACT_KEY, PARENT_BINDING


def fixture(arm='FRONTIER_P'):
    with patch.object(fixtures, 'BINDING', PARENT_BINDING):
        state, arguments = fixtures.state_fixture('PAIR', event=2, maximum_events=1)
    binding = 'b' * 64
    event = state['events'][1]
    event.pop(OLD_CONTRACT_KEY)
    event.update(method=arm, **{FORK_CONTRACT_KEY: production_contract(arm, binding)})
    state['algorithm_contract'][FORK_CONTRACT_KEY] = production_contract(arm, binding)
    trainer = SimpleNamespace(config_text=state['config_text'],
        contract=deepcopy(state['algorithm_contract']), fork_arm=arm, fork_binding=binding,
        parent_validation=dict(parent_history_digest=history_digest(state['events'][:1])),
        original_config_text=arguments['original_config_text'], expected_cuda_rng_states=0,
        backend=SimpleNamespace(optimizer=SimpleNamespace(recipe=arguments['expected_recipe'])))
    return state, trainer


class AllocationTrainingTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_new_boundary_keeps_original_pair_history_and_all58_adam_entries(self):
        for arm in ('CONTROL', 'FRONTIER_P'):
            state, trainer = fixture(arm)
            original = deepcopy(state)
            result = validate_boundary(state, trainer=trainer)
            self.assertEqual(result['global_step'], 256)
            self.assertEqual(result['adam_state_entries'], 58)
            self.assertEqual(history_digest(state['events']), history_digest(original['events']))

    def test_rewritten_history_or_optimizer_reset_rejected(self):
        for mutation in (
            lambda state: state['events'][0].update(method='BASE'),
            lambda state: state.update(global_step=128),
            lambda state: state['backend']['optimizer']['optimizer_state_dict']['state'].clear(),
            lambda state: state['events'][1]['records'][0]['response_observation']['ratio_metrics'].clear(),
        ):
            state, trainer = fixture()
            mutation(state)
            with self.assertRaises(ValueError):
                validate_boundary(state, trainer=trainer)

    def test_candidate_and_control_contracts_cannot_be_exchanged(self):
        state, trainer = fixture()
        with self.assertRaisesRegex(ValueError, 'fork identity'):
            validate_event(state['events'][1], arm='CONTROL', binding=trainer.fork_binding,
                           require_completed=False)

    def test_new_runner_stops_before_third_event(self):
        trainer = object.__new__(ForkTrainer)
        trainer.qualification_only = False
        trainer.global_step, trainer.presentations, trainer.events = 256, 512, [{}, {}]
        with self.assertRaisesRegex(ValueError, 'one warm continuation'):
            trainer._prepare_event('event_003', ['fixture'] * 128, 2)


if __name__ == '__main__':
    unittest.main()
