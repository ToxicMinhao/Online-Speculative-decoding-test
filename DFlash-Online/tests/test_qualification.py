"""Exact qualification checks remain separate from the normal training loop."""

import unittest
from types import SimpleNamespace
import numpy as np
import torch

from online_feedback.qualify import assert_state_equal, check_saved_training_state
from online_feedback.rng import capture_rng


class StateComparisonTests(unittest.TestCase):
    def test_nested_optimizer_and_rng_state(self):
        state = {'optimizer': [torch.tensor([1., 2.])], 'rng': (np.array([3, 4]), 5)}
        other = {'optimizer': [torch.tensor([1., 2.])], 'rng': (np.array([3, 4]), 5)}
        assert_state_equal(state, other)
        other['optimizer'][0][1] = 4
        with self.assertRaisesRegex(AssertionError, 'optimizer'):
            assert_state_equal(state, other)

    def test_dtype_changes_are_not_exact_restoration(self):
        with self.assertRaises(AssertionError):
            assert_state_equal(torch.ones(2, dtype=torch.bfloat16), torch.ones(2))

    def test_changed_counters_are_detected(self):
        with self.assertRaisesRegex(AssertionError, 'global_step'):
            assert_state_equal({'global_step': 1}, {'global_step': 2})

    def test_live_weights_must_match_saved_state_before_restart(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = SimpleNamespace(
            bundle=SimpleNamespace(draft_model=model),
            backend=SimpleNamespace(optimizer=optimizer),
            global_step=0, presentations=0,
        )
        saved = {
            'draft_state_dict': {name: value.clone() for name, value in model.state_dict().items()},
            'backend': {'optimizer': optimizer.state_dict()},
            'rng': capture_rng(), 'global_step': 0, 'presentations': 0,
        }
        check_saved_training_state(trainer, saved)
        with torch.no_grad():
            model.weight.add_(1)
        with self.assertRaisesRegex(AssertionError, 'draft'):
            check_saved_training_state(trainer, saved)

    def test_rng_drift_since_checkpoint_is_visible(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = SimpleNamespace(
            bundle=SimpleNamespace(draft_model=model),
            backend=SimpleNamespace(optimizer=optimizer),
            global_step=0, presentations=0,
        )
        saved = {
            'draft_state_dict': model.state_dict(),
            'backend': {'optimizer': optimizer.state_dict()},
            'rng': capture_rng(), 'global_step': 0, 'presentations': 0,
        }
        torch.rand(1)
        with self.assertRaisesRegex(AssertionError, 'rng'):
            check_saved_training_state(trainer, saved)


if __name__ == '__main__':
    unittest.main()
