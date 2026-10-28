"""Direct BF16 AdamW arithmetic and restart checks, without models or GPUs."""

import copy
import unittest

import torch

from online_feedback.optimizer import DirectBF16AdamW


def model():
    torch.manual_seed(17)
    return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)).bfloat16()


def assign_gradients(module, step):
    for index, param in enumerate(module.parameters()):
        if param.requires_grad:
            param.grad = torch.linspace(-3, 4, param.numel()).reshape(param.shape).to(param.dtype)
            param.grad.mul_(step + index + 1)


class DirectBF16AdamWTests(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_updates_and_moments_match_direct_torch_adamw_after_clipping(self):
        actual_model = model()
        reference_model = copy.deepcopy(actual_model)
        initial = copy.deepcopy(actual_model.state_dict())
        adapter = DirectBF16AdamW(actual_model)
        reference = torch.optim.AdamW(reference_model.parameters(), lr=3e-4, weight_decay=0.01)
        self.assertFalse(adapter.checks_loss_denominator)

        for step in range(3):
            assign_gradients(actual_model, step)
            assign_gradients(reference_model, step)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
            reference.step()
            reference.zero_grad(set_to_none=True)
            norm = adapter.step()
            self.assertGreater(float(norm), 1.0)
            self.assert_tree_equal(norm, expected_norm)
            self.assert_tree_equal(actual_model.state_dict(), reference_model.state_dict())
            self.assert_tree_equal(adapter.optimizer.state_dict(), reference.state_dict())
            self.assertTrue(all(param.grad is None for param in actual_model.parameters()))
            self.assertEqual(adapter.get_learning_rate(), 3e-4)

        self.assertTrue(any(not torch.equal(initial[name], value)
                            for name, value in actual_model.state_dict().items()))
        self.assertFalse(hasattr(adapter, "fp32_params"))
        self.assertFalse(hasattr(adapter, "scheduler"))
        for actual_param, optimizer_param in zip(actual_model.parameters(), adapter.optimizer.param_groups[0]["params"]):
            self.assertIs(actual_param, optimizer_param)
            moments = adapter.optimizer.state[actual_param]
            self.assertEqual(moments["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(moments["exp_avg_sq"].dtype, torch.bfloat16)
            self.assertEqual(moments["step"].item(), 3)

    def test_exact_restart_preserves_next_update_and_moments(self):
        original_model = model()
        original = DirectBF16AdamW(original_model)
        for step in range(2):
            assign_gradients(original_model, step)
            original.step()
        saved_model = copy.deepcopy(original_model.state_dict())
        saved_optimizer = copy.deepcopy(original.state_dict())

        resumed_model = model()
        resumed_model.load_state_dict(saved_model)
        resumed = DirectBF16AdamW(resumed_model)
        resumed.load_state_dict(saved_optimizer)
        for step in (2, 3):
            assign_gradients(original_model, step)
            assign_gradients(resumed_model, step)
            self.assert_tree_equal(original.step(), resumed.step())
            self.assert_tree_equal(original_model.state_dict(), resumed_model.state_dict())
            self.assert_tree_equal(original.state_dict(), resumed.state_dict())

    def test_frozen_parameters_are_excluded_and_zero_grad_clears_trainable_gradients(self):
        draft = model()
        draft.register_parameter("frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False))
        adapter = DirectBF16AdamW(draft)
        self.assertNotIn(id(draft.frozen), {id(p) for p in adapter.model_params})
        assign_gradients(draft, 0)
        adapter.zero_grad()
        self.assertTrue(all(p.grad is None for p in draft.parameters()))

    def test_empty_or_non_bf16_trainable_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "trainable"):
            DirectBF16AdamW(torch.nn.Identity())
        with self.assertRaisesRegex(ValueError, "BF16 parameters"):
            DirectBF16AdamW(torch.nn.Linear(2, 2))
        frozen = model().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "trainable"):
            DirectBF16AdamW(frozen)

    def test_incompatible_resume_is_rejected_without_mutation(self):
        draft = model()
        adapter = DirectBF16AdamW(draft)
        assign_gradients(draft, 0)
        adapter.step()
        before = copy.deepcopy(adapter.state_dict())
        initial_weights = copy.deepcopy(draft.state_dict())
        invalid_states = []
        missing_recipe = copy.deepcopy(before)
        del missing_recipe["recipe"]
        invalid_states.append(missing_recipe)
        for key, value in (("type", "fp32_master_adamw"), ("max_grad_norm", 2.0), ("parameters", ())):
            state = copy.deepcopy(before)
            state["recipe"][key] = value
            invalid_states.append(state)
        for key, value in (("lr", 1e-4), ("weight_decay", 0.0), ("betas", (0.8, 0.9))):
            state = copy.deepcopy(before)
            state["optimizer_state_dict"]["param_groups"][0][key] = value
            invalid_states.append(state)
        for moment in ("exp_avg", "exp_avg_sq"):
            state = copy.deepcopy(before)
            first = next(iter(state["optimizer_state_dict"]["state"].values()))
            first[moment] = first[moment].float()
            invalid_states.append(state)

        for state in invalid_states:
            with self.subTest(recipe=state.get("recipe")):
                with self.assertRaisesRegex(ValueError, "Checkpoint"):
                    adapter.load_state_dict(state)
                self.assert_tree_equal(adapter.state_dict(), before)
                self.assert_tree_equal(draft.state_dict(), initial_weights)

    def test_new_recipe_settings_cannot_adopt_old_checkpoint(self):
        saved = DirectBF16AdamW(model()).state_dict()
        for settings in ({"learning_rate": 1e-4}, {"weight_decay": 0.0}, {"max_grad_norm": 0.5}):
            with self.subTest(settings=settings):
                destination = DirectBF16AdamW(model(), **settings)
                before = copy.deepcopy(destination.state_dict())
                with self.assertRaisesRegex(ValueError, "recipe differs"):
                    destination.load_state_dict(saved)
                self.assert_tree_equal(destination.state_dict(), before)

    def test_nonfinite_gradients_fail_before_weights_or_adam_state_change(self):
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(gradient=nonfinite):
                draft = model()
                adapter = DirectBF16AdamW(draft)
                assign_gradients(draft, 0)
                adapter.step()
                weights = copy.deepcopy(draft.state_dict())
                state = copy.deepcopy(adapter.state_dict())
                assign_gradients(draft, 1)
                next(draft.parameters()).grad.flatten()[0] = nonfinite
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    adapter.step()
                self.assert_tree_equal(draft.state_dict(), weights)
                self.assert_tree_equal(adapter.state_dict(), state)


if __name__ == "__main__":
    unittest.main()
