"""Tiny real DFlash CPU checks; native A800 qualification remains separate."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from online_feedback.rng import capture_rng, restore_rng
from online_feedback.survival_head_runtime import install_survival_head_core
from test_adversarial_block_runtime import build


def assert_nested_equal(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_nested_equal(case, first, second)
    else:
        case.assertEqual(left, right)


class SurvivalHeadRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_disabled_is_original_native_endpoint_optimizer_and_rng(self):
        endpoints = []
        for install in (False, True):
            model, batch, _, backend, core = build()
            if install:
                native = core
                before = torch.get_rng_state().clone()
                core = install_survival_head_core(core, model, arm="KPLUS")
                self.assertIs(core, native)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))
            self.assertFalse(hasattr(model, "survival_head_extension"))
            first, second = core.train_step(batch), core.train_step(batch)
            self.assertFalse(first.optimizer_stepped)
            self.assertTrue(second.optimizer_stepped)
            endpoints.append((deepcopy(model.state_dict()),
                              deepcopy(backend.optimizer.state_dict()),
                              torch.get_rng_state().clone(), second.loss))
        assert_nested_equal(self, endpoints[0], endpoints[1])

    def test_head_initialization_export_and_optimizer_parameter_isolation(self):
        model, _, _, backend, native = build()
        model_before = deepcopy(model.state_dict())
        native_before = deepcopy(backend.optimizer.state_dict())
        native_parameter_ids = {id(p) for p in backend.optimizer.model_params}
        before = torch.get_rng_state().clone()
        core = install_survival_head_core(native, model, arm="LOCAL")
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertIs(core.backend.optimizer, backend.optimizer)
        assert_nested_equal(self, model.state_dict(), model_before)
        assert_nested_equal(self, backend.optimizer.state_dict(), native_before)
        head_ids = {id(p) for p in core.extension.head.parameters()}
        self.assertFalse(head_ids & native_parameter_ids)
        self.assertFalse(head_ids & {id(p) for p in model.parameters()})
        self.assertEqual(set(core.extension.head.state_dict()), {"weight", "bias"})
        self.assertEqual(core.extension.optimizer.get_learning_rate(),
                         backend.optimizer.get_learning_rate())

    def test_one_clean_forward_per_response_and_two_separate_clips_per_window(self):
        for arm in ("LOCAL", "SURVIVAL"):
            model, batch, _, backend, native = build()
            core = install_survival_head_core(native, model, arm=arm)
            initial_head = deepcopy(core.extension.head.state_dict())
            with patch.object(model.draft_model, "forward", wraps=model.draft_model.forward) as draft:
                with patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                    first = core.train_step(batch)
                    self.assertFalse(first.optimizer_stepped)
                    self.assertEqual(core.extension.steps, 0)
                    self.assertFalse(core.extension.optimizer.optimizer.state)
                    assert_nested_equal(self, core.extension.head.state_dict(), initial_head)
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in core.extension.head.parameters()))
                    second = core.train_step(batch)
                    self.assertTrue(second.optimizer_stepped)
                    self.assertEqual(clip.call_count, 2)
                self.assertEqual(draft.call_count, 2)
            self.assertEqual(backend.steps, 1)
            self.assertEqual(backend.backward_boundaries, [False, True])
            self.assertEqual(backend.scalings, [1.0])
            self.assertEqual(core.extension.steps, 1)
            self.assertGreater(second.metrics["head_grad_norm"], 0)
            self.assertEqual(second.metrics["head_optimizer_steps"], 1)
            self.assertTrue(any(not torch.equal(value, initial_head[name])
                                for name, value in core.extension.head.state_dict().items()))
            for optimizer in (backend.optimizer, core.extension.optimizer):
                for parameter in optimizer.model_params:
                    state = optimizer.optimizer.state[parameter]
                    self.assertEqual(state["step"].item(), 1)
                    self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
                    self.assertIsNone(parameter.grad)

    def test_auxiliary_alone_reaches_real_backbone_with_finite_nonzero_gradient(self):
        for arm in ("LOCAL", "SURVIVAL"):
            model, batch, author, backend, native = build()
            core = install_survival_head_core(native, model, arm=arm)
            augment = core.extension.augment_response

            def auxiliary_only(native_loss, terms):
                # Keep the native chunk computation, but isolate the auxiliary
                # gradient at response reduction for this disposable CPU check.
                return augment(native_loss.detach() * 0, terms)

            core.extension.augment_response = auxiliary_only
            output = author.forward_loss(batch)
            output.loss.backward()
            gradients = [p.grad.float() for p in model.draft_model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.square().sum().item() for gradient in gradients), 0)
            self.assertTrue(all(p.grad is None for p in model.lm_head.parameters()))
            self.assertTrue(all(p.grad is None for p in model.embed_tokens.parameters()))

    def test_all_arms_match_anchor_draws_and_rng(self):
        snapshots, heads = [], []
        for arm in ("KPLUS", "LOCAL", "SURVIVAL"):
            model, batch, _, _, native = build()
            draws = []
            original_sampler = model._sample_anchor_positions

            def sampler(*args, **kwargs):
                sampled = original_sampler(*args, **kwargs)
                draws.append(tuple(value.clone() for value in sampled))
                return sampled

            model._sample_anchor_positions = sampler
            core = install_survival_head_core(native, model, arm=arm)
            if arm != "KPLUS":
                heads.append(deepcopy(core.extension.head.state_dict()))
            core.train_step(batch)
            core.train_step(batch)
            snapshots.append((draws, torch.get_rng_state().clone()))
        assert_nested_equal(self, snapshots[0], snapshots[1])
        assert_nested_equal(self, snapshots[0], snapshots[2])
        assert_nested_equal(self, heads[0], heads[1])

    def test_head_average_matches_two_response_manual_backward_and_separate_adam(self):
        model, batch, _, _, native = build()
        core = install_survival_head_core(native, model, arm="SURVIVAL")
        initial_rng = capture_rng()
        initial_model = deepcopy(model.state_dict())
        initial_head = deepcopy(core.extension.state_dict())
        core.train_step(batch)
        core.train_step(batch)
        expected = deepcopy(core.extension.state_dict())

        model2, batch2, _, _, native2 = build()
        model2.load_state_dict(initial_model)
        manual = install_survival_head_core(native2, model2, arm="SURVIVAL")
        manual.extension.load_state_dict(initial_head)
        restore_rng(initial_rng)
        for _ in range(2):
            output = manual.strategy.forward_loss(batch2)
            (output.loss / 2).backward()
        manual.extension.step()
        assert_nested_equal(self, manual.extension.state_dict(), expected)

    def test_head_state_roundtrip_is_separate_and_rejects_other_label_contract(self):
        model, batch, _, _, native = build()
        core = install_survival_head_core(native, model, arm="LOCAL")
        core.train_step(batch)
        core.train_step(batch)
        state = deepcopy(core.extension.state_dict())
        model2, _, _, _, native2 = build()
        restored = install_survival_head_core(native2, model2, arm="LOCAL")
        restored.extension.load_state_dict(state)
        assert_nested_equal(self, restored.extension.state_dict(), state)
        self.assertEqual(restored.extension.steps, 1)
        model3, _, _, _, native3 = build()
        other = install_survival_head_core(native3, model3, arm="SURVIVAL")
        with self.assertRaisesRegex(ValueError, "contract differs"):
            other.extension.load_state_dict(state)

    def test_boundary_and_optimizer_mismatch_are_rejected(self):
        model, batch, _, _, native = build()
        with self.assertRaisesRegex(ValueError, "must match"):
            install_survival_head_core(native, model, arm="LOCAL", learning_rate=9)
        native.train_step(batch)
        with self.assertRaisesRegex(ValueError, "complete two-response"):
            install_survival_head_core(native, model, arm="LOCAL")


if __name__ == "__main__":
    unittest.main()
