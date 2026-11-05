"""CPU integration checks for reference-relative pressure on real DFlash blocks.

Small explicit position oracles keep KL, boundary masks, auxiliary normalization
and diagnostic counts independently reviewable. These are not GPU qualification.
"""
import copy
import math
import random
import unittest
import weakref
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from online_feedback.objective import TargetObjectiveModel
from online_feedback.rng import capture_rng, restore_rng
from test_objective_integration import parameter_gradients, tiny_author
from test_osd_objective import osd_batch
from test_training_contract import trees_equal


ARM = "osd_reference_margin"


def independent_softplus(logits, labels):
    """Select non-label rivals by explicit rows, independent of production code."""
    values = []
    for row, label in zip(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)):
        token = int(label)
        competitors = [index for index in range(row.numel()) if index != token]
        rival = competitors[int(row.detach()[competitors].argmax())]
        values.append(F.softplus(row[rival].float() - row[token].float()))
    return torch.stack(values).reshape_as(labels)


def independent_prefix(proposals, labels, weights, eos_token_ids=()):
    prefix = torch.zeros_like(labels, dtype=torch.bool)
    for batch in range(labels.shape[0]):
        for block in range(labels.shape[1]):
            alive = int(labels[batch, block, 0]) not in eos_token_ids
            for offset in range(1, labels.shape[2]):
                if not alive or weights[batch, block, offset] <= 0:
                    alive = False
                    continue
                if proposals[batch, block, offset] != labels[batch, block, offset]:
                    alive = False
                    continue
                prefix[batch, block, offset] = True
                if int(labels[batch, block, offset]) in eos_token_ids:
                    alive = False
    return prefix


class ReferenceMarginObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, dtype=torch.float32, **kwargs):
        model = TargetObjectiveModel(tiny_author(dtype=dtype), ARM, **kwargs)
        self.addCleanup(model.end_event)
        return model

    def test_weighted_loss_gradients_and_diagnostics_match_independent_position_oracle(self):
        model = self.model(ranking_lambda=1.7)
        model.begin_event()
        batch = osd_batch(model)
        batch["loss_mask"][0, 6] = 0
        batch["target_last_hidden_states"].requires_grad_(True)
        # Include prompt-only, interrupted-response, clipped-tail and disabled blocks.
        anchors = torch.tensor([[0, 4, 8, 10, 6]])
        keep = torch.tensor([[True, True, True, True, False]])
        prefix = torch.tensor([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                                [0, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool)
        reference_softplus = torch.zeros_like(prefix, dtype=torch.float32)
        reference_softplus[0, 2, 2] = 1000.0  # A protected but inactive position.
        reference_softplus.requires_grad_(True)
        # Later matches after the interrupted prefix must enter correctness metrics.
        # Deliberate true values at zero-weight positions also test reduction masks.
        reference_correct = torch.ones_like(prefix)
        feedback = (prefix, reference_softplus, reference_correct)
        draft_inputs = {key: batch[key] for key in ("input_ids", "hidden_states", "loss_mask")}
        with mock.patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)), \
                mock.patch.object(model, "_reference_feedback", return_value=feedback):
            actual, accuracy, metrics = model(**batch)
            actual.backward()
            actual_gradients = parameter_gradients(model)
            model.zero_grad(set_to_none=True)
            _, _, hidden = model._forward_draft_blocks(**draft_inputs)
            hidden = hidden.reshape(1, anchors.shape[1], model.block_size, -1)
            kl_sum = hidden.new_zeros(())
            auxiliary_sum = hidden.new_zeros(())
            weight_sum = hidden.new_zeros(())
            protected_weight = hidden.new_zeros(())
            positions = active = regressed = correct = 0
            for block, anchor in enumerate(anchors[0].tolist()):
                if not keep[0, block]:
                    continue
                for offset in range(1, model.block_size):
                    position = anchor + offset
                    if position >= batch["input_ids"].shape[1] or not batch["loss_mask"][0, position]:
                        continue
                    label = batch["input_ids"][0, position]
                    logits = model.lm_head(hidden[0, block, offset])
                    with torch.no_grad():
                        teacher = model.lm_head(batch["target_last_hidden_states"][0, position - 1]).softmax(-1)
                    weight = torch.exp(torch.tensor(-(offset - 1) / model.loss_decay_gamma))
                    kl_sum = kl_sum + weight * F.kl_div(logits.log_softmax(-1), teacher, reduction="sum")
                    competing = [token for token in range(logits.numel()) if token != int(label)]
                    rival = competing[int(logits.detach()[competing].argmax())]
                    delta = F.softplus(logits[rival] - logits[label]) - reference_softplus[0, block, offset].detach()
                    if prefix[0, block, offset]:
                        auxiliary_sum = auxiliary_sum + weight * delta.relu()
                        protected_weight = protected_weight + weight
                        active += int(delta.detach() > 0)
                    weight_sum = weight_sum + weight
                    positions += 1
                    is_correct = int(logits.detach().argmax()) == int(label)
                    correct += int(is_correct)
                    regressed += int(not is_correct)
            denominator = weight_sum + 1e-6
            expected = (kl_sum + model.ranking_lambda * auxiliary_sum) / denominator
            expected.backward()

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(accuracy, torch.tensor(correct / positions))
        self.assertEqual(positions, 6)
        self.assertEqual(active, 4)
        self.assertGreater(regressed, 0)
        self.assertIsNone(reference_softplus.grad)
        self.assertIsNone(batch["target_last_hidden_states"].grad)
        for name, gradient in parameter_gradients(model).items():
            torch.testing.assert_close(actual_gradients[name], gradient, atol=2e-6, rtol=2e-5)
        sums = metrics["sum_metrics"]
        expected_sums = {
            "kl_loss_numerator": kl_sum.detach(),
            "auxiliary_loss_numerator": model.ranking_lambda * auxiliary_sum.detach(),
            "loss_denominator": denominator,
            "supervised_position_exposure": positions,
            "protected_active_positions": active,
            "protected_positions": int(prefix.sum()),
            "reference_correct_positions": positions,
            "reference_correct_current_incorrect_positions": regressed,
            "reference_prefix_weight": protected_weight,
            "ranking_coefficient_weight": protected_weight,
            "ranking_rho": protected_weight / weight_sum,
        }
        for name, value in expected_sums.items():
            torch.testing.assert_close(sums[name], torch.as_tensor(value).float(), atol=2e-6, rtol=2e-6)
            self.assertFalse(sums[name].requires_grad)
        ratios = metrics["ratio_metrics"]
        expected_ratios = {
            "forward_kl": (kl_sum.detach(), denominator),
            "ranking_loss": (auxiliary_sum.detach(), denominator),
            "auxiliary_loss": (model.ranking_lambda * auxiliary_sum.detach(), denominator),
            "protected_activation_rate": (active, int(prefix.sum())),
            "reference_correct_to_incorrect_rate": (regressed, positions),
        }
        for name, pair in expected_ratios.items():
            for actual_term, expected_term in zip(ratios[name], pair):
                torch.testing.assert_close(actual_term, torch.as_tensor(expected_term).float(), atol=2e-6, rtol=2e-6)
        # The correctness denominator includes the match beyond the prefix interruption.
        self.assertGreater(positions, int(prefix.sum()))

    def test_equal_current_and_reference_margins_leave_positive_kl_and_its_gradients(self):
        model = self.model()
        model.lm_head = torch.nn.Identity()
        hidden = torch.tensor([[[[2., 0., -1.], [1.5, 0.1, -0.3],
                                 [1.2, 0.8, -0.2], [2., 0.5, -0.1]]]], requires_grad=True)
        labels = torch.zeros(1, 1, 4, dtype=torch.long)
        weights = torch.tensor([[[0., 1., 0.6, 0.3]]])
        coefficients = (weights > 0).float()
        reference = independent_softplus(hidden.detach(), labels).requires_grad_(True)
        teacher = torch.tensor([[[[0., 1., 2.]]]]).expand_as(hidden).clone().requires_grad_(True)
        terms = model._chunk_loss(hidden, None, labels, weights, teacher, coefficients,
                                  reference, weights > 0)
        denominator = weights.sum() + 1e-6
        actual = terms[0] / denominator
        actual.backward()
        actual_gradient = hidden.grad.detach().clone()
        hidden.grad = None
        expected = (F.kl_div(hidden.log_softmax(-1), teacher.detach().softmax(-1),
                             reduction="none").sum(-1) * weights).sum() / denominator
        expected.backward()
        self.assertGreater(float(actual.detach()), 0)
        self.assertEqual(float(terms[5]), 0.0)
        self.assertEqual(float(terms[6]), 0.0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_gradient, hidden.grad, atol=0, rtol=0)
        self.assertGreater(float(actual_gradient.abs().max()), 0)
        self.assertTrue(torch.equal(actual_gradient[0, 0, 0], torch.zeros(3)))
        self.assertIsNone(reference.grad)
        self.assertIsNone(teacher.grad)

    def test_feedback_uses_one_full_reference_forward_and_one_projection_per_chunk(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.model(dtype=dtype, eos_token_ids=(10,))
                model.objective_chunk_blocks = 2
                model.train()
                model.draft_model.layers[0].eval()
                modes = {name: module.training for name, module in model.named_modules()}
                rng_before_setup = capture_rng()
                model.begin_event()
                self.assertTrue(trees_equal(rng_before_setup, capture_rng()))
                batch = osd_batch(model)
                batch["hidden_states"].requires_grad_(True)
                batch["loss_mask"][0, 6] = 0
                anchors = torch.tensor([[0, 4, 8, 10, 6]])
                keep = torch.tensor([[True, True, True, True, False]])
                labels, weights, _ = model._block_labels_and_weights(
                    batch["input_ids"], batch["loss_mask"], anchors, keep)
                logits = torch.full((*labels.shape, 29), -2.0, dtype=dtype)
                logits.scatter_(-1, labels.unsqueeze(-1), 1.0)
                # Mismatch closes this prefix; the later match remains diagnostic evidence.
                logits[0, 1, 1, (int(labels[0, 1, 1]) + 1) % 29] = 2.0
                expected_prefix = independent_prefix(logits.argmax(-1), labels, weights, (10,))
                expected_softplus = independent_softplus(logits, labels)
                expected_correct = (logits.argmax(-1) == labels) & (weights > 0)
                projection_shapes = []
                hidden_refs = []
                full_shapes = []
                original_forward = model.reference.model.draft_model.forward

                def reference_call(*args, **kwargs):
                    self.assertFalse(torch.is_grad_enabled())
                    full_shapes.append((kwargs["position_ids"].shape, kwargs["noise_embedding"].shape))
                    random.random()
                    np.random.rand()
                    torch.rand(4)
                    output = original_forward(*args, **kwargs)
                    hidden_refs.append(weakref.ref(output))
                    return output

                def project(hidden):
                    self.assertFalse(torch.is_grad_enabled())
                    start = sum(shape[1] for shape in projection_shapes)
                    projection_shapes.append(hidden.shape)
                    return logits[:, start:start + hidden.shape[1]].clone()

                rng = capture_rng()
                with mock.patch.object(model.reference.model.draft_model, "forward", side_effect=reference_call), \
                        mock.patch.object(model.lm_head, "forward", new=project), \
                        mock.patch.object(model.draft_model, "forward", side_effect=AssertionError("unexpected current forward")):
                    actual = model._reference_feedback(
                        batch["input_ids"], batch["hidden_states"], batch["loss_mask"],
                        anchors, keep, labels, weights)
                self.assertEqual(len(full_shapes), 1)
                self.assertEqual(full_shapes[0][0][-1], 12 + 5 * 4)
                self.assertEqual(full_shapes[0][1][-2], 5 * 4)
                self.assertEqual(len(projection_shapes), math.ceil(5 / 2))
                self.assertEqual([shape[1] for shape in projection_shapes], [2, 2, 1])
                for observed, expected in zip(actual, (expected_prefix, expected_softplus, expected_correct)):
                    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                    self.assertFalse(observed.requires_grad)
                    self.assertIsNone(observed.grad_fn)
                    self.assertEqual(observed.numel(), labels.numel())
                self.assertTrue(bool(actual[2][0, 1, 3]))
                self.assertFalse(bool(actual[0][0, 1, 3]))
                self.assertTrue(bool(actual[0][0, 2, 1]))  # Include the matched EOS itself.
                self.assertFalse(bool(actual[0][0, 2, 2]))
                self.assertTrue(trees_equal(rng, capture_rng()))
                self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
                self.assertTrue(all(reference() is None for reference in hidden_refs))
                self.assertIsNone(batch["hidden_states"].grad)

    def test_reference_finishes_before_current_graph_and_stays_fixed_during_backward(self):
        model = self.model()
        model.begin_event()
        batch = osd_batch(model)
        reference = model.reference.model.draft_model
        frozen_state = copy.deepcopy(reference.state_dict())
        hidden_refs = []
        phases = []
        reference_forward = reference.forward
        current_forward = model.draft_model.forward

        def reference_call(*args, **kwargs):
            phases.append("reference")
            self.assertFalse(torch.is_grad_enabled())
            output = reference_forward(*args, **kwargs)
            hidden_refs.append(weakref.ref(output))
            return output

        def current_call(*args, **kwargs):
            phases.append("current")
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(all(reference() is None for reference in hidden_refs))
            return current_forward(*args, **kwargs)

        rng = capture_rng()
        with mock.patch.object(reference, "forward", side_effect=reference_call), \
                mock.patch.object(model.draft_model, "forward", side_effect=current_call), \
                mock.patch.object(model, "_sample_anchor_positions", wraps=model._sample_anchor_positions) as sampler:
            loss, _, _ = model(**batch)
            loss.backward()
            self.assertEqual(sampler.call_count, 1)
            self.assertEqual(phases, ["reference", "current"])
        rng_after = capture_rng()
        restore_rng(rng)
        model._sample_anchor_positions(12, batch["loss_mask"], torch.device("cpu"))
        self.assertTrue(trees_equal(rng_after, capture_rng()))
        self.assertTrue(trees_equal(frozen_state, reference.state_dict()))
        for name, parameter in reference.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotEqual(parameter.data_ptr(), dict(model.draft_model.named_parameters())[name].data_ptr())
        self.assertFalse(any("reference" in name for name in model.state_dict()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.draft_model.parameters()))

    def test_chunk_sizes_preserve_active_auxiliary_loss_gradients_counts_and_rng(self):
        author = tiny_author(dtype=torch.float32)
        outputs = []
        for chunk_size in (1, 2, 64):
            model = TargetObjectiveModel(copy.deepcopy(author), ARM)
            self.addCleanup(model.end_event)
            model.objective_chunk_blocks = chunk_size
            model.begin_event()

            def feedback(*args):
                weights = args[-1]
                prefix = weights > 0
                prefix[:, 1::2] = False
                return prefix, torch.zeros_like(weights), weights > 0

            torch.manual_seed(17)
            with mock.patch.object(model, "_reference_feedback", side_effect=feedback):
                loss, _, metrics = model(**osd_batch(model))
                loss.backward()
            self.assertGreater(float(metrics["sum_metrics"]["auxiliary_loss_numerator"]), 0)
            outputs.append((loss.detach(), parameter_gradients(model), metrics, capture_rng()))
        for actual in outputs[1:]:
            torch.testing.assert_close(actual[0], outputs[0][0], atol=2e-6, rtol=2e-6)
            self.assertTrue(trees_equal(actual[3], outputs[0][3]))
            for name, value in actual[2]["sum_metrics"].items():
                torch.testing.assert_close(value, outputs[0][2]["sum_metrics"][name], atol=2e-6, rtol=2e-6)
            for name, gradient in actual[1].items():
                torch.testing.assert_close(gradient, outputs[0][1][name], atol=2e-6, rtol=2e-5)

    def test_empty_reference_sets_have_zero_raw_diagnostic_denominators_and_keep_kl(self):
        model = self.model()
        model.begin_event()

        def empty_feedback(*args):
            weights = args[-1]
            return torch.zeros_like(weights, dtype=torch.bool), torch.zeros_like(weights), torch.zeros_like(weights, dtype=torch.bool)

        with mock.patch.object(model, "_reference_feedback", side_effect=empty_feedback):
            loss, _, metrics = model(**osd_batch(model))
            loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertEqual(float(metrics["sum_metrics"]["auxiliary_loss_numerator"]), 0)
        for name in ("protected_activation_rate", "reference_correct_to_incorrect_rate"):
            self.assertEqual(tuple(float(value) for value in metrics["ratio_metrics"][name]), (0., 0.))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in parameter_gradients(model).values()))

    def test_empty_response_raises_existing_error_after_one_sampler_call(self):
        model = self.model()
        model.begin_event()
        batch = osd_batch(model)
        batch["loss_mask"].zero_()
        with mock.patch.object(model, "_sample_anchor_positions", wraps=model._sample_anchor_positions) as sampler:
            with self.assertRaisesRegex(ValueError, "two consecutive supervised tokens"):
                model(**batch)
        self.assertEqual(sampler.call_count, 1)


if __name__ == "__main__":
    unittest.main()
