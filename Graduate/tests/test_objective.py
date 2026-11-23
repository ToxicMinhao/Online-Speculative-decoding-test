"""Independent CPU formula/gradient checks; no model runtime or GPU qualification."""

import math
import random
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from graduate.config import ObjectiveSettings
from graduate.objective import (
    ResponseTerms, _prefix_weights, compute_chunk, reduce_response, response_loss,
)


SETTINGS = ObjectiveSettings(
    first_error_redistribution_fraction=0.5,
    prefix_probability_smoothing=0.5,
    prefix_relative_cap=0.25,
    denominator_epsilon=1e-6,
)
FP32_TOLERANCE = {"rtol": 2e-6, "atol": 2e-7}
# The independent effective-weight oracle combines backward paths. Native K+
# deliberately retains separate BF16 paths; use its historical gradient bounds.
BF16_GRADIENT_TOLERANCE = {"rtol": 0.02, "atol": 2e-4}


def scalar_prefix_weights(probability, base, valid, prefix):
    """Scalar definition, without production masks or tensor prefix reductions."""
    product = 1.0
    scores = [0.0] * len(base)
    for end, available in enumerate(valid):
        if available:
            product *= 0.5 + 0.5 * probability[end]
            for start in range(end + 1):
                scores[start] += product
    positions = [index for index, selected in enumerate(prefix) if selected]
    result = list(base)
    if len(positions) <= 1:
        return result
    mass = sum(base[index] for index in positions)
    total_score = sum(scores[index] for index in positions)
    relative = {index: mass * scores[index] / total_score / base[index] - 1
                for index in positions}
    maximum = max(abs(value) for value in relative.values())
    factor = min(1.0, 0.25 / maximum) if maximum else 0.0
    for index in positions:
        result[index] *= 1 + factor * relative[index]
    return result


def explicit_response_weights(logits, labels, weights, stop_token_ids=()):
    """Traverse blocks and construct equivalent final coefficients independently."""
    proposals = logits.detach().argmax(-1)
    probabilities = F.log_softmax(logits.detach(), -1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1).float().exp()
    selected = torch.zeros_like(weights)
    prefix_delta = torch.zeros_like(weights)
    errors = []
    for block in range(labels.shape[1]):
        target = labels[0, block].tolist()
        proposal = proposals[0, block].tolist()
        base = weights[0, block].tolist()
        valid = [False] * len(target)
        prefix = [False] * len(target)
        if target[0] not in stop_token_ids:
            for offset in range(1, len(target)):
                if base[offset] <= 0:
                    break
                valid[offset] = True
                if target[offset] in stop_token_ids:
                    break
        for offset in range(1, len(target)):
            if not valid[offset]:
                break
            selected[0, block, offset] = base[offset]
            if proposal[offset] != target[offset]:
                following = offset + 1
                priority = 2 if (following < len(target) and valid[following]
                                 and proposal[following] == target[following]) else 1
                errors.append((block, offset, priority))
                break
            prefix[offset] = True
        allocated = scalar_prefix_weights(probabilities[0, block].tolist(), base, valid, prefix)
        prefix_delta[0, block] = torch.tensor(allocated) - weights[0, block]

    modified = selected.clone()
    if {priority for _, _, priority in errors} == {1, 2}:
        mass = sum(float(weights[0, block, offset]) for block, offset, _ in errors)
        mean_priority = sum(float(weights[0, block, offset]) * priority
                            for block, offset, priority in errors) / mass
        for block, offset, priority in errors:
            modified[0, block, offset] *= 0.5 + 0.5 * priority / mean_priority
    return selected, modified + prefix_delta, errors


def reference_loss(logits, teacher, labels, weights, stop_token_ids=()):
    selected, modified, _ = explicit_response_weights(logits, labels, weights, stop_token_ids)
    rows = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher.detach(), -1),
                    reduction="none").sum(-1)
    return (rows * modified).sum() / (selected.sum() + 1e-6)


def mixed_fixture():
    """Place the two first-error priorities in different eight-block chunks."""
    generator = torch.Generator().manual_seed(20261103)
    logits = torch.randn(1, 17, 16, 31, generator=generator).to(torch.bfloat16)
    teacher = torch.randn(logits.shape, generator=generator).to(torch.bfloat16)
    labels = logits.argmax(-1)
    offsets = torch.arange(16).float()
    weights = torch.exp(-(offsets - 1).clamp_min(0) / 7).expand(1, 17, 16).clone()
    weights[..., 0] = 0
    for block in range(16):
        offset = 1 + block % 5
        labels[0, block, offset] = (labels[0, block, offset] + 1) % 31
        weights[0, block] *= 0.15 + 0.06 * block
        if block < 8:
            labels[0, block, offset + 1] = (labels[0, block, offset + 1] + 1) % 31
    return logits, teacher, labels, weights


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_gradient_matches(self, actual, expected):
        torch.testing.assert_close(actual, expected, **BF16_GRADIENT_TOLERANCE)
        relative = torch.linalg.vector_norm((actual - expected).float()) / torch.linalg.vector_norm(expected.float())
        self.assertLessEqual(float(relative), 0.01)

    def test_mixed_priorities_across_chunks_match_independent_loss_and_gradients(self):
        logits, teacher, labels, weights = mixed_fixture()
        teacher.requires_grad_()
        actual_logits = logits.clone().requires_grad_()
        expected_logits = logits.clone().requires_grad_()
        actual, metrics = response_loss(actual_logits, teacher, labels, weights,
                                        torch.nn.Identity(), (), SETTINGS, 8)
        expected = reference_loss(expected_logits, teacher, labels, weights)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, **FP32_TOLERANCE)
        self.assert_gradient_matches(actual_logits.grad, expected_logits.grad)
        self.assertIsNone(teacher.grad)
        selected, _, errors = explicit_response_weights(logits, labels, weights)
        self.assertEqual([sum(priority == value for _, _, priority in errors) for value in (1, 2)], [8, 8])
        torch.testing.assert_close(metrics["denominator"], selected.sum() + 1e-6, **FP32_TOLERANCE)
        self.assertTrue(torch.equal(actual_logits.grad[selected == 0], torch.zeros_like(actual_logits.grad[selected == 0])))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        self.assertGreater(abs(float(metrics["kplus_numerator"] - metrics["native_numerator"])), 0.01)

    def test_loss_does_not_backpropagate_into_position_weights(self):
        logits, teacher, labels, weights = mixed_fixture()
        logits.requires_grad_()
        weights.requires_grad_()
        loss, _ = response_loss(logits, teacher, labels, weights,
                                torch.nn.Identity(), (), SETTINGS, 8)
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertGreater(float(logits.grad.float().abs().sum()), 0)

    def test_prefix_weights_match_scalar_oracle_and_preserve_mass_bounds_support(self):
        rng = random.Random(20261116)
        for length in range(1, 16):
            for prefix_length in (0, 1, length):
                with self.subTest(length=length, prefix_length=prefix_length):
                    probability = torch.tensor([0.0] + [rng.random() for _ in range(length)])
                    base = torch.tensor([0.0] + [math.exp(-index / 7) for index in range(length)])
                    valid = torch.arange(length + 1) > 0
                    prefix = valid & (torch.arange(length + 1) <= prefix_length)
                    actual = _prefix_weights(probability, base, valid, prefix, SETTINGS)
                    expected = scalar_prefix_weights(probability.tolist(), base.tolist(), valid.tolist(), prefix.tolist())
                    torch.testing.assert_close(actual, torch.tensor(expected), **FP32_TOLERANCE)
                    torch.testing.assert_close(actual[prefix].sum(), base[prefix].sum(), **FP32_TOLERANCE)
                    self.assertTrue(bool((actual[prefix] >= 0.75 * base[prefix] - 2e-7).all()))
                    self.assertTrue(bool((actual[prefix] <= 1.25 * base[prefix] + 2e-7).all()))
                    self.assertTrue(torch.equal(actual[~prefix], base[~prefix]))
                    self.assertFalse(actual.requires_grad)

    def test_outside_prefix_probability_affects_scores_without_changing_its_weight(self):
        probability = torch.tensor([0.0, 0.9, 0.8, 0.6, 0.7, 0.2])
        base = torch.tensor([0.0] + [math.exp(-index / 7) for index in range(5)])
        valid = base > 0
        prefix = valid.clone()
        prefix[-1] = False
        before = _prefix_weights(probability, base, valid, prefix, SETTINGS)
        probability[-1] = 0.99
        after = _prefix_weights(probability, base, valid, prefix, SETTINGS)
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.equal(before[~prefix], base[~prefix]))
        self.assertTrue(torch.equal(after[~prefix], base[~prefix]))

    def test_gap_eos_seed_and_eos_mismatch_match_scalar_selection(self):
        logits = torch.zeros(1, 5, 6, 10, dtype=torch.bfloat16)
        labels = torch.arange(6).reshape(1, 1, 6).expand(1, 5, 6).clone()
        weights = torch.ones(1, 5, 6)
        weights[..., 0] = 0
        weights[0, 0, 3] = 0
        labels[0, 1, 2] = 9
        labels[0, 2, 0] = 9
        labels[0, 3, 2] = 9
        logits.scatter_(-1, labels.unsqueeze(-1), 3)
        logits[0, 3, 2, 8] = 4  # Wrong EOS has no eligible successor.
        logits[0, 4, 1, 8] = 4  # Correct immediate successor gives priority two.
        teacher = -logits
        actual_logits = logits.clone().requires_grad_()
        expected_logits = logits.clone().requires_grad_()
        actual, metrics = response_loss(actual_logits, teacher, labels, weights,
                                        torch.nn.Identity(), (9,), SETTINGS, 2)
        expected = reference_loss(expected_logits, teacher, labels, weights, (9,))
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, **FP32_TOLERANCE)
        self.assert_gradient_matches(actual_logits.grad, expected_logits.grad)
        selected, _, _ = explicit_response_weights(logits, labels, weights, (9,))
        self.assertEqual(selected.sum(-1).flatten().tolist(), [2, 2, 0, 2, 1])
        self.assertEqual(float(metrics["priority1_weight"]), 1)
        self.assertEqual(float(metrics["priority2_weight"]), 1)

    def test_no_eligible_positions_produce_zero_loss_and_zero_gradient(self):
        logits, teacher, labels, weights = mixed_fixture()
        weights.zero_()
        logits.requires_grad_()
        loss, metrics = response_loss(logits, teacher, labels, weights, torch.nn.Identity(), (), SETTINGS, 8)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertAlmostEqual(float(metrics["denominator"]), 1e-6, places=12)

    def test_neutral_all_correct_and_single_priority_keep_native_bf16_gradient(self):
        logits, teacher, labels, weights = mixed_fixture()
        logits, teacher = logits[:, :2], teacher[:, :2]
        for first_error in (False, True):
            with self.subTest(first_error=first_error):
                labels = logits.argmax(-1)
                weights = torch.zeros_like(labels, dtype=torch.float32)
                weights[..., 1] = 1
                if first_error:
                    labels[..., 1] = (labels[..., 1] + 1) % logits.shape[-1]
                actual_logits = logits.clone().requires_grad_()
                expected_logits = logits.clone().requires_grad_()
                actual, metrics = response_loss(actual_logits, teacher, labels, weights,
                                                torch.nn.Identity(), (), SETTINGS, 8)
                rows = F.kl_div(F.log_softmax(expected_logits, -1), F.softmax(teacher, -1),
                                reduction="none").sum(-1)
                expected = (rows * weights).sum() / (weights.sum() + 1e-6)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=0, atol=0)
                self.assertEqual(float(metrics["changed_prefix_positions"]), 0)

    def test_zero_valued_prefix_delta_preserves_its_nonzero_derivative(self):
        rows = torch.zeros(3, requires_grad=True)
        delta = (rows * torch.tensor([0.25, -0.25, 0])).sum()
        native = torch.tensor(3.0, requires_grad=True)
        zero = torch.tensor(0.0)
        terms = ResponseTerms(native, torch.tensor(2.0), zero, zero, zero, zero,
                              delta, torch.tensor(2.0))
        loss, _ = reduce_response(terms, SETTINGS)
        loss.backward()
        self.assertEqual(float(delta.detach()), 0)
        torch.testing.assert_close(rows.grad, torch.tensor([0.25, -0.25, 0]) / (2 + 1e-6))

    def test_probability_is_exponentiated_after_bf16_log_probability_cast(self):
        logits, teacher, labels, weights = mixed_fixture()
        with patch("graduate.objective._prefix_weights", wraps=_prefix_weights) as coefficients:
            terms = compute_chunk(logits, teacher, labels, weights, (), SETTINGS)
        probability = coefficients.call_args.args[0]
        native_log_probability = F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(probability, native_log_probability.float().exp(), rtol=0, atol=0)
        self.assertFalse(torch.equal(probability, native_log_probability.exp().float()))
        self.assertEqual(terms.native_numerator.dtype, torch.float32)

    def test_checkpointed_projections_keep_physical_shape_and_do_not_consume_rng(self):
        logits, teacher, labels, weights = mixed_fixture()
        logits.requires_grad_()
        shapes = []

        class RecordingHead(torch.nn.Module):
            def forward(self, hidden):
                shapes.append(tuple(hidden.shape))
                return hidden

        rng = torch.get_rng_state().clone()
        loss, _ = response_loss(logits, teacher, labels, weights, RecordingHead(), (), SETTINGS, 8)
        self.assertEqual(shapes, [(1, 8, 16, 31)] * 4 + [(1, 1, 16, 31)] * 2)
        loss.backward()
        self.assertGreater(len(shapes), 6)  # Backward recomputes projections within chunks.
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_two_responses_are_normalized_separately_before_accumulation(self):
        logits, teacher, labels, weights = mixed_fixture()
        actual_losses, expected_losses, metrics = [], [], []
        actual_inputs, expected_inputs = [], []
        for target, local_weights in ((teacher, weights), (-teacher, weights * 0.31)):
            actual_logits = logits.clone().requires_grad_()
            expected_logits = logits.clone().requires_grad_()
            actual, report = response_loss(actual_logits, target, labels, local_weights,
                                           torch.nn.Identity(), (), SETTINGS, 8)
            actual_losses.append(actual)
            expected_losses.append(reference_loss(expected_logits, target, labels, local_weights))
            metrics.append(report)
            actual_inputs.append(actual_logits)
            expected_inputs.append(expected_logits)
        actual = sum(actual_losses) / 2
        expected = sum(expected_losses) / 2
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, **FP32_TOLERANCE)
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            self.assert_gradient_matches(actual_input.grad, expected_input.grad)
        pooled = sum(report["numerator"] for report in metrics) / sum(report["denominator"] for report in metrics)
        self.assertGreater(abs(float(actual.detach() - pooled)), 1e-4)


if __name__ == "__main__":
    unittest.main()
