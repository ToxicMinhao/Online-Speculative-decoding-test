"""CPU value and gradient checks for the reference-margin auxiliary penalty."""
import unittest

import torch
import torch.nn.functional as F

from online_feedback.ranking import (
    OSD_OBJECTIVES,
    RANKING_OBJECTIVES,
    allocation_coefficients,
    reference_margin_penalty,
    rival_softplus,
)


class ReferenceMarginTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_reference_margin_uses_exact_prefix_allocation(self):
        self.assertIn("osd_reference_margin", RANKING_OBJECTIVES)
        self.assertIn("osd_reference_margin", OSD_OBJECTIVES)
        weights = torch.tensor(
            [[[0., 3., 2., 1.], [0., 4., 2., 1.]]], requires_grad=True,
        )
        prefix = torch.tensor(
            [[[False, True, True, False], [False, False, False, False]]],
        )
        reference_coefficients, reference_rho = allocation_coefficients(
            prefix, weights, "osd_reference_margin",
        )
        prefix_coefficients, prefix_rho = allocation_coefficients(
            prefix, weights, "osd_prefix",
        )
        self.assertTrue(torch.equal(reference_coefficients, prefix_coefficients))
        self.assertTrue(torch.equal(reference_rho, prefix_rho))
        self.assertTrue(torch.equal(reference_coefficients, prefix.float()))
        self.assertFalse(reference_coefficients.requires_grad)
        self.assertFalse(reference_rho.requires_grad)

    def test_equal_margins_have_zero_value_and_zero_auxiliary_gradient(self):
        current = torch.tensor(
            [[3., 2., 1.], [1., 3., 2.]], requires_grad=True,
        )
        reference = current.detach().clone().requires_grad_()
        labels = torch.tensor([0, 1])
        penalty = reference_margin_penalty(
            rival_softplus(current, labels), rival_softplus(reference, labels),
        )
        penalty.sum().backward()
        self.assertTrue(torch.equal(penalty, torch.zeros_like(penalty)))
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))
        self.assertIsNone(reference.grad)

    def test_improved_margins_have_zero_value_and_zero_auxiliary_gradient(self):
        current = torch.tensor(
            [[4., 0., 2.], [1., 5., 2.]], requires_grad=True,
        )
        reference = torch.tensor(
            [[3., 2., 1.], [2., 4., 3.]], requires_grad=True,
        )
        labels = torch.tensor([0, 1])
        penalty = reference_margin_penalty(
            rival_softplus(current, labels), rival_softplus(reference, labels),
        )
        penalty.sum().backward()
        self.assertTrue(torch.equal(penalty, torch.zeros_like(penalty)))
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))
        self.assertIsNone(reference.grad)

    def test_worse_margins_have_exact_original_prefix_logit_gradients(self):
        current = torch.tensor(
            [[3., 2.5, 0.], [1., 2., 4.]], requires_grad=True,
        )
        original_prefix = current.detach().clone().requires_grad_()
        reference = torch.tensor(
            [[3., 1., 2.], [0., 3., 4.]], requires_grad=True,
        )
        labels = torch.tensor([0, 1])
        weights = torch.tensor([2., 7.])
        denominator = weights.sum() + 1e-6
        penalty = reference_margin_penalty(
            rival_softplus(current, labels), rival_softplus(reference, labels),
        )
        self.assertTrue((penalty > 0).all())
        (weights * penalty).sum().div(denominator).backward()
        (weights * rival_softplus(original_prefix, labels)).sum().div(
            denominator,
        ).backward()
        self.assertTrue(torch.equal(current.grad, original_prefix.grad))
        self.assertIsNone(reference.grad)

    def test_each_model_uses_its_own_rival_when_rivals_change(self):
        current = torch.tensor(
            [[4., 3., -2.], [4., -3., 3.75]], requires_grad=True,
        )
        reference = torch.tensor(
            [[4., -3., 3.5], [4., 3., -2.]], requires_grad=True,
        )
        labels = torch.tensor([0, 0])
        penalty = reference_margin_penalty(
            rival_softplus(current, labels), rival_softplus(reference, labels),
        )
        expected = torch.tensor([
            0., F.softplus(torch.tensor(-0.25)) - F.softplus(torch.tensor(-1.)),
        ])
        torch.testing.assert_close(penalty, expected, rtol=0, atol=0)
        penalty.sum().backward()
        expected_gradient = torch.zeros_like(current)
        slope = torch.sigmoid(torch.tensor(-0.25))
        expected_gradient[1, 0] = -slope
        expected_gradient[1, 2] = slope
        torch.testing.assert_close(current.grad, expected_gradient)
        self.assertIsNone(reference.grad)
        # Using current's rival ID for reference makes row 0 incorrectly active;
        # using reference's rival ID for current makes row 1 incorrectly inactive.

    def test_full_vocabulary_rivals_and_first_index_ties_keep_prefix_gradients(self):
        current = torch.tensor(
            [[1., 5., 4., 4., -2.], [8., 8., 7., 9., 9.]],
            requires_grad=True,
        )
        reference = torch.tensor(
            [[1., 7., 4., 4., -2.], [8., 8., 7., 10., 9.]],
            requires_grad=True,
        )
        labels = torch.tensor([1, 3])
        penalty = reference_margin_penalty(
            rival_softplus(current, labels), rival_softplus(reference, labels),
        )
        expected = F.softplus(torch.tensor([-1., 0.])) - F.softplus(
            torch.tensor([-3., -1.]),
        )
        torch.testing.assert_close(penalty, expected, rtol=0, atol=0)
        penalty.sum().backward()
        expected_gradient = torch.zeros_like(current)
        slope = torch.sigmoid(torch.tensor(-1.))
        expected_gradient[0, 1], expected_gradient[0, 2] = -slope, slope
        expected_gradient[1, 3], expected_gradient[1, 4] = -0.5, 0.5
        torch.testing.assert_close(current.grad, expected_gradient)
        self.assertIsNone(reference.grad)

    def test_reference_is_detached_even_when_supplied_with_gradient_history(self):
        current = torch.tensor([1., 3.], requires_grad=True)
        reference = torch.tensor([0.5, 1.], requires_grad=True)
        reference_with_history = reference * 2
        penalty = reference_margin_penalty(current, reference_with_history)
        current_gradient, reference_gradient = torch.autograd.grad(
            penalty.sum(), (current, reference), allow_unused=True,
        )
        self.assertTrue(torch.equal(current_gradient, torch.tensor([0., 1.])))
        self.assertIsNone(reference_gradient)

    def test_relu_precedes_unequal_weight_reduction(self):
        current = torch.tensor([2., 1., 3.], requires_grad=True)
        reference = torch.tensor([1., 3., 3.], requires_grad=True)
        weights = torch.tensor([2., 7., 11.])
        denominator = weights.sum() + 1e-6
        penalty = reference_margin_penalty(current, reference)
        loss = (weights * penalty).sum() / denominator
        wrong_order = torch.relu((weights * (current - reference)).sum()) / denominator
        self.assertTrue(torch.equal(penalty, torch.tensor([1., 0., 0.])))
        torch.testing.assert_close(loss, 2 / denominator, rtol=0, atol=0)
        self.assertEqual(float(wrong_order.detach()), 0.)
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        expected_gradient = torch.tensor([2., 0., 0.]) / denominator
        self.assertTrue(torch.equal(current.grad, expected_gradient))
        self.assertIsNone(reference.grad)

    def test_bf16_logits_use_fp32_margin_softplus_and_reference_difference(self):
        current = torch.tensor(
            [[0.5, 256., -9.], [0.99609375, 1., -9.]],
            dtype=torch.bfloat16, requires_grad=True,
        )
        reference = torch.tensor(
            [[1., 256., -9.], [1., 1., -9.]],
            dtype=torch.bfloat16, requires_grad=True,
        )
        labels = torch.tensor([0, 0])
        current_softplus = rival_softplus(current, labels)
        reference_softplus = rival_softplus(reference, labels)
        penalty = reference_margin_penalty(current_softplus, reference_softplus)
        expected = F.softplus(torch.tensor([255.5, 0.00390625])) - F.softplus(
            torch.tensor([255., 0.]),
        )
        self.assertEqual(current_softplus.dtype, torch.float32)
        self.assertEqual(reference_softplus.dtype, torch.float32)
        self.assertEqual(penalty.dtype, torch.float32)
        torch.testing.assert_close(penalty, expected, rtol=0, atol=0)
        self.assertEqual(float(penalty[0].detach()), 0.5)
        self.assertGreater(float(penalty[1].detach()), 0.)
        penalty.sum().backward()
        expected_gradient = torch.zeros_like(current)
        slopes = torch.sigmoid(torch.tensor([255.5, 0.00390625])).to(torch.bfloat16)
        expected_gradient[:, 0] = -slopes
        expected_gradient[:, 1] = slopes
        self.assertTrue(torch.equal(current.grad, expected_gradient))
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertIsNone(reference.grad)


if __name__ == "__main__":
    unittest.main()
