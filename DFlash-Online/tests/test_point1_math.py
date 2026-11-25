"""CPU-only checks of the diagnostic's derivative order and reported arithmetic."""

import unittest

import torch

from experiments.frontier_preservation_pilot.math import adamw_shadow, optimizer_vjp
from experiments.frontier_preservation_pilot.point1_math import (
    DifferenceVectors,
    adjoint_reconciliation,
    fp32_difference,
    matched_assessment_contrast,
    optimizer_jvp,
    optimizer_jvp_view,
    prediction_diagnostics,
    score_contrast,
    vector_metrics,
)


# Preserve the existing test_math.py derivative criterion, before GPU work.
DERIVATIVE_RTOL = 2e-12
DERIVATIVE_ATOL = 2e-12


def independent_update(parameters, gradients, means, squares, steps, options, max_norm):
    """Differentiable warm Adam formula independent of the analytic context."""
    norm = torch.cat([gradient.flatten() for gradient in gradients]).norm()
    clip = torch.minimum(norm.new_tensor(1.0), max_norm / (norm + 1e-6))
    beta1, beta2 = options['betas']
    result = []
    for theta, gradient, mean, square, step in zip(parameters, gradients, means, squares, steps):
        clipped = gradient * clip
        corrected_mean = (beta1 * mean + (1 - beta1) * clipped) / (1 - beta1 ** (step + 1))
        corrected_square = (beta2 * square + (1 - beta2) * clipped.square()) / (
            1 - beta2 ** (step + 1)
        )
        result.append(theta * (1 - options['lr'] * options['weight_decay'])
                      - options['lr'] * corrected_mean / (corrected_square.sqrt() + options['eps']))
    return tuple(result)


class OptimizerJVPTests(unittest.TestCase):
    def fixture(self, scale):
        generator = torch.Generator().manual_seed(31)
        parameters = [torch.randn(3, generator=generator, dtype=torch.float64),
                      torch.randn(2, 2, generator=generator, dtype=torch.float64)]
        gradients = [torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
                     for p in parameters]
        means = [torch.randn(p.shape, generator=generator, dtype=torch.float64) / 10
                 for p in parameters]
        squares = [torch.rand(p.shape, generator=generator, dtype=torch.float64) + 0.1
                   for p in parameters]
        directions = [torch.randn(p.shape, generator=generator, dtype=torch.float64)
                      for p in parameters]
        cotangents = [torch.randn(p.shape, generator=generator, dtype=torch.float64)
                      for p in parameters]
        steps = [128, 7]
        options = dict(lr=0.03, weight_decay=0.07, betas=(0.8, 0.91), eps=0.02)
        _, context = adamw_shadow(parameters, gradients, means, squares, steps, options,
                                  max_grad_norm=1.0, dtype=torch.float64)
        return parameters, gradients, means, squares, steps, options, context, directions, cotangents

    def test_clipped_and_unclipped_jvp_matches_independent_autodiff(self):
        for scale in (0.01, 4.0):
            with self.subTest(scale=scale):
                parameters, gradients, means, squares, steps, options, context, directions, _ = self.fixture(scale)
                _, expected = torch.autograd.functional.jvp(
                    lambda *g: independent_update(parameters, g, means, squares, steps, options, 1.0),
                    tuple(gradients), tuple(directions), strict=True,
                )
                actual = optimizer_jvp(context, directions)
                self.assertEqual(context['clipping_active'], scale == 4.0)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)

    def test_jvp_vjp_duality_and_wrong_order_is_detected(self):
        for scale in (0.01, 4.0):
            with self.subTest(scale=scale):
                *_, context, directions, cotangents = self.fixture(scale)
                jvp = optimizer_jvp(context, directions)
                vjp = optimizer_vjp(context, cotangents)
                forward = sum((r * value).sum() for r, value in zip(cotangents, jvp))
                reverse = sum((d * value).sum() for d, value in zip(directions, vjp))
                torch.testing.assert_close(forward, reverse, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)
                if context['clipping_active']:
                    wrong = optimizer_vjp(context, directions)
                    self.assertGreater(sum(float((a - b).abs().sum()) for a, b in zip(jvp, wrong)), 1e-5)

    def test_lazy_view_repeats_without_mutating_context_or_inputs(self):
        *_, context, directions, _ = self.fixture(4.0)
        before = [value.clone() for value in directions]
        view = optimizer_jvp_view(context, directions)
        self.assertEqual(len(view), len(directions))
        first, second = list(view), list(view)
        for a, b, original, current in zip(first, second, before, directions):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(original, current, rtol=0, atol=0)

    def test_zero_gradient_and_zero_direction(self):
        zero = torch.zeros(3, dtype=torch.float64)
        options = dict(lr=0.03, weight_decay=0.01, betas=(0.8, 0.91), eps=0.02)
        _, context = adamw_shadow([zero], [zero], [zero], [zero], [5], options,
                                  max_grad_norm=1.0, dtype=torch.float64)
        self.assertFalse(context['clipping_active'])
        actual = optimizer_jvp(context, [torch.ones_like(zero)])[0]
        expected = -0.03 / (1 - 0.8**6) * (1 - 0.8) / 0.02
        torch.testing.assert_close(actual, torch.full_like(zero, expected),
                                   rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)
        torch.testing.assert_close(optimizer_jvp(context, [zero])[0], zero, rtol=0, atol=0)

    def test_inherits_exact_clipping_boundary_branch(self):
        gradient = torch.tensor([1.0 - 1e-6], dtype=torch.float64)
        zero = torch.zeros_like(gradient)
        options = dict(lr=0.03, weight_decay=0.01, betas=(0.8, 0.91), eps=0.02)
        _, context = adamw_shadow([zero], [gradient], [zero], [torch.ones_like(zero)],
                                  [5], options, max_grad_norm=1.0, dtype=torch.float64)
        self.assertEqual(float(context['clip']), 1.0)
        self.assertTrue(context['clipping_active'])
        direction = torch.ones_like(gradient)
        clipped = optimizer_jvp(context, [direction])[0]
        norm = context['norm']
        expected = context['adam_diagonal'][0] * (
            direction - gradient * (gradient * direction).sum() / (norm * (norm + 1e-6))
        )
        torch.testing.assert_close(clipped, expected, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)
        self.assertGreater(float((clipped - context['adam_diagonal'][0] * direction).abs().sum()), 0)

    def test_nonfinite_and_shape_failures_are_not_hidden(self):
        *_, context, directions, _ = self.fixture(4.0)
        with self.assertRaisesRegex(ValueError, 'shape'):
            optimizer_jvp(context, [directions[0][:1], directions[1]])
        directions[0][0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            optimizer_jvp(context, directions)


class VectorSummaryTests(unittest.TestCase):
    def test_bf16_operands_promoted_before_difference_and_lazy_repeat(self):
        candidate = [torch.tensor([256.0], dtype=torch.bfloat16)]
        reference = [torch.tensor([-1.0], dtype=torch.bfloat16)]
        self.assertEqual(float((candidate[0] - reference[0]).float()), 256.0)
        self.assertEqual(float(fp32_difference(candidate[0], reference[0])), 257.0)
        view = DifferenceVectors(candidate, reference)
        self.assertEqual(len(view), 1)
        self.assertEqual(float(list(view)[0]), 257.0)
        self.assertEqual(float(list(view)[0]), 257.0)

    def test_streamed_metrics_match_flat_fp64_and_keep_layer_contributions(self):
        reference = [torch.tensor([1.0, 2.0, -3.0]), torch.tensor([[4.0, 5.0]])]
        candidate = [torch.tensor([1.0, -2.0, -2.0]), torch.tensor([[5.0, 7.0]])]
        cotangent = [torch.tensor([2.0, -3.0, 1.0]), torch.tensor([[1.0, -2.0]])]
        result = vector_metrics(iter(reference), iter(candidate), reference_name='linear',
                                candidate_name='finite', cotangent=iter(cotangent),
                                parameter_names=['model.layers.0.a', 'model.layers.1.b'],
                                chunk_elements=2)
        a = torch.cat([x.flatten().double() for x in reference])
        b = torch.cat([x.flatten().double() for x in candidate])
        r = torch.cat([x.flatten().double() for x in cotangent])
        self.assertEqual(result['reference_norm'], float(a.norm()))
        self.assertEqual(result['candidate_norm'], float(b.norm()))
        self.assertEqual(result['difference_norm'], float((b - a).norm()))
        self.assertEqual(result['difference_signed_projection'], float(-r.dot(b - a)))
        self.assertEqual(result['differing_coordinates'], 4)
        self.assertEqual(result['coordinates'], 5)
        self.assertEqual(result['relative_difference_denominator'], 'norm(linear)')
        self.assertEqual(set(result['per_layer']), {'layers.0', 'layers.1'})
        self.assertEqual(sum(x['reference_signed_projection'] for x in result['per_layer'].values()),
                         result['reference_signed_projection'])

    def test_zero_reference_and_both_zero_have_no_invented_denominator(self):
        zero, one = [torch.zeros(2)], [torch.ones(2)]
        both = vector_metrics(zero, zero, reference_name='zero', candidate_name='also_zero')
        self.assertEqual(both['zero_status'], 'both_zero')
        self.assertIsNone(both['cosine'])
        self.assertIsNone(both['relative_difference'])
        single = vector_metrics(zero, one, reference_name='zero', candidate_name='one')
        self.assertEqual(single['zero_status'], 'reference_zero')
        self.assertIsNone(single['cosine'])
        self.assertIsNone(single['relative_difference'])
        reversed_result = vector_metrics(one, zero, reference_name='one', candidate_name='zero')
        self.assertEqual(reversed_result['zero_status'], 'candidate_zero')
        self.assertEqual(reversed_result['relative_difference'], 1.0)

    def test_fp64_products_preserve_small_and_cancelling_contributions(self):
        small = torch.tensor([1e-30, -1e-30], dtype=torch.float32)
        result = vector_metrics([small], [small], reference_name='a', candidate_name='b')
        self.assertGreater(result['reference_norm'], 0)
        self.assertEqual(result['difference_norm'], 0)
        self.assertAlmostEqual(result['cosine'], 1)

    def test_nonfinite_vectors_and_length_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            vector_metrics([torch.tensor([torch.inf])], [torch.ones(1)],
                           reference_name='a', candidate_name='b')
        with self.assertRaises(ValueError):
            vector_metrics([torch.ones(1)], [torch.ones(1), torch.ones(1)],
                           reference_name='a', candidate_name='b')


class ScalarContrastTests(unittest.TestCase):
    def test_predictions_report_sign_error_and_zero_without_new_gate(self):
        row = prediction_diagnostics(-0.25, 0.125, prediction_name='native_point')
        self.assertFalse(row['signs_match'])
        self.assertEqual(row['prediction_sign'], 'negative')
        self.assertEqual(row['finite_benefit_sign'], 'positive')
        self.assertEqual(row['absolute_prediction_error'], 0.375)
        self.assertEqual(row['relative_absolute_error'], 3.0)
        self.assertFalse(row['admission_gate'])
        zero = prediction_diagnostics(1.0, 0.0, prediction_name='shadow_point')
        self.assertIsNone(zero['relative_absolute_error'])
        self.assertTrue(zero['zero_benefit'])
        dual = adjoint_reconciliation(0.125, 0.25)
        self.assertEqual(dual['difference'], 0.125)
        self.assertEqual(dual['relative_difference'], 1.0)
        self.assertIsNone(adjoint_reconciliation(0.0, 1.0)['relative_difference'])
        self.assertTrue(adjoint_reconciliation(0.0, 0.0)['both_zero'])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            prediction_diagnostics(float('nan'), 1.0, prediction_name='bad')

    def test_score_signs_and_mass_offset_identity(self):
        beta = torch.tensor([[1.0, 0.8, 0.6, 0.0], [1.0, 0.7, 0.0, 0.0]])
        prefix = beta > 0
        candidate = beta + torch.tensor([[0.1, -0.04, -0.06, 0.0], [-0.03, 0.03, 0.0, 0.0]])
        scores = torch.tensor([[1e6, 1e6 + 0.125, 1e6 + 0.25, 17.0],
                               [-2.0, 3.0, 18.0, 19.0]])
        result = score_contrast(candidate, beta, scores, beta, prefix)
        self.assertGreater(result['raw']['positive'], 0)
        self.assertLess(result['raw']['negative'], 0)
        self.assertAlmostEqual(result['raw']['total'],
                               result['raw']['positive'] + result['raw']['negative'], places=10)
        self.assertAlmostEqual(result['raw_minus_centered'], result['mass_offset_contribution'], places=10)
        self.assertEqual(result['prefix_occurrences'], 5)
        self.assertEqual(result['nonempty_pools'], 2)

    def test_primary_prediction_may_be_negative_and_empty_pool_is_zero(self):
        beta = torch.ones((2, 3))
        prefix = torch.tensor([[True, True, False], [False, False, False]])
        candidate = torch.tensor([[1.25, 0.75, 10.0], [2.0, 2.0, 2.0]])
        scores = torch.tensor([[-2.0, 2.0, 99.0], [99.0, 99.0, 99.0]])
        result = score_contrast(candidate, beta, scores, beta, prefix)
        self.assertEqual(result['raw']['total'], -1.0)
        self.assertEqual(result['beta_centered']['total'], -1.0)
        self.assertEqual(result['per_pool']['raw'][1], 0.0)

    def test_matched_assessment_reports_cancellation_and_requires_shapes(self):
        reference = torch.tensor([1e6, 2.0, 4.0], dtype=torch.float32)
        candidate = torch.tensor([1e6, 1.75, 4.125], dtype=torch.float32)
        result = matched_assessment_contrast(reference, candidate)
        self.assertEqual(result['benefit']['total'], 0.125)
        self.assertEqual(result['benefit']['positive'], 0.25)
        self.assertEqual(result['benefit']['negative'], -0.125)
        self.assertEqual(result['occurrences'], 3)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            matched_assessment_contrast(reference, candidate[:1])


if __name__ == '__main__':
    unittest.main()
