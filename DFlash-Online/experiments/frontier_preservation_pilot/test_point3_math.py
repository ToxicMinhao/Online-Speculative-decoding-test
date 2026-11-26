"""Independent CPU checks for the fixed-K cumulative-maximum assessment."""

import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from . import math as legacy
from .point3_math import (
    assessment_pair,
    max_terms_from_margins,
    native_assessment_numerator,
)


def make_roles(matches):
    labels = torch.zeros((1, len(matches) + 1), dtype=torch.long)
    proposals = labels.clone()
    proposals[:, 1:] = torch.tensor([not match for match in matches]).long()
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, 0] = 0
    return legacy.current_roles(proposals, labels, weights)


def logits_for_margins(margins, dtype=torch.float32):
    canonical = torch.as_tensor(margins, dtype=dtype)
    return torch.stack((canonical, torch.zeros_like(canonical)), dim=-1)


def slice_roles(roles, start, stop):
    return {
        key: value[start:stop] if torch.is_tensor(value) and value.ndim else value
        for key, value in roles.items()
    }


class Point3AssessmentTests(unittest.TestCase):
    def test_bound_exhaustive_binary_outcomes_and_fixed_origin(self):
        # Different original frontiers, then every possible evaluated outcome.
        for origin in ([1], [0], [1, 1, 1], [0, 1, 1], [1, 0, 1, 0, 1]):
            roles = make_roles(origin)
            endpoint = int(roles['K'].item())
            for evaluated in itertools.product((False, True), repeat=len(origin)):
                margins = [[0] + [2 if correct else -2 for correct in evaluated]]
                values, _ = assessment_pair(logits_for_margins(margins), roles)
                prefix = next(
                    (position for position, correct in enumerate(evaluated) if not correct),
                    len(evaluated),
                )
                shortfall = endpoint - min(prefix, endpoint)
                binary_max = sum(not all(evaluated[:k]) for k in range(1, endpoint + 1))
                self.assertEqual(shortfall, binary_max)
                self.assertLessEqual(shortfall, float(values['MAX']))
                self.assertLessEqual(float(values['MAX']), float(values['SUM']) + 1e-6)

    def test_single_position_sum_and_max_are_identical(self):
        for origin in ([True], [False]):
            roles = make_roles(origin)
            values, details = assessment_pair(logits_for_margins([[3, -2]]), roles)
            torch.testing.assert_close(values['SUM'], values['MAX'], rtol=0, atol=0)
            self.assertEqual(details['bottleneck_multiplicity'].tolist(), [[0, 1]])

    def test_no_triangular_weight_is_applied_to_max(self):
        roles = make_roles([1, 1, 1, 1])
        values, _ = assessment_pair(logits_for_margins([[0, 0, 0, 0, 0]]), roles)
        self.assertEqual(float(values['MAX']), 4)
        self.assertEqual(float(values['SUM']), 10)

    def test_fixed_branch_multiplicity_gradient(self):
        margins = torch.tensor([[0, 2.0, 1.0, 3.0, -1.0]], dtype=torch.float64,
                               requires_grad=True)
        mask = torch.tensor([[False, True, True, True, True]])
        terms, details = max_terms_from_margins(margins, mask)
        self.assertEqual(details['bottleneck_indices'].tolist(), [[-1, 1, 2, 2, 4]])
        self.assertEqual(details['bottleneck_multiplicity'].tolist(), [[0, 1, 2, 0, 1]])
        actual = torch.autograd.grad(terms.sum(), margins)[0]
        expected = -details['bottleneck_multiplicity'] * torch.sigmoid(-margins) / math.log(2)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(int(details['bottleneck_multiplicity'].sum()), int(mask.sum()))

    def test_fp64_finite_difference_only_on_unchanged_smooth_branches(self):
        values = torch.tensor([[0, 1.2, 0.4, 0.9, -0.7]], dtype=torch.float64,
                              requires_grad=True)
        mask = torch.tensor([[False, True, True, True, True]])
        terms, original = max_terms_from_margins(values, mask)
        derivative = torch.autograd.grad(terms.sum(), values)[0]
        epsilon = 1e-6
        for position in range(values.shape[-1]):
            plus, minus = values.detach().clone(), values.detach().clone()
            plus[0, position] += epsilon
            minus[0, position] -= epsilon
            upper, upper_branch = max_terms_from_margins(plus, mask)
            lower, lower_branch = max_terms_from_margins(minus, mask)
            self.assertTrue(torch.equal(original['bottleneck_indices'],
                                        upper_branch['bottleneck_indices']))
            self.assertTrue(torch.equal(original['bottleneck_indices'],
                                        lower_branch['bottleneck_indices']))
            numeric = (upper.sum() - lower.sum()) / (2 * epsilon)
            self.assertAlmostEqual(float(numeric), float(derivative[0, position]), places=8)

    def test_equal_margins_choose_earliest_and_declared_derivative(self):
        margins = torch.tensor([[0, 2.0, 2.0, -1.0, -1.0]], dtype=torch.float64,
                               requires_grad=True)
        mask = torch.tensor([[False, True, True, True, True]])
        terms, details = max_terms_from_margins(margins, mask)
        self.assertEqual(details['bottleneck_indices'].tolist(), [[-1, 1, 1, 3, 3]])
        self.assertEqual(details['tied_bottleneck_prefixes'].tolist(),
                         [[False, False, True, False, True]])
        actual = torch.autograd.grad(terms.sum(), margins)[0]
        expected = -details['bottleneck_multiplicity'] * torch.sigmoid(-margins) / math.log(2)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_selection_uses_margin_before_softplus_rounding(self):
        margins = torch.tensor([[0.0, 150.0, 140.0, 145.0]])
        mask = torch.tensor([[False, True, True, True]])
        self.assertTrue(torch.equal(F.softplus(-margins[mask]), torch.zeros(3)))
        _, details = max_terms_from_margins(margins, mask)
        self.assertEqual(details['bottleneck_indices'].tolist(), [[-1, 1, 2, 2]])

    def test_vocabulary_ties_use_lowest_wrong_id_and_live_gradients(self):
        roles = make_roles([1, 1])
        logits = torch.tensor([[[0, 0, 0], [2, 2, 2], [3, 0, 0]]],
                              dtype=torch.bfloat16, requires_grad=True)
        value, details = legacy.assessment(logits, roles, kind='MAX')
        self.assertEqual(details['rivals'].tolist(), [1, 1])
        self.assertEqual(details['wrong_rival_ties'], 2)
        gradient = torch.autograd.grad(value, logits)[0]
        self.assertLess(float(gradient[0, 1, 0]), 0)
        self.assertGreater(float(gradient[0, 1, 1]), 0)
        self.assertTrue(torch.equal(gradient[0, :, 2], torch.zeros(3, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(gradient[0, 2], torch.zeros(3, dtype=torch.bfloat16)))

    def test_wrong_native_argmax_tie_remains_covered_by_bound(self):
        roles = make_roles([0, 1])
        roles['labels'].fill_(1)
        logits = torch.zeros((1, 3, 2), dtype=torch.bfloat16)
        values, details = assessment_pair(logits, roles)
        self.assertFalse(bool(details['correct'].any()))
        self.assertEqual(details['margin_ties'], 2)
        self.assertEqual(float(values['MAX']), 2)

    def test_gathers_native_before_margin_promotion(self):
        roles = make_roles([1])
        logits = torch.tensor([[[0, 0], [256, -1]]], dtype=torch.bfloat16)
        _, details = assessment_pair(logits, roles)
        self.assertEqual(float(details['margins'][0]), 257)

    def test_masks_eos_gap_and_empty_seed(self):
        labels = torch.tensor([[3, 1, 9, 2, 4], [3, 1, 2, 3, 4], [9, 1, 2, 3, 4]])
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[:, 0] = 0
        weights[1, 2] = 0
        roles = legacy.current_roles(labels, labels, weights, (9,))
        self.assertEqual(roles['K'].tolist(), [2, 1, 0])
        logits = torch.zeros((3, 5, 10), requires_grad=True)
        values, details = assessment_pair(logits, roles)
        self.assertEqual(float(values['MAX']), 1.5)  # divide by two nonempty blocks
        self.assertEqual(details['bottleneck_multiplicity'].sum(-1).tolist(), [2, 1, 0])
        gradient = torch.autograd.grad(values['MAX'], logits)[0]
        self.assertFalse(bool(gradient[~roles['assessment_mask']].any()))

    def test_empty_response_is_graph_connected_zero_even_with_unused_nonfinites(self):
        labels = torch.zeros((2, 4), dtype=torch.long)
        roles = legacy.current_roles(labels, labels, torch.zeros_like(labels).float())
        logits = torch.full((2, 4, 3), torch.nan, requires_grad=True)
        values, details = assessment_pair(logits, roles)
        self.assertEqual(details['nonempty_blocks'], 0)
        for kind in ('SUM', 'MAX'):
            self.assertEqual(float(values[kind].detach()), 0)
            gradient = torch.autograd.grad(values[kind], logits, retain_graph=True)[0]
            self.assertTrue(torch.equal(gradient, torch.zeros_like(logits)))
        self.assertTrue(bool((details['bottleneck_indices'] == -1).all()))
        self.assertEqual(float(legacy.window_mean((torch.tensor(4.0), values['MAX'])).detach()), 2)

    def test_nonfinite_assessed_margin_stops_before_selecting_an_invalid_branch(self):
        roles = make_roles([1])
        for margin in (torch.inf, -torch.inf, torch.nan):
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(ValueError, 'Nonfinite assessed margins'):
                    max_terms_from_margins(torch.tensor([[0., margin]]), roles['assessment_mask'])
        # Both native BF16 values are finite; their FP32 difference can overflow.
        huge = torch.finfo(torch.bfloat16).max
        logits = torch.tensor([[[0, 0], [huge, -huge]]], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'Nonfinite assessed margins'):
            assessment_pair(logits, roles)

    def test_fixed_origin_K_does_not_expand_after_repair(self):
        roles = make_roles([1, 0, 1, 0, 1])
        self.assertEqual(int(roles['K']), 3)
        logits = logits_for_margins([[0, 1, 2, 3, -100, -100]])
        value, _ = legacy.assessment(logits, roles, kind='MAX')
        changed = logits.clone()
        changed[:, 4:, 0] = 100
        updated, _ = legacy.assessment(changed, roles, kind='MAX')
        torch.testing.assert_close(value, updated, rtol=0, atol=0)

    def test_fresh_preview_recomputes_bottlenecks_without_changing_roles(self):
        roles = make_roles([1, 1, 0, 1])
        _, first = assessment_pair(logits_for_margins([[0, 2, 1, -1, 3]]), roles)
        _, second = assessment_pair(logits_for_margins([[0, -2, 1, 1, 3]]), roles)
        self.assertEqual(first['bottleneck_indices'].tolist(), [[-1, 1, 2, 3, 3]])
        self.assertEqual(second['bottleneck_indices'].tolist(), [[-1, 1, 1, 1, 1]])
        self.assertEqual(int(roles['K']), 4)

    def test_blocks_stay_independent(self):
        margins = torch.tensor([[0, -5, 2], [0, 1, 0]], dtype=torch.float64,
                               requires_grad=True)
        mask = torch.tensor([[False, True, True], [False, True, True]])
        terms, details = max_terms_from_margins(margins, mask)
        self.assertEqual(details['bottleneck_indices'].tolist(), [[-1, 1, 1], [-1, 1, 2]])
        first_gradient = torch.autograd.grad(terms[0].sum(), margins)[0]
        self.assertTrue(torch.equal(first_gradient[1], torch.zeros(3, dtype=torch.float64)))

    def test_chunk_numerators_preserve_full_response_normalization(self):
        generator = torch.Generator().manual_seed(17)
        labels = torch.zeros((11, 16), dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[:, 0] = 0
        weights[-1] = 0
        roles = legacy.current_roles(labels, labels, weights)
        logits = torch.randn((11, 16, 3), generator=generator, requires_grad=True)
        full, _ = assessment_pair(logits, roles)
        count = int(roles['valid'].any(-1).sum())
        for kind in ('SUM', 'MAX'):
            total = 0
            for start, stop in ((0, 8), (8, 11)):
                chunk = slice_roles(roles, start, stop)
                value, _ = native_assessment_numerator(
                    logits[start:stop], chunk['labels'], chunk['assessment_mask'],
                    chunk['assessment_weight'], assessment=kind,
                )
                total = total + value
            chunked = total / count
            torch.testing.assert_close(chunked, full[kind], rtol=1e-6, atol=1e-6)
            chunk_gradient = torch.autograd.grad(chunked, logits, retain_graph=True)[0]
            full_gradient = torch.autograd.grad(full[kind], logits, retain_graph=True)[0]
            # Different CPU vector lengths can round softplus derivatives at
            # the FP32 ulp. This is algebra coverage, not the native GPU gate.
            torch.testing.assert_close(chunk_gradient, full_gradient, rtol=1e-6, atol=1e-7)

    def test_default_sum_value_and_gradient_reproduce_legacy_expression(self):
        roles = make_roles([1, 1, 0, 1, 1])
        logits = logits_for_margins([[0, 3, 2, -1, 1, 0.5]]).requires_grad_()
        old, _ = legacy.assessment(logits, roles)
        values, _ = assessment_pair(logits, roles)
        torch.testing.assert_close(old, values['SUM'], rtol=0, atol=0)
        old_gradient = torch.autograd.grad(old, logits, retain_graph=True)[0]
        sum_gradient = torch.autograd.grad(values['SUM'], logits)[0]
        torch.testing.assert_close(old_gradient, sum_gradient, rtol=0, atol=0)

    def test_continuation_has_no_direct_teaching_loss(self):
        roles = make_roles([1, 1, 0, 1, 1, 0])
        loss = torch.arange(7.0).reshape(1, 7).requires_grad_()
        coefficients = roles['beta'] * 1.1
        value = legacy.response_loss(loss, coefficients, roles['beta'], roles)
        derivative = torch.autograd.grad(value, loss)[0]
        self.assertTrue(torch.equal(derivative[:, 4:], torch.zeros((1, 3))))
        self.assertTrue(bool((derivative[:, 1:4] > 0).all()))

    def test_role_multiplicity_identifies_sparse_continuation_gradient(self):
        roles = make_roles([1, 1, 0, 1, 1])
        _, details = assessment_pair(logits_for_margins([[0, 2, 1, -1, 3, 2]]), roles)
        counts = details['active_bottlenecks_by_role']
        self.assertEqual(counts['P'], {'distinct_positions': 2, 'multiplicity': 2})
        self.assertEqual(counts['E'], {'distinct_positions': 1, 'multiplicity': 3})
        self.assertEqual(counts['C'], {'distinct_positions': 0, 'multiplicity': 0})


if __name__ == '__main__':
    unittest.main()
