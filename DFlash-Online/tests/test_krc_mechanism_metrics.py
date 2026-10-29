"""CPU accounting checks for native-prefix mechanism measurements."""
from copy import deepcopy
import json
import math
import unittest

import torch

from online_feedback.krc_mechanism_metrics import (
    fitted_objective_predictions, gradient_alignment, pooled_summary, response_metrics,
)


def captured(original, weights, *, labels=None, validity=None, keep=None):
    original = torch.tensor([original])
    count, width = original.shape[1:]
    if labels is None:
        labels = torch.arange(width).repeat(1, count, 1)
    else:
        labels = torch.tensor([labels])
    if validity is None:
        validity = torch.ones_like(labels).float()
        validity[..., 0] = 0
    else:
        validity = torch.tensor([validity], dtype=torch.float32)
    return {
        'record_id': 'fixture-response', 'labels': labels,
        'original_proposals': original, 'protected_weights': torch.tensor([weights]).float(),
        'validity_weights': validity, 'anchors': torch.arange(count).unsqueeze(0) * 7,
        'keep': torch.tensor([keep if keep is not None else [True] * count]),
    }


def evaluate(response, ordinary, candidate=None, *, ties=(), candidate_ties=(), **kwargs):
    ordinary = torch.tensor([ordinary])
    candidate = ordinary.clone() if candidate is None else torch.tensor([candidate])
    margins = torch.where(ordinary == response['labels'], 1., -1.)
    candidate_margins = torch.where(candidate == response['labels'], 1., -1.)
    for block, offset in ties:
        margins[0, block, offset] = 0
    for block, offset in candidate_ties:
        candidate_margins[0, block, offset] = 0
    return response_metrics(response, ordinary, candidate, margins, candidate_margins, **kwargs)


class PrefixMetricsTests(unittest.TestCase):
    def test_first_oracle_repairs_only_one_error_and_stratifies_native_tie(self):
        response = captured([[0, 1, 2, 3, 4, 9]], [[0, 4, 3, 2, 1, 0]])
        result = evaluate(response, [[0, 9, 2, 9, 4, 9]], ties=((0, 1),))
        block = result['blocks'][0]
        self.assertEqual(block['oracle_first_protected_error_offset'], 1)
        self.assertEqual(block['oracle_first_prefix_gain'], 2)
        self.assertEqual(block['oracle_all_prefix_gain'], 4)
        oracle = result['oracles']['FIRST_PROTECTED_ERROR']
        self.assertEqual(oracle['wrong_native_tie_prefix_gain_sum'], 2)
        self.assertEqual(oracle['strict_negative_margin_prefix_gain_sum'], 0)
        self.assertEqual(result['oracles']['all_minus_first_prefix_gain_sum'], 2)
        self.assertEqual(result['summary']['K_wrong_native_tie_errors'], 1)
        self.assertEqual(result['summary']['K_strict_negative_margin_errors'], 1)
        self.assertEqual(result['summary']['K_hinge_numerator_fp64'], 2)

    def test_strict_blocking_error_and_zero_opportunity_blocks_are_retained(self):
        response = captured([[0, 1, 2], [0, 1, 2]], [[0, 2, 1], [0, 2, 1]])
        result = evaluate(response, [[0, 9, 2], [0, 1, 2]])
        first = result['oracles']['FIRST_PROTECTED_ERROR']
        self.assertEqual(first['strict_negative_margin_blocking_errors'], 1)
        self.assertEqual(first['strict_negative_margin_prefix_gain_sum'], 2)
        self.assertEqual(first['zero_opportunity_blocks'], 1)
        self.assertEqual(first['valid_blocks'], 2)
        self.assertEqual(first['prefix_gain_mean'], 1)
        self.assertIsNone(result['blocks'][1]['oracle_first_protected_error_offset'])

    def test_binary_repair_retention_does_not_hide_lost_longer_K_progress(self):
        response = captured([[0, 1, 9, 3, 4]], [[0, 1, 0, 0, 0]])
        result = evaluate(response, [[0, 1, 2, 3, 4]], [[0, 1, 2, 9, 4]])
        summary = result['summary']
        self.assertEqual(summary['useful_repairs_retained'], 1)
        self.assertEqual(summary['K_new_prefix_gain'], 3)
        self.assertEqual(summary['K_prefix_gain_retained'], 1)
        self.assertEqual(summary['K_prefix_gain_lost'], 2)
        self.assertEqual(summary['old_prefix_tokens_lost'], 0)
        self.assertEqual(summary['gross_prefix_losses'], 2)
        block = result['blocks'][0]
        self.assertEqual(block['original_first_error_offset'], 2)
        self.assertEqual(block['original_first_error_canonical_token'], 2)

    def test_first_error_repairs_are_matched_by_identity_not_total(self):
        response = captured([[0, 1, 9, 3], [0, 1, 9, 3]],
                            [[0, 1, 0, 0], [0, 1, 0, 0]])
        result = evaluate(response, [[0, 1, 2, 3], [0, 1, 9, 3]],
                          [[0, 1, 9, 3], [0, 1, 2, 3]])
        summary = result['summary']
        self.assertEqual(summary['prefix_change_sum'], 0)
        self.assertEqual(summary['gross_prefix_gains'], 2)
        self.assertEqual(summary['gross_prefix_losses'], 2)
        self.assertEqual(summary['useful_repairs_retained'], 0)
        self.assertEqual(summary['useful_repairs_lost'], 1)
        self.assertEqual(summary['useful_repairs_newly_obtained'], 1)

    def test_new_protected_errors_and_old_repairs_are_both_counted(self):
        response = captured([[0, 1, 2, 3]], [[0, 3, 2, 1]])
        result = evaluate(response, [[0, 9, 2, 3]], [[0, 1, 9, 3]],
                          candidate_ties=((0, 2),))
        summary = result['summary']
        self.assertEqual(summary['protected_errors_repaired'], 1)
        self.assertEqual(summary['new_protected_errors'], 1)
        self.assertEqual(summary['candidate_wrong_native_tie_errors'], 1)
        self.assertEqual(summary['candidate_strict_negative_margin_errors'], 0)
        self.assertEqual(summary['old_prefix_tokens_lost'], 2)
        self.assertEqual(summary['prefix_change_sum'], 1)

    def test_EOS_gaps_and_empty_anchors_use_the_qualified_valid_block_set(self):
        response = captured(
            [[0, 1, 2, 9], [0, 1, 9, 3], [0, 1, 2, 3]],
            [[0, 2, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            validity=[[0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
            keep=[True, True, False],
        )
        result = evaluate(response,
                          [[0, 9, 2, 9], [0, 1, 9, 3], [0, 9, 9, 9]],
                          [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                          eos_token_ids=(2,))
        self.assertEqual(len(result['blocks']), 3)
        self.assertEqual(result['summary']['valid_blocks'], 2)
        self.assertEqual(result['summary']['candidate_prefix_tokens'], 3)
        self.assertEqual(result['summary']['prefix_change_sum'], 2)
        self.assertEqual(result['summary']['prefix_change_mean'], 1)
        self.assertEqual(result['summary']['tied_blocks'], 1)
        self.assertEqual(result['oracles']['FIRST_PROTECTED_ERROR']['zero_opportunity_blocks'], 1)

    def test_pair_hinge_uses_pooled_weights_not_response_mean(self):
        first = captured([[0, 1, 2]], [[0, 2, 1]])
        second = captured([[0, 1, 9]], [[0, 1, 0]])
        a = evaluate(first, [[0, 9, 9]])
        b = evaluate(second, [[0, 1, 9]])
        pooled = pooled_summary([a, b])
        self.assertEqual(pooled['summary']['protected_weight'], 4)
        self.assertEqual(pooled['summary']['K_hinge_numerator_fp64'], 3)
        self.assertEqual(pooled['summary']['K_hinge_fp64'], .75)
        self.assertNotEqual(pooled['summary']['K_hinge_fp64'], .5)

    def test_empty_protection_and_empty_valid_blocks_have_explicit_null_ratios(self):
        response = captured([[0, 9, 9]], [[0, 0, 0]], validity=[[0, 0, 0]], keep=[False])
        result = evaluate(response, [[0, 9, 9]])
        summary = result['summary']
        self.assertIsNone(summary['prefix_change_mean'])
        self.assertEqual(summary['prefix_change_mean_null_reason'], 'no_valid_blocks')
        self.assertIsNone(summary['candidate_hinge_fp64'])
        self.assertEqual(summary['candidate_hinge_null_reason'], 'empty_protection')
        self.assertIsNone(result['oracles']['ALL_PROTECTED_ERRORS']['prefix_gain_mean'])
        json.dumps(result, allow_nan=False)

    def test_metrics_preserve_input_arrays_and_identity(self):
        response = captured([[0, 1, 9]], [[0, 1, 0]])
        before = deepcopy(response)
        identity = {'event': 4, 'pass': 2, 'local_step': 128, 'global_step': 512,
                    'role': 'D', 'candidate': 'R_HALF'}
        result = evaluate(response, [[0, 1, 2]], identity=identity)
        for key in identity:
            self.assertEqual(result['blocks'][0][key], identity[key])
        for key, value in response.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, before[key]))
        json.dumps(result, allow_nan=False)

    def test_mismatched_masks_native_decisions_and_shapes_fail_closed(self):
        response = captured([[0, 1, 2]], [[0, 2, 0]])
        with self.assertRaisesRegex(ValueError, 'membership'):
            evaluate(response, [[0, 1, 2]])
        response['protected_weights'][0, 0, 2] = 1
        with self.assertRaisesRegex(ValueError, 'share the labels shape'):
            response_metrics(response, torch.tensor([[[0, 1]]]), response['labels'],
                             torch.ones_like(response['labels']), torch.ones_like(response['labels']))
        predictions = response['labels'].clone()
        predictions[..., 1] = 9
        with self.assertRaisesRegex(ValueError, 'margins disagree'):
            response_metrics(response, predictions, predictions,
                             torch.ones_like(predictions), torch.ones_like(predictions))


class PredictionTests(unittest.TestCase):
    def test_predictions_keep_rounding_and_actual_residuals_distinct(self):
        result = fitted_objective_predictions(2, .5, 'applied', -1, -.75, 1.5)
        self.assertEqual(result['ideal'], 1)
        self.assertEqual(result['proposed_linearized_V'], 1)
        self.assertEqual(result['realized_linearized_V'], 1.25)
        self.assertEqual(result['realized_minus_proposed'], .25)
        self.assertEqual(result['actual_minus_realized_linearized'], .25)

    def test_positive_violation_zero_gradient_does_not_predict_ideal_zero(self):
        result = fitted_objective_predictions(2, 1, 'unresolved_zero_gradient', 0, 0, 2)
        self.assertEqual(result['formal_ideal'], 0)
        self.assertIsNone(result['ideal'])
        self.assertEqual(result['realized_linearized_V'], 2)
        self.assertIn('unresolved_zero_gradient', result['ideal_null_reason'])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            fitted_objective_predictions(math.inf, 1, 'applied', 0, 0, 0)

    def test_gradient_alignment_uses_existing_gradients_without_mutation(self):
        reference = [torch.tensor([1., 2.]), torch.tensor([3.])]
        diagnostic = [torch.tensor([-2., 1.]), torch.tensor([0.])]
        before = [tensor.clone() for tensor in reference + diagnostic]
        result = gradient_alignment(reference, diagnostic)
        self.assertEqual(result['g_R_dot_g_D'], 0)
        self.assertEqual(result['cosine'], 0)
        self.assertAlmostEqual(result['g_R_norm_fp64'], math.sqrt(14))
        for actual, saved in zip(reference + diagnostic, before):
            self.assertTrue(torch.equal(actual, saved))
        empty = gradient_alignment([torch.zeros(2)], [torch.ones(2)])
        self.assertIsNone(empty['cosine'])
        self.assertEqual(empty['cosine_null_reason'], 'zero_gradient_norm')
        with self.assertRaisesRegex(ValueError, 'lengths'):
            gradient_alignment(reference, [])


if __name__ == '__main__':
    unittest.main()
