import copy
import json
import unittest
from unittest.mock import patch

import torch

from . import allocation_contract as C
from .math import allocate_prefix


def prefix_mask(lengths):
    positions = torch.arange(16)
    return (positions > 0) & (positions <= torch.tensor(lengths).unsqueeze(-1))


def response_fixture():
    prefix = prefix_mask([3, 2]).unsqueeze(0)
    repair = torch.zeros_like(prefix)
    repair[0, 0, 4] = True
    repair[0, 1, 3] = True
    beta = torch.exp(-torch.arange(16).float() / 7).expand_as(prefix).clone()
    scores = torch.arange(16).float().expand_as(beta).clone()
    coefficients, _ = allocate_prefix(scores, beta, prefix)
    z = (beta * (prefix | repair)).sum() + 1e-6
    roles = dict(beta=beta, prefix=prefix, repair=repair, denominator=z)
    return dict(
        scores=scores,
        coefficients=coefficients,
        roles=roles,
        repair_coefficients=beta.clone(),
        reference_repair_coefficients=beta.clone(),
        direct_loss_mask=prefix | repair,
        response_denominator=z,
    )


class AllocationContractTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(
            torch.cuda.is_initialized(), 'CPU contract test initialized CUDA'
        )

    def test_prospective_contract_literal_and_json_roundtrip(self):
        contract = C.numerical_contract()
        self.assertEqual(json.loads(json.dumps(contract, allow_nan=False)), contract)
        self.assertEqual((contract['mass_atol'], contract['mass_rtol']), (2e-7, 2e-6))
        self.assertEqual(
            (
                contract['relative_cap'],
                contract['gate_b_relative_beta'],
                contract['gate_c_alignment_fraction'],
            ),
            (0.25, 0.025, 0.9),
        )
        self.assertEqual(
            contract['historical_hidden_state_gate'],
            dict(atol=1e-5, rtol=0.016, status='failed', production_veto=False),
        )
        self.assertEqual(
            contract['maximum_reference_backwards_including_one_correction'], 120
        )
        contract['relative_cap'] = 0.8
        self.assertEqual(C.numerical_contract()['relative_cap'], 0.25)

    def test_full_pool_selection_lowest_then_longest_distinct_and_ties(self):
        selected = C.select_pools(
            [prefix_mask([0, 2, 5, 5, 1]), prefix_mask([15, 3, 15]).unsqueeze(0)],
            ['first', 'second'],
        )
        self.assertEqual(
            [(x['response_index'], x['block_index']) for x in selected['pools']],
            [(0, 1), (0, 2), (1, 0), (1, 2)],
        )
        self.assertEqual(selected['reference_backward_count'], 37)
        self.assertEqual(selected['pools'][3]['positions'], list(range(1, 16)))
        self.assertTrue(selected['passed'])
        self.assertEqual(
            selected,
            C.select_pools(
                [prefix_mask([0, 2, 5, 5, 1]), prefix_mask([15, 3, 15]).unsqueeze(0)],
                ['first', 'second'],
            ),
        )

    def test_coverage_blocker_keeps_available_pools_and_never_substitutes(self):
        result = C.select_pools([prefix_mask([1, 0]), prefix_mask([3])])
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'coverage_blocked')
        self.assertEqual(result['reference_backward_count'], 3)
        self.assertEqual(len(result['pools']), 1)
        maximum = C.select_pools([prefix_mask([15, 15]), prefix_mask([15, 15])])
        self.assertEqual(maximum['reference_backward_count'], 60)

    def test_selection_rejects_anchor_gaps_nonboolean_and_wrong_window(self):
        bad = prefix_mask([3])
        bad[0, 0] = True
        with self.assertRaises(ValueError):
            C.select_pools([bad, prefix_mask([2])])
        bad = prefix_mask([3])
        bad[0, 2] = False
        with self.assertRaises(ValueError):
            C.select_pools([bad, prefix_mask([2])])
        with self.assertRaises(ValueError):
            C.select_pools([prefix_mask([2]).float(), prefix_mask([2])])
        with self.assertRaises(ValueError):
            C.select_pools([prefix_mask([2])])

    def test_gate_a_native_mass_support_repair_and_normalization(self):
        original = response_fixture()
        self.assertTrue(C.validate_native_invariants(**original)['passed'])
        cases = []
        changed = copy.deepcopy(original)
        changed['repair_coefficients'][changed['roles']['repair']] *= 1.01
        cases.append((changed, 'repair_unchanged'))
        changed = copy.deepcopy(original)
        changed['direct_loss_mask'][0, 0, 6] = True
        cases.append((changed, 'direct_support_exact'))
        changed = copy.deepcopy(original)
        changed['response_denominator'] *= 2
        cases.append((changed, 'native_response_denominator'))
        changed = copy.deepcopy(original)
        changed['accumulation_factor'] = 1
        cases.append((changed, 'accumulation_once'))
        changed = copy.deepcopy(original)
        changed['coefficients'][0, 0, 1] *= 1.2
        cases.append((changed, 'block_mass'))
        changed = copy.deepcopy(original)
        changed['scores'][0, 0, 1] = torch.nan
        cases.append((changed, 'finite'))
        changed = copy.deepcopy(original)
        changed['coefficients'][0, 0, 6] += 0.01
        cases.append((changed, 'nonprefix_preservation_unchanged'))
        for data, failure in cases:
            result = C.validate_native_invariants(**data)
            self.assertFalse(result['passed'])
            self.assertIn(failure, result['failures'])
        for key in ['response_denominator', 'accumulation_factor']:
            data = copy.deepcopy(original)
            data[key] = True
            with self.assertRaises(ValueError):
                C.validate_native_invariants(**data)
        data = copy.deepcopy(original)
        data['scores'] = data['scores'].bool()
        with self.assertRaises(ValueError):
            C.validate_native_invariants(**data)
        data = copy.deepcopy(original)
        data['coefficients'] = data['coefficients'].bfloat16()
        self.assertFalse(
            C.validate_native_invariants(**data)['checks']['allocator_fp32']
        )

    def test_native_chunk_denominator_is_exact_binding_with_roundoff_diagnostic(self):
        data = response_fixture()
        native_z = torch.nextafter(
            data['roles']['denominator'], torch.tensor(float('inf'))
        )
        data['roles']['denominator'] = native_z
        data['response_denominator'] = native_z.clone()
        result = C.validate_native_invariants(**data)
        self.assertTrue(result['passed'])
        self.assertGreater(result['native_vs_full_sum_difference'], 0)
        data['response_denominator'] = torch.nextafter(
            native_z, torch.tensor(float('inf'))
        )
        self.assertFalse(
            C.validate_native_invariants(**data)['checks'][
                'native_response_denominator'
            ]
        )

    def test_matching_active_scores_pass_and_offset_arithmetic_is_stable(self):
        beta = torch.tensor([1.0, 0.8, 0.6])
        ref = torch.tensor([0.0, 0.125, 0.25], dtype=torch.float64)
        base = C.evaluate_pool(beta, ref, ref.float(), ref.float())
        shifted = C.evaluate_pool(
            beta, ref + 1e6, (ref + 1e6).float(), (ref + 1e6).float()
        )
        self.assertTrue(base['passed'])
        self.assertTrue(base['nonconstant_reference_and_both_fast_active'])
        self.assertEqual(
            base['reference_coefficients_fp32'], shifted['reference_coefficients_fp32']
        )
        self.assertEqual(base['gate_c'], shifted['gate_c'])
        self.assertLess(base['independent_fp64_maximum_coefficient_difference'], 2e-7)

    def test_gate_b_requires_both_fast_passes_and_mutual_fidelity(self):
        beta = torch.ones(3)
        reference = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        with patch.object(
            C,
            'allocate_prefix',
            side_effect=[
                (torch.tensor([0.75, 1.0, 1.25]), {}),
                (torch.tensor([0.77, 0.98, 1.25]), {}),
                (torch.tensor([0.73, 1.02, 1.25]), {}),
            ],
        ):
            result = C.evaluate_pool(
                beta, reference, reference.float(), reference.float()
            )
        self.assertEqual(result['gate_b']['fast_vs_reference'], [True, True])
        self.assertFalse(result['gate_b']['fast_repeat_vs_first'])
        self.assertFalse(result['passed'])
        result = C.evaluate_pool(beta, reference, reference.float(), -reference.float())
        self.assertEqual(result['gate_b']['fast_vs_reference'], [True, False])

    def test_gate_c_is_separate_from_small_coefficient_error(self):
        beta = torch.ones(10)
        reference = torch.tensor(
            [-1.0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=torch.float64
        )
        coefficient_error = torch.tensor(
            [0.0, 0.025, 0.025, 0.025, 0.025, -0.025, -0.025, -0.025, 0.0, -0.025]
        )
        # These actual signed allocations conserve mass and retain one cap
        # endpoint. Every coefficient is faithful under B, but their combined
        # local alignment retains only 88.75%, so independent C must reject.
        fast = 0.25 * reference.float() + coefficient_error
        result = C.evaluate_pool(beta, reference, fast, fast)
        self.assertTrue(result['gate_b']['passed'])
        self.assertFalse(result['gate_c']['passed'])
        self.assertAlmostEqual(result['gate_c']['reference_alignment'], 1.0)
        self.assertAlmostEqual(result['gate_c']['fast_alignment'][0], 0.8875, places=6)
        self.assertAlmostEqual(result['gate_c']['tolerance'], 13.2e-6)

    def test_constant_reference_no_nearly_constant_threshold_or_inactive_pass(self):
        beta = torch.ones(2)
        constant = torch.ones(2, dtype=torch.float64)
        result = C.evaluate_pool(beta, constant, constant.float(), constant.float())
        self.assertTrue(result['passed'])
        self.assertTrue(result['gate_c']['reference_constant'])
        self.assertFalse(result['nonconstant_reference_and_both_fast_active'])
        result = C.evaluate_pool(
            beta, constant, torch.tensor([1.0, 2.0]), constant.float()
        )
        self.assertFalse(result['gate_c']['passed'])
        tiny = torch.tensor([0.0, 1e-30], dtype=torch.float64)
        result = C.evaluate_pool(beta, tiny, tiny.float(), tiny.float())
        self.assertFalse(result['gate_c']['reference_constant'])
        self.assertTrue(result['nonconstant_reference_and_both_fast_active'])
        rounded = torch.tensor([1.0, 1.0 + 1e-12], dtype=torch.float64)
        result = C.evaluate_pool(beta, rounded, rounded.float(), rounded.float())
        self.assertTrue(result['raw_nonconstant_became_fp32_constant'])
        self.assertGreater(result['raw_fp64_score_contrast'], 0)
        self.assertEqual(result['fp32_score_contrast'], 0)

    def test_all_selected_pools_and_both_response_repeat_invariants_required(self):
        selection = C.select_pools([prefix_mask([2]), prefix_mask([3])])
        receipts = []
        for pool in selection['pools']:
            scores = torch.arange(len(pool['positions'])).float()
            receipts.append(
                dict(
                    C.evaluate_pool(
                        torch.ones_like(scores), scores.double(), scores, scores
                    ),
                    **pool
                )
            )
        invariants = [[dict(passed=True), dict(passed=True)] for _ in range(2)]
        result = C.qualify_selected_pools(selection, receipts, invariants)
        self.assertTrue(result['passed'])
        self.assertFalse(result['production_qualified'])
        invariants[1][1]['passed'] = False
        self.assertFalse(
            C.qualify_selected_pools(selection, receipts, invariants)['passed']
        )
        with self.assertRaises(ValueError):
            C.qualify_selected_pools(selection, receipts[:1], invariants)
        with self.assertRaises(ValueError):
            C.qualify_selected_pools(selection, receipts[::-1], invariants)
        for row in receipts:
            row['nonconstant_reference_and_both_fast_active'] = False
        self.assertIn(
            'active_nonconstant_pool',
            C.qualify_selected_pools(
                selection, receipts, [[dict(passed=True)] * 2] * 2
            )['failures'],
        )

    def test_nonfinite_and_boolean_scores_fail_and_direction_comparisons_keep_target(
        self,
    ):
        beta = torch.ones(2)
        scores = torch.tensor([1.0, 2.0])
        for bad in [
            torch.tensor([True, False]),
            torch.tensor([float('nan'), 1.0]),
            torch.tensor([float('inf'), 1.0]),
        ]:
            with self.assertRaises(ValueError):
                C.evaluate_pool(beta, bad, scores, scores)
        result = C.direction_rounding_diagnostics(
            scores.double(),
            scores.double() + 1,
            scores + 3,
            metadata=dict(
                primary_dtype='torch.float32',
                effective_dtype='torch.bfloat16',
                scale=1.0,
            ),
        )
        self.assertEqual(result['direction_cast_score_difference'], [1.0, 1.0])
        self.assertEqual(result['derivative_route_score_difference'], [2.0, 2.0])
        self.assertEqual(result['total_score_difference'], [3.0, 3.0])
        self.assertFalse(result['admission_gate'])
        self.assertFalse(result['comparison_is_pure_derivative_route'])
        coincident = C.direction_rounding_diagnostics(
            scores, scores, scores, metadata=dict(effective_dtype='torch.bfloat16')
        )
        self.assertFalse(coincident['comparison_is_pure_derivative_route'])


if __name__ == '__main__':
    unittest.main()
