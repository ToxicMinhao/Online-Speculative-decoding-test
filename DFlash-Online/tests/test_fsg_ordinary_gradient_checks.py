"""CPU evidence checks; no model, sampling, optimizer or GPU execution."""
import json
import math
import unittest
from unittest.mock import patch

import torch

from experiments.functional_step_geometry import ordinary_gradient_checks as checks


class OrdinaryGradientChecksTests(unittest.TestCase):
    def test_exact_values_none_and_empty_sets(self):
        left = {'weight': torch.tensor([1., -2.], dtype=torch.bfloat16), 'unused': None}
        result = checks.compare_gradients(left, left)
        self.assertTrue(result['exact_equal'])
        self.assertEqual(result['global']['compared_coordinates'], 2)
        self.assertEqual(result['global']['left_l2'], math.sqrt(5))
        self.assertEqual(result['global']['relative_l2'], 0)
        self.assertTrue(checks.compare_gradients({}, {})['exact_equal'])

    def test_norms_include_equal_coordinates_and_reference_orientation(self):
        result = checks.compare_gradients({'w': torch.tensor([3., 4., 2.])},
                                         {'w': torch.tensor([3., 4., 0.])})
        total = result['global']
        self.assertFalse(result['exact_equal'])
        self.assertEqual(total['differing_coordinates'], 1)
        self.assertAlmostEqual(total['left_l2'], math.sqrt(29))
        self.assertEqual(total['right_l2'], 5)
        self.assertEqual(total['discrepancy_l2'], 2)
        self.assertEqual(total['relative_l2'], .4)
        self.assertEqual(total['maximum_absolute_error'], 2)

    def test_zero_reference_norm_is_explicit(self):
        result = checks.compare_gradients({'w': torch.ones(1)}, {'w': torch.zeros(1)})
        self.assertEqual(result['global']['relative_l2'], 'Infinity')
        json.dumps(result, allow_nan=False)

    def test_shape_dtype_and_membership_mismatches_fail(self):
        left = {'shape': torch.ones(2), 'dtype': torch.ones(1), 'missing': None}
        right = {'shape': torch.ones(3), 'dtype': torch.ones(1, dtype=torch.float64)}
        result = checks.compare_gradients(left, right)
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['global']['metadata_mismatches'], 3)
        self.assertEqual(result['global']['discrepancy_l2'], 'undefined')
        self.assertEqual(result['global']['compared_coordinates'], 1)
        self.assertEqual(result['global']['left']['coordinates'], 3)
        self.assertEqual(result['global']['right']['coordinates'], 4)

    def test_none_is_not_a_zero_gradient(self):
        result = checks.compare_gradients({'w': None}, {'w': torch.zeros(2)})
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['global']['compared_coordinates'], 0)

    def test_nonfinite_values_preserved_even_if_equal(self):
        left = torch.tensor([float('nan'), float('inf'), -float('inf'), 3., 4.])
        right = left.clone()
        result = checks.compare_gradients({'w': left}, {'w': right})
        self.assertFalse(result['exact_equal'])
        total = result['global']
        self.assertEqual(total['nonfinite_coordinates'], 3)
        self.assertEqual(total['differing_coordinates'], 3)
        self.assertEqual(total['left']['finite_l2'], 5)
        self.assertEqual(total['left']['nan_coordinates'], 1)
        self.assertEqual(total['left_l2'], 'NaN')
        self.assertEqual(total['discrepancy_l2'], 'undefined')
        self.assertEqual(total['samples'][0]['left'], 'NaN')
        self.assertEqual(total['samples'][1]['left'], 'Infinity')
        json.dumps(result, allow_nan=False)

    def test_float64_subtraction_overflow_remains_explicit(self):
        result = checks.compare_gradients({'w': torch.tensor([1e308], dtype=torch.float64)},
                                         {'w': torch.tensor([-1e308], dtype=torch.float64)})
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['global']['nonfinite_difference_coordinates'], 1)
        self.assertEqual(result['global']['discrepancy_l2'], 'Infinity')
        json.dumps(result, allow_nan=False)

    def test_unpaired_nonfinite_values_reach_global_failure_counter(self):
        cases = [
            ({'w': torch.tensor([float('nan')])}, {}),
            ({'w': None}, {'w': torch.tensor([float('inf')])}),
            ({'w': torch.tensor([float('nan')])},
             {'w': torch.tensor([float('inf'), -float('inf')])}),
        ]
        for (left, right), expected_count in zip(cases, (1, 1, 3)):
            with self.subTest(left=left, right=right):
                result = checks.compare_gradients(left, right)
                self.assertFalse(result['exact_equal'])
                self.assertEqual(result['global']['compared_coordinates'], 0)
                self.assertEqual(result['global']['nonfinite_coordinates'], expected_count)
                self.assertEqual(result['parameters'][0]['nonfinite_coordinates'], expected_count)
                json.dumps(result, allow_nan=False)

    def test_samples_are_bounded_sorted_and_order_independent(self):
        a, b = torch.ones(20), torch.zeros(20)
        left, right = {'z': a, 'a': a}, {'a': b, 'z': b}
        result = checks.compare_gradients(left, right)
        reverse = checks.compare_gradients(dict(reversed(list(left.items()))),
                                           dict(reversed(list(right.items()))))
        self.assertEqual(result, reverse)
        self.assertEqual(len(result['global']['samples']), 8)
        self.assertTrue(all(s['parameter'] == 'a' for s in result['global']['samples']))
        self.assertEqual([s['index'] for s in result['parameters'][0]['samples']], list(range(8)))
        self.assertLess(len(json.dumps(result)), 10000)

    def test_chunking_bounds_all_promoted_vectors_and_counts_boundaries(self):
        left, right = torch.arange(23, dtype=torch.float32), torch.arange(23, dtype=torch.float32)
        right[[6, 7, 14, 22]] += 1
        sizes = []
        original = checks._add_values

        def inspect(total, values):
            sizes.append(values.numel())
            self.assertEqual(values.dtype, torch.float64)
            return original(total, values)

        with patch.object(checks, 'CHUNK_COORDINATES', 7), patch.object(checks, '_add_values', inspect):
            result = checks.compare_gradients({'w': left}, {'w': right})
        self.assertLessEqual(max(sizes), 7)
        self.assertEqual(result['global']['differing_coordinates'], 4)
        self.assertEqual(result['global']['discrepancy_l2'], 2)
        self.assertEqual([s['index'] for s in result['global']['samples']], [6, 7, 14, 22])
        self.assertLessEqual(checks.CHUNK_COORDINATES, 1_000_000)

    def test_no_mutation_autograd_or_rng_change(self):
        left = torch.arange(12., requires_grad=True).reshape(3, 4).transpose(0, 1)
        right = left.detach().clone()
        before_left, before_right = left.detach().clone(), right.clone()
        rng = torch.random.get_rng_state().clone()
        version = left._version
        result = checks.compare_gradients({'w': left}, {'w': right})
        self.assertTrue(result['exact_equal'])
        self.assertTrue(torch.equal(left, before_left))
        self.assertTrue(torch.equal(right, before_right))
        self.assertEqual(left._version, version)
        self.assertTrue(left.requires_grad)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_global_norms_pool_parameters_before_ratio(self):
        result = checks.compare_gradients({'a': torch.tensor([3.]), 'b': torch.tensor([4.])},
                                         {'a': torch.tensor([0.]), 'b': torch.tensor([4.])})
        self.assertEqual(result['global']['left_l2'], 5)
        self.assertEqual(result['global']['right_l2'], 4)
        self.assertEqual(result['global']['discrepancy_l2'], 3)
        self.assertEqual(result['global']['relative_l2'], .75)


if __name__ == '__main__':
    unittest.main()
