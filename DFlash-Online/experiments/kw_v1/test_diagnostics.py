import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from .diagnostics import compare_exact, tensor_difference


class ExactDiagnosticsTests(unittest.TestCase):
    def test_failure_is_written_before_raising_without_path_truncation(self):
        left = {str(i): torch.tensor([1, 2], dtype=torch.bfloat16) for i in range(105)}
        right = {name: value + 1 for name, value in left.items()}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checks.json'
            with self.assertRaisesRegex(ValueError, 'boundary'):
                compare_exact(left, right, 'boundary', [], receipt_path=path)
            row = json.loads(path.read_text())[0]
        self.assertEqual(row['differing_path_count'], 105)
        self.assertEqual(len(row['tensor_differences']), 105)
        self.assertEqual(row['tensor_differences']['state.0']['differing_coordinates'], 2)
        self.assertEqual(row['tensor_differences']['state.0']['max_abs_error_finite_pairs'], 1)

    def test_signed_zero_and_nonfinite_values_are_explicit(self):
        left = torch.tensor([0., float('nan'), float('inf'), 4.])
        right = torch.tensor([-0., 1., float('-inf'), 3.])
        row = tensor_difference(left, right)
        self.assertEqual(row['differing_coordinates'], 4)
        self.assertEqual((row['nonfinite_left'], row['nonfinite_right']), (2, 1))
        self.assertEqual(row['max_abs_error_finite_pairs'], 1)
        self.assertEqual(row['samples'][1]['left'], 'nan')
        json.dumps(row, allow_nan=False)

    def test_exact_success_and_structural_failures_keep_the_original_gate(self):
        checks = []
        compare_exact({'x': torch.tensor([1.])}, {'x': torch.tensor([1.])}, 'same', checks)
        self.assertEqual(checks[0]['status'], 'passed')
        with self.assertRaises(ValueError):
            compare_exact({'x': None}, {'x': torch.tensor([0.])}, 'active set', checks)
        self.assertEqual(checks[-1]['differences'], ['state.x'])
        with self.assertRaises(ValueError):
            compare_exact(torch.tensor([1.]), torch.tensor([1., 1.]), 'shape', checks)
        self.assertFalse(checks[-1]['tensor_differences']['state']['comparable'])

    def test_finite_fp64_norm_overflow_still_retains_failure_evidence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checks.json'
            with self.assertRaisesRegex(ValueError, 'overflow'):
                compare_exact(torch.tensor([1e308], dtype=torch.float64),
                              torch.zeros(1, dtype=torch.float64), 'overflow', [], receipt_path=path)
            row = json.loads(path.read_text())[0]['tensor_differences']['state']
        self.assertEqual(row['nonfinite_left'], 0)
        self.assertEqual(row['left_l2_finite_coordinates'], 'inf')


if __name__ == '__main__':
    unittest.main()
