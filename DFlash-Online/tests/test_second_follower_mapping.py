"""Mapping normalization must not weaken exact tensor identity."""

from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
import torch
from experiments.second_follower.qualification import compare_weight_mappings


class MappingCheckTests(unittest.TestCase):
    def test_ordered_native_state_compares_actual_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
            result = compare_weight_mappings(
                Path(d) / "check.json", OrderedDict(weight=x), {"weight": x.clone()}
            )
            self.assertTrue(result["exact_equal"])
            self.assertEqual(result["compared_tensors"], 1)
            self.assertEqual(result["compared_coordinates"], 2)

    def test_changed_weight_and_dtype_still_fail(self):
        with tempfile.TemporaryDirectory() as d:
            x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
            for y in (torch.tensor([1.0, 3.0], dtype=torch.bfloat16), x.float()):
                with self.assertRaises(AssertionError):
                    compare_weight_mappings(
                        Path(d) / "check.json", OrderedDict(weight=x), {"weight": y}
                    )


if __name__ == "__main__":
    unittest.main()
