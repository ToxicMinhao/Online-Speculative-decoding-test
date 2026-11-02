import unittest
from experiments.full_k_gsm8k.costs import compiler_delta


class CompilerCostTests(unittest.TestCase):
    def test_nested_timers_are_kept_separate_and_old_calls_not_recounted(self):
        before = {"compile": [2.0], "inner": [1.0]}
        after = {"compile": [2.0, 3.0], "inner": [1.0, 2.5], "cache": [0.1]}
        result = compiler_delta(before, after)
        self.assertEqual(result["compile"], dict(calls=1, durations_s=[3.0], total_s=3.0))
        self.assertEqual(result["inner"]["total_s"], 2.5)
        self.assertEqual(result["cache"]["total_s"], 0.1)
        self.assertNotIn("total", result)
        self.assertEqual(compiler_delta(after, after), {})

    def test_rewritten_or_nonfinite_history_is_not_a_valid_cost(self):
        for before, after in [({"a": [1.]}, {}), ({"a": [1.]}, {"a": [2.]}),
                              ({}, {"a": [float("nan")]}), ({}, {"a": [-1.]})]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                compiler_delta(before, after)


if __name__ == "__main__":
    unittest.main()
