import unittest
import numpy as np
from experiments.longer_run_headroom.analysis import summarize, interval, FIELDS


def pair(rid, group, **changes):
    row = dict(
        dataset="fixture",
        request_id=rid,
        group=group,
        j=1,
        v=14,
        status="valid",
        **{k: 0 for k in FIELDS}
    )
    row.update(changes)
    return row


class AnalysisTests(unittest.TestCase):
    def test_census_share_and_sample_ratio_have_different_denominators(self):
        census = [
            dict(dataset="fixture", request_id=r, group=g, j=1, v=14, opening=False)
            for r in ["a", "b"]
            for g in ["LONG", "SHORT", "R0", "R0"]
        ]
        paired = [
            pair("a", "LONG", blocked_with_both=1),
            pair("b", "LONG", repair=1, repair_with_two_accepted=1),
            pair("a", "SHORT", native_prefix_change=-2, prefix_lost=2),
            pair("b", "SHORT", native_prefix_change=3, prefix_gained=3),
        ]
        x = summarize(census, paired, draws=200)["datasets"]["fixture"]
        self.assertEqual(x["A"]["estimate"], 0.125)
        self.assertEqual(x["B"]["estimate"], 1)
        self.assertEqual(x["B"]["denominator"], 1)
        self.assertGreater(x["B"]["interval"]["undefined_draws"], 0)
        self.assertEqual(x["groups"]["SHORT"]["native_prefix_change"]["total"], 1)
        self.assertEqual(
            x["standardized"]["estimates"]["native_prefix_change"]["SHORT"], 0.5
        )

    def test_no_repairs_remain_undefined_not_zero(self):
        census = [
            dict(
                dataset="fixture",
                request_id="a",
                group="LONG",
                j=1,
                v=14,
                opening=False,
            )
        ]
        x = summarize(census, [pair("a", "LONG", blocked_with_both=1)], draws=10)[
            "datasets"
        ]["fixture"]
        self.assertIsNone(x["B"]["estimate"])
        self.assertEqual(x["B"]["interval"]["undefined_draws"], 10)
        self.assertEqual(x["unconditional_realized"]["fraction"], 0)
        self.assertTrue(x["unconditional_realized"]["interval"]["degenerate"])

    def test_invalid_pairs_excluded_but_not_removed(self):
        census = [
            dict(
                dataset="fixture",
                request_id="a",
                group="LONG",
                j=1,
                v=14,
                opening=False,
            )
        ]
        x = summarize(census, [pair("a", "LONG", status="invalid")], draws=10)[
            "datasets"
        ]["fixture"]
        self.assertEqual(x["validity"], {"invalid": 1})
        self.assertIsNone(x["A"]["estimate"])

    def test_same_seed_reproduces(self):
        census = [
            dict(
                dataset="fixture", request_id=r, group="LONG", j=1, v=14, opening=False
            )
            for r in ["a", "b"]
        ]
        rows = [pair("a", "LONG", repair=1), pair("b", "LONG", blocked_with_both=1)]
        self.assertEqual(
            summarize(census, rows, draws=50), summarize(census, rows, draws=50)
        )

    def test_nonfinite_intervals_explicit(self):
        x = interval([0, np.nan, 1])
        self.assertEqual(x["defined_draws"], 2)
        self.assertEqual(x["undefined_draws"], 1)


if __name__ == "__main__":
    unittest.main()
