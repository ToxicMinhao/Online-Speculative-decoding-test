import unittest
from experiments.longer_run_headroom.census import classify, digest, sample


class CensusTests(unittest.TestCase):
    def test_short_requires_wrong_second_follower(self):
        r = classify([1, 2, 9, 4, 8], [1, 2, 3, 4, 5], 0, [99])
        self.assertEqual((r["group"], r["j"], r["R"], r["v"]), ("SHORT", 3, 1, 2))

    def test_long_and_seed_exclusion(self):
        r = classify([9, 2, 3, 8], [1, 2, 3, 4], 0, [99])
        self.assertEqual((r["group"], r["j"], r["R"]), ("LONG", 1, 2))

    def test_last_error_is_insufficient(self):
        self.assertEqual(
            classify([1, 9], [1, 2], 0, [99])["group"], "insufficient_lookahead"
        )

    def test_one_follower_is_not_short(self):
        self.assertEqual(
            classify([9, 2], [1, 2], 0, [99])["group"], "insufficient_lookahead"
        )

    def test_valid_eos_included_post_eos_excluded(self):
        r = classify([9, 2, 99, 4], [1, 2, 99, 4], 0, [99])
        self.assertEqual((r["group"], r["R"], r["valid_slots"]), ("LONG", 2, 3))

    def test_cap_censors_available_suffix(self):
        self.assertEqual(
            classify([9, 2, 3, 4], [1, 2], 0, [99])["group"], "insufficient_lookahead"
        )

    def test_all_correct_and_zero_run(self):
        self.assertEqual(
            classify([1, 2, 3], [1, 2, 3], 0, [99])["group"], "all_correct"
        )
        self.assertEqual(classify([9, 8, 3], [1, 2, 3], 0, [99])["group"], "R0")
        self.assertEqual(classify([9], [1], 99, [99])["group"], "seed_eos")

    def test_sample_excludes_opening_and_has_fixed_hash_order(self):
        rows = [
            dict(
                dataset="gsm8k",
                request_id="r",
                parent_round_ordinal=i,
                opening=i == 1,
                group="LONG",
                selection_sha256=digest(["kplus-run-probe-v1", "gsm8k", "r", i]),
            )
            for i in range(1, 80)
        ]
        chosen, groups = sample(rows)
        self.assertEqual(len(chosen), 64)
        self.assertTrue(all(not r["opening"] for r in chosen))
        self.assertEqual(chosen, sample(list(reversed(rows)))[0])
        self.assertAlmostEqual(chosen[0]["inclusion_fraction"], 64 / 78)


if __name__ == "__main__":
    unittest.main()
