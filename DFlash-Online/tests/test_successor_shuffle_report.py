from copy import deepcopy
import unittest
import numpy as np
from experiments.successor_shuffle import report as R


def rows(n=64):
    values = []
    for i in range(n):
        for method, calls, latency, prefix in [
            ("K", 10, 2.0, 1),
            ("SHUFFLED", 9, 1.9, 2),
            ("TRUE", 8, 1.8, 3),
        ]:
            for repeat in range(2):
                values.append(
                    dict(
                        phase="final",
                        event=5,
                        dataset="gsm8k",
                        seed=17,
                        method=method,
                        validation_status="valid",
                        valid_measurement=True,
                        errors=[],
                        panel_ordinal=i,
                        request_id=f"r{i}",
                        repeat=repeat,
                        verification_calls=calls,
                        accepted_draft_tokens=64 - calls,
                        native_accepted_draft_tokens=64 - calls,
                        committed_progress=64,
                        output_tokens=65,
                        checkpoint_id="/" + method,
                        finish_reason="eos",
                        cap_hit=False,
                        request_latency_s=latency + repeat * 0.2,
                        native_generate_s=latency - 0.1 + repeat * 0.2,
                        cache_maintenance_s=0.01,
                        first_round_present=True,
                        first_round_start_digest="a" * 64,
                        first_round_accepted_prefix_tokens=prefix,
                        first_round_raw_prefix_tokens=prefix,
                        first_round_output_start=0,
                    )
                )
    return values


class ReportTests(unittest.TestCase):
    def test_real_contrasts_counts_once_and_mean_time(self):
        result, pairs = R.endpoint_from_audit(
            dict(status="passed", errors=[], rows=rows()), 64
        )
        summary = result["summary"]
        self.assertEqual(
            set(summary["comparisons"]),
            {"TRUE_vs_SHUFFLED", "TRUE_vs_K", "SHUFFLED_vs_K"},
        )
        self.assertEqual(summary["method_totals"]["K"]["verification_calls"], 640)
        self.assertAlmostEqual(
            summary["method_totals"]["K"]["request_latency_s"], 64 * 2.1
        )
        self.assertEqual(
            R.decision(summary)["outcome"], "informative_placement_supported"
        )
        self.assertTrue(R.decision(summary)["incremental_verifier_benefit_vs_K"])

    def test_joint_bootstrap_independent_numpy_recomputation(self):
        sample = rows()
        for r in sample:
            i = r["panel_ordinal"]
            base = {"K": 10, "TRUE": 8, "SHUFFLED": 9}[r["method"]]
            calls = base + i % {"K": 4, "TRUE": 3, "SHUFFLED": 5}[r["method"]]
            r.update(
                verification_calls=calls,
                accepted_draft_tokens=64 - calls,
                native_accepted_draft_tokens=64 - calls,
            )
        summary = R.endpoint_from_audit(
            dict(status="passed", errors=[], rows=sample), 64
        )[0]["summary"]
        draw = np.random.Generator(np.random.PCG64(20261111)).integers(
            0, 64, size=(10000, 64)
        )
        a = np.array([8 + i % 3 for i in range(64)])
        b = np.array([9 + i % 5 for i in range(64)])
        values = 100 * (1 - a[draw].sum(axis=1) / b[draw].sum(axis=1))
        expected = np.quantile(values, [0.025, 0.975], method="linear")
        np.testing.assert_array_equal(
            summary["comparisons"]["TRUE_vs_SHUFFLED"]["call_reduction"]["ci95_pct"],
            expected,
        )

    def test_primary_unresolved_does_not_become_equivalence_or_k_gain(self):
        summary = R.endpoint_from_audit(
            dict(status="passed", errors=[], rows=rows()), 64
        )[0]["summary"]
        changed = deepcopy(summary)
        changed["comparisons"]["TRUE_vs_SHUFFLED"]["call_reduction"]["ci95_pct"] = [
            -1,
            1,
        ]
        d = R.decision(changed)
        self.assertEqual(d["outcome"], "successor_placement_unresolved")
        self.assertTrue(d["incremental_verifier_benefit_vs_K"])
        self.assertFalse(d["informative_placement_and_incremental_verifier_benefit"])
        changed["comparisons"]["TRUE_vs_SHUFFLED"]["call_reduction"]["ci95_pct"] = [
            -2,
            -0.1,
        ]
        self.assertEqual(
            R.decision(changed)["outcome"], "prioritization_unfavorable_for_this_fork"
        )

    def test_latency_requires_own_interval_and_both_sweeps(self):
        summary = R.endpoint_from_audit(
            dict(status="passed", errors=[], rows=rows()), 64
        )[0]["summary"]
        for alteration in ("interval", "sweep"):
            changed = deepcopy(summary)
            if alteration == "interval":
                changed["comparisons"]["TRUE_vs_SHUFFLED"][
                    "request_latency_s_reduction"
                ]["ci95_pct"] = [-0.1, 1]
            else:
                changed["repeat_totals"]["1"]["TRUE"]["request_latency_s"] = 1000
            self.assertFalse(
                R.decision(changed)["latency"]["TRUE_vs_SHUFFLED"]["benefit_supported"]
            )

    def test_invalid_incomplete_or_duplicate_repeat_blocks_inference(self):
        for sample in (rows()[:-1], rows()[:-1] + [rows()[0]]):
            result, paired = R.endpoint_from_audit(
                dict(status="passed", errors=[], rows=sample), 64
            )
            self.assertFalse(result["inference_permitted"])
            self.assertEqual(paired, [])
        result, _ = R.endpoint_from_audit(
            dict(status="failed", errors=["native"], rows=rows()), 64
        )
        self.assertFalse(result["inference_permitted"])
        self.assertEqual(len(R.fallacy_checks()["checks"]), 11)


if __name__ == "__main__":
    unittest.main()
