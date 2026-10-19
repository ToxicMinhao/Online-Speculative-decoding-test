"""Focused accounting/paired-validity checks, retaining harmful observations."""
import copy
import tempfile
from pathlib import Path
import unittest

from .experiment import run_stream
from .test_experiment import FakeBackend, stream
from . import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.result = run_stream(stream(), FakeBackend())
        self.assertEqual(self.result["status"], "complete", self.result.get("error"))

    def test_ratios_pool_sums_not_average_percentages(self):
        rows = self.result["scores"]
        for row in rows:
            row["request_s"] = (10. if row["turn"] == 1 else 1.) * (2. if row["arm"] == "BANK" else 1.)
        summary = report.summarize([self.result])
        paired = summary["overall"][0]
        self.assertAlmostEqual(paired["request_time_reduction_percent"], -100.)
        self.assertEqual(paired["valid_contexts"], 20)
        self.assertEqual(paired["candidate_metrics"]["executions"], 60)

    def test_invalid_pair_excluded_without_deleting_other_valid_comparison(self):
        row = next(r for r in self.result["scores"] if r["turn"] == 3 and r["arm"] == "OSD" and r["repeat"] == 0)
        row["output_ids"] = [99]; row["canonical_output_match"] = False; row["same_generation_valid"] = False
        summary = report.summarize([self.result])
        by_pair = {(p["candidate"], p["control"]): p for p in summary["overall"]}
        self.assertEqual(by_pair[("BANK", "F")]["valid_contexts"], 20)
        self.assertEqual(by_pair[("BANK", "OSD")]["valid_contexts"], 19)
        self.assertEqual(by_pair[("OSD", "F")]["valid_contexts"], 19)
        self.assertEqual(len(self.result["scores"]), 180)

    def test_complete_costs_count_outer_update_and_activation_once(self):
        costs = report.cost_breakdown(self.result)
        bank = costs["arms"]["BANK"]
        self.assertTrue(bank["complete_nonoverlapping_accounting"])
        self.assertAlmostEqual(bank["serving_plus_adaptation_episode_s"], 20*(1.1+.2+.01)+19*1.2)
        self.assertAlmostEqual(bank["serving_only_s"], 22.)
        self.assertEqual(len(costs["final_feedback_analysis_only"]), 1)
        self.assertTrue(costs["benchmark_excluded_from_deployment"])
        # Neither three benchmark repeats nor nested update stage timers are added again.
        self.assertGreater(costs["benchmark_request_s"], bank["serving_plus_adaptation_episode_s"])

    def test_missing_cost_not_silently_zero(self):
        del self.result["updates"][0]["recurring_adaptation_s"]
        bank = report.cost_breakdown(self.result)["arms"]["BANK"]
        self.assertFalse(bank["complete_nonoverlapping_accounting"])
        self.assertIsNone(bank["serving_plus_adaptation_episode_s"])

    def test_benchmark_method_restoration_is_not_deployment_activation(self):
        for row in self.result["canonical"]:
            if row["arm"] == "BANK":
                row["activation_s"] = 5.2  # Includes an artificial restore from the other arm.
        bank = report.cost_breakdown(self.result)["arms"]["BANK"]
        self.assertAlmostEqual(bank["nonoverlapping_components"]["external_activation_s"], 4.)
        self.assertAlmostEqual(bank["actual_shared_worker_activation_s"], 104.)
        self.assertAlmostEqual(bank["serving_plus_adaptation_episode_s"], 20*(1.1+.2+.01)+19*1.2)

    def test_reconstructed_frozen_episode_uses_repeat_mean_not_all_replays(self):
        costs = report.cost_breakdown(self.result)
        frozen = costs["frozen_reference_on_bank_trajectory"]
        self.assertTrue(frozen["complete_valid_context_coverage"])
        self.assertFalse(frozen["actual_frozen_canonical_session"])
        self.assertEqual(frozen["timing_executions_used"], 60)
        self.assertAlmostEqual(frozen["serving_only_s"], 20.)
        self.assertAlmostEqual(frozen["cache_maintenance_s"], .2)
        self.assertAlmostEqual(frozen["serving_plus_adaptation_episode_s"], 20.2)
        self.assertEqual(costs["arms"]["OSD"]["canonical_input_trajectory"], "OSD")
        row = next(r for r in self.result["scores"] if r["arm"] == "F")
        row["same_generation_valid"] = False
        frozen = report.cost_breakdown(self.result)["frozen_reference_on_bank_trajectory"]
        self.assertFalse(frozen["complete_valid_context_coverage"])
        self.assertIsNone(frozen["serving_plus_adaptation_episode_s"])
        self.assertEqual(len(frozen["invalid_contexts"]), 1)

    def test_zero_fallback_is_still_measured_and_source_fit_separate(self):
        bank = report.bank_usage(self.result)
        self.assertEqual(bank["zero_count"], 1)
        self.assertEqual(bank["singleton_count"], 19)
        self.assertAlmostEqual(bank["active_use_rate"], .95)
        self.assertIsNone(bank["decisions"][0]["realized_served_gain"])
        self.assertGreater(bank["decisions"][0]["post_write_fit_gain"], 0)
        summary = report.summarize([self.result])
        self.assertAlmostEqual(summary["overall"][0]["request_time_reduction_percent"], -10.)

    def test_source_separation_and_every_repeat_delivery(self):
        persona = run_stream(stream("personamem_persona42"), FakeBackend())
        summary = report.summarize([self.result, persona])
        self.assertEqual(set(summary["by_source"]), {"wildchat", "personamem"})
        with tempfile.TemporaryDirectory() as temporary:
            report.write_reports([self.result, persona], temporary)
            self.assertTrue((Path(temporary)/"REPORT.json").exists())
            import csv
            with (Path(temporary)/"timing_repeats.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9*(20+18))


if __name__ == "__main__":
    unittest.main()
