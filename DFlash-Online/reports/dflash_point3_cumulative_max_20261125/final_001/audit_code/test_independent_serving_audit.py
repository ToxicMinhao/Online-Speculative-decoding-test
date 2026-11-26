"""Synthetic checks for the standalone audit; no native/model execution."""
from copy import deepcopy
import unittest

from independent_serving_audit import ARMS, audit, collect, recompute, select


def fixture():
    rows = []
    for repeat in range(3):
        for group in range(2):
            order = ARMS[(repeat + group) % 3:] + ARMS[:(repeat + group) % 3]
            for position, arm in enumerate(order):
                arm_index = ARMS.index(arm)
                for item in range(66):
                    warmup = item < 2
                    request = item if warmup else group * 64 + item - 2
                    latency = (2.0 - arm_index * .1) * (1 + repeat * .1)
                    calls = 20 - arm_index
                    rows.append(dict(
                        activation=(repeat * 2 + group) * 3 + position,
                        repeat=repeat, group=group, method=arm,
                        phase="warmup" if warmup else "scored",
                        request_id=("warmup_" if warmup else "request_") + f"{request:03d}",
                        source_id=str(request), physical_gpu_uuid="GPU-test",
                        valid_measurement=True, errors=[], checkpoint_id=arm + "/export",
                        output_tokens=101, committed_progress=100,
                        verification_calls=calls, accepted_draft_tokens=100-calls,
                        native_accepted_draft_tokens=100-calls, finish_reason="eos",
                        request_latency_s=latency, native_generate_s=latency * .8,
                        cache_maintenance_s=.125,
                    ))
    return rows


def summary_fixture():
    rows = fixture()
    result = recompute(rows)
    result["serving_identities"] = {a: dict(parameters=a, inference={"block_size": 16}) for a in ARMS}
    result["parameter_identities"] = {a: dict(sha256=a) for a in ARMS}
    result["freeze"] = select(result, result["serving_identities"])
    return rows, result


class ServingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.summary = summary_fixture()

    def test_known_constant_ratio_and_counts(self):
        result = self.summary
        self.assertEqual(result["totals"]["C"]["verification_calls"], 2560)
        self.assertEqual(result["totals"]["F_C_MAX"]["verification_calls"], 2304)
        self.assertAlmostEqual(result["totals"]["C"]["request_latency_s"], 281.6)
        comparison = result["contrasts"]["F_C_MAX_vs_C"]
        self.assertEqual((comparison["wins"], comparison["regressions"], comparison["ties"]), (128, 0, 0))
        self.assertEqual(comparison["gross_calls_saved"], 256)
        for field in ("verification_call_reduction_pct", "request_latency_s_reduction_pct"):
            self.assertAlmostEqual(comparison[field]["value"], 10)
            for bound in comparison[field]["ci95"]:
                self.assertAlmostEqual(bound, 10)
        self.assertEqual(result["freeze"]["selected"], "F_C_MAX")

    def test_unchanged_summary_passes_and_tampering_fails(self):
        self.assertEqual(audit(self.rows, self.summary)["status"], "passed")
        altered = deepcopy(self.summary)
        altered["contrasts"]["F_C_MAX_vs_C"]["request_latency_s_reduction_pct"]["ci95"][0] += .01
        result = audit(self.rows, altered)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("ci95/0" in e for e in result["errors"]))

    def test_invalid_or_inconsistent_measurements_stop(self):
        for key, value in (("valid_measurement", False), ("verification_calls", 99),
                           ("native_generate_s", 1000)):
            rows = deepcopy(self.rows)
            scored = next(r for r in rows if r["phase"] == "scored")
            scored[key] = value
            with self.assertRaises(ValueError):
                collect(rows)

    def test_identical_models_ignore_timing_noise(self):
        result = deepcopy(self.summary)
        ids = result["serving_identities"]
        ids["F_C_MAX"] = deepcopy(ids["F_C_SUM"])
        self.assertEqual(select(result, ids)["selected"], "F_C_SUM")
        ids["F_C_SUM"] = ids["F_C_MAX"] = deepcopy(ids["C"])
        decision = select(result, ids)
        self.assertEqual(decision["selected"], "C")
        self.assertTrue(decision["no_demonstrated_speedup"])

    def test_call_regression_disqualifies_faster_candidate(self):
        result = deepcopy(self.summary)
        result["totals"]["F_C_MAX"]["verification_calls"] = 2561
        self.assertEqual(select(result, result["serving_identities"])["selected"], "F_C_SUM")

    def test_exact_selection_ties_and_provisional_interval(self):
        result = deepcopy(self.summary)
        result["totals"]["F_C_MAX"] = deepcopy(result["totals"]["F_C_SUM"])
        decision = select(result, result["serving_identities"])
        self.assertEqual(decision["selected"], "F_C_SUM")
        result["contrasts"]["F_C_SUM_vs_C"]["request_latency_s_reduction_pct"]["ci95"] = [-.01, .1]
        decision = select(result, result["serving_identities"])
        self.assertEqual(decision["status"], "provisional_development_selection")
        self.assertTrue(decision["no_demonstrated_speedup"])

    def test_zero_cache_time_is_undefined_not_nan(self):
        rows = deepcopy(self.rows)
        for row in rows:
            row["cache_maintenance_s"] = 0
        result = recompute(rows)
        for comparison in result["contrasts"].values():
            self.assertEqual(comparison["cache_maintenance_s_reduction_pct"],
                             dict(value=None, ci95=None, status="undefined"))

    def test_external_cpu_identity_receipt_matches_execution_and_exports(self):
        receipt = dict(status="passed", gpu_work=0, experimental_modules_imported=False,
                       weights_loaded_into_framework=False, source={"commit": "fixture"},
                       tensor_content_identities=deepcopy(self.summary["parameter_identities"]))
        result = audit(self.rows, self.summary, identity_receipt=receipt, execution_commit="fixture")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["identity_status"], "independently_verified_cpu_postflight")
        with self.assertRaises(ValueError):
            audit(self.rows, self.summary, identity_receipt=receipt, execution_commit="another")
        receipt["tensor_content_identities"]["C"]["sha256"] = "changed"
        self.assertEqual(audit(self.rows, self.summary, identity_receipt=receipt,
                               execution_commit="fixture")["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
