"""Invalid native measurements remain evidence, never comparison gains."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.dflash_per_turn.experiment import validate_measurement
from . import experiment as e
from .measurement import measurement_validation
from .report import comparison, summarize, write_reports
from .test_experiment import Backend, references, stream


def invalid_measurements():
    cases = [
        ("accepted_exceeds_output", {"accepted_draft_tokens": 4}),
        ("missing_output", {"output_ids": None}),
        ("empty_output", {"output_ids": []}),
        ("negative_token", {"output_ids": [-1]}),
        ("out_of_vocabulary", {"output_ids": [128256]}),
        ("boolean_token", {"output_ids": [True]}),
        ("output_over_cap", {"output_ids": [1] * 1025}),
        ("missing_text", {"text": None}),
        ("feedback_capture", {"feedback": {}}),
        ("capture_costs", {"capture_costs": {}}),
        ("capture_enabled", {"capture": True}),
    ]
    for key in ("request_s", "activation_s", "native_activation_s", "cache_maintenance_s"):
        for label, value in (("missing", None), ("negative", -1.), ("nonfinite", float("nan")), ("boolean", True)):
            cases.append((key + "_" + label, {key: value}))
    for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
        for label, value in (("missing", None), ("negative", -1), ("fractional", .5), ("boolean", True)):
            cases.append((key + "_" + label, {key: value}))
    return cases


class InvalidScoreBackend(Backend):
    def __init__(self, arm, changes):
        super().__init__(arm)
        self.changes = changes

    def serve(self, *args, **kwargs):
        output = super().serve(*args, **kwargs)
        if kwargs.get("repeat") is not None:
            output.update(copy.deepcopy(self.changes))
        return output


def completed_result():
    source = stream()
    return e.run_stream(source, references(source), lambda arm, name: Backend(arm))


class MeasurementTests(unittest.TestCase):
    def test_valid_receipt_and_independent_reporting(self):
        result = completed_result()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(all(row["measurement_validation"] == {"status": "passed", "errors": []}
                            for row in result["scores"]))
        self.assertTrue(all(row["valid"] for row in summarize([result])["comparisons"]))

    def test_all_shared_validator_failures_persist_before_execution_rejects(self):
        source = stream()
        reference = references(source)[2]
        for label, changes in invalid_measurements():
            with self.subTest(label=label):
                backend = InvalidScoreBackend("H", changes)
                destination, emitted = [], []
                with self.assertRaises(ValueError):
                    e.score(backend, reference, "F", "F", destination=destination,
                            emit=lambda kind, row: emitted.append((kind, copy.deepcopy(row))))
                self.assertEqual(len(destination), 1)
                saved = destination[0]
                self.assertEqual(emitted[0][0], "score")
                self.assertEqual(saved["measurement_validation"]["status"], "failed")
                self.assertTrue(saved["measurement_validation"]["errors"])
                self.assertEqual(emitted[0][1]["measurement_validation"], saved["measurement_validation"])
                # The wrapper preserves the qualified validator's actual failure.
                with self.assertRaises(ValueError) as error:
                    validate_measurement(saved)
                self.assertEqual(saved["measurement_validation"]["errors"], [str(error.exception)])
                if "capture" in changes:
                    self.assertIs(saved["capture"], True)

    def test_forged_pass_status_cannot_bypass_any_shared_validator_rule(self):
        original = completed_result()
        for label, changes in invalid_measurements():
            with self.subTest(label=label):
                result = copy.deepcopy(original)
                row = next(r for r in result["scores"] if r["arm"] == "G" and r["turn"] == 2)
                row.update(changes)
                row["measurement_validation"] = {"status": "passed", "errors": []}
                value, _ = comparison(result, 2, "G", "F")
                self.assertFalse(value["valid"])
                self.assertIsNone(value["verification_calls_saved"])
                self.assertIsNone(value["verification_call_reduction_percent"])
                self.assertIsNone(value["diagnostic_latency_reduction_percent"])
                self.assertTrue(any(reason.startswith("G:measurement_validation:")
                                    for reason in value["invalid_reasons"]))
                report = summarize([result])
                pooled = next(r for r in report["pooled"] if r["source"] == "all" and r["turn"] == "all"
                              and r["candidate"] == "G" and r["control"] == "F")
                self.assertEqual((pooled["valid_requests"], pooled["invalid_requests"]), (1, 1))
                self.assertEqual(pooled["candidate_metrics"]["verification_calls"], 2)
                self.assertEqual(pooled["verification_calls_saved"], 0)

    def test_absent_receipt_does_not_bypass_revalidation(self):
        result = completed_result()
        row = next(r for r in result["scores"] if r["arm"] == "G" and r["turn"] == 2)
        del row["measurement_validation"]
        row["accepted_draft_tokens"] = 4
        row["verification_calls"] = 0
        self.assertFalse(comparison(result, 2, "G", "F")[0]["valid"])

    def test_recorded_failure_stays_invalid_if_fields_are_later_changed(self):
        result = completed_result()
        row = next(r for r in result["scores"] if r["arm"] == "G" and r["turn"] == 2)
        row["measurement_validation"] = {"status": "failed", "errors": ["Accepted count exceeds output"]}
        self.assertEqual(measurement_validation(row)["status"], "passed")
        value, _ = comparison(result, 2, "G", "F")
        self.assertFalse(value["valid"])
        self.assertIn("G:recorded_measurement_validation_failed", value["invalid_reasons"])

    def test_controller_rejection_keeps_receipt_and_reports_it_without_gain(self):
        source = stream()
        backends, emitted = {}, []
        def factory(arm, name):
            backends[arm] = (InvalidScoreBackend(arm, {"accepted_draft_tokens": 4, "verification_calls": 0})
                             if arm == "G" else Backend(arm))
            return backends[arm]
        result = e.run_stream(source, references(source), factory,
                              emit=lambda kind, row: emitted.append((kind, copy.deepcopy(row))))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_arm"], "G")
        self.assertNotIn("N", backends)
        self.assertFalse(backends["G"].finalized)
        bad = next(r for r in result["scores"] if r["arm"] == "G")
        self.assertTrue(bad["same_generation_valid"])
        self.assertEqual(bad["measurement_validation"],
                         {"status": "failed", "errors": ["Accepted count exceeds output"]})
        self.assertTrue(any(kind == "score" and row.get("arm") == "G" and
                            row.get("measurement_validation") == bad["measurement_validation"]
                            for kind, row in emitted))
        with tempfile.TemporaryDirectory() as directory:
            report = write_reports([json.loads(json.dumps(result))], directory)
            disk = json.loads((Path(directory) / "SUMMARY.json").read_text())
            retained = next(r for r in disk["scores"] if r["arm"] == "G")
            self.assertEqual(retained["measurement_validation"], bad["measurement_validation"])
            self.assertEqual(retained["accepted_draft_tokens"], 4)
            self.assertTrue(all(not r["valid"] for r in report["comparisons"] if r["candidate"] == "G"))
            self.assertTrue(all(r["valid_requests"] == 0 and r["verification_calls_saved"] is None
                                for r in report["pooled"] if r["candidate"] == "G"))

    def test_nonfinite_rejection_is_durable_and_losslessly_encoded(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=repr(value)):
                source = stream()
                emitted = []
                def factory(arm, name):
                    return (InvalidScoreBackend(arm, {"request_s": value, "verification_calls": 0,
                                                     "native_trace": {"values": [value]}})
                            if arm == "G" else Backend(arm))
                result = e.run_stream(source, references(source), factory,
                    emit=lambda kind, row: emitted.append(json.dumps({"event": kind, "data": row}, allow_nan=False)))
                self.assertEqual(result["status"], "failed")
                bad = next(r for r in result["scores"] if r["arm"] == "G")
                marker = bad["request_s"]
                self.assertEqual(marker["nonfinite_float"], repr(value))
                self.assertEqual(marker["ieee754_binary64_hex"], struct.pack(">d", value).hex())
                self.assertEqual(bad["native_trace"]["values"], [marker])
                self.assertEqual(bad["measurement_validation"],
                                 {"status": "failed", "errors": ["Invalid/missing cost: request_s"]})
                # Actual strict event output and complete report files survive.
                retained_event = next(json.loads(record)["data"] for record in emitted
                    if json.loads(record)["event"] == "score" and json.loads(record)["data"]["arm"] == "G")
                self.assertEqual(retained_event["request_s"], marker)
                with tempfile.TemporaryDirectory() as directory:
                    report = write_reports([json.loads(json.dumps(result, allow_nan=False))], directory)
                    saved = json.loads((Path(directory) / "SUMMARY.json").read_text())
                    retained = next(r for r in saved["scores"] if r["arm"] == "G")
                    self.assertEqual(retained["request_s"], marker)
                    self.assertEqual(retained["measurement_validation"], bad["measurement_validation"])
                    self.assertTrue(all(r["valid_requests"] == 0 and r["verification_calls_saved"] is None
                                        for r in report["pooled"] if r["candidate"] == "G"))


if __name__ == "__main__":
    unittest.main()
