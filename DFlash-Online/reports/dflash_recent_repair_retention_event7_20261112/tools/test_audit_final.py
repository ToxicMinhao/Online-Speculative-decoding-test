"""Synthetic CPU fixtures for the separately maintained result auditor."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import audit_final


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    evaluation, report = root / "evaluation", root / "final_report"
    requests = [dict(request_id=f"r{i}", prompt_ids=[i + 3], stop_token_ids=[99], max_new_tokens=2048)
                for i in range(2)]
    refs = {row["request_id"]: dict(response_ids=[100, 11, 12, 99], finish_reason="eos") for row in requests}
    exports = {arm: "/export/" + arm for arm in audit_final.ARMS}
    write(root / "STATUS.json", {"status": "completed"})
    write(root / "EXPORTS.json", exports)
    write(evaluation / "REQUESTS.json", requests)
    write(evaluation / "CANONICAL.json", refs)
    measurements, per_request = [], []
    for ordinal, request in enumerate(requests):
        for arm_index, arm in enumerate(audit_final.ARMS):
            times = []
            for repeat in (0, 1):
                trace = [dict(output_start=0, anchor_token=100, verify_size=4,
                    draft_tokens=[11, 12, 99], target_argmax=[11, 12, 99, 4],
                    committed_tokens=[11, 12, 99], committed_progress=3,
                    accepted_draft_tokens=3, native_accepted_draft_tokens=2,
                    raw_accepted_draft_tokens_before_stop=3, stop_clipped=True)]
                latency = float(arm_index + 1) + ordinal * .1 + repeat * .2
                times.append(latency)
                capture = dict(prompt_ids=request["prompt_ids"], response_ids=refs[request["request_id"]]["response_ids"],
                    blocks=trace, cache_maintenance_s=.01)
                metadata = dict(trace=trace, capture=capture, feedback_capture_in_timed_request=False,
                    verification_calls=1, accepted_draft_tokens=3, committed_progress=3,
                    request_latency_s=latency, native_generate_s=latency - .02)
                record = dict(prompt_ids=request["prompt_ids"], response_ids=capture["response_ids"],
                    finish_reason="eos", checkpoint_id=exports[arm], metadata=metadata)
                write(evaluation / f"{ordinal:04d}_r{repeat}_{arm}.json", record)
                measurements.append(dict(request_id=request["request_id"], method=arm, repeat=repeat,
                    panel_ordinal=ordinal, verification_calls=1, accepted_draft_tokens=3, output_tokens=4,
                    committed_progress=3, native_accepted_draft_tokens=2, finish_reason="eos",
                    request_latency_s=latency, native_generate_s=latency - .02, cache_maintenance_s=.01,
                    valid_measurement=True, validation_status="valid", errors=[], structural_errors=[]))
            per_request.append(dict(request_id=request["request_id"], method=arm,
                verification_calls=1, accepted_draft_tokens=3, output_tokens=4, committed_progress=3,
                finish_reason="eos", request_latency_s=sum(times)/2,
                native_generate_s=sum(times)/2-.02, cache_maintenance_s=.01))
    write(evaluation / "MEASUREMENTS.jsonl", measurements, lines=True)
    table(report / "per_request.csv", per_request)
    table(report / "all_repeats.csv", measurements)
    assigned = {"RECENT": [], "CONTROL": []}
    for pid in (0, 1):
        for arm, start, margin in (("RECENT", 0, .6), ("CONTROL", 7, .7)):
            assigned[arm].append(dict(request_id="r_train", canonical_start=start, proposal_slot=2,
                presentation_id=pid, margin=margin, floor=.5*margin, weight=.75))
    matches = [dict(source_slot_id="source", destination_slot_id="destination", presentation_ids=[0, 1],
                    source_margins=[.6, .6], destination_margins=[.7, .7])]
    write(root / "ASSIGNMENTS.json", dict(arms=assigned, matches=matches,
        summary=dict(recent_occurrences=2, assigned_raw_weight=1.5, matched_distinct_slots=1)))
    table(report / "assigned_slots.csv", [dict(arm=arm, **row) for arm, rows in assigned.items() for row in rows])
    table(report / "matching.csv", [dict(source_slot_id="source", destination_slot_id="destination",
        presentation_id=pid, source_margin=.6, destination_margin=.7) for pid in (0, 1)])
    comparisons = {}
    for other in ("NATIVE", "CONTROL"):
        recent_time = sum(row["request_latency_s"] for row in per_request if row["method"] == "RECENT")
        other_time = sum(row["request_latency_s"] for row in per_request if row["method"] == other)
        comparisons[other] = dict(recent_calls=2, reference_calls=2, recent_accepted=6, reference_accepted=6,
            recent_request_s=recent_time, reference_request_s=other_time, accepted_length_relative_gain_pct=0.,
            time_reduction_pct=100*(1-recent_time/other_time))
    write(report / "SUMMARY.json", dict(complete_valid_comparison=True, comparisons=comparisons))
    return measurements


class IndependentAuditTests(unittest.TestCase):
    def test_counts_once_times_average_repeats_and_slot_invariants(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            result = audit_final.run(root)
            self.assertEqual(result["status"], "passed", result["errors"])
            self.assertEqual(result["serving"]["totals"]["RECENT"]["verification_calls"], 2)
            self.assertEqual(result["serving"]["totals"]["RECENT"]["accepted_draft_tokens"], 6)
            self.assertAlmostEqual(result["serving"]["totals"]["RECENT"]["request_latency_s"], 4.3)
            self.assertEqual(result["matching"]["assigned_raw_weight"], 1.5)

    def test_wrong_clipped_acceptance_is_invalid_even_when_flag_claims_valid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "evaluation/0000_r0_RECENT.json"
            raw = json.loads(path.read_text())
            raw["metadata"]["accepted_draft_tokens"] = 2
            write(path, raw)
            result = audit_final.run(root)
            self.assertEqual(result["status"], "failed")
            self.assertGreater(result["serving"]["invalid_raw"], 0)

    def test_count_table_regression_is_detected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "final_report/SUMMARY.json"
            summary = json.loads(path.read_text())
            summary["comparisons"]["NATIVE"]["recent_calls"] = 4
            write(path, summary)
            result = audit_final.run(root)
            self.assertEqual(result["status"], "failed")
            self.assertIn("summary:NATIVE:recent_calls", result["errors"])

    def test_assigned_weight_difference_is_detected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "ASSIGNMENTS.json"
            assigned = json.loads(path.read_text())
            assigned["arms"]["CONTROL"][0]["weight"] = .5
            write(path, assigned)
            result = audit_final.run(root)
            self.assertEqual(result["status"], "failed")
            self.assertIn("matching:presentation_depth_count_and_weight_invariant", result["errors"])

    def test_incomplete_work_remains_incomplete_without_gain_claim(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixture(root)
            write(root / "STATUS.json", {"status": "failed"})
            write(root / "evaluation/MEASUREMENTS.jsonl", rows[:1], lines=True)
            table(root / "final_report/all_repeats.csv", rows[:1])
            table(root / "final_report/per_request.csv", [])
            write(root / "final_report/SUMMARY.json", dict(complete_valid_comparison=False, comparisons={}))
            result = audit_final.run(root)
            self.assertEqual(result["status"], "incomplete", result["errors"])
            self.assertEqual(result["serving"]["unavailable_or_unattempted"], 11)
            self.assertEqual(result["serving"]["paired_requests"], 0)

    def test_token_mismatch_is_not_hidden_by_matching_counts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "evaluation/0000_r1_CONTROL.json"
            raw = json.loads(path.read_text())
            raw["response_ids"] = [100, 11, 13, 99]
            write(path, raw)
            result = audit_final.run(root)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any(error.endswith(":tokens") for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
