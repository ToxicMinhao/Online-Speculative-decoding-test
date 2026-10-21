"""Synthetic report checks, including adversarial validation and weighting."""
from copy import deepcopy
import json

import pytest

from .report import summarize, write_reports
from .test_protocol import anchor_fixture, row_fixture


def request_locations(request_id, count, frozen_length, adapted_length, *, population="heldout"):
    output = [10] + list(range(100, 180))
    anchors = [anchor_fixture(request_id=request_id, start=index + 1, output=output,
                              frozen_length=frozen_length, population=population) for index in range(count)]
    request = dict(group_id="g", domain="gsm8k", request_id=request_id, population=population,
                   source_control=population == "source", selected_location_count=count, anchors=anchors,
                   retained_measurements={arm: dict(verification_calls=n, output_tokens=80, accepted_draft_tokens=30,
                                                  retained_response_kl=1.2) for arm, n in (("F", 10), ("S8", 11), ("L8", 10))})
    locations = [dict(anchor=anchor, rows={arm: row_fixture(anchor, arm, prefix_length=None if arm == "F" else adapted_length)
                                         for arm in ("F", "S8", "L8")}, comparisons={"S8": {"valid": True}}) for anchor in anchors]
    return request, locations


def result_fixture(*specs):
    requests, locations = [], []
    for spec in specs or (("r", 1, 2, 4),):
        request, rows = request_locations(*spec)
        requests.append(request)
        locations.extend(rows)
    return dict(status="completed", groups=[dict(group_id="g", domain="gsm8k", status="completed", requests=requests,
                                                  locations=locations, costs={"loading_seconds": 1.25})],
                counters={"actual_draft_block_evaluations": 3 * len(locations)}, costs={"wall_seconds": 5.})


def overall(analysis, arm="S8", population="heldout"):
    return next(row for row in analysis["overall"] if row["arm"] == arm and row["population"] == population)


def test_equal_request_weight_not_equal_location_weight():
    analysis = summarize(result_fixture(("many", 8, 2, 4), ("few", 1, 4, 0)))
    row = overall(analysis)
    assert row["selected_locations"] == row["valid_locations"] == 9
    assert row["mean_correct_prefix_gain"] == -1
    assert row["prefix_gain_blocks"] == 8 and row["prefix_loss_blocks"] == 1
    assert row["mean_correct_prefix_gain_request_denominator"] == 2
    assert next(row for row in analysis["group_summaries"] if row["arm"] == "S8")["mean_correct_prefix_gain"] == -1
    assert next(row for row in analysis["domain_summaries"] if row["arm"] == "S8")["mean_correct_prefix_gain"] == -1


def test_report_revalidates_raw_rows_instead_of_trusting_cached_comparisons():
    result = result_fixture()
    result["groups"][0]["locations"][0]["rows"]["S8"]["target_verifications"] = 1
    row = overall(summarize(result))
    assert row["invalid_locations"] == 1 and row["valid_locations"] == 0
    assert row["mean_correct_prefix_gain"] is None and row["useful_first_error_fixes"] == 0


def test_failed_partial_group_preserves_valid_completed_rows_and_missing_denominators():
    result = result_fixture(("r", 3, 2, 4))
    group = result["groups"][0]
    group["status"] = "failed"
    group["locations"][1]["rows"]["S8"]["status"] = "failed"
    group["locations"][2]["rows"] = {}
    row = overall(summarize(result))
    assert (row["selected_locations"], row["valid_locations"], row["invalid_locations"], row["unattempted_locations"]) == (3, 1, 1, 1)
    assert row["mean_correct_prefix_gain"] == 2


def test_checkpoint_major_partial_rows_keep_missing_and_invalid_denominators_separate():
    result = result_fixture(("r", 3, 2, 4))
    locations = result["groups"][0]["locations"]
    for location in locations:
        location["rows"].pop("S8")
        location["rows"].pop("L8")
    locations[1]["rows"]["F"]["native_logits_finite"] = False
    row = overall(summarize(result))
    assert row["selected_locations"] == 3 and row["valid_locations"] == 0
    assert row["unattempted_locations"] == 2 and row["invalid_locations"] == 1
    assert row["pairs_with_missing_arms"] == 3 and row["pairs_with_observed_invalid_arms"] == 1


def test_campaign_and_group_measured_wall_times_are_explicit_cost_records(tmp_path):
    result = result_fixture()
    result["actual_wall_s"] = 42.25
    result["groups"][0]["actual_wall_s"] = 40.125
    result["costs"] = {}
    analysis = write_reports(result, tmp_path)
    campaign = next(row for row in analysis["costs"] if row["scope"] == "campaign")
    group = next(row for row in analysis["costs"] if row["scope"] == "group")
    assert campaign["actual_wall_s"] == 42.25 and campaign["costs"] == {}
    assert group["actual_wall_s"] == 40.125 and group["costs"]["loading_seconds"] == 1.25
    assert "actual_wall_s" in (tmp_path / "costs.csv").read_text()


def test_group_level_state_failure_excludes_all_its_completed_pairs():
    result = result_fixture()
    result["groups"][0]["validity_errors"] = ["state mutation discovered at close"]
    analysis = summarize(result)
    row = overall(analysis)
    assert row["valid_locations"] == 0 and row["invalid_locations"] == 1
    assert all(pair["useful_first_error_fix"] is None for pair in analysis["paired_blocks"])
    assert not any(position["valid"] for position in analysis["paired_positions"])


def test_zero_round_requests_remain_missing_not_zero_effect():
    analysis = summarize(result_fixture(("no_rounds", 0, 0, 0), ("valid", 1, 2, 4)))
    row = overall(analysis)
    assert row["selected_requests"] == 2 and row["requests_with_no_selected_rounds"] == 1
    assert row["mean_correct_prefix_gain"] == 2 and row["mean_correct_prefix_gain_request_denominator"] == 1
    empty = next(row for row in analysis["request_summaries"] if row["request_id"] == "no_rounds")
    assert empty["mean_correct_prefix_gain"] is None and empty["selected_locations"] == 0


def test_source_controls_are_never_pooled_with_heldout():
    result = result_fixture()
    request, locations = request_locations("source", 1, 0, 8, population="source")
    result["groups"][0]["requests"].append(request)
    result["groups"][0]["locations"].extend(locations)
    analysis = summarize(result)
    assert overall(analysis)["mean_correct_prefix_gain"] == 2
    assert overall(analysis, population="source")["mean_correct_prefix_gain"] == 8
    assert all(row["population"] in ("source", "heldout") for row in analysis["overall"])


@pytest.mark.parametrize("mutation", ["anchor", "start", "request", "missing"])
def test_duplicate_or_missing_selected_evidence_cannot_change_denominators(mutation):
    result = result_fixture()
    group = result["groups"][0]
    if mutation in ("anchor", "start"):
        new = deepcopy(group["locations"][0])
        if mutation == "start":
            new["anchor"]["anchor_id"] = "different_label_same_start"
        group["locations"].append(new)
    elif mutation == "request":
        group["requests"].append(deepcopy(group["requests"][0]))
    else:
        group["locations"] = []
    with pytest.raises(ValueError):
        summarize(result)


def test_write_reports_retains_raw_invalid_nonfinite_and_costs(tmp_path):
    result = result_fixture()
    result["groups"][0]["locations"][0]["rows"]["S8"]["positions"][0]["native_canonical_log_probability"] = float("nan")
    analysis = write_reports(result, tmp_path)
    assert overall(analysis)["invalid_locations"] == 1
    saved = json.loads((tmp_path / "REPORT.json").read_text())
    assert saved["result"]["groups"][0]["locations"][0]["rows"]["S8"]["positions"][0]["native_canonical_log_probability"] == "NaN"
    assert saved["analysis"]["costs"][1]["costs"]["loading_seconds"] == 1.25
    text = (tmp_path / "REPORT.md").read_text()
    assert "not saved verification calls or speedup" in text
    assert "cannot locate the previous response-level KL reduction" in text
    assert "full nine-token block ending exactly at that cap" in text
    assert (tmp_path / "paired_positions.csv").exists() and (tmp_path / "request_summaries.csv").exists()
    assert (tmp_path / "raw_native_rows.json").exists()


def test_retained_calls_are_independent_and_missing_is_unavailable():
    result = result_fixture()
    result["groups"][0]["locations"][0]["rows"] = {}
    analysis = summarize(result)
    hist = next(row for row in analysis["retained_measurement_summaries"] if row["arm"] == "S8" and row["population"] == "heldout")
    assert hist["verification_calls"] == 11 and overall(analysis)["valid_locations"] == 0
    missing = next(row for row in analysis["retained_measurement_summaries"] if row["population"] == "source")
    assert missing["verification_calls"] is None


def test_full_retained_source_campaign_is_separate_from_selected_controls():
    result = result_fixture()
    request, locations = request_locations("source_control", 1, 0, 8, population="source")
    result["groups"][0]["requests"].append(request)
    result["groups"][0]["locations"].extend(locations)
    result["plan"] = {"groups": [dict(group_id="g", domain="gsm8k", requests=deepcopy(result["groups"][0]["requests"]),
        retained_campaign_measurements={"source": [dict(request_id=f"source_{i}", retained_measurements=request["retained_measurements"])
                                                    for i in range(4)]})]}
    analysis = summarize(result)
    campaign = next(row for row in analysis["retained_campaign_summaries"] if row["population"] == "source" and row["arm"] == "F")
    controls = next(row for row in analysis["retained_measurement_summaries"] if row["population"] == "source" and row["arm"] == "F")
    assert campaign["verification_calls"] == 40 and campaign["recorded_requests"] == 4
    assert controls["verification_calls"] == 10 and controls["selected_requests"] == 1


def test_report_rejects_changed_sealed_request_and_excludes_changed_anchor():
    result = result_fixture()
    result["plan"] = {"groups": [dict(group_id="g", requests=deepcopy(result["groups"][0]["requests"]))]}
    result["groups"][0]["requests"][0]["request_id"] = "other"
    with pytest.raises(ValueError, match="sealed plan"):
        summarize(result)
    result = result_fixture()
    result["groups"][0]["requests"][0]["anchors"] = deepcopy(result["groups"][0]["requests"][0]["anchors"])
    location = result["groups"][0]["locations"][0]
    location["anchor"] = deepcopy(location["anchor"])
    location["anchor"]["unsealed_extra"] = True
    assert overall(summarize(result))["valid_locations"] == 0
