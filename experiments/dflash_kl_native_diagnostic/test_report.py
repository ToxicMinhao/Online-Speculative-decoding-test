"""Report arithmetic and exclusion tests, independent of GPU artifact fixtures."""
import copy
import json

import pytest

from . import report

REAL_VALIDATE = report._validate


def pair(request_id="r0", *, population="source", group_id="gsm8k_00", domain="gsm8k",
         f_calls=10, s_calls=8, f_loss=2.0, s_loss=1.0):
    value = dict(request_id=request_id, group_id=group_id, domain=domain, population=population,
                 validation={"status": "passed", "errors": []})
    for checkpoint, calls, loss in (("F", f_calls, f_loss), ("S8", s_calls, s_loss)):
        value[checkpoint] = dict(
            native=dict(verification_calls=calls, output_tokens=100, accepted_draft_tokens=99-calls,
                        verification_committed_tokens=99, initial_output_tokens=1, cap_hit=False,
                        finish_reason="stop", checkpoint_identity_sha256=checkpoint,
                        request_s=1000.0 if checkpoint == "F" else 0.001),
            kl=dict(loss=loss, eligible_positions=10, zero_supervision=False,
                    unique_supervised_positions=10, normalization_denominator=10,
                    anchor_positions=[2, 3], loss_mask_sha256="mask", weights_sha256="weights"))
    return value


def campaign(*pairs, status="complete"):
    groups = {}
    for value in pairs:
        group = groups.setdefault(value["group_id"], dict(group_id=value["group_id"], domain=value["domain"],
                                 status="complete", source=[], heldout=[], costs={"teacher_preparation_s": 2., "loss_forward_s": 3.}))
        group[value["population"]].append(value)
    return dict(status=status, groups=list(groups.values()), actual_wall_s=10.)


@pytest.fixture(autouse=True)
def isolated_validation(monkeypatch):
    # Unit tests below exercise report arithmetic. Real shared-validator wiring has
    # a separate integration test once the controller's complete pair fixture exists.
    def fake_validate(value):
        errors = []
        if value.get("validation") != {"status": "passed", "errors": []}:
            errors.append("recorded_failure")
        if value.get("reject_during_revalidation"):
            errors.append("recomputed_failure")
        return errors
    monkeypatch.setattr(report, "_validate", fake_validate)


def overall(summary, population):
    return next(row for row in summary["pooled"] if row["scope"] == "overall" and row["population"] == population)


def test_calls_pool_by_counts_not_mean_of_request_reductions():
    summary = report.summarize(campaign(pair("r1", f_calls=10, s_calls=5), pair("r2", f_calls=90, s_calls=90)))
    row = overall(summary, "source")
    assert row["F_verification_calls"] == 100
    assert row["S8_verification_calls"] == 95
    assert row["verification_call_reduction_percent"] == pytest.approx(5.)
    assert row["native_better_requests"] == 1 and row["native_tied_requests"] == 1


def test_loss_average_gives_each_normalized_response_equal_weight():
    first, second = pair("r1", f_loss=1., s_loss=.8), pair("r2", f_loss=9., s_loss=4.2)
    first["F"]["kl"]["normalization_denominator"] = first["S8"]["kl"]["normalization_denominator"] = 1.
    second["F"]["kl"]["normalization_denominator"] = second["S8"]["kl"]["normalization_denominator"] = 1000.
    row = overall(report.summarize(campaign(first, second)), "source")
    assert row["F_mean_kl"] == 5. and row["S8_mean_kl"] == 2.5
    assert row["kl_reduction_percent"] == 50.


def test_source_and_heldout_are_never_pooled_together():
    summary = report.summarize(campaign(pair("s", f_calls=10, s_calls=5), pair("h", population="heldout", f_calls=90, s_calls=99)))
    assert overall(summary, "source")["verification_calls_saved"] == 5
    assert overall(summary, "heldout")["verification_calls_saved"] == -9
    assert all(row["population"] in report.POPULATIONS for row in summary["pooled"])


def test_zero_supervision_excludes_only_loss_claim_and_keeps_native_pair():
    value = pair()
    for checkpoint in ("F", "S8"):
        value[checkpoint]["kl"].update(loss=None, eligible_positions=0, zero_supervision=True,
                                      normalization_denominator=1e-6)
    summary = report.summarize(campaign(value))
    row = overall(summary, "source")
    assert row["valid_requests"] == 1 and row["kl_pairs"] == 0 and row["zero_supervision_pairs"] == 1
    assert row["F_mean_kl"] is None and row["verification_calls_saved"] == 2


def test_rejected_pair_cannot_reenter_even_with_passed_recorded_status():
    good, invalid = pair("good"), pair("bad")
    invalid["reject_during_revalidation"] = True
    summary = report.summarize(campaign(good, invalid))
    row = overall(summary, "source")
    assert row["valid_requests"] == 1 and row["invalid_requests"] == 1
    bad = summary["comparisons"][1]
    assert bad["F_verification_calls"] == 10 and bad["S8_loss"] == 1.
    assert bad["verification_calls_saved"] is None and bad["kl_change_S8_minus_F"] is None


def test_recorded_failed_validation_is_preserved():
    value = pair()
    value["validation"] = {"status": "failed", "errors": ["output_mismatch"]}
    assert not report.comparison(value)["valid"]


def test_duplicates_invalidate_both_instead_of_double_counting():
    value = pair()
    summary = report.summarize(campaign(value, copy.deepcopy(value)))
    row = overall(summary, "source")
    assert row["valid_requests"] == 0 and row["invalid_requests"] == 2
    assert all("duplicate_request_pair" in row["invalid_reasons"] for row in summary["comparisons"])


def test_population_or_group_identity_mismatch_is_excluded():
    data = campaign(pair())
    data["groups"][0]["source"][0]["population"] = "heldout"
    result = report.summarize(data)
    assert result["comparisons"][0]["valid"] is False
    assert "pair_group_or_population_identity_mismatch" in result["comparisons"][0]["invalid_reasons"]


def test_null_and_harmful_results_remain_visible():
    summary = report.summarize(campaign(pair("w", s_calls=12), pair("t", s_calls=10)))
    row = overall(summary, "source")
    assert row["native_worse_requests"] == 1 and row["native_tied_requests"] == 1
    assert row["verification_call_reduction_percent"] == pytest.approx(-10)


def test_all_four_evidence_cells_are_explicit():
    summary = report.summarize(campaign(pair("source"), pair("heldout", population="heldout")))
    source, heldout = summary["comparisons"]
    assert (source["F_native_origin"], source["S8_native_origin"]) == ("retained", "fresh")
    assert source["F_kl_origin"] == source["S8_kl_origin"] == "retained"
    assert heldout["F_native_origin"] == heldout["S8_native_origin"] == "retained"
    assert heldout["F_kl_origin"] == heldout["S8_kl_origin"] == "fresh"


def test_no_historical_fresh_timing_comparison_is_produced(tmp_path):
    summary = report.write_reports(campaign(pair()), tmp_path)
    assert all("request_s" not in row and "request_time_reduction_percent" not in row for row in summary["comparisons"] + summary["pooled"])
    assert summary["costs"][0]["components"] == {"teacher_preparation_s": 2., "loss_forward_s": 3.}
    text = (tmp_path / "REPORT.md").read_text()
    assert "not a latency comparison" in text and "does not establish a serving-time improvement" in text
    assert "No latency-reduction or serving-speedup estimate" in text


def test_interpretation_does_not_claim_underlying_cause(tmp_path):
    summary = report.write_reports(campaign(pair("source", s_calls=10), pair("heldout", population="heldout", s_calls=12)), tmp_path)
    descriptions = " ".join(summary["interpretations"][0]["patterns"])
    assert "mismatch in this setup" in descriptions
    assert "unproven even on these source requests" in descriptions
    assert "not the underlying cause" in summary["interpretation_boundary"]
    assert "forward KL is generally unsuitable" in summary["interpretation_boundary"]


def test_report_files_empty_status_and_json_safety(tmp_path):
    summary = report.write_reports({"groups": [], "status": "planned"}, tmp_path)
    assert summary["status"] == "not_executed"
    assert {"SUMMARY.json", "PAIRED_REQUESTS.csv", "POOLED.csv", "COSTS.csv", "REPORT.md"} == {path.name for path in tmp_path.iterdir()}
    assert "No diagnostic measurements have been executed" in (tmp_path / "REPORT.md").read_text()
    assert json.loads((tmp_path / "SUMMARY.json").read_text())["status"] == "not_executed"


def test_nonfinite_raw_evidence_is_explicit_valid_json(tmp_path):
    value = pair()
    value["F"]["kl"]["loss"] = float("nan")
    value["reject_during_revalidation"] = True
    report.write_reports(campaign(value), tmp_path)
    saved = json.loads((tmp_path / "SUMMARY.json").read_text())
    assert saved["comparisons"][0]["F_loss"] == "NaN"
    assert saved["comparisons"][0]["kl_change_S8_minus_F"] is None


def test_full_campaign_status_requires_all_six_complete_groups():
    rows = []
    for index in range(6):
        domain = ("gsm8k", "codealpaca", "sharegpt")[index // 2]
        for population in report.POPULATIONS:
            rows.extend(pair(f"{index}_{population}_{j}", population=population, group_id=f"{domain}_{index % 2}", domain=domain) for j in range(4))
    data = campaign(*rows)
    assert report.summarize(data)["status"] == "complete"
    data["groups"][-1]["status"] = "failed"
    assert report.summarize(data)["status"] == "partial_or_failed"


def test_report_does_not_mutate_private_raw_records_or_publish_text():
    value = pair()
    value["F"]["native"].update(input_ids=[1, 2], output_ids=[3, 4], text="private source text")
    before = copy.deepcopy(value)
    summary = report.summarize(campaign(value))
    assert value == before
    assert "private source text" not in json.dumps(summary)


@pytest.mark.parametrize("mutation", ["none", "counter", "anchors", "recorded_failure"])
def test_shared_validator_integration(monkeypatch, mutation):
    from .test_experiment import pair_fixture

    monkeypatch.setattr(report, "_validate", REAL_VALIDATE)
    value = pair_fixture("heldout")
    if mutation == "counter":
        value["S8"]["native"]["accepted_draft_tokens"] = 99
    elif mutation == "anchors":
        value["S8"]["kl"]["anchor_positions"][0] += 1
    elif mutation == "recorded_failure":
        value["validation"] = {"status": "failed", "errors": ["execution rejected measurement"]}
    summary = report.summarize(campaign(value))
    row = summary["comparisons"][0]
    assert row["valid"] is (mutation == "none")
    assert row["kl_valid"] is (mutation == "none")
    if mutation != "none":
        assert row["invalid_reasons"] and row["verification_calls_saved"] is None
        assert overall(summary, "heldout")["valid_requests"] == 0


def test_complete_controller_output_can_be_reported_without_loosening_validation(monkeypatch):
    from .experiment import run_campaign
    from .test_experiment import FakeBackend, groups_fixture

    monkeypatch.setattr(report, "_validate", REAL_VALIDATE)
    summary = report.summarize(run_campaign(groups_fixture(), lambda _: FakeBackend()))
    assert summary["status"] == "complete"
    assert len(summary["comparisons"]) == 48
    assert overall(summary, "source")["valid_requests"] == 24
    assert overall(summary, "heldout")["kl_pairs"] == 24
