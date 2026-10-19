"""Synthetic-only report tests; no retained scientific evidence is executed."""
import csv
import json

import pytest

from experiments.dflash_osd_attribution.report import summarize_diagnostic, write_report


def decision(position, classification, *, valid=True, repair=True, eligible=True):
    return dict(anchor_id="a", stream="wildchat_00", turn=20, absolute_output_position=position,
        block_position=position, valid=valid, repair=repair, introduced_error=False,
        frozen_native_margin=-0.125, osd_native_margin=0.25,
        capacity=dict(eligible=eligible, classification=classification, deficit=1.0, bound=0.1,
                      tolerance=1.1e-7, native_margin_used_for_classification=False))


def comparison(anchor_id="a", *, valid=True):
    rows = [decision(1, "excluded"), decision(2, "borderline"), decision(3, "not_excluded"),
            decision(4, "not_evaluated", eligible=False), decision(5, "excluded", valid=False)]
    for row in rows:
        row["anchor_id"] = anchor_id
    return dict(anchor_id=anchor_id, stream="wildchat_00", turn=20, valid=valid,
        invalid_reasons=[] if valid else ["frozen reproduction mismatch"],
        frozen_correct_prefix_length=1, osd_correct_prefix_length=3,
        correct_prefix_gain=2 if valid else None, decisions=rows)


def test_bound_denominator_preserves_missing_and_borderline():
    result = summarize_diagnostic([comparison()])
    total = result["summary"][-1]
    assert total["valid_repair_denominator"] == 4
    assert total["evaluated_repair_denominator"] == 3
    assert total["mathematically_excluded"] == 1
    assert total["borderline_inconclusive"] == 1
    assert total["not_excluded"] == 1
    assert total["bound_missing_or_unavailable"] == 1
    assert total["excluded_fraction_of_all_valid_repairs"] == .25
    assert total["excluded_fraction_of_evaluated_repairs"] == pytest.approx(1/3)


def test_invalid_frozen_reproduction_overrides_child_flags():
    result = summarize_diagnostic([comparison(valid=False)])
    assert result["summary"][-1]["invalid_anchors"] == 1
    assert result["summary"][-1]["valid_repair_denominator"] == 0
    assert result["summary"][-1]["mathematically_excluded"] == 0
    assert not any(row["repair"] for row in result["decisions"])


def test_duplicate_anchors_cannot_inflate_denominators():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_diagnostic([comparison(), comparison()])


def test_overlapping_anchors_are_not_reported_as_distinct_positions():
    result = summarize_diagnostic([comparison("a"), comparison("b")])
    total = result["summary"][-1]
    assert total["valid_repair_denominator"] == 8
    assert total["distinct_repaired_output_positions"] == 4


def test_report_retains_native_margins_separately_and_discloses_missing(tmp_path):
    inventory = dict(cases=[dict(stream="personamem_persona42", status="unavailable",
                               issues=["exact checkpoint missing"])])
    result = write_report(tmp_path, diagnostic=[comparison()], inventory=inventory)
    assert result["analysis"] is None
    text = (tmp_path/"REPORT.md").read_text()
    assert "Native campaign analysis is unavailable" in text
    assert "exact checkpoint missing" in text
    assert "radius constrains the combined effective" in text
    assert "Native BF16 canonical margins" in text
    assert "converted into a percentage of total time saved" in text
    rows = list(csv.DictReader((tmp_path/"diagnostic_decisions.csv").open()))
    assert rows[0]["frozen_native_margin"] == "-0.125"
    assert rows[0]["capacity_deficit"] == "1.0"
    assert rows[0]["capacity_bound"] == "0.1"
    assert rows[0]["capacity_native_margin_used_for_classification"] == "False"
    assert json.loads((tmp_path/"REPORT.json").read_text())["diagnostic"] == [comparison()]


def test_missing_diagnostic_is_unavailable_not_zero_effect(tmp_path):
    result = write_report(tmp_path)
    assert result["diagnostic"] is None
    text = (tmp_path/"REPORT.md").read_text()
    assert "No local diagnostic comparisons are available" in text
    assert "no correction-space conclusion is justified" in text
    assert "capacity bound" not in text


def test_native_repeat_rows_and_signed_gains_preserved(tmp_path):
    aggregate = dict(group="wildchat_00", requests=1, valid_requests=1, saved_request_s=-1.0,
        request_time_reduction_percent=-10.0, saved_verification_calls=-2,
        verification_call_reduction_percent=-5.0, share_of_net_saved_time_percent=100.0,
        share_of_net_saved_calls_percent=100.0)
    repeats = [dict(episode_id="wildchat_00", arm=arm, repeat=i, request_s=10+i)
               for arm in ("F", "OSD") for i in range(3)]
    analysis = dict(overall=aggregate, conversations=[aggregate], sources=[aggregate],
        timing_repeats=repeats, pattern_summary=[], costs=[])
    result = write_report(tmp_path, analysis=analysis)
    assert result["analysis"]["timing_repeats"] == repeats
    rows = list(csv.DictReader((tmp_path/"timing_repeats.csv").open()))
    assert len(rows) == 6 and sorted({r["repeat"] for r in rows}) == ["0", "1", "2"]
    text = (tmp_path/"REPORT.md").read_text()
    assert "-10.000" in text
    assert "Teacher time is already included in training_s" in text
    assert "No local diagnostic comparisons are available" in text
    assert "lexical heuristics" in text


def test_no_exclusion_does_not_claim_learnability(tmp_path):
    row = comparison()
    row["decisions"] = [decision(1, "not_excluded")]
    write_report(tmp_path, diagnostic=dict(comparisons=[row], diagnostic_seconds=1.25))
    text = (tmp_path/"REPORT.md").read_text()
    assert "does not identify insufficient pairwise capacity" in text
    assert "learnability, joint" in text
    assert "full-backbone training" in text
    assert json.loads((tmp_path/"REPORT.json").read_text())["diagnostic"]["diagnostic_seconds"] == 1.25


def test_report_missing_tokenizer_and_stream_keeps_planned_denominator(tmp_path):
    aggregate = dict(group="all_available", requests=20, valid_requests=19)
    analysis = dict(overall=aggregate, conversations=[], sources=[], pattern_summary=[], costs=[],
        planned_requests=94, tokenizer=dict(status="unavailable", reason="missing pinned vocabulary"),
        missing_streams=[dict(stream="wildchat_01", reason="source RESULT missing")])
    write_report(tmp_path, analysis=analysis)
    text = (tmp_path/"REPORT.md").read_text()
    assert "19 valid paired requests out of 20 supplied requests" in text
    assert "94 planned requests" in text
    assert "Lexical categorization is unavailable" in text
    assert "No substitute tokenizer was used" in text
    assert "source RESULT missing" in text
    rows = list(csv.DictReader((tmp_path/"missing_streams.csv").open()))
    assert rows == analysis["missing_streams"]
