"""Small independent fixtures for retained-record continuation arithmetic."""

import copy
import json
import math

import pytest

from experiments.continuation_credit_cpu.records import (
    block_roles, continuation_scores, load_records, paired_block, read_receipt, validate_block,
)


def block(pattern, supports=None):
    """C/E describe valid proposals; all remaining slots are padding."""
    labels = list(range(100, 116))
    proposals = labels.copy()
    supports = supports or [0.8] * len(pattern)
    valid = [False] + [True] * len(pattern) + [False] * (15 - len(pattern))
    positions = []
    for j, (symbol, support) in enumerate(zip(pattern, supports), 1):
        correct = symbol == "C"
        margin = 2 * math.atanh(support) if correct else -1.0
        if not correct:
            proposals[j] += 100
        positions.append(dict(position=j, canonical=labels[j], prediction=proposals[j],
            rival=labels[j] + 100, correct=correct, margin=margin,
            support_fp32=support if correct else 0.0, margin_tie=margin == 0))
    result = dict(labels=labels, proposals=proposals, valid=valid,
                  positions=positions, m=len(pattern))
    roles = block_roles(result)
    result["roles"] = dict(labels=labels.copy(), valid=valid.copy(),
        prefix=[0 < j <= roles["L"] for j in range(16)],
        repair=[j == roles["e"] for j in range(16)],
        H=roles["K"] - roles["L"] if roles["has_error"] else 0,
        **{key: roles[key] for key in ("L", "e", "K", "m")})
    return result


@pytest.mark.parametrize("pattern,supports,old,shared", [
    ("ECECC", None, .8, 1.6),
    ("EECC", None, 0, .8),
    ("ECCC", None, 2.4, 2.4),
    ("EEEE", None, 0, 0),
    ("ECEC", [.8, .2, .8, .9], .2, .3),
])
def test_literal_areas(pattern, supports, old, shared):
    data = block(pattern, supports)
    actual_old, actual_shared, _ = continuation_scores(data["positions"], 1)
    assert actual_old * 14 == pytest.approx(old)
    assert actual_shared * 14 == pytest.approx(shared)


def test_correct_tie_keeps_native_role_and_zeroes_future_credit():
    data = block("ECEC", [.8, 0, .8, .9])
    roles = validate_block(data)
    old, shared, followers = continuation_scores(data["positions"], roles["e"])
    assert roles["high"] and roles["r"] == 1
    assert old == shared == 0
    assert [f["w"] for f in followers] == [0, 0]


def test_wrong_tie_still_counts_as_error():
    data = block("ECEC")
    data["positions"][2].update(margin=0, margin_tie=True)
    validate_block(data)
    _, _, followers = continuation_scores(data["positions"], 1)
    assert followers[-1]["error_count"] == 2


def test_outcomes_include_progress_beyond_old_cutoff_and_followers():
    parent = block("CECECC", [.8, .8, .2, .8, .9, .8])
    child = block("CCCCCE")
    row, followers = paired_block(parent, child, response_ordinal=0, record_id="one", anchor=7)
    assert (row["L"], row["e"], row["r"], row["n"], row["K"]) == (1, 2, 1, 4, 3)
    assert row["Y_full"] == 4 and row["Y_old"] == 2
    assert row["beyond_K"] == 2 and row["crossed_next"] and row["useful_repair"]
    assert row["lost_prefix"] == 0 and row["later_error_count"] == 1
    assert [f["position"] for f in followers] == [5, 6]
    assert [f["target"] for f in followers] == [1, 0]
    assert [f["w"] for f in followers] == [.2, .2]
    assert followers[0]["credit"] == pytest.approx(.2 / 2 / 14)
    assert followers[0]["later_mean_deficit"] == followers[0]["later_max_deficit"] == 1


def test_no_error_parent_retains_loss_but_has_no_repair_denominator():
    row, followers = paired_block(block("CCCC"), block("CECC"),
        response_ordinal=0, record_id="one", anchor=7)
    assert row["L"] == row["K"] == row["m"] == 4
    assert row["Y_full"] == row["Y_old"] == -3 and row["lost_prefix"] == 3
    assert row["u_old"] is row["u_shared"] is row["delta_u"] is row["useful_repair"] is None
    assert not row["crossed_next"] and not followers


def test_lost_prefix_is_not_a_useful_repair():
    row, followers = paired_block(block("CCECC"), block("CECCC"),
        response_ordinal=0, record_id="one", anchor=7)
    assert row["lost_prefix"] == 1 and row["Y_full"] == -1
    assert not row["useful_repair"] and not followers


def test_valid_boundary_includes_last_proposal_but_not_padding():
    data = block("CEC")
    data["proposals"][4] += 1  # Excluded padding cannot create a later error.
    roles = validate_block(data)
    assert roles["K"] == roles["m"] == 3 and roles["r"] == 1


def test_noncontiguous_validity_rejected():
    data = block("CCCC")
    data["valid"][2] = False
    with pytest.raises(ValueError, match="contiguous"):
        validate_block(data)


@pytest.mark.parametrize("field,value", [("margin", float("nan")), ("support_fp32", float("inf"))])
def test_nonfinite_preserved_as_failure(field, value):
    data = block("CC")
    data["positions"][0][field] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_block(data)


def test_alignment_rejects_changed_canonical_data():
    parent, child = block("CEC"), block("CCC")
    child["labels"][2] += 1000
    child["proposals"][2] += 1000
    child["positions"][1]["canonical"] += 1000
    child["positions"][1]["prediction"] += 1000
    child["roles"]["labels"] = child["labels"].copy()
    with pytest.raises(ValueError, match="canonical"):
        paired_block(parent, child, response_ordinal=0, record_id="one", anchor=7)


def test_position_and_native_role_disagreements_rejected():
    original = block("CEC")
    changed = copy.deepcopy(original)
    changed["positions"].pop()
    with pytest.raises(ValueError, match="every valid"):
        validate_block(changed)
    changed = copy.deepcopy(original)
    changed["roles"]["L"] = 2
    with pytest.raises(ValueError, match="Saved roles"):
        validate_block(changed)


def test_new_low_confidence_match_can_reduce_shared_score():
    before = block("ECEC", [.8, .8, .8, .8])
    after = block("ECCC", [.8, .8, .01, .8])
    first = continuation_scores(before["positions"], 1)[1]
    second = continuation_scores(after["positions"], 1)[1]
    assert second < first


def test_receipt_requires_original_anchor_order_and_endpoint(tmp_path):
    binding = dict(record_id="one", anchors=list(range(64)))
    receipt = dict(status="passed", endpoint="parent", **binding, blocks=[block("CEC")] * 64)
    path = tmp_path / "response.json"
    path.write_text(json.dumps(receipt))
    assert len(read_receipt(path, binding, endpoint="parent")["blocks"]) == 64
    with pytest.raises(ValueError, match="misaligned"):
        read_receipt(path, binding, endpoint="child")
    receipt["anchors"] = list(reversed(receipt["anchors"]))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="misaligned"):
        read_receipt(path, binding, endpoint="parent")


def test_loader_rejects_incomplete_response_cohort_without_resampling(tmp_path):
    (tmp_path / "CONTRACT.json").write_text(json.dumps(dict(features=[])))
    with pytest.raises(ValueError, match="128 distinct"):
        load_records(tmp_path)
