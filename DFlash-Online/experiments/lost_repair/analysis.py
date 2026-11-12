"""CPU census, training exposure, and paper-only H/M feasibility matching.

One natural start is identified by response identity and the absolute canonical
output index of its known seed. Training occurrences additionally carry a
presentation ID. We never infer independent support from repeated presentations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math


MATCH_SEED = 20261114
PRIORITIES = ("high", "low", "ineligible")
# Same native K+ mass contract, reproduced here without importing model code.
MASS_RTOL, MASS_ATOL = 2e-6, 2e-7


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def anchor_key(row):
    return str(row["request_id"]), int(row["canonical_start"])


def stable_key(request_id, canonical_start, seed=MATCH_SEED):
    """Same key orders H and breaks equally distant M candidates."""
    payload = json.dumps(
        [int(seed), str(request_id), int(canonical_start)],
        ensure_ascii=False, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def classify_start(row):
    """Apply the historical definitions to EOS/cap-clipped native prefixes.

    ``genuine_k6_rejection`` must be established from the saved native proposal
    and canonical token, rather than from a possibly truncated acceptance count.
    Missing or invalid replays stay in the census with no H/M/converse label.
    """
    result = dict(row)
    result.update(
        history="missing_replay", H=False, M=False, converse=False,
        j=None, previously_attained_then_lost=None,
        k6_earlier_break_vs_k4=None, k6_earlier_break_vs_k5=None,
        k4_earlier_break_vs_k6=None, k5_earlier_break_vs_k6=None,
    )
    if any(row.get(name) is None for name in ("a4", "a5", "a6")):
        return result
    if not row.get("parity_valid", False):
        result["history"] = "invalid_replay"
        return result
    valid = int(row["valid_proposals"])
    a4, a5, a6 = (int(row[name]) for name in ("a4", "a5", "a6"))
    if not 0 <= valid <= 15 or any(not 0 <= a <= valid for a in (a4, a5, a6)):
        raise ValueError("Native accepted prefixes exceed canonical EOS/cap bounds")
    genuine = bool(row["genuine_k6_rejection"])
    if genuine and a6 >= valid:
        raise ValueError("Genuine K+6 rejection requires a valid canonical slot")
    genuine4 = bool(row.get("genuine_k4_rejection", a4 < valid))
    genuine5 = bool(row.get("genuine_k5_rejection", a5 < valid))
    if (genuine4 and a4 >= valid) or (genuine5 and a5 >= valid):
        raise ValueError("Older genuine rejection requires a valid canonical slot")
    result.update(
        j=a6 + 1 if genuine else None,
        H=genuine and genuine4 and a4 == a6 and a5 > a6,
        M=genuine and genuine4 and genuine5 and a4 == a5 == a6,
        converse=genuine4 and genuine5 and a4 == a5 and a6 > a4,
        converse_j=a4 + 1 if genuine4 and genuine5 and a4 == a5 and a6 > a4 else None,
        converse_prefix_gain=a6 - a4 if genuine4 and genuine5 and a4 == a5 and a6 > a4 else 0,
        k6_earlier_break_vs_k4=a6 < a4,
        k6_earlier_break_vs_k5=a6 < a5,
        k4_earlier_break_vs_k6=a4 < a6,
        k5_earlier_break_vs_k6=a5 < a6,
    )
    result["history"] = (
        "H" if result["H"] else "M" if result["M"] else
        "other_rejection_history" if genuine else "all_correct_or_terminal"
    )
    if result["H"]:
        result["previously_attained_then_lost"] = a5 - a6
    return result


def match_occurrences(rows, *, seed=MATCH_SEED):
    """Pair H/M without replacement within presentation/response/j/priority.

    The no-successor ``ineligible`` class is retained as its own stratum. Its
    measured weight still uses native K+'s low coefficient; it is neither
    merged with wrong-valid-successor rows nor omitted from denominators.
    """
    strata = defaultdict(lambda: {"H": [], "M": []})
    for row in rows:
        if row["matched_history"] not in ("H", "M"):
            continue
        key = (row["presentation_id"], str(row["request_id"]), row["j"], row["priority"])
        strata[key][row["matched_history"]].append(row)
    pairs = []
    for key, groups in strata.items():
        remaining = list(groups["M"])
        ordered_h = sorted(groups["H"], key=lambda r: stable_key(*anchor_key(r), seed))
        for h in ordered_h:
            if not remaining:
                break
            m = min(remaining, key=lambda r: (
                abs(r["wrong_margin"] - h["wrong_margin"]),
                stable_key(*anchor_key(r), seed),
            ))
            remaining.remove(m)
            pairs.append(dict(
                presentation_id=key[0], request_id=key[1], j=key[2], priority=key[3],
                H_start=h["canonical_start"], M_start=m["canonical_start"],
                H_margin=h["wrong_margin"], M_margin=m["wrong_margin"],
                margin_difference=h["wrong_margin"] - m["wrong_margin"],
                absolute_margin_difference=abs(h["wrong_margin"] - m["wrong_margin"]),
                H_weight=h["first_error_weight"], M_weight=m["first_error_weight"],
                projected_weight=0.25 * min(h["first_error_weight"], m["first_error_weight"]),
            ))
    return pairs


def _support(rows):
    return dict(
        occurrences=len(rows),
        distinct_anchors=len({anchor_key(r) for r in rows}),
        distinct_requests=len({str(r["request_id"]) for r in rows}),
    )


def analyze(
    census_rows, training_rows, presentation_rows, scheduled_request_ids, *,
    census_complete, training_complete, expected_natural_starts=None,
):
    """Return report-ready tables and a complete/incomplete feasibility verdict.

    ``training_rows`` includes EVERY sampled block occurrence, not only joined
    H/M starts. Each row supplies j (or None), priority, first_error_weight and
    wrong_margin (or None with no error). ``presentation_rows`` has one native
    selected-weight denominator and expected_anchor_count per presentation,
    including empty ones.
    It is the full scheduled two-pass event denominator, never a matched subset.
    """
    request_ids = [str(x) for x in scheduled_request_ids]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Scheduled feedback identities must be distinct")
    request_set = set(request_ids)
    census = [classify_start(row) for row in census_rows]
    lookup = {anchor_key(r): r for r in census}
    if len(lookup) != len(census):
        raise ValueError("Duplicate natural canonical start in census")
    if any(r["request_id"] not in request_set for r in census):
        raise ValueError("Census contains an unscheduled request")
    presentations = {r["presentation_id"]: dict(r) for r in presentation_rows}
    if len(presentations) != len(presentation_rows):
        raise ValueError("Duplicate presentation denominator")
    if any(str(r["request_id"]) not in request_set for r in presentation_rows):
        raise ValueError("Presentation contains an unscheduled request")
    for row in presentation_rows:
        if not math.isfinite(row["selected_weight"]) or row["selected_weight"] < 0:
            raise ValueError("Selected weight must be finite and nonnegative")

    occurrences = []
    seen = set()
    for supplied in training_rows:
        row = dict(supplied)
        identity = (row["presentation_id"], *anchor_key(row))
        if identity in seen:
            raise ValueError("Duplicate sampled anchor within one presentation")
        seen.add(identity)
        presentation = presentations[row["presentation_id"]]
        if str(row["request_id"]) != str(presentation["request_id"]):
            raise ValueError("Training anchor and presentation response differ")
        weight = row["first_error_weight"]
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Native first-error weight must be finite and nonnegative")
        if row["j"] is not None:
            if row["priority"] not in PRIORITIES or not 1 <= row["j"] <= 15:
                raise ValueError("Training first-error slot or priority is invalid")
            margin = row["wrong_margin"]
            if margin is None or not math.isfinite(margin) or margin < 0:
                raise ValueError("Wrong-top-1 minus canonical margin must be finite/nonnegative")
        elif weight != 0:
            raise ValueError("No-error block cannot contribute first-error weight")
        parent = lookup.get(anchor_key(row))
        same_j = bool(parent and parent.get("j") is not None and row["j"] == parent["j"])
        row.update(
            natural_history=parent["history"] if parent else "not_a_natural_start",
            same_first_error_j=same_j,
            matched_history=parent["history"] if same_j and parent["history"] in ("H", "M") else None,
        )
        occurrences.append(row)
    pairs = match_occurrences(occurrences)
    matched_keys = {(p["presentation_id"], p["request_id"], p["H_start"]) for p in pairs}
    matched_m_keys = {(p["presentation_id"], p["request_id"], p["M_start"]) for p in pairs}
    m_strata = {
        (r["presentation_id"], str(r["request_id"]), r["j"], r["priority"])
        for r in occurrences if r["matched_history"] == "M"
    }
    for row in occurrences:
        key = (row["presentation_id"], *anchor_key(row))
        row["matched_H"] = key in matched_keys
        row["matched_M"] = key in matched_m_keys
        stratum = (row["presentation_id"], str(row["request_id"]), row["j"], row["priority"])
        row["comparable_M_available"] = row["matched_history"] == "H" and stratum in m_strata
        row["unmatched_H_reason"] = (
            "no_comparable_M" if not row["comparable_M_available"] else "M_exhausted_without_replacement"
        ) if row["matched_history"] == "H" and not row["matched_H"] else None
    h_natural = [r for r in census if r["H"]]
    h_scheduled = [r for r in occurrences if r["natural_history"] == "H"]
    h_agreed = [r for r in occurrences if r["matched_history"] == "H"]
    h_comparable = [r for r in h_agreed if r["comparable_M_available"]]
    h_matched = [r for r in h_agreed if r["matched_H"]]
    m_agreed = [r for r in occurrences if r["matched_history"] == "M"]

    coverage = []
    by_anchor = defaultdict(list)
    for row in occurrences:
        by_anchor[anchor_key(row)].append(row)
    for row in h_natural:
        rows = by_anchor[anchor_key(row)]
        coverage.append(dict(
            request_id=row["request_id"], canonical_start=row["canonical_start"], j=row["j"],
            a4=row["a4"], a5=row["a5"], a6=row["a6"],
            previously_attained_then_lost=row["previously_attained_then_lost"],
            scheduled_occurrences=len(rows),
            same_j_occurrences=sum(r["same_first_error_j"] for r in rows),
            comparable_M_occurrences=sum(r["comparable_M_available"] for r in rows),
            matched_occurrences=sum(r["matched_H"] for r in rows),
            native_H_weight=sum(r["first_error_weight"] for r in rows if r["matched_history"] == "H"),
            priorities=sorted({r["priority"] for r in rows if r["matched_history"] == "H"}),
        ))
    occurrence_by_presentation = defaultdict(list)
    pairs_by_presentation = defaultdict(list)
    for row in occurrences:
        occurrence_by_presentation[row["presentation_id"]].append(row)
    for row in pairs:
        pairs_by_presentation[row["presentation_id"]].append(row)
    weight_rows = []
    for pid, presentation in presentations.items():
        rows = occurrence_by_presentation[pid]
        serialized_weight = sum(r["first_error_weight"] for r in rows)
        native_weight = presentation.get("first_error_weight")
        mass_checked = native_weight is not None and presentation.get("expected_anchor_count") == len(rows)
        if native_weight is not None and (not math.isfinite(native_weight) or native_weight < 0):
            raise ValueError("Native response first-error denominator must be finite and nonnegative")
        if mass_checked and abs(serialized_weight - native_weight) > MASS_ATOL + MASS_RTOL * abs(native_weight):
            raise ValueError("Serialized anchor weights differ from native response first-error mass")
        weight_rows.append(dict(
            **presentation,
            sampled_anchor_occurrences=len(rows),
            first_error_occurrences=sum(r["j"] is not None for r in rows),
            native_first_error_weight=native_weight,
            serialized_anchor_first_error_weight=serialized_weight,
            first_error_mass_consistency_checked=mass_checked,
            first_error_mass_difference=(serialized_weight - native_weight) if native_weight is not None else None,
            H_same_j_occurrences=sum(r["matched_history"] == "H" for r in rows),
            native_H_weight_before_matching=sum(r["first_error_weight"] for r in rows if r["matched_history"] == "H"),
            matched_H_occurrences=len(pairs_by_presentation[pid]),
            projected_matched_weight=sum(p["projected_weight"] for p in pairs_by_presentation[pid]),
        ))
    native_denominators_available = bool(weight_rows) and all(r["native_first_error_weight"] is not None for r in weight_rows)
    first_error_weight = (sum(r["native_first_error_weight"] for r in weight_rows)
                          if native_denominators_available else None)
    serialized_weight = sum(r["serialized_anchor_first_error_weight"] for r in weight_rows)
    selected_weight = sum(r["selected_weight"] for r in weight_rows)
    h_weight = sum(r["native_H_weight_before_matching"] for r in weight_rows)
    projected = sum(p["projected_weight"] for p in pairs)
    matched_support = _support(h_matched)

    missing = [r for r in census if r["history"] == "missing_replay"]
    invalid = [r for r in census if r["history"] == "invalid_replay"]
    present_request_counts = Counter(str(r["request_id"]) for r in presentation_rows)
    count_complete = all(present_request_counts[r] == 2 for r in request_ids)
    row_count_complete = expected_natural_starts is None or len(census) == expected_natural_starts
    census_valid_complete = bool(census_complete and not missing and not invalid
                                and row_count_complete
                                and {str(r["request_id"]) for r in census} == request_set)
    anchor_count_complete = all(
        p.get("expected_anchor_count") == len(occurrence_by_presentation[pid])
        for pid, p in presentations.items()
    )
    full_training = bool(training_complete and count_complete and anchor_count_complete
                         and native_denominators_available)
    full_first_error_weight = first_error_weight if full_training else None
    full_selected_weight = selected_weight if full_training else None
    complete = bool(census_valid_complete and full_training)
    checks = (
        ("distinct_matched_H_anchors", matched_support["distinct_anchors"], 32),
        ("distinct_matched_H_requests", matched_support["distinct_requests"], 16),
        ("projected_weight_fraction_all_native_first_error", ratio(projected, full_first_error_weight), 0.0025),
    )
    gate = {
        name: dict(value=value, threshold=threshold,
                   observed_threshold_met=value is not None and value >= threshold,
                   passed=(value is not None and value >= threshold) if complete else None)
        for name, value, threshold in checks
    }
    decision = "INCOMPLETE" if not complete else (
        "PASS" if all(x["passed"] for x in gate.values()) else "FAIL"
    )
    valid_rejections = [r for r in census if r["history"] in ("H", "M", "other_rejection_history")]
    per_request = []
    for rid in request_ids:
        native = [r for r in census if str(r["request_id"]) == rid]
        cov = [r for r in coverage if str(r["request_id"]) == rid]
        pp = [p for p in pairs if p["request_id"] == rid]
        per_request.append(dict(
            request_id=rid, natural_starts=len(native),
            valid_genuine_k6_rejections=sum(r["history"] in ("H", "M", "other_rejection_history") for r in native),
            H=sum(r["H"] for r in native), M=sum(r["M"] for r in native),
            converse=sum(r["converse"] for r in native),
            H_lost_prefix_occurrences=sum(r["previously_attained_then_lost"] or 0 for r in native),
            missing_replays=sum(r["history"] == "missing_replay" for r in native),
            invalid_replays=sum(r["history"] == "invalid_replay" for r in native),
            H_scheduled_distinct_anchors=sum(r["scheduled_occurrences"] > 0 for r in cov),
            H_same_j_distinct_anchors=sum(r["same_j_occurrences"] > 0 for r in cov),
            H_comparable_M_distinct_anchors=sum(r["comparable_M_occurrences"] > 0 for r in cov),
            H_matched_distinct_anchors=sum(r["matched_occurrences"] > 0 for r in cov),
            native_H_weight_before_matching=sum(r["native_H_weight"] for r in cov),
            projected_matched_weight=sum(p["projected_weight"] for p in pp),
        ))
    margins = [p["margin_difference"] for p in pairs]
    summary = dict(
        status="COMPLETE" if complete else "INCOMPLETE", gate_decision=decision, gate=gate,
        scheduled_requests=len(request_ids), observed_presentations=len(presentation_rows),
        expected_presentations=2 * len(request_ids),
        observed_natural_starts=len(census), expected_natural_starts=expected_natural_starts,
        full_natural_denominator=len(census) if census_valid_complete else None,
        known_unobserved_starts=(max(0, expected_natural_starts - len(census)) if expected_natural_starts is not None else None),
        missing_replays=len(missing), invalid_replays=len(invalid),
        census_complete_declared=bool(census_complete), training_complete_declared=bool(training_complete),
        presentation_coverage_complete=count_complete,
        training_anchor_coverage_complete=anchor_count_complete,
        native_first_error_denominators_available=native_denominators_available,
        history_counts=dict(Counter(r["history"] for r in census)),
        observed_genuine_k6_rejections=len(valid_rejections),
        H_observed_count=len(h_natural),
        H_share_all_natural_starts=ratio(len(h_natural), len(census)) if census_valid_complete else None,
        H_share_genuine_k6_rejections=ratio(len(h_natural), len(valid_rejections)) if census_valid_complete else None,
        H_measured_share_observed_starts=ratio(len(h_natural), len(census)),
        H_measured_share_valid_observed_rejections=ratio(len(h_natural), len(valid_rejections)),
        M_observed_count=sum(r["M"] for r in census),
        converse_observed_count=sum(r["converse"] for r in census),
        H_contributing_requests=len({str(r["request_id"]) for r in h_natural}),
        H_by_depth=dict(sorted(Counter(r["j"] for r in h_natural).items())),
        H_lost_prefix_histogram=dict(sorted(Counter(r["previously_attained_then_lost"] for r in h_natural).items())),
        H_lost_prefix_total=sum(r["previously_attained_then_lost"] for r in h_natural),
        H_largest_request_count=max((r["H"] for r in per_request), default=0),
        H_largest_request_fraction=ratio(max((r["H"] for r in per_request), default=0), len(h_natural)),
        exposure_funnel=dict(native_replay=_support(h_natural), scheduled=_support(h_scheduled),
                             same_j=_support(h_agreed), comparable_M=_support(h_comparable),
                             matched=matched_support),
        M_same_j_support=_support(m_agreed),
        H_unmatched_occurrences=len(h_agreed) - len(pairs),
        H_unmatched_distinct_anchors=len({anchor_key(r) for r in h_agreed} - {anchor_key(r) for r in h_matched}),
        H_unmatched_reasons=dict(Counter(r["unmatched_H_reason"] for r in h_agreed if not r["matched_H"])),
        H_priority_occurrences=dict(Counter(r["priority"] for r in h_agreed)),
        matched_priority_occurrences=dict(Counter(p["priority"] for p in pairs)),
        native_first_error_weight_by_priority={
            priority: sum(r["first_error_weight"] for r in occurrences if r["j"] is not None and r["priority"] == priority)
            for priority in PRIORITIES
        },
        native_H_weight_by_priority={
            priority: sum(r["first_error_weight"] for r in h_agreed if r["priority"] == priority)
            for priority in PRIORITIES
        },
        native_first_error_weight=first_error_weight, selected_weight=selected_weight,
        full_native_first_error_weight=full_first_error_weight, full_selected_weight=full_selected_weight,
        serialized_anchor_first_error_weight=serialized_weight,
        first_error_mass_consistency=dict(rtol=MASS_RTOL, atol=MASS_ATOL,
            checked_presentations=sum(r["first_error_mass_consistency_checked"] for r in weight_rows)),
        native_H_weight_before_matching=h_weight,
        H_weight_fraction_all_first_error=ratio(h_weight, full_first_error_weight),
        H_weight_fraction_all_selected=ratio(h_weight, full_selected_weight),
        H_weight_fraction_observed_first_error=ratio(h_weight, first_error_weight),
        H_weight_fraction_observed_selected=ratio(h_weight, selected_weight),
        projected_matched_weight=projected,
        projected_weight_fraction_all_first_error=ratio(projected, full_first_error_weight),
        projected_weight_fraction_observed_first_error=ratio(projected, first_error_weight),
        margin_balance=dict(
            pairs=len(pairs), mean_signed_difference=ratio(sum(margins), len(margins)),
            mean_absolute_difference=ratio(sum(abs(x) for x in margins), len(margins)),
            maximum_absolute_difference=max((abs(x) for x in margins), default=None),
            mean_H_margin=ratio(sum(p["H_margin"] for p in pairs), len(pairs)),
            mean_M_margin=ratio(sum(p["M_margin"] for p in pairs), len(pairs)),
        ),
        matching_seed=MATCH_SEED, projected_weight_factor=0.25,
        matching_contract="same presentation, response, j and high/low/ineligible class; H hash order; nearest M margin; hash ties; no replacement",
        interpretation="Paper-only feasibility screen. No weights changed; local historical prefixes are not serving speedups.",
    )
    return dict(summary=summary, census_rows=census, anchor_coverage_rows=coverage,
                native_weight_rows=weight_rows, training_anchor_rows=occurrences,
                pairs=pairs, per_request=per_request)
