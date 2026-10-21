"""Reports rederive shared validity and give requests equal weight.

Source controls remain a separate illustrative population. Raw observations,
invalid records, and unattempted selected locations are always retained.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from .protocol import ADAPTED_ARMS, ARMS, POPULATIONS, SCHEMA, compare_anchor, digest, require

BOUNDARY = (
    "Matching length includes a correctly matched EOS; non-EOS length is reported separately. "
    "A fully matching suffix exhausted at the response cap is censored, including a full nine-token "
    "block ending exactly at that cap. Unavailable continuation is never scored. After an earlier "
    "mismatch, canonical-token agreement is only positional evidence and is not a target verification "
    "decision. Local prefix gains are not saved verification calls or speedup. Margins and log "
    "probabilities cannot locate the previous response-level KL reduction; that explanation remains "
    "untested. No efficacy threshold or new mechanism is selected by this report."
)

MEAN_FIELDS = (
    "correct_prefix_gain", "non_eos_prefix_gain", "frozen_correct_prefix_length", "adapted_correct_prefix_length",
    "frozen_non_eos_prefix_length", "adapted_non_eos_prefix_length", "changed_proposal_count_all_nine",
    "changed_proposal_count_scored", "frozen_first_error_fixed", "useful_first_error_fix",
    "first_error_fix_blocked_by_earlier_error", "introduced_errors_within_frozen_prefix",
    "has_introduced_earlier_error", "mean_canonical_log_probability_change", "mean_canonical_margin_change",
)
COUNT_FLAGS = {
    "frozen_first_error_blocks": "frozen_first_error_available",
    "first_error_fixes": "frozen_first_error_fixed",
    "useful_first_error_fixes": "useful_first_error_fix",
    "blocked_first_error_fixes": "first_error_fix_blocked_by_earlier_error",
    "blocks_with_new_earlier_errors": "has_introduced_earlier_error",
    "frozen_censored_blocks": "frozen_prefix_censored",
    "adapted_censored_blocks": "adapted_prefix_censored",
    "frozen_matched_eos_blocks": "frozen_matched_eos",
    "adapted_matched_eos_blocks": "adapted_matched_eos",
}
COUNT_SUMS = {
    "scored_positions": "scored_positions",
    "changed_proposals_all_nine": "changed_proposal_count_all_nine",
    "changed_proposals_scored": "changed_proposal_count_scored",
    "new_errors_within_frozen_prefix": "introduced_errors_within_frozen_prefix",
}


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _mean(values):
    values = [value for value in values if type(value) in (bool, int, float) and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _request_summary(request, pairs, arm, group):
    valid = [pair for pair in pairs if pair["valid"]]
    row = {key: request.get(key, group.get(key)) for key in ("group_id", "domain", "request_id", "population")}
    row.update(arm=arm, source_control=request.get("source_control", request.get("population") == "source"),
               selected_locations=len(pairs), valid_locations=len(valid),
               invalid_locations=sum(pair["status"] == "invalid" for pair in pairs),
               unattempted_locations=sum(pair["status"] == "unattempted" for pair in pairs),
               pairs_with_missing_arms=sum(bool(pair["missing_arms"]) for pair in pairs),
               pairs_with_observed_invalid_arms=sum(bool(pair["invalid_observed_arms"]) for pair in pairs),
               no_selected_rounds=not pairs, group_status=group.get("status"))
    for field in MEAN_FIELDS:
        row["mean_" + field] = _mean(pair.get(field) for pair in valid)
    for outcome in ("gain", "loss", "tie"):
        row["prefix_" + outcome + "_blocks"] = sum(pair["prefix_outcome"] == outcome for pair in valid)
    for key, field in COUNT_FLAGS.items():
        row[key] = sum(pair[field] is True for pair in valid)
    for key, field in COUNT_SUMS.items():
        row[key] = sum(pair[field] for pair in valid)
    rows = [position for pair in valid for position in pair["positions"]]
    for field in ("positional_fix", "positional_new_error", "positional_only_after_earlier_mismatch"):
        row[field + "_positions"] = sum(position[field] for position in rows)
    row["useful_fix_fraction_of_frozen_first_errors"] = _ratio(row["useful_first_error_fixes"], row["frozen_first_error_blocks"])
    row["valid_fraction_of_selected_locations"] = _ratio(len(valid), len(pairs))
    return row


def _aggregate(requests, **identity):
    """Each available request mean receives exactly one vote, never one per block."""
    row = dict(identity, selected_requests=len(requests), requests_with_valid_locations=sum(r["valid_locations"] > 0 for r in requests),
               requests_without_valid_locations=sum(r["valid_locations"] == 0 for r in requests),
               requests_with_no_selected_rounds=sum(r["no_selected_rounds"] for r in requests))
    count_keys = ("selected_locations", "valid_locations", "invalid_locations", "unattempted_locations",
                  "pairs_with_missing_arms", "pairs_with_observed_invalid_arms",
                  "prefix_gain_blocks", "prefix_loss_blocks", "prefix_tie_blocks", *COUNT_FLAGS, *COUNT_SUMS,
                  "positional_fix_positions", "positional_new_error_positions", "positional_only_after_earlier_mismatch_positions")
    for key in count_keys:
        row[key] = sum(request[key] for request in requests)
    for field in MEAN_FIELDS:
        key = "mean_" + field
        row[key] = _mean(request[key] for request in requests)
        row[key + "_request_denominator"] = sum(request[key] is not None for request in requests)
    row["valid_fraction_of_selected_locations"] = _ratio(row["valid_locations"], row["selected_locations"])
    row["useful_fix_fraction_of_frozen_first_errors_raw"] = _ratio(row["useful_first_error_fixes"], row["frozen_first_error_blocks"])
    row["equal_request_mean_useful_fix_fraction_of_frozen_first_errors"] = _mean(request["useful_fix_fraction_of_frozen_first_errors"] for request in requests)
    row["weighting"] = "Mean within each request, then equal weight per request with available valid evidence"
    return row


def _retained(request, group):
    measurements = request.get("retained_measurements", {})
    rows = []
    for arm in ARMS:
        row = {key: request.get(key, group.get(key)) for key in ("group_id", "domain", "request_id", "population")}
        row.update(arm=arm, origin="retained_end_to_end_measurement", source_control=request.get("population") == "source")
        values = measurements.get(arm)
        row["available"] = isinstance(values, dict)
        if isinstance(values, dict):
            row.update(values)
        rows.append(row)
    return rows


def _cost_rows(result):
    counters = dict(result.get("counters", {}))
    for key in ("selected_locations", "planned_arm_evaluations", "attempted_arm_evaluations",
                "actual_draft_block_evaluations", "unattempted_arm_evaluations", "unknown_work_receipts",
                "planned_reconstruction_target_verifications", "reconstruction_target_verifications",
                "reconstruction_target_verify_attempts", "target_verifications", "scored_target_verifications",
                "unknown_reconstruction_work_receipts", "unattempted_reconstruction_target_verifications",
                "regenerated_response_tokens", "training_steps"):
        if key in result:
            counters[key] = result[key]
    rows = [{"scope": "campaign", "costs": result.get("costs", {}), "counters": counters,
             "actual_wall_s": result.get("actual_wall_s")}]
    for group in result.get("groups", []):
        rows.append({"scope": "group", "group_id": group.get("group_id"), "costs": group.get("costs", {}),
                     "actual_wall_s": group.get("actual_wall_s")})
        for location in group.get("locations", []):
            for arm, value in location.get("rows", {}).items():
                if not isinstance(value, dict):
                    continue
                row = {"scope": "native_block", "group_id": group.get("group_id"),
                       "anchor_id": location.get("anchor", {}).get("anchor_id"), "arm": arm,
                       "costs": value.get("costs", {})}
                for key, item in value.items():
                    if key.endswith(("_seconds", "_s", "_bytes")) or key in (
                        "target_prefill_chunks", "native_head_calls", "draft_block_evaluations",
                        "target_verifications", "reconstruction_target_verifications", "reconstruction_target_verify_attempts",
                        "scored_target_verifications", "reconstruction_schedule_sha256", "reconstruction_rounds",
                        "regenerated_response_tokens", "training_steps",
                        "optimizer_creations", "optimizer_loads"):
                        row[key] = item
                rows.append(row)
    return rows


def summarize(result):
    groups = result.get("groups", [])
    require(isinstance(groups, list), "Result groups must be a list")
    group_ids = [group.get("group_id") for group in groups]
    require(len(set(group_ids)) == len(group_ids), "Duplicate group identity")
    blocks, positions, requests, retained, raw_rows = [], [], [], [], []
    seen_anchors, seen_requests, seen_starts = set(), set(), set()
    sealed_groups = {group["group_id"]: group for group in result.get("plan", {}).get("groups", [])}
    if sealed_groups:
        require(set(group_ids) == set(sealed_groups), "Result must preserve every selected group, including unattempted groups")
    for group in groups:
        group_id = group.get("group_id")
        group_requests = group.get("requests", [])
        require(isinstance(group_requests, list), "Group requests must be a list")
        if group_id in sealed_groups:
            require(group_requests == sealed_groups[group_id]["requests"], "Selected requests differ from sealed plan")
        request_map = {}
        for request in group_requests:
            identity = (group_id, request.get("population"), request.get("request_id"))
            require(identity not in seen_requests, "Duplicate request identity")
            require(request.get("population") in POPULATIONS, "Unknown request population")
            seen_requests.add(identity)
            request_map[identity] = request
            retained.extend(_retained(request, group))
        paired_by_request = {(identity, arm): [] for identity in request_map for arm in ADAPTED_ARMS}
        group_errors = list(group.get("validity_errors", []))
        if isinstance(group.get("validation"), dict) and group["validation"].get("status") == "failed":
            group_errors.extend(group["validation"].get("errors", ["group_validation_failed"]))
        for location in group.get("locations", []):
            anchor = location["anchor"]
            identity = (group_id, anchor.get("population"), anchor.get("request_id"))
            require(identity in request_map, "Location has no selected request record")
            require(anchor.get("anchor_id") not in seen_anchors, "Duplicate anchor identity")
            start_key = identity + (anchor.get("block_output_start"),)
            require(start_key not in seen_starts, "Duplicate canonical block start")
            seen_starts.add(start_key)
            seen_anchors.add(anchor.get("anchor_id"))
            rows = location.get("rows", {})
            for arm in ARMS:
                raw_rows.append(dict(anchor_id=anchor.get("anchor_id"), arm=arm, row=rows.get(arm),
                                     group_id=group_id, request_id=anchor.get("request_id"), population=anchor.get("population")))
            for arm in ADAPTED_ARMS:
                frozen, adapted = rows.get("F") or {}, rows.get(arm) or {}
                pair, paired_positions = compare_anchor(anchor, frozen, adapted, arm)
                errors = list(group_errors)
                if anchor.get("group_id") != group_id or anchor.get("domain") != group.get("domain"):
                    errors.append("anchor_group_identity_mismatch")
                request = request_map[identity]
                if "anchors" in request and anchor not in request["anchors"]:
                    errors.append("anchor_differs_from_selected_request")
                for key in ("input_ids", "output_ids"):
                    if key + "_sha256" in request and digest(anchor.get(key)) != request[key + "_sha256"]:
                        errors.append("anchor_request_hash_mismatch:" + key)
                if errors:
                    pair["valid"] = False
                    pair["invalid_reasons"].extend("group:" + str(error) for error in errors)
                    if pair["status"] != "unattempted":
                        pair["status"] = "invalid"
                    for field in (*MEAN_FIELDS, *COUNT_FLAGS.values(), *COUNT_SUMS.values(), "prefix_outcome"):
                        pair[field] = None
                    for position in paired_positions:
                        position["valid"] = False
                        position["invalid_reasons"] = pair["invalid_reasons"]
                        position["useful_first_error_fix"] = False
                pair["group_status"] = group.get("status")
                blocks.append(pair)
                positions.extend(paired_positions)
                paired_by_request[(identity, arm)].append(pair)
        for identity, request in request_map.items():
            for arm in ADAPTED_ARMS:
                pairs = paired_by_request[(identity, arm)]
                expected = request.get("selected_location_count", len(request.get("anchors", pairs)))
                require(expected == len(pairs), "Selected request locations missing from result; preserve unattempted records")
                requests.append(_request_summary(request, pairs, arm, group))
    group_summary, domain_summary, overall = [], [], []
    for arm in ADAPTED_ARMS:
        for population in POPULATIONS:
            subset = [row for row in requests if row["arm"] == arm and row["population"] == population]
            overall.append(_aggregate(subset, arm=arm, population=population, scope="overall"))
            for key, destination in (("group_id", group_summary), ("domain", domain_summary)):
                for value in sorted({row[key] for row in subset}):
                    destination.append(_aggregate([row for row in subset if row[key] == value], arm=arm,
                                                  population=population, scope=key, **{key: value}))
    historical_summary = []
    for population in POPULATIONS:
        for arm in ARMS:
            subset = [row for row in retained if row["population"] == population and row["arm"] == arm]
            available = [row for row in subset if row.get("available") and type(row.get("verification_calls")) is int]
            historical_summary.append(dict(population=population, arm=arm, selected_requests=len(subset),
                available_requests=len(available), verification_calls=sum(row["verification_calls"] for row in available) if available else None,
                output_tokens=sum(row["output_tokens"] for row in available) if available and all(type(row.get("output_tokens")) is int for row in available) else None,
                mean_retained_response_kl=_mean(row.get("retained_response_kl") for row in available),
                scope="Selected source controls only" if population == "source" else "All selected held-out requests",
                origin="Retained end-to-end records; independent of local-prefix outcomes"))
    campaign_rows = []
    for group_id, group in sealed_groups.items():
        for population, items in group.get("retained_campaign_measurements", {}).items():
            require(population in POPULATIONS, "Unknown retained campaign population")
            identities = [item.get("request_id") for item in items]
            require(len(set(identities)) == len(identities), "Duplicate retained campaign request")
            for item in items:
                request = dict(item, group_id=group_id, domain=group["domain"], population=population)
                for row in _retained(request, group):
                    row["source_control"] = False
                    row["origin"] = "retained_full_campaign_measurement"
                    campaign_rows.append(row)
    campaign_summary = []
    for population in POPULATIONS:
        for arm in ARMS:
            subset = [row for row in campaign_rows if row["population"] == population and row["arm"] == arm]
            available = [row for row in subset if row.get("available") and type(row.get("verification_calls")) is int]
            campaign_summary.append(dict(population=population, arm=arm, recorded_requests=len(subset),
                available_requests=len(available), verification_calls=sum(row["verification_calls"] for row in available) if available else None,
                mean_retained_response_kl=_mean(row.get("retained_response_kl") for row in available),
                scope="Full retained campaign; not the six selected source controls",
                origin="Retained end-to-end records; no new complete-response generation"))
    return dict(schema=SCHEMA, status=result.get("status"), paired_blocks=blocks, paired_positions=positions,
                raw_rows=raw_rows, request_summaries=requests, group_summaries=group_summary,
                domain_summaries=domain_summary, overall=overall, retained_measurements=retained,
                retained_measurement_summaries=historical_summary, costs=_cost_rows(result),
                retained_campaign_measurements=campaign_rows, retained_campaign_summaries=campaign_summary,
                counters=_cost_rows(result)[0]["counters"], interpretation_boundary=BOUNDARY,
                group_statuses=[{key: group.get(key) for key in ("group_id", "domain", "status", "error", "validity_errors")}
                                for group in groups])


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_clean(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else _clean(value)
                             for key, value in row.items()})


def _fmt(value):
    return "unavailable" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def write_reports(result, out):
    analysis = summarize(result)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("paired_blocks", analysis["paired_blocks"]), ("paired_positions", analysis["paired_positions"]),
                       ("request_summaries", analysis["request_summaries"]), ("group_summaries", analysis["group_summaries"]),
                       ("domain_summaries", analysis["domain_summaries"]), ("overall", analysis["overall"]),
                       ("retained_measurements", analysis["retained_measurements"]),
                       ("retained_measurement_summaries", analysis["retained_measurement_summaries"]), ("costs", analysis["costs"])):
        _csv(out / (name + ".csv"), rows)
    _csv(out / "retained_campaign_measurements.csv", analysis["retained_campaign_measurements"])
    _csv(out / "retained_campaign_summaries.csv", analysis["retained_campaign_summaries"])
    for name, value in (("REPORT.json", {"result": result, "analysis": analysis}),
                        ("paired_blocks.json", analysis["paired_blocks"]), ("paired_positions.json", analysis["paired_positions"]),
                        ("raw_native_rows.json", analysis["raw_rows"])):
        (out / name).write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    text = ["# Matched-prefix draft diagnostic", "", "## Material Passport", "",
            "ARS experiment-agent; bounded read-only local diagnostic. This report creates no new model evaluations. "
            "Scientific results are available only for rows passing the shared native validity gate.", "", BOUNDARY, "",
            "Location effects are averaged within each request, then equally across requests with valid evidence. "
            "Raw counts and selected/valid denominators are separate. Source controls are illustrative lower-median "
            "examples, not estimates for all 24 source requests. Partial groups and requests with no rounds remain visible.", "",
            "| Population | Arm/F | Valid / selected locations | Requests with valid / selected | Equal-request mean prefix change | Useful first-error fixes / F first errors | Blocks with new earlier errors |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in analysis["overall"]:
        text.append(f"| {row['population']} | {row['arm']} | {row['valid_locations']} / {row['selected_locations']} | "
                    f"{row['requests_with_valid_locations']} / {row['selected_requests']} | {_fmt(row['mean_correct_prefix_gain'])} | "
                    f"{row['useful_first_error_fixes']} / {row['frozen_first_error_blocks']} | {row['blocks_with_new_earlier_errors']} |")
    text.extend(["", "The paired records distinguish useful fixes at F's first mismatch from positional fixes blocked by an "
                 "earlier new error. Gain/loss/tie counts, censoring, EOS, margin changes and all-nine versus scored proposal "
                 "changes are preserved in the CSV/JSON files. No numerical cutoff defines 'few' or 'useful' at the population level.", "",
                 "| Retained population | Arm | Available / selected requests | End-to-end verification calls | Retained mean response KL |", "|---|---|---:|---:|---:|"])
    for row in analysis["retained_measurement_summaries"]:
        text.append(f"| {row['population']} | {row['arm']} | {row['available_requests']} / {row['selected_requests']} | "
                    f"{_fmt(row['verification_calls'])} | {_fmt(row['mean_retained_response_kl'])} |")
    text.extend(["", "Retained call measurements are shown alongside local observations; they are never inferred from prefix "
                 "changes. Source rows cover only the six selected controls when the full plan is present. Missing "
                 "measurements are unavailable, not zero.", "",
                 "| Full retained campaign | Arm | Available / recorded requests | End-to-end verification calls | Retained mean response KL |",
                 "|---|---|---:|---:|---:|"])
    for row in analysis["retained_campaign_summaries"]:
        text.append(f"| {row['population']} | {row['arm']} | {row['available_requests']} / {row['recorded_requests']} | "
                    f"{_fmt(row['verification_calls'])} | {_fmt(row['mean_retained_response_kl'])} |")
    text.extend(["", "Full-campaign source counts include all original source requests when their retained records are available; "
                 "they are separate from the illustrative source-control local sample.", "", "Group status and missing evidence:", ""])
    for group in analysis["group_statuses"]:
        text.append(f"- {group['group_id']}: {group['status']}; {_fmt(group.get('error'))}; validity errors: {_fmt(group.get('validity_errors'))}")
    if not any(row["valid_locations"] for row in analysis["overall"]):
        text.extend(["", "Insufficient valid local evidence; no decision-change conclusion is available."])
    else:
        text.extend(["", "Assess the reported gains, losses, useful fixes and new earlier errors together. Positive local means "
                     "describe sampled prefix improvement; mixed or incomplete patterns do not establish an end-to-end benefit."])
    text.extend(["", "Costs preserve original-prompt target-prefill chunks, historical reconstruction target verifications, "
                 "head inspection, loading, cache maintenance, wall time and memory "
                 "under their original scopes. Historical verification forwards replay retained proposal blocks only, "
                 "without extra draft evaluations, scored target verification or regenerated histories. "
                 "reconstruction_target_verify_s is included in reconstruction_s; these nested/subset costs "
                 "are not summed into a new latency measure. The raw result "
                 "retains plan, implementation/configuration bindings, invalid rows and unattempted locations.", ""])
    (out / "REPORT.md").write_text("\n".join(text))
    return analysis
