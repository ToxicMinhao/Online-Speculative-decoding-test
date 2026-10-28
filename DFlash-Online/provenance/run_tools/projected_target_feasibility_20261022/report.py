"""Report the sealed F/C/P screen without dropping missing or invalid requests.

No model/runtime imports are needed. Negative percentage changes mean fewer
verification calls. Full-planned ratios remain null until every pair is valid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = ("F", "C", "P")
COMPARISONS = (("C", "F"), ("P", "F"), ("P", "C"))


def _read(path):
    if not path.exists():
        return None, "missing_file"
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return None, "json_is_not_an_object"
        return value, ""
    except (OSError, ValueError) as error:
        return None, f"unreadable_json:{type(error).__name__}"


def _integer(value):
    return type(value) is int and value >= 0


def _token_ids(value):
    return isinstance(value, list) and bool(value) and all(_integer(x) for x in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate(record, request, cap):
    """Check native output, termination and count consistency before pairing."""
    errors = []
    prompt, response = record.get("prompt_ids"), record.get("response_ids")
    if not _token_ids(prompt) or prompt != request["prompt_ids"]:
        errors.append("prompt_ids_mismatch")
    if record.get("record_id", request["request_id"]) != request["request_id"]:
        errors.append("record_id_mismatch")
    if not _token_ids(response):
        errors.append("invalid_response_ids")
        response = []
    if record.get("max_new_tokens") != cap:
        errors.append("max_new_tokens_mismatch")
    if not response or len(response) > cap:
        errors.append("invalid_response_length")
    finish = record.get("finish_reason")
    if finish not in ("eos", "length"):
        errors.append("invalid_finish_reason")
    if finish == "length" and len(response) != cap:
        errors.append("length_finish_before_cap")
    stops = record.get("stop_token_ids")
    if isinstance(stops, list) and response:
        if any(token in stops for token in response[:-1]):
            errors.append("continued_after_eos")
        if (response[-1] in stops) != (finish == "eos"):
            errors.append("eos_token_finish_mismatch")

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    calls = metadata.get("verification_calls")
    accepted = metadata.get("accepted_draft_tokens")
    progress = metadata.get("committed_progress")
    for key, value in (("verification_calls", calls),
                       ("accepted_draft_tokens", accepted),
                       ("committed_progress", progress)):
        if not _integer(value):
            errors.append("invalid_" + key)
    if _integer(progress) and progress != len(response) - 1:
        errors.append("progress_length_mismatch")
    if _integer(calls) and _integer(progress):
        if calls > progress or bool(calls) != bool(progress):
            errors.append("calls_progress_mismatch")
    if _integer(accepted) and _integer(progress) and accepted > progress:
        errors.append("accepted_exceeds_progress")
    trace = metadata.get("trace")
    if trace is not None:
        try:
            if len(trace) != calls:
                errors.append("trace_call_count_mismatch")
            if sum(block["accepted_draft_tokens"] for block in trace) != accepted:
                errors.append("trace_accepted_count_mismatch")
            offset = 0
            for block in trace:
                committed = block["committed_progress"]
                if (not _integer(committed) or committed < 1
                        or block["output_start"] != offset
                        or block["anchor_token"] != response[offset]
                        or block["committed_tokens"] != response[offset + 1:offset + 1 + committed]):
                    raise ValueError("trace does not reconstruct output")
                offset += committed
            if offset != progress:
                errors.append("trace_progress_mismatch")
        except (KeyError, TypeError, IndexError, ValueError):
            errors.append("invalid_verification_trace")
    return errors, metadata


def _qualification(root, group):
    receipt, error = _read(root / group / "QUALIFICATION.json")
    reasons = []
    if error:
        reasons.append("qualification_" + error)
    else:
        if receipt.get("status") != "passed":
            reasons.append("qualification_not_passed")
        for key in ("matched_anchors", "matched_exposure", "matched_final_rng"):
            if receipt.get(key) is not True:
                reasons.append("qualification_" + key + "_not_passed")
    return {"passed": not reasons, "reasons": reasons, "receipt": receipt}


def _paired_row(group, role, arm, baseline, requests, records, qualification):
    pairs = []
    invalid = []
    for request in requests:
        current = records[(group, request["request_id"], arm)]
        reference = records[(group, request["request_id"], baseline)]
        if current["comparison_valid"] and reference["comparison_valid"]:
            pairs.append((current, reference))
        else:
            invalid.append(request["request_id"])
    current_sum = sum(current["verification_calls"] for current, _ in pairs)
    baseline_sum = sum(reference["verification_calls"] for _, reference in pairs)
    change = 100 * (current_sum / baseline_sum - 1) if baseline_sum else None
    complete = bool(requests) and len(pairs) == len(requests)
    return {
        "group": group, "role": role, "arm": arm, "baseline": baseline,
        "planned_pairs": len(requests), "valid_pairs": len(pairs),
        "invalid_or_missing_pairs": len(invalid), "complete": complete,
        "qualification_passed": qualification["passed"],
        "arm_calls_sum": current_sum if pairs else None,
        "baseline_calls_sum": baseline_sum if pairs else None,
        "change_pct": change if complete else None,
        "available_valid_pairs_change_pct": change,
        "gains": sum(a["verification_calls"] < b["verification_calls"] for a, b in pairs),
        "regressions": sum(a["verification_calls"] > b["verification_calls"] for a, b in pairs),
        "ties": sum(a["verification_calls"] == b["verification_calls"] for a, b in pairs),
        "invalid_request_ids": ";".join(invalid),
        "ratio_note": "zero_baseline_calls" if pairs and not baseline_sum else "",
    }


def _pool(rows, role, arm, baseline):
    selected = [r for r in rows if (r["role"], r["arm"], r["baseline"]) == (role, arm, baseline)]
    result = {"group": "ALL", "role": role, "arm": arm, "baseline": baseline}
    for key in ("planned_pairs", "valid_pairs", "invalid_or_missing_pairs", "gains", "regressions", "ties"):
        result[key] = sum(r[key] for r in selected)
    for key in ("arm_calls_sum", "baseline_calls_sum"):
        result[key] = sum(r[key] or 0 for r in selected) if result["valid_pairs"] else None
    result["complete"] = bool(selected) and all(r["complete"] for r in selected)
    result["qualification_passed"] = bool(selected) and all(r["qualification_passed"] for r in selected)
    denominator = result["baseline_calls_sum"]
    change = 100 * (result["arm_calls_sum"] / denominator - 1) if denominator else None
    result["change_pct"] = change if result["complete"] else None
    result["available_valid_pairs_change_pct"] = change
    result["invalid_request_ids"] = ";".join(
        f'{r["group"]}:{r["invalid_request_ids"]}' for r in selected if r["invalid_request_ids"])
    result["ratio_note"] = "zero_baseline_calls" if result["valid_pairs"] and not denominator else ""
    return result


def _costs(root, groups, rows, status):
    """Keep nested timing boundaries and exposure; never add overlapping costs."""
    output = {
        "timing_note": "Single, instrumented request timings; no speedup inference. "
                       "Reference forward/head/projection are nested in training_s. "
                       "Reference setup is outside training_s but inside event wall time. "
                       "Event wall time also includes checkpoint/export; do not add these boundaries.",
        "status_receipt": status, "groups": {},
    }
    for group in groups:
        name = group["name"]
        group_cost = {}
        preparation, prep_error = _read(root / name / "F" / "PREPARATION.json")
        prep_rows = preparation.get("records", []) if preparation else []
        group_cost["shared_teacher_preparation"] = {
            "read_error": prep_error or None,
            "teacher_preparation_s": sum(r["teacher_preparation_s"] for r in prep_rows)
            if prep_rows and all(_number(r.get("teacher_preparation_s")) for r in prep_rows) else None,
            "feature_persistence_s": sum(r["feature_persistence_s"] for r in prep_rows)
            if prep_rows and all(_number(r.get("feature_persistence_s")) for r in prep_rows) else None,
            "receipt": preparation,
        }
        for arm in ARMS:
            serving_rows = [r for r in rows if r["group"] == name and r["arm"] == arm]
            timed = [r for r in serving_rows if _number(r["request_latency_s"])]
            arm_cost = {"serving": {
                "planned_requests": len(serving_rows), "timed_requests": len(timed),
                "observed_instrumented_request_s": sum(r["request_latency_s"] for r in timed) if timed else None,
                "invalid_timed_requests": sum(not r["comparison_valid"] for r in timed),
            }}
            if arm != "F":
                training, error = _read(root / name / arm / "TRAINING.json")
                activation, activation_error = _read(root / name / arm / "ACTIVATION.json")
                training = training or {}
                costs = training.get("costs", {})
                arm_cost.update({
                    "training_read_error": error or None,
                    "training": {key: value for key, value in costs.items() if key != "export_s"},
                    "reference_and_projection": training.get("objective_work"),
                    "export": {"export_s": costs.get("export_s")},
                    "activation": activation, "activation_read_error": activation_error or None,
                    "exposure": {
                        "response_presentations": training.get("response_presentations"),
                        "optimizer_steps": training.get("optimizer_steps"),
                        "actual_learning_rates": training.get("actual_learning_rates"),
                        "supervision": training.get("exposure"),
                        "presentations": [{key: r.get(key) for key in (
                            "record_id", "pass", "optimizer_stepped", "optimizer_step_after", "sampled_anchors")}
                            for r in training.get("records", [])],
                    },
                    "peak_memory": {key: value for key, value in training.items()
                                    if "memory" in key or "peak" in key},
                })
            group_cost[arm] = arm_cost
        output["groups"][name] = group_cost
    return output


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def build_report(root):
    """Write REPORT.md, SUMMARY.json, PER_REQUEST.csv, POOLED.csv and COSTS.json."""
    root = Path(root)
    manifest, error = _read(root / "manifest.json")
    if error:
        raise ValueError("Cannot report without sealed manifest.json: " + error)
    groups = manifest["groups"]
    qualification = {group["name"]: _qualification(root, group["name"]) for group in groups}
    status, status_error = _read(root / "STATUS.json")
    rows, records, comparisons = [], {}, []
    default_cap = manifest.get("max_new_tokens", manifest.get("generation", {}).get("max_new_tokens", 512))
    for group in groups:
        name = group["name"]
        for request in group["requests"]:
            loaded = {}
            for arm in ARMS:
                record, error = _read(root / name / arm / "records" / (request["request_id"] + ".json"))
                loaded[arm] = record
                reasons, metadata = _validate(record, request, request.get("max_new_tokens", default_cap)) if record else ([error], {})
                if record and arm != "F":
                    frozen = loaded["F"]
                    if frozen is None:
                        reasons.append("canonical_F_missing")
                    else:
                        for key in ("prompt_ids", "response_ids", "finish_reason", "max_new_tokens"):
                            if record.get(key) != frozen.get(key):
                                reasons.append("canonical_F_" + key + "_mismatch")
                        if not records[(name, request["request_id"], "F")]["native_valid"]:
                            reasons.append("canonical_F_invalid")
                native_valid = not reasons
                reasons += qualification[name]["reasons"]
                record = record or {}
                response = record.get("response_ids")
                row = {
                    "group": name, "role": request["role"], "request_id": request["request_id"], "arm": arm,
                    "present": loaded[arm] is not None, "native_valid": native_valid,
                    "comparison_valid": not reasons, "validity_reason": ";".join(reasons) or "valid",
                    "verification_calls": metadata.get("verification_calls"),
                    "accepted_draft_tokens": metadata.get("accepted_draft_tokens"),
                    "committed_progress": metadata.get("committed_progress"),
                    "response_tokens": len(response) if isinstance(response, list) else None,
                    "finish_reason": record.get("finish_reason"), "max_new_tokens": record.get("max_new_tokens"),
                    "request_latency_s": metadata.get("request_latency_s"),
                }
                rows.append(row)
                records[(name, request["request_id"], arm)] = row
        for role in ("A", "B"):
            planned = [r for r in group["requests"] if r["role"] == role]
            for arm, baseline in COMPARISONS:
                comparisons.append(_paired_row(name, role, arm, baseline, planned, records, qualification[name]))
    comparisons += [_pool(comparisons, role, arm, baseline)
                    for role in ("A", "B") for arm, baseline in COMPARISONS]
    outcomes = []
    for name in [g["name"] for g in groups] + ["ALL"]:
        for role in ("A", "B"):
            for arm in ARMS:
                planned = [r for r in rows if r["role"] == role and r["arm"] == arm
                           and (name == "ALL" or r["group"] == name)]
                valid = [r for r in planned if r["comparison_valid"]]
                outcomes.append({
                    "group": name, "role": role, "arm": arm, "planned_requests": len(planned),
                    "valid_requests": len(valid),
                    "accepted_draft_tokens_sum": sum(r["accepted_draft_tokens"] for r in valid) if valid else None,
                    "output_tokens_sum": sum(r["response_tokens"] for r in valid) if valid else None,
                    "output_tokens_mean": sum(r["response_tokens"] for r in valid) / len(valid) if valid else None,
                    "eos_count": sum(r["finish_reason"] == "eos" for r in valid),
                    "cap_count": sum(r["finish_reason"] == "length" for r in valid),
                    "eos_rate_valid": sum(r["finish_reason"] == "eos" for r in valid) / len(valid) if valid else None,
                    "cap_rate_valid": sum(r["finish_reason"] == "length" for r in valid) / len(valid) if valid else None,
                })
    summary = {
        "scope": "Fixed F/C/P feasibility screen; A is trained source, B is held out.",
        "planned_generations": len(rows), "present_generations": sum(r["present"] for r in rows),
        "valid_generations": sum(r["comparison_valid"] for r in rows),
        "all_planned_valid": bool(rows) and all(r["comparison_valid"] for r in rows),
        "qualification": qualification, "run_status": status, "run_status_read_error": status_error or None,
        "comparisons": comparisons, "outcomes": outcomes,
        "ratio_definition": "100 * (sum(arm calls) / sum(baseline calls) - 1); negative is fewer calls. "
                            "change_pct requires all planned pairs; available_valid_pairs_change_pct names its reduced denominator.",
    }
    costs = _costs(root, groups, rows, status)
    _write_csv(root / "PER_REQUEST.csv", rows)
    _write_csv(root / "POOLED.csv", comparisons)
    for filename, payload in (("SUMMARY.json", summary), ("COSTS.json", costs)):
        (root / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Projected-target feasibility report", "",
             f'Planned {len(rows)} generations; present {summary["present_generations"]}; '
             f'qualified and valid {summary["valid_generations"]}.', "",
             "A is the trained source set; B is held out. Negative changes mean fewer verification calls. "
             "Ratios use summed calls, never the mean of request ratios. Incomplete rows retain their planned denominator; "
             "their full-planned percentage is unavailable.", "",
             "| Group | Role | Comparison | Valid/planned | Arm/base calls | Change | Gains/regressions/ties |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in comparisons:
        change = f'{row["change_pct"]:+.2f}%' if row["change_pct"] is not None else "unavailable"
        calls = f'{row["arm_calls_sum"]}/{row["baseline_calls_sum"]}' if row["valid_pairs"] else "unavailable"
        lines.append(f'| {row["group"]} | {row["role"]} | {row["arm"]} vs {row["baseline"]} | '
                     f'{row["valid_pairs"]}/{row["planned_pairs"]} | {calls} | {change} | '
                     f'{row["gains"]}/{row["regressions"]}/{row["ties"]} |')
    lines += ["", "Token and termination totals across groups (qualified valid requests only):", "",
              "| Role | Arm | Valid/planned | Accepted draft tokens | Output tokens (mean) | EOS/cap |",
              "|---|---|---:|---:|---:|---:|"]
    for row in outcomes:
        if row["group"] != "ALL":
            continue
        length = f'{row["output_tokens_sum"]} ({row["output_tokens_mean"]:.1f})' if row["valid_requests"] else "unavailable"
        accepted = row["accepted_draft_tokens_sum"] if row["valid_requests"] else "unavailable"
        lines.append(f'| {row["role"]} | {row["arm"]} | {row["valid_requests"]}/{row["planned_requests"]} | '
                     f'{accepted} | {length} | {row["eos_count"]}/{row["cap_count"]} |')
    lines += ["", "Qualification:", ""]
    lines += [f'- {name}: {"passed" if value["passed"] else "; ".join(value["reasons"])}.'
              for name, value in qualification.items()]
    lines += ["", "PER_REQUEST.csv preserves every planned arm/request and its validity reason. "
              "SUMMARY.json records accepted draft tokens, output lengths and EOS/cap rates with explicit denominators. "
              "POOLED.csv includes the explicitly labeled available-valid-pair ratios for incomplete comparisons.", "",
              "COSTS.json separates shared teacher preparation, training, nested reference/projection work, exposure, "
              "checkpoint/export, activation and serving. Timings are single instrumented observations and do not establish speedup. "
              "Losses from different objectives are not compared. Peak-memory and end-to-end receipts are retained where supplied.", "",
              "Benefit versus F supports adaptation on this screen. Beating C while remaining worse than F is reduced harm. "
              "Little source learning leaves preservation inconclusive; this report does not select a new direction.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = build_report(args.root)
    print(f'{result["valid_generations"]}/{result["planned_generations"]} qualified valid generations')
