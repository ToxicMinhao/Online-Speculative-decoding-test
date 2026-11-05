"""Fresh F/OSD comparison on retained feedback, with explicit validity and costs."""
import json
from pathlib import Path

from .report import _number, _read, _validate, _write_csv


ARMS = ("F", "O")
PAIR_FIELDS = ("prompt_ids", "response_ids", "finish_reason", "stop_token_ids", "max_new_tokens")
QUALIFICATION_FIELDS = (
    "matched_anchors", "matched_exposure", "matched_initial_rng", "matched_final_rng",
    "frozen_reproduction", "target_frozen",
)
ARM_CHECKS = (
    "initial_released_equal", "publication_equal", "outputs_equal", "fresh_optimizer",
    "target_excluded_from_optimizer",
)


def _qualification(root, group):
    receipt, error = _read(root / group / "QUALIFICATION.json")
    reasons = ["qualification_" + error] if error else []
    if receipt is not None:
        if receipt.get("status") != "passed":
            reasons.append("qualification_not_passed")
        reasons += ["qualification_" + key for key in QUALIFICATION_FIELDS
                    if receipt.get(key) is not True]
        checks = receipt.get("arms", {}).get("O", {})
        reasons += ["qualification_O_" + key for key in ARM_CHECKS if checks.get(key) is not True]
    return {"passed": not reasons, "reasons": reasons, "receipt": receipt}


def _sum(rows, field):
    return sum(row[field] for row in rows) if rows else None


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _change(current, baseline):
    return 100 * (current / baseline - 1) if current is not None and baseline else None


def _comparison(group, role, requests, records):
    pairs = [(records[(r["group"], r["request_id"], "F")],
              records[(r["group"], r["request_id"], "O")]) for r in requests]
    valid = [(f, o) for f, o in pairs if f["comparison_valid"] and o["comparison_valid"]]
    timed = [(f, o) for f, o in valid if f["timing_valid"] and o["timing_valid"]]
    result = {
        "group": group, "role": role, "planned_pairs": len(pairs), "valid_pairs": len(valid),
        "invalid_or_missing_pairs": len(pairs) - len(valid), "timed_pairs": len(timed),
        "complete": bool(pairs) and len(valid) == len(pairs),
        "timing_complete": bool(pairs) and len(timed) == len(pairs),
        "gains": sum(o["verification_calls"] < f["verification_calls"] for f, o in valid),
        "regressions": sum(o["verification_calls"] > f["verification_calls"] for f, o in valid),
        "ties": sum(o["verification_calls"] == f["verification_calls"] for f, o in valid),
        "invalid_request_ids": ";".join(f'{f["group"]}:{f["request_id"]}' for f, o in pairs
                                        if not (f["comparison_valid"] and o["comparison_valid"])),
    }
    for index, arm in enumerate(ARMS):
        rows = [pair[index] for pair in valid]
        for field in ("verification_calls", "accepted_draft_tokens", "committed_progress", "response_tokens"):
            result[f"{arm}_{field}_sum"] = _sum(rows, field)
        result[f"{arm}_request_latency_s_sum"] = _sum([pair[index] for pair in timed], "request_latency_s")
        calls = result[f"{arm}_verification_calls_sum"]
        result[f"{arm}_calls_per_1000_output_tokens"] = _ratio(calls, result[f"{arm}_response_tokens_sum"], 1000)
        result[f"{arm}_accepted_draft_tokens_per_call"] = _ratio(result[f"{arm}_accepted_draft_tokens_sum"], calls)
        result[f"{arm}_committed_progress_per_call"] = _ratio(result[f"{arm}_committed_progress_sum"], calls)
        result[f"{arm}_cap_hits"] = sum(row["finish_reason"] == "length" for row in rows)
        result[f"{arm}_eos_count"] = sum(row["finish_reason"] == "eos" for row in rows)
    calls_change = _change(result["O_verification_calls_sum"], result["F_verification_calls_sum"])
    time_change = _change(result["O_request_latency_s_sum"], result["F_request_latency_s_sum"])
    result.update(
        call_change_pct=calls_change if result["complete"] else None,
        available_valid_pairs_call_change_pct=calls_change,
        request_time_change_pct=time_change if result["timing_complete"] else None,
        available_valid_timed_pairs_time_change_pct=time_change,
    )
    return result


def _costs(root, source, groups, rows, status):
    result = {
        "actual_campaign_wall_s": status.get("wall_s") if status else None,
        "run_status": status,
        "timing_note": "Fresh single instrumented request timings only. Teacher-head work is nested in "
                       "training_s. Event wall includes training, checkpoint save and export; do not sum "
                       "these boundaries. Activation and native_activation describe nested/same work. "
                       "Historical preparation is reused work, not fresh campaign time.",
        "groups": {},
    }
    for group in groups:
        name = group["name"]
        preparation, prep_error = _read(source / name / "F/PREPARATION.json") if source else (None, "source_campaign_missing")
        prep_rows = preparation.get("records", []) if preparation else []
        historical = {}
        for field in ("teacher_preparation_s", "feature_persistence_s"):
            historical[field] = sum(r[field] for r in prep_rows) if prep_rows and all(
                _number(r.get(field)) for r in prep_rows) else None
        training, training_error = _read(root / name / "O/TRAINING.json")
        activation, activation_error = _read(root / name / "O/ACTIVATION.json")
        training = training or {}
        costs = training.get("costs", {})
        serving = {}
        for arm in ARMS:
            arm_rows = [r for r in rows if r["group"] == name and r["arm"] == arm]
            timed = [r for r in arm_rows if r["timing_valid"]]
            serving[arm] = {
                "planned_requests": len(arm_rows), "observed_timed_requests": len(timed),
                "observed_request_latency_s": _sum(timed, "request_latency_s"),
                "invalid_or_unqualified_timed_requests": sum(not r["comparison_valid"] for r in timed),
            }
        result["groups"][name] = {
            "serving": serving,
            "teacher_preparation": {
                "fresh_teacher_preparation_s": 0.0, "reused_retained_features": True,
                "historical_cost_for_reconstructed_workflow": historical,
                "source_receipt": str(source / name / "F/PREPARATION.json") if source else None,
                "read_error": prep_error or None,
            },
            "training": {key: costs.get(key) for key in ("training_s", "checkpoint_save_s", "export_s", "event_wall_s_before_commit")},
            "nested_objective_work": training.get("objective_work"),
            "activation": activation,
            "training_read_error": training_error or None,
            "activation_read_error": activation_error or None,
            "model_loading_s": training.get("model_loading_s"),
            "diagnostic_validation_s": training.get("diagnostic_validation_s"),
            "exposure": {key: training.get(key) for key in (
                "response_presentations", "optimizer_steps", "actual_learning_rates", "exposure")},
            "peak_memory": {key: value for key, value in training.items() if key.startswith("peak_")},
        }
    return result


def build_report(root):
    """Preserve every planned measurement; only qualified identical-output pairs enter gains."""
    root = Path(root)
    manifest, error = _read(root / "manifest.json")
    if error:
        raise ValueError("Cannot report without manifest.json: " + error)
    plan, plan_error = _read(root / "PLAN.json")
    status, status_error = _read(root / "STATUS.json")
    source = Path(plan["source_campaign"]) if plan and plan.get("source_campaign") else None
    groups = manifest["groups"]
    qualification = {g["name"]: _qualification(root, g["name"]) for g in groups}
    rows, records, requests = [], {}, []
    cap = (plan or {}).get("output_cap", manifest.get("max_new_tokens", 512))
    for group in groups:
        name = group["name"]
        for request in group["requests"]:
            requests.append({**request, "group": name})
            original, original_error = _read(source / name / "F/records" / (request["request_id"] + ".json")) if source else (None, "source_campaign_missing")
            frozen = None
            for arm in ARMS:
                record, error = _read(root / name / arm / "records" / (request["request_id"] + ".json"))
                reasons, metadata = _validate(record, request, cap) if record is not None else ([error], {})
                if record is not None:
                    reference = original if arm == "F" else frozen
                    if reference is None:
                        reasons.append("retained_F_" + original_error if arm == "F" else "fresh_F_missing")
                    else:
                        reasons += [f'{arm}_reference_{key}_mismatch' for key in PAIR_FIELDS
                                    if record.get(key) != reference.get(key)]
                        if arm == "F" and metadata.get("verification_calls") != reference.get("metadata", {}).get("verification_calls"):
                            reasons.append("F_retained_verification_calls_mismatch")
                    if arm == "O" and not records[(name, request["request_id"], "F")]["native_valid"]:
                        reasons.append("fresh_F_invalid")
                native_valid = not reasons
                reasons = reasons + qualification[name]["reasons"]
                response = record.get("response_ids") if record else None
                row = {
                    "group": name, "role": request["role"], "request_id": request["request_id"], "arm": arm,
                    "present": record is not None, "native_valid": native_valid,
                    "comparison_valid": not reasons, "validity_reason": ";".join(reasons) or "valid",
                    "verification_calls": metadata.get("verification_calls"),
                    "accepted_draft_tokens": metadata.get("accepted_draft_tokens"),
                    "committed_progress": metadata.get("committed_progress"),
                    "response_tokens": len(response) if isinstance(response, list) else None,
                    "finish_reason": record.get("finish_reason") if record else None,
                    "request_latency_s": metadata.get("request_latency_s"),
                    "timing_valid": _number(metadata.get("request_latency_s")),
                }
                rows.append(row)
                records[(name, request["request_id"], arm)] = row
                if arm == "F":
                    frozen = record
    comparisons = [_comparison(name, role, [r for r in requests if r["role"] == role and (
        name == "ALL" or r["group"] == name)], records)
        for name in [g["name"] for g in groups] + ["ALL"] for role in ("A", "B")]
    summary = {
        "scope": "Fresh F versus OSD on fixed A feedback and held-out B; OSD objective only changes learning signal.",
        "planned_generations": len(rows), "present_generations": sum(r["present"] for r in rows),
        "valid_generations": sum(r["comparison_valid"] for r in rows),
        "all_planned_valid": bool(rows) and all(r["comparison_valid"] for r in rows),
        "qualification": qualification, "comparisons": comparisons, "run_status": status,
        "plan_read_error": plan_error or None, "status_read_error": status_error or None,
        "ratio_definition": "100 * (sum(OSD) / sum(F) - 1); negative means fewer calls or less time. "
                            "Full-planned ratios require complete valid pairs. Available-pair ratios retain their reduced denominator.",
    }
    costs = _costs(root, source, groups, rows, status)
    _write_csv(root / "PER_REQUEST.csv", rows)
    _write_csv(root / "POOLED.csv", comparisons)
    for filename, payload in (("SUMMARY.json", summary), ("COSTS.json", costs)):
        (root / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_markdown(root, summary)
    return summary


def _write_markdown(root, summary):
    def percent(value):
        return f"{value:+.2f}%" if value is not None else "unavailable"

    lines = ["# OSD baseline follow-up", "",
             f'Planned {summary["planned_generations"]} generations; observed {summary["present_generations"]}; '
             f'qualified valid measurements {summary["valid_generations"]}.', "",
             "A measures fit to the trained source requests; B measures transfer to held-out requests. "
             "Negative changes mean fewer verification calls or less request time. Ratios use sums, not mean request percentages. "
             "Incomplete comparisons have no full-planned percentage; the machine-readable tables label available-pair results separately.", "",
             "| Group | Set | Valid/planned | F / OSD calls | Call change | Request-time change | Gains/regressions/ties |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["comparisons"]:
        calls = f'{row["F_verification_calls_sum"]} / {row["O_verification_calls_sum"]}' if row["valid_pairs"] else "unavailable"
        lines.append(f'| {row["group"]} | {row["role"]} | {row["valid_pairs"]}/{row["planned_pairs"]} | '
                     f'{calls} | {percent(row["call_change_pct"])} | {percent(row["request_time_change_pct"])} | '
                     f'{row["gains"]}/{row["regressions"]}/{row["ties"]} |')
    lines += ["", "Qualification:", ""]
    lines += [f'- {name}: {"passed" if q["passed"] else "; ".join(q["reasons"])}.'
              for name, q in summary["qualification"].items()]
    lines += ["", "Only fresh F/OSD timings are compared. Each request has one instrumented observation, including "
              "prefill, drafting, verification and compact tracing; large feedback capture, training, publication and file "
              "persistence are outside that timer. These measurements do not establish robust serving speedup or "
              "adaptation-inclusive acceleration. No historical C/P timings enter these ratios.", "",
              "PER_REQUEST.csv retains missing, invalid and unqualified work. POOLED.csv includes calls per 1,000 output tokens, "
              "accepted draft tokens and committed progress per call, cap/EOS counts and separate timing denominators. "
              "COSTS.json preserves actual campaign wall time, all observed serving costs, reused historical teacher preparation, "
              "new training, checkpoint/export and activation. Teacher-head work is nested inside training and is not added twice.", "",
              "A source improvement alone does not establish transfer. A B improvement over frozen supports useful adaptation "
              "under this fixed recipe. Null or harmful results remain evidence; this report does not retune the method or "
              "establish general superiority of forward KL.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
