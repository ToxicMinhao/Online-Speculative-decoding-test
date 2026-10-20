"""Pure descriptive reporting for the bounded agreement-filter development screen."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .experiment import ARMS, LEARNING_ARMS, digest
from .controls import n_checkpoint_control, n_update_control, N_NORM_POLICY
from .measurement import measurement_validation

SCOPE = ("Existing development requests, one capture-disabled replay per checkpoint/request. "
         "Latency is diagnostic, not deployment speedup. Checkpoints are compared on identical "
         "historical canonical inputs; each learning arm separately generated its own training history.")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _ratio(a, b, scale=1):
    return scale * a / b if a is not None and b else None


def _reduction(a, b):
    return 100 * (1 - a / b) if a is not None and b else None


def totals(rows):
    rows = list(rows)
    def measured(key):
        return sum(r[key] for r in rows) if rows and all(_number(r.get(key)) for r in rows) else None
    tokens = sum(len(r["output_ids"]) for r in rows) if rows and all(isinstance(r.get("output_ids"), list) for r in rows) else None
    calls = measured("verification_calls")
    accepted = measured("accepted_draft_tokens")
    committed = measured("verification_committed_tokens")
    return {"requests": len(rows), "output_tokens": tokens, "verification_calls": calls,
            "accepted_draft_tokens": accepted, "verification_committed_tokens": committed,
            "calls_per_1000_output_tokens": _ratio(calls, tokens, 1000),
            "accepted_draft_tokens_per_call": _ratio(accepted, calls),
            "committed_progress_per_call": _ratio(committed, calls),
            "diagnostic_request_s": measured("request_s")}


def _reference(result, turn):
    refs = result.get("references", {})
    return refs.get(turn, refs.get(str(turn)))


def _invalid(row, reference):
    validation = measurement_validation(row)
    reasons = ["measurement_validation:" + error for error in validation["errors"]]
    recorded = row.get("measurement_validation")
    if recorded is not None and (not isinstance(recorded, dict) or
            recorded.get("status") != "passed" or recorded.get("errors") != []):
        reasons.append("recorded_measurement_validation_failed")
    if reference is None:
        return reasons + ["missing_immutable_reference"]
    for key in ("same_generation_valid", "input_match", "canonical_output_match", "checkpoint_identity_match"):
        if row.get(key) is not True:
            reasons.append(key)
    if row.get("capture") is not False or row.get("feedback") is not None or row.get("capture_costs") is not None:
        reasons.append("capture_not_disabled")
    if row.get("canonical") is not False or row.get("training_data") is not False or row.get("repeat") != 0:
        reasons.append("not_single_benchmark_replay")
    for key in ("input_ids", "output_ids"):
        if row.get(key) != reference.get(key) or row.get("reference_" + key + "_sha256") != digest(reference.get(key)):
            reasons.append("exact_" + key + "_mismatch")
    if not isinstance(row.get("checkpoint_identity_sha256"), str) or not row["checkpoint_identity_sha256"]:
        reasons.append("checkpoint_identity_missing")
    for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
        if type(row.get(key)) is not int or row[key] < 0:
            reasons.append("invalid_" + key)
    if not _number(row.get("request_s")) or not row["request_s"]:
        reasons.append("invalid_request_s")
    return reasons


def comparison(result, turn, candidate, control):
    rows = result.get("scores", [])
    reference = _reference(result, turn)
    groups = [[r for r in rows if r.get("turn") == turn and r.get("arm") == arm] for arm in (candidate, control)]
    invalid = []
    for arm, group in zip((candidate, control), groups):
        if len(group) != 1:
            invalid.append(arm + ":missing_or_duplicate_score")
        for row in group:
            invalid.extend(arm + ":" + p for p in _invalid(row, reference))
    valid = not invalid
    new, old = (totals(g) for g in groups)
    calls_saved = old["verification_calls"] - new["verification_calls"] if valid else None
    norm_control = n_checkpoint_control(result, turn)
    return {"episode_id": result["episode_id"], "source": result["source"], "synthetic": result["synthetic"],
            "turn": turn, "candidate": candidate, "control": control, "valid": valid,
            "invalid_reasons": sorted(set(invalid)), "candidate_metrics": new, "control_metrics": old,
            "N_control_status": norm_control["status"], "N_control_errors": norm_control["errors"],
            "selective_filter_attribution_eligible": valid and norm_control["status"] == "passed",
            "comparison_scope": "Descriptive native measurements; selective attribution additionally requires all N controls to qualify",
            "verification_calls_saved": calls_saved,
            "verification_call_reduction_percent": _reduction(new["verification_calls"], old["verification_calls"]) if valid else None,
            "diagnostic_latency_reduction_percent": _reduction(new["diagnostic_request_s"], old["diagnostic_request_s"]) if valid else None,
            "drafting_result": ("improvement" if calls_saved > 0 else "regression" if calls_saved < 0 else "tie") if valid else "invalid"}, groups


def _pool(pairs):
    valid = [(row, groups) for row, groups in pairs if row["valid"]]
    candidate_rows = [r for _, groups in valid for r in groups[0]]
    control_rows = [r for _, groups in valid for r in groups[1]]
    new, old = totals(candidate_rows), totals(control_rows)
    attributable = [(row, groups) for row, groups in valid if row["selective_filter_attribution_eligible"]]
    return {"expected_requests": len(pairs), "valid_requests": len(valid), "invalid_requests": len(pairs) - len(valid),
            "norm_control_qualified_requests": len(attributable),
            "norm_control_failed_requests": len(valid) - len(attributable),
            "selective_filter_attribution_eligible": bool(valid) and len(attributable) == len(valid),
            "comparison_scope": "Descriptive only when any contributing N control fails; not a norm-matched selective-filter result",
            "improved_requests": sum(row["drafting_result"] == "improvement" for row, _ in valid),
            "regressed_requests": sum(row["drafting_result"] == "regression" for row, _ in valid),
            "tied_requests": sum(row["drafting_result"] == "tie" for row, _ in valid),
            "candidate_metrics": new, "control_metrics": old,
            "verification_calls_saved": old["verification_calls"] - new["verification_calls"] if valid else None,
            "verification_call_reduction_percent": _reduction(new["verification_calls"], old["verification_calls"]),
            "diagnostic_latency_reduction_percent": _reduction(new["diagnostic_request_s"], old["diagnostic_request_s"])}


def _costs(result):
    records = []
    for arm, value in result.get("arms", {}).items():
        for update in value.get("updates", []):
            account = update.get("accounting", {})
            records.append({"episode_id": result["episode_id"], "source": result["source"], "arm": arm,
                "event": update.get("event"), "training_s": update.get("training_s"),
                "export_s": update.get("export_s"), "optimizer_state_save_s": update.get("optimizer_state_save_s"),
                "trainer_startup_s": update.get("trainer_startup_s"),
                "observed_update_call_s": update.get("observed_update_call_s"),
                "teacher_preparation_s": account.get("costs", {}).get("teacher_preparation_s"),
                "eligible_position_presentations": account.get("eligible_position_presentations_this_event"),
                "eligible_block_presentations": account.get("eligible_block_presentations_this_event"),
                "optimizer_steps_this_event": account.get("optimizer_steps_this_event"),
                "anchor_exposure_sha256": account.get("anchor_exposure_sha256"),
                "peak_memory_allocated_bytes": account.get("peak_memory_allocated_bytes"),
                "peak_memory_reserved_bytes": account.get("peak_memory_reserved_bytes"),
                "source_losses": [b.get("loss") for b in account.get("batches", [])],
                "agreement_steps": [b.get("agreement") for b in account.get("batches", [])],
                "N_numerical_control": n_update_control(update) if arm == "N" else None,
                "fixed_groups": account.get("fixed_groups"),
                "cost_scope": "Teacher preparation is nested in training; observed update wall includes nested components."})
    return records


def summarize(results):
    results = list(results)
    all_pairs = []
    comparisons = [(arm, "F") for arm in LEARNING_ARMS] + [("G", c) for c in ("H", "L", "N")]
    for result in results:
        for turn in (2, 3):
            for candidate, control in comparisons:
                all_pairs.append(comparison(result, turn, candidate, control))
    pooled = []
    sources = sorted({r["source"] for r in results})
    for source in ["all"] + sources:
        for turn in (2, 3, "all"):
            for candidate, control in comparisons:
                matches = [(row, groups) for row, groups in all_pairs if row["candidate"] == candidate and
                    row["control"] == control and (source == "all" or row["source"] == source) and
                    (turn == "all" or row["turn"] == turn)]
                pooled.append({"source": source, "turn": turn, "candidate": candidate, "control": control,
                               **_pool(matches)})
    interpretations = []
    for result in results:
        for turn in (2, 3):
            selected = {row["control"]: row for row, _ in all_pairs if row["episode_id"] == result["episode_id"] and
                        row["turn"] == turn and row["candidate"] == "G"}
            norm_control = n_checkpoint_control(result, turn)
            if not all(selected.get(c, {}).get("valid") for c in ("F", "H", "L", "N")):
                label = "invalid_comparison"
            elif selected["H"]["verification_calls_saved"] < 0:
                label = ("tradeoff_not_improvement_over_ordinary_OSD" if all(selected[c]["verification_calls_saved"] > 0 for c in ("L", "N"))
                         else "worse_than_ordinary_OSD")
            elif norm_control["status"] != "passed":
                label = "selective_filter_attribution_unavailable_N_norm_control"
            elif all(selected[c]["verification_calls_saved"] > 0 for c in ("H", "L", "N")):
                label = "useful_filter_signal" if selected["F"]["verification_calls_saved"] > 0 else "control_advantage_without_frozen_gain"
            else:
                label = "no_clear_filter_advantage"
            interpretations.append({"episode_id": result["episode_id"], "turn": turn, "interpretation": label,
                "N_numerical_control": norm_control,
                "selective_filter_attribution_eligible": all(selected.get(c, {}).get("valid") for c in ("F", "H", "L", "N"))
                    and norm_control["status"] == "passed"})
    scores = [{"episode_id": r["episode_id"], "source": r["source"], "synthetic": r["synthetic"], **s}
              for r in results for s in r.get("scores", [])]
    per_request = []
    for result in results:
        for turn in (2, 3):
            for arm in ARMS:
                records = [row for row in result.get("scores", []) if row.get("turn") == turn and row.get("arm") == arm]
                invalid = [] if len(records) == 1 else ["missing_or_duplicate_score"]
                for row in records:
                    invalid.extend(_invalid(row, _reference(result, turn)))
                per_request.append({"episode_id": result["episode_id"], "source": result["source"],
                    "synthetic": result["synthetic"], "turn": turn, "arm": arm,
                    "valid": not invalid, "invalid_reasons": sorted(set(invalid)), **totals(records)})
    return {"schema_version": 2, "scope": SCOPE, "primary_control": "H", "N_control_policy": dict(N_NORM_POLICY), "results": [
                {"episode_id": r["episode_id"], "status": r.get("status"),
                 "implementation_checks": r.get("implementation_checks"), "actual_wall_s": r.get("actual_wall_s")} for r in results],
            "comparisons": [r for r, _ in all_pairs], "pooled": pooled, "interpretations": interpretations,
            "scores": scores, "per_request": per_request, "updates": [row for r in results for row in _costs(r)],
            "canonical_costs": [{"episode_id": r["episode_id"], "arm": arm, "turn": c["turn"],
                **{k: c.get(k) for k in ("request_s", "activation_s", "native_activation_s", "cache_maintenance_s")},
                "matches_common_input": c.get("matches_common_input"), "matches_common_output": c.get("matches_common_output")}
                for r in results for arm, value in r.get("arms", {}).items() for c in value.get("canonical", [])],
            "cost_note": "Serving, training, export and activation remain separate. Native activation is nested in activation. "
                         "Benchmark-only replay/activation is not deployment cost. No timing-based speedup conclusion."}


def _flatten(row):
    result = {}
    for key, value in row.items():
        if key in ("candidate_metrics", "control_metrics"):
            result.update({key.removesuffix("_metrics") + "_" + k: v for k, v in value.items()})
        elif isinstance(value, (list, dict)):
            result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        else:
            result[key] = value
    return result


def _csv(path, rows):
    rows = [_flatten(row) for row in rows]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(results, out):
    summary = summarize(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("comparisons", "pooled", "scores", "per_request", "updates", "canonical_costs", "interpretations"):
        _csv(out / (key.upper() + ".csv"), summary[key])
    lines = ["# OSD agreement-filter development screen", "", SCOPE, "",
             "H is the primary ordinary-OSD control. Beating L and N while losing to H is a tradeoff, not an improvement over OSD.", "",
             "N's applied norm must match its own intended filtered norm within 1% at every ancestral step (zero requires exact zero). "
             "Failed/missing controls preserve descriptive measurements but block selective-filter attribution. No tolerance or precision retuning.", "",
             "Verification calls on each fixed request:", "",
             "| Conversation | Turn | F | H | L | G | N |", "|---|---|---:|---:|---:|---:|---:|"]
    keys = list(dict.fromkeys((r["episode_id"], r["turn"]) for r in summary["per_request"]))
    for episode, turn in keys:
        rows = {r["arm"]: r for r in summary["per_request"] if (r["episode_id"], r["turn"]) == (episode, turn)}
        values = [str(rows[arm]["verification_calls"]) if rows[arm]["valid"] else "invalid / unavailable" for arm in ARMS]
        lines.append(f"| {episode} | {turn} | " + " | ".join(values) + " |")
    lines += ["", "PER_REQUEST.csv also reports accepted draft tokens, committed progress per call, token lengths and validity for every arm.", "",
             "| Source | Turn | Comparison | Valid / expected | N control qualified / valid | Calls saved | Call reduction | Improved / regressed / tied |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["pooled"]:
        if row["candidate"] != "G":
            continue
        reduction = row["verification_call_reduction_percent"]
        reduction = "unavailable" if reduction is None else f"{reduction:+.2f}%"
        lines.append(f"| {row['source']} | {row['turn']} | G / {row['control']} | {row['valid_requests']} / {row['expected_requests']} | "
                     f"{row['norm_control_qualified_requests']} / {row['valid_requests']} | {row['verification_calls_saved']} | {reduction} | {row['improved_requests']} / {row['regressed_requests']} / {row['tied_requests']} |")
    lines += ["", "Every per-request alternative, regression, mismatch and native measurement is retained in the CSV/JSON tables. "
              "Pooled ratios use summed measurements over valid pairs, not averages of request percentages.", "",
              "Source fitting and a smaller step alone do not establish transfer. Turn 2 tests the first update; turn 3 tests the accumulated two-update state, "
              "not an isolated causal estimate of update 2. Positive control comparisons are descriptive signals on existing development cases.", "",
              summary["cost_note"], "", "Stop after the declared report. No retuning, repeated timing campaign or automatic v0.2 resumption."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
