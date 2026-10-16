"""Descriptive per-turn accounting. This module cannot load models or run updates."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


INFERENCE_SCOPE = (
    "Performance with an available checkpoint: complete request latency includes prefill, "
    "drafting and target verification, with feedback capture disabled. Training, activation "
    "and external cache maintenance are reported separately."
)
SELECTION_CRITERION = (
    "Use all three valid paired repeats on the next request. More verification calls in "
    "all three identifies drafting harm, even if timing improves (reported as a tradeoff). "
    "Equal calls in all three and slower timing in all three identifies timing-only harm. "
    "Fewer calls in all three with no timing regression, or equal calls with strictly faster "
    "timing in all three, identifies a beneficial control. No percentage threshold is used. "
    "Choose chronologically at most two harmful events and one beneficial event."
)
METRICS = (
    "executions", "output_tokens", "complete_request_s", "verification_calls",
    "calls_per_1000_output_tokens", "accepted_draft_tokens",
    "accepted_draft_tokens_per_verification", "verification_committed_tokens",
    "verification_committed_tokens_per_call", "committed_output_tokens_per_verification",
    "output_tokens_per_second",
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def measured_sum(rows, key):
    return sum(row[key] for row in rows) if rows and all(_number(row.get(key)) for row in rows) else None


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def _reduction(candidate, control):
    return 100 * (1 - candidate / control) if candidate is not None and control else None


def _tokens(value):
    return isinstance(value, list) and bool(value) and all(type(token) is int and token >= 0 for token in value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def totals(rows):
    rows = list(rows)
    tokens = sum(len(row["output_ids"]) for row in rows) if rows and all(_tokens(row.get("output_ids")) for row in rows) else None
    calls, seconds, accepted = (measured_sum(rows, key) for key in
                                ("verification_calls", "request_s", "accepted_draft_tokens"))
    progress = [dict(row, verification_committed_tokens=row.get("verification_committed_tokens", row.get("committed_tokens")))
                for row in rows]
    committed = measured_sum(progress, "verification_committed_tokens")
    return {"executions": len(rows), "output_tokens": tokens, "complete_request_s": seconds,
            "verification_calls": calls, "accepted_draft_tokens": accepted,
            "calls_per_1000_output_tokens": _ratio(calls, tokens, 1000),
            "accepted_draft_tokens_per_verification": _ratio(accepted, calls),
            "verification_committed_tokens": committed,
            "verification_committed_tokens_per_call": _ratio(committed, calls),
            "committed_output_tokens_per_verification": _ratio(tokens, calls),
            "output_tokens_per_second": _ratio(tokens, seconds)}


def _planned(result):
    count = result.get("planned_requests")
    if type(count) is not int or count < 1:
        raise ValueError("planned_requests must be a positive integer, including for partial results")
    return count


def _methods(turn):
    current = "F" if turn == 1 else f"D{turn - 1}"
    previous = None if turn == 1 else "F" if turn == 2 else f"D{turn - 2}"
    return current, previous


def _identity_problems(result):
    identities = defaultdict(set)
    for row in result.get("canonical", []) + result.get("scores", []):
        identities[row.get("checkpoint_id")].add(row.get("checkpoint_identity_sha256"))
    return {key for key, values in identities.items()
            if len(values) != 1 or any(not isinstance(value, str) or not value for value in values)}


def _validity(row, canonical, identity_problems=()):
    reasons = []
    if canonical is None:
        return ["missing_or_duplicate_canonical_request"]
    for key in ("same_generation_valid", "input_match", "canonical_output_match", "checkpoint_identity_match"):
        if row.get(key) is not True:
            reasons.append(key)
    if row.get("capture") is not False or row.get("feedback") is not None or row.get("capture_costs") is not None:
        reasons.append("capture_not_disabled")
    if row.get("status") in ("error", "failed"):
        reasons.append("failed_execution")
    for key in ("input_ids", "output_ids"):
        if not _tokens(row.get(key)) or not _tokens(canonical.get(key)) or row[key] != canonical[key]:
            reasons.append("exact_" + key + "_mismatch")
    if row.get("checkpoint_id") in identity_problems:
        reasons.append("checkpoint_identity_changed_or_missing")
    if not _number(row.get("request_s")) or not row["request_s"]:
        reasons.append("invalid_request_time")
    if type(row.get("verification_calls")) is not int or row["verification_calls"] < 0:
        reasons.append("invalid_verification_calls")
    return reasons


def _canonical(result, turn):
    matches = [row for row in result.get("canonical", []) if row.get("turn") == turn]
    return matches[0] if len(matches) == 1 else None


def _paired(result, turn, candidate, control, repeat, rows=None, canonical=None):
    records = result.get("scores", []) if rows is None else rows
    canonical = _canonical(result, turn) if canonical is None else canonical
    repeats = (0, 1, 2) if repeat == "pooled" else (repeat,)
    groups = [[row for row in records if row.get("turn") == turn and row.get("repeat") in repeats
               and row.get("checkpoint_id") == method] for method in (candidate, control)]
    reasons = []
    identity_problems = _identity_problems(result)
    for method, group in zip((candidate, control), groups):
        if len(group) != len(repeats) or {row.get("repeat") for row in group} != set(repeats):
            reasons.append(method + ":missing_or_duplicate_repeats")
        for row in group:
            reasons.extend(method + ":" + value for value in _validity(row, canonical, identity_problems))
    valid = not reasons
    new, old = (totals(group) for group in groups)
    value = {"candidate": candidate, "control": control, "repeat": repeat,
             "paired_same_generation_valid": valid, "invalid_reasons": sorted(set(reasons)),
             "expected_pairs": len(repeats), "candidate_metrics": new, "control_metrics": old,
             "latency_reduction_percent": _reduction(new["complete_request_s"], old["complete_request_s"]) if valid else None,
             "verification_call_reduction_percent": _reduction(new["verification_calls"], old["verification_calls"]) if valid else None,
             "calls_per_1000_token_reduction_percent": _reduction(new["calls_per_1000_output_tokens"], old["calls_per_1000_output_tokens"]) if valid else None}
    return value, groups


def classify_update(result, turn):
    """Pure, predeclared case selection; caller owns global quotas and branch state."""
    current, previous = _methods(turn)
    response = {"kind": None, "event": turn - 1, "next_turn": turn,
                "metric_basis": "invalid", "reason": "No previous update at turn one",
                "repeat_evidence": [], "valid_repeats": 0, "tradeoff": False,
                "selection_criterion": SELECTION_CRITERION}
    if previous is None:
        return response
    for repeat in range(3):
        value, _ = _paired(result, turn, current, previous, repeat)
        response["repeat_evidence"].append(value)
    valid = [row for row in response["repeat_evidence"] if row["paired_same_generation_valid"]]
    response["valid_repeats"] = len(valid)
    if len(valid) != 3:
        response["reason"] = "Missing, mismatched or invalid repeats; no case-selection claim"
        return response
    # Raw differences preserve direction when the reference needs zero calls
    # (e.g. an immediate target EOS). Its percentage reduction is undefined.
    calls = [row["control_metrics"]["verification_calls"] - row["candidate_metrics"]["verification_calls"]
             for row in valid]
    latency = [row["latency_reduction_percent"] for row in valid]
    response["metric_basis"] = "mixed_or_inconclusive"
    response["reason"] = "No consistent predeclared harmful or beneficial result across all three repeats"
    if all(value < 0 for value in calls):
        response.update(kind="harmful", metric_basis="verification_calls",
                        reason="Latest update increases verification calls on the next request in all three repeats",
                        tradeoff=any(value > 0 for value in latency))
    elif all(value == 0 for value in calls) and all(value < 0 for value in latency):
        response.update(kind="harmful", metric_basis="timing_only",
                        reason="Equal calls but slower requests in all three repeats; no demonstrated drafting harm")
    elif all(value > 0 for value in calls) and all(value >= 0 for value in latency):
        response.update(kind="beneficial", metric_basis="verification_calls",
                        reason="Fewer calls in all three repeats without a request-latency regression")
    elif all(value == 0 for value in calls) and all(value > 0 for value in latency):
        response.update(kind="beneficial", metric_basis="timing_only",
                        reason="Equal calls but faster requests in all three repeats; timing-only control")
    elif any(value > 0 for value in calls) and any(value < 0 for value in latency):
        response["tradeoff"] = True
    return response


def select_diagnostic_events(results):
    """Select only in input stream order and then turn order, never by effect size."""
    selected, used = [], {"harmful": 0, "beneficial": 0}
    for result in results:
        for turn in range(2, _planned(result) + 1):
            decision = classify_update(result, turn)
            kind = decision["kind"]
            if kind and used[kind] < {"harmful": 2, "beneficial": 1}[kind]:
                selected.append({"episode_id": result["episode_id"], **decision})
                used[kind] += 1
    return selected


def _timers(rows):
    keys = sorted({key for row in rows for key in row if key.endswith("_s")})
    return {key: {"sum_s": measured_sum(rows, key), "records": len(rows),
                  "measured_records": sum(_number(row.get(key)) for row in rows)} for key in keys}


def cost_breakdown(result):
    canonical, updates, support = (result.get(key, []) for key in ("canonical", "updates", "support"))
    n = _planned(result)
    complete_schedule = ([row.get("turn") for row in canonical] == list(range(1, n + 1)) and
                         [row.get("event") for row in updates] == list(range(1, n)))
    observed_update = measured_sum(updates, "observed_update_call_s")
    trainer_startup = measured_sum(updates, "trainer_startup_s")
    recurring_update = (observed_update - trainer_startup if observed_update is not None and
                        trainer_startup is not None and observed_update >= trainer_startup else None)
    components = {"canonical_serving_s": measured_sum(canonical, "request_s"),
                  "canonical_activation_s": measured_sum(canonical, "activation_s"),
                  "canonical_cache_maintenance_s": measured_sum(canonical, "cache_maintenance_s"),
                  "synchronous_update_excluding_startup_s": recurring_update,
                  "separate_support_preparation_s": measured_sum(support, "observed_support_prepare_s")}
    complete = (result.get("status") == "complete" and complete_schedule and
                [row.get("support_turn", row.get("turn")) for row in support] == list(range(1, n)) and
                all(value is not None for value in components.values()))
    per_update = []
    for update in updates:
        accounting = update.get("accounting", {})
        exposure = {key: value for key, value in accounting.items()
                    if key == "response_audits" or any(word in key for word in
                        ("anchor", "eligible", "supervis", "presentation", "response_order", "optimizer_steps"))}
        presentations = [row for batch in accounting.get("batches", []) for row in batch.get("presentations", [])]
        exposure["sampled_presentation_totals"] = {key: measured_sum(presentations, key) for key in
            ("sampled_blocks", "eligible_blocks", "eligible_positions", "unique_supervised_positions", "weighted_position_sum")}
        per_update.append({"event": update.get("event"), "checkpoint": update.get("checkpoint", update.get("checkpoint_id")),
                           "optimizer_steps": update.get("optimizer_steps"),
                           "observed_update_call_s": update.get("observed_update_call_s"),
                           "top_level_timers": {key: value for key, value in update.items() if key.endswith("_s")},
                           "costs": update.get("costs"), "accounting_costs": accounting.get("costs"),
                           "checkpoint_transfer": update.get("checkpoint_transfer"), "supervised_exposure": exposure})
    return {"complete_update_schedule": complete_schedule, "complete_nonoverlapping_accounting": complete,
            "nonoverlapping_components": components,
            "reconstructed_recurring_session_s": sum(components.values()) if complete else None,
            "unmeasured_components": [key for key, value in components.items() if value is None],
            "canonical_timers": _timers(canonical), "support_timers": _timers(support), "per_update": per_update,
            "benchmark_timers": _timers(result.get("scores", [])), "actual_experiment_wall_s": result.get("actual_wall_s"),
            "observed_update_calls_including_startup_s": observed_update,
            "one_time_loading": {"trainer_startup_s": trainer_startup, "serving_startup_receipt": result.get("startup")},
            "diagnostic_recovery_io": {
                "optimizer_state_save_s": measured_sum(updates, "optimizer_state_save_s"),
                "scope": "Nested in the observed pilot update wait, not added again. Durable optimizer/RNG snapshots and their hashing support recovery/isolated diagnostics; do not claim this is an inherent training-algorithm cost."},
            "cleanup": result.get("cleanup"),
            "scope": ("Verification-feature capture is disabled. Canonical responses supply post-serving token metadata; "
                      "its separately measured support preparation is counted once. The synchronous update call is counted once "
                      "after subtracting its explicitly measured one-time trainer startup. Teacher replay, draft training, export, "
                      "state-save and transfer are nested in that call; "
                      "optimizer_state_save_s separately identifies diagnostic/recovery snapshot I/O while remaining in the actual pilot blocking total; "
                      "native publication is nested in activation. Diagnostic branches and benchmark replays/switches are not serving costs. "
                      "Nested timers are explanatory, never added again. Missing measurements are not assumed zero. "
                      "Each new canonical turn publishes a previously unused D(t-1); its activation is the production publication, "
                      "not a benchmark restoration. Initial model loading and diagnostic validation remain separate from recurring serving costs."),
            "training_inclusive_speedup": "not established: no complete frozen own-history session is measured"}


def _aggregate(results, pair):
    candidate_rows, control_rows, excluded, valid_turns = [], [], [], []
    expected = 0
    for result in results:
        for turn in range(1 if pair == "online_vs_frozen" else 2, _planned(result) + 1):
            expected += 1
            current, previous = _methods(turn)
            value, groups = _paired(result, turn, current, "F" if pair == "online_vs_frozen" else previous, "pooled")
            identity = {"episode_id": result["episode_id"], "turn": turn}
            if value["paired_same_generation_valid"]:
                candidate_rows.extend(groups[0])
                control_rows.extend(groups[1])
                valid_turns.append(identity)
            else:
                excluded.append({**identity, "reasons": value["invalid_reasons"]})
    candidate, control = totals(candidate_rows), totals(control_rows)
    return {"comparison": pair, "expected_contexts": expected, "expected_pairs": 3 * expected,
            "valid_contexts": len(valid_turns), "valid_pairs": 3 * len(valid_turns),
            "coverage_complete": len(valid_turns) == expected, "excluded_contexts": excluded,
            "candidate_metrics": candidate, "control_metrics": control,
            "latency_reduction_percent": _reduction(candidate["complete_request_s"], control["complete_request_s"]),
            "verification_call_reduction_percent": _reduction(candidate["verification_calls"], control["verification_calls"]),
            "calls_per_1000_token_reduction_percent": _reduction(candidate["calls_per_1000_output_tokens"], control["calls_per_1000_output_tokens"]),
            "scope": "Ratio of sums over fully valid three-repeat contexts only; see excluded contexts and denominator. "
                     "Turn one aliases online to frozen and adds no extra execution. Repeats are not independent requests."}


def _diagnostics(result):
    output = []
    for case in result.get("diagnostics", []):
        record = {key: value for key, value in case.items() if key != "scores"}
        record["comparisons"] = []
        next_turn = case.get("next_turn")
        if type(next_turn) is int and next_turn >= 2:
            current, _ = _methods(next_turn)
            record["accumulated_online_vs_frozen"], _ = _paired(result, next_turn, current, "F", "pooled")
            record["marginal_latest_update"] = classify_update(result, next_turn)
        for window, turn_key in (("next", "next_turn"), ("earlier", "earlier_turn")):
            turn = case.get(turn_key)
            canonical = _canonical(result, turn)
            rows = [row for row in case.get("scores", []) if row.get("window") == window]
            roles = {role: {row.get("checkpoint_id") for row in rows if row.get("role") == role}
                     for role in ("pre", "post", "low_lr")}
            for candidate_role, control_role in (("post", "pre"), ("low_lr", "pre"), ("low_lr", "post")):
                if len(roles[candidate_role]) != 1 or len(roles[control_role]) != 1:
                    record["comparisons"].append({"window": window, "candidate_role": candidate_role,
                        "control_role": control_role, "paired_same_generation_valid": False,
                        "invalid_reasons": ["missing_or_changed_diagnostic_checkpoint"]})
                    continue
                candidate, control = next(iter(roles[candidate_role])), next(iter(roles[control_role]))
                # Include branch scores in the identity-consistency check.
                diagnostic_result = {**result, "scores": result.get("scores", []) + rows}
                value, _ = _paired(diagnostic_result, turn, candidate, control, "pooled", rows=rows, canonical=canonical)
                record["comparisons"].append({"window": window, "candidate_role": candidate_role,
                                              "control_role": control_role, **value})
        record["earlier_probe_scope"] = "Earlier request is an in-sample retention diagnostic, not a held-out generalization test."
        record["interpretation"] = "An isolated smaller-rate rerun tests this event; it does not change the canonical trajectory or establish a generally optimal rate."
        output.append(record)
    return output


def summarize(results):
    results = list(results)
    if len({row["episode_id"] for row in results}) != len(results):
        raise ValueError("Duplicate conversation would double-count the same evidence")
    conversations = []
    for result in results:
        n = _planned(result)
        per_turn = []
        for turn in range(1, n + 1):
            canonical = _canonical(result, turn)
            current, previous = _methods(turn)
            comparisons = []
            for name, control in (("online_vs_frozen", "F"), ("current_vs_previous", previous)):
                if control is None:
                    continue
                for repeat in (0, 1, 2, "pooled"):
                    value, _ = _paired(result, turn, current, control, repeat)
                    comparisons.append({"comparison": name, **value})
            per_turn.append({"turn": turn, "context_tokens": len(canonical["input_ids"]) if canonical and _tokens(canonical.get("input_ids")) else None,
                             "output_tokens": len(canonical["output_ids"]) if canonical and _tokens(canonical.get("output_ids")) else None,
                             "cap_hit": canonical.get("cap_hit") if canonical else None,
                             "current_checkpoint": current, "previous_checkpoint": previous,
                             "comparisons": comparisons, "latest_update_classification": classify_update(result, turn)})
        expected_scores = 3 * sum(len(set(["F", *_methods(turn)]) - {None}) for turn in range(1, n + 1))
        conversations.append({"episode_id": result["episode_id"], "source": result["source"],
                              "synthetic": result.get("synthetic"), "status": result.get("status"),
                              "error": result.get("error"), "planned_requests": n,
                              "canonical_requests_observed": len(result.get("canonical", [])),
                              "expected_score_executions": expected_scores, "score_executions_observed": len(result.get("scores", [])),
                              "planned_update_events": n - 1, "planned_optimizer_steps": 2 * (n - 1),
                              "update_events_observed": len(result.get("updates", [])),
                              "all_raw_score_metrics_including_invalid": totals(result.get("scores", [])),
                              "comparisons": [_aggregate([result], pair) for pair in ("online_vs_frozen", "current_vs_previous")],
                              "turns": per_turn, "cost_breakdown": cost_breakdown(result),
                              "diagnostics": _diagnostics(result)})
    sources = []
    for source, synthetic in dict.fromkeys((row["source"], row.get("synthetic")) for row in results):
        subset = [row for row in results if (row["source"], row.get("synthetic")) == (source, synthetic)]
        sources.append({"source": source, "synthetic": synthetic, "conversations": len(subset),
                        "comparisons": [_aggregate(subset, pair) for pair in ("online_vs_frozen", "current_vs_previous")]})
    return {"schema_version": 1, "measurement_scope": INFERENCE_SCOPE, "conversations": conversations,
            "sources": sources, "selection_criterion": SELECTION_CRITERION,
            "eligible_diagnostic_selection": select_diagnostic_events(results),
            "interpretation_limits": [
                "The two datasets are reported separately; PersonaMem is synthetic. Six conversations and one training seed do not establish broad generalization.",
                "Every raw repeat and mismatch remains in the result files and CSV. Invalid contexts are excluded from gain claims with their full denominator disclosed.",
                "Online versus frozen measures accumulated adaptation. Current versus previous isolates the latest update on the same next request; erosion of an earlier gain is not necessarily underperformance versus frozen.",
                "Accepted draft tokens exclude correction/bonus tokens. Verification-committed progress includes them; all-output progress may additionally include an initial target token.",
                "The three timing repeats reuse identical inputs/checkpoints; do not treat them as three independent scientific samples.",
                "Canonical responses alone supply history and learning. Checkpoint replays and isolated diagnostic branches supply neither.",
                "Lower training loss alone is not success. Cost reporting does not imply training-inclusive or asynchronous deployment speedup."]}


def _csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items() if key in fields} for row in rows)


def _raw_csv_rows(results, diagnostics=False):
    for result in results:
        groups = [(None, result.get("scores", []))] if not diagnostics else [(case, case.get("scores", [])) for case in result.get("diagnostics", [])]
        for case, records in groups:
            identity_result = result if case is None else {**result, "scores": result.get("scores", []) + records}
            bad_ids = _identity_problems(identity_result)
            for row in records:
                canonical = _canonical(result, row.get("turn"))
                reasons = _validity(row, canonical, bad_ids)
                yield {"episode_id": result["episode_id"], "source": result["source"], "synthetic": result.get("synthetic"),
                       **{key: value for key, value in row.items() if key not in ("input_ids", "output_ids")},
                       "event": case.get("event") if case else None,
                       "context_tokens": len(row["input_ids"]) if _tokens(row.get("input_ids")) else None,
                       "output_tokens": len(row["output_ids"]) if _tokens(row.get("output_ids")) else None,
                       "exact_input_ids_sha256": _digest(row["input_ids"]) if _tokens(row.get("input_ids")) else None,
                       "exact_output_ids_sha256": _digest(row["output_ids"]) if _tokens(row.get("output_ids")) else None,
                       "report_valid": not reasons, "invalid_reasons": reasons,
                       "additional_record_json": {key: value for key, value in row.items() if key not in ("input_ids", "output_ids")}}


def _format(value):
    return "unmeasured/invalid" if value is None else f"{value:+.2f}%"


def write_reports(results, directory):
    """Write a fresh report directory; this is read-only with respect to input evidence."""
    results = list(results)
    summary = summarize(results)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    raw_fields = ["episode_id", "source", "synthetic", "event", "turn", "repeat", "window", "role", "checkpoint_id",
                  "checkpoint_identity_sha256", "context_tokens", "output_tokens", "cap_hit", "request_s", "verification_calls",
                  "accepted_draft_tokens", "verification_committed_tokens", "committed_tokens", "capture", "input_match",
                  "canonical_output_match", "checkpoint_identity_match", "same_generation_valid", "report_valid", "invalid_reasons",
                  "exact_input_ids_sha256", "exact_output_ids_sha256", "additional_record_json"]
    _csv(directory / "raw_repeats.csv", _raw_csv_rows(results), raw_fields)
    _csv(directory / "diagnostic_repeats.csv", _raw_csv_rows(results, diagnostics=True), raw_fields)
    per_turn = []
    for conversation in summary["conversations"]:
        for turn in conversation["turns"]:
            for comparison in turn["comparisons"]:
                row = {"episode_id": conversation["episode_id"], "source": conversation["source"],
                       "synthetic": conversation["synthetic"], "turn": turn["turn"],
                       "context_tokens": turn["context_tokens"], "output_tokens": turn["output_tokens"],
                       "cap_hit": turn["cap_hit"], **comparison}
                for side in ("candidate", "control"):
                    row.update({side + "_" + key: comparison[side + "_metrics"].get(key) for key in METRICS})
                per_turn.append(row)
    fields = ["episode_id", "source", "synthetic", "turn", "context_tokens", "output_tokens", "cap_hit",
              "comparison", "candidate", "control", "repeat", "expected_pairs", "paired_same_generation_valid", "invalid_reasons",
              "latency_reduction_percent", "verification_call_reduction_percent", "calls_per_1000_token_reduction_percent"]
    fields += [side + "_" + key for side in ("candidate", "control") for key in METRICS]
    _csv(directory / "per_turn.csv", per_turn, fields)
    lines = ["# Per-turn DFlash adaptation pilot", "", INFERENCE_SCOPE, "",
             "Positive percentages mean reductions. Ratios are computed from sums, never averaged per-request percentages. "
             "Only complete valid three-repeat contexts enter pooled comparisons; exclusions remain below and in the CSV.", ""]
    for source in summary["sources"]:
        lines.extend([f"## {source['source']}" + (" (synthetic)" if source["synthetic"] else ""), "",
                      "| Comparison | Valid/planned contexts | Request-time reduction | Call reduction |",
                      "|---|---:|---:|---:|"])
        for value in source["comparisons"]:
            lines.append(f"| {value['comparison']} | {value['valid_contexts']}/{value['expected_contexts']} | "
                         f"{_format(value['latency_reduction_percent'])} | {_format(value['verification_call_reduction_percent'])} |")
        lines.append("")
    for conversation in summary["conversations"]:
        lines.extend([f"## {conversation['episode_id']}", "",
                      f"Status: {conversation['status']}; canonical requests {conversation['canonical_requests_observed']}/{conversation['planned_requests']}; "
                      f"timed executions {conversation['score_executions_observed']}/{conversation['expected_score_executions']}; "
                      f"updates {conversation['update_events_observed']}/{conversation['planned_update_events']}.", ""])
        for value in conversation["comparisons"]:
            lines.append(f"- {value['comparison']}: request time {_format(value['latency_reduction_percent'])}, "
                         f"calls {_format(value['verification_call_reduction_percent'])}; "
                         f"{value['valid_contexts']}/{value['expected_contexts']} valid contexts.")
            if value["excluded_contexts"]:
                lines.append("  Excluded contexts (retained, not silently removed): " + json.dumps(value["excluded_contexts"]) + ".")
        costs = conversation["cost_breakdown"]
        lines.extend(["", "Recurring components (seconds; null means unmeasured): `" + json.dumps(costs["nonoverlapping_components"]) + "`.",
                      "Reconstructed recurring total: " + str(costs["reconstructed_recurring_session_s"]) + " s. "
                      "Actual experiment wall time (includes checks/replays/diagnostics): " + str(costs["actual_experiment_wall_s"]) + " s.",
                      "Diagnostic/recovery optimizer-state snapshot I/O: " + str(costs["diagnostic_recovery_io"]["optimizer_state_save_s"]) +
                      " s, already included in the observed pilot update wait. This is measured experiment infrastructure, not an inherent training-algorithm cost.", ""])
    lines.extend(["## Interpretation limits", "", *["- " + value for value in summary["interpretation_limits"]], ""])
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    case_lines = ["# Selected update case analyses", "", SELECTION_CRITERION, "",
                  "Earlier-request replay is an in-sample retention diagnostic. Branches must remain isolated from the main weights, optimizer and RNG state.", ""]
    for conversation in summary["conversations"]:
        for case in conversation["diagnostics"]:
            selection = case.get("selection", {})
            case_lines.extend([f"## {conversation['episode_id']} event {case.get('event')}", "",
                               f"Status: {case.get('status')}. Selection: {case.get('kind')} "
                               f"({selection.get('metric_basis', 'not recorded')}). " + selection.get("reason", ""), ""])
            accumulated = case.get("accumulated_online_vs_frozen", {})
            case_lines.append("On this next request, the main online checkpoint versus frozen reduces request time by "
                              + _format(accumulated.get("latency_reduction_percent")) + " and calls by "
                              + _format(accumulated.get("verification_call_reduction_percent")) + ".")
            if case.get("kind") == "harmful" and accumulated.get("paired_same_generation_valid"):
                call_gain = accumulated.get("verification_call_reduction_percent")
                if selection.get("metric_basis") == "timing_only":
                    case_lines.append("The latest update increases measured request time with unchanged verification calls. This timing-only result does not establish reduced draft-target agreement or erosion of an earlier drafting gain.")
                elif call_gain is not None and call_gain > 0:
                    case_lines.append("The latest update is harmful under the stated selection criterion, but accumulated adaptation still needs fewer verification calls than frozen. This is erosion of a useful adaptation, not evidence that all learning was harmful.")
                elif call_gain is not None and call_gain < 0:
                    case_lines.append("The latest-update regression coexists with more verification calls than frozen. The two comparisons remain distinct; this event alone does not explain all accumulated underperformance.")
            case_lines.extend(["", "| Saved request | Comparison | Valid | Request-time reduction | Call reduction |",
                               "|---|---|---|---:|---:|"])
            for value in case["comparisons"]:
                case_lines.append(f"| {value['window']} | {value['candidate_role']} / {value['control_role']} | "
                                  f"{value['paired_same_generation_valid']} | {_format(value.get('latency_reduction_percent'))} | "
                                  f"{_format(value.get('verification_call_reduction_percent'))} |")
                if not value["paired_same_generation_valid"]:
                    case_lines.append("\nInvalid comparison retained: " + json.dumps(value.get("invalid_reasons")) + ".\n")
            case_lines.extend(["", "`pre` is the exact pre-update checkpoint; `post` is the main 3e-4 update; "
                               "`low_lr` repeats the same feedback/anchors/exposure at 1e-4. "
                               f"Next request: {case.get('next_turn')}; earlier request: {case.get('earlier_turn')}.",
                               "Main state restored: " + str(case.get("branch", {}).get("main_state_restored")) + ". "
                               "Diagnostic wall time: " + str(case.get("actual_wall_s")) + " s (excluded from serving).",
                               case["earlier_probe_scope"], case["interpretation"],
                               "Complete isolation receipts, exposure and all paired metrics are retained in SUMMARY.json and diagnostic_repeats.csv.", ""])
    if not any(conversation["diagnostics"] for conversation in summary["conversations"]):
        complete = bool(summary["conversations"]) and all(
            conversation["status"] == "complete" and
            all(comparison["coverage_complete"] for comparison in conversation["comparisons"])
            for conversation in summary["conversations"])
        harmful = any(row["kind"] == "harmful" for row in summary["eligible_diagnostic_selection"])
        case_lines.append("No diagnostic branch results are recorded. This does not by itself establish that no harmful update exists; inspect completion and eligible-selection evidence in SUMMARY.json.")
        if complete and not harmful:
            case_lines.append("No reproducibly harmful update meets the predeclared criterion in these completed, fully valid conversations. This is not proof that every update helps or that no other workload can regress.")
    (directory / "CASES.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    return summary
