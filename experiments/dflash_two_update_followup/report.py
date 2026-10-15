"""Descriptive follow-up accounting; repeated timings are not independent data."""
from collections import defaultdict
import math


INFERENCE_SCOPE = ("Available-checkpoint request latency includes prefill, drafting and verification. "
                   "All scores use original saved inputs with capture disabled. Learning, publication "
                   "and cache maintenance are outside this timer.")
WINDOWS = {"17-24": list(range(17, 25)), "25-32": list(range(25, 33))}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def measured_sum(records, key):
    """Absence is unmeasured; an empty observation set is not measured zero."""
    return sum(row[key] for row in records) if records and all(_number(row.get(key)) for row in records) else None


def _ratio(numerator, denominator, scale=1):
    return scale * numerator / denominator if numerator is not None and denominator else None


def totals(rows):
    valid_outputs = rows and all(isinstance(row.get("output_ids"), list) and row["output_ids"] and
                                all(type(token) is int and token >= 0 for token in row["output_ids"]) for row in rows)
    tokens = sum(len(row["output_ids"]) for row in rows) if valid_outputs else None
    calls, accepted, seconds, committed = (measured_sum(rows, field) for field in
                                           ("verification_calls", "accepted_draft_tokens", "request_s", "verification_committed_tokens"))
    contexts = {(row.get("episode_id"), row.get("turn"), row.get("input_ids_sha256")) for row in rows}
    questions = {row.get("question_id") for row in rows if row.get("question_id") is not None}
    return {"measurement_scope": INFERENCE_SCOPE, "executions": len(rows),
            "distinct_request_contexts": len(contexts), "distinct_question_ids": len(questions),
            "output_tokens": tokens, "verification_calls": calls, "accepted_draft_tokens": accepted,
            "complete_request_s": seconds, "mean_request_s": _ratio(seconds, len(rows)),
            "calls_per_1000_output_tokens": _ratio(calls, tokens, 1000),
            "accepted_draft_tokens_per_verification": _ratio(accepted, calls),
            "committed_output_tokens_per_verification": _ratio(tokens, calls),
            "verification_committed_tokens": committed,
            "verification_committed_tokens_per_call": _ratio(committed, calls),
            "output_tokens_per_second": _ratio(tokens, seconds)}


def _methods(mode, window):
    if mode == "checkpoints-only":
        return ["F", "C1", "C2"] if window == "25-32" else []
    return ["F", "C1", "A1"] if window == "17-24" else ["F", "C1", "C2", "A2"]


def _keys(rows):
    return [(row.get("episode_id"), row.get("turn"), row.get("repeat")) for row in rows]


def _valid_pair(left, right, expected):
    if not expected or len(left) != len(expected) or len(right) != len(expected):
        return False
    if len(set(_keys(left))) != len(left) or len(set(_keys(right))) != len(right):
        return False
    if set(_keys(left)) != expected or set(_keys(right)) != expected:
        return False
    indexed = {key: row for key, row in zip(_keys(right), right)}
    for key, row in zip(_keys(left), left):
        other = indexed[key]
        if not (row.get("same_generation_valid") is True and other.get("same_generation_valid") is True and
                row.get("input_match") is True and other.get("input_match") is True and
                row.get("checkpoint_identity_match") is True and other.get("checkpoint_identity_match") is True and
                row.get("question_id") == other.get("question_id") and
                isinstance(row.get("input_ids_sha256"), str) and
                row.get("input_ids_sha256") == other.get("input_ids_sha256") and
                row.get("input_ids_sha256") == row.get("reference_input_ids_sha256") and
                other.get("input_ids_sha256") == other.get("reference_input_ids_sha256") and
                isinstance(row.get("output_ids"), list) and row.get("output_ids") == other.get("output_ids")):
            return False
    # A method's checkpoint must stay fixed across requests and repeats in an episode.
    for records in (left, right):
        by_episode = defaultdict(set)
        for row in records:
            by_episode[row["episode_id"]].add(row.get("checkpoint_identity_sha256"))
        if any(len(values) != 1 or None in values for values in by_episode.values()):
            return False
    return True


def _reduction(new, base):
    return 100 * (1 - new / base) if new is not None and base else None


def _window(episodes, window, repeat):
    mode = episodes[0]["mode"]
    methods = _methods(mode, window)
    repeats = range(3) if repeat == "pooled" else [repeat]
    expected = {(episode["episode_id"], turn, value) for episode in episodes
                for turn in WINDOWS[window] for value in repeats}
    groups = {method: [row for episode in episodes for row in episode["scores"]
                       if row.get("turn") in WINDOWS[window] and row.get("repeat") in repeats
                       and row.get("checkpoint_id") == method] for method in methods}
    metrics = {method: totals(rows) for method, rows in groups.items()}
    comparisons = []
    pairs = [(method, "F") for method in methods if method != "F"]
    pairs += [(candidate, control) for candidate, control in (("C2", "C1"), ("A1", "C1"), ("A2", "C2"))
              if candidate in methods and control in methods]
    for candidate, control in pairs:
        valid = _valid_pair(groups[candidate], groups[control], expected)
        left, right = metrics[candidate], metrics[control]
        comparisons.append({"candidate": candidate, "control": control,
                            "paired_same_generation_valid": valid,
                            "latency_reduction_percent": _reduction(left["complete_request_s"], right["complete_request_s"]) if valid else None,
                            "verification_call_reduction_percent": _reduction(left["verification_calls"], right["verification_calls"]) if valid else None,
                            "calls_per_1000_token_reduction_percent": _reduction(left["calls_per_1000_output_tokens"], right["calls_per_1000_output_tokens"]) if valid else None})
    return {"mode": mode, "window": window, "repeat": repeat,
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "expected_distinct_contexts": len(episodes) * len(WINDOWS[window]),
            "metrics": metrics, "comparisons": comparisons,
            "pooled_repeat_note": "Ratio of summed measurements; repeated contexts are not independent observations." if repeat == "pooled" else None}


def _timers(records):
    keys = sorted({key for row in records for key in row if key.endswith("_s")})
    return {key: {"sum_s": measured_sum(records, key),
                  "measured_records": sum(_number(row.get(key)) for row in records),
                  "records": len(records)} for key in keys}


def cost_breakdown(episode):
    canonical, updates, support = episode.get("canonical", []), episode.get("updates", []), episode.get("support", [])
    full_canonical = ([row.get("turn") for row in canonical] == list(range(1, 33)) and
                      [row.get("checkpoint_id") for row in canonical] == ["F"] * 16 + ["A1"] * 8 + ["A2"] * 8)
    full_updates = ([row.get("checkpoint_id") for row in updates] == ["A1", "A2"] and
                    [row.get("optimizer_steps") for row in updates] == [8, 16])
    full_support = [row.get("support_turn") for row in support] == list(range(9, 25))
    components = {"canonical_serving_s": measured_sum(canonical, "request_s"),
                  "canonical_cache_maintenance_s": measured_sum(canonical, "cache_maintenance_s"),
                  "canonical_publication_s": measured_sum(canonical, "activation_s"),
                  "observed_support_preparation_s": measured_sum(support, "observed_support_prepare_s"),
                  "observed_synchronous_update_wait_s": measured_sum(updates, "observed_update_call_s")}
    complete = (episode.get("status") == "complete" and episode.get("mode") == "author-path" and
                full_canonical and full_updates and full_support and all(value is not None for value in components.values()))
    per_update = []
    for update in updates:
        accounting = update.get("accounting", update)
        exposure = {key: value for key, value in accounting.items()
                    if key == "response_audits" or any(term in key for term in
                        ("anchor", "eligible", "presentation", "supervis", "response_order", "support_turn", "optimizer_steps"))}
        presentations = [item for batch in accounting.get("batches", []) for item in batch.get("presentations", [])]
        exposure["sampled_presentation_totals"] = {key: measured_sum(presentations, key) for key in
            ("sampled_blocks", "eligible_blocks", "eligible_positions", "unique_supervised_positions", "weighted_position_sum")}
        per_update.append({"checkpoint_id": update.get("checkpoint_id"), "optimizer_steps": update.get("optimizer_steps"),
                           "observed_update_call_s": update.get("observed_update_call_s"),
                           "top_level_timers": {key: value for key, value in update.items() if key.endswith("_s")},
                           "accounting_costs": accounting.get("costs"), "costs": update.get("costs"),
                           "checkpoint_transfer": update.get("checkpoint_transfer"),
                           "supervised_exposure": exposure})
    return {"complete_canonical_trajectory": full_canonical, "complete_update_schedule": full_updates,
            "complete_support_schedule": full_support, "complete_nonoverlapping_accounting": complete,
            "nonoverlapping_components": components,
            "accounted_author_episode_s": sum(components.values()) if complete else None,
            "unmeasured_required_components": [name for name, value in components.items() if value is None],
            "capture": {"proposed_block_tensor_capture": "disabled",
                        "observed_unexpected_capture_turns": [row.get("turn") for row in canonical
                                                              if row.get("feedback") is not None or row.get("capture_costs") is not None]},
            "support_preparation_timers": _timers(support),
            "canonical_timers": _timers(canonical), "per_update": per_update,
            "lifecycle_receipt": episode.get("cleanup"), "experiment_wall_s": episode.get("experiment_wall_s"),
            "scope": ("Sum only the five nonoverlapping components. Canonical request includes any inline token collection; "
                      "observed support preparation includes separate token metadata serialization. Teacher preparation/replay is within training_s; "
                      "training/export/state-save/transfer and trainer startup are within observed_update_call_s. "
                      "Native activation is within canonical publication. Nested timers are diagnostic and are never added again. "
                      "Benchmark serving/checkpoint switches and initial serving-engine lifecycle are excluded and retained separately. "
                      "Missing timers are unmeasured, never assumed free. A partial canonical trajectory has no whole-session total."),
            "frozen_full_session_comparison": "unavailable: requested common-input benchmarks do not measure a complete frozen own-history session"}


def _trajectory(episode):
    rows = episode.get("trajectory_comparisons", [])
    return {"compared_turns": len(rows), "input_match_turns": [row["turn"] for row in rows if row.get("input_match") is True],
            "input_mismatch_turns": [row["turn"] for row in rows if row.get("input_match") is False],
            "output_match_turns": [row["turn"] for row in rows if row.get("output_match") is True],
            "output_mismatch_turns": [row["turn"] for row in rows if row.get("output_match") is False],
            "interpretation": "Own-history divergence is retained evidence; it is distinct from required common-input native output preservation."}


def summarize(episodes):
    episodes = list(episodes)
    seen = set()
    for episode in episodes:
        key = (episode.get("mode"), episode.get("episode_id"))
        if key in seen:
            raise ValueError("Duplicate episode/mode would double-count repeated evidence")
        if episode.get("mode") not in ("checkpoints-only", "author-path"):
            raise ValueError("Unknown follow-up result mode")
        seen.add(key)
    reports = []
    for episode in episodes:
        windows = [window for window in WINDOWS if _methods(episode["mode"], window)]
        reports.append({"episode_id": episode["episode_id"], "scenario": episode["scenario"],
                        "replica": episode["replica"], "mode": episode["mode"], "status": episode["status"],
                        "error": episode.get("error"), "cleanup_error": episode.get("cleanup_error"),
                        "reference_source_binding": episode.get("reference_source_binding"),
                        "original_update_costs": episode.get("original_update_costs"),
                        "original_capture_costs": episode.get("original_capture_costs"),
                        "original_cost_scope": episode.get("original_cost_scope"),
                        "windows": [_window([episode], window, repeat) for window in windows for repeat in (0, 1, 2, "pooled")],
                        "cost_breakdown": cost_breakdown(episode), "trajectory": _trajectory(episode)})
    scenarios, pooled = [], []
    for mode in sorted({episode["mode"] for episode in episodes}):
        selected = [episode for episode in episodes if episode["mode"] == mode]
        for window in WINDOWS:
            if not _methods(mode, window):
                continue
            for repeat in (0, 1, 2, "pooled"):
                pooled.append(_window(selected, window, repeat))
                for scenario in sorted({episode["scenario"] for episode in selected}):
                    subset = [episode for episode in selected if episode["scenario"] == scenario]
                    scenarios.append({"scenario": scenario, **_window(subset, window, repeat)})
    return {"schema_version": 1, "episodes": reports, "scenarios": scenarios, "pooled": pooled,
            "interpretation_limits": [INFERENCE_SCOPE,
                "C2 versus C1 is measured on identical later saved contexts; previous early-versus-late percentages did not isolate this comparison.",
                "A1/C1 and A2/C2 compare complete training recipes under the common protocol; they do not isolate an individual sampling or weighting component.",
                "Three timing repeats share prompts, checkpoints and training seed17. Descriptive pooling does not create independent prompts, replicas or training seeds.",
                "Keep every episode, scenario, timing repeat, failed row and trajectory discrepancy; invalid pairs suppress comparison percentages, not raw measurements.",
                "Actual anchors and supervised exposure can differ between recipes; raw training losses with different reductions are not compared.",
                "Accepted proposal tokens, all committed output progress and trace-derived verification progress have separate denominators. Initial target tokens occur in all-output progress.",
                "Full cost is reconstructed only for complete canonical trajectories with all required timers; benchmark replays and nested timers are excluded from that sum.",
                "Model/context adaptations preclude calling the author-path arm an exact paper reproduction. R3/R4 remain reserved; no result selects a mechanism or advances the stage."]}
