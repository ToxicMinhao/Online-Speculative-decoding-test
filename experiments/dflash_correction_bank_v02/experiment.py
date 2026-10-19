"""Causal orchestration without model imports or scientific parameter tuning.

Backend states are durable immutable references. BANK states precede reading, so
every measured replay includes descriptor extraction, routing and compilation.
The two learning trajectories never borrow each other's responses or feedback.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

from experiments.dflash_per_turn.experiment import (
    BLOCK_SIZE, CONTEXT_LIMIT, EXPECTED_STREAMS, INPUT_LIMIT, OUTPUT_CAP,
    digest, require, valid_ids, validate_stream, validate_update as validate_osd_update,
)

ARMS = ("F", "OSD", "BANK")
LEARNING_ARMS = ("BANK", "OSD")
REPEATS = 3
QUALIFICATION_CHECKS = (
    "zero_preserves_outputs", "compiled_fusion", "masks_and_conditioning",
    "compressed_feedback", "direct_retention_bounds", "current_response_fit_reported",
    "native_precision_reported", "cold_start", "rank_preserving_updates",
    "cumulative_protection_fixtures", "versioned_action_cache",
    "native_cost_profile_available",
)


def plan(streams):
    require([s["stream_id"] for s in streams] == list(EXPECTED_STREAMS),
            "Approved stream order changed; qualification uses the first stream")
    for stream in streams:
        validate_stream(stream)
    n = sum(len(s["requests"]) for s in streams)
    updates = n - len(streams)
    return dict(schema_version=1, mode="correction_bank_v02", status="review_only", gpu_started=False,
        seed=17, streams=[dict(stream_id=s["stream_id"], source=s["source"], synthetic=s["synthetic"],
                              requests=len(s["requests"])) for s in streams],
        request_contexts=n, canonical_requests_per_learning_arm=n, canonical_executions=2*n,
        update_events_per_learning_arm=updates, osd_optimizer_steps=2*updates,
        bank_scheduled_steps=2*updates, bank_steps_are_compute_matched=False,
        bank_feedback_summaries=n, timed_replays=9*n, timing_repeats=3,
        qualification=dict(stream_id=streams[0]["stream_id"], separate_from_campaign=True,
                           restart_all_streams=True, performance_win_required=False,
                           requests=3, timed_replays=27),
        block_size=BLOCK_SIZE, output_cap=OUTPUT_CAP, input_limit=INPUT_LIMIT,
        context_limit=CONTEXT_LIMIT, arms=list(ARMS), paired_input_trajectory="BANK",
        notes=["No checkpoint deduplication: a zero BANK correction still incurs read overhead.",
               "1044 timed executions reuse 116 contexts; they are not independent examples.",
               "Both learning arms supply their own canonical responses and feedback.",
               "Qualification includes correctness and fit measurement, not an efficacy threshold."])


def method_order(repeat, stream_index, turn):
    require(repeat in range(REPEATS), "Invalid repeat")
    offset = (repeat + stream_index + turn - 1) % len(ARMS)
    return list(ARMS[offset:] + ARMS[:offset])


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_measurement(output, *, bank=False):
    valid_ids(output.get("output_ids"), output=True)
    require(isinstance(output.get("text"), str), "Missing generated text")
    for key in ("request_s", "activation_s", "native_activation_s", "cache_maintenance_s"):
        require(_finite(output.get(key)), "Invalid/missing timer: " + key)
    for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
        require(type(output.get(key)) is int and output[key] >= 0, "Missing native counter: " + key)
    require(output["accepted_draft_tokens"] <= len(output["output_ids"]), "Invalid accepted-token count")
    require(output.get("capture", False) is False and output.get("feedback") is None and
            output.get("capture_costs") is None, "Canonical/replay serving must disable feedback capture")
    if bank:
        read = output.get("read", {})
        for key in ("forecast_saved_before_response", "descriptor_extracted", "routing_recomputed", "fusion_recomputed"):
            require(read.get(key) is True, "BANK read not causally recomputed: " + key)
        coefficients = read.get("coefficients")
        require(isinstance(coefficients, list) and len(coefficients) == 4 and
                all(_finite(a) and a <= 1 for a in coefficients) and
                sum(a > 0 for a in coefficients) <= 2, "Invalid bank coefficients")
        require(read.get("cost_in_request_s") is True, "Descriptor/routing/fusion omitted from request latency")


def validate_state(state, arm, turn):
    require(isinstance(state, dict) and state.get("arm") == arm and
            state.get("after_turn") == (0 if arm == "F" else turn-1), "Wrong pre-request state")
    require(isinstance(state.get("state_sha256"), str) and len(state["state_sha256"]) == 64,
            "State identity must be an immutable digest")
    if arm == "BANK":
        require(state.get("before_read") is True, "BANK snapshot was taken after the read decision")


def score_input(backend, item, states, *, stream_index, emit, destination):
    ids = item["input_ids"]
    for repeat in range(REPEATS):
        order = method_order(repeat, stream_index, item["turn"])
        emit("method_order", dict(turn=item["turn"], repeat=repeat, methods=order))
        for arm in order:
            state = copy.deepcopy(states[arm])
            state_digest = digest(state)
            output = backend.serve(list(ids), arm, state, capture=False,
                turn_index=item["turn"], repeat=repeat, canonical=False)
            input_match = output.get("input_ids") == ids and output.get("input_ids_sha256") == digest(ids)
            output_match = output.get("output_ids") == item["output_ids"]
            state_match = output.get("state_sha256") == states[arm]["state_sha256"] and digest(state) == state_digest
            if arm == "BANK":
                state_match = state_match and output.get("read", {}).get("snapshot_sha256") == state["state_sha256"]
            row = {**output, "episode_id": item["episode_id"], "source": item["source"],
                "synthetic": item["synthetic"], "turn": item["turn"], "question_id": item["question_id"],
                "arm": arm, "repeat": repeat, "capture": False, "training_data": False,
                "paired_input_trajectory": "BANK", "context_tokens": len(ids),
                "output_tokens": len(output.get("output_ids", [])),
                "cap_hit": len(output.get("output_ids", [])) == OUTPUT_CAP,
                "reference_input_ids_sha256": digest(ids), "reference_output_ids_sha256": digest(item["output_ids"]),
                "input_match": input_match, "canonical_output_match": output_match,
                "checkpoint_identity_match": state_match,
                "same_generation_valid": input_match and output_match and state_match}
            destination.append(row)
            emit("score", row)  # Preserve failures before raising a structural error.
            validate_measurement(output, bank=arm == "BANK")
            require(output.get('bank_head_capture') is None and not output.get('correction_bank_cycle_observations'),
                    'Timed replay collected bank features or causal cycle observations')
            if not row["same_generation_valid"]:
                emit("output_or_input_mismatch", row)


def validate_bank_update(update, item, state, *, final):
    require(update.get("arm") == "BANK" and update.get("event") == item["turn"], "Wrong bank event")
    require(update.get("state_before_sha256") == state["state_sha256"], "Bank feedback used another pre-read state")
    require(update.get("scheduled_steps") == (0 if final else 2), "Wrong scheduled bank write count")
    require(type(update.get("accepted_steps")) is int and 0 <= update["accepted_steps"] <= (0 if final else 2),
            "Invalid accepted-step count")
    for key in ("current_response_only", "frozen_reference", "protection_records_immutable", "served_record_precedes_trials"):
        require(update.get(key) is True, "Bank feedback/write validity failure: " + key)
    require(update.get("input_ids_sha256") == digest(item["input_ids"]) and
            update.get("output_ids_sha256") == digest(item["output_ids"]), "Bank feedback trajectory differs")


def implementation_checks(result):
    n = result["planned_requests"]
    bad_scores = sum(row.get("same_generation_valid") is not True for row in result["scores"])
    counts = {arm: sum(row["arm"] == arm for row in result["canonical"]) for arm in LEARNING_ARMS}
    updates = {arm: sum(row["arm"] == arm for row in result["updates"]) for arm in LEARNING_ARMS}
    cold = [row for row in result["canonical"] if row["arm"] == "BANK" and row["turn"] == 1]
    cold_ok = len(cold) == 1 and all(not any(row["read"]["coefficients"]) for row in cold)
    structural = (result["status"] == "complete" and all(c == n for c in counts.values()) and
        len(result["scores"]) == 9*n and not bad_scores and all(c == n-1 for c in updates.values()) and
        len(result["feedback_summaries"]) == n and cold_ok and
        all(row.get("mask_and_schedule_check") is True for row in result["updates"]) and
        result.get("cleanup", {}).get("status") == "passed")
    return dict(status="passed" if structural else "failed", canonical_per_arm=counts,
        update_events_per_arm=updates, timed_replays=len(result["scores"]), expected_timed_replays=9*n,
        invalid_scores=bad_scores, cold_start_passed=cold_ok,
        performance_win_required=False, paired_input_trajectory="BANK")


def run_stream(stream, backend, *, stream_index=0, qualification=False,
               emit=lambda kind, data: None, snapshot=lambda result: None):
    rows = validate_stream(stream)
    if qualification:
        require(stream["stream_id"] == "wildchat_00", "Qualification uses first planned stream")
        rows = rows[:3]
    result = dict(schema_version=1, mode="correction_bank_v02", status="running", episode_id=stream["stream_id"],
        source=stream["source"], synthetic=stream["synthetic"], planned_requests=len(rows), seed=17,
        qualification=qualification, paired_input_trajectory="BANK", canonical=[], scores=[], updates=[],
        feedback_summaries=[], pre_request_states=[], history_comparisons=[], qualification_checks=None)
    histories = {arm: [] for arm in LEARNING_ARMS}
    started = time.monotonic()
    try:
        result["startup"] = backend.start_episode(stream["stream_id"], seed=17, qualification=qualification)
        for request in rows:
            turn = request["turn"]
            states = {arm: backend.snapshot_state(arm, turn) for arm in ARMS}
            for arm, state in states.items():
                validate_state(state, arm, turn)
            result["pre_request_states"].append(dict(turn=turn, states=copy.deepcopy(states)))
            emit("pre_request_states", result["pre_request_states"][-1])
            items = {}
            for arm in LEARNING_ARMS:
                histories[arm].append(dict(role="user", content=request["prompt"]))
                ids = backend.encode(copy.deepcopy(histories[arm]))
                valid_ids(ids)
                state = copy.deepcopy(states[arm]); before = digest(state)
                output = backend.serve(list(ids), arm, state, capture=False,
                    turn_index=turn, repeat=None, canonical=True)
                item = {**output, "episode_id": stream["stream_id"], "source": stream["source"],
                    "synthetic": stream["synthetic"], "turn": turn, "arm": arm, "canonical": True,
                    "record_id": (f'{stream["stream_id"]}:t{turn:02d}' if arm == "OSD" else
                                  f'{stream["stream_id"]}:BANK:t{turn:02d}'),
                    "question_id": request.get("request_id", f'{stream["stream_id"]}:t{turn}'),
                    "source_user_text": request["prompt"], "source_message_ids": request.get("source_message_ids"),
                    "source_user_text_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                    "context_tokens": len(ids), "output_tokens": len(output.get("output_ids", [])),
                    "cap_hit": len(output.get("output_ids", [])) == OUTPUT_CAP, "capture": False}
                result["canonical"].append(item); emit("canonical", item)
                validate_measurement(output, bank=arm == "BANK")
                require(output.get("input_ids") == ids and output.get("input_ids_sha256") == digest(ids),
                        "Canonical worker changed input token IDs")
                require(output.get("state_sha256") == state["state_sha256"] and digest(state) == before,
                        "Canonical serving changed or used incorrect pre-request state")
                if arm == "BANK":
                    require(output["read"].get("snapshot_sha256") == state["state_sha256"], "Wrong bank read snapshot")
                    require(turn > 1 or not any(output["read"]["coefficients"]), "Specified turn-1 frozen fallback failed")
                items[arm] = item
            comparison = dict(turn=turn, inputs_match=items["BANK"]["input_ids"] == items["OSD"]["input_ids"],
                outputs_match=items["BANK"]["output_ids"] == items["OSD"]["output_ids"],
                bank_input_sha256=digest(items["BANK"]["input_ids"]), osd_input_sha256=digest(items["OSD"]["input_ids"]),
                bank_output_sha256=digest(items["BANK"]["output_ids"]), osd_output_sha256=digest(items["OSD"]["output_ids"]))
            result["history_comparisons"].append(comparison); emit("history_comparison", comparison)
            score_input(backend, items["BANK"], states, stream_index=stream_index, emit=emit, destination=result["scores"])
            # No arm learns until every paired pre-update measurement is saved.
            for arm in LEARNING_ARMS:
                final = turn == len(rows)
                if final and arm == "OSD":
                    continue
                update = backend.learn(arm, copy.deepcopy(items[arm]), copy.deepcopy(states[arm]),
                    turn_index=turn, final=final, qualification=qualification)
                update = {**update, "arm": arm, "event": turn}
                if arm == "BANK":
                    validate_bank_update(update, items[arm], states[arm], final=final)
                    result["feedback_summaries"].append(update); emit("feedback_summary", update)
                else:
                    validate_osd_update(update, turn, items[arm])
                update["mask_and_schedule_check"] = True
                if not final:
                    result["updates"].append(update); emit("update", update)
            for arm in LEARNING_ARMS:
                item = items[arm]
                histories[arm].append(dict(role="assistant", content=item["text"], output_ids=list(item["output_ids"])))
            snapshot(result)
        if qualification:
            checks = backend.qualify(copy.deepcopy(result))
            require(isinstance(checks, dict), "Missing direct qualification evidence")
            result["qualification_checks"] = checks; emit("qualification", checks)
            require(all(checks.get(key) is True for key in QUALIFICATION_CHECKS), "Implementation qualification failed")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        emit("failure", dict(error=result["error"]))
    finally:
        try:
            result["cleanup"] = backend.close_episode(copy.deepcopy(result))
        except Exception as error:
            result["cleanup"] = dict(status="failed", error=f"{type(error).__name__}: {error}")
            result["status"] = "failed"
        result["actual_wall_s"] = time.monotonic()-started
        result["implementation_checks"] = implementation_checks(result)
        snapshot(result)
    return result
