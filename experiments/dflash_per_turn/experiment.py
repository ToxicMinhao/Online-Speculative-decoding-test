"""Per-turn learning and matched replay protocol, independent of GPU execution.

Only canonical serving produces history/support. Score first, learn second.
Diagnostic branches are disposable and never become the current online draft.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

OUTPUT_CAP, BLOCK_SIZE, INPUT_LIMIT, CONTEXT_LIMIT, REPEATS = 1024, 10, 31744, 32768, 3
ARM = "AdaFlash OSD training path under our session protocol"
EXPECTED_STREAMS = {
    "wildchat_00": 20, "wildchat_01": 16, "wildchat_02": 17,
    "wildchat_19": 23, "personamem_persona42": 18, "personamem_persona404": 22,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def valid_ids(ids, *, output=False):
    require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 128256 for x in ids),
            "Exact nonempty token IDs required")
    require(len(ids) <= (OUTPUT_CAP if output else INPUT_LIMIT), "Token limit exceeded; no truncation")
    if not output:
        require(len(ids) + OUTPUT_CAP + BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
                "Complete history/output/proposals exceed context envelope")


def validate_stream(stream):
    require(isinstance(stream, dict) and stream.get("stream_id") in EXPECTED_STREAMS, "Unapproved stream")
    rows = stream.get("requests", [])
    require(len(rows) == EXPECTED_STREAMS[stream["stream_id"]] and
            [row.get("turn") for row in rows] == list(range(1, len(rows) + 1)),
            "Selected consecutive request count/order changed")
    require(all(isinstance(row.get("prompt"), str) and row["prompt"] for row in rows), "Missing source user text")
    require(isinstance(stream.get("source"), str) and type(stream.get("synthetic")) is bool,
            "Keep source and synthetic provenance explicit")
    return rows


def plan(streams):
    require([s["stream_id"] for s in streams] == list(EXPECTED_STREAMS), "Approved first-stream/order changed")
    for stream in streams:
        validate_stream(stream)
    return dict(schema_version=1, mode="per_turn_pilot", status="review_only", gpu_started=False,
                arm=ARM, seed=17, streams=[dict(stream_id=s["stream_id"], source=s["source"],
                    synthetic=s["synthetic"], requests=len(s["requests"])) for s in streams],
                canonical_requests=116, update_events=110, optimizer_steps=220,
                timed_replays=990, timing_repeats=REPEATS, output_cap=OUTPUT_CAP,
                block_size=BLOCK_SIZE, input_limit=INPUT_LIMIT, context_limit=CONTEXT_LIMIT,
                learning_rate=3e-4, passes=2, effective_response_batch_size=1, steps_per_event=2,
                diagnostics=dict(max_harmful=2, max_beneficial=1, learning_rate=1e-4,
                    earlier_request_rule="max(1, update_event - 1); previously observed, not held-out",
                    maximum_extra_updates=3, maximum_extra_steps=6, maximum_extra_timed_replays=54),
                notes=["990 repeats reuse 116 request contexts; they are not independent examples.",
                       "Only canonical online answers enter training/history; evaluation precedes learning.",
                       "First complete stream must pass implementation checks before subsequent streams.",
                       "Only the declared lower-LR diagnostics may change learning rate; they never enter main state."])


def methods_for_turn(turn):
    require(type(turn) is int and turn >= 1, "Invalid turn")
    current = "F" if turn == 1 else f"D{turn - 1}"
    previous = None if turn == 1 else ("F" if turn == 2 else f"D{turn - 2}")
    return list(dict.fromkeys(m for m in ("F", current, previous) if m is not None))


def method_order(methods, repeat, stream_index, turn):
    require(methods and len(set(methods)) == len(methods) and repeat in range(REPEATS), "Invalid score matrix")
    offset = (repeat + stream_index + turn - 1) % len(methods)
    return methods[offset:] + methods[:offset]


def validate_measurement(output):
    valid_ids(output.get("output_ids"), output=True)
    require(isinstance(output.get("text"), str), "Missing generated text")
    for key in ("request_s", "activation_s", "native_activation_s", "cache_maintenance_s"):
        value = output.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid/missing cost: " + key)
    for key in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens"):
        require(type(output.get(key)) is int and output[key] >= 0, "Missing native count: " + key)
    require(output["accepted_draft_tokens"] <= len(output["output_ids"]), "Accepted count exceeds output")
    require(output.get("feedback") is None and output.get("capture_costs") is None and
            output.get("capture", False) is False, "Paired inference must disable feedback capture")


def score_input(backend, item, methods, *, stream_index, emit, destination, extra=None):
    """Retain each raw repeat, including mismatches, without rerendering its input."""
    ids = item["input_ids"]
    valid_ids(ids)
    for repeat in range(REPEATS):
        order = method_order(methods, repeat, stream_index, item["turn"])
        emit("method_order", dict(turn=item["turn"], repeat=repeat, methods=order, **(extra or {})))
        for method in order:
            expected = backend.checkpoints[method]["checkpoint_identity_sha256"]
            output = backend.serve(list(ids), method, capture=False, turn_index=item["turn"], repeat=repeat)
            require(isinstance(output, dict), "Missing native measurement")
            input_match = output.get("input_ids") == ids and output.get("input_ids_sha256") == digest(ids)
            output_match = output.get("output_ids") == item["output_ids"]
            identity_match = output.get("checkpoint_identity_sha256") == expected
            scored = {**output, "episode_id": item["episode_id"], "turn": item["turn"],
                "question_id": item["question_id"], "source": item["source"], "repeat": repeat,
                "checkpoint_id": method, "capture": False, "training_data": False,
                "reference_input_ids_sha256": digest(ids),
                "reference_output_ids_sha256": digest(item["output_ids"]),
                "context_tokens": len(ids), "output_tokens": len(output.get("output_ids", [])),
                "cap_hit": len(output.get("output_ids", [])) == OUTPUT_CAP,
                "input_match": input_match, "canonical_output_match": output_match,
                "checkpoint_identity_match": identity_match,
                "same_generation_valid": input_match and output_match and identity_match,
                **(extra or {})}
            destination.append(scored)
            emit("score" if extra is None else "diagnostic_score", scored)
            # Preserve the row before validating. Equality failures do not become
            # speedup claims; reports retain all failures and their denominators.
            validate_measurement(output)
            if not scored["same_generation_valid"]:
                emit("output_or_input_mismatch", scored)


def validate_update(update, event, support):
    checkpoint = update.get("checkpoint", {})
    require(checkpoint.get("checkpoint_id") == f"D{event}" and update.get("optimizer_steps") == 2 * event,
            "Wrong cumulative checkpoint/optimizer steps")
    account = update.get("accounting", {})
    require(account.get("optimizer_steps_this_event") == 2 and
            account.get("optimizer_steps_total") == 2 * event and
            account.get("response_presentations_this_event") == 2 and
            account.get("support_turns") == [event], "Not exactly two passes/steps over newest response")
    recipe = account.get("recipe", {})
    require(recipe.get("responses_per_step") == 1 and recipe.get("passes") == 2 and
            recipe.get("learning_rate") == 3e-4 and recipe.get("loss") == "forward_KL",
            "Per-turn main learning recipe changed")
    audits = account.get("response_audits", [])
    require(len(audits) == 1, "Update includes old responses")
    start = len(support["input_ids"])
    end = start + len(support["output_ids"])
    audit = audits[0]
    mask = [0] * start + [1] * (end - start) + [0] * (BLOCK_SIZE - 1)
    require(audit.get("response_start") == start and audit.get("response_end_exclusive") == end and
            audit.get("loss_mask_sha256") == digest(mask), "History/prefix received loss or response boundary differs")
    presentations = [p for batch in account.get("batches", []) for p in batch.get("presentations", [])]
    require(len(presentations) == 2, "Missing actual training exposures")
    for presentation in presentations:
        positions = presentation.get("unique_supervised_position_ids")
        require(isinstance(positions, list) and all(start <= pos < end for pos in positions),
                "Supervised positions include previous answers or user input")
        require(len(set(positions)) == len(positions) and
                presentation.get("unique_supervised_positions") == len(positions) and
                type(presentation.get("eligible_positions")) is int and
                presentation["eligible_positions"] >= len(positions) and
                presentation.get("zero_supervision") is (not positions),
                "Supervised exposure counters or zero-supervision flag differ")
        require(presentation.get("turn") == event and presentation.get("record_id") == support["record_id"],
                "Training replay used noncanonical or old feedback")
    return True


def implementation_checks(result):
    n = result["planned_requests"]
    ids = [row["turn"] for row in result["canonical"]]
    return dict(status="passed" if result.get("status") == "complete" and
                ids == list(range(1, n + 1)) and len(result["scores"]) == 9 * (n - 1) and
                all(row.get("same_generation_valid") for row in result["scores"]) and
                len(result["updates"]) == n - 1 and
                all(u.get("mask_and_schedule_check") is True for u in result["updates"]) and
                result.get("cleanup", {}).get("status") == "passed" else "failed",
                canonical_requests=len(ids), expected_requests=n, timing_replays=len(result["scores"]),
                expected_timing_replays=9 * (n - 1), update_events=len(result["updates"]),
                invalid_scores=sum(not s.get("same_generation_valid") for s in result["scores"]),
                scope="First-stream implementation gate; no positive performance threshold.")


def run_stream(stream, backend, *, stream_index=0, case_budget=None, emit=lambda kind, data: None,
               snapshot=lambda result: None, classify=None):
    """No automatic retries. Failed/partial streams preserve events and recovery."""
    rows = validate_stream(stream)
    case_budget = case_budget if case_budget is not None else {"harmful": 0, "beneficial": 0}
    require(set(case_budget) == {"harmful", "beneficial"}, "Invalid global case budget")
    if classify is None:
        if __package__:
            from .report import classify_update as classify
        else:
            from report import classify_update as classify
    result = dict(schema_version=1, mode="per_turn", status="running", episode_id=stream["stream_id"],
        source=stream["source"], synthetic=stream["synthetic"], planned_requests=len(rows), seed=17,
        canonical=[], scores=[], support=[], updates=[], diagnostics=[], retirement=[], implementation_checks=None,
        stream_binding=stream.get("binding"), arm=ARM, output_cap=OUTPUT_CAP)
    messages, protected_events, protected_checkpoints = [], set(), {"F"}
    started = time.monotonic()
    try:
        result["startup"] = backend.start_episode(stream["stream_id"])
        for row in rows:
            turn = row["turn"]
            current = "F" if turn == 1 else f"D{turn - 1}"
            messages.append(dict(role="user", content=row["prompt"]))
            ids = backend.encode(copy.deepcopy(messages))
            valid_ids(ids)
            expected = backend.checkpoints[current]["checkpoint_identity_sha256"]
            output = backend.serve(list(ids), current, capture=False, turn_index=turn, repeat=None)
            item = {**output, "episode_id": stream["stream_id"], "source": stream["source"],
                "turn": turn, "question_id": row.get("request_id", f'{stream["stream_id"]}:t{turn:02d}'),
                "checkpoint_id": current, "canonical": True, "capture": False,
                "source_user_text": row["prompt"], "source_user_text_sha256":
                    hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                "source_message_ids": row.get("source_message_ids"),
                "context_tokens": len(ids), "output_tokens": len(output.get("output_ids", [])),
                "cap_hit": len(output.get("output_ids", [])) == OUTPUT_CAP,
                "reference_input_ids_sha256": digest(ids)}
            result["canonical"].append(item)
            emit("canonical", item)
            validate_measurement(output)
            require(output.get("input_ids") == ids and output.get("input_ids_sha256") == digest(ids),
                    "Canonical worker changed exact input IDs")
            require(output.get("checkpoint_identity_sha256") == expected, "Canonical weights differ")
            messages.append(dict(role="assistant", content=output["text"], output_ids=list(output["output_ids"])))
            score_input(backend, item, methods_for_turn(turn), stream_index=stream_index,
                        emit=emit, destination=result["scores"])
            snapshot(result)
            if turn > 1:
                selection = classify(result, turn)
                emit("latest_update_classification", selection)
                kind = selection.get("kind")
                maximum = {"harmful": 2, "beneficial": 1}.get(kind, 0)
                if kind in case_budget and case_budget[kind] < maximum:
                    event = turn - 1
                    earlier = result["canonical"][max(1, event - 1) - 1]
                    diagnostic = dict(event=event, kind=kind, selection=selection, next_turn=turn,
                        earlier_turn=earlier["turn"], status="running", scores=[],
                        retention_scope="Earlier already-observed request, not independent held-out generalization.")
                    result["diagnostics"].append(diagnostic)
                    protected_events.add(event)
                    pre = "F" if event == 1 else f"D{event - 1}"
                    protected_checkpoints.update((pre, current))
                    case_budget[kind] += 1
                    snapshot(result)  # preserve the selected event before branching
                    before = copy.deepcopy(backend.checkpoints[current])
                    diagnostic_started = time.monotonic()
                    branch = backend.diagnostic_event(event, result["updates"][event - 1])
                    diagnostic["branch"] = branch
                    require(branch.get("main_state_restored") is True and backend.checkpoints[current] == before,
                            "Diagnostic failed to restore main weights/optimizer/RNG")
                    low = branch["checkpoint"]["checkpoint_id"]
                    require(low == f"L{event}", "Diagnostic checkpoint must have a separate identity")
                    protected_checkpoints.add(low)
                    roles = {pre: "pre", current: "post", low: "low_lr"}
                    for window, saved in (("next", item), ("earlier", earlier)):
                        start_index = len(diagnostic["scores"])
                        score_input(backend, saved, [pre, current, low], stream_index=stream_index,
                            emit=emit, destination=diagnostic["scores"], extra={"window": window, "event": event})
                        for score in diagnostic["scores"][start_index:]:
                            score["role"] = roles[score["checkpoint_id"]]
                    diagnostic["actual_wall_s"] = time.monotonic() - diagnostic_started
                    diagnostic["status"] = "complete"
                    emit("diagnostic_complete", diagnostic)
                    snapshot(result)
            if turn < len(rows):
                support_started = time.monotonic()
                support = backend.prepare_support(copy.deepcopy(item))
                support["observed_support_prepare_s"] = time.monotonic() - support_started
                require(support.get("input_ids") == item["input_ids"] and
                        support.get("output_ids") == item["output_ids"] and support.get("turn") == turn,
                        "Support must be the actual newest online-serving answer")
                result["support"].append(support)
                emit("support", support)
                update_started = time.monotonic()
                update = backend.train_event(turn, [support])
                update["observed_update_call_s"] = time.monotonic() - update_started
                result["updates"].append(update)
                emit("update", update)  # preserve receipts even if validation fails
                update["mask_and_schedule_check"] = validate_update(update, turn, support)
                snapshot(result)
                keep = protected_checkpoints | {"F", f"D{turn}", current}
                if hasattr(backend, "retire"):
                    receipt = backend.retire(keep_ids=keep, protected_event_ids=protected_events)
                    result["retirement"].append(receipt)
                    emit("retirement", receipt)
            snapshot(result)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        emit("failure", dict(error=repr(error), completed_turns=len(result["canonical"])))
    finally:
        try:
            result["cleanup"] = backend.finish_episode()
        except Exception as error:
            result.update(status="failed", cleanup_error=repr(error))
        result["actual_wall_s"] = time.monotonic() - started
        result["experiment_wall_s"] = result["actual_wall_s"]
        result["implementation_checks"] = implementation_checks(result)
        snapshot(result)
    return result
