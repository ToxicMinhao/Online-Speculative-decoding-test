"""Bounded causal controller. Replays cannot become history or training feedback."""
from __future__ import annotations

import copy
import hashlib
import math
import time

from experiments.dflash_per_turn.experiment import (BLOCK_SIZE, CONTEXT_LIMIT, EXPECTED_STREAMS, INPUT_LIMIT,
    OUTPUT_CAP, digest, require, valid_ids, validate_measurement, validate_stream)

LEARNING_ARMS = ("H", "L", "G", "N")
ARMS = ("F",) + LEARNING_ARMS
LEARNING_RATES = {"H": 3e-4, "L": 1e-4, "G": 3e-4, "N": 3e-4}


def plan(streams):
    require([s["stream_id"] for s in streams] == list(EXPECTED_STREAMS), "Approved six-stream order changed")
    for stream in streams:
        validate_stream(stream)
    return {"schema_version": 1, "mode": "osd_agreement_filter", "status": "review_only", "gpu_started": False,
            "seed": 17, "learning_arms": list(LEARNING_ARMS), "benchmark_arms": list(ARMS),
            "streams": [{"stream_id": s["stream_id"], "source": s["source"], "synthetic": s["synthetic"],
                         "requests": 3} for s in streams],
            "own_history_canonical_generations": 72, "update_events": 48, "optimizer_steps": 96,
            "benchmark_contexts": 12, "scored_replays": 60, "replays_per_method_request": 1,
            "qualification": {"stream_id": "wildchat_00", "requests_per_learning_arm": 3,
                "own_history_canonical_generations": 12, "update_events": 8, "optimizer_steps": 16,
                "scored_replays": 10, "clean_restart_before_campaign": True},
            "output_cap": OUTPUT_CAP, "block_size": BLOCK_SIZE, "input_limit": INPUT_LIMIT,
            "context_limit": CONTEXT_LIMIT, "passes": 2, "steps_per_event": 2,
            "learning_rates": LEARNING_RATES, "primary_control": "H",
            "scope": "Development screen; single-shot latency is diagnostic, not deployment speedup."}


def _references(stream, references):
    require(set(references) == {2, 3}, "Require immutable canonical references for turns 2 and 3")
    for turn in (2, 3):
        item = references[turn]
        require(item.get("episode_id") == stream["stream_id"] and item.get("turn") == turn,
                "Canonical reference stream/turn mismatch")
        require(item.get("source_user_text") == stream["requests"][turn - 1]["prompt"],
                "Saved canonical user text differs from approved source")
        for key in ("input_ids", "output_ids"):
            valid_ids(item.get(key), output=key == "output_ids")
            require(item.get(key + "_sha256") == digest(item[key]), "Immutable canonical token hash mismatch")


def score(backend, reference, checkpoint, arm, *, emit, destination):
    expected = backend.checkpoints[checkpoint]["checkpoint_identity_sha256"]
    output = backend.serve(list(reference["input_ids"]), checkpoint, capture=False,
                           turn_index=reference["turn"], repeat=0)
    require(isinstance(output, dict), "Missing benchmark output")
    input_match = output.get("input_ids") == reference["input_ids"] and output.get("input_ids_sha256") == digest(reference["input_ids"])
    output_match = output.get("output_ids") == reference["output_ids"]
    identity_match = output.get("checkpoint_identity_sha256") == expected
    row = {**output, "episode_id": reference["episode_id"], "turn": reference["turn"],
           "source": reference["source"], "synthetic": reference["synthetic"], "arm": arm,
           "checkpoint_id": checkpoint, "repeat": 0, "canonical": False, "training_data": False,
           "capture": False, "reference_input_ids_sha256": digest(reference["input_ids"]),
           "reference_output_ids_sha256": digest(reference["output_ids"]),
           "reference_context_sha256": reference.get("context_sha256"),
           "input_match": input_match, "canonical_output_match": output_match,
           "checkpoint_identity_match": identity_match,
           "same_generation_valid": input_match and output_match and identity_match,
           "output_tokens": len(output.get("output_ids", [])),
           "context_tokens": len(reference["input_ids"]),
           "latency_scope": "Single-shot capture-disabled diagnostic; no deployment-speedup claim."}
    destination.append(row)
    emit("score", row)  # Preserve mismatches and malformed receipts before validation.
    validate_measurement(output)
    if not row["same_generation_valid"]:
        emit("benchmark_mismatch", row)
    return row


def validate_update(update, event, support, arm):
    checkpoint = update.get("checkpoint", {})
    require(checkpoint.get("checkpoint_id") == f"D{event}" and update.get("optimizer_steps") == 2 * event,
            "Update checkpoint or cumulative optimizer steps differ")
    account = update.get("accounting", {})
    require(account.get("optimizer_steps_this_event") == 2 and account.get("optimizer_steps_total") == 2 * event and
            account.get("response_presentations_this_event") == 2 and account.get("support_turns") == [event],
            "Each event must use two passes/two steps on only the newest response")
    recipe = account.get("recipe", {})
    require(recipe.get("responses_per_step") == 1 and recipe.get("passes") == 2 and
            recipe.get("learning_rate") == LEARNING_RATES[arm] and recipe.get("loss") == "forward_KL",
            "Qualified objective or arm learning rate changed")
    audits = account.get("response_audits", [])
    require(len(audits) == 1, "Unexpected training responses")
    start, count = len(support["input_ids"]), len(support["output_ids"])
    require(audits[0].get("response_start") == start and audits[0].get("response_end_exclusive") == start + count and
            audits[0].get("loss_mask_sha256") == digest([0] * start + [1] * count + [0] * (BLOCK_SIZE - 1)),
            "Only newest response may receive loss")
    presentations = [p for b in account.get("batches", []) for p in b.get("presentations", [])]
    require(len(presentations) == 2, "Exactly two response presentations required")
    for p in presentations:
        positions = p.get("unique_supervised_position_ids")
        require(isinstance(positions, list) and positions and len(set(positions)) == len(positions) and
                all(start <= pos < start + count for pos in positions), "Empty or out-of-response supervision")
        require(p.get("record_id") == support["record_id"] and p.get("turn") == event and
                p.get("unique_supervised_positions") == len(positions) and p.get("zero_supervision") is False and
                type(p.get("eligible_positions")) is int and p["eligible_positions"] >= len(positions),
                "Anchor/exposure metadata does not match newest canonical response")
    if arm in ("G", "N"):
        require(account.get("agreement_arm") == arm and account.get("filter_enabled") is True,
                "Filtered/scaled arm metadata differs")
        groups = account.get("fixed_groups", {})
        eligible = list(range(start + 1, start + count))
        split = len(eligible) // 2
        require(groups.get("a") == eligible[:split] and groups.get("b") == eligible[split:] and
                groups.get("a") and groups.get("b"), "Fixed chronological groups missing or changed")
        for index, (batch, p) in enumerate(zip(account["batches"], presentations)):
            group = p.get("grouping", {})
            require(group.get("identity_sha256") == groups.get("identity_sha256") and
                    group.get("group_a_positions") == groups["a"] and group.get("group_b_positions") == groups["b"] and
                    group.get("shared_denominator") == p.get("normalization_denominator") and
                    group.get("group_a_presentations", 0) > 0 and group.get("group_b_presentations", 0) > 0,
                    "Group overlap assignments, full denominator or nonempty supervision changed")
            stats = batch.get("agreement", {})
            require(stats.get("mode") == arm and stats.get("optimizer_step_calls") == 1 and
                    stats.get("optimizer_steps_before") == 2 * (event - 1) + index and
                    stats.get("optimizer_steps_after") == 2 * (event - 1) + index + 1,
                    "Agreement diagnostics do not certify exactly one optimizer-state advancement")
            for key in ("proposed_step_norm", "hypothetical_filtered_step_norm", "applied_step_norm", "mask_keep_fraction"):
                value = stats.get(key)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        "Missing/nonfinite agreement diagnostic: " + key)
    return True


def implementation_checks(result):
    arms = result.get("arms", {})
    problems = []
    for arm in LEARNING_ARMS:
        value = arms.get(arm, {})
        if value.get("status") != "complete" or [r.get("turn") for r in value.get("canonical", [])] != [1, 2, 3]:
            problems.append(arm + ":incomplete_canonical_trajectory")
        if [r.get("event") for r in value.get("updates", [])] != [1, 2] or not all(
                u.get("mask_and_schedule_check") is True for u in value.get("updates", [])):
            problems.append(arm + ":update_schedule_or_mask")
        if value.get("cleanup", {}).get("status") != "passed":
            problems.append(arm + ":cleanup")
    scores = result.get("scores", [])
    for turn in (2, 3):
        for arm in ARMS:
            rows = [r for r in scores if r.get("turn") == turn and r.get("arm") == arm]
            if len(rows) != 1 or rows[0].get("same_generation_valid") is not True:
                problems.append(f"{arm}:turn{turn}:missing_duplicate_or_invalid_score")
    return {"status": "passed" if not problems and result.get("status") == "complete" else "failed",
            "problems": problems, "canonical_generations": sum(len(a.get("canonical", [])) for a in arms.values()),
            "scored_replays": len(scores), "expected_scored_replays": 10,
            "update_events": sum(len(a.get("updates", [])) for a in arms.values()),
            "scope": "Protocol/output gate; includes no positive-performance criterion."}


def run_stream(stream, references, backend_factory, *, emit=lambda kind, data: None,
               snapshot=lambda result: None, qualification=False):
    rows = validate_stream(stream)[:3]
    _references(stream, references)
    result = {"schema_version": 1, "mode": "osd_agreement_filter", "status": "running",
              "episode_id": stream["stream_id"], "source": stream["source"], "synthetic": stream["synthetic"],
              "seed": 17, "planned_requests": 3, "arms": {}, "scores": [], "updates": [],
              "qualification": qualification,
              "references": copy.deepcopy(references), "stream_binding": copy.deepcopy(stream.get("binding"))}
    started = time.monotonic()
    for arm in LEARNING_ARMS:
        value = {"arm": arm, "episode_id": stream["stream_id"], "status": "running", "canonical": [],
                 "support": [], "updates": [], "scores": [], "retirement": [], "qualification": qualification}
        result["arms"][arm] = value
        backend = None
        messages = []
        try:
            backend = backend_factory(arm, stream["stream_id"])
            value["startup"] = backend.start_episode(stream["stream_id"])
            for row in rows:
                turn = row["turn"]
                current = "F" if turn == 1 else f"D{turn - 1}"
                messages.append({"role": "user", "content": row["prompt"]})
                ids = backend.encode(copy.deepcopy(messages))
                valid_ids(ids)
                expected = backend.checkpoints[current]["checkpoint_identity_sha256"]
                output = backend.serve(list(ids), current, capture=False, turn_index=turn, repeat=None)
                item = {**output, "episode_id": stream["stream_id"], "source": stream["source"],
                        "synthetic": stream["synthetic"], "arm": arm, "turn": turn, "checkpoint_id": current,
                        "canonical": True, "capture": False, "training_data": turn < 3,
                        "question_id": row.get("request_id", f'{stream["stream_id"]}:t{turn:02d}'),
                        "source_user_text": row["prompt"], "source_user_text_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                        "source_message_ids": copy.deepcopy(row.get("source_message_ids")),
                        "context_tokens": len(ids), "output_tokens": len(output.get("output_ids", [])),
                        "cap_hit": len(output.get("output_ids", [])) == OUTPUT_CAP,
                        "reference_input_ids_sha256": digest(ids)}
                value["canonical"].append(item)
                emit("canonical", item)
                validate_measurement(output)
                require(output.get("input_ids") == ids and output.get("input_ids_sha256") == digest(ids) and
                        output.get("checkpoint_identity_sha256") == expected, "Own canonical input or weights changed")
                if turn > 1:
                    item["matches_common_input"] = ids == references[turn]["input_ids"]
                    item["matches_common_output"] = output["output_ids"] == references[turn]["output_ids"]
                messages.append({"role": "assistant", "content": output["text"], "output_ids": list(output["output_ids"])})
                if turn in (2, 3):
                    if arm == "H":
                        value["scores"].append(score(backend, references[turn], "F", "F", emit=emit, destination=result["scores"]))
                    value["scores"].append(score(backend, references[turn], current, arm, emit=emit, destination=result["scores"]))
                snapshot(result)
                if turn < 3:
                    support_started = time.monotonic()
                    support = backend.prepare_support(copy.deepcopy(item))
                    support["observed_support_prepare_s"] = time.monotonic() - support_started
                    require(support.get("input_ids") == item["input_ids"] and support.get("output_ids") == item["output_ids"] and
                            support.get("turn") == turn, "Training support must be this arm's actual newest response")
                    value["support"].append(support)
                    emit("support", {"arm": arm, **support})
                    update_started = time.monotonic()
                    update = backend.train_event(turn, [support])
                    update.update(arm=arm, event=turn, observed_update_call_s=time.monotonic() - update_started)
                    value["updates"].append(update)
                    result["updates"].append(update)
                    emit("update", update)
                    update["mask_and_schedule_check"] = validate_update(update, turn, support, arm)
                    if qualification and arm == "G":
                        require(update.get("qualification_checks", {}).get("status") == "passed",
                                "Native agreement qualification receipt missing or failed")
                    snapshot(result)
                    if hasattr(backend, "retire"):
                        receipt = backend.retire(keep_ids={"F", current, f"D{turn}"}, protected_event_ids=set())
                        value["retirement"].append(receipt)
                        emit("retirement", {"arm": arm, **receipt})
                snapshot(result)
            value["status"] = "complete"
            require(all(s.get("same_generation_valid") is True for s in value["scores"]),
                    "Invalid common-input benchmark: preserve recovery state and stop")
        except Exception as error:
            value.update(status="failed", error=repr(error))
            result.update(status="failed", error=repr(error), failed_arm=arm)
            emit("failure", {"arm": arm, "error": repr(error)})
        finally:
            if backend is not None:
                try:
                    value["cleanup"] = backend.finish_episode()
                    require(value["cleanup"].get("status") == "passed", "Worker cleanup failed")
                except Exception as error:
                    value.update(status="failed", cleanup_error=repr(error))
                    result.update(status="failed", cleanup_error=repr(error), failed_arm=arm)
            if backend is not None and value["status"] == "complete" and result["status"] != "failed":
                # Native finalization persists/validates the compact arm result
                # before retiring any now-unneeded rolling checkpoint payload.
                if hasattr(backend, "finalize_retention"):
                    try:
                        receipt = backend.finalize_retention(result=copy.deepcopy(value))
                        value["final_retention"] = receipt
                        emit("final_retention", {"arm": arm, **receipt})
                    except Exception as error:
                        value.update(status="failed", retention_error=repr(error))
                        result.update(status="failed", retention_error=repr(error), failed_arm=arm)
            snapshot(result)
        if result["status"] == "failed":
            break
    else:
        result["status"] = "complete"
    result["actual_wall_s"] = time.monotonic() - started
    result["implementation_checks"] = implementation_checks(result)
    snapshot(result)
    return result
