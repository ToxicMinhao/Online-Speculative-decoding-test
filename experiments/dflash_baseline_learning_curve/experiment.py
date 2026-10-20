"""Bounded independent-request learning curve; no automatic GPU launch.

The first complete group is a correctness gate, never a positive-gain gate.
Raw measurements precede validation. Every checkpoint comparison uses the same
held-out token IDs, with the training process stopped before warmup or timing.
"""
from __future__ import annotations

import copy
import hashlib
import math
import json
import struct
import time

CHECKPOINTS = ("F", "S2", "S8")
STEPS = {"F": 0, "S2": 2, "S8": 8}
OUTPUT_CAP, BLOCK_SIZE, CONTEXT_LIMIT, INPUT_LIMIT = 1024, 10, 32768, 31735
STOP_IDS = (128001, 128008, 128009)
METRIC_DEFINITIONS = {
    "accepted_draft_tokens": "Committed accepted draft tokens; excludes correction and bonus tokens.",
    "accepted_prefix_length": "Correct proposed prefix before EOS/cap clipping; excludes correction/bonus.",
    "verification_committed_tokens": "Tokens emitted by verification, including correction/bonus; excludes initial prefill token.",
    "committed_progress_per_call": "Verification-committed tokens divided by verification calls; includes correction/bonus.",
    "emitted_tokens_per_verification_call": "All response output tokens divided by verification calls, including initial prefill token.",
    "request_s": "Complete serving request with prefill/drafting/verification, capture disabled; excludes training, warmup and checkpoint loading.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def preserve_invalid_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value), "ieee754_binary64_hex": struct.pack(">d", value).hex()}
    if isinstance(value, dict):
        return {key: preserve_invalid_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [preserve_invalid_numbers(item) for item in value]
    return value


def _ids(value, maximum):
    return isinstance(value, list) and 0 < len(value) <= maximum and all(
        type(token) is int and 0 <= token < 128256 for token in value)


def validate_group(group):
    require(isinstance(group.get("group_id", group.get("id")), str) and group.get("domain"), "Missing group/domain identity")
    rows = []
    for role in ("feedback", "evaluation"):
        selected = group.get(role)
        require(isinstance(selected, list) and len(selected) == 4, "Each role requires four independent requests")
        for row in selected:
            require(isinstance(row.get("request_id"), str) and row["request_id"], "Missing request identity")
            require(_ids(row.get("input_ids"), INPUT_LIMIT), "Invalid full prompt IDs; truncation is forbidden")
            require(len(row["input_ids"]) + OUTPUT_CAP + BLOCK_SIZE - 1 <= CONTEXT_LIMIT, "Prompt exceeds context budget")
            rows.append(row)
    require(len({r["request_id"] for r in rows}) == 8 and len({digest(r["input_ids"]) for r in rows}) == 8,
            "Feedback/evaluation prompts must be distinct")
    return rows


def validate_groups(groups):
    require(len(groups) == 6, "Exactly six preselected groups are required")
    rows = [r for group in groups for r in validate_group(group)]
    require(len({g.get("group_id", g.get("id")) for g in groups}) == 6, "Duplicate group")
    domains = {g["domain"] for g in groups}
    require(len(domains) == 3 and all(sum(g["domain"] == domain for g in groups) == 2 for domain in domains),
            "Two groups per each of three domains required")
    require(len({r["request_id"] for r in rows}) == 48 and len({digest(r["input_ids"]) for r in rows}) == 48,
            "All 48 requests must be distinct across groups and roles")
    return groups


def checkpoint_order(group_index, request_index):
    offset = (group_index + request_index) % len(CHECKPOINTS)
    return list(CHECKPOINTS[offset:] + CHECKPOINTS[:offset])


def validate_quiescence(receipt):
    require(isinstance(receipt, dict) and receipt.get("status") == "passed" and
            receipt.get("training_quiescent") is True and receipt.get("trainer_processes") == [],
            "Training must be stopped, with no background trainer during evaluation")
    return receipt


def validate_measurement(row):
    require(row.get("status") == "passed", "Native measurement did not pass")
    require(_ids(row.get("input_ids"), INPUT_LIMIT) and _ids(row.get("output_ids"), OUTPUT_CAP), "Invalid exact token IDs")
    require(row.get("input_ids_sha256") == digest(row["input_ids"]), "Input-token identity differs")
    n = len(row["output_ids"])
    require(row.get("output_tokens") == n and row.get("cap_hit") is (n == OUTPUT_CAP), "Output length/cap accounting differs")
    require(isinstance(row.get("finish_reason"), (str, dict)) and bool(row["finish_reason"]), "Missing EOS/stop reason")
    if "eos_hit" in row:
        require(row["eos_hit"] is (row["output_ids"][-1] in STOP_IDS), "EOS flag differs from exact tokens")
    require(type(row.get("request_s")) in (int, float) and math.isfinite(row["request_s"]) and row["request_s"] > 0,
            "Request latency must be finite and positive")
    require(row.get("capture") is False and row.get("feedback") is None and row.get("capture_costs") is None,
            "Serving measurements must disable training-data capture")
    for name in ("verification_calls", "accepted_draft_tokens", "verification_committed_tokens", "initial_output_tokens"):
        require(type(row.get(name)) is int and row[name] >= 0, "Invalid native counter: " + name)
    require(row["accepted_draft_tokens"] <= row["verification_committed_tokens"] <= n, "Accepted/committed counts exceed output")
    require(row["initial_output_tokens"] + row["verification_committed_tokens"] == n, "Initial/verification progress does not cover output")
    rounds = row.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == row["verification_calls"], "Round count differs from native verification calls")
    offset = row["initial_output_tokens"]
    accepted = committed = 0
    for index, event in enumerate(rounds):
        require(event.get("round_index") == index and event.get("output_offset") == offset, "Noncontiguous verification offsets")
        for name in ("accepted_prefix_length", "accepted_draft_tokens", "committed_tokens", "proposed_tokens"):
            require(type(event.get(name)) is int and event[name] >= 0, "Invalid round counter: " + name)
        require(event["accepted_draft_tokens"] <= event["accepted_prefix_length"] <= event["proposed_tokens"] <= BLOCK_SIZE - 1,
                "Round acceptance/proposal counts differ")
        require(event["accepted_draft_tokens"] <= event["committed_tokens"] <= BLOCK_SIZE and event["committed_tokens"] > 0,
                "Round committed progress differs")
        require(type(event.get("stop_clipped")) is bool, "Missing round stop-clipping flag")
        rejection = event.get("first_rejection_offset")
        require((rejection is None and event["accepted_prefix_length"] == event["proposed_tokens"]) or
                (type(rejection) is int and 0 <= rejection < event["proposed_tokens"] and rejection == event["accepted_prefix_length"]),
                "Round first rejection disagrees with accepted prefix")
        require(event.get("committed_token_ids") == row["output_ids"][offset:offset + event["committed_tokens"]],
                "Round committed tokens do not match saved output segment")
        offset += event["committed_tokens"]
        accepted += event["accepted_draft_tokens"]
        committed += event["committed_tokens"]
    require(offset == n and accepted == row["accepted_draft_tokens"] and committed == row["verification_committed_tokens"],
            "Per-round counters do not reconcile with request")
    for key, numerator in (("committed_progress_per_call", committed), ("emitted_tokens_per_verification_call", n)):
        if key in row:
            expected = numerator / len(rounds) if rounds else None
            require((expected is None and row[key] is None) or
                    (type(row[key]) in (int, float) and math.isclose(row[key], expected, rel_tol=1e-12, abs_tol=1e-12)),
                    "Derived progress metric differs: " + key)
    require(isinstance(row.get("checkpoint_identity_sha256"), str) and len(row["checkpoint_identity_sha256"]) == 64,
            "Missing checkpoint identity")
    return True


def measurement_validation(row):
    try:
        validate_measurement(row)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return {"status": "failed", "errors": [str(error)]}
    return {"status": "passed", "errors": []}


def _checkpoint_identity(binding):
    require(isinstance(binding, dict) and isinstance(binding.get("checkpoint_identity_sha256"), str) and
            len(binding["checkpoint_identity_sha256"]) == 64, "Missing bound checkpoint")
    return binding["checkpoint_identity_sha256"]


def _all_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return 'nonfinite_float' not in value and all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def validate_training(training, support):
    require(training.get("status") == "passed", "Training pool did not complete")
    require(_all_finite(training), "Nonfinite training/source-loss/cost evidence")
    checkpoints = training.get("checkpoints", {})
    require(set(checkpoints) == set(CHECKPOINTS), "Continuous step-0/2/8 checkpoints are required")
    for binding in checkpoints.values():
        _checkpoint_identity(binding)
    account = training.get("accounting", {})
    require(account.get("optimizer_steps") == 8 and account.get("response_presentations") == 16,
            "Training must be eight cumulative steps and sixteen presentations")
    recipe = account.get("recipe", {})
    require(recipe.get("passes") == 4 and recipe.get("responses_per_step") == 2 and recipe.get("learning_rate") == 3e-4 and
            recipe.get("loss") == "forward_KL", "Qualified four-pass accumulated training recipe differs")
    batches = account.get("batches", [])
    require(len(batches) == 8, "Missing optimizer-step receipts")
    expected_ids = [row["record_id"] for row in support]
    by_id = {row["record_id"]: row for row in support}
    for index, batch in enumerate(batches):
        require(batch.get("step", batch.get("optimizer_steps_after")) == index + 1 and batch.get("pass_index") == index // 2,
                "Steps/passes are not one continuous trajectory")
        presentations = batch.get("presentations", [])
        expected = expected_ids[2 * (index % 2):2 * (index % 2) + 2]
        require([p.get("record_id") for p in presentations] == expected, "Training feedback order or pool changed")
        for p in presentations:
            record = by_id[p["record_id"]]
            start, end = len(record["input_ids"]), len(record["input_ids"]) + len(record["output_ids"])
            require(p.get("response_start") == start and p.get("response_end_exclusive") == end, "Response-loss boundary differs")
            require(p.get("loss_mask_sha256") == digest([0] * start + [1] * (end - start) + [0] * (BLOCK_SIZE - 1)),
                    "Training loss mask does not match its independent response")
            positions = p.get("unique_supervised_position_ids")
            require(isinstance(positions, list) and len(positions) == len(set(positions)) and
                    all(type(pos) is int and start < pos < end for pos in positions), "Training supervises prompt/foreign positions")
            require(p.get("unique_supervised_positions") == len(positions) and type(p.get("eligible_positions")) is int and
                    p["eligible_positions"] >= len(positions), "Actual supervised exposure is inconsistent")
            require(bool(positions) or (p.get("zero_supervision") is True and p["eligible_positions"] == 0),
                    "Empty exposure requires the qualified explicit zero-supervision receipt")
            require(isinstance(p.get("anchor_positions"), list) and p["anchor_positions"] and
                    type(p.get("normalization_denominator")) in (int, float) and
                    math.isfinite(p["normalization_denominator"]) and p["normalization_denominator"] > 0,
                    "Missing sampled-anchor/normalization evidence")
    losses = training.get("source_losses", [])
    require(len(losses) == 3 and {row.get("checkpoint_id") for row in losses} == set(CHECKPOINTS),
            "Fixed-anchor source losses required at all three checkpoints")
    require(len({row.get("fixed_anchor_identity_sha256") for row in losses}) == 1 and
            all(isinstance(row.get("fixed_anchor_identity_sha256"), str) and len(row["fixed_anchor_identity_sha256"]) == 64
                and row.get("training_rng_preserved") is True and type(row.get("loss")) in (int, float) and
                math.isfinite(row["loss"]) for row in losses), "Source diagnostics changed anchors/RNG or have invalid loss")
    return True


def implementation_checks(result):
    good = (result.get("status") == "complete" and len(result.get("feedback", [])) == 4 and
            len(result.get("scores", [])) == 12 and len(result.get("warmups", [])) == 3 and
            result.get("training_valid") is True and
            all(row.get("same_generation_valid") is True for row in result.get("scores", [])) and
            result.get("cleanup", {}).get("status") == "passed")
    return {"status": "passed" if good else "failed", "feedback_generations": len(result.get("feedback", [])),
            "evaluation_generations": len(result.get("scores", [])), "warmup_generations": len(result.get("warmups", [])),
            "scope": "Correctness only; null or harmful performance is not a gate."}


def run_group(group, backend, *, group_index=0, emit=lambda kind, value: None, snapshot=lambda value: None):
    validate_group(group)
    result = dict(schema_version=1, group_id=group.get("group_id", group.get("id")), domain=group["domain"],
                  status="running", group_index=group_index, group_binding=digest(group),
                  feedback=[], support=[], scores=[], warmups=[], publications=[], quiescence_checks=[],
                  training=None, training_valid=False, evaluation_requests=copy.deepcopy(group["evaluation"]),
                  metric_definitions=METRIC_DEFINITIONS, latency_scope="preliminary one-shot serving-only timing")
    started = time.monotonic()
    def persist(kind, value):
        emit(kind, preserve_invalid_numbers(value)); snapshot(preserve_invalid_numbers(result))
    def quiescent():
        receipt = backend.assert_training_quiescent()
        result["quiescence_checks"].append(receipt); persist("training_quiescence", receipt)
        return validate_quiescence(receipt)
    def serve(request, checkpoint, role, destination):
        output = backend.serve(list(request["input_ids"]), checkpoint, role=role,
                               request_id=request["request_id"], capture=False)
        require(isinstance(output, dict), "Native serving produced no record")
        row = {**output, "group_id": result["group_id"], "domain": result["domain"],
               "request_id": request["request_id"], "checkpoint_id": checkpoint, "step": STEPS[checkpoint],
               "role": role, "reference_input_ids_sha256": digest(request["input_ids"]),
               "training_data": role == "feedback", "timing_repeats": 1,
               "eos_hit": bool(output.get("output_ids")) and output["output_ids"][-1] in STOP_IDS}
        if role == "evaluation":
            row["training_quiescence"] = copy.deepcopy(result["quiescence_checks"][-1])
        destination.append(row)
        persist("raw_measurement", row)
        validation = measurement_validation(row)
        row["measurement_validation"] = validation
        row["input_match"] = row.get("input_ids") == request["input_ids"]
        row["checkpoint_identity_match"] = row.get("checkpoint_identity_sha256") == _checkpoint_identity(result["checkpoints"][checkpoint])
        persist("measurement_validation", row)
        require(validation["status"] == "passed", "; ".join(validation["errors"]))
        require(row["input_match"] and row["checkpoint_identity_match"], "Exact prompt/checkpoint identity mismatch")
        return row
    try:
        result["startup"] = backend.start_group(group)
        require(result["startup"].get("status") == "passed" and result["startup"].get("fresh_optimizer") is True and
                result["startup"].get("released_reset") is True and result["startup"].get("seed") == 17,
                "Group startup/reset/fresh optimizer contract did not pass")
        result["checkpoints"] = copy.deepcopy(result["startup"].get("checkpoints", getattr(backend, "checkpoints", {})))
        _checkpoint_identity(result["checkpoints"].get("F"))
        for request in group["feedback"]:
            serve(request, "F", "feedback", result["feedback"])
        support_started = time.monotonic()
        result["support"] = backend.prepare_support(copy.deepcopy(result["feedback"]))
        result["support_prepare_wall_s"] = time.monotonic() - support_started
        require(len(result["support"]) == 4, "Feedback support must contain exactly four responses")
        for support, canonical in zip(result["support"], result["feedback"]):
            require(support.get("input_ids") == canonical["input_ids"] and support.get("output_ids") == canonical["output_ids"] and
                    support.get("record_id"), "Support differs from frozen-generated exact tokens")
        require(len({r["record_id"] for r in result["support"]}) == 4, "Duplicate support record")
        persist("support", result["support"])
        training_started = time.monotonic()
        result["training_started"] = True
        result["training"] = backend.train_pool(copy.deepcopy(result["support"]))
        result["training_call_wall_s"] = time.monotonic() - training_started
        persist("training", result["training"])
        result["training_valid"] = validate_training(result["training"], result["support"])
        require(_checkpoint_identity(result["training"]["checkpoints"]["F"]) ==
                _checkpoint_identity(result["checkpoints"]["F"]), "Released reference checkpoint changed during training")
        result["checkpoints"] = copy.deepcopy(result["training"]["checkpoints"])
        result["trainer_stop"] = backend.stop_trainer()
        persist("trainer_stopped", result["trainer_stop"])
        validate_quiescence(result["trainer_stop"])
        for checkpoint in CHECKPOINTS:
            quiescent()
            publication = {"checkpoint_id": checkpoint, "purpose": "warmup", **backend.publish(checkpoint)}
            result["publications"].append(publication); persist("publication", publication)
            require(publication.get("status") == "passed", "Warmup checkpoint loading failed")
            warmup = {"checkpoint_id": checkpoint, **backend.warmup(checkpoint)}
            result["warmups"].append(warmup); persist("warmup", warmup)
            require(warmup.get("status") == "passed", "Warmup failed")
        for index, request in enumerate(group["evaluation"]):
            order = checkpoint_order(group_index, index)
            persist("evaluation_order", {"request_id": request["request_id"], "order": order})
            rows = []
            for checkpoint in order:
                quiescent()
                publication = {"checkpoint_id": checkpoint, "purpose": "evaluation",
                               "request_id": request["request_id"], **backend.publish(checkpoint)}
                result["publications"].append(publication); persist("publication", publication)
                require(publication.get("status") == "passed", "Evaluation checkpoint loading failed")
                quiescent()
                row = serve(request, checkpoint, "evaluation", result["scores"])
                row["order"] = list(order); row["order_index"] = len(rows); rows.append(row)
            frozen = next(row for row in rows if row["checkpoint_id"] == "F")
            for row in rows:
                row["canonical_output_match"] = row["output_ids"] == frozen["output_ids"]
                row["reference_output_ids_sha256"] = digest(frozen["output_ids"])
                row["same_generation_valid"] = row["canonical_output_match"] and row["input_match"] and row["checkpoint_identity_match"]
                persist("paired_validation", row)
            require(all(row["same_generation_valid"] for row in rows), "Adapted output differs from frozen; affected group stopped")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error)); persist("failure", {"error": repr(error)})
    finally:
        try:
            result["cleanup"] = backend.finish_group()
            require(result["cleanup"].get("status") == "passed", "Group cleanup failed")
        except Exception as error:
            result.update(status="failed", cleanup_error=repr(error))
        result["actual_wall_s"] = time.monotonic() - started
        result["implementation_checks"] = implementation_checks(result)
        snapshot(preserve_invalid_numbers(result))
    return preserve_invalid_numbers(result)


def run_campaign(groups, backend_factory, *, emit=lambda kind, value: None, snapshot=lambda value: None):
    validate_groups(groups)
    result = dict(schema_version=1, status="running", groups=[], expected_groups=6,
                  planned_core_generations=96, planned_optimizer_steps=48,
                  planned_training_response_presentations=96, planned_explicit_warmups=18,
                  main_scope="baseline learning curve; agreement-filter campaign remains paused")
    started = time.monotonic()
    for index, group in enumerate(groups):
        current = run_group(group, backend_factory(group), group_index=index, emit=emit,
                            snapshot=lambda value: snapshot({**result, "active_group": value}))
        result["groups"].append(current)
        if index == 0:
            result["first_group_qualification"] = current["implementation_checks"]
        snapshot(result)
        if current["implementation_checks"]["status"] != "passed":
            result["status"] = "stopped_after_correctness_failure"
            break
    else:
        result["status"] = "complete"
    result["actual_wall_s"] = time.monotonic() - started
    result["unexecuted_groups"] = [g.get("group_id", g.get("id")) for g in groups[len(result["groups"]):]]
    snapshot(result)
    return result
