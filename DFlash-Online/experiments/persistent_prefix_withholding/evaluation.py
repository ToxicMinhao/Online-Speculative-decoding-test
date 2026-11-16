"""Fixed event-eight serving: 36 activations, 72 warmups, 1,152 scored calls.

This module owns ordering and validity. It uses the existing native backend and
does not receive a trainer, feedback buffer, or a mutable canonical reference.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.fresh_confirmation.evaluate import append_json, safe_scalars
from experiments.fresh_confirmation.protocol import (
    measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from experiments.o_half_matched_lr.archive import validate_record
from experiments.osd_persistent.run import score_record

METHODS = ("N", "E", "S")
PANEL_SIZE = 128
BLOCK_SIZE = 32
REPEATS = 3
SCOPE = "DEVELOPMENT_FEASIBILITY_REUSED_DATA"
SESSION = "persistent_prefix_withholding_event8"


def arm_order(repeat, block):
    base = METHODS if block % 2 == 0 else ("N", "S", "E")
    shift = (repeat + block // 2) % 3
    return base[shift:] + base[:shift]


def evaluation_schedule(requests, within_block_permutations):
    """Use the sealed permutations; never choose a new order during evaluation."""
    if (len(requests) != PANEL_SIZE or len(within_block_permutations) != REPEATS
            or [r["panel_ordinal"] for r in requests] != list(range(PANEL_SIZE))):
        raise ValueError("Require the sealed 128-request panel and three repeats")
    schedule = []
    for repeat, blocks in enumerate(within_block_permutations):
        if len(blocks) != PANEL_SIZE // BLOCK_SIZE:
            raise ValueError("Require four sealed request blocks per repeat")
        for block, ordinals in enumerate(blocks):
            if sorted(ordinals) != list(range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)):
                raise ValueError("A request permutation changed its fixed block")
            for position, method in enumerate(arm_order(repeat, block)):
                schedule.append(dict(activation=len(schedule), repeat=repeat,
                    block_ordinal=block, position=position, method=method,
                    request_ordinals=list(ordinals)))
    return schedule


def schedule_balance(schedule):
    orders = defaultdict(dict)
    for entry in schedule:
        orders[entry["repeat"], entry["block_ordinal"]][entry["position"]] = entry["method"]
    counts = Counter(tuple(row[i] for i in range(3)) for row in orders.values())
    return dict(logical_activations=len(schedule), warmups=2 * len(schedule),
        scored_generations=sum(len(row["request_ordinals"]) for row in schedule),
        orders=[dict(order=list(order), count=count) for order, count in sorted(counts.items())],
        positions={method: [sum(row[p] == method for row in orders.values()) for p in range(3)]
                   for method in METHODS})


def validate_references(requests, canonicals):
    """Require a provenance-sealed reference, not agreement among candidates."""
    if len({r["request_id"] for r in requests}) != len(requests) or set(canonicals) != {r["request_id"] for r in requests}:
        raise ValueError("References must cover each distinct input exactly")
    for request in requests:
        reference = canonicals[request["request_id"]]
        tokens = reference["response_ids"]
        if (not request["prompt_ids"] or request["max_new_tokens"] != 2048
                or len(request["prompt_ids"]) + 2048 > 4096
                or not request["stop_token_ids"]
                or reference.get("prompt_ids") != request["prompt_ids"]
                or reference.get("verified") is not True
                or not reference.get("binding_digest")
                or reference.get("reference_kind") not in ("target_only_AR", "native_greedy_target_verifier")
                or not reference.get("source_record_path")
                or not tokens or len(tokens) > 2048
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Canonical identity, provenance or serving geometry is incomplete")
        finish = reference["finish_reason"]
        if (finish not in ("eos", "length")
                or any(token in request["stop_token_ids"] for token in tokens[:-1])
                or (finish == "eos" and tokens[-1] not in request["stop_token_ids"])
                or (finish == "length" and (len(tokens) != 2048 or tokens[-1] in request["stop_token_ids"]))):
            raise ValueError("Canonical termination violates the fixed EOS/output cap")


def label_for(slot):
    if slot["phase"] == "warmup":
        return f"warmup_a{slot['activation']:02d}_{slot['panel_ordinal']}_{slot['method']}"
    return f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"


def measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid):
    """Validate raw native decisions before admitting any returned measurement."""
    label = label_for(slot)
    try:
        errors = validate_record(record, request, checkpoint, canonical, label=label, session=SESSION)
        row = score_record(record, request_id=request["request_id"], repeat=slot["repeat"], method=slot["method"])
        trace = record["metadata"]["trace"]
        native_accepted = sum(block["native_accepted_draft_tokens"] for block in trace)
        row.update(finish_reason=record["finish_reason"],
            cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"],
            native_accepted_draft_tokens=native_accepted,
            native_acceptance_histogram={str(k): v for k, v in sorted(Counter(block["native_accepted_draft_tokens"] for block in trace).items())})
        errors.extend(measurement_errors(row, request, canonical))
        if native_accepted != row["committed_progress"] - row["verification_calls"]:
            errors.append("native_acceptance_accounting")
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        row = dict(request_id=request["request_id"], checkpoint_id=str(checkpoint))
        errors = ["malformed_native_record: " + repr(error)]
    row = safe_scalars(row)
    row.update(**slot, label=label, source_id=request["source_id"],
        original_index=request["original_index"], physical_gpu_uuid=physical_gpu_uuid,
        scope=SCOPE)
    return row, sorted(set(errors))


def boundary_errors(check, physical_gpu_uuid):
    """The caller inspects both cards and host activity outside request timing."""
    errors = []
    if check.get("status") != "passed" or check.get("physical_gpu_uuid") != physical_gpu_uuid:
        errors.append("boundary_or_device_identity")
    for key in ("both_cards_checked", "other_gpu_idle", "host_quiet", "storage_admitted", "time_admitted"):
        if check.get(key) is not True:
            errors.append(key)
    return errors


def evaluate_final(backend, requests, checkpoints, root, *, canonicals,
                   warmup_requests, warmup_canonicals, schedule,
                   within_block_permutations, physical_gpu_uuid, boundary_check,
                   execution_commit):
    """Execute once, preserving invalid/unknown work before stopping on failure."""
    planned = evaluation_schedule(requests, within_block_permutations)
    if schedule != planned:
        raise ValueError("The sealed activation or request order changed")
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    if set(checkpoints) != set(METHODS) or len(set(checkpoints.values())) != 3:
        raise ValueError("Require three distinct N8/E8/S8 endpoints")
    if (len(warmup_requests) != 2 or [r["panel_ordinal"] for r in warmup_requests] != [0, 1]
            or {r["source_id"] for r in requests} & {r["source_id"] for r in warmup_requests}):
        raise ValueError("Require two fixed old-development warmups outside the final panel")
    validate_references(requests, canonicals)
    validate_references(warmup_requests, warmup_canonicals)
    if not physical_gpu_uuid or boundary_check is None:
        raise ValueError("Fixed GPU identity and quiet-boundary admission are required")
    if len(execution_commit) != 40 or any(c not in "0123456789abcdef" for c in execution_commit):
        raise ValueError("Require the actual clean execution commit")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "PROTOCOL.json", dict(schedule=planned, checkpoints=checkpoints,
        physical_gpu_uuid=physical_gpu_uuid, scope=SCOPE, capture=False,
        within_block_permutations=within_block_permutations, event=8,
        execution_commit=execution_commit,
        planned_scored=1152, planned_warmups=72, canonical_source="sealed_authenticated_references"))
    write_json(root / "REQUESTS.json", requests)
    write_json(root / "CANONICAL.json", canonicals)
    write_json(root / "WARMUPS.json", dict(requests=warmup_requests, canonicals=warmup_canonicals))
    rows, grouped, checked = [], defaultdict(list), set()
    progress = dict(status="running", planned_generations=1224, attempted_generations=0,
        returned_generations=0, publications=0, logical_activations=0,
        activation_s=0.0, weight_validation_s=0.0, optimizer_steps=0,
        feedback_collections=0, teacher_preparations=0, automatic_retry=False)
    pending = None
    started = time.perf_counter()

    def persist_status():
        progress.update(completed_valid_generations=sum(r["valid_measurement"] for r in rows),
            invalid_generations=sum(not r["valid_measurement"] for r in rows),
            scored_generations=sum(r["phase"] == "scored" for r in rows),
            warmup_generations=sum(r["phase"] == "warmup" for r in rows),
            unknown_generation_attempts=int(pending is not None),
            unattempted_generations=1224-progress["attempted_generations"], interrupted_attempt=pending)
        write_json(root / "STATUS.json", progress)

    try:
        for entry in planned:
            boundary = boundary_check()
            errors = boundary_errors(boundary, physical_gpu_uuid)
            append_json(root / "BOUNDARY_CHECKS.jsonl", dict(activation=entry["activation"], check=boundary, errors=errors))
            if errors:
                raise RuntimeError("Evaluation boundary failed: " + repr(errors))
            method, checkpoint = entry["method"], checkpoints[entry["method"]]
            # Publish at every logical activation, even adjacent identical arms.
            # The qualified backend owns cache invalidation and native refresh.
            receipt = backend.publish(checkpoint)
            errors = publication_errors(receipt, checkpoint)
            progress["publications"] += 1
            progress["logical_activations"] += 1
            progress["activation_s"] += receipt["activation_s"]
            append_json(root / "ACTIVATIONS.jsonl", {**entry, **receipt, "errors": errors})
            if errors:
                raise RuntimeError("Checkpoint publication failed: " + repr(errors))
            if method not in checked:
                before = time.perf_counter()
                reply = backend.check_weights(checkpoint)
                errors = weight_check_errors(reply, checkpoint)
                elapsed = time.perf_counter() - before
                progress["weight_validation_s"] += elapsed
                write_json(root / f"WEIGHTS_{method}.json", dict(native_reply=reply, errors=errors, validation_s=elapsed))
                if errors:
                    raise RuntimeError("Published weights differ: " + repr(errors))
                checked.add(method)
            work = [("warmup", r, warmup_canonicals[r["request_id"]]) for r in warmup_requests]
            work.extend(("scored", requests[i], canonicals[requests[i]["request_id"]]) for i in entry["request_ordinals"])
            for phase, request, canonical in work:
                slot = {key: value for key, value in entry.items() if key != "request_ordinals"}
                slot.update(phase=phase, event=8, panel_ordinal=request["panel_ordinal"], request_id=request["request_id"])
                label = label_for(slot)
                path = root / (label + ".json")
                pending = {**slot, "label": label}
                progress["attempted_generations"] += 1
                persist_status()
                record = backend.generate(request["prompt_ids"], max_new_tokens=2048,
                    request_id=label, session_id=SESSION, request_index=request["panel_ordinal"],
                    purpose="evaluation", save_to=path)
                progress["returned_generations"] += 1
                # Persist raw values, including nonfinite evidence, before validation.
                with _temporary_file(path) as temporary:
                    temporary.write_text(json.dumps(record, allow_nan=True) + "\n")
                row, errors = measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid)
                key = phase, method, request["request_id"]
                previous = grouped[key]
                errors.extend(repeat_errors([*previous, row], method))
                if previous and row.get("native_accepted_draft_tokens") != previous[0].get("native_accepted_draft_tokens"):
                    errors.append("repeat_native_acceptance_mismatch")
                if previous and row.get("native_acceptance_histogram") != previous[0].get("native_acceptance_histogram"):
                    errors.append("repeat_native_histogram_mismatch")
                row.update(record_path=str(path), errors=sorted(set(errors)),
                    validation_status="invalid" if errors else "valid", valid_measurement=not errors)
                rows.append(row)
                previous.append(row)
                pending = None
                append_json(root / "MEASUREMENTS.jsonl", row)
                write_json(root / (label + "_VALIDATION.json"), dict(status=row["validation_status"], errors=row["errors"]))
                persist_status()
                if errors:
                    raise RuntimeError("Native correctness failure: " + repr(errors))
        final_boundary = boundary_check()
        errors = boundary_errors(final_boundary, physical_gpu_uuid)
        append_json(root / "BOUNDARY_CHECKS.jsonl", dict(activation="after_final", check=final_boundary, errors=errors))
        if errors:
            raise RuntimeError("Final evaluation boundary failed: " + repr(errors))
        progress["status"] = "completed"
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        raise
    finally:
        progress["evaluation_wall_s"] = time.perf_counter() - started
        persist_status()
    return progress


def _audit_raw_records(root, *, native_trace_resolver=None):
    """Recompute returned measurements from retained raw outputs; no GPU work."""
    root = Path(root)
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    requests = json.loads((root / "REQUESTS.json").read_text())
    canonicals = json.loads((root / "CANONICAL.json").read_text())
    warmups = json.loads((root / "WARMUPS.json").read_text())
    rows = [json.loads(line) for line in (root / "MEASUREMENTS.jsonl").read_text().splitlines()]
    errors, rebuilt = [], []
    resolve_native = native_trace_resolver or Path
    status = json.loads((root / "STATUS.json").read_text())
    planned = evaluation_schedule(requests, protocol["within_block_permutations"])
    if (protocol["schedule"] != planned or protocol["scope"] != SCOPE
            or protocol["capture"] is not False or protocol["event"] != 8
            or protocol["planned_scored"] != 1152 or protocol["planned_warmups"] != 72):
        errors.append("sealed_protocol_changed")
    for key, expected_value in dict(status="completed", planned_generations=1224,
            attempted_generations=1224, returned_generations=1224,
            completed_valid_generations=1224, invalid_generations=0,
            scored_generations=1152, warmup_generations=72, publications=36,
            logical_activations=36, unknown_generation_attempts=0,
            unattempted_generations=0, optimizer_steps=0,
            feedback_collections=0, teacher_preparations=0).items():
        if status.get(key) != expected_value:
            errors.append("status_" + key)
    activations = [json.loads(line) for line in (root / "ACTIVATIONS.jsonl").read_text().splitlines()]
    if len(activations) != 36:
        errors.append("activation_count")
    for saved, entry in zip(activations, planned):
        if (any(saved.get(key) != value for key, value in entry.items()) or saved.get("errors")
                or publication_errors(saved, protocol["checkpoints"][entry["method"]])):
            errors.append("activation_identity")
    for method in METHODS:
        weights = json.loads((root / f"WEIGHTS_{method}.json").read_text())
        if weights["errors"] or weight_check_errors(weights["native_reply"], protocol["checkpoints"][method]):
            errors.append("loaded_weights_" + method)
    boundaries = [json.loads(line) for line in (root / "BOUNDARY_CHECKS.jsonl").read_text().splitlines()]
    if [row["activation"] for row in boundaries] != [*range(36), "after_final"]:
        errors.append("boundary_schedule")
    for boundary in boundaries:
        errors.extend(boundary_errors(boundary["check"], protocol["physical_gpu_uuid"]))
        errors.extend(boundary["errors"])
    validate_references(requests, canonicals)
    validate_references(warmups["requests"], warmups["canonicals"])
    expected = []
    for entry in planned:
        for phase, ordinal in [("warmup", 0), ("warmup", 1), *[("scored", i) for i in entry["request_ordinals"]]]:
            slot = {k: v for k, v in entry.items() if k != "request_ordinals"}
            selected = warmups["requests"] if phase == "warmup" else requests
            slot.update(phase=phase, event=8, panel_ordinal=ordinal, request_id=selected[ordinal]["request_id"])
            expected.append(slot)
    if len(rows) != len(expected):
        errors.append("incomplete_measurements")
    grouped = defaultdict(list)
    for saved, slot in zip(rows, expected):
        request = (warmups["requests"] if slot["phase"] == "warmup" else requests)[slot["panel_ordinal"]]
        refs = warmups["canonicals"] if slot["phase"] == "warmup" else canonicals
        record = json.loads((root / (label_for(slot) + ".json")).read_text())
        native_path = resolve_native(record["metadata"]["native_trace_path"])
        if json.loads(Path(native_path).read_text()) != record["metadata"]["capture"]:
            errors.append(label_for(slot) + ": native_capture_changed")
        rebuilt_row, invalid = measure(record, request, protocol["checkpoints"][slot["method"]],
            refs[request["request_id"]], slot, protocol["physical_gpu_uuid"])
        if any(saved.get(key) != value for key, value in rebuilt_row.items()):
            invalid.append("saved_measurement_differs_from_raw")
        if saved.get("valid_measurement") is not True or saved.get("errors"):
            invalid.append("saved_invalid")
        group = grouped[slot["phase"], slot["method"], request["request_id"]]
        invalid.extend(repeat_errors([*group, rebuilt_row], slot["method"]))
        if group and group[0].get("native_accepted_draft_tokens") != rebuilt_row.get("native_accepted_draft_tokens"):
            invalid.append("repeat_native_acceptance_mismatch")
        if group and group[0].get("native_acceptance_histogram") != rebuilt_row.get("native_acceptance_histogram"):
            invalid.append("repeat_native_histogram_mismatch")
        group.append(rebuilt_row)
        errors.extend(label_for(slot) + ": " + error for error in sorted(set(invalid)))
        rebuilt_row.update(valid_measurement=not invalid, errors=invalid, validation_status="invalid" if invalid else "valid")
        rebuilt.append(rebuilt_row)
    return dict(status="passed" if not errors else "invalid_or_incomplete", errors=errors,
        execution_commit=protocol["execution_commit"],
        rows=rebuilt, extra_model_work=0)


def audit_raw_records(root, *, native_trace_resolver=None):
    """Missing or malformed evidence produces an incomplete result, never a pass."""
    try:
        return _audit_raw_records(root, native_trace_resolver=native_trace_resolver)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return dict(status="invalid_or_incomplete", errors=["evidence_read_or_schema: " + repr(error)],
            rows=[], extra_model_work=0)
