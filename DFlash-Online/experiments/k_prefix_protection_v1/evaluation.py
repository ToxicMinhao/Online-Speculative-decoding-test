"""Fixed F/K/K_HALF/KP serving cells, with one global first-F output reference."""

import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
    publication_errors, weight_check_errors,
)
from experiments.kw_v1.evaluate import append_json, safe_scalars, save_measurements, update_counts
from .contract import CELLS, METHODS, REPEATS, evaluation_schedule, method_order, validate_requests


def validate_rows(root, request, rows, records, canonical):
    """Resolve earlier candidate rows when this request's first valid F is available."""
    for row in rows:
        errors = list(row["structural_errors"])
        if canonical is not None:
            record = records[row["label"]]
            for name in ("response_ids", "finish_reason"):
                if record.get(name) != canonical[name]:
                    errors.append("reference_" + name)
            if row["method"] == "F":
                if any(row.get(name) != canonical["F_counts"][name] for name in COUNT_FIELDS):
                    errors.append("global_F_count_mismatch")
        errors.extend(measurement_errors(row, request, canonical))
        errors.extend(repeat_errors(rows, row["method"]))
        errors = sorted(set(errors))
        status = "invalid" if errors else "canonical_pending" if canonical is None else "valid"
        previous = row.get("validation_status"), row.get("errors")
        row.update(validation_status=status, errors=errors,
                   valid_measurement=True if status == "valid" else False if status == "invalid" else None)
        if previous != (status, errors):
            receipt = {"label": row["label"], "request_id": request["request_id"],
                       "method": row["method"], "repeat": row["repeat"],
                       "status": status, "errors": errors,
                       "canonical_source": canonical["source_record_path"] if canonical else None}
            append_json(root / "VALIDATION_EVENTS.jsonl", receipt)
            write_json(root / (row["label"] + "_VALIDATION.json"), receipt)


def evaluate(backend, requests, checkpoints, root, *, seed, event, canonical_path,
             qualification_requests=8):
    """Run one presealed cell; tiny CPU fixtures can use fewer panel requests."""
    if (seed, event) not in CELLS:
        raise ValueError("Evaluation cell is outside the fixed comparison")
    if set(checkpoints) != set(METHODS) or len(set(map(str, checkpoints.values()))) != 4:
        raise ValueError("Require distinct released F and retained K and new K_HALF/KP checkpoints")
    if not 1 <= qualification_requests <= len(requests):
        raise ValueError("The included correctness gate must fit the panel")
    validate_requests(requests, evaluation=True)
    checkpoints = {method: str(value) for method, value in checkpoints.items()}
    canonical_path = Path(canonical_path)
    first_cell = (seed, event) == CELLS[0]
    if first_cell and canonical_path.exists():
        raise ValueError("First-cell canonical exists; no automatic evaluation retry")
    canonicals = {} if first_cell else json.loads(canonical_path.read_text())
    if not first_cell and set(canonicals) != {request["request_id"] for request in requests}:
        raise ValueError("Later cells require the entire first-cell canonical panel")
    if any(value["F_checkpoint"] != checkpoints["F"] for value in canonicals.values()):
        raise ValueError("Released F identity changed across evaluation cells")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, checked = [], set()
    published = False
    active_attempt = None
    started = time.perf_counter()
    progress = {"status": "running", "seed": seed, "event": event,
                "planned_generations": len(requests) * len(METHODS) * REPEATS,
                "attempted_generations": 0, "returned_generations": 0,
                "qualification": "pending", "qualification_requests": qualification_requests,
                "activation_s": 0.0, "weight_validation_s": 0.0,
                "extra_generations": 0, "automatic_retry": False}
    write_json(root / "PROTOCOL.json", {
        "seed": seed, "event": event, "methods": list(METHODS), "repeats": REPEATS,
        "checkpoints": checkpoints, "schedule": evaluation_schedule(requests, seed=seed, event=event),
        "canonical_path": str(canonical_path), "canonical_source_cell": list(CELLS[0]),
        "canonical_source_method": "F", "count_trajectory_repeat": 0,
        "qualification_requests": qualification_requests, "extra_warmup_generations": 0,
        "capture": False, "performance_win_required": False})
    write_json(root / "REQUESTS.json", requests)
    write_json(root / "CANONICAL_INPUT.json", canonicals)
    try:
        for request in requests:
            ordinal = request["panel_ordinal"]
            request_rows, records = [], {}
            for repeat in range(REPEATS):
                for position, method in enumerate(method_order(ordinal, repeat)):
                    checkpoint = checkpoints[method]
                    if not published or backend.checkpoint_id != checkpoint:
                        activation = backend.publish(checkpoint)
                        errors = publication_errors(activation, checkpoint)
                        append_json(root / "ACTIVATIONS.jsonl", {
                            "panel_ordinal": ordinal, "repeat": repeat, "method": method,
                            **activation, "errors": errors})
                        progress["activation_s"] += activation["activation_s"]
                        if errors:
                            raise RuntimeError("Native publication failed: " + str(errors))
                        published = True
                    if method not in checked:
                        checking = time.perf_counter()
                        equality = backend.check_weights(checkpoint)
                        errors = weight_check_errors(equality, checkpoint)
                        elapsed = time.perf_counter() - checking
                        write_json(root / ("WEIGHTS_" + method + ".json"), {
                            "native_reply": equality, "errors": errors, "validation_s": elapsed})
                        progress["weight_validation_s"] += elapsed
                        if errors:
                            raise RuntimeError("Native weight equality failed: " + str(errors))
                        checked.add(method)
                    label = f"{ordinal:04d}_r{repeat}_{method}"
                    active_attempt = {"label": label, "request_id": request["request_id"],
                                      "method": method, "repeat": repeat}
                    progress["attempted_generations"] += 1
                    update_counts(progress, rows, active_attempt)
                    write_json(root / "STATUS.json", progress)
                    path = root / (label + ".json")
                    record = backend.generate(
                        request["prompt_ids"], max_new_tokens=request["max_new_tokens"],
                        request_id=label, session_id="k_prefix_protection_v1_evaluation", request_index=ordinal,
                        purpose="evaluation", save_to=path)
                    progress["returned_generations"] += 1
                    active_attempt = None
                    with _temporary_file(path) as temporary:
                        temporary.write_text(json.dumps(record, allow_nan=True) + "\n")
                    records[label] = record
                    try:
                        errors = record_errors(record, request, checkpoint, canonicals.get(request["request_id"]))
                        if (record["record_id"] != label or record["request_index"] != ordinal
                                or record["session_id"] != "k_prefix_protection_v1_evaluation"):
                            errors.append("native_identity")
                        row = score_record(record, request_id=request["request_id"], repeat=repeat, method=method)
                        row.update(finish_reason=record["finish_reason"],
                                   cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"])
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        errors = ["malformed_native_record: " + str(error)]
                        row = {"request_id": request["request_id"], "repeat": repeat,
                               "method": method, "checkpoint_id": checkpoint}
                    row = safe_scalars(row)
                    row.update(label=label, seed=seed, event=event, position=position,
                               panel_ordinal=ordinal, original_index=request["original_index"],
                               record_path=str(path), structural_errors=list(errors),
                               validation_status="invalid", valid_measurement=False,
                               errors=["validation_not_completed"])
                    rows.append(row)
                    request_rows.append(row)
                    errors.extend(measurement_errors(row, request))
                    if method == "F" and request["request_id"] not in canonicals and not errors:
                        canonicals[request["request_id"]] = {
                            "response_ids": list(record["response_ids"]), "finish_reason": record["finish_reason"],
                            "source_record_path": str(path), "source_method": "F", "source_repeat": repeat,
                            "source_cell": [seed, event], "F_checkpoint": checkpoint,
                            "F_counts": {name: row[name] for name in COUNT_FIELDS}}
                        write_json(canonical_path, canonicals)
                    validate_rows(root, request, request_rows, records, canonicals.get(request["request_id"]))
                    save_measurements(root, rows)
                    update_counts(progress, rows, active_attempt)
                    write_json(root / "STATUS.json", progress)
                    invalid = [row for row in request_rows if row["validation_status"] == "invalid"]
                    if invalid:
                        raise RuntimeError("Evaluation correctness failed: " + str(invalid[-1]["errors"]))
            if ordinal + 1 == qualification_requests:
                progress["qualification"] = "passed"
                write_json(root / "QUALIFICATION.json", {
                    "status": "passed", "requests": qualification_requests,
                    "generations": progress["returned_generations"],
                    "methods_with_native_equality": sorted(checked),
                    "performance_win_required": False, "extra_generations": 0})
        progress["status"] = "completed"
    except BaseException as error:
        progress.update(status="failed", error_type=type(error).__name__, error=str(error))
        if progress["qualification"] == "pending":
            progress["qualification"] = "incomplete"
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress["evaluation_wall_s"] = time.perf_counter() - started
        save_measurements(root, rows)
        write_json(root / "CANONICAL.json", canonicals)
        write_json(root / "STATUS.json", progress)
    return progress
