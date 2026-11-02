"""Fixed capture-off endpoint with durable attempts and native validity checks."""
from collections import defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.kw_v1.evaluate import append_json, safe_scalars, update_counts
from .contract import ARMS, final_schedule

SESSION = "k_workload_replay_v1_final"
REPEATS = 3


def validate_inputs(requests, checkpoints, canonicals):
    if (len(requests) != 256 or [r["panel_ordinal"] for r in requests] != list(range(256))
            or len({r["request_id"] for r in requests}) != 256
            or set(canonicals) != {r["request_id"] for r in requests}):
        raise ValueError("Keep the complete sealed 256-request panel and canonical identities")
    if set(checkpoints) != set(ARMS) or len(set(checkpoints.values())) != 3:
        raise ValueError("Require exactly three distinct final K/U/R checkpoints")
    for ordinal, request in enumerate(requests):
        if request["domain"] != ("math" if ordinal % 2 == 0 else "coding"):
            raise ValueError("Keep the fixed alternating math/coding panel")
        if (request["max_new_tokens"] != 2048 or not request["prompt_ids"]
                or len(request["prompt_ids"]) + 2048 > 4096 or not request["stop_token_ids"]):
            raise ValueError("Keep the qualified full-output cap, context and stop settings")
        if any(type(token) is not int or token < 0 for token in request["prompt_ids"] + request["stop_token_ids"]):
            raise ValueError("Invalid input token identity")
        canonical = canonicals[request["request_id"]]
        if (not canonical["response_ids"] or canonical["finish_reason"] not in ("eos", "length")
                or not canonical["source_record_path"]):
            raise ValueError("Complete retained canonical outputs and source paths are required")
    return final_schedule(requests)


def measurement(record, request, checkpoint, canonical, slot):
    label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
    try:
        errors = validate_record(record, request, checkpoint, canonical, label=label, session=SESSION)
        row = score_record(record, request_id=request["request_id"], repeat=slot["repeat"], method=slot["method"])
        row.update(finish_reason=record["finish_reason"],
                   cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"])
        errors.extend(measurement_errors(row, request, canonical))
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors = ["malformed_native_record: " + repr(error)]
        row = {"checkpoint_id": str(checkpoint)}
    row = safe_scalars(row)
    row.update(**slot, seed=17, event=8, phase="final", capture=False, label=label)
    return row, errors


def evaluate_final(backend, requests, checkpoints, root, *, canonicals, boundary_check=None):
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = validate_inputs(requests, checkpoints, canonicals)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, grouped, checked = [], defaultdict(list), set()
    active_attempt = None
    progress = dict(status="running", planned_generations=len(planned), attempted_generations=0,
                    returned_generations=0, activation_s=0.0, weight_validation_s=0.0,
                    extra_generations=0, optimizer_steps=0, feedback_collections=0, automatic_retry=False,
                    boundary_checks=[])
    write_json(root / "PROTOCOL.json", dict(checkpoints=checkpoints, schedule=planned,
        capture=False, extra_warmup_generations=0, canonical_source="retained_exact_math_and_coding_outputs",
        count_trajectory_repeat=0, timing_scope="contemporaneous_checkpoint_ready_serving",
        publication_and_weight_checks_outside_request_timing=True,
        boundary_check_enabled=boundary_check is not None))
    write_json(root / "REQUESTS.json", requests)
    write_json(root / "CANONICAL_INPUT.json", canonicals)
    try:
        current, active_batch = None, None
        for slot in planned:
            method, ordinal = slot["method"], slot["panel_ordinal"]
            checkpoint = checkpoints[method]
            if current != checkpoint:
                activation = backend.publish(checkpoint)
                errors = publication_errors(activation, checkpoint)
                append_json(root / "ACTIVATIONS.jsonl", {**slot, **activation, "errors": errors})
                progress["activation_s"] += activation["activation_s"]
                if errors:
                    raise RuntimeError("Native checkpoint publication failed: " + repr(errors))
                current = checkpoint
            if method not in checked:
                checking = time.perf_counter()
                weights = backend.check_weights(checkpoint)
                errors = weight_check_errors(weights, checkpoint)
                elapsed = time.perf_counter() - checking
                write_json(root / f"WEIGHTS_{method}.json", dict(native_reply=weights, errors=errors, validation_s=elapsed))
                progress["weight_validation_s"] += elapsed
                if errors:
                    raise RuntimeError("Loaded/exported weight mismatch: " + repr(errors))
                checked.add(method)
            batch = (slot["block_ordinal"], slot["repeat"], slot["position"])
            if batch != active_batch:
                if boundary_check is not None:
                    receipt = boundary_check()
                    progress["boundary_checks"].append(dict(boundary="before_batch", **slot, receipt=receipt))
                active_batch = batch
            request = requests[ordinal]
            label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
            path = root / (label + ".json")
            active_attempt = {**slot, "label": label}
            progress["attempted_generations"] += 1
            update_counts(progress, rows, active_attempt)
            write_json(root / "STATUS.json", progress)
            record = backend.generate(request["prompt_ids"], max_new_tokens=request["max_new_tokens"],
                request_id=label, session_id=SESSION, request_index=ordinal, purpose="evaluation", save_to=path)
            progress["returned_generations"] += 1
            # Keep the attempt unresolved until its raw result is durable.
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + "\n")
            row, errors = measurement(record, request, checkpoint, canonicals[request["request_id"]], slot)
            row.update(record_path=str(path), structural_errors=list(errors))
            group = grouped[request["request_id"], method]
            errors.extend(repeat_errors([*group, row], method))
            errors = sorted(set(errors))
            row.update(errors=errors, valid_measurement=not errors,
                       validation_status="invalid" if errors else "valid")
            append_json(root / "MEASUREMENTS.jsonl", row)
            rows.append(row)
            group.append(row)
            active_attempt = None
            write_json(root / (label + "_VALIDATION.json"), dict(status=row["validation_status"], errors=errors,
                canonical_source=canonicals[request["request_id"]]["source_record_path"]))
            update_counts(progress, rows, active_attempt)
            write_json(root / "STATUS.json", progress)
            if errors:
                raise RuntimeError("Native evaluation correctness failed: " + repr(errors))
        if boundary_check is not None:
            receipt = boundary_check()
            progress["boundary_checks"].append(dict(boundary="after_final_batch", receipt=receipt))
        progress["status"] = "completed"
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress["evaluation_wall_s"] = time.perf_counter() - started
        write_json(root / "STATUS.json", progress)
    return progress


def audit_final(folder, requests, checkpoints, *, canonicals):
    """Reconstruct endpoint validity from raw native evidence before reporting."""
    folder = Path(folder)
    rows, errors, validated, progress = [], [], [], {}
    try:
        checkpoints = {method: str(path) for method, path in checkpoints.items()}
        planned = validate_inputs(requests, checkpoints, canonicals)
        protocol = json.loads((folder / "PROTOCOL.json").read_text())
        progress = json.loads((folder / "STATUS.json").read_text())
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        if (protocol["schedule"] != planned or protocol["checkpoints"] != checkpoints
                or protocol["capture"] is not False or protocol["extra_warmup_generations"] != 0
                or json.loads((folder / "REQUESTS.json").read_text()) != requests
                or json.loads((folder / "CANONICAL_INPUT.json").read_text()) != canonicals):
            errors.append("sealed_inputs_or_protocol_changed")
        if len(rows) != len(planned) or [{k: row.get(k) for k in slot} for row, slot in zip(rows, planned)] != planned:
            errors.append("incomplete_or_changed_schedule")
        required = dict(status="completed", planned_generations=len(planned), attempted_generations=len(planned),
            returned_generations=len(planned), completed_valid_generations=len(planned), invalid_generations=0,
            unknown_generation_attempts=0, unattempted_generations=0, pending_canonical_generations=0,
            extra_generations=0, optimizer_steps=0, feedback_collections=0, automatic_retry=False)
        errors.extend("status_" + key for key, value in required.items() if progress.get(key) != value)
        if protocol.get("boundary_check_enabled"):
            expected_batches = list(dict.fromkeys((s["block_ordinal"], s["repeat"], s["position"]) for s in planned))
            boundaries = progress.get("boundary_checks", [])
            observed_batches = [(b.get("block_ordinal"), b.get("repeat"), b.get("position"))
                                for b in boundaries if b.get("boundary") == "before_batch"]
            if (observed_batches != expected_batches or len(boundaries) != len(expected_batches) + 1
                    or not boundaries or boundaries[-1].get("boundary") != "after_final_batch"):
                errors.append("incomplete_boundary_checks")
        expected_publications, current = [], None
        for slot in planned:
            if checkpoints[slot["method"]] != current:
                expected_publications.append(slot)
                current = checkpoints[slot["method"]]
        activations = [json.loads(line) for line in (folder / "ACTIVATIONS.jsonl").read_text().splitlines()]
        if len(activations) != len(expected_publications):
            errors.append("publication_count")
        for saved, slot in zip(activations, expected_publications):
            if (any(saved.get(key) != value for key, value in slot.items()) or saved["errors"]
                    or publication_errors(saved, checkpoints[slot["method"]])):
                errors.append("checkpoint_publication")
        for method, checkpoint in checkpoints.items():
            weights = json.loads((folder / f"WEIGHTS_{method}.json").read_text())
            if weights["errors"] or weight_check_errors(weights["native_reply"], checkpoint):
                errors.append("loaded_weights_" + method)
        reader = ArchiveReader(folder)
        grouped = defaultdict(list)
        for saved, slot in zip(rows, planned):
            ordinal, method = slot["panel_ordinal"], slot["method"]
            label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
            raw = json.loads((folder / (label + ".json")).read_text())
            native_path = raw["metadata"]["native_trace_path"]
            if "/campaign_001/" in native_path:
                reader.origin = Path(native_path.split("/campaign_001/", 1)[0])
                reader.root = Path(str(folder).split("/campaign_001/", 1)[0])
            native = reader.read(reader.saved_path(native_path))
            if native != raw["metadata"]["capture"]:
                errors.append(label + ": original_native_capture_changed")
            row, invalid = measurement(raw, requests[ordinal], checkpoints[method], canonicals[slot["request_id"]], slot)
            group = grouped[slot["request_id"], method]
            invalid.extend(repeat_errors([*group, row], method))
            if (invalid or saved["validation_status"] != "valid" or saved["valid_measurement"] is not True
                    or saved["errors"] or saved["structural_errors"]
                    or any(saved.get(key) != value for key, value in row.items())):
                errors.append(label + ": " + repr(invalid or ["saved_measurement_disagreement"]))
            group.append(row)
            validated.append({**saved, **row})
    except (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors.append("missing_or_malformed_evidence: " + repr(error))
    return dict(status="passed" if not errors else "invalid_or_incomplete", rows=validated,
                saved_rows=rows, errors=errors, progress=progress)
