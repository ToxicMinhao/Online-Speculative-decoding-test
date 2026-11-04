"""Five real policy exports, one native serving session, and strict raw audits."""

from collections import defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    measurement_errors,
    publication_errors,
    repeat_errors,
    weight_check_errors,
)
from experiments.fresh_confirmation.evaluate import save_measurements, update_counts
from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.kw_v1.evaluate import append_json, safe_scalars
from experiments.mathqa_transfer.evaluation import activate

METHODS = ("FULL", "ONCE", "RANDOM64", "STATIC64", "PROGRESS64")
REPEATS = 2
PANEL_SIZE = 200
BLOCK_SIZE = 20
SESSION = "vfd_second_pass_evaluation"


def final_schedule(requests):
    """Two sweeps; rotate by request block and reverse the same order in sweep2."""
    slots = []
    for repeat in range(REPEATS):
        for start in range(0, len(requests), BLOCK_SIZE):
            block = start // BLOCK_SIZE
            shift = block % len(METHODS)
            order = METHODS[shift:] + METHODS[:shift]
            if repeat:
                order = tuple(reversed(order))
            for position, method in enumerate(order):
                for ordinal in range(start, min(start + BLOCK_SIZE, len(requests))):
                    slots.append(
                        dict(
                            seed=17,
                            event=4,
                            phase="final",
                            repeat=repeat,
                            block_ordinal=block,
                            panel_ordinal=ordinal,
                            position=position,
                            request_id=requests[ordinal]["request_id"],
                            method=method,
                        )
                    )
    return slots


def validate_inputs(requests, checkpoints, canonicals, *, production=True):
    if set(checkpoints) != set(METHODS) or len(
        set(map(str, checkpoints.values()))
    ) != len(METHODS):
        raise ValueError("Require five distinct real policy checkpoint identities")
    if not requests or (production and len(requests) != PANEL_SIZE):
        raise ValueError("Require the sealed 200-request endpoint")
    if [r["panel_ordinal"] for r in requests] != list(range(len(requests))):
        raise ValueError("Require consecutive panel ordinals")
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate request identity")
    if set(canonicals) != {r["request_id"] for r in requests}:
        raise ValueError(
            "Every timed request must have a previously validated canonical reference"
        )
    bindings = {value.get("binding_digest") for value in canonicals.values()}
    if len(bindings) != 1 or not next(iter(bindings)):
        raise ValueError(
            "Every canonical must share the same authenticated target/runtime binding"
        )
    for request in requests:
        if (
            request["max_new_tokens"] != 2048
            or not request["prompt_ids"]
            or len(request["prompt_ids"]) + 2048 > 4096
            or any(type(v) is not int or v < 0 for v in request["prompt_ids"])
            or request["stop_token_ids"] != [151645, 151643]
            or type(request["original_index"]) is not int
            or request["original_index"] < 0
        ):
            raise ValueError("Request differs from pinned native runtime contract")
        reference = canonicals[request["request_id"]]
        if (
            reference.get("prompt_ids") != request["prompt_ids"]
            or reference.get("verified") is not True
        ):
            raise ValueError("Canonical prompt or validity binding missing")
        ids = reference["response_ids"]
        if (
            not ids
            or len(ids) > 2048
            or any(type(v) is not int or v < 0 for v in ids)
            or reference["finish_reason"] not in ("eos", "length")
        ):
            raise ValueError("Malformed canonical output")
    return final_schedule(requests)


def label_for(slot):
    return f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"


def measure(record, request, checkpoint, canonical, slot):
    label = label_for(slot)
    try:
        errors = validate_record(
            record, request, checkpoint, canonical, label=label, session=SESSION
        )
        row = score_record(
            record,
            request_id=request["request_id"],
            repeat=slot["repeat"],
            method=slot["method"],
        )
        row.update(
            finish_reason=record["finish_reason"],
            cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"],
        )
        errors.extend(measurement_errors(row, request, canonical))
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors = ["malformed_native_record: " + repr(error)]
        row = dict(
            request_id=request["request_id"],
            repeat=slot["repeat"],
            method=slot["method"],
            checkpoint_id=str(checkpoint),
        )
    row = safe_scalars(row)
    row.update(**slot, label=label, original_index=request["original_index"])
    return row, sorted(set(errors))


def evaluate(
    backend,
    requests,
    checkpoints,
    root,
    *,
    canonicals,
    boundary_check=None,
    production=True,
):
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = validate_inputs(requests, checkpoints, canonicals, production=production)
    if production and boundary_check is None:
        raise ValueError(
            "Require storage and training/transfer quiescence check at every block boundary"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, grouped, checked = [], defaultdict(list), set()
    active_attempt, active_batch = None, None
    started = time.perf_counter()
    progress = dict(
        status="running",
        planned_generations=len(planned),
        attempted_generations=0,
        returned_generations=0,
        activation_s=0.0,
        weight_validation_s=0.0,
        publications=0,
        optimizer_steps=0,
        feedback_collections=0,
        teacher_preparations=0,
        extra_generations=0,
        automatic_retry=False,
    )
    write_json(
        root / "PROTOCOL.json",
        dict(
            schedule=planned,
            checkpoints=checkpoints,
            capture=False,
            extra_warmup_generations=0,
            canonical_source="sealed_verified_references",
            count_trajectory_repeat=0,
        ),
    )
    write_json(root / "REQUESTS.json", requests)
    write_json(root / "CANONICAL.json", canonicals)
    try:
        for slot in planned:
            method, ordinal = slot["method"], slot["panel_ordinal"]
            batch = (slot["block_ordinal"], slot["repeat"], slot["position"])
            if batch != active_batch and boundary_check is not None:
                check = boundary_check()
                append_json(
                    root / "BOUNDARY_CHECKS.jsonl", dict(batch=batch, check=check)
                )
                if boundary_errors(check):
                    raise RuntimeError(
                        "Failed evaluation boundary: " + repr(boundary_errors(check))
                    )
            active_batch = batch
            checkpoint = checkpoints[method]
            activate(backend, checkpoint, slot, root, progress, checked)
            request = requests[ordinal]
            label = label_for(slot)
            path = root / (label + ".json")
            active_attempt = {**slot, "label": label}
            progress["attempted_generations"] += 1
            update_counts(progress, rows, active_attempt)
            write_json(root / "STATUS.json", progress)
            record = backend.generate(
                request["prompt_ids"],
                max_new_tokens=request["max_new_tokens"],
                request_id=label,
                session_id=SESSION,
                request_index=ordinal,
                purpose="evaluation",
                save_to=path,
            )
            progress["returned_generations"] += 1
            active_attempt = None
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + "\n")
            row, errors = measure(
                record, request, checkpoint, canonicals[request["request_id"]], slot
            )
            group = grouped[request["request_id"], method]
            errors.extend(repeat_errors([*group, row], method))
            row.update(
                record_path=str(path),
                structural_errors=sorted(set(errors)),
                errors=sorted(set(errors)),
                validation_status="invalid" if errors else "valid",
                valid_measurement=not errors,
            )
            rows.append(row)
            group.append(row)
            write_json(
                root / (label + "_VALIDATION.json"),
                dict(
                    label=label, status=row["validation_status"], errors=row["errors"]
                ),
            )
            save_measurements(root, rows)
            update_counts(progress, rows, active_attempt)
            write_json(root / "STATUS.json", progress)
            if errors:
                raise RuntimeError("Native comparison invalid: " + repr(errors))
        progress["status"] = "completed"
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress["evaluation_wall_s"] = time.perf_counter() - started
        save_measurements(root, rows)
        write_json(root / "STATUS.json", progress)
    return progress


def boundary_errors(check):
    errors = []
    if (
        check["storage"]["status"] != "passed"
        or check["storage"]["free_bytes"] < check["storage"]["required_bytes"]
    ):
        errors.append("storage_boundary")
    if check["quiescence"]["status"] != "passed" or any(
        check["quiescence"][key]
        for key in ("training", "transfers", "foreign_gpu_processes")
    ):
        errors.append("quiescence_boundary")
    return errors


def audit_stage(
    folder, requests, checkpoints, *, canonicals, evidence_root=None, production=True
):
    """Reconstruct scalars from raw outputs and sidecars; never trust a validity flag."""
    folder = Path(folder)
    errors, validated, progress = [], [], {}
    try:
        checkpoints = {method: str(path) for method, path in checkpoints.items()}
        planned = validate_inputs(
            requests, checkpoints, canonicals, production=production
        )
        protocol = json.loads((folder / "PROTOCOL.json").read_text())
        rows = [
            json.loads(line)
            for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()
        ]
        progress = json.loads((folder / "STATUS.json").read_text())
        if (
            protocol
            != dict(
                schedule=planned,
                checkpoints=checkpoints,
                capture=False,
                extra_warmup_generations=0,
                canonical_source="sealed_verified_references",
                count_trajectory_repeat=0,
            )
            or json.loads((folder / "REQUESTS.json").read_text()) != requests
            or json.loads((folder / "CANONICAL.json").read_text()) != canonicals
        ):
            errors.append("sealed_protocol_changed")
        if (
            len(rows) != len(planned)
            or [{k: r.get(k) for k in slot} for r, slot in zip(rows, planned)]
            != planned
        ):
            errors.append("incomplete_or_changed_schedule")
        for key, expected in dict(
            status="completed",
            planned_generations=len(planned),
            attempted_generations=len(planned),
            returned_generations=len(planned),
            completed_valid_generations=len(planned),
            invalid_generations=0,
            unknown_generation_attempts=0,
            unattempted_generations=0,
            pending_canonical_generations=0,
            optimizer_steps=0,
            feedback_collections=0,
            teacher_preparations=0,
            extra_generations=0,
        ).items():
            if progress.get(key) != expected:
                errors.append("status_" + key)
        for method in METHODS:
            weights = json.loads((folder / f"WEIGHTS_{method}.json").read_text())
            if weights["errors"] or weight_check_errors(
                weights["native_reply"], checkpoints[method]
            ):
                errors.append("loaded_weights_" + method)
        activations = [
            json.loads(line)
            for line in (folder / "ACTIVATIONS.jsonl").read_text().splitlines()
        ]
        expected_slots, current = [], None
        for slot in planned:
            if slot["method"] != current:
                expected_slots.append(slot)
                current = slot["method"]
        if len(activations) != len(expected_slots):
            errors.append("publication_count")
        for saved, slot in zip(activations, expected_slots):
            if (
                any(saved.get(k) != v for k, v in slot.items())
                or saved["errors"]
                or publication_errors(saved, checkpoints[slot["method"]])
            ):
                errors.append("publication_identity")
        boundary_file = folder / "BOUNDARY_CHECKS.jsonl"
        if production or boundary_file.exists():
            boundaries = [
                json.loads(line) for line in boundary_file.read_text().splitlines()
            ]
            batches = []
            for slot in planned:
                batch = [slot["block_ordinal"], slot["repeat"], slot["position"]]
                if not batches or batch != batches[-1]:
                    batches.append(batch)
            if [row["batch"] for row in boundaries] != batches:
                errors.append("boundary_check_schedule")
            for row in boundaries:
                errors.extend(boundary_errors(row["check"]))
        reader = ArchiveReader(evidence_root or folder)
        grouped = defaultdict(list)
        for saved, slot in zip(rows, planned):
            request, method = requests[slot["panel_ordinal"]], slot["method"]
            label = label_for(slot)
            raw = json.loads((folder / (label + ".json")).read_text())
            native_path = raw["metadata"]["native_trace_path"]
            if "/campaign_001/" in native_path:
                reader.origin = Path(native_path.split("/campaign_001/", 1)[0])
            if (
                reader.read(reader.saved_path(native_path))
                != raw["metadata"]["capture"]
            ):
                errors.append(label + ": native_capture_changed")
            row, invalid = measure(
                raw,
                request,
                checkpoints[method],
                canonicals[request["request_id"]],
                slot,
            )
            group = grouped[request["request_id"], method]
            invalid.extend(repeat_errors([*group, row], method))
            if (
                invalid
                or saved.get("validation_status") != "valid"
                or saved.get("valid_measurement") is not True
                or saved.get("errors")
                or saved.get("structural_errors")
                or any(saved.get(k) != v for k, v in row.items())
            ):
                errors.append(
                    label + ": " + repr(invalid or ["saved_measurement_disagreement"])
                )
            group.append(row)
            validated.append({**saved, **row})
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        RuntimeError,
    ) as error:
        errors.append("missing_or_malformed_evidence: " + repr(error))
    return dict(
        status="passed" if not errors else "invalid_or_incomplete",
        errors=errors,
        rows=validated,
        progress=progress,
        model_work=0,
    )


def prepare_canonicals(backend, requests, root, *, canonicals, binding_digest):
    """Generate each missing canonical once, untimed, with the qualified verifier.

    Native timers remain in raw evidence but are excluded from every endpoint
    estimate. This is verifier-derived reference generation, not a separate
    target-only replication. The campaign authenticates binding_digest against
    its target/runtime preflight before invoking this function.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(canonicals)
    missing = [request for request in requests if request["request_id"] not in result]
    if len(missing) > PANEL_SIZE or set(result) - {r["request_id"] for r in requests}:
        raise ValueError("Canonical generation budget or identity mismatch")
    if any(value.get("binding_digest") != binding_digest for value in result.values()):
        raise ValueError("Retained canonical target/runtime binding differs")
    status = dict(
        status="running",
        planned_generations=len(missing),
        attempted_generations=0,
        returned_generations=0,
        verified_generations=0,
        capture=False,
        automatic_retry=False,
    )
    write_json(root / "STATUS.json", status)
    write_json(root / "INPUT_CANONICALS.json", result)
    try:
        for request in missing:
            label = f"canonical_{request['panel_ordinal']:04d}"
            path = root / (label + ".json")
            status["attempted_generations"] += 1
            status["active_request"] = request["request_id"]
            write_json(root / "STATUS.json", status)
            record = backend.generate(
                request["prompt_ids"],
                max_new_tokens=request["max_new_tokens"],
                request_id=label,
                session_id=SESSION,
                request_index=request["panel_ordinal"],
                purpose="evaluation",
                save_to=path,
            )
            status["returned_generations"] += 1
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + "\n")
            errors = validate_record(
                record,
                request,
                backend.checkpoint_id,
                record,
                label=label,
                session=SESSION,
            )
            row = score_record(
                record, request_id=request["request_id"], repeat=0, method="canonical"
            )
            row.update(
                finish_reason=record["finish_reason"],
                cache_maintenance_s=record["metadata"]["capture"][
                    "cache_maintenance_s"
                ],
            )
            errors.extend(measurement_errors(row, request, record))
            write_json(
                root / (label + "_VALIDATION.json"),
                dict(errors=errors, valid=not errors),
            )
            if errors:
                raise RuntimeError(
                    "Canonical structural integrity failed: " + repr(errors)
                )
            result[request["request_id"]] = dict(
                prompt_ids=request["prompt_ids"],
                response_ids=record["response_ids"],
                finish_reason=record["finish_reason"],
                verified=True,
                binding_digest=binding_digest,
                source="fresh_qualified_verifier_once",
                source_record_path=str(path),
            )
            status["verified_generations"] += 1
            status["active_request"] = None
            write_json(root / "CANONICALS.json", result)
            write_json(root / "STATUS.json", status)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(root / "STATUS.json", status)
        write_json(root / "CANONICALS.json", result)
    return result


def audit_canonicals(requests, canonicals, *, binding_digest, evidence_root=None):
    """Recheck retained hashes and newly generated native references, without model work."""
    from .data import file_identity

    errors, checked = [], []
    reader = ArchiveReader(evidence_root or ".")
    for request in requests:
        try:
            reference = canonicals[request["request_id"]]
            if (
                reference["binding_digest"] != binding_digest
                or reference["verified"] is not True
                or reference["prompt_ids"] != request["prompt_ids"]
            ):
                raise ValueError("canonical binding mismatch")
            if reference["source"] == "audited_PR25_exact_prompt":
                source_path = Path(reference["source_record"]["path"])
                native_path = Path(reference["source_native"]["path"])
                if not source_path.is_file():
                    source_path = Path(reference["source_original_record_path"])
                if not native_path.is_file():
                    native_path = Path(reference["source_original_native_path"])
                # Remote sealing may provide explicitly relocated source paths;
                # identities remain path-independent hashes and byte sizes.
                for name, path in [
                    ("source_record", source_path),
                    ("source_native", native_path),
                ]:
                    identity = file_identity(path)
                    if any(
                        identity[key] != reference[name][key]
                        for key in ("bytes", "sha256")
                    ):
                        raise ValueError("retained reference source changed")
                raw = json.loads(source_path.read_text())
                native = json.loads(native_path.read_text())
                validation_request = dict(request, panel_ordinal=raw["request_index"])
                label, session = raw["record_id"], raw["session_id"]
            elif reference["source"] == "fresh_qualified_verifier_once":
                recorded = reference["source_record_path"]
                if "/campaign_001/" in recorded:
                    reader.origin = Path(recorded.split("/campaign_001/", 1)[0])
                source_path = reader.saved_path(recorded)
                raw = reader.read(source_path)
                native = reader.read(
                    reader.saved_path(raw["metadata"]["native_trace_path"])
                )
                validation_request = request
                label, session = f"canonical_{request['panel_ordinal']:04d}", SESSION
            else:
                raise ValueError("Unknown canonical provenance")
            if native != raw["metadata"]["capture"]:
                raise ValueError("canonical native sidecar differs")
            invalid = validate_record(
                raw,
                validation_request,
                raw["checkpoint_id"],
                reference,
                label=label,
                session=session,
            )
            if invalid:
                raise ValueError("canonical record invalid: " + repr(invalid))
            checked.append(
                dict(
                    request_id=request["request_id"],
                    source=reference["source"],
                    source_record_path=str(source_path),
                )
            )
        except (
            OSError,
            KeyError,
            ValueError,
            TypeError,
            IndexError,
            RuntimeError,
        ) as error:
            errors.append(dict(request_id=request["request_id"], error=repr(error)))
    return dict(
        status="passed" if not errors else "invalid_or_incomplete",
        errors=errors,
        references_checked=len(checked),
        checked=checked,
        model_work=0,
    )
