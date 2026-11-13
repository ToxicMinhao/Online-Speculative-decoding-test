"""Recent-repair retention serving with NATIVE/RECENT/CONTROL arms and native raw audits.

Inherited PR39 complete native validity/cache checks; historical callers remain unchanged.
"""

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

from experiments.survival_head.contract import SERVING

METHODS = ("NATIVE", "RECENT", "CONTROL")
DATASETS = ("gsm8k",)
PANEL_SIZE = 256
BLOCK_SIZE = 8
EVENT = 7
ENDPOINT_REPEATS = 2
QUALIFICATION_REQUESTS = 8

SESSION = "recent_repair_retention_evaluation"


def _native_evaluation_schedule(requests, dataset="gsm8k", event=EVENT):
    """Input-only eight-request blocks: cyclic arm order, then reverse.

    Each arm runs a whole block before the next publication. Sweep two keeps
    request/block order and reverses each block's arm order, balancing pair
    precedence. When the block count is not divisible by3, positions are approximately balanced;
    this fixed imbalance is reported and never changed in response to outcomes.
    """
    if dataset != "gsm8k" or event != EVENT or not requests:
        raise ValueError("Only the prescribed GSM8K event-seven panel is evaluated")
    slots = []
    for repeat in range(ENDPOINT_REPEATS):
        for start in range(0, len(requests), BLOCK_SIZE):
            block = start // BLOCK_SIZE
            shift = block % len(METHODS)
            order = METHODS[shift:] + METHODS[:shift]
            if repeat:
                order = order[::-1]
            for position, method in enumerate(order):
                for ordinal in range(start, min(start + BLOCK_SIZE, len(requests))):
                    slots.append(
                        dict(
                            seed=17,
                            dataset=dataset,
                            event=EVENT,
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


def schedule_balance(slots):
    from collections import Counter

    blocks = {}
    for slot in slots:
        key = slot["repeat"], slot["block_ordinal"]
        blocks.setdefault(key, {})[slot["position"]] = slot["method"]
    orders = [tuple(value[p] for p in range(len(METHODS))) for value in blocks.values()]
    return dict(
        block_repeat_orders=len(orders),
        generations=len(slots),
        positions={
            m: [sum(order[p] == m for order in orders) for p in range(len(METHODS))]
            for m in METHODS
        },
        orders=[
            dict(order=list(order), count=count)
            for order, count in sorted(Counter(orders).items())
        ],
        pair_precedence={
            a + "_before_" + b: sum(order.index(a) < order.index(b) for order in orders)
            for a in METHODS
            for b in METHODS
            if a != b
        },
    )


def _native_validate_evaluation_inputs(
    requests, checkpoints, canonicals, *, dataset, event, schedule=None, production=True
):
    if set(checkpoints) != set(METHODS):
        raise ValueError("Require three explicitly identified arm exports; identical CONTROL reuse is declared separately")
    if not 0 < len(requests) <= PANEL_SIZE:
        raise ValueError("Require the sealed retention panel of at most256 requests")
    if [r["panel_ordinal"] for r in requests] != list(range(len(requests))):
        raise ValueError("Require consecutive panel ordinals")
    if len({r["request_id"] for r in requests}) != len(requests) or set(canonicals) != {
        r["request_id"] for r in requests
    }:
        raise ValueError("Missing or duplicate canonical request identity")
    bindings = {value.get("binding_digest") for value in canonicals.values()}
    if len(bindings) != 1 or not next(iter(bindings)):
        raise ValueError("Require one authenticated target/runtime canonical binding")
    for request in requests:
        reference = canonicals[request["request_id"]]
        ids = reference["response_ids"]
        if (
            request["max_new_tokens"] != 2048
            or not request["prompt_ids"]
            or len(request["prompt_ids"]) + 2048 > 4096
            or any(type(v) is not int or v < 0 for v in request["prompt_ids"])
            or not request["stop_token_ids"]
            or type(request["original_index"]) is not int
            or request["original_index"] < 0
            or reference.get("prompt_ids") != request["prompt_ids"]
            or reference.get("verified") is not True
            or not ids
            or len(ids) > 2048
            or any(type(v) is not int or v < 0 for v in ids)
            or reference["finish_reason"] not in ("eos", "length")
        ):
            raise ValueError("Input or canonical differs from pinned native contract")
        if (reference["finish_reason"] == "length" and len(ids) != 2048) or (
            reference["finish_reason"] == "eos"
            and ids[-1] not in request["stop_token_ids"]
        ):
            raise ValueError("Canonical termination differs from EOS/output cap")
    planned = _native_evaluation_schedule(requests, dataset, event)
    if production and schedule is None:
        raise ValueError("Production requires the presealed timing schedule")
    if schedule is not None and schedule != planned:
        raise ValueError("Sealed timing order changed")
    return planned


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
            native_accepted_draft_tokens=sum(
                b["native_accepted_draft_tokens"] for b in record["metadata"]["trace"]
            ),
        )
        row.update(first_round(record))
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


def first_round(record):
    """Only the first native round has a shared conditioning prefix across arms.

    The seed is generated by the target before this round. Keep the complete
    prefix identity and both raw and stop-clipped acceptance; later call indices
    are never used as aligned mechanism observations.
    """
    import hashlib

    trace = record["metadata"]["trace"]
    if not trace:
        # EOS may be the target-generated seed, before any verification round.
        if len(record["response_ids"]) != 1 or record["finish_reason"] != "eos":
            raise ValueError("No first native round without immediate seed EOS")
        return dict(
            first_round_present=False,
            first_round_start_digest=None,
            first_round_accepted_prefix_tokens=0,
            first_round_raw_prefix_tokens=0,
            first_round_output_start=None,
        )
    block = trace[0]
    if block["output_start"] != 0 or block["anchor_token"] != record["response_ids"][0]:
        raise ValueError("First native round must follow the shared target seed")
    # Native output_start indexes the anchor in the response (seed=0), rather
    # than counting all prefix tokens. Include that anchor in the shared prefix.
    prefix = record["prompt_ids"] + record["response_ids"][: block["output_start"] + 1]
    digest = hashlib.sha256(
        json.dumps(prefix, separators=(",", ":")).encode()
    ).hexdigest()
    return dict(
        first_round_present=True,
        first_round_start_digest=digest,
        first_round_accepted_prefix_tokens=block["accepted_draft_tokens"],
        first_round_raw_prefix_tokens=block["raw_accepted_draft_tokens_before_stop"],
        first_round_output_start=block["output_start"],
    )


def qualification_receipt(rows, request_count):
    size = min(QUALIFICATION_REQUESTS, request_count)
    selected = [row for row in rows if row["panel_ordinal"] < size]
    expected = size * len(METHODS) * ENDPOINT_REPEATS
    slots = {(r["panel_ordinal"], r["method"], r["repeat"]) for r in selected}
    passed = len(selected) == len(slots) == expected and all(
        r.get("valid_measurement") is True and not r.get("errors") for r in selected
    )
    return dict(
        status="passed" if passed else "pending_or_failed",
        requests=size,
        generations=len(selected),
        planned_generations=expected,
        extra_generations=0,
        scope=f"First{size} input-selected requests within the{request_count * len(METHODS) * ENDPOINT_REPEATS}-completion budget; correctness only",
    )


def serving_protocol(planned, checkpoints):
    return dict(
        schedule=planned,
        checkpoints=checkpoints,
        capture=False,
        extra_warmup_generations=0,
        canonical_source="sealed_verified_references",
        count_trajectory_repeat=0,
        serving=SERVING,
        first8_check="included_in_main_budget",
        order_assignment="input_only_cyclic8_request_blocks",
    )


def _native_evaluate_stage(
    backend,
    requests,
    checkpoints,
    root,
    *,
    canonicals,
    dataset,
    event,
    schedule=None,
    boundary_check=None,
    production=True,
):
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = _native_validate_evaluation_inputs(
        requests,
        checkpoints,
        canonicals,
        dataset=dataset,
        event=event,
        schedule=schedule,
        production=production,
    )
    if production and boundary_check is None:
        raise ValueError(
            "Require storage and training/transfer quiescence check at every block boundary"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, grouped, checked = [], defaultdict(list), set()
    active_attempt, active_batch, physical_gpu = None, None, None
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
        serving_protocol(planned, checkpoints),
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
                if _native_boundary_errors(check):
                    raise RuntimeError(
                        "Failed evaluation boundary: "
                        + repr(_native_boundary_errors(check))
                    )
                if production:
                    observed = check.get("physical_gpu")
                    if (
                        not isinstance(observed, str)
                        or not observed
                        or (physical_gpu is not None and observed != physical_gpu)
                    ):
                        raise RuntimeError(
                            "Every timing block must use the same authenticated physical GPU"
                        )
                    physical_gpu = observed
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
            if group and row.get("native_accepted_draft_tokens") != group[0].get(
                "native_accepted_draft_tokens"
            ):
                errors.append("repeat_native_accepted_count_mismatch")
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
            write_json(
                root / "QUALIFICATION.json", qualification_receipt(rows, len(requests))
            )
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
        write_json(
            root / "QUALIFICATION.json", qualification_receipt(rows, len(requests))
        )
        write_json(root / "STATUS.json", progress)
    return progress


def _native_boundary_errors(check):
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


def _native_audit_stage(
    folder,
    requests,
    checkpoints,
    *,
    canonicals,
    dataset,
    event,
    schedule=None,
    evidence_root=None,
    production=True,
):
    """Reconstruct scalars from raw outputs and sidecars; never trust a validity flag."""
    folder = Path(folder)
    errors, validated, progress = [], [], {}
    try:
        checkpoints = {method: str(path) for method, path in checkpoints.items()}
        planned = _native_validate_evaluation_inputs(
            requests,
            checkpoints,
            canonicals,
            dataset=dataset,
            event=event,
            schedule=schedule,
            production=production,
        )
        protocol = json.loads((folder / "PROTOCOL.json").read_text())
        rows = [
            json.loads(line)
            for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()
        ]
        progress = json.loads((folder / "STATUS.json").read_text())
        if (
            protocol != serving_protocol(planned, checkpoints)
            or json.loads((folder / "REQUESTS.json").read_text()) != requests
            or json.loads((folder / "CANONICAL.json").read_text()) != canonicals
        ):
            errors.append("sealed_protocol_changed")
        gate = json.loads((folder / "QUALIFICATION.json").read_text())
        if (
            gate != qualification_receipt(rows, len(requests))
            or gate["status"] != "passed"
        ):
            errors.append("first8_correctness_gate")
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
            if checkpoints[slot["method"]] != current:
                expected_slots.append(slot)
                current = checkpoints[slot["method"]]
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
                errors.extend(_native_boundary_errors(row["check"]))
            if production:
                identities = [row["check"].get("physical_gpu") for row in boundaries]
                if (
                    not identities
                    or any(
                        not isinstance(value, str) or not value for value in identities
                    )
                    or len(set(identities)) != 1
                ):
                    errors.append("physical_gpu_changed_or_unbound")
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
            if group and row.get("native_accepted_draft_tokens") != group[0].get(
                "native_accepted_draft_tokens"
            ):
                invalid.append("repeat_native_accepted_count_mismatch")
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



evaluation_schedule = _native_evaluation_schedule
evaluate_stage = _native_evaluate_stage
audit_stage = _native_audit_stage
