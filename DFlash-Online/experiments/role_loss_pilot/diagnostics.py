"""Fixed canonical-context prefix diagnostics, isolated from all training decisions."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch

from online_feedback.first_error import valid_prefix_mask
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json
from online_feedback.vfd_scoring import (
    _scoring_runtime, _synchronize, _tensor_digest, native_eligible_anchors,
)

POLICY_VERSION = "vfd-roleloss-pilot-v1"
ANCHOR_HASH_TEMPLATE = "vfd-roleloss-pilot-v1|{dataset}|diagnostic-anchor|{source_id}|{absolute_position}"
DATASETS = ("gsm8k", "mathqa")
ARMS = ("KPLUS", "MIX", "ROLE")


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def select_anchors(loss_mask, dataset, source_id):
    """Hash choice, then native positional order; no observed prediction is used."""
    if dataset not in DATASETS:
        raise ValueError("Unknown diagnostic dataset")
    eligible = native_eligible_anchors(loss_mask)
    ranked = sorted(eligible, key=lambda position: (
        hashlib.sha256(ANCHOR_HASH_TEMPLATE.format(dataset=dataset, source_id=source_id,
            absolute_position=position).encode("utf-8")).hexdigest(), position))
    return sorted(ranked[:8])


def diagnostic_manifest(requests, canonicals, dataset):
    chosen = requests[:16]
    return dict(dataset=dataset, policy_version=POLICY_VERSION,
        request_ids=[r["request_id"] for r in chosen], source_ids=[r["source_id"] for r in chosen],
        anchor_cap=8, anchor_hash_template=ANCHOR_HASH_TEMPLATE,
        anchor_order="ascending_absolute_position_after_hash_selection",
        native_eligibility="adjacent_response_loss_mask_gt_0.5",
        canonical_context="full_prompt_plus_canonical_response_no_truncation")


def _read_reference(request, reference):
    """Load the exact retained native record, keeping old native request ordinals."""
    from experiments.o_half_matched_lr.archive import validate_record
    path = Path(reference.get("source_record", {}).get("path") or reference["source_record_path"])
    raw = json.loads(path.read_text())
    expected = reference.get("source_record")
    if expected and any(file_identity(path)[k] != expected[k] for k in ("bytes", "sha256")):
        raise ValueError("Retained diagnostic canonical record identity differs")
    native_path = Path(reference.get("source_native", {}).get("path") or raw["metadata"]["native_trace_path"])
    if json.loads(native_path.read_text()) != raw["metadata"]["capture"]:
        raise ValueError("Retained diagnostic native sidecar differs")
    native_expected = reference.get("source_native")
    if native_expected and any(file_identity(native_path)[k] != native_expected[k] for k in ("bytes", "sha256")):
        raise ValueError("Retained diagnostic native identity differs")
    errors = validate_record(raw, dict(request, panel_ordinal=raw["request_index"]),
        raw["checkpoint_id"], reference, label=raw["record_id"], session=raw["session_id"])
    if errors:
        raise ValueError("Invalid diagnostic canonical: " + repr(errors))
    return raw


def prepare_pool(backend, requests, canonicals, root, *, dataset, diagnostic_manifest):
    """Prepare each canonical once by the qualified native block-replay adapter.

    This counts response preparations separately from target prefill, native
    verification replay and terminal forwards. There are no new generations.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    expected = globals()["diagnostic_manifest"](requests, canonicals, dataset)
    if isinstance(diagnostic_manifest, (str, Path)):
        diagnostic_manifest = json.loads(Path(diagnostic_manifest).read_text())
    if diagnostic_manifest != expected or len(expected["request_ids"]) != 16:
        raise ValueError("Fixed diagnostic manifest differs from the first16 sealed panel inputs")
    pool = dict(status="running", dataset=dataset, manifest=expected, features=[],
        planned_preparations=16, attempted_preparations=0, completed_preparations=0,
        unknown_preparations=0, unattempted_preparations=16, new_generations=0,
        target_prefill_forwards=0, target_verification_replay_forwards=0, terminal_target_forwards=0,
        active_request=None, automatic_retry=False, training_use=False)
    started = time.perf_counter()
    try:
        with preserve_rng():
            for ordinal, request in enumerate(requests[:16]):
                source = _read_reference(request, canonicals[request["request_id"]])
                pool["attempted_preparations"] += 1
                pool["unknown_preparations"] = 1
                pool["active_request"] = request["request_id"]
                pool["unattempted_preparations"] -= 1
                write_json(root / "POOL.json", pool)
                replay = backend._call("roleloss_reconstruct", record=source,
                    outputfile=str((root / f"replay_{ordinal:02d}.json").absolute()))
                if replay["status"] != "passed" or replay["new_generations"] != 0:
                    raise ValueError("Canonical native reconstruction did not pass")
                pool["target_prefill_forwards"] += replay["prefill_forwards"]
                pool["target_verification_replay_forwards"] += replay["verification_forwards"]
                request_root = root / "requests" / f"{ordinal:02d}"
                prepared = backend.prepare_feedback(replay["record"], request_root / "loop/features" / (source["record_id"] + ".pt"))
                if prepared["status"] != "prepared":
                    raise ValueError("Fixed diagnostic input cannot be replaced when ineligible")
                path = prepared["feature_path"]
                raw = torch.load(path, map_location="cpu", weights_only=True)
                combined = request["prompt_ids"] + canonicals[request["request_id"]]["response_ids"]
                if raw["input_ids"].tolist() != [combined] or len(combined) > 4096:
                    raise ValueError("Diagnostic preparation changed full canonical context")
                anchors = select_anchors(raw["loss_mask"], dataset, request["source_id"])
                pool["terminal_target_forwards"] += 1
                pool["features"].append(dict(request_id=request["request_id"], source_id=request["source_id"],
                    path=str(path), identity=file_identity(path), anchor_ids=anchors,
                    context_length=len(combined), preparation=prepared, reconstruction=str(root / f"replay_{ordinal:02d}.json")))
                from experiments.kplus_full_gsm8k.storage import retire_redundant_features
                retirement = retire_redundant_features(replay["record"], prepared, root, request_root)
                pool["features"][-1]["redundant_retirement"] = retirement
                pool["completed_preparations"] += 1
                pool["unknown_preparations"] = 0
                pool["active_request"] = None
                write_json(root / "POOL.json", pool)
        pool["status"] = "completed"
    except BaseException as error:
        pool.update(status="failed", error=repr(error))
        raise
    finally:
        pool["preparation_wall_s"] = time.perf_counter() - started
        write_json(root / "POOL.json", pool)
    return pool


def snapshot_response(model, raw, *, dataset, source_id, request_id, anchor_ids, conditioning_id):
    """One ordinary draft block forward plus native chunked vocabulary projections."""
    inputs, features, mask = (raw[k] for k in ("input_ids", "hidden_states", "loss_mask"))
    if inputs.ndim != 2 or inputs.shape[0] != 1 or mask.shape != inputs.shape or features.shape[:2] != inputs.shape:
        raise ValueError("Diagnostic conditioning geometry differs")
    selected = select_anchors(mask, dataset, source_id)
    if list(anchor_ids) != selected:
        raise ValueError("Fixed diagnostic anchor identity changed")
    result = dict(dataset=dataset, request_id=request_id, source_id=source_id,
        policy_version=POLICY_VERSION, conditioning_id=conditioning_id, anchor_ids=selected,
        input_ids_sha256=_tensor_digest(inputs), loss_mask_sha256=_tensor_digest(mask),
        context_length=inputs.shape[1], block_size=model.block_size, eos_token_ids=list(model.eos_token_ids),
        evaluation_mode=True, no_backward=True, blocks=[],
        costs=dict(response_forwards=0, draft_forwards=0, projection_calls=0, forward_s=0., projection_s=0.))
    started = time.perf_counter()
    if not selected:
        result["costs"]["total_s"] = time.perf_counter() - started
        return result
    device = next(model.draft_model.parameters()).device
    inputs, features, mask = (x.to(device) for x in (inputs, features, mask))
    anchors = torch.tensor([selected], dtype=torch.long, device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    with _scoring_runtime(model, anchors, keep):
        labels, weights, _ = model._block_labels_and_weights(inputs, mask, anchors, keep)
        valid = valid_prefix_mask(labels, weights, model.eos_token_ids)
        _synchronize(features)
        forward_started = time.perf_counter()
        actual, actual_keep, hidden = model._forward_draft_blocks(inputs, features, mask)
        _synchronize(hidden)
        result["costs"]["forward_s"] = time.perf_counter() - forward_started
        if not torch.equal(actual, anchors) or not torch.equal(actual_keep, keep):
            raise ValueError("Author forward changed diagnostic anchors")
        result["costs"].update(response_forwards=1, draft_forwards=1)
        hidden = hidden.reshape(1, len(selected), model.block_size, -1)
        projection_started = time.perf_counter()
        chunk = int(model.objective_chunk_blocks)
        if chunk < 1:
            raise ValueError("Native projection chunk size must be positive")
        for start in range(0, len(selected), chunk):
            stop = min(start + chunk, len(selected))
            logits = model.lm_head(hidden[:, start:stop])
            if logits.dtype != features.dtype or not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite or changed-precision diagnostic logits")
            predictions = logits.argmax(-1)[0].cpu().tolist()
            result["costs"]["projection_calls"] += 1
            for index in range(start, stop):
                result["blocks"].append(dict(anchor_id=selected[index], labels=labels[0,index].cpu().tolist(),
                    valid=valid[0,index].cpu().tolist(), weights=weights[0,index].cpu().tolist(),
                    predictions=predictions[index-start]))
            del logits
        _synchronize(hidden)
        result["costs"]["projection_s"] = time.perf_counter() - projection_started
    result["costs"]["total_s"] = time.perf_counter() - started
    return result


def snapshot_panel(trainer, pool, root, *, dataset, arm, event):
    from online_feedback.rng import capture_rng
    from experiments.ok_continuation.resume import differences
    if (arm, event) != ("INITIAL", 0) and (arm not in ARMS or event not in (2, 4)):
        raise ValueError("Unplanned fixed-context snapshot")
    if pool["status"] != "completed" or pool["dataset"] != dataset or len(pool["features"]) != 16:
        raise ValueError("Require complete immutable diagnostic feature pool")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", dataset=dataset, arm=arm, event=event, snapshots={},
        attempted_responses=0, returned_responses=0, unattempted_responses=16,
        unknown_responses=0, draft_forwards=0, projection_calls=0, active_request=None,
        training_use=False, automatic_retry=False)
    before_rng = capture_rng()
    model = trainer.bundle.model
    modes = [(module, module.training) for module in model.modules()]
    started = time.perf_counter()
    try:
        with preserve_rng():
            for feature in pool["features"]:
                if file_identity(feature["path"]) != feature["identity"]:
                    raise ValueError("Shared diagnostic target feature identity changed")
                result["attempted_responses"] += 1
                result["unknown_responses"] = 1
                result["unattempted_responses"] -= 1
                result["active_request"] = feature["request_id"]
                write_json(root / "RESULT.json", result)
                raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                snapshot = snapshot_response(model, raw, dataset=dataset, source_id=feature["source_id"],
                    request_id=feature["request_id"], anchor_ids=feature["anchor_ids"],
                    conditioning_id=feature["identity"]["sha256"])
                result["snapshots"][feature["request_id"]] = snapshot
                result["returned_responses"] += 1
                result["unknown_responses"] = 0
                result["active_request"] = None
                result["draft_forwards"] += snapshot["costs"]["draft_forwards"]
                result["projection_calls"] += snapshot["costs"]["projection_calls"]
                write_json(root / "RESULT.json", result)
        if differences(before_rng, capture_rng()) or any(module.training != mode for module,mode in modes):
            raise ValueError("Diagnostic changed RNG or model mode")
        result.update(status="completed", rng_restored=True, model_mode_restored=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        write_json(root / "RESULT.json", result)
    return result


def prefix_length(block):
    if not (len(block["labels"]) == len(block["predictions"]) == len(block["valid"]) == len(block["weights"])) or block["valid"][0]:
        raise ValueError("Malformed diagnostic proposal block")
    ended = False
    for valid in block["valid"][1:]:
        if ended and valid:
            raise ValueError("Diagnostic validity reopens after EOS/gap")
        ended |= not valid
    length = 0
    for label, prediction, valid in zip(block["labels"][1:], block["predictions"][1:], block["valid"][1:]):
        if not valid or label != prediction:
            break
        length += 1
    return length


def compare_snapshots(initial, event2, event4):
    """Useful repair means the original error and its entire earlier prefix match."""
    keys = ("dataset", "request_id", "source_id", "policy_version", "conditioning_id", "anchor_ids",
            "input_ids_sha256", "loss_mask_sha256", "context_length", "block_size", "eos_token_ids")
    if any(initial[k] != other[k] for other in (event2,event4) for k in keys):
        raise ValueError("Fixed diagnostic conditioning differs between checkpoints")
    if any(len(s["blocks"]) != len(initial["anchor_ids"]) for s in (initial,event2,event4)):
        raise ValueError("Diagnostic block count differs")
    rows = []
    for b0,b2,b4 in zip(initial["blocks"],event2["blocks"],event4["blocks"]):
        if any(b0[k] != b[k] for b in (b2,b4) for k in ("anchor_id","labels","valid","weights")):
            raise ValueError("Diagnostic labels, validity or anchors changed")
        n0,n2,n4 = map(prefix_length, (b0,b2,b4))
        available = sum(b0["valid"])
        error = n0 < available
        gain2,gain4 = max(n2-n0,0),max(n4-n0,0)
        rows.append(dict(anchor_id=b0["anchor_id"], valid_proposals=available,
            initial_prefix=n0,event2_prefix=n2,event4_prefix=n4, original_first_error=error,
            event2_broken_prefix=n2<n0,event4_broken_prefix=n4<n0,
            event2_repair_with_prefix_intact=error and n2>n0,
            event4_repair_with_prefix_intact=error and n4>n0,
            event2_gained_prefix_tokens=gain2,event4_gained_prefix_tokens=gain4,
            repair_persists=error and n2>n0 and n4>n0,
            event2_gain_fully_persists=gain2>0 and n4>=n2,
            persisted_gain_tokens=min(gain2,gain4), lost_gain_tokens=max(gain2-gain4,0)))
    fields = [k for k in rows[0] if k != "anchor_id"] if rows else []
    return dict(request_id=initial["request_id"], anchors=rows,
                totals={k:sum(row[k] for row in rows) for k in fields}, anchor_count=len(rows))


def summarize_diagnostics(initial, event2, event4):
    for receipt in (initial,event2,event4):
        if (receipt["status"] != "completed" or receipt["returned_responses"] != 16
                or receipt["unknown_responses"] or receipt["unattempted_responses"]
                or not receipt["rng_restored"] or not receipt["model_mode_restored"]):
            raise ValueError("Incomplete or nonisolated diagnostics")
    if (initial["arm"], initial["event"]) != ("INITIAL",0) or event2["event"] != 2 or event4["event"] != 4 or event2["arm"] != event4["arm"]:
        raise ValueError("Wrong diagnostic trajectory/checkpoint identities")
    if set(initial["snapshots"]) != set(event2["snapshots"]) or set(initial["snapshots"]) != set(event4["snapshots"]):
        raise ValueError("Diagnostic requests changed")
    rows = [compare_snapshots(initial["snapshots"][key], event2["snapshots"][key], event4["snapshots"][key]) for key in initial["snapshots"]]
    fields = set().union(*(r["totals"] for r in rows))
    total = {k:sum(r["totals"].get(k,0) for r in rows) for k in sorted(fields)}
    return dict(status="passed", dataset=event2["dataset"], arm=event2["arm"], requests=16,
        anchors=sum(r["anchor_count"] for r in rows), totals=total, per_request=rows,
        scope="Local canonical-context changes; no speedup estimate or isolated causal claim.")
