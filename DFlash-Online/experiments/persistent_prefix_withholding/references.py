"""Authenticate retained target-greedy verifier references after panel sealing.

This is a CPU evidence check, not an independent target-only reproduction.
Installed runtime source parity and current model identities remain explicit
campaign gates; historical output availability never changes the sealed panel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.o_half_matched_lr.archive import full_trace_errors
from online_feedback.native import validate_generation


def file_identity(path):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def resolve_missing_requests(panel, missing, reused):
    """Bind the audit's identity-only rows to the unchanged 128-request panel.

    The audit intentionally does not duplicate prompts in MISSING_REFERENCES.
    Resolve the complete list on CPU, before allocating the reference backend.
    """
    from .data import token_digest
    from .evaluation import validate_references

    requests = panel["requests"]
    if len(requests) != 128 or len(reused) != 104 or len(missing) != 24:
        raise ValueError("Require exactly 104 reused and 24 missing references for 128 requests")
    by_id = {}
    source_ids = set()
    for ordinal, request in enumerate(requests):
        request_id, source_id = request.get("request_id"), request.get("source_id")
        prompt = request.get("prompt_ids")
        if (not request_id or not source_id or request_id in by_id or source_id in source_ids
                or request.get("panel_ordinal") != ordinal
                or not isinstance(prompt, list) or not prompt or len(prompt) + 2048 > 4096
                or any(type(token) is not int or token < 0 for token in prompt)
                or request.get("prompt_ids_sha256") != token_digest(prompt)
                or request.get("max_new_tokens") != 2048
                or request.get("stop_token_ids") != [151645, 151643]
                or request.get("collect_feedback") is not False):
            raise ValueError("Sealed reference panel has incomplete identity or changed generation settings")
        by_id[request_id] = request
        source_ids.add(source_id)
    resolved, missing_ids, missing_sources = [], set(), set()
    for entry in missing:
        request_id, source_id = entry.get("request_id"), entry.get("source_id")
        request = by_id.get(request_id)
        if (entry.get("role") != "panel" or request is None
                or source_id != request["source_id"]
                or request_id in missing_ids or source_id in missing_sources):
            raise ValueError("Missing reference identity, role or uniqueness differs from the sealed panel")
        # Do not silently ignore a stale, privately enriched execution manifest.
        for field in ("prompt_ids", "stop_token_ids", "panel_ordinal", "max_new_tokens", "collect_feedback"):
            if field in entry and entry[field] != request[field]:
                raise ValueError("Missing-reference fields contradict the sealed panel: " + field)
        missing_ids.add(request_id)
        missing_sources.add(source_id)
        resolved.append(deepcopy(request))
    reused_ids = set(reused)
    if reused_ids & missing_ids or reused_ids | missing_ids != set(by_id):
        raise ValueError("Reused and missing references must cover the fixed panel exactly once")
    for request_id, reference in reused.items():
        if reference.get("source_id") != by_id[request_id]["source_id"]:
            raise ValueError("Reused reference source identity differs from the sealed panel")
        content = {key: value for key, value in reference.items() if key != "binding_digest"}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if reference.get("binding_digest") != digest:
            raise ValueError("Reused canonical reference binding is invalid")
    validate_references([by_id[key] for key in reused], reused)
    return resolved


def load_missing_requests(plan):
    """Authenticate immutable inputs and reject any partial-generation retry."""
    values = {}
    for key in ("data_panel", "missing_references", "reused_references"):
        path = plan[key]
        if file_identity(path) != plan["input_identities"].get(path):
            raise ValueError("Reference input differs from its sealed identity: " + key)
        values[key] = json.loads(Path(path).read_text())
    requests = resolve_missing_requests(values["data_panel"], values["missing_references"],
                                        values["reused_references"])
    merged = json.loads(Path(plan["references"]).read_text())
    if merged != values["reused_references"]:
        raise ValueError("Reference destination must contain exactly the 104 authenticated reused references")
    return requests, merged


def record_errors(record, request):
    """Validate actual target predictions and the complete native commit path."""
    errors = full_trace_errors(record)
    meta = record["metadata"]
    capture = meta["capture"]
    if record["prompt_ids"] != request["prompt_ids"]:
        errors.append("prompt_identity")
    if record["stop_token_ids"] != request["stop_token_ids"] or record["max_new_tokens"] != 2048:
        errors.append("generation_limits")
    if (meta["block_size"] != 16 or meta["context_limit"] != 4096
            or meta["runtime"] != "pinned_SGLang_FA3"
            or meta["cache_policy"] != "idle_full_flush_before_every_request_and_publication"):
        errors.append("native_serving_contract")
    if (meta["feedback_capture_in_timed_request"] or capture["collect_features"]
            or capture["feature_bytes"] != 0):
        errors.append("feedback_capture_enabled")
    if (capture["status"] != "passed" or not capture["native_capture_validated"]
            or capture["label_binding_scope"] != "native_prefill_and_committed_verification"
            or capture["target_argmax_ids"] != record["response_ids"]
            or capture["covered_token_ids"] != record["prompt_ids"] + record["response_ids"][:-1]
            or capture["covered_rows"] != len(capture["covered_token_ids"])):
        errors.append("captured_target_argmax_binding")
    try:
        _, termination = validate_generation(record["prompt_ids"],
            record["prompt_ids"] + record["response_ids"], meta["trace"],
            max_new_tokens=2048, stop_tokens=request["stop_token_ids"])
        if termination != record["finish_reason"]:
            errors.append("termination")
    except RuntimeError as error:
        errors.append(str(error))
    calls = len(meta["trace"])
    progress = sum(block["committed_progress"] for block in meta["trace"])
    accepted = sum(block["native_accepted_draft_tokens"] for block in meta["trace"])
    if (meta["verification_calls"] != calls or meta["committed_progress"] != progress
            or progress != len(record["response_ids"]) - 1 or accepted != progress - calls):
        errors.append("native_count_identity")
    return errors


def canonical_record(request, record, source_path, source_identity, provenance):
    reference = dict(prompt_ids=request["prompt_ids"], response_ids=record["response_ids"],
        finish_reason=record["finish_reason"], verified=True,
        reference_kind="native_greedy_target_verifier", source_record_path=str(source_path),
        source_record_identity=source_identity, source_id=request["source_id"],
        native_checkpoint=record["checkpoint_id"], provenance=provenance)
    reference["binding_digest"] = hashlib.sha256(json.dumps(
        reference, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return reference


def audit_reused(workspace, data_directory, output):
    research = Path(workspace) / "research"
    data = Path(data_directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}

    def read(path):
        path = Path(path)
        files[str(path)] = file_identity(path)
        return json.loads(path.read_text())

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    seal = read(data / "SEAL.json")
    for name, identity in seal.items():
        if file_identity(data / name) != identity:
            raise ValueError("Sealed data changed: " + name)
    panel = read(data / "PANEL.json")["requests"]
    warmups = read(data / "WARMUPS.json")["requests"]
    original_root = research / "dflash_full_k_gsm8k_20261031/production_002/sealed"
    original = read(original_root / "ORIGINAL_MANIFEST.json")
    canonical = read(original_root / "CANONICALS.json")
    prior_audit = read(original_root / "REUSE_AUDIT.json")
    model_pin = read(original_root / "MODEL_PIN.json")
    runtime_pin = read(original_root / "RUNTIME_PIN.json")
    if prior_audit["status"] != "passed" or prior_audit["model_work"] != 0:
        raise ValueError("Prior independent native archive audit is not valid")
    original_run = research / "dflash_online_osd_rerun_20261023"
    original_seal = read(original_run / "sealed/SEAL.json")
    qualification_root = original_run / "first_event/events/event_001"
    replay = read(qualification_root / "NATIVE_REPLAY_CHECK.json")
    publication = read(qualification_root / "PUBLICATION_CHECK.json")
    if not (replay["same_input"] and replay["same_output"] and replay["reloaded_calls_equal"]
            and not replay["feedback_collected"] and publication["all_equal"]
            and publication["status"] == "passed"):
        raise ValueError("Prior native publication/output qualification failed")
    # Pin the actual historical source identities; runtime equality with today's
    # installed path is independently checked by the campaign before new work.
    for key in ("online_feedback/native.py", "online_feedback/sglang_backend.py",
                "online_feedback/sglang_hooks.py", "online_feedback/sglang_worker.py"):
        if original_seal["source_sha256"][key] != runtime_pin["source_sha256"][key]:
            raise ValueError("Historical native-source provenance changed: " + key)
    qualification_records = [read(qualification_root / f"qualification_{name}.json")
                             for name in ("frozen", "updated", "reloaded")]
    if len({tuple(record["response_ids"]) for record in qualification_records}) != 1:
        raise ValueError("Retained qualification output equality is not reproduced")
    for record in qualification_records:
        errors = record_errors(record, {"prompt_ids": record["prompt_ids"],
                                        "stop_token_ids": record["stop_token_ids"]})
        if errors:
            raise ValueError("Retained qualification trace errors: " + repr(errors))

    by_prompt = {tuple(row["prompt_ids"]): (index, row)
                 for index, row in enumerate(original["eval_requests"])}
    reused, reused_warmups, missing, checks = {}, {}, [], []
    source_root = original_run / "status_check_20261024/raw/campaign_001/evaluation"
    provenance = dict(target_pin=model_pin["target"], runtime_pin=runtime_pin,
        original_source_seal=files[str(original_run / "sealed/SEAL.json")],
        prior_archive_audit=files[str(original_root / "REUSE_AUDIT.json")],
        prior_qualification=files[str(qualification_root / "NATIVE_REPLAY_CHECK.json")],
        prior_publication=files[str(qualification_root / "PUBLICATION_CHECK.json")],
        greedy=True, temperature=0, block_size=16, precision="bfloat16", attention="fa3",
        context_limit=4096, max_new_tokens=2048, stop_token_ids=[151645, 151643],
        reasoning="off", cache_policy="idle_full_flush_before_every_request_and_publication",
        claim="Qualified native target-greedy verifier reference, not independent target-only AR")
    for role, requests, destination in (("panel", panel, reused), ("warmup", warmups, reused_warmups)):
        for request in requests:
            inherited = by_prompt.get(tuple(request["prompt_ids"]))
            if inherited is None:
                missing.append({"request_id": request["request_id"], "source_id": request["source_id"], "role": role})
                continue
            index, old_request = inherited
            old_ref = canonical[old_request["request_id"]]
            repeat_counts = []
            for repeat in range(3):
                path = source_root / f"{index:04d}_r{repeat}_F.json"
                raw = read(path)
                old_path = f"{prior_audit['original_run']}/campaign_001/evaluation/{path.name}"
                if files[str(path)] != prior_audit["files"][old_path]:
                    raise ValueError("Retained reference raw identity differs from prior audit")
                errors = record_errors(raw, request)
                if (raw["response_ids"] != old_ref["response_ids"]
                        or raw["finish_reason"] != old_ref["finish_reason"]
                        or raw["checkpoint_id"] != old_ref["F_checkpoint"]):
                    errors.append("retained_canonical_binding")
                checks.append(dict(request_id=request["request_id"], source_id=request["source_id"],
                    role=role, repeat=repeat, source_path=str(path), errors=errors,
                    native_blocks=len(raw["metadata"]["trace"])))
                if errors:
                    write("REUSE_CHECKS.json", checks)
                    raise ValueError("Invalid retained reference: " + request["source_id"] + repr(errors))
                repeat_counts.append((raw["metadata"]["verification_calls"],
                    sum(block["native_accepted_draft_tokens"] for block in raw["metadata"]["trace"])))
                if repeat == 0:
                    expected_path = f"{prior_audit['original_run']}/campaign_001/evaluation/{path.name}"
                    if old_ref["source_record_path"] != expected_path:
                        raise ValueError("Canonical source record is not original frozen repeat zero")
                    destination[request["request_id"]] = canonical_record(request, raw, path, files[str(path)],
                                                                         {"receipt": "REUSE_PROVENANCE.json"})
            if len(set(repeat_counts)) != 1:
                raise ValueError("Retained frozen repeat counts disagree")
    if any(row["role"] == "warmup" for row in missing):
        raise ValueError("Fixed warmups lack independently retained compatible references")
    write("REUSED_REFERENCES.json", reused)
    write("WARMUP_REFERENCES.json", reused_warmups)
    write("MISSING_REFERENCES.json", missing)
    write("REUSE_CHECKS.json", checks)
    write("REUSE_PROVENANCE.json", provenance)
    receipt = dict(status="RETAINED_REFERENCES_VALIDATED", reference_kind="native_greedy_target_verifier",
        panel_reused=len(reused), warmup_reused=len(reused_warmups), panel_missing=len(missing),
        checked_raw_records=len(checks), checked_native_blocks=sum(row["native_blocks"] for row in checks),
        invalid_records=0, extra_model_work=0, source_proof_status="historical_pins_bound; current_installed_source_gate_separate",
        files=files, no_historical_times_or_counts_in_new_arm_comparison=True)
    write("REUSE_RECEIPT.json", receipt)
    return {key: receipt[key] for key in ("status", "panel_reused", "warmup_reused", "panel_missing", "checked_raw_records", "checked_native_blocks")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_reused(args.workspace, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
