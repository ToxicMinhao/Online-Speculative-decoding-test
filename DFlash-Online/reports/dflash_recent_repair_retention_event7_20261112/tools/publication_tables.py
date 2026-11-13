#!/usr/bin/env python3
"""Add redacted provenance tables without importing or changing experiment code.

This is post-run metadata publication, not another model execution or numerical
qualification. It reads scalar receipts and the sealed original anchor schedule.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    content = Path(path).read_bytes()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def canonical_digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def table(path, rows):
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pick(value, names):
    return {key: value[key] for key in names if key in value
            and (value[key] is None or isinstance(value[key], (str, bool, int, float)))}


def resolve(path, root, original_root, retained_root=None):
    """Only exact paths/mappings, never a basename search across experiments."""
    path = Path(path)
    candidates = []
    if path.is_relative_to(Path(original_root)):
        candidates.append(root / path.relative_to(original_root))
    if retained_root is not None and path.is_absolute():
        candidates.append(Path(retained_root) / path.relative_to("/"))
    candidates.append(path)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def schedule_tables(schedule, requests):
    """Publish positions/keep flags, not the tokens stored at those positions."""
    source = {row["request_id"]: row for row in requests}
    records = schedule["records"]
    if len(source) != len(requests) or schedule["passes"] != 2:
        raise ValueError("Invalid sealed response population or pass count")
    if len(records) != 2 * len(source) or schedule["response_presentations"] != len(records):
        raise ValueError("Sealed presentation count differs from the two-pass schedule")
    order_rows, anchor_rows, coverage = [], [], Counter()
    for ordinal, row in enumerate(records):
        request_id = row["record_id"]
        pass_index = int(row["pass_index"])
        if row["presentation_id"] != ordinal or pass_index != ordinal // len(source) or request_id not in source:
            raise ValueError("Presentation identity/order differs from the original sealed schedule")
        coverage[pass_index, request_id] += 1
        prompt_length = len(source[request_id]["prompt_ids"])
        begin = len(anchor_rows)
        for draw, sampled in enumerate(row["sampled_anchors"]):
            if len(sampled["positions"]) != 1 or len(sampled["keep_mask"]) != 1:
                raise ValueError("Expected the qualified one-response sampler shape")
            positions, keep = sampled["positions"][0], sampled["keep_mask"][0]
            if len(positions) != len(keep):
                raise ValueError("Anchor positions and keep-mask length differ")
            for anchor, (position, kept) in enumerate(zip(positions, keep)):
                if type(position) is not int or type(kept) is not bool:
                    raise ValueError("Malformed anchor offset/keep value")
                anchor_rows.append(dict(request_id=request_id, presentation_id=ordinal,
                    pass_index=pass_index, sampler_draw=draw, anchor_ordinal=anchor,
                    sequence_offset=position, canonical_response_offset=position-prompt_length,
                    kept=kept))
        selected = anchor_rows[begin:]
        order_rows.append(dict(request_id=request_id, presentation_id=ordinal, pass_index=pass_index,
            event_local_window=ordinal // 2 + 1, accumulation_position=ordinal % 2,
            global_adam_step_for_window=769 + ordinal // 2,
            prompt_length=prompt_length, sampler_draws=len(row["sampled_anchors"]),
            anchor_positions=len(selected), kept_anchors=sum(r["kept"] for r in selected),
            sampled_anchor_payload_sha256=canonical_digest(row["sampled_anchors"])))
    if coverage != Counter({(p, rid):1 for p in (0, 1) for rid in source}):
        raise ValueError("Each response must appear exactly once per pass")
    return order_rows, anchor_rows


def input_tables(plan):
    requests = plan["original_plan"]["requests"]
    rows = []
    for ordinal, row in enumerate(requests):
        rows.append(dict(training_pool_ordinal=ordinal, **pick(row, (
            "request_id", "dataset_id", "dataset_index", "request_index", "event_id", "max_new_tokens")),
            prompt_length=len(row["prompt_ids"]),
            prompt_tokens_canonical_json_sha256=canonical_digest(row["prompt_ids"])))
    inputs = []
    labels = {plan[name]:name for name in ("origin_state", "origin_config", "feature_manifest", "schedule",
        "candidate_slots", "all_starts", "native_weights", "panel_manifest", "timing_schedule") if name in plan}
    for ordinal, (path, pinned) in enumerate(sorted(plan["input_identities"].items())):
        source = Path(path)
        request_id = source.parent.name if source.name == "features.pt" else ""
        role = "teacher_features" if request_id else labels.get(path, "sealed_dependency")
        inputs.append(dict(artifact_id=f"input_{ordinal:04d}", role=role, request_id=request_id,
                          **pick(pinned, ("bytes", "sha256"))))
    checkpoints = []
    for name, checkpoint in plan["checkpoints"].items():
        for kind in ("weights_identity", "config_identity"):
            checkpoints.append(dict(checkpoint=name, artifact=kind, **pick(checkpoint[kind], ("bytes", "sha256"))))
    if plan["origin_state"] in plan["input_identities"]:
        checkpoints.append(dict(checkpoint="K6", artifact="complete_recovery_state",
            **pick(plan["input_identities"][plan["origin_state"]], ("bytes", "sha256"))))
    return rows, inputs, checkpoints


def comparison_summary(value, prefix):
    if not isinstance(value, dict):
        return {prefix + "_exact_equal": None}
    result = {prefix+"_"+key: value.get(key) for key in (
        "exact_equal", "compared_tensors", "compared_coordinates")}
    result[prefix+"_tensor_failure_count"] = len(value.get("tensor_differences", []))
    result[prefix+"_metadata_failure_count"] = len(value.get("metadata_differences", []))
    return result


def qualification_rows(root, plan, retained_root=None, training_commit=None):
    result = []
    training_commit=training_commit or plan["execution_commit"]
    for case in ("NATIVE", "DISABLED", "ZERO", "ACTIVE"):
        original = Path(plan["qualification_root"]) / case / "RESULT.json"
        path = resolve(original, root, plan["root"], retained_root)
        if path is None:
            result.append(dict(case=case, status="unavailable", execution_commit=training_commit))
            continue
        receipt = load(path)
        if receipt.get("execution_commit") != training_commit:
            raise ValueError("Qualification execution commit differs from sealed experiment")
        row = dict(case=case, **pick(receipt, ("status", "execution_commit", "wall_s",
            "peak_allocated_bytes", "peak_reserved_bytes", "finite", "optimizer_steps")),
            receipt_sha256=identity(path)["sha256"],
            ordinary_response_observations=len(receipt.get("observations", [])))
        for name in ("exact_native", "exact_selections", "active_restored", "long_restored"):
            row.update(comparison_summary(receipt.get(name), name))
        for phase in ("attempts", "completed", "unknown"):
            for name in ("response_forwards", "response_backwards", "optimizer_steps"):
                row[phase+"_"+name] = receipt.get("work", {}).get(phase, {}).get(name)
                row["startup_"+phase+"_"+name] = receipt.get("startup", {}).get("work", {}).get(phase, {}).get(name)
        startup_exact = path.parent / "startup/EXACT_RESTORATION.json"
        row.update(comparison_summary(load(startup_exact) if startup_exact.exists() else None, "startup_restored"))
        for component in ("active_fixture", "long_response", "startup"):
            for key in ("response_forwards", "response_backwards", "optimizer_steps", "wall_s",
                        "sequence_tokens", "peak_allocated_bytes", "peak_reserved_bytes",
                        "preceding_sampler_only_draws", "actual_presentation_id", "nonzero_draft_gradient_tensors",
                        "loss", "canonical_logit_gradient", "rival_logit_gradient"):
                value = receipt.get(component, {}).get(key)
                if value is None or isinstance(value, (int,float,bool)):
                    row[component+"_"+key] = value
        for key, value in receipt.get("physical_work", {}).items():
            if key in ("response_forwards", "draft_forwards", "vocabulary_head_calls", "layer_forwards"):
                row["physical_"+key] = value
        result.append(row)
    return result


def publish(root, output, *, schedule_path=None, retained_root=None):
    root, output = Path(root), Path(output)
    plan_path = next((path for path in (root/"TRAINING_PLAN.json", root/"sealed/PLAN.json", root/"PLAN.json")
                      if path.exists()), None)
    if plan_path is None:
        raise FileNotFoundError("No retained sealed plan")
    plan = load(plan_path)
    continuation=load(root/"CONTINUATION.json") if (root/"CONTINUATION.json").exists() else {}
    training_commit=continuation.get("training_execution_commit",plan["execution_commit"])
    evaluation_commit=continuation.get("evaluation_execution_commit",plan["execution_commit"])
    schedule_path = Path(schedule_path) if schedule_path else resolve(plan["schedule"],root,plan["root"],retained_root)
    if schedule_path is None or not schedule_path.is_file():
        raise FileNotFoundError("Original sealed SCHEDULE.json unavailable; supply --schedule with its exact retained copy")
    expected = plan["input_identities"].get(plan["schedule"])
    if expected is None or identity(schedule_path) != expected:
        raise ValueError("Original schedule does not match the sealed file identity")
    schedule = load(schedule_path)
    request_rows, inputs, checkpoint_rows = input_tables(plan)
    order_rows, anchors = schedule_tables(schedule,plan["original_plan"]["requests"])
    qualifications = qualification_rows(root,plan,retained_root,training_commit)
    for arm in ("NATIVE", "RECENT", "CONTROL"):
        path = root/"training"/arm/"RESULT.json"
        if path.exists():
            endpoint = load(path)
            for kind in ("weights_identity", "state_identity"):
                if kind in endpoint:
                    checkpoint_rows.append(dict(checkpoint=arm, artifact=kind, **pick(endpoint[kind],("bytes","sha256"))))
    output.mkdir(parents=True,exist_ok=True)
    tables = {"training_input_ids.csv": request_rows, "sealed_input_identities.csv": inputs,
        "parent_and_endpoint_identities.csv": checkpoint_rows, "training_response_order.csv": order_rows,
        "training_anchor_bindings.csv": anchors, "qualification_checks.csv": qualifications}
    for name, values in tables.items():
        table(output/name,[dict(row,execution_commit=row.get("execution_commit",training_commit),
            training_execution_commit=training_commit,evaluation_execution_commit=evaluation_commit) for row in values])
    metadata = dict(schema="retention-event7-publication-supplement-v1", execution_commit=plan["execution_commit"],
        training_execution_commit=training_commit,evaluation_execution_commit=evaluation_commit,
        publication_helper_sha256=identity(__file__)["sha256"], scientific_code_modified=False,model_operations=0,
        plan_identity=identity(plan_path), original_schedule_identity=expected,
        response_count=len(request_rows), response_presentations=len(order_rows), anchor_occurrences=len(anchors),
        kept_anchor_occurrences=sum(r["kept"] for r in anchors),
        scope="Request/source identifiers, offsets, hashes and scalar receipts only; no prompt/response/token arrays or tensors.",
        hash_definition="Derived token/sampler hashes use UTF8 json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False); sealed artifact hashes use raw bytes.",
        authorship="Same-team publication helper; author also contributed objective/report code. This is not an independent native replication.",
        qualification_counter_scope="Work-ledger attempts/completions/unknown counts are observed; no tensors are opened to reconstruct counters.",
        output_identities={name:identity(output/name) for name in tables})
    save(output/"PUBLICATION_METADATA.json",metadata)
    (output/"PUBLICATION_METADATA.md").write_text("## Retained input and qualification provenance\n\n"
        f"Training execution commit: `{training_commit}`; evaluation execution commit: `{evaluation_commit}`. "
        "This post-run helper performs no model operations and changes no scientific code.\n\n"
        f"The original schedule identity is verified before publishing {len(order_rows)} response presentations and {len(anchors)} scalar anchor offsets. "
        "Keep flags distinguish sampled positions from eligible retained blocks. Source text, tokens and tensors remain private. "
        "Parent K5/K6, K6 full recovery, input dependencies and completed endpoint identities are recorded separately.\n\n"
        "Qualification tables distinguish native/disabled/zero equality, active-fixture and long-response restoration, "
        "observed work-ledger counts, physical calls and peak memory. Missing receipts remain unavailable. "
        "Reported startup fields are taken from receipts; unavailable counters are not inferred.\n\n"
        "Cost scope: this event reuses PR42 K6 feedback/teacher packets and K5/K6 label trajectories. "
        "The new-work timers do not newly measure feedback collection, teacher preparation or the historical "
        "checkpoint replays used to create those labels. Future events need those labels and must account for "
        "their replay cost; the reused work is not free. Full online deployment costs and training-inclusive "
        "speedup are therefore not established.\n\n"
        + metadata["authorship"] + "\n")
    return metadata


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--schedule",type=Path,help="Exact retained original schedule when outside the downloaded root")
    parser.add_argument("--retained-root",type=Path,help="Optional archive preserving the original absolute directory tree")
    args=parser.parse_args()
    result=publish(args.root,args.output,schedule_path=args.schedule,retained_root=args.retained_root)
    print(json.dumps(dict(execution_commit=result["execution_commit"],presentations=result["response_presentations"],
                          anchors=result["anchor_occurrences"])))
