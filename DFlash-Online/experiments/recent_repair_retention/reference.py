"""Frozen K+6 measurements on the retained event-seven training presentations.

This preparation never constructs an optimizer. It naturally redraws the saved
schedule, projects eight blocks at a time, and emits detached slot scalars only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.lost_repair.training_probe import (
    _anchor_record, _rng_equal, _read_features, build_readonly_model,
    frozen_training_state, prepare_batch, validate_recovery,
)
from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def feature_paths(plan):
    if "feature_paths" in plan:
        return [Path(path) for path in plan["feature_paths"]]
    value = read(plan["feature_manifest"])
    if isinstance(value, list):
        return [Path(path) for path in value]
    if "feature_paths" in value:
        return [Path(path) for path in value["feature_paths"]]
    return [Path(row.get("feature_path", row.get("feature", {}).get("path")))
            for row in value["records"]]


def validate_schedule(schedule, record_ids):
    rows = schedule["records"]
    if schedule["passes"] != 2 or len(rows) != 256 or len(set(record_ids)) != 128:
        raise ValueError("Require the original 128-response two-pass schedule")
    for index, row in enumerate(rows):
        if (row["presentation_id"] != index or row["pass_index"] != index // 128
                or len(row["sampled_anchors"]) != 1):
            raise ValueError("Schedule presentation identity differs")
    for offset in (0, 128):
        if sorted(row["record_id"] for row in rows[offset:offset+128]) != sorted(record_ids):
            raise ValueError("Each original response must appear once in each pass")
    return rows


@torch.no_grad()
def slot_measurements(logits, labels, weights, *, anchors, keep, prompt_length,
                      request_id, presentation_id, pass_index, eos_token_ids=(),
                      candidates=None, expected_tokens=None):
    """Independent native head observations for eligible natural block starts."""
    if logits.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("Reference preparation requires native BF16 head and FP32 weights")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite frozen-parent logits")
    proposals = logits.argmax(-1)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    selected = first_error_mask(proposals, labels, weights, eos_token_ids)
    correct = selected & (proposals == labels)
    prefixes = correct.sum(-1)
    rows = []
    for local in range(labels.shape[1]):
        if not bool(keep[0, local]):
            continue
        start = int(anchors[0, local]) - prompt_length
        for slot in range(1, labels.shape[-1]):
            key = (str(request_id), start, slot)
            if candidates is not None and key not in candidates:
                continue
            label = int(labels[0, local, slot])
            native = logits[0, local, slot]
            competitors = native.clone()
            competitors[label] = -torch.inf
            rival = int(competitors.argmax())
            margin = float(native[label].float() - native[rival].float())
            expected = label if expected_tokens is None else expected_tokens[key]
            if expected != label:
                raise ValueError("Retained verifier label and teacher-packet label disagree")
            rows.append(dict(
                presentation_id=presentation_id, pass_index=pass_index,
                request_id=str(request_id), canonical_start=start,
                sequence_start=int(anchors[0, local]), local_anchor_ordinal=local,
                proposal_slot=slot, canonical_token=label, expected_canonical_token=expected,
                label_match=True, margin=margin, weight=float(weights[0, local, slot]),
                correct_prefix=int(prefixes[0, local]), valid=bool(valid[0, local, slot]),
                native_correct=bool(proposals[0, local, slot] == label),
                rival_token=rival, correct_tie=bool(margin == 0 and proposals[0, local, slot] == label),
            ))
    return rows


@torch.no_grad()
def observe_response(model, raw, binding, *, max_length, candidates, expected_tokens):
    """Preserve full-block attention and the production head's physical shapes."""
    device = next(model.parameters()).device
    batch = prepare_batch(raw, max_length, device)
    original = model._sample_anchor_positions
    sampled_rng = None

    def sample(*args, **kwargs):
        nonlocal sampled_rng
        result = original(*args, **kwargs)
        if _anchor_record(*result) != binding["sampled_anchors"][0]:
            raise ValueError("Frozen preparation drew different anchors or keep masks")
        sampled_rng = capture_rng()
        return result

    model._sample_anchor_positions = sample
    try:
        anchors, keep, hidden = model._forward_draft_blocks(
            batch["input_ids"], batch["hidden_states"], batch["loss_mask"],
            max_valid_anchors=raw["input_ids"].shape[1] - raw["prompt_length"] - 1,
        )
    finally:
        model._sample_anchor_positions = original
    labels, weights, _ = model._block_labels_and_weights(
        batch["input_ids"], batch["loss_mask"], anchors, keep,
    )
    hidden = hidden.reshape(1, anchors.shape[1], model.block_size, -1)
    rows, chunks, selected_weight = [], 0, 0.0
    for start in range(0, anchors.shape[1], 8):
        stop = start + 8
        logits = model.lm_head(hidden[:, start:stop]).to(model.loss_dtype)
        local_labels, local_weights = labels[:, start:stop], weights[:, start:stop]
        selected = first_error_mask(logits.argmax(-1), local_labels, local_weights, model.eos_token_ids)
        selected_weight += float((local_weights * selected).sum())
        observed = slot_measurements(
            logits, local_labels, local_weights, anchors=anchors[:, start:stop],
            keep=keep[:, start:stop], prompt_length=raw["prompt_length"],
            request_id=raw["record_id"], presentation_id=binding["presentation_id"],
            pass_index=binding["pass_index"], eos_token_ids=model.eos_token_ids,
            candidates=candidates, expected_tokens=expected_tokens,
        )
        for row in observed:
            row["anchor_ordinal"] = start + row.pop("local_anchor_ordinal")
        rows.extend(observed)
        chunks += 1
        del logits
    if sampled_rng is None or not _rng_equal(sampled_rng, capture_rng()):
        raise ValueError("Frozen forward consumed RNG beyond the retained anchor draw")
    return rows, dict(presentation_id=binding["presentation_id"],
        pass_index=binding["pass_index"], request_id=raw["record_id"],
        sampled_anchors=binding["sampled_anchors"], selected_weight=selected_weight,
        response_forwards=1, projection_chunks=chunks, backward_calls=0,
        optimizer_steps=0, emitted_slot_rows=len(rows))


def execute(plan_path, output):
    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import init_distributed
    from online_feedback.recipe import load_event_config
    import torch.distributed as dist

    plan, output = read(plan_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = feature_paths(plan)
    schedule = read(plan["schedule"])
    # Candidate labels are recovered from the private verifier trace before any
    # model work, rather than inferred from a future outcome or a parent argmax.
    candidate_rows = read(plan["candidate_slots"])
    if isinstance(candidate_rows, dict):
        candidate_rows = candidate_rows["slots"]
    expected = {(str(r["request_id"]), int(r["canonical_start"]), int(r["proposal_slot"])):
                int(r["canonical_token"]) for r in candidate_rows}
    if len(expected) != len(candidate_rows):
        raise ValueError("Duplicate candidate slot identity")
    status = dict(status="running", response_forwards=0, projection_chunks=0,
                  backward_calls=0, optimizer_steps=0, execution_commit=plan["execution_commit"])
    started = time.perf_counter()
    owned_dist = False
    try:
        config, _ = load_event_config(plan["origin_config"])
        os.environ["FSDP_SHARDING"] = config.training.fsdp_sharding
        if not dist.is_initialized():
            _bootstrap_single_process_env()
            init_distributed(timeout=config.training.dist_timeout, tp_size=1,
                             sp_ulysses_size=1, sp_ring_size=1)
            owned_dist = True
        recovery = torch.load(plan["origin_state"], map_location="cpu", weights_only=False, mmap=True)
        status["origin"] = validate_recovery(plan["origin_config"], recovery)
        model, cfg = build_readonly_model(plan["origin_config"], recovery)
        features = _read_features(paths, cfg.data.max_length)
        bindings = validate_schedule(schedule, [r["record_id"] for r in features])
        torch.cuda.reset_peak_memory_stats()
        ordinal = 0
        with frozen_training_state(model, recovery["rng"]):
            for pass_index in range(2):
                for index in torch.randperm(len(features)).tolist():
                    feature = features[index]
                    binding = bindings[ordinal]
                    if (feature["record_id"], pass_index) != (binding["record_id"], binding["pass_index"]):
                        raise ValueError("Frozen reference order differs from retained native schedule")
                    raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                    began = time.perf_counter()
                    rows, receipt = observe_response(model, raw, binding,
                        max_length=cfg.data.max_length, candidates=set(expected), expected_tokens=expected)
                    receipt["wall_s"] = time.perf_counter() - began
                    write_json(output / "presentations" / f"{ordinal:03d}.json",
                               dict(rows=rows, presentation=receipt))
                    ordinal += 1
                    status["response_forwards"] = ordinal
                    status["projection_chunks"] += receipt["projection_chunks"]
                    write_json(output / "STATUS.json", status)
        if ordinal != 256:
            raise ValueError("Frozen reference preparation did not cover all presentations")
        status.update(status="completed", caller_rng_and_modes_restored=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["wall_s"] = time.perf_counter() - started
        write_json(output / "STATUS.json", status)
        if owned_dist:
            dist.destroy_process_group()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(args.plan, args.output)
