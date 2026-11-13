"""Score one immutable step-896 endpoint on the original training presentations.

Run NATIVE7 and RECENT7 in separate processes. The original K+6 configuration
and RNG define the geometry; only the draft's installed endpoint weights change.
No optimizer, backward pass, teacher projection, or generation is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from experiments.lost_repair.training_probe import (
    _read_features, _rng_equal, build_readonly_model, frozen_training_state, validate_recovery,
)
from experiments.recent_repair_retention.reference import (
    feature_paths, observe_response, read, validate_schedule,
)
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json


ENDPOINT_DIGESTS = {
    "NATIVE7": "30123d8de0bab8462a7afb77669079a65f62e771437232dcb0cba21413bed2cb",
    "RECENT7": "d873ae4455f355d8bb11bfbe771071519044155b42db051feb0d28d585eaacc9",
}


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify_inputs(plan):
    """Authenticate the inherited recipe, RNG, schedule and every teacher packet."""
    required = {str(Path(plan[name])) for name in ("origin_config", "origin_state", "schedule")}
    required.update(str(path) for path in feature_paths(plan))
    identities = plan["input_identities"]
    if not required.issubset(identities):
        raise ValueError("Missing bound input identity for frozen replay")
    for path, expected in identities.items():
        actual = file_identity(path)
        if any(actual[name] != expected[name] for name in ("bytes", "sha256")):
            raise ValueError(f"Retained replay input identity differs: {path}")
    occurrence_identity = file_identity(plan["sealed_occurrences"])
    if any(occurrence_identity[name] != plan["sealed_occurrences_identity"][name]
           for name in ("bytes", "sha256")):
        raise ValueError("Sealed occurrence identity differs")
    return dict(files=len(identities), all_identities_match=True,
                sealed_occurrences=occurrence_identity)


def slot_key(row):
    return (str(row["request_id"]), int(row["canonical_start"]), int(row["proposal_slot"]))


def occurrence_key(row):
    return (int(row["presentation_id"]), *slot_key(row))


def index_occurrences(rows, *, enforce_population=True):
    """Keep each pass's eligibility, including the slot eligible only once."""
    indexed = defaultdict(dict)
    multiplicities = Counter()
    for row in rows:
        presentation = int(row["presentation_id"])
        key = slot_key(row)
        if key in indexed[presentation]:
            raise ValueError("Duplicate sealed occurrence")
        if int(row["pass_index"]) != presentation // 128:
            raise ValueError("Occurrence pass differs from presentation")
        if not 1 <= key[2] < 16:
            raise ValueError("Sealed tag must be a proposal, excluding the seed")
        margin, floor = float(row["parent_margin"]), float(row["floor"])
        if not math.isfinite(margin) or margin < 0 or floor != .5 * min(margin, 1):
            raise ValueError("Original parent margin and floor disagree")
        indexed[presentation][key] = row
        multiplicities[key] += 1
    if enforce_population:
        checks = (
            len(rows) == 1629, len(multiplicities) == 815,
            len({key[0] for key in multiplicities}) == 128,
            Counter(multiplicities.values()) == Counter({2: 814, 1: 1}),
            Counter(int(row["pass_index"]) for row in rows) == Counter({0: 814, 1: 815}),
        )
        if not all(checks):
            raise ValueError("Sealed 128-request/815-slot/1629-occurrence population differs")
        sentinel = [r for r in rows if slot_key(r) == ("train_00768", 3, 7)]
        if len(sentinel) != 2 or any(float(r["parent_margin"]) != .75 or float(r["floor"]) != .375 for r in sentinel):
            raise ValueError("Known train_00768 parent-margin binding differs")
    return dict(indexed)


def verify_endpoint(draft, endpoint, expected_sha256):
    """Compare every installed draft tensor with the authenticated export."""
    identity = file_identity(endpoint)
    if identity["sha256"] != expected_sha256:
        raise ValueError("Endpoint file identity differs")
    state = draft.state_dict()
    coordinates, dtypes = 0, Counter()
    with safe_open(str(endpoint), framework="pt", device="cpu") as saved:
        if set(state) != set(saved.keys()):
            raise ValueError("Installed draft and export tensor names differ")
        for name, actual in state.items():
            expected = saved.get_tensor(name)
            actual = actual.detach().cpu()
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or not torch.equal(actual, expected)):
                raise ValueError(f"Installed endpoint tensor differs: {name}")
            coordinates += actual.numel()
            dtypes[str(actual.dtype)] += actual.numel()
    return dict(**identity, tensors=len(state), coordinates=coordinates,
                dtypes=dict(dtypes), exact_export_equality=True)


def shared_piece_identities(model):
    """Hash frozen head/embedding in bounded slices, without copying full modules."""
    result = {}
    for component in ("embed_tokens", "lm_head"):
        module = getattr(model, component)
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError(f"Shared target component is not frozen: {component}")
        digest = hashlib.sha256()
        coordinates = 0
        for name, tensor in module.state_dict().items():
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            flat = tensor.detach().reshape(-1)
            for chunk in flat.split(8 * 1024 * 1024):
                digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            coordinates += tensor.numel()
        result[component] = dict(sha256=digest.hexdigest(), coordinates=coordinates)
    return result


def verify_runtime_model(model):
    if (model.block_size != 16 or model.objective_chunk_blocks != 8
            or model.num_anchors != 512 or model.loss_dtype != torch.bfloat16
            or model.attention_backend != "flex_attention"):
        raise ValueError("Native BF16/Flex/eight-block scoring configuration differs")
    if any(not module.training for module in model.modules()):
        raise ValueError("Frozen scoring must retain ordinary training mode")
    if any(p.dtype != torch.bfloat16 for p in model.draft_model.parameters()):
        raise ValueError("Draft parameter precision differs from BF16")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen scoring found a model gradient")
    return dict(all_modules_training=True, draft_dtype="torch.bfloat16",
                loss_dtype="torch.bfloat16", block_size=16, head_chunk_blocks=8,
                num_anchors=512, attention_backend="flex_attention", model_gradients=0)


def attach_sealed_rows(observed, sealed, arm):
    """Validate the joined tag at every outcome, including wrong/unreachable tags."""
    seen, result = set(), []
    for observation in observed:
        key = slot_key(observation)
        if key in seen or key not in sealed:
            raise ValueError("Observed a duplicate or unsealed occurrence")
        seen.add(key)
        binding = sealed[key]
        for field in ("presentation_id", "pass_index", "anchor_ordinal", "canonical_token"):
            if int(observation[field]) != int(binding[field]):
                raise ValueError(f"Observed occurrence binding differs: {field}")
        if not observation["valid"] or not observation["label_match"]:
            raise ValueError("Sealed verifier-approved tag became invalid")
        if float(observation["weight"]) != float(binding["weight"]):
            raise ValueError("Position weight differs from the retained training layout")
        prefix, slot = int(observation["correct_prefix"]), key[2]
        outcome = ("earlier_break" if prefix < slot - 1 else
                   "first_mismatch_at_j" if prefix == slot - 1 else "accepts_through")
        result.append(dict(observation, arm=arm, parent_margin=float(binding["parent_margin"]),
                           floor=float(binding["floor"]), outcome=outcome,
                           accepts_through=prefix >= slot,
                           floor_breach=float(observation["margin"]) < float(binding["floor"]),
                           wrong_at_tie=observation["margin"] == 0 and not observation["native_correct"]))
    if seen != set(sealed):
        raise ValueError("Missing sealed occurrences; no endpoint-dependent filtering is allowed")
    return result


def score_presentations(model, features, bindings, occurrences, origin_rng, *,
                        max_length, arm, output, status, synchronize=None, timer=None):
    """One natural sampling traversal; persist every completed presentation."""
    synchronize = synchronize or (lambda: None)
    timer = timer or time.perf_counter
    caller_rng = capture_rng()
    with frozen_training_state(model, origin_rng):
        for pass_index in range(2):
            for index in torch.randperm(len(features)).tolist():
                ordinal = status["response_forwards"]
                feature, binding = features[index], bindings[ordinal]
                if (feature["record_id"], pass_index) != (binding["record_id"], binding["pass_index"]):
                    raise ValueError("Naturally drawn response order differs from retained schedule")
                raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                sealed = occurrences.get(ordinal, {})
                expected = {key: int(row["canonical_token"]) for key, row in sealed.items()}
                status.update(current_presentation=ordinal, response_forwards_attempted=ordinal + 1)
                write_json(output / "STATUS.json", status)
                synchronize()
                began = timer()
                rows, receipt = observe_response(
                    model, raw, binding, max_length=max_length,
                    candidates=set(sealed), expected_tokens=expected,
                )
                synchronize()
                receipt["synchronized_forward_wall_s"] = timer() - began
                # Preserve completed raw observations before checking membership.
                write_json(output / "raw_presentations" / f"{ordinal:03d}.json",
                           dict(rows=rows, presentation=receipt))
                # The forward has happened even if a later binding check fails.
                status["response_forwards"] += 1
                status["projection_chunks"] += receipt["projection_chunks"]
                status["synchronized_forward_wall_s"] += receipt["synchronized_forward_wall_s"]
                write_json(output / "STATUS.json", status)
                rows = attach_sealed_rows(rows, sealed, arm)
                write_json(output / "presentations" / f"{ordinal:03d}.json",
                           dict(rows=rows, presentation=receipt))
                status["emitted_occurrences"] += len(rows)
                write_json(output / "STATUS.json", status)
    if not _rng_equal(caller_rng, capture_rng()):
        raise ValueError("Frozen replay failed to restore caller RNG")
    status["caller_rng_and_modes_restored"] = True


def execute(plan_path, output):
    """One process owns one newly built endpoint model and no optimizer."""
    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import init_distributed
    from online_feedback.recipe import load_event_config
    import torch.distributed as dist

    plan, output = read(plan_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, owned_dist = time.perf_counter(), False
    status = dict(status="preflight", arm=plan["arm"], execution_commit=plan["execution_commit"],
                  response_forwards=0, response_forwards_attempted=0, projection_chunks=0,
                  emitted_occurrences=0, backward_calls=0, optimizer_steps=0,
                  teacher_head_forwards=0, generations=0, startup_response_forwards=0,
                  synchronized_forward_wall_s=0.0)
    write_json(output / "STATUS.json", status)
    try:
        arm = plan["arm"]
        if arm not in ENDPOINT_DIGESTS or plan["endpoint_sha256"] != ENDPOINT_DIGESTS[arm]:
            raise ValueError("Endpoint is not the prespecified step-896 export")
        status["input_validation"] = verify_inputs(plan)
        write_json(output / "STATUS.json", status)
        occurrence_rows = read(plan["sealed_occurrences"])
        occurrences = index_occurrences(occurrence_rows)
        paths, schedule = feature_paths(plan), read(plan["schedule"])
        if len(paths) != 128:
            raise ValueError("Expected all original 128 teacher-feature packets")
        status["endpoint_file_before"] = file_identity(plan["endpoint_path"])
        if status["endpoint_file_before"]["sha256"] != ENDPOINT_DIGESTS[arm]:
            raise ValueError("Endpoint file identity differs before model construction")
        write_json(output / "PLAN.json", plan)
        config, _ = load_event_config(plan["origin_config"])
        os.environ["FSDP_SHARDING"] = config.training.fsdp_sharding
        _bootstrap_single_process_env()
        if int(os.environ["WORLD_SIZE"]) != 1:
            raise ValueError("Frozen replay requires one process on one A800")
        if not dist.is_initialized():
            init_distributed(timeout=config.training.dist_timeout, tp_size=1,
                             sp_ulysses_size=1, sp_ring_size=1)
            owned_dist = True
        if dist.get_world_size() != 1:
            raise ValueError("Frozen replay requires a single rank")
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        recovery = torch.load(plan["origin_state"], map_location="cpu", weights_only=False, mmap=True)
        status["origin"] = validate_recovery(plan["origin_config"], recovery)
        model, cfg = build_readonly_model(plan["origin_config"], recovery)
        endpoint_state = load_file(plan["endpoint_path"], device="cpu")
        model.draft_model.load_state_dict(endpoint_state, strict=True)
        del endpoint_state
        status["endpoint_before"] = verify_endpoint(model.draft_model, plan["endpoint_path"], ENDPOINT_DIGESTS[arm])
        status["runtime_before"] = verify_runtime_model(model)
        status["shared_before"] = shared_piece_identities(model)
        torch.cuda.synchronize()
        status["model_loading_and_integrity_s"] = time.perf_counter() - load_started
        features = _read_features(paths, cfg.data.max_length)
        bindings = validate_schedule(schedule, [r["record_id"] for r in features])
        status["status"] = "running"
        score_presentations(model, features, bindings, occurrences, recovery["rng"],
                            max_length=cfg.data.max_length, arm=arm, output=output, status=status,
                            synchronize=torch.cuda.synchronize)
        if (status["response_forwards"], status["projection_chunks"], status["emitted_occurrences"]) != (256, 8170, 1629):
            raise ValueError("Completed replay work/population differs from the sealed budget")
        verify_started = time.perf_counter()
        status["endpoint_after"] = verify_endpoint(model.draft_model, plan["endpoint_path"], ENDPOINT_DIGESTS[arm])
        status["shared_after"] = shared_piece_identities(model)
        status["runtime_after"] = verify_runtime_model(model)
        if status["shared_after"] != status["shared_before"] or status["runtime_after"] != status["runtime_before"]:
            raise ValueError("Shared frozen components, modes or precision changed during replay")
        status["post_scoring_integrity_s"] = time.perf_counter() - verify_started
        status.update(status="completed", exact_endpoint_before_and_after=True,
                      shared_components_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device_name=torch.cuda.get_device_name())
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      uncompleted_presentations=256 - status["response_forwards"],
                      partial_attempt_unknown_chunks=status["response_forwards_attempted"] > status["response_forwards"])
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
