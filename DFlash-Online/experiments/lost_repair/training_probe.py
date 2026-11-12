"""Read-only native K+ sampling and exposure on the fixed K+6 checkpoint.

No TrainerCore, optimizer, loss backward or publication is constructed here.
The draft conditioning, sampler, validity and eight-block vocabulary projection
are the existing training operations. Only compact per-anchor observations leave
each forward; native response-wide K+ coefficients are reduced after all chunks.
"""

from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.kplus import first_error_groups, group_terms, install_kplus, redistribute_response
from online_feedback.rng import capture_rng, preserve_rng, restore_rng


def _torch_rng_equal(left, right):
    return torch.equal(left["torch"], right["torch"]) and len(left["cuda"]) == len(
        right["cuda"]
    ) and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def _rng_equal(left, right):
    import numpy as np

    return (_torch_rng_equal(left, right) and left["python"] == right["python"]
            and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1])
            and left["numpy"][2:] == right["numpy"][2:])


@contextmanager
def frozen_training_state(model, origin_rng):
    """Use a disposable RNG trajectory while leaving caller modes/gradients intact."""
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter._version, parameter.grad,
                   parameter.grad._version if parameter.grad is not None else None)
                  for parameter in model.parameters()]
    caches = {
        name: getattr(model, name)
        for name in ("_cached_block_mask", "_cached_seq_len", "_cached_bsz")
        if hasattr(model, name)
    }
    # Native Flex Attention has no attention dropout. Enforce that reason for
    # reproducing the sampler trajectory without running optimizer/backward work.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError("Read-only schedule requires the native zero-dropout draft")
        if getattr(module, "attention_dropout", 0) != 0:
            raise ValueError("Native attention dropout differs from zero")
    with preserve_rng():
        restore_rng(origin_rng)
        model.train()
        try:
            with torch.no_grad():
                yield
        finally:
            for module, training in modes:
                module.training = training
            for name, value in caches.items():
                setattr(model, name, value)
            if any(parameter._version != version or parameter.grad is not gradient
                   or (gradient is not None and gradient._version != gradient_version)
                   for parameter, version, gradient, gradient_version in parameters):
                raise RuntimeError("Read-only probe modified a parameter or ordinary gradient")


def prepare_batch(raw, max_length, device):
    """Use the native normalizer after its complete-response validation."""
    from online_feedback.training import validate_feature, validate_target_binding
    from specforge.algorithms.common.hidden_states_data import normalize_offline_sample

    validate_feature(raw, max_length=max_length)
    validate_target_binding(raw)
    sample = dict(raw, input_ids=raw["input_ids"][0], loss_mask=raw["loss_mask"][0])
    tensors = normalize_offline_sample(sample, max_length)
    return {name: value.to(device, non_blocking=True) for name, value in tensors.items()}


def _read_features(feature_paths, max_length):
    """Inspect identities once; stream feature tensors one response at a time."""
    from online_feedback.training import validate_feature, validate_target_binding

    features = []
    for path in feature_paths:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        row = validate_feature(raw, max_length=max_length)
        validate_target_binding(raw)
        features.append(dict(row, path=str(Path(path))))
    if len({row["record_id"] for row in features}) != len(features):
        raise ValueError("Probe requires distinct response identities")
    return features


def _anchor_record(anchors, keep):
    return dict(positions=anchors.detach().cpu().tolist(), keep_mask=keep.detach().cpu().tolist())


def sample_schedule(model, feature_paths, origin_rng, *, max_length, passes=2):
    """Draw the two-pass native schedule without a model forward or RNG leakage."""
    features = _read_features(feature_paths, max_length)
    device = next(model.parameters()).device
    records = []
    with frozen_training_state(model, origin_rng):
        for pass_index in range(passes):
            for index in torch.randperm(len(features)).tolist():
                feature = features[index]
                length, prompt = feature["length"], feature["prompt_length"]
                mask = torch.zeros((1, length), device=device)
                mask[:, prompt:] = 1
                anchors, keep = model._sample_anchor_positions(
                    length, mask, device, max_valid_anchors=length - prompt - 1,
                )
                records.append(dict(
                    record_id=feature["record_id"], pass_index=pass_index,
                    presentation_id=len(records), sampled_anchors=[_anchor_record(anchors, keep)],
                ))
        final_rng = capture_rng()
    return dict(records=records, final_rng=final_rng, response_presentations=len(records),
                passes=passes, sampler_draws=len(records), model_forwards=0, optimizer_steps=0)


@torch.no_grad()
def observe_logits(logits, labels, weights, *, eos_token_ids=()):
    """Observe one native head chunk, including EOS, ties and unavailable followers."""
    if logits.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("Probe requires native BF16 logits and FP32 position weights")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite native training logits")
    proposals = logits.argmax(-1)
    selection = first_error_mask(proposals, labels, weights, eos_token_ids)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    low, high = first_error_groups(proposals, labels, weights, selection, eos_token_ids)
    errors = low | high
    # The winning token is already noncanonical at a true mismatch, including
    # wrong-at-tie positions. Use the native deterministic argmax winner.
    wrong_margin = logits.gather(-1, proposals.unsqueeze(-1)).squeeze(-1).float() - (
        logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    )
    ordinary = group_terms(
        torch.zeros_like(weights), proposals, labels, weights, selection, eos_token_ids,
    )
    rows = []
    for block in range(labels.shape[1]):
        positions = errors[0, block].nonzero().flatten()
        j = int(positions[0]) if positions.numel() else None
        row = dict(j=j, priority="ineligible", native_priority=None,
                   ordinary_first_error_weight=0.0, wrong_margin=None,
                   valid_proposals=int(valid[0, block].sum()),
                   correct_prefix=int((valid & (proposals == labels))[0, block, :j].sum())
                   if j is not None else int(valid[0, block].sum()))
        if j is not None:
            successor_valid = j + 1 < labels.shape[-1] and bool(valid[0, block, j + 1])
            native_priority = "high" if bool(high[0, block, j]) else "low"
            row.update(
                priority=native_priority if successor_valid else "ineligible",
                native_priority=native_priority, successor_valid=successor_valid,
                ordinary_first_error_weight=float(weights[0, block, j]),
                wrong_margin=float(wrong_margin[0, block, j]),
                wrong_at_tie=bool(wrong_margin[0, block, j] == 0),
                canonical_token=int(labels[0, block, j]),
                wrong_token=int(proposals[0, block, j]),
            )
        rows.append(row)
    return rows, ((weights * selection).sum(), *ordinary)


@torch.no_grad()
def observe_response(model, raw, *, presentation_id, pass_index, max_length,
                     expected_anchors=None):
    """One full-block draft forward and native-shaped head chunks, no teacher head."""
    device = next(model.parameters()).device
    batch = prepare_batch(raw, max_length, device)
    original_sampler = model._sample_anchor_positions
    sampled_rng = None

    def observe_sampler(*args, **kwargs):
        nonlocal sampled_rng
        result = original_sampler(*args, **kwargs)
        if expected_anchors is not None and _anchor_record(*result) != expected_anchors:
            raise ValueError("Native anchor draw differs from the sealed disposable schedule")
        sampled_rng = capture_rng()
        return result

    model._sample_anchor_positions = observe_sampler
    try:
        anchors, keep, hidden = model._forward_draft_blocks(
            batch["input_ids"], batch["hidden_states"], batch["loss_mask"],
            max_valid_anchors=raw["input_ids"].shape[1] - raw["prompt_length"] - 1,
        )
    finally:
        model._sample_anchor_positions = original_sampler
    labels, weights, _ = model._block_labels_and_weights(
        batch["input_ids"], batch["loss_mask"], anchors, keep,
    )
    hidden = hidden.reshape(1, anchors.shape[1], model.block_size, -1)
    rows, totals, chunks = [], None, 0
    for start in range(0, anchors.shape[1], model.objective_chunk_blocks):
        stop = start + model.objective_chunk_blocks
        logits = model.lm_head(hidden[:, start:stop]).to(model.loss_dtype)
        observed, terms = observe_logits(
            logits, labels[:, start:stop], weights[:, start:stop], eos_token_ids=model.eos_token_ids,
        )
        totals = terms if totals is None else tuple(a + b for a, b in zip(totals, terms))
        for offset, row in enumerate(observed):
            block = start + offset
            if not bool(keep[0, block]):
                continue
            row.update(
                presentation_id=presentation_id, pass_index=pass_index,
                request_id=raw["record_id"], anchor_ordinal=block,
                canonical_start=int(anchors[0, block]) - raw["prompt_length"],
                sequence_start=int(anchors[0, block]), kept=bool(keep[0, block]),
            )
            rows.append(row)
        chunks += 1
        del logits
    selected, *native_terms = totals
    _, native = redistribute_response(selected.new_zeros(()), selected, tuple(native_terms))
    coefficients = {"low": float(native["kplus_coefficient1"]),
                    "high": float(native["kplus_coefficient2"])}
    for row in rows:
        coefficient = coefficients[row["native_priority"]] if row["j"] is not None else 0.0
        row["native_coefficient"] = coefficient
        row["first_error_weight"] = float(
            weights.new_tensor(row["ordinary_first_error_weight"]) * coefficient
        )
    after_rng = capture_rng()
    if sampled_rng is None or not _rng_equal(sampled_rng, after_rng):
        raise RuntimeError("Native read-only forward consumed RNG outside anchor sampling")
    summary = dict(
        presentation_id=presentation_id, pass_index=pass_index, request_id=raw["record_id"],
        selected_weight=float(selected), first_error_weight=float(native["kplus_first_error_weight_after"]),
        ordinary_low_weight=float(native["kplus_W1"]), ordinary_high_weight=float(native["kplus_W2"]),
        low_coefficient=coefficients["low"], high_coefficient=coefficients["high"],
        anchors=len(rows), expected_anchor_count=int(keep.sum()),
        sampled_anchor_slots=int(anchors.numel()), sampled_anchors=[_anchor_record(anchors, keep)],
        response_forwards=1, projection_chunks=chunks, backwards=0, optimizer_steps=0,
        no_rng_consumption_after_sampling=True,
    )
    return rows, summary


def run_readonly_probe(model, feature_paths, origin_rng, *, max_length, schedule, sink=None, budget=None):
    """Replay all scheduled presentations at fixed weights and emit each durably."""
    features = _read_features(feature_paths, max_length)
    all_rows, presentations = [], []
    expected = schedule["records"]
    began = time.perf_counter()
    with frozen_training_state(model, origin_rng):
        for pass_index in range(schedule["passes"]):
            for index in torch.randperm(len(features)).tolist():
                feature = features[index]
                ordinal = len(presentations)
                binding = expected[ordinal]
                if (binding["record_id"], binding["pass_index"], binding["presentation_id"]) != (
                    feature["record_id"], pass_index, ordinal,
                ):
                    raise ValueError("Native presentation order differs from the sealed schedule")
                ticket = budget.attempt("training_forward", presentation_id=ordinal,
                                        request_id=feature["record_id"]) if budget is not None else None
                response_began = time.perf_counter()
                try:
                    raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                    rows, summary = observe_response(
                        model, raw, presentation_id=ordinal, pass_index=pass_index, max_length=max_length,
                        expected_anchors=binding["sampled_anchors"][0],
                    )
                    summary["wall_s"] = time.perf_counter() - response_began
                    if sink is not None:
                        sink(rows, summary)
                    if budget is not None:
                        budget.complete(ticket, projection_chunks=summary["projection_chunks"],
                                        wall_s=summary["wall_s"])
                except BaseException as error:
                    if budget is not None:
                        budget.fail(ticket, repr(error))
                    raise
                all_rows.extend(rows)
                presentations.append(summary)
        if len(presentations) != len(expected):
            raise ValueError("Read-only event did not cover the complete sealed schedule")
        if not _rng_equal(capture_rng(), schedule["final_rng"]):
            raise ValueError("Final disposable sampler state differs from the sealed schedule")
    return dict(
        status="completed", training_rows=all_rows, presentation_rows=presentations,
        response_forwards=len(presentations), projection_chunks=sum(x["projection_chunks"] for x in presentations),
        backwards=0, optimizer_steps=0, wall_s=time.perf_counter() - began,
        caller_rng_and_modes_restored=True, parameters_and_gradients_unchanged=True,
        teacher_head_forwards=0,
    )


def validate_recovery(config_path, recovery):
    """Bind the read-only model and sampler to the exact completed K+6 event."""
    if Path(config_path).read_text() != recovery["config_text"]:
        raise ValueError("Read-only configuration differs from the exact K+6 recovery")
    expected = dict(global_step=768, presentations=1536, horizon=7424,
                    accumulation_steps=2, reduction="response_mean")
    for name, value in expected.items():
        if recovery[name] != value:
            raise ValueError(f"K+6 recovery {name} differs: {recovery[name]!r} != {value!r}")
    return dict(config_matches_recovery=True, accumulation_remainder=0, **expected)


def build_readonly_model(config_path, recovery):
    """Build only native model pieces; caller owns device/distributed setup."""
    from specforge.application import resolve_run
    from specforge.training.assembly import build_model_bundle
    from online_feedback.objective import TargetObjectiveModel
    from online_feedback.recipe import load_event_config
    from online_feedback.training import ranking_eos_token_ids

    validate_recovery(config_path, recovery)
    cfg, recipe = load_event_config(config_path)
    if (cfg.model.torch_dtype != "bfloat16" or recipe.kl_precision != "bfloat16"
            or cfg.training.attention_backend != "flex_attention" or not recipe.shuffle_responses):
        raise ValueError("Read-only probe requires the pinned native BF16/Flex/shuffled recipe")
    with preserve_rng():
        bundle = build_model_bundle(cfg, algorithm=resolve_run(cfg).algorithm)
        bundle.draft_model.load_state_dict(recovery["draft_state_dict"], strict=True)
        model = TargetObjectiveModel(
            bundle.model, "osd_first_error", loss_precision=recipe.kl_precision,
            eos_token_ids=ranking_eos_token_ids(cfg.model.target_model_path),
        )
        install_kplus(model)
    if model.block_size != 16 or model.objective_chunk_blocks != 8 or model.num_anchors != 512:
        raise ValueError("Native block, head chunk or anchor setting differs")
    model.train()
    return model, cfg


def _file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def execute(features_path, config_path, recovery_path, output, budget_connection):
    """Isolated GPU worker; a parent supervisor owns the aggregate GPU clock."""
    from experiments.verification_transition.budget import connect_budget
    from online_feedback.storage import save_torch, write_json
    import torch.distributed as dist

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "STATUS.json"
    if status_path.exists():
        raise FileExistsError("Fresh training-probe destination required; no automatic retry")
    began = time.perf_counter()
    owned_dist = False
    completed = []
    write_json(status_path, dict(status="preflight", completed_presentations=0,
                                 planned_presentations=256, optimizer_steps=0, backwards=0))
    try:
        budget = connect_budget(json.loads(Path(budget_connection).read_text()))
        budget.register_process(os.getpid(), os.getpgrp())
        feature_list = json.loads(Path(features_path).read_text())
        feature_paths = feature_list if isinstance(feature_list, list) else feature_list["feature_paths"]
        if len(feature_paths) != 128:
            raise ValueError("The sealed event-7 pool must contain exactly 128 responses")
        sources = [_file_identity(path) for path in (features_path, config_path, recovery_path)]
        sources.extend(_file_identity(path) for path in feature_paths)
        write_json(output / "INPUTS.json", dict(inputs=sources, optimizer_creation=False,
                                                config_matches_recovery=None))
        load_ticket = budget.attempt("model_load", operation="readonly_training_bundle")
        load_began = time.perf_counter()
        try:
            from specforge.cli import _bootstrap_single_process_env
            from specforge.distributed import init_distributed
            from online_feedback.recipe import load_event_config

            config, _ = load_event_config(config_path)
            os.environ["FSDP_SHARDING"] = config.training.fsdp_sharding
            if not dist.is_initialized():
                _bootstrap_single_process_env()
                if int(os.environ["WORLD_SIZE"]) != 1:
                    raise ValueError("Read-only training probe requires one process")
                init_distributed(timeout=config.training.dist_timeout,
                                 tp_size=1, sp_ulysses_size=1, sp_ring_size=1)
                owned_dist = True
            if dist.get_world_size() != 1:
                raise ValueError("Read-only training probe requires a single rank")
            torch.cuda.reset_peak_memory_stats()
            recovery = torch.load(recovery_path, map_location="cpu", weights_only=False, mmap=True)
            restoration = validate_recovery(config_path, recovery)
            write_json(output / "INPUTS.json", dict(inputs=sources, optimizer_creation=False,
                                                    **restoration))
            model, cfg = build_readonly_model(config_path, recovery)
            torch.cuda.synchronize()
            budget.complete(load_ticket, wall_s=time.perf_counter() - load_began)
        except BaseException as error:
            budget.fail(load_ticket, repr(error))
            raise
        schedule_began = time.perf_counter()
        schedule = sample_schedule(model, feature_paths, recovery["rng"], max_length=cfg.data.max_length)
        write_json(output / "SCHEDULE.json", {key: value for key, value in schedule.items() if key != "final_rng"})
        save_torch(output / "SCHEDULE_RNG.pt", schedule["final_rng"])
        schedule_s = time.perf_counter() - schedule_began

        def persist(rows, summary):
            ordinal = summary["presentation_id"]
            write_json(output / "presentations" / f"{ordinal:03d}.json",
                       dict(training_rows=rows, presentation=summary))
            completed.append(summary)
            write_json(status_path, dict(status="running", completed_presentations=len(completed),
                                         planned_presentations=256, optimizer_steps=0, backwards=0))

        result = run_readonly_probe(
            model, feature_paths, recovery["rng"], max_length=cfg.data.max_length,
            schedule=schedule, sink=persist, budget=budget,
        )
        torch.cuda.synchronize()
        result.update(
            initialization_s=time.perf_counter() - began - result["wall_s"] - schedule_s,
            schedule_s=schedule_s, phase_wall_s=time.perf_counter() - began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            startup_response_forwards=0, startup_backwards=0,
            temporary_optimizer_steps=0, schedule_identity=_file_identity(output / "SCHEDULE.json"),
        )
        write_json(output / "RESULT.json", result)
        write_json(status_path, dict(status="completed", completed_presentations=256,
                                     planned_presentations=256, optimizer_steps=0, backwards=0))
        return result
    except BaseException as error:
        write_json(status_path, dict(status="failed", error=repr(error), traceback=traceback.format_exc(),
                                     completed_presentations=len(completed), planned_presentations=256,
                                     uncompleted_presentations=256 - len(completed),
                                     optimizer_steps=0, backwards=0, phase_wall_s=time.perf_counter() - began))
        raise
    finally:
        if owned_dist:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "config", "recovery", "output", "budget-connection"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    execute(args.features, args.config, args.recovery, args.output, args.budget_connection)


if __name__ == "__main__":
    main()
