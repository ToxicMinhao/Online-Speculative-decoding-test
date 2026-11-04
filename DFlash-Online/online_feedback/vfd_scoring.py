"""Fixed, bounded second-pass selection for the November 4 VFD screen.

This module never enters the training objective or consumes its native sampler.
Snapshots contain per-position IDs, validity and FP32 margins, not vocabulary
logits. Counts of repairs, persistent errors and broken prefixes refer to each
original first error; repair and prefix-break flags can overlap.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import time

import torch

from .first_error import valid_prefix_mask
from .rng import preserve_rng

POLICY_VERSION = "vfd-s2-v1"
MAX_SCORE_ANCHORS = 16
_CACHE_NAMES = ("_cached_block_mask", "_cached_seq_len", "_cached_bsz")
_ROTARY_STATE_NAMES = ("max_seq_len_cached", "attention_scaling", "original_inv_freq")


def selection_hash(kind, event_id, response_id, anchor_id=None):
    fields = [POLICY_VERSION, str(kind), str(event_id), str(response_id)]
    if anchor_id is not None:
        fields.append(str(anchor_id))
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


def native_eligible_anchors(loss_mask):
    """The author's adjacent >0.5 supervision rule for one response."""
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError("Scoring requires exactly one response")
    if not bool(torch.isfinite(loss_mask).all()):
        raise ValueError("Nonfinite supervision mask")
    valid = (loss_mask[0, :-1] > 0.5) & (loss_mask[0, 1:] > 0.5)
    return valid.nonzero(as_tuple=False).flatten().cpu().tolist()


def select_score_anchors(loss_mask, event_id, response_id, cap=MAX_SCORE_ANCHORS):
    """Hash-select distinct native anchors, then use native positional order."""
    if (
        not isinstance(cap, int)
        or isinstance(cap, bool)
        or not 1 <= cap <= MAX_SCORE_ANCHORS
    ):
        raise ValueError("Score anchor cap must be an integer from 1 to 16")
    eligible = native_eligible_anchors(loss_mask)
    ranked = sorted(
        eligible,
        key=lambda anchor: (
            selection_hash("anchor", event_id, response_id, anchor),
            anchor,
        ),
    )
    return sorted(ranked[:cap])


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _scoring_runtime(model, anchors, keep):
    """Restore mode, sampler binding, RNG, buffers and known transient caches.

    No weights, gradients, optimizer or reference model are copied. The pinned
    author's no-grad draft forward has no parameter-writing path. Small registered
    buffers are retained because rotary implementations can change them in eval.
    """
    absent = object()
    sampler = model.__dict__.get("_sample_anchor_positions", absent)
    modules = tuple(model.modules())
    modes = [(module, module.training) for module in modules]
    buffers = [
        (
            module,
            dict(module._buffers),
            {
                name: (value.detach().clone(), value._version)
                for name, value in module._buffers.items()
                if value is not None
            },
        )
        for module in modules
        if module._buffers
    ]
    caches = [(model, name, model.__dict__.get(name, absent)) for name in _CACHE_NAMES]
    rotary = getattr(getattr(model, "draft_model", None), "rotary_emb", None)
    if rotary is not None:
        caches.extend(
            (rotary, name, rotary.__dict__.get(name, absent))
            for name in _ROTARY_STATE_NAMES
        )
    with preserve_rng(), torch.no_grad():
        try:
            model.eval()
            model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
            for name in _CACHE_NAMES:
                if hasattr(model, name):
                    setattr(model, name, None)
            yield
        finally:
            if sampler is absent:
                model.__dict__.pop("_sample_anchor_positions", None)
            else:
                model._sample_anchor_positions = sampler
            for module, training in modes:
                module.training = training
            for module, originals, values in buffers:
                module._buffers.clear()
                module._buffers.update(originals)
                for name, (saved, version) in values.items():
                    if originals[name]._version != version:
                        originals[name].copy_(saved)
            for module, name, value in caches:
                if value is absent:
                    module.__dict__.pop(name, None)
                else:
                    setattr(module, name, value)


def native_predictions_and_margins(logits, labels):
    """Native argmax first; gathered canonical/rival logits subtract in FP32."""
    if logits.shape[:-1] != labels.shape or logits.shape[-1] < 2:
        raise ValueError("Expected vocabulary logits with matching canonical labels")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite native scoring logits")
    predictions = logits.argmax(-1)
    canonical = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    competitors = logits.detach().clone()
    competitors.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
    rival = competitors.max(-1).values.float()
    margins = canonical - rival
    if not bool(torch.isfinite(margins).all()):
        raise ValueError("Nonfinite FP32 scoring margin")
    return predictions, margins


def _synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def score_response_snapshot(
    model, batch, *, event_id, response_id, anchor_ids=None, conditioning_id=None
):
    """One no-grad, eval-mode author block forward with at most 16 anchors.

    ``batch`` is a prepared tensor mapping or TrainingBatch. ``conditioning_id``
    should identify the immutable prepared packet already authenticated by the
    runner. IDs/masks are hashed here; large teacher feature tensors are not
    rehashed or retained on every score operation.
    """
    started = time.perf_counter()
    tensors = batch.tensors if hasattr(batch, "tensors") else batch
    inputs, features, mask = (
        tensors[name] for name in ("input_ids", "hidden_states", "loss_mask")
    )
    if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] == 0:
        raise ValueError("Scoring requires one nonempty prepared response")
    if (
        mask.shape != inputs.shape
        or features.shape[:2] != inputs.shape
        or features.ndim != 3
    ):
        raise ValueError("Scoring conditioning geometry differs")
    selected = select_score_anchors(mask, event_id, response_id)
    if anchor_ids is not None and list(anchor_ids) != selected:
        raise ValueError("Scoring anchors differ from the fixed event hash selection")
    chunk_size = int(model.objective_chunk_blocks)
    if chunk_size < 1:
        raise ValueError("Scoring requires a positive native vocabulary chunk size")
    block_size = int(model.block_size)
    result = {
        "policy_version": POLICY_VERSION,
        "event_id": event_id,
        "response_id": response_id,
        "anchor_ids": selected,
        "eligible_anchors": len(native_eligible_anchors(mask)),
        "sampled_anchors": len(selected),
        "context_length": int(inputs.shape[1]),
        "response_length": int((mask > 0).sum()),
        "conditioning_id": conditioning_id,
        "input_ids_sha256": _tensor_digest(inputs),
        "loss_mask_sha256": _tensor_digest(mask),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "block_size": block_size,
        "eos_token_ids": list(model.eos_token_ids),
        "evaluation_mode": True,
        "blocks": [],
        "costs": {
            "response_forwards": 0,
            "draft_forwards": 0,
            "projection_calls": 0,
            "forward_s": 0.0,
            "projection_s": 0.0,
        },
    }
    if not selected:
        result["native_logits_dtype"] = None
        result["costs"]["total_s"] = time.perf_counter() - started
        return result
    device = next(model.draft_model.parameters()).device
    inputs, features, mask = (value.to(device) for value in (inputs, features, mask))
    anchors = torch.tensor([selected], device=device, dtype=torch.long)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    with _scoring_runtime(model, anchors, keep):
        labels, weights, _ = model._block_labels_and_weights(
            inputs, mask, anchors, keep
        )
        valid = valid_prefix_mask(labels, weights, model.eos_token_ids)
        _synchronize(features)
        forward_started = time.perf_counter()
        actual_anchors, actual_keep, hidden = model._forward_draft_blocks(
            inputs, features, mask
        )
        _synchronize(hidden)
        result["costs"]["forward_s"] = time.perf_counter() - forward_started
        if not torch.equal(actual_anchors, anchors) or not torch.equal(
            actual_keep, keep
        ):
            raise ValueError("Author draft forward changed fixed scoring anchors")
        result["costs"]["response_forwards"] = 1
        result["costs"]["draft_forwards"] = 1
        hidden = hidden.reshape(1, len(selected), block_size, -1)
        projection_started = time.perf_counter()
        label_rows = labels[0].cpu().tolist()
        weight_rows = weights[0].cpu().tolist()
        valid_rows = valid[0].cpu().tolist()
        for start in range(0, len(selected), chunk_size):
            stop = min(start + chunk_size, len(selected))
            logits = model.lm_head(hidden[:, start:stop])
            if str(logits.dtype) != str(features.dtype):
                raise ValueError("Scoring logits unexpectedly changed native precision")
            result["native_logits_dtype"] = str(logits.dtype)
            predictions, margins = native_predictions_and_margins(
                logits, labels[:, start:stop]
            )
            result["costs"]["projection_calls"] += 1
            prediction_rows = predictions[0].cpu().tolist()
            margin_rows = margins[0].cpu().tolist()
            for index in range(start, stop):
                local = index - start
                result["blocks"].append(
                    {
                        "anchor_id": selected[index],
                        "labels": label_rows[index],
                        "weights": weight_rows[index],
                        "valid": valid_rows[index],
                        "predictions": prediction_rows[local],
                        "margins": margin_rows[local],
                    }
                )
            del logits, predictions, margins
        _synchronize(hidden)
        result["costs"]["projection_s"] = time.perf_counter() - projection_started
    result["costs"]["total_s"] = time.perf_counter() - started
    return result


def _validate_block(block):
    length = len(block["labels"])
    if length < 2 or any(
        len(block[name]) != length
        for name in ("weights", "valid", "predictions", "margins")
    ):
        raise ValueError("Invalid compact scoring block geometry")
    if block["valid"][0]:
        raise ValueError("The known anchor cannot be a scored proposal")
    if any(not math.isfinite(float(value)) for value in block["margins"]):
        raise ValueError("Nonfinite compact scoring margin")
    ended = False
    for valid in block["valid"][1:]:
        if ended and valid:
            raise ValueError("Block validity cannot reopen after a gap or EOS")
        ended |= not valid


def _credit(before_margin, after_margin):
    margins = torch.tensor([before_margin, after_margin], dtype=torch.float32)
    before, after = margins.unbind()
    movement = (after - before).clamp_min(0)
    gap = (-after).clamp_min(0)
    denominator = movement + gap
    credit = (
        2 * after.sigmoid() * movement / denominator
        if float(denominator) > 0
        else after.new_zeros(())
    )
    if not bool(
        torch.isfinite(movement) & torch.isfinite(gap) & torch.isfinite(credit)
    ):
        raise ValueError("Nonfinite progress-score arithmetic")
    return float(credit), float(movement), float(gap)


def score_response_pair(before, after=None):
    """Compute the sealed formula; average over all sampled valid blocks."""
    binding = (
        "policy_version",
        "event_id",
        "response_id",
        "anchor_ids",
        "eligible_anchors",
        "sampled_anchors",
        "context_length",
        "response_length",
        "conditioning_id",
        "input_ids_sha256",
        "loss_mask_sha256",
        "feature_shape",
        "feature_dtype",
        "block_size",
        "eos_token_ids",
        "evaluation_mode",
        "native_logits_dtype",
    )
    if after is not None and any(before[key] != after[key] for key in binding):
        raise ValueError("Before/after scoring conditioning or precision differs")
    if before["policy_version"] != POLICY_VERSION or not before["evaluation_mode"]:
        raise ValueError("Unknown scoring policy or non-evaluation snapshot")
    if len(before["blocks"]) != len(before["anchor_ids"]):
        raise ValueError("Snapshot block count differs from fixed anchor count")
    if after is not None and len(after["blocks"]) != len(before["blocks"]):
        raise ValueError("Before/after scoring block count differs")
    counts = dict(
        valid_blocks=0,
        fully_matching_blocks=0,
        initial_errors=0,
        initial_opportunities=0,
        repaired_errors=0,
        persistent_errors=0,
        broken_prefixes=0,
        broken_successors=0,
        qualifying_opportunities=0,
        opportunity_repairs=0,
        opportunity_broken_prefixes=0,
        opportunity_persistent_errors=0,
        positive_credit_blocks=0,
    )
    blocks = []
    credit_sum = 0.0
    for index, pre in enumerate(before["blocks"]):
        _validate_block(pre)
        post = None if after is None else after["blocks"][index]
        if pre["anchor_id"] != before["anchor_ids"][index]:
            raise ValueError("Scoring block is not bound to its selected anchor")
        if post is not None:
            _validate_block(post)
            if any(
                pre[key] != post[key]
                for key in ("anchor_id", "labels", "weights", "valid")
            ):
                raise ValueError("Before/after canonical labels or validity differ")
        positions = [j for j, valid in enumerate(pre["valid"]) if valid]
        errors = [j for j in positions if pre["predictions"][j] != pre["labels"][j]]
        counts["valid_blocks"] += bool(positions)
        counts["fully_matching_blocks"] += bool(positions) and not errors
        row = {
            "anchor_id": pre["anchor_id"],
            "valid_positions": len(positions),
            "first_error": errors[0] if errors else None,
            "initial_opportunity": False,
            "credit": 0.0,
        }
        if errors:
            counts["initial_errors"] += 1
            first = errors[0]
            successor = first + 1
            opportunity = (
                successor in positions
                and pre["predictions"][successor] == pre["labels"][successor]
            )
            counts["initial_opportunities"] += opportunity
            row.update(
                initial_opportunity=opportunity, before_margin=pre["margins"][first]
            )
            if post is not None:
                broken = any(
                    post["predictions"][j] != pre["labels"][j]
                    for j in positions
                    if j < first
                )
                repaired = post["predictions"][first] == pre["labels"][first]
                successor_matches = (
                    successor in positions
                    and post["predictions"][successor] == pre["labels"][successor]
                )
                qualifying = (
                    opportunity and not broken and not repaired and successor_matches
                )
                counts["repaired_errors"] += repaired
                counts["persistent_errors"] += not repaired
                counts["broken_prefixes"] += broken
                counts["broken_successors"] += opportunity and not successor_matches
                counts["opportunity_repairs"] += opportunity and repaired
                counts["opportunity_broken_prefixes"] += opportunity and broken
                counts["opportunity_persistent_errors"] += opportunity and not repaired
                counts["qualifying_opportunities"] += qualifying
                after_margin = post["margins"][first]
                value, movement, gap = _credit(pre["margins"][first], after_margin)
                credit = value if qualifying else 0.0
                if qualifying and after_margin > 0:
                    raise ValueError(
                        "Native mismatch cannot have positive canonical margin"
                    )
                row.update(
                    after_margin=after_margin,
                    positive_movement=movement,
                    remaining_gap=gap,
                    prefix_broken=broken,
                    repaired=repaired,
                    persistent=not repaired,
                    successor_matches=successor_matches,
                    qualifying=qualifying,
                    credit=credit,
                )
                counts["positive_credit_blocks"] += credit > 0
                credit_sum += credit
        blocks.append(row)
    denominator = counts["valid_blocks"]
    return {
        "policy_version": POLICY_VERSION,
        "event_id": before["event_id"],
        "response_id": before["response_id"],
        "anchor_ids": before["anchor_ids"],
        "response_length": before["response_length"],
        "context_length": before["context_length"],
        "eligible_anchors": before["eligible_anchors"],
        "sampled_anchors": before["sampled_anchors"],
        "static_score": (
            counts["initial_opportunities"] / denominator if denominator else 0.0
        ),
        "progress_score": credit_sum / denominator if denominator else 0.0,
        "has_after_snapshot": after is not None,
        "counts": counts,
        "blocks": blocks,
    }


def select_responses(scores, full_order, *, event_id, policy, count=64):
    """Select by fixed hashes/scores, then train in FULL's induced pass order."""
    order = list(full_order)
    if len(order) != len(set(order)) or len(order) < count or count < 0:
        raise ValueError(
            "Selection requires distinct IDs and a feasible response count"
        )
    if policy not in {"FULL", "ONCE", "RANDOM64", "STATIC64", "PROGRESS64"}:
        raise ValueError("Unknown second-pass policy")
    score_map = {row["response_id"]: row for row in scores}
    if policy in {"STATIC64", "PROGRESS64"}:
        if len(score_map) != len(scores) or set(score_map) != set(order):
            raise ValueError("Every response needs exactly one current-event score")
        key = "static_score" if policy == "STATIC64" else "progress_score"
        for row in scores:
            if (
                row["event_id"] != event_id
                or row.get("policy_version") != POLICY_VERSION
            ):
                raise ValueError("Scores do not belong to this event/policy version")
            if policy == "PROGRESS64" and not row["has_after_snapshot"]:
                raise ValueError(
                    "Progress selection requires completed pass-one snapshots"
                )
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Selection score is outside its finite unit interval")
        ranked = sorted(
            order,
            key=lambda item: (
                -score_map[item][key],
                selection_hash("tie", event_id, item),
                str(item),
            ),
        )
    elif policy == "RANDOM64":
        key = None
        ranked = sorted(
            order,
            key=lambda item: (selection_hash("random", event_id, item), str(item)),
        )
    else:
        key = None
        ranked = order
    selected = (
        ranked[:count]
        if policy not in {"FULL", "ONCE"}
        else (order if policy == "FULL" else [])
    )
    selected_ids = set(selected)
    induced = [item for item in order if item in selected_ids]
    values = [score_map[item][key] for item in order] if key else []
    boundary = score_map[ranked[count - 1]][key] if key and count else None
    return {
        "policy_version": POLICY_VERSION,
        "event_id": event_id,
        "policy": policy,
        "selected_ids": selected,
        "training_order": induced,
        "selected_count": len(selected),
        "ranked_ids": ranked,
        "score_key": key,
        "zero_scores": sum(value == 0 for value in values),
        "zero_rate": (
            sum(value == 0 for value in values) / len(values) if values else None
        ),
        "tied_responses": sum(values.count(value) > 1 for value in values),
        "tie_rate": (
            sum(values.count(value) > 1 for value in values) / len(values)
            if values
            else None
        ),
        "cutoff_score": boundary,
        "cutoff_tie_count": values.count(boundary) if boundary is not None else 0,
        "selected_response_lengths": (
            [score_map[item]["response_length"] for item in induced]
            if score_map
            else []
        ),
        "selected_valid_blocks": (
            sum(score_map[item]["counts"]["valid_blocks"] for item in induced)
            if score_map
            else None
        ),
    }
