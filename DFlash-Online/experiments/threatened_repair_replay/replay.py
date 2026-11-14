"""Read immutable memory layouts and add sparse teacher-KL before native clipping.

The complete original block layout reaches both the draft and frozen head. Only
the final loss weights are sparse. Auxiliary autograd does not accumulate into
the native DDP gradient buckets; its ordered BF16 sum is added once afterward.
"""
from contextlib import contextmanager
from copy import deepcopy
import time

import torch
import torch.nn.functional as F

from experiments.lost_repair.training_probe import prepare_batch
from online_feedback.first_error import first_error_mask, valid_prefix_mask
from online_feedback.functional_geometry import isolated_geometry_runtime
from online_feedback.ranking import fixed_anchors
from specforge.core.chunking import checkpointed_chunk_reduce

from .core import computation_key, tag_key


@contextmanager
def replay_runtime(model):
    """Preserve the caller's training modes, buffers, caches, RNG and cost book."""
    modes = [(module, module.training) for module in model.modules()]
    costs = deepcopy(getattr(model, "costs", None))
    with isolated_geometry_runtime(model):
        # The ordinary reference layouts were measured in training mode. The
        # qualified draft has no dropout, but mode is still part of the contract.
        for module, training in modes:
            module.training = training
        try:
            yield
        finally:
            if costs is not None:
                model.costs.clear()
                model.costs.update(costs)


@contextmanager
def _work(ledger, name, context, **fields):
    # The controller labels memory observations separately from replay work.
    # Merge explicitly: forwarding category twice would fail before the ledger
    # could persist the attempted operation.
    metadata = {"category": "replay", **(context or {}), **fields}
    if ledger is None:
        yield {}
        return
    # operation metadata label its receipt; scope also labels physical hooks
    # nested inside it, without installing another layer/head hook.
    with ledger.scope(**metadata), ledger.operation(name) as receipt:
        yield receipt


def _layout_inputs(model, layout, max_length):
    raw = torch.load(layout["feature_path"], map_location="cpu", weights_only=True)
    binding = layout["binding"]
    if (str(raw["record_id"]), str(binding["record_id"]), binding["presentation_id"]) != (
            str(layout["request_id"]), str(layout["request_id"]), layout["reference_presentation_id"]):
        raise ValueError("Replay feature identity differs from its original layout")
    device = next(model.parameters()).device
    batch = prepare_batch(raw, max_length, device)
    observation, = binding["sampled_anchors"]
    anchors = torch.tensor(observation["positions"], device=device, dtype=torch.long)
    keep = torch.tensor(observation["keep_mask"], device=device, dtype=torch.bool)
    labels, weights, indices = model._block_labels_and_weights(
        batch["input_ids"], batch["loss_mask"], anchors, keep)
    valid = valid_prefix_mask(labels, weights, model.eos_token_ids)
    for tag in layout["candidate_tags"]:
        ordinal, slot = int(tag["anchor_ordinal"]), int(tag["proposal_slot"])
        if (not 1 <= slot < model.block_size or not bool(valid[0, ordinal, slot])
                or int(anchors[0, ordinal]) - raw["prompt_length"] != tag["canonical_start"]
                or int(labels[0, ordinal, slot]) != tag["canonical_token"]
                or float(weights[0, ordinal, slot]) != tag["weight"]):
            raise ValueError("Replay tag no longer matches its original token/anchor/weight")
    return raw, batch, anchors, keep, labels, weights, indices


def _hidden(model, batch, anchors, keep):
    with fixed_anchors(model, anchors, keep):
        actual_anchors, actual_keep, hidden = model._forward_draft_blocks(
            batch["input_ids"], batch["hidden_states"], batch["loss_mask"])
    if not torch.equal(actual_anchors, anchors) or not torch.equal(actual_keep, keep):
        raise ValueError("Replay changed the immutable full anchor layout")
    return hidden.reshape(1, anchors.shape[1], model.block_size, -1)


@torch.no_grad()
def observation_rows(logits, labels, weights, tags, *, anchor_offset, eos_token_ids=()):
    """Native decisions include matched EOS and retain wrong argmax ties."""
    if logits.dtype != torch.bfloat16 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Memory observation requires finite native BF16 logits")
    proposals = logits.argmax(-1)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    selected = first_error_mask(proposals, labels, weights, eos_token_ids)
    prefix = (selected & (proposals == labels)).sum(-1)
    rows = []
    for tag in sorted(tags, key=computation_key):
        local = int(tag["anchor_ordinal"]) - anchor_offset
        if not 0 <= local < labels.shape[1]:
            continue
        slot, canonical = int(tag["proposal_slot"]), int(tag["canonical_token"])
        values = logits[0, local, slot]
        rivals = values.clone()
        rivals[canonical] = -torch.inf
        rival = int(rivals.argmax())
        margin = float(values[canonical].float() - values[rival].float())
        length, available = int(prefix[0, local]), int(valid[0, local].sum())
        correct = bool(proposals[0, local, slot] == canonical)
        rows.append(dict(request_id=str(tag["request_id"]),
            canonical_start=int(tag["canonical_start"]), proposal_slot=slot,
            reference_presentation_id=int(tag["reference_presentation_id"]),
            anchor_ordinal=int(tag["anchor_ordinal"]), canonical_token=canonical,
            weight=float(weights[0, local, slot]), accepted_prefix=length,
            valid_proposals=available, first_error=length + 1 if length < available else None,
            valid=bool(valid[0, local, slot]), native_correct=correct, direct_correct=correct,
            proposed_token=int(proposals[0, local, slot]), rival_token=rival,
            margin=margin, wrong_tie=margin == 0 and not correct))
    return rows


def _check_layout(model):
    if (model.block_size != 16 or model.objective_chunk_blocks != 8
            or model.loss_dtype != torch.bfloat16):
        raise ValueError("Replay requires the pinned block-16/eight-block BF16 layout")


def _ordered_layouts(layouts):
    if len({row["layout_id"] for row in layouts}) != len(layouts):
        raise ValueError("A full replay layout was scheduled twice")
    return sorted(layouts, key=lambda row: (
        str(row["request_id"]), int(row["reference_presentation_id"])))


@torch.no_grad()
def observe_memory(model, layouts, *, max_length, ledger=None, context=None):
    """One complete original response forward per layout; no teacher projection."""
    _check_layout(model)
    rows, receipts = [], []
    with replay_runtime(model):
        for layout in _ordered_layouts(layouts):
            started = time.perf_counter()
            with _work(ledger, "response_forward", context, layout_id=layout["layout_id"]):
                _, batch, anchors, keep, labels, weights, _ = _layout_inputs(model, layout, max_length)
                hidden = _hidden(model, batch, anchors, keep)
                chunks = 0
                for begin in range(0, anchors.shape[1], 8):
                    stop = begin + 8
                    logits = model.lm_head(hidden[:, begin:stop]).to(model.loss_dtype)
                    rows.extend(observation_rows(logits, labels[:, begin:stop],
                        weights[:, begin:stop], layout["candidate_tags"], anchor_offset=begin,
                        eos_token_ids=model.eos_token_ids))
                    chunks += 1
                del hidden, logits
            receipts.append(dict(layout_id=layout["layout_id"], response_forwards=1,
                logical_head_chunks=chunks, head_forward_calls=chunks, backwards=0,
                wall_s=time.perf_counter() - started))
    if len(rows) != sum(len(layout["candidate_tags"]) for layout in layouts):
        raise ValueError("Memory observation omitted or repeated a sealed tag")
    return rows, receipts


@torch.no_grad()
def qualify_memory_observer(model, layouts, *, max_length, ledger=None, context=None):
    """Compare independent retained observation logic on the very same logits.

    The temporary head hook performs scalar extraction only. It never invokes
    the head or draft again, and is removed even when the observation fails.
    """
    from experiments.recent_repair_retention.reference import slot_measurements

    started = time.perf_counter()
    comparisons, work, mismatches = [], [], []
    for layout in _ordered_layouts(layouts):
        with replay_runtime(model):
            raw, _, anchors, keep, labels, weights, _ = _layout_inputs(model, layout, max_length)
        expected = {tag_key(tag): tag["canonical_token"] for tag in layout["candidate_tags"]}
        independent, shapes = [], []
        ordinal = [0]

        def observe_head(_module, _inputs, logits):
            start = ordinal[0] * 8
            ordinal[0] += 1
            stop = min(start + 8, anchors.shape[1])
            shape = tuple(logits.shape)
            if logits.shape[:3] != (1, stop-start, model.block_size):
                raise ValueError("Memory head no longer uses the retained full eight-block grouping")
            shapes.append(list(shape))
            original = slot_measurements(logits, labels[:, start:stop], weights[:, start:stop],
                anchors=anchors[:, start:stop], keep=keep[:, start:stop],
                prompt_length=raw["prompt_length"], request_id=layout["request_id"],
                presentation_id=layout["reference_presentation_id"],
                pass_index=layout["binding"]["pass_index"], eos_token_ids=model.eos_token_ids,
                candidates=set(expected), expected_tokens=expected)
            for row in original:
                anchor = start + row["local_anchor_ordinal"]
                # Independently count the contiguous canonical suffix, including
                # EOS and stopping at missing supervision or a sequence end.
                available = 0
                if int(labels[0, anchor, 0]) not in model.eos_token_ids:
                    for slot in range(1, model.block_size):
                        if float(weights[0, anchor, slot]) <= 0:
                            break
                        available += 1
                        if int(labels[0, anchor, slot]) in model.eos_token_ids:
                            break
                independent.append(dict(request_id=row["request_id"],
                    canonical_start=row["canonical_start"], proposal_slot=row["proposal_slot"],
                    reference_presentation_id=row["presentation_id"],
                    anchor_ordinal=anchor, canonical_token=row["canonical_token"],
                    weight=row["weight"], accepted_prefix=row["correct_prefix"],
                    valid_proposals=available, valid=row["valid"],
                    first_error=row["correct_prefix"]+1 if row["correct_prefix"] < available else None,
                    native_correct=row["native_correct"], direct_correct=row["native_correct"],
                    wrong_tie=row["margin"] == 0 and not row["native_correct"], margin=row["margin"]))

        handle = model.lm_head.register_forward_hook(observe_head)
        try:
            observed, receipts = observe_memory(model, [layout], max_length=max_length,
                ledger=ledger, context=context)
        finally:
            handle.remove()
        work.extend(receipts)
        actual = {tag_key(row): row for row in observed}
        reference = {tag_key(row): row for row in independent}
        if actual.keys() != reference.keys():
            mismatches.append(dict(layout_id=layout["layout_id"], field="tag_membership"))
        for key in sorted(actual.keys() & reference.keys()):
            for field, value in reference[key].items():
                if actual[key][field] != value:
                    mismatches.append(dict(layout_id=layout["layout_id"], tag=list(key),
                        field=field, actual=actual[key][field], reference=value))
        expected_chunks = (anchors.shape[1] + 7) // 8
        if ordinal[0] != expected_chunks:
            mismatches.append(dict(layout_id=layout["layout_id"], field="head_count",
                actual=ordinal[0], reference=expected_chunks))
        comparisons.append(dict(layout_id=layout["layout_id"], compared_tags=len(reference),
            physical_head_shapes=shapes, response_forwards=1, extra_model_or_head_forwards=0))
    return dict(status="passed" if not mismatches else "failed", comparisons=comparisons,
        mismatches=mismatches, work=work, response_forwards=len(layouts),
        independent_reference="retained PR44 slot_measurements plus direct canonical boundary counting",
        additional_head_forwards=0, optimizer_steps=0, wall_s=time.perf_counter()-started)


def sparse_weights(weights, labels, tags, selected_keys, *, eos_token_ids=()):
    """Freeze sparse row membership before constructing checkpointed graphs."""
    mask = torch.zeros_like(weights)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    for tag in sorted(tags, key=computation_key):
        ordinal, slot = int(tag["anchor_ordinal"]), int(tag["proposal_slot"])
        if not bool(valid[0, ordinal, slot]) or int(labels[0, ordinal, slot]) != tag["canonical_token"]:
            raise ValueError("Sparse replay selected an invalid or changed canonical row")
        if tag_key(tag) in selected_keys:
            mask[0, ordinal, slot] = weights[0, ordinal, slot]
    return mask.detach().clone()


def teacher_kl_terms(logits, teacher_logits, row_weights):
    """Native BF16 forward KL, detached teacher, FP32 scalar weighting."""
    probability = F.softmax(teacher_logits.detach(), dim=-1)
    token_loss = F.kl_div(F.log_softmax(logits, dim=-1), probability,
                        reduction="none").sum(-1)
    return (token_loss * row_weights).sum()


def positional_weight_sum(selected_tags, *, device):
    return torch.tensor([tag["weight"] for tag in sorted(selected_tags,
        key=lambda tag: (int(tag["proposal_slot"]), computation_key(tag)))],
        dtype=torch.float32, device=device).sum()


def replay_denominator(selected_tags, *, device):
    """Pin one FP32 denominator from the shared ascending-position multiset."""
    return positional_weight_sum(selected_tags, device=device) + 1e-6


@torch.no_grad()
def add_gradient_sum(parameters, replay_gradients):
    """One BF16 addition before the original clip; retain exact native zero path."""
    totals = torch.zeros(5, dtype=torch.float64, device=parameters[0].device)
    changed = 0
    for parameter, extra in zip(parameters, replay_gradients):
        native = parameter.grad
        if native is None:
            raise ValueError("Native averaged gradient is missing at the replay boundary")
        if extra is None:
            extra = torch.zeros_like(native)
        if not bool(torch.isfinite(native).all() & torch.isfinite(extra).all()):
            raise ValueError("Nonfinite native/replay gradient before clipping")
        before = native.float()
        totals[0] += before.double().square().sum()
        totals[1] += extra.double().square().sum()
        totals[2] += (before.double() * extra.double()).sum()
        # Padded zero layouts really run backward, but do not rewrite buckets.
        if bool(torch.count_nonzero(extra)):
            native.add_(extra.to(native.dtype))
        realized = native.float() - before
        totals[3] += realized.double().square().sum()
        totals[4] += native.double().square().sum()
        changed += int(torch.count_nonzero(realized))
    n2, r2, dot, a2, combined2 = totals.tolist()
    return dict(native_preclip_norm=n2 ** .5, replay_gradient_norm=r2 ** .5,
        native_replay_dot=dot, realized_added_gradient_norm=a2 ** .5,
        combined_preclip_norm=combined2 ** .5, changed_gradient_entries=changed)


def add_replay_gradients(trainer, layouts, selected_tags, *, coefficient=.10,
                        ledger=None, context=None, force_padded_zero=False,
                        denominator=None):
    """Add one global sparse replay loss to an already averaged native pair.

    Each full-layout backward finishes before the next layout. Its gradients
    accumulate in BF16, matching native parameter arithmetic. No optimizer,
    clipping, response averaging, sampler draw or target model call occurs here.
    """
    receipt = dict(selected_tags=len(selected_tags), coefficient=coefficient,
        response_forwards=0, response_backwards=0, logical_head_chunks=0,
        head_forward_calls=0, teacher_head_calls=0, recomputed_head_chunks=0,
        layout_receipts=[], status="direct_native")
    if (coefficient == 0 or not selected_tags) and not force_padded_zero:
        return receipt
    model = trainer.bundle.model
    _check_layout(model)
    parameters = tuple(trainer.backend.optimizer.model_params)
    selected = {tag_key(tag) for tag in selected_tags}
    candidates = {tag_key(tag): tag for layout in layouts for tag in layout["candidate_tags"]}
    if len(selected) != len(selected_tags) or not selected <= candidates.keys():
        raise ValueError("Replay selection is not a unique subset of the padded union")
    if any(tag != candidates[tag_key(tag)] for tag in selected_tags):
        raise ValueError("Replay selection modified immutable candidate metadata")
    # One deterministic global selected-row normalization, not a response mean.
    # Sorting the same-j multiset makes the paired denominators exactly equal.
    selected_weight = positional_weight_sum(selected_tags, device=parameters[0].device)
    expected_denominator = selected_weight + 1e-6
    if denominator is None:
        denominator = expected_denominator
    elif (denominator.dtype != torch.float32 or denominator.requires_grad
            or not torch.equal(denominator, expected_denominator)):
        raise ValueError("Replay denominator differs from the shared positional budget")
    replay_gradients = [None] * len(parameters)
    total_loss = 0.0
    started = time.perf_counter()
    with replay_runtime(model), torch.enable_grad():
        for layout in _ordered_layouts(layouts):
            with _work(ledger, "response_forward", context, layout_id=layout["layout_id"]):
                raw, batch, anchors, keep, labels, weights, indices = _layout_inputs(
                    model, layout, trainer.cfg.data.max_length)
                row_weights = sparse_weights(weights, labels, layout["candidate_tags"], selected,
                    eos_token_ids=model.eos_token_ids)
                hidden = _hidden(model, batch, anchors, keep)
                # The shared DFlash normalizer returns draft-conditioning inputs
                # only. As in OnlineTrainer._batch, retain the separate frozen
                # teacher field from the validated capture, without retokenizing
                # or changing the predecessor-row alignment.
                teacher_states = raw["target_last_hidden_states"].detach().to(
                    hidden.device, non_blocking=True)
                teacher = model._aligned_target_hidden(teacher_states, indices)
                del teacher_states
                # Bound arguments, rather than a mutable per-response closure,
                # are retained by each activation-checkpoint node.
                seen = [0]
                logical = (anchors.shape[1] + 7) // 8
                def chunk(current, frozen, sparse, _layout=layout["layout_id"],
                          _seen=seen, _logical=logical):
                    recompute = _seen[0] >= _logical
                    _seen[0] += 1
                    receipt["recomputed_head_chunks"] += int(recompute)
                    receipt["head_forward_calls"] += 1
                    receipt["teacher_head_calls"] += 1
                    logits = model.lm_head(current).to(model.loss_dtype)
                    with torch.no_grad():
                        teacher_logits = model.lm_head(frozen).to(model.loss_dtype)
                    return (teacher_kl_terms(logits, teacher_logits, sparse),)
                numerator, = checkpointed_chunk_reduce(chunk, hidden, teacher, row_weights,
                    chunk_size=8, dim=1, checkpoint=True)
                loss = coefficient * numerator / denominator
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite replay loss")
                receipt["response_forwards"] += 1
                receipt["logical_head_chunks"] += logical
                total_loss += float(loss.detach())
                receipt["layout_receipts"].append(dict(layout_id=layout["layout_id"],
                    selected_rows=int(torch.count_nonzero(row_weights)),
                    selected_weight=float(row_weights.sum()), numerator=float(numerator.detach()),
                    denominator=float(denominator), auxiliary_loss=float(loss.detach()),
                    logical_head_chunks=logical))
            with _work(ledger, "response_backward", context, layout_id=layout["layout_id"]):
                gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
                for index, gradient in enumerate(gradients):
                    if not bool(torch.isfinite(gradient).all()):
                        raise ValueError("Nonfinite replay gradient")
                    if replay_gradients[index] is None:
                        replay_gradients[index] = gradient.detach().clone()
                    else:
                        replay_gradients[index].add_(gradient.detach())
                receipt["response_backwards"] += 1
            del gradients, loss, hidden, teacher, numerator
        receipt.update(add_gradient_sum(parameters, replay_gradients))
    receipt.update(status="padded_replay_added", selected_weight=float(selected_weight),
        denominator=float(denominator), auxiliary_loss=total_loss,
        replay_loss=total_loss/coefficient if coefficient else 0.,
        wall_s=time.perf_counter() - started,
        gradient_accumulation="ordered full-layout BF16; one BF16 native-gradient addition",
        normalization="global selected rows; position-sorted weight sum plus 1e-6")
    return receipt
