"""Response-wide successor-bit permutation on the unchanged native K+ loss.

Only the detached assignment changes. The ordinary BF16 KL rows, native masks,
FP32 chunk reductions and K+ redistribution formula remain shared.
"""

import hashlib
import json
import time

import torch

from .first_error import first_error_mask, valid_prefix_mask
from .kplus import (
    KPlusRedistribution,
    MASS_ATOL,
    MASS_RTOL,
    first_error_groups,
    group_terms,
    install_kplus,
    redistribute_response,
)


SEED = 20261111
CHUNK_BLOCKS = 8
SCHEMA = "kplus_successor_shuffle_v1"


def successor_shuffle_contract(*, identity=False):
    return {
        "name": SCHEMA,
        "identity": bool(identity),
        "seed": SEED,
        "stratum": "one_response_and_first_error_proposal_position",
        "eligible": "first_error_with_immediate_valid_successor_including_canonical_EOS",
        "recipients": "ascending_canonical_anchor_offset",
        "donors": "SHA256(UTF8(compact_JSON([seed,response_id,position,anchor_offset]))),offset",
        "serialization": "ensure_ascii=False,separators=(',',':'),no_trailing_newline",
        "fixed_points": "allowed_without_reroll",
        "single_class_and_singleton": "assigned_bits_unchanged",
        "projection_chunk_blocks": CHUNK_BLOCKS,
        "additional_backbone_forwards": 0,
        "additional_teacher_projections": 0,
        "additional_student_projections": "one_detached_projection_per_eight_block_chunk",
        "training_rng_draws": 0,
        "reduction": "unchanged_native_KPlusRedistribution",
        "reassigned_weight": "half_L1_change_in_first_error_coefficients_times_original_weights",
    }


def donor_key(response_id, position, anchor_offset):
    encoded = json.dumps(
        [SEED, response_id, int(position), int(anchor_offset)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), int(anchor_offset)


@torch.no_grad()
def successor_assignment(
    proposals,
    labels,
    weights,
    anchors,
    response_id,
    *,
    eos_token_ids=(),
    identity=False
):
    """Return four immutable mask channels and a compact manipulation receipt.

    Channels are native selection, selected error, true high, assigned high.
    No input tensor or RNG state is changed. Native positional decay is equal
    within a stratum; reject weights for which a bit permutation changes mass.
    """
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("A stable nonempty canonical response identity is required")
    if (
        labels.ndim != 3
        or labels.shape[0] != 1
        or proposals.shape != labels.shape
        or weights.shape != labels.shape
        or anchors.shape != labels.shape[:2]
    ):
        raise ValueError("Require aligned blocks from exactly one response")
    if anchors.dtype not in (torch.int32, torch.int64) or bool((anchors < 0).any()):
        raise ValueError("Canonical anchor offsets must be nonnegative integers")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Require finite nonnegative native weights")
    selection = first_error_mask(proposals, labels, weights, eos_token_ids)
    low, high = first_error_groups(proposals, labels, weights, selection, eos_token_ids)
    errors = low | high
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    eligible = torch.zeros_like(valid)
    eligible[..., :-1] = errors[..., :-1] & valid[..., 1:]
    assigned = high.clone()
    offsets = anchors[0].tolist()
    active = valid.any(-1)[0].tolist()
    active_offsets = [offset for offset, present in zip(offsets, active) if present]
    if len(active_offsets) != len(set(active_offsets)):
        raise ValueError("Active canonical anchor offsets must be unique")
    donors_by_block = {i: offsets[i] for i in range(len(offsets))}
    strata = []
    for position in range(1, labels.shape[-1]):
        members = (
            torch.nonzero(eligible[0, :, position], as_tuple=False).flatten().tolist()
        )
        if not members:
            continue
        recipients = sorted(members, key=lambda block: offsets[block])
        values = weights[0, recipients, position]
        if not torch.equal(values, values[0].expand_as(values)):
            raise ValueError("Successor strata require equal native positional weights")
        bits = high[0, recipients, position].tolist()
        n, k = len(recipients), sum(bits)
        donors = recipients
        if not identity and n > 1:
            donors = sorted(
                recipients,
                key=lambda block: donor_key(response_id, position, offsets[block]),
            )
        assigned[0, recipients, position] = high[0, donors, position]
        for recipient, donor in zip(recipients, donors):
            donors_by_block[recipient] = offsets[donor]
        changed = int(
            (assigned[0, recipients, position] != high[0, recipients, position]).sum()
        )
        strata.append(
            {
                "position": position,
                "n": n,
                "k": k,
                "recipients": [offsets[i] for i in recipients],
                "donors": [offsets[i] for i in donors],
                "true_high_count": k,
                "assigned_high_count": int(assigned[0, recipients, position].sum()),
                "flipped_bits": changed,
                "uniform_expected_flips": 2.0 * k * (n - k) / n,
            }
        )
    blocks = []
    for block, offset in enumerate(offsets):
        positions = torch.nonzero(errors[0, block], as_tuple=False).flatten().tolist()
        position = positions[0] if positions else None
        blocks.append(
            {
                "anchor_offset": offset,
                "first_error_position": position,
                "eligible_successor": (
                    bool(eligible[0, block, position])
                    if position is not None
                    else False
                ),
                "true_priority": (
                    1 + int(high[0, block, position]) if position is not None else None
                ),
                "assigned_priority": (
                    1 + int(assigned[0, block, position])
                    if position is not None
                    else None
                ),
                "donor_anchor_offset": (
                    donors_by_block[block] if position is not None else None
                ),
            }
        )
    packed = torch.stack((selection, errors, high, assigned), dim=-1).detach()
    receipt = {
        "schema": SCHEMA,
        "status": "prepared",
        "response_id": response_id,
        "seed": SEED,
        "identity": bool(identity),
        "anchor_offsets": offsets,
        "blocks": blocks,
        "strata": strata,
        "eligible_errors": int(eligible.sum()),
        "ineligible_errors": int((errors & ~eligible).sum()),
        "mixed_strata": sum(0 < item["k"] < item["n"] for item in strata),
        "flipped_bits": int((high != assigned).sum()),
        "uniform_expected_flips": sum(
            item["uniform_expected_flips"] for item in strata
        ),
        "class_counts_exact": all(
            item["k"] == item["assigned_high_count"] for item in strata
        ),
        "position_counts": [
            {
                "position": position,
                "errors": int(errors[..., position].sum()),
                "eligible": int(eligible[..., position].sum()),
                "true_high": int(high[..., position].sum()),
                "assigned_high": int(assigned[..., position].sum()),
            }
            for position in range(1, labels.shape[-1])
        ],
    }
    return packed, receipt


def assigned_group_terms(
    token_loss, proposals, labels, weights, selection, packed, eos_token_ids=()
):
    """Pure chunk function: checkpoint replay reads explicit masks, never state."""
    expected_selection, errors, true_high, assigned_high = packed.unbind(-1)
    low, high = first_error_groups(proposals, labels, weights, selection, eos_token_ids)
    if (
        not torch.equal(selection, expected_selection)
        or not torch.equal(errors, low | high)
        or not torch.equal(true_high, high)
    ):
        raise RuntimeError(
            "Detached prepass and native current-forward first-error masks differ"
        )
    weight1 = weights.detach() * (errors & ~assigned_high)
    weight2 = weights.detach() * assigned_high
    assigned = (
        weight1.sum(),
        weight2.sum(),
        (token_loss * weight1).sum(),
        (token_loss * weight2).sum(),
        (errors & ~assigned_high).sum().float(),
        assigned_high.sum().float(),
    )
    # Detached unpermuted terms are manipulation evidence on this very state.
    original = group_terms(
        token_loss.detach(), proposals, labels, weights, selection, eos_token_ids
    )
    return (*assigned, *original)


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class SuccessorShuffle:
    def __init__(self, *, identity=False):
        self.identity = bool(identity)
        self.response_id = None
        self.last_receipt = None
        self._snapshot = None
        self._eos_token_ids = ()

    def set_response_id(self, response_id):
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(
                "A stable nonempty canonical response identity is required"
            )
        self.response_id = response_id
        self._snapshot = None
        self.last_receipt = dict(
            schema=SCHEMA,
            status="pending",
            response_id=response_id,
            seed=SEED,
            identity=self.identity,
        )

    @torch.no_grad()
    def prepare_response(self, model, hidden, labels, weights, anchors):
        if self.response_id is None:
            raise ValueError("Set response identity before every shuffled presentation")
        response_id, self.response_id = self.response_id, None
        self._eos_token_ids = tuple(model.eos_token_ids)
        self._snapshot = None
        self.last_receipt.update(
            status="preparing", projection_calls=0, projection_completed=0
        )
        model.costs["shuffle_presentations"] = (
            model.costs.get("shuffle_presentations", 0) + 1
        )
        _sync(hidden)
        started = time.perf_counter()
        proposals = []
        try:
            for start in range(0, hidden.shape[1], CHUNK_BLOCKS):
                self.last_receipt["projection_calls"] += 1
                model.costs["shuffle_student_projection_calls"] = (
                    model.costs.get("shuffle_student_projection_calls", 0) + 1
                )
                logits = model.lm_head(
                    hidden[:, start : start + CHUNK_BLOCKS].detach()
                ).to(model.loss_dtype)
                proposals.append(logits.argmax(-1))
                self.last_receipt["projection_completed"] += 1
                del logits
            proposed = torch.cat(proposals, dim=1)
            _sync(proposed)
        except BaseException as error:
            self.last_receipt.update(status="failed", error=repr(error))
            raise
        finally:
            projection_s = time.perf_counter() - started
            self.last_receipt["projection_s"] = projection_s
            model.costs["shuffle_student_projection_s"] = (
                model.costs.get("shuffle_student_projection_s", 0.0) + projection_s
            )
        started = time.perf_counter()
        try:
            packed, receipt = successor_assignment(
                proposed,
                labels,
                weights,
                anchors,
                response_id,
                eos_token_ids=model.eos_token_ids,
                identity=self.identity,
            )
            self._snapshot = {
                "anchors": anchors.detach().clone(),
                "labels": labels.detach().clone(),
                "weights": weights.detach().clone(),
                **{
                    name: value.detach().clone()
                    for name, value in zip(
                        ("selection", "errors", "true_high", "assigned_high"),
                        packed.unbind(-1),
                    )
                },
            }
            valid = valid_prefix_mask(labels, weights, model.eos_token_ids)
            eligible = torch.zeros_like(valid)
            eligible[..., :-1] = self._snapshot["errors"][..., :-1] & valid[..., 1:]
            self._snapshot["eligible"] = eligible
            _sync(packed)
        except BaseException as error:
            self._snapshot = None
            self.last_receipt.update(status="failed", error=repr(error))
            raise
        finally:
            assignment_s = time.perf_counter() - started
            self.last_receipt["assignment_s"] = assignment_s
            model.costs["shuffle_assignment_s"] = (
                model.costs.get("shuffle_assignment_s", 0.0) + assignment_s
            )
        receipt.update(
            projection_calls=len(proposals),
            projection_completed=len(proposals),
            projection_s=projection_s,
            assignment_s=assignment_s,
        )
        self.last_receipt = receipt
        return packed

    @torch.no_grad()
    def finish_response(self, terms, assigned_metrics):
        true_numerator, truth = redistribute_response(
            terms[0].detach(), terms[1].detach(), terms[11:17]
        )
        receipt = self.last_receipt
        receipt["selected_weight"] = float(terms[1])
        receipt["denominator"] = float(terms[1] + 1e-6)
        receipt["true_loss"] = float(true_numerator / (terms[1] + 1e-6))
        receipt["assigned_loss"] = float(
            assigned_metrics["kplus_N_new"] / (terms[1] + 1e-6)
        )
        for label, metrics in (("true", truth), ("assigned", assigned_metrics)):
            for name in ("W1", "W2", "coefficient1", "coefficient2"):
                receipt[label + "_" + name] = float(metrics["kplus_" + name])
            receipt[label + "_error_mass"] = float(
                metrics["kplus_first_error_weight_after"]
            )
        checked = (
            "W1",
            "W2",
            "coefficient1",
            "coefficient2",
            "first_error_weight_after",
        )
        mass_preserved = all(
            bool(
                torch.isclose(
                    truth["kplus_" + name],
                    assigned_metrics["kplus_" + name],
                    rtol=MASS_RTOL,
                    atol=MASS_ATOL,
                )
            )
            for name in checked
        )
        if not mass_preserved:
            receipt.update(status="failed", mass_preserved=False)
            raise RuntimeError(
                "Successor permutation changed native K+ class mass or coefficients"
            )
        values = self._snapshot
        true_coeff = torch.where(
            values["true_high"],
            truth["kplus_coefficient2"],
            truth["kplus_coefficient1"],
        )
        assigned_coeff = torch.where(
            values["assigned_high"],
            assigned_metrics["kplus_coefficient2"],
            assigned_metrics["kplus_coefficient1"],
        )
        l1 = (
            values["weights"] * values["errors"] * (assigned_coeff - true_coeff).abs()
        ).sum()
        receipt.update(
            status="passed",
            mass_preserved=True,
            selection_preserved=True,
            denominator_preserved=True,
            coefficient_l1_weight=float(l1),
            reassigned_weight=float(0.5 * l1),
        )
        numeric = (
            "eligible_errors",
            "ineligible_errors",
            "mixed_strata",
            "flipped_bits",
            "uniform_expected_flips",
            "reassigned_weight",
            "coefficient_l1_weight",
            "projection_calls",
            "projection_s",
            "assignment_s",
            "true_loss",
            "assigned_loss",
        )
        metrics = {
            "shuffle_" + name: terms[1].new_tensor(receipt[name]) for name in numeric
        }
        metrics["shuffle_response_count"] = terms[1].new_ones(())
        return metrics

    def qualification_snapshot(self):
        if self._snapshot is None:
            raise RuntimeError("No response assignment has been prepared")
        return {
            **{
                name: value.detach().cpu().tolist()
                for name, value in self._snapshot.items()
            },
            "eos_token_ids": list(self._eos_token_ids),
        }


def install_successor_shuffle(model, *, identity=False):
    if model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16:
        raise ValueError("Successor shuffle requires native BF16 osd_first_error")
    if model.objective_chunk_blocks != CHUNK_BLOCKS:
        raise ValueError(
            "Successor shuffle requires the fixed eight-block projection chunks"
        )
    if any(
        getattr(model, name, None) is not None
        for name in (
            "successor_shuffle",
            "prefix_guard",
            "point_loss_extension",
            "role_loss_extension",
            "survival_head_extension",
            "adversarial_block",
        )
    ):
        raise ValueError("Successor shuffle cannot combine another objective extension")
    reducer = getattr(model, "first_error_redistribution", None)
    if reducer is None:
        install_kplus(model)
    elif type(reducer) is not KPlusRedistribution:
        raise ValueError("Successor shuffle requires the unmodified native K+ reducer")
    model.successor_shuffle = SuccessorShuffle(identity=identity)
    return successor_shuffle_contract(identity=identity)
