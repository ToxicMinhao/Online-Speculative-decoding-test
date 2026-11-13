"""A small, response-normalized margin floor on sealed historical repairs.

The native K+ loss, sampler and optimizer remain the owners of training.  This
extension binds detached metadata to the anchors that the sampler actually drew
and adds one hinge term.  It neither samples blocks nor retains model features.
"""

import math

import torch

from .first_error import first_error_mask, valid_prefix_mask
from .prefix_protection import canonical_margin


COEFFICIENT = 0.10
FLOOR_FRACTION = 0.50
FLOOR_CAP = 1.0

# Additive statistics, summed across chunks before a response is normalized.
STAT_NAMES = (
    "assigned_count", "assigned_weight", "reachable_count", "reachable_weight",
    "firing_count", "firing_weight", "firing_incorrect_count",
    "firing_incorrect_weight", "firing_correct_count", "firing_correct_weight",
    "correct_tie_count", "wrong_tie_count", "strict_incorrect_count",
)


def retention_contract(*, enabled=True, coefficient=COEFFICIENT):
    return {
        "name": "kplus_recent_repair_retention_event7_v1",
        "enabled": bool(enabled), "coefficient": float(coefficient),
        "reference_floor": "0.50 * min(parent_margin, 1.0)",
        "reach": "native_first_error_mask_at_assigned_slot",
        "hinge": "relu(detached_floor - live_canonical_margin)",
        "normalization": "native_live_selected_weight_sum_plus_1e-6_per_response",
        "logits": "qualified_native_BF16",
        "margin_and_reduction": "FP32",
        "zero_coefficient": "exact_native_loss_graph",
        "zero_hinge": "exact_native_loss_graph",
        "reference_model": "none_in_training",
        "extra_model_forwards": 0,
    }


@torch.no_grad()
def parent_slot_measurements(logits, labels, weights, eos_token_ids=()):
    """Frozen preparation's scalar view of one native vocabulary-head chunk.

    Call on the actual eight-block projections, including the actual final
    partial chunk.  The caller owns anchor identities and schedule validation.
    Only returned scalars/masks need survive; no full vocabulary scores do.
    """
    proposals = logits.argmax(-1)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    prefix = first_error_mask(proposals, labels, weights, eos_token_ids)
    prefix &= proposals == labels
    return {
        "margins": canonical_margin(logits, labels).detach(),
        "valid": valid,
        "correct_prefix": prefix.sum(-1),
        "proposals": proposals,
        "correct": proposals == labels,
    }


def retention_chunk_terms(logits, labels, weights, selection, assigned, floors):
    """Return a differentiable numerator and detached additive statistics.

    Selection already is K's prefix-through-first-error mask.  Gathering only
    assigned rows avoids a second full-vocabulary clone for unassigned slots.
    All operations are functional so activation-checkpoint recomputation cannot
    change assignments or double-count logging.
    """
    assigned = assigned.detach()
    positions = torch.nonzero(assigned, as_tuple=True)
    if not positions[0].numel():
        zero = weights.new_zeros(())
        return zero, weights.new_zeros(len(STAT_NAMES))
    margin = canonical_margin(logits[positions], labels[positions])
    raw_weight = weights[positions].detach()
    reachable = selection[positions].detach()
    hinge = torch.relu(floors[positions].detach() - margin)
    numerator = (raw_weight * reachable * hinge).sum()
    with torch.no_grad():
        correct = logits[positions].argmax(-1) == labels[positions]
        firing = reachable & (hinge > 0)
        wrong_fire = firing & ~correct
        correct_fire = firing & correct
        ties = margin == 0
        statistics = torch.stack((
            raw_weight.new_tensor(raw_weight.numel()), raw_weight.sum(),
            reachable.sum().float(), (raw_weight * reachable).sum(),
            firing.sum().float(), (raw_weight * firing).sum(),
            wrong_fire.sum().float(), (raw_weight * wrong_fire).sum(),
            correct_fire.sum().float(), (raw_weight * correct_fire).sum(),
            (ties & correct).sum().float(), (ties & ~correct).sum().float(),
            (margin < 0).sum().float(),
        ))
    return numerator, statistics


class RecentRepairRetention:
    """Presentation binding and response reduction, outside the module tree."""

    def __init__(self, *, enabled=True, coefficient=COEFFICIENT):
        if coefficient not in (0.0, COEFFICIENT):
            raise ValueError("Retention permits its fixed dose or zero for qualification")
        self.enabled = bool(enabled)
        self.coefficient = float(coefficient)
        self.presentation = None

    @property
    def active(self):
        return self.enabled and self.coefficient != 0

    def contract(self):
        return retention_contract(enabled=self.enabled, coefficient=self.coefficient)

    def set_presentation(self, rows, *, request_id, presentation_id, prompt_length,
                         expected_anchors, expected_keep):
        """Copy compact sealed records before the ordinary response forward.

        Rows use response-relative canonical starts and proposal slots.  The
        expected sampler output is a check, never a substitute for sampling.
        Tensor metadata is cloned now, so caller mutation cannot alter backward.
        """
        self.presentation = {
            "rows": tuple(dict(row) for row in rows),
            "request_id": str(request_id), "presentation_id": presentation_id,
            "prompt_length": int(prompt_length),
            "anchors": torch.as_tensor(expected_anchors).detach().cpu().clone(),
            "keep": torch.as_tensor(expected_keep).detach().cpu().bool().clone(),
        }

    @torch.no_grad()
    def prepare_response(self, anchors, keep, labels, weights, eos_token_ids=()):
        """Bind by exact start identity, rejecting stale or ineligible metadata."""
        if self.presentation is None:
            raise RuntimeError("Bind the sealed retention presentation before forwarding")
        record = self.presentation
        expected_anchors = record["anchors"].reshape_as(anchors)
        expected_keep = record["keep"].reshape_as(keep)
        if not torch.equal(anchors.detach().cpu(), expected_anchors):
            raise ValueError("Retention presentation's naturally sampled anchors differ")
        if not torch.equal(keep.detach().cpu().bool(), expected_keep):
            raise ValueError("Retention presentation's naturally sampled keep mask differs")
        if labels.shape[0] != 1:
            raise ValueError("Retention requires native one-response microbatches")
        assigned = torch.zeros_like(labels, dtype=torch.bool)
        floors = torch.zeros_like(weights, dtype=torch.float32)
        valid = valid_prefix_mask(labels, weights, eos_token_ids)
        positions = {}
        for ordinal, (anchor, kept) in enumerate(zip(anchors[0].tolist(), keep[0].tolist())):
            if kept:
                start = int(anchor) - record["prompt_length"]
                if start in positions:
                    raise ValueError("Retention requires unique kept anchor starts")
                positions[start] = ordinal
        for row in record["rows"]:
            if str(row.get("request_id", record["request_id"])) != record["request_id"]:
                raise ValueError("Retention assignment belongs to another request")
            if row.get("presentation_id", record["presentation_id"]) != record["presentation_id"]:
                raise ValueError("Retention assignment belongs to another presentation")
            start, slot = int(row["canonical_start"]), int(row["proposal_slot"])
            block = positions.get(start)
            if block is None or not 0 < slot < labels.shape[-1]:
                raise ValueError("Assigned retention slot has no kept proposal anchor")
            index = (0, block, slot)
            if bool(assigned[index]) or not bool(valid[index]):
                raise ValueError("Assigned retention slot is duplicate or not valid")
            if "anchor_ordinal" in row and int(row["anchor_ordinal"]) != block:
                raise ValueError("Retention anchor ordinal changed")
            if int(labels[index]) != int(row["canonical_token"]):
                raise ValueError("Retention canonical token differs from live labels")
            if float(weights[index]) != float(row["position_weight"]):
                raise ValueError("Retention raw positional weight changed")
            margin = float(row["parent_margin"])
            if not math.isfinite(margin) or margin < 0:
                raise ValueError("Retention parent must have finite nonnegative margin")
            assigned[index] = True
            floors[index] = FLOOR_FRACTION * min(margin, FLOOR_CAP)
        return assigned, floors

    chunk_terms = staticmethod(retention_chunk_terms)

    def augment_response(self, native_loss, denominator, terms):
        """Normalize once by native selected mass, never by tagged mass."""
        numerator, statistics = terms
        auxiliary = self.coefficient * numerator / denominator
        # A zero ReLU has zero derivative.  Keep the native graph untouched in
        # that case, avoiding a numerically different zero-addition backward.
        loss = native_loss if not bool(numerator.detach() > 0) else native_loss + auxiliary
        sums = {f"retention_{name}": value for name, value in zip(STAT_NAMES, statistics)}
        sums.update({
            "retention_hinge_numerator": numerator.detach(),
            "retention_native_denominator": denominator.detach(),
            "retention_auxiliary_loss": auxiliary.detach(),
            "retention_native_kl_loss": native_loss.detach(),
            "retention_coefficient": denominator.new_tensor(self.coefficient),
            "retention_response_count": denominator.new_ones(()),
        })
        ratios = {
            "retention_hinge_loss": (numerator.detach(), denominator.detach()),
            "retention_auxiliary_weighted_loss": (
                self.coefficient * numerator.detach(), denominator.detach()),
            "retention_reach_rate": (statistics[2], statistics[0]),
            "retention_firing_rate": (statistics[4], statistics[0]),
            "retention_weighted_reach_rate": (statistics[3], statistics[1]),
            "retention_weighted_firing_rate": (statistics[5], statistics[1]),
        }
        return loss, sums, ratios


def install_recent_repair_retention(model, *, enabled=True, coefficient=COEFFICIENT):
    """Attach only to native K+; never register historical weights or features."""
    if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
            or getattr(model, "first_error_redistribution", None) is None):
        raise ValueError("Recent retention requires the qualified native BF16 K+ objective")
    incompatible = ("prefix_guard", "successor_shuffle", "survival_head_extension",
                    "role_loss_extension", "point_loss_extension", "adversarial_block",
                    "recent_repair_retention")
    if any(getattr(model, name, None) is not None for name in incompatible):
        raise ValueError("Recent retention cannot combine other experimental extensions")
    extension = RecentRepairRetention(enabled=enabled, coefficient=coefficient)
    model.recent_repair_retention = extension
    return extension
