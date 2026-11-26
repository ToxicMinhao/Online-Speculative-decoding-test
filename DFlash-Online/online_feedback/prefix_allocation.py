"""Bounded redistribution of native K+'s existing correct-prefix KL mass.

Scores use detached canonical probabilities from the same BF16 log-softmax as
ordinary KL. The first-error objective and response denominator are untouched.
Only additive scalars and compact allocation summaries cross head chunks.
"""

import torch

from .first_error import valid_prefix_mask
from .kplus import MASS_ATOL, MASS_RTOL


SMOOTHING = 0.5
RELATIVE_CAP = 0.25
SUMMARY_FIELDS = (
    "blocks", "valid_positions", "prefix_positions", "changed_blocks",
    "changed_positions", "prefix_weight_before", "prefix_weight_after",
    "weight_moved", "mass_error_abs_sum", "mass_violations",
    "bounds_violations", "support_violations", "nonfinite_values",
)


def prefix_allocation_contract(relative_cap=RELATIVE_CAP):
    """Fixed candidate definition; cap zero exists only for qualification."""
    if relative_cap not in (0.0, RELATIVE_CAP):
        raise ValueError("Prefix allocation permits cap 0.25 or qualification-only cap 0")
    return {
        "name": "kplus_bounded_prefix_allocation_v1",
        "smoothing": SMOOTHING,
        "relative_cap": float(relative_cap),
        "score": "reverse_cumsum(cumprod(0.5+0.5*q)) over every valid proposal",
        "probability": "exp(FP32(detached canonical gather from native BF16 log_softmax))",
        "allocation_precision": "float32",
        "support": "current correct prefix; no new first-error or post-error loss",
        "numerator": "native_N_plus + sum_prefix((a-b)*native_KL_row)",
        "denominator": "unchanged native selected_weight_sum+1e-6 once per response",
        "response_averaging": "unchanged two separately normalized responses",
        "mass_rtol": MASS_RTOL,
        "mass_atol": MASS_ATOL,
        "no_op": "cap zero or exactly unchanged coefficients returns native numerator object",
        "additional_model_forwards": 0,
        "additional_vocabulary_projections": 0,
    }


@torch.no_grad()
def bounded_prefix_coefficients(probability, weights, valid, prefix, *, relative_cap=RELATIVE_CAP):
    """Return FP32 coefficients; invalid slots never extend the score horizon.

    All operations retain the block's position dimension. Denominator ones below
    belong only to masked empty sets; smoothing makes every nonempty score sum
    strictly positive, so no epsilon changes the allocation rule.
    """
    base = weights.detach().float()
    if relative_cap == 0:
        return base
    probability = probability.detach().float()
    factors = torch.where(valid, SMOOTHING + (1 - SMOOTHING) * probability, 1.0)
    products = factors.cumprod(-1) * valid
    scores = products.flip(-1).cumsum(-1).flip(-1)
    return coefficients_from_scores(scores, base, prefix, relative_cap=relative_cap)


@torch.no_grad()
def coefficients_from_scores(scores, weights, prefix, *, relative_cap=RELATIVE_CAP):
    """Normalize the prefix scores, then bound relative movement toward them."""
    base = weights.detach().float()
    if relative_cap == 0:
        return base
    scores = scores.detach().float()
    mass = (base * prefix).sum(-1, keepdim=True)
    score_mass = (scores * prefix).sum(-1, keepdim=True)
    score_denominator = torch.where(score_mass > 0, score_mass, 1.0)
    desired = mass * scores / score_denominator
    relative = torch.where(prefix, desired / torch.where(prefix, base, 1.0) - 1, 0.0)
    maximum = relative.abs().amax(-1, keepdim=True)
    safe_maximum = torch.where(maximum > 0, maximum, 1.0)
    factor = torch.where(maximum > 0, (relative_cap / safe_maximum).clamp(max=1), 0.0)
    candidate = base * (1 + factor * relative)
    movable = prefix & (prefix.sum(-1, keepdim=True) > 1)
    return torch.where(movable, candidate, base)


@torch.no_grad()
def allocation_summaries(base, candidate, valid, prefix):
    """Additive diagnostics: no mutable chunk counters or vocabulary retention."""
    difference = candidate - base
    lengths = prefix.sum(-1)
    before = (base * prefix).sum(-1)
    after = (candidate * prefix).sum(-1)
    changed = (difference != 0) & prefix
    lower, upper = (1 - RELATIVE_CAP) * base, (1 + RELATIVE_CAP) * base
    tolerance = MASS_ATOL + MASS_RTOL * base.abs()
    outside = prefix & ((candidate < lower - tolerance) | (candidate > upper + tolerance))
    scalars = torch.stack((
        base.new_tensor(lengths.numel()), valid.sum().float(), prefix.sum().float(),
        changed.any(-1).sum().float(), changed.sum().float(), before.sum(), after.sum(),
        0.5 * difference.abs().sum(), (after - before).abs().sum(),
        (~torch.isclose(after, before, rtol=MASS_RTOL, atol=MASS_ATOL)).sum().float(),
        outside.sum().float(), ((difference != 0) & ~prefix).sum().float(),
        (~torch.isfinite(candidate)).sum().float(),
    ))
    # A small [prefix length, slot, statistic] table, summed across response chunks.
    slots = base.shape[-1]
    values = torch.stack((torch.ones_like(base), base, candidate, difference, difference.abs()), -1)
    # Keep a fixed reduction layout. A floating-point scatter into repeated
    # length/slot indices can reorder CUDA additions and spoil exact replay of
    # these diagnostics even when the scientific state itself is unchanged.
    groups = torch.arange(slots + 1, device=base.device).reshape(-1, 1, 1)
    membership = (groups == lengths.reshape(1, -1, 1)) & prefix.reshape(1, -1, slots)
    by_slot = torch.where(membership.unsqueeze(-1), values.reshape(1, -1, slots, 5), 0.).sum(1)
    by_length = torch.bincount(lengths.reshape(-1), minlength=slots + 1).float()
    return scalars, by_slot, by_length


class PrefixAllocation:
    """Persistent fixed rule, with one refreshed summary per original response."""

    def __init__(self, relative_cap=RELATIVE_CAP):
        self.contract = prefix_allocation_contract(relative_cap)
        self.relative_cap = float(relative_cap)
        self.last_summary = None
        self.coefficient_probe = None

    def chunk_terms(self, log_probability, token_loss, proposals, labels, weights,
                    selection, eos_token_ids=()):
        with torch.no_grad():
            valid = valid_prefix_mask(labels, weights, eos_token_ids)
            prefix = selection.detach() & valid & (proposals.detach() == labels)
            canonical_log_probability = log_probability.detach().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            probability = canonical_log_probability.float().exp()
            base = weights.detach().float()
            candidate = bounded_prefix_coefficients(
                probability, base, valid, prefix, relative_cap=self.relative_cap)
            summaries = allocation_summaries(base, candidate, valid, prefix)
            summaries[0][-1] += (~torch.isfinite(probability) & valid).sum()
            if self.coefficient_probe is not None:
                self.coefficient_probe(dict(base=base, candidate=candidate, prefix=prefix, valid=valid))
        # Do not branch on this scalar value: cancellation can leave a nonzero
        # gradient even when the correction evaluates to zero.
        delta = ((candidate - base) * token_loss).sum()
        return delta, *summaries

    def reduce_response(self, native_numerator, terms):
        """Append Delta after native K+ has consumed its original eleven fields."""
        delta, counts, by_slot, by_length = terms
        values = counts.detach().cpu().tolist()
        summary = dict(zip(SUMMARY_FIELDS, values))
        self.last_summary = summary
        if any(summary[name] for name in (
                "mass_violations", "bounds_violations", "support_violations", "nonfinite_values")):
            raise RuntimeError("Prefix allocation invariant failed: " + repr(summary))
        summary["prefix_length_counts"] = by_length.detach().cpu().tolist()
        summary["by_prefix_length_and_slot"] = [
            dict(prefix_length=length, slot=slot, occurrences=row[0],
                 native_weight_sum=row[1], candidate_weight_sum=row[2],
                 signed_weight_change=row[3], absolute_weight_change=row[4])
            for length, slots in enumerate(by_slot.detach().cpu().tolist())
            for slot, row in enumerate(slots) if row[0]
        ]
        summary["delta_numerator"] = float(delta.detach())
        self.last_summary = summary
        # Native object identity depends on coefficient movement, never on Delta.
        numerator = native_numerator if summary["changed_positions"] == 0 else native_numerator + delta
        diagnostics = {"prefix_allocation_" + name: counts[i].detach()
                       for i, name in enumerate(SUMMARY_FIELDS)}
        diagnostics["prefix_allocation_delta_numerator"] = delta.detach()
        diagnostics["prefix_allocation_native_numerator"] = native_numerator.detach()
        return numerator, diagnostics


def install_prefix_allocation(model, *, relative_cap=RELATIVE_CAP):
    """Install after native K+, with no other objective modifier enabled."""
    if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
            or getattr(model, "first_error_redistribution", None) is None):
        raise ValueError("Prefix allocation requires installed native BF16 K+")
    conflicts = ("prefix_allocation", "native_prefix_roles", "prefix_guard", "role_loss_extension",
                 "point_loss_extension", "recent_repair_retention", "survival_head_extension",
                 "adversarial_block", "successor_shuffle")
    if any(getattr(model, name, None) is not None for name in conflicts):
        raise ValueError("Prefix allocation cannot combine objective extensions")
    extension = PrefixAllocation(relative_cap)
    model.prefix_allocation = extension
    return dict(extension.contract)
