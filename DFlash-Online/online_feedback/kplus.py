"""Response-wide first-error redistribution on the unchanged native K loss.

The experiment installs this reducer explicitly. It consumes the existing
current-draft proposals and ordinary KL rows; no model or vocabulary projection
is added. Only additive scalars cross projection-chunk boundaries.
"""

import torch

from .first_error import valid_prefix_mask


MASS_RTOL = 2e-6
MASS_ATOL = 2e-7


def kplus_contract():
    """Serializable, fixed scientific rule for experiment/state receipts."""
    return {
        "name": "kplus_first_error_redistribution_v1",
        "base_objective": "osd_first_error",
        "redistribution_fraction": 0.5,
        "priorities": [1, 2],
        "priority2": "immediate_valid_canonical_successor_matches_current_argmax",
        "pooling": "all_sampled_blocks_in_one_response",
        "numerator": "N_K + 0.5*(W1*N2-W2*N1)/(W1+2*W2)",
        "neutral_branch": "original_N_K_for_no_errors_or_one_priority_class",
        "denominator": "unchanged_native_selected_weight_sum_plus_1e-6",
        "mass_conservation_rtol": MASS_RTOL,
        "mass_conservation_atol": MASS_ATOL,
        "additional_model_forwards": 0,
        "additional_vocabulary_projections": 0,
    }


@torch.no_grad()
def first_error_groups(proposals, labels, weights, selection, eos_token_ids=()):
    """Partition K's selected mismatches using only their immediate successor.

    The inherited contiguous validity mask stops at the first gap and includes
    canonical EOS but excludes every position after it. It also excludes the
    seed slot. Consequently an EOS mismatch cannot receive priority two.
    """
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    successor_correct = torch.zeros_like(valid)
    successor_correct[..., :-1] = (
        valid[..., 1:] & (proposals.detach()[..., 1:] == labels[..., 1:])
    )
    errors = selection & (proposals.detach() != labels)
    priority2 = errors & successor_correct
    priority1 = errors & ~successor_correct
    return priority1, priority2


def group_terms(token_loss, proposals, labels, weights, selection, eos_token_ids=()):
    """Return W1,W2,N1,N2 and error counts as additive chunk scalars.

    Group membership and weights are detached; N1/N2 retain the very same
    ordinary KL graph as the native numerator. FP32 weights retain the native
    BF16-row/FP32-reduction arithmetic in production.
    """
    priority1, priority2 = first_error_groups(
        proposals, labels, weights, selection, eos_token_ids,
    )
    weights1 = weights.detach() * priority1
    weights2 = weights.detach() * priority2
    return (
        weights1.sum(), weights2.sum(),
        (token_loss * weights1).sum(), (token_loss * weights2).sum(),
        priority1.sum().float(), priority2.sum().float(),
    )


def redistribute_response(native_numerator, selected_weight, terms):
    """Apply one correction after all response chunks have been summed.

    Returning native_numerator itself in the neutral branch is intentional:
    algebraically adding zero can still alter BF16 backward rounding.
    """
    weight1, weight2, numerator1, numerator2, count1, count2 = terms
    mixed = bool((weight1 > 0) & (weight2 > 0))
    coefficient1 = weight1.new_ones(())
    coefficient2 = weight2.new_ones(())
    numerator = native_numerator
    if mixed:
        priority_weight = weight1 + 2 * weight2
        coefficient1 = 1 - 0.5 * weight2 / priority_weight
        coefficient2 = 1 + 0.5 * weight1 / priority_weight
        numerator = native_numerator + 0.5 * (
            weight1 * numerator2 - weight2 * numerator1
        ) / priority_weight

    original_mass = weight1 + weight2
    new_mass = weight1 * coefficient1 + weight2 * coefficient2
    if not bool(torch.isclose(new_mass, original_mass, rtol=MASS_RTOL, atol=MASS_ATOL)):
        raise RuntimeError("K+ first-error coefficient mass was not conserved")
    moved_mass = 0.5 * (
        weight1 * (coefficient1 - 1).abs() + weight2 * (coefficient2 - 1).abs()
    )
    moved_share = moved_mass / original_mass if mixed else original_mass.new_zeros(())
    diagnostics = {
        "kplus_N_K": native_numerator.detach(),
        "kplus_S": selected_weight.detach(),
        "kplus_W1": weight1.detach(),
        "kplus_W2": weight2.detach(),
        "kplus_N1": numerator1.detach(),
        "kplus_N2": numerator2.detach(),
        "kplus_N_new": numerator.detach(),
        "kplus_priority1_error_count": count1.detach(),
        "kplus_priority2_error_count": count2.detach(),
        "kplus_coefficient1": coefficient1.detach(),
        "kplus_coefficient2": coefficient2.detach(),
        "kplus_first_error_weight_after": new_mass.detach(),
        "kplus_conservation_error": (new_mass - original_mass).detach(),
        "kplus_error_weight_moved": moved_mass.detach(),
        "kplus_error_weight_moved_share": moved_share.detach(),
        "kplus_mixed_response": original_mass.new_tensor(float(mixed)),
        "kplus_response_count": original_mass.new_ones(()),
    }
    return numerator, diagnostics


class KPlusRedistribution:
    """Narrow optional extension used only by the K+ experiment installer."""

    @staticmethod
    def validate_response(input_ids):
        if input_ids.shape[0] != 1:
            raise ValueError("K+ redistribution requires one response per microbatch")

    chunk_terms = staticmethod(group_terms)

    @staticmethod
    def reduce_response(terms):
        return redistribute_response(terms[0], terms[1], terms[5:])


def install_kplus(model):
    """Attach K+ to an existing native K model without replacing any module.

    Call once after building the experiment's native runtime. The return value
    is the fixed contract to retain in its configuration and state receipts.
    """
    if model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16:
        raise ValueError("K+ requires native BF16 osd_first_error")
    if getattr(model, "prefix_guard", None) is not None:
        raise ValueError("K+ cannot be combined with a prefix correction")
    if getattr(model, "first_error_redistribution", None) is not None:
        raise ValueError("K+ redistribution is already installed")
    model.first_error_redistribution = KPlusRedistribution()
    return kplus_contract()
