"""Capped SHORT-to-LONG redistribution inside native K+'s high-priority mass.

Only per-depth sufficient statistics cross head chunks. Membership and transfer
arithmetic are detached; the ordinary KL numerators retain their native graph.
The existing objective applies its denominator and epsilon after this reducer.
"""

import torch

from .first_error import valid_prefix_mask
from .kplus import (
    KPlusRedistribution, MASS_ATOL, MASS_RTOL, first_error_groups,
    group_terms, kplus_contract, redistribute_response,
)


DEPTH_TERM_NAMES = (
    "short_count", "long_count", "censored_count",
    "short_weight", "long_weight", "censored_weight",
    "short_numerator", "long_numerator",
)


def second_follower_contract(*, disabled=False):
    """Fixed scientific rule; disabled mode is for discarded qualification only."""
    if type(disabled) is not bool:
        raise ValueError("disabled must be a boolean qualification switch")
    return {
        "name": "kplus_second_follower_v1",
        "native": kplus_contract(),
        "disabled": disabled,
        "disabled_scope": "qualification_only_never_production",
        "short": "first_follower_correct_second_valid_follower_wrong",
        "long": "first_and_second_valid_followers_correct",
        "censored": "first_follower_correct_second_slot_unavailable",
        "pooling": "same_first_error_depth_across_all_response_chunks",
        "transfer": "min(c*B_S/8,c*B_L/8,(c-1)*B_S)",
        "relative_cap": 0.125,
        "donor_floor": "ordinary_K_first_error_weight",
        "numerator": "N_Kplus+sum_j(t_j*(N_LONG_j/B_LONG_j-N_SHORT_j/B_SHORT_j))",
        "neutral_branch": "return_original_native_Kplus_numerator_tensor",
        "denominator": "unchanged_native_selected_weight_sum_plus_1e-6_once",
        "coefficient_arithmetic": "detached_FP32",
        "kl_arithmetic": "unchanged_native_BF16_rows_FP32_weighted_reductions",
        "additional_model_forwards": 0,
        "additional_vocabulary_projections": 0,
        "mass_conservation_rtol": MASS_RTOL,
        "mass_conservation_atol": MASS_ATOL,
    }


@torch.no_grad()
def second_follower_groups(proposals, labels, weights, selection, eos_token_ids=()):
    """Classify only native high-priority first errors, never their followers."""
    if labels.ndim != 3 or labels.shape[0] != 1:
        raise ValueError("Second follower requires one response of complete blocks")
    if any(value.shape != labels.shape for value in (proposals, weights, selection)):
        raise ValueError("Second-follower block tensors must be aligned")
    if weights.dtype != torch.float32 or weights.requires_grad:
        raise ValueError("Second follower requires detached native FP32 weights")
    _, high = first_error_groups(proposals, labels, weights, selection, eos_token_ids)
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    second_valid = torch.zeros_like(valid)
    second_correct = torch.zeros_like(valid)
    second_valid[..., :-2] = valid[..., 2:]
    second_correct[..., :-2] = (
        valid[..., 2:] & (proposals.detach()[..., 2:] == labels[..., 2:])
    )
    short = high & second_valid & ~second_correct
    long = high & second_correct
    censored = high & ~second_valid
    return short, long, censored


def second_follower_terms(token_loss, proposals, labels, weights, selection, eos_token_ids=()):
    """Six unchanged native scalars followed by eight additive depth vectors."""
    native = group_terms(token_loss, proposals, labels, weights, selection, eos_token_ids)
    short, long, censored = second_follower_groups(
        proposals, labels, weights, selection, eos_token_ids,
    )
    axes = tuple(range(weights.ndim - 1))
    class_weights = tuple(weights.detach() * mask for mask in (short, long, censored))
    counts = tuple(mask.sum(dim=axes).float() for mask in (short, long, censored))
    masses = tuple(value.sum(dim=axes) for value in class_weights)
    numerators = tuple((token_loss * value).sum(dim=axes) for value in class_weights[:2])
    return (*native, *counts, *masses, *numerators)


@torch.no_grad()
def _transfer_terms(depths, coefficient, *, disabled):
    short, long = depths["short_weight"], depths["long_weight"]
    native_short, native_long = coefficient * short, coefficient * long
    mixed = (short > 0) & (long > 0)
    eligible = mixed & (coefficient > 1)
    proposed = torch.minimum(torch.minimum(native_short / 8, native_long / 8),
                             (coefficient - 1) * short)
    proposed = torch.where(eligible, proposed, torch.zeros_like(proposed))
    transferred = torch.zeros_like(proposed) if disabled else proposed
    active = transferred > 0
    short_after, long_after = native_short - transferred, native_long + transferred
    tolerance = MASS_ATOL + MASS_RTOL * (native_short + native_long)
    valid = (
        torch.isfinite(transferred) & (transferred >= 0)
        & (transferred <= native_short / 8 + tolerance)
        & (transferred <= native_long / 8 + tolerance)
        & (short_after >= short - tolerance)
        & ((short_after + long_after - native_short - native_long).abs() <= tolerance)
    )
    if not bool(valid.all()):
        raise RuntimeError("Second-follower cap, donor floor or depth mass failed")
    short_scale, long_scale = torch.ones_like(short), torch.ones_like(long)
    short_scale[active] = 1 - transferred[active] / native_short[active]
    long_scale[active] = 1 + transferred[active] / native_long[active]
    return dict(
        native_short_weight=native_short, native_long_weight=native_long,
        native_censored_weight=coefficient * depths["censored_weight"],
        rule_transfer_weight=proposed, transferred_weight=transferred,
        short_weight_after=short_after, long_weight_after=long_after,
        short_relative_scale=short_scale, long_relative_scale=long_scale,
        short_coefficient=coefficient * short_scale,
        long_coefficient=coefficient * long_scale,
        mixed=mixed.float(), active=active.float(),
        mass_balance_error=short_after + long_after - native_short - native_long,
    )


def reduce_second_follower(terms, *, disabled=False):
    """Return candidate numerator, flat detached metrics, and a depth receipt.

``terms`` are the existing objective's five base terms plus this module's
fourteen extension terms. No epsilon is applied to the learning numerator.
``kplus_*`` diagnostics continue to describe the same-state native control.
The optional disabled mode still records proposed exposure but applies none.
"""
    second_follower_contract(disabled=disabled)
    if len(terms) != 19:
        raise ValueError("Second-follower reducer requires all response depth statistics")
    native, diagnostics = redistribute_response(terms[0], terms[1], terms[5:11])
    depths = dict(zip(DEPTH_TERM_NAMES, terms[11:]))
    shapes = {tuple(value.shape) for value in depths.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("Second-follower response statistics must be equal depth vectors")
    with torch.no_grad():
        high_weight = sum(depths[name].sum() for name in
                          ("short_weight", "long_weight", "censored_weight"))
        high_count = sum(depths[name].sum() for name in
                         ("short_count", "long_count", "censored_count"))
        if not bool(torch.isclose(high_weight, terms[6], rtol=MASS_RTOL, atol=MASS_ATOL)):
            raise RuntimeError("Second-follower classes do not partition native high weight")
        if not bool(high_count == terms[10]):
            raise RuntimeError("Second-follower classes do not partition native high counts")
    allocation = _transfer_terms(depths, diagnostics["kplus_coefficient2"], disabled=disabled)
    active = allocation["transferred_weight"] > 0
    candidate = native
    if bool(active.any()):
        correction = (allocation["transferred_weight"][active] * (
            depths["long_numerator"][active] / depths["long_weight"][active]
            - depths["short_numerator"][active] / depths["short_weight"][active]
        )).sum()
        candidate = native + correction

    with torch.no_grad():
        detached_depths = {name: value.detach() for name, value in {**depths, **allocation}.items()}
        moved = allocation["transferred_weight"].sum()
        native_high = diagnostics["kplus_coefficient2"] * diagnostics["kplus_W2"]
        prefix = terms[1].detach() - diagnostics["kplus_W1"] - diagnostics["kplus_W2"]
        one = moved.new_ones(())
        totals = dict(
            response_count=one, disabled=one.new_tensor(float(disabled)),
            active_response=active.any().float(), mixed_strata=allocation["mixed"].sum(),
            active_strata=active.sum().float(), transferred_weight=moved,
            rule_transfer_weight=allocation["rule_transfer_weight"].sum(),
            selected_weight=terms[1].detach(), selected_denominator=terms[1].detach() + 1e-6,
            native_high_weight=native_high,
            native_low_weight=diagnostics["kplus_coefficient1"] * diagnostics["kplus_W1"],
            prefix_weight=prefix, prefix_weight_change=moved.new_zeros(()),
            low_weight_change=moved.new_zeros(()), selected_weight_change=moved.new_zeros(()),
            moved_share_selected=moved / (terms[1].detach() + 1e-6),
            moved_share_native_high=(moved / native_high if bool(native_high > 0) else moved.new_zeros(())),
            native_numerator=native.detach(), candidate_numerator=candidate.detach(),
            numerator_correction=(candidate.detach() - native.detach()),
            mass_balance_error=allocation["mass_balance_error"].sum(),
        )
        for name in ("short_count", "long_count", "censored_count", "short_weight", "long_weight", "censored_weight"):
            totals[name] = depths[name].sum().detach()
        for name in ("native_short_weight", "native_long_weight", "native_censored_weight"):
            totals[name] = allocation[name].sum()
        diagnostics.update({"second_follower_" + name: value.detach() for name, value in totals.items()})
        # Flat scalar fields use the existing original-response observation path.
        for position in range(next(iter(depths.values())).numel()):
            for name, value in detached_depths.items():
                diagnostics[f"second_follower_depth_{position}_{name}"] = value[position]
        receipt = dict(
            contract=second_follower_contract(disabled=disabled),
            totals={name: value.detach() for name, value in totals.items()},
            depths=detached_depths,
            native_high_coefficient=diagnostics["kplus_coefficient2"].detach(),
        )
    return candidate, diagnostics, receipt


class SecondFollowerRedistribution:
    """Existing first_error_redistribution API; no model hook or projection."""

    validate_response = staticmethod(KPlusRedistribution.validate_response)
    chunk_terms = staticmethod(second_follower_terms)

    def __init__(self, *, disabled=False):
        self.contract = second_follower_contract(disabled=disabled)
        self.disabled = disabled
        self.last_receipt = None

    def reduce_response(self, terms):
        numerator, metrics, self.last_receipt = reduce_second_follower(terms, disabled=self.disabled)
        return numerator, metrics


def install_second_follower(model, *, disabled=False):
    """Replace only the exact ordinary native K+ reducer (or an empty slot)."""
    contract = second_follower_contract(disabled=disabled)
    if model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16:
        raise ValueError("Second follower requires native BF16 osd_first_error")
    if any(getattr(model, name, None) is not None for name in (
            "prefix_guard", "point_loss_extension", "role_loss_extension",
            "survival_head_extension", "successor_shuffle", "adversarial_block")):
        raise ValueError("Second follower cannot combine objective extensions")
    existing = getattr(model, "first_error_redistribution", None)
    if existing is not None and type(existing) is not KPlusRedistribution:
        raise ValueError("Second follower may replace only the exact native K+ reducer")
    model.first_error_redistribution = SecondFollowerRedistribution(disabled=disabled)
    return contract
