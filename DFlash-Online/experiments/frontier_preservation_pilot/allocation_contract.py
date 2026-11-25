"""Prospective native-token contraction qualification, independent of efficacy.

Only the declared FP32 allocator is used to admit fast scores. FP64 arithmetic
and effective-direction contractions remain diagnostics, not alternate targets.
"""

import math

import torch

from .math import MASS_ATOL, MASS_RTOL, RELATIVE_CAP, allocate_prefix, require


def numerical_contract():
    """JSON-compatible rules sealed before any new real-model score audit."""
    return dict(
        name='native_token_contraction_v2',
        reference='-dot_FP64(unweighted_native_token_FKL_gradient, FP32_w)/(2*Z_response)',
        allocator_dtype='torch.float32',
        reference_dot_dtype='torch.float64',
        relative_cap=RELATIVE_CAP,
        mass_atol=MASS_ATOL,
        mass_rtol=MASS_RTOL,
        gate_a='finite; positive; block mass/cap; unchanged repair; P_union_E only; native Z and factor 1/2 once',
        gate_b_relative_beta=0.025,
        gate_b='abs(a_fast-a_reference)<=0.025*beta+MASS_ATOL+MASS_RTOL*abs(beta)',
        repeatability='both fast evaluations vs reference and each other satisfy Gate B',
        gate_c_alignment_fraction=0.9,
        gate_c='G_fast>=0.9*G_reference-sum((MASS_ATOL+MASS_RTOL*abs(beta))*abs(c_reference))',
        reference_center='FP32 reference scores promoted to FP64; subtract pool minimum before weighted centering',
        constant_reference='exactly constant only; abs(a_fast-beta)<=MASS_ATOL+MASS_RTOL*abs(beta)',
        activity='at least one selected nonconstant reference pool has nonzero coefficient movement in both fast evaluations',
        selection='per response: lowest native block index with >=2 prefix positions, then longest distinct eligible pool; ties by block index',
        responses=2,
        native_block_size=16,
        maximum_selected_pools=4,
        maximum_reference_backwards_per_audit=60,
        maximum_reference_backwards_including_one_correction=120,
        selection_requires_each_response=True,
        fast_evaluations=2,
        direction_cast='record represented effective direction and scale; undo scale; secondary oracle contraction never replaces FP32-w target',
        historical_hidden_state_gate=dict(
            atol=1e-5, rtol=0.016, status='failed', production_veto=False
        ),
        aggregate_gradient_diagnostic_is_admission_gate=False,
        efficacy_is_admission_gate=False,
    )


def _flat_prefix(prefix):
    require(
        isinstance(prefix, torch.Tensor) and prefix.dtype == torch.bool,
        'Prefix must be boolean',
    )
    require(
        prefix.ndim >= 2 and prefix.shape[-1] == 16, 'Require native 16-slot blocks'
    )
    flat = prefix.detach().reshape(-1, 16)
    positions = torch.arange(16, device=flat.device)
    expected = (positions > 0) & (positions <= flat.sum(-1, keepdim=True))
    require(
        torch.equal(flat, expected),
        'Prefix must be consecutive native positions after the anchor',
    )
    return flat


@torch.no_grad()
def select_pools(prefixes, response_ids=None):
    """Select complete pools from roles only; no score argument is accepted."""
    require(len(prefixes) == 2, 'Require exactly two responses')
    if response_ids is not None:
        require(
            len(response_ids) == 2 and all(isinstance(x, str) for x in response_ids),
            'Invalid response IDs',
        )
    pools, coverage = [], []
    for response_index, prefix in enumerate(prefixes):
        flat = _flat_prefix(prefix)
        lengths = flat.sum(-1).cpu().tolist()
        eligible = [i for i, length in enumerate(lengths) if length >= 2]
        chosen = eligible[:1]
        distinct = [i for i in eligible if i not in chosen]
        if distinct:
            chosen.append(min(distinct, key=lambda i: (-lengths[i], i)))
        coverage.append(
            dict(
                response_index=response_index,
                eligible_pools=len(eligible),
                selected_pools=len(chosen),
            )
        )
        for index in chosen:
            pools.append(
                dict(
                    response_index=response_index,
                    block_index=index,
                    positions=list(range(1, lengths[index] + 1)),
                )
            )
    count = sum(len(pool['positions']) for pool in pools)
    require(count <= 60, 'Selected pools exceed the declared reference-backward cap')
    covered = all(item['selected_pools'] > 0 for item in coverage)
    return dict(
        contract='native_token_contraction_v2',
        status='selected' if covered else 'coverage_blocked',
        passed=covered,
        response_ids=None if response_ids is None else list(response_ids),
        pools=pools,
        coverage=coverage,
        reference_backward_count=count,
        selected_before_scores=True,
    )


@torch.no_grad()
def validate_native_invariants(
    scores,
    coefficients,
    roles,
    repair_coefficients,
    reference_repair_coefficients,
    *,
    direct_loss_mask,
    response_denominator,
    accumulation_factor=0.5,
):
    """Gate A on one full response and one fast evaluation, including repair."""
    beta, prefix, repair = (roles[key].detach() for key in ('beta', 'prefix', 'repair'))
    arrays = (
        scores,
        coefficients,
        beta,
        prefix,
        repair,
        repair_coefficients,
        reference_repair_coefficients,
        direct_loss_mask,
    )
    require(
        all(isinstance(x, torch.Tensor) and x.shape == beta.shape for x in arrays),
        'Native invariant shapes differ',
    )
    require(
        prefix.dtype == repair.dtype == direct_loss_mask.dtype == torch.bool,
        'Teaching masks must be boolean',
    )
    require(
        all(
            value.dtype != torch.bool
            for value in (
                scores,
                coefficients,
                beta,
                repair_coefficients,
                reference_repair_coefficients,
            )
        ),
        'Scores and teaching weights cannot be boolean',
    )
    require(
        not isinstance(accumulation_factor, bool),
        'Accumulation factor cannot be boolean',
    )
    require(
        not isinstance(response_denominator, bool),
        'Response denominator cannot be boolean',
    )
    actual_z = torch.as_tensor(response_denominator, device=beta.device)
    frozen_z = torch.as_tensor(roles['denominator'], device=beta.device)
    require(
        actual_z.numel() == frozen_z.numel() == 1, 'Response denominator must be scalar'
    )
    require(
        actual_z.dtype != torch.bool and frozen_z.dtype != torch.bool,
        'Response denominator cannot be boolean',
    )
    base, candidate = beta.float(), coefficients.detach().float()
    # The authored eight-block reduction fixes the actual denominator. A full
    # response reduction may differ by FP32 rounding; it is only a mathematical
    # cross-check under the inherited invariant tolerance.
    expected_z = (base * (prefix | repair)).sum() + 1e-6
    before, after = (base * prefix).sum(-1), (candidate * prefix).sum(-1)
    tolerance = MASS_ATOL + MASS_RTOL * base.abs()
    checks = dict(
        allocator_fp32=scores.dtype
        == coefficients.dtype
        == beta.dtype
        == torch.float32,
        finite=all(
            bool(torch.isfinite(x).all())
            for x in (
                scores,
                candidate,
                base,
                repair_coefficients,
                reference_repair_coefficients,
                actual_z,
                frozen_z,
            )
        ),
        disjoint_teaching_roles=not bool((prefix & repair).any()),
        positive_weights=bool(
            (base[prefix | repair] > 0).all()
            and (candidate[prefix] > 0).all()
            and (repair_coefficients[repair] > 0).all()
        ),
        block_mass=bool(
            torch.isclose(after, before, atol=MASS_ATOL, rtol=MASS_RTOL).all()
        ),
        relative_cap=bool(
            (
                (candidate[prefix] >= 0.75 * base[prefix] - tolerance[prefix])
                & (candidate[prefix] <= 1.25 * base[prefix] + tolerance[prefix])
            ).all()
        ),
        nonprefix_preservation_unchanged=torch.equal(candidate[~prefix], base[~prefix]),
        repair_unchanged=torch.equal(
            repair_coefficients[repair], reference_repair_coefficients[repair]
        ),
        direct_support_exact=torch.equal(direct_loss_mask, prefix | repair),
        native_response_denominator=bool(
            (actual_z == frozen_z).all()
            and torch.isclose(
                frozen_z, expected_z, atol=MASS_ATOL, rtol=MASS_RTOL
            ).all()
        ),
        accumulation_once=math.isfinite(float(accumulation_factor))
        and float(accumulation_factor) == 0.5,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        failures=[key for key, passed in checks.items() if not passed],
        denominator=float(frozen_z),
        full_sum_denominator=float(expected_z),
        native_vs_full_sum_difference=float(frozen_z - expected_z),
        prefix_positions=int(prefix.sum()),
        repair_positions=int(repair.sum()),
        maximum_block_mass_error=(
            float((after - before).abs().max()) if before.numel() else 0.0
        ),
    )


def _pool_vector(value, name, *, dtype):
    require(
        isinstance(value, torch.Tensor) and value.ndim == 1,
        name + ' must be a one-dimensional tensor',
    )
    require(value.dtype != torch.bool, name + ' cannot be boolean')
    result = value.detach().to(device='cpu', dtype=dtype)
    require(torch.isfinite(result).all(), name + ' is nonfinite')
    return result


def _center64(scores, beta):
    offsets = scores.double() - scores.double().min()
    return offsets - (beta.double() * offsets).sum() / beta.double().sum()


def _allocate64(scores, beta):
    center = _center64(scores, beta)
    maximum = center.abs().max()
    return (
        beta.double()
        if maximum == 0
        else beta.double() * (1 + RELATIVE_CAP * center / maximum)
    )


@torch.no_grad()
def evaluate_pool(beta, reference_scores_fp64, fast_scores, repeated_fast_scores):
    """Gates B/C on every coordinate, plus a separate FP64 arithmetic record."""
    base = _pool_vector(beta, 'beta', dtype=torch.float32)
    raw = _pool_vector(reference_scores_fp64, 'reference', dtype=torch.float64)
    first = _pool_vector(fast_scores, 'first fast score', dtype=torch.float32)
    second = _pool_vector(
        repeated_fast_scores, 'repeated fast score', dtype=torch.float32
    )
    require(
        2 <= base.numel() <= 15
        and raw.shape == first.shape == second.shape == base.shape,
        'Require one complete eligible native pool',
    )
    require((base > 0).all(), 'Pool beta must be positive')
    ref = raw.float()
    require(torch.isfinite(ref).all(), 'FP32 reference conversion overflowed')
    mask = torch.ones_like(base, dtype=torch.bool)
    reference, _ = allocate_prefix(ref, base, mask)
    allocations = [allocate_prefix(value, base, mask)[0] for value in (first, second)]
    center = _center64(ref, base)
    constant = bool((ref == ref[0]).all())
    t = MASS_ATOL + MASS_RTOL * base.double().abs()
    bound = 0.025 * base.double() + t
    fidelity = [
        bool(((a.double() - reference.double()).abs() <= bound).all())
        for a in allocations
    ]
    mutual = bool(
        ((allocations[0].double() - allocations[1].double()).abs() <= bound).all()
    )
    g_ref = float(((reference.double() - base.double()) * center).sum())
    tol_g = float((t * center.abs()).sum())
    gains = [float(((a.double() - base.double()) * center).sum()) for a in allocations]
    constant_checks = [
        bool(((a.double() - base.double()).abs() <= t).all()) for a in allocations
    ]
    alignment = (
        constant_checks if constant else [gain >= 0.9 * g_ref - tol_g for gain in gains]
    )
    active = [bool((a != base).any()) for a in allocations]
    raw_center = _center64(raw, base)
    fp64_ref = _allocate64(ref, base)
    return dict(
        passed=all(fidelity) and mutual and all(alignment),
        gate_b=dict(
            passed=all(fidelity) and mutual,
            fast_vs_reference=fidelity,
            fast_repeat_vs_first=mutual,
            per_position_bound=bound.tolist(),
        ),
        gate_c=dict(
            passed=all(alignment),
            reference_constant=constant,
            fast_passed=alignment,
            reference_alignment=g_ref,
            fast_alignment=gains,
            tolerance=tol_g,
            required_alignment=0.9 * g_ref - tol_g,
            constant_reference_checks=constant_checks if constant else None,
        ),
        active_fast=active,
        nonconstant_reference_and_both_fast_active=(not constant and all(active)),
        beta=base.tolist(),
        reference_scores_fp64=raw.tolist(),
        reference_scores_fp32=ref.tolist(),
        fast_scores_fp32=first.tolist(),
        repeated_fast_scores_fp32=second.tolist(),
        reference_coefficients_fp32=reference.tolist(),
        fast_coefficients_fp32=[a.tolist() for a in allocations],
        independent_fp64_coefficients_from_fp32_scores=fp64_ref.tolist(),
        independent_fp64_coefficients_from_raw_scores=_allocate64(raw, base).tolist(),
        independent_fp64_maximum_coefficient_difference=float(
            (reference.double() - fp64_ref).abs().max()
        ),
        raw_fp64_score_contrast=float(raw.max() - raw.min()),
        fp32_score_contrast=float(ref.double().max() - ref.double().min()),
        raw_nonconstant_became_fp32_constant=bool((raw != raw[0]).any()) and constant,
        maximum_centered_reference_score=float(center.abs().max()),
        maximum_centered_raw_reference_score=float(raw_center.abs().max()),
        maximum_coefficient_differences=[
            float((a.double() - reference.double()).abs().max()) for a in allocations
        ],
        repeat_maximum_coefficient_difference=float(
            (allocations[0].double() - allocations[1].double()).abs().max()
        ),
        reference_mass_residual=float((reference.double() - base.double()).sum()),
        fast_mass_residuals=[
            float((a.double() - base.double()).sum()) for a in allocations
        ],
    )


def qualify_selected_pools(selection, pool_receipts, invariant_receipts):
    """No successful subset can replace the prospectively selected full pools."""
    require(
        selection.get('selected_before_scores') is True,
        'Selection was not sealed before scores',
    )
    expected = [
        (p['response_index'], p['block_index'], tuple(p['positions']))
        for p in selection['pools']
    ]
    observed = [
        (p['response_index'], p['block_index'], tuple(p['positions']))
        for p in pool_receipts
    ]
    require(
        expected == observed,
        'Pool receipts must cover the exact selected pools in order',
    )
    require(
        len(invariant_receipts) == 2
        and all(len(row) == 2 for row in invariant_receipts),
        'Require both response invariants for both fast evaluations',
    )
    gate_a = all(
        row['passed'] is True for repeat in invariant_receipts for row in repeat
    )
    gate_b = all(row['gate_b']['passed'] is True for row in pool_receipts)
    gate_c = all(row['gate_c']['passed'] is True for row in pool_receipts)
    active = any(
        row['nonconstant_reference_and_both_fast_active'] is True
        for row in pool_receipts
    )
    coverage = selection.get('passed') is True and all(
        row['selected_pools'] >= 1 for row in selection['coverage']
    )
    checks = dict(
        coverage=coverage,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        active_nonconstant_pool=active,
    )
    return dict(
        status=(
            'numerically_qualified'
            if all(checks.values())
            else 'numerical_qualification_failed'
        ),
        passed=all(checks.values()),
        checks=checks,
        failures=[name for name, passed in checks.items() if not passed],
        selected_pools=len(pool_receipts),
        reference_backward_count=selection['reference_backward_count'],
        old_hidden_state_gate='failed_retained_diagnostic',
        production_qualified=False,
    )


@torch.no_grad()
def direction_rounding_diagnostics(
    reference_scores_fp64, effective_reference_scores_fp64, fast_scores, *, metadata
):
    """Separate direction casting from route error without introducing a gate.

    The caller binds the exact represented direction and inverse scaling. Both
    reference vectors already include the same response normalization once.
    """
    primary = _pool_vector(
        reference_scores_fp64, 'FP32-w reference', dtype=torch.float64
    )
    effective = _pool_vector(
        effective_reference_scores_fp64,
        'effective-direction reference',
        dtype=torch.float64,
    )
    fast = _pool_vector(fast_scores, 'fast score', dtype=torch.float64)
    require(
        primary.shape == effective.shape == fast.shape, 'Direction score shapes differ'
    )
    require(
        isinstance(metadata, dict) and metadata,
        'Exact direction binding metadata is required',
    )
    cast, route, total = effective - primary, fast - effective, fast - primary
    return dict(
        direction_metadata=metadata,
        primary_target='represented_FP32_w',
        secondary_target='identical_effective_fast_direction_with_rescaling_undone',
        direction_cast_score_difference=cast.tolist(),
        derivative_route_score_difference=route.tolist(),
        total_score_difference=total.tolist(),
        maximum_absolute_direction_cast_difference=float(cast.abs().max()),
        maximum_absolute_route_difference=float(route.abs().max()),
        comparison_is_pure_derivative_route=metadata.get(
            'effective_direction_equals_primary'
        )
        is True,
        admission_gate=False,
    )
