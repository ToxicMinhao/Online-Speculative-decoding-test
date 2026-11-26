"""Bounded C-reference qualification and scalar Point2A comparisons.

The native oracle, allocator, and numerical admission criteria are inherited.
Only the reference binding and the prospectively smaller pool sample change.
No full parameter vector is written by these helpers.
"""

import math
from pathlib import Path

import torch

from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json
from .allocation_contract import (
    direction_rounding_diagnostics, evaluate_pool, numerical_contract,
    qualify_selected_pools, select_pools, validate_native_invariants,
)
from .allocation_qualify import (
    direction_metadata, json_diagnostics, selected_values, serial_references,
    vector_identity,
)
from .math import require
from .native import rng_digest
from .point1_math import fp32_difference, score_contrast


CONTRASTS = (('F_C', 'F_BETA'), ('F_C', 'C'), ('F_BETA', 'C'))


def point2a_numerical_contract():
    result = numerical_contract()
    result.update(
        name='native_token_contraction_v2_point2a_C_reference',
        reference='-dot_FP64(unweighted_native_token_FKL_gradient, FP32_w_C)/(2*Z_response)',
        selection='per response: lowest native block index with >=2 prefix positions; complete pool',
        maximum_selected_pools=2,
        maximum_reference_backwards_per_audit=30,
        maximum_reference_backwards_including_one_correction=60,
        expected_selected_pool_sizes=[13, 15],
    )
    return result


def require_c_reference(*, reference_name, parent_identity, current_identity,
                        captured_gradient_identity, supplied_gradient_identity):
    """Reject a beta gradient or a post-C optimizer context before previewing."""
    require(reference_name == 'C', 'Point2A requires the observed native C reference')
    require(bool(parent_identity) and parent_identity == current_identity,
            'C shadow must use the fully restored parent state, not post-C state')
    require(bool(captured_gradient_identity)
            and captured_gradient_identity == supplied_gradient_identity,
            'C shadow gradient differs from the captured native preclip gradient')
    return dict(reference='C', parent_identity=parent_identity,
                native_preclip_gradient=captured_gradient_identity,
                parent_restored_before_shadow=True)


@torch.no_grad()
def select_reference_pools(snapshots, response_ids=None, expected_counts=(13, 15)):
    """Select from frozen roles before either reference's scores exist."""
    inherited = select_pools([item['roles']['prefix'] for item in snapshots], response_ids)
    pools = []
    for index in range(2):
        rows = [p for p in inherited['pools'] if p['response_index'] == index]
        if rows:
            pools.append(rows[0])
    counts = [len(p['positions']) for p in pools]
    require(inherited['passed'] and len(pools) == 2,
            'Each frozen response requires an eligible complete prefix pool')
    require(sum(counts) <= 30, 'Point2A reference allowance exceeds 30 backwards')
    if expected_counts is not None:
        require(counts == list(expected_counts),
                'Selected pool coverage changed; authenticate the original window')
    coverage = []
    for index, item in enumerate(inherited['coverage']):
        lengths = snapshots[index]['roles']['prefix'].reshape(-1, 16).sum(-1)
        coverage.append(dict(item, selected_pools=1,
                             singleton_pools_in_response=int((lengths == 1).sum()),
                             selected_singleton_pools=0))
    return dict(contract=point2a_numerical_contract()['name'], status='selected',
                passed=True, response_ids=inherited['response_ids'], pools=pools,
                coverage=coverage, reference_backward_count=sum(counts),
                expected_position_counts=None if expected_counts is None else list(expected_counts),
                selected_before_scores=True)


class TokenLedgerAdapter:
    """Keep the oracle unchanged and reserve at its actual backward boundary."""

    def __init__(self, ledger):
        self.ledger, self.pending, self.reservation = ledger, None, None

    def reserve(self, occurrence):
        require(self.pending is None, 'Previous oracle occurrence is unfinished')
        self.pending = dict(occurrence)

    def mark(self, status):
        require(self.pending is not None, 'No oracle occurrence is pending')
        if status == 'backward_started':
            require(self.reservation is None, 'Oracle backward was already reserved')
            self.reservation = self.ledger.reserve_token(dict(self.pending))
        elif status == 'backward_completed':
            require(self.reservation is not None, 'Oracle backward has no reservation')
            self.ledger.complete_token(self.reservation)
        elif status == 'completed':
            require(self.reservation is not None, 'Oracle occurrence has no backward')
            self.pending, self.reservation = None, None
        else:
            raise ValueError('Unexpected oracle ledger status: ' + status)


def qualify_direction(scorer, direction, output, ledger, work, selection,
                      verify_callback=None):
    """Qualify fixed w_C at the restored parent; return receipt and first fast pass."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = scorer.snapshots
    expected = select_reference_pools(snapshots, selection['response_ids'],
                                      selection['expected_position_counts'])
    require(selection == expected, 'Point2A pool selection changed after sealing')
    require(len(direction) == len(scorer.parameters)
            and all(w.dtype == torch.float32 for w in direction),
            'Primary C direction must contain all represented FP32 parameters')
    primary_before = vector_identity(scorer.parameter_names, direction)
    frozen_repair = [s['repair_coefficients'].detach().clone() for s in snapshots]
    repeats, invariants = [], []
    fast_rng = capture_rng()
    for repeat in range(2):
        restore_rng(fast_rng)
        work.start('point2a_fast_scoring_window')
        fast = scorer.fast_scores(direction)
        work.finish('point2a_fast_scoring_window')
        require(rng_digest(capture_rng()) == rng_digest(fast_rng),
                'Fast scoring changed the frozen parent RNG')
        if repeat:
            require(fast['direction_scale'] == repeats[0]['direction_scale']
                    and len(fast['effective_direction']) == len(repeats[0]['effective_direction'])
                    and all(torch.equal(a, b) for a, b in zip(
                        fast['effective_direction'], repeats[0]['effective_direction'])),
                    'Repeated fast scoring changed its represented direction')
        checks = [validate_native_invariants(
            fast['scores'][i], fast['allocations'][i], s['roles'],
            fast['repair_coefficients'][i], frozen_repair[i],
            direct_loss_mask=fast['direct_loss_masks'][i],
            response_denominator=fast['response_denominators'][i],
            accumulation_factor=fast['accumulation_factor'],
        ) for i, s in enumerate(snapshots)]
        invariants.append(checks)
        write_json(output / f'FAST_{repeat + 1}.json', dict(
            invariants=checks, allocation_diagnostics=json_diagnostics(fast['allocation_diagnostics']),
            attention_work=fast['attention_work']))
        if verify_callback is not None:
            verify_callback(f'FAST_{repeat + 1}_STATE.json')
        if repeat:
            del fast['effective_direction']
        repeats.append(fast)
    effective = repeats[0]['effective_direction']
    scale = float(repeats[0]['direction_scale'])
    metadata = direction_metadata(scorer.parameter_names, direction, effective, scale)
    require(metadata['primary'] == primary_before, 'Fast scoring mutated fixed w_C')
    metadata['fixed_primary_unchanged'] = True
    metadata['reference'] = 'C'
    write_json(output / 'DIRECTIONS.json', metadata)
    write_json(output / 'REFERENCE_BINDING.json', dict(
        selection=selection, directions=metadata,
        response_denominators=[float(s['roles']['denominator']) for s in snapshots],
        loss='Native BF16 unweighted token FKL; create_graph=False',
        head_layout='Native full eight-block chunk, including unchanged final remainder',
        normalization='Divide the FP64 native-gradient dot by 2*Z_response once outside backward'))
    rows = serial_references(scorer, snapshots, selection, direction, effective,
                             scale, TokenLedgerAdapter(ledger), work, output)
    pools = []
    for pool in selection['pools']:
        matching = [r for r in rows if r['response_index'] == pool['response_index']
                    and r['block_index'] == pool['block_index']]
        require([r['position'] for r in matching] == pool['positions'],
                'Serial oracle records lost selected pool order')
        reference = torch.tensor([r['reference_score_fp64'] for r in matching], dtype=torch.float64)
        secondary = torch.tensor([r['effective_reference_score_fp64'] for r in matching], dtype=torch.float64)
        i = pool['response_index']
        first, second = [selected_values(f['scores'][i], pool) for f in repeats]
        receipt = evaluate_pool(selected_values(snapshots[i]['roles']['beta'], pool),
                                reference, first, second)
        receipt.update(pool)
        receipt['direction_diagnostic'] = direction_rounding_diagnostics(
            reference, secondary, first, metadata=metadata)
        pools.append(receipt)
    gates = qualify_selected_pools(selection, pools, invariants)
    exact = {key: all(torch.equal(a, b) for a, b in zip(repeats[0][key], repeats[1][key]))
             for key in ('scores', 'allocations')}
    result = dict(**gates, selection=selection, selected_pool_receipts=pools,
                  invariants=invariants, direction_metadata=metadata,
                  repeat_exact=exact, reference='C',
                  singleton_reference_coverage=False)
    result['numerical_status'] = result['status']
    result['status'] = 'passed' if result['passed'] else 'failed'
    write_json(output / 'NUMERICAL_GATES.json', result)
    write_json(output / 'QUALIFICATION.json', result)
    if verify_callback is not None:
        verify_callback('COMPLETE_AUDIT_STATE.json')
    del repeats[0]['effective_direction']
    return result, repeats[0]


@torch.no_grad()
def coefficient_movement(candidate, reference, beta, prefix):
    require(candidate.shape == reference.shape == beta.shape == prefix.shape,
            'Coefficient movement shapes differ')
    require(prefix.dtype == torch.bool, 'Coefficient support must be boolean')
    require(all(torch.isfinite(x).all() for x in (candidate, reference, beta)),
            'Nonfinite coefficient movement')
    difference = fp32_difference(candidate, reference).double().reshape(-1, 16)
    base = beta.detach().float().double().reshape(-1, 16)
    mask = prefix.reshape(-1, 16)
    pools = []
    for index in range(mask.shape[0]):
        if not bool(mask[index].any()):
            continue
        delta, weights = difference[index][mask[index]], base[index][mask[index]]
        require((weights > 0).all(), 'Prefix pool beta must be positive')
        budget = float(weights.sum())
        pools.append(dict(block_index=index, positions=int(mask[index].sum()),
                          beta_budget=budget, coefficients_equal=bool((delta == 0).all()),
                          changed_positions=int((delta != 0).sum()),
                          maximum_relative_movement=float((delta.abs() / weights).max()),
                          half_l1_mass_moved_over_beta_budget=float(delta.abs().sum()) / (2 * budget),
                          signed_mass_residual=float(delta.sum())))
    return dict(coefficients_equal=torch.equal(candidate, reference),
                changed_positions=int((difference[mask] != 0).sum()),
                changed_nonprefix_positions=int((difference[~mask] != 0).sum()),
                maximum_relative_movement=max((p['maximum_relative_movement'] for p in pools), default=0.0),
                maximum_half_l1_mass_moved_over_beta_budget=max(
                    (p['half_l1_mass_moved_over_beta_budget'] for p in pools), default=0.0),
                per_pool=pools, movement_denominator='frozen beta_i; pool budget=sum_prefix beta_i',
                products_and_reductions='FP64 after FP32 coefficient subtraction')


@torch.no_grad()
def score_changes(candidate_scores, reference_scores, beta, prefix):
    require(candidate_scores.shape == reference_scores.shape == beta.shape == prefix.shape,
            'Score comparison shapes differ')
    require(prefix.dtype == torch.bool, 'Score support must be boolean')
    require(all(torch.isfinite(x).all() for x in (candidate_scores, reference_scores, beta)),
            'Nonfinite score comparison')
    candidate, reference, base = [x.detach().to(device='cpu', dtype=torch.float32).double().reshape(-1, 16)
                                  for x in (candidate_scores, reference_scores, beta)]
    masks = prefix.detach().cpu().reshape(-1, 16)
    pools = []
    for index, mask in enumerate(masks):
        if not bool(mask.any()):
            continue
        a, b, weights = candidate[index][mask], reference[index][mask], base[index][mask]
        require((weights > 0).all(), 'Prefix pool beta must be positive')
        centered = []
        for values in (a, b):
            offsets = values - values.min()
            centered.append(offsets - (weights * offsets).sum() / weights.sum())
        ca, cb = centered
        pairs = torch.triu(torch.ones((len(a), len(a)), dtype=torch.bool, device=a.device), diagonal=1)
        order_a = torch.sign(ca[:, None] - ca[None, :])[pairs]
        order_b = torch.sign(cb[:, None] - cb[None, :])[pairs]
        pools.append(dict(
            block_index=index, positions=len(a), raw_scores_equal=torch.equal(a, b),
            raw_maximum_absolute_difference=float((a - b).abs().max()),
            candidate_raw_maximum_magnitude=float(a.abs().max()),
            reference_raw_maximum_magnitude=float(b.abs().max()),
            centered_maximum_absolute_difference=float((ca - cb).abs().max()),
            centered_sign_changes=int((torch.sign(ca) != torch.sign(cb)).sum()),
            centered_strict_sign_reversals=int((ca * cb < 0).sum()),
            ordered_pairs=len(order_a), ordering_changes=int((order_a != order_b).sum()),
            ordering_strict_reversals=int((order_a * order_b < 0).sum()),
            ties_created=int(((order_a == 0) & (order_b != 0)).sum()),
            ties_removed=int(((order_a != 0) & (order_b == 0)).sum()),
        ))
    return dict(per_pool=pools, centered_sign_changes=sum(p['centered_sign_changes'] for p in pools),
                ordering_changes=sum(p['ordering_changes'] for p in pools),
                raw_scores_equal=torch.equal(candidate_scores, reference_scores),
                center='stable beta-weighted FP64 center of represented FP32 scores',
                comparison='candidate C-reference scores versus original beta-reference scores')


def _signed_receipt(total, positive, negative, absolute):
    return dict(total=total, positive=positive, negative=negative, absolute=absolute,
                sign=1 if total > 0 else -1 if total < 0 else 0,
                cancellation_fraction=(1 - abs(total) / absolute if absolute else None),
                cancellation_denominator='sum absolute signed contributions',
                zero_absolute_contribution=absolute == 0)


@torch.no_grad()
def gradient_contraction(candidate, reference, direction, chunk_elements=1_048_576):
    require(len(candidate) == len(reference) == len(direction) > 0,
            'Gradient contrast parameter membership differs')
    require(chunk_elements > 0, 'Contraction chunk size must be positive')
    sums = {key: [] for key in ('total', 'positive', 'negative', 'absolute')}
    changed, count = 0, 0
    for a, b, w in zip(candidate, reference, direction):
        require(a.shape == b.shape == w.shape, 'Gradient contrast parameter shapes differ')
        a, b, w = [x.detach().reshape(-1) for x in (a, b, w)]
        for start in range(0, len(a), chunk_elements):
            part = slice(start, start + chunk_elements)
            delta = fp32_difference(a[part], b[part])
            weights = w[part].float()
            require(torch.isfinite(delta).all() and torch.isfinite(weights).all(),
                    'Nonfinite native gradient contraction')
            terms = -weights.double() * delta.double()
            for key, value in (('total', terms), ('positive', terms.clamp_min(0)),
                               ('negative', terms.clamp_max(0)), ('absolute', terms.abs())):
                sums[key].append(float(value.sum()))
            changed += int((delta != 0).sum())
            count += delta.numel()
    result = _signed_receipt(**{k: math.fsum(v) for k, v in sums.items()})
    result.update(changed_coordinates=changed, coordinates=count, reference='C',
                  normalization='Native gradients already normalized; no additional 1/(2Z)',
                  products_and_reductions='FP64 after FP32 gradient subtraction')
    return result


def three_contrasts(allocations, scores_c, snapshots, gradients=None, direction=None):
    require(set(allocations) == {'C', 'F_BETA', 'F_C'}, 'Point2A requires exactly three cases')
    require(len(scores_c) == len(snapshots) == 2 and all(len(v) == 2 for v in allocations.values()),
            'Point2A requires two frozen responses for every case')
    require((gradients is None) == (direction is None), 'Supply both gradients and C direction')
    if gradients is not None:
        require(set(gradients) == set(allocations), 'Native gradient cases differ')
    result = {}
    for candidate, reference in CONTRASTS:
        rows = []
        for i, snapshot in enumerate(snapshots):
            roles = snapshot['roles']
            a, b, scores, beta, prefix = [x.detach().cpu() for x in (
                allocations[candidate][i], allocations[reference][i], scores_c[i],
                roles['beta'], roles['prefix'])]
            rows.append(dict(response_index=i, score_contrast=score_contrast(
                a, b, scores, beta, prefix), coefficient_movement=coefficient_movement(
                    a, b, beta, prefix)))
        prediction = {}
        for convention in ('raw', 'beta_centered'):
            prediction[convention] = _signed_receipt(**{
                key: math.fsum(row['score_contrast'][convention][key] for row in rows)
                for key in ('total', 'positive', 'negative', 'absolute')})
        prediction.update(reference='C', normalization='Scores already include 1/(2Z); sum responses once',
                          products_and_reductions='FP64 after FP32 coefficient subtraction')
        item = dict(candidate=candidate, reference=reference, per_response=rows, prediction=prediction,
                    coefficient_equality=all(row['coefficient_movement']['coefficients_equal'] for row in rows),
                    changed_positions=sum(row['coefficient_movement']['changed_positions'] for row in rows))
        if gradients is not None:
            item['actual_gradient_contraction'] = gradient_contraction(
                gradients[candidate], gradients[reference], direction)
        result[candidate + '-' + reference] = item
    return result
