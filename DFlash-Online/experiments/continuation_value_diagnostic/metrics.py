"""Frozen-parent continuation information and signed, fixed-boundary outcomes.

Native BF16 head values define predictions and rivals. Only their subtraction
and tanh supports use FP32; compact scalar reductions use Python FP64. Nothing
in this module changes a model, loss, coefficient or training trajectory.
"""

from collections import defaultdict
import math

import torch

from online_feedback.first_error import valid_prefix_mask


BLOCK_SIZE = 16
MAX_FOLLOWERS = 14
ROLE_KEYS = ('labels', 'valid', 'prefix', 'repair', 'L', 'e', 'H', 'K', 'm')


def roles_from_predictions(labels, valid, proposals):
    """Recover native roles from contiguous valid proposals; slot zero is a seed."""
    if not (len(labels) == len(valid) == len(proposals) == BLOCK_SIZE):
        raise ValueError('A diagnostic block must have sixteen aligned slots')
    m = sum(valid)
    if valid != [False] + [True] * m + [False] * (BLOCK_SIZE - m - 1):
        raise ValueError('Validity must be the contiguous canonical proposal prefix')
    first_error = next((j for j in range(1, m + 1) if proposals[j] != labels[j]), -1)
    length = m if first_error == -1 else first_error - 1
    endpoint = m
    if first_error >= 1:
        endpoint = next(
            (j - 1 for j in range(first_error + 1, m + 1)
             if proposals[j] != labels[j]), m,
        )
    return dict(
        labels=list(labels), valid=list(valid),
        prefix=[0 < j <= length for j in range(BLOCK_SIZE)],
        repair=[j == first_error for j in range(BLOCK_SIZE)],
        L=length, e=first_error, H=endpoint - length if first_error >= 1 else 0,
        K=endpoint, m=m,
    )


@torch.no_grad()
def compact_blocks(logits, labels, weights, eos_token_ids=()):
    """Discard a native vocabulary chunk after retaining all valid positions.

    Native argmax chooses the lowest token ID at a tie, including when the
    canonical token is tied. The strongest rival always excludes that token.
    """
    if (logits.dtype != torch.bfloat16 or labels.ndim != 3
            or labels.shape[0] != 1 or labels.shape[-1] != BLOCK_SIZE
            or logits.shape[:-1] != labels.shape or weights.shape != labels.shape):
        raise ValueError('Expected one response of aligned native BF16 16-slot blocks')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite native diagnostic logits')
    valid = valid_prefix_mask(labels, weights, eos_token_ids)
    proposals = logits.argmax(-1)
    output = []
    for block in range(labels.shape[1]):
        canonical = labels[0, block].tolist()
        mask = valid[0, block].tolist()
        predictions = proposals[0, block].tolist()
        roles = roles_from_predictions(canonical, mask, predictions)
        count = roles['m']
        native = logits[0, block, 1:count + 1]
        token_ids = labels[0, block, 1:count + 1]
        wrong = native.clone()
        wrong.scatter_(-1, token_ids.unsqueeze(-1), -torch.inf)
        rivals = wrong.argmax(-1)
        true_values = native.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).float()
        wrong_values = native.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
        margins = true_values - wrong_values
        supports = torch.tanh(margins / 2).clamp_min(0)
        rival_ties = (wrong == wrong_values.to(native.dtype).unsqueeze(-1)).sum(-1) > 1
        positions = [dict(
            position=j, canonical=canonical[j], prediction=predictions[j],
            rival=int(rival), correct=predictions[j] == canonical[j],
            margin=float(margin), support_fp32=float(support),
            margin_tie=bool(margin == 0), wrong_rival_tie=bool(tie),
        ) for j, (rival, margin, support, tie) in enumerate(
            zip(rivals.tolist(), margins.tolist(), supports.tolist(), rival_ties.tolist()), 1,
        )]
        output.append(dict(
            labels=canonical, valid=mask, proposals=predictions, m=count,
            roles=roles, positions=positions, support_source='native_fp32_tanh',
        ))
    return output


def continuation_area(supports):
    """Area of a running minimum, with earliest ties retained for attribution."""
    minima, bottlenecks, ties = [], [], []
    best = None
    for index, support in enumerate(supports):
        if not math.isfinite(support) or not 0 <= support <= 1:
            raise ValueError('Continuation supports must be finite and bounded by [0, 1]')
        if best is None or support < supports[best]:
            best = index
        minima.append(supports[best])
        bottlenecks.append(best)
        ties.append(sum(value == supports[best] for value in supports[:index + 1]) > 1)
    area = math.fsum(minima)
    return dict(A=area, u=area / MAX_FOLLOWERS,
                running_minima=minima, bottleneck_indices=bottlenecks, tied_terms=ties)


def parent_predictors(parent):
    """Only parent information determines HIGH membership and the frozen K.

    Legacy witnesses stop storing margins at K. Their first later mismatch is
    nevertheless known from retained predictions, and makes every later area
    term exactly zero. Those zero *terms* are inferred; missing margins/supports
    are left as None rather than fabricated.
    """
    role = roles_from_predictions(parent['labels'], parent['valid'], parent['proposals'])
    if 'roles' in parent and parent['roles'] != role:
        raise ValueError('Recorded parent roles disagree with native predictions')
    m, error, length, endpoint = (role[key] for key in ('m', 'e', 'L', 'K'))
    positions = {p['position']: p for p in parent['positions']}
    if len(positions) != len(parent['positions']) or set(positions) - set(range(1, m + 1)):
        raise ValueError('Parent position records are duplicated or outside validity')
    if set(range(1, endpoint + 1)) - positions.keys():
        raise ValueError('Missing a parent margin inside the frozen endpoint')
    for index, position in positions.items():
        margin = position['margin']
        correct = parent['proposals'][index] == parent['labels'][index]
        if not math.isfinite(margin) or (correct and margin < 0) or (not correct and margin > 0):
            raise ValueError('A native margin is nonfinite or contradicts argmax correctness')
    has_error = error >= 1
    run = endpoint - error if has_error else 0
    available = m - error if has_error else 0
    prefix_margins = [positions[j]['margin'] for j in range(1, length + 1)]
    error_margin = positions[error]['margin'] if has_error else 0.0
    followers = [positions.get(j, {}).get('support_fp32')
                 for j in range(error + 1, m + 1)] if has_error else []
    matching_supports = followers[:run]
    if any(value is None for value in matching_supports):
        raise ValueError('Missing a support inside the original matching follower run')
    area = continuation_area(matching_supports)
    h1 = followers[0] if available and followers[0] is not None else 0.0
    # Outside the correct run, the next mismatch ends every further area term.
    if available > run and followers[run] not in (None, 0.0):
        raise ValueError('An incorrect native follower has positive support')
    full_minima = area['running_minima'] + [0.0] * (available - run)
    deficit = math.fsum(h1 - value for value in area['running_minima']) / MAX_FOLLOWERS
    if not math.isclose(area['A'], run * h1 - MAX_FOLLOWERS * deficit,
                        rel_tol=1e-14, abs_tol=1e-14):
        raise ValueError('Continuation area and later-bottleneck identity disagree')
    complete = None
    if len(positions) == m and has_error:
        complete = continuation_area(followers)
        if complete['running_minima'] != full_minima:
            raise ValueError('Native full-suffix area disagrees with frozen run reduction')
    immediate_terms = sum(index == 0 for index in area['bottleneck_indices'])
    later_terms = run - immediate_terms
    full_immediate = (sum(index == 0 for index in complete['bottleneck_indices'])
                      if complete is not None else None)
    return dict(
        m=m, L=length, e=error, r=run, n=available, K=endpoint,
        s=bool(run), high=has_error and run >= 1,
        beta=math.exp(-(error - 1) / 7) if has_error else 0.0,
        A=area['A'], u=area['u'], D=deficit, h1=h1,
        error_margin=error_margin, d=max(0.0, -error_margin),
        minimum_original_prefix_margin=min(prefix_margins, default=0.0),
        empty_prefix=length == 0, error_tie=has_error and error_margin == 0,
        prefix_tie=any(margin == 0 for margin in prefix_margins),
        follower_supports_fp32=followers, running_minima_fp64=full_minima,
        run_bottleneck_indices=area['bottleneck_indices'],
        run_bottleneck_tied_terms=area['tied_terms'],
        run_immediate_controlled_terms=immediate_terms,
        run_later_controlled_terms=later_terms,
        run_immediate_controlled_fraction=immediate_terms / run if run else None,
        run_later_controlled_fraction=later_terms / run if run else None,
        all_follower_bottleneck_indices=None if complete is None else complete['bottleneck_indices'],
        all_follower_bottleneck_tied_terms=None if complete is None else complete['tied_terms'],
        all_follower_immediate_controlled_fraction=(full_immediate / available
                                                   if complete is not None and available else None),
        all_follower_later_controlled_fraction=((available - full_immediate) / available
                                               if complete is not None and available else None),
        full_suffix_ties_observed=complete is not None,
        inferred_zero_area_terms=available - run if len(positions) < m else 0,
    )


def paired_block(parent, child, *, record_id, anchor, cohort='primary'):
    """Signed progress and decompositions at the immutable parent's endpoint."""
    if parent['labels'] != child['labels'] or parent['valid'] != child['valid']:
        raise ValueError('Paired blocks changed canonical tokens or validity')
    origin = parent_predictors(parent)
    current = roles_from_predictions(child['labels'], child['valid'], child['proposals'])
    if 'roles' in child and child['roles'] != current:
        raise ValueError('Recorded child roles disagree with native predictions')
    error, old_length, endpoint = (origin[key] for key in ('e', 'L', 'K'))
    new_length = current['L']
    signed = min(new_length, endpoint) - old_length
    positive = max(signed, 0)
    labels, predictions = parent['labels'], child['proposals']
    damage = sum(predictions[j] != labels[j] for j in range(1, old_length + 1))
    followers = range(error + 1, endpoint + 1) if error >= 1 else ()
    return dict(
        record_id=str(record_id), anchor=int(anchor), cohort=cohort, **origin,
        child_L=new_length, Y=signed,
        raw_repair=error >= 1 and predictions[error] == labels[error],
        useful_repair=signed > 0, prefix_damaged=signed < 0,
        positive_progress=positive, lost_prefix=max(-signed, 0),
        reachable_origin_followers=max(positive - 1, 0),
        damaged_prefix_positions=damage,
        surviving_origin_followers=sum(predictions[j] == labels[j] for j in followers),
        uncapped_progress=new_length - old_length,
        parent=parent, child=child,
    )


def legacy_compact_block(block):
    """Promote saved native margins to CPU FP32 supports; make no model call."""
    current = block['current']
    roles = roles_from_predictions(current['labels'], current['valid'], block['proposals'])
    if any(current[key] != roles[key] for key in ROLE_KEYS):
        raise ValueError('Legacy current roles do not reproduce from retained predictions')
    positions = [dict(position) for position in block['positions']]
    margins = torch.tensor([position['margin'] for position in positions],
                           dtype=torch.float32, device='cpu')
    supports = torch.tanh(margins / 2).clamp_min(0).tolist()
    for position, support in zip(positions, supports):
        position['support_fp32'] = support
    return dict(
        labels=roles['labels'], valid=roles['valid'], proposals=block['proposals'],
        m=roles['m'], roles=roles, positions=positions,
        support_source='offline_cpu_fp32_tanh_of_retained_native_margins',
    )


def legacy_pairs(parent_witness, child_witness, cohort):
    """Reanalyze one legacy cohort and reproduce its retained integer summaries."""
    if cohort not in ('new_buffer', 'parent_training'):
        raise ValueError('Legacy cohorts must remain separate')
    parents = parent_witness['sets'][cohort]
    children = child_witness['sets'][cohort]
    if len(parents) != len(children):
        raise ValueError('Legacy response count changed')
    rows = []
    for parent, child in zip(parents, children):
        if (parent['record_id'] != child['record_id'] or parent['anchors'] != child['anchors']
                or len(parent['blocks']) != len(child['blocks'])
                or len(parent['anchors']) != len(parent['blocks'])):
            raise ValueError('Legacy response/anchor binding changed')
        for anchor, first, second in zip(parent['anchors'], parent['blocks'], child['blocks']):
            compact_parent, compact_child = legacy_compact_block(first), legacy_compact_block(second)
            if first['origin'] != compact_parent['roles'] or second['origin'] != first['origin']:
                raise ValueError('Legacy origin is not the fixed parent role assignment')
            row = paired_block(compact_parent, compact_child, record_id=parent['record_id'],
                               anchor=anchor, cohort='legacy_' + cohort)
            for observed, compact in ((first, compact_parent), (second, compact_child)):
                reproduced = paired_block(compact_parent, compact, record_id=parent['record_id'],
                                          anchor=anchor, cohort='legacy_' + cohort)
                expected = dict(
                    raw_repair=int(reproduced['raw_repair']),
                    useful_repair=int(reproduced['useful_repair']),
                    damaged_prefix_positions=reproduced['damaged_prefix_positions'],
                    damaged_prefix_blocks=int(reproduced['prefix_damaged']),
                    surviving_followers=reproduced['surviving_origin_followers'],
                    reachable_followers=reproduced['reachable_origin_followers'],
                    matching_prefix_length=reproduced['child_L'],
                    fixed_K_shortfall=max(0, row['K'] - reproduced['child_L']),
                )
                if any(observed['metrics'][key] != value for key, value in expected.items()):
                    raise ValueError('Legacy integer outcome summary does not reproduce')
            rows.append(row)
    return rows


def coverage_table(rows, response_ids):
    """Per-response coverage, including empty responses and diagnostic pool support."""
    if len(set(response_ids)) != len(response_ids):
        raise ValueError('Coverage response identities must be unique')
    identities = [(row['record_id'], row['anchor']) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Coverage must not double-count one response/anchor block')
    groups = defaultdict(list)
    for row in rows:
        groups[row['record_id']].append(row)
    if set(groups) - set(response_ids):
        raise ValueError('Coverage rows contain an unsealed response')
    output = []
    for record_id in response_ids:
        blocks = groups[record_id]
        high = [row for row in blocks if row['high']]
        low = sum(row['e'] >= 1 and not row['high'] for row in blocks)
        distinct_scores = len({row['u'] for row in high})
        output.append(dict(
            record_id=record_id, blocks=len(blocks), nonempty_blocks=sum(row['m'] > 0 for row in blocks),
            empty_blocks=sum(row['m'] == 0 for row in blocks),
            no_error_blocks=sum(row['m'] > 0 and row['e'] == -1 for row in blocks),
            error_blocks=len(high) + low, high_blocks=len(high), low_blocks=low,
            mixed_high_low_pool=bool(high) and low > 0, at_least_two_high=len(high) >= 2,
            high_score_variation=distinct_scores > 1,
        ))
    return output
