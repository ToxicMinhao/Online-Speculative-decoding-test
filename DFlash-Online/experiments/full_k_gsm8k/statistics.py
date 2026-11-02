"""Prespecified final effects and count-only intermediate observations."""

from collections import defaultdict

import numpy as np

from .evaluation import METHODS, REPEATS

COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COMPARISONS = (('K_HALF', 'O'), ('K', 'O'), ('K_HALF', 'K'), ('K_HALF', 'F'), ('K', 'F'))
BOOTSTRAP_SEED = 20261102
RESAMPLES = 10000


def reduce_final(rows):
    """Use one count trajectory and three latency measurements per request/method."""
    grouped = defaultdict(list)
    identities = {}
    for row in rows:
        if (row['phase'] != 'final' or row['event'] != 58 or row['seed'] != 17
                or row['method'] not in METHODS or row['validation_status'] != 'valid'
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Only valid final executions enter the final comparison')
        ordinal, request_id = row['panel_ordinal'], row['request_id']
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError('A panel ordinal has multiple request identities')
        identities[ordinal] = request_id
        grouped[ordinal, row['method']].append(row)
    if (not identities or sorted(identities) != list(range(len(identities)))
            or len(set(identities.values())) != len(identities)):
        raise ValueError('Missing or duplicated paired request identities')
    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(seed=17, event=58, panel_ordinal=ordinal, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row['repeat'])
            if [row['repeat'] for row in records] != list(range(REPEATS)):
                raise ValueError('Final comparison requires all three distinct repeats for every method')
            first = records[0]
            for field in COUNT_FIELDS:
                if (type(first[field]) is not int or first[field] < 0
                        or any(row[field] != first[field] for row in records)):
                    raise ValueError('Invalid or inconsistent repeated counts')
            if first['verification_calls'] <= 0:
                raise ValueError('Verification denominator must be positive')
            if any(row['checkpoint_id'] != first['checkpoint_id'] or row['finish_reason'] != first['finish_reason']
                   for row in records):
                raise ValueError('Checkpoint or termination changes within a method')
            times = {field: float(np.mean([row[field] for row in records])) for field in TIMINGS}
            if any(not np.isfinite(row[field]) or row[field] < 0 for row in records for field in TIMINGS):
                raise ValueError('Invalid latency measurement')
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(**times, cap_hit=first['cap_hit'], finish_reason=first['finish_reason'],
                                 checkpoint_id=first['checkpoint_id'], count_trajectory_repeat=0, timing_repeats=3)
        if len({(value[m]['output_tokens'], value[m]['finish_reason']) for m in METHODS}) != 1:
            raise ValueError('Paired output length or termination differs')
        paired.append(value)
    return paired


def interval(values):
    finite = values[np.isfinite(values)]
    return dict(ci95=np.quantile(finite, [.025, .975], method='linear').tolist() if len(finite) else None,
                undefined_bootstrap_draws=int(len(values)-len(finite)))


def reduction(candidate, reference, indices):
    """Ratio of sums; zero aggregate cache timers are undefined, not a failure."""
    denominator = float(reference.sum())
    sampled_reference = reference[indices].sum(axis=1)
    draws = np.full(len(indices), np.nan)
    np.divide(candidate[indices].sum(axis=1), sampled_reference, out=draws, where=sampled_reference > 0)
    uncertainty = interval(100*(1-draws))
    return dict(point_pct=100*(1-float(candidate.sum())/denominator) if denominator else None,
        ci95_pct=uncertainty['ci95'], undefined_bootstrap_draws=uncertainty['undefined_bootstrap_draws'],
        candidate_sum=float(candidate.sum()), reference_sum=denominator)


def comparison(candidate, comparator, indices):
    delta = comparator['verification_calls']-candidate['verification_calls']
    result = dict(call_reduction=reduction(candidate['verification_calls'], comparator['verification_calls'], indices),
        request_wins=int((delta > 0).sum()), request_regressions=int((delta < 0).sum()),
        request_ties=int((delta == 0).sum()), request_denominator=int(len(delta)),
        gross_calls_saved=int(np.maximum(delta, 0).sum()), gross_calls_added=int(np.maximum(-delta, 0).sum()))
    for field in TIMINGS:
        result[field + '_reduction'] = reduction(candidate[field], comparator[field], indices)
    return result


def method_totals(values):
    total = {field: int(values[field].sum()) for field in COUNT_FIELDS}
    total.update({field: float(values[field].sum()) for field in TIMINGS})
    total.update(calls_per_1000_output_tokens=1000*total['verification_calls']/total['output_tokens'],
        accepted_tokens_per_call=total['accepted_draft_tokens']/total['verification_calls'],
        committed_progress_per_call=total['committed_progress']/total['verification_calls'])
    return total


def interpret(primary, *, valid=True):
    if not valid:
        return dict(endpoint_label='invalid_or_incomplete', supported_endpoint_gain=False,
                    next_action='Resolve the concrete validity failure before interpreting efficacy.',
                    automatic_followup_authorized=False)
    effect = primary['call_reduction']
    low, high = effect['ci95_pct']
    if effect['point_pct'] > 0 and low > 0:
        label = 'supported endpoint call reduction'
        action = ('Keep the fixed trained candidate for review; propose one separately agreed '
                  'confirmation addressing portability beyond this reused benchmark and fitted seed.')
    elif effect['point_pct'] < 0 and high < 0:
        label = 'supported endpoint call regression'
        action = ('Review the retained learning curve to choose one bounded investigation of '
                  'the observed long-horizon limitation; do not retune or extend automatically.')
    else:
        label = 'inconclusive endpoint difference'
        action = ('Review whether one separately agreed precision or training-seed check could '
                  'resolve the endpoint uncertainty before changing the algorithm.')
    return dict(endpoint_label=label, supported_endpoint_gain=effect['point_pct'] > 0 and low > 0,
        practical_effect_pct=effect['point_pct'], next_action=action, automatic_followup_authorized=False,
        scope='Conditional on seed17 and the previously used benchmark; no equivalence or general robustness claim.')


def summarize(rows):
    paired = reduce_final(rows)
    arrays = {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
                      for field in (*COUNT_FIELDS, *TIMINGS)} for method in METHODS}
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired)))
    effects = {a + '_vs_' + b: comparison(arrays[a], arrays[b], indices) for a, b in COMPARISONS}
    gross = {}
    frozen = arrays['F']['verification_calls']
    for method in ('O', 'K', 'K_HALF'):
        difference = frozen-arrays[method]['verification_calls']
        gain, harm = np.maximum(difference, 0), np.maximum(-difference, 0)
        gross[method] = dict(gross_gain_calls=int(gain.sum()), gross_harm_calls=int(harm.sum()),
                            net_calls_saved=int(difference.sum()))
    totals = {method: method_totals(values) for method, values in arrays.items()}
    repeats = {str(repeat): {method: {field: sum(row[field] for row in rows
        if row['repeat'] == repeat and row['method'] == method) for field in (*COUNT_FIELDS, *TIMINGS)}
        for method in METHODS} for repeat in range(REPEATS)}
    return dict(request_count=len(paired), bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED,
        generator='PCG64', percentiles=[2.5,97.5], method='linear',
        unit='One request carrying every method and its three-repeat mean; repeats are not samples.'),
        method_totals=totals, comparisons=effects, gain_harm_vs_F=gross, repeat_totals=repeats,
        interpretation=interpret(effects['K_HALF_vs_O']),
        secondary_intervals='Descriptive and unadjusted; conditional on one fitted seed and a reused benchmark.',
        primary_scope='K_HALF/O is a full-updater different-recipe comparison; K/O is the same-LR control.',
        timing_scope='Fresh contemporaneous checkpoint-ready serving; not net asynchronous or training-inclusive speedup.')


def intermediate_summary(rows, canonicals):
    if not rows or any(row['phase'] != 'intermediate' or not row['valid_measurement']
                       or row['validation_status'] != 'valid' or row['errors'] for row in rows):
        raise ValueError('Intermediate summaries require every planned valid record')
    identities = {(row['panel_ordinal'], row['request_id']) for row in rows}
    if len(identities) != len(rows) or len({(r['method'], r['event']) for r in rows}) != 1:
        raise ValueError('An intermediate cell contains duplicated requests or mixed checkpoints')
    candidate = sum(row['verification_calls'] for row in rows)
    frozen = sum(canonicals[row['request_id']]['F_counts']['verification_calls'] for row in rows)
    deltas = [canonicals[row['request_id']]['F_counts']['verification_calls']-row['verification_calls'] for row in rows]
    return dict(arm=rows[0]['method'], event=rows[0]['event'], requests=len(rows),
        verification_calls=candidate, frozen_verification_calls=frozen,
        call_reduction_vs_F_pct=100*(1-candidate/frozen), request_wins=sum(v > 0 for v in deltas),
        request_regressions=sum(v < 0 for v in deltas), request_ties=sum(v == 0 for v in deltas),
        output_tokens=sum(row['output_tokens'] for row in rows),
        baseline_scope='Retained original F counts on the identical first128 requests; historical count-only context.',
        interpretation='Fixed development observation, not checkpoint selection or evidence of faster learning than O.')
