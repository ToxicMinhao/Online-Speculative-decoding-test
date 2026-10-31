"""Prespecified paired-request statistics; repeats and fitted seeds are not samples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEEDS = (17, 29)
METHODS = ('F', 'K', 'K_HALF', 'KP')
COMPARISONS = (('KP', 'K'), ('KP', 'F'), ('KP', 'K_HALF'),
               ('K_HALF', 'K'), ('K_HALF', 'F'), ('K', 'F'))
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
RESAMPLES = 10000
BOOTSTRAP_SEED = 20261102
PRIORITY_TARGET_PCT = 0.5
EQUIVALENCE_BAND_PCT = (-0.5, 0.5)


def bootstrap_indices(request_count, *, seed=BOOTSTRAP_SEED, resamples=RESAMPLES):
    """One draw carries both fitted seeds and every method for a request."""
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(0, request_count, size=(resamples, request_count))


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {'ci95': np.quantile(finite, [0.025, 0.975], method='linear').tolist()
            if len(finite) else None, 'undefined_bootstrap_draws': int(len(values) - len(finite))}


def summed_interval(per_request, indices):
    values = np.asarray(per_request, dtype=np.float64)
    return {'point': float(values.sum()), **interval(values[indices].sum(axis=1))}


def ratio_reduction(candidate, reference, indices):
    """Percentage reduction of summed paired measurements, never mean percentages."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = reference.sum()
    if denominator <= 0:
        raise ValueError('Reduction requires a positive reference total')
    drawn_reference = reference[indices].sum(axis=1)
    drawn_candidate = candidate[indices].sum(axis=1)
    draws = np.full(len(indices), np.nan)
    np.divide(drawn_candidate, drawn_reference, out=draws, where=drawn_reference > 0)
    summary = interval(100.0 * (1.0 - draws))
    return {'point_pct': float(100.0 * (1.0 - candidate.sum() / denominator)),
            'ci95_pct': summary['ci95'],
            'undefined_bootstrap_draws': summary['undefined_bootstrap_draws'],
            'candidate_sum': float(candidate.sum()), 'reference_sum': float(denominator)}


def aligned_arrays(rows, methods=METHODS, fields=COUNT_FIELDS + TIMINGS):
    """Validate one reduced trajectory per seed/request and align by sealed ordinal."""
    ordered = {}
    for seed in SEEDS:
        ordered[seed] = sorted((row for row in rows if row['seed'] == seed),
                               key=lambda row: row['panel_ordinal'])
    if len(rows) != sum(len(values) for values in ordered.values()):
        raise ValueError('Unexpected fitted seed in paired results')
    keys = [(row['panel_ordinal'], row['request_id']) for row in ordered[SEEDS[0]]]
    if not keys or len({item[0] for item in keys}) != len(keys) or len({item[1] for item in keys}) != len(keys):
        raise ValueError('Missing or duplicated paired requests')
    for seed in SEEDS[1:]:
        if [(row['panel_ordinal'], row['request_id']) for row in ordered[seed]] != keys:
            raise ValueError('Seeds do not carry identical paired request identities/order')
    arrays = {}
    for method in methods:
        arrays[method] = {}
        for field in fields:
            values = np.array([[row[method][field] for row in ordered[seed]] for seed in SEEDS], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Nonfinite or negative measurement: ' + method + '/' + field)
            if field in COUNT_FIELDS and not (values == np.floor(values)).all():
                raise ValueError('Fractional count measurement: ' + method + '/' + field)
            arrays[method][field] = values
        if (arrays[method]['verification_calls'] <= 0).any():
            raise ValueError('Every validated trajectory must have positive verification calls')
    return keys, arrays


def comparison(candidate, reference, indices):
    """Keep request/seed decisions separate while bootstrapping joint requests."""
    candidate_calls = candidate['verification_calls']
    reference_calls = reference['verification_calls']
    delta = reference_calls - candidate_calls
    result = {'call_reduction': ratio_reduction(candidate_calls.sum(axis=0), reference_calls.sum(axis=0), indices),
              'request_seed_wins': int((delta > 0).sum()),
              'request_seed_regressions': int((delta < 0).sum()),
              'request_seed_ties': int((delta == 0).sum()),
              'request_seed_denominator': int(delta.size),
              'gross_calls_saved': int(np.maximum(delta, 0).sum()),
              'gross_calls_added': int(np.maximum(-delta, 0).sum())}
    for field in TIMINGS:
        result[field + '_reduction'] = ratio_reduction(candidate[field].sum(axis=0), reference[field].sum(axis=0), indices)
    return result


def gain_harm(frozen, incumbent, candidate, indices):
    """Positive parts are formed within seed before any sum across fitted seeds."""
    gain_k = np.maximum(frozen - incumbent, 0)
    harm_k = np.maximum(incumbent - frozen, 0)
    gain_m = np.maximum(frozen - candidate, 0)
    harm_m = np.maximum(candidate - frozen, 0)
    retained = np.minimum(gain_k, gain_m)
    components = {'K_gross_gain': gain_k, 'K_gross_harm': harm_k,
                  'candidate_gross_gain': gain_m, 'candidate_gross_harm': harm_m,
                  'retained_gain': retained, 'lost_gain': gain_k - retained,
                  'added_gain': gain_m - retained,
                  'prevented_harm': np.maximum(harm_k - harm_m, 0),
                  'added_harm': np.maximum(harm_m - harm_k, 0)}
    explained = components['added_gain'] - components['lost_gain'] + components['prevented_harm'] - components['added_harm']
    if not np.array_equal(incumbent - candidate, explained):
        raise AssertionError('Gain/harm accounting identity does not hold per seed/request')
    result = {name: summed_interval(value.sum(axis=0), indices) for name, value in components.items()}
    denominator = gain_k.sum()
    drawn_denominator = gain_k.sum(axis=0)[indices].sum(axis=1)
    drawn_retained = retained.sum(axis=0)[indices].sum(axis=1)
    fractions = np.full(len(indices), np.nan)
    np.divide(drawn_retained, drawn_denominator, out=fractions, where=drawn_denominator > 0)
    result['retained_gain_fraction'] = {
        'point': float(retained.sum() / denominator) if denominator else None,
        **interval(fractions), 'denominator_K_gross_gain': float(denominator)}
    result['net_calls_saved_vs_K'] = int((incumbent - candidate).sum())
    result['identity_verified_at_each_seed_request'] = True
    return result


def method_totals(values):
    totals = {field: float(value.sum()) for field, value in values.items()}
    for field in COUNT_FIELDS:
        totals[field] = int(totals[field])
    totals['calls_per_1000_output_tokens'] = (
        1000 * totals['verification_calls'] / totals['output_tokens'] if totals['output_tokens'] else None)
    totals['accepted_tokens_per_call'] = totals['accepted_draft_tokens'] / totals['verification_calls']
    totals['committed_progress_per_call'] = totals['committed_progress'] / totals['verification_calls']
    return totals


def interpret(combined, per_seed):
    """Apply the declared endpoint labels without claiming a causal mechanism."""
    primary = combined['KP_vs_K']['call_reduction']
    versus_f = combined['KP_vs_F']['call_reduction']
    versus_half = combined['KP_vs_K_HALF']['call_reduction']
    seed_primary = [per_seed[str(seed)]['KP_vs_K']['call_reduction']['point_pct'] for seed in SEEDS]
    seed_f = [per_seed[str(seed)]['KP_vs_F']['call_reduction']['point_pct'] for seed in SEEDS]
    useful = (primary['ci95_pct'][0] > 0 and primary['point_pct'] >= PRIORITY_TARGET_PCT
              and all(value > 0 for value in seed_primary)
              and versus_f['ci95_pct'][0] > 0 and all(value > 0 for value in seed_f))
    if useful:
        label = 'supported useful improvement at this budget'
    elif min(seed_primary) < 0 < max(seed_primary):
        label = 'mixed fitted-seed directions'
    elif primary['point_pct'] >= PRIORITY_TARGET_PCT and primary['ci95_pct'][0] <= 0 <= primary['ci95_pct'][1]:
        label = 'promising but unresolved'
    elif primary['ci95_pct'][0] > 0 and (versus_f['ci95_pct'][0] <= 0 or not all(value > 0 for value in seed_f)):
        label = 'relative improvement over K; useful improvement over F unresolved'
    elif primary['point_pct'] < 0:
        label = 'negative primary point estimate'
    elif primary['point_pct'] < PRIORITY_TARGET_PCT:
        label = 'small primary effect; useful-improvement criterion not met'
    else:
        label = 'useful-improvement criterion not met'
    equivalent = (versus_half['ci95_pct'][0] >= EQUIVALENCE_BAND_PCT[0]
                  and versus_half['ci95_pct'][1] <= EQUIVALENCE_BAND_PCT[1])
    return {'endpoint_label': label, 'supported_useful_improvement': useful,
            'primary_seed_effects_pct': seed_primary, 'versus_F_seed_effects_pct': seed_f,
            'superiority_over_tested_K_HALF_supported': versus_half['ci95_pct'][0] > 0,
            'KP_K_HALF_practical_equivalence_supported': equivalent,
            'equivalence_band_pct': list(EQUIVALENCE_BAND_PCT),
            'KP_vs_K_HALF_seed_effects_pct': [per_seed[str(seed)]['KP_vs_K_HALF']['call_reduction']['point_pct'] for seed in SEEDS],
            'secondary_intervals': 'Descriptive/unadjusted, not multiplicity-adjusted confirmatory claims.',
            'mechanism_boundary': 'Gain/harm accounting and this lower-LR control do not uniquely identify a causal mechanism.'}


def summarize(rows):
    """Return fixed comparisons and joint-request uncertainty from validated rows."""
    keys, arrays = aligned_arrays(rows)
    indices = bootstrap_indices(len(keys))
    combined = {left + '_vs_' + right: comparison(arrays[left], arrays[right], indices)
                for left, right in COMPARISONS}
    per_seed, totals, decompositions = {}, {'combined': {}}, {'combined': {}}
    for method in METHODS:
        totals['combined'][method] = method_totals(arrays[method])
    for method in ('KP', 'K_HALF'):
        decompositions['combined'][method] = gain_harm(
            arrays['F']['verification_calls'], arrays['K']['verification_calls'],
            arrays[method]['verification_calls'], indices)
    for seed_index, seed in enumerate(SEEDS):
        subset = {method: {field: values[seed_index:seed_index + 1]
                           for field, values in fields.items()} for method, fields in arrays.items()}
        per_seed[str(seed)] = {left + '_vs_' + right: comparison(subset[left], subset[right], indices)
                              for left, right in COMPARISONS}
        totals[str(seed)] = {method: method_totals(subset[method]) for method in METHODS}
        decompositions[str(seed)] = {method: gain_harm(
            subset['F']['verification_calls'], subset['K']['verification_calls'],
            subset[method]['verification_calls'], indices) for method in ('KP', 'K_HALF')}
    return {'request_count': len(keys), 'request_seed_count': 2 * len(keys),
            'bootstrap': {'resamples': RESAMPLES, 'seed': BOOTSTRAP_SEED, 'generator': 'PCG64',
                          'percentiles': [2.5, 97.5], 'method': 'linear',
                          'unit': 'One request carrying both fitted seeds and every method.',
                          'scope': 'Conditional on two fixed fitted trajectories and the selected panel.'},
            'combined': combined, 'per_seed': per_seed, 'method_totals': totals,
            'gain_harm': decompositions, 'interpretation': interpret(combined, per_seed)}


def precision_note(path):
    """Reproduce historical joint-request intervals as a planning proxy only."""
    with Path(path).open(newline='') as stream:
        records = list(csv.DictReader(stream))
    result = {}
    for event in (2, 4):
        rows = [{'seed': int(row['seed']), 'request_id': row['request_id'],
                 'panel_ordinal': int(row['panel_ordinal']),
                 **{method: {'verification_calls': int(row[method + '_verification_calls'])}
                    for method in ('F', 'O', 'K')}}
                for row in records if int(row['event']) == event]
        keys, arrays = aligned_arrays(rows, methods=('F', 'O', 'K'), fields=('verification_calls',))
        if len(keys) != 128:
            raise ValueError('Original precision proxy must contain all 128 paired requests')
        indices = bootstrap_indices(len(keys), seed=20261030)
        result[str(event)] = {f'K_vs_{reference}': ratio_reduction(
            arrays['K']['verification_calls'].sum(axis=0),
            arrays[reference]['verification_calls'].sum(axis=0), indices) for reference in ('O', 'F')}
    return {'status': 'CPU_only_before_model_work', 'source_csv': str(path),
            'bootstrap': {'resamples': RESAMPLES, 'seed': 20261030, 'generator': 'PCG64',
                          'unit': 'Request identity carrying both seeds', 'percentile_method': 'linear'},
            'combined_historical_intervals': result, 'priority_target_pct': PRIORITY_TARGET_PCT,
            'interpretation': 'Historical intervals are planning proxies. Candidate-versus-K variance is unknown; '
                              'these do not guarantee power. The fixed 0.5% target is a research-priority choice, '
                              'not a measured deployment break-even. No panel resizing, tuning or outcome-based selection.'}


def write_precision_note(source_csv, output):
    from experiments.ok_coding.prepare import identity, save
    note = precision_note(source_csv)
    note['source_identity'] = identity(source_csv)
    save(Path(output) / 'PRECISION_NOTE.json', note)
    lines = ['# Prelaunch precision note', '',
             'CPU analysis of the completed coding study; no model execution.', '',
             '| Original endpoint | Comparison | Call reduction | Joint-request 95% interval |',
             '|---|---|---:|---:|']
    for event, comparisons in note['combined_historical_intervals'].items():
        for name, value in comparisons.items():
            low, high = value['ci95_pct']
            lines.append(f"| Event {event} | {name} | {value['point_pct']:+.4f}% | [{low:+.4f}, {high:+.4f}]% |")
    lines.extend(['', note['interpretation'], '',
                  'Both historical endpoint intervals are reported; the new campaign evaluates only event 4. '
                  'Their uncertainty should be compared with the 0.5% priority target, rather than read as '
                  'a power estimate for KP versus K. Each bootstrap sample jointly retains both seeds; '
                  'timing repeats and seed/request rows are not independent observations.', '',
                  'Source: `' + str(source_csv) + '`. Exact source bytes and bootstrap configuration are in PRECISION_NOTE.json.'])
    (Path(output) / 'PRECISION_NOTE.md').write_text('\n'.join(lines) + '\n')
    return note
