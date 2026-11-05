"""Count-only matched-LR effects with joint-request uncertainty.

Archived and new serving times deliberately do not enter these statistics.
Each resampled request carries both fixed fitted seeds and all five methods.
"""
from __future__ import annotations

import numpy as np

from experiments.k_prefix_protection_v1.statistics import (
    aligned_arrays, bootstrap_indices, interval, method_totals, ratio_reduction,
    summed_interval,
)

SEEDS = (17, 29)
METHODS = ('F', 'O', 'K', 'O_HALF', 'K_HALF')
COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens',
                'committed_progress', 'output_tokens')
COMPARISONS = (('K_HALF', 'O_HALF'), ('K', 'O'), ('K_HALF', 'K'),
               ('O_HALF', 'O'), ('O', 'F'), ('K', 'F'),
               ('O_HALF', 'F'), ('K_HALF', 'F'))
DECOMPOSITIONS = (('K_HALF', 'O_HALF'), ('O_HALF', 'O'), ('K_HALF', 'K'))
RESAMPLES = 10000
BOOTSTRAP_SEED = 20261102
PRIORITY_PCT = 0.5
EQUIVALENCE_BAND_PCT = (-0.5, 0.5)


def comparison(candidate, reference, indices):
    """One trajectory per seed/request; wins are descriptive outcome counts."""
    delta = reference - candidate
    return {
        'call_reduction': ratio_reduction(candidate.sum(axis=0), reference.sum(axis=0), indices),
        'request_seed_wins': int((delta > 0).sum()),
        'request_seed_regressions': int((delta < 0).sum()),
        'request_seed_ties': int((delta == 0).sum()),
        'request_seed_denominator': int(delta.size),
        'gross_calls_saved': int(np.maximum(delta, 0).sum()),
        'gross_calls_added': int(np.maximum(-delta, 0).sum()),
    }


def objective_lr_interaction(calls, indices):
    """Use F as a COMMON denominator, not differently normalized effects."""
    numerator = ((calls['O_HALF'] - calls['K_HALF'])
                 - (calls['O'] - calls['K'])).sum(axis=0)
    frozen = calls['F'].sum(axis=0)
    draws = 100.0 * numerator[indices].sum(axis=1) / frozen[indices].sum(axis=1)
    uncertainty = interval(draws)
    return {
        'point_pct': float(100.0 * numerator.sum() / frozen.sum()),
        'ci95_pct': uncertainty['ci95'],
        'undefined_bootstrap_draws': uncertainty['undefined_bootstrap_draws'],
        'numerator_call_difference': int(numerator.sum()),
        'denominator_F_calls': int(frozen.sum()),
        'formula': '100 * ((O_HALF - K_HALF) - (O - K)) / F, using summed calls',
        'positive_means': 'K has a larger call advantage over O at the lower LR.',
    }


def gain_harm(frozen, comparator, candidate, indices):
    """Compute positive parts within seed before pooling paired requests."""
    gain_b = np.maximum(frozen - comparator, 0)
    harm_b = np.maximum(comparator - frozen, 0)
    gain_a = np.maximum(frozen - candidate, 0)
    harm_a = np.maximum(candidate - frozen, 0)
    retained = np.minimum(gain_a, gain_b)
    values = {
        'comparator_gross_gain': gain_b, 'comparator_gross_harm': harm_b,
        'candidate_gross_gain': gain_a, 'candidate_gross_harm': harm_a,
        'retained_gain': retained, 'lost_gain': gain_b - retained,
        'added_gain': gain_a - retained,
        'prevented_harm': np.maximum(harm_b - harm_a, 0),
        'added_harm': np.maximum(harm_a - harm_b, 0),
    }
    explained = (values['added_gain'] - values['lost_gain']
                 + values['prevented_harm'] - values['added_harm'])
    if not np.array_equal(comparator - candidate, explained):
        raise AssertionError('Gain/harm identity failed within seed/request')
    result = {name: summed_interval(value.sum(axis=0), indices)
              for name, value in values.items()}
    denominator = gain_b.sum()
    drawn_denominator = gain_b.sum(axis=0)[indices].sum(axis=1)
    drawn_retained = retained.sum(axis=0)[indices].sum(axis=1)
    fractions = np.full(len(indices), np.nan)
    np.divide(drawn_retained, drawn_denominator, out=fractions, where=drawn_denominator > 0)
    result['retained_gain_fraction'] = {
        'point': float(retained.sum() / denominator) if denominator else None,
        **interval(fractions), 'denominator_comparator_gross_gain': int(denominator),
    }
    result['net_calls_saved_vs_comparator'] = int((comparator - candidate).sum())
    result['identity_verified_at_each_seed_request'] = True
    return result


def interpret(combined, per_seed, *, valid=True):
    """Apply exactly the handoff's A/B/C/D/E rules; authorize no later work."""
    if not valid:
        return {
            'branch': 'E', 'label': 'Invalid or incomplete experiment',
            'next_action': 'Stop and resolve the concrete validity failure before interpreting efficacy.',
            'automatic_followup_authorized': False,
        }
    primary = combined['K_HALF_vs_O_HALF']['call_reduction']
    point = primary['point_pct']
    low, high = primary['ci95_pct']
    seeds = [per_seed[str(seed)]['K_HALF_vs_O_HALF']['call_reduction']['point_pct']
             for seed in SEEDS]
    pooled_equivalence = low >= -0.5 and high <= 0.5
    seed_consistency = all(-0.5 <= value <= 0.5 for value in seeds)
    useful_k = point >= 0.5 and low > 0 and all(value > 0 for value in seeds)
    useful_o = point <= -0.5 and high < 0 and all(value < 0 for value in seeds)
    if useful_k:
        branch, label = 'A', 'Useful matched-LR advantage for K_HALF'
        action = ('Keep K_HALF as the provisional coding base; recommend a separately agreed '
                  'bounded portability or online-setting check of this fixed base.')
    elif pooled_equivalence and seed_consistency:
        branch, label = 'B', 'Practical equivalence with consistent seed point estimates'
        action = ('Prefer ordinary O_HALF as the simpler provisional coding base; retire the '
                  'claim of an established incremental K benefit at this recipe.')
    elif useful_o:
        branch, label = 'C', 'Useful advantage for O_HALF'
        action = ('Adopt O_HALF provisionally as the coding base; the implemented K selection '
                  'and normalization rule is not supported as the core improvement here.')
    else:
        branch = 'D'
        if min(seeds) < 0 < max(seeds) or (pooled_equivalence and not seed_consistency):
            label = 'Mixed fitted-seed directions or pooled equivalence without seed consistency'
        elif abs(point) < 0.5:
            label = 'Small directional effect; no useful-superiority or equivalence decision'
        else:
            label = 'Unresolved matched-LR difference'
        action = ('Keep K_HALF provisional and review whether a separately scoped precision '
                  'or seed-consistency check could resolve the base choice; no new run is authorized.')
    return {
        'branch': branch, 'label': label, 'primary_seed_effects_pct': seeds,
        'useful_K_HALF_advantage': useful_k, 'useful_O_HALF_advantage': useful_o,
        'pooled_practical_equivalence': pooled_equivalence,
        'equivalence_seed_point_consistency': seed_consistency,
        'equivalence_band_pct': list(EQUIVALENCE_BAND_PCT),
        'priority_target_pct': PRIORITY_PCT,
        'true_effect_above_priority_target_supported': low > PRIORITY_PCT,
        'within_objective_LR_point_effects_pct': {
            name: combined[name]['call_reduction']['point_pct']
            for name in ('K_HALF_vs_K', 'O_HALF_vs_O')},
        'next_action': action, 'automatic_followup_authorized': False,
        'scope': ('Development/ablation evidence conditional on this reused panel and two fixed '
                  'fitted seeds; equal LR does not match displacement or isolate selection from normalization.'),
    }


def _scope_summary(arrays, indices):
    calls = {method: values['verification_calls'] for method, values in arrays.items()}
    return {
        'comparisons': {left + '_vs_' + right: comparison(calls[left], calls[right], indices)
                        for left, right in COMPARISONS},
        'reverse_primary': comparison(calls['O_HALF'], calls['K_HALF'], indices),
        'objective_lr_interaction': objective_lr_interaction(calls, indices),
        'method_totals': {method: method_totals(values) for method, values in arrays.items()},
        'gain_harm': {left + '_vs_' + right: gain_harm(calls['F'], calls[right], calls[left], indices)
                      for left, right in DECOMPOSITIONS},
    }


def summarize(rows):
    """Join validated counts without ever importing archived/new serving times."""
    keys, arrays = aligned_arrays(rows, methods=METHODS, fields=COUNT_FIELDS)
    indices = bootstrap_indices(len(keys), seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
    combined_scope = _scope_summary(arrays, indices)
    per_seed_scopes = {}
    for position, seed in enumerate(SEEDS):
        subset = {method: {field: value[position:position + 1] for field, value in fields.items()}
                  for method, fields in arrays.items()}
        per_seed_scopes[str(seed)] = _scope_summary(subset, indices)
    per_seed = {seed: value['comparisons'] for seed, value in per_seed_scopes.items()}
    return {
        'request_count': len(keys), 'request_seed_count': len(SEEDS) * len(keys),
        'bootstrap': {'resamples': RESAMPLES, 'seed': BOOTSTRAP_SEED, 'generator': 'PCG64',
                      'percentiles': [2.5, 97.5], 'method': 'linear',
                      'unit': 'One request identity carrying both fitted seeds and every method.',
                      'scope': ('Nominal descriptive intervals conditional on two fitted trajectories and '
                                'a reused development panel; no adjustment for prior K_HALF selection.')},
        'combined': combined_scope['comparisons'], 'per_seed': per_seed,
        'reverse_primary': {'combined': combined_scope['reverse_primary'],
                            **{seed: value['reverse_primary'] for seed, value in per_seed_scopes.items()}},
        'objective_lr_interaction': {'combined': combined_scope['objective_lr_interaction'],
                                    **{seed: value['objective_lr_interaction'] for seed, value in per_seed_scopes.items()}},
        'method_totals': {'combined': combined_scope['method_totals'],
                          **{seed: value['method_totals'] for seed, value in per_seed_scopes.items()}},
        'gain_harm': {'combined': combined_scope['gain_harm'],
                      **{seed: value['gain_harm'] for seed, value in per_seed_scopes.items()}},
        'interpretation': interpret(combined_scope['comparisons'], per_seed),
        'secondary_intervals': 'Descriptive and unadjusted; no fresh confirmatory significance claim.',
        'latency_comparison': 'Not supported: checkpoint-batched new counts and archived measurements.',
    }
