"""Prespecified paired latency analysis for the three frozen MathQA checkpoints.

Timing repeats estimate a request's mean latency. The bootstrap resamples the
1,024 paired requests, carrying all three methods together in every draw.
"""
from collections import defaultdict

import numpy as np

from experiments.full_k_gsm8k.statistics import comparison, method_totals

from .contract import ANALYSIS, METHODS, PANEL_SIZE, REPEATS

COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
BOOTSTRAP_SEED = ANALYSIS['seed']
RESAMPLES = ANALYSIS['resamples']


def reduce_final(rows, requests=None):
    """Require every valid execution; retain one count trajectory and mean time.

    Native token/trace validation belongs to ``evaluation.audit_stage``. This
    reducer repeats the scalar and identity checks so a saved validity flag
    alone cannot admit an inconsistent, incomplete or nonfinite measurement.
    """
    grouped = defaultdict(list)
    identities, checkpoint_ids = {}, {}
    for row in rows:
        if (row['phase'] != 'final' or row['event'] != 58 or row['seed'] != 17
                or row['method'] not in METHODS or row['validation_status'] != 'valid'
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Only valid frozen event-58 final executions enter analysis')
        ordinal, request_id, method = row['panel_ordinal'], row['request_id'], row['method']
        if type(ordinal) is not int or not isinstance(request_id, str) or not request_id:
            raise ValueError('Invalid request identity')
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError('A panel ordinal has multiple request identities')
        if method in checkpoint_ids and checkpoint_ids[method] != row['checkpoint_id']:
            raise ValueError('A method must use one unchanged checkpoint throughout')
        identities[ordinal] = request_id
        checkpoint_ids[method] = row['checkpoint_id']
        grouped[ordinal, method].append(row)
    if (len(rows) != PANEL_SIZE * len(METHODS) * REPEATS
            or sorted(identities) != list(range(PANEL_SIZE))
            or len(set(identities.values())) != PANEL_SIZE):
        raise ValueError('Require all 1024 distinct requests and all 9216 executions')
    if len(set(checkpoint_ids.values())) != len(METHODS):
        raise ValueError('Require distinct O58, K58 and KPLUS58 checkpoint identities')
    if requests is not None and [(r['panel_ordinal'], r['request_id']) for r in requests] != sorted(identities.items()):
        raise ValueError('Final identities differ from the sealed request panel')
    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(seed=17, event=58, panel_ordinal=ordinal, request_id=request_id)
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row['repeat'])
            if [row['repeat'] for row in records] != list(range(REPEATS)):
                raise ValueError('Every method/request requires three distinct repeats')
            first = records[0]
            for field in COUNT_FIELDS:
                if any(type(row[field]) is not int or row[field] < 0 or row[field] != first[field] for row in records):
                    raise ValueError('Invalid or inconsistent repeated counts')
            if (first['verification_calls'] <= 0 or first['output_tokens'] <= 0
                    or first['accepted_draft_tokens'] > first['output_tokens']
                    or first['committed_progress'] > first['output_tokens']):
                raise ValueError('Invalid count denominator or progress/acceptance accounting')
            if (first['finish_reason'] not in ('eos', 'length')
                    or type(first['cap_hit']) is not bool
                    or first['cap_hit'] != (first['finish_reason'] == 'length')
                    or any(row['finish_reason'] != first['finish_reason']
                           or row['cap_hit'] != first['cap_hit'] for row in records)):
                raise ValueError('Invalid or inconsistent termination/cap status')
            if any(not np.isfinite(row[field]) or row[field] < 0 for row in records for field in TIMINGS):
                raise ValueError('Invalid latency measurement')
            if any(row['request_latency_s'] <= 0 for row in records):
                raise ValueError('Complete-request latency must be positive')
            times = {field: float(np.mean([row[field] for row in records])) for field in TIMINGS}
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(**times, cap_hit=first['cap_hit'], finish_reason=first['finish_reason'],
                checkpoint_id=first['checkpoint_id'], count_trajectory_repeat=0, timing_repeats=REPEATS)
        if len({(value[m]['output_tokens'], value[m]['finish_reason']) for m in METHODS}) != 1:
            raise ValueError('Paired output lengths or termination differ')
        paired.append(value)
    return paired


def ordered_interpretation(comparisons):
    """Apply the fixed latency testing order; a failed primary closes confirmation."""
    primary = comparisons['KPLUS_vs_O']['request_latency_s_reduction']
    secondary = comparisons['KPLUS_vs_K']['request_latency_s_reduction']
    primary_pass = primary['ci95_pct'][0] > 0
    secondary_positive = secondary['ci95_pct'][0] > 0
    return dict(
        primary=dict(contrast='KPLUS_vs_O', measure='complete_request_time',
            pass_criterion='lower endpoint of paired 95% interval > 0',
            passed=primary_pass,
            conclusion=('KPLUS58 has lower complete request time than O58 on this panel.' if primary_pass
                        else 'Lower complete request time than O58 is not established.')),
        secondary=dict(contrast='KPLUS_vs_K', role='confirmatory' if primary_pass else 'descriptive',
            interval_above_zero=secondary_positive,
            passed=secondary_positive if primary_pass else None,
            conclusion=('KPLUS58 also improves complete request time over K58 on this panel.'
                        if primary_pass and secondary_positive else
                        'An incremental complete-request-time improvement over K58 is not established.'
                        if primary_pass else
                        'The KPLUS58/K58 estimate is descriptive because the primary test did not pass.')),
        scope='Frozen checkpoints trained on GSM8K; MathQA serving transfer only. No new adaptation or training-seed replication.',
        automatic_followup_authorized=False,
    )


def summarize(rows, requests=None):
    paired = reduce_final(rows, requests)
    arrays = {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
              for field in (*COUNT_FIELDS, *TIMINGS)} for method in METHODS}
    # Generate ONCE. Both contrasts and all descriptive metrics use these draws.
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired)))
    effects = {f'KPLUS_vs_{reference}': comparison(arrays['KPLUS'], arrays[reference], indices)
               for reference in ('O', 'K')}
    totals = {method: method_totals(values) for method, values in arrays.items()}
    for method in METHODS:
        totals[method].update(
            mean_request_ms=1000 * totals[method]['request_latency_s'] / PANEL_SIZE,
            cap_hits=sum(row[method]['cap_hit'] for row in paired),
            eos_responses=sum(row[method]['finish_reason'] == 'eos' for row in paired),
            output_tokens_min=min(row[method]['output_tokens'] for row in paired),
            output_tokens_max=max(row[method]['output_tokens'] for row in paired),
            output_tokens_mean=totals[method]['output_tokens'] / PANEL_SIZE,
        )
    repeats = {str(repeat): {method: {field: sum(row[field] for row in rows
        if row['repeat'] == repeat and row['method'] == method) for field in (*COUNT_FIELDS, *TIMINGS)}
        for method in METHODS} for repeat in range(REPEATS)}
    return dict(request_count=len(paired), execution_count=len(rows),
        bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED, generator='PCG64',
            percentiles=[2.5, 97.5], method='linear', shared_draws_across_contrasts=True,
            unit='One paired request carrying all three methods; repeats form one mean latency.'),
        method_totals=totals, comparisons=effects, repeat_totals=repeats,
        interpretation=ordered_interpretation(effects),
        primary_measure='complete_request_time',
        other_intervals='Call and component-time intervals are descriptive; only ordered total-time tests are confirmatory.',
        timing_scope='Fresh contemporaneous frozen-checkpoint serving. Loading/switching excluded; cache maintenance included.',
        canonical_scope='Cross-arm agreement against fresh O58 repeat0 under the qualified native verifier, not independent target-only generation.',
    )
