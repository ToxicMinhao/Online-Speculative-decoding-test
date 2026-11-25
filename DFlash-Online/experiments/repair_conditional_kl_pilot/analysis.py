"""Paired whole-request pilot analysis; repetitions estimate timing only."""
from collections import defaultdict
import hashlib
import math

import numpy as np

from experiments.opencode_native_comparison.contract import digest
from experiments.opencode_native_comparison.analysis import _interval
from .evaluation import COUNT_FIELDS, METHODS, REQUESTS, REPEATS

TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261124, 10000


def reduce_requests(rows, *, expected_requests=REQUESTS):
    groups = defaultdict(list)
    for row in rows:
        if (row['phase'] not in ('warmup', 'scored') or row['method'] not in METHODS
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Invalid evidence cannot enter pilot gain calculations')
        if row['phase'] == 'scored':
            groups[row['request_id'], row['method']].append(row)
    ids = sorted({key[0] for key in groups})
    if len(ids) != expected_requests or len(groups) != len(ids) * len(METHODS):
        raise ValueError('Incomplete two-arm request panel')
    for method in METHODS:
        endpoints = {row['checkpoint_id'] for group in groups.values() for row in group
                     if row['method'] == method}
        if len(endpoints) != 1:
            raise ValueError('A method changed endpoint across requests')
    if len({row['physical_gpu_uuid'] for row in rows}) != 1:
        raise ValueError('All timing must use one fixed physical card')
    result = []
    for request_id in ids:
        if len({row['source_id'] for method in METHODS for row in groups[request_id, method]}) != 1:
            raise ValueError('Source identity differs across arms')
        item = dict(request_id=request_id)
        for method in METHODS:
            records = sorted(groups[request_id, method], key=lambda row: row['repeat'])
            if [row['repeat'] for row in records] != list(range(REPEATS)):
                raise ValueError('Exactly three distinct timing repeats are required')
            first = records[0]
            for key in (*COUNT_FIELDS, 'finish_reason', 'checkpoint_id', 'physical_gpu_uuid', 'source_id'):
                if any(row[key] != first[key] for row in records):
                    raise ValueError('Repeat counts or identity changed: ' + key)
            for key in TIMES:
                if any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                       or row[key] < 0 or (key != 'cache_maintenance_s' and row[key] == 0)
                       for row in records):
                    raise ValueError('Invalid latency: ' + key)
            if any(row['native_generate_s'] > row['request_latency_s'] for row in records):
                raise ValueError('Native generation exceeds complete-request timing')
            if any(type(first[key]) is not int or first[key] < 0 for key in COUNT_FIELDS):
                raise ValueError('Native counts must be nonnegative integers')
            calls, progress, output = first['verification_calls'], first['committed_progress'], first['output_tokens']
            if (not 1 <= output <= 2048 or first['finish_reason'] not in ('eos', 'length')
                    or (first['finish_reason'] == 'length' and output != 2048)
                    or progress != output - 1
                    or first['native_accepted_draft_tokens'] != progress - calls
                    or first['accepted_draft_tokens'] > progress
                    or (calls == 0 and (progress != 0 or first['accepted_draft_tokens'] != 0))):
                raise ValueError('Native count identities or termination do not reconcile')
            item[method] = {key: first[key] for key in (*COUNT_FIELDS, 'finish_reason')}
            item[method].update({key: float(np.mean([row[key] for row in records])) for key in TIMES})
            item[method]['repeat_times'] = {key: [row[key] for row in records] for key in TIMES}
        if len({(item[method]['output_tokens'], item[method]['finish_reason']) for method in METHODS}) != 1:
            raise ValueError('Paired output length or termination differs')
        result.append(item)
    return result


def summarize(rows, *, expected_requests=REQUESTS):
    paired = reduce_requests(rows, expected_requests=expected_requests)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)))
    values = {method: {key: np.asarray([row[method][key] for row in paired], dtype=np.float64)
                       for key in (*TIMES, *COUNT_FIELDS)} for method in METHODS}
    totals = {}
    for method, metrics in values.items():
        sums = {key: float(value.sum()) for key, value in metrics.items()}
        calls = sums['verification_calls']
        for name, key in (('native_accepted_per_call', 'native_accepted_draft_tokens'),
                          ('stop_clipped_accepted_per_call', 'accepted_draft_tokens'),
                          ('committed_per_call', 'committed_progress')):
            sums[name] = sums[key] / calls if calls else None
        sums['tokens_per_s'] = sums['output_tokens'] / sums['request_latency_s']
        sums['repeat_totals'] = {key: [sum(row[method]['repeat_times'][key][r] for row in paired)
                                       for r in range(REPEATS)] for key in TIMES}
        sums['termination_counts'] = {kind: sum(row[method]['finish_reason'] == kind for row in paired)
                                     for kind in ('eos', 'length')}
        sums['output_length'] = dict(min=int(metrics['output_tokens'].min()), max=int(metrics['output_tokens'].max()),
            mean=float(metrics['output_tokens'].mean()), median=float(np.median(metrics['output_tokens'])))
        totals[method] = sums
    comparison = dict(candidate='PAIR', baseline='BASE')
    for output_name, key in [(key + '_reduction_pct', key) for key in TIMES] + [
            ('verification_call_reduction_pct', 'verification_calls')]:
        current, reference = values['PAIR'][key], values['BASE'][key]
        point = 100 * (1 - current.sum() / reference.sum()) if reference.sum() else float('nan')
        with np.errstate(divide='ignore', invalid='ignore'):
            samples = 100 * (1 - current[draws].sum(1) / reference[draws].sum(1))
        comparison[output_name] = _interval(point, samples)
    if comparison['request_latency_s_reduction_pct']['status'] != 'undefined':
        comparison['request_latency_s_reduction_pct']['status'] = 'prespecified_primary'
    difference = values['BASE']['verification_calls'] - values['PAIR']['verification_calls']
    comparison.update(wins=int((difference > 0).sum()), regressions=int((difference < 0).sum()),
        ties=int((difference == 0).sum()), gross_calls_saved=int(np.maximum(difference, 0).sum()),
        gross_calls_added=int(np.maximum(-difference, 0).sum()), net_calls_saved=int(difference.sum()))
    ordered_ids = [row['request_id'] for row in paired]
    return dict(requests=len(paired), totals=totals, contrasts={'PAIR_vs_BASE': comparison}, paired=paired,
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, generator='PCG64', seed=BOOTSTRAP_SEED,
            input_order=ordered_ids, input_order_digest=digest(ordered_ids),
            draw_sha256=hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5; paired whole requests; average timing repeats within each request',
            counts='one validated repeat per request; timing repeats are not independent training runs'))
