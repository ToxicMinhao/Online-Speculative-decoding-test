"""Prespecified K+58/A58 primary latency comparison and O58 context."""
from collections import defaultdict
import numpy as np

from .contract import METHODS
from .evaluation import REPEATS
from experiments.full_k_gsm8k.statistics import comparison, method_totals

COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMINGS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
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
    if (len(identities) != 1024 or sorted(identities) != list(range(1024))
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


def interpret(effect, *, valid=True):
    """The primary decision is the paired complete-request latency interval."""
    if not valid:
        return dict(endpoint_label='invalid_or_incomplete', automatic_followup_authorized=False)
    low, high = effect['request_latency_s_reduction']['ci95_pct']
    label = 'favors_KPLUS' if low > 0 else 'favors_A' if high < 0 else 'inconclusive'
    return dict(endpoint_label=label, primary='KPLUS_vs_A_complete_request_time',
                automatic_followup_authorized=False,
                scope='One seed, previously used panel, endpoint serving latency.')


def summarize(rows):
    paired = reduce_final(rows)
    arrays = {method: {field: np.array([row[method][field] for row in paired], dtype=np.float64)
              for field in (*COUNT_FIELDS, *TIMINGS)} for method in METHODS}
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired)))
    effects = {a + '_vs_' + b: comparison(arrays[a], arrays[b], indices)
               for a, b in (('KPLUS', 'A'), ('KPLUS', 'O'), ('A', 'O'))}
    repeats = {str(repeat): {method: {field: sum(row[field] for row in rows
        if row['repeat'] == repeat and row['method'] == method) for field in (*COUNT_FIELDS, *TIMINGS)}
        for method in METHODS} for repeat in range(REPEATS)}
    return dict(request_count=len(paired), bootstrap=dict(resamples=RESAMPLES, seed=BOOTSTRAP_SEED,
        generator='PCG64', percentiles=[2.5, 97.5], method='linear',
        unit='One paired request carrying every arm; repeats form one mean time.'),
        method_totals={method: method_totals(values) for method, values in arrays.items()},
        comparisons=effects, repeat_totals=repeats,
        interpretation=interpret(effects['KPLUS_vs_A']),
        timing_scope='Fresh contemporaneous checkpoint-ready serving; not training-inclusive speedup.',
        comparison_scope='Own-feedback recipe comparison, not identical-feature replay.')
