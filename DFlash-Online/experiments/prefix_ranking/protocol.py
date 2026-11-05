"""Four-arm one-event evaluation contracts, independent of training."""

import json
import math
from pathlib import Path

from experiments.osd_components.protocol import (
    publication_errors, record_errors as native_record_errors, weight_check_errors,
)

ARMS = ('F', 'O', 'U', 'P')
REQUESTS = 64
REPEATS = 3
QUALIFICATION_REQUESTS = 8
GENERATIONS = REQUESTS * len(ARMS) * REPEATS
COMPARISONS = (('P', 'O'), ('P', 'U'), ('P', 'F'), ('O', 'F'))
COUNT_FIELDS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')


def method_order(ordinal, repeat, arms=ARMS):
    shift = (ordinal + repeat) % len(arms)
    return arms[shift:] + arms[:shift]


def load_ordinary_counts(path, requests):
    """Read the completed first-event F/O counts, never the old final endpoint."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    request_ids = {request['request_id'] for request in requests}
    if {row['request_id'] for row in rows} != request_ids:
        raise ValueError('Retained ordinary results do not cover the identical request panel')
    counts = {}
    for request in requests:
        selected = [row for row in rows if row['request_id'] == request['request_id']]
        slots = {(row['method'], row['repeat']) for row in selected}
        expected = {(method, repeat) for method in ('F', 'O') for repeat in range(REPEATS)}
        if len(selected) != len(expected) or slots != expected:
            raise ValueError('Retained ordinary results require all three F/O repeats')
        counts[request['request_id']] = {}
        for method in ('F', 'O'):
            group = [row for row in selected if row['method'] == method]
            if any(measurement_errors(row, request) or not row['valid_measurement'] or row['errors']
                   for row in group):
                raise ValueError('Retained ordinary measurements contain invalid evidence')
            values = {row['verification_calls'] for row in group}
            if len(values) != 1:
                raise ValueError('Retained ordinary verification counts do not reproduce')
            counts[request['request_id']][method] = values.pop()
    return counts


def validate_inputs(requests, checkpoints, ordinary_counts, arms=ARMS, required_arms=('F', 'O')):
    """Allow shorter CPU fixtures; the campaign runner seals exactly 64 requests."""
    if (len(arms) != 4 or len(set(arms)) != 4 or not set(required_arms) <= set(arms)
            or set(checkpoints) != set(arms) or len(set(map(str, checkpoints.values()))) != len(arms)):
        raise ValueError('Four distinct checkpoint identities ' + '/'.join(arms) + ' are required')
    if [row['panel_ordinal'] for row in requests] != list(range(len(requests))):
        raise ValueError('Keep the sealed development-panel order')
    ids = [row['request_id'] for row in requests]
    if len(set(ids)) != len(ids) or set(ordinary_counts) != set(ids):
        raise ValueError('Unique requests and complete retained one-event counts are required')
    for request in requests:
        if request['max_new_tokens'] != 2048:
            raise ValueError('Keep the qualified 2048-token-or-EOS limit')
        if set(ordinary_counts[request['request_id']]) != {'F', 'O'}:
            raise ValueError('Both retained first-event F and O counts are required')
        if any(type(value) is not int or value < 0
               for value in ordinary_counts[request['request_id']].values()):
            raise ValueError('Retained counts must be nonnegative integers')


def record_errors(record, request, method, checkpoint, ordinary_counts):
    # Bypass both legacy endpoint names, then apply the retained FIRST-event gate.
    errors = native_record_errors(record, request, 'prefix_ranking_' + method, checkpoint)
    if method in ('F', 'O'):
        if record['metadata']['verification_calls'] != ordinary_counts[request['request_id']][method]:
            errors.append('retained_one_event_count_mismatch')
    return errors


def measurement_errors(row, request, checkpoint=None):
    """Recheck scalar structure even if a durable row claims it is valid."""
    errors = []
    for name in COUNT_FIELDS:
        if type(row.get(name)) is not int or row[name] < 0:
            errors.append('invalid_' + name)
    if not errors:
        if row['output_tokens'] != len(request['response_ids']):
            errors.append('output_token_count')
        if row['committed_progress'] != row['output_tokens'] - 1:
            errors.append('committed_progress_accounting')
        if not 0 <= row['accepted_draft_tokens'] <= row['committed_progress']:
            errors.append('accepted_token_bounds')
        if row['committed_progress'] > 0 and row['verification_calls'] == 0:
            errors.append('zero_verification_calls_with_progress')
    for name in ('request_latency_s', 'native_generate_s'):
        value = row.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append('invalid_' + name)
    if row.get('prompt_tokens') != len(request['prompt_ids']):
        errors.append('prompt_token_count')
    if row.get('cap_hit') != (request['finish_reason'] == 'length'):
        errors.append('cap_hit')
    if checkpoint is not None and row.get('checkpoint_id') != str(checkpoint):
        errors.append('checkpoint_identity')
    return errors


def repeat_errors(rows, method):
    counts = {tuple(row.get(key) for key in COUNT_FIELDS)
              for row in rows if row['method'] == method}
    return ['repeat_count_mismatch'] if len(counts) > 1 else []
