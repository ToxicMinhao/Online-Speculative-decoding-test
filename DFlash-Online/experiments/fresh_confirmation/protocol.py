"""Two-arm contracts without historical counts or precomputed output labels."""

import math

from experiments.osd_components.protocol import (
    publication_errors,
    record_errors as native_record_errors,
    weight_check_errors,
)
from experiments.prefix_ranking.protocol import (
    COUNT_FIELDS,
    measurement_errors as scalar_errors,
)

ARMS = ('O', 'K')
REPEATS = 3
REQUESTS = 64
GENERATIONS = 384
QUALIFICATION_REQUESTS = 8
COMPARISONS = (('K', 'O'),)


def method_order(ordinal, repeat):
    shift = (ordinal + repeat) % 2
    return ARMS[shift:] + ARMS[:shift]


def repeat_errors(rows, method):
    selected = [row for row in rows if row['method'] == method]
    if any(type(row.get(name)) is not int or row[name] < 0 for row in selected for name in COUNT_FIELDS):
        return ['invalid_repeat_counts']
    counts = {tuple(row[name] for name in COUNT_FIELDS) for row in selected}
    return ['repeat_count_mismatch'] if len(counts) > 1 else []


def validate_inputs(requests, checkpoints, qualification_requests):
    """The campaign seals64 requests; short CPU fixtures may use fewer."""
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != 2:
        raise ValueError('Require exactly two distinct O/K checkpoints')
    if not requests or [r['panel_ordinal'] for r in requests] != list(range(len(requests))):
        raise ValueError('Keep consecutive zero-based request ordinals')
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Request identities must be unique')
    if type(qualification_requests) is not int or not 1 <= qualification_requests <= len(requests):
        raise ValueError('Qualification must be inside the request budget')
    for request in requests:
        if 'response_ids' in request or 'finish_reason' in request:
            raise ValueError('Fresh canonical output must come from the first O execution')
        if request['max_new_tokens'] != 2048:
            raise ValueError('Keep the qualified2048-token-or-EOS cap')
        if not request['prompt_ids'] or len(request['prompt_ids']) + 2048 > 4096:
            raise ValueError('Prompt must fit the full serving context without truncation')
        if not request['stop_token_ids'] or any(type(v) is not int or v < 0 for v in request['stop_token_ids']):
            raise ValueError('Require the pinned stop-token identities')
        if any(type(v) is not int or v < 0 for v in request['prompt_ids']):
            raise ValueError('Prompt IDs must be nonnegative integers')
        if type(request['original_index']) is not int or request['original_index'] < 0:
            raise ValueError('Keep the source dataset index')


def expected_output(request, record, canonical=None):
    """Use a record's own output only for structural validation before O exists."""
    source = canonical if canonical is not None else record
    return {**request, 'response_ids': source['response_ids'], 'finish_reason': source['finish_reason']}


def record_errors(record, request, checkpoint, canonical=None):
    expected = expected_output(request, record, canonical)
    # None disables the helper's legacy F/O historical-count branch. Actual
    # measured arms remain O/K; no placeholder arm or count is introduced.
    errors = native_record_errors(record, expected, None, checkpoint)
    if any(type(token) is not int or token < 0 for token in record['response_ids']):
        errors.append('invalid_response_token')
    return errors


def measurement_errors(row, request, canonical=None):
    """Validate scalar structure even when canonical comparison is pending."""
    expected = dict(request)
    if canonical is None:
        output_count = row.get('output_tokens')
        expected['response_ids'] = range(min(output_count, 2049) if type(output_count) is int and output_count >= 0 else 0)
        expected['finish_reason'] = row.get('finish_reason')
    else:
        expected.update(response_ids=canonical['response_ids'], finish_reason=canonical['finish_reason'])
    errors = scalar_errors(row, expected)
    for name in ('request_latency_s', 'native_generate_s'):
        if type(row.get(name)) not in (int, float):
            errors.append('invalid_' + name)
    if row.get('finish_reason') not in ('eos', 'length'):
        errors.append('invalid_finish_reason')
    if canonical is not None and row.get('finish_reason') != canonical['finish_reason']:
        errors.append('canonical_finish_reason')
    value = row.get('cache_maintenance_s')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        errors.append('invalid_cache_maintenance_s')
    if type(row.get('output_tokens')) is int and row['output_tokens'] > 2048:
        errors.append('output_cap')
    return errors
