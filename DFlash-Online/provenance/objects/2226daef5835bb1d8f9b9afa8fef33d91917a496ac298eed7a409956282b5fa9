"""Native K/KC contracts: exact retained F outputs, no historical-count gate."""

from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, publication_errors, weight_check_errors, repeat_errors,
    record_errors as validate_native_record,
    measurement_errors as validate_measurement,
)

ARMS = ('K', 'KC')
REPEATS = 3
REQUESTS = 64
GENERATIONS = REQUESTS * len(ARMS) * REPEATS
QUALIFICATION_REQUESTS = 8
COMPARISONS = (('KC', 'K'),)


def method_order(ordinal, repeat):
    shift = (ordinal + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def prepare_requests(requests):
    """The retained manifest's order is the measured panel order."""
    return [dict(request, panel_ordinal=request.get('panel_ordinal', ordinal))
            for ordinal, request in enumerate(requests)]


def validate_inputs(requests, checkpoints, frozen_counts, qualification_requests):
    """The sealed campaign has 64 requests; smaller CPU fixtures are allowed."""
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != 2:
        raise ValueError('Require exactly two distinct K/KC checkpoints')
    if not requests or len(requests) > REQUESTS:
        raise ValueError('Require requests within the retained 64-request panel')
    if [r['panel_ordinal'] for r in requests] != list(range(len(requests))):
        raise ValueError('Keep consecutive zero-based request ordinals')
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Request identities must be unique')
    if type(qualification_requests) is not int or not 1 <= qualification_requests <= len(requests):
        raise ValueError('Qualification must be inside the request budget')
    if frozen_counts is not None:
        if set(frozen_counts) != {r['request_id'] for r in requests}:
            raise ValueError('F counts must match the exact request identities')
        if any(type(value) is not int or value <= 0 for value in frozen_counts.values()):
            raise ValueError('F counts must be positive integers')
    for request in requests:
        if request['max_new_tokens'] != 2048:
            raise ValueError('Keep the qualified 2048-token-or-EOS cap')
        prompt = request['prompt_ids']
        if not prompt or len(prompt) + 2048 > 4096:
            raise ValueError('Prompt must fit the full serving context without truncation')
        if any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError('Prompt IDs must be nonnegative integers')
        if type(request['original_index']) is not int or request['original_index'] < 0:
            raise ValueError('Keep the source dataset index')
        stops = request['stop_token_ids']
        if not stops or any(type(token) is not int or token < 0 for token in stops):
            raise ValueError('Require the pinned stop-token identities')
        response = request.get('response_ids')
        if (not isinstance(response, list) or not 1 <= len(response) <= 2048
                or any(type(token) is not int or token < 0 for token in response)):
            raise ValueError('Retain the canonical F output tokens')
        if any(token in stops for token in response[:-1]):
            raise ValueError('Retained output crosses an EOS boundary')
        finish = request.get('finish_reason')
        if finish == 'eos':
            if response[-1] not in stops:
                raise ValueError('Retained EOS termination must end in a stop token')
        elif finish == 'length':
            if len(response) != 2048 or response[-1] in stops:
                raise ValueError('Retained length termination must reach the output cap')
        else:
            raise ValueError('Retain canonical F termination')


def canonical_outputs(requests):
    return {r['request_id']: {
        'response_ids': list(r['response_ids']), 'finish_reason': r['finish_reason'],
        'source_method': 'F', 'source': 'retained_manifest_eval_requests',
    } for r in requests}


def record_errors(record, request, checkpoint):
    # This helper has no hard-coded measured-arm set. Supplying the canonical
    # request checks outputs while leaving its legacy historical O-count gate off.
    return validate_native_record(record, request, checkpoint, canonical=request)


def measurement_errors(row, request):
    return validate_measurement(row, request, canonical=request)
