"""Fixed K1/KRC arms with the existing fresh-input validity contract."""

from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, REPEATS, REQUESTS, GENERATIONS, QUALIFICATION_REQUESTS,
    measurement_errors, record_errors, repeat_errors,
    publication_errors, weight_check_errors,
)

ARMS = ('K1', 'KRC')
COMPARISONS = (('KRC', 'K1'),)


def method_order(ordinal, repeat):
    shift = (ordinal + repeat) % 2
    return ARMS[shift:] + ARMS[:shift]


def validate_inputs(requests, checkpoints, qualification_requests):
    """Validate sealed inputs once; short CPU fixtures may use fewer requests."""
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != 2:
        raise ValueError('Require exactly two distinct K1/KRC checkpoints')
    if not requests or [row['panel_ordinal'] for row in requests] != list(range(len(requests))):
        raise ValueError('Keep consecutive zero-based request ordinals')
    if len({row['request_id'] for row in requests}) != len(requests):
        raise ValueError('Request identities must be unique')
    if type(qualification_requests) is not int or not 1 <= qualification_requests <= len(requests):
        raise ValueError('Qualification must be inside the request budget')
    for request in requests:
        if 'response_ids' in request or 'finish_reason' in request:
            raise ValueError('Fresh canonical output must come from the first K1 execution')
        if request['max_new_tokens'] != 2048:
            raise ValueError('Keep the qualified 2048-token-or-EOS cap')
        if not request['prompt_ids'] or len(request['prompt_ids']) + 2048 > 4096:
            raise ValueError('Prompt must fit the full context without truncation')
        if not request['stop_token_ids'] or any(type(v) is not int or v < 0 for v in request['stop_token_ids']):
            raise ValueError('Require the pinned stop-token identities')
        if any(type(v) is not int or v < 0 for v in request['prompt_ids']):
            raise ValueError('Prompt IDs must be nonnegative integers')
        if type(request['original_index']) is not int or request['original_index'] < 0:
            raise ValueError('Keep the source dataset index')
