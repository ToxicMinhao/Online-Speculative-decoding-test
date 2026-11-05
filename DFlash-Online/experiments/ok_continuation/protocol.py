"""Two measured arms, retained F outputs, and reporting-only F call counts."""

from experiments.fresh_confirmation.protocol import (
    ARMS, REPEATS, REQUESTS, GENERATIONS, QUALIFICATION_REQUESTS, COMPARISONS,
    COUNT_FIELDS, method_order, publication_errors, weight_check_errors,
    repeat_errors, validate_inputs as validate_prompt_inputs,
    record_errors as validate_native_record,
    measurement_errors as validate_measurement,
)


def prepare_requests(requests):
    """Use manifest order when older retained inputs have no panel ordinal."""
    return [dict(request, panel_ordinal=request.get('panel_ordinal', ordinal))
            for ordinal, request in enumerate(requests)]


def validate_inputs(requests, checkpoints, frozen_counts, qualification_requests):
    prompts = [{key: value for key, value in request.items()
                if key not in ('response_ids', 'finish_reason')} for request in requests]
    validate_prompt_inputs(prompts, checkpoints, qualification_requests)
    if len(requests) > REQUESTS:
        raise ValueError('Evaluation exceeds the retained 64-request panel')
    if set(frozen_counts) != {request['request_id'] for request in requests}:
        raise ValueError('F counts must match the exact request identities')
    if any(type(value) is not int or value <= 0 for value in frozen_counts.values()):
        raise ValueError('F counts must be positive integers')
    for request in requests:
        response = request.get('response_ids')
        if (not isinstance(response, list) or not 1 <= len(response) <= 2048
                or any(type(token) is not int or token < 0 for token in response)):
            raise ValueError('Retain the canonical F output tokens')
        stops = request['stop_token_ids']
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
    return {request['request_id']: {
        'response_ids': list(request['response_ids']),
        'finish_reason': request['finish_reason'],
        'source_method': 'F',
        'source': 'retained_manifest_eval_requests',
    } for request in requests}


def record_errors(record, request, checkpoint):
    # The reviewed helper validates canonical tokens and native trajectories.
    # Its historical O-count branch stays disabled for both measured arms.
    return validate_native_record(record, request, checkpoint, canonical=request)


def measurement_errors(row, request):
    return validate_measurement(row, request, canonical=request)
