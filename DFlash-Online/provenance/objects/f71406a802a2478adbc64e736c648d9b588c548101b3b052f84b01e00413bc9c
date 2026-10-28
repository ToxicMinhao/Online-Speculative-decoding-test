"""Native KR/KRC contracts for two unchanged panels and retained KRC counts."""

from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, publication_errors, weight_check_errors, repeat_errors,
    record_errors as validate_native_record,
    measurement_errors as validate_measurement,
)

ARMS = ('KR', 'KRC')
REPEATS = 3
REQUESTS = 64
GENERATIONS = REQUESTS * len(ARMS) * REPEATS
QUALIFICATION_REQUESTS = 8
COMPARISONS = (('KR', 'KRC'),)


def method_order(ordinal, repeat):
    shift = (ordinal + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def prepare_requests(requests, retained_krc_counts):
    """The retained manifest's order is the measured panel order."""
    return [dict(request, panel_ordinal=request.get('panel_ordinal', ordinal),
                 retained_krc_calls=retained_krc_counts[request['request_id']])
            for ordinal, request in enumerate(requests)]


def validate_inputs(requests, checkpoints, retained_krc_counts, qualification_requests):
    """The sealed campaign has 64 requests; smaller CPU fixtures are allowed."""
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != 2:
        raise ValueError('Require exactly two distinct KR/KRC checkpoints')
    if not requests or len(requests) > REQUESTS:
        raise ValueError('Require requests within the retained 64-request panel')
    if [r['panel_ordinal'] for r in requests] != list(range(len(requests))):
        raise ValueError('Keep consecutive zero-based request ordinals')
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Request identities must be unique')
    if type(qualification_requests) is not int or not 0 <= qualification_requests <= len(requests):
        raise ValueError('Qualification must be inside the request budget')
    if retained_krc_counts is None:
        raise ValueError('Retained per-request KRC counts are required')
    else:
        if set(retained_krc_counts) != {r['request_id'] for r in requests}:
            raise ValueError('KRC counts must match the exact request identities')
        if any(type(value) is not int or value <= 0 for value in retained_krc_counts.values()):
            raise ValueError('KRC counts must be positive integers')
    for request in requests:
        if request.get('canonical_source_method') not in ('F', 'K1'):
            raise ValueError('Bind each retained canonical to F or K1')
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
            raise ValueError('Retain the canonical output tokens')
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
            raise ValueError('Retain canonical termination')


def canonical_outputs(requests):
    return {r['request_id']: {
        'response_ids': list(r['response_ids']), 'finish_reason': r['finish_reason'],
        'source_method': r['canonical_source_method'], 'source': 'retained_panel_canonical',
    } for r in requests}


def record_errors(record, request, checkpoint):
    # This helper has no hard-coded measured-arm set. Supplying the canonical
    # request checks outputs while leaving its legacy historical O-count gate off.
    return validate_native_record(record, request, checkpoint, canonical=request)


def measurement_errors(row, request):
    errors = validate_measurement(row, request, canonical=request)
    if (row.get('method') == 'KRC'
            and row.get('verification_calls') != request['retained_krc_calls']):
        errors.append('retained_KRC_verification_calls')
    return errors


def validate_plan(plan):
    """Validate the sealed two-panel schema before model construction.

    PLAN contains config/ordinary_receipt/k_receipt/pinned_initial/reference_schedule/replay_schedule,
    target/draft/worker_python, retained_krc_checkpoint, and panels in this order:
      [{name: 'development', requests: [...64 rows...], retained_krc_counts: {id: n}},
       {name: 'additional', requests: [...64 rows...], retained_krc_counts: {id: n}}].
    Each request retains prompt_ids, response_ids, finish_reason, original_index,
    stop_token_ids, max_new_tokens and canonical_source_method ('F' or 'K1').
    Checkpoint/source hashes and canonical-file provenance belong to the outer seal.
    """
    for key in ('config', 'ordinary_receipt', 'k_receipt', 'pinned_initial',
                'reference_schedule', 'replay_schedule', 'target', 'draft', 'worker_python',
                'retained_krc_checkpoint'):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError('Missing sealed path: ' + key)
    panels = plan['panels']
    if [panel['name'] for panel in panels] != ['development', 'additional']:
        raise ValueError('Evaluate development then additional, separately')
    all_ids, all_prompts = set(), set()
    for panel, canonical in zip(panels, ('F', 'K1'), strict=True):
        requests = prepare_requests(panel['requests'], panel['retained_krc_counts'])
        if len(requests) != REQUESTS:
            raise ValueError('Both sealed panels must contain exactly 64 requests')
        if any(row['canonical_source_method'] != canonical for row in requests):
            raise ValueError('Wrong panel-specific retained canonical source')
        validate_inputs(requests, {'KR': '__new_KR__', 'KRC': plan['retained_krc_checkpoint']},
                        panel['retained_krc_counts'], 8 if canonical == 'F' else 0)
        ids = {row['request_id'] for row in requests}
        prompts = {tuple(row['prompt_ids']) for row in requests}
        if ids & all_ids or prompts & all_prompts or len(prompts) != REQUESTS:
            raise ValueError('The two fixed panels must remain distinct and deduplicated')
        all_ids.update(ids); all_prompts.update(prompts)
    return dict(passed=True, panels=['development', 'additional'], requests=128,
                planned_generations=768, included_gate_generations=48)
