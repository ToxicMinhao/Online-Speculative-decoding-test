"""Fixed three-arm KW-v1 evaluation contracts for events 10 and 12."""
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, publication_errors, weight_check_errors, repeat_errors,
    record_errors, measurement_errors,
)

ARMS = ('O', 'K', 'KW')
REPEATS = 3
REQUESTS = 64
GENERATIONS = 576
TOTAL_GENERATIONS = 4032
QUALIFICATION_REQUESTS = 8
COMPARISONS = (('KW', 'K'), ('KW', 'O'), ('K', 'O'))
PANELS = ('development', 'additional', 'fresh', 'final')
CANONICAL_SOURCES = dict(development='F', additional='K1', fresh='O8')


def method_order(ordinal, repeat):
    base = ARMS if ordinal % 2 == 0 else ('O', 'KW', 'K')
    shift = (ordinal + repeat) % 3
    return base[shift:] + base[:shift]


def prepare_requests(requests):
    return [dict(request, panel_ordinal=request.get('panel_ordinal', index))
            for index, request in enumerate(requests)]


def validate_inputs(requests, checkpoints, qualification_requests, *, event_number,
                    panel_name, canonical_mode):
    if event_number not in (10, 12) or panel_name not in PANELS:
        raise ValueError('Evaluate only fixed event-10/12 panels')
    if panel_name == 'final' and (event_number != 12 or canonical_mode != 'fresh'):
        raise ValueError('Final panel is evaluated only at event 12 with first-O12 canonicals')
    if panel_name != 'final' and canonical_mode != 'retained':
        raise ValueError('Existing panels require retained exact canonical outputs')
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != 3:
        raise ValueError('Require exactly three distinct O/K/KW checkpoints')
    if (not requests or len(requests) > REQUESTS
            or [row['panel_ordinal'] for row in requests] != list(range(len(requests)))
            or len({row['request_id'] for row in requests}) != len(requests)):
        raise ValueError('Require 1..64 unique consecutively ordered requests')
    if (type(qualification_requests) is not int or qualification_requests not in (0, QUALIFICATION_REQUESTS)
            or qualification_requests > len(requests)
            or (qualification_requests and (event_number != 10 or panel_name != 'development'))):
        raise ValueError('The included correctness gate belongs only to event-10 development')
    if len(requests) == REQUESTS and event_number == 10 and panel_name == 'development' and qualification_requests != 8:
        raise ValueError('Event-10 development requires its included eight-request gate')
    for row in requests:
        if row['max_new_tokens'] != 2048:
            raise ValueError('Keep the qualified 2048-token-or-EOS cap')
        if not row['prompt_ids'] or len(row['prompt_ids']) + 2048 > 4096:
            raise ValueError('Prompt must fit the full context without truncation')
        if any(type(token) is not int or token < 0 for token in row['prompt_ids']):
            raise ValueError('Prompt tokens must be nonnegative integers')
        if not row['stop_token_ids'] or any(type(token) is not int or token < 0 for token in row['stop_token_ids']):
            raise ValueError('Require pinned stop-token identities')
        if type(row['original_index']) is not int or row['original_index'] < 0:
            raise ValueError('Keep source dataset index')
    if canonical_mode == 'retained':
        source = CANONICAL_SOURCES[panel_name]
        if any(row.get('canonical_source_method') != source for row in requests):
            raise ValueError('Retained panel canonical source differs')
        for row in requests:
            tokens, stops = row.get('response_ids'), row['stop_token_ids']
            if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 2048
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or any(token in stops for token in tokens[:-1])):
                raise ValueError('Require retained canonical tokens through the first EOS')
            if row.get('finish_reason') == 'eos':
                if tokens[-1] not in stops:
                    raise ValueError('Retained EOS output must end with the pinned stop token')
            elif row.get('finish_reason') == 'length':
                if len(tokens) != 2048 or tokens[-1] in stops:
                    raise ValueError('Retained capped output must reach 2048 without EOS')
            else:
                raise ValueError('Require retained canonical termination')
    else:
        if any('response_ids' in row or 'finish_reason' in row for row in requests):
            raise ValueError('Fresh canonicals come from the first O12 execution')


def canonical_outputs(requests, canonical_mode):
    if canonical_mode == 'fresh':
        return {}
    return {row['request_id']: dict(response_ids=list(row['response_ids']),
        finish_reason=row['finish_reason'], source_method=row['canonical_source_method'],
        source_record_path='retained_panel_canonical', source='retained_panel_canonical')
        for row in requests}


def evaluation_schedule(requests):
    """Pin every ordered slot before the first generation."""
    return [dict(panel_ordinal=row['panel_ordinal'], request_id=row['request_id'],
                 repeat=repeat, method=method, position=position)
            for row in requests for repeat in range(REPEATS)
            for position, method in enumerate(method_order(row['panel_ordinal'], repeat))]


def validate_panels(panels):
    if [panel['name'] for panel in panels] != list(PANELS):
        raise ValueError('Seal development, additional, fresh and final panels in that order')
    ids, prompts = set(), set()
    for panel in panels:
        mode = 'fresh' if panel['name'] == 'final' else 'retained'
        if panel.get('canonical_mode') != mode:
            raise ValueError('Panel canonical mode differs from its fixed role')
        requests = prepare_requests(panel['requests'])
        if len(requests) != REQUESTS:
            raise ValueError('Every sealed evaluation panel requires exactly 64 requests')
        validate_inputs(requests, {arm: '/new_' + arm for arm in ARMS}, 0,
                        event_number=12, panel_name=panel['name'], canonical_mode=mode)
        new_ids = {row['request_id'] for row in requests}
        new_prompts = {tuple(row['prompt_ids']) for row in requests}
        if len(new_prompts) != REQUESTS or ids & new_ids or prompts & new_prompts:
            raise ValueError('Evaluation panels overlap by identity or prompt content')
        ids.update(new_ids)
        prompts.update(new_prompts)
        if mode == 'fresh' and panel.get('selection_seed') != 20261030:
            raise ValueError('Final panel must use the sealed seed 20261030')
    return dict(status='passed', evaluation_events=[10, 12], panel_requests=256,
                main_generations=TOTAL_GENERATIONS, included_gate_generations=72,
                extra_reference_generations=0, historical_count_gates=False)
