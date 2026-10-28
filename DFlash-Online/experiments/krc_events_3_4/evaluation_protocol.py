"""Three-arm native serving contract for the two already retained panels."""

from collections import Counter

from experiments.krc_continuation.evaluation_protocol import (
    canonical_outputs, record_errors,
    validate_inputs as validate_retained_inputs,
)
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, publication_errors, weight_check_errors, repeat_errors,
    measurement_errors as validate_measurement,
)

ARMS = ('O4', 'K4', 'KRC4')
COMPARISONS = (('KRC4', 'O4'), ('K4', 'O4'), ('KRC4', 'K4'))
REPEATS = 3
REQUESTS = 64
GENERATIONS = REQUESTS * len(ARMS) * REPEATS
QUALIFICATION_REQUESTS = 8
SESSION_ID = 'krc_events_3_4_evaluation'
HISTORICAL_ARMS = ('O1', 'K1', 'KRC1', 'O2', 'K2', 'KRC2')


def method_order(ordinal, repeat):
    shift = (ordinal + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def rotation_balance(request_count):
    """64 is not divisible by three; the rotating surplus cancels over repeats."""
    starts = [dict(Counter(method_order(index, repeat)[0]
                           for index in range(request_count)))
              for repeat in range(REPEATS)]
    return dict(
        starts_by_repeat=starts,
        starts_all_repeats={arm: sum(row.get(arm, 0) for row in starts) for arm in ARMS},
        per_request_each_arm_starts_once=True,
        per_request_each_arm_occupies_each_position_once=True,
        note='Per-repeat starts are 22/21/21 for 64 requests; combined repeats are exactly balanced.',
    )


def prepare_requests(requests, expected_control_counts=None):
    return [dict(request, panel_ordinal=request.get('panel_ordinal', ordinal),
                 expected_control_calls={} if expected_control_counts is None else {
                     arm: counts[request['request_id']]
                     for arm, counts in expected_control_counts.items()})
            for ordinal, request in enumerate(requests)]


def validate_count_maps(values, request_ids, *, allowed_arms, required_arms=None):
    if not isinstance(values, dict) or not set(values) <= set(allowed_arms):
        raise ValueError('Unexpected historical/control count arms')
    if required_arms is not None and set(values) != set(required_arms):
        raise ValueError('Require exact historical/control count arms: ' + str(required_arms))
    for arm, counts in values.items():
        if (not isinstance(counts, dict) or set(counts) != request_ids
                or any(type(value) is not int or value <= 0 for value in counts.values())):
            raise ValueError('Count map must bind positive integer counts to exact request identities: ' + arm)


def validate_inputs(requests, checkpoints, expected_control_counts, qualification_requests):
    if set(checkpoints) != set(ARMS) or len(set(map(str, checkpoints.values()))) != len(ARMS):
        raise ValueError('Require exactly three distinct O4/K4/KRC4 checkpoints')
    # Reuse the qualified canonical, prompt, stop-token, context, ordinal and cap
    # contracts. Those checks are independent of which checkpoints are measured.
    validate_retained_inputs(requests, {'K2': checkpoints['K4'], 'KRC2': checkpoints['KRC4']},
                             None, qualification_requests)
    if expected_control_counts is not None:
        validate_count_maps(expected_control_counts, {r['request_id'] for r in requests},
                            allowed_arms=('O4', 'K4'), required_arms=('O4', 'K4'))


def measurement_errors(row, request):
    errors = validate_measurement(row, request, canonical=request)
    expected = request['expected_control_calls'].get(row.get('method'))
    if expected is not None and row.get('verification_calls') != expected:
        errors.append('retained_' + row['method'] + '_verification_calls')
    return errors


def validate_plan(plan):
    """Validate only evaluation fields; the campaign owns training contracts."""
    for key in ('retained_o4_checkpoint', 'retained_k4_checkpoint'):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError('Missing sealed path: ' + key)
    panels = plan['panels']
    if [panel['name'] for panel in panels] != ['development', 'additional']:
        raise ValueError('Evaluate development then additional, separately')
    all_ids, all_prompts = set(), set()
    for index, panel in enumerate(panels):
        counts = panel['expected_control_counts']
        if (index == 0 and counts is None) or (index == 1 and counts is not None):
            raise ValueError('Only development has retained O4/K4 count gates')
        requests = prepare_requests(panel['requests'], counts)
        if len(requests) != REQUESTS:
            raise ValueError('Both sealed panels require exactly 64 requests')
        if any(row['canonical_source_method'] != ('F', 'K1')[index] for row in requests):
            raise ValueError('Wrong panel-specific retained canonical source')
        validate_inputs(requests, {'O4': plan['retained_o4_checkpoint'],
                                  'K4': plan['retained_k4_checkpoint'], 'KRC4': '__new_KRC4__'},
                        counts, QUALIFICATION_REQUESTS if index == 0 else 0)
        ids = {row['request_id'] for row in requests}
        prompts = {tuple(row['prompt_ids']) for row in requests}
        if ids & all_ids or prompts & all_prompts or len(prompts) != REQUESTS:
            raise ValueError('Both fixed panels must remain distinct and deduplicated')
        all_ids.update(ids)
        all_prompts.update(prompts)
        required = HISTORICAL_ARMS if index == 0 else ('K1', 'KRC1', 'K2', 'KRC2')
        validate_count_maps(panel['historical_counts'], ids, allowed_arms=HISTORICAL_ARMS,
                            required_arms=required)
    return dict(passed=True, panels=['development', 'additional'], requests=128,
                planned_generations=2 * GENERATIONS,
                included_gate_generations=QUALIFICATION_REQUESTS * len(ARMS) * REPEATS,
                historical_count_gate={'development': ['O4', 'K4'], 'additional': None},
                rotation_balance=rotation_balance(REQUESTS))
