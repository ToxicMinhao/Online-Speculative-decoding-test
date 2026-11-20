"""The handoff's ten-order, five-arm timing schedule and its exact balance."""
from collections import Counter
from .contract import METHODS, EVALUATION_REQUESTS


def orders():
    result = []
    for shift in range(len(METHODS)):
        rotation = METHODS[shift:]+METHODS[:shift]
        result.extend((rotation, tuple(reversed(rotation))))
    return result


def evaluation_schedule(requests):
    if len(requests) != EVALUATION_REQUESTS:
        raise ValueError('Final timing requires all 1,024 sealed requests')
    result = []
    for repeat in range(3):
        for group in range(8):
            selected = requests[group*128:(group+1)*128]
            for position, method in enumerate(orders()[(8*repeat+group) % 10]):
                result.append(dict(activation=len(result), repeat=repeat,
                    group=group, position=position, method=method, warmups=2,
                    request_ids=[row['request_id'] for row in selected]))
    return result


def schedule_summary(schedule):
    positions = {method: [0]*5 for method in METHODS}
    precedence = {first: {second: 0 for second in METHODS if second != first}
                  for first in METHODS}
    coverage = Counter()
    for start in range(0, len(schedule), 5):
        batch = schedule[start:start+5]
        if len(batch) != 5 or {row['method'] for row in batch} != set(METHODS):
            raise ValueError('Each scheduling group must contain five different arms')
        for position, row in enumerate(batch):
            positions[row['method']][position] += 1
            for request_id in row['request_ids']:
                coverage[request_id, row['repeat'], row['method']] += 1
            for later in batch[position+1:]:
                precedence[row['method']][later['method']] += 1
    if (len(schedule) != 120 or len(coverage) != 15360
            or any(count != 1 for count in coverage.values())
            or any(value != 12 for row in precedence.values() for value in row.values())
            or any(not 4 <= value <= 6 for row in positions.values() for value in row)):
        raise ValueError('Five-arm coverage or aggregate balance differs from the handoff')
    return dict(activations=120, scored_generations=15360, warmup_generations=240,
                groups=24, precedence=precedence, positions=positions,
                per_request_position_balance=False)
