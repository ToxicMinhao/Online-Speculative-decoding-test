"""Prespecified serving-only contract; no training configuration or trainer."""
from collections import Counter

from experiments.full_k_gsm8k.contract import SERVING

METHODS = ('O', 'K', 'KPLUS')
PANEL_SIZE = 1024
BLOCK = 64
REPEATS = 3
SCHEMA = 'mathqa_transfer_v1'
BUDGET = dict(timed_generations=9216, historical_control_generations=3,
              total_generations=9219, optimizer_steps=0, feedback_collections=0,
              teacher_preparations=0, extra_warmup_generations=0)
ANALYSIS = dict(primary=['KPLUS', 'O'], secondary=['KPLUS', 'K'],
                outcome='request_latency_s', resamples=10000, seed=20261103,
                generator='PCG64', percentile_method='linear',
                pass_rule='lower_95_percent_endpoint_gt_zero',
                secondary_confirmatory_only_if_primary_passes=True)


def method_order(block, repeat):
    order = ('O', 'K', 'KPLUS') if block % 2 == 0 else ('O', 'KPLUS', 'K')
    shift = (block // 2 + repeat) % 3
    return order[shift:] + order[:shift]


def final_schedule(requests):
    slots = []
    for start in range(0, len(requests), BLOCK):
        block = start // BLOCK
        for repeat in range(REPEATS):
            for position, method in enumerate(method_order(block, repeat)):
                for request in requests[start:start + BLOCK]:
                    slots.append(dict(seed=17, event=58, phase='final', block_ordinal=block,
                        panel_ordinal=request['panel_ordinal'], request_id=request['request_id'],
                        repeat=repeat, method=method, position=position))
    return slots


def schedule_balance():
    orders = Counter(method_order(block, repeat) for block in range(16) for repeat in range(3))
    return {'orders': [{'order': list(order), 'count': count} for order, count in sorted(orders.items())],
            'each_order_count': 8, 'block_repeat_orders': sum(orders.values()),
            'generations': BUDGET['timed_generations']}


def validate_inputs(requests, checkpoints, *, production=True):
    if set(checkpoints) != set(METHODS) or len(set(map(str, checkpoints.values()))) != 3:
        raise ValueError('Require the three distinct authenticated O58/K58/K+58 exports')
    if (not requests or (production and len(requests) != PANEL_SIZE)
            or [r['panel_ordinal'] for r in requests] != list(range(len(requests)))
            or len({r['request_id'] for r in requests}) != len(requests)):
        raise ValueError('Require the complete ordered unique MathQA panel')
    for request in requests:
        if 'response_ids' in request or 'finish_reason' in request:
            raise ValueError('New outputs must not enter the sealed request panel')
        if (not request['prompt_ids'] or len(request['prompt_ids']) > 2048
                or request['max_new_tokens'] != 2048 or not request['stop_token_ids']
                or any(type(t) is not int or t < 0 for t in request['prompt_ids'] + request['stop_token_ids'])):
            raise ValueError('Full prompt, output cap or tokenizer identity is invalid')
    return final_schedule(requests)


def validate_plan(plan):
    fixed = dict(schema=SCHEMA, methods=list(METHODS), serving=SERVING,
                 budget=BUDGET, analysis=ANALYSIS, automatic_retry=False)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('Sealed serving-only experiment contract changed')
    planned = validate_inputs(plan['evaluation_requests'], plan['checkpoints'])
    if plan['evaluation_schedule'] != planned:
        raise ValueError('Sealed balanced order changed')
    if plan['execution_source']['dirty'] or len(plan['execution_source']['commit']) != 40:
        raise ValueError('Seal the clean published execution source')
    if (plan['control_request']['panel_ordinal'] != 0 or
            set(plan['control_counts']) != set(METHODS)):
        raise ValueError('Require the three historical ordinal-zero controls')
    return dict(status='passed', budget=BUDGET, balance=schedule_balance())
