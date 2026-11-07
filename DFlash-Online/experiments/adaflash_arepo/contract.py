"""One new own-feedback arm and the prespecified fresh four-arm endpoint."""
from collections import Counter
from itertools import permutations
import json
from pathlib import Path

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.osd_persistent.data import validate_manifest

SCHEMA = 'adaflash_arepo_v1'
ARM = 'AREPO'
ARMS = (ARM,)
METHODS = ('O', 'KPLUS', 'A', ARM)
EVENTS = 58
PANEL_SIZE = 1024
REQUEST_BLOCK = 64
REPEATS = 3
LEARNING_RATES = {ARM: 3e-4}
NEW_BUDGET = dict(training_events=58, optimizer_steps=7424,
    response_presentations=14848, response_forwards=14848, response_backwards=14848,
    online_generations=7473, terminal_preparation_forwards=7424,
    intermediate_generations=0, final_generations=12288, correctness_generations=4,
    total_generations=19765, disposable_optimizer_steps=6, baseline_optimizer_steps=0)
ANALYSIS = dict(primary=['KPLUS', ARM],
    comparisons=[['KPLUS', ARM], ['KPLUS', 'O'], [ARM, 'O'], [ARM, 'A']],
    outcome='request_latency_s', resamples=10000, seed=20261102, generator='PCG64',
    percentile_method='linear', decision='positive_CI_favors_KPLUS_negative_CI_favors_AREPO_otherwise_inconclusive')


def method_order(block, repeat):
    """All 24 orders twice; every position is balanced within each repeat."""
    orbits = ((0,1,2,3), (0,1,4,5), (2,3,4,5))
    tails = tuple(permutations(METHODS[1:]))
    order = ('O', *tails[orbits[repeat][block // 4]])
    shift = block % 4
    return order[shift:] + order[:shift]


def final_schedule(requests):
    slots = []
    for start in range(0, len(requests), REQUEST_BLOCK):
        block = start // REQUEST_BLOCK
        for repeat in range(REPEATS):
            for position, method in enumerate(method_order(block, repeat)):
                for request in requests[start:start + REQUEST_BLOCK]:
                    slots.append(dict(seed=17, event=58, phase='final', block_ordinal=block,
                        panel_ordinal=request['panel_ordinal'], request_id=request['request_id'],
                        repeat=repeat, method=method, position=position))
    return slots


def schedule_balance(slots=None):
    orders = Counter(method_order(block, repeat) for block in range(16) for repeat in range(REPEATS))
    return dict(orders=[dict(order=list(order), count=count) for order,count in sorted(orders.items())],
        each_order_count=2, block_repeat_orders=48,
        each_method_per_position_per_repeat=4,
        generations=len(slots) if slots is not None else NEW_BUDGET['final_generations'])


def validate_inputs(requests, checkpoints, canonicals, *, production=True):
    # Use the unchanged PR29 request/canonical validator through a local view.
    from experiments.adaflash_c1.contract import validate_inputs as original
    if production and len(requests) != PANEL_SIZE:
        raise ValueError('Preserve the full 1,024-request endpoint panel')
    if set(checkpoints) != set(METHODS) or len(set(map(str, checkpoints.values()))) != 4:
        raise ValueError('Require distinct retained O/KPLUS/paper-A and new AREPO exports')
    original(requests, {name: checkpoints[name] for name in ('O','KPLUS','A')},
             canonicals, production=False)
    return final_schedule(requests)


def validate_plan(plan):
    from online_feedback.adaflash_repo import adaflash_repo_contract
    fixed = dict(schema=SCHEMA, arms=list(ARMS), methods=list(METHODS), seed=17,
        events=EVENTS, budget=NEW_BUDGET, serving=SERVING, learning_rates=LEARNING_RATES,
        analysis=ANALYSIS, loss=adaflash_repo_contract(), intermediate_events=[], automatic_retry=False)
    if any(plan.get(key) != value for key,value in fixed.items()):
        raise ValueError('The released-loss supplement contract changed')
    if plan['execution_source']['dirty'] or len(plan['execution_source']['commit']) != 40:
        raise ValueError('Publish and seal clean execution source')
    if set(plan['configs']) != set(ARMS) or set(plan['references']) != {'O','KPLUS','A'}:
        raise ValueError('Only AREPO trains; all three prior exports remain frozen')
    manifest = json.loads(Path(plan['manifest']).read_text())
    checked = validate_manifest(manifest)
    if (len(manifest['train_requests']) != 7473 or manifest['policy']['cap_policy'] != 'fixed2048'
            or manifest['policy']['train_context_limit'] != 2272
            or manifest['policy']['serving_context_limit'] != 4096
            or stable_digest(manifest) != plan['manifest_binding']):
        raise ValueError('Preserve the complete PR29 feedback stream and preparation')
    final_export = str(Path(plan['root']) / 'campaign_001/arms/AREPO/trainer/events/event_058/export')
    checkpoints = dict(plan['references'], AREPO=final_export)
    canonicals = json.loads(Path(plan['canonicals']).read_text())
    schedule = validate_inputs(plan['evaluation_requests'], checkpoints, canonicals)
    if plan['evaluation_schedule'] != schedule:
        raise ValueError('Four-arm balanced endpoint order changed')
    if (plan['qualification']['maximum_optimizer_steps'] != 6
            or plan['qualification']['timeout_s'] != 1800):
        raise ValueError('Qualification is bounded to six disposable Adam steps')
    return dict(status='passed', manifest=checked, budget=NEW_BUDGET, balance=schedule_balance(schedule))
