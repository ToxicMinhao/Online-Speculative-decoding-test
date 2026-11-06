"""Fixed paper-defined AdaFlash trajectory and fresh three-arm endpoint."""
from collections import Counter
import json
from pathlib import Path

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.osd_persistent.data import validate_manifest

SCHEMA = 'adaflash_c1_v1'
ARM = 'A'
ARMS = (ARM,)
METHODS = ('O', 'KPLUS', 'A')
EVENTS = 58
PANEL_SIZE = 1024
REQUEST_BLOCK = 64
REPEATS = 3
LEARNING_RATES = {ARM: 3e-4}
NEW_BUDGET = dict(training_events=58, optimizer_steps=7424,
    response_presentations=14848, response_forwards=14848, response_backwards=14848,
    online_generations=7473, terminal_preparation_forwards=7424,
    intermediate_generations=0, final_generations=9216, correctness_generations=4,
    total_generations=16693, disposable_optimizer_steps=0, baseline_optimizer_steps=0)
ANALYSIS = dict(primary=['KPLUS', 'A'], comparisons=[['KPLUS', 'A'], ['KPLUS', 'O'], ['A', 'O']],
    outcome='request_latency_s', resamples=10000, seed=20261102, generator='PCG64',
    percentile_method='linear', decision='positive_CI_favors_KPLUS_negative_CI_favors_A_otherwise_inconclusive')


def method_order(block, repeat):
    """Each of the six orders occurs eight times over sixteen blocks."""
    order = METHODS if block % 2 == 0 else ('O', 'A', 'KPLUS')
    shift = (block // 2 + repeat) % 3
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
    return dict(orders=[dict(order=list(order), count=count) for order, count in sorted(orders.items())],
        each_order_count=8, block_repeat_orders=sum(orders.values()),
        generations=len(slots) if slots is not None else NEW_BUDGET['final_generations'])


def validate_inputs(requests, checkpoints, canonicals, *, production=True):
    ids = [request['request_id'] for request in requests]
    if (not ids or (production and len(ids) != PANEL_SIZE)
            or [r['panel_ordinal'] for r in requests] != list(range(len(ids)))
            or len(set(ids)) != len(ids) or set(canonicals) != set(ids)):
        raise ValueError('Preserve the complete original GSM8K panel and canonical identities')
    if set(checkpoints) != set(METHODS) or len(set(map(str, checkpoints.values()))) != 3:
        raise ValueError('Require distinct O58, KPLUS58 and new A58 exports')
    for request in requests:
        if (request['max_new_tokens'] != 2048 or not request['prompt_ids']
                or len(request['prompt_ids']) + 2048 > 4096 or not request['stop_token_ids']
                or any(type(t) is not int or t < 0 for t in request['prompt_ids'] + request['stop_token_ids'])):
            raise ValueError('The original full prompt, output cap or tokenizer identity changed')
        canonical = canonicals[request['request_id']]
        if not canonical['response_ids'] or canonical['finish_reason'] not in ('eos', 'length'):
            raise ValueError('Missing immutable canonical output and termination')
    return final_schedule(requests)


def validate_plan(plan):
    from online_feedback.adaflash_opd import adaflash_opd_contract
    fixed = dict(schema=SCHEMA, arms=list(ARMS), methods=list(METHODS), seed=17,
        events=EVENTS, budget=NEW_BUDGET, serving=SERVING, learning_rates=LEARNING_RATES,
        analysis=ANALYSIS, loss=adaflash_opd_contract(), intermediate_events=[], automatic_retry=False)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('The fixed paper-defined AdaFlash C1 contract changed')
    if plan['execution_source']['dirty'] or len(plan['execution_source']['commit']) != 40:
        raise ValueError('Publish and seal a clean execution commit')
    if set(plan['configs']) != set(ARMS) or set(plan['references']) != {'O', 'KPLUS'}:
        raise ValueError('Only A trains; O58 and KPLUS58 are retained serving controls')
    manifest = json.loads(Path(plan['manifest']).read_text())
    checked = validate_manifest(manifest)
    if (len(manifest['train_requests']) != 7473 or manifest['policy']['cap_policy'] != 'fixed2048'
            or manifest['policy']['train_context_limit'] != 2272
            or manifest['policy']['serving_context_limit'] != 4096
            or stable_digest(manifest) != plan['manifest_binding']):
        raise ValueError('Preserve the original full GSM8K stream and sequence preparation')
    final_export = str(Path(plan['root']) / 'campaign_001/arms/A/trainer/events/event_058/export')
    checkpoints = dict(plan['references'], A=final_export)
    canonicals = json.loads(Path(plan['canonicals']).read_text())
    planned = validate_inputs(plan['evaluation_requests'], checkpoints, canonicals)
    if plan['evaluation_schedule'] != planned:
        raise ValueError('The predeclared balanced endpoint order changed')
    return dict(status='passed', manifest=checked, budget=NEW_BUDGET, balance=schedule_balance(planned))
