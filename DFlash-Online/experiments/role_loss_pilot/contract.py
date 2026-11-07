"""Finite work, scientific identity, and input boundaries for the role-loss pilot."""
import json
from pathlib import Path

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from online_feedback.kplus import kplus_contract

SCHEMA = 'role_loss_pilot_v1'
ARMS = ('KPLUS', 'MIX', 'ROLE')
DATASETS = ('gsm8k', 'mathqa')
EVENTS = 4
PANEL_SIZE = 256
REQUEST_BLOCK = 32
SEED = 17
NEW_BUDGET = dict(training_events=24, optimizer_steps=3072, response_presentations=6144,
    response_forwards=6144, response_backwards=6144, online_generations=3072,
    terminal_preparation_forwards=3072, intermediate_generations=1536,
    final_generations=3072, canonical_generations=512, qualification_generations=4,
    correctness_generations=8, disposable_optimizer_steps=6,
    qualification_response_forwards=12, qualification_response_backwards=12,
    diagnostic_draft_forwards=224, diagnostic_target_preparations=32)
ANALYSIS = dict(primary=['ROLE', 'KPLUS'], comparisons=[['ROLE','KPLUS'],['MIX','KPLUS'],['ROLE','MIX']],
    resamples=10000, seed=20261107, generator='PCG64', percentile_method='linear',
    endpoint_repeats=2, intermediate_repeats=1, repeated_requests_are_independent=False)


def loss_contract(arm):
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    if arm == 'KPLUS':
        return kplus_contract()
    from online_feedback.role_loss import role_loss_contract
    return role_loss_contract(arm)


def arm_limits(arm):
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    return dict(online_generations=512, correctness_generations=4 if arm == 'KPLUS' else 0,
        terminal_preparation_forwards=512, response_forwards=1024,
        response_backwards=1024, optimizer_steps=512)


def expected_work(events, arm):
    if type(events) is not int or not 0 <= events <= EVENTS:
        raise ValueError('Pilot event boundary is 0..4')
    return dict(online_generations=events*128,
        correctness_generations=4 if arm == 'KPLUS' and events else 0,
        terminal_preparation_forwards=events*128, response_forwards=events*256,
        response_backwards=events*256, optimizer_steps=events*128)


def trajectory_root(root, dataset, arm):
    if dataset not in DATASETS or arm not in ARMS:
        raise ValueError('Trajectory is outside the fixed six cells')
    return Path(root)/'campaign_001/trajectories'/dataset/arm


def read_requests(path, kind):
    value = json.loads(Path(path).read_text())
    if isinstance(value, list):
        return value
    keys = ('train_requests','requests') if kind == 'train' else ('evaluation_requests','requests','panel_requests')
    for key in keys:
        if key in value:
            return value[key]
    raise ValueError('Manifest does not contain '+kind+' requests')


def validate_plan(plan):
    fixed = dict(schema=SCHEMA, arms=list(ARMS), seed=SEED, events=EVENTS,
                 budget=NEW_BUDGET, serving=SERVING, automatic_retry=False)
    if any(plan.get(key) != value for key,value in fixed.items()):
        raise ValueError('The fixed role-loss pilot contract changed')
    source = plan['execution_source']
    if source.get('dirty') is not False or len(source['commit']) != 40:
        raise ValueError('Pilot execution requires a published clean source identity')
    binding = plan['binding']
    if len(binding) != 64 or any(c not in '0123456789abcdef' for c in binding):
        raise ValueError('Malformed sealed pilot binding')
    if plan['loss_contracts'] != {arm:loss_contract(arm) for arm in ARMS}:
        raise ValueError('Sealed candidate or unchanged K+ loss contract differs')
    if set(plan['datasets']) != set(DATASETS):
        raise ValueError('Require independent GSM8K and MathQA trajectories')
    for dataset, item in plan['datasets'].items():
        if set(item['configs']) != set(ARMS):
            raise ValueError('Require one fresh config per dataset/arm')
        train = read_requests(item['training_manifest'], 'train')
        panel = read_requests(item['panel_manifest'], 'test')
        for rows, size in ((train,512),(panel,256)):
            if len(rows) != size or len({r['request_id'] for r in rows}) != size:
                raise ValueError('Input count or unique source request identity differs')
        train_prompts = {stable_digest(r['prompt_ids']) for r in train}
        if train_prompts.intersection(stable_digest(r['prompt_ids']) for r in panel):
            raise ValueError('Adaptation inputs overlap the development panel')
    q = plan['qualification']
    if q.get('maximum_optimizer_steps',6) != 6 or q.get('timeout_s',1800) != 1800:
        raise ValueError('Qualification bounds changed')
    return dict(status='passed', trajectories=6, events=24, budget=NEW_BUDGET)
