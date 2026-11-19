"""Fixed scientific identities and budgets for released-start C58."""
import json
from pathlib import Path

from experiments.full_k_gsm8k.contract import SERVING, stable_digest
from experiments.osd_persistent.data import validate_manifest

SCHEMA = 'full_c_gsm8k_v1'
ARM = 'KPLUS_C'
EVENTS = 58
METHODS = ('O58', 'KPLUS58', 'KPLUS_C58')
PRODUCTION = dict(adam=7424, response_forward=14848, response_backward=14848,
    training_initialization=1, feedback_generation=7473, teacher_preparation=7424,
    correctness_generation=4, canonical_generation=0, qualification_generation=0,
    warmup_generation=576, scored_generation=18432)
QUALIFICATION = dict(adam=6, response_forward=14, response_backward=14,
    training_initialization=1, feedback_generation=0, teacher_preparation=0,
    correctness_generation=0, canonical_generation=0, qualification_generation=24,
    warmup_generation=0, scored_generation=0)
SERVING_QUALIFICATION = [dict(dataset=dataset,repeat=repeat,method=method,panel_ordinal=ordinal)
    for dataset in ('gsm8k','mathqa') for repeat in range(2)
    for method in METHODS for ordinal in (0,1)]


def plan_binding(plan):
    return stable_digest({k:v for k,v in plan.items() if k != 'binding'})


def validate_plan(plan,*,package_root=None):
    from .evaluation import evaluation_schedule,validate_inputs,validate_historical_counts
    fixed = dict(schema=SCHEMA, seed=17, events=58, arms=[ARM], methods=list(METHODS),
        serving=SERVING, learning_rates={ARM:3e-4}, intermediate_events=[],
        automatic_retry=False, production_work=PRODUCTION, qualification_work=QUALIFICATION,
        serving_qualification_schedule=SERVING_QUALIFICATION,
        forecast=dict(factor=1.3,fixed_margin_s=1800))
    if any(plan.get(k) != v for k,v in fixed.items()) or plan_binding(plan) != plan['binding']:
        raise ValueError('Fixed full C58 scientific plan or binding changed')
    if (plan['execution_source']['dirty'] or plan['execution_source']['commit'] != plan['execution_commit']
            or len(plan['execution_commit']) != 40 or set(plan['references']) != {'O58','KPLUS58'}):
        raise ValueError('Require a clean execution identity and both retained controls')
    def local(path):
        return Path(path) if package_root is None else Path(package_root)/Path(path).relative_to(plan['root'])
    manifest = json.loads(local(plan['manifest']).read_text())
    result = validate_manifest(manifest)
    if (stable_digest(manifest) != plan['manifest_binding'] or len(manifest['train_requests']) != 7473
            or manifest['policy']['cap_policy'] != 'fixed2048'
            or manifest['policy']['train_context_limit'] != 2272):
        raise ValueError('Original sealed full GSM8K stream changed')
    for dataset, sub in plan['datasets'].items():
        requests = json.loads(local(sub['requests']).read_text())['requests']
        if dataset not in ('gsm8k','mathqa') or len(requests) != 1024:
            raise ValueError('Both original 1024-request panels are required')
        if json.loads(local(sub['schedule']).read_text()) != evaluation_schedule(requests):
            raise ValueError('Prespecified six-permutation activation schedule changed')
        load=lambda key:json.loads(local(sub[key]).read_text())
        validate_inputs(requests,load('canonicals'),load('warmups')['requests'],
            load('warmup_canonicals'),sub['reference_checkpoints'])
        validate_historical_counts(requests,load('historical_counts'))
    if set(plan['datasets']) != {'gsm8k','mathqa'} or len(plan['qualification_feature_paths']) != 4:
        raise ValueError('Missing dataset or bounded qualification feature fixture')
    return dict(status='passed', original_manifest=result, endpoint=58)
