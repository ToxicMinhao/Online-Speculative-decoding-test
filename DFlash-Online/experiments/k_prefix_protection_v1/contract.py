"""Fixed KP/K_HALF experiment boundaries; historical O/K contracts stay unchanged."""
from experiments.ok_coding.contract import validate_requests
from .prepare import evaluation_schedule, METHODS, SEEDS, UPSTREAM_BINDING

ARMS = ('K_HALF', 'KP')
CELLS = ((17, 4), (29, 4))
REPEATS = 3
ORIGINAL_COMMIT = '2705d992b133196be162229e1d5a08dd9c4f9cbb'
NEW_BUDGET = dict(optimizer_steps=2048, presentations=4096, reference_forwards=2048,
                  evaluation=3072, feedback=0, terminal_teacher_forwards=0)
QUALIFICATION_BUDGET = dict(forwards=8, backwards=8, reference_forwards=6,
                            optimizer_steps=4, generations=0)


def method_order(ordinal, repeat):
    base = METHODS if (ordinal // 4) % 2 == 0 else ('F', 'KP', 'K_HALF', 'K')
    shift = (ordinal + repeat) % 4
    return base[shift:] + base[:shift]


def validate_plan(plan):
    if plan['new_budget'] != NEW_BUDGET or plan['qualification_budget'] != QUALIFICATION_BUDGET:
        raise ValueError('Only the prespecified work is authorized')
    if plan['training_binding']['upstream_data_binding'] != UPSTREAM_BINDING:
        raise ValueError('The original coding feedback binding changed')
    if plan['original_execution_commit'] != ORIGINAL_COMMIT:
        raise ValueError('The original coding execution source changed')
    if (plan['training_seeds'] != list(SEEDS) or plan['stop_step'] != 512
            or plan['inherited_horizon'] != 7424 or plan['no_automatic_retry'] is not True):
        raise ValueError('Preserve seed order, stop and native horizon')
    requests = plan['evaluation_requests']
    if len(requests) != 128:
        raise ValueError('Exactly128 new evaluation requests required')
    validate_requests(requests, evaluation=True)
    if plan['evaluation_schedule'] != evaluation_schedule(requests):
        raise ValueError('Sealed evaluation order changed')
    if set(plan['configs']) != {'17', '29'} or any(set(v) != set(ARMS) for v in plan['configs'].values()):
        raise ValueError('Four independent new trajectories required')
    if any(set(plan['pools'][str(seed)]) != {'1','2','3','4'} for seed in SEEDS):
        raise ValueError('All eight retained pools required')
    if plan['serving'] != dict(block_size=16, max_new_tokens=2048, context_limit=4096,
            concurrency=1, temperature=0, capture=False, precision='bfloat16', attention_backend='fa3'):
        raise ValueError('Qualified serving settings changed')
    if len(plan['execution_source']['commit']) != 40 or plan['execution_source']['dirty']:
        raise ValueError('Require the clean published execution source')
    return {'status':'passed', 'budget':NEW_BUDGET, 'qualification':QUALIFICATION_BUDGET}
