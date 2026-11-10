"""Fixed one-event prefix-survival screen; no implicit extra model work."""
from copy import deepcopy

SCHEMA = 'kplus_prefix_survival_v1'
ARMS = METHODS = ('KPLUS', 'LOCAL', 'SURVIVAL')
DATASET = 'gsm8k'
DATASETS = (DATASET,)
EVENT = 5
EVENTS = (EVENT,)
START_EVENT, FINAL_EVENT = 4, 5
START_STEP, END_STEP = 512, 640
PANEL_SIZE, SOURCE_PANEL_SIZE, ENDPOINT_REPEATS = 128, 256, 2
BLOCK_SIZE, QUALIFICATION_REQUESTS = 8, 8
SEED, HEAD_SEED, HEAD_COEFFICIENT = 17, 17, 1.0
ANALYSIS = dict(resamples=10000, seed=20261110, generator='PCG64',
    percentile_method='linear', primary=[['SURVIVAL', 'KPLUS'], ['SURVIVAL', 'LOCAL']],
    controls=[['LOCAL', 'KPLUS']], intervals_adjusted=False,
    uncertainty_unit='paired_request', repeats_reduction='within_request_mean',
    effect='percent_reduction_of_paired_totals', first_round='shared_start_only')
MAIN_BUDGET = dict(training_events=3, optimizer_steps=384, head_optimizer_steps=256,
    ordinary_presentations=768, response_forwards=768, response_backwards=768,
    endpoint_generations=768, feedback_generations=0, canonical_generations=0,
    unique_feedback_responses=128, unique_teacher_packets=128)
QUALIFICATION_CASES = ('KPLUS', 'DISABLED', 'LOCAL', 'SURVIVAL')
QUALIFICATION_BUDGET = dict(cases=4, optimizer_steps=4, head_optimizer_steps=2,
    ordinary_presentations=8, case_forwards=8, case_backwards=8,
    auxiliary_backwards=4, startup_forwards=8, startup_backwards=8,
    response_forwards=16, response_backwards=20, serving_generations=6,
    feedback_generations=0, canonical_generations=0, automatic_retry=False)
STARTUP_BUDGET = dict(runtimes=3, response_forwards=6, response_backwards=6,
    optimizer_steps=0, head_optimizer_steps=0)
SERVING = dict(block_size=16, max_new_tokens=2048, context_limit=4096,
    concurrency=1, greedy=True, capture=False, extra_warmup_generations=0,
    cache_policy='idle_full_flush_before_every_request_and_publication',
    same_physical_card=True)


def experiment_contract(binding, arm, dataset=DATASET, *, origin_binding):
    if arm not in ARMS or dataset != DATASET:
        raise ValueError('Require a prescribed GSM8K survival pilot arm')
    for value in (binding, origin_binding):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Require sealed experiment and PR32 origin bindings')
    return dict(schema=SCHEMA, binding=binding, arm=arm, dataset=dataset,
        origin_binding=origin_binding, origin_method='KPLUS', origin_event=START_EVENT,
        start_adam_step=START_STEP, final_adam_step=END_STEP, events=list(EVENTS),
        initialization='exact_complete_PR32_GSM8K_KPLUS4', seed_reset=False,
        native_sampling=True, inherited_horizon=7424,
        feedback='same_128_saved_KPLUS4_responses_and_teacher_packets',
        head=None if arm == 'KPLUS' else dict(label=arm, coefficient=HEAD_COEFFICIENT,
            seed=HEAD_SEED, initialization='independent_generator_nonzero_weight_zero_bias',
            optimizer='separate_native_BF16_AdamW', export=False),
        serving=deepcopy(SERVING))


def expected_work(arm, completed_event=EVENT):
    if arm not in ARMS or type(completed_event) is not int or completed_event not in (4, 5):
        raise ValueError('Require an original or event-five boundary')
    steps = 128 * (completed_event - 4)
    return dict(optimizer_steps=steps, ordinary_presentations=steps*2,
        response_forwards=steps*2, response_backwards=steps*2,
        head_optimizer_steps=steps if arm != 'KPLUS' else 0)


def budgets():
    """Qualification/startup receipts are separate, never included as main work."""
    return deepcopy(dict(main=MAIN_BUDGET, qualification=QUALIFICATION_BUDGET,
                         startup=STARTUP_BUDGET, serving=SERVING))
