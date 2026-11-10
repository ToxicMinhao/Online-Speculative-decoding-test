"""Fixed ABT pilot design and separately charged native work."""
from copy import deepcopy

SCHEMA = 'kplus_adversarial_block_training_v1'
ARMS = ('KPLUS', 'CLEAN2', 'RANDOM', 'ADV')
DATASETS = ('gsm8k', 'mathqa')
EVENTS = (5, 6)
START_EVENT, FINAL_EVENT = 4, 6
PANEL_SIZE, ENDPOINT_REPEATS = 256, 2
DIAGNOSTIC_WINDOWS = (1, 32, 65, 128)
RHO = 0.01
QUALIFICATION_CASES = ('KPLUS', 'DISABLED', 'ADV_ZERO', 'CLEAN2', 'RANDOM', 'ADV')
ANALYSIS = dict(resamples=10000, seed=20261110, generator='PCG64',
    percentile_method='linear', primary=[['ADV', 'KPLUS']],
    controls=[['ADV', 'CLEAN2'], ['ADV', 'RANDOM']],
    datasets_pooled=False, intervals_adjusted=False)
MAIN_BUDGET = dict(continuations=8, training_events=16,
    feedback_generations=512, teacher_preparations=512, optimizer_steps=2048,
    ordinary_presentations=4096, response_forwards=7168,
    response_backwards=7168, endpoint_generations=4096)
QUALIFICATION_BUDGET = dict(cases=12, optimizer_steps=12,
    ordinary_presentations=24, case_forwards=36, case_backwards=36,
    startup_forwards=24, startup_backwards=24,
    resource_forwards=8, resource_backwards=8,
    response_forwards=68, response_backwards=68,
    serving_generations=8, canonical_generations=0,
    feedback_generations=4, teacher_preparations=4,
    feedback_included_in_main_unique_total=True, automatic_retry=False)
STARTUP_BUDGET = dict(runtimes=16, response_forwards=32,
    response_backwards=32, optimizer_steps=0)
DIAGNOSTIC_BUDGET = dict(windows=64, response_forwards=128,
    response_backwards=0, optimizer_steps=0, serving_generations=0)


def augmentation_contract(arm):
    if arm not in ARMS:
        raise ValueError('Unknown ABT arm')
    if arm == 'KPLUS':
        return None
    return dict(name=SCHEMA, arm=arm, rho=0. if arm == 'CLEAN2' else RHO,
        views=2, view_weights=[0.5, 0.5], ascent_steps=1 if arm == 'ADV' else 0,
        clean_initialization=True, eligible='valid_artificial_mask_slots_only',
        radius='rho_times_clean_eligible_block_embedding_frobenius_norm',
        zero_gradient_block='zero_perturbation', selection='detached_clean_current_support',
        teacher='unchanged_frozen_canonical_distributions',
        numerator='native_aggregate_KPLUS_graph', denominator='native_clean_selected_weight_plus_epsilon',
        perturbation_dtype='float32_auxiliary', model_input_dtype='bfloat16',
        model_gradient_optimizer_dtype='native_bfloat16',
        attack_detached=True, stochastic_realization='same_across_response_views',
        perturbation_rng='separate_dataset_event_pass_presentation_block_key',
        optimizer='one_native_clip_and_AdamW_per_two_response_window',
        serving_augmentation=False, candidate_rejection=False, backtracking=False)


def experiment_contract(binding, arm, dataset, *, origin_binding):
    if arm not in ARMS or dataset not in DATASETS:
        raise ValueError('Unknown ABT dataset or arm')
    for value in (binding, origin_binding):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Require sealed experiment and PR32 origin bindings')
    return dict(schema=SCHEMA, binding=binding, arm=arm, dataset=dataset,
        origin_binding=origin_binding, origin_method='KPLUS', origin_event=4,
        start_adam_step=512, final_adam_step=768, events=list(EVENTS),
        initialization='exact_complete_dataset_specific_PR32_KPLUS4',
        feedback='byte_identical_shared_evolving_KPLUS_packets',
        native_sampling=True, seed_reset=False, inherited_horizon=7424,
        augmentation=augmentation_contract(arm))


def expected_work(arm, completed_event):
    if arm not in ARMS or type(completed_event) is not int or not 4 <= completed_event <= 6:
        raise ValueError('ABT boundary must be event 4, 5 or 6')
    events = completed_event - 4
    responses = events * 256
    views = responses * (1 if arm == 'KPLUS' else 2)
    return dict(optimizer_steps=events*128, ordinary_presentations=responses,
                response_forwards=views, response_backwards=views)


def budgets():
    return deepcopy(dict(main=MAIN_BUDGET, qualification=QUALIFICATION_BUDGET,
                         startup=STARTUP_BUDGET, diagnostic=DIAGNOSTIC_BUDGET))
