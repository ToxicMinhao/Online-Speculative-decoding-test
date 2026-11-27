"""Fixed scientific counts for the fresh BASE/TAIL confirmation trajectories."""
from experiments.opencode_native_comparison.contract import digest, validate_requests

SCHEMA = 'continuation_repair_fullrun_v1'
ARMS = ('BASE', 'TAIL')
METHODS = ('BASE', 'TAIL', 'AREPO', 'F')
EVENTS, WINDOW, PASSES = 58, 128, 2
TRAIN_REQUESTS, EVALUATION_REQUESTS = 7424, 1024
SEED, SELECTION_SEED = 17, 20261126
INPUT_IDENTITY = dict(bytes=32309622,
    sha256='df7194e667126f34f8e2ab2f6e22e2e5480cd7921728ceb4f946bd3a0cce6581')
SOURCE_REVISION = '9819cc0464bf7312883763f93751749007b5bbcf'
SERVING = dict(precision='bfloat16', attention='fa3', greedy=True,
    draft_layers=5, block_size=16, max_new_tokens=2048, context_limit=4096,
    physical_context_length=4112, concurrency=1, feedback_capture=False, cuda_graphs=True)
PRODUCTION_WORK = dict(feedback_generation=14848, teacher_preparation=14848,
    correctness_generation=8, canonical_generation=1024, canonical_warmup_generation=0,
    scored_generation=12288, warmup_generation=96, adam=14848,
    response_forward=29696, response_backward=29696, training_initialization=2,
    witness_response_forward=72)
QUALIFICATION_CEILINGS = dict(adam=8, generation=16)
# Unused because this sealed plan authenticates both original frozen-F warmups.
OPTIONAL_WORK_CEILINGS = dict(canonical_warmup_generation=2)
RESOURCE_LIMITS = dict(elapsed_s=72*3600, occupied_card_s=120*3600,
    new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000,
    forecast_multiplier=1.25, forecast_fixed_s=3600)
FORMULA_FILES = ('online_feedback/objective.py', 'online_feedback/kplus.py',
    'online_feedback/prefix_allocation.py', 'online_feedback/repair_conditional_kl.py',
    'online_feedback/continuation_repair.py')
WITNESS_EVENTS = (1, 16, 32, 58)


def experiment_contract(arm, binding):
    """Persist the actual method and fresh trajectory, never a foreign-arm view."""
    from online_feedback.kplus import kplus_contract
    from online_feedback.prefix_allocation import prefix_allocation_contract
    from online_feedback.repair_conditional_kl import repair_conditional_kl_contract
    from online_feedback.continuation_repair import continuation_repair_contract
    if arm not in ARMS or len(binding) != 64 or any(c not in '0123456789abcdef' for c in binding):
        raise ValueError('Require explicit BASE/TAIL arm and sealed scientific binding')
    return dict(schema=SCHEMA, arm=arm, binding=binding, dataset='OpenCodeInstruct',
        initialization='released_weights_fresh_Adam_seed17', initial_counters=[0,0,0],
        final_counters=[7424,14848,58], seed=17, context_limit=4096,
        feedback='own_current_draft_native_responses', maximum_events=58, horizon=7424,
        responses_per_event=128, passes_per_event=2, accumulation_steps=2,
        learning_rate=3e-4, objective='osd_first_error',
        redistribution=kplus_contract(), allocation=prefix_allocation_contract(.25),
        repair_pair=repair_conditional_kl_contract(),
        continuation_repair=continuation_repair_contract(enabled=arm == 'TAIL'))


def event_requests(original_inputs, event):
    if type(event) is not int or not 1 <= event <= EVENTS:
        raise ValueError('Event must belong to the fixed 58-event training stream')
    rows = original_inputs['train_requests'][(event-1)*WINDOW:event*WINDOW]
    if len(rows) != WINDOW:
        raise ValueError('Original sealed feedback window is incomplete')
    return rows


def validate_inputs(original, evaluation):
    """Validate the new panel without reselection or replacement of the stream."""
    train_ids, train_tokens = validate_requests(original['train_requests'], count=TRAIN_REQUESTS, role='train')
    eval_ids, eval_tokens = validate_requests(evaluation['evaluation_requests'], count=EVALUATION_REQUESTS, role='evaluation')
    if train_ids & eval_ids or train_tokens & eval_tokens:
        raise ValueError('Fresh evaluation overlaps the preserved training stream')
    for role, count in [('warmup',2), ('qualification',16)]:
        ids, tokens = validate_requests(original[role+'_requests'], count=count, role=role)
        if ids & eval_ids or tokens & eval_tokens:
            raise ValueError('Fresh evaluation overlaps reserved execution checks')
    if evaluation['warmup_requests'] != original['warmup_requests']:
        raise ValueError('Original separate warmups changed')
    old_ids = {row['request_id'] for row in original['evaluation_requests']}
    old_tokens = {tuple(row['prompt_ids']) for row in original['evaluation_requests']}
    if eval_ids & old_ids or eval_tokens & old_tokens:
        raise ValueError('Fresh confirmation reused original development requests')
    return dict(status='passed', training_requests=TRAIN_REQUESTS,
        evaluation_requests=EVALUATION_REQUESTS, training_reselected=False)
