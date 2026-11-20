"""Fixed scientific counts and shared configuration for the CODE comparison."""
import hashlib
import json

SCHEMA = 'opencode_native_comparison_v1'
ARMS = ('O_CODE58', 'AREPO_CODE58', 'C_CODE58')
METHODS = (*ARMS, 'F', 'AR')
EVENTS, WINDOW, PASSES = 58, 128, 2
TRAIN_REQUESTS, EVALUATION_REQUESTS = 7424, 1024
SEED, SELECTION_SEED = 17, 20261120
SERVING = dict(precision='bfloat16', attention='fa3', greedy=True,
    draft_layers=5, block_size=16, max_new_tokens=2048, context_limit=4096,
    physical_context_length=4112, concurrency=1, feedback_capture=False,
    cuda_graphs=True)
PRODUCTION_WORK = dict(feedback_generation=22272, correctness_generation=12,
    canonical_generation=1024, frozen_precheck_generation=1024,
    warmup_generation=240, scored_generation=15360, adam=22272,
    response_forward=44544, response_backward=44544,
    teacher_preparation=22272, training_initialization=3)
QUALIFICATION_CEILINGS = dict(generation=64, adam=48, response_forward=128,
    response_backward=128, teacher_preparation=32, training_initialization=8)
RESOURCE_LIMITS = dict(elapsed_s=48*3600, occupied_card_s=90*3600,
    new_allocated_bytes=90_000_000_000, free_reserve_bytes=400_000_000_000,
    forecast_multiplier=1.25, forecast_fixed_s=3600)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_requests(requests, *, count, role):
    """Validate an input-only ordered set, without relaxing older GSM gates."""
    if len(requests) != count:
        raise ValueError(f'{role} requires exactly {count} requests')
    identifiers, prompts = set(), set()
    for ordinal, request in enumerate(requests):
        ids = request['prompt_ids']
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 2048
                or any(type(token) is not int or token < 0 for token in ids)
                or request['max_new_tokens'] != 2048
                or request['stop_token_ids'] != [151645, 151643]):
            raise ValueError('Require complete bounded prompts and pinned stopping tokens')
        if request['request_index'] != ordinal:
            raise ValueError('Request order differs from its sealed ordinal')
        if role == 'evaluation' and request['panel_ordinal'] != ordinal:
            raise ValueError('Evaluation panel order changed')
        if any(key in request for key in ('output', 'answer', 'solution',
                                         'response_ids', 'finish_reason')):
            raise ValueError('Input manifests cannot contain reference or generated answers')
        identity, prompt = request['request_id'], tuple(ids)
        if identity in identifiers or prompt in prompts:
            raise ValueError('Repeated request identity or rendered prompt')
        identifiers.add(identity)
        prompts.add(prompt)
    return identifiers, prompts


def validate_manifest(manifest):
    groups = (('train', TRAIN_REQUESTS), ('evaluation', EVALUATION_REQUESTS),
              ('warmup', 2), ('qualification', 16))
    identifiers, prompts = set(), set()
    for role, count in groups:
        current_ids, current_prompts = validate_requests(
            manifest[role+'_requests'], count=count, role=role)
        if identifiers & current_ids or prompts & current_prompts:
            raise ValueError('Training/evaluation/warmup/qualification inputs overlap')
        identifiers |= current_ids
        prompts |= current_prompts
    return dict(status='passed', requests=len(identifiers), training_tail=0,
                training_context=4096)


def event_requests(manifest, event):
    if type(event) is not int or not 1 <= event <= EVENTS:
        raise ValueError('Only the fixed 58 CODE update windows are authorized')
    rows = manifest['train_requests'][(event-1)*WINDOW:event*WINDOW]
    if len(rows) != WINDOW:
        raise ValueError('Incomplete CODE feedback window')
    return rows
