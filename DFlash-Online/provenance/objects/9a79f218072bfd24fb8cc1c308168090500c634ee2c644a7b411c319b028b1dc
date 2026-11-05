"""The bounded protocol; ordinary learning remains in online_feedback."""

from experiments.osd_components.protocol import (
    OUTPUT_REFERENCE_ERRORS,
    publication_errors,
    record_errors as component_record_errors,
    weight_check_errors,
)

ARMS = ('F', 'O')
FEEDBACK_REQUESTS = 128
EVALUATION_REQUESTS = 64
REPEATS = 3
QUALIFICATION_REQUESTS = 8
OPTIMIZER_STEPS = 128
RESPONSE_PRESENTATIONS = 256
EVALUATION_GENERATIONS = EVALUATION_REQUESTS * len(ARMS) * REPEATS
TOTAL_GENERATIONS = FEEDBACK_REQUESTS + EVALUATION_GENERATIONS


def method_order(ordinal, repeat):
    return ARMS if (ordinal + repeat) % 2 == 0 else ARMS[::-1]


def record_errors(record, request, method, checkpoint):
    """Reuse native checks, but never compare new O against the old final O.

    The component validator applies historical counts only to the exact names
    F and O. A distinct identity bypasses only that obsolete O count check;
    canonical outputs, geometry, capture, trace and measurement checks remain.
    """
    reference_method = 'F' if method == 'F' else 'one_event_O'
    return component_record_errors(record, request, reference_method, checkpoint)


def validate_manifest(manifest):
    feedback = manifest['train_requests']
    evaluation = manifest['eval_requests']
    if len(feedback) != FEEDBACK_REQUESTS or len(evaluation) != EVALUATION_REQUESTS:
        raise ValueError('The one-event protocol requires 128 feedback and 64 evaluation requests')
    if [row['panel_ordinal'] for row in evaluation] != list(range(EVALUATION_REQUESTS)):
        raise ValueError('Evaluation must preserve the existing development-panel order')
    feedback_inputs = {tuple(row['prompt_ids']) for row in feedback}
    evaluation_inputs = {tuple(row['prompt_ids']) for row in evaluation}
    if (len(feedback_inputs) != FEEDBACK_REQUESTS
            or len(evaluation_inputs) != EVALUATION_REQUESTS
            or feedback_inputs & evaluation_inputs):
        raise ValueError('Feedback and evaluation inputs must be distinct and disjoint')
    for rows in (feedback, evaluation):
        if len({row['request_id'] for row in rows}) != len(rows):
            raise ValueError('Request IDs must be unique within each phase')
        if any(row['max_new_tokens'] != 2048 for row in rows):
            raise ValueError('The output cap must remain 2048 or EOS')
    if (manifest['policy']['train_context_limit'] != 2272
            or manifest['policy']['serving_context_limit'] != 4096):
        raise ValueError('The qualified training and serving limits must remain unchanged')


def validate_training_accounting(receipt, status):
    expected = {
        'optimizer_steps': OPTIMIZER_STEPS,
        'response_presentations': RESPONSE_PRESENTATIONS,
        'start_step': 0,
        'end_step': OPTIMIZER_STEPS,
    }
    for key, value in expected.items():
        if receipt['training'][key] != value:
            raise RuntimeError(f'One-event training accounting failed: {key}')
    if (status['global_step'] != OPTIMIZER_STEPS
            or status['presentations'] != RESPONSE_PRESENTATIONS
            or status['completed_events'] != ['event_001']
            or status['accumulation_remainder'] != 0
            or status['objective'] != 'osd'
            or status['reduction'] != 'response_mean'):
        raise RuntimeError('The trainer did not finish exactly one ordinary OSD event')
