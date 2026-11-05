"""Fixed scheduling and scientific checks for the checkpoint-only comparison."""

import math

from online_feedback.native import validate_generation

ARMS = ('F', 'O', 'C-only', 'B-only')
SEED = 20261024
REQUESTS = 64
REPEATS = 3
QUALIFICATION_REQUESTS = 8
OUTPUT_CAP = 2048
CONTEXT_LIMIT = 4096
BLOCK_SIZE = 16
GENERATIONS = REQUESTS * len(ARMS) * REPEATS
OUTPUT_REFERENCE_ERRORS = frozenset({'reference_response_ids', 'reference_finish_reason'})


def method_order(panel_ordinal, repeat):
    """Rotate by panel position, never by the original manifest index."""
    shift = (3 * panel_ordinal + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def publication_errors(receipt, checkpoint):
    status = receipt['native_status']
    publication = status['publication']
    checks = {
        'checkpoint_structure': receipt['checkpoint_structure_valid'] is True,
        'checkpoint_identity': receipt['checkpoint_id'] == str(checkpoint),
        'worker_status': status['status'] == 'passed' and status['poisoned'] is None,
        'no_pending_capture': status['capture_pending'] is False,
        'cache_empty': status['cache_tokens'] == publication['cache_after_tokens'] == 0,
        'runtime': status['attention_backend'] == 'fa3' and status['block_size'] == BLOCK_SIZE,
        'publication': publication['status'] == 'passed',
        'published_path': publication['checkpoint_path'] == str(checkpoint),
        'helpers': all(publication[key] is True for key in (
            'helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')),
    }
    return [name for name, passed in checks.items() if not passed]


def weight_check_errors(receipt, checkpoint):
    checks = {
        'native_equality': receipt['status'] == 'passed' and receipt['all_equal'] is True,
        'exported_tensors': receipt['tensors'] == receipt['source_tensor_count'] == 58,
        'native_tensors': receipt['native_tensor_count'] == 43,
        'checkpoint_identity': receipt.get('checkpoint_path') == str(checkpoint),
        'helpers': all(receipt[key] is True for key in (
            'helper_matches_current_layers', 'helper_norm_matches_current_layers', 'fused_enabled')),
    }
    return [name for name, passed in checks.items() if not passed]


def record_errors(record, request, method, checkpoint):
    """Validate the native record, then compare against the retained reference.

    Return errors instead of dropping the raw record. Structural generation
    failures are already raised by the backend and leave its native trace intact.
    This function is also used when inspecting retained baseline records.
    """
    meta = record['metadata']
    trace = meta['trace']
    response = record['response_ids']
    errors = []
    for name in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
        if record[name] != request[name]:
            errors.append('reference_' + name)
    if record['checkpoint_id'] != str(checkpoint):
        errors.append('checkpoint_identity')
    if record['purpose'] != 'evaluation' or meta['feedback_capture_in_timed_request']:
        errors.append('feedback_capture_or_purpose')
    if meta['block_size'] != BLOCK_SIZE or meta['context_limit'] != CONTEXT_LIMIT:
        errors.append('runtime_geometry')
    if meta['cache_policy'] != 'idle_full_flush_before_every_request_and_publication':
        errors.append('cache_policy')
    if meta['runtime'] != 'pinned_SGLang_FA3':
        errors.append('runtime')
    if (meta['verification_calls'] != len(trace)
            or meta['accepted_draft_tokens'] != sum(b['accepted_draft_tokens'] for b in trace)
            or meta['committed_progress'] != len(response) - 1):
        errors.append('measurement_accounting')
    if not 0 <= meta['accepted_draft_tokens'] <= meta['committed_progress']:
        errors.append('accepted_token_bounds')
    for name in ('request_latency_s', 'native_generate_s'):
        if not math.isfinite(meta[name]) or meta[name] <= 0:
            errors.append('invalid_' + name)
    capture = meta['capture']
    if (capture['collect_features'] is not False or capture['feature_bytes'] != 0
            or capture['native_capture_validated'] is not True):
        errors.append('capture_contract')
    try:
        _, finish = validate_generation(
            record['prompt_ids'], record['prompt_ids'] + response, trace,
            max_new_tokens=OUTPUT_CAP, stop_tokens=request['stop_token_ids'])
        if finish != record['finish_reason']:
            errors.append('termination_accounting')
    except RuntimeError as error:
        errors.append('native_trace: ' + str(error))
    if method in ('F', 'O') and meta['verification_calls'] != request['reference_calls'][method]:
        errors.append('historical_count_mismatch')
    return errors
