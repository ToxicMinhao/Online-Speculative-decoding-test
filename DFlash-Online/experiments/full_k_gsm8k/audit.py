"""Reconstruct training evidence without reloading retired tensors or running models.

The raw work ledger, committed buffers, native traces and event observations are
checked independently of STATUS. State equality itself remains scoped to the
recorded live/tensor checks made before each boundary was retired.
"""
from collections import Counter
import json
import math
from pathlib import Path
import time

from online_feedback.native import validate_generation
from online_feedback.storage import write_json
from experiments.ok_coding.campaign import read_work
from experiments.o_half_matched_lr.archive import ArchiveReader, full_trace_errors
from experiments.osd_components.protocol import publication_errors, weight_check_errors
from .contract import ARMS, EVENTS, INTERMEDIATE_EVENTS, NEW_BUDGET, validate_plan
from .training import LEARNING_RATES, validate_event_receipt

ARM_WORK = {
    'online_generations': 7473, 'correctness_generations': 4,
    'intermediate_generations': 384, 'terminal_preparation_forwards': 7424,
    'response_forwards': 14848, 'response_backwards': 14848, 'optimizer_steps': 7424,
}
ERROR_TYPES = (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid nonnegative measurement: ' + name)
    return value


def close_number(actual, expected, name):
    # Materialized FP32 accumulation can round once before becoming Python float.
    # This checks accounting only; it is not a training-qualification tolerance.
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-5),
            'Materialized accumulation differs: ' + name)


def audit_work(path):
    """Reconstruct both action totals and the causal collection/update ordering."""
    work = read_work(path, ARM_WORK)
    completed = dict.fromkeys(ARM_WORK, 0)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        kind, ordinal = row['kind'], row['ordinal']
        if row['phase'] == 'completed':
            completed[kind] += 1
            continue
        steps = completed['optimizer_steps']
        if kind == 'online_generations':
            expected_steps = min((ordinal - 1) // 128, EVENTS) * 128
            require(steps == expected_steps, 'Online response used the wrong update boundary')
        elif kind == 'terminal_preparation_forwards':
            event = (ordinal - 1) // 128 + 1
            require(completed['online_generations'] == event * 128
                    and steps == (event - 1) * 128,
                    'Teacher preparation did not follow complete own feedback')
        elif kind in ('response_forwards', 'response_backwards'):
            event = (ordinal - 1) // 256 + 1
            require(completed['terminal_preparation_forwards'] == event * 128,
                    'Training preceded this event\'s complete teacher preparation')
            if kind == 'response_forwards':
                require(steps == (ordinal - 1) // 2, 'Unexpected ordinary forward order')
            else:
                require(completed['response_forwards'] == ordinal,
                        'Backward lacks its one original response forward')
        elif kind == 'optimizer_steps':
            require(completed['response_forwards'] == completed['response_backwards'] == 2 * ordinal,
                    'Adam step did not average exactly two new response losses')
        elif kind == 'correctness_generations':
            require(completed['online_generations'] == 128
                    and steps == (0 if ordinal == 1 else 128),
                    'Inherited correctness generation moved outside the first real event')
        elif kind == 'intermediate_generations':
            endpoint = INTERMEDIATE_EVENTS[(ordinal - 1) // 128]
            require(steps == endpoint * 128 and completed['online_generations'] == endpoint * 128,
                    'Intermediate evaluation changed its prescribed endpoint')
    require(work['attempts'] == work['completed'] == ARM_WORK
            and not any(work['unknown'].values()), 'Incomplete or unknown scientific work')
    return work


def audit_native(reader, path, request, checkpoint, *, capture, label=None,
                 session=None, request_index=None):
    """Validate the raw native decision trace and recompute every reported count."""
    record = reader.read(path)
    metadata = record['metadata']
    response = record['response_ids']
    expected_label = request['request_id'] if label is None else label
    expected_index = request['request_index'] if request_index is None else request_index
    require(record['record_id'] == expected_label and record['request_index'] == expected_index
            and record['prompt_ids'] == request['prompt_ids']
            and record['checkpoint_id'] == str(checkpoint)
            and record['max_new_tokens'] == 2048
            and record['purpose'] == ('serving' if capture else 'evaluation'),
            'Native identity, prompt, checkpoint or purpose differs')
    if session is not None:
        require(record['session_id'] == session, 'Native session differs')
    if 'stop_token_ids' in request:
        require(record['stop_token_ids'] == request['stop_token_ids'], 'Stop tokens differ')
    require(all(type(token) is int and token >= 0 for token in record['prompt_ids'] + response)
            and len(record['prompt_ids']) + 2048 <= 4096, 'Invalid token IDs or full context')
    _, finish = validate_generation(record['prompt_ids'], record['prompt_ids'] + response,
        metadata['trace'], max_new_tokens=2048, stop_tokens=record['stop_token_ids'])
    require(finish == record['finish_reason'] and not full_trace_errors(record),
            'Native termination or proposal/verification trace differs')
    capture_record = metadata['capture']
    require(metadata['feedback_capture_in_timed_request'] is capture
            and capture_record['collect_features'] is capture
            and capture_record['native_capture_validated'] is True
            and metadata['block_size'] == 16 and metadata['context_limit'] == 4096
            and metadata['runtime'] == 'pinned_SGLang_FA3'
            and metadata['cache_policy'] == 'idle_full_flush_before_every_request_and_publication',
            'Native capture or runtime contract differs')
    native = reader.read(reader.saved_path(metadata['native_trace_path']))
    require(native == capture_record, 'Retained native trace differs from serving record')
    calls = len(metadata['trace'])
    accepted = sum(block['accepted_draft_tokens'] for block in metadata['trace'])
    progress = len(response) - 1
    require(metadata['verification_calls'] == calls
            and metadata['accepted_draft_tokens'] == accepted
            and metadata['committed_progress'] == progress
            and 0 <= accepted <= progress, 'Native scalar count does not reconstruct')
    if 'calls_per_1000_output_tokens' in metadata:
        require(metadata['calls_per_1000_output_tokens'] == 1000 * calls / len(response),
                'Native output-normalized count differs')
    if 'committed_progress_per_call' in metadata:
        require(metadata['committed_progress_per_call'] == (progress / calls if calls else None),
                'Native progress-per-call differs')
    for field in ('request_latency_s', 'native_generate_s'):
        finite_nonnegative(metadata[field], field)
    finite_nonnegative(capture_record['cache_maintenance_s'], 'cache_maintenance_s')
    if capture:
        require(capture_record['covered_token_ids'] == record['prompt_ids'] + response[:-1]
                and capture_record['target_argmax_ids'] == response
                and capture_record['label_binding_scope'] == 'native_prefill_and_committed_verification',
                'Captured native target conditioning differs')
    return record


def audit_response_observations(result, responses, windows):
    """Cross-check original response rows against once-only accumulation windows."""
    require(len(responses) == 256 and len(windows) == 128, 'Missing original response/window rows')
    original_totals = Counter()
    anchors = 0
    for index, (row, observed) in enumerate(zip(result['records'], responses)):
        require(row['response_observation'] == observed
                and observed['scope'] == 'one_original_response_forward'
                and observed['record_id'] == row['record_id']
                and observed['event_id'] == result['event_id']
                and observed['pass_index'] == row['pass'] == index // 128
                and observed['presentation_ordinal'] == index
                and observed['sampled_anchors'] == row['sampled_anchors'],
                'Original response observation identity/anchors differs')
        require(math.isfinite(observed['loss']), 'Nonfinite original response loss')
        for name, value in observed['sum_metrics'].items():
            require(math.isfinite(value), 'Nonfinite original response statistic: ' + name)
            original_totals[name] += value
        selected = observed['sum_metrics']['selected_position_occurrences']
        available = observed['sum_metrics']['available_position_occurrences']
        require(0 <= selected <= available, 'Invalid selected versus available exposure')
        anchors += sum(sum(draw['keep_mask'][0]) for draw in observed['sampled_anchors'])
    totals = result['selection_totals']
    require(result['selection_totals_source'] == 'optimizer_boundary_accumulation_windows_once',
            'Unrecognized exposure reduction')
    for name, expected in totals.items():
        boundary_sum = sum(row['metrics'][name] for row in result['records'] if row['optimizer_stepped'])
        require(boundary_sum == expected, 'Selection totals double-counted or altered: ' + name)
        if name in original_totals:
            close_number(original_totals[name], expected, name)
    require(totals['response_presentations'] == 256, 'Accumulated response count differs')
    for index, window in enumerate(windows):
        pair = responses[2 * index:2 * index + 2]
        require(window['scope'] == 'optimizer_window' and window['event_id'] == result['event_id']
                and window['window'] == index + 1
                and window['global_step'] == result['start_step'] + index + 1
                and window['record_ids'] == [row['record_id'] for row in pair],
                'Optimizer window identity differs')
        finite_nonnegative(window['gradient_norm'], 'gradient_norm')
        close_number(window['mean_response_loss'], sum(row['loss'] for row in pair) / 2, 'window loss')
    work = result['objective_work']
    require(work['reference_operations'] == 0 and work['work']['response_forwards'] == 256
            and work['work']['draft_forwards'] == 256, 'Extra or missing current/reference forward')
    for name, value in work['work'].items():
        require(type(value) is int and value >= 0, 'Malformed actual native work: ' + name)
    return dict(selection_totals=dict(totals), sampled_anchor_occurrences=anchors,
                original_response_observations=256, optimizer_windows=128,
                actual_native_work=work['work'])


def audit_preparation(reader, prepared, feature, record):
    """Audit retained exact tokens/mask metadata; retired tensor values are not reread."""
    prompt, response = record['prompt_ids'], record['response_ids']
    combined = prompt + response
    require(prepared == reader.read(reader.saved_path(prepared['feature_path']).with_suffix('.json')),
            'Prepared metadata differs from controller receipt')
    expected = dict(record_id=record['record_id'], prompt_length=len(prompt),
        response_length=len(response), input_ids=combined, target_argmax_ids=response,
        label_mismatches=[], capture_backend='same_frozen_SGLang_native_capture',
        native_capture_validated=True, label_binding_scope='native_prefill_and_committed_verification',
        final_feature_logits_used_for_labels=False, response_only_loss=True,
        status='prepared', supervised_positions=len(response))
    require(all(prepared.get(key) == value for key, value in expected.items()),
            'Prepared full-context response-only target binding differs')
    require(len(response) >= 2 and len(combined) <= 2272
            and prepared['feature_shape'][:2] == [1, len(combined)]
            and feature['record_id'] == record['record_id']
            and feature['path'] == prepared['feature_path']
            and feature['length'] == len(combined)
            and feature['prompt_length'] == len(prompt)
            and feature['response_tokens'] == len(response)
            and feature['adjacent_response_positions'] == len(response) - 1,
            'Prepared feature shape, exposure or no-truncation contract differs')
    terminal = prepared['terminal_feature']
    require(reader.read(reader.saved_path(terminal['tensor_path']).with_suffix('.json')) == terminal
            and terminal['input_ids'] == combined
            and terminal['positions'] == [len(combined) - 1]
            and terminal['final_feature_logits_used_for_labels'] is False,
            'Terminal completion feature supplied a label or changed identity')
    for key in ('teacher_preparation_s', 'feature_persistence_s'):
        finite_nonnegative(prepared[key], key)


def audit_retirement(reader, event_root, result, records, previous_checkpoint):
    retired = reader.read(event_root / 'RETIREMENT.json')
    planned = reader.read(event_root / 'RETIREMENT_PLAN.json')
    require(retired['status'] == 'completed' and planned['status'] == 'planned'
            and retired['event_id'] == result['event_id']
            and retired['files'] == retired['removed'] == planned['files']
            and retired['compact_evidence'] == planned['compact_evidence']
            and retired['retained_current_state'] == result['full_state_path']
            and retired['retained_current_export'] == result['checkpoint_path'],
            'Retirement is not the complete committed plan')
    expected = {record['metadata']['capture']['feature_path']: 'native_capture' for record in records}
    controller = reader.read(event_root / 'loop/events' / (result['event_id'] + '.json'))
    for prepared in controller['preparation']:
        expected[prepared['feature_path']] = 'prepared_feature'
        expected[prepared['terminal_feature']['tensor_path']] = 'terminal_feature'
    if previous_checkpoint is not None:
        expected[str(Path(previous_checkpoint) / 'model.safetensors')] = 'previous_export_weights'
    require({row['path']: row['kind'] for row in retired['files']} == expected
            and len(retired['files']) == len(expected)
            and retired['removed_bytes'] == sum(row['bytes'] for row in retired['files']),
            'Retired files exceed or omit the qualified own-artifact set')
    for entry in retired['compact_evidence']:
        path = reader.saved_path(entry['path'])
        if str(path) not in reader.files:
            reader.read(path)
        require(reader.files[str(path)] == {key: entry[key] for key in ('bytes', 'sha256')},
                'Compact evidence changed since retirement: ' + entry['path'])
    return dict(files=len(retired['removed']), bytes=retired['removed_bytes'])


def check_boundary(marker, validation, arm, event, result):
    expected = dict(status='passed', arm=arm, completed_event=event,
        global_step=event * 128, presentations=event * 256, adam_state_entries=58,
        cuda_rng_states=1, extra_model_passes=0, extra_optimizer_steps=0, export_exact=True)
    require(all(validation.get(key) == value for key, value in expected.items()),
            'Full-state boundary validation is incomplete')
    live = validation['live']
    require(live['status'] == 'passed' and live['exact_resume'] is True
            and live['mismatches'] == [] and live['global_step'] == event * 128
            and live['presentations'] == event * 256 and live['no_rng_reset'] is True,
            'Live model/Adam/RNG boundary did not match persisted state')
    require(marker['status'] == 'passed' and marker['arm'] == arm and marker['event'] == event
            and marker['scientific_check'] == validation
            and Path(marker['weights']).parent == Path(result['checkpoint_path'])
            and Path(marker['state']).parent == Path(result['checkpoint_path']).parent,
            'Boundary identity differs from the trained export')
    for name in ('state_identity', 'weights_identity'):
        require(marker[name]['bytes'] > 0 and len(marker[name]['sha256']) == 64,
                'Missing retained tensor identity')


def audit_resources(resource):
    require(all(resource[name] == 0 for name in
                ('extra_forwards', 'extra_backwards', 'extra_optimizer_steps')),
            'Resource observation introduced model work')
    allocated = resource['training_peak_allocated_bytes']
    reserved = resource['training_peak_reserved_bytes']
    require(type(allocated) is int and type(reserved) is int
            and 0 < allocated <= reserved, 'Invalid training memory peaks')
    for name, timer in resource['training_compiler_timers'].items():
        require(timer['calls'] == len(timer['durations_s'])
                and timer['total_s'] == sum(timer['durations_s']),
                'Compiler timer accounting differs: ' + name)
        for value in timer['durations_s']:
            finite_nonnegative(value, 'compiler timer ' + name)
    return resource


def audit_correctness(reader, base, requests, first_result, first_record, plan):
    root = base / 'events/event_001'
    replay = audit_native(reader, root / 'loop/evaluation/first_capture_disabled.json',
        requests[0], plan['draft'], capture=False, label='first_capture_disabled',
        session='evaluation', request_index=1)
    comparison = reader.read(root / 'loop/evaluation/first_capture_disabled.comparison.json')
    require(comparison['record'] == replay and comparison['same_input'] is True
            and comparison['output_equal'] is True and comparison['learning_unchanged'] is True
            and comparison['canonical_response_ids'] == first_record['response_ids']
            and replay['response_ids'] == first_record['response_ids']
            and replay['finish_reason'] == first_record['finish_reason']
            and replay['metadata']['verification_calls'] == first_record['metadata']['verification_calls'],
            'First capture-disabled control does not reproduce the actual first response')
    records = {}
    for name in ('updated', 'frozen', 'reloaded'):
        checkpoint = plan['draft'] if name == 'frozen' else first_result['checkpoint_path']
        records[name] = audit_native(reader, root / f'qualification_{name}.json', requests[128],
            checkpoint, capture=False, label='qualification_' + name, session='qualification', request_index=0)
    require(len({tuple(row['response_ids']) for row in records.values()}) == 1
            and len({row['finish_reason'] for row in records.values()}) == 1
            and records['updated']['metadata']['verification_calls'] == records['reloaded']['metadata']['verification_calls'],
            'Updated/frozen/reloaded native controls differ')
    check = reader.read(root / 'INHERITED_CORRECTNESS.json')
    require(all(check[key] is True for key in ('same_input', 'same_output', 'reloaded_calls_equal'))
            and check['feedback_collected'] is False and check['performance_win_required'] is False
            and check['generations'] == 4 and check['disposable_optimizer_steps'] == 0
            and check['first_capture_disabled_in_loop'] is True, 'Inherited control receipt differs')
    return {'generations': 4, 'disposable_optimizer_steps': 0,
            'publication_replay_wall_s': finite_nonnegative(check['wall_s'], 'publication replay wall'),
            'capture_disabled_replay_wall_s': finite_nonnegative(
                reader.read(root / 'loop/evaluation/first_capture_disabled.costs.json')['evaluation_call_s'],
                'capture-disabled replay wall')}


def _audit_event(reader, base, plan, arm, event, requests, previous_checkpoint):
    event_id = f'event_{event:03d}'
    root = base / 'events' / event_id
    trainer_root = base / 'trainer/events' / event_id
    result = reader.read(trainer_root / 'RESULT.json')
    validate_event_receipt(result, arm=arm, event_number=event)
    pending = reader.read(trainer_root / 'PENDING.json')
    require(pending == {key: result[key] for key in ('event_id', 'features', 'passes', 'start_step')}
            and not (trainer_root / 'FAILED.json').exists(),
            'Pending event identity changed or a failure was hidden')
    require(result['checkpoint_path'] == str(Path(plan['checkpoints'][arm]).parent.parent / event_id / 'export'),
            'Event export path is not this own chronological endpoint')
    controller = reader.read(root / 'loop/events' / (event_id + '.json'))
    require(controller['event_id'] == event_id and controller['training'] == result,
            'Controller and native trainer results differ')
    require(not publication_errors(controller['activation'], result['checkpoint_path']),
            'Native activation failed')
    publication = reader.read(root / 'PUBLICATION_CHECK.json')
    require(not weight_check_errors(publication, result['checkpoint_path'])
            and publication['event_id'] == event_id
            and publication['trainer_status']['global_step'] == event * 128,
            'Published native weights did not equal this export')
    marker = reader.read(base / 'boundaries' / (event_id + '.json'))
    validation = reader.read(base / 'boundaries' / (event_id + '_validation.json'))
    check_boundary(marker, validation, arm, event, result)
    group = requests[(event - 1) * 128:event * 128]
    records = [audit_native(reader, root / 'loop/serving' / (request['request_id'] + '.json'),
        request, previous_checkpoint or plan['draft'], capture=True, session='gsm8k_train') for request in group]
    buffer = reader.read(root / 'loop/feedback_buffer/state.json')
    require(buffer['schema_version'] == 2 and buffer['capacity'] == 128 and buffer['records'] == records
            and len(buffer['events']) == 1, 'Buffer lost chronological own feedback')
    committed = buffer['events'][0]
    require(committed['event_id'] == event_id and committed['status'] == 'committed'
            and committed['record_ids'] == [request['request_id'] for request in group]
            and committed['checkpoint_id'] == result['checkpoint_path']
            and committed['receipt'] == controller and committed['error'] is None,
            'Feedback was not committed after this own trained publication')
    require(len(controller['preparation']) == len(result['features']) == 128,
            'Wrong teacher preparation count')
    for prepared, feature, record in zip(controller['preparation'], result['features'], records):
        audit_preparation(reader, prepared, feature, record)
    responses = reader.read(trainer_root / 'RESPONSE_OBSERVATIONS.jsonl', lines=True)
    windows = reader.read(trainer_root / 'WINDOW_OBSERVATIONS.jsonl', lines=True)
    exposure = audit_response_observations(result, responses, windows)
    require(all(row['learning_rate'] == LEARNING_RATES[arm] for row in windows),
            'Optimizer window learning rate differs')
    retirement = audit_retirement(reader, root, result, records, previous_checkpoint)
    if event > 1:
        prior = reader.read(base / 'boundaries' / f'event_{event-1:03d}.json')
        retired = reader.read(base / 'boundaries' / f'event_{event-1:03d}_retirement.json')
        require(retired['status'] == 'completed' and retired['retired_state'] == prior['state']
                and retired['identity'] == prior['state_identity'] and retired['successor'] == marker['state'],
                'Own predecessor state retirement lacks its verified successor')
    costs = dict(result['costs'])
    costs['blocking_update_s'] = reader.read(root / 'loop/events' / (event_id + '.costs.json'))['update_call_s']
    costs['teacher_preparation_s'] = sum(row['teacher_preparation_s'] for row in controller['preparation'])
    costs['feature_persistence_s'] = sum(row['feature_persistence_s'] for row in controller['preparation'])
    costs['activation_s'] = controller['activation']['activation_s']
    costs['controller_serving_s'] = 0.
    costs['native_request_s'] = 0.
    costs['native_generate_s'] = 0.
    costs['cache_maintenance_s'] = 0.
    costs['capture_host_s'] = 0.
    costs['capture_persistence_s'] = 0.
    for request, record in zip(group, records):
        receipt = reader.read(root / 'loop/serving' / (request['request_id'] + '.costs.json'))
        costs['controller_serving_s'] += finite_nonnegative(receipt['serve_call_s'], 'serve_call_s')
        metadata, capture = record['metadata'], record['metadata']['capture']
        costs['native_request_s'] += metadata['request_latency_s']
        costs['native_generate_s'] += metadata['native_generate_s']
        costs['cache_maintenance_s'] += capture['cache_maintenance_s']
        costs['capture_host_s'] += finite_nonnegative(capture['capture_host_s'], 'capture_host_s')
        costs['capture_persistence_s'] += finite_nonnegative(capture['feature_persistence_s'], 'capture_persistence_s')
    close_number(costs['training_s'], sum(row['training_s'] for row in result['records']), 'training time')
    for name, value in result['objective_work'].items():
        if name.endswith('_s'):
            costs['objective_' + name] = value
    for name, value in costs.items():
        finite_nonnegative(value, name)
    resources = audit_resources(reader.read(root / 'RESOURCE_COSTS.json'))
    return dict(event=event, result=result, first_record=records[0], costs=costs,
                exposure=exposure, retirement=retirement, raw_online_generations=len(records),
                native_blocks=sum(len(row['metadata']['trace']) for row in records),
                boundary=marker, resources=resources)


def audit_training(root, *, raise_on_error=True):
    """Always preserve a compact audit; reject invalid completion before final serving."""
    root = Path(root)
    started = time.perf_counter()
    reader = ArchiveReader(root)
    errors = []
    audit = dict(status='running', budget=NEW_BUDGET, arms={}, errors=errors, extra_model_work=0,
        scope='Compact record reconstruction; no retired feature/state tensor reload. '
              'Numerical state equality is supported by original live boundary receipts.')

    def checked(label, action):
        try:
            return action()
        except ERROR_TYPES as error:
            errors.append(label + ': ' + repr(error))
            return None

    def load_plan():
        value = reader.read(root / 'sealed/PLAN.json')
        for name in ('manifest', 'canonicals', 'checkpoints', 'evaluation_requests', 'binding', 'draft'):
            require(name in value, 'Missing sealed plan field: ' + name)
        require(all(arm in value['checkpoints'] for arm in ARMS), 'Missing own endpoint paths')
        return value

    plan = checked('sealed plan', load_plan)
    if plan is not None:
        reader.origin = Path(plan['manifest']).parent.parent
        checked('plan contract', lambda: validate_plan({**plan, 'manifest': str(reader.saved_path(plan['manifest']))}))
        manifest = checked('manifest', lambda: reader.read(reader.saved_path(plan['manifest'])))
        canonicals = checked('canonical inputs', lambda: reader.read(reader.saved_path(plan['canonicals'])))
        requests = manifest['train_requests'] if manifest else []
        for arm in ARMS:
            base = root / 'campaign_001/arms' / arm
            totals, exposure, actual_native = Counter(), Counter(), Counter()
            resource_records = []
            row = dict(events_completed=0, optimizer_steps=0, response_presentations=0,
                online_generations=0, terminal_preparation_forwards=0, correctness_generations=0,
                intermediate_generations=0, actual_work={}, costs={}, exposure={}, final_checkpoint=None,
                event_audits=[], invalid_event_numbers=[])
            audit['arms'][arm] = row
            work = checked(arm + ' work', lambda: audit_work(base / 'WORK.jsonl'))
            row['actual_work'] = work or checked(arm + ' partial work', lambda: read_work(base / 'WORK.jsonl', ARM_WORK)) or {}
            status = checked(arm + ' status', lambda: reader.read(base / 'STATUS.json'))
            initial = checked(arm + ' initial state receipt', lambda: reader.read(base / 'trainer/INITIAL_CHECK.json'))
            if initial:
                expected = dict(status='passed', arm=arm, completed_event=0, global_step=0,
                    presentations=0, adam_state_entries=0, cuda_rng_states=1, export_exact=True,
                    extra_model_passes=0, extra_optimizer_steps=0)
                checked(arm + ' initialization', lambda: require(all(initial.get(key) == value for key, value in expected.items())
                    and (arm == 'K' or initial['paired_initialization']['exact_released_weights_empty_adam_rng'] is True),
                    'Genuine paired released initialization is not established'))
                if arm == 'K_HALF':
                    checked('K_HALF independent RNG', lambda: require(
                        initial['paired_initialization']['rng_overwritten'] is False,
                        'Independent initialization was forced by replacing RNG'))
            previous = None
            for event in range(1, EVENTS + 1):
                evidence = checked(f'{arm}/event_{event:03d}', lambda event=event, previous=previous:
                    _audit_event(reader, base, plan, arm, event, requests, previous))
                expected_export = str(Path(plan['checkpoints'][arm]).parent.parent / f'event_{event:03d}/export')
                previous = expected_export
                if evidence is None:
                    row['invalid_event_numbers'].append(event)
                    continue
                row['events_completed'] += 1
                row['optimizer_steps'] += 128
                row['response_presentations'] += 256
                row['online_generations'] += 128
                row['terminal_preparation_forwards'] += 128
                totals.update(evidence['costs'])
                exposure.update(evidence['exposure']['selection_totals'])
                exposure['sampled_anchor_occurrences'] += evidence['exposure']['sampled_anchor_occurrences']
                actual_native.update(evidence['exposure']['actual_native_work'])
                row['event_audits'].append(dict(event=event, status='passed', costs=evidence['costs'],
                    exposure=evidence['exposure'], retirement=evidence['retirement'], native_blocks=evidence['native_blocks']))
                resource_records.append(dict(event=event, **evidence['resources']))
                if event == 1:
                    gate = checked(arm + ' inherited controls', lambda: audit_correctness(
                        reader, base, requests, evidence['result'], evidence['first_record'], plan))
                    if gate:
                        row['correctness_generations'] = gate['generations']
                        totals['correctness_publication_replay_s'] += gate['publication_replay_wall_s']
                        totals['correctness_capture_disabled_replay_s'] += gate['capture_disabled_replay_wall_s']
                if event in INTERMEDIATE_EVENTS and canonicals is not None:
                    from .evaluation import audit_stage
                    folder = base / 'evaluation' / f'event_{event:03d}'
                    stage = checked(f'{arm} intermediate {event}', lambda: audit_stage(folder,
                        plan['evaluation_requests'], {arm: expected_export}, canonicals=canonicals,
                        phase='intermediate', arm=arm, event=event, evidence_root=root))
                    if stage:
                        errors.extend(f'{arm} intermediate {event}: ' + error for error in stage['errors'])
                        row['intermediate_generations'] += len(stage['rows'])
                        for measured in stage['rows']:
                            raw = checked(f'{arm} intermediate native binding', lambda measured=measured:
                                reader.read(reader.saved_path(measured['record_path'])))
                            if raw:
                                checked(f'{arm} intermediate native source', lambda raw=raw:
                                    reader.read(reader.saved_path(raw['metadata']['native_trace_path'])))
                    unchanged = checked(f'{arm} intermediate {event} state', lambda:
                        reader.read(folder / 'TRAINING_STATE_UNCHANGED.json'))
                    if unchanged:
                        checked(f'{arm} intermediate {event} exact state', lambda:
                            require(unchanged['status'] == 'passed' and unchanged['export_exact'] is True
                                and unchanged['live']['exact_resume'] is True and unchanged['live']['mismatches'] == []
                                and unchanged['global_step'] == event * 128, 'Intermediate evaluation changed own state'))
            for request in requests[7424:]:
                record = checked(arm + ' tail ' + request['request_id'], lambda request=request:
                    audit_native(reader, base / 'tail' / (request['request_id'] + '.json'), request,
                        plan['checkpoints'][arm], capture=False, session='gsm8k_tail'))
                if record:
                    row['online_generations'] += 1
                    totals['tail_native_request_s'] += record['metadata']['request_latency_s']
                    totals['tail_native_generate_s'] += record['metadata']['native_generate_s']
                    totals['tail_cache_maintenance_s'] += record['metadata']['capture']['cache_maintenance_s']
            final = checked(arm + ' final check', lambda: reader.read(base / 'FINAL_STATE_CHECK.json'))
            if final:
                checked(arm + ' final state', lambda: require(final['status'] == 'passed'
                    and final['completed_event'] == EVENTS and final['global_step'] == 7424
                    and final['presentations'] == 14848 and final['export_exact'] is True
                    and final['live']['exact_resume'] is True and final['live']['mismatches'] == [],
                    'Final state is not the own complete event-58 endpoint'))
            if status:
                checked(arm + ' status reconciliation', lambda: require(status['status'] == 'completed'
                    and status['phase'] == 'completed' and status['events_completed'] == 58
                    and status['tail_completed'] == 49 and status['binding'] == plan['binding']
                    and status['final_checkpoint'] == plan['checkpoints'][arm]
                    and status['actual_work'] == row['actual_work'] and not status['invalid_online_records'],
                    'Status is inconsistent with independently reconstructed records'))
                totals['loading_s'] = sum(segment['backend_loading_s'] + segment['trainer_loading_s'] for segment in status['segments'])
                totals['online_wall_s'] = sum(status['worker_segments_s'])
                totals['initial_validation_s'] = sum(segment.get('initial_validation_s', 0.) for segment in status['segments'])
                totals['publication_validation_s'] = sum(entry['costs']['publication_validation_s'] for entry in status['event_costs'])
                totals['retention_s'] = sum(entry['costs']['retention_s'] for entry in status['event_costs'])
                checked(arm + ' status event costs', lambda: require(
                    [entry['event'] for entry in status['event_costs']] == list(range(1, 59))
                    and all(entry['costs']['training_s'] == row['event_audits'][index]['costs']['training_s']
                            for index, entry in enumerate(status['event_costs'])), 'Status event costs changed'))
            executed = checked(arm + ' worker receipt', lambda: reader.read(root / f'campaign_001/EXEC_{arm}.json'))
            if executed:
                checked(arm + ' worker completion', lambda: require(executed['arm'] == arm
                    and executed['returncode'] == 0, 'Own arm worker did not exit successfully'))
            native_files = {str(path) for path in base.glob('native_attempt_*/native/request_[0-9]*.json')}
            native_read = {path for path in reader.files if path in native_files}
            checked(arm + ' native execution inventory', lambda: require(
                native_files == native_read and len(native_files) == 7473 + 4 + 384,
                'Native trace inventory contains missing or undeclared generations'))
            row['native_trace_files'] = len(native_files)
            checked(arm + ' finite costs', lambda: [finite_nonnegative(value, name)
                for name, value in totals.items()])
            row.update(costs=dict(totals), exposure=dict(exposure), actual_native_work=dict(actual_native),
                       final_checkpoint=plan['checkpoints'][arm] if final else None,
                       cost_scope='Nested component clocks must not be added to controller update or worker totals.',
                       resource_records=resource_records,
                       peak_allocated_bytes=max((r['training_peak_allocated_bytes'] for r in resource_records), default=None),
                       peak_reserved_bytes=max((r['training_peak_reserved_bytes'] for r in resource_records), default=None),
                       unavailable_costs=['Native serving-worker compilation separated from loading/serving'])
            checked(arm + ' completion', lambda: require(row['events_completed'] == 58
                and row['optimizer_steps'] == 7424 and row['response_presentations'] == 14848
                and row['online_generations'] == 7473 and row['terminal_preparation_forwards'] == 7424
                and row['correctness_generations'] == 4 and row['intermediate_generations'] == 384,
                'Incomplete verified own trajectory'))
    audit.update(status='passed' if not errors else 'invalid_or_incomplete',
                 compact_files_checked=len(reader.files), audit_wall_s=time.perf_counter() - started)
    destination = root / 'campaign_001/TRAINING_AUDIT.json'
    write_json(destination, audit)
    if errors and raise_on_error:
        raise ValueError('Training evidence audit failed; preserved ' + str(destination) + ': ' + errors[0])
    return audit
