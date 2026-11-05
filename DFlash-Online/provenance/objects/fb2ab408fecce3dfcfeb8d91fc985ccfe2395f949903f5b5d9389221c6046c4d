"""Local descriptive analysis of retained event 1 and continuation events 2/4.

This program reads compact evidence only. It never starts inference/training,
loads a tensor checkpoint, repairs a record, or compares latency across events.
The independent execution/state audit remains a separate required artifact.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ARMS = ('O', 'K')
EVENTS = (1, 2, 4)
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
HISTOGRAM = tuple('first_error_position_' + str(index) for index in range(1, 16))
RESEARCH = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reduction(reference, candidate):
    return 100 * (reference - candidate) / reference if reference else None


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def direction(reference, candidate):
    return 'improved' if candidate < reference else 'regressed' if candidate > reference else 'tied'


def check(condition, issues, scope, message):
    if not condition:
        issues.append({'scope': scope, 'message': message})


def fingerprint(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def bind_panel(campaign, requests, issues):
    """Bind the retained panel to this continuation's sealed execution plan."""
    path = campaign.parent / 'sealed/PLAN.json'
    seal_path = campaign.parent / 'sealed/SEAL.json'
    if not path.is_file() or not seal_path.is_file():
        check(False, issues, 'provenance', 'Sealed continuation PLAN/SEAL unavailable')
        return None
    plan, seal = load(path), load(seal_path)
    identities = [identity for name, identity in seal.get('files', {}).items()
                  if name.endswith('/sealed/PLAN.json')]
    actual = fingerprint(path)
    check(len(identities) == 1 and all(actual[key] == identities[0][key] for key in ('bytes', 'sha256')),
          issues, 'provenance', 'Continuation PLAN bytes differ from its seal')
    check(plan.get('eval_requests') == requests, issues, 'provenance',
          'Retained event1 panel differs from sealed continuation panel')
    return plan


def validate_row(row, request, directory, event, checkpoints):
    """Recheck each returned row against retained F and its local native record."""
    errors = list(row.get('errors', []))
    if row.get('valid_measurement') is not True:
        errors.append('runner_rejected_or_pending')
    for key in COUNTS:
        if type(row.get(key)) is not int or row[key] < 0:
            errors.append('invalid_' + key)
    for key in ('request_latency_s', 'native_generate_s'):
        if not finite(row.get(key), positive=True):
            errors.append('invalid_' + key)
    if not finite(row.get('cache_maintenance_s')):
        errors.append('invalid_cache_maintenance_s')
    arm, repeat = row.get('method'), row.get('repeat')
    if arm not in ARMS or type(repeat) is not int or repeat not in range(3):
        errors.append('invalid_slot')
    ordinal = request['panel_ordinal']
    label = f'{ordinal:04d}_r{repeat}_{arm}'
    path = directory / (label + '.json')
    if Path(row.get('record_path', '')).name != path.name:
        errors.append('native_path_identity')
    if row.get('panel_ordinal') != ordinal or row.get('original_index') != request['original_index']:
        errors.append('request_identity')
    try:
        record = load(path)
        metadata, response = record['metadata'], record['response_ids']
        if any(record[key] != request[key] for key in
               ('prompt_ids', 'response_ids', 'finish_reason', 'max_new_tokens', 'stop_token_ids')):
            errors.append('retained_F_input_output_or_termination')
        session = 'prefix_ranking_evaluation' if event == 1 else 'ok_continuation_evaluation'
        if (record['record_id'] != label or record['request_index'] != ordinal
                or record['session_id'] != session or record['purpose'] != 'evaluation'):
            errors.append('native_identity_or_purpose')
        expected_checkpoint = checkpoints.get(arm)
        if (not expected_checkpoint or record['checkpoint_id'] != expected_checkpoint
                or row.get('checkpoint_id') != expected_checkpoint):
            errors.append('checkpoint_identity')
        if (metadata['runtime'] != 'pinned_SGLang_FA3' or metadata['block_size'] != 16
                or metadata['context_limit'] != 4096
                or metadata['cache_policy'] != 'idle_full_flush_before_every_request_and_publication'):
            errors.append('runtime_contract')
        capture = metadata['capture']
        if (metadata['feedback_capture_in_timed_request'] is not False
                or capture['collect_features'] is not False or capture['feature_bytes'] != 0
                or capture['native_capture_validated'] is not True):
            errors.append('capture_contract')
        for key in COUNTS[:3] + ('request_latency_s', 'native_generate_s'):
            if row.get(key) != metadata[key]:
                errors.append('native_scalar_' + key)
        if (row.get('output_tokens') != len(response) or row.get('finish_reason') != record['finish_reason']
                or row.get('prompt_tokens') != len(record['prompt_ids'])
                or row.get('cache_maintenance_s') != capture['cache_maintenance_s']
                or row.get('cap_hit') != (record['finish_reason'] == 'length')):
            errors.append('native_output_scalars')
        if (not response or len(response) > 2048
                or any(type(token) is not int or token < 0 for token in response)
                or any(token in request['stop_token_ids'] for token in response[:-1])):
            errors.append('output_boundary')
        finish = 'eos' if response[-1] in request['stop_token_ids'] else 'length'
        if finish != record['finish_reason'] or (finish == 'length' and len(response) != 2048):
            errors.append('termination_accounting')
        offset, accepted = 0, 0
        for block in metadata['trace']:
            progress = block['committed_progress']
            kept = block['accepted_draft_tokens']
            if (type(progress) is not int or not 1 <= progress <= 16
                    or type(kept) is not int or not 0 <= kept <= min(progress, 15)
                    or block['output_start'] != offset or block['anchor_token'] != response[offset]
                    or block['committed_tokens'] != response[offset + 1:offset + 1 + progress]):
                errors.append('native_trace')
                break
            offset += progress
            accepted += kept
        if (offset != len(response) - 1 or metadata['committed_progress'] != offset
                or metadata['accepted_draft_tokens'] != accepted
                or metadata['verification_calls'] != len(metadata['trace'])):
            errors.append('native_count_accounting')
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append('unavailable_or_malformed_native_record: ' + str(error))
    return {**row, 'event': event, 'local_record_path': str(path),
            'analysis_valid': not errors, 'analysis_errors': sorted(set(errors))}


def paired_comparison(rows, candidate, reference, timed):
    reference_calls = sum(row[reference + '_verification_calls'] for row in rows)
    candidate_calls = sum(row[candidate + '_verification_calls'] for row in rows)
    differences = [row[reference + '_verification_calls'] - row[candidate + '_verification_calls'] for row in rows]
    result = {'call_reduction_pct': reduction(reference_calls, candidate_calls),
              'call_request_directions': {sign: sum(direction(row[reference + '_verification_calls'],
                  row[candidate + '_verification_calls']) == sign for row in rows)
                  for sign in ('improved', 'regressed', 'tied')},
              'call_gross_saved': sum(max(value, 0) for value in differences),
              'call_gross_extra': sum(max(-value, 0) for value in differences),
              'call_net_saved': sum(differences)}
    if timed:
        before = sum(row[reference + '_mean_request_latency_s'] for row in rows)
        after = sum(row[candidate + '_mean_request_latency_s'] for row in rows)
        result['request_time_reduction_pct'] = reduction(before, after)
        result['request_time_net_saved_s'] = before - after
        result['time_scope'] = 'Contemporaneous O/K measurements within this event only.'
    return result


def analyze_event(directory, event, requests, frozen_counts, issues, expected_checkpoints=None):
    directory = Path(directory)
    measurements = directory / 'MEASUREMENTS.jsonl'
    status = load(directory / 'STATUS.json') if (directory / 'STATUS.json').is_file() else None
    protocol = load(directory / 'PROTOCOL.json') if (directory / 'PROTOCOL.json').is_file() else {}
    checkpoints = protocol.get('checkpoints', {})
    if expected_checkpoints is not None and (measurements.exists() or status is not None):
        check(set(expected_checkpoints) == set(ARMS)
              and all(checkpoints.get(arm) == expected_checkpoints[arm] for arm in ARMS),
              issues, f'event_{event}', 'Evaluation checkpoints differ from this event training receipts')
    returned = lines(measurements) if measurements.is_file() else []
    measured = [row for row in returned if row.get('method') in ARMS]
    if event != 1:
        check(len(measured) == len(returned), issues, f'event_{event}', 'Unexpected measured arm')
    known = {request['request_id'] for request in requests}
    check(all(row.get('request_id') in known for row in measured), issues, f'event_{event}', 'Unknown request identity')
    checked_rows, per_request, repeat_totals = [], [], []
    for request in requests:
        selected = [validate_row(row, request, directory, event, checkpoints)
                    for row in measured if row['request_id'] == request['request_id']]
        checked_rows.extend(selected)
        expected = {(arm, repeat) for arm in ARMS for repeat in range(3)}
        complete = (len(selected) == 6
                    and all(row.get('method') in ARMS and type(row.get('repeat')) is int for row in selected)
                    and {(row['method'], row['repeat']) for row in selected} == expected)
        errors = {error for row in selected for error in row['analysis_errors']}
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        for arm in ARMS:
            own = [row for row in selected if row['method'] == arm]
            if all(type(row.get(key)) is int for row in own for key in COUNTS):
                if len({tuple(row[key] for key in COUNTS) for row in own}) > 1:
                    errors.add(arm + '_repeat_count_mismatch')
        result = {'event': event, 'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'complete': complete, 'valid': complete and not errors, 'errors': sorted(errors)}
        if result['valid']:
            result.update(F_verification_calls=frozen_counts[request['request_id']],
                          output_tokens=len(request['response_ids']), finish_reason=request['finish_reason'])
            for arm in ARMS:
                own = [row for row in selected if row['method'] == arm]
                primary = next(row for row in own if row['repeat'] == 0)
                for key in COUNTS:
                    result[arm + '_' + key] = primary[key]
                result[arm + '_mean_request_latency_s'] = sum(row['request_latency_s'] for row in own) / 3
            for candidate, reference in (('K', 'O'), ('O', 'F'), ('K', 'F')):
                result[candidate + '_vs_' + reference + '_call_direction'] = direction(
                    result[reference + '_verification_calls'], result[candidate + '_verification_calls'])
        per_request.append(result)
    valid = [row for row in per_request if row['valid']]
    valid_ids = {row['request_id'] for row in valid}
    for repeat in range(3):
        totals = {'event': event, 'repeat': repeat, 'valid_requests': len(valid)}
        for arm in ARMS:
            selected = [row for row in checked_rows if row['request_id'] in valid_ids
                        and row['method'] == arm and row['repeat'] == repeat]
            totals[arm + '_request_latency_s'] = sum(row['request_latency_s'] for row in selected)
            totals[arm + '_verification_calls'] = sum(row['verification_calls'] for row in selected)
        totals['K_vs_O_request_time_reduction_pct'] = reduction(totals['O_request_latency_s'], totals['K_request_latency_s'])
        repeat_totals.append(totals)
    summary = {'event': event, 'measurement_status': 'missing' if not measurements.exists() else
               'complete_valid' if len(valid) == len(requests) else 'partial_or_invalid',
               'execution_status': status.get('status') if status else 'unavailable',
               'planned_requests': len(requests), 'valid_requests': len(valid),
               'invalid_or_incomplete_requests': len(requests) - len(valid),
               'returned_O_K_measurements': len(measured), 'expected_O_K_measurements': len(requests) * 6,
               'native_rows_with_errors': sum(not row['analysis_valid'] for row in checked_rows),
               'F_verification_calls': sum(row['F_verification_calls'] for row in valid) if valid else None,
               'costs': {key: status.get(key) for key in ('activation_s', 'weight_validation_s', 'evaluation_wall_s')} if status else {},
               'execution_accounting': status}
    for arm in ARMS:
        totals = {key: sum(row[arm + '_' + key] for row in valid) if valid else None for key in COUNTS}
        totals['summed_mean_request_latency_s'] = sum(row[arm + '_mean_request_latency_s'] for row in valid) if valid else None
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        summary[arm] = totals
    summary['comparisons'] = {candidate + '_vs_' + reference: paired_comparison(valid, candidate, reference, reference == 'O')
                              for candidate, reference in (('K', 'O'), ('O', 'F'), ('K', 'F'))}
    lengths = [row['output_tokens'] for row in valid]
    summary['output_lengths'] = {'minimum': min(lengths) if lengths else None, 'maximum': max(lengths) if lengths else None,
        'median': median(lengths) if lengths else None, 'total': sum(lengths),
        'eos_requests': sum(row['finish_reason'] == 'eos' for row in valid),
        'cap_hits': sum(length == 2048 for length in lengths)}
    if status and status.get('status') == 'completed':
        check(summary['measurement_status'] == 'complete_valid', issues, f'event_{event}', 'Completed evaluator lacks full valid cohort')
    if event != 1 and status:
        check(status.get('returned_generations') == len(returned), issues, f'event_{event}', 'Returned count differs from durable measurements')
    if event != 1 and (directory / 'FROZEN_COUNTS.json').is_file():
        check(load(directory / 'FROZEN_COUNTS.json') == frozen_counts, issues, f'event_{event}', 'Historical F count mapping differs')
    return summary, per_request, repeat_totals, checked_rows


def training_summary(path, event, arm, issues):
    """Summarize each receipt once; additive metrics use optimizer boundaries."""
    path = Path(path)
    if not path.is_file():
        return {'event': event, 'arm': arm, 'status': 'unavailable', 'source': str(path)}, None
    receipt = load(path)
    scope = f'training_{event}_{arm}'
    records, features = receipt['records'], receipt['features']
    by_id = {feature['record_id']: feature for feature in features}
    anchors, slots, occurrences, weighted = 0, 0, 0, 0.
    schedule = []
    for index, row in enumerate(records):
        check(row['optimizer_stepped'] == (index % 2 == 1), issues, scope, 'Optimizer boundaries differ')
        check(row['optimizer_step_after'] == (event - 1) * 128 + (index + 1) // 2, issues, scope, 'Cumulative optimizer counter differs')
        feature = by_id[row['record_id']]
        schedule.append({key: row[key] for key in ('record_id', 'pass', 'sampled_anchors')})
        for draw in row['sampled_anchors']:
            for positions, keep in zip(draw['positions'], draw['keep_mask']):
                slots += len(positions)
                check(len(positions) == len(keep), issues, scope, 'Anchor mask length differs')
                for anchor, selected in zip(positions, keep):
                    if selected:
                        anchors += 1
                        offsets = [offset for offset in range(1, 16)
                                   if feature['prompt_length'] <= anchor + offset < feature['length']]
                        occurrences += len(offsets)
                        weighted += sum(math.exp(-(offset - 1) / 7.) for offset in offsets)
    boundaries = [row for row in records if row['optimizer_stepped']]
    metrics = {key: sum(row['metrics'].get(key, 0) for row in boundaries)
               for key in ('response_presentations', 'supervised_position_exposure')}
    if arm == 'K':
        names = ('available_position_occurrences', 'selected_position_occurrences', 'available_position_weight',
                 'selected_position_weight', 'main_loss_numerator', 'loss_denominator', 'no_error_blocks', 'empty_blocks') + HISTOGRAM
        metrics.update({key: sum(row['metrics'][key] for row in boundaries) for key in names})
        check(receipt.get('selection_totals_source') == 'optimizer_boundary_accumulation_windows_once', issues, scope, 'Selection totals scope differs')
        for key, value in metrics.items():
            check(receipt['selection_totals'].get(key) == value, issues, scope, 'Selection total differs: ' + key)
        selected_occurrences = metrics['selected_position_occurrences']
        selected_weight = metrics['selected_position_weight']
        check(metrics['available_position_occurrences'] == occurrences, issues, scope, 'Available occurrence count differs')
        check(math.isclose(metrics['available_position_weight'], weighted, rel_tol=1.3e-6, abs_tol=1e-5), issues, scope, 'Available weighted exposure differs')
    else:
        selected_occurrences, selected_weight = occurrences, weighted
    check(metrics['supervised_position_exposure'] == selected_occurrences, issues, scope, 'Supervised exposure differs')
    check(metrics['response_presentations'] == len(records), issues, scope, 'Accumulation rows double counted or omitted')
    check(0 <= selected_occurrences <= occurrences and 0 <= selected_weight <= weighted * (1 + 1.3e-6),
          issues, scope, 'Selection outside available exposure')
    if receipt.get('status') == 'completed':
        check(len(records) == 256 and len(features) == len(by_id) == 128 and len(boundaries) == 128,
              issues, scope, 'Completed event budget differs')
        expected = dict(event_id=f'event_{event:03d}', objective='osd' if arm == 'O' else 'osd_first_error',
                        passes=2, response_presentations=256, start_step=(event - 1) * 128,
                        end_step=event * 128, optimizer_steps=128)
        check(all(receipt.get(key) == value for key, value in expected.items()),
              issues, scope, 'Completed event identity/objective/counters differ')
    result = {'event': event, 'arm': arm, 'status': receipt.get('status'), 'source': str(path),
              'start_step': receipt['start_step'], 'end_step': receipt['end_step'],
              'optimizer_steps': receipt['optimizer_steps'], 'response_presentations': len(records),
              'sampled_block_slots': slots, 'kept_anchors': anchors,
              'available_occurrences': occurrences, 'selected_occurrences': selected_occurrences,
              'selected_occurrence_fraction': selected_occurrences / occurrences if occurrences else None,
              'available_weight': weighted, 'selected_weight': selected_weight,
              'selected_weight_fraction': selected_weight / weighted if weighted else None,
              'boundary_metrics': metrics, 'costs': receipt.get('costs'),
              'objective_work': receipt.get('objective_work'), 'source_identity': fingerprint(path)}
    return result, {'features': features, 'schedule': schedule}


def work_accounting(campaign, status, event_summaries, training, issues):
    """Use receipts for observed work; preserve unknown interrupted counters."""
    counters = {'O': 128, 'K': 128}
    presentations = {'O': 256, 'K': 256}
    unknown = []
    for event in (2, 3, 4):
        for arm in ARMS:
            folder = campaign / f'resume_{event:03d}_{arm}'
            closed, failure = folder / 'CLOSED.json', folder / 'FAILED.json'
            saved = load(closed) if closed.is_file() else load(failure).get('status') if failure.is_file() else None
            if saved is not None:
                counters[arm] = saved['global_step']
                presentations[arm] = saved['presentations']
            elif folder.exists():
                unknown.append({'event': event, 'arm': arm})
    # A returned counter cannot establish whether an interrupted optimizer call
    # changed tensors before TrainerCore updated its reported counters.
    if status.get('status') == 'failed' and status.get('phase', '').startswith('training_'):
        arm = status['phase'].removeprefix('training_')
        event = status.get('event')
        if arm in ARMS and event in (2, 3, 4) and not any(row == {'event': event, 'arm': arm} for row in unknown):
            unknown.append({'event': event, 'arm': arm})
    observed_feedback = sum(len(list((campaign / f'event_{event:03d}' / 'responses').glob('*.json'))) for event in (2, 3, 4))
    evaluations = {str(event): event_summaries[str(event)]['execution_accounting'] for event in (2, 4)}
    result = {'planned': {'feedback_generations': 384, 'evaluation_generations': 768,
                          'optimizer_steps': 768, 'presentations': 1536},
              'campaign_status': status, 'durable_feedback_response_files': observed_feedback,
              'returned_evaluation_rows': sum(event_summaries[str(event)]['returned_O_K_measurements'] for event in (2, 4)),
              'observed_arm_steps': counters, 'unknown_training_counters': unknown,
              'reported_optimizer_steps': sum(counters.values()) - 256,
              'reported_presentations': sum(presentations.values()) - 512,
              'optimizer_steps_exact': None if unknown else sum(counters.values()) - 256,
              'presentations_exact': None if unknown else sum(presentations.values()) - 512,
              'completed_training_receipts': sum(row['status'] == 'completed' for row in training if row['event'] > 1),
              'evaluation_statuses': evaluations,
              'scope': 'Compact receipt accounting; full checkpoint/RNG equality and worker-wide work counts require the independent execution audit.'}
    if status.get('status') == 'completed':
        check(all(value and value.get('status') == 'completed' for value in evaluations.values()),
              issues, 'campaign', 'Completed campaign contains incomplete/failed evaluator status')
        for key, expected in (('optimizer_steps', 768), ('presentations', 1536),
                              ('feedback_returned', 384), ('evaluation_returned', 768)):
            check(status.get(key) == expected, issues, 'campaign', 'Completed reported budget differs: ' + key)
        check(result['optimizer_steps_exact'] == 768 and result['presentations_exact'] == 1536
              and observed_feedback == 384 and result['returned_evaluation_rows'] == 768
              and result['completed_training_receipts'] == 6,
              issues, 'campaign', 'Completed observed receipt budget differs')
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def fallacy_scan(events, repeats):
    """Bound interpretation of these descriptive, repeated-panel measurements."""
    disagreements = []
    for event in EVENTS:
        pooled = events[str(event)]['comparisons']['K_vs_O']['request_time_reduction_pct']
        values = [row['K_vs_O_request_time_reduction_pct'] for row in repeats if row['event'] == event]
        if pooled is not None and any(value is not None and value * pooled < 0 for value in values):
            disagreements.append(event)
    return [
        {'type': "Simpson's paradox", 'assessment': 'Events are kept separate; paired request and timing-repeat results are retained.',
         'events_with_timing_repeat_direction_disagreement': disagreements},
        {'type': 'Ecological fallacy', 'assessment': 'Aggregate savings do not imply that every request improves; request directions are reported.'},
        {'type': "Berkson's paradox", 'assessment': 'Results are conditional on the fixed development panel, not a random population estimate.'},
        {'type': 'Collider bias', 'assessment': 'No performance-based selection or covariate adjustment; correctness exclusions and incomplete cohorts remain visible.'},
        {'type': 'Base rate neglect', 'assessment': 'Call totals, output tokens, valid-request denominators and per-request directions accompany percentages.'},
        {'type': 'Regression to the mean', 'assessment': 'Both retained arms continue on matched chronological feedback; this single trajectory does not establish robustness.'},
        {'type': 'Survivorship bias', 'assessment': 'All planned requests and missing events are retained; gains use an explicitly counted complete valid paired cohort.'},
        {'type': 'Look-elsewhere effect', 'assessment': 'All prespecified events 1/2/4 and K/O, O/F, K/F call comparisons are reported; no significance test is claimed.'},
        {'type': 'Garden of forking paths', 'assessment': 'Fixed repeat-0 counts and three-repeat means use pooled ratios without outlier removal or outcome-dependent reruns.'},
        {'type': 'Correlation versus causation', 'assessment': 'Mask and selected-weight renormalization are jointly tested; serving measurements do not establish adaptation-inclusive deployment acceleration.'},
        {'type': 'Reverse causality', 'assessment': 'Updates precede evaluation, but this ordering does not identify which internal mechanism caused an observed difference.'},
    ]


def analyze(campaign, event_one, ordinary_training, output):
    campaign, event_one, output = Path(campaign), Path(event_one), Path(output)
    if not campaign.is_dir() or not (campaign / 'STATUS.json').is_file():
        raise FileNotFoundError('Retrieved continuation campaign/STATUS.json is required; no experiment is started')
    manifest_path = event_one / 'MANIFEST.json'
    requests = load(manifest_path)['eval_requests']
    if len(requests) != 64 or [row['panel_ordinal'] for row in requests] != list(range(64)):
        raise ValueError('Require the unchanged original 64-request panel')
    frozen_counts = {row['request_id']: row['reference_calls']['F'] for row in requests}
    issues, event_summaries, per_request, repeats, all_rows, training, schedules = [], {}, [], [], [], [], {}
    plan = bind_panel(campaign, requests, issues)
    for event in EVENTS:
        directory = event_one / 'evaluation' if event == 1 else campaign / f'evaluation_{event:03d}'
        expected_checkpoints = {}
        for arm in ARMS:
            receipt_path = (Path(ordinary_training) if arm == 'O' else event_one / 'K/events/event_001/RESULT.json') if event == 1 else campaign / arm / 'events' / f'event_{event:03d}' / 'RESULT.json'
            if receipt_path.is_file():
                receipt = load(receipt_path)
                expected_checkpoints[arm] = receipt.get('export_path')
                if event == 1 and plan is not None:
                    check(receipt.get('export_path') == plan.get('initial', {}).get(arm, {}).get('export'),
                          issues, 'provenance', 'Event1 export differs from sealed parent: ' + arm)
        summary, requests_out, repeats_out, rows_out = analyze_event(
            directory, event, requests, frozen_counts, issues, expected_checkpoints=expected_checkpoints)
        event_summaries[str(event)] = summary
        per_request.extend(requests_out)
        repeats.extend(repeats_out)
        all_rows.extend(rows_out)
    for event in (1, 2, 3, 4):
        for arm in ARMS:
            path = (Path(ordinary_training) if arm == 'O' else event_one / 'K/events/event_001/RESULT.json') if event == 1 else campaign / arm / 'events' / f'event_{event:03d}' / 'RESULT.json'
            result, schedule = training_summary(path, event, arm, issues)
            training.append(result)
            schedules[(event, arm)] = schedule
        ordinary, selected = schedules[(event, 'O')], schedules[(event, 'K')]
        if ordinary is not None and selected is not None:
            check(ordinary == selected, issues, f'training_{event}', 'O/K features or normal presentation/anchor draws differ')
    status = load(campaign / 'STATUS.json')
    costs = load(campaign / 'COSTS.json') if (campaign / 'COSTS.json').is_file() else None
    work = work_accounting(campaign, status, event_summaries, training, issues)
    complete = (status.get('status') == 'completed'
                and all(item['measurement_status'] == 'complete_valid' and item['execution_status'] == 'completed'
                        for item in event_summaries.values()) and not issues)
    report = {'verification_status': 'ANALYZED', 'status': 'complete' if complete else 'incomplete_or_issues',
              'events': event_summaries, 'training': training, 'work_accounting': work,
              'campaign_costs': costs, 'issues': issues,
              'source_manifest': fingerprint(manifest_path),
              'count_scope': 'One repeat-0 trajectory per arm/request after all three repeats pass; F counts use exactly the same valid cohort.',
              'timing_scope': 'Only O/K within each event are compared. No F latency or cross-event latency change is computed.',
              'cost_scope': 'Campaign wall includes nested loading, collection, preparation, training, persistence, evaluation and publication timers. Do not add nested costs. Shared feedback is collected once, not charged twice.',
              'limits': ['One seed and one matched shared-feedback trajectory; no seed/window robustness claim.',
                         'Original development panel is reused; timing repeats are repeated measurements, not new requests.',
                         'Mask and selected-weight renormalization remain jointly tested.',
                         'Available-checkpoint serving gains exclude adaptation and do not establish net deployment acceleration.',
                         'This continuation tests the retained O/K states; exact resume validation does not certify equivalence to an uninterrupted GPU execution.']}
    report['fallacy_scan'] = fallacy_scan(event_summaries, repeats)
    report['fallacy_scan_coverage'] = '11/11 checked for descriptive interpretation; this is not a general validity certificate.'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'ANALYSIS.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_csv(output / 'PER_REQUEST.csv', per_request)
    write_csv(output / 'REPEATS.csv', repeats)
    write_csv(output / 'ALL_REPEATS.csv', all_rows)
    write_csv(output / 'TRAINING.csv', training)
    event_table = []
    for event, row in event_summaries.items():
        item = {'event': int(event), 'valid_requests': row['valid_requests'], 'status': row['measurement_status'],
                'F_calls': row['F_verification_calls'], 'O_calls': row['O']['verification_calls'], 'K_calls': row['K']['verification_calls']}
        for label, values in row['comparisons'].items():
            item[label + '_call_reduction_pct'] = values['call_reduction_pct']
            item[label + '_call_directions'] = values['call_request_directions']
        item['K_vs_O_same_event_time_reduction_pct'] = row['comparisons']['K_vs_O']['request_time_reduction_pct']
        event_table.append(item)
    write_csv(output / 'EVENTS.csv', event_table)
    text = ['## Material Passport', '', 'ARS experiment-agent validate; local compact-evidence analysis. Verification Status: **ANALYZED**.', '',
            f"Analysis status: **{report['status']}**; campaign status: **{status.get('status', 'unavailable')}**; issues: **{len(issues)}**.", '',
            '| Event | Valid requests | F calls (historical) | O calls | K calls | K/O calls | O/F calls | K/F calls | Same-event K/O time |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def shown(value):
        return 'unavailable' if value is None else f'{value:+.4f}%'
    for item in event_table:
        calls = [str(item[key]) if item[key] is not None else 'unavailable' for key in ('F_calls', 'O_calls', 'K_calls')]
        ratios = [shown(item[key]) for key in ('K_vs_O_call_reduction_pct', 'O_vs_F_call_reduction_pct', 'K_vs_F_call_reduction_pct', 'K_vs_O_same_event_time_reduction_pct')]
        text.append('| ' + ' | '.join([str(item['event']), str(item['valid_requests']) + '/64', *calls, *ratios]) + ' |')
    text.extend(['', report['count_scope'], '', report['timing_scope'], '', report['cost_scope'], '',
                 'Incomplete or invalid requests remain in PER_REQUEST.csv. Any partial cohort is explicitly counted; missing events are unavailable.', '',
                 *['- ' + limitation for limitation in report['limits']], '',
                 'Detailed outcomes, repeated measurements, training exposure/costs and work accounting are in the accompanying JSON and CSV files. '
                 'This analysis does not rerun a scientific experiment or certify tensor resume/worker accounting; see the independent final execution audit.'])
    text.extend(['', report['fallacy_scan_coverage'], ''])
    text.extend('- ' + row['type'] + ': ' + row['assessment'] for row in report['fallacy_scan'])
    (output / 'ANALYSIS.md').write_text('\n'.join(text) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--event-one', type=Path,
                        default=RESEARCH / 'dflash_first_error_execution_20261026/raw/campaign_001')
    parser.add_argument('--ordinary-training', type=Path, default=RESEARCH / 'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.campaign, args.event_one, args.ordinary_training, args.output)
    print(json.dumps({'status': report['status'], 'issues': len(report['issues']),
                      'campaign_status': report['work_accounting']['campaign_status'].get('status'),
                      'output': str(args.output)}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
