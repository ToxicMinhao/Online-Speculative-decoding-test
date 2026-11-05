"""Independent, standard-library-only audit of retrieved U/P campaign evidence.

Reads compact JSON artifacts only. It never loads weights, imports Torch, reaches
the server, trains a model or modifies the running campaign. Remote record paths
are resolved by basename inside the retrieved evaluation directory.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ARMS = ('F', 'O', 'U', 'P')
PAIRS = (('P', 'O'), ('P', 'U'), ('P', 'F'), ('O', 'F'))
WORK = {'current_training_forward': 10, 'reference_training_forward': 4,
        'detached_comparison_forward': 4, 'temporary_optimizer_step': 5}


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).is_file() else []


def check(condition, issues, message):
    if not condition:
        issues.append(message)


def numeric_equal(actual, expected):
    return (actual is None and expected is None) or (
        isinstance(actual, (int, float)) and isinstance(expected, (int, float))
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9))


def qualification(root, ordinary, issues):
    result = read_json(root / 'qualification/RESULT.json')
    work = read_lines(root / 'qualification/WORK.jsonl')
    if result is None:
        return {'status': 'unavailable', 'completed': False, 'work_records': len(work)}
    passed = result['status'] == 'passed'
    counters = {}
    for kind, planned in WORK.items():
        attempted = sum(row['kind'] == kind and row.get('status') == 'started' for row in work)
        completed = sum(row['kind'] == kind and row.get('status') == 'completed' for row in work)
        failed = sum(row['kind'] == kind and row.get('status') == 'failed' for row in work)
        counters[kind] = dict(planned=planned, attempted=attempted, completed=completed,
                             explicit_failed=failed, incomplete_attempts=attempted - completed,
                             unattempted=max(planned - attempted, 0))
        check(0 <= completed <= attempted <= planned, issues, 'Qualification work exceeds budget: ' + kind)
        for suffix, count in (('_attempted', attempted), ('_completed', completed)):
            check(result.get('work_counts', {}).get(kind + suffix, 0) == count, issues,
                  'Qualification work counter mismatch: ' + kind + suffix)
        if passed:
            check(completed == planned, issues, 'Passed qualification has incomplete work: ' + kind)
    presentations = [row for row in work if row['kind'] == 'presentation_complete']
    check(len(presentations) == result.get('work_counts', {}).get('response_presentations', 0),
          issues, 'Qualification presentation counter mismatch')
    trials = result['trials']
    comparisons = []
    baseline = next((trial for trial in trials if trial['label'] == 'O'), None)
    for trial in trials:
        records = trial.get('records', [])
        for index, row in enumerate(records):
            expected = ordinary['records'][index]
            for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'):
                check(row[key] == expected[key], issues, f"Qualification {trial['label']} row {index}: {key}")
        reference = baseline.get('records', []) if baseline is not None else []
        changes = []
        for index, (row, previous) in enumerate(zip(records, reference)):
            for metric in sorted(set(row['metrics']) | set(previous['metrics'])):
                actual, expected = row['metrics'].get(metric), previous['metrics'].get(metric)
                if actual != expected:
                    changes.append({'presentation': index, 'metric': metric, 'actual': actual,
                                    'ordinary': expected,
                                    'difference': actual - expected if isinstance(actual, (int, float))
                                    and isinstance(expected, (int, float)) else None})
        comparisons.append({'trial': trial['label'], 'status': trial['status'], 'presentations': len(records),
                            'optimizer_steps_from_records': sum(row['optimizer_stepped'] for row in records),
                            'retained_first_window_schedule_equal': all(
                                all(row[key] == original[key] for key in
                                    ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after'))
                                for row, original in zip(records, ordinary['records'][:2])),
                            'recorded_metrics_equal_O': not changes and len(records) == len(reference) == 2,
                            'metric_differences_from_O': changes,
                            'reported_exact_compatibility': trial.get('exact_baseline_compatibility'),
                            'peak_allocated_bytes': trial.get('peak_cuda_allocated_bytes'),
                            'peak_reserved_bytes': trial.get('peak_cuda_reserved_bytes')})
    if baseline is not None:
        check(all(row['metrics'] == expected['metrics'] for row, expected in
                  zip(baseline.get('records', []), ordinary['records'][:2])), issues,
              'Qualification O metrics differ from original first window')
    if passed:
        check([trial['label'] for trial in trials] == ['O', 'U0', 'P0', 'U1', 'P1'], issues,
              'Passed qualification does not contain all five ordered trials')
        check(all(trial['status'] == 'passed' for trial in trials), issues, 'Qualification trial not passed')
        check(result.get('temporary_states_discarded') is True, issues, 'Temporary states not marked discarded')
    return {'status': result['status'], 'completed': passed, 'work': counters,
            'completed_top_level_draft_forwards': sum(counters[k]['completed'] for k in WORK if k != 'temporary_optimizer_step'),
            'planned_top_level_draft_forwards': 18, 'presentation_count': len(presentations),
            'recorded_internal_work': {key: value for key, value in result.get('work_counts', {}).items()
                                       if key in ('current_layer_forwards_including_recompute',
                                                  'reference_layer_forwards',
                                                  'vocabulary_head_forwards_including_recompute')},
            'trials': comparisons, 'failure': result.get('error'), 'wall_s': result.get('wall_s'),
            'state_tensor_difference_magnitude': 'Unavailable from compact receipts; no tensor-state comparison is performed here.',
            'interpretation': 'A failed correctness qualification supplies no efficacy evidence; do not infer the numerical cause from the gate message alone.'}


def train_receipt(root, arm, ordinary, issues):
    receipt = read_json(root / arm / 'events/event_001/RESULT.json')
    if receipt is None:
        failure = read_json(root / arm / 'events/event_001/FAILED.json')
        pending = read_json(root / arm / 'events/event_001/PENDING.json')
        return {'status': 'failed' if failure else 'pending_without_receipt' if pending else 'unavailable',
                'completed': False, 'failure': failure, 'pending': pending}
    expected = {'status': 'completed', 'passes': 2, 'response_presentations': 256,
                'optimizer_steps': 128, 'start_step': 0, 'end_step': 128,
                'objective': {'U': 'osd_uniform', 'P': 'osd_prefix'}[arm]}
    for key, value in expected.items():
        check(receipt.get(key) == value, issues, f'{arm} receipt {key} mismatch')
    check(receipt['features'] == ordinary['features'], issues, arm + ' feature pool differs')
    check(len(receipt['records']) == 256, issues, arm + ' presentation rows differ from 256')
    features = {row['record_id']: row for row in receipt['features']}
    anchors, direct_positions = 0, 0
    for index, (row, original) in enumerate(zip(receipt['records'], ordinary['records'])):
        for key in ('record_id', 'pass', 'sampled_anchors', 'optimizer_stepped', 'optimizer_step_after', 'learning_rate'):
            check(row.get(key) == original.get(key), issues, f'{arm} presentation {index}: {key} differs')
        feature = features[row['record_id']]
        for draw in row['sampled_anchors']:
            for positions, keep in zip(draw['positions'], draw['keep_mask']):
                check(len(positions) == len(keep), issues, f'{arm} anchor/keep geometry differs at {index}')
                for anchor, active in zip(positions, keep):
                    if active:
                        anchors += 1
                        check(feature['prompt_length'] <= anchor < feature['length'] - 1, issues,
                              f'{arm} ineligible anchor at presentation {index}')
                        direct_positions += sum(feature['prompt_length'] <= anchor + j < feature['length'] for j in range(1, 16))
    boundary = [row for row in receipt['records'] if row['optimizer_stepped']]
    metric_names = ('supervised_position_exposure', 'reference_prefix_positions', 'reference_prefix_weight',
                    'ranking_coefficient_weight')
    exposure = {name: sum(row['metrics'].get(name, 0) for row in boundary) for name in metric_names}
    check(len(boundary) == 128, issues, arm + ' optimizer boundaries differ from 128')
    check(exposure['supervised_position_exposure'] == direct_positions, issues, arm + ' exposure reconstruction differs')
    check(math.isclose(exposure['reference_prefix_weight'], exposure['ranking_coefficient_weight'],
                       rel_tol=1.3e-6, abs_tol=1e-5), issues, arm + ' auxiliary coefficient mass differs from prefix mass')
    return {'status': receipt['status'], 'completed': receipt['status'] == 'completed',
            'response_presentations': len(receipt['records']), 'optimizer_steps': len(boundary),
            'anchors': anchors, 'reconstructed_supervised_positions': direct_positions,
            'optimizer_boundary_exposure': exposure, 'costs': receipt['costs'],
            'objective_work': receipt['objective_work']}


def trace_record(record, expected, checkpoint, issues, label):
    """Reconstruct greedy commits directly, independently of the runner validator."""
    before = len(issues)
    for key in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
        check(record[key] == expected[key], issues, label + ': canonical ' + key)
    check(record['checkpoint_id'] == checkpoint, issues, label + ': checkpoint identity')
    metadata = record['metadata']
    trace, response = metadata['trace'], record['response_ids']
    capture = metadata['capture']
    check(record['purpose'] == 'evaluation' and metadata['feedback_capture_in_timed_request'] is False,
          issues, label + ': capture/purpose')
    check(capture['collect_features'] is False and capture['feature_bytes'] == 0
          and capture['native_capture_validated'] is True, issues, label + ': capture fields')
    check(metadata['block_size'] == 16 and metadata['context_limit'] == 4096
          and metadata['runtime'] == 'pinned_SGLang_FA3', issues, label + ': runtime')
    check(metadata['cache_policy'] == 'idle_full_flush_before_every_request_and_publication', issues, label + ': cache policy')
    check(response and len(response) <= 2048, issues, label + ': response length')
    stops = [i for i, token in enumerate(response) if token in expected['stop_token_ids']]
    check((not stops and len(response) == 2048 and record['finish_reason'] == 'length') or
          (stops == [len(response) - 1] and record['finish_reason'] == 'eos'), issues, label + ': termination')
    progress, accepted = 0, 0
    for index, block in enumerate(trace):
        prefix = label + f': block {index}'
        check(block['output_start'] == progress, issues, prefix + ' start')
        check(progress < len(response) and block['anchor_token'] == response[progress], issues, prefix + ' seed')
        proposals, teacher = block['draft_tokens'], block['target_argmax']
        check(len(proposals) == 15 and len(teacher) == block['verify_size'] == 16, issues, prefix + ' geometry')
        raw = 0
        for proposal, target in zip(proposals, teacher):
            if proposal != target:
                break
            raw += 1
        check(block['raw_accepted_draft_tokens_before_stop'] == raw, issues, prefix + ' raw accepted')
        full = proposals[:raw] + [teacher[raw]]
        committed = full[:max(2048 - 1 - progress, 0)]
        for j, token in enumerate(committed):
            if token in expected['stop_token_ids']:
                committed = committed[:j + 1]
                break
        check(block['committed_tokens'] == committed, issues, prefix + ' native target commit')
        check(response[progress + 1:progress + 1 + len(committed)] == committed, issues, prefix + ' canonical commit')
        check(block['committed_progress'] == len(committed), issues, prefix + ' progress')
        check(block['accepted_draft_tokens'] == min(raw, len(committed)), issues, prefix + ' stop-clipped accepted')
        check(block['positions'] == list(range(len(record['prompt_ids']) + progress,
                                               len(record['prompt_ids']) + progress + 16)), issues, prefix + ' positions')
        progress += len(committed)
        accepted += block['accepted_draft_tokens']
    check(progress == len(response) - 1 == metadata['committed_progress'], issues, label + ': summed progress')
    check(len(trace) == metadata['verification_calls'], issues, label + ': calls')
    check(accepted == metadata['accepted_draft_tokens'], issues, label + ': accepted sum')
    for key in ('request_latency_s', 'native_generate_s'):
        check(math.isfinite(metadata[key]) and metadata[key] > 0, issues, label + ': ' + key)
    return {'valid': len(issues) == before, 'verification_calls': len(trace),
            'accepted_draft_tokens': accepted, 'committed_progress': progress,
            'output_tokens': len(response), 'request_latency_s': metadata['request_latency_s']}


def evaluation(root, requests, ordinary_measurements, issues):
    directory = root / 'evaluation'
    status = read_json(directory / 'STATUS.json')
    if status is None:
        return {'status': 'not_started_or_unavailable', 'completed': False, 'raw_measurements': 0}
    rows = read_lines(directory / 'MEASUREMENTS.jsonl')
    protocol = read_json(directory / 'PROTOCOL.json')
    summary = read_json(directory / 'report/SUMMARY.json')
    retained = {(row['request_id'], row['method']): row['verification_calls']
                for row in ordinary_measurements if row['repeat'] == 0}
    expected_slots = {(request['request_id'], method, repeat) for request in requests for method in ARMS for repeat in range(3)}
    slots = [(row['request_id'], row['method'], row['repeat']) for row in rows]
    check(len(slots) == len(set(slots)), issues, 'Duplicate evaluation slots')
    check(set(slots) <= expected_slots, issues, 'Unexpected evaluation slots')
    by_id = {request['request_id']: request for request in requests}
    checked = {}
    block_count = 0
    for row in rows:
        label = f"{row['panel_ordinal']:04d}_r{row['repeat']}_{row['method']}"
        path = directory / Path(row['record_path']).name
        record = read_json(path)
        if record is None:
            issues.append(label + ': native record missing after retrieval')
            continue
        result = trace_record(record, by_id[row['request_id']], protocol['checkpoints'][row['method']], issues, label)
        for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens', 'request_latency_s'):
            check(row[key] == result[key], issues, label + ': score differs from raw ' + key)
        if row['method'] in ('F', 'O'):
            check(result['verification_calls'] == retained[(row['request_id'], row['method'])], issues,
                  label + ': retained first-event count mismatch')
        result['valid'] &= row['valid_measurement'] is True and not row['errors']
        checked[(row['request_id'], row['method'], row['repeat'])] = result
        block_count += result['verification_calls']
    valid_requests = []
    for request in requests:
        request_id = request['request_id']
        group = [checked.get((request_id, method, repeat)) for method in ARMS for repeat in range(3)]
        if not all(row and row['valid'] for row in group):
            continue
        consistent = all(len({checked[(request_id, method, repeat)]['verification_calls'] for repeat in range(3)}) == 1
                         for method in ARMS)
        check(consistent, issues, request_id + ': repeated calls differ')
        if consistent:
            valid_requests.append(request_id)
    totals = {}
    for method in ARMS:
        totals[method] = {key: sum(checked[(request_id, method, 0)][key] for request_id in valid_requests)
                          for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')}
        totals[method]['summed_mean_request_latency_s'] = sum(
            sum(checked[(request_id, method, repeat)]['request_latency_s'] for repeat in range(3)) / 3
            for request_id in valid_requests)
    comparisons = {}
    for candidate, reference in PAIRS:
        entry = {}
        for metric, key in (('call', 'verification_calls'), ('request_time', 'summed_mean_request_latency_s')):
            denominator = totals[reference][key]
            entry[metric + '_reduction_pct'] = 100 * (1 - totals[candidate][key] / denominator) if denominator else None
        comparisons[candidate + '_vs_' + reference] = entry
    if summary is not None:
        check(summary['valid_requests'] == len(valid_requests), issues, 'Summary valid-request count differs')
        for method in ARMS:
            for key, value in totals[method].items():
                check(numeric_equal(summary[method][key], value), issues, f'Summary {method}.{key} differs')
        for pair, values in comparisons.items():
            for key, value in values.items():
                check(numeric_equal(summary['comparisons'][pair][key], value), issues, f'Summary {pair}.{key} differs')
    else:
        issues.append('Evaluation summary unavailable')
    attempted, returned = status['attempted_generations'], status['returned_generations']
    check(status['planned_generations'] == 768, issues, 'Evaluation plan differs from 768')
    check(returned == len(rows), issues, 'Returned count differs from durable score count')
    check(status['unattempted_generations'] == 768 - attempted, issues, 'Unattempted count differs')
    check(returned + status['unknown_generation_attempts'] == attempted, issues, 'Unknown-generation accounting differs')
    if status['status'] == 'completed':
        check(set(slots) == expected_slots and len(valid_requests) == 64, issues, 'Completed evaluation lacks all 768 valid slots')
    return {'status': status['status'], 'completed': status['status'] == 'completed',
            'raw_measurements': len(rows), 'native_blocks': block_count, 'valid_common_requests': len(valid_requests),
            'unattempted': status['unattempted_generations'], 'unknown_attempts': status['unknown_generation_attempts'],
            'totals': totals, 'comparisons': comparisons}


def audit(campaign, manifest, ordinary_receipt, ordinary_measurements):
    issues = []
    requests = manifest['eval_requests']
    check(len(requests) == 64, issues, 'Original evaluation panel must have 64 requests')
    status = read_json(campaign / 'STATUS.json') or {'status': 'unavailable'}
    q = qualification(campaign, ordinary_receipt, issues)
    training = {arm: train_receipt(campaign, arm, ordinary_receipt, issues) for arm in ('U', 'P')}
    e = evaluation(campaign, requests, ordinary_measurements, issues)
    if q['status'] == 'failed':
        check(not any(row['completed'] for row in training.values()) and e['raw_measurements'] == 0,
              issues, 'Main work occurred after failed qualification')
    complete = q['completed'] and all(row['completed'] for row in training.values()) and e['completed']
    check(status['status'] != 'completed' or complete, issues, 'Campaign claims completion without complete evidence')
    return {'audit_status': 'inconsistent' if issues else 'completed_and_consistent' if complete else 'incomplete_evidence_consistent',
            'campaign_status': status['status'], 'complete_campaign': complete, 'issues': issues,
            'reported_main_optimizer_steps': status.get('main_optimizer_steps'),
            'reported_main_response_presentations': status.get('main_response_presentations'),
            'reported_new_feedback_generations': status.get('new_feedback_generations'),
            'reported_new_teacher_preparation_forwards': status.get('new_teacher_preparation_forwards'),
            'qualification': q, 'training': training, 'evaluation': e,
            'limits': ['No weights, optimizer tensors or GPU state were loaded.',
                       'Remote original-O artifact hashes are checked separately by the parent.',
                       'A failed qualification is not a ranking efficacy result.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ordinary-receipt', type=Path, required=True)
    parser.add_argument('--ordinary-measurements', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measurements = args.ordinary_measurements or args.ordinary_receipt.parents[3] / 'evaluation/MEASUREMENTS.jsonl'
    result = audit(args.campaign, read_json(args.manifest), read_json(args.ordinary_receipt), read_lines(measurements))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = [f"Audit status: {result['audit_status']}", f"Campaign complete: {result['complete_campaign']}",
             f"Qualification: {result['qualification']['status']}",
             f"Evaluation rows: {result['evaluation']['raw_measurements']}", '']
    lines.extend(result['issues'] or ['No compact-artifact accounting contradiction found.'])
    if result['qualification'].get('failure'):
        lines.extend(['', 'Recorded qualification failure:', result['qualification']['failure']['message'],
                      'No efficacy conclusion; discrepancy magnitude and cause remain unestablished.'])
    args.output.with_name(args.output.stem + '_ISSUES.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'audit_status': result['audit_status'], 'complete_campaign': result['complete_campaign'],
                      'issues': len(result['issues']), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
