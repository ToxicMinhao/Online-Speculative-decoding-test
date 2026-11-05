"""Independent compact-evidence audit. No Torch, scientific execution or network."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SOURCE = RESEARCH / 'dflash_online_publication_20261028/github_checkout/DFlash-Online'
ARMS = ('O4', 'K4', 'KRC4')
COMPARISONS = (('KRC4', 'O4'), ('K4', 'O4'), ('KRC4', 'K4'))
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMERS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')

def read(path):
    return json.loads(Path(path).read_text())

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def check(ok, issues, name):
    if not ok:
        issues.append(name)

def close(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)

def compare(actual, expected, issues, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), issues, label + ': mapping required')
        if isinstance(actual, dict):
            for k, v in expected.items():
                compare(actual.get(k), v, issues, label + '.' + k)
    else:
        check(close(actual, expected) if type(expected) in (int, float) else actual == expected,
              issues, label + ': differs')

def functions_from(path, names, namespace=None):
    namespace = {} if namespace is None else namespace
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mapped(path, raw):
    """Resolve retained bytes locally; never change a saved remote identity."""
    path = str(path)
    source_prefix = '/data/sources/minhao/DFlash-Online/'
    runs_prefix = '/data/sources/minhao/runs/'
    candidates = [Path(path)]
    if path.startswith(source_prefix):
        relative = path[len(source_prefix):]
        candidates += [HERE / 'worktree' / relative, SOURCE / relative]
    elif path.startswith(runs_prefix):
        name, relative = path[len(runs_prefix):].split('/', 1)
        candidates += [RESEARCH / name / 'raw' / relative, RESEARCH / name / relative,
                       RESEARCH / name / 'final_review/raw' / relative]
        if name == HERE.name:
            candidates.insert(0, raw / relative)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def slots(requests):
    result = []
    for request in requests:
        for repeat in range(3):
            shift = (request['panel_ordinal'] + repeat) % 3
            for arm in ARMS[shift:] + ARMS[:shift]:
                result.append((request['request_id'], repeat, arm))
    return result


def pool(rows, requests):
    """Require exact repeated slots; pool counts once, timing means per request."""
    expected = {(r['request_id'], repeat, arm) for r in requests
                for repeat in range(3) for arm in ARMS}
    actual = [(r['request_id'], r['repeat'], r['method']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete, duplicate or unexpected repeat set')
    values, totals, comparisons = {}, {}, {}
    for request in requests:
        for arm in ARMS:
            selected = [r for r in rows if r['request_id'] == request['request_id']
                        and r['method'] == arm]
            if len({tuple(r[key] for key in COUNTS) for r in selected}) != 1:
                raise ValueError('Repeat-inconsistent deterministic counts')
            if any(type(r[key]) is not int or r[key] < 0 for r in selected for key in COUNTS):
                raise ValueError('Counts must be nonnegative integers')
            if any(type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] < 0
                   for r in selected for key in TIMERS):
                raise ValueError('Timers must be finite and nonnegative')
            first = next(r for r in selected if r['repeat'] == 0)
            values[request['request_id'], arm] = {
                **{key: first[key] for key in COUNTS},
                **{key: sum(r[key] for r in selected) / 3 for key in TIMERS}}
    for arm in ARMS:
        total = {key: sum(values[r['request_id'], arm][key] for r in requests) for key in COUNTS}
        for timer in TIMERS:
            total['summed_mean_' + timer] = sum(values[r['request_id'], arm][timer] for r in requests)
            total['all_repeats_' + timer] = sum(r[timer] for r in rows if r['method'] == arm)
        total['calls_per_1000_output_tokens'] = 1000 * total['verification_calls'] / total['output_tokens']
        total['accepted_draft_tokens_per_call'] = total['accepted_draft_tokens'] / total['verification_calls']
        total['committed_progress_per_call'] = total['committed_progress'] / total['verification_calls']
        totals[arm] = total
    for candidate, reference in COMPARISONS:
        comparison = dict(candidate=candidate, reference=reference)
        for label, metric, reduction in (('call', 'verification_calls', 'call_reduction_pct'),
                                         ('latency', 'request_latency_s', 'request_time_reduction_pct')):
            before = [values[r['request_id'], reference][metric] for r in requests]
            after = [values[r['request_id'], candidate][metric] for r in requests]
            differences = [old - new for old, new in zip(before, after)]
            comparison[reduction] = 100 * (1 - sum(after) / sum(before))
            comparison[label + '_request_directions'] = directions(differences)
            comparison[label + '_gross_saved'] = sum(max(0, value) for value in differences)
            comparison[label + '_gross_extra'] = sum(max(0, -value) for value in differences)
            comparison[label + '_net_saved'] = sum(differences)
        comparisons[candidate + '_vs_' + reference] = comparison
    return totals, comparisons, values


def directions(differences):
    return dict(improved=sum(value > 0 for value in differences),
                regressed=sum(value < 0 for value in differences),
                tied=sum(value == 0 for value in differences))


def count_history(values, requests, historical):
    vectors = {arm: [values[r['request_id'], arm]['verification_calls'] for r in requests]
               for arm in ARMS}
    vectors.update({arm: [counts[r['request_id']] for r in requests]
                    for arm, counts in historical.items()})
    curves = []
    for event in (1, 2, 4):
        curves.append(dict(event=event, measurement='fresh' if event == 4 else 'historical',
            **{arm: sum(vectors[arm + str(event)]) if arm + str(event) in vectors else None
               for arm in ('O', 'K', 'KRC')}))
    pairs = [(family + str(new), family + str(old)) for family in ('O', 'K', 'KRC')
             for old, new in ((1, 2), (2, 4), (1, 4))]
    pairs += [(candidate + str(event), reference + str(event)) for event in (1, 2)
              for candidate, reference in (('KRC', 'O'), ('K', 'O'), ('KRC', 'K'))]
    comparisons = {}
    for candidate, reference in pairs:
        if candidate not in vectors or reference not in vectors:
            continue
        before, after = vectors[reference], vectors[candidate]
        differences = [old - new for old, new in zip(before, after)]
        comparisons[candidate + '_vs_' + reference] = dict(
            candidate_calls=sum(after), reference_calls=sum(before),
            call_reduction_pct=100 * (1 - sum(after) / sum(before)),
            request_directions=directions(differences),
            gross_saved=sum(max(0, value) for value in differences),
            gross_extra=sum(max(0, -value) for value in differences), net_saved=sum(differences))
    changes = {}
    for candidate, reference in (('KRC', 'O'), ('KRC', 'K'), ('K', 'O')):
        current = 100 * (1 - sum(vectors[candidate + '4']) / sum(vectors[reference + '4']))
        for event in (1, 2):
            old = comparisons.get(candidate + str(event) + '_vs_' + reference + str(event))
            changes[f'{candidate}_vs_{reference}_event{event}_to4'] = (
                None if old is None else current - old['call_reduction_pct'])
    return dict(event_count_curve=curves, count_learning_curve=comparisons,
                relative_advantage_change_percentage_points=changes)


def count_map(path, method):
    return {row['request_id']: row['verification_calls'] for row in lines(path)
            if row['method'] == method and row['repeat'] == 0}


def audit_historical_inputs(panel, issues):
    name = panel['name']
    history = panel['historical_counts']
    old_root = RESEARCH / 'dflash_ok_continuation_20261027/raw/campaign_001'
    first_name = 'dflash_krc_event_20261027' if name == 'development' else 'dflash_krc_confirmation_20261028'
    first = RESEARCH / first_name / 'raw/campaign_001/evaluation/MEASUREMENTS.jsonl'
    second = RESEARCH / f'dflash_krc_continuation_20261028/raw/campaign_001/evaluation/{name}/MEASUREMENTS.jsonl'
    expected = {
        'K1': count_map(first, 'K' if name == 'development' else 'K1'),
        'KRC1': count_map(first, 'KRC'),
        'K2': count_map(second, 'K2'), 'KRC2': count_map(second, 'KRC2')}
    if name == 'development':
        # The O1 source is the original qualified first event, not the O checkpoint
        # embedded in older frozen/OSD panel preparation.
        expected['O1'] = count_map(RESEARCH /
            'dflash_osd_one_event_20261025/final_review/raw/campaign_001/evaluation/MEASUREMENTS.jsonl', 'O')
        expected['O2'] = count_map(old_root / 'evaluation_002/MEASUREMENTS.jsonl', 'O')
        gates = {arm + '4': count_map(old_root / 'evaluation_004/MEASUREMENTS.jsonl', arm)
                 for arm in ('O', 'K')}
        check(panel['expected_control_counts'] == gates, issues, 'development: exact retained O4/K4 count gates')
        check([sum(gates[arm].values()) for arm in ('O4', 'K4')] == [2994, 2942], issues,
              'development: retained control pooled counts')
    else:
        check(panel['expected_control_counts'] is None, issues, 'additional: no historical O4/K4 count gate')
    check(history == expected, issues, name + ': exact per-request historical count maps')


def native_helpers():
    module = imported(RESEARCH / 'dflash_prefix_ranking_one_event_20261025/audit_results.py',
                      'qualified_native_record_auditor')
    native = functions_from(SOURCE / 'online_feedback/native.py', ['validate_generation'])
    return module.trace_record, native['validate_generation']


def audit_publication(directory, requests, checkpoints, status, issues):
    helper = functions_from(SOURCE / 'experiments/osd_components/protocol.py',
                            ['publication_errors', 'weight_check_errors'], {'BLOCK_SIZE': 16})
    activations = lines(directory / 'ACTIVATIONS.jsonl')
    previous, expected = None, []
    by_id = {row['request_id']: row for row in requests}
    for request_id, repeat, arm in slots(requests):
        if arm != previous:
            expected.append((by_id[request_id]['panel_ordinal'], repeat, arm))
            previous = arm
    check([(row['panel_ordinal'], row['repeat'], row['method']) for row in activations] == expected,
          issues, str(directory) + ': publication order')
    for row in activations:
        check(not row['errors'] and not helper['publication_errors'](row, checkpoints[row['method']]),
              issues, 'native publication validity')
    activation_s = sum(row['activation_s'] for row in activations)
    check(close(activation_s, status['activation_s']), issues, 'publication timer sum')
    weight_s = 0
    for arm in ARMS:
        receipt = read(directory / ('WEIGHTS_' + arm + '.json'))
        check(not receipt['errors'] and not helper['weight_check_errors'](receipt['native_reply'], checkpoints[arm]),
              issues, arm + ': loaded tensor equality')
        weight_s += receipt['validation_s']
    check(close(weight_s, status['weight_validation_s']), issues, 'weight validation timer sum')
    return dict(activations=len(activations), activation_s=activation_s,
                weight_checks=3, weight_validation_s=weight_s)


def audit_panel(raw, panel, issues):
    name = panel['name']
    directory = raw / 'campaign_001/evaluation' / name
    gates = panel['expected_control_counts']
    requests = [dict(request, panel_ordinal=request.get('panel_ordinal', index),
                     expected_control_calls={} if gates is None else {
                         arm: counts[request['request_id']] for arm, counts in gates.items()})
                for index, request in enumerate(panel['requests'])]
    retained_name = 'dflash_krc_event_20261027' if name == 'development' else 'dflash_krc_confirmation_20261028'
    retained_root = RESEARCH / retained_name / 'raw/campaign_001/evaluation'
    retained_canonical = read(retained_root / 'CANONICAL.json')
    retained_rows = {row['request_id']: row for row in lines(retained_root / 'MEASUREMENTS.jsonl')
                     if row['method'] == 'KRC' and row['repeat'] == 0}
    for request in requests:
        retained = retained_canonical[request['request_id']]
        original_row = retained_rows[request['request_id']]
        original_native = read(retained_root / Path(original_row['record_path']).name)
        check(request['response_ids'] == retained['response_ids']
              and request['finish_reason'] == retained['finish_reason']
              and request['prompt_ids'] == original_native['prompt_ids'], issues,
              name + '/' + request['request_id'] + ': original canonical/input provenance')
    rows = lines(directory / 'MEASUREMENTS.jsonl')
    protocol, status = read(directory / 'PROTOCOL.json'), read(directory / 'STATUS.json')
    checkpoints = protocol['checkpoints']
    check(read(directory / 'REQUESTS.json') == requests, issues, name + ': sealed input identity')
    check(read(directory / 'EXPECTED_CONTROL_COUNTS.json') == gates, issues, name + ': control gate identity')
    check(read(directory / 'HISTORICAL_COUNTS.json') == panel['historical_counts'], issues, name + ': history identity')
    audit_historical_inputs(panel, issues)
    compare(protocol, dict(arms=list(ARMS), comparisons=[list(pair) for pair in COMPARISONS],
        primary_comparison='KRC4_vs_O4', repeats=3, rotation='(panel_ordinal + repeat) % 3',
        count_trajectory_repeat=0, historical_count_gates=['O4', 'K4'] if name == 'development' else [],
        capture=False, extra_warmup_generations=0), issues, name + ': protocol')
    expected = slots(requests)
    check([(row['request_id'], row['repeat'], row['method']) for row in rows] == expected
          and len(rows) == 576, issues, name + ': exact 576 chronological slots')
    starts = [Counter(arm for position, (_, current, arm) in enumerate(expected)
                      if current == repeat and position % 3 == 0) for repeat in range(3)]
    check(all(sorted(row.values()) == [21, 21, 22] for row in starts)
          and sum(starts, Counter()) == Counter({arm: 64 for arm in ARMS}),
          issues, name + ': exact rotation balance')
    compare(protocol['rotation_balance'], dict(starts_by_repeat=[dict(row) for row in starts],
        starts_all_repeats=dict(sum(starts, Counter())), per_request_each_arm_starts_once=True,
        per_request_each_arm_occupies_each_position_once=True), issues, name + ': declared rotation balance')
    compare(status, dict(status='completed', planned_generations=576, attempted_generations=576,
        returned_generations=576, completed_valid_generations=576, invalid_generations=0,
        pending_canonical_generations=0, unknown_generation_attempts=0, unattempted_generations=0,
        optimizer_steps=0, feedback_collections=0, automatic_retry=False, interrupted_attempt=None),
        issues, name + ': completed exact evaluation budget')
    if name == 'development':
        compare(read(directory / 'QUALIFICATION.json'), dict(status='passed', requests=8, generations=72,
            arms_with_native_equality=sorted(ARMS), performance_win_required=False, extra_generations=0),
            issues, 'included 72-generation gate')
        check(status['qualification'] == 'passed' and status['qualification_requests'] == 8,
              issues, 'development: included gate status')
    else:
        check(not (directory / 'QUALIFICATION.json').exists() and status['qualification_requests'] == 0
              and status['qualification'] == 'not_required_prior_panel_gate', issues,
              'additional: no duplicated qualification campaign')
    expected_canonical = {request['request_id']: dict(response_ids=request['response_ids'],
        finish_reason=request['finish_reason'], source_method=request['canonical_source_method'],
        source='retained_panel_canonical') for request in requests}
    check(read(directory / 'CANONICAL.json') == expected_canonical, issues, name + ': canonical bindings')
    by_id = {request['request_id']: request for request in requests}
    trace_record, validate_generation = native_helpers()
    native_paths, total_blocks, checked = set(), 0, []
    histograms = {arm: Counter() for arm in ARMS}
    for row in rows:
        request = by_id[row['request_id']]
        arm, repeat = row['method'], row['repeat']
        label = f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        record = read(directory / (label + '.json'))
        compare(row, dict(label=label, validation_status='valid', valid_measurement=True,
            structural_errors=[], errors=[]), issues, name + '/' + label)
        check(Path(row['record_path']).name == label + '.json', issues, label + ': record path')
        derived = trace_record(record, request, checkpoints[arm], issues, name + '/' + label)
        meta, capture = record['metadata'], record['metadata']['capture']
        blocks = meta['trace']
        check(record['record_id'] == label and record['request_index'] == request['panel_ordinal']
              and record['session_id'] == 'krc_events_3_4_evaluation', issues, label + ': native identity')
        check(capture['blocks'] == blocks and capture['prompt_ids'] == request['prompt_ids']
              and capture['response_ids'] == capture['target_argmax_ids'] == record['response_ids']
              and capture['covered_token_ids'] == request['prompt_ids'] + record['response_ids'][:-1]
              and capture['covered_rows'] == len(capture['covered_token_ids']), issues,
              label + ': complete native capture consistency')
        check(capture['status'] == 'passed' and capture['block_size'] == 16
              and capture['feature_persistence_s'] == 0 and not capture.get('feature_path')
              and not meta.get('feature_path'), issues, label + ': no feedback capture')
        actual, finish = validate_generation(request['prompt_ids'], record['prompt_ids'] + record['response_ids'],
            blocks, max_new_tokens=2048, stop_tokens=request['stop_token_ids'])
        check(actual == request['response_ids'] and finish == request['finish_reason'],
              issues, label + ': exact canonical generation')
        for index, block in enumerate(blocks):
            check(block['native_accepted_draft_tokens'] == max(block['committed_progress'] - 1, 0)
                  and block['stop_clipped'] == (block['committed_progress'] < block['raw_accepted_draft_tokens_before_stop'] + 1)
                  and block['finished'] == (index == len(blocks) - 1), issues,
                  label + ': exact native block counts/termination')
            if repeat == 0:
                histograms[arm][block['accepted_draft_tokens']] += 1
        for key in COUNTS[:-1] + TIMERS[:2]:
            check(close(row[key], meta[key]), issues, label + ': native measurement ' + key)
        check(row['output_tokens'] == len(actual) and row['prompt_tokens'] == len(request['prompt_ids'])
              and row['finish_reason'] == finish and row['cap_hit'] == (finish == 'length'),
              issues, label + ': output bookkeeping')
        check(close(row['cache_maintenance_s'], capture['cache_maintenance_s']), issues, label + ': cache timer')
        check(all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0 for key in TIMERS),
              issues, label + ': finite nonnegative timers')
        check(row['native_generate_s'] + row['cache_maintenance_s'] <= row['request_latency_s'] + 1e-9,
              issues, label + ': nested timers fit complete request')
        if arm in request['expected_control_calls']:
            check(derived['verification_calls'] == request['expected_control_calls'][arm], issues,
                  label + ': retained per-request development control')
        native_path = raw / 'campaign_001/native' / Path(meta['native_trace_path']).name
        check(read(native_path) == capture, issues, label + ': raw worker capture identity')
        check(native_path.name not in native_paths, issues, label + ': unique native execution')
        native_paths.add(native_path.name)
        total_blocks += len(blocks)
        compare(read(directory / (label + '_VALIDATION.json')), dict(label=label,
            request_id=request['request_id'], method=arm, repeat=repeat, status='valid', valid=True,
            errors=[], canonical_source='retained_panel_canonical',
            canonical_method=request['canonical_source_method']), issues, label + ': validation receipt')
        checked.append({**row, **{key: derived[key] for key in COUNTS}})
    complete_files = {path.name for path in directory.iterdir()
                      if re.fullmatch(r'\d{4}_r[012]_(?:O4|K4|KRC4)\.json', path.name)}
    check(complete_files == {row['label'] + '.json' for row in rows}, issues, name + ': no extra generation files')
    events = lines(directory / 'VALIDATION_EVENTS.jsonl')
    check(len(events) == 576 and all(row['status'] == 'valid' and not row['errors'] for row in events),
          issues, name + ': 576 native validation events')
    totals, comparisons, values = pool(checked, requests)
    summary = read(directory / 'report/SUMMARY.json')
    compare(summary, dict(planned_requests=64, valid_requests=64, invalid_or_incomplete_requests=0,
        returned_measurements=576, valid_timed_generations=576, primary_comparison='KRC4_vs_O4',
        timing_repeats_per_method_request=3, **totals, comparisons=comparisons), issues, name + ': reported aggregates')
    with (directory / 'report/PER_REQUEST.csv').open(newline='') as stream:
        per_request = list(csv.DictReader(stream))
    check([row['request_id'] for row in per_request] == [request['request_id'] for request in requests],
          issues, name + ': reported request order')
    for row in per_request:
        check(row['valid'] == row['complete'] == 'True' and row['errors'] == '', issues, name + ': request validity')
        for arm in ARMS:
            for key in COUNTS:
                check(int(row[arm + '_' + key]) == values[row['request_id'], arm][key],
                      issues, name + ': reported per-request ' + key)
            for key in TIMERS:
                check(close(float(row[arm + '_mean_' + key]), values[row['request_id'], arm][key]),
                      issues, name + ': reported per-request ' + key)
    repeat_totals = []
    for repeat in range(3):
        by_arm = {arm: {key: sum(row[key] for row in checked
                    if row['method'] == arm and row['repeat'] == repeat) for key in TIMERS} for arm in ARMS}
        repeat_totals.append(dict(repeat=repeat, **by_arm, comparisons={
            candidate + '_vs_' + reference: 100 * (1 - by_arm[candidate]['request_latency_s'] /
                                                       by_arm[reference]['request_latency_s'])
            for candidate, reference in COMPARISONS}))
    history = count_history(values, requests, panel['historical_counts'])
    compare(summary, history, issues, name + ': count-only history and relative gaps')
    publication = audit_publication(directory, requests, checkpoints, status, issues)
    return dict(records=len(rows), requests=len(requests), native_captures=sorted(native_paths),
        reconstructed_native_blocks=total_blocks, totals=totals, comparisons=comparisons,
        accepted_count_histograms_repeat0={arm: dict(histograms[arm]) for arm in ARMS},
        repeat_totals=repeat_totals, publication=publication, status=status, history=history)


def correction_helpers():
    return imported(RESEARCH / 'dflash_krc_event_20261027/audit.py', 'qualified_correction_record_auditor')


def audit_training_event(raw, plan, entry, issues):
    helper = correction_helpers()
    number = entry['event_number']
    start = (number - 1) * 128
    root = raw / f'campaign_001/events/event_{number:03d}'
    event = read(raw / f'campaign_001/KRC/events/event_{number:03d}/RESULT.json')
    ordinary = read(mapped(entry['ordinary_receipt'], raw))
    retained_k = read(mapped(entry['k_receipt'], raw))
    retained_pool = read(mapped(entry['pool'], raw))
    report = read(root / 'CORRECTION_EVIDENCE.json')
    schedule = read(mapped(entry['reference_schedule'], raw))
    expected = helper.expected_schedule(ordinary)
    check(schedule['windows'] == expected, issues, f'event{number}: independently rebuilt cyclic reference windows')
    check(event['features'] == ordinary['features'] == retained_k['features'] == schedule['features']
          and [feature['path'] for feature in event['features']] == retained_pool['features'],
          issues, f'event{number}: exact immutable O-generated feature pool')
    compare(event, dict(status='completed', event_id=f'event_{number:03d}', start_step=start,
        end_step=start + 128, optimizer_steps=128, response_presentations=256, passes=2,
        objective='osd_first_error', reduction='response_mean'), issues, f'event{number}: event receipt')
    feature_ids = {feature['record_id'] for feature in event['features']}
    check(len(feature_ids) == len(event['features']) == 128, issues, f'event{number}: 128 distinct responses')
    check(set(retained_pool['record_ids']) == feature_ids and len(retained_pool['identities']) == 128,
          issues, f'event{number}: pool record identities')
    evaluation = [request for panel in plan['panels'] for request in panel['requests']]
    check(not feature_ids.intersection(request['request_id'] for request in evaluation),
          issues, f'event{number}: evaluation IDs excluded')
    prompts = {tuple(request['prompt_ids']) for request in evaluation}
    response_root = RESEARCH / f'dflash_ok_continuation_20261027/raw/campaign_001/event_{number:03d}/responses'
    responses = [read(path) for path in sorted(response_root.glob('*.json'))]
    check(len(responses) == 128 and not prompts.intersection(tuple(record['prompt_ids']) for record in responses),
          issues, f'event{number}: both evaluation prompt panels excluded')
    capped = [feature for feature in event['features'] if feature['response_tokens'] == 2048]
    if number == 4:
        check(len(capped) == 1 and capped[0]['record_id'] == 'train_00399'
              and capped[0]['length'] == 2231 and capped[0]['prompt_length'] == 183,
              issues, 'event4: original capped record preserved')
    else:
        check(capped == [], issues, 'event3: no substituted cap record')
    exposure = helper.TRACE.training_exposure(event, number, 'K', issues, ordinary)
    for reference in (ordinary, retained_k):
        check(len(event['records']) == 256 and all(all(actual[key] == bound[key] for key in
            ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after', 'optimizer_stepped'))
            for actual, bound in zip(event['records'], reference['records'], strict=True)),
            issues, f'event{number}: natural ordinary RNG/order/anchor identity')
    contract = read(root / 'correction/CONTINUATION_CONTRACT.json')
    check(contract == report['diagnostic_migration'] and contract['changed_fields'] == []
          and contract['before'] == contract['after'] and contract['before']['probe_steps'] == []
          and not contract['reseeded'], issues, f'event{number}: unchanged no-probe correction contract')
    first = read(root / 'correction/FIRST_WINDOW.json')
    check(first == report['first_window'] and all(first['checks'].values()),
          issues, f'event{number}: actual first window integrity')
    compare(first, dict(status='passed', event_local_step=1, global_step_before=start,
        global_step_after_native_Adam=start + 1, optimizer_counters=[float(start + 1)],
        extra_optimizer_steps=0, extra_forwards=0), issues, f'event{number}: included first-window work')
    compare(report, dict(status='completed', start_step=start, native_global_step=start + 128, probes=[]),
            issues, f'event{number}: final correction boundary')
    compare(report['counts'], dict(K_backward_attempted=256, K_backward_completed=256, K_backward_failed=0,
        Adam_attempted=128, Adam_completed=128, Adam_failed=0, presentations_completed=256,
        windows_completed=128), issues, f'event{number}: exact ordinary work')
    compare(report['work'], dict(original_response_forwards=256, pre_reference_response_forwards=256,
        post_k_reference_response_forwards=256, post_c_reference_response_forwards=0,
        post_k_learning_response_forwards=0, post_c_learning_response_forwards=0,
        violation_backward_failed=0), issues, f'event{number}: exact 768 core forwards, no probes')
    check(0 <= report['work']['violation_backward_completed'] <= 256
          and report['work']['direct_corrections'] <= 128,
          issues, f'event{number}: bounded reference backwards/corrections')
    check([row['step'] for row in report['steps']] == list(range(1, 129)),
          issues, f'event{number}: 128 local decision windows')
    evidence = SimpleNamespace(raw=raw, issues=issues, read=read)
    eos = helper.BASE.eos_ids(evidence)
    summed_work, summed_costs, branches = Counter(), Counter(), Counter()
    reference_anchors = reference_available = reference_protected = nonzero = 0
    for summary in report['steps']:
        step = summary['step']
        window = read(root / f'correction/window_{step:03d}.json')
        check(window['summary'] == summary, issues, f'event{number}/window{step}: durable summary')
        details, binding = window['details'], expected[step - 1]
        check(details['schedule_binding'] == binding and
              not set(binding['learning_record_ids']) & set(binding['reference_record_ids']),
              issues, f'event{number}/window{step}: disjoint own A/R pair')
        compare(summary, dict(status='completed', actual_Adam_steps_completed=step,
            trainer_step_before_return=start + step - 1), issues, f'event{number}/window{step}: global accounting')
        compare(summary['integrity']['step_coordinates'], dict(event_local_step=step,
            global_step_before=start + step - 1, global_step_after_native_Adam=start + step),
            issues, f'event{number}/window{step}: exact local/global coordinates')
        for phase, checks in summary['integrity'].items():
            if phase != 'step_coordinates':
                check(checks is True or all(checks.values()), issues,
                      f'event{number}/window{step}: {phase} state preservation')
        check(all(details['reference_pre_integrity'].values()), issues,
              f'event{number}/window{step}: pre-reference state preservation')
        check(details['learning'] is None and details['post_c'] is None
              and details['reference']['post_c'] is None, issues,
              f'event{number}/window{step}: no diagnostic model work')
        reference = details['reference']
        check(reference['correction'] == details['correction'] and reference['post_k'] == details['post_k'],
              issues, f'event{number}/window{step}: only reference pair supplies correction')
        check([row['record_id'] for row in reference['responses']] == binding['reference_record_ids'],
              issues, f'event{number}/window{step}: reference identity')
        for source, bound in zip(reference['responses'], binding['reference'], strict=True):
            draw, = bound['sampled_anchors']
            check(source['anchors'] == draw['positions'] and source['keep'] == draw['keep_mask'],
                  issues, f'event{number}/window{step}: reference own recorded anchors')
            reference_anchors += sum(source['keep'][0])
            reference_available += sum(helper.BASE.valid_length(labels, validity, eos)
                for labels, validity in zip(source['labels'][0], source['validity_weights'][0]))
        totals, _ = helper.BASE.score_window(reference, step, eos, issues, collect_blocks=False)
        reference_protected += totals['K']['protected_positions']
        correction = details['correction']
        branches[correction['status']] += 1
        nonzero += correction['realized_correction_norm'] > 0
        summed_work.update(details['work'])
        summed_costs.update(summary['costs'])
        summed_costs['integrity_s'] += summary['integrity_s']
    check(dict(summed_work) == report['work'], issues, f'event{number}: window work sums exactly')
    check(all(close(value, report['costs'][key]) for key, value in summed_costs.items()),
          issues, f'event{number}: nested window cost sums')
    compare(read(root / 'TRAINING_CHECK.json'), dict(status='passed'), issues, f'event{number}: training check')
    compare(read(root / 'CORRECTION_CHECK.json'), dict(passed=True), issues, f'event{number}: correction check')
    saved = read(root / 'SAVED_STATE_CHECK.json')
    compare(saved, dict(status='passed', completed_event=number, objective='osd_first_error',
        global_step=number * 128, presentations=number * 256, horizon=7424, draft_tensors=58,
        optimizer_parameters=58, adam_step=number * 128, cuda_rng_states=1,
        scientific_extra_steps=0, export_exact=True, state_path=event['full_state_path'],
        export_path=event['checkpoint_path']), issues, f'event{number}: complete saved state/export gate')
    check(Path(event['full_state_path']).parent.name == f'event_{number:03d}'
          and event['full_state_path'] != event['latest_state_path'], issues,
          f'event{number}: complete state retained apart from mutable latest slot')
    return dict(exposure=exposure, reference_presentations=256, reference_distinct_responses=128,
        reference_phase_forwards=512, reference_anchors=reference_anchors,
        reference_available_occurrences=reference_available, reference_protected_occurrences=reference_protected,
        counts=report['counts'], work=report['work'], correction_branches=dict(branches),
        realized_nonzero_corrections=nonzero, core_response_forwards=768,
        core_backwards=256 + report['work']['violation_backward_completed'], diagnostic_forwards=0,
        contract=contract, first_window=first, event_costs=event['costs'],
        correction_costs=report['costs'], saved_state=saved,
        interpretation='Pre-correction reference statistics; final local prefix effects were not replayed.')


def audit_training(raw, plan, issues):
    root = raw / 'campaign_001'
    snapshot = read(root / 'RESUME_SNAPSHOT.json')
    compare(snapshot, dict(status='passed', completed_event=2, objective='osd_first_error',
        global_step=256, presentations=512, horizon=7424, draft_tensors=58, optimizer_parameters=58,
        adam_step=256, cuda_rng_states=1, scientific_extra_steps=0, export_exact=True,
        state_path=plan['krc2_state'], export_path=plan['krc2_checkpoint']),
        issues, 'complete corrected own KRC2 snapshot')
    old = read(mapped(plan['krc2_receipt'], raw))
    check(old['full_state_path'] == plan['krc2_state'] and old['checkpoint_path'] == plan['krc2_checkpoint'],
          issues, 'retained source receipt resolves KRC2 state and export')
    resume = read(root / 'LIVE_RESUME.json')
    compare(resume, dict(status='passed', exact_resume=True, mismatches=[], mismatch_count=0,
        global_step=256, presentations=512, no_rng_reset=True, scientific_extra_steps=0),
        issues, 'exact KRC2 live weights/Adam/RNG/config/history restoration')
    startup = read(root / 'startup/STARTUP.json')
    compare(startup, dict(status='passed', response_evaluations=2, backwards=2, optimizer_steps=0,
        exact_initial_state_restored=True, separate_disposable_runtime=True,
        runtime_closed_before_K_construction=True), issues, 'separate two-response startup, zero Adam')
    check([entry['event_number'] for entry in plan['events']] == [3, 4], issues, 'only consecutive events 3 and 4')
    events = {}
    for entry in plan['events']:
        number = entry['event_number']
        events[f'event_{number:03d}'] = audit_training_event(raw, plan, entry, issues)
        boundary = read(root / f'events/event_{number:03d}/LIVE_BOUNDARY.json')
        compare(boundary, dict(status='passed', exact_boundary=True, mismatches=[],
            global_step=number * 128, presentations=number * 256, no_rng_restore=True,
            extra_model_passes=0, extra_optimizer_steps=0), issues,
            f'event{number}: checkpoint/export preserves live next-event state')
    check(sum(event['core_response_forwards'] for event in events.values()) == 1536,
          issues, 'exact 1536 core response forwards')
    check(512 <= sum(event['core_backwards'] for event in events.values()) <= 1024,
          issues, 'bounded 1024 core backwards')
    closed = read(root / 'TRAINER_CLOSED.json')
    compare(closed, dict(closed=True, failed=False, global_step=512, presentations=1024,
        completed_events=['event_001', 'event_002', 'event_003', 'event_004'], reduction='response_mean',
        objective='osd_first_error', accumulation_remainder=0, horizon=7424),
        issues, 'closed complete KRC4 state before native timing')
    retention = read(root / 'STATE_RETENTION.json')
    check(retention['status'] == 'passed' and retention['preserved_both_complete_states'] is True
          and set(retention['before']) == set(retention['after']) == {'event_003', 'event_004'}
          and retention['before'] == retention['after'], issues,
          'both complete state/export identities preserved after event 4')
    for number in (3, 4):
        event_id = f'event_{number:03d}'
        retained = retention['before'][event_id]
        check(retained == read(root / f'events/{event_id}/STATE_IDENTITIES.json'), issues,
              event_id + ': immediate identity equals final retention basis')
        state = events[event_id]['saved_state']
        expected_paths = {state['state_path'], state['export_path'] + '/config.json',
                          state['export_path'] + '/model.safetensors'}
        check({row['path'] for row in retained} == expected_paths
              and all(type(row['bytes']) is int and row['bytes'] > 0
                      and re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in retained),
              issues, event_id + ': complete state and serving export identity membership')
    return dict(events=events, resume_snapshot=snapshot, live_resume=resume,
                startup=startup, trainer_closed=closed, state_retention=retention)


def audit_integrity(raw, plan, issues):
    seal = read(raw / 'sealed/SEAL.json')
    preflight, postflight = read(raw / 'PREFLIGHT.json'), read(raw / 'POSTFLIGHT.json')
    check(seal['sealed_before_training'], issues, 'source/dependencies sealed before training')
    for label, receipt in (('preflight', preflight), ('postflight', postflight)):
        compare(receipt, dict(status='passed', files_checked=len(seal['files'])), issues,
                label + ': all remote sealed source/input identities verified')
    storage = preflight['storage']
    check(storage['admitted'] and storage['free_bytes'] >= storage['required_bytes']
          and storage['reserve_bytes'] == 8 * 2**30
          and storage['host_available_bytes'] >= storage['host_required_bytes'] == 72 * 2**30,
          issues, 'actual peak disk and host-memory admission')
    check(storage['tensor_peak_bytes'] >= int(9 * storage['draft_bytes'] * 1.10)
          and storage['compiler_staging_logs_bytes'] >= 4 * 2**30
          and storage['required_bytes'] == storage['tensor_peak_bytes']
              + storage['compiler_staging_logs_bytes'] + storage['reserve_bytes'],
          issues, 'two complete states/exports and temporary-overlap peak budget')
    runtime = read(raw / 'sealed/RUNTIME_PIN.json')
    for receipt in (preflight, postflight):
        compare(receipt['runtime'], {key: runtime[key] for key in
            ('python_version', 'torch', 'transformers', 'cuda', 'native_runtime_sha256')},
            issues, 'qualified native runtime identity')
        compare(receipt['training_runtime'], dict(torch='2.13.0+cu130', transformers='5.12.1'),
                issues, 'qualified training runtime identity')
    local_count, remote_only = 0, []
    for name, identity in seal['files'].items():
        local = mapped(name, raw)
        if local is None:
            remote_only.append(name)
            continue
        check(local.stat().st_size == identity['bytes'] and digest(local) == identity['sha256'],
              issues, 'local sealed identity: ' + str(local))
        local_count += 1
    source = plan['execution_source']
    check(re.fullmatch('[0-9a-f]{40}', source['execution_git_commit']) is not None
          and source['dirty_files'] == [] and source['branch'] == 'codex/krc-events-3-4-20261028'
          and source['working_source'] == '/data/sources/minhao/DFlash-Online',
          issues, 'actual execution commit and clean source contract')
    check(int((raw / 'EXIT_CODE').read_text()) == 0, issues, 'supervisor exit zero')
    status, costs = read(raw / 'campaign_001/STATUS.json'), read(raw / 'campaign_001/COSTS.json')
    compare(status, dict(status='completed', phase='finished', global_optimizer_step=512,
        new_optimizer_steps=256, new_presentations=512, new_feedback_generations=0,
        new_qualification_optimizer_steps=0, automatic_retry=False, execution_source=source,
        actual_new_Adam_steps_completed=256, actual_global_Adam_step=512,
        actual_K_presentations_completed=512),
        issues, 'completed bounded campaign')
    compare(status['evaluation_totals'], dict(planned_generations=1152, attempted_generations=1152,
        returned_generations=1152, completed_valid_generations=1152, invalid_generations=0,
        unknown_generation_attempts=0, unattempted_generations=0), issues, 'exact 1152 native executions')
    check(not status.get('cleanup_errors'), issues, 'all native/trainer cleanup completed')
    compare(costs, dict(new_qualification_optimizer_steps=0, execution_source=source), issues,
            'startup and execution identity cost accounting')
    check(all(type(costs[key]) in (int, float) and math.isfinite(costs[key]) and costs[key] > 0 for key in
        ('loading_and_live_resume_s', 'snapshot_validation_s', 'backend_loading_s', 'actual_campaign_wall_s')),
        issues, 'finite separated campaign costs')
    for number in (3, 4):
        event_cost = costs['events'][f'event_{number:03d}']
        check(all(type(event_cost[key]) in (int, float) and math.isfinite(event_cost[key])
                  and event_cost[key] > 0 for key in ('event_wall_s', 'saved_state_validation_s',
                                                     'peak_allocated_bytes', 'peak_reserved_bytes')),
              issues, f'event{number}: finite cost and GPU peak records')
        check(event_cost['peak_allocated_bytes'] <= event_cost['peak_reserved_bytes'],
              issues, f'event{number}: allocated memory bounded by reserved')
    engine = read(raw / 'campaign_001/native/engine_settings.json')
    compare(engine, dict(model_path=plan['target'], speculative_draft_model_path=plan['draft'],
        speculative_algorithm='DFLASH', speculative_num_draft_tokens=16, dtype='bfloat16',
        attention_backend='fa3', tp_size=1, max_running_requests=1, context_length=4112,
        chunked_prefill_size=8192, random_seed=17, skip_server_warmup=True),
        issues, 'qualified native engine; 4096 token context plus draft margin')
    retrieval = read(raw / 'RETRIEVAL_MANIFEST.json')
    retrieved_bytes = 0
    for name, identity in retrieval.items():
        local = raw / name
        check(local.is_file() and local.stat().st_size == identity['bytes'] and digest(local) == identity['sha256'],
              issues, 'retrieved identity: ' + name)
        retrieved_bytes += identity['bytes']
    source_receipts = read(HERE / 'AUDIT_SOURCE_RECEIPTS.json')
    for name, identity in source_receipts['files'].items():
        path = Path(name)
        check(path.stat().st_size == identity['bytes'] and digest(path) == identity['sha256'],
              issues, 'retained audit source receipt identity: ' + name)
    return dict(sealed_files=len(seal['files']), local_files_rehashed=local_count,
        remote_only_files=len(remote_only), retrieved_files_rehashed=len(retrieval),
        retrieved_bytes=retrieved_bytes, execution_source=source,
        source_integrity_basis='Local counterparts rehashed; remote pre/postflight verifies nonlocal tensors.',
        preflight=preflight, postflight=postflight, status=status, costs=costs)


def run_audit(raw):
    raw, issues = Path(raw), []
    result = dict(verification_status='ANALYZED', scope='KRC own-state events 3/4 and separate fixed panels',
        auditor_training_steps=0, auditor_generations=0, issues=issues,
        provenance='Independent same-team retained-data/source audit, not blinded external review or rerun.')
    try:
        plan = read(raw / 'sealed/PLAN.json')
        result['training'] = audit_training(raw, plan, issues)
        panels = {panel['name']: audit_panel(raw, panel, issues) for panel in plan['panels']}
        result['panels'] = panels
        captures = [capture for panel in panels.values() for capture in panel['native_captures']]
        check(len(captures) == len(set(captures)) == 1152 and set(captures) ==
              {path.name for path in (raw / 'campaign_001/native').glob('request_*.json')},
              issues, 'exact 1152 unique native worker captures across both panels')
        result['native_blocks'] = sum(panel['reconstructed_native_blocks'] for panel in panels.values())
        result['integrity'] = audit_integrity(raw, plan, issues)
    except (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError, StopIteration) as error:
        issues.append(type(error).__name__ + ': ' + str(error))
    result['status'] = 'passed' if not issues else 'issues_found'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=HERE / 'raw')
    parser.add_argument('--output', type=Path, default=HERE / 'FINAL_AUDIT.json')
    arguments = parser.parse_args()
    result = run_audit(arguments.raw)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], issues=result['issues'],
        native_blocks=result.get('native_blocks'), panels={name: panel['comparisons']
            for name, panel in result.get('panels', {}).items()}), indent=2))
    raise SystemExit(bool(result['issues']))
