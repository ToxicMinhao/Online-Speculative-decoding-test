"""Audit native evidence and report the fixed two-seed released-prefix comparison."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from online_feedback.storage import write_json
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
    publication_errors, weight_check_errors,
)
from experiments.osd_persistent.run import score_record
from experiments.ok_coding.report import write_csv
from .contract import ARMS, CELLS, METHODS, SEEDS, evaluation_schedule, validate_plan
from .statistics import TIMINGS, summarize


def read_json(path):
    return json.loads(Path(path).read_text())


def local_artifact(root, recorded):
    """Resolve a retained raw path after compact retrieval without changing its identity."""
    path = Path(recorded)
    if path.exists():
        return path
    marker = "/campaign_001/"
    if marker not in str(path):
        raise FileNotFoundError(recorded)
    return Path(root) / "campaign_001" / str(path).split(marker, 1)[1]


def audit_cell(root, plan, seed, event):
    """Independently validate raw outputs/counts and reduce repeats by request."""
    root = Path(root)
    folder = root / "campaign_001" / f"seed_{seed}" / f"event_{event:03d}" / "evaluation"
    requests = plan["evaluation_requests"]
    errors, rows = [], []
    try:
        protocol = read_json(folder / "PROTOCOL.json")
        status = read_json(folder / "STATUS.json")
        canonical = read_json(root / "campaign_001" / "EVALUATION_CANONICAL.json")
        rows = [json.loads(line) for line in (folder / "MEASUREMENTS.jsonl").read_text().splitlines()]
        expected_schedule = evaluation_schedule(requests, seed=seed, event=event)
        schedule_keys = ("seed", "event", "request_id", "panel_ordinal", "repeat", "method", "position")
        if (protocol["schedule"] != expected_schedule
                or [{key: row.get(key) for key in schedule_keys} for row in rows] != expected_schedule):
            errors.append("schedule_or_chronological_rows")
        expected_count = len(requests) * 12
        for key, expected in {"status": "completed", "qualification": "passed",
                "attempted_generations": expected_count, "returned_generations": expected_count,
                "completed_valid_generations": expected_count, "invalid_generations": 0,
                "pending_canonical_generations": 0, "unknown_generation_attempts": 0,
                "unattempted_generations": 0}.items():
            if status.get(key) != expected:
                errors.append("status_" + key)
        gate = read_json(folder / "QUALIFICATION.json")
        if gate != {"status": "passed", "requests": 8, "generations": 96,
                    "methods_with_native_equality": ["F", "K", "KP", "K_HALF"],
                    "performance_win_required": False, "extra_generations": 0}:
            errors.append("included_gate")
        checkpoints = {"F": plan["draft"], "K": plan["original_k4"][str(seed)]["export"]}
        for arm in ARMS:
            endpoint = read_json(root / "campaign_001" / f"seed_{seed}" / arm / "events"
                                 / f"event_{event:03d}" / "RESULT.json")
            checkpoints[arm] = endpoint["export_path"]
        if protocol["checkpoints"] != checkpoints:
            errors.append("checkpoint_binding")
        for method in METHODS:
            equality = read_json(folder / ("WEIGHTS_" + method + ".json"))
            if equality["errors"] or weight_check_errors(equality["native_reply"], checkpoints[method]):
                errors.append("weights_" + method)
        for line in (folder / "ACTIVATIONS.jsonl").read_text().splitlines():
            activation = json.loads(line)
            if activation["errors"] or publication_errors(activation, checkpoints[activation["method"]]):
                errors.append("publication")
        if set(canonical) != {request["request_id"] for request in requests}:
            errors.append("canonical_panel")
        for request in requests:
            rid = request["request_id"]
            reference = canonical[rid]
            ordinal = request["panel_ordinal"]
            expected_source = (root / "campaign_001" / "seed_17" / "event_004" / "evaluation"
                               / f"{ordinal:04d}_r0_F.json")
            if (reference["source_method"] != "F" or reference["source_repeat"] != 0
                    or reference["source_cell"] != [17, 4]
                    or reference["F_checkpoint"] != plan["draft"]
                    or local_artifact(root, reference["source_record_path"]).resolve() != expected_source.resolve()):
                errors.append(rid + ":canonical_source")
            source = read_json(expected_source)
            if (source["response_ids"] != reference["response_ids"]
                    or source["finish_reason"] != reference["finish_reason"]):
                errors.append(rid + ":canonical_source_output")
            source_score = score_record(source, request_id=rid, repeat=0, method="F")
            if any(source_score[name] != reference["F_counts"][name] for name in COUNT_FIELDS):
                errors.append(rid + ":canonical_F_counts")
            selected = [row for row in rows if row["request_id"] == rid]
            for row in selected:
                path = folder / (row["label"] + ".json")
                if local_artifact(root, row["record_path"]).resolve() != path.resolve():
                    errors.append(row["label"] + ":raw_path")
                record = read_json(path)
                invalid = record_errors(record, request, checkpoints[row["method"]], reference)
                if (record["record_id"] != row["label"] or record["request_index"] != ordinal
                        or record["session_id"] != "k_prefix_protection_v1_evaluation"):
                    invalid.append("native_identity")
                recalculated = score_record(record, request_id=rid, repeat=row["repeat"], method=row["method"])
                recalculated.update(finish_reason=record["finish_reason"],
                                    cache_maintenance_s=record["metadata"]["capture"]["cache_maintenance_s"])
                if any(row.get(key) != value for key, value in recalculated.items()):
                    invalid.append("raw_measurement_disagreement")
                invalid.extend(measurement_errors(row, request, reference))
                invalid.extend(repeat_errors(selected, row["method"]))
                if row["method"] == "F" and any(row[name] != reference["F_counts"][name] for name in COUNT_FIELDS):
                    invalid.append("global_F_counts")
                if (row.get("valid_measurement") is not True or row.get("errors")
                        or row.get("validation_status") != "valid"):
                    invalid.append("measurement_not_valid")
                errors.extend(row["label"] + ":" + error for error in sorted(set(invalid)))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append("missing_or_malformed_evidence: " + repr(error))
    if errors:
        return [], errors
    return reduce_requests(rows, requests), []


def reduce_requests(rows, requests):
    reduced = []
    for request in requests:
        value = {"request_id": request["request_id"], "panel_ordinal": request["panel_ordinal"]}
        for method in METHODS:
            selected = [row for row in rows if row["request_id"] == request["request_id"] and row["method"] == method]
            selected.sort(key=lambda row: row["repeat"])
            if [row["repeat"] for row in selected] != [0, 1, 2] or repeat_errors(selected, method):
                raise ValueError("Three identical-count timing repeats are required")
            value[method] = {name: selected[0][name] for name in COUNT_FIELDS}
            value[method].update({name: sum(row[name] for row in selected) / 3 for name in TIMINGS})
            value[method]["cap_hit"] = selected[0]["cap_hit"]
            value[method]["finish_reason"] = selected[0]["finish_reason"]
        reduced.append(value)
    return reduced


def response_observations(result, *, seed, event, arm):
    """Use records captured before accumulation merges; count each original forward once."""
    responses, windows = [], []
    for ordinal, row in enumerate(result['records']):
        observation = row['response_observation']
        if (observation['record_id'] != row['record_id']
                or observation['scope'] != 'one_original_response_forward'):
            raise ValueError('Response observation identity or scope differs')
        if not math.isfinite(observation['loss']):
            raise ValueError('Nonfinite response loss')
        for pair in observation['ratio_metrics'].values():
            if len(pair) != 2 or not all(math.isfinite(value) for value in pair) or pair[1] < 0:
                raise ValueError('Malformed response ratio metric')
        if not all(math.isfinite(value) for value in observation['sum_metrics'].values()):
            raise ValueError('Nonfinite response sum metric')
        base = dict(seed=seed, event=event, arm=arm, pass_index=row['pass'],
                    window=ordinal // 2 + 1, response_in_window=ordinal % 2,
                    presentation_ordinal=ordinal, record_id=row['record_id'])
        prefix = observation.get('prefix_protection')
        if arm == 'KP':
            if not prefix or not prefix['protection_evaluated'] or not prefix['finite']:
                raise ValueError('Missing or nonfinite production prefix-protection evidence')
            if prefix['qualification_forced_active']:
                raise ValueError('Disposable forced-active override entered production')
            if (sum(prefix['protected_prefix_histogram']) != prefix['protected_prefix_histogram_denominator']
                    or prefix['active_positions'] > prefix['protected_positions']
                    or prefix['incorrect_protected_positions'] > prefix['protected_positions']
                    or prefix['wrong_tie_positions'] > prefix['incorrect_protected_positions']):
                raise ValueError('Protection occurrence accounting differs')
            for name in ('reference_margins', 'current_margins', 'thresholds'):
                if prefix[name]['count'] != prefix['protected_positions']:
                    raise ValueError('Protected margin count differs')
            if not math.isclose(observation['loss'], prefix['total_loss'], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('Response total loss differs from objective record')
        elif prefix is not None:
            raise ValueError('K_HALF must not execute reference protection')
        responses.append({**base, 'training_s': row['training_s'],
                          'sampled_anchors': row['sampled_anchors'], **observation})
        if row['optimizer_stepped']:
            gradient = row['metrics']['grad_norm']
            if not math.isfinite(gradient):
                raise ValueError('Nonfinite native window gradient norm')
            windows.append({**base, 'scope': 'optimizer_window',
                'optimizer_step_after': row['optimizer_step_after'],
                'gradient_norm': gradient, 'clip_threshold': 1.,
                'clipping_active': gradient > 1., 'learning_rate': row['learning_rate']})
    if len(responses) != 256 or len(windows) != 128:
        raise ValueError('Event response/window observation counts differ')
    return responses, windows


def audit_training(root, plan):
    """Keep failed/incomplete events visible and out of valid training aggregates."""
    from experiments.ok_coding.campaign import read_work, require_complete_work
    from experiments.ok_coding.training import validate_event_receipt

    output = Path(root) / 'campaign_001'
    events, errors, pids, devices, caches = [], [], [], [], []
    for seed in SEEDS:
        seed_root = output / f'seed_{seed}'
        for event in range(1, 5):
            for arm in ARMS:
                label = f'{seed}/{event}/{arm}'
                audit = seed_root / f'resume_{event:03d}_{arm}'
                entry = dict(seed=seed, event=event, arm=arm, valid=False, errors=[])
                try:
                    result = read_json(seed_root / arm / 'events' / f'event_{event:03d}' / 'RESULT.json')
                    schedule = plan['original_schedules'][str(seed)][str(event)]
                    schedule = schedule['path'] if isinstance(schedule, dict) else schedule
                    original = read_json(schedule)
                    validate_event_receipt(result, event_number=event, objective='osd_first_error',
                                           reference_schedule=original)
                    worker = read_json(audit / 'WORKER.json')
                    limits = dict(forwards=256, backwards=256, optimizer_steps=128)
                    if arm == 'KP':
                        limits['reference_forwards'] = 256
                    work = read_work(audit / 'WORK.jsonl', limits)
                    require_complete_work(work, limits)
                    if (worker['status'] != 'completed' or worker['work'] != work
                            or worker['additional_qualification_steps'] != 0):
                        raise ValueError('Worker completion or scientific accounting differs')
                    startup = worker['startup']
                    if startup['cuda_initialized_on_entry'] or startup['warmup_model_work'] != 0:
                        raise ValueError('Native cold startup differs')
                    if any(not value['initially_empty'] for value in startup['caches'].values()):
                        raise ValueError('Compiler/kernel cache was not initially empty')
                    live = read_json(audit / 'LIVE.json')
                    boundary = read_json(audit / 'LIVE_BOUNDARY.json')
                    endpoint = read_json(audit / 'SAVED_ENDPOINT.json')
                    if (not live['exact_resume'] or not boundary['exact_resume']
                            or not endpoint['export_exact'] or endpoint['status'] != 'passed'
                            or endpoint['global_step'] != event * 128):
                        raise ValueError('Own-state restoration or endpoint equality failed')
                    if event == 1:
                        migration = read_json(audit / 'MIGRATION.json')
                        if (migration['status'] != 'passed'
                                or not migration['source_weights_rng_moments_counters_history_exact']):
                            raise ValueError('Initial scientific state migration differs')
                    responses, windows = response_observations(result, seed=seed, event=event, arm=arm)
                    closed = read_json(audit / 'CLOSED.json')
                    if (not closed['closed'] or closed['failed'] or closed['accumulation_remainder'] != 0
                            or closed['global_step'] != event * 128 or closed['presentations'] != event * 256):
                        raise ValueError('Trainer did not close at the exact empty boundary')
                    costs = read_json(audit / 'COSTS.json')
                    if costs['response_observations'] != 256:
                        raise ValueError('Response cost-accounting denominator differs')
                    if arm == 'KP' and costs['objective_work']['work']['reference_response_forwards'] != 256:
                        raise ValueError('Reference response work differs')
                    entry.update(valid=True, costs=costs, response_records=responses,
                                 window_records=windows, work=work, startup=startup,
                                 selection_totals=result.get('selection_totals'))
                    pids.append(startup['pid'])
                    devices.append((startup['cuda_visible_devices'], startup['device_uuid']))
                    caches.extend(value['path'] for value in startup['caches'].values())
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    entry['errors'].append(repr(error))
                    errors.append(label + ':' + repr(error))
                events.append(entry)
    if len(pids) != 16 or len(set(pids)) != 16 or len(set(devices)) != 1:
        errors.append('fresh_process_or_same_updater')
    if len(caches) != 96 or len(set(caches)) != 96:
        errors.append('separate_empty_compiler_caches')
    return events, errors


def summarize_observations(records):
    """Average normalized response losses; sum exposure without boundary-row duplication."""
    result = {'responses': len(records), 'mean_total_response_loss':
              sum(row['loss'] for row in records) / len(records)}
    sum_names = set().union(*(row['sum_metrics'] for row in records))
    result['sum_metrics'] = {name: sum(row['sum_metrics'].get(name, 0) for row in records)
                             for name in sorted(sum_names)}
    protected = [row['prefix_protection'] for row in records if 'prefix_protection' in row]
    if protected:
        for name in ('k_loss', 'protection_loss', 'weighted_protection_loss'):
            result['mean_response_' + name] = sum(row[name] for row in protected) / len(protected)
        for name in ('available_blocks', 'protected_blocks', 'empty_protected_blocks',
                     'protected_positions', 'active_positions', 'incorrect_protected_positions',
                     'wrong_tie_positions', 'current_zero_margin_positions', 'active_weight',
                     'protection_weight', 'protection_numerator', 'protection_denominator',
                     'k_numerator', 'k_selected_weight', 'k_denominator'):
            result[name] = sum(row[name] for row in protected)
        result['protected_prefix_histogram'] = [sum(row['protected_prefix_histogram'][i] for row in protected)
                                                for i in range(len(protected[0]['protected_prefix_histogram']))]
        result['protected_prefix_histogram_denominator'] = sum(row['protected_prefix_histogram_denominator'] for row in protected)
        denominator = result['protected_positions']
        result['active_position_fraction'] = result['active_positions'] / denominator if denominator else None
        result['incorrect_protected_fraction'] = result['incorrect_protected_positions'] / denominator if denominator else None
        weight = result['protection_weight']
        result['active_weight_fraction'] = result['active_weight'] / weight if weight else None
        for name in ('reference_margins', 'current_margins', 'thresholds'):
            observations = [row[name] for row in protected]
            nonempty = [value for value in observations if value['count']]
            result[name] = {'count': sum(value['count'] for value in observations),
                            'sum': sum(value['sum'] for value in observations),
                            'min': min(value['min'] for value in nonempty) if nonempty else None,
                            'max': max(value['max'] for value in nonempty) if nonempty else None}
        result['reference_zero_margin_count'] = sum(row['reference_margins']['zero_margin_count'] for row in protected)
    return result


def aggregate_training(events):
    groups = defaultdict(list)
    for event in events:
        if not event['valid']:
            continue
        for row in event['response_records']:
            prefix = f"{row['seed']}/{row['arm']}/{row['event']}"
            groups[prefix].append(row)
            groups[prefix + f"/pass_{row['pass_index']}"].append(row)
            groups[prefix + f"/window_{row['window']:03d}"].append(row)
    return {name: summarize_observations(rows) for name, rows in groups.items()}


def flatten_record(row):
    return {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()}


def audit_qualification(root):
    from experiments.ok_coding.campaign import read_work, require_complete_work
    from .contract import QUALIFICATION_BUDGET
    from .qualification import CASES, case_limits, qualification_work, require_neutral_equal
    folder = Path(root) / 'qualification_001'
    try:
        result = read_json(folder / 'RESULT.json')
        expected = {name: value for name, value in QUALIFICATION_BUDGET.items() if name != 'generations'}
        work = qualification_work(root)
        require_complete_work(work, expected)
        case_rows = []
        for name, _options, references in CASES:
            case = read_json(folder / name / 'CASE.json')
            case_work = read_work(folder / name / 'WORK.jsonl', case_limits(references))
            require_complete_work(case_work, case_limits(references))
            if (case['status'] != 'completed' or case['work'] != case_work
                    or not case['serialized_restored_exact'] or case['frozen_changes']
                    or case['global_step'] != 513 or case['presentations'] != 1026):
                raise ValueError('Disposable case evidence differs: ' + name)
            case_rows.append(case)
        if (len({row['startup']['pid'] for row in case_rows}) != 4
                or len({row['startup']['device_uuid'] for row in case_rows}) != 1):
            raise ValueError('Qualification did not use four cold processes on one updater')
        require_neutral_equal(read_json(folder / 'Q2/VERSUS_Q1.json'))
        forced = read_json(folder / 'Q4/FORCED_ACTIVE.json')
        if forced['gradients']['exact'] or forced['moments']['exact']:
            raise ValueError('Forced active gradients/moments did not change')
        if result['status'] != 'passed' or result['work'] != work or result['budget'] != expected:
            raise ValueError('Bounded numerical qualification did not pass exactly')
        return result, []
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, ['qualification:' + repr(error)]


def length_and_termination_summary(reduced):
    summary = {}
    for seed, rows in reduced.items():
        summary[seed] = {}
        for method in METHODS:
            counts = [row[method]['output_tokens'] for row in rows]
            caps = sum(row[method]['cap_hit'] for row in rows)
            summary[seed][method] = dict(
                requests=len(rows), minimum_output_tokens=min(counts), maximum_output_tokens=max(counts),
                mean_output_tokens=sum(counts)/len(counts), output_cap_requests=caps,
                output_cap_frequency=caps/len(rows),
                termination_counts=dict(Counter(row[method]['finish_reason'] for row in rows)))
    return summary


def cost_summary(events, plan):
    """Keep parent/subtimers separate and identify historical K as unmatched context."""
    groups = defaultdict(list)
    for event in events:
        if event['valid']:
            groups[f"{event['seed']}/{event['arm']}"].append(event['costs'])
    output = {'current': {}, 'historical_K': {},
        'scope': 'Cold native update timings; historical K is not a fresh matched overhead measurement.',
        'timer_overlap': 'Loading includes reference setup; event wall includes training/save/export; '
                         'objective subtimers are included in training or loading. Do not add nested timers.'}
    for key, rows in groups.items():
        value = {'events': len(rows)}
        for field in ('training_s', 'event_wall_s', 'loading_initialization_resume_validation_s',
                      'frozen_reference_setup_s', 'checkpoint_save_s', 'export_s', 'state_validation_hash_s'):
            value[field] = sum(row.get(field, 0.) for row in rows)
        for field in ('peak_allocated_bytes', 'peak_reserved_bytes', 'loading_peak_allocated_bytes',
                      'loading_peak_reserved_bytes'):
            value[field] = max(row.get(field, 0) for row in rows)
        value['objective_subtimers'] = {field: sum(row['objective_work'].get(field, 0.) for row in rows)
            for field in ('reference_setup_s', 'reference_forward_s', 'reference_head_s',
                          'extra_loss_s', 'teacher_head_s', 'selection_s')}
        output['current'][key] = value
    if 'original_run' in plan:
        path = Path(plan['original_run']) / 'final_report/TRAINING_COSTS.csv'
        if path.exists():
            with path.open(newline='') as stream:
                historical = [row for row in csv.DictReader(stream) if row['arm'] == 'K']
            for seed in SEEDS:
                selected = [row for row in historical if int(row['seed']) == seed and int(row['event']) in (1,2,3,4)]
                if len(selected) == 4:
                    output['historical_K'][str(seed)] = {
                        field: sum(float(row[field]) for row in selected)
                        for field in ('training_s', 'event_wall_s', 'loading_initialization_resume_validation_s',
                                      'checkpoint_save_s', 'export_s')}
            output['historical_source'] = str(path)
        else:
            output['historical_source_unavailable'] = str(path)
    return output


def write_campaign_report(root, output, *, execution_commit=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root / 'sealed/PLAN.json')
    errors = []
    try:
        validate_plan(plan)
        if execution_commit is not None and execution_commit != plan['execution_source']['commit']:
            raise ValueError('Execution commit does not match sealed source')
    except (ValueError, KeyError, TypeError) as error:
        errors.append('plan:' + repr(error))
    qualification, qualification_errors = audit_qualification(root)
    errors.extend(qualification_errors)
    training, training_errors = audit_training(root, plan)
    errors.extend(training_errors)
    reduced, cell_errors = {}, {}
    for seed, event in CELLS:
        rows, invalid = audit_cell(root, plan, seed, event)
        cell_errors[str(seed)] = invalid
        errors.extend(f'{seed}/{event}:' + reason for reason in invalid)
        if not invalid:
            reduced[str(seed)] = [{**row, 'seed': seed, 'event': event} for row in rows]
    try:
        from .campaign import reconcile, require_complete
        status = read_json(root / 'campaign_001/STATUS.json')
        reconstructed = reconcile(root)
        require_complete(reconstructed)
        if any(status.get(name) != value for name, value in reconstructed.items()):
            errors.append('campaign_counter_reconstruction_disagreement')
        if status['status'] != 'completed':
            errors.append('campaign_not_completed')
    except (OSError, KeyError, ValueError) as error:
        status = {'status': 'unavailable', 'error': repr(error)}
        errors.append('campaign_status_unavailable')
    analysis = summarize([row for values in reduced.values() for row in values]) if len(reduced) == 2 else None
    summary = {'status': 'valid' if not errors else 'invalid_or_incomplete', 'errors': errors,
               'execution_source': plan['execution_source'], 'campaign_status': status,
               'qualification': qualification, 'analysis': analysis,
               'cell_errors': cell_errors,
               'interpretation': analysis['interpretation'] if not errors else
                    {'endpoint_label': 'Invalid/incomplete; no efficacy interpretation'},
               'training_scope': 'Only independently validated complete events enter aggregates.',
               'lengths_and_termination': length_and_termination_summary(reduced),
               'costs': cost_summary(training, plan)}
    write_json(output / 'SUMMARY.json', summary)
    write_json(output / 'TRAINING_AUDIT.json', {'errors': training_errors, 'events': training})
    aggregates = aggregate_training(training)
    write_json(output / 'TRAINING_AGGREGATES.json', aggregates)
    write_json(output / 'COST_SUMMARY.json', summary['costs'])
    write_csv(output / 'PAIRED_REQUESTS.csv', [
        {key: value for key, value in row.items() if key not in METHODS} |
        {method + '_' + field: value for method in METHODS for field, value in row[method].items()}
        for rows in reduced.values() for row in rows])
    responses = [row for event in training if event['valid'] for row in event['response_records']]
    windows = [row for event in training if event['valid'] for row in event['window_records']]
    write_csv(output / 'TRAINING_RESPONSES.csv', [flatten_record(row) for row in responses])
    write_csv(output / 'TRAINING_WINDOWS.csv', windows)
    write_csv(output / 'TRAINING_COSTS.csv', [flatten_record({
        'seed': event['seed'], 'event': event['event'], 'arm': event['arm'], **event['costs']})
        for event in training if event['valid']])
    all_repeats, repeat_totals, evaluation_costs = [], [], []
    for seed, event in CELLS:
        folder = root / 'campaign_001' / f'seed_{seed}' / f'event_{event:03d}' / 'evaluation'
        measurements = folder / 'MEASUREMENTS.jsonl'
        if not measurements.exists():
            continue
        rows = [json.loads(line) for line in measurements.read_text().splitlines()]
        all_repeats.extend(flatten_record(row) for row in rows)
        if (folder / 'STATUS.json').exists():
            evaluation_costs.append({'seed': seed, 'event': event, **read_json(folder / 'STATUS.json')})
        for method in METHODS:
            for repeat in range(3):
                selected = [row for row in rows if row['method'] == method and row['repeat'] == repeat]
                valid = selected if str(seed) in reduced else []
                repeat_totals.append({'seed': seed, 'event': event, 'method': method, 'repeat': repeat,
                    'returned_rows': len(selected), 'independently_valid_rows': len(valid),
                    **{name: sum(row[name] for row in valid) for name in (*COUNT_FIELDS, *TIMINGS)}})
    write_csv(output / 'ALL_REPEATS.csv', all_repeats)
    write_csv(output / 'REPEAT_TOTALS.csv', repeat_totals)
    write_json(output / 'EVALUATION_COSTS.json', evaluation_costs)
    lines = ['# Released-prefix protection v1: fixed coding comparison', '',
             'Execution source: `' + plan['execution_source']['commit'] + '`.', '',
             'Hypothesis: protecting released-DFlash prefixes may remove more harmful changes than useful '
             'K gains. The expected pattern allowed partial harm reduction together with lost gains; '
             'a modest or unresolved effect was anticipated.', '', summary['interpretation']['endpoint_label'] + '.', '']
    if analysis is not None:
        lines += ['| Scope | Method | Calls | Output tokens | Calls / 1,000 tokens | Accepted draft tokens | Progress / call |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for scope, methods in analysis['method_totals'].items():
            for method, totals in methods.items():
                lines.append(f"| {scope} | {method} | {totals['verification_calls']} | {totals['output_tokens']} | "
                    f"{totals['calls_per_1000_output_tokens']:.4f} | {totals['accepted_draft_tokens']} | "
                    f"{totals['committed_progress_per_call']:.4f} |")
        lines += ['', '| Scope | Comparison | Call reduction | Paired 95% interval | Request-time reduction | Wins / regressions / ties |',
                  '|---|---|---:|---:|---:|---:|']
        for scope, comparisons in [('combined', analysis['combined']), *analysis['per_seed'].items()]:
            for name, value in comparisons.items():
                metric = value['call_reduction']
                low, high = metric['ci95_pct']
                lines.append(f"| {scope} | {name} | {metric['point_pct']:+.4f}% | [{low:+.4f}, {high:+.4f}]% | "
                    f"{value['request_latency_s_reduction']['point_pct']:+.4f}% | "
                    f"{value['request_seed_wins']} / {value['request_seed_regressions']} / {value['request_seed_ties']} |")
        lines += ['', 'Combined decisions count 256 seed/request pairs, while uncertainty resamples 128 joint '
                  'request identities carrying both fitted seeds and all methods. The same 10,000 PCG64 '
                  'seed-20261102 draws are reused everywhere. Intervals are conditional on these two '
                  'trajectories/panel; secondary intervals are descriptive and unadjusted.', '',
                  '| Candidate | Retained K gain | Lost gain | Added gain | Prevented harm | Added harm | Retained fraction |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method, values in analysis['gain_harm']['combined'].items():
            fraction = values['retained_gain_fraction']['point']
            formatted = 'undefined' if fraction is None else f'{fraction:.4f}'
            lines.append(f"| {method} | {values['retained_gain']['point']:.0f} | {values['lost_gain']['point']:.0f} | "
                f"{values['added_gain']['point']:.0f} | {values['prevented_harm']['point']:.0f} | "
                f"{values['added_harm']['point']:.0f} | {formatted} |")
    if any(event['valid'] and event['arm'] == 'KP' for event in training):
        lines += ['', '| Seed / event | Mean K loss | Mean protection loss | Protected positions | Active / protected | Incorrect / protected | Wrong ties |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for event in training:
            if not event['valid'] or event['arm'] != 'KP':
                continue
            value = aggregates[f"{event['seed']}/KP/{event['event']}"]
            active = value['active_position_fraction']
            incorrect = value['incorrect_protected_fraction']
            lines.append(f"| {event['seed']} / {event['event']} | {value['mean_response_k_loss']:.6f} | "
                f"{value['mean_response_protection_loss']:.6f} | {value['protected_positions']} | "
                f"{active if active is not None else 'undefined'} | {incorrect if incorrect is not None else 'undefined'} | "
                f"{value['wrong_tie_positions']} |")
    if summary['costs']['current']:
        lines += ['', '| Trajectory | Completed events | Training seconds | Reference forward/head seconds (included) | Peak allocated GiB |',
                  '|---|---:|---:|---:|---:|']
        for key, cost in summary['costs']['current'].items():
            timers = cost['objective_subtimers']
            lines.append(f"| {key} | {cost['events']} | {cost['training_s']:.3f} | "
                f"{timers['reference_forward_s'] + timers['reference_head_s']:.3f} | {cost['peak_allocated_bytes']/2**30:.3f} |")
        if summary['costs']['historical_K']:
            lines += ['', 'Historical original K training seconds (unmatched context): ' + ', '.join(
                f"seed {seed}: {cost['training_s']:.3f}" for seed, cost in summary['costs']['historical_K'].items()) + '.']
    if errors:
        lines += ['', 'Evidence failures (raw records retained; no efficacy claim):', '', *['- ' + error for error in errors]]
    lines += ['', 'All losses/exposures in TRAINING_RESPONSES.csv are original-response observations before '
              'accumulation merging. TRAINING_WINDOWS.csv records one native gradient norm/clipping decision '
              'per optimizer window. TRAINING_AGGREGATES.json averages normalized response losses and sums '
              'exposure without counting checkpoint recomputation twice. These are training-block metrics, '
              'not native acceptance or a measured matched-K preservation counterfactual.', '',
              'Serving totals use one validated trajectory per request; times average its three repeats. '
              'Complete-request, native-generation and cache-maintenance scopes remain separate. '
              'TRAINING_COSTS.csv preserves cold compilation, reference setup/forwards/heads, extra loss, '
              'teacher heads, persistence, export, state validation and memory scopes. Nested subcosts '
              'overlap parent timers and must not be added to them. EVALUATION_COSTS.json separates '
              'publication/switching from serving. Historical K training costs are context, not fresh matched costs.', '',
              'This is a new short-prompt CodeAlpaca panel under the same raw-row project partition. Source '
              'answers were unused; all selected IDs/tokens and exclusions were sealed before model work. '
              'No O comparison, general convergence, stochastic-decoding, multi-turn domain-switching, '
              'asynchronous deployment or training-inclusive speedup claim follows.', '',
              'Next action: stop at this fixed comparison. Review its paired uncertainty, gain/harm profile '
              'and lower-LR control before agreeing to any subsequent experiment. No automatic tuning or extension.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--execution-commit')
    args = parser.parse_args()
    report = write_campaign_report(args.root, args.output or str(Path(args.root) / 'final_report'), execution_commit=args.execution_commit)
    print(json.dumps({'status': report['status'], 'errors': report['errors']}))
    raise SystemExit(0 if report['status'] == 'valid' else 2)
