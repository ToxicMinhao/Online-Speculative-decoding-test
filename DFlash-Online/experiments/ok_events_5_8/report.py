"""O/K endpoint reports independently revalidate retained or first-O8 canonicals."""

import csv
import hashlib
import json
from pathlib import Path
import re

from online_feedback.storage import write_json
from experiments.osd_one_event.report import direction, reduction
from .evaluation_protocol import (
    ARMS, REPEATS, COMPARISONS, COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
)


def validated_row(row, request, checkpoints, canonical):
    errors = list(row.get('errors', []))
    errors.extend(measurement_errors(row, request, canonical))
    if row.get('method') not in ARMS or row.get('repeat') not in range(REPEATS):
        errors.append('invalid_evaluation_slot')
    if row.get('valid_measurement') is not True:
        errors.append('execution_rejected_or_pending_measurement')
    if canonical is None:
        errors.append('canonical_unavailable')
    path = Path(row['record_path'])
    if not path.is_file():
        errors.append('native_record_unavailable')
    else:
        try:
            record = json.loads(path.read_text())
            errors.extend(record_errors(record, request, checkpoints[row['method']], canonical))
            metadata = record['metadata']
            for name in ('verification_calls', 'accepted_draft_tokens', 'committed_progress',
                         'request_latency_s', 'native_generate_s'):
                if row.get(name) != metadata[name]:
                    errors.append('row_native_disagreement_' + name)
            if row.get('cache_maintenance_s') != metadata['capture']['cache_maintenance_s']:
                errors.append('row_native_disagreement_cache_maintenance_s')
            if (record['record_id'] != row['label'] or record['request_index'] != request['panel_ordinal']
                    or record['session_id'] != 'ok_events_5_8_evaluation'):
                errors.append('native_identity')
            if row.get('output_tokens') != len(record['response_ids']):
                errors.append('row_native_disagreement_output_tokens')
            if row.get('checkpoint_id') != checkpoints[row['method']]:
                errors.append('checkpoint_identity')
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append('malformed_native_record: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def canonical_errors(canonical, request, rows, checkpoints, canonical_mode):
    if canonical is None:
        return ['canonical_unavailable']
    if canonical_mode == 'retained':
        if (canonical.get('source') != 'retained_panel_canonical'
                or canonical.get('source_method') != request['canonical_source_method']
                or any(canonical.get(key) != request[key] for key in ('response_ids', 'finish_reason'))):
            return ['retained_canonical_binding']
        return []
    sources = [row for row in rows if row['method'] == 'O' and row['repeat'] == 0]
    if len(sources) != 1 or canonical['source_record_path'] != sources[0]['record_path']:
        return ['canonical_first_O_binding']
    try:
        record = json.loads(Path(canonical['source_record_path']).read_text())
        errors = record_errors(record, request, checkpoints['O'])
        if any(canonical[key] != record[key] for key in ('response_ids', 'finish_reason')):
            errors.append('canonical_source_output')
        if canonical['source_method'] != 'O' or canonical['source_repeat'] != 0:
            errors.append('canonical_source_slot')
        return errors
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return ['canonical_source_unavailable_or_malformed: ' + str(error)]


def summarize(rows, requests, *, checkpoints, canonicals, canonical_mode):
    arms, comparisons = ARMS, COMPARISONS
    per_request, checked_rows, valid_rows = [], [], []
    for request in requests:
        raw_rows = [row for row in rows if row['request_id'] == request['request_id']]
        canonical = canonicals.get(request['request_id'])
        canonical_issues = canonical_errors(canonical, request, raw_rows, checkpoints, canonical_mode)
        selected = [validated_row(row, request, checkpoints, canonical) for row in raw_rows]
        checked_rows.extend(selected)
        expected = {(method, repeat) for method in arms for repeat in range(REPEATS)}
        complete = (len(selected) == len(expected)
                    and {(row['method'], row['repeat']) for row in selected} == expected)
        errors = set(canonical_issues)
        errors.update(error for row in selected for error in row['report_errors'])
        for method in arms:
            errors.update(repeat_errors(selected, method))
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        valid = complete and not errors
        result = {'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'complete': complete, 'valid': valid, 'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
            for method in arms:
                repeats = [row for row in selected if row['method'] == method]
                primary = next(row for row in repeats if row['repeat'] == 0)
                for key in COUNT_FIELDS:
                    result[f'{method}_{key}'] = primary[key]
                for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
                    result[f'{method}_mean_{timer}'] = sum(row[timer] for row in repeats) / REPEATS
                calls, tokens = primary['verification_calls'], primary['output_tokens']
                result[f'{method}_calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
                result[f'{method}_accepted_draft_tokens_per_call'] = primary['accepted_draft_tokens'] / calls if calls else None
                result[f'{method}_committed_progress_per_call'] = primary['committed_progress'] / calls if calls else None
            for candidate, reference in comparisons:
                label = f'{candidate}_vs_{reference}'
                result[label + '_call_reduction_pct'] = reduction(
                    result[f'{reference}_verification_calls'], result[f'{candidate}_verification_calls'])
                result[label + '_request_time_reduction_pct'] = reduction(
                    result[f'{reference}_mean_request_latency_s'], result[f'{candidate}_mean_request_latency_s'])
                result[label + '_call_direction'] = direction(
                    result[f'{reference}_verification_calls'], result[f'{candidate}_verification_calls'])
                result[label + '_latency_direction'] = direction(
                    result[f'{reference}_mean_request_latency_s'], result[f'{candidate}_mean_request_latency_s'])
        per_request.append(result)

    valid_requests = [row for row in per_request if row['valid']]
    summary = {
        'planned_requests': len(requests), 'valid_requests': len(valid_requests),
        'invalid_or_incomplete_requests': len(requests) - len(valid_requests),
        'returned_measurements': len(rows), 'valid_timed_generations': len(valid_rows),
        'primary_comparison': '_vs_'.join(comparisons[0]), 'timing_repeats_per_method_request': REPEATS,
        'cohort': 'One common cohort of complete, valid ' + '/'.join(arms) + ' requests for all comparisons.',
        'count_scope': 'One validated repeat-0 trajectory per method/request; timing repeats are not extra examples.',
        'latency_scope': 'Capture-disabled complete request with checkpoint available: prefill, draft, verification and request cache work; training and activation excluded.',
    }
    for method in arms:
        totals = {key: sum(row[f'{method}_{key}'] for row in valid_requests) for key in COUNT_FIELDS}
        for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
            totals['summed_mean_' + timer] = sum(row[f'{method}_mean_{timer}'] for row in valid_requests)
            totals['all_repeats_' + timer] = sum(row[timer] for row in valid_rows if row['method'] == method)
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        summary[method] = totals
    summary['comparisons'] = {}
    for candidate, reference in comparisons:
        label = f'{candidate}_vs_{reference}'
        comparison = {
            'candidate': candidate, 'reference': reference,
            'call_reduction_pct': reduction(summary[reference]['verification_calls'], summary[candidate]['verification_calls']),
            'request_time_reduction_pct': reduction(summary[reference]['all_repeats_request_latency_s'], summary[candidate]['all_repeats_request_latency_s']),
            'native_generate_time_reduction_pct': reduction(summary[reference]['all_repeats_native_generate_s'], summary[candidate]['all_repeats_native_generate_s']),
        }
        for metric, key in (('call', 'verification_calls'), ('latency', 'mean_request_latency_s')):
            differences = [row[f'{reference}_{key}'] - row[f'{candidate}_{key}'] for row in valid_requests]
            comparison[metric + '_request_directions'] = {
                sign: sum(row[label + '_' + metric + '_direction'] == sign for row in valid_requests)
                for sign in ('improved', 'regressed', 'tied')}
            comparison[metric + '_gross_saved'] = sum(max(value, 0) for value in differences)
            comparison[metric + '_gross_extra'] = sum(max(-value, 0) for value in differences)
            comparison[metric + '_net_saved'] = sum(differences)
        summary['comparisons'][label] = comparison
    return summary, per_request, checked_rows, valid_rows


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(rows, requests, root, *, checkpoints, canonicals, execution, event_number,
                  panel_name, canonical_mode):
    arms, comparisons = ARMS, COMPARISONS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(
        rows, requests, checkpoints=checkpoints, canonicals=canonicals, canonical_mode=canonical_mode)
    summary.update(execution=execution, event_number=event_number, panel_name=panel_name,
                   canonical_mode=canonical_mode, historical_count_gate=False)
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'ALL_REPEATS.csv', checked_rows)
    write_csv(root / 'PER_REQUEST.csv', per_request)
    repeat_totals = []
    for repeat in range(REPEATS):
        for method in arms:
            selected = [row for row in valid_rows if row['repeat'] == repeat and row['method'] == method]
            repeat_totals.append({'repeat': repeat, 'method': method, 'requests': len(selected),
                                  **{timer: sum(row[timer] for row in selected) for timer in
                                     ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')},
                                  'verification_calls': sum(row['verification_calls'] for row in selected)})
    write_csv(root / 'REPEAT_TOTALS.csv', repeat_totals)
    title = f'# O{event_number}/K{event_number}: {panel_name} panel'
    text = [
        title, '',
        f"Execution: **{execution['status']}**. Valid common requests: **{summary['valid_requests']}/{len(requests)}**.", '',
        'All values use fresh capture-disabled measurements at this checkpoint endpoint. '
        'Retained panels use their original canonical outputs; fresh uses its first O8 output. '
        'Request latency includes prefill, drafting, target verification and request cache maintenance. '
        'Training and activation are excluded; these are results with an available checkpoint.', '',
    ]
    text.extend(['| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |',
                 '|---|---:|---:|---:|'])
    for method in arms:
        totals = summary[method]
        rate = totals['calls_per_1000_output_tokens']
        shown_rate = 'unavailable' if rate is None else f'{rate:.4f}'
        text.append(f"| {method} | {totals['verification_calls']} | "
                    f"{totals['summed_mean_request_latency_s']:.6f} | {shown_rate} |")
    text.extend(['', '| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |',
                 '|---|---:|---:|---:|'])
    for name, comparison in summary['comparisons'].items():
        values = [comparison['call_reduction_pct'], comparison['request_time_reduction_pct']]
        shown = ['unavailable' if value is None else f'{value:+.4f}%' for value in values]
        signs = comparison['call_request_directions']
        text.append(f"| {name} | {shown[0]} | {shown[1]} | {signs['improved']} / {signs['regressed']} / {signs['tied']} |")
    primary_note = f'**K versus O is the sole comparison.** Canonical mode: {canonical_mode}; no historical call count is a gate. '
    text.extend([
        '', primary_note +
        'Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.', '',
        'Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, '
        'not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. '
        'Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.', '',
        f"Benchmark-only activation: {execution['activation_s']:.6f}s; native weight validation: "
        f"{execution['weight_validation_s']:.6f}s. These are not charged to inference-only latency. "
        'Update time and initial backend loading are reported separately from serving latency.', '',
        'Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. '
        'Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.',
    ])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary


def curve_direction(initial, final, *, complete):
    if not complete or initial is None or final is None:
        return 'inconclusive: incomplete comparable panel'
    if final < 0:
        return 'reverses: K has more calls than O at event 8'
    if final == 0:
        return 'parity at event 8'
    if initial <= 0:
        return 'K has a positive event-8 advantage'
    if final > initial:
        return 'persists and grows from event 4 to event 8'
    if final < initial:
        return 'persists but shrinks from event 4 to event 8'
    return 'persists with the same aggregate call reduction'


def _load(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _local_records(directory, rows, canonicals):
    """Rebind retrieved record locations without altering immutable raw files."""
    rebound = [dict(row, record_path=str(directory / Path(row['record_path']).name)) for row in rows]
    canonical_copy = {key: dict(value) for key, value in canonicals.items()}
    for value in canonical_copy.values():
        if value.get('source_method') == 'O' and 'source_record_path' in value:
            value['source_record_path'] = str(directory / Path(value['source_record_path']).name)
    return rebound, canonical_copy


def receipt_index(root):
    """Hash compact provenance receipts without copying input or token arrays."""
    paths = list(root.glob('*.json')) + list((root / 'sealed').glob('*.json'))
    campaign = root / 'campaign_001'
    paths += list(campaign.glob('*.json')) + list(campaign.glob('resume_*/*.json'))
    paths += list(campaign.glob('retention/*.json'))
    paths += list(campaign.glob('[OK]/events/event_*/RESULT.json'))
    return [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set(paths)) if path.is_file()]


def write_campaign_report(root, output, *, execution_commit):
    """Reaudit all raw tables and keep endpoint/panel units separate."""
    from .evaluation_protocol import validate_panels
    root, output = Path(root), Path(output)
    plan = _load(root / 'sealed/PLAN.json')
    if not re.fullmatch('[0-9a-f]{40}', execution_commit):
        raise ValueError('Require the actual complete execution Git revision')
    source = plan['execution_source']
    if source.get('execution_git_commit') != execution_commit or source.get('dirty_files') != []:
        raise ValueError('Execution source receipt differs from the requested clean revision')
    validate_panels(plan['panels'])
    campaign = root / 'campaign_001'
    status = _load(campaign / 'STATUS.json', {})
    costs = _load(campaign / 'COSTS.json', {})
    output.mkdir(parents=True, exist_ok=True)
    panel_rows, request_rows, repeat_rows, audit_rows, curve = [], [], [], [], []
    summaries = {}
    for event in (6, 8):
        for panel in plan['panels'][:2 if event == 6 else 3]:
            name, mode = panel['name'], panel['canonical_mode']
            directory = campaign / f'event_{event:03d}/evaluation' / name
            progress = _load(directory / 'STATUS.json', {'status': 'unattempted'})
            protocol = _load(directory / 'PROTOCOL.json', {})
            requests = [dict(row, panel_ordinal=row.get('panel_ordinal', index))
                        for index, row in enumerate(panel['requests'])]
            measurement_path = directory / 'MEASUREMENTS.jsonl'
            native_rows = ([json.loads(line) for line in measurement_path.read_text().splitlines()]
                           if measurement_path.exists() else [])
            canonical = _load(directory / 'CANONICAL.json', {})
            rows, canonical = _local_records(directory, native_rows, canonical)
            checkpoints = protocol.get('checkpoints', {})
            summary, per_request, checked, valid_rows = summarize(rows, requests,
                checkpoints=checkpoints, canonicals=canonical, canonical_mode=mode)
            binding_errors = []
            if protocol:
                for arm in ARMS:
                    endpoint = _load(campaign / arm / 'events' / f'event_{event:03d}/RESULT.json', {})
                    if endpoint.get('export_path') != checkpoints.get(arm) or not checkpoints.get(arm):
                        binding_errors.append('own_event_endpoint_binding_' + arm)
            if protocol and (protocol.get('event_number') != event or protocol.get('panel_name') != name
                             or protocol.get('canonical_mode') != mode or protocol.get('historical_count_gate') is not False):
                binding_errors.append('evaluation_protocol_binding')
            if protocol and _load(directory / 'REQUESTS.json') != requests:
                binding_errors.append('sealed_request_manifest_binding')
            if len(rows) != progress.get('returned_generations', 0):
                binding_errors.append('returned_measurement_count')
            if len(checked) != len(rows):
                binding_errors.append('foreign_request_measurements')
                known = {item['request_id'] for item in requests}
                checked.extend(dict(item, report_valid=False, report_errors=['foreign_request_measurement'])
                               for item in rows if item['request_id'] not in known)
            complete = (progress.get('status') == 'completed' and not binding_errors
                        and summary['valid_requests'] == 64 and summary['valid_timed_generations'] == 384)
            if event == 6 and name == 'development':
                gate = _load(directory / 'QUALIFICATION.json', {})
                if (gate.get('status') != 'passed' or gate.get('requests') != 8
                        or gate.get('generations') != 48 or gate.get('extra_generations') != 0):
                    binding_errors.append('included_event6_gate')
                    complete = False
            summary.update(event_number=event, panel_name=name, canonical_mode=mode,
                           complete_valid_panel=complete, binding_errors=binding_errors, execution=progress)
            summaries[f'{event}_{name}'] = summary
            comparison = summary['comparisons']['K_vs_O']
            row = dict(event=event, panel=name, complete_valid_panel=complete,
                valid_requests=summary['valid_requests'], planned_requests=64,
                O_calls=summary['O']['verification_calls'], K_calls=summary['K']['verification_calls'],
                call_reduction_pct=comparison['call_reduction_pct'],
                request_time_reduction_pct=comparison['request_time_reduction_pct'],
                native_generate_time_reduction_pct=comparison['native_generate_time_reduction_pct'],
                **{key: comparison['call_request_directions'][key] for key in ('improved', 'regressed', 'tied')},
                gross_calls_saved=comparison['call_gross_saved'], gross_calls_added=comparison['call_gross_extra'],
                evaluation_wall_s=progress.get('evaluation_wall_s'), activation_s=progress.get('activation_s'),
                weight_validation_s=progress.get('weight_validation_s'), binding_errors=';'.join(binding_errors))
            for arm in ARMS:
                for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
                    row[arm + '_summed_mean_' + timer] = summary[arm]['summed_mean_' + timer]
                for key in ('output_tokens', 'accepted_draft_tokens', 'committed_progress',
                            'accepted_draft_tokens_per_call', 'committed_progress_per_call'):
                    row[arm + '_' + key] = summary[arm][key]
            panel_rows.append(row)
            request_rows.extend(dict(event=event, panel=name, **item) for item in per_request)
            audit_rows.extend(dict(event=event, panel=name, **item) for item in checked)
            for repeat in range(REPEATS):
                for arm in ARMS:
                    selected = [item for item in valid_rows if item['repeat'] == repeat and item['method'] == arm]
                    repeat_rows.append(dict(event=event, panel=name, repeat=repeat, method=arm,
                        requests=len(selected), verification_calls=sum(item['verification_calls'] for item in selected),
                        **{timer: sum(item[timer] for item in selected) for timer in
                           ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')}))
    interpretation = {}
    for panel in plan['panels'][:2]:
        name, historical = panel['name'], panel['historical_counts']
        old_o, old_k = sum(historical['O4'].values()), sum(historical['K4'].values())
        initial = reduction(old_o, old_k)
        differences = [historical['O4'][row['request_id']]-historical['K4'][row['request_id']]
                       for row in panel['requests']]
        curve.append(dict(event=4, panel=name, source='retained_PR16_counts_only', requests=64,
            O_calls=old_o, K_calls=old_k, call_reduction_pct=initial,
            improved=sum(value > 0 for value in differences), regressed=sum(value < 0 for value in differences),
            tied=sum(value == 0 for value in differences), complete_comparable_panel=True))
        for event in (6, 8):
            current = next(row for row in panel_rows if row['event'] == event and row['panel'] == name)
            curve.append(dict(event=event, panel=name, source='fresh_current_endpoint',
                requests=current['valid_requests'], **{key: current[key] for key in
                    ('O_calls', 'K_calls', 'call_reduction_pct', 'improved', 'regressed', 'tied')},
                complete_comparable_panel=current['complete_valid_panel']))
        final = summaries['8_'+name]
        complete = summaries['6_'+name]['complete_valid_panel'] and final['complete_valid_panel']
        interpretation[name] = curve_direction(initial, final['comparisons']['K_vs_O']['call_reduction_pct'], complete=complete)
    fresh = summaries['8_fresh']
    fresh_value = fresh['comparisons']['K_vs_O']['call_reduction_pct']
    interpretation['fresh'] = ('inconclusive: incomplete fresh panel' if not fresh['complete_valid_panel'] else
        'K has fewer calls on the fresh event-8 panel' if fresh_value > 0 else
        'K has more calls on the fresh event-8 panel' if fresh_value < 0 else 'O/K have equal fresh-panel call totals')
    def numeric_tree(value):
        if isinstance(value, dict):
            return {key: numeric_tree(item) for key, item in value.items() if isinstance(item, (dict, int, float))}
        return value
    cost_summary = numeric_tree(costs)
    cost_summary['scope'] = 'Training/save/export/loading costs are separate from serving latency; native generation and cache times are nested in complete request time.'
    actual_work = {key: status.get(key) for key in (
        'feedback_attempted', 'feedback_returned', 'evaluation_attempted', 'evaluation_returned',
        'evaluation_valid', 'optimizer_steps', 'presentations', 'arm_counters', 'arm_presentations',
        'extra_optimizer_steps', 'extra_generations', 'events_completed')}
    cost_summary['actual_work'] = actual_work
    retention = _load(campaign / 'retention/FINAL.json', {})
    retention_summary = dict(status=retention.get('status', 'unavailable'),
        required_events=retention.get('required_events'), destination=retention.get('destination'),
        verified_file_count=len(retention.get('verified_files', {})))
    campaign_summary = dict(execution_git_commit=execution_commit, execution_source=source,
        campaign_status=status.get('status'), planned_main_generations=1920,
        reaudit_returned_measurements=len(audit_rows),
        reaudit_valid_timed_generations=sum(value['valid_timed_generations'] for value in summaries.values()),
        complete_valid_panels=sum(value['complete_valid_panel'] for value in summaries.values()),
        interpretation=interpretation, panels=summaries, actual_work=actual_work,
        endpoint_retention=retention_summary,
        historical_curve_scope='Only event-4 call counts; no historical timing enters a latency learning curve.',
        no_seed_replication=True, no_integrated_asynchronous_speedup_claim=True,
        raw_table_reaudit='Every returned record is read again; invalid/incomplete request and generation rows remain visible.')
    provenance = dict(execution_git_commit=execution_commit, receipts=receipt_index(root),
        scope='Byte identities of retained source/input/launch/state/work receipts. Receipt contents and execution commands remain in the named artifacts; this index does not duplicate private arrays or independently rerun state restoration.')
    write_json(output / 'SUMMARY.json', campaign_summary)
    write_json(output / 'COSTS.json', cost_summary)
    write_json(output / 'PROVENANCE.json', provenance)
    for name, rows in (('EVENT_PANEL', panel_rows), ('PER_REQUEST', request_rows), ('ALL_REPEATS', audit_rows),
                       ('REPEAT_TOTALS', repeat_rows), ('CALL_CURVE', curve)):
        write_csv(output / (name + '.csv'), rows)
    lines = ['# OSD/K continuation through event 8', '', f'Execution revision: `{execution_commit}`.', '',
        ' '.join(name.capitalize() + ': ' + text + '.' for name, text in interpretation.items()), '',
        '| Event | Panel | Valid requests | O calls | K calls | K call reduction | Wins/losses/ties | Request-time reduction | Native-time reduction |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def shown(value):
        return 'unavailable' if value is None else f'{value:+.4f}%'
    for row in panel_rows:
        lines.append(f"| {row['event']} | {row['panel']} | {row['valid_requests']}/64 | {row['O_calls']} | {row['K_calls']} | "
            f"{shown(row['call_reduction_pct'])} | {row['improved']}/{row['regressed']}/{row['tied']} | "
            f"{shown(row['request_time_reduction_pct'])} | {shown(row['native_generate_time_reduction_pct'])} |")
    lines += ['', '| Event | Arm | Event wall (s) | Training (s) | State save (s) | Export (s) | Peak allocated (GiB) |',
              '|---:|---|---:|---:|---:|---:|---:|']
    def number(value, divisor=1):
        return 'unavailable' if value is None else f'{value/divisor:.3f}'
    for event in (5, 6, 7, 8):
        for arm in ARMS:
            item = costs.get(f'event_{event:03d}_{arm}', {})
            lines.append(f"| {event} | {arm} | " + ' | '.join(number(item.get(key)) for key in
                ('event_wall_s', 'training_s', 'checkpoint_save_s', 'export_s')) +
                ' | ' + number(item.get('peak_allocated_bytes'), 2**30) + ' |')
    lines += ['', 'Training, state save and export are components of event wall time; they are not added to it. '
        'Loading/resume checks and their memory peaks remain separate in COSTS.json. '
        f"Campaign wall time: {number(costs.get('campaign_wall_s'))} s; backend loading: {number(costs.get('backend_loading_s'))} s."]
    lines += ['', f"Recorded work: {actual_work['feedback_returned']} returned feedback responses, "
        f"{actual_work['optimizer_steps']} added optimizer steps, {actual_work['presentations']} added presentations, "
        f"{actual_work['evaluation_returned']} returned evaluation generations. "
        f"Required event-6/8 retention receipt: {retention_summary['status']}. "
        'These are recorded execution counters; null denotes unavailable evidence.']
    lines += ['', 'Primary reduction is 100 × (sum O calls − sum K calls) / sum O calls, using one validated repeat-0 trajectory per request. Three timing repeats are repeated measurements, not independent examples. Panels are never pooled.', '',
        'CALL_CURVE.csv retains event-4 historical O4/K4 counts beside fresh events 6 and 8. Historical times are excluded. Each timing comparison uses O/K measured together at the same fresh endpoint; native generation and request-cache time are nested in complete request time.', '',
        'PER_REQUEST.csv retains incomplete and rejected requests; ALL_REPEATS.csv independently rechecks raw tokens, termination, native traces, counters, timings and identities. EVENT_PANEL.csv also reports accepted draft tokens per call and committed progress per call. COSTS.json preserves numeric update/save/export/loading and memory costs separately from serving latency. PROVENANCE.json indexes source/input/launch/state/work receipts by byte identity, including available execution-command evidence.', '',
        f"Coverage: {campaign_summary['reaudit_valid_timed_generations']}/1920 valid timed executions; {campaign_summary['complete_valid_panels']}/5 complete valid panels. The event-6 first-eight-request gate is included in the main budget. Fresh-panel canonical outputs come from the first O8 execution, with no extra F/reference generation.", '',
        'Validity and interpretation limits: one continued O/K trajectory, no seed replication; all-valid-position O and current-prefix-plus-first-error K retain their qualified objectives. No KRC arm, tuning or asynchronous deployment speedup claim. Update costs do not establish adaptation-inclusive acceleration.', '',
        'ARS fallacy scan, 11/11 design checks: Simpson’s paradox (panels separate); ecological fallacy (paired requests retained); Berkson’s paradox (fixed selection limits generalization); collider bias (no performance filtering); base-rate neglect (pooled denominators shown); regression to the mean (no extreme-request selection); survivorship bias (missing/invalid cases retained); look-elsewhere effect (all fixed endpoints shown); garden of forking paths (sealed design, no tuning); correlation/causation (no training-age causal claim); reverse causality (no causal claim from the curve). No significance test or seed-population inference is made.', '',
        'Material Passport: ARS experiment-agent validate; ANALYZED by raw-record reconstruction, not an independent GPU replication. Stop after event 8 and this report; Stage 1 / project None unchanged.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return campaign_summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    write_campaign_report(args.root, args.output, execution_commit=args.execution_commit)
