"""Local compact-evidence reporting; never loads models or runs experiments."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ARMS = ('O4', 'K4', 'KRC4')
PAIRS = (('KRC4', 'O4'), ('K4', 'O4'), ('KRC4', 'K4'))
TIMERS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def show(value, digits=3):
    return f'{value:.{digits}f}' if number(value) else 'unavailable'


def reduction(reference, candidate):
    return 100 * (reference - candidate) / reference if number(reference) and reference > 0 and number(candidate) else None


def comparison_rows(panel):
    rows = []
    if not panel.get('valid_requests'):
        return rows
    for candidate, reference in PAIRS:
        measured = panel['comparisons'][candidate + '_vs_' + reference]
        rows.append(dict(candidate=candidate, reference=reference,
            baseline_calls=panel[reference]['verification_calls'], method_calls=panel[candidate]['verification_calls'],
            call_reduction_pct=reduction(panel[reference]['verification_calls'], panel[candidate]['verification_calls']),
            request_time_reduction_pct=reduction(panel[reference]['summed_mean_request_latency_s'],
                                                 panel[candidate]['summed_mean_request_latency_s']),
            **measured['call_request_directions'], gross_saved=measured['call_gross_saved'],
            gross_added=measured['call_gross_extra']))
    return rows


def persistence(panel):
    """Describe relative gaps separately from movement of KRC itself."""
    by_event = {row['event']: row for row in panel['event_count_curve']}
    current, prior = by_event[4], by_event[2]
    sentences = []
    for baseline in ('O', 'K'):
        new = reduction(current[baseline], current['KRC'])
        old = reduction(prior[baseline], prior['KRC'])
        if new is None:
            sentences.append(f'KRC4 versus {baseline}4 is unavailable.')
            continue
        if old is None:
            result = 'improves on' if new > 0 else 'ties' if new == 0 else 'regresses against'
            sentences.append(f'KRC4 {result} {baseline}4 ({new:+.4f}%). Earlier {baseline} counts are absent, so persistence against {baseline} cannot be assessed.')
        else:
            if new > 0:
                trend = 'widens' if new > old else 'shrinks' if new < old else 'is unchanged'
                result = f'advantage persists and {trend}' if old > 0 else 'advantage emerges'
            else:
                result = ('advantage disappears into a tie' if new == 0 else 'advantage reverses') if old > 0 else 'advantage does not emerge'
            sentences.append(f'Against {baseline}, the {result}: event 2 {old:+.4f}% → event 4 {new:+.4f}% ({new-old:+.4f} percentage points).')
    own = reduction(prior['KRC'], current['KRC'])
    sentences.append(f'KRC4 versus KRC2 alone changes calls by {show(own, 4)}% reduction; that within-arm change does not establish relative benefit.')
    return ' '.join(sentences)


def training_rows(raw, costs, audit):
    rows = []
    for number_ in (3, 4):
        event_id = f'event_{number_:03d}'
        measured = costs.get('events', {}).get(event_id, {})
        receipt = read(raw / f'KRC/events/{event_id}/RESULT.json', {})
        correction = read(raw / f'events/{event_id}/CORRECTION_EVIDENCE.json', measured.get('correction', {}))
        counts, work, timers = (correction.get(key, {}) for key in ('counts', 'work', 'costs'))
        event_cost = receipt.get('costs', measured.get('event', {}))
        intrinsic_values = [timers.get(key) for key in ('pre_reference_s', 'post_k_s', 'correction_s')]
        intrinsic = sum(intrinsic_values) if all(number(value) for value in intrinsic_values) else None
        integrity_values = [timers.get(key) for key in ('integrity_s', 'pre_reference_integrity_s', 'first_window_gate_s')]
        integrity = sum(integrity_values) if all(number(value) for value in integrity_values) else None
        inclusive = event_cost.get('training_s')
        residual = inclusive - intrinsic - integrity if all(number(value) for value in (inclusive, intrinsic, integrity)) else None
        audited = audit.get('training', {}).get('events', {}).get(event_id, {})
        exposure = audited.get('exposure', {})
        branches = audited.get('correction_branches')
        if branches is None:
            branches = dict(Counter((row.get('correction') or {}).get('status', 'unavailable')
                                    for row in correction.get('steps', [])))
        rows.append(dict(event=number_, status=receipt.get('status', correction.get('status', 'unattempted')),
            start_step=receipt.get('start_step'), end_step=receipt.get('end_step'),
            Adam_steps=counts.get('Adam_completed'), ordinary_presentations=counts.get('presentations_completed'),
            ordinary_forwards=work.get('original_response_forwards'), ordinary_backwards=counts.get('K_backward_completed'),
            ordinary_anchors=exposure.get('kept_anchors'), ordinary_available_occurrences=exposure.get('available_occurrences'),
            ordinary_selected_occurrences=exposure.get('selection', {}).get('selected_position_occurrences'),
            reference_capture_forwards=work.get('pre_reference_response_forwards'),
            reference_replay_forwards=work.get('post_k_reference_response_forwards'),
            reference_backwards=work.get('violation_backward_completed'),
            reference_presentations=audited.get('reference_presentations'), reference_anchors=audited.get('reference_anchors'),
            reference_available_occurrences=audited.get('reference_available_occurrences'),
            reference_protected_occurrences=audited.get('reference_protected_occurrences'),
            BF16_nonzero_corrections=audited.get('realized_nonzero_corrections'),
            correction_branches=json.dumps(branches, sort_keys=True),
            event_wall_s=measured.get('event_wall_s'), training_windows_inclusive_s=inclusive,
            intrinsic_reference_correction_s=intrinsic, integrity_checks_s=integrity,
            ordinary_K_and_other_overhead_residual_s=residual,
            **{key: event_cost.get(key) for key in ('checkpoint_save_s', 'export_s', 'state_retention_s')},
            **{key: measured.get(key) for key in ('saved_state_validation_s', 'artifact_identity_s',
                                                'peak_allocated_bytes', 'peak_reserved_bytes')},
            full_state_path=receipt.get('full_state_path'), export_path=receipt.get('checkpoint_path')))
    return rows


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'])
    lines.extend('| ' + ' | '.join(str(value) if value is not None else 'absent' for value in row) + ' |' for row in rows)


def complete_execution(status, generations):
    return status.get('status') == 'completed' and all(status.get(key) == value for key, value in {
        'completed_valid_generations': generations, 'invalid_generations': 0,
        'unknown_generation_attempts': 0, 'unattempted_generations': 0}.items())


def complete_panel(panel):
    """Missing or contradictory report artifacts cannot qualify an endpoint."""
    if (panel.get('valid_requests') != 64 or panel['missing_report_files']
            or not complete_execution(panel['execution'], 576)):
        return False
    rows = panel['repeats']
    try:
        if len(rows) != 9 or {(row['method'], int(row['repeat'])) for row in rows} != {
                (arm, repeat) for arm in ARMS for repeat in range(3)}:
            return False
        for arm in ARMS:
            selected = [row for row in rows if row['method'] == arm]
            if any(int(row['requests']) != 64 or int(row['verification_calls']) != panel[arm]['verification_calls']
                   for row in selected):
                return False
            for timer in TIMERS:
                times = [float(row[timer]) for row in selected]
                expected = panel[arm]['summed_mean_' + timer]
                if (not all(number(value) and value >= 0 for value in times) or not number(expected)
                        or not math.isclose(sum(times) / 3, expected, rel_tol=1e-9, abs_tol=1e-9)):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def render(summary):
    valid = summary['completed_audited_endpoint']
    lines = ['# KRC continuation through event 4', '',
        f"Execution **{summary['status'].get('status', 'unavailable')}**; independent audit **{summary['audit'].get('status', 'unavailable')}**.", '',
        'KRC2’s complete corrected weights, own BF16 Adam state, RNG and history continue through the retained O-generated event-3/4 feedback. O is the main OSD-on-DFlash baseline; K is the correction ablation. The panels remain separate.', '',
        'At every ordinary K update, the next disjoint response pair in the same pass supplies its current correct prefixes. The qualified post-K reference correction adjusts draft weights while leaving Adam moments/counters unchanged. Optional diagnostic probes remain disabled.', '',
        ('Both panels completed the declared endpoint comparison and passed the retained-evidence audit.' if valid else
         '**No completed endpoint conclusion:** execution, full-panel measurements or audit verification is incomplete. Available valid cohorts and all failed/unknown/unattempted work remain visible.')]
    for name, panel in summary['panels'].items():
        lines.extend(['', f'## {name}', '', f"Valid common requests: {panel.get('valid_requests', 0)}/64."])
        if panel['missing_report_files']:
            lines.append('Missing panel report files: ' + ', '.join(panel['missing_report_files']) + '.')
        if valid:
            lines.append(persistence(panel))
        comparisons = comparison_rows(panel)
        table(lines, ['Comparison', 'Baseline / method calls', 'Call reduction', 'Request-time reduction', 'Wins / losses / ties', 'Saved / added'],
            [[row['candidate'] + ' vs ' + row['reference'], f"{row['baseline_calls']} / {row['method_calls']}",
              show(row['call_reduction_pct'], 4) + '%', show(row['request_time_reduction_pct'], 4) + '%',
              f"{row['improved']} / {row['regressed']} / {row['tied']}", f"{row['gross_saved']} / {row['gross_added']}"] for row in comparisons])
        table(lines, ['Event', 'Measurement', 'O calls', 'K calls', 'KRC calls'],
            [[row[key] for key in ('event', 'measurement', 'O', 'K', 'KRC')] for row in panel.get('event_count_curve', [])])
        table(lines, ['Arm', 'Complete request s', 'Native generation s', 'Cache maintenance s'],
            [[arm, *[show(panel[arm].get('summed_mean_' + timer), 6) for timer in TIMERS]]
             for arm in ARMS if panel.get('valid_requests') and arm in panel])
        table(lines, ['Repeat', 'Arm', 'Requests', 'Calls', 'Request s', 'Native s', 'Cache s'],
            [[row.get(key) for key in ('repeat', 'method', 'requests', 'verification_calls', *TIMERS)] for row in panel['repeats']])
        execution = panel['execution']
        lines.extend(['', 'Valid / invalid / unknown / unattempted executions: ' + ' / '.join(str(execution.get(key, 'unavailable')) for key in
            ('completed_valid_generations', 'invalid_generations', 'unknown_generation_attempts', 'unattempted_generations')) + '.',
            f"Publication/switching {show(execution.get('activation_s'))} s and native weight checks {show(execution.get('weight_validation_s'))} s are outside request latency."])
    lines.extend(['', 'Counts use one validated trajectory per request; reductions are 100 × (baseline total − method total) / baseline total. Three timing repeats are not independent training runs. Request totals sum each request’s repeat mean. Native-generation and cache-maintenance timers are nested inside request time. Historical counts support the event-1/2/4 curves; historical and fresh latency are never used as a learning curve.', '', '## Training and retained states'])
    table(lines, ['Event', 'Adam / presentations', 'Ordinary forwards / backwards', 'Reference capture / replay / backwards', 'BF16 nonzero', 'Branches'],
        [[r['event'], f"{r['Adam_steps']} / {r['ordinary_presentations']}", f"{r['ordinary_forwards']} / {r['ordinary_backwards']}",
          f"{r['reference_capture_forwards']} / {r['reference_replay_forwards']} / {r['reference_backwards']}",
          r['BF16_nonzero_corrections'], r['correction_branches']] for r in summary['training']])
    table(lines, ['Event', 'Wall s', 'Inclusive train s', 'Reference/correction s', 'Integrity s', 'K + other residual s', 'Save s', 'Export s', 'Retention s', 'Peak allocated / reserved GB'],
        [[r['event'], *[show(r[key]) for key in ('event_wall_s', 'training_windows_inclusive_s', 'intrinsic_reference_correction_s',
          'integrity_checks_s', 'ordinary_K_and_other_overhead_residual_s', 'checkpoint_save_s', 'export_s', 'state_retention_s')],
          ' / '.join(show(r[key] / 1e9 if number(r[key]) else None) for key in ('peak_allocated_bytes', 'peak_reserved_bytes'))] for r in summary['training']])
    for row in summary['training']:
        lines.extend(['', f"KRC{row['event']} state: `{row['full_state_path']}`; serving export: `{row['export_path']}`.",
            f"Audited reference presentations / anchors / available / protected occurrences: {row['reference_presentations']} / {row['reference_anchors']} / {row['reference_available_occurrences']} / {row['reference_protected_occurrences']}."])
        lines.append(f"Audited ordinary anchors / available / selected position occurrences: {row['ordinary_anchors']} / {row['ordinary_available_occurrences']} / {row['ordinary_selected_occurrences']}. These overlapping occurrences are not independent examples.")
    lines.extend(['', 'Inclusive training time sums the presentation timers, including ordinary K, reference/correction and integrity hooks. Response-tensor loading occurs outside those timers and remains in event wall time. The residual subtracts measured reference/correction and integrity timers; it includes ordinary K and unseparated overhead, not an independently measured pure-K time. Save/export/retention are nested in event wall time. Saved-state validation and artifact hashing follow the event timer and are separately retained in TRAINING.csv and campaign costs. The complete states retain the 7,424-step horizon; task orchestration stops at 512. Reference exposure adds computation, so training is not compute matched.',
        f"Disposable startup: `{json.dumps(summary['costs'].get('startup', {}), sort_keys=True)}`.",
        f"Backend loading: {show(summary['costs'].get('backend_loading_s'))} s; campaign wall: {show(summary['costs'].get('actual_campaign_wall_s'))} s.", '', '## Validity and provenance', '',
        f"KRC2 live restoration: `{summary['resume'].get('exact_resume', 'unavailable')}`; final state retention: `{summary['retention'].get('status', 'unavailable')}`; storage admission: `{summary['storage'].get('admitted', 'unavailable')}`.",
        'Correctness requires exact canonical output tokens/termination and repeat-consistent counts. Development additionally reproduces retained O4/K4 per-request counts; additional-panel O4/K4 counts are new. The first eight development requests supply the included 72-execution native gate within the 1,152-generation budget.',
        f"Execution commit: `{summary['source'].get('execution_git_commit', 'unavailable')}`. Postprocessing commit: `{summary['postprocessing'].get('git_commit') or 'not supplied; exact script SHA-256 recorded'}`; script SHA-256 `{summary['postprocessing']['script_sha256']}`.",
        f"Execution source: `{summary['source'].get('working_source', 'unavailable')}`; dirty files: `{json.dumps(summary['source'].get('dirty_files'))}`.",
        f"Audit issues: `{json.dumps(summary['audit'].get('issues', []))}`. Source/retrieval verification: `{json.dumps(summary['audit_integrity_counts'], sort_keys=True)}`.",
        'Exact inputs, model/config/runtime pins and commands are retained in [sealed plan](raw/sealed/PLAN.json), [preflight](raw/PREFLIGHT.json), [postflight](raw/POSTFLIGHT.json), [launch record](LAUNCH.json), [launch script](run_campaign.sh), [independent audit](FINAL_AUDIT.json), and [complete costs](raw/campaign_001/COSTS.json). Machine-readable root SUMMARY.json/TRAINING.csv and analysis panel files preserve all figures, paths, request rows and repeats.', '',
        'KRC4 versus K4 compares their complete correction histories; it does not isolate corrections in events 3 and 4 because the event-2 states already differ. This is one shared-feedback trajectory and fixed-checkpoint serving evidence, not seed robustness or training-inclusive asynchronous deployment speedup. No tuning, new arm, event-3 serving evaluation or continuation beyond event 4 follows. Stage 1 / project None remains unchanged.'])
    return '\n'.join(lines) + '\n'


def build(root=HERE, *, postprocess_commit=None):
    root = Path(root)
    raw = root / 'raw/campaign_001'
    plan = read(root / 'raw/sealed/PLAN.json')
    if plan is None:
        raise FileNotFoundError('A retained sealed plan is required; no results are inferred without it')
    status, costs, audit = read(raw / 'STATUS.json', {}), read(raw / 'COSTS.json', {}), read(root / 'FINAL_AUDIT.json', {})
    panels = {}
    for declared in plan['panels']:
        name = declared['name']
        source = raw / 'evaluation' / name / 'report'
        panel = read(source / 'SUMMARY.json', dict(valid_requests=0, comparisons={}))
        panel['missing_report_files'] = [filename for filename in
            ('SUMMARY.json', 'PER_REQUEST.csv', 'ALL_REPEATS.csv', 'REPEAT_TOTALS.csv')
            if not (source / filename).is_file()]
        panel['execution'] = read(raw / 'evaluation' / name / 'STATUS.json', status.get('panels', {}).get(name, {}))
        panel['repeats'] = list(csv.DictReader((source / 'REPEAT_TOTALS.csv').read_text().splitlines())) if (source / 'REPEAT_TOTALS.csv').is_file() else []
        panels[name] = panel
        destination = root / 'analysis' / name
        write_csv(destination / 'COMPARISONS.csv', comparison_rows(panel))
        write_csv(destination / 'EVENT_COUNT_CURVE.csv', panel.get('event_count_curve', []))
        for filename in ('PER_REQUEST.csv', 'ALL_REPEATS.csv', 'REPEAT_TOTALS.csv'):
            if (source / filename).is_file():
                shutil.copyfile(source / filename, destination / filename)
    integrity = audit.get('integrity', {})
    summary = dict(status=status, completed_audited_endpoint=complete_execution(
        {**status.get('evaluation_totals', {}), 'status': status.get('status')}, 1152)
        and audit.get('status') == 'passed' and not audit.get('issues')
        and set(panels) == {'development', 'additional'} and all(complete_panel(panel) for panel in panels.values()),
        panels=panels, training=training_rows(raw, costs, audit), costs={key: value for key, value in costs.items() if key not in ('events', 'interrupted_event')},
        source=plan.get('execution_source', read(root / 'EXECUTION_SOURCE.json', {})),
        postprocessing=dict(git_commit=postprocess_commit, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        resume=read(raw / 'LIVE_RESUME.json', {}), retention=read(raw / 'STATE_RETENTION.json', {}),
        storage=read(root / 'raw/STORAGE_ADMISSION.json', integrity.get('preflight', {}).get('storage', {})),
        audit={key: audit[key] for key in ('status', 'issues', 'provenance', 'native_blocks') if key in audit},
        audit_integrity_counts={key: integrity[key] for key in ('sealed_files', 'local_files_rehashed', 'remote_only_files', 'retrieved_files_rehashed') if key in integrity})
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'TRAINING.csv', summary['training'])
    (root / 'REPORT.md').write_text(render(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, nargs='?', default=HERE)
    parser.add_argument('--postprocess-commit')
    arguments = parser.parse_args()
    result = build(arguments.root, postprocess_commit=arguments.postprocess_commit)
    print(json.dumps({'completed_audited_endpoint': result['completed_audited_endpoint'], 'output': str(arguments.root / 'REPORT.md')}))
