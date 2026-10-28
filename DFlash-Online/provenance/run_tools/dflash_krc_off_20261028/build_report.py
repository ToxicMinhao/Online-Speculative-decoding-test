"""Local-only KRC2_OFF evidence report. Never loads models or executes training."""
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARMS = ('KRC2_OFF', 'KRC2')
TIMERS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def csv_rows(path):
    return list(csv.DictReader(path.read_text().splitlines())) if path.is_file() else []


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')


def write_csv(path, records):
    keys = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def show(value, digits=4):
    return f'{value:.{digits}f}' if number(value) else 'unavailable'


def reduction(reference, candidate):
    return 100 * (1 - candidate / reference) if reference else None


def compare(reference, candidate):
    differences = [left - right for left, right in zip(reference, candidate, strict=True)]
    return dict(reference_calls=sum(reference), candidate_calls=sum(candidate),
        call_reduction_pct=reduction(sum(reference), sum(candidate)),
        improved=sum(value > 0 for value in differences),
        regressed=sum(value < 0 for value in differences), tied=sum(value == 0 for value in differences),
        gross_saved=sum(max(value, 0) for value in differences),
        gross_added=sum(max(-value, 0) for value in differences), net_saved=sum(differences))


def panel_summary(raw, panel):
    name = panel['name']
    directory = raw / 'evaluation' / name
    source = read(directory / 'report/SUMMARY.json', {})
    per_request = csv_rows(directory / 'report/PER_REQUEST.csv')
    by_id = {row['request_id']: row for row in per_request}
    if len(by_id) != len(per_request):
        raise ValueError(name + ': duplicate request summary identities')
    for ordinal, request in enumerate(panel['requests']):
        if request['request_id'] not in by_id:
            per_request.append(dict(request_id=request['request_id'], panel_ordinal=ordinal,
                                    complete=False, valid=False, errors='unattempted_or_report_unavailable'))
    valid = [row for row in per_request if row['valid'] in ('True', True)]
    totals = {}
    for arm in ARMS:
        totals[arm] = {key: sum(int(row[f'{arm}_{key}']) for row in valid) for key in COUNTS}
        for timer in TIMERS:
            totals[arm]['summed_mean_' + timer] = sum(float(row[f'{arm}_mean_{timer}']) for row in valid)
        if source:
            for key, value in totals[arm].items():
                expected = source[arm][key]
                if (key in COUNTS and value != expected) or (key not in COUNTS and abs(value - expected) > 1e-8):
                    raise ValueError(f'{name}: report CSV/JSON mismatch at {arm}/{key}')
    call_values = {arm: [int(row[f'{arm}_verification_calls']) for row in valid] for arm in ARMS}
    historical = {arm: [int(row[f'{arm}_historical_verification_calls']) for row in valid]
                  for arm in ('KRC1', 'K2')}
    comparisons = {
        'KRC2_vs_KRC2_OFF': compare(call_values['KRC2_OFF'], call_values['KRC2']),
        'KRC2_OFF_vs_KRC1': compare(historical['KRC1'], call_values['KRC2_OFF']),
        'KRC2_vs_KRC1': compare(historical['KRC1'], call_values['KRC2']),
        'KRC2_OFF_vs_K2': compare(historical['K2'], call_values['KRC2_OFF'])}
    repeat_rows = csv_rows(directory / 'report/REPEAT_TOTALS.csv')
    repeats = []
    for repeat in range(3):
        selected = {row['method']: row for row in repeat_rows if int(row['repeat']) == repeat}
        if set(selected) != set(ARMS):
            continue
        result = dict(repeat=repeat)
        for arm in ARMS:
            result[arm + '_requests'] = int(selected[arm]['requests'])
            for timer in TIMERS:
                result[arm + '_' + timer] = float(selected[arm][timer])
        result['request_time_reduction_pct'] = reduction(
            result['KRC2_OFF_request_latency_s'], result['KRC2_request_latency_s'])
        repeats.append(result)
    execution = read(directory / 'STATUS.json', source.get('execution', {}))
    if not execution:
        execution = dict(status='unattempted', planned_generations=384, attempted_generations=0,
            completed_valid_generations=0, invalid_generations=0, unknown_generation_attempts=0,
            returned_generations=0, unattempted_generations=384)
    activations = json_rows(directory / 'ACTIVATIONS.jsonl')
    activation_s = sum(row['activation_s'] for row in activations)
    if abs(activation_s - execution.get('activation_s', 0)) > 1e-8:
        raise ValueError(name + ': activation records and total disagree')
    summary = dict(name=name, planned_requests=len(panel['requests']), valid_requests=len(valid),
        invalid_or_incomplete_requests=len(panel['requests']) - len(valid), totals=totals,
        historical_totals={arm: sum(values) for arm, values in historical.items()}, comparisons=comparisons,
        request_time_reduction_pct=reduction(totals['KRC2_OFF']['summed_mean_request_latency_s'],
                                            totals['KRC2']['summed_mean_request_latency_s']),
        repeats=repeats, execution=execution, activation_count=len(activations), activation_s=activation_s,
        native_weight_validation_s=execution.get('weight_validation_s'), canonical_outputs=source.get('canonical_outputs'),
        canonical_scope='retained F' if name == 'development' else 'retained first K1 on this panel',
        historical_count_gate='Retained per-request KRC2 calls, on every repeat',
        count_scope='One validated repeat-0 trajectory per request; historical context uses the same valid cohort.')
    return summary, per_request, repeat_rows, activations


def flatten(value):
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def training_summary(raw, costs):
    event = read(raw / 'KRC2_OFF/events/event_002/RESULT.json', {})
    off = read(raw / 'OFF_EVIDENCE.json', costs.get('off', read(raw / 'off/WORK.json', {})))
    journal = json_rows(raw / 'off/PRESENTATIONS.jsonl')
    records = event.get('records', journal)
    if event and len(records) != len(journal):
        raise ValueError('Event receipt and presentation journal have different lengths')
    if event:
        fields = ('record_id', 'pass', 'sampled_anchors', 'optimizer_step_after',
                  'optimizer_stepped', 'learning_rate', 'metrics', 'metrics_scope')
        for index, (row, entry) in enumerate(zip(records, journal, strict=True)):
            if any(row[key] != entry[key] for key in fields):
                raise ValueError('Event/journal scientific discrepancy at presentation ' + str(index))
    presentations = []
    for ordinal, source in enumerate(records):
        row = {key: source.get(key) for key in ('record_id', 'pass', 'optimizer_step_after',
               'optimizer_stepped', 'learning_rate', 'training_s', 'metrics_scope')}
        row.update(presentation_ordinal=ordinal, event_local_optimizer_step=ordinal // 2 + 1,
            sampled_anchor_occurrences=sum(sum(bool(value) for value in flatten(anchor['keep_mask']))
                                           for anchor in source.get('sampled_anchors', [])))
        row.update(source.get('metrics', {}))
        presentations.append(row)
    boundary = [row for row in records if row['optimizer_stepped']]
    selected_names = set(event.get('selection_totals', {}))
    if not selected_names and boundary:
        selected_names = {key for key in boundary[0]['metrics'] if key.startswith('first_error_position_')}
        selected_names.update(key for key in ('available_position_occurrences', 'selected_position_occurrences',
            'available_position_weight', 'selected_position_weight', 'no_error_blocks', 'empty_blocks',
            'main_loss_numerator', 'loss_denominator', 'response_presentations', 'supervised_position_exposure')
            if key in boundary[0]['metrics'])
    exposure = {key: sum(row['metrics'][key] for row in boundary) for key in sorted(selected_names)}
    if event and exposure != event.get('selection_totals', {}):
        raise ValueError('Boundary-only selection/exposure sums differ from event receipt')
    work = off.get('work', {})
    objective = event.get('objective_work', costs.get('objective_work', {}))
    summary = dict(event_status=event.get('status', 'incomplete'), event_id=event.get('event_id', 'event_002'),
        start_step=128, end_step=event.get('end_step', off.get('end_global_step')),
        optimizer_steps=work.get('Adam_steps', 0), presentations=work.get('presentations', len(records)),
        event_receipt_presentations=event.get('response_presentations'), work=work, counts=off.get('counts', {}),
        zero_reference_work=off.get('zero_reference_work', {}),
        core_response_forwards=work.get('response_forwards', 0), core_backwards=work.get('K_backwards', 0),
        sampled_anchor_occurrences=sum(row['sampled_anchor_occurrences'] for row in presentations),
        distinct_feedback_responses=len({row['record_id'] for row in records}),
        selection_totals=exposure, selection_scope='Each completed optimizer-boundary accumulation row is counted once.',
        complete_boundary_rows=len(boundary), first_A=off.get('first_A'), migration=off.get('migration'),
        objective_work=objective, costs=dict(
            event_wall_s=costs.get('event_wall_s'), training_windows_s=event.get('costs', {}).get('training_s'),
            checkpoint_save_s=event.get('costs', {}).get('checkpoint_save_s'),
            export_s=event.get('costs', {}).get('export_s'),
            journal_s=off.get('costs', {}).get('journal_s'), first_A_check_s=off.get('costs', {}).get('first_A_check_s'),
            startup_s=costs.get('startup', {}).get('wall_s'), snapshot_validation_s=costs.get('snapshot_validation_s'),
            loading_and_live_resume_s=costs.get('loading_and_live_resume_s'), backend_loading_s=costs.get('backend_loading_s'),
            peak_allocated_bytes=costs.get('peak_allocated_bytes'), peak_reserved_bytes=costs.get('peak_reserved_bytes'),
            actual_campaign_wall_s=costs.get('actual_campaign_wall_s'),
            nesting='Training, objective and journal timers are nested. Persistence/export belong to event wall time; startup and backend loading are separate.'),
        work_counter_scopes=dict(
            response_forwards='Complete response-level model evaluations; module invocations and recomputation are separate.',
            head_forwards_including_recompute='Native head invocations, including teacher projections and checkpoint recomputation; not unique sampled blocks.',
            layer_forwards_including_recompute='Native layer invocations including checkpoint recomputation.',
            objective_work='Existing K-objective timers, nested in training time. No new teacher-feedback preparation is performed.'))
    return summary, presentations


def render(summary):
    status = summary['status']
    training = summary['training']
    panels = summary['panels']
    complete = status.get('status') == 'completed' and all(panel['valid_requests'] == 64 for panel in panels.values())
    text = ['# KRC2 correction-removal control', '',
        f"Execution: **{status.get('status', 'unavailable')}**. Development and additional panels remain separate.", '',
        'KRC2_OFF restores the exact corrected KRC1 weights, AdamW moments/counters, configuration/history and training RNG at step 128. '
        'Only then are reference processing, correction and optional probes disabled. Ordinary K continues on the unchanged event-2 pool to step 256. '
        'Retained KRC2 is the contemporaneously evaluated control.', '']
    if complete:
        gains = [panel['comparisons']['KRC2_vs_KRC2_OFF']['net_saved'] for panel in panels.values()]
        if all(gain > 0 for gain in gains):
            text.append('**KRC2 uses fewer verification calls on both panels:** event-2 correction adds endpoint benefit from the same KRC1 state in this test.')
        elif all(gain <= 0 for gain in gains):
            text.append('**OFF matches or improves verification counts on both panels:** removing correction is a simplification candidate for this tested continuation.')
        else:
            text.append('**The call-count comparison is mixed across panels.** This test does not establish a consistent endpoint benefit from event-2 correction.')
        text.extend(['Differences are descriptive for these fixed checkpoints and requests. Small or mixed differences do not establish equivalence or a universally optimal event-1-only correction policy.', ''])
    else:
        text.extend(['**Incomplete execution:** available valid pairs are summarized below, but no completed endpoint conclusion is drawn. Failures and unattempted work remain in the records.', ''])
    text.extend(['| Panel | OFF calls | KRC2 calls | Call reduction favoring KRC2 | Request-time reduction | KRC2 wins / losses / ties | Gross saved / added |',
                 '|---|---:|---:|---:|---:|---:|---:|'])
    for name, panel in panels.items():
        comparison = panel['comparisons']['KRC2_vs_KRC2_OFF']
        text.append(f"| {name} | {comparison['reference_calls']} | {comparison['candidate_calls']} | {show(comparison['call_reduction_pct'])}% | {show(panel['request_time_reduction_pct'])}% | "
                    f"{comparison['improved']} / {comparison['regressed']} / {comparison['tied']} | {comparison['gross_saved']} / {comparison['gross_added']} |")
    text.extend(['', 'Primary reduction is `1 − ΣKRC2_calls / ΣOFF_calls`; positive values favor retaining correction. '
        'Counts use one validated trajectory per request after repeat-consistency checks. Ratios use summed counts, not averages of request-level percentages.', '',
        '| Panel | Retained KRC1 calls | OFF calls | OFF reduction versus KRC1 | Retained K2 calls (context) |',
        '|---|---:|---:|---:|---:|'])
    for name, panel in panels.items():
        comparison = panel['comparisons']['KRC2_OFF_vs_KRC1']
        text.append(f"| {name} | {comparison['reference_calls']} | {comparison['candidate_calls']} | {show(comparison['call_reduction_pct'])}% | {panel['historical_totals']['K2']} |")
    text.extend(['', 'Historical context uses count measurements on the same valid cohort. K2 and KRC1 are not freshly generated or timed here.', '',
        '| Panel | Arm | Complete request (s) | Native generation (s) | Cache maintenance (s) | Accepted draft tokens | Output tokens |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for name, panel in panels.items():
        for arm, total in panel['totals'].items():
            text.append(f"| {name} | {arm} | {show(total['summed_mean_request_latency_s'], 6)} | {show(total['summed_mean_native_generate_s'], 6)} | "
                        f"{show(total['summed_mean_cache_maintenance_s'], 6)} | {total['accepted_draft_tokens']} | {total['output_tokens']} |")
    text.extend(['', 'Request totals sum three-repeat means. Native-generation and cache-maintenance times are nested components of complete request latency. Loading and checkpoint switching are separate.', '',
        '| Panel | Repeat | OFF request time (s) | KRC2 request time (s) | KRC2 reduction |', '|---|---:|---:|---:|---:|'])
    for name, panel in panels.items():
        for row in panel['repeats']:
            text.append(f"| {name} | {row['repeat']} | {show(row['KRC2_OFF_request_latency_s'], 6)} | {show(row['KRC2_request_latency_s'], 6)} | {show(row['request_time_reduction_pct'])}% |")
    text.extend(['', 'Per-repeat native-generation and cache-maintenance totals are also retained in `REPEAT_TOTALS.csv`.', '',
        '## Restoration, training and cost', '',
        f"Exact KRC1 live restoration: **{summary['resume'].get('exact_resume', 'unavailable')}**. "
        f"First actual A-update reproduction: **{(training.get('first_A') or {}).get('status', 'unavailable')}**. "
        'This first update is part of the main event; only its pre-correction K metrics, anchors and counters are matched. No later trajectory equality is required.', '',
        f"Actual new work: **{training['optimizer_steps']} AdamW steps, {training['presentations']} presentations, "
        f"{training['core_response_forwards']} core response forwards and {training['core_backwards']} backwards**. "
        f"Global steps: 128→{training['end_step']}. "
        f"Reference/correction work: `{json.dumps(training['zero_reference_work'], sort_keys=True)}`.", '',
        f"Ordinary sampled-anchor occurrences: {training['sampled_anchor_occurrences']}. "
        f"Exposure statistics sum {training['complete_boundary_rows']} complete accumulation-boundary rows once. "
        'They do not add the individual presentation metrics again. No reference-response exposure is introduced.', '',
        '| Cost | Seconds |', '|---|---:|'])
    for key, value in training['costs'].items():
        if key.endswith('_s'):
            text.append(f'| {key} | {show(value, 3)} |')
    allocated = training['costs']['peak_allocated_bytes']
    reserved = training['costs']['peak_reserved_bytes']
    text.extend(['', f"Peak allocated/reserved memory: {show(allocated / 1e9 if number(allocated) else None, 3)} / "
        f"{show(reserved / 1e9 if number(reserved) else None, 3)} GB.", '',
        'Qualified disposable startup is separately counted: two response forwards/backwards and zero Adam steps. '
        'Core response counts are separate from projection/module invocations and recomputation. Startup, first-A journal checks, persistence and export are recorded separately; nested costs must not be added twice.', '',
        'Historical KRC2 training costs are context, not fresh matched overhead measurements. Serving gains exclude adaptation cost; historical-versus-fresh serving latency is not a learning curve.', '',
        '| Panel | Valid / invalid / unknown / unattempted executions | Publications | Switching time (s) | Weight checks (s) |',
        '|---|---:|---:|---:|---:|'])
    for name, panel in panels.items():
        execution = panel['execution']
        counts = [execution.get(key, 'unavailable') for key in ('completed_valid_generations', 'invalid_generations',
                   'unknown_generation_attempts', 'unattempted_generations')]
        text.append(f"| {name} | {' / '.join(map(str, counts))} | {panel['activation_count']} | {show(panel['activation_s'], 3)} | {show(panel['native_weight_validation_s'], 3)} |")
    text.extend(['', 'Both panels require exact retained canonical output tokens and termination, repeat-consistent counts, and reproduction of retained per-request KRC2 verification counts. '
        'Only the first eight development requests provide the included 48-execution gate. Training closes before timed evaluation; no additional generation warmups are used.', '',
        'This control isolates removal of event-2 reference processing and correction from the same KRC1 state. It does not establish seed robustness, equivalence, training-inclusive acceleration or a general optimal correction schedule. '
        'No tuning or further events follow from this report.', '',
        'Evidence: `analysis/SUMMARY.json`, `TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `WORK_COUNTERS.csv`, '
        'panel-specific request/repeat tables, and raw resume/migration/first-A receipts and native outputs under `raw/campaign_001`.'])
    historical = summary.get('historical_KRC2_costs', {})
    if historical:
        text.extend(['', 'Historical cost context: the retained KRC2 event took '
            f"{show(historical.get('event_wall_s'), 3)} s, including "
            f"{show(historical.get('intrinsic_reference_correction_s'), 3)} s of reference/correction work. "
            'These earlier measurements are not a contemporaneous training-cost control.'])
    text.extend(['', 'Implementation and reproducibility: [independent implementation review](REVIEW.md), '
        '[implementation diff](IMPLEMENTATION.diff), [sealed actual-path plan](raw/sealed/PLAN.json), '
        '[configuration](raw/sealed/config.yaml), [model pins](raw/sealed/MODEL_PIN.json), '
        '[runtime pins](raw/sealed/RUNTIME_PIN.json), [full cost receipt](raw/campaign_001/COSTS.json), '
        '[first-update check](raw/campaign_001/off/FIRST_A.json), '
        '[retained final checkpoint/recovery identities](raw/FINAL_ARTIFACTS.json). '
        'Nineteen focused CPU tests passed locally and on A800. Inherited training and inference source remained unchanged.', '',
        'Both checkpoints use Qwen3-8B with the qualified DFlash implementation, BF16, greedy block-16 decoding, '
        '2,048 output tokens or EOS, context limit 4,096, concurrency one and capture disabled. '
        'The original capped feedback response was preserved. '
        f"Ordinary training sampled {training['sampled_anchor_occurrences']:,} anchor occurrences; "
        f"selection admitted {show(training['selection_totals'].get('selected_position_occurrences'), 0)} "
        f"of {show(training['selection_totals'].get('available_position_occurrences'), 0)} available proposal occurrences. "
        'These are overlapping training occurrences, not independent examples.'])
    return '\n'.join(text) + '\n'


def build(root=HERE):
    root = Path(root)
    raw = root / 'raw/campaign_001'
    if not raw.is_dir():
        raise FileNotFoundError('Raw OFF evidence has not arrived: ' + str(raw))
    plan = read(root / 'raw/sealed/PLAN.json', read(root / 'sealed/PLAN.json'))
    if plan is None:
        raise FileNotFoundError('Sealed plan is required to retain unattempted requests')
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    status = read(raw / 'STATUS.json', {})
    costs = read(raw / 'COSTS.json', {})
    training, presentations = training_summary(raw, costs)
    panels = {}
    for panel in plan['panels']:
        summary, per_request, repeats, activations = panel_summary(raw, panel)
        name = panel['name']; panels[name] = summary
        destination = output / name; destination.mkdir(exist_ok=True)
        write(destination / 'SUMMARY.json', summary)
        write_csv(destination / 'PER_REQUEST.csv', per_request)
        write_csv(destination / 'REPEAT_TOTALS.csv', repeats)
        write(destination / 'ACTIVATIONS.json', activations)
        source = raw / 'evaluation' / name / 'report/ALL_REPEATS.csv'
        if source.is_file():
            (destination / 'ALL_REPEATS.csv').write_bytes(source.read_bytes())
    summary = dict(status=status, resume=read(raw / 'LIVE_RESUME.json', {}),
        snapshot_check=read(raw / 'RESUME_SNAPSHOT.json', {}),
        bindings=read(raw / 'PRESENTATION_SCHEDULE_CHECK.json', {}),
        training_check=read(raw / 'TRAINING_CHECK.json', {}), off_check=read(raw / 'OFF_CHECK.json', {}),
        training=training, startup=costs.get('startup', {}), startup_work=costs.get('startup_work_counts', {}), panels=panels)
    historical = read(root.parent / 'dflash_krc_continuation_20261028/analysis/TRAINING_SUMMARY.json', {})
    summary['historical_KRC2_costs'] = historical.get('costs', {})
    write(output / 'SUMMARY.json', summary)
    write(output / 'TRAINING_SUMMARY.json', training)
    write_csv(output / 'PRESENTATIONS.csv', presentations)
    counters = [dict(counter=name, value=value, scope=training['work_counter_scopes'].get(name,
                'Exact work observer counter; module/chunk calls may include recomputation.'))
                for name, value in training['counts'].items()]
    counters.extend(dict(counter=name, value=value, scope='Explicit zero-reference contract')
                    for name, value in training['zero_reference_work'].items())
    write_csv(output / 'WORK_COUNTERS.csv', counters)
    (root / 'REPORT.md').write_text(render(summary))
    return summary


if __name__ == '__main__':
    result = build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
    print(json.dumps(dict(status=result['status'].get('status'),
                         comparisons={name: panel['comparisons'] for name, panel in result['panels'].items()}), indent=2))
